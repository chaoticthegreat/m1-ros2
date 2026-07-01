"""Interactive keyboard interface for the M1 robot (sim AND real).

This is the operator console. Unlike ``isaac/teleop.py`` (which drives the Isaac
Sim articulation directly and therefore only works in simulation), this node
talks *only* to the ``m1_controller`` brain over the standard ``/m1/*`` ROS 2
topics. Those same topics are consumed by the controller whether the low-level
driver is Isaac Sim or the real hardware, so the exact same console drives both:

    out  /m1/<arm>/target_pose   geometry_msgs/PoseStamped   (Cartesian reach)
    out  /m1/cmd_vel             geometry_msgs/Twist         (swerve base)
    out  /m1/<arm>/gripper       std_msgs/Float64            (0=closed..1=open)
    in   /joint_states           sensor_msgs/JointState      (to seed targets)

Each arm keeps an internal Cartesian target point (in the ``base_link`` frame).
On startup the target is seeded from the live ``/joint_states`` via the URDF
forward kinematics, so the arm does not jump when the console connects -- the
target starts exactly where the fingertip already is, and the keys nudge it
from there.

Run it in its own terminal (it needs keyboard focus; it works fine over SSH and
headless because it is pure text):

    ros2 run m1_control m1_teleop

-------------------------------------------------------------------------------
KEYBOARD MAP
-------------------------------------------------------------------------------
  Base (swerve body velocity, hold to drive)
    w / s ............ drive forward / backward
    a / d ............ turn (yaw) left / right
    q / e ............ strafe (crab) left / right
    space ............ stop the base now

  Active arm gripper target (base_link frame: x fwd, y left, z up)
    i / k ............ move target +x / -x  (forward / backward)
    j / l ............ move target +y / -y  (left / right)
    u / o ............ move target +z / -z  (up / down)

  Arm select + grippers
    t ................ switch active arm (LEFT <-> RIGHT)
    [ / ] ............ close / open the active arm's gripper

  Global
    + / - ............ increase / decrease motion step size
    h ................ re-seed arm targets to the current pose & stop base
    ESC or Ctrl-C .... quit
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import select
import sys
import termios
import tty

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from m1_control.kinematics import (
    ARM_JOINTS,
    LIFT_JOINT,
    ReachController,
    UrdfModel,
)

# --- Motion increments (per key press; hold = key auto-repeat) ---------------
ARM_STEP = 0.02          # Cartesian target nudge per press (m)
GRIP_STEP = 0.08         # gripper open/close per press (0..1)
STEP_SCALE_MIN = 0.25    # bounds on the +/- step multiplier
STEP_SCALE_MAX = 4.0

# --- Base body-velocity command ----------------------------------------------
MAX_LINEAR = 0.5         # forward / reverse speed (m/s)
MAX_STRAFE = 0.4         # sideways crab speed (m/s)
MAX_YAW = 1.0            # yaw rate (rad/s, +ve = left/CCW)
LINEAR_ACCEL = 1.5       # ramp toward the commanded linear speed (m/s^2)
YAW_ACCEL = 3.0          # ramp toward the commanded yaw rate (rad/s^2)
BASE_HOLD = 0.4          # s without a base key before the command zeros out

# --- Soft workspace clamp on the target point (base_link frame, m) -----------
# Height (z) is intentionally unbounded: the operator may aim the target at any
# height and the controller's IK reaches as close as the joint limits allow.
TARGET_LIMITS = {
    "x": (-0.9, 0.9),
    "y": (-0.9, 0.9),
    "z": (float("-inf"), float("inf")),
}

DEFAULT_TARGET = {
    "left": np.array([0.40, 0.25, 0.70], dtype=np.float64),
    "right": np.array([0.40, -0.25, 0.70], dtype=np.float64),
}

LOOP_PERIOD = 0.02       # s (50 Hz key poll + publish loop)
ESC = "\x1b"


def _default_urdf_path() -> str:
    try:
        share = get_package_share_directory("ranger_air_description")
        return os.path.join(share, "urdf", "ranger_air_description.urdf")
    except Exception:  # noqa: BLE001
        return ""


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _move_toward(current: float, target: float, max_delta: float) -> float:
    delta = target - current
    if abs(delta) <= max_delta:
        return target
    return current + (max_delta if delta > 0 else -max_delta)


class M1Teleop(Node):
    """Publishes high-level commands to the m1_controller from the keyboard."""

    def __init__(self):
        super().__init__("m1_teleop")

        self.declare_parameter("urdf_path", _default_urdf_path())
        urdf_path = self.get_parameter("urdf_path").value

        # The URDF lets us seed each arm's target at its current fingertip, so
        # the arm holds still when the console connects instead of snapping to a
        # default point. If it is missing we fall back to fixed default targets.
        self.reach = None
        if urdf_path and os.path.isfile(urdf_path):
            try:
                with open(urdf_path, "r") as fh:
                    self.reach = ReachController(UrdfModel.from_string(fh.read()))
                self.get_logger().info(f"loaded URDF kinematics from {urdf_path}")
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"URDF FK unavailable ({exc}); using default targets")
        else:
            self.get_logger().warn("no URDF found; arm targets start at defaults")

        # --- state ---------------------------------------------------------
        self.q_meas: dict = {}
        self.target = {a: DEFAULT_TARGET[a].copy() for a in ("left", "right")}
        self.seeded = {"left": False, "right": False}
        self.grip = {"left": 0.0, "right": 0.0}
        self.active = "left"
        self.cmd_vel = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}
        self.cmd_target = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}
        self.step_scale = 1.0
        self.quit = False
        self._last_base_key = self.get_clock().now()
        self._last_loop = self.get_clock().now()

        # --- ROS interface -------------------------------------------------
        self.pose_pub = {
            a: self.create_publisher(PoseStamped, f"/m1/{a}_arm/target_pose", 10)
            for a in ("left", "right")
        }
        self.grip_pub = {
            a: self.create_publisher(Float64, f"/m1/{a}_arm/gripper", 10)
            for a in ("left", "right")
        }
        self.cmd_vel_pub = self.create_publisher(Twist, "/m1/cmd_vel", 10)
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)

    # --- feedback ----------------------------------------------------------
    def _on_joint_states(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.q_meas[name] = float(pos)
        # Once we have the full chain for an arm, snap its (still un-nudged)
        # target onto the current fingertip so nothing jumps on connect.
        if self.reach is None:
            return
        for arm in ("left", "right"):
            if self.seeded[arm]:
                continue
            needed = ARM_JOINTS[arm] + [LIFT_JOINT]
            if all(j in self.q_meas for j in needed):
                try:
                    tip = self.reach.fingertip(arm, self.q_meas)
                    self.target[arm] = np.asarray(tip, dtype=np.float64)
                    self.seeded[arm] = True
                except Exception:  # noqa: BLE001
                    pass

    # --- key handling ------------------------------------------------------
    def handle_key(self, key: str):
        now = self.get_clock().now()

        # Base driving (hold to move; auto-repeat keeps it alive).
        base_keys = {
            "w": ("vx", MAX_LINEAR), "s": ("vx", -MAX_LINEAR),
            "q": ("vy", MAX_STRAFE), "e": ("vy", -MAX_STRAFE),
            "a": ("yaw", MAX_YAW), "d": ("yaw", -MAX_YAW),
        }
        if key in base_keys:
            axis, value = base_keys[key]
            self.cmd_target[axis] = value
            self._last_base_key = now
            return
        if key == " ":
            self.cmd_target = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}
            self.cmd_vel = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}
            return

        # Active-arm Cartesian target nudges.
        step = ARM_STEP * self.step_scale
        arm_keys = {
            "i": (0, step), "k": (0, -step),    # +x / -x
            "j": (1, step), "l": (1, -step),    # +y / -y
            "u": (2, step), "o": (2, -step),    # +z / -z
        }
        if key in arm_keys:
            idx, delta = arm_keys[key]
            self.target[self.active][idx] += delta
            self._clamp_target(self.active)
            return

        # Grippers.
        if key == "]":
            g = self.grip[self.active] + GRIP_STEP
            self.grip[self.active] = _clamp(g, 0.0, 1.0)
            return
        if key == "[":
            g = self.grip[self.active] - GRIP_STEP
            self.grip[self.active] = _clamp(g, 0.0, 1.0)
            return

        # Misc.
        if key == "t":
            self.active = "right" if self.active == "left" else "left"
            return
        if key == "+" or key == "=":
            self.step_scale = _clamp(self.step_scale * 1.5, STEP_SCALE_MIN, STEP_SCALE_MAX)
            return
        if key == "-" or key == "_":
            self.step_scale = _clamp(self.step_scale / 1.5, STEP_SCALE_MIN, STEP_SCALE_MAX)
            return
        if key == "h":
            # Re-seed each arm's target onto its live fingertip (no jump). Only
            # fall back to the fixed default when q_meas/reach are unavailable.
            for arm in ("left", "right"):
                tip = None
                if self.reach is not None:
                    needed = ARM_JOINTS[arm] + [LIFT_JOINT]
                    if all(j in self.q_meas for j in needed):
                        try:
                            tip = self.reach.fingertip(arm, self.q_meas)
                        except Exception:  # noqa: BLE001
                            tip = None
                if tip is not None:
                    self.target[arm] = np.asarray(tip, dtype=np.float64)
                    self.seeded[arm] = True
                else:
                    self.target[arm] = DEFAULT_TARGET[arm].copy()
                    self.seeded[arm] = False
            self.cmd_target = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}
            self.cmd_vel = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}
            return
        if key == ESC or key == "\x03":  # ESC or Ctrl-C
            self.quit = True
            return

    def _clamp_target(self, arm: str):
        t = self.target[arm]
        t[0] = _clamp(t[0], *TARGET_LIMITS["x"])
        t[1] = _clamp(t[1], *TARGET_LIMITS["y"])
        t[2] = _clamp(t[2], *TARGET_LIMITS["z"])

    # --- publish loop ------------------------------------------------------
    def publish(self):
        now = self.get_clock().now()
        dt = (now - self._last_loop).nanoseconds * 1e-9
        self._last_loop = now
        dt = _clamp(dt, 1.0 / 240.0, 0.1)

        # Stop the base if no driving key arrived recently (key released).
        idle = (now - self._last_base_key).nanoseconds * 1e-9
        if idle > BASE_HOLD:
            self.cmd_target = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}

        self.cmd_vel["vx"] = _move_toward(self.cmd_vel["vx"], self.cmd_target["vx"], LINEAR_ACCEL * dt)
        self.cmd_vel["vy"] = _move_toward(self.cmd_vel["vy"], self.cmd_target["vy"], LINEAR_ACCEL * dt)
        self.cmd_vel["yaw"] = _move_toward(self.cmd_vel["yaw"], self.cmd_target["yaw"], YAW_ACCEL * dt)

        twist = Twist()
        twist.linear.x = self.cmd_vel["vx"]
        twist.linear.y = self.cmd_vel["vy"]
        twist.angular.z = self.cmd_vel["yaw"]
        self.cmd_vel_pub.publish(twist)

        stamp = now.to_msg()
        for arm in ("left", "right"):
            # Don't publish a target until the arm is seeded from live feedback
            # (so a URDF-backed session never jumps the arm to DEFAULT on
            # connect). With no URDF the best-effort default is still published.
            if self.seeded[arm] or self.reach is None:
                msg = PoseStamped()
                msg.header.stamp = stamp
                msg.header.frame_id = "base_link"
                msg.pose.position.x = float(self.target[arm][0])
                msg.pose.position.y = float(self.target[arm][1])
                msg.pose.position.z = float(self.target[arm][2])
                msg.pose.orientation.w = 1.0
                self.pose_pub[arm].publish(msg)

            g = Float64()
            g.data = float(self.grip[arm])
            self.grip_pub[arm].publish(g)

    def status_line(self) -> str:
        t = self.target[self.active]
        return (
            f"\r[{self.active.upper():5s}] target x{t[0]:+.2f} y{t[1]:+.2f} z{t[2]:+.2f} m | "
            f"grip {self.grip[self.active]:.2f} | "
            f"base vx{self.cmd_vel['vx']:+.2f} vy{self.cmd_vel['vy']:+.2f} yaw{self.cmd_vel['yaw']:+.2f} | "
            f"step x{self.step_scale:.2f}   "
        )


HELP = __doc__.split("KEYBOARD MAP")[1]


def main(args=None):
    rclpy.init(args=args)
    node = M1Teleop()

    print("M1 teleop -- drives the m1_controller (works in sim and on the real robot).")
    print("KEYBOARD MAP" + HELP)

    if not sys.stdin.isatty():
        node.get_logger().error("m1_teleop needs an interactive terminal (a TTY).")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        return

    settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())  # single-char reads, keep Ctrl-C working
        ticks = 0
        while rclpy.ok() and not node.quit:
            rlist, _, _ = select.select([sys.stdin], [], [], LOOP_PERIOD)
            if rlist:
                key = sys.stdin.read(1)
                if key:
                    node.handle_key(key.lower() if key.isalpha() else key)
            node.publish()
            rclpy.spin_once(node, timeout_sec=0.0)
            ticks += 1
            if ticks % 5 == 0:  # refresh status ~10 Hz
                sys.stdout.write(node.status_line())
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        # Leave the base stopped on exit.
        try:
            node.cmd_vel_pub.publish(Twist())
        except Exception:  # noqa: BLE001
            pass
        print("\nm1_teleop: stopped.")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

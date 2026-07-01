"""M1 whole-body controller (the deployable "brain").

Consumes high-level commands and emits a single ``/m1/joint_command`` that the
simulator (Isaac Sim ROS 2 bridge) or the real robot driver applies:

  Inputs
    /joint_states                 sensor_msgs/JointState   (feedback)
    /m1/left_arm/target_pose      geometry_msgs/PoseStamped (reach target)
    /m1/right_arm/target_pose     geometry_msgs/PoseStamped
    /m1/cmd_vel                   geometry_msgs/Twist       (swerve base)
    /m1/left_arm/gripper          std_msgs/Float64          (0=closed..1=open)
    /m1/right_arm/gripper         std_msgs/Float64

  Output
    /m1/joint_command             sensor_msgs/JointState
        position -> steer / lift / arms / fingers   (wheels get 0, ignored)
        velocity -> wheels                          (others get 0 = hold)

The wheels are velocity-driven (their drive stiffness is 0 in sim), so their
position entry has no effect; every other joint is position-driven with a 0
target velocity so it holds its pose. This keeps the message free of NaNs,
which the Isaac articulation controller handles more reliably.

Given a target pose, the arms + shared lift run a Drake-backed Cartesian reach
toward the requested point. The reach is **position-only**: the gripper
fingertip is driven to the target point and the pose's orientation is ignored.
Targets are interpreted in the robot ``base_link`` frame.

Each tick the solver (Drake InverseKinematics + amortized multi-start) finds the
optimal joint configuration for the active target(s) and leads the measured pose
toward it by a bounded step, so reachable targets converge to sub-millimetre error and an
unreachable one settles at the closest the joints allow. The shared prismatic
lift is recruited automatically to help the arms reach; when both arms have
targets they are solved together and the single lift is resolved as the
least-squares compromise that best serves both grippers. The target poses are
also published as RViz markers for inspection.
"""

from __future__ import annotations

import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, PoseStamped, Twist
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray

from m1_control import swerve
from m1_control.kinematics import (
    ARM_JOINTS,
    LIFT_JOINT,
    ReachController,
    UrdfModel,
)

GRIPPER_OPEN = 0.7854  # finger travel (rad) at fully-open

# Joints we actively command, grouped so we know how to fill the message.
STEER_JOINTS = swerve.STEER_JOINTS
WHEEL_JOINTS = swerve.WHEEL_JOINTS
LEFT_FINGERS = ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]
RIGHT_FINGERS = ["openarm_right_finger_joint1", "openarm_right_finger_joint2"]

POSITION_JOINTS = (
    STEER_JOINTS
    + [LIFT_JOINT]
    + ARM_JOINTS["left"]
    + ARM_JOINTS["right"]
    + LEFT_FINGERS
    + RIGHT_FINGERS
)
ALL_JOINTS = POSITION_JOINTS + WHEEL_JOINTS

# Joints the brain must see in /joint_states before it can seed its command pose
# and begin actuating: the upper body it actually controls (lift + both arms +
# fingers). The 4 base STEER_JOINTS are deliberately EXCLUDED -- the base is
# open-loop swerve (their command is overwritten from cmd_vel every tick and never
# read back), and on the real-hardware / mock arms-only path (use_base:=false)
# nothing publishes them, so requiring them would wedge the controller
# uninitialized forever (the arms would never move).
INIT_JOINTS = (
    [LIFT_JOINT]
    + ARM_JOINTS["left"]
    + ARM_JOINTS["right"]
    + LEFT_FINGERS
    + RIGHT_FINGERS
)


def _default_urdf_path() -> str:
    try:
        share = get_package_share_directory("ranger_air_description")
        return os.path.join(share, "urdf", "ranger_air_description.urdf")
    except Exception:  # noqa: BLE001
        return ""


class M1Controller(Node):
    def __init__(self):
        super().__init__("m1_controller")

        self.declare_parameter("urdf_path", _default_urdf_path())
        self.declare_parameter("control_rate", 60.0)
        self.declare_parameter("command_topic", "/m1/joint_command")

        urdf_path = self.get_parameter("urdf_path").value
        rate = float(self.get_parameter("control_rate").value)
        cmd_topic = self.get_parameter("command_topic").value

        if not urdf_path or not os.path.isfile(urdf_path):
            raise RuntimeError(f"URDF not found at urdf_path='{urdf_path}'")
        with open(urdf_path, "r") as fh:
            urdf_xml = fh.read()
        self.model = UrdfModel.from_string(urdf_xml)
        self.reach = ReachController(self.model)
        self.get_logger().info(f"loaded URDF kinematics from {urdf_path}")

        # --- command / feedback state ---
        self.q_meas: dict = {}                 # joint -> measured position
        self.pos_cmd: dict = {}                # position targets we publish
        self.targets = {"left": None, "right": None}   # reach points (base frame)
        self.grip = {"left": 0.0, "right": 0.0}
        self.cmd_vel = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}
        self.swerve = swerve.SwerveSolver()
        self._initialized = False
        self._last_time = self.get_clock().now()

        # --- ROS interface ---
        self.cmd_pub = self.create_publisher(JointState, cmd_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/m1/target_markers", 10)
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)
        self.create_subscription(
            PoseStamped, "/m1/left_arm/target_pose",
            lambda m: self._on_target("left", m), 10)
        self.create_subscription(
            PoseStamped, "/m1/right_arm/target_pose",
            lambda m: self._on_target("right", m), 10)
        self.create_subscription(Twist, "/m1/cmd_vel", self._on_cmd_vel, 10)
        self.create_subscription(
            Float64, "/m1/left_arm/gripper",
            lambda m: self._on_gripper("left", m), 10)
        self.create_subscription(
            Float64, "/m1/right_arm/gripper",
            lambda m: self._on_gripper("right", m), 10)

        self.timer = self.create_timer(1.0 / rate, self._control_tick)
        self.get_logger().info(
            f"M1 controller up: publishing {cmd_topic} at {rate:.0f} Hz")

    # --- callbacks ---------------------------------------------------------
    def _on_joint_states(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            p = float(pos)
            if not np.isfinite(p):
                continue                       # drop a glitched/garbage encoder sample
            self.q_meas[name] = p
        if not self._initialized and all(j in self.q_meas for j in INIT_JOINTS):
            for j in POSITION_JOINTS:
                self.pos_cmd[j] = self.q_meas.get(j, 0.0)   # steer joints may be absent
            self._initialized = True
            self.get_logger().info("initialized command pose from /joint_states")

    def _on_target(self, arm: str, msg: PoseStamped):
        p = msg.pose.position
        new = np.array([p.x, p.y, p.z], dtype=np.float64)
        if not np.isfinite(new).all():
            # A non-finite target would trip Drake's NaN assertion (SystemExit) or
            # the DLS SVD (LinAlgError) and kill the control brain -- drop it here.
            self.get_logger().warning(
                f"ignoring non-finite {arm} reach target ({p.x}, {p.y}, {p.z})")
            return
        prev = self.targets[arm]
        # Position-only reach: the pose's orientation is ignored. Log only when
        # the target meaningfully changes; operator bridges (web/teleop) re-
        # publish the same target every tick.
        if prev is None or float(np.linalg.norm(new - prev)) > 1e-4:
            self.get_logger().info(
                f"{arm} reach target -> ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})")
        self.targets[arm] = new

    def _on_cmd_vel(self, msg: Twist):
        vx, vy, yaw = msg.linear.x, msg.linear.y, msg.angular.z
        if not np.isfinite([vx, vy, yaw]).all():
            # A garbage (NaN/Inf) velocity is treated as a dead-man: stop the base
            # rather than forwarding it -- swerve would propagate NaN into the wheel
            # velocities, violating the NaN-free /m1/joint_command contract.
            self.get_logger().warning("ignoring non-finite cmd_vel; stopping base")
            self.cmd_vel["vx"] = self.cmd_vel["vy"] = self.cmd_vel["yaw"] = 0.0
            return
        self.cmd_vel["vx"] = vx
        self.cmd_vel["vy"] = vy
        self.cmd_vel["yaw"] = yaw

    def _on_gripper(self, arm: str, msg: Float64):
        self.grip[arm] = max(0.0, min(1.0, float(msg.data)))

    # --- control loop ------------------------------------------------------
    def _control_tick(self):
        now = self.get_clock().now()
        dt = (now - self._last_time).nanoseconds * 1e-9
        self._last_time = now
        dt = min(max(dt, 1.0 / 240.0), 0.05)

        if not self._initialized:
            return

        # 1) Cartesian reach. Each arm with a target is solved to its optimal
        #    joint configuration; the command then leads the measured pose toward
        #    that goal by a bounded step. When both arms reach, they share one
        #    stacked solve so the single lift is the least-squares compromise
        #    that best serves both grippers.
        if any(t is not None for t in self.targets.values()):
            # Position-only reach: each active arm's target is a bare point.
            reach_targets = {arm: self.targets[arm] for arm in ("left", "right")}
            try:
                result = self.reach.solve_step(self.q_meas, reach_targets)
            except Exception as exc:  # noqa: BLE001 - one bad solve must not kill the loop
                self.get_logger().error(
                    f"solve_step failed ({exc}); holding last command",
                    throttle_duration_sec=1.0)
                result = {}
            for jname, q in result.items():
                if jname.startswith("_"):       # meta keys (e.g. "_dist")
                    continue
                self.pos_cmd[jname] = q

        # 2) Grippers (right side mirrored negative, like the URDF mimic).
        for j in LEFT_FINGERS:
            self.pos_cmd[j] = self.grip["left"] * GRIPPER_OPEN
        for j in RIGHT_FINGERS:
            self.pos_cmd[j] = -self.grip["right"] * GRIPPER_OPEN

        # 3) Swerve base -> steer positions + wheel velocities.
        steer_targets, wheel_vel = self.swerve.solve(
            self.cmd_vel["vx"], self.cmd_vel["vy"], self.cmd_vel["yaw"], dt)
        for jn, val in steer_targets.items():
            self.pos_cmd[jn] = val

        # 4) Compose and publish the joint command.
        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = list(ALL_JOINTS)
        positions = []
        velocities = []
        for j in ALL_JOINTS:
            if j in WHEEL_JOINTS:
                positions.append(0.0)                     # ignored (kp=0)
                velocities.append(wheel_vel.get(j, 0.0))
            else:
                positions.append(self.pos_cmd.get(j, 0.0))
                velocities.append(0.0)                    # hold target velocity
        msg.position = positions
        msg.velocity = velocities
        self.cmd_pub.publish(msg)

        # 5) Publish target / fingertip markers for RViz.
        self._publish_markers(now)

    def _publish_markers(self, now):
        arr = MarkerArray()
        colors = {"left": (0.20, 0.55, 1.0), "right": (1.0, 0.45, 0.20)}
        stamp = now.to_msg()
        for arm in ("left", "right"):
            target = self.targets[arm]
            r, g, b = colors[arm]
            # Target sphere.
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp = stamp
            m.ns = f"{arm}_target"
            m.id = 0
            m.type = Marker.SPHERE
            if target is None:
                m.action = Marker.DELETE
            else:
                m.action = Marker.ADD
                m.pose.position.x = float(target[0])
                m.pose.position.y = float(target[1])
                m.pose.position.z = float(target[2])
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = 0.06
                m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.9
            arr.markers.append(m)

            # Text label above the target.
            txt = Marker()
            txt.header.frame_id = "base_link"
            txt.header.stamp = stamp
            txt.ns = f"{arm}_target_label"
            txt.id = 1
            txt.type = Marker.TEXT_VIEW_FACING
            if target is None:
                txt.action = Marker.DELETE
            else:
                txt.action = Marker.ADD
                txt.pose.position.x = float(target[0])
                txt.pose.position.y = float(target[1])
                txt.pose.position.z = float(target[2]) + 0.08
                txt.pose.orientation.w = 1.0
                txt.scale.z = 0.05
                txt.color.r, txt.color.g, txt.color.b, txt.color.a = r, g, b, 1.0
                txt.text = f"{arm} target"
            arr.markers.append(txt)

            # Current fingertip (smaller, translucent) + line to the target.
            tip = None
            try:
                tip = self.reach.fingertip(arm, self.q_meas)
            except Exception:  # noqa: BLE001
                tip = None
            dot = Marker()
            dot.header.frame_id = "base_link"
            dot.header.stamp = stamp
            dot.ns = f"{arm}_fingertip"
            dot.id = 2
            dot.type = Marker.SPHERE
            if tip is None:
                dot.action = Marker.DELETE
            else:
                dot.action = Marker.ADD
                dot.pose.position.x = float(tip[0])
                dot.pose.position.y = float(tip[1])
                dot.pose.position.z = float(tip[2])
                dot.pose.orientation.w = 1.0
                dot.scale.x = dot.scale.y = dot.scale.z = 0.035
                dot.color.r, dot.color.g, dot.color.b, dot.color.a = r, g, b, 0.45
            arr.markers.append(dot)

            line = Marker()
            line.header.frame_id = "base_link"
            line.header.stamp = stamp
            line.ns = f"{arm}_error"
            line.id = 3
            line.type = Marker.LINE_STRIP
            if tip is None or target is None:
                line.action = Marker.DELETE
            else:
                line.action = Marker.ADD
                line.scale.x = 0.008
                line.color.r, line.color.g, line.color.b, line.color.a = r, g, b, 0.6
                p0, p1 = Point(), Point()
                p0.x, p0.y, p0.z = float(tip[0]), float(tip[1]), float(tip[2])
                p1.x, p1.y, p1.z = float(target[0]), float(target[1]), float(target[2])
                line.points = [p0, p1]
            arr.markers.append(line)
        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = M1Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

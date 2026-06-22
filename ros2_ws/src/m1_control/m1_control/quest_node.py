"""Meta Quest (WebXR) teleop for the M1 robot (sim AND real).

Lets you drive the two gripper target poses with the Quest's hand controllers,
using nothing but the headset's built-in browser -- no app install, no
sideloading, no ADB, no Unity. The node serves a single WebXR page; you open it
in the Quest browser, tap "Enter VR", and each controller streams its 6-DoF pose
and buttons back here over the LAN. We map that onto the SAME ``/m1/*`` topics
the web/teleop nodes use, so the identical interface drives the Isaac Sim robot
and the real hardware:

    out  /m1/<arm>/target_pose   geometry_msgs/PoseStamped   (Cartesian reach)
    out  /m1/<arm>/gripper       std_msgs/Float64            (0=closed..1=open)
    out  /m1/cmd_vel             geometry_msgs/Twist         (swerve base)
    in   /joint_states           sensor_msgs/JointState      (feedback + seeding)

Controls (per hand):
    * Grip (squeeze) ......... CLUTCH. Hold to "grab" that arm; while held the
                               gripper target follows your hand's MOTION (relative)
                               and its ORIENTATION mirrors your controller
                               ABSOLUTELY (full 6-DOF). Release to freeze the arm
                               and reposition your hand (like lifting a mouse off
                               the desk).
    * Trigger ................ that arm's gripper, analog 0 (open) .. 1 (closed).
    * Thumbstick CLICK ....... lock / unlock that gripper's ROTATION -- while
                               locked, hand twist no longer rotates the gripper, so
                               you can re-orient your wrist (or translate) freely.
    * Thumbstick (push) ...... LEFT hand drives the base: forward/back to drive
                               forward/back, left/right to strafe (crab). RIGHT
                               hand left/right turns (yaw). The robot model in the
                               headset drives through the room to match.
    * A / X button ........... re-seed that arm's target to its live fingertip AND
                               re-zero its rotation reference -- "home to here".

Why relative position but ABSOLUTE orientation? Relative + clutched *position*
lets you move a small real distance, release, recenter, and continue (standard VR
teleop) without matching the robot's whole reach with your shoulders. *Orientation*
is instead absolute: the gripper mirrors the controller's actual orientation, so
pointing the controller "up 90 deg" puts the gripper up 90 deg -- it does NOT add
90 deg to wherever it already was each grab. The reference is zeroed to the live
pose on first grab / A-X (no snap), and the thumbstick-click LOCK freezes it when
you want to re-orient your wrist without the gripper following.

WebXR requires a SECURE CONTEXT. ``http://<ip>`` is not secure, so this server
speaks HTTPS with a self-signed certificate (auto-generated on first run via
openssl). You accept the browser's "not private" warning once on the Quest.

    ros2 run m1_control m1_quest
    # then on the Quest browser open: https://<this-machine-ip>:8443

Safety: like the web panel, the base is dead-man'd, and arm targets only move
while you hold the clutch. If the headset stops sending updates (took it off,
left the page, network drop) the arms freeze at their last target and the base
zeroes after BASE_HOLD seconds.
"""

from __future__ import annotations

import json
import math
import os
import ssl
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
    mat_to_quat,
    quat_to_mat,
)
from m1_control.swerve import SwerveOdometry

# Vendored web assets (three.js + converted glTF meshes + manifest) served by
# this node so the headset page is fully self-contained over the LAN.
WEB_ASSETS = os.path.join(os.path.dirname(__file__), "web_assets")
_STATIC_TYPES = {
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".bin": "application/octet-stream",
    ".png": "image/png",
    ".html": "text/html; charset=utf-8",
}

# --- Command / safety limits -------------------------------------------------
MAX_LINEAR = 0.5         # forward / reverse speed (m/s)
MAX_STRAFE = 0.4         # sideways crab speed (m/s)
MAX_YAW = 1.0            # yaw rate (rad/s, +ve = left)
BASE_HOLD = 0.5          # s without a base refresh before the base zeros out
STICK_DEADZONE = 0.15    # ignore tiny thumbstick noise
PUBLISH_RATE = 60.0      # Hz the joint command / targets are streamed at
MOTION_SCALE = 1.0       # hand metres -> target metres while clutched (1:1)

# Soft workspace clamp on the target point (base_link frame, m). Height (z) is
# intentionally unbounded; the controller's IK reaches as close as the joint
# limits allow.
TARGET_LIMITS = {
    "x": (-0.9, 0.9),
    "y": (-0.9, 0.9),
    "z": (float("-inf"), float("inf")),
}
DEFAULT_TARGET = {
    "left": [0.40, 0.25, 0.70],
    "right": [0.40, -0.25, 0.70],
}


def _default_urdf_path() -> str:
    try:
        share = get_package_share_directory("ranger_air_description")
        return os.path.join(share, "urdf", "ranger_air_description.urdf")
    except Exception:  # noqa: BLE001
        return ""


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _heading_basis(head_fwd) -> tuple[np.ndarray, np.ndarray]:
    """Build a horizontal WebXR-space basis aligned with where the operator is
    looking, so hand motion maps intuitively to the robot regardless of how the
    Quest's reference-space yaw happens to be oriented.

    WebXR axes are x=right, y=up, -z=forward, but the reference space's yaw is
    arbitrary (set by the headset/guardian pose at session start), so a fixed
    axis map makes "your left" land on the wrong robot axis. Instead we project
    the headset forward direction onto the floor and derive:

        F = headset forward (horizontal)  -> robot +x (away from you = forward)
        L = up x F                        -> robot +y (your left)
        up (WebXR y)                      -> robot +z

    Returns (F, L) as WebXR-space unit vectors. With no head pose we fall back
    to F=-z (facing the reference-space default), which reduces to the old map.
    """
    if head_fwd is None:
        F = np.array([0.0, 0.0, -1.0])
    else:
        f = np.array([float(head_fwd[0]), 0.0, float(head_fwd[2])])
        n = np.linalg.norm(f)
        F = f / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    # L = up x F  with up=(0,1,0)  ->  (Fz, 0, -Fx); this makes F x L = up so the
    # {F, L, up} frame is right-handed like the robot's {x_fwd, y_left, z_up}.
    L = np.array([F[2], 0.0, -F[0]])
    return F, L


def _webxr_to_robot_basis(F: np.ndarray, L: np.ndarray) -> np.ndarray:
    """3x3 matrix C mapping a WebXR-space vector to robot base_link coordinates.

    It is exactly the rotation behind the position map ``robot = [-(d.L), d.F,
    d_up]`` (rows ``-L``, ``F``, ``up``), so a WebXR-space *rotation* ``R`` is
    expressed in the robot frame by the similarity transform ``C @ R @ C.T``.
    With ``F``/``L`` horizontal-orthonormal and ``F x L = up`` this C is a proper
    rotation (det +1), so the operator's hand twist maps to the gripper with the
    same handedness (no mirror-inverted feel).
    """
    return np.array(
        [[-L[0], -L[1], -L[2]],
         [F[0], F[1], F[2]],
         [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )


def _ensure_cert(node: "M1QuestNode") -> tuple[str, str]:
    """Return (certfile, keyfile), generating a self-signed pair if needed."""
    cert = node.get_parameter("certfile").value
    key = node.get_parameter("keyfile").value
    if cert and key and os.path.isfile(cert) and os.path.isfile(key):
        return cert, key

    cache = os.path.expanduser("~/.cache/m1_quest")
    os.makedirs(cache, exist_ok=True)
    cert = os.path.join(cache, "cert.pem")
    key = os.path.join(cache, "key.pem")
    if os.path.isfile(cert) and os.path.isfile(key):
        return cert, key

    node.get_logger().info(f"generating a self-signed TLS cert in {cache} (openssl)")
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", key, "-out", cert, "-days", "3650",
            "-subj", "/CN=m1-quest",
            # SANs so the cert is valid for any LAN IP/hostname the Quest uses.
            "-addext", "subjectAltName=DNS:localhost,IP:0.0.0.0,IP:127.0.0.1",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return cert, key


class M1QuestNode(Node):
    """Bridges Quest controller poses (over WebXR) to the m1_controller."""

    def __init__(self):
        super().__init__("m1_quest")
        self.declare_parameter("urdf_path", _default_urdf_path())
        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 8443)
        self.declare_parameter("certfile", "")
        self.declare_parameter("keyfile", "")
        self.declare_parameter("motion_scale", MOTION_SCALE)
        self.declare_parameter("enable_base", True)

        urdf_path = self.get_parameter("urdf_path").value
        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)
        self.motion_scale = float(self.get_parameter("motion_scale").value)
        self.enable_base = bool(self.get_parameter("enable_base").value)

        self.reach = None
        if urdf_path and os.path.isfile(urdf_path):
            try:
                with open(urdf_path, "r") as fh:
                    self.reach = ReachController(UrdfModel.from_string(fh.read()))
                self.get_logger().info(f"loaded URDF kinematics from {urdf_path}")
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"URDF FK unavailable ({exc}); using default targets")

        # --- shared state (guarded by _lock) -------------------------------
        self._lock = threading.Lock()
        self.q_meas: dict = {}
        self.target = {a: list(DEFAULT_TARGET[a]) for a in ("left", "right")}
        # Per-arm target gripper orientation (base frame) as a quaternion
        # [x,y,z,w]. Seeded to the live gripper orientation so the arm holds its
        # pose on connect; an identity quaternion (the un-seeded default) reads
        # as position-only at the controller, so nothing rotates until seeded.
        self.target_quat = {a: [0.0, 0.0, 0.0, 1.0] for a in ("left", "right")}
        self.seeded = {"left": False, "right": False}
        self.grip = {"left": 0.0, "right": 0.0}
        self.cmd_vel = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}
        # Clutch bookkeeping per arm: where the hand was and where the target
        # was at the instant the grip was squeezed.
        self.clutch = {"left": False, "right": False}
        self.clutch_hand0 = {"left": None, "right": None}
        self.clutch_target0 = {"left": None, "right": None}
        # ABSOLUTE gripper orientation. The target orientation is a fixed function
        # of the controller's CURRENT orientation, not a twist accumulated each
        # grab: ``R_target = C @ hand_R @ C.T @ ori_align``. The heading basis ``C``
        # and the alignment ``ori_align`` are captured once per arm (preserving the
        # live gripper orientation, so nothing snaps) and re-zeroed only on an
        # explicit recenter (A/X) or when the rotation lock is released. So holding
        # the controller a given way always commands the same gripper orientation
        # -- controller up 90 deg -> gripper up 90 deg -- instead of adding 90 deg
        # to wherever it already was.
        self.ori_C = {"left": None, "right": None}
        self.ori_align = {"left": None, "right": None}
        # Rotation lock (thumbstick click, per arm): while set, hand twist no longer
        # moves the gripper orientation, so the wrist can be re-oriented / the hand
        # translated without rotating the gripper. Releasing it re-zeros ori_align
        # to the held orientation so tracking resumes from there without a jump.
        self.ori_locked = {"left": False, "right": False}
        self.last_lock = {"left": False, "right": False}  # lock-toggle edge detect
        # Heading-relative basis (WebXR-space F/L vectors) captured at the
        # instant the clutch is squeezed, so the position mapping stays fixed for
        # the duration of that grab even if the operator turns their head.
        self.clutch_F = {"left": None, "right": None}
        self.clutch_L = {"left": None, "right": None}
        self.last_btn = {"left": False, "right": False}  # A/X edge detect
        self._last_base_cmd = 0.0
        self._last_update = 0.0
        # Dead-reckoned base pose (odom frame), integrated from the commanded
        # body velocity each tick so the headset model actually drives through
        # the room as the operator commands the swerve. Reset to the origin
        # whenever the hologram is (re)placed so the robot sits at the anchor.
        self.odom = SwerveOdometry()
        self._last_tick = 0.0

        # --- ROS interface --------------------------------------------------
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
        self.create_timer(1.0 / PUBLISH_RATE, self._tick)

    # --- feedback ----------------------------------------------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_joint_states(self, msg: JointState):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                self.q_meas[name] = float(pos)
            if self.reach is None:
                return
            for arm in ("left", "right"):
                # Seed each arm's target onto its current fingertip exactly once
                # so the arm holds still on connect instead of snapping.
                if self.seeded[arm]:
                    continue
                if all(j in self.q_meas for j in ARM_JOINTS[arm] + [LIFT_JOINT]):
                    try:
                        tip, R = self.reach.gripper_pose(arm, self.q_meas)
                        self.target[arm] = [float(tip[0]), float(tip[1]), float(tip[2])]
                        self.target_quat[arm] = [float(c) for c in mat_to_quat(R)]
                        self.seeded[arm] = True
                    except Exception:  # noqa: BLE001
                        pass

    def _reseed(self, arm: str):
        if self.reach is None:
            return
        if all(j in self.q_meas for j in ARM_JOINTS[arm] + [LIFT_JOINT]):
            try:
                tip, R = self.reach.gripper_pose(arm, self.q_meas)
                self.target[arm] = [float(tip[0]), float(tip[1]), float(tip[2])]
                self.target_quat[arm] = [float(c) for c in mat_to_quat(R)]
                self.seeded[arm] = True
            except Exception:  # noqa: BLE001
                pass

    def _calibrate_ori(self, arm: str, hand_R, head_fwd):
        """Zero the absolute-orientation map for ``arm`` to the current pose.

        Captures the heading basis ``C`` and an alignment so that, right now, the
        controller's orientation maps to the gripper target's CURRENT orientation
        -- nothing jumps at calibration. Afterwards the target orientation is the
        fixed function ``C @ hand_R @ C.T @ ori_align`` of the live controller
        orientation, so a given controller pose always commands the same gripper
        orientation (absolute, not accumulated). Re-called only on an explicit
        recenter (A/X) or when the rotation lock is released, so it never drifts
        grab-to-grab. ``ori_align = C @ hand_R^T @ C^T @ G`` inverts the map at the
        current controller pose so it yields the current target orientation ``G``.
        """
        if hand_R is None:
            return
        F, L = _heading_basis(head_fwd)
        C = _webxr_to_robot_basis(F, L)
        G = quat_to_mat(self.target_quat[arm])
        self.ori_C[arm] = C
        self.ori_align[arm] = C @ hand_R.T @ C.T @ G

    # --- WebXR frame ingest (called from the HTTP handler thread) -----------
    def on_xr_frame(self, data: dict):
        """Apply one batch of controller states posted by the headset."""
        with self._lock:
            self._last_update = self._now()
            controllers = data.get("controllers", {})
            head_fwd = data.get("head")  # headset forward vector (WebXR space)
            # The page raises ``place`` when it (re)anchors the hologram (first
            # frame or a B/Y recenter). Zero the dead-reckoned pose so the robot
            # snaps back to the anchor in front of the operator instead of
            # drifting off by however far it had already been driven.
            if data.get("place"):
                self.odom.reset()
            # Physical controllers are cross-mapped: the LEFT controller drives
            # the RIGHT arm and vice versa.
            controller_for_arm = {"left": "right", "right": "left"}
            for arm in ("left", "right"):
                c = controllers.get(controller_for_arm[arm])
                if not c or not c.get("valid"):
                    # Lost tracking for this hand: drop the clutch so we don't
                    # jump when it returns.
                    self.clutch[arm] = False
                    continue

                hand = np.array(c.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
                hand_quat = c.get("quat")
                hand_R = quat_to_mat(hand_quat) if hand_quat else None
                squeeze = bool(c.get("squeeze"))
                trigger = _clamp(float(c.get("trigger", 0.0)), 0.0, 1.0)
                button = bool(c.get("button"))
                lock = bool(c.get("lock"))

                # A/X edge: re-seed this arm's target to the live fingertip AND
                # re-zero the absolute-orientation map there ("recenter rotation").
                if button and not self.last_btn[arm]:
                    self._reseed(arm)
                    self._calibrate_ori(arm, hand_R, head_fwd)
                    self.clutch[arm] = False
                self.last_btn[arm] = button

                # Rotation-lock toggle (thumbstick click): freeze / unfreeze this
                # gripper's target orientation so the wrist can be re-oriented (or
                # the hand translated) without rotating the gripper. On UNLOCK,
                # re-zero the map so hand twist resumes from the held orientation
                # without a jump.
                if lock and not self.last_lock[arm]:
                    self.ori_locked[arm] = not self.ori_locked[arm]
                    if not self.ori_locked[arm]:
                        self._calibrate_ori(arm, hand_R, head_fwd)
                self.last_lock[arm] = lock

                # Clutch logic: target follows hand delta only while squeezed. The
                # delta (raw WebXR metres) is projected into the operator's heading
                # frame captured at the squeeze, so "forward/left/up relative to
                # where you're looking" map to robot x/y/z. Orientation is separate
                # and ABSOLUTE (below), not a per-grab delta.
                if squeeze and not self.clutch[arm]:
                    self.clutch[arm] = True
                    self.clutch_hand0[arm] = hand
                    self.clutch_target0[arm] = np.array(self.target[arm])
                    F, L = _heading_basis(head_fwd)
                    self.clutch_F[arm], self.clutch_L[arm] = F, L
                    self.seeded[arm] = True
                    # First grab for this arm: zero the absolute-orientation map to
                    # the live pose so the gripper doesn't snap when tracking begins.
                    if self.ori_align[arm] is None:
                        self._calibrate_ori(arm, hand_R, head_fwd)
                elif squeeze and self.clutch[arm]:
                    d = (hand - self.clutch_hand0[arm]) * self.motion_scale
                    F, L = self.clutch_F[arm], self.clutch_L[arm]
                    robot_delta = np.array([
                        float(-(d @ L)),  # right    -> +x  (x/y swapped, x reversed)
                        float(d @ F),     # forward  -> +y  (x/y swapped)
                        float(d[1]),      # up       -> +z
                    ])
                    self._set_target(arm, self.clutch_target0[arm] + robot_delta)
                    # ABSOLUTE orientation: the gripper target orientation is a
                    # fixed function of the controller's CURRENT orientation (see
                    # _calibrate_ori), so it mirrors the controller rather than
                    # accumulating a twist each grab. The lock freezes it.
                    if (hand_R is not None and self.ori_C[arm] is not None
                            and not self.ori_locked[arm]):
                        C = self.ori_C[arm]
                        R_tgt = C @ hand_R @ C.T @ self.ori_align[arm]
                        self.target_quat[arm] = [float(v) for v in mat_to_quat(R_tgt)]
                elif not squeeze:
                    self.clutch[arm] = False

                self.grip[arm] = trigger

            # Base from the thumbsticks:
            #   LEFT stick  forward/back -> drive forward/back (vx)
            #   LEFT stick  left/right   -> strafe (crab) left/right (vy)
            #   RIGHT stick left/right   -> turn (yaw)
            # Driven every frame (so centring the stick stops the base at once);
            # the dead-man (BASE_HOLD) then only guards a lost connection.
            if self.enable_base:
                lx, ly = self._stick(controllers.get("left") or {})
                rx, _ = self._stick(controllers.get("right") or {})
                self.cmd_vel = {
                    # gamepad stick y is +down -> push up (negative) = forward
                    "vx": _clamp(-ly * MAX_LINEAR, -MAX_LINEAR, MAX_LINEAR),
                    # stick x is +right -> push left (negative) = +y (robot left)
                    "vy": _clamp(-lx * MAX_STRAFE, -MAX_STRAFE, MAX_STRAFE),
                    # push right (positive) = turn right = clockwise = -yaw
                    "yaw": _clamp(-rx * MAX_YAW, -MAX_YAW, MAX_YAW),
                }
                self._last_base_cmd = self._now()
            viz = self._viz_locked()
        return {"ok": True, "viz": viz}

    def _viz_locked(self) -> dict:
        """Geometry for the headset's 3D overlay (assumes ``_lock`` held).

        For each arm we send the target point, the live fingertip, the
        base->tip skeleton points (FK of the *measured* joints), and the
        fingertip->target distance so the page can colour the target by how
        well the arm is reaching (green=on target, red=out of reach). All
        points are in the robot ``base_link`` frame; the page anchors that
        frame to a fixed spot in the room, then offsets it by ``base`` (the
        dead-reckoned swerve pose) so the model drives around as commanded.
        """
        x, y, theta = self.odom.pose
        viz = {
            "frame": "base_link", "arms": {}, "links": {},
            # base_link pose in the odom frame (x fwd, y left, +yaw = CCW about
            # +z). The page applies this to the robot group so the whole model
            # translates/turns through the room as the base drives.
            "base": {
                "p": [round(x, 4), round(y, 4), 0.0],
                "q": [round(c, 5) for c in self.odom.quaternion()],
            },
        }
        have_js = bool(self.q_meas)
        if self.reach is not None and have_js:
            # Full-robot link poses (base_link frame) so the page can drive each
            # mesh; cheap matrix walk over the URDF tree.
            try:
                Ts = self.reach.model.link_transforms(self.q_meas)
                for name, T in Ts.items():
                    viz["links"][name] = {
                        "p": [round(float(T[i, 3]), 4) for i in range(3)],
                        "q": [round(float(c), 5) for c in mat_to_quat(T[:3, :3])],
                    }
            except Exception:  # noqa: BLE001
                pass
        for arm in ("left", "right"):
            a = {
                "target": [round(float(v), 4) for v in self.target[arm]],
                # Target gripper orientation (quaternion x,y,z,w) so the page can
                # draw the target's rotation, not just its point.
                "target_quat": [round(float(v), 5) for v in self.target_quat[arm]],
            }
            if (
                self.reach is not None
                and have_js
                and all(j in self.q_meas for j in ARM_JOINTS[arm] + [LIFT_JOINT])
            ):
                try:
                    pts = self.reach.chains[arm].link_points(self.q_meas)
                    tip = pts[-1]
                    _, R_tip = self.reach.gripper_pose(arm, self.q_meas)
                    a["points"] = [[round(float(c), 4) for c in p] for p in pts]
                    a["tip"] = [round(float(c), 4) for c in tip]
                    # Live gripper orientation -> a second (current-pose) triad,
                    # so the operator can see how close the gripper is to the
                    # commanded orientation.
                    a["tip_quat"] = [round(float(c), 5) for c in mat_to_quat(R_tip)]
                    a["dist"] = round(
                        float(np.linalg.norm(np.asarray(self.target[arm]) - tip)), 4
                    )
                except Exception:  # noqa: BLE001
                    pass
            viz["arms"][arm] = a
        return viz

    @staticmethod
    def _deadzone(v: float) -> float:
        """Smooth radial deadzone: ignore tiny noise, then rescale so the live
        range maps 0..1 with no jump at the deadzone edge (no sudden lurch as
        the stick crosses the threshold)."""
        a = abs(v)
        if a < STICK_DEADZONE:
            return 0.0
        scaled = (a - STICK_DEADZONE) / (1.0 - STICK_DEADZONE)
        return math.copysign(min(scaled, 1.0), v)

    @classmethod
    def _stick(cls, c: dict) -> tuple[float, float]:
        s = c.get("stick", [0.0, 0.0]) if c else [0.0, 0.0]
        x = float(s[0]) if len(s) > 0 else 0.0
        y = float(s[1]) if len(s) > 1 else 0.0
        return cls._deadzone(x), cls._deadzone(y)

    def _set_target(self, arm: str, xyz):
        self.target[arm] = [
            _clamp(float(xyz[0]), *TARGET_LIMITS["x"]),
            _clamp(float(xyz[1]), *TARGET_LIMITS["y"]),
            _clamp(float(xyz[2]), *TARGET_LIMITS["z"]),
        ]

    # --- publish loop ------------------------------------------------------
    def _tick(self):
        with self._lock:
            now = self._now()
            if now - self._last_base_cmd > BASE_HOLD:
                self.cmd_vel = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}
            cmd = dict(self.cmd_vel)
            # Dead-reckon the base pose from the command we are about to publish,
            # so the headset model drives through the room in lockstep with the
            # swerve. dt is clamped to the same band the controller uses.
            dt = now - self._last_tick if self._last_tick else 0.0
            self._last_tick = now
            self.odom.update(cmd["vx"], cmd["vy"], cmd["yaw"],
                             min(max(dt, 0.0), 0.05))
            targets = {a: list(self.target[a]) for a in ("left", "right")}
            quats = {a: list(self.target_quat[a]) for a in ("left", "right")}
            grip = dict(self.grip)

        tw = Twist()
        tw.linear.x = cmd["vx"]
        tw.linear.y = cmd["vy"]
        tw.angular.z = cmd["yaw"]
        self.cmd_vel_pub.publish(tw)

        stamp = self.get_clock().now().to_msg()
        for arm in ("left", "right"):
            ps = PoseStamped()
            ps.header.stamp = stamp
            ps.header.frame_id = "base_link"
            ps.pose.position.x = targets[arm][0]
            ps.pose.position.y = targets[arm][1]
            ps.pose.position.z = targets[arm][2]
            q = quats[arm]
            ps.pose.orientation.x = q[0]
            ps.pose.orientation.y = q[1]
            ps.pose.orientation.z = q[2]
            ps.pose.orientation.w = q[3]
            self.pose_pub[arm].publish(ps)

            g = Float64()
            g.data = grip[arm]
            self.grip_pub[arm].publish(g)

    # --- API used by the HTTP handler (thread-safe) ------------------------
    def snapshot(self) -> dict:
        with self._lock:
            now = self._now()
            connected = (now - getattr(self, "_last_update", 0.0)) < 1.0
            js_live = bool(self.q_meas)
            out = {"xr_connected": connected, "joint_states": js_live, "arms": {}}
            for arm in ("left", "right"):
                out["arms"][arm] = {
                    "target": [round(v, 3) for v in self.target[arm]],
                    "grip": round(self.grip[arm], 3),
                    "clutch": self.clutch[arm],
                    "rot_locked": self.ori_locked[arm],
                }
            return out


def _make_handler(node: M1QuestNode):
    class Handler(BaseHTTPRequestHandler):
        # Keep-alive so the headset's high-rate POSTs reuse one TLS connection
        # instead of a fresh handshake every XR frame.
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # silence per-request logging
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _send_static(self, rel: str):
            """Serve a vendored asset (three.js / glTF / manifest) safely."""
            rel = rel.split("?", 1)[0].lstrip("/")
            full = os.path.normpath(os.path.join(WEB_ASSETS, rel))
            if not full.startswith(os.path.realpath(WEB_ASSETS) + os.sep) and \
               not full.startswith(WEB_ASSETS + os.sep):
                self._send(403, json.dumps({"error": "forbidden"}))
                return
            if not os.path.isfile(full):
                self._send(404, json.dumps({"error": "not found"}))
                return
            ctype = _STATIC_TYPES.get(os.path.splitext(full)[1].lower(),
                                      "application/octet-stream")
            try:
                with open(full, "rb") as fh:
                    self._send(200, fh.read(), ctype)
            except OSError:
                self._send(404, json.dumps({"error": "not found"}))

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, INDEX_HTML, "text/html; charset=utf-8")
            elif path == "/api/state":
                self._send(200, json.dumps(node.snapshot()))
            elif path == "/manifest.json" or path.startswith("/vendor/") \
                    or path.startswith("/meshes/"):
                self._send_static(path)
            else:
                self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            if self.path != "/api/xr":
                self._send(404, json.dumps({"error": "not found"}))
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                self._send(200, json.dumps(node.on_xr_frame(payload)))
            except Exception as exc:  # noqa: BLE001
                self._send(400, json.dumps({"ok": False, "error": str(exc)}))

    return Handler


class _Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _bind_server(node: M1QuestNode, handler, ctx, tries: int = 10):
    last_exc = None
    for port in range(node.port, node.port + tries):
        try:
            server = _Server((node.host, port), handler)
            server.socket = ctx.wrap_socket(server.socket, server_side=True)
            node.port = port
            return server
        except OSError as exc:  # noqa: PERF203
            last_exc = exc
            node.get_logger().warn(f"port {port} busy, trying {port + 1}…")
    raise last_exc


def _lan_ips() -> list[str]:
    ips = []
    try:
        out = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, check=False
        ).stdout
        ips = [ip for ip in out.split() if ":" not in ip]
    except Exception:  # noqa: BLE001
        pass
    return ips


def main(args=None):
    rclpy.init(args=args)
    node = M1QuestNode()

    try:
        certfile, keyfile = _ensure_cert(node)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        node.get_logger().error(
            f"could not create a TLS certificate ({exc}). Install openssl, or "
            "pass an existing cert/key: "
            "`ros2 run m1_control m1_quest --ros-args -p certfile:=cert.pem -p keyfile:=key.pem`")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        return

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=certfile, keyfile=keyfile)

    try:
        server = _bind_server(node, _make_handler(node), ctx)
    except OSError as exc:
        node.get_logger().error(
            f"could not open a port near {node.port} ({exc}). Another m1_quest "
            f"may be running (`pkill -f m1_quest`) or pick another port "
            "(`--ros-args -p port:=9443`).")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        return

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    urls = [f"https://{ip}:{node.port}" for ip in _lan_ips()] or [
        f"https://<this-machine-ip>:{node.port}"
    ]
    node.get_logger().info(
        "M1 Quest WebXR teleop running. On the Quest browser open one of:\n    "
        + "\n    ".join(urls)
        + "\n  (accept the self-signed certificate warning, then tap 'Enter VR')."
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        try:
            node.cmd_vel_pub.publish(Twist())  # leave the base stopped
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# ---------------------------------------------------------------------------
# WebXR page (served at "/"). Self-contained: the only third-party code is
# three.js, vendored locally under web_assets/ and served by this node (no CDN,
# no build step, no install on the headset). Requests an immersive-ar session
# (passthrough so you still see your room), falling back to immersive-vr.
#
# It renders an RViz-like hologram of the robot: each link's converted glTF mesh
# is posed every frame from the per-link FK transforms streamed in the /api/xr
# response, with a coloured target sphere per arm (green=reaching, red=out of
# reach). Each XR frame it also reads both controllers' grip poses + buttons and
# POSTs them to /api/xr (one request in flight at a time).
# ---------------------------------------------------------------------------
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>M1 Quest Teleop</title>
<style>
  :root{ --bg:#f0eee6; --panel:#f6f4ec; --line:#dcd7c8; --ink:#181613;
         --muted:#73706a; --accent:#c96442; --good:#4f7a4a; --bad:#b4432f;
         --serif:'Iowan Old Style',Georgia,'Times New Roman',serif;
         --sans:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif; }
  *{box-sizing:border-box;}
  body{margin:0;font-family:var(--sans);background:var(--bg);color:var(--ink);}
  header{padding:18px 24px;border-bottom:1px solid var(--line);}
  h1{font-family:var(--serif);font-size:22px;margin:0;font-weight:600;}
  main{max-width:760px;margin:0 auto;padding:22px 24px;}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:8px;
        padding:18px 20px;margin-bottom:18px;}
  button.big{font-size:20px;padding:16px 26px;border-radius:8px;cursor:pointer;
        background:var(--accent);color:#fff;border:none;font-family:var(--sans);}
  button.big:disabled{background:#bdb8a8;cursor:not-allowed;}
  .status{display:flex;gap:8px;align-items:center;font-size:14px;color:var(--muted);}
  .dot{width:10px;height:10px;border-radius:50%;background:var(--bad);}
  .dot.on{background:var(--good);}
  table{width:100%;border-collapse:collapse;font-size:14px;}
  td{padding:4px 6px;border-top:1px solid var(--line);font-variant-numeric:tabular-nums;}
  td:first-child{color:var(--muted);text-transform:capitalize;}
  kbd{background:#fbfaf5;border:1px solid var(--line);border-bottom-width:2px;
      border-radius:5px;padding:1px 6px;font-size:13px;}
  .muted{color:var(--muted);font-size:13px;line-height:1.6;}
  #log{font-family:ui-monospace,monospace;font-size:12px;color:var(--muted);
       white-space:pre-wrap;margin-top:10px;}
</style>
<script type="importmap">
{ "imports": {
    "three": "/vendor/three.module.js",
    "three/addons/": "/vendor/addons/"
} }
</script>
</head>
<body>
<header><h1>M1 — Quest controller teleop</h1></header>
<main>
  <div class="card">
    <button id="enter" class="big" disabled>Checking WebXR…</button>
    <p class="muted" id="xrhint" style="margin-top:14px"></p>
  </div>

  <div class="card">
    <div class="status">
      <span class="dot" id="jsDot"></span>
      <span id="jsText">robot feedback: unknown</span>
    </div>
    <table>
      <tr><td>left target</td><td id="lt">–</td><td id="lg">grip –</td><td id="lc"></td></tr>
      <tr><td>right target</td><td id="rt">–</td><td id="rg">grip –</td><td id="rc"></td></tr>
    </table>
  </div>

  <div class="card">
    <h3 style="margin:0 0 8px;font-family:var(--serif)">Controls</h3>
    <p class="muted">
      <kbd>Grip</kbd> (squeeze) — hold to "grab" that arm; the gripper follows your
      hand's motion, and its rotation <em>mirrors your controller absolutely</em>
      (point the controller a way → the gripper points that way). Release to
      reposition your hand.<br/>
      <kbd>Trigger</kbd> — that arm's gripper (squeeze to close).<br/>
      <kbd>Stick click</kbd> — lock / unlock that gripper's rotation (twist your
      wrist freely while locked).<br/>
      <kbd>Left stick</kbd> — drive: forward/back = drive fwd/back, left/right = strafe.<br/>
      <kbd>Right stick</kbd> — left/right = turn (yaw).<br/>
      <kbd>A</kbd>/<kbd>X</kbd> — re-home that arm's target <em>and</em> rotation to where it is now.<br/>
      <kbd>B</kbd>/<kbd>Y</kbd> — recenter the robot hologram in front of you.
    </p>
    <p class="muted">
      In the headset you'll see a 3D model of the robot (its real meshes) drawn
      over your room in passthrough, posed live from the robot's joints, plus a
      coloured sphere at each Cartesian target with an <b>RGB orientation triad</b>
      showing the target gripper rotation (a smaller, fainter triad tracks the
      live gripper so you can see how well it's aligned). The target turns
      <span style="color:var(--good)">green</span> when the arm reaches it and
      <span style="color:var(--bad)">red</span> when the pose is out of reach,
      so you can see impossible goals immediately. (First load streams ~4 MB of
      meshes; it's cached afterward.)
    </p>
  </div>
  <div id="log"></div>
</main>

<script>
window.log = (m)=>{ const el=document.getElementById("log");
  el.textContent=(new Date().toLocaleTimeString()+"  "+m+"\n"+el.textContent).slice(0,2000); };

/* ---- live state from the robot (2D page poll) ---- */
async function poll(){
  try{
    const s=await (await fetch("/api/state")).json();
    const d=document.getElementById("jsDot"), t=document.getElementById("jsText");
    if(s.joint_states){ d.classList.add("on"); t.textContent="robot feedback: live"; }
    else { d.classList.remove("on"); t.textContent="robot feedback: none — start sim/robot"; }
    const f=(a)=>{ const arm=s.arms[a]; if(!arm) return;
      document.getElementById(a[0]+"t").textContent="["+arm.target.join(", ")+"]";
      document.getElementById(a[0]+"g").textContent="grip "+arm.grip.toFixed(2);
      document.getElementById(a[0]+"c").textContent=
        (arm.clutch?"● clutch":"")+(arm.rot_locked?"  🔒 rot":""); };
    f("left"); f("right");
  }catch(e){}
}
setInterval(poll,300); poll();

</script>

<script type="module">
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const enterBtn=document.getElementById("enter");
const hint=document.getElementById("xrhint");

let renderer=null, scene=null, camera=null, robotRoot=null, robotBase=null;
let xrSession=null, refSpace=null, refIsFloor=false, robotPlaced=false;
let lastViz=null, inFlight=false, meshesLoaded=false;
// Per-link bookkeeping so a mesh that fails to load is *visible as a gap* and
// gets a wireframe fallback for that arm, instead of silently vanishing.
let armLinks={left:[], right:[]};
const armComplete={left:false, right:false};
const linkNodes={}, targetMesh={}, tipMesh={}, errLine={}, wire={};
const targetAxes={}, tipAxes={};
const recenterPrev={left:false, right:false};
// Sticky "(re)anchor the robot" signal. Set when we place/recenter, sent to the
// node so it zeroes the dead-reckoned base pose, and cleared only once a POST
// that carried it actually reaches the server -- so a recenter that lands while
// a POST is in flight is NOT dropped (it rides the next POST instead).
let pendingPlace=false;

// Robot materials. The converted meshes now carry per-solid VERTEX COLOURS
// baked from the real CAD materials (white body, black tyres/trim, red accents),
// so links with colour data use a vertex-colour material to render the real
// part; links without (e.g. plain STL solids) fall back to a neutral grey. Both
// are matte and double-sided so heavy decimation can't leave see-through holes.
const ROBOT_MAT=new THREE.MeshStandardMaterial({
  color:0xc7ccd4, metalness:0.15, roughness:0.62, side:THREE.DoubleSide,
});
const ROBOT_MAT_VC=new THREE.MeshStandardMaterial({
  color:0xffffff, vertexColors:true, metalness:0.12, roughness:0.65,
  side:THREE.DoubleSide,
});

async function chooseMode(){
  if(!navigator.xr) return null;
  if(await navigator.xr.isSessionSupported("immersive-ar")) return "immersive-ar";
  if(await navigator.xr.isSessionSupported("immersive-vr")) return "immersive-vr";
  return null;
}

function reachColor(d){
  if(d<0.02) return new THREE.Color(0x4cd964);   // reaching
  if(d<0.08) return new THREE.Color(0xf2b33a);   // marginal
  return new THREE.Color(0xe64d40);              // out of reach
}

/* ---- three.js scene: robot meshes + target markers, drawn over passthrough ---- */
function initThree(){
  renderer=new THREE.WebGLRenderer({antialias:true, alpha:true});
  renderer.setClearAlpha(0);                  // transparent -> passthrough shows through
  renderer.xr.enabled=true;
  renderer.domElement.style.display="none";   // only used inside the XR session
  document.body.appendChild(renderer.domElement);

  scene=new THREE.Scene();
  camera=new THREE.PerspectiveCamera(70, 1, 0.02, 50);
  // Soft, even lighting so a matte robot reads well over passthrough: sky/ground
  // hemisphere fill + a touch of ambient + two opposed directionals for shape.
  scene.add(new THREE.HemisphereLight(0xffffff, 0x555a60, 1.1));
  scene.add(new THREE.AmbientLight(0xffffff, 0.35));
  const key=new THREE.DirectionalLight(0xffffff, 1.1); key.position.set(1.5,3,1.5);
  scene.add(key);
  const fill=new THREE.DirectionalLight(0xffffff, 0.5); fill.position.set(-1.5,1,-1.5);
  scene.add(fill);

  // Room anchor: its matrix maps robot axes (x fwd, y left, z up) -> XR world
  // and pins the odom origin to a fixed spot in the room.
  robotRoot=new THREE.Group(); robotRoot.matrixAutoUpdate=false; scene.add(robotRoot);
  // base_link group: lives inside the anchor and carries the dead-reckoned
  // swerve pose, so the whole robot (meshes, targets, wires) drives through the
  // room as the base is commanded. Everything below hangs off robotBase.
  robotBase=new THREE.Group(); robotRoot.add(robotBase);

  for(const arm of ["left","right"]){
    const tg=new THREE.Mesh(new THREE.SphereGeometry(0.05,24,16),
      new THREE.MeshStandardMaterial({color:0x4cd964, transparent:true, opacity:0.42}));
    tg.visible=false; robotBase.add(tg); targetMesh[arm]=tg;
    const tp=new THREE.Mesh(new THREE.SphereGeometry(0.02,16,12),
      new THREE.MeshStandardMaterial({color:0x33ccee, emissive:0x114455}));
    tp.visible=false; robotBase.add(tp); tipMesh[arm]=tp;
    const g=new THREE.BufferGeometry().setFromPoints(
      [new THREE.Vector3(), new THREE.Vector3()]);
    const ln=new THREE.Line(g, new THREE.LineBasicMaterial({color:0x4cd964}));
    ln.visible=false; robotBase.add(ln); errLine[arm]=ln;
    // Orientation triads (RGB = gripper X/Y/Z). The larger one is the TARGET
    // pose's rotation; the smaller one tracks the live gripper, so the operator
    // can see how close the gripper's rotation is to what was commanded. Drawn
    // on top (depthTest off) so they read clearly over the robot mesh. On
    // robotBase so they drive with the robot through the room.
    const tax=new THREE.AxesHelper(0.15);
    tax.material.depthTest=false; tax.material.linewidth=2;
    tax.renderOrder=5; tax.visible=false; robotBase.add(tax); targetAxes[arm]=tax;
    const pax=new THREE.AxesHelper(0.10);
    pax.material.transparent=true; pax.material.opacity=0.6;
    pax.renderOrder=4; pax.visible=false; robotBase.add(pax); tipAxes[arm]=pax;
  }
}

/* ---- fetch one mesh with a few retries (a single dropped request used to
       silently leave that link missing with no fallback) ---- */
async function loadMeshWithRetry(loader, url, tries){
  let lastErr;
  for(let i=0;i<tries;i++){
    try{ return await loader.loadAsync(url); }
    catch(e){ lastErr=e; await new Promise(r=>setTimeout(r, 150*(i+1))); }
  }
  throw lastErr;
}

/* ---- load the converted robot meshes once (cached by the browser after) ----
   Unique meshes are fetched in parallel WITH RETRY, then each link is instanced
   from the cache. Anything that still fails is reported by name and left to the
   per-arm wireframe fallback, so a missing mesh is obvious instead of a silent
   hole in the model. */
async function loadRobot(){
  let manifest;
  try{ manifest=await (await fetch("/manifest.json")).json(); }
  catch(e){ log("no mesh manifest — using wireframe fallback"); return; }
  const entries=manifest.links||[];
  // Which manifest links belong to each arm chain (for the per-arm wireframe
  // fallback if any of them fail to load).
  armLinks={
    left: entries.filter(e=>e.link.indexOf("openarm_left")>=0).map(e=>e.link),
    right: entries.filter(e=>e.link.indexOf("openarm_right")>=0).map(e=>e.link),
  };

  const loader=new GLTFLoader(); const cache={};
  const urls=[...new Set(entries.map(e=>e.mesh))];
  await Promise.all(urls.map(async (u)=>{
    try{ cache[u]=await loadMeshWithRetry(loader, u, 3); }
    catch(e){ cache[u]=null; log("mesh load failed after retries: "+u+" ("+e+")"); }
  }));

  let ok=0; const failed=[];
  for(const e of entries){
    const g=cache[e.mesh];
    if(!g){ failed.push(e.link); continue; }
    try{
      const obj=g.scene.clone(true);
      // GLTFLoader hands back three's default fully-metallic material, which
      // renders dark/patchy and looks see-through. Replace it: meshes that carry
      // baked vertex colours get the vertex-colour material (so the real part
      // colours show); colourless ones get the neutral grey. Double-sided so
      // decimated winding never leaves holes; frustumCulled off so a stale/bad
      // bounding sphere under the non-standard anchor matrix can't cull a link.
      obj.traverse((o)=>{ if(o.isMesh){
        if(!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
        o.geometry.computeBoundingSphere();
        const hasColor=!!o.geometry.attributes.color;
        if(o.material && o.material.dispose) o.material.dispose();
        o.material=hasColor?ROBOT_MAT_VC:ROBOT_MAT;
        o.frustumCulled=false;
      }});
      let node=linkNodes[e.link];
      // Start hidden: a link mesh is only shown once it has a real FK pose, so a
      // freshly-loaded link can't render clumped at base_link origin (0,0,0)
      // before the first /joint_states frame streams its transform. On robotBase
      // so it drives with the robot through the room.
      if(!node){ node=new THREE.Group(); node.visible=false; linkNodes[e.link]=node; robotBase.add(node); }
      node.add(obj); ok++;
    }catch(err){ failed.push(e.link); log("mesh instance failed "+e.link+": "+err); }
  }

  const loaded=new Set(Object.keys(linkNodes));
  armComplete.left = armLinks.left.length>0 && armLinks.left.every(l=>loaded.has(l));
  armComplete.right = armLinks.right.length>0 && armLinks.right.every(l=>loaded.has(l));
  meshesLoaded = loaded.size>0;
  log("robot meshes: "+ok+"/"+entries.length+" links"+
      (failed.length ? (" — MISSING "+failed.join(", ")+" (wireframe fallback)") : " (all present)"));
}

/* ---- per-frame: pose every link mesh + target markers from lastViz ---- */
function updateRobot(){
  if(!lastViz) return;
  // Offset the whole robot inside the room anchor by the dead-reckoned swerve
  // pose (base_link in the odom frame) so it drives around as commanded. While a
  // place/recenter reset is still in flight the cached base is stale (not yet
  // zeroed), so hold robotBase at the local anchor snap until it lands.
  if(robotBase && lastViz.base && !pendingPlace){
    const b=lastViz.base;
    if(b.p) robotBase.position.set(b.p[0], b.p[1], b.p[2]);
    if(b.q) robotBase.quaternion.set(b.q[0], b.q[1], b.q[2], b.q[3]);
  }
  if(lastViz.links){
    for(const name in lastViz.links){
      const node=linkNodes[name]; if(!node) continue;
      const t=lastViz.links[name];
      node.position.set(t.p[0], t.p[1], t.p[2]);
      node.quaternion.set(t.q[0], t.q[1], t.q[2], t.q[3]);
      node.visible=true;   // now it has a real pose -> safe to show
    }
  }
  updateWire(!meshesLoaded);   // both arms before load; per-arm gap fallback after
  for(const arm of ["left","right"]){
    const a=lastViz.arms ? lastViz.arms[arm] : null;
    const tg=targetMesh[arm], tp=tipMesh[arm], ln=errLine[arm];
    const tax=targetAxes[arm], pax=tipAxes[arm];
    if(!a || !a.target){
      tg.visible=tp.visible=ln.visible=tax.visible=pax.visible=false; continue; }
    const col=reachColor(a.dist==null?9:a.dist);
    tg.position.set(a.target[0],a.target[1],a.target[2]);
    tg.material.color.copy(col); tg.visible=true;
    // Target orientation triad at the target point.
    if(a.target_quat){
      tax.position.set(a.target[0],a.target[1],a.target[2]);
      tax.quaternion.set(a.target_quat[0],a.target_quat[1],a.target_quat[2],a.target_quat[3]);
      tax.visible=true;
    } else tax.visible=false;
    if(a.tip){
      tp.position.set(a.tip[0],a.tip[1],a.tip[2]); tp.visible=true;
      const pos=ln.geometry.attributes.position;
      pos.setXYZ(0, a.tip[0],a.tip[1],a.tip[2]);
      pos.setXYZ(1, a.target[0],a.target[1],a.target[2]);
      pos.needsUpdate=true; ln.material.color.copy(col); ln.visible=true;
      // Live gripper orientation triad at the fingertip.
      if(a.tip_quat){
        pax.position.set(a.tip[0],a.tip[1],a.tip[2]);
        pax.quaternion.set(a.tip_quat[0],a.tip_quat[1],a.tip_quat[2],a.tip_quat[3]);
        pax.visible=true;
      } else pax.visible=false;
    } else { tp.visible=false; ln.visible=false; pax.visible=false; }
  }
}

/* ---- wireframe fallback: drawn for both arms before meshes load, and after
       load only for an arm whose mesh(es) failed (so a gap is never silent) ---- */
function updateWire(forceBoth){
  for(const arm of ["left","right"]){
    const a=lastViz.arms ? lastViz.arms[arm] : null;
    const show = forceBoth || !armComplete[arm];
    if(!show || !a || !a.points){ if(wire[arm]) wire[arm].visible=false; continue; }
    const pts=[];
    for(let i=0;i<a.points.length-1;i++) pts.push(...a.points[i], ...a.points[i+1]);
    let w=wire[arm];
    if(!w){ w=new THREE.LineSegments(new THREE.BufferGeometry(),
        new THREE.LineBasicMaterial({color:0x8c99b8}));
      robotBase.add(w); wire[arm]=w; }
    w.geometry.setAttribute("position", new THREE.Float32BufferAttribute(pts,3));
    w.geometry.attributes.position.needsUpdate=true; w.visible=true;
  }
}

/* ---- place base_link ~1.1 m in front of the operator, on the floor ---- */
function setAnchor(vpose, head){
  const p=vpose.transform.position;
  const n=Math.hypot(head[0],head[2]);
  const F=n>1e-4?[head[0]/n,0,head[2]/n]:[0,0,-1];   // horizontal forward
  const fy=refIsFloor?0:(p.y-1.1);                    // floor height
  const m=new THREE.Matrix4();
  // robot axes -> world: x=F (fwd), y=(Fz,0,-Fx) (left), z=(0,1,0) (up)
  m.makeBasis(new THREE.Vector3(F[0],0,F[2]),
              new THREE.Vector3(F[2],0,-F[0]),
              new THREE.Vector3(0,1,0));
  m.setPosition(p.x+F[0]*1.1, fy, p.z+F[2]*1.1);
  robotRoot.matrix.copy(m); robotRoot.matrixWorldNeedsUpdate=true;
  log("placed robot model");
}

function readController(src, frame){
  if(!src || !src.gripSpace || !src.gamepad) return null;
  const pose=frame.getPose(src.gripSpace, refSpace);
  if(!pose) return {valid:false};
  const p=pose.transform.position; const o=pose.transform.orientation; const gp=src.gamepad;
  // Quest mapping: buttons[0]=trigger,[1]=grip,[3]=stick click,[4]=A/X,[5]=B/Y; axes[2,3]=stick.
  const trigger=gp.buttons[0]?gp.buttons[0].value:0;
  const squeeze=gp.buttons[1]?gp.buttons[1].pressed:false;
  const button=gp.buttons[4]?gp.buttons[4].pressed:false;
  const recenter=gp.buttons[5]?gp.buttons[5].pressed:false;
  const lock=gp.buttons[3]?gp.buttons[3].pressed:false;   // thumbstick click -> rotation lock
  const ax=gp.axes||[];
  const stick=[ax.length>2?ax[2]:(ax[0]||0), ax.length>3?ax[3]:(ax[1]||0)];
  // grip-space orientation (quaternion x,y,z,w) drives the gripper's rotation.
  return {valid:true, pos:[p.x,p.y,p.z], quat:[o.x,o.y,o.z,o.w],
          trigger, squeeze, button, recenter, lock, stick};
}

/* ---- three.js animation loop: read controllers, POST, pose meshes, render ---- */
function onFrame(t, frame){
  if(frame && xrSession){
    refSpace=renderer.xr.getReferenceSpace();
    const out={left:null, right:null};
    for(const src of xrSession.inputSources){
      if(src.handedness==="left"||src.handedness==="right")
        out[src.handedness]=readController(src, frame);
    }
    let recenter=false;
    for(const h of ["left","right"]){
      const c=out[h]; const now=!!(c&&c.recenter);
      if(now && !recenterPrev[h]) recenter=true; recenterPrev[h]=now;
    }
    let head=null;
    const vpose=frame.getViewerPose(refSpace);
    if(vpose){
      const q=vpose.transform.orientation;     // rotate (0,0,-1) by q
      head=[ -2*(q.x*q.z + q.w*q.y),
             -2*(q.y*q.z - q.w*q.x),
             -(1 - 2*(q.x*q.x + q.y*q.y)) ];
      if(!robotPlaced || recenter){
        setAnchor(vpose, head); robotPlaced=true; pendingPlace=true;
        // Snap the model back to the anchor immediately; updateRobot() holds it
        // here (it skips the stale cached base while pendingPlace) until the
        // server's zeroed pose comes back.
        if(robotBase){ robotBase.position.set(0,0,0); robotBase.quaternion.set(0,0,0,1); }
      }
    }
    if(!inFlight){
      inFlight=true;
      // place tells the node to zero its dead-reckoned base pose so the robot
      // sits exactly at the (re)anchored spot. Sticky: clear pendingPlace only
      // after the POST that carried it succeeds, so a reset is never dropped.
      const sentPlace=pendingPlace;
      fetch("/api/xr",{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({controllers:out, head:head, place:sentPlace}),keepalive:true})
        .then(r=>r.json()).then(j=>{ if(j&&j.viz) lastViz=j.viz; if(sentPlace) pendingPlace=false; })
        .catch(()=>{}).finally(()=>{inFlight=false;});
    }
    updateRobot();
  }
  if(renderer) renderer.render(scene, camera);
}

async function startXR(mode){
  try{ xrSession=await navigator.xr.requestSession(mode,
        {optionalFeatures:["local-floor"]}); }
  catch(e){ log("requestSession failed: "+e); return; }
  robotPlaced=false;
  renderer.xr.setReferenceSpaceType("local-floor"); refIsFloor=true;
  try{ await renderer.xr.setSession(xrSession); }
  catch(e){ log("setSession failed: "+e); xrSession=null; return; }
  xrSession.addEventListener("end",()=>{
    xrSession=null; renderer.setAnimationLoop(null); log("session ended"); });
  log("session started ("+mode+")");
  renderer.setAnimationLoop(onFrame);
}

(async ()=>{
  initThree();
  loadRobot();                                 // kicks off mesh download in the background
  if(!navigator.xr){
    enterBtn.textContent="WebXR not available";
    hint.textContent="Open this page in the Meta Quest browser over HTTPS. If you see a "+
      "'connection not private' warning, choose Advanced → Proceed (the cert is self-signed).";
    return;
  }
  const mode=await chooseMode();
  if(!mode){ enterBtn.textContent="No immersive session supported";
    hint.textContent="Your browser exposes WebXR but no immersive mode. Use the Quest browser."; return; }
  enterBtn.disabled=false;
  enterBtn.textContent=(mode==="immersive-ar"?"Enter (passthrough)":"Enter VR");
  enterBtn.onclick=()=>startXR(mode);
  hint.textContent="Tip: the robot appears ~1 m in front of you. Squeeze a Grip to move an arm; press B/Y to recenter the model.";
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()

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

The reach is POSITION-ONLY: each hand drives its arm's target POINT; gripper
orientation is not controlled (the controller's orientation is not used). An
in-headset "REACH ERROR" HUD shows each arm's target-to-fingertip distance.

Controls (per hand):
    * Grip (squeeze) ......... CLUTCH. Hold to "grab" that arm; while held the
                               gripper target follows your hand's MOTION (relative).
                               Release to freeze the arm and reposition your hand
                               (like lifting a mouse off the desk).
    * Trigger ................ that arm's gripper, analog 0 (open) .. 1 (closed).
    * Thumbstick CLICK ....... toggle PRECISION mode for that arm -- hand motion is
                               scaled down (PRECISION_SCALE) for fine, sub-cm target
                               placement; click again to return to 1:1.
    * Thumbstick (push) ...... LEFT hand drives the base: forward/back to drive
                               forward/back, left/right to strafe (crab). RIGHT
                               hand left/right turns (yaw). The robot model in the
                               headset drives through the room to match.
    * A / X button ........... re-seed that arm's target to its live fingertip --
                               "home to here".

Why relative (clutched) position? It lets you move a small real distance, release,
recenter, and continue (standard VR teleop) without matching the robot's whole
reach with your shoulders. Precision mode then scales that motion down so you can
nudge the target onto a point with sub-cm accuracy.

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
PRECISION_SCALE = 0.25   # hand->target scale while an arm is in precision mode
                         # (thumbstick-click), for fine sub-cm target placement

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
        # Per-arm reach error (m): target point - live fingertip. Computed in
        # _viz_locked each frame and surfaced to the headset HUD + the 2D page.
        self.err = {"left": None, "right": None}
        self.seeded = {"left": False, "right": False}
        self.grip = {"left": 0.0, "right": 0.0}
        self.cmd_vel = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}
        # Clutch bookkeeping per arm: where the hand was and where the target
        # was at the instant the grip was squeezed.
        self.clutch = {"left": False, "right": False}
        self.clutch_hand0 = {"left": None, "right": None}
        self.clutch_target0 = {"left": None, "right": None}
        # Precision mode (thumbstick click, per arm): while set, the clutch maps
        # hand motion to target motion at PRECISION_SCALE for fine sub-cm placement.
        # The reach is position-only, so the thumbstick-click no longer locks any
        # gripper rotation -- it is repurposed here for accuracy of placement.
        self.fine = {"left": False, "right": False}
        self.last_precision = {"left": False, "right": False}  # toggle edge detect
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

        # --- viz FK memoization --------------------------------------------
        # The full-robot link poses + per-arm skeleton points depend ONLY on the
        # measured joints, but a headset renders ~72-90 Hz and POSTs each frame --
        # far faster than /joint_states arrives. Recomputing the ~36-link FK +
        # serialization every POST is the bulk of the old per-frame cost, so we
        # memoize that q-only payload and reuse it for every POST that lands
        # between two joint updates. _q_ver is bumped whenever q_meas changes
        # (in _on_joint_states); _viz_fk_cache holds the (already-rounded) links
        # + per-arm points/tip computed at that version. The cheap, per-frame,
        # hand-dependent bits (target sphere, dist, base pose, error line) are
        # always rebuilt fresh on top of the cached FK.
        self._q_ver = 0
        self._viz_fk_cache = None        # (q_ver, {"links":..., "arms":{...}})

        # --- trajectory preview (planned in a background thread) ------------
        # The Cartesian path preview is moderately expensive (tens of ms, more
        # when it must avoid a collision), so it MUST NOT run on the request /
        # _tick path. A daemon worker replans on a throttle (only when a target
        # has actually moved) and stashes the latest result here under the lock;
        # _viz_locked just reads + serializes it. The planner is built lazily
        # (it needs self.reach) on the worker's first pass.
        self._traj = {"left": None, "right": None}   # arm -> latest Trajectory
        self._traj_planner = None
        self._traj_stop = threading.Event()
        # Last target each arm was planned FROM, so the worker can skip replanning
        # while the operator isn't moving the goal (saves the IK every tick).
        self._traj_last_goal = {"left": None, "right": None}
        self._traj_thread = None

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

        # Path-preview worker: only useful with FK, but starting it always (it
        # idles cheaply until joints + the planner are available) keeps the
        # lifecycle simple. Daemon so it dies with the process.
        if self.reach is not None:
            self._traj_thread = threading.Thread(
                target=self._traj_worker, name="m1_quest_traj", daemon=True)
            self._traj_thread.start()

    # --- feedback ----------------------------------------------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_joint_states(self, msg: JointState):
        with self._lock:
            changed = False
            for name, pos in zip(msg.name, msg.position):
                v = float(pos)
                if self.q_meas.get(name) != v:
                    changed = True
                self.q_meas[name] = v
            # Invalidate the memoized FK payload only when the joints actually
            # moved, so back-to-back POSTs at the same q reuse the cache.
            if changed:
                self._q_ver += 1
            if self.reach is None:
                return
            for arm in ("left", "right"):
                # Seed each arm's target onto its current fingertip exactly once
                # so the arm holds still on connect instead of snapping.
                if self.seeded[arm]:
                    continue
                if all(j in self.q_meas for j in ARM_JOINTS[arm] + [LIFT_JOINT]):
                    try:
                        tip = self.reach.fingertip(arm, self.q_meas)
                        self.target[arm] = [float(tip[0]), float(tip[1]), float(tip[2])]
                        self.seeded[arm] = True
                    except Exception:  # noqa: BLE001
                        pass

    def _reseed(self, arm: str):
        if self.reach is None:
            return
        if all(j in self.q_meas for j in ARM_JOINTS[arm] + [LIFT_JOINT]):
            try:
                tip = self.reach.fingertip(arm, self.q_meas)
                self.target[arm] = [float(tip[0]), float(tip[1]), float(tip[2])]
                self.seeded[arm] = True
            except Exception:  # noqa: BLE001
                pass

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
                squeeze = bool(c.get("squeeze"))
                trigger = _clamp(float(c.get("trigger", 0.0)), 0.0, 1.0)
                button = bool(c.get("button"))
                precision = bool(c.get("lock"))  # thumbstick click (gamepad btn 3)

                # A/X edge: re-seed this arm's target to the live fingertip
                # ("home to here") and drop the clutch.
                if button and not self.last_btn[arm]:
                    self._reseed(arm)
                    self.clutch[arm] = False
                self.last_btn[arm] = button

                # Thumbstick-click edge: toggle this arm's PRECISION mode (fine
                # placement). Position-only reach, so there is no rotation to lock;
                # the click now scales hand motion down for sub-cm accuracy.
                if precision and not self.last_precision[arm]:
                    self.fine[arm] = not self.fine[arm]
                self.last_precision[arm] = precision

                # Clutch logic: target follows hand delta only while squeezed. The
                # delta (raw WebXR metres) is projected into the operator's heading
                # frame captured at the squeeze, so "forward/left/up relative to
                # where you're looking" map to robot x/y/z.
                if squeeze and not self.clutch[arm]:
                    self.clutch[arm] = True
                    self.clutch_hand0[arm] = hand
                    self.clutch_target0[arm] = np.array(self.target[arm])
                    F, L = _heading_basis(head_fwd)
                    self.clutch_F[arm], self.clutch_L[arm] = F, L
                    self.seeded[arm] = True
                elif squeeze and self.clutch[arm]:
                    scale = self.motion_scale * (PRECISION_SCALE if self.fine[arm] else 1.0)
                    d = (hand - self.clutch_hand0[arm]) * scale
                    F, L = self.clutch_F[arm], self.clutch_L[arm]
                    robot_delta = np.array([
                        float(-(d @ L)),  # right    -> +x  (x/y swapped, x reversed)
                        float(d @ F),     # forward  -> +y  (x/y swapped)
                        float(d[1]),      # up       -> +z
                    ])
                    self._set_target(arm, self.clutch_target0[arm] + robot_delta)
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
            # Take a cheap SNAPSHOT (copies) of everything the viz needs while we
            # still hold the lock, then release it BEFORE the multi-millisecond
            # FK / serialization runs -- so the heavy work no longer serializes
            # against the 60 Hz _tick publish and the /joint_states callback.
            snap = self._viz_snapshot_locked()
        # Heavy FK + serialization happen OUTSIDE the lock, from the snapshot.
        viz = self._build_viz(snap)
        return {"ok": True, "viz": viz}

    def _viz_snapshot_locked(self) -> dict:
        """Cheap copy of the viz inputs (assumes ``_lock`` held).

        Copies only the small, hand/odom-dependent state -- the expensive FK is
        done afterwards, off the lock, by :meth:`_build_viz`. Also snapshots the
        latest planned trajectory (read here so the background planner's writes
        are seen atomically with the rest of the state).
        """
        x, y, _ = self.odom.pose
        return {
            "q_meas": dict(self.q_meas),
            "q_ver": getattr(self, "_q_ver", 0),
            "target": {a: list(self.target[a]) for a in ("left", "right")},
            "fine": {a: self.fine[a] for a in ("left", "right")},
            "base": {
                "p": [round(x, 4), round(y, 4), 0.0],
                "q": [round(c, 5) for c in self.odom.quaternion()],
            },
            "traj": {a: getattr(self, "_traj", {}).get(a)
                     for a in ("left", "right")},
        }

    def _viz_locked(self) -> dict:
        """Backwards-compatible viz build (snapshot + build in one call).

        Kept for the position test, which calls this directly. In the live path
        :meth:`on_xr_frame` snapshots under the lock and builds outside it; here
        we just do both back-to-back (the caller is single-threaded).
        """
        snap = self._viz_snapshot_locked()
        return self._build_viz(snap)

    def _fk_payload(self, q_meas: dict, q_ver: int) -> dict:
        """The q-only viz payload: full-robot ``links`` + per-arm skeleton
        ``points``/``tip``, all already rounded for JSON.

        Memoized on ``q_ver`` (bumped in _on_joint_states): a headset POSTs far
        faster than joints update, so every POST that lands between two joint
        frames reuses this instead of re-running the ~36-link FK + per-arm
        ``link_points`` + ``mat_to_quat`` + rounding. This is the bulk of the
        old per-frame cost. The hand-dependent bits (target/dist/base) are NOT
        cached -- they are rebuilt fresh by :meth:`_build_viz` every call.
        """
        cache = getattr(self, "_viz_fk_cache", None)
        if cache is not None and cache[0] == q_ver:
            return cache[1]

        payload = {"links": {}, "arms": {}}
        have_js = bool(q_meas)
        if self.reach is not None and have_js:
            # Full-robot link poses (base_link frame) so the page can drive each
            # mesh; cheap matrix walk over the (origin-cached) URDF tree.
            try:
                Ts = self.reach.model.link_transforms(q_meas)
                for name, T in Ts.items():
                    payload["links"][name] = {
                        "p": [round(float(T[i, 3]), 4) for i in range(3)],
                        "q": [round(float(c), 5) for c in mat_to_quat(T[:3, :3])],
                    }
            except Exception:  # noqa: BLE001
                pass
            for arm in ("left", "right"):
                if not all(j in q_meas for j in ARM_JOINTS[arm] + [LIFT_JOINT]):
                    continue
                try:
                    pts = self.reach.chains[arm].link_points(q_meas)
                    tip = pts[-1]
                    payload["arms"][arm] = {
                        "points": [[round(float(c), 4) for c in p] for p in pts],
                        "tip": [round(float(c), 4) for c in tip],
                        # raw fingertip kept (unrounded) so the per-frame dist is
                        # exact; not serialized.
                        "_tip": np.asarray(tip, dtype=np.float64),
                    }
                except Exception:  # noqa: BLE001
                    pass
        self._viz_fk_cache = (q_ver, payload)
        return payload

    def _build_viz(self, snap: dict) -> dict:
        """Assemble the headset overlay from a snapshot (NO lock held).

        For each arm we send the target point, the live fingertip, the
        base->tip skeleton points (FK of the *measured* joints), and the
        fingertip->target distance so the page can colour the target by how
        well the arm is reaching (green=on target, red=out of reach). All
        points are in the robot ``base_link`` frame; the page anchors that
        frame to a fixed spot in the room, then offsets it by ``base`` (the
        dead-reckoned swerve pose) so the model drives around as commanded.

        The heavy, q-only part (links + skeleton) comes from the memoized
        :meth:`_fk_payload`; only the cheap, per-frame target/dist/base/traj are
        built here. ``self.err`` is updated under a short lock so the 2D
        snapshot readout stays thread-safe.
        """
        fk = self._fk_payload(snap["q_meas"], snap["q_ver"])
        viz = {
            "frame": "base_link",
            # links is q-only and already serialized -> reuse the cached dict.
            "links": fk["links"],
            "arms": {},
            # base_link pose in the odom frame (x fwd, y left, +yaw = CCW about
            # +z). The page applies this to the robot group so the whole model
            # translates/turns through the room as the base drives.
            "base": snap["base"],
        }
        err = {"left": None, "right": None}
        for arm in ("left", "right"):
            target = snap["target"][arm]
            a = {
                "target": [round(float(v), 4) for v in target],
                "fine": snap["fine"][arm],
            }
            fa = fk["arms"].get(arm)
            if fa is not None:
                # Skeleton + fingertip are q-only (cached); copy the serialized
                # forms straight through.
                a["points"] = fa["points"]
                a["tip"] = fa["tip"]
                # Reach error (target point - fingertip), in metres. Cheap: one
                # norm. Recomputed every frame so a target move updates the
                # colour/HUD immediately even though the FK is memoized.
                dist = float(np.linalg.norm(np.asarray(target) - fa["_tip"]))
                a["dist"] = round(dist, 4)
                err[arm] = dist
            viz["arms"][arm] = a
        # Trajectory preview (planned off-thread). Per-arm polyline + per-waypoint
        # colliding flags, in base_link frame like ``arms[..]["points"]``.
        traj = self._traj_payload(snap.get("traj", {}))
        if traj:
            viz["traj"] = traj
        # Surface the reach error to the 2D /api/state readout. Short lock so it
        # never races _on_joint_states / snapshot.
        with self._lock:
            self.err = err
        return viz

    @staticmethod
    def _traj_payload(traj: dict) -> dict:
        """Serialize each arm's latest Trajectory into the viz form:
        ``{arm: {"points": [[x,y,z]...], "colliding": [bool...], "free": bool}}``.

        Points are the fingertip polyline (base_link frame), rounded to 4 dp like
        the rest of the viz. Returns ``{}`` when nothing is planned yet so the
        page simply hides the path.
        """
        out = {}
        for arm in ("left", "right"):
            tr = traj.get(arm)
            if tr is None:
                continue
            try:
                pts = tr.points_for(arm)
                if not pts:
                    continue
                out[arm] = {
                    "points": [[round(float(c), 4) for c in p] for p in pts],
                    "colliding": [bool(wp.colliding) for wp in tr.waypoints],
                    "free": bool(tr.collision_free),
                }
            except Exception:  # noqa: BLE001
                continue
        return out

    # --- background trajectory planner -------------------------------------
    def _traj_worker(self):
        """Replan the per-arm Cartesian preview off the request/_tick path.

        Throttled to ~3-4 Hz and only when a target has actually moved (> ~1 cm)
        since its last plan, so the (tens-of-ms) IK never piles up. Snapshots the
        inputs under the lock, plans OUTSIDE it, then stores the result under the
        lock. The whole loop is guarded so a planner hiccup can't kill the thread.
        """
        # Threshold (m) a goal must move before we bother replanning. Smaller than
        # the operator could meaningfully see in the preview.
        REPLAN_EPS = 0.01
        while not self._traj_stop.is_set():
            try:
                # Lazy planner construction (needs self.reach).
                if self._traj_planner is None:
                    from m1_control.trajectory import TrajectoryPlanner
                    self._traj_planner = TrajectoryPlanner(self.reach)

                with self._lock:
                    q_meas = dict(self.q_meas)
                    targets = {a: list(self.target[a]) for a in ("left", "right")}
                    seeded = dict(self.seeded)
                    last_goal = dict(self._traj_last_goal)
                # Only plan for arms whose joints are live AND whose target moved
                # enough since the last plan -- otherwise reuse the stored result.
                goals = {}
                for arm in ("left", "right"):
                    if not seeded.get(arm):
                        continue
                    if not all(j in q_meas for j in ARM_JOINTS[arm] + [LIFT_JOINT]):
                        continue
                    g = np.asarray(targets[arm], dtype=np.float64)
                    lg = last_goal.get(arm)
                    if lg is None or float(np.linalg.norm(g - np.asarray(lg))) > REPLAN_EPS:
                        goals[arm] = g
                if goals:
                    new_traj = {}
                    for arm, g in goals.items():
                        # Plan each active arm on its own so one arm's goal move
                        # doesn't disturb the other's preview.
                        tr = self._traj_planner.plan(
                            q_meas, {"left": g if arm == "left" else None,
                                     "right": g if arm == "right" else None})
                        new_traj[arm] = (tr, [float(c) for c in g])
                    with self._lock:
                        for arm, (tr, g) in new_traj.items():
                            self._traj[arm] = tr
                            self._traj_last_goal[arm] = g
            except Exception:  # noqa: BLE001
                # Never let a planner hiccup kill the worker; just retry next pass.
                pass
            # ~3-4 Hz. interruptible so shutdown is immediate.
            self._traj_stop.wait(0.28)

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
            # Position-only reach: orientation is ignored by the controller, so
            # publish a valid identity quaternion.
            ps.pose.orientation.w = 1.0
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
                    "fine": self.fine[arm],
                    # Reach error (mm) for the 2D readout; None until joints stream.
                    "err_mm": (round(self.err[arm] * 1e3, 1)
                               if self.err[arm] is not None else None),
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
        node._traj_stop.set()   # ask the path-preview worker to exit (daemon)
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
      <tr><td>left target</td><td id="lt">–</td><td id="lg">grip –</td><td id="le">err –</td><td id="lc"></td></tr>
      <tr><td>right target</td><td id="rt">–</td><td id="rg">grip –</td><td id="re">err –</td><td id="rc"></td></tr>
    </table>
  </div>

  <div class="card">
    <h3 style="margin:0 0 8px;font-family:var(--serif)">Controls</h3>
    <p class="muted">
      <kbd>Grip</kbd> (squeeze) — hold to "grab" that arm; the gripper's target
      follows your hand's motion. Release to reposition your hand.<br/>
      <kbd>Trigger</kbd> — that arm's gripper (squeeze to close).<br/>
      <kbd>Stick click</kbd> — toggle <em>precision mode</em> for that arm (hand
      motion scaled down for fine, sub-cm placement).<br/>
      <kbd>Left stick</kbd> — drive: forward/back = drive fwd/back, left/right = strafe.<br/>
      <kbd>Right stick</kbd> — left/right = turn (yaw).<br/>
      <kbd>A</kbd>/<kbd>X</kbd> — re-home that arm's target to where it is now.<br/>
      <kbd>B</kbd>/<kbd>Y</kbd> — recenter the robot hologram in front of you.
    </p>
    <p class="muted">
      In the headset you'll see a 3D model of the robot (its real meshes) drawn
      over your room in passthrough, posed live from the robot's joints, plus a
      coloured sphere at each Cartesian target and a floating <b>REACH ERROR</b>
      panel showing each arm's target↔fingertip distance in mm. The target and the
      error readout turn <span style="color:var(--good)">green</span> when the arm
      reaches it and <span style="color:var(--bad)">red</span> when it's out of
      reach, so you can see impossible goals immediately. (First load streams
      ~4 MB of meshes; it's cached afterward.)
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
      document.getElementById(a[0]+"e").textContent=
        (arm.err_mm==null?"err –":"err "+arm.err_mm.toFixed(1)+"mm");
      document.getElementById(a[0]+"c").textContent=
        (arm.clutch?"● clutch":"")+(arm.fine?"  ◎ fine":""); };
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
// Per-arm trajectory PREVIEW: a vertex-coloured polyline (green=clear, red where
// a waypoint self-collides) plus camera-facing dots at the waypoints, both under
// robotBase so they drive with the robot. Preallocated once in initThree(),
// updated in place from lastViz.traj (no per-frame allocation).
const trajLine={}, trajDots={};
const TRAJ_MAX_PTS=64;   // >> planner's ~25 waypoints; fixed buffer capacity
let errHud=null, hudCanvas=null, hudCtx=null, hudTex=null;   // in-VR REACH ERROR panel
// Last distances actually drawn into the HUD canvas (mm, quantized to 0.1). The
// canvas redraw + GPU texture upload is the costly part of the HUD, so we skip
// it while the displayed values are unchanged and only re-billboard the panel.
let hudDrawnL=null, hudDrawnR=null;
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

// CSS colour for the error HUD, matching reachColor's thresholds.
function errCss(d){
  if(d==null) return "#8c99b8";
  if(d<0.02) return "#4cd964";
  if(d<0.08) return "#f2b33a";
  return "#e64d40";
}

/* ---- redraw the REACH ERROR HUD canvas from each arm's distance (m) ----
   Skips the (expensive) canvas clear+draw+texture upload when both displayed
   values are unchanged since the last draw, quantized to 0.1 mm -- the panel is
   still re-billboarded every frame by updateHud(), which is cheap. ---- */
function drawHud(dl, dr){
  if(!hudCtx) return;
  // Quantize to 0.1 mm (null stays null) so sub-tenth jitter doesn't redraw.
  const ql = dl==null?null:Math.round(dl*1e4);
  const qr = dr==null?null:Math.round(dr*1e4);
  if(ql===hudDrawnL && qr===hudDrawnR) return;   // nothing visible changed
  hudDrawnL=ql; hudDrawnR=qr;
  const W=hudCanvas.width, H=hudCanvas.height, c=hudCtx;
  c.clearRect(0,0,W,H);
  c.fillStyle="rgba(18,18,22,0.72)";
  c.strokeStyle="rgba(255,255,255,0.18)"; c.lineWidth=4;
  c.beginPath(); c.roundRect(6,6,W-12,H-12,18); c.fill(); c.stroke();
  c.textBaseline="middle"; c.textAlign="left";
  c.fillStyle="#e8e6df";
  c.font="600 34px system-ui,-apple-system,'Segoe UI',sans-serif";
  c.fillText("REACH ERROR",36,46);
  const row=(label,d,y)=>{
    const col=errCss(d);
    c.beginPath(); c.arc(54,y,14,0,Math.PI*2); c.fillStyle=col; c.fill();
    c.fillStyle="#cfccc4"; c.font="500 34px system-ui,sans-serif";
    c.textAlign="left"; c.fillText(label,84,y);
    c.fillStyle=col; c.textAlign="right";
    c.font="600 40px ui-monospace,Menlo,Consolas,monospace";
    c.fillText(d==null?"– mm":(d*1000).toFixed(1)+" mm",W-40,y);
  };
  row("left", dl, 120);
  row("right", dr, 186);
  hudTex.needsUpdate=true;
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

    // Trajectory preview polyline: fixed-capacity position + per-vertex colour
    // buffers, drawn via setDrawRange so we never reallocate. vertexColors lets
    // a colliding stretch turn red while the rest stays green.
    const pg=new THREE.BufferGeometry();
    pg.setAttribute("position", new THREE.Float32BufferAttribute(
      new Float32Array(TRAJ_MAX_PTS*3), 3).setUsage(THREE.DynamicDrawUsage));
    pg.setAttribute("color", new THREE.Float32BufferAttribute(
      new Float32Array(TRAJ_MAX_PTS*3), 3).setUsage(THREE.DynamicDrawUsage));
    const pl=new THREE.Line(pg, new THREE.LineBasicMaterial({vertexColors:true}));
    pl.visible=false; robotBase.add(pl); trajLine[arm]=pl;
    // Waypoint dots: camera-facing points sharing the same colour scheme. Its own
    // buffers (sizeAttenuation off so the dots stay legible at any robot range).
    const dg=new THREE.BufferGeometry();
    dg.setAttribute("position", new THREE.Float32BufferAttribute(
      new Float32Array(TRAJ_MAX_PTS*3), 3).setUsage(THREE.DynamicDrawUsage));
    dg.setAttribute("color", new THREE.Float32BufferAttribute(
      new Float32Array(TRAJ_MAX_PTS*3), 3).setUsage(THREE.DynamicDrawUsage));
    const dots=new THREE.Points(dg, new THREE.PointsMaterial(
      {size:0.018, vertexColors:true, sizeAttenuation:true, depthTest:true}));
    dots.visible=false; robotBase.add(dots); trajDots[arm]=dots;
  }

  // In-VR REACH ERROR HUD: a canvas-textured panel that floats above the robot
  // and billboards to face the operator, showing each arm's target->fingertip
  // error in mm (green/amber/red). Attached to the scene (not the robot groups)
  // so billboarding is a plain camera-facing quaternion and it ignores odom drift.
  hudCanvas=document.createElement("canvas"); hudCanvas.width=512; hudCanvas.height=256;
  hudCtx=hudCanvas.getContext("2d");
  hudTex=new THREE.CanvasTexture(hudCanvas);
  const hudMat=new THREE.MeshBasicMaterial({map:hudTex, transparent:true,
    depthTest:false, side:THREE.DoubleSide});
  errHud=new THREE.Mesh(new THREE.PlaneGeometry(0.40,0.20), hudMat);
  errHud.renderOrder=10; errHud.visible=false; scene.add(errHud);
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
    if(!a || !a.target){
      tg.visible=tp.visible=ln.visible=false; continue; }
    const col=reachColor(a.dist==null?9:a.dist);
    tg.position.set(a.target[0],a.target[1],a.target[2]);
    tg.material.color.copy(col); tg.visible=true;
    if(a.tip){
      tp.position.set(a.tip[0],a.tip[1],a.tip[2]); tp.visible=true;
      const pos=ln.geometry.attributes.position;
      pos.setXYZ(0, a.tip[0],a.tip[1],a.tip[2]);
      pos.setXYZ(1, a.target[0],a.target[1],a.target[2]);
      pos.needsUpdate=true; ln.material.color.copy(col); ln.visible=true;
    } else { tp.visible=false; ln.visible=false; }
  }
  updateTraj();
  updateHud();
}

/* ---- per-frame: draw each arm's planned path preview from lastViz.traj ----
   Fills the preallocated polyline + waypoint-dot buffers in place. A waypoint's
   colour is green when clear and red when its `colliding` flag is set; a polyline
   vertex is red if either it or its predecessor collides, so a bad stretch reads
   red. Hidden whenever there's no trajectory for that arm. Robust to a missing
   lastViz.traj (older response / nothing planned yet). ---- */
const TRAJ_GREEN=[0.30,0.85,0.39], TRAJ_RED=[0.90,0.30,0.25];
function updateTraj(){
  const traj=lastViz.traj||null;
  for(const arm of ["left","right"]){
    const pl=trajLine[arm], dots=trajDots[arm];
    const t=traj?traj[arm]:null;
    if(!t || !t.points || t.points.length<1){
      if(pl) pl.visible=false; if(dots) dots.visible=false; continue;
    }
    const pts=t.points, col=t.colliding||[];
    const n=Math.min(pts.length, TRAJ_MAX_PTS);
    const lp=pl.geometry.attributes.position.array, lc=pl.geometry.attributes.color.array;
    const dp=dots.geometry.attributes.position.array, dc=dots.geometry.attributes.color.array;
    for(let i=0;i<n;i++){
      const p=pts[i], b=i*3;
      lp[b]=dp[b]=p[0]; lp[b+1]=dp[b+1]=p[1]; lp[b+2]=dp[b+2]=p[2];
      // dot: its own waypoint flag; line vertex: red if it or the prior collides
      // so the offending segment is unambiguously red.
      const dotC = col[i] ? TRAJ_RED : TRAJ_GREEN;
      const lnC = (col[i] || (i>0 && col[i-1])) ? TRAJ_RED : TRAJ_GREEN;
      dc[b]=dotC[0]; dc[b+1]=dotC[1]; dc[b+2]=dotC[2];
      lc[b]=lnC[0]; lc[b+1]=lnC[1]; lc[b+2]=lnC[2];
    }
    pl.geometry.setDrawRange(0, n);
    dots.geometry.setDrawRange(0, n);
    pl.geometry.attributes.position.needsUpdate=true;
    pl.geometry.attributes.color.needsUpdate=true;
    dots.geometry.attributes.position.needsUpdate=true;
    dots.geometry.attributes.color.needsUpdate=true;
    // A single-waypoint path has no segment to draw; show the dot, hide the line.
    pl.visible = n>=2; dots.visible=true;
  }
}

/* ---- pose + redraw the REACH ERROR HUD: float it above the robot and face the
       camera (billboard), refreshing the per-arm error text/colours ---- */
function updateHud(){
  if(!errHud) return;
  const al=lastViz.arms?lastViz.arms.left:null, ar=lastViz.arms?lastViz.arms.right:null;
  const dl=al&&al.dist!=null?al.dist:null, dr=ar&&ar.dist!=null?ar.dist:null;
  if(dl==null && dr==null){ errHud.visible=false; return; }
  drawHud(dl, dr);
  // Anchor above the robot base (its world position) and billboard to the camera.
  const base=new THREE.Vector3();
  (robotBase||robotRoot).getWorldPosition(base);
  base.y += 1.35;                       // float above the robot
  errHud.position.copy(base);
  // Billboard: copy the (XR head) camera's world orientation. A camera looks down
  // its -z, so the plane's +z (front, unmirrored) face ends up toward the viewer.
  const cam = (renderer.xr.isPresenting ? renderer.xr.getCamera() : camera);
  if(cam){ const q=new THREE.Quaternion(); cam.getWorldQuaternion(q); errHud.quaternion.copy(q); }
  errHud.visible=true;
}

/* ---- wireframe fallback: drawn for both arms before meshes load, and after
       load only for an arm whose mesh(es) failed (so a gap is never silent).
       The position buffer is PREALLOCATED once per arm (ample fixed capacity)
       and updated in place + drawn via setDrawRange, instead of allocating a new
       Float32BufferAttribute every frame while loading / for a failed arm. ---- */
const WIRE_MAX_PTS=32;   // arm skeleton is well under this; segments=(pts-1)
function updateWire(forceBoth){
  for(const arm of ["left","right"]){
    const a=lastViz.arms ? lastViz.arms[arm] : null;
    const show = forceBoth || !armComplete[arm];
    if(!show || !a || !a.points){ if(wire[arm]) wire[arm].visible=false; continue; }
    let w=wire[arm];
    if(!w){
      const geom=new THREE.BufferGeometry();
      // (WIRE_MAX_PTS-1) segments * 2 endpoints * 3 floats, allocated once.
      const buf=new THREE.Float32BufferAttribute(
        new Float32Array((WIRE_MAX_PTS-1)*2*3), 3);
      buf.setUsage(THREE.DynamicDrawUsage);
      geom.setAttribute("position", buf);
      w=new THREE.LineSegments(geom, new THREE.LineBasicMaterial({color:0x8c99b8}));
      robotBase.add(w); wire[arm]=w;
    }
    // Fill the existing array in place (consecutive points -> line segments).
    const arr=w.geometry.attributes.position.array;
    const n=Math.min(a.points.length, WIRE_MAX_PTS);
    let k=0;
    for(let i=0;i<n-1;i++){
      arr[k++]=a.points[i][0];   arr[k++]=a.points[i][1];   arr[k++]=a.points[i][2];
      arr[k++]=a.points[i+1][0]; arr[k++]=a.points[i+1][1]; arr[k++]=a.points[i+1][2];
    }
    w.geometry.setDrawRange(0, Math.max(0, (n-1)*2));
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
  const p=pose.transform.position; const gp=src.gamepad;
  // Quest mapping: buttons[0]=trigger,[1]=grip,[3]=stick click,[4]=A/X,[5]=B/Y; axes[2,3]=stick.
  const trigger=gp.buttons[0]?gp.buttons[0].value:0;
  const squeeze=gp.buttons[1]?gp.buttons[1].pressed:false;
  const button=gp.buttons[4]?gp.buttons[4].pressed:false;
  const recenter=gp.buttons[5]?gp.buttons[5].pressed:false;
  const lock=gp.buttons[3]?gp.buttons[3].pressed:false;   // thumbstick click -> precision toggle
  const ax=gp.axes||[];
  const stick=[ax.length>2?ax[2]:(ax[0]||0), ax.length>3?ax[3]:(ax[1]||0)];
  // Position-only: grip-space orientation is no longer sent (the gripper rotation
  // is not controlled). 'lock' carries the thumbstick-click for precision mode.
  return {valid:true, pos:[p.x,p.y,p.z],
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

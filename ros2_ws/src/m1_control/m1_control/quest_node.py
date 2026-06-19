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
    * Grip (squeeze) ......... CLUTCH. Hold to "grab" that arm; the gripper
                               target follows your hand's MOTION while held.
                               Release to freeze the arm and reposition your
                               hand (like lifting a mouse off the desk).
    * Trigger ................ that arm's gripper, analog 0 (open) .. 1 (closed).
    * Thumbstick ............. (left hand) drive the base: push to translate,
                               (right hand) push left/right to yaw.
    * A / X button ........... re-seed that arm's target to its live fingertip
                               (undo drift) -- safe "home to current pose".

Why relative + clutch? Absolute 1:1 mapping forces your shoulders to match the
robot's workspace exactly. Clutched relative motion lets you move a small real
distance, release, recenter, and continue -- standard for VR teleop.

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
)

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
        self.seeded = {"left": False, "right": False}
        self.grip = {"left": 0.0, "right": 0.0}
        self.cmd_vel = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}
        # Clutch bookkeeping per arm: where the hand was and where the target
        # was at the instant the grip was squeezed.
        self.clutch = {"left": False, "right": False}
        self.clutch_hand0 = {"left": None, "right": None}
        self.clutch_target0 = {"left": None, "right": None}
        # Heading-relative basis (WebXR-space F/L vectors) captured at the
        # instant the clutch is squeezed, so the mapping stays fixed for the
        # duration of that grab even if the operator turns their head.
        self.clutch_F = {"left": None, "right": None}
        self.clutch_L = {"left": None, "right": None}
        self.last_btn = {"left": False, "right": False}  # A/X edge detect
        self._last_base_cmd = 0.0
        self._last_update = 0.0

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

                # A/X edge: re-seed this arm's target to the live fingertip.
                if button and not self.last_btn[arm]:
                    self._reseed(arm)
                    self.clutch[arm] = False
                self.last_btn[arm] = button

                # Clutch logic: target follows hand delta only while squeezed.
                # The delta (raw WebXR metres) is projected into the operator's
                # heading frame captured at the squeeze, so "forward/left/up
                # relative to where you're looking" map to robot x/y/z.
                if squeeze and not self.clutch[arm]:
                    self.clutch[arm] = True
                    self.clutch_hand0[arm] = hand
                    self.clutch_target0[arm] = np.array(self.target[arm])
                    self.clutch_F[arm], self.clutch_L[arm] = _heading_basis(head_fwd)
                    self.seeded[arm] = True
                elif squeeze and self.clutch[arm]:
                    d = (hand - self.clutch_hand0[arm]) * self.motion_scale
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

            # Base from thumbsticks (left = translate, right.x = yaw).
            if self.enable_base:
                left = controllers.get("left") or {}
                right = controllers.get("right") or {}
                lx, ly = self._stick(left)
                rx, _ = self._stick(right)
                if abs(lx) > 0 or abs(ly) > 0 or abs(rx) > 0:
                    self.cmd_vel = {
                        # stick y is +down in gamepad convention -> push up = fwd
                        "vx": _clamp(-ly * MAX_LINEAR, -MAX_LINEAR, MAX_LINEAR),
                        "vy": _clamp(-lx * MAX_STRAFE, -MAX_STRAFE, MAX_STRAFE),
                        "yaw": _clamp(-rx * MAX_YAW, -MAX_YAW, MAX_YAW),
                    }
                    self._last_base_cmd = self._now()
        return {"ok": True}

    @staticmethod
    def _stick(c: dict) -> tuple[float, float]:
        s = c.get("stick", [0.0, 0.0]) if c else [0.0, 0.0]
        x = float(s[0]) if len(s) > 0 else 0.0
        y = float(s[1]) if len(s) > 1 else 0.0
        if abs(x) < STICK_DEADZONE:
            x = 0.0
        if abs(y) < STICK_DEADZONE:
            y = 0.0
        return x, y

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

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, INDEX_HTML, "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._send(200, json.dumps(node.snapshot()))
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
# WebXR page (served at "/"). Plain HTML/JS, no build step, no dependencies.
# Requests an immersive-ar session (passthrough so you still see your room),
# falling back to immersive-vr. Each XR frame it reads both controllers' grip
# poses + buttons and POSTs them to /api/xr (one request in flight at a time).
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
      hand's motion. Release to freeze &amp; recenter.<br/>
      <kbd>Trigger</kbd> — that arm's gripper (squeeze to close).<br/>
      <kbd>Left stick</kbd> — drive the base (push to translate).<br/>
      <kbd>Right stick</kbd> — turn (yaw).<br/>
      <kbd>A</kbd>/<kbd>X</kbd> — re-home that arm's target to where it is now.
    </p>
  </div>
  <div id="log"></div>
</main>

<script>
const log = (m)=>{ const el=document.getElementById("log");
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
      document.getElementById(a[0]+"c").textContent=arm.clutch?"● clutch":""; };
    f("left"); f("right");
  }catch(e){}
}
setInterval(poll,300); poll();

/* ---- WebXR teleop ---- */
let xrSession=null, refSpace=null, gl=null, inFlight=false;
const enterBtn=document.getElementById("enter");
const hint=document.getElementById("xrhint");

async function chooseMode(){
  if(!navigator.xr){ return null; }
  if(await navigator.xr.isSessionSupported("immersive-ar")) return "immersive-ar";
  if(await navigator.xr.isSessionSupported("immersive-vr")) return "immersive-vr";
  return null;
}

(async ()=>{
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
  hint.textContent="Tip: stand with arms comfortable, then squeeze a Grip button to start moving that arm.";
})();

async function startXR(mode){
  try{
    xrSession=await navigator.xr.requestSession(mode,{optionalFeatures:["local-floor"]});
  }catch(e){ log("requestSession failed: "+e); return; }

  const canvas=document.createElement("canvas");
  gl=canvas.getContext("webgl",{xrCompatible:true,alpha:true});
  await gl.makeXRCompatible();
  xrSession.updateRenderState({baseLayer:new XRWebGLLayer(xrSession,gl)});
  try{ refSpace=await xrSession.requestReferenceSpace("local-floor"); }
  catch(e){ refSpace=await xrSession.requestReferenceSpace("local"); }

  xrSession.addEventListener("end",()=>{ xrSession=null; log("session ended"); });
  log("session started ("+mode+")");
  xrSession.requestAnimationFrame(onXRFrame);
}

function readController(src, frame){
  if(!src || !src.gripSpace || !src.gamepad) return null;
  const pose=frame.getPose(src.gripSpace, refSpace);
  if(!pose) return {valid:false};
  const p=pose.transform.position;
  const gp=src.gamepad;
  // Quest mapping: buttons[0]=trigger, [1]=squeeze/grip, [4]=A/X; axes[2,3]=stick.
  const trigger=gp.buttons[0]?gp.buttons[0].value:0;
  const squeeze=gp.buttons[1]?gp.buttons[1].pressed:false;
  const button=gp.buttons[4]?gp.buttons[4].pressed:false;
  const ax=gp.axes||[];
  const stick=[ax.length>2?ax[2]:(ax[0]||0), ax.length>3?ax[3]:(ax[1]||0)];
  return {valid:true, pos:[p.x,p.y,p.z], trigger, squeeze, button, stick};
}

function onXRFrame(t, frame){
  if(!xrSession) return;
  xrSession.requestAnimationFrame(onXRFrame);

  // Required for an immersive session: bind + clear the XR framebuffer.
  const glLayer=xrSession.renderState.baseLayer;
  gl.bindFramebuffer(gl.FRAMEBUFFER, glLayer.framebuffer);
  gl.clearColor(0,0,0,0);            // transparent -> passthrough shows through
  gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);

  const out={left:null, right:null};
  for(const src of xrSession.inputSources){
    if(src.handedness==="left"||src.handedness==="right"){
      out[src.handedness]=readController(src, frame);
    }
  }

  // Headset forward vector (WebXR space) so the server can map hand motion
  // relative to where you're looking, not to the arbitrary reference yaw.
  let head=null;
  const vp=frame.getViewerPose(refSpace);
  if(vp){
    const q=vp.transform.orientation;     // rotate (0,0,-1) by q
    head=[ -2*(q.x*q.z + q.w*q.y),
           -2*(q.y*q.z - q.w*q.x),
           -(1 - 2*(q.x*q.x + q.y*q.y)) ];
  }

  // Throttle: at most one POST in flight so we never queue up under load.
  if(!inFlight){
    inFlight=true;
    fetch("/api/xr",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({controllers:out, head:head}),keepalive:true})
      .catch(()=>{}).finally(()=>{inFlight=false;});
  }
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()

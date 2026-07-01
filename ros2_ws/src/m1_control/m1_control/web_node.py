"""Web control panel for the M1 robot (sim AND real).

Starts a small HTTP server that serves a single-page control panel and bridges
the browser to the ``m1_controller`` brain over the standard ``/m1/*`` ROS 2
topics. Because it only uses those topics, the exact same panel drives the Isaac
Sim robot and the real hardware -- whatever is publishing ``/joint_states`` and
applying ``/m1/joint_command`` underneath.

    out  /m1/<arm>/target_pose   geometry_msgs/PoseStamped   (Cartesian reach)
    out  /m1/cmd_vel             geometry_msgs/Twist         (swerve base)
    out  /m1/<arm>/gripper       std_msgs/Float64            (0=closed..1=open)
    in   /joint_states           sensor_msgs/JointState      (feedback + seeding)

No extra dependencies: the server is Python's stdlib ``http.server`` and the UI
is plain HTML/JS that polls a JSON state endpoint and POSTs commands. Run it in
its own terminal, then open the printed URL in a browser:

    ros2 run m1_control m1_web
    # then browse to http://localhost:8080

Safety: the base command is dead-man'd. The browser refreshes the velocity
while a drive key/button is held; if it stops refreshing (key released, tab
closed, network drop) the node zeroes the base after ``BASE_HOLD`` seconds.
"""

from __future__ import annotations

import json
import math
import os
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
PUBLISH_RATE = 60.0      # Hz the joint command / targets are streamed at

# Soft workspace clamp on the target point (base_link frame, m). Height (z) is
# intentionally unbounded: the operator may aim anywhere vertically and the
# controller's IK reaches as close as the joint limits physically allow.
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


def _finite(v) -> float:
    """Coerce to float and reject NaN/+-inf.

    A non-finite value must never reach a publisher (PoseStamped.position would
    carry inf/nan) nor the snapshot JSON (json.dumps would emit invalid
    ``Infinity``/``NaN`` tokens that break the browser's strict ``JSON.parse``).
    Raises ``ValueError`` so it propagates to do_POST's except block -> HTTP 400.
    """
    f = float(v)
    if not math.isfinite(f):
        raise ValueError("non-finite value")
    return f


class M1WebNode(Node):
    """Bridges the browser control panel to the m1_controller over ROS topics."""

    def __init__(self):
        super().__init__("m1_web")
        self.declare_parameter("urdf_path", _default_urdf_path())
        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 8080)

        urdf_path = self.get_parameter("urdf_path").value
        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)

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
        self._last_base_cmd = 0.0
        self._last_js = None

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
                p = float(pos)
                if not math.isfinite(p):
                    continue            # drop a glitched/garbage encoder sample
                self.q_meas[name] = p
            self._last_js = self._now()
            if self.reach is None:
                return
            for arm in ("left", "right"):
                # Seed each arm's target onto its current fingertip exactly
                # once, before the operator has touched it, so the arm holds
                # still on connect instead of snapping to a default point.
                # After that the target ONLY changes on explicit user input --
                # it is never re-synced to the live fingertip. (An idle arm
                # rides the shared lift when the other arm reaches; re-syncing
                # used to silently move that arm's stored target, which made
                # commanding one arm appear to move the other arm's target.)
                if self.seeded[arm]:
                    continue
                if all(j in self.q_meas for j in ARM_JOINTS[arm] + [LIFT_JOINT]):
                    try:
                        tip = self.reach.fingertip(arm, self.q_meas)
                        self.target[arm] = [float(tip[0]), float(tip[1]), float(tip[2])]
                        self.seeded[arm] = True
                    except Exception:  # noqa: BLE001
                        pass

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
            # `is not None`, not a truthiness test: a sim clock at t=0.0 makes
            # self._last_js == 0.0, which would falsely read as "never received".
            connected = self._last_js is not None and (now - self._last_js) < 1.0
            out = {
                "connected": connected,
                "base": dict(self.cmd_vel),
                "arms": {},
            }
            for arm in ("left", "right"):
                tip = None
                dist = None
                if self.reach is not None and all(
                    j in self.q_meas for j in ARM_JOINTS[arm] + [LIFT_JOINT]
                ):
                    try:
                        t = self.reach.fingertip(arm, self.q_meas)
                        tip = [round(float(t[0]), 3), round(float(t[1]), 3), round(float(t[2]), 3)]
                        tgt = np.array(self.target[arm])
                        dist = round(float(np.linalg.norm(np.array(t) - tgt)), 3)
                    except Exception:  # noqa: BLE001
                        pass
                out["arms"][arm] = {
                    "target": [round(v, 3) for v in self.target[arm]],
                    "grip": round(self.grip[arm], 3),
                    "fingertip": tip,
                    "dist": dist,
                    "seeded": self.seeded[arm],
                }
            lift = self.q_meas.get(LIFT_JOINT)
            out["lift"] = round(float(lift), 3) if lift is not None else None
            return out

    def apply(self, cmd: dict) -> dict:
        ctype = cmd.get("type")
        with self._lock:
            if ctype == "base":
                self.cmd_vel = {
                    "vx": _clamp(_finite(cmd.get("vx", 0.0)), -MAX_LINEAR, MAX_LINEAR),
                    "vy": _clamp(_finite(cmd.get("vy", 0.0)), -MAX_STRAFE, MAX_STRAFE),
                    "yaw": _clamp(_finite(cmd.get("yaw", 0.0)), -MAX_YAW, MAX_YAW),
                }
                self._last_base_cmd = self._now()
            elif ctype == "base_stop":
                self.cmd_vel = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}
                self._last_base_cmd = self._now()
            elif ctype == "target_set":
                arm = cmd.get("arm")
                if arm in self.target:
                    xyz = cmd.get("xyz", self.target[arm])
                    self._set_target(arm, xyz)
                    self.seeded[arm] = True
            elif ctype == "target_nudge":
                arm = cmd.get("arm")
                if arm in self.target:
                    d = cmd.get("dxyz", [0, 0, 0])
                    cur = self.target[arm]
                    self._set_target(arm, [cur[0] + d[0], cur[1] + d[1], cur[2] + d[2]])
                    self.seeded[arm] = True
            elif ctype == "gripper":
                arm = cmd.get("arm")
                if arm in self.grip:
                    self.grip[arm] = _clamp(_finite(cmd.get("value", 0.0)), 0.0, 1.0)
            elif ctype == "reset":
                self.seeded = {"left": False, "right": False}
                self.target = {a: list(DEFAULT_TARGET[a]) for a in ("left", "right")}
                self.grip = {"left": 0.0, "right": 0.0}
                self.cmd_vel = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}
            else:
                return {"ok": False, "error": f"unknown command '{ctype}'"}
        return {"ok": True}

    def _set_target(self, arm: str, xyz):
        # Validate finiteness BEFORE clamping: _clamp(nan, -inf, inf) returns
        # +inf, which would otherwise be published and poison the snapshot JSON.
        self.target[arm] = [
            _clamp(_finite(xyz[0]), *TARGET_LIMITS["x"]),
            _clamp(_finite(xyz[1]), *TARGET_LIMITS["y"]),
            _clamp(_finite(xyz[2]), *TARGET_LIMITS["z"]),
        ]


def _make_handler(node: M1WebNode):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence per-request logging
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, INDEX_HTML, "text/html; charset=utf-8")
            elif self.path == "/api/state":
                # allow_nan=False: belt-and-braces guard so a stray non-finite
                # value can never silently emit invalid `Infinity`/`NaN` JSON
                # (which would break the browser's strict JSON.parse).
                self._send(200, json.dumps(node.snapshot(), allow_nan=False))
            else:
                self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            if self.path != "/api/command":
                self._send(404, json.dumps({"error": "not found"}))
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                result = node.apply(payload)
                self._send(200, json.dumps(result))
            except Exception as exc:  # noqa: BLE001
                self._send(400, json.dumps({"ok": False, "error": str(exc)}))

    return Handler


class _Server(ThreadingHTTPServer):
    allow_reuse_address = True   # reuse sockets sitting in TIME_WAIT
    daemon_threads = True


def _bind_server(node: M1WebNode, handler, tries: int = 10):
    """Bind on the requested port, falling back to the next few if it is busy."""
    last_exc = None
    for port in range(node.port, node.port + tries):
        try:
            server = _Server((node.host, port), handler)
            node.port = port
            return server
        except OSError as exc:  # noqa: PERF203
            last_exc = exc
            if port + 1 < node.port + tries:
                node.get_logger().warn(f"port {port} busy, trying {port + 1}…")
            else:
                node.get_logger().warn(f"port {port} busy")
    raise last_exc


def main(args=None):
    rclpy.init(args=args)
    node = M1WebNode()

    try:
        server = _bind_server(node, _make_handler(node))
    except OSError as exc:
        node.get_logger().error(
            f"could not open a web port near {node.port} ({exc}). "
            "Another m1_web may be running. Find it with "
            f"`ss -ltnp | grep {node.port}` (or `lsof -i :{node.port}`), `kill` "
            "that PID, or pick another port: "
            "`ros2 run m1_control m1_web --ros-args -p port:=9000`.")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        return

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    shown = "localhost" if node.host in ("0.0.0.0", "") else node.host
    node.get_logger().info(
        f"M1 web panel running -> http://{shown}:{node.port}  "
        f"(drives sim and the real robot; open it in a browser)")

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
# Single-page control panel (served at "/"). Plain HTML/CSS/JS, no build step.
# ---------------------------------------------------------------------------
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>M1 Control Panel</title>
<style>
  :root {
    /* Anthropic-inspired warm palette */
    --bg:#f0eee6; --panel:#f6f4ec; --panel2:#ebe8dd; --line:#dcd7c8;
    --ink:#181613; --txt:#181613; --muted:#73706a; --accent:#c96442;
    --accent-ink:#b4543a; --good:#4f7a4a; --bad:#b4432f; --warn:#b8862f;
    --serif:'Tiempos Headline','Iowan Old Style',Georgia,'Times New Roman',serif;
    --sans:'Styrene B',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  }
  * { box-sizing:border-box; }
  body { margin:0; font-family:var(--sans); background:var(--bg); color:var(--txt);
         -webkit-font-smoothing:antialiased; }
  header { display:flex; align-items:center; gap:14px; padding:18px 28px;
           border-bottom:1px solid var(--line); background:var(--bg); }
  .logo { width:26px; height:26px; flex:none; }
  header h1 { font-family:var(--serif); font-size:21px; margin:0; font-weight:600;
              letter-spacing:-.01em; color:var(--ink); }
  .status { display:flex; align-items:center; gap:8px; margin-left:auto;
            font-size:13px; color:var(--muted); }
  .dot { width:9px; height:9px; border-radius:50%; background:var(--bad); }
  .dot.on { background:var(--good); }
  main { display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr));
         gap:20px; padding:26px 28px; max-width:1140px; margin:0 auto; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px;
          padding:20px 22px; }
  .card h2 { font-family:var(--serif); font-size:19px; margin:0 0 14px; color:var(--ink);
             font-weight:600; letter-spacing:-.01em; }
  .row { display:flex; align-items:center; gap:8px; margin:8px 0; }
  .row label { width:78px; color:var(--muted); font-size:13px; }
  button { background:#fbfaf5; color:var(--ink); border:1px solid var(--line);
           border-radius:6px; padding:9px 12px; font-size:14px; cursor:pointer;
           font-family:var(--sans); user-select:none;
           transition:background .1s,border-color .1s,color .1s; }
  button:hover { border-color:var(--accent); }
  button:active, button.held { background:var(--accent); border-color:var(--accent);
           color:#fff; }
  button.stop { background:var(--ink); border-color:var(--ink); color:#f0eee6; }
  button.stop:hover { background:var(--bad); border-color:var(--bad); }
  .pad { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; max-width:250px; }
  .pad button { padding:14px 0; }
  .grid2 { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }
  input[type=number]{ width:72px; background:#fbfaf5; color:var(--ink);
           border:1px solid var(--line); border-radius:6px; padding:6px;
           font-family:var(--sans); }
  input[type=range]{ flex:1; accent-color:var(--accent); }
  .val { font-variant-numeric:tabular-nums; color:var(--ink); }
  .muted { color:var(--muted); font-size:12px; }
  code { background:var(--panel2); border-radius:4px; padding:1px 5px; font-size:.92em; }
  .pill { font-size:11px; padding:3px 9px; border-radius:999px; background:var(--panel2);
          border:1px solid var(--line); color:var(--muted); text-transform:uppercase;
          letter-spacing:.05em; }
  .pill.active { background:var(--accent); border-color:var(--accent); color:#fff; }
  .arm-head { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
  .arm-head h2 { text-transform:capitalize; }
  .readout { font-variant-numeric:tabular-nums; font-size:13px; color:var(--muted);
             margin-top:8px; line-height:1.6; border-top:1px solid var(--line);
             padding-top:8px; }
  .span { grid-column:1/-1; }
  kbd { background:#fbfaf5; border:1px solid var(--line); border-bottom-width:2px;
        border-radius:5px; padding:1px 6px; font-size:12px; color:var(--ink); }
  footer { text-align:center; color:var(--muted); font-size:12px; padding:14px 0 30px;
           font-family:var(--serif); font-style:italic; }
</style>
</head>
<body>
<header>
  <svg class="logo" viewBox="0 0 100 100" aria-hidden="true">
    <path fill="#181613" d="M34 18 L8 82 H24 L29 68 H53 L58 82 H74 L48 18 Z M34.5 54 L41 36 L47.5 54 Z"/>
    <path fill="#c96442" d="M66 18 L92 82 H77 L51 18 Z"/>
  </svg>
  <h1>M1 Control Panel</h1>
  <span class="muted">drives simulation &amp; the real robot</span>
  <div class="status"><span id="dot" class="dot"></span><span id="statusText">connecting…</span></div>
</header>

<main>
  <!-- BASE -->
  <section class="card">
    <h2>Mobile base (swerve)</h2>
    <div class="pad">
      <button data-base="strafe_l">⟸ strafe</button>
      <button data-base="fwd">▲ fwd</button>
      <button data-base="strafe_r">strafe ⟹</button>
      <button data-base="turn_l">◀ turn</button>
      <button class="stop" id="baseStop">■ stop</button>
      <button data-base="turn_r">turn ▶</button>
      <span></span>
      <button data-base="back">▼ back</button>
      <span></span>
    </div>
    <div class="readout" id="baseRead">vx 0.00  vy 0.00  yaw 0.00</div>
    <div class="muted">Hold a button to drive. Keys: <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> drive/turn, <kbd>Q</kbd><kbd>E</kbd> strafe, <kbd>Space</kbd> stop.</div>
  </section>

  <!-- GLOBAL -->
  <section class="card">
    <h2>Session</h2>
    <div class="row"><span class="muted">Robot feedback (<code>/joint_states</code>)</span></div>
    <div class="readout" id="globalRead">lift: –</div>
    <div class="row" style="margin-top:14px;">
      <button id="resetBtn">Reset targets &amp; stop</button>
      <button class="stop" id="estop">E-STOP base</button>
    </div>
    <div class="muted" style="margin-top:10px;">If the dot is red, nothing will move: start the
      simulator (or real robot) so <code>/joint_states</code> is published. The arm only begins
      tracking once feedback arrives.</div>
  </section>
</main>

<main id="arms"></main>
<footer id="foot">M1 web panel</footer>

<script>
const armNames = ["left","right"];
let stepSize = 0.05;

async function postCmd(obj){
  try { await fetch("/api/command",{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify(obj)}); } catch(e){}
}

/* ---------- base driving (held buttons + keyboard, dead-man refresh) ------- */
const keysDown = new Set();
const heldBtns = new Set();
function baseVector(){
  let vx=0,vy=0,yaw=0;
  const L=0.5,S=0.4,Y=1.0;
  if(keysDown.has("w")||heldBtns.has("fwd"))      vx+=L;
  if(keysDown.has("s")||heldBtns.has("back"))     vx-=L;
  if(keysDown.has("a")||heldBtns.has("turn_l"))   yaw+=Y;
  if(keysDown.has("d")||heldBtns.has("turn_r"))   yaw-=Y;
  if(keysDown.has("q")||heldBtns.has("strafe_l")) vy+=S;
  if(keysDown.has("e")||heldBtns.has("strafe_r")) vy-=S;
  return {vx,vy,yaw};
}
let baseActive=false;
setInterval(()=>{
  const anyHeld = keysDown.size>0 || heldBtns.size>0;
  if(anyHeld){ const v=baseVector(); postCmd({type:"base",...v}); baseActive=true; }
  else if(baseActive){ postCmd({type:"base_stop"}); baseActive=false; }
}, 100);

function bindHold(btn,name){
  const down=e=>{e.preventDefault(); heldBtns.add(name); btn.classList.add("held");};
  const up=e=>{e.preventDefault(); heldBtns.delete(name); btn.classList.remove("held");};
  btn.addEventListener("mousedown",down); btn.addEventListener("mouseup",up);
  btn.addEventListener("mouseleave",up);
  btn.addEventListener("touchstart",down,{passive:false});
  btn.addEventListener("touchend",up);
  // A cancelled touch / pointer (gesture interrupted, finger dragged off,
  // OS takeover) otherwise leaves the name stuck in heldBtns and the base
  // keeps driving (and keeps refreshing the server dead-man). Bind cancels too.
  btn.addEventListener("touchcancel",up,{passive:false});
  btn.addEventListener("pointercancel",up);
}
document.querySelectorAll("[data-base]").forEach(b=>bindHold(b,b.dataset.base));
document.getElementById("baseStop").onclick=()=>{heldBtns.clear();postCmd({type:"base_stop"});};
document.getElementById("estop").onclick=()=>{keysDown.clear();heldBtns.clear();postCmd({type:"base_stop"});};
document.getElementById("resetBtn").onclick=()=>postCmd({type:"reset"});

window.addEventListener("keydown",e=>{
  const tag=(e.target&&e.target.tagName)||""; if(tag==="INPUT"||tag==="TEXTAREA") return;
  const k=e.key.toLowerCase();
  if(["w","a","s","d","q","e"].includes(k)){ keysDown.add(k); e.preventDefault(); }
  else if(k===" "){ keysDown.clear(); heldBtns.clear(); postCmd({type:"base_stop"}); e.preventDefault(); }
});
window.addEventListener("keyup",e=>keysDown.delete(e.key.toLowerCase()));

/* Lost focus / hidden tab never delivers keyup/touchend, so a held key/button
   would stay stuck and the base would keep driving (and keep refreshing the
   server BASE_HOLD dead-man). Clear all held state and stop the base. */
window.addEventListener("blur",()=>{keysDown.clear();heldBtns.clear();postCmd({type:"base_stop"});});
document.addEventListener("visibilitychange",()=>{ if(document.hidden){keysDown.clear();heldBtns.clear();postCmd({type:"base_stop"});} });

/* ---------- arm cards (built dynamically) ---------------------------------- */
const armsRoot=document.getElementById("arms");
function armCard(arm){
  const c=document.createElement("section"); c.className="card";
  c.innerHTML=`
    <div class="arm-head"><h2 style="margin:0">${arm} arm</h2>
      <span class="pill" id="${arm}-seed">target</span></div>
    <div class="grid2">
      <button data-nudge="${arm},0,+">+X (fwd)</button>
      <button data-nudge="${arm},1,+">+Y (left)</button>
      <button data-nudge="${arm},2,+">+Z (up)</button>
      <button data-nudge="${arm},0,-">−X (back)</button>
      <button data-nudge="${arm},1,-">−Y (right)</button>
      <button data-nudge="${arm},2,-">−Z (down)</button>
    </div>
    <div class="readout" id="${arm}-read">target –</div>
    <div class="row span" style="margin-top:10px">
      <label>Go to</label>
      <input type="number" step="0.01" id="${arm}-x" placeholder="x"/>
      <input type="number" step="0.01" id="${arm}-y" placeholder="y"/>
      <input type="number" step="0.01" id="${arm}-z" placeholder="z"/>
      <button id="${arm}-go">Send</button>
    </div>
    <div class="row"><label>Gripper</label>
      <input type="range" min="0" max="1" step="0.01" id="${arm}-grip"/>
      <span class="val" id="${arm}-gripv">0.00</span></div>
  `;
  armsRoot.appendChild(c);

  c.querySelectorAll("[data-nudge]").forEach(b=>{
    b.onclick=()=>{ const [a,i,s]=b.dataset.nudge.split(",");
      const d=[0,0,0]; d[+i]=(s==="+"?1:-1)*stepSize; postCmd({type:"target_nudge",arm:a,dxyz:d}); };
  });
  c.querySelector(`#${arm}-go`).onclick=()=>{
    const x=parseFloat(document.getElementById(`${arm}-x`).value);
    const y=parseFloat(document.getElementById(`${arm}-y`).value);
    const z=parseFloat(document.getElementById(`${arm}-z`).value);
    if([x,y,z].some(isNaN)) return;
    postCmd({type:"target_set",arm:arm,xyz:[x,y,z]});
  };
  const slider=c.querySelector(`#${arm}-grip`);
  slider.addEventListener("input",()=>{ postCmd({type:"gripper",arm:arm,value:parseFloat(slider.value)});
    document.getElementById(`${arm}-gripv`).textContent=parseFloat(slider.value).toFixed(2); });
}
armNames.forEach(armCard);

/* step size control appended to the base card footer */
(function(){
  const sel=document.createElement("div"); sel.className="row"; sel.style.marginTop="10px";
  sel.innerHTML=`<label>Step</label>`;
  [[0.01,"1 cm"],[0.05,"5 cm"],[0.1,"10 cm"]].forEach(([v,t])=>{
    const b=document.createElement("button"); b.textContent=t;
    if(v===stepSize) b.classList.add("held");
    b.onclick=()=>{ stepSize=v; sel.querySelectorAll("button").forEach(x=>x.classList.remove("held")); b.classList.add("held"); };
    sel.appendChild(b);
  });
  document.querySelectorAll(".card")[1].appendChild(sel);
})();

/* ---------- live state polling -------------------------------------------- */
const gripTouched={left:false,right:false};
armNames.forEach(a=>document.getElementById(`${a}-grip`).addEventListener("pointerdown",()=>gripTouched[a]=true));
armNames.forEach(a=>document.getElementById(`${a}-grip`).addEventListener("pointerup",()=>gripTouched[a]=false));

async function poll(){
  try{
    const s=await (await fetch("/api/state")).json();
    const dot=document.getElementById("dot"), st=document.getElementById("statusText");
    if(s.connected){ dot.classList.add("on"); st.textContent="robot feedback OK"; }
    else { dot.classList.remove("on"); st.textContent="no /joint_states — start sim/robot"; }
    document.getElementById("baseRead").textContent=
      `vx ${s.base.vx.toFixed(2)}  vy ${s.base.vy.toFixed(2)}  yaw ${s.base.yaw.toFixed(2)}`;
    document.getElementById("globalRead").textContent=
      "lift: "+(s.lift==null?"–":s.lift.toFixed(3)+" m");
    for(const a of armNames){
      const arm=s.arms[a]; if(!arm) continue;
      const t=arm.target;
      const tip = arm.fingertip ? `  tip [${arm.fingertip.join(", ")}]` : "  tip –";
      const d = arm.dist==null ? "" : `  dist ${(arm.dist*100).toFixed(1)} cm`;
      document.getElementById(`${a}-read`).textContent=
        `target [${t.join(", ")}]${tip}${d}`;
      document.getElementById(`${a}-seed`).className="pill"+(arm.seeded?" active":"");
      document.getElementById(`${a}-seed`).textContent=arm.seeded?"tracking":"default";
      if(!gripTouched[a]){
        const g=document.getElementById(`${a}-grip`); g.value=arm.grip;
        document.getElementById(`${a}-gripv`).textContent=arm.grip.toFixed(2);
      }
    }
  }catch(e){
    document.getElementById("dot").classList.remove("on");
    document.getElementById("statusText").textContent="panel disconnected";
  }
}
setInterval(poll,150); poll();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()

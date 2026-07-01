"""M1 hardware configuration / test web node (``m1_hwconfig``).

A maintenance-mode console for the Damiao CAN motors of the arms + lift: scan the
bus and assign motor IDs, edit per-joint limits, jog/test motors with a dead-man,
calibrate zero, and watch live telemetry -- all from a browser, no extra deps
(stdlib ``http.server`` + an embedded/served HTML page), themed to match
``m1_control/web_node.py`` (warm cream / clay).

Bus ownership is mutually exclusive with live ros2_control (see the deployment
design's safety section):

* **maintenance** mode -- this node owns the CAN bus through a
  :class:`~m1_can_tools.motor_bus.MotorBus`; every write/jog/zero/assign endpoint
  is allowed.
* **run** mode -- ros2_control owns the bus; this node refuses all writes and
  the page is read-only telemetry (from ``/joint_states``).

Architecture mirrors ``web_node`` / ``quest_node``: a stdlib
``ThreadingHTTPServer`` in a daemon thread, an ``rclpy`` node, and a jog dead-man
(``JOG_HOLD``, the analogue of ``BASE_HOLD``) so motion only continues while the
browser keeps refreshing. **The HTTP request handlers delegate to plain
``api_*`` methods that return ``(status_code, body_dict)``**, so the whole data
path is unit-testable headless against a ``MotorBus(FakeTransport)`` with no
server and no hardware (see ``test/test_hwconfig_datapath.py``).
"""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional, Tuple

from m1_can_tools import dm_protocol as dm
from m1_can_tools.motor_bus import MotorBus, load_map, save_map
from m1_can_tools.transport import make_transport

# --- Safety / timing --------------------------------------------------------
JOG_HOLD = 0.5          # s without a jog refresh before a held jog is dropped
DEADMAN_RATE = 20.0     # Hz the jog dead-man / telemetry loop ticks
DEFAULT_JOG_KP = 10.0
DEFAULT_JOG_KD = 1.0

# --- Default MIT-mode impedance gains per DM model --------------------------
# The C++ M1SystemInterface plugin reads kp/kd from the persisted motor map and
# refuses to open the CAN bus if any commanded joint has kp == 0 (limp-arm
# guard). The page may not always supply kp/kd, so api_map persists a sensible
# NON-ZERO default keyed by motor model, mirroring the conventions in
# ``m1_hardware/config/control_gains.yaml`` (lift/proximal DM8009 kp70/kd2.5,
# DM4340 kp70/kd2, DM4310 wrist kp10/kd0.7, gripper DMH3510 kp5/kd0.1).
# These are bring-up defaults -- the operator MUST tune per control_gains.yaml.
_MODEL_DEFAULT_GAINS: Dict[str, Tuple[float, float]] = {
    "DM8009": (70.0, 2.5),
    "DM8006": (70.0, 2.5),
    "DM4340": (70.0, 2.0),
    "DM4340_48V": (70.0, 2.0),
    "DM6006": (40.0, 1.5),
    "DM4310": (10.0, 0.7),
    "DM4310_48V": (10.0, 0.7),
    "DM3507": (10.0, 0.7),
    "DMH3510": (5.0, 0.1),
    "DMH6215": (5.0, 0.1),
    "DMG6220": (5.0, 0.1),
    "DM10010": (120.0, 3.0),
    "DM10010L": (120.0, 3.0),
}
# Safe non-zero fallback for any model not in the table above.
_FALLBACK_KP, _FALLBACK_KD = 10.0, 1.0


def _default_gains(model: str) -> Tuple[float, float]:
    """Sensible NON-ZERO (kp, kd) bring-up gains for a DM *model*.

    Keeps a commanded joint from defaulting to kp==0 (which would trip the C++
    plugin's limp-arm guard and refuse the bus). Operator should tune these per
    ``m1_hardware/config/control_gains.yaml``.
    """
    return _MODEL_DEFAULT_GAINS.get(model, (_FALLBACK_KP, _FALLBACK_KD))

# Where the limits editor writes (consumed by ros2_control in Phase 1+).
DEFAULT_LIMITS_PATH = "m1_joint_limits.yaml"

ApiResult = Tuple[int, dict]


class M1HwConfigNode:
    """The data-path owner behind the config web page.

    Subclasses :class:`rclpy.node.Node` at runtime via :meth:`main`; constructed
    bare (``__new__`` + :meth:`_init_state`) in tests so the data path runs with
    no ROS init / DDS. The ``api_*`` methods are the HTTP API; they each return
    ``(status_code, body)`` and never touch the socket.
    """

    # --- construction ------------------------------------------------------
    def _init_state(
        self,
        bus: MotorBus,
        motor_map: Dict[str, dict],
        mode: str = "maintenance",
        limits_path: str = DEFAULT_LIMITS_PATH,
        map_path: str = "",
    ) -> None:
        """Wire up the shared state. Used by both ``main`` and the tests."""
        self._lock = threading.Lock()
        self.bus = bus
        self.motor_map = motor_map
        self.mode = mode
        self.limits_path = limits_path
        self.map_path = map_path
        self.bus_ok = bus is not None
        # Held-jog dead-man: joint -> (pos, vel, kp, kd, last_refresh_time) | None.
        self._jog_active: Dict[str, Optional[tuple]] = {j: None for j in motor_map}
        # Last telemetry per joint (for run-mode read-only / inventory display).
        self._telem: Dict[str, dict] = {}

    def _now(self) -> float:
        """Monotonic seconds. Tests stub via ``self._clock[0]``; the ROS node
        overrides this with the node clock."""
        clk = getattr(self, "_clock", None)
        if clk is not None:
            return float(clk[0])
        return 0.0

    # --- mode guard --------------------------------------------------------
    def _require_maintenance(self) -> Optional[ApiResult]:
        if self.mode != "maintenance":
            return (403, {"ok": False, "error": "robot is in run mode; "
                          "writes are disabled (ros2_control owns the bus)"})
        return None

    # --- API: state / mode -------------------------------------------------
    def api_state(self, _payload: dict) -> ApiResult:
        """Snapshot: mode, bus health, per-motor telemetry, and the map."""
        with self._lock:
            motors = []
            for joint, info in self.motor_map.items():
                t = self._telem.get(joint, {})
                motors.append({
                    "joint": joint,
                    "id": info["id"],
                    "master_id": info.get("master_id", dm.master_id(info["id"])),
                    "model": info["model"],
                    "pos": t.get("pos"),
                    "vel": t.get("vel"),
                    "torque": t.get("torque"),
                    "t_mos": t.get("t_mos"),
                    "t_rotor": t.get("t_rotor"),
                    "err": t.get("err"),
                    "enabled": self._jog_active.get(joint) is not None,
                })
            return (200, {
                "mode": self.mode,
                "bus_ok": self.bus_ok,
                "motors": motors,
                "map": self.motor_map,
            })

    def api_mode(self, payload: dict) -> ApiResult:
        """Switch maintenance<->run. Leaving maintenance disables all motors."""
        mode = payload.get("mode")
        if mode not in ("maintenance", "run"):
            return (400, {"ok": False, "error": f"bad mode {mode!r}"})
        with self._lock:
            if mode == "run" and self.mode == "maintenance":
                # Hand the bus to ros2_control: stop everything we were holding.
                for joint in list(self._jog_active):
                    self._jog_active[joint] = None
                try:
                    self.bus.disable_all()
                except Exception:  # noqa: BLE001
                    pass
            self.mode = mode
        return (200, {"ok": True, "mode": mode})

    # --- API: scan / assign / map ------------------------------------------
    def api_scan(self, payload: dict) -> ApiResult:
        """Ping a candidate id range and list the responders."""
        guard = self._require_maintenance()
        if guard:
            return guard
        lo = int(payload.get("from", 1))
        hi = int(payload.get("to", lo))
        with self._lock:
            motors = self.bus.scan(range(lo, hi + 1))
        return (200, {"ok": True, "motors": motors})

    def api_assign(self, payload: dict) -> ApiResult:
        """Set a motor's CAN/master id (placeholder: param-write frame).

        On the real bus this writes the id parameter (arb ``0x7FF``) and re-scans;
        offline it just records the new ids in the map for the responder.
        """
        guard = self._require_maintenance()
        if guard:
            return guard
        old_id = payload.get("old_id")
        new_id = payload.get("new_id")
        master_id = payload.get("master_id", None)
        if old_id is None or new_id is None:
            return (400, {"ok": False, "error": "old_id and new_id required"})
        # Validate the requested CAN id (standard 11-bit range; id 0 reserved).
        err = self._validate_ids(new_id, master_id)
        if err is not None:
            return err
        with self._lock:
            for info in self.motor_map.values():
                if info["id"] == old_id:
                    info["id"] = int(new_id)
                    # A falsy/0 master_id (the page sends +('') === 0 when blank)
                    # is reserved/invalid -> treat as absent and default to
                    # id + 0x10.
                    info["master_id"] = (int(master_id) if master_id
                                         else dm.master_id(int(new_id)))
                    break
        return (200, {"ok": True})

    # --- id validation -----------------------------------------------------
    _CAN_ID_MIN = 1
    _CAN_ID_MAX = 0x7FF      # standard 11-bit CAN identifier

    def _validate_ids(self, new_id, master_id) -> Optional[ApiResult]:
        """Reject an out-of-range CAN id (and explicit master id) with a 400.

        ``new_id`` must be in ``[1, 0x7FF]`` (standard CAN; id 0 reserved). A
        ``master_id`` is optional, but if supplied (non-falsy) it must be in the
        same range -- a falsy/0 master_id is handled as ABSENT by the caller.
        Returns an ``ApiResult`` to return on failure, or ``None`` if valid.
        """
        try:
            nid = int(new_id)
        except (TypeError, ValueError):
            return (400, {"ok": False, "error": f"id must be an integer, got {new_id!r}"})
        if nid < self._CAN_ID_MIN or nid > self._CAN_ID_MAX:
            return (400, {"ok": False, "error":
                          f"id {nid} out of CAN range [{self._CAN_ID_MIN}, "
                          f"{self._CAN_ID_MAX}]"})
        if master_id:   # falsy/0/None -> treated as absent (defaulted elsewhere)
            try:
                mid = int(master_id)
            except (TypeError, ValueError):
                return (400, {"ok": False, "error":
                              f"master_id must be an integer, got {master_id!r}"})
            if mid < self._CAN_ID_MIN or mid > self._CAN_ID_MAX:
                return (400, {"ok": False, "error":
                              f"master_id {mid} out of CAN range "
                              f"[{self._CAN_ID_MIN}, {self._CAN_ID_MAX}]"})
        else:
            # No explicit master_id -> the caller derives id + 0x10. Reject an id
            # whose auto-derived master would overflow the CAN range (id > 0x7EF),
            # so a boundary id can't silently persist a master_id > 0x7FF.
            derived = dm.master_id(nid)
            if derived > self._CAN_ID_MAX:
                return (400, {"ok": False, "error":
                              f"id {nid} too large: auto master_id {derived} exceeds "
                              f"{self._CAN_ID_MAX} (supply an explicit master_id, or "
                              f"use id <= {self._CAN_ID_MAX - 0x10})"})
        return None

    def api_map(self, payload: dict) -> ApiResult:
        """Map a logical joint -> motor (id/model/limits/dir/offset) and persist."""
        guard = self._require_maintenance()
        if guard:
            return guard
        joint = payload.get("joint")
        if not joint:
            return (400, {"ok": False, "error": "joint required"})
        model = payload.get("model", "DM4310")
        if model not in dm.LIMITS:
            return (400, {"ok": False, "error": f"unknown model {model!r}"})
        with self._lock:
            entry = self.motor_map.get(joint, {})
            new_id = payload.get("id", entry.get("id", 0))
            # Validate the CAN id (and an explicit master_id) before persisting,
            # so the derived master_id is never garbage.
            err = self._validate_ids(new_id, payload.get("master_id"))
            if err is not None:
                return err
            entry["id"] = int(new_id)
            # A falsy/0 master_id (page sends +('') === 0 when blank) is
            # reserved/invalid -> treat as absent and default to id + 0x10.
            mid = payload.get("master_id", entry.get("master_id"))
            entry["master_id"] = int(mid) if mid else dm.master_id(entry["id"])
            entry["model"] = model
            entry.setdefault("soft_limits", {"pos": [-1.0, 1.0], "vel": 1.0, "effort": 1.0})
            # Persist dir/offset with payload-takes-precedence (NOT setdefault,
            # which silently drops a re-map's new value). Defaults to current
            # then 1 / 0.0.
            entry["dir"] = int(payload.get("dir", entry.get("dir", 1)))
            entry["offset"] = float(payload.get("offset", entry.get("offset", 0.0)))
            # Persist NON-ZERO kp/kd: the C++ plugin defaults missing gains to
            # 0.0 and refuses the bus when a commanded joint has kp == 0
            # (limp-arm guard). Default per-model from control_gains.yaml
            # conventions; operator should tune. (See _default_gains.)
            d_kp, d_kd = _default_gains(model)
            entry["kp"] = float(payload.get("kp", entry.get("kp", d_kp)))
            entry["kd"] = float(payload.get("kd", entry.get("kd", d_kd)))
            self.motor_map[joint] = entry
            self._jog_active.setdefault(joint, None)
            if self.map_path:
                try:
                    save_map(self.map_path, self.motor_map)
                except Exception as exc:  # noqa: BLE001
                    return (500, {"ok": False, "error": f"map save failed: {exc}"})
        return (200, {"ok": True})

    # --- API: limits -------------------------------------------------------
    def api_limits(self, payload: dict) -> ApiResult:
        """Edit a joint's limits: validate vs model max, update soft limits,
        write ``m1_joint_limits.yaml``."""
        guard = self._require_maintenance()
        if guard:
            return guard
        joint = payload.get("joint")
        if joint not in self.motor_map:
            return (400, {"ok": False, "error": f"unknown joint {joint!r}"})
        info = self.motor_map[joint]
        p_max, v_max, t_max = dm.limits(info["model"])

        sl = info.get("soft_limits", {})
        pos = payload.get("pos", sl.get("pos", [-p_max, p_max]))
        vel = float(payload.get("vel", sl.get("vel", v_max)))
        eff = float(payload.get("effort", sl.get("effort", t_max)))
        lo, hi = float(pos[0]), float(pos[1])

        # Validate against the per-model [P, V, T]MAX.
        if lo < -p_max or hi > p_max or lo > hi:
            return (400, {"ok": False, "error":
                          f"pos limits out of model range [-{p_max}, {p_max}]"})
        if vel < 0 or vel > v_max:
            return (400, {"ok": False, "error": f"vel exceeds model V_MAX {v_max}"})
        if eff < 0 or eff > t_max:
            return (400, {"ok": False, "error": f"effort exceeds model T_MAX {t_max}"})

        with self._lock:
            info["soft_limits"] = {"pos": [lo, hi], "vel": vel, "effort": eff}
            self._write_limits_yaml()
        return (200, {"ok": True})

    def _write_limits_yaml(self) -> None:
        """Write the full ros2_control-style ``joint_limits.yaml``."""
        import yaml
        doc = {"joint_limits": {}}
        for joint, info in self.motor_map.items():
            sl = info.get("soft_limits", {})
            pos = sl.get("pos", [None, None])
            doc["joint_limits"][joint] = {
                "has_position_limits": pos[0] is not None,
                "min_position": pos[0],
                "max_position": pos[1],
                "has_velocity_limits": "vel" in sl,
                "max_velocity": sl.get("vel"),
                "has_effort_limits": "effort" in sl,
                "max_effort": sl.get("effort"),
            }
        with open(self.limits_path, "w") as fh:
            yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False)

    # --- API: enable / zero / jog ------------------------------------------
    def api_enable(self, payload: dict) -> ApiResult:
        guard = self._require_maintenance()
        if guard:
            return guard
        joint = payload.get("joint")
        if joint not in self.motor_map:
            return (400, {"ok": False, "error": f"unknown joint {joint!r}"})
        on = bool(payload.get("on", True))
        with self._lock:
            if on:
                self.bus.enable(joint)
            else:
                self.bus.disable(joint)
                self._jog_active[joint] = None
        return (200, {"ok": True, "enabled": on})

    def api_zero(self, payload: dict) -> ApiResult:
        guard = self._require_maintenance()
        if guard:
            return guard
        joint = payload.get("joint")
        if joint not in self.motor_map:
            return (400, {"ok": False, "error": f"unknown joint {joint!r}"})
        with self._lock:
            self.bus.set_zero(joint)
        return (200, {"ok": True})

    def api_jog(self, payload: dict) -> ApiResult:
        """Send one clamped MIT jog frame; requires ``hold:true`` (dead-man)."""
        guard = self._require_maintenance()
        if guard:
            return guard
        joint = payload.get("joint")
        if joint not in self.motor_map:
            return (400, {"ok": False, "error": f"unknown joint {joint!r}"})
        if not payload.get("hold", False):
            # No live dead-man -> refuse to start motion.
            return (400, {"ok": False, "error": "jog requires hold:true (dead-man)"})
        pos = float(payload.get("pos", 0.0))
        vel = float(payload.get("vel", 0.0))
        kp = float(payload.get("kp", DEFAULT_JOG_KP))
        kd = float(payload.get("kd", DEFAULT_JOG_KD))
        with self._lock:
            self.bus.jog(joint, pos, vel=vel, kp=kp, kd=kd)
            self._jog_active[joint] = (pos, vel, kp, kd, self._now())
        return (200, {"ok": True})

    def tick_deadman(self) -> None:
        """Re-send held jogs; drop (and disable) ones whose refresh went stale.

        Called at ``DEADMAN_RATE`` from the ROS timer. A jog kept alive by the
        browser (each ``api_jog`` refreshes its timestamp) is re-commanded; one
        whose browser stopped refreshing for ``JOG_HOLD`` s is disabled so the
        motor halts -- the maintenance analogue of ``BASE_HOLD``.
        """
        now = self._now()
        with self._lock:
            for joint, st in list(self._jog_active.items()):
                if st is None:
                    continue
                pos, vel, kp, kd, ts = st
                if now - ts > JOG_HOLD:
                    self._jog_active[joint] = None
                    try:
                        self.bus.disable(joint)
                    except Exception:  # noqa: BLE001
                        pass
                else:
                    try:
                        self.bus.jog(joint, pos, vel=vel, kp=kp, kd=kd)
                    except Exception:  # noqa: BLE001
                        pass

    def poll_telemetry(self) -> None:
        """Drain feedback frames and refresh ``self._telem`` (maintenance only)."""
        if self.mode != "maintenance" or self.bus is None:
            return
        for joint in self.motor_map:
            with self._lock:
                try:
                    fb = self.bus.telemetry(joint, timeout=0.0)
                except Exception:  # noqa: BLE001
                    fb = None
                if fb is not None:
                    self._telem[joint] = fb


# ---------------------------------------------------------------------------
# HTTP handler -- thin dispatch onto the api_* methods above.
# ---------------------------------------------------------------------------
_POST_ROUTES = {
    "/api/scan": "api_scan",
    "/api/assign": "api_assign",
    "/api/map": "api_map",
    "/api/limits": "api_limits",
    "/api/jog": "api_jog",
    "/api/enable": "api_enable",
    "/api/zero": "api_zero",
    "/api/mode": "api_mode",
}


def _make_handler(node: M1HwConfigNode, html: bytes):
    class Handler(BaseHTTPRequestHandler):
        disable_nagle_algorithm = True   # TCP_NODELAY (snappy on a real network)

        def log_message(self, *args):    # silence per-request logging
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html", "/hwconfig.html"):
                self._send(200, html, "text/html; charset=utf-8")
            elif self.path == "/api/state":
                code, body = node.api_state({})
                self._send(code, json.dumps(body))
            else:
                self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            method = _POST_ROUTES.get(self.path)
            if method is None:
                self._send(404, json.dumps({"error": "not found"}))
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                code, body = getattr(node, method)(payload)
                self._send(code, json.dumps(body))
            except Exception as exc:  # noqa: BLE001
                self._send(400, json.dumps({"ok": False, "error": str(exc)}))

    return Handler


class _Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _load_html() -> bytes:
    """Read the served HTML page from the package's ``web/`` dir."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "web", "hwconfig.html")
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return b"<!DOCTYPE html><h1>M1 hwconfig</h1><p>page asset missing</p>"


# ---------------------------------------------------------------------------
# ROS entrypoint -- the bare data-path class above is wrapped in a rclpy Node.
# ---------------------------------------------------------------------------
def main(args=None):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    class _RosNode(Node, M1HwConfigNode):
        def __init__(self):
            Node.__init__(self, "m1_hwconfig")
            self.declare_parameter("host", "0.0.0.0")
            self.declare_parameter("port", 8090)
            self.declare_parameter("mode", "maintenance")
            self.declare_parameter("transport", "fake")  # fake|socketcan|serial
            self.declare_parameter("can_channel", "can0")
            self.declare_parameter("can_fd", False)
            self.declare_parameter("serial_dev", "/dev/ttyACM0")
            self.declare_parameter("motor_map", "")
            self.declare_parameter("limits_path", DEFAULT_LIMITS_PATH)

            self.host = self.get_parameter("host").value
            self.port = int(self.get_parameter("port").value)
            mode = self.get_parameter("mode").value
            map_path = self.get_parameter("motor_map").value
            limits_path = self.get_parameter("limits_path").value

            motor_map = {}
            if map_path and os.path.isfile(map_path):
                try:
                    motor_map = load_map(map_path)
                    self.get_logger().info(f"loaded motor map from {map_path}")
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warn(f"motor map load failed: {exc}")

            # Build the transport only in maintenance mode (run mode reads
            # /joint_states instead of owning the bus).
            bus = None
            if mode == "maintenance":
                kind = self.get_parameter("transport").value
                spec = {
                    "kind": kind,
                    "channel": self.get_parameter("can_channel").value,
                    "fd": bool(self.get_parameter("can_fd").value),
                    "dev": self.get_parameter("serial_dev").value,
                }
                if kind == "sim":
                    # Seed virtual motors from the map (small distinct start poses
                    # so the demo page isn't all-zeros), keyed by slave id.
                    motors = {}
                    for i, (joint, info) in enumerate(motor_map.items()):
                        motors[int(info["id"])] = {
                            "master_id": int(info.get(
                                "master_id", dm.master_id(int(info["id"])))),
                            "model": info.get("model", "DM4310"),
                            "pos": round(0.05 * ((i % 5) - 2), 3),
                        }
                    spec["motors"] = motors
                    self.get_logger().info(
                        f"SIM transport: {len(motors)} virtual motors")
                try:
                    transport = make_transport(spec)
                    bus = MotorBus(transport, motor_map)
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().error(f"transport init failed: {exc}")

            self._init_state(bus=bus, motor_map=motor_map, mode=mode,
                             limits_path=limits_path, map_path=map_path)

            # Run-mode telemetry source: /joint_states (read-only).
            self.create_subscription(JointState, "/joint_states",
                                     self._on_joint_states, 10)
            self.create_timer(1.0 / DEADMAN_RATE, self._loop)

        def _now(self) -> float:
            return self.get_clock().now().nanoseconds * 1e-9

        def _on_joint_states(self, msg):
            with self._lock:
                for name, pos in zip(msg.name, msg.position):
                    if name in self.motor_map:
                        self._telem.setdefault(name, {})["pos"] = float(pos)

        def _loop(self):
            self.tick_deadman()
            self.poll_telemetry()

    rclpy.init(args=args)
    node = _RosNode()
    handler = _make_handler(node, _load_html())

    server = None
    for port in range(node.port, node.port + 10):
        try:
            server = _Server((node.host, port), handler)
            node.port = port
            break
        except OSError:
            node.get_logger().warn(f"port {port} busy, trying {port + 1}…")
    if server is None:
        node.get_logger().error("could not bind a hwconfig port")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        return

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    shown = "localhost" if node.host in ("0.0.0.0", "") else node.host
    node.get_logger().info(
        f"M1 hardware config page -> http://{shown}:{node.port}  "
        f"(mode={node.mode}; maintenance owns the CAN bus)")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        try:
            if node.bus is not None and node.mode == "maintenance":
                node.bus.disable_all()
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

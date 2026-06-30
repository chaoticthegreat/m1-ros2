"""Quest viz performance bench (no DDS).

Measures the SERVER side of the Quest WebXR teleop, to back the perf fix:

  A. per-frame cost of the REAL ``M1QuestNode.on_xr_frame`` (FK cache hit vs.
     recompute) + the response payload size.
  B. END-TO-END over the REAL HTTPS server: one keep-alive TLS connection (like
     the headset) hammering ``/api/xr``; reports sustained req/s + per-request
     latency. This exercises the rewritten single-write ``_send`` + ``do_POST``
     and confirms the server is nowhere near the bottleneck.

NB: the ~40 ms Nagle/delayed-ACK stall the fix removes is a property of a REAL
network path; it does NOT reproduce on loopback. So this bench proves the server
is fast and correct; the real RTT win is read off the in-headset ``/?perf`` HUD.

Run:  /usr/bin/python3 _quest_perf_bench.py     (with /opt/ros/jazzy sourced)
"""
import http.client
import json
import os
import ssl
import subprocess
import sys
import threading
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ros2_ws/src/m1_control"))

from m1_control.quest_node import M1QuestNode, _make_handler, _Server  # noqa: E402
from m1_control.kinematics import (  # noqa: E402
    ARM_JOINTS, LIFT_JOINT, ReachController, UrdfModel,
)
from m1_control.swerve import SwerveOdometry  # noqa: E402

ARMS = ("left", "right")
URDF = "assets/ranger_air_description/urdf/ranger_air_description.urdf"


def make_node(reach):
    """Headless node carrying just what the HTTP path + on_xr_frame touch."""
    n = object.__new__(M1QuestNode)
    n._lock = threading.Lock()
    n._now = lambda: 0.0
    n.reach = reach
    n.enable_base = True
    n.motion_scale = 1.0
    n.host = "127.0.0.1"
    n.port = 0
    n.get_logger = lambda: types.SimpleNamespace(
        warn=lambda *a, **k: None, info=lambda *a, **k: None,
        error=lambda *a, **k: None)
    n.q_meas = {}
    n.target = {a: [0.40, 0.0, 0.70] for a in ARMS}
    n.err = {a: None for a in ARMS}
    n.seeded = {a: True for a in ARMS}
    n.grip = {a: 0.0 for a in ARMS}
    n.cmd_vel = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}
    n.clutch = {a: False for a in ARMS}
    n.clutch_hand0 = {a: None for a in ARMS}
    n.clutch_target0 = {a: None for a in ARMS}
    n.clutch_F = {a: None for a in ARMS}
    n.clutch_L = {a: None for a in ARMS}
    n.fine = {a: False for a in ARMS}
    n.last_precision = {a: False for a in ARMS}
    n.last_btn = {a: False for a in ARMS}
    n.control_mode = "relative"
    n._last_mode_chord = False
    n._last_base_cmd = 0.0
    n._last_update = 0.0
    n.odom = SwerveOdometry()
    n._last_tick = 0.0
    n._q_ver = 0
    n._viz_fk_cache = None
    n._traj = {a: None for a in ARMS}
    return n


def full_q(lift=0.4, jitter=0.0):
    q = {j: 0.0 for j in ARM_JOINTS["left"] + ARM_JOINTS["right"]}
    q[LIFT_JOINT] = lift
    q[ARM_JOINTS["left"][1]] = 0.5 + jitter
    q[ARM_JOINTS["left"][3]] = 0.8
    q[ARM_JOINTS["right"][1]] = -0.5
    q[ARM_JOINTS["right"][3]] = 0.8 - jitter
    return q


def ctrl():
    return {"valid": True, "pos": [0, 0, 0], "squeeze": False, "lock": False,
            "button": False, "recenter": False, "trigger": 0.0, "stick": [0, 0]}


PAYLOAD = {"controllers": {"left": ctrl(), "right": ctrl()}, "head": [0, 0, -1.0]}


def stats(ts):
    ts = sorted(ts)
    n = len(ts)
    return (sum(ts) / n, ts[n // 2], ts[int(n * 0.95)], ts[int(n * 0.99)], ts[-1])


def _ensure_cert():
    """Reuse the node's cached cert if present, else make a throwaway one."""
    cache = os.path.expanduser("~/.cache/m1_quest")
    cert, key = os.path.join(cache, "cert.pem"), os.path.join(cache, "key.pem")
    if os.path.isfile(cert) and os.path.isfile(key):
        return cert, key
    cert, key = "/tmp/_qbench_cert.pem", "/tmp/_qbench_key.pem"
    if not (os.path.isfile(cert) and os.path.isfile(key)):
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
             "-keyout", key, "-out", cert, "-days", "3650", "-subj", "/CN=bench"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return cert, key


def bench_inproc(node):
    print("=== A. server per-frame cost (in-process, no socket) ===")
    node.q_meas = full_q()
    res = node.on_xr_frame(PAYLOAD)
    blob = json.dumps(res)
    print(f"  response payload: {len(blob)} bytes ({len(blob)/1024:.1f} KB), "
          f"{len(res['viz']['links'])} links")

    def micro(label, fn, iters=600):
        fn(); fn()
        t = []
        for _ in range(iters):
            t0 = time.perf_counter(); fn(); t.append((time.perf_counter() - t0) * 1e3)
        mean, p50, p95, p99, mx = stats(t)
        print(f"  {label:34s} mean {mean:6.3f}  p50 {p50:6.3f}  "
              f"p95 {p95:6.3f}  max {mx:6.3f} ms")

    micro("on_xr_frame (FK cache HIT)", lambda: node.on_xr_frame(PAYLOAD))
    i = {"k": 0}

    def miss():
        i["k"] += 1
        node.q_meas = full_q(jitter=0.001 * (i["k"] % 50))
        node._q_ver += 1
        node.on_xr_frame(PAYLOAD)
    micro("on_xr_frame (FK recompute/frame)", miss)
    node.q_meas = full_q()
    micro("on_xr_frame + json.dumps", lambda: json.dumps(node.on_xr_frame(PAYLOAD)))
    print()


def bench_e2e(node, iters=2000):
    print("=== B. end-to-end over the REAL HTTPS server (one keep-alive conn) ===")
    assert _make_handler(node).disable_nagle_algorithm is True, \
        "Handler must set disable_nagle_algorithm (TCP_NODELAY)"
    cert, key = _ensure_cert()
    sctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    sctx.load_cert_chain(certfile=cert, keyfile=key)
    server = _Server((node.host, 0), _make_handler(node))
    server.socket = sctx.wrap_socket(server.socket, server_side=True)
    port = server.socket.getsockname()[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        cctx = ssl.create_default_context()
        cctx.check_hostname = False
        cctx.verify_mode = ssl.CERT_NONE
        conn = http.client.HTTPSConnection(node.host, port, context=cctx, timeout=10)
        body = json.dumps(PAYLOAD).encode()
        hdr = {"Content-Type": "application/json"}

        # warmup + correctness check (round-trips through do_POST/_send)
        ok = False
        for _ in range(10):
            conn.request("POST", "/api/xr", body, hdr)
            r = conn.getresponse()
            data = r.read()
            ok = (r.status == 200 and b'"links"' in data
                  and r.getheader("Content-Length") == str(len(data)))
        print(f"  round-trip correctness (200, viz.links, Content-Length): "
              f"{'OK' if ok else 'FAIL'}")

        lat = []
        i = {"k": 0}
        t_all = time.perf_counter()
        for _ in range(iters):
            # change joints each request to model the worst case (FK recompute)
            i["k"] += 1
            with node._lock:
                node.q_meas = full_q(jitter=0.001 * (i["k"] % 50))
                node._q_ver += 1
            t0 = time.perf_counter()
            conn.request("POST", "/api/xr", body, hdr)
            r = conn.getresponse()
            r.read()
            lat.append((time.perf_counter() - t0) * 1e3)
        wall = time.perf_counter() - t_all
        conn.close()
        mean, p50, p95, p99, mx = stats(lat)
        print(f"  {iters} POSTs in {wall*1e3:.0f} ms  -> {iters/wall:8.0f} req/s "
              f"sustained on one keep-alive TLS connection")
        print(f"  per-request latency:  mean {mean:6.3f}  p50 {p50:6.3f}  "
              f"p95 {p95:6.3f}  p99 {p99:6.3f}  max {mx:6.3f} ms (loopback)")
        return ok, iters / wall
    finally:
        server.shutdown()
        server.server_close()


def main():
    print("=== Quest viz server perf bench ===\n")
    reach = ReachController(UrdfModel.from_string(open(URDF).read()))
    node = make_node(reach)
    bench_inproc(node)
    ok, rps = bench_e2e(node)

    print("\n---- GATES ----")
    gates = {
        "Handler sets TCP_NODELAY": _make_handler(node).disable_nagle_algorithm is True,
        "end-to-end round-trip correct (viz over real TLS)": ok,
        "server sustains >> 90 req/s (not the bottleneck)": rps > 200,
    }
    for k, v in gates.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    npass = sum(1 for v in gates.values() if v)
    print(f"\n{npass}/{len(gates)} gates passed")
    sys.exit(0 if npass == len(gates) else 1)


if __name__ == "__main__":
    main()

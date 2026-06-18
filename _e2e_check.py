"""End-to-end check: fake robot + real controller + web node, driven over HTTP.

Simulates the closed loop (robot follows /m1/joint_command) so we can confirm
the arm reaches a target and the base wheels spin -- the same path the browser
panel uses, without needing Isaac Sim.
"""
import json
import threading
import time
import urllib.request

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from m1_control.controller_node import (ALL_JOINTS, LIFT_JOINT, POSITION_JOINTS,
                                        WHEEL_JOINTS, M1Controller)
from m1_control.web_node import M1WebNode
from m1_control.kinematics import ARM_JOINTS, ReachController, UrdfModel
from ament_index_python.packages import get_package_share_directory
import os

PORT = 8092


class FakeRobot(Node):
    """Publishes /joint_states; instantly follows position commands (closed loop)."""

    def __init__(self):
        super().__init__("fake_robot")
        self.q = {j: 0.0 for j in ALL_JOINTS}
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self.create_subscription(JointState, "/m1/joint_command", self._cmd, 10)
        self.create_timer(0.02, self._tick)

    def _cmd(self, msg):
        for n, p in zip(msg.name, msg.position):
            if n in self.q and n not in WHEEL_JOINTS:
                self.q[n] = float(p)  # perfect position tracking

    def _tick(self):
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = list(self.q.keys())
        m.position = [self.q[n] for n in m.name]
        m.velocity = [0.0] * len(m.name)
        self.pub.publish(m)


class Recorder(Node):
    def __init__(self):
        super().__init__("recorder")
        self.last = None
        self.create_subscription(JointState, "/m1/joint_command", self._cb, 10)

    def _cb(self, msg):
        self.last = {n: (p, v) for n, p, v in
                     zip(msg.name, msg.position, msg.velocity)}


class Probe(Node):
    """Independent subscriber to confirm web messages are on the wire."""

    def __init__(self):
        super().__init__("probe")
        from geometry_msgs.msg import Twist
        from std_msgs.msg import Float64
        self.cv = None
        self.gl = None
        self.create_subscription(Twist, "/m1/cmd_vel", lambda m: setattr(self, "cv", m.linear.x), 10)
        self.create_subscription(Float64, "/m1/left_arm/gripper", lambda m: setattr(self, "gl", m.data), 10)


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/api/cmd",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=2).read()


def get(path):
    return urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=2).read()


def main():
    rclpy.init()
    fake = FakeRobot()
    ctrl = M1Controller()
    rec = Recorder()
    probe = Probe()
    web = M1WebNode()

    from http.server import ThreadingHTTPServer
    from m1_control.web_node import _make_handler, _resolve_web_dir
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), _make_handler(web, _resolve_web_dir()))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    ex = MultiThreadedExecutor()
    for n in (fake, ctrl, rec, probe, web):
        ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()

    results = []

    def check(name, ok, detail=""):
        results.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    # 0) Let everything settle / initialize.
    time.sleep(1.5)

    # 1) HTTP serves the page + state shows connected.
    html = get("/").decode()
    check("web serves index.html", "M1 Control Panel" in html, f"{len(html)} bytes")
    st = json.loads(get("/api/state"))
    check("web sees robot connected", st["connected"] and st["njoints"] >= len(POSITION_JOINTS),
          f"njoints={st['njoints']}")

    # 2) Reach: command a high-left target that needs the lift.
    share = get_package_share_directory("ranger_air_description")
    urdf = os.path.join(share, "urdf", "ranger_air_description.urdf")
    reach = ReachController(UrdfModel.from_string(open(urdf).read()))
    q0 = dict(fake.q)
    tip0 = reach.fingertip("left", q0)
    target = [0.45, 0.28, 1.05]
    post({"type": "target", "arm": "left", "xyz": target})
    time.sleep(4.0)
    q1 = dict(fake.q)
    tip1 = reach.fingertip("left", q1)
    d0 = float(np.linalg.norm(np.array(target) - tip0))
    d1 = float(np.linalg.norm(np.array(target) - tip1))
    lift_cmd = rec.last.get(LIFT_JOINT, (0, 0))[0] if rec.last else 0.0
    check("arm reaches target (error shrinks)", d1 < d0 - 0.05,
          f"err {d0*100:.1f}cm -> {d1*100:.1f}cm")
    check("shared lift recruited for high target", abs(lift_cmd) > 0.03,
          f"lift_cmd={lift_cmd:.3f} m")
    arm_moved = any(abs(q1[j] - q0[j]) > 0.01 for j in ARM_JOINTS["left"])
    check("left arm joints moved", arm_moved)

    # 3) Base: drive forward (streamed like the browser), wheels should spin.
    for _ in range(12):
        post({"type": "cmd_vel", "vx": 0.3, "vy": 0.0, "yaw": 0.0})
        time.sleep(0.1)
    wv = [rec.last[w][1] for w in WHEEL_JOINTS] if rec.last else []
    check("base drive spins wheels", any(abs(v) > 0.5 for v in wv),
          f"wheel vels={[round(v,2) for v in wv]}")
    print("   DEBUG web.cmd_vel=", web.cmd_vel, " ctrl.cmd_vel=", ctrl.cmd_vel,
          " probe.cv=", probe.cv)
    post({"type": "stop"})
    time.sleep(1.0)
    wv2 = [rec.last[w][1] for w in WHEEL_JOINTS] if rec.last else []
    check("stop zeros wheels", all(abs(v) < 0.05 for v in wv2),
          f"wheel vels={[round(v,2) for v in wv2]}")

    # 4) Gripper command flows through to the finger command.
    post({"type": "gripper", "arm": "left", "value": 1.0})
    time.sleep(0.8)
    print("   DEBUG web.grip=", web.grip, " ctrl.grip=", ctrl.grip, " probe.gl=", probe.gl)
    fcmd = rec.last.get("openarm_left_finger_joint1", (0, 0))[0] if rec.last else 0.0
    lf = fake.q.get("openarm_left_finger_joint1", 0.0)
    check("gripper opens", lf > 0.3, f"cmd={fcmd:.3f} finger={lf:.3f} rad")

    httpd.shutdown()
    rclpy.shutdown()

    npass = sum(1 for _, ok, _ in results if ok)
    print(f"\n==== {npass}/{len(results)} checks passed ====")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

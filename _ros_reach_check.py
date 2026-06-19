"""End-to-end reach check over real ROS/DDS (no Isaac, no web node).

Wires a fake robot (follows /m1/joint_command) to the real M1Controller and an
in-graph driver that streams target poses the way an operator bridge
(Quest/web/teleop) does. Everything runs on ROS timers under a single-threaded
executor, so the message flow mirrors the real (multi-process) system instead of
being starved by a main-thread sleep loop.

Confirms, over the wire, the behaviours the solver work targeted:

  1. a static target converges (single arm),
  2. a smoothly moving target is tracked smoothly (no goal/fingertip jumps),
  3. moving one arm does not disturb the other (held arm stays put).

Run:
  source /opt/ros/jazzy/setup.bash && source ros2_ws/install/setup.bash
  ROS_LOG_DIR=/tmp/ros_log /usr/bin/python3 _ros_reach_check.py
"""
import math
import os
import sys

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ros2_ws/src/m1_control"))

from m1_control.controller_node import ALL_JOINTS, WHEEL_JOINTS, M1Controller
from m1_control.kinematics import ReachController, UrdfModel

RATE = 60.0


class FakeRobot(Node):
    """Publishes /joint_states; follows position commands instantly."""

    def __init__(self):
        super().__init__("fake_robot")
        self.q = {j: 0.0 for j in ALL_JOINTS}
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        self.create_subscription(JointState, "/m1/joint_command", self._cmd, 10)
        self.create_timer(1.0 / 120.0, self._tick)

    def _cmd(self, msg):
        for n, p in zip(msg.name, msg.position):
            if n in self.q and n not in WHEEL_JOINTS:
                self.q[n] = float(p)

    def _tick(self):
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = list(self.q.keys())
        m.position = [self.q[n] for n in m.name]
        m.velocity = [0.0] * len(m.name)
        self.pub.publish(m)


class Scenario(Node):
    """Drives targets + checks results, all on one 60 Hz timer (in-graph)."""

    def __init__(self, fake, reach):
        super().__init__("scenario")
        self.fake = fake
        self.reach = reach
        self.pub = {
            a: self.create_publisher(PoseStamped, f"/m1/{a}_arm/target_pose", 10)
            for a in ("left", "right")
        }
        self.target = {"left": None, "right": None}
        self.phase = 0
        self.k = 0
        self.results = []
        self.max_step = 0.0
        self.max_err = 0.0
        self.max_right_dev = 0.0
        self.right_anchor = None
        self.prev_tip = None
        self.base_l = reach.fingertip("left", self._cfg("left"))
        self.done = False
        self.create_timer(1.0 / RATE, self._tick)

    @staticmethod
    def _cfg(arm, lift=0.35):
        q = {"lift_joint": lift}
        for j, v in zip([f"openarm_{arm}_joint{i}" for i in range(1, 8)],
                        [0.0, 0.5, 0.0, 0.9, 0.0, 0.4, 0.0]):
            q[j] = v
        return q

    def _publish(self):
        stamp = self.get_clock().now().to_msg()
        for a in ("left", "right"):
            t = self.target[a]
            if t is None:
                continue
            ps = PoseStamped()
            ps.header.stamp = stamp
            ps.header.frame_id = "base_link"
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (
                float(t[0]), float(t[1]), float(t[2]))
            ps.pose.orientation.w = 1.0
            self.pub[a].publish(ps)

    def _check(self, name, ok, detail=""):
        self.results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)

    def _tick(self):
        self.k += 1
        if self.phase == 0:
            # Let the controller initialize its command pose from /joint_states.
            if self.k > 60:
                self.target["left"] = self.reach.fingertip("left", self._cfg("left"))
                self.phase, self.k = 1, 0
        elif self.phase == 1:
            self._publish()
            if self.k > 180:
                tip = self.reach.fingertip("left", dict(self.fake.q))
                err = float(np.linalg.norm(self.target["left"] - tip))
                self._check("static single-arm converges", err < 0.005,
                            f"err={err*1e3:.2f}mm")
                self.prev_tip = tip
                self.phase, self.k = 2, 0
        elif self.phase == 2:
            t = self.k / RATE
            self.target["left"] = self.base_l + np.array([
                0.10 * math.sin(0.8 * t), 0.12 * math.sin(0.6 * t),
                0.08 * math.sin(1.1 * t)])
            self._publish()
            tip = self.reach.fingertip("left", dict(self.fake.q))
            if self.k > 20:  # let it lock on
                self.max_step = max(self.max_step,
                                    float(np.linalg.norm(tip - self.prev_tip)))
                self.max_err = max(self.max_err,
                                   float(np.linalg.norm(self.target["left"] - tip)))
            self.prev_tip = tip
            if self.k > 300:
                self._check("moving target tracked smoothly",
                            self.max_step < 0.02 and self.max_err < 0.03,
                            f"max fingertip step={self.max_step*1e3:.1f}mm "
                            f"max err={self.max_err*1e3:.1f}mm")
                self.target["right"] = self.reach.fingertip("right", self._cfg("right"))
                self.phase, self.k = 3, 0
        elif self.phase == 3:
            self._publish()  # both: left at last value, right static
            if self.k == 150:
                self.right_anchor = self.reach.fingertip("right", dict(self.fake.q))
            if self.k > 150:
                t = (self.k - 150) / RATE
                self.target["left"] = self.base_l + np.array([
                    0.12 * math.sin(0.9 * t), 0.14 * math.sin(0.7 * t),
                    0.10 * math.sin(1.2 * t)])
                rtip = self.reach.fingertip("right", dict(self.fake.q))
                self.max_right_dev = max(
                    self.max_right_dev,
                    float(np.linalg.norm(rtip - self.right_anchor)))
            if self.k > 150 + 300:
                self._check("held arm undisturbed while other moves",
                            self.max_right_dev < 0.01,
                            f"max right deviation={self.max_right_dev*1e3:.1f}mm")
                self.phase = 4

        if self.phase == 4 and not self.done:
            self.done = True
            npass = sum(self.results)
            print(f"\n==== {npass}/{len(self.results)} checks passed ====",
                  flush=True)
            raise KeyboardInterrupt


def main():
    rclpy.init()
    fake = FakeRobot()
    ctrl = M1Controller()
    share = get_package_share_directory("ranger_air_description")
    urdf = os.path.join(share, "urdf", "ranger_air_description.urdf")
    reach = ReachController(UrdfModel.from_string(open(urdf).read()))
    scn = Scenario(fake, reach)

    from rclpy.executors import SingleThreadedExecutor
    ex = SingleThreadedExecutor()
    for n in (fake, ctrl, scn):
        ex.add_node(n)
    rc = 0
    try:
        ex.spin()
    except KeyboardInterrupt:
        rc = 0 if (scn.results and all(scn.results)) else 1
    ex.shutdown()
    if rclpy.ok():
        rclpy.shutdown()
    os._exit(rc)


if __name__ == "__main__":
    main()

#!/usr/bin/env /usr/bin/python3
"""One-shot LIVE probe: is the brain COMMANDING the target, or is the SIM not
tracking it?  (The decisive split for the reach failures.)

Grabs the latest /m1/joint_command (what the brain TELLS the arm to do),
/joint_states (what the arm ACTUALLY did in sim), and the per-arm target, then
computes the fingertip of BOTH joint sets with the SAME FK the brain uses:

  cmd_err  = || target - fingertip(commanded joints) ||   should be ~0 if the
             solver converged (the command leads onto the target).
  meas_err = || target - fingertip(measured joints) ||    what the operator sees.

Interpretation:
  cmd_err ~0, meas_err large  -> SIM not tracking the command (gravity sag/gains);
                                 the IK is fine.  (isaac/ros_sim.py ARM_KP / drive.)
  cmd_err large               -> the LIVE solve is returning a bad command (a
                                 live-only solver bug, e.g. mimic-finger infeasible
                                 -> SNOPT garbage); the IK IS the problem.

Run:  PYTHONPATH=ros2_ws/src/m1_control /usr/bin/python3 _probe_cmd_vs_meas.py
"""
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

sys.path.insert(0, "ros2_ws/src/m1_control")
from m1_control.kinematics import (UrdfModel, ReachController, ARM_JOINTS,
                                    LIFT_JOINT)


def find_urdf():
    for c in ("ros2_ws/install/ranger_air_description/share/ranger_air_description/"
              "urdf/ranger_air_description.urdf",
              "assets/ranger_air_description/urdf/ranger_air_description.urdf"):
        if os.path.isfile(c):
            return c
    raise SystemExit("URDF not found")


class Probe(Node):
    def __init__(self):
        super().__init__("m1_cmd_meas_probe")
        self.reach = ReachController(UrdfModel.from_string(open(find_urdf()).read()))
        self.cmd = {}
        self.meas = {}
        self.target = {"left": None, "right": None}
        self.create_subscription(JointState, "/m1/joint_command", self._cmd, 10)
        self.create_subscription(JointState, "/joint_states", self._meas, 10)
        self.create_subscription(PoseStamped, "/m1/left_arm/target_pose",
                                 lambda m: self._tg("left", m), 10)
        self.create_subscription(PoseStamped, "/m1/right_arm/target_pose",
                                 lambda m: self._tg("right", m), 10)

    def _cmd(self, m):
        for n, p in zip(m.name, m.position):
            self.cmd[n] = float(p)

    def _meas(self, m):
        for n, p in zip(m.name, m.position):
            self.meas[n] = float(p)

    def _tg(self, arm, m):
        self.target[arm] = [m.pose.position.x, m.pose.position.y, m.pose.position.z]

    def ready(self):
        need = ARM_JOINTS["left"] + ARM_JOINTS["right"] + [LIFT_JOINT]
        return all(j in self.cmd for j in need) and all(j in self.meas for j in need)


def main():
    rclpy.init()
    p = Probe()
    t0 = time.time()
    while time.time() - t0 < 4.0 and not (p.ready() and any(p.target.values())):
        rclpy.spin_once(p, timeout_sec=0.1)
    if not p.ready():
        print("did not receive /m1/joint_command + /joint_states (is the brain up?)")
        return
    print(f"{'arm':5s} {'target':>26s} {'cmd_err':>9s} {'meas_err':>9s}   verdict")
    for arm in ("left", "right"):
        tg = p.target[arm]
        if tg is None:
            continue
        tg = np.array(tg)
        cmd_tip = p.reach.fingertip(arm, p.cmd)
        meas_tip = p.reach.fingertip(arm, p.meas)
        ce = float(np.linalg.norm(tg - cmd_tip)) * 1e3
        me = float(np.linalg.norm(tg - meas_tip)) * 1e3
        if ce < 8 and me > 25:
            verdict = "SIM NOT TRACKING (IK ok, command not executed)"
        elif ce >= 25:
            verdict = "LIVE SOLVE BAD (brain commands wrong config)"
        else:
            verdict = "ok / converged"
        print(f"{arm:5s} {str([round(v,3) for v in tg]):>26s} "
              f"{ce:8.1f}m {me:8.1f}m   {verdict}")
    # also dump the lift command vs measured + finger joints (mimic check)
    print(f"\nlift: cmd={p.cmd.get(LIFT_JOINT):.4f}  meas={p.meas.get(LIFT_JOINT):.4f}")
    fingers = [j for j in p.cmd if "finger" in j]
    print("finger joints (cmd vs meas):")
    for j in sorted(fingers):
        print(f"  {j:28s} cmd={p.cmd.get(j):+.4f}  meas={p.meas.get(j, float('nan')):+.4f}")
    # per-joint cmd vs meas for the worst arm, to see WHERE it sags
    for arm in ("left", "right"):
        if p.target[arm] is None:
            continue
        print(f"\n{arm} arm joints (cmd -> meas, delta):")
        for j in ARM_JOINTS[arm]:
            c, mv = p.cmd.get(j, 0.0), p.meas.get(j, 0.0)
            print(f"  {j:26s} {c:+.4f} -> {mv:+.4f}  (Δ={mv-c:+.4f})")
    p.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

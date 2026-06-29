#!/usr/bin/env /usr/bin/python3
"""Live contact/collision diagnostic for a stuck reach.

Grabs the brain's command, the sim's measured joints + efforts, and the targets,
then: (a) cmd vs meas fingertip error, (b) self-collision clearance of BOTH the
commanded and the measured config (repo CollisionModel: arm<->arm + arm<->body
column), (c) measured joint efforts (a contact reaction / drive saturation shows
as a large steady effort). If the commanded config self-collides, or the measured
config sits at a contact, the arm is BLOCKED (not sagging) and the IK target is
self-colliding -- a different fix than gains.
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
from m1_control.collision import CollisionModel


def find_urdf():
    for c in ("ros2_ws/install/ranger_air_description/share/ranger_air_description/"
              "urdf/ranger_air_description.urdf",
              "assets/ranger_air_description/urdf/ranger_air_description.urdf"):
        if os.path.isfile(c):
            return c
    raise SystemExit("URDF not found")


class D(Node):
    def __init__(self):
        super().__init__("m1_contact_diag")
        self.reach = ReachController(UrdfModel.from_string(open(find_urdf()).read()))
        self.cm = CollisionModel(self.reach)
        self.cmd, self.meas, self.eff = {}, {}, {}
        self.target = {"left": None, "right": None}
        self.create_subscription(JointState, "/m1/joint_command", self._cmd, 10)
        self.create_subscription(JointState, "/joint_states", self._meas, 10)
        self.create_subscription(PoseStamped, "/m1/left_arm/target_pose",
                                 lambda m: self._t("left", m), 10)
        self.create_subscription(PoseStamped, "/m1/right_arm/target_pose",
                                 lambda m: self._t("right", m), 10)

    def _cmd(self, m):
        for n, p in zip(m.name, m.position):
            self.cmd[n] = float(p)

    def _meas(self, m):
        for n, p in zip(m.name, m.position):
            self.meas[n] = float(p)
        for n, e in zip(m.name, m.effort or []):
            self.eff[n] = float(e)

    def _t(self, a, m):
        self.target[a] = [m.pose.position.x, m.pose.position.y, m.pose.position.z]

    def ready(self):
        need = ARM_JOINTS["left"] + ARM_JOINTS["right"] + [LIFT_JOINT]
        return all(j in self.cmd for j in need) and all(j in self.meas for j in need)


def clr(cm, qdict):
    try:
        r = cm.clearance(qdict)
        # clearance() may return a scalar or (min_clr, pair) -- normalize
        if isinstance(r, (tuple, list)):
            return float(r[0]), (r[1] if len(r) > 1 else None)
        return float(r), None
    except Exception as e:  # noqa: BLE001
        return None, f"err:{e}"


def main():
    rclpy.init()
    d = D()
    t0 = time.time()
    while time.time() - t0 < 4.0 and not (d.ready() and any(d.target.values())):
        rclpy.spin_once(d, timeout_sec=0.1)
    if not d.ready():
        print("no command/joint_states (brain up?)")
        return
    for arm in ("left", "right"):
        tg = d.target[arm]
        if tg is None:
            continue
        tg = np.array(tg)
        ce = float(np.linalg.norm(tg - d.reach.fingertip(arm, d.cmd))) * 1e3
        me = float(np.linalg.norm(tg - d.reach.fingertip(arm, d.meas))) * 1e3
        print(f"{arm}: target={[round(v,3) for v in tg]}  cmd_err={ce:.1f}mm  meas_err={me:.1f}mm")
    cmd_clr, cmd_pair = clr(d.cm, d.cmd)
    meas_clr, meas_pair = clr(d.cm, d.meas)
    print(f"\nself-collision clearance (negative/near-0 = colliding):")
    print(f"  COMMANDED config: clearance={cmd_clr}  pair={cmd_pair}")
    print(f"  MEASURED  config: clearance={meas_clr}  pair={meas_pair}")
    print("\nmeasured joint efforts (large steady effort => contact/saturation):")
    if not d.eff:
        print("  (/joint_states carries no effort field)")
    for arm in ("left", "right"):
        for j in ARM_JOINTS[arm]:
            if j in d.eff:
                print(f"  {j:26s} effort={d.eff[j]:+.3f}")
    d.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

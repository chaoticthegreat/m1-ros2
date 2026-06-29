#!/usr/bin/env /usr/bin/python3
"""Static gravity torque at the live COMMANDED config vs the joint effort limits.

If a joint's gravity torque is WELL UNDER its effort limit yet the live measured
effort is AT the limit (saturated), the extra resisting torque is a CONTACT (the
arm is jammed against structure), not gravity -> the IK is commanding a
self-colliding pose. If gravity torque >= the limit, the target is statically
torque-infeasible (needs gravity-comp feedforward).

tau_grav_j = d/dq_j [ sum_k m_k * g * z_com_k(q) ]  (finite-difference of the
gravitational potential energy), using the same FK the brain uses.
"""
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

sys.path.insert(0, "ros2_ws/src/m1_control")
from m1_control.kinematics import UrdfModel, ARM_JOINTS, LIFT_JOINT

G = 9.81
URDF = ("ros2_ws/install/ranger_air_description/share/ranger_air_description/"
        "urdf/ranger_air_description.urdf")
if not os.path.isfile(URDF):
    URDF = "assets/ranger_air_description/urdf/ranger_air_description.urdf"


def parse_inertials(path):
    """link name -> (mass, com xyz in link frame)."""
    r = ET.fromstring(open(path).read())
    out = {}
    for lk in r.iter("link"):
        ine = lk.find("inertial")
        if ine is None:
            continue
        m = float(ine.find("mass").get("value"))
        o = ine.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0").split())] if o is not None else [0, 0, 0]
        out[lk.get("name")] = (m, np.array(xyz))
    return out


def potential_energy(model, inert, q):
    Ts = model.link_transforms(q)
    U = 0.0
    for name, (m, com) in inert.items():
        T = Ts.get(name)
        if T is None:
            continue
        z = (T[:3, :3] @ com + T[:3, 3])[2]
        U += m * G * z
    return U


def grav_torques(model, inert, q, joints, eps=1e-4):
    tau = {}
    for j in joints:
        qp, qm = dict(q), dict(q)
        qp[j] = q.get(j, 0.0) + eps
        qm[j] = q.get(j, 0.0) - eps
        tau[j] = (potential_energy(model, inert, qp)
                  - potential_energy(model, inert, qm)) / (2 * eps)
    return tau


class Grab(Node):
    def __init__(self):
        super().__init__("m1_grav_grab")
        self.cmd, self.eff = {}, {}
        self.create_subscription(JointState, "/m1/joint_command", self._c, 10)
        self.create_subscription(JointState, "/joint_states", self._m, 10)

    def _c(self, m):
        for n, p in zip(m.name, m.position):
            self.cmd[n] = float(p)

    def _m(self, m):
        for n, e in zip(m.name, m.effort or []):
            self.eff[n] = float(e)

    def ready(self):
        return all(j in self.cmd for j in ARM_JOINTS["left"] + [LIFT_JOINT])


def main():
    model = UrdfModel.from_string(open(URDF).read())
    inert = parse_inertials(URDF)
    limits = {j: float(model.joints[j].effort)
              for a in ("left", "right") for j in ARM_JOINTS[a]
              if hasattr(model.joints[j], "effort")}

    rclpy.init()
    g = Grab()
    t0 = time.time()
    while time.time() - t0 < 4.0 and not g.ready():
        rclpy.spin_once(g, timeout_sec=0.1)
    if not g.ready():
        print("no /m1/joint_command")
        return
    allj = ARM_JOINTS["left"] + ARM_JOINTS["right"]
    tau = grav_torques(model, inert, g.cmd, allj)
    print(f"{'joint':24s} {'grav_tau':>9s} {'limit':>7s} {'meas_eff':>9s}  flag")
    for j in allj:
        lim = limits.get(j, float("nan"))
        gt = tau[j]
        me = g.eff.get(j, float("nan"))
        flag = ""
        if abs(me) >= lim - 0.5:
            flag += "SATURATED "
        if abs(gt) >= lim - 0.5:
            flag += "GRAV>=LIMIT "
        if abs(me) >= lim - 0.5 and abs(gt) < lim - 2.0:
            flag += "<- excess torque = CONTACT"
        print(f"{j:24s} {gt:+9.2f} {lim:7.1f} {me:+9.2f}  {flag}")
    g.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

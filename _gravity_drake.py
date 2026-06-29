#!/usr/bin/env /usr/bin/python3
"""Decisive gravity-infeasibility test: Drake's exact static gravity torque at the
live COMMANDED config vs each joint's effort limit. If |tau_grav| >= effort limit,
the effort-limited PD sim drive (no gravity feedforward) cannot hold that posture
-> the arm sags to a steady offset (the observed jam), and it is NOT collision and
NOT a solver bug. Uses Drake CalcGravityGeneralizedForces (trusted; uses the URDF
inertials), not the hand finite-difference."""
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

sys.path.insert(0, "ros2_ws/src/m1_control")
URDF = ("ros2_ws/install/ranger_air_description/share/ranger_air_description/"
        "urdf/ranger_air_description.urdf")
BASE_LINK = "base_link"
from m1_control.kinematics import ARM_JOINTS, LIFT_JOINT


def effort_limits():
    r = ET.fromstring(open(URDF).read())
    out = {}
    for j in r.iter("joint"):
        lim = j.find("limit")
        if lim is not None and lim.get("effort") is not None:
            out[j.get("name")] = float(lim.get("effort"))
    return out


def build_plant():
    from pydrake.multibody.plant import MultibodyPlant
    from pydrake.multibody.parsing import Parser
    root = ET.fromstring(open(URDF).read())
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for e in list(link.findall(tag)):
                link.remove(e)
    plant = MultibodyPlant(0.0)
    Parser(plant).AddModelsFromString(ET.tostring(root, encoding="unicode"), "urdf")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName(BASE_LINK))
    plant.Finalize()
    return plant


class Grab(Node):
    def __init__(self):
        super().__init__("m1_grav_drake")
        self.cmd = {}
        self.create_subscription(JointState, "/m1/joint_command", self._c, 10)

    def _c(self, m):
        for n, p in zip(m.name, m.position):
            self.cmd[n] = float(p)

    def ready(self):
        return all(j in self.cmd for j in ARM_JOINTS["left"] + [LIFT_JOINT])


def main():
    plant = build_plant()
    ctx = plant.CreateDefaultContext()
    jstart = {plant.get_joint(i).name(): plant.get_joint(i).position_start()
              for i in plant.GetJointIndices() if plant.get_joint(i).num_positions() == 1}
    lim = effort_limits()

    rclpy.init()
    g = Grab()
    t0 = time.time()
    while time.time() - t0 < 4.0 and not g.ready():
        rclpy.spin_once(g, timeout_sec=0.1)
    if not g.ready():
        print("no /m1/joint_command (brain up?)"); return

    qv = np.zeros(plant.num_positions())
    for n, s in jstart.items():
        qv[s] = g.cmd.get(n, 0.0)
    plant.SetPositions(ctx, qv)
    tau_g = plant.CalcGravityGeneralizedForces(ctx)   # generalized gravity force
    # holding torque the drive must supply = -gravity force
    print(f"{'joint':24s} {'hold_tau(N·m)':>13s} {'effort_lim':>10s}  flag")
    for arm in ("left", "right"):
        for j in ARM_JOINTS[arm]:
            s = jstart.get(j)
            if s is None:
                continue
            hold = -tau_g[s]
            L = lim.get(j, float("nan"))
            flag = "  <<< EXCEEDS LIMIT -> sags" if abs(hold) >= L - 0.5 else (
                "  (near limit)" if abs(hold) >= 0.7 * L else "")
            print(f"{j:24s} {hold:+13.2f} {L:10.1f}{flag}")
    # lift too
    s = jstart.get(LIFT_JOINT)
    if s is not None:
        print(f"{LIFT_JOINT:24s} {-tau_g[s]:+13.2f} {lim.get(LIFT_JOINT,float('nan')):10.1f}")
    g.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()

"""Absolute-orientation + rotation-lock validation for the Quest teleop (no DDS).

Drives the REAL ``M1QuestNode.on_xr_frame`` state machine (the node is built with
``__new__`` so we exercise the actual clutch/orientation/lock code without ROS
init or DDS sockets) with synthetic controller frames, and checks:

  1. Controller orientation maps to the gripper target ABSOLUTELY -- a given
     controller pose always commands the same gripper orientation, and a 90 deg
     controller rotation makes a 90 deg gripper rotation.
  2. It does NOT accumulate: grab/rotate/release, recenter the wrist, re-grab and
     rotate the SAME way -> the gripper ends at 90 deg, not 180 deg (the reported
     "it just adds 90 deg to the current rotation each time" bug).
  3. The rotation LOCK freezes the gripper orientation against hand twist, and
     unlocking resumes tracking with no jump.
  4. Position clutch still works (orientation change didn't break translation).

Run:  /usr/bin/python3 _quest_orientation_test.py   (with /opt/ros/jazzy sourced)
"""
import math
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ros2_ws/src/m1_control"))

from m1_control.quest_node import M1QuestNode  # noqa: E402
from m1_control.kinematics import mat_to_quat, quat_to_mat  # noqa: E402

ARMS = ("left", "right")


def make_node():
    """A node instance with just the state on_xr_frame touches (no ROS init)."""
    n = object.__new__(M1QuestNode)
    n._lock = threading.Lock()
    n._now = lambda: 0.0          # override the ROS-clock helper
    n.reach = None                # _viz_locked stays minimal
    n.enable_base = False
    n.motion_scale = 1.0
    n.q_meas = {}
    n.target = {a: [0.40, 0.0, 0.70] for a in ARMS}
    n.target_quat = {a: [0.0, 0.0, 0.0, 1.0] for a in ARMS}
    n.seeded = {a: False for a in ARMS}
    n.grip = {a: 0.0 for a in ARMS}
    n.cmd_vel = {"vx": 0.0, "vy": 0.0, "yaw": 0.0}
    n.clutch = {a: False for a in ARMS}
    n.clutch_hand0 = {a: None for a in ARMS}
    n.clutch_target0 = {a: None for a in ARMS}
    n.clutch_F = {a: None for a in ARMS}
    n.clutch_L = {a: None for a in ARMS}
    n.ori_C = {a: None for a in ARMS}
    n.ori_align = {a: None for a in ARMS}
    n.ori_locked = {a: False for a in ARMS}
    n.last_lock = {a: False for a in ARMS}
    n.last_btn = {a: False for a in ARMS}
    n._last_base_cmd = 0.0
    n._last_update = 0.0
    return n


def quat_axis_angle(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    s = math.sin(angle / 2.0)
    return [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2.0)]


def ctrl(pos, quat, squeeze=False, lock=False, button=False, trigger=0.0):
    return {"valid": True, "pos": list(pos), "quat": list(quat),
            "squeeze": squeeze, "lock": lock, "button": button,
            "trigger": trigger, "recenter": False, "stick": [0.0, 0.0]}


def frame(node, arm, c):
    """One XR frame driving ``arm`` (its controller is the cross-mapped one)."""
    phys = {"left": "right", "right": "left"}[arm]
    other = {"left": "right", "right": "left"}[phys]
    controllers = {phys: c, other: ctrl([0, 0, 0], [0, 0, 0, 1])}
    node.on_xr_frame({"controllers": controllers, "head": [0.0, 0.0, -1.0]})


def quat_angle_between(qa, qb):
    Ra, Rb = quat_to_mat(qa), quat_to_mat(qb)
    Rd = Ra @ Rb.T
    c = max(-1.0, min(1.0, (np.trace(Rd) - 1.0) * 0.5))
    return math.degrees(math.acos(c))


def tgt(node, arm="left"):
    return list(node.target_quat[arm])


def main():
    print("=== Quest absolute-orientation + rotation-lock validation ===")
    gates = {}
    H0 = [0.0, 0.0, 0.0, 1.0]                    # controller "straight"
    H90 = quat_axis_angle([1.0, 0.0, 0.0], math.radians(90))   # pitched up 90 deg
    p0 = [0.0, 0.0, 0.0]

    # --- 1. magnitude: controller +90 deg -> gripper +90 deg -----------------
    n = make_node()
    frame(n, "left", ctrl(p0, H0, squeeze=True))     # grab: calibrate at straight
    base = tgt(n)
    frame(n, "left", ctrl(p0, H90, squeeze=True))    # rotate controller up 90
    rotated = tgt(n)
    ang = quat_angle_between(rotated, base)
    print(f"1. controller +90deg -> gripper rotated {ang:.1f}deg (want ~90)")
    gates["+90deg controller -> ~90deg gripper"] = abs(ang - 90.0) < 3.0

    # --- 2. absolute, no accumulation ---------------------------------------
    # grab@straight, rotate to +90, RELEASE, recenter wrist to straight while
    # released, re-grab@straight, rotate to +90 again. Absolute -> 90deg from
    # zero; additive -> 180deg.
    n = make_node()
    frame(n, "left", ctrl(p0, H0, squeeze=True))     # grab, calibrate zero
    zero = tgt(n)
    frame(n, "left", ctrl(p0, H90, squeeze=True))    # rotate to +90
    frame(n, "left", ctrl(p0, H90, squeeze=False))   # release (still +90)
    frame(n, "left", ctrl(p0, H0, squeeze=False))    # recenter wrist (released)
    frame(n, "left", ctrl(p0, H0, squeeze=True))     # re-grab at straight (engage)
    frame(n, "left", ctrl(p0, H0, squeeze=True))     # held at straight -> tracks
    regrab = tgt(n)
    frame(n, "left", ctrl(p0, H90, squeeze=True))    # rotate to +90 again
    again = tgt(n)
    a_regrab = quat_angle_between(regrab, zero)
    a_again = quat_angle_between(again, zero)
    print(f"2. re-grab@straight -> {a_regrab:.1f}deg (want ~0, no snap), "
          f"then +90 again -> {a_again:.1f}deg (want ~90 ABS, not ~180 additive)")
    gates["re-grab at straight returns to zero (absolute)"] = a_regrab < 3.0
    gates["no accumulation (90deg not 180deg)"] = abs(a_again - 90.0) < 5.0

    # --- 3. same controller pose -> same gripper orientation -----------------
    # Two different paths to the same controller orientation must give the same
    # gripper orientation (path-independence == absolute).
    n = make_node()
    frame(n, "left", ctrl(p0, H0, squeeze=True))
    Hother = quat_axis_angle([0, 1, 0], math.radians(70))
    frame(n, "left", ctrl(p0, Hother, squeeze=True))   # wander off
    frame(n, "left", ctrl(p0, H90, squeeze=True))      # arrive at H90 via Hother
    viaA = tgt(n)
    frame(n, "left", ctrl(p0, H0, squeeze=True))       # straight
    frame(n, "left", ctrl(p0, H90, squeeze=True))      # arrive at H90 directly
    viaB = tgt(n)
    spread = quat_angle_between(viaA, viaB)
    print(f"3. same controller pose via 2 paths -> gripper differs {spread:.2f}deg (want ~0)")
    gates["path-independent (absolute) orientation"] = spread < 1.0

    # --- 4. rotation lock freezes orientation, unlock resumes w/o jump -------
    n = make_node()
    frame(n, "left", ctrl(p0, H0, squeeze=True))
    frame(n, "left", ctrl(p0, H90, squeeze=True))      # at +90
    locked_at = tgt(n)
    frame(n, "left", ctrl(p0, H90, squeeze=True, lock=True))   # toggle LOCK on
    Hwild = quat_axis_angle([0, 0, 1], math.radians(120))
    frame(n, "left", ctrl(p0, Hwild, squeeze=True))    # twist hard while locked
    frame(n, "left", ctrl(p0, H0, squeeze=True))       # and back to straight
    during_lock = tgt(n)
    drift = quat_angle_between(during_lock, locked_at)
    print(f"4a. gripper drift while locked + twisting {drift:.2f}deg (want ~0)")
    gates["lock freezes orientation"] = drift < 0.5
    # Unlock (controller currently at straight); orientation must not jump, then
    # resume tracking from the held orientation.
    frame(n, "left", ctrl(p0, H0, squeeze=True, lock=True))    # toggle LOCK off
    after_unlock = tgt(n)
    jump = quat_angle_between(after_unlock, locked_at)
    frame(n, "left", ctrl(p0, H90, squeeze=True))      # now twisting works again
    resumed = tgt(n)
    moved = quat_angle_between(resumed, locked_at)
    print(f"4b. unlock jump {jump:.2f}deg (want ~0) | then twist moves it {moved:.1f}deg (want >10)")
    gates["unlock without jump"] = jump < 0.5
    gates["tracking resumes after unlock"] = moved > 10.0

    # --- 5. position clutch still works -------------------------------------
    n = make_node()
    frame(n, "left", ctrl([0, 0, 0], H0, squeeze=True))        # grab at origin
    frame(n, "left", ctrl([0.0, 0.10, 0.0], H0, squeeze=True))  # raise hand 10cm (WebXR +y = up)
    dz = n.target["left"][2] - 0.70
    print(f"5. raise hand 10cm -> target z moved {dz*100:.1f}cm (want ~+10)")
    gates["position clutch intact"] = abs(dz - 0.10) < 0.02

    print("\n---- GATES ----")
    npass = 0
    for k, v in gates.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        npass += int(v)
    print(f"\n{npass}/{len(gates)} gates passed")
    return 0 if npass == len(gates) else 1


if __name__ == "__main__":
    sys.exit(main())

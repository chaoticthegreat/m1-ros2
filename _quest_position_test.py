"""Position-only Quest teleop validation (no DDS).

The Quest reach is position-only now: hands drive the target POINT, the gripper
rotation is not controlled, the thumbstick-click toggles a PRECISION mode, and an
in-headset HUD shows each arm's target->fingertip error. This drives the REAL
``M1QuestNode.on_xr_frame`` / ``_viz_locked`` / ``snapshot`` (built via ``__new__``
so no ROS init / DDS) with synthetic controller frames and checks:

  1. Position clutch: hand motion while gripped moves the target point.
  2. Precision mode (thumbstick click) scales hand motion down, and is edge-
     triggered (holding the click doesn't toggle every frame).
  3. A/X re-seeds the target to the live fingertip ("home to here").
  4. Thumbsticks drive the base (vx / vy / yaw).
  5. The error-window data path: _viz_locked + snapshot report the per-arm
     target<->fingertip error the HUD renders.

Run:  /usr/bin/python3 _quest_position_test.py   (with /opt/ros/jazzy sourced)
"""
import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ros2_ws/src/m1_control"))

from m1_control.quest_node import M1QuestNode, PRECISION_SCALE  # noqa: E402
from m1_control.kinematics import (  # noqa: E402
    ARM_JOINTS, LIFT_JOINT, ReachController, UrdfModel,
)
from m1_control.swerve import SwerveOdometry  # noqa: E402

ARMS = ("left", "right")
URDF = "assets/ranger_air_description/urdf/ranger_air_description.urdf"


def make_node(reach=None, enable_base=False):
    """A node with just the state on_xr_frame / _viz_locked / snapshot touch."""
    n = object.__new__(M1QuestNode)
    n._lock = threading.Lock()
    n._now = lambda: 0.0
    n.reach = reach
    n.enable_base = enable_base
    n.motion_scale = 1.0
    n.q_meas = {}
    n.target = {a: [0.40, 0.0, 0.70] for a in ARMS}
    n.err = {a: None for a in ARMS}
    n.seeded = {a: False for a in ARMS}
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
    n._last_base_cmd = 0.0
    n._last_update = 0.0
    n.odom = SwerveOdometry()
    n._last_tick = 0.0
    return n


def ctrl(pos, squeeze=False, lock=False, button=False, trigger=0.0, stick=(0.0, 0.0)):
    return {"valid": True, "pos": list(pos), "squeeze": squeeze, "lock": lock,
            "button": button, "trigger": trigger, "recenter": False,
            "stick": list(stick)}


def frame(node, arm, c, head=(0.0, 0.0, -1.0)):
    """One XR frame driving ``arm`` (its controller is the cross-mapped one)."""
    phys = {"left": "right", "right": "left"}[arm]
    other = {"left": "right", "right": "left"}[phys]
    controllers = {phys: c, other: ctrl([0, 0, 0])}
    node.on_xr_frame({"controllers": controllers, "head": list(head)})


def _cfg(reach, arm, lift=0.4):
    q = {j: 0.0 for j in ARM_JOINTS["left"] + ARM_JOINTS["right"]}
    q[LIFT_JOINT] = lift
    q[ARM_JOINTS[arm][1]] = 0.5
    q[ARM_JOINTS[arm][3]] = 0.8
    return q


def main():
    print("=== Quest position-only teleop validation ===")
    gates = {}

    # --- 1. position clutch ---------------------------------------------------
    n = make_node()
    frame(n, "left", ctrl([0, 0, 0], squeeze=True))            # grab at origin
    frame(n, "left", ctrl([0.0, 0.10, 0.0], squeeze=True))     # raise hand 10cm (+y up)
    dz = n.target["left"][2] - 0.70
    print(f"1. raise hand 10cm -> target z moved {dz*100:.1f}cm (want ~+10)")
    gates["position clutch intact"] = abs(dz - 0.10) < 0.02

    # --- 2. precision mode scales motion + is edge-triggered ------------------
    n = make_node()
    frame(n, "left", ctrl([0, 0, 0], squeeze=True, lock=True))  # grab + click (fine ON)
    frame(n, "left", ctrl([0, 0, 0], squeeze=True, lock=True))  # HOLD click (no re-toggle)
    fine_on = n.fine["left"]
    frame(n, "left", ctrl([0.0, 0.10, 0.0], squeeze=True))      # raise 10cm in fine mode
    dz_fine = n.target["left"][2] - 0.70
    print(f"2. fine mode {'ON' if fine_on else 'off'} (held click, single toggle) | "
          f"raise 10cm -> target z {dz_fine*100:.2f}cm (want ~{10*PRECISION_SCALE:.1f})")
    gates["precision toggles once on click (edge)"] = fine_on is True
    gates["precision scales motion down"] = abs(dz_fine - 0.10 * PRECISION_SCALE) < 0.01

    # toggle OFF again restores 1:1
    n2 = make_node()
    frame(n2, "left", ctrl([0, 0, 0], lock=True))   # click on (released)
    frame(n2, "left", ctrl([0, 0, 0]))              # release click
    frame(n2, "left", ctrl([0, 0, 0], lock=True))   # click off
    gates["precision toggles back off"] = n2.fine["left"] is False

    # --- 3. A/X re-seeds target to live fingertip -----------------------------
    reach = ReachController(UrdfModel.from_string(open(URDF).read()))
    n = make_node(reach=reach)
    n.q_meas = _cfg(reach, "left")
    n.target["left"] = [0.0, 0.0, 0.0]              # deliberately wrong
    frame(n, "left", ctrl([0, 0, 0], button=True))  # A/X -> reseed
    tip = np.asarray(reach.fingertip("left", n.q_meas))
    err = float(np.linalg.norm(np.asarray(n.target["left"]) - tip))
    print(f"3. A/X reseed -> target at fingertip, err {err*1e3:.3f}mm (want ~0)")
    gates["A/X reseeds to fingertip"] = err < 1e-6

    # --- 4. thumbsticks drive the base ---------------------------------------
    n = make_node(enable_base=True)
    # left stick pushed forward (y=-1), right stick pushed right (x=+1)
    controllers = {
        "left": ctrl([0, 0, 0], stick=(0.0, -1.0)),
        "right": ctrl([0, 0, 0], stick=(1.0, 0.0)),
    }
    n.on_xr_frame({"controllers": controllers, "head": [0.0, 0.0, -1.0]})
    print(f"4. base cmd vx {n.cmd_vel['vx']:+.2f} (want >0, fwd) | "
          f"yaw {n.cmd_vel['yaw']:+.2f} (want <0, right turn)")
    gates["left stick fwd -> vx>0"] = n.cmd_vel["vx"] > 0.1
    gates["right stick -> yaw"] = abs(n.cmd_vel["yaw"]) > 0.1

    # --- 5. error-window data path (_viz_locked + snapshot) ------------------
    reach = ReachController(UrdfModel.from_string(open(URDF).read()))
    n = make_node(reach=reach)
    n.q_meas = _cfg(reach, "left")
    n.q_meas.update(_cfg(reach, "right"))
    tipL = np.asarray(reach.fingertip("left", n.q_meas))
    n.target["left"] = [float(tipL[0]), float(tipL[1]) + 0.05, float(tipL[2])]  # 50mm off
    viz = n._viz_locked()
    dist = viz["arms"]["left"].get("dist")
    snap = n.snapshot()
    err_mm = snap["arms"]["left"]["err_mm"]
    print(f"5. viz dist {dist*1e3 if dist is not None else None:.1f}mm | "
          f"snapshot err_mm {err_mm}mm (want ~50)")
    gates["viz reports per-arm reach error"] = dist is not None and abs(dist - 0.05) < 1e-3
    gates["snapshot err_mm matches (HUD source)"] = err_mm is not None and abs(err_mm - 50.0) < 1.0
    gates["snapshot exposes precision flag"] = "fine" in snap["arms"]["left"]

    print("\n---- GATES ----")
    npass = 0
    for k, v in gates.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        npass += int(v)
    print(f"\n{npass}/{len(gates)} gates passed")
    return 0 if npass == len(gates) else 1


if __name__ == "__main__":
    sys.exit(main())

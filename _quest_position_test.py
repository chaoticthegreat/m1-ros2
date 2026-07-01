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
    # RLock to match the real node: on_xr_frame/on_ctrl_frame hold the lock and
    # call the shared _apply_place/_apply_controls helpers, which re-acquire it.
    n._lock = threading.RLock()
    n._last_ctrl_seq = None
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
    n.control_mode = "relative"
    n._last_mode_chord = False
    n._last_base_cmd = 0.0
    n._last_update = 0.0
    n.odom = SwerveOdometry()
    n._last_tick = 0.0
    return n


def ctrl(pos, squeeze=False, lock=False, button=False, trigger=0.0, stick=(0.0, 0.0),
         recenter=False, pos_base=None):
    c = {"valid": True, "pos": list(pos), "squeeze": squeeze, "lock": lock,
         "button": button, "trigger": trigger, "recenter": recenter,
         "stick": list(stick)}
    if pos_base is not None:
        # ABSOLUTE scheme: hand position in base_link frame (the page computes this
        # via the inverse hologram anchor; here we supply it directly).
        c["pos_base"] = list(pos_base)
    return c


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

    # --- 2b. relative-mode precision toggle is continuous (no mid-grab snap) ---
    # BUG7 regression guard: toggling precision while gripped in the DEFAULT
    # relative scheme must re-anchor (clutch_hand0/clutch_target0) so the target
    # does NOT jump -- previously the re-anchor was gated to absolute mode only,
    # so toggling precision mid-grab in relative re-scaled the whole accumulated
    # delta and snapped the target by (1-PRECISION_SCALE)*|delta|.
    nr = make_node()                                    # control_mode defaults to "relative"
    frame(nr, "left", ctrl([0, 0, 0], squeeze=True))    # grab at origin
    frame(nr, "left", ctrl([0.0, 0.10, 0.0], squeeze=True))   # raise hand 10cm (1:1)
    before = np.asarray(nr.target["left"], dtype=float).copy()
    frame(nr, "left", ctrl([0.0, 0.10, 0.0], squeeze=True, lock=True))  # toggle precision, hand STILL
    after = np.asarray(nr.target["left"], dtype=float)
    snap_mm = float(np.linalg.norm(after - before)) * 1000.0
    print(f"2b. relative precision toggle, hand held still -> target moved "
          f"{snap_mm:.4f}mm (want ~0)")
    gates["relative precision toggle is continuous (no snap)"] = snap_mm < 1e-2
    # and a subsequent small hand move now scales by PRECISION_SCALE
    frame(nr, "left", ctrl([0.0, 0.12, 0.0], squeeze=True))   # +2cm hand, fine mode
    dz_after = nr.target["left"][2] - after[2]
    gates["relative precision then scales motion"] = abs(dz_after - 0.02 * PRECISION_SCALE) < 0.005

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
    gates["snapshot exposes control scheme"] = snap.get("mode") == "relative"

    # --- 6. ABSOLUTE scheme: the hand's ACTUAL position IS the target ----------
    # Embodiment: CROSS-mapped (the LEFT arm is driven by the physical RIGHT
    # controller -- verified live in the headset), and the gripper target follows
    # the controller's body-frame position, not a delta.
    n = make_node()
    n.control_mode = "absolute"

    def aframe(pos_base, squeeze=True, lock=False):
        controllers = {
            "right": ctrl([0, 0, 0], squeeze=squeeze, lock=lock, pos_base=pos_base),
            "left": ctrl([0, 0, 0]),
        }
        n.on_xr_frame({"controllers": controllers, "head": [0.0, 0.0, -1.0]})

    aframe([0.30, 0.10, 0.80])                 # squeeze (engage): snap target to hand
    t0 = list(n.target["left"])
    aframe([0.35, 0.10, 0.80])                 # move hand +5cm in x: target follows 1:1
    t1 = list(n.target["left"])
    print(f"6. absolute engage -> target {[round(v,3) for v in t0]} (want [.30,.10,.80]); "
          f"move hand +5cm x -> {[round(v,3) for v in t1]} (want [.35,.10,.80])")
    gates["absolute snaps target to the hand on engage"] = (
        max(abs(t0[0]-0.30), abs(t0[1]-0.10), abs(t0[2]-0.80)) < 1e-6)
    gates["absolute tracks the hand 1:1 (actual pose)"] = (
        max(abs(t1[0]-0.35), abs(t1[1]-0.10), abs(t1[2]-0.80)) < 1e-6)

    # --- 7. ABSOLUTE + precision scales motion around the engage point ---------
    n = make_node()
    n.control_mode = "absolute"
    n.fine["left"] = True
    controllers = {"right": ctrl([0, 0, 0], squeeze=True, pos_base=[0.30, 0.10, 0.80]),
                   "left": ctrl([0, 0, 0])}
    n.on_xr_frame({"controllers": controllers, "head": [0.0, 0.0, -1.0]})   # engage (fine)
    controllers = {"right": ctrl([0, 0, 0], squeeze=True, pos_base=[0.40, 0.10, 0.80]),
                   "left": ctrl([0, 0, 0])}
    n.on_xr_frame({"controllers": controllers, "head": [0.0, 0.0, -1.0]})   # +10cm x
    dx = n.target["left"][0] - 0.30
    print(f"7. absolute fine: move hand +10cm x -> target x moved {dx*100:.2f}cm "
          f"(want ~{10*PRECISION_SCALE:.1f})")
    gates["absolute precision scales around engage"] = abs(dx - 0.10 * PRECISION_SCALE) < 1e-6

    # --- 8. A/X+B/Y chord toggles the scheme (edge-triggered, no re-home) ------
    reach = ReachController(UrdfModel.from_string(open(URDF).read()))
    n = make_node(reach=reach)
    n.q_meas = _cfg(reach, "left")
    n.q_meas.update(_cfg(reach, "right"))
    for a in ARMS:
        n.seeded[a] = True
        n.target[a] = [0.0, 0.0, 0.0]          # if a re-home leaked, this would move
    chord = ctrl([0, 0, 0], button=True, recenter=True)
    idle = ctrl([0, 0, 0])

    def cframe(left, right):
        n.on_xr_frame({"controllers": {"left": left, "right": right},
                       "head": [0.0, 0.0, -1.0]})

    cframe(chord, idle)                          # chord down: relative -> absolute
    mode_after = n.control_mode
    cframe(chord, idle)                          # held: no re-toggle (edge)
    held_mode = n.control_mode
    cframe(idle, idle)                           # release
    cframe(chord, idle)                          # chord again: absolute -> relative
    mode_back = n.control_mode
    homed = max(max(abs(v) for v in n.target[a]) for a in ARMS)
    print(f"8. chord: relative -> {mode_after} (held -> {held_mode}) -> {mode_back}; "
          f"max |target| after (re-home suppressed, want 0): {homed:.3f}")
    gates["chord toggles relative -> absolute"] = mode_after == "absolute"
    gates["chord is edge-triggered (held doesn't re-toggle)"] = held_mode == "absolute"
    gates["chord does NOT also re-home an arm"] = homed < 1e-9
    gates["chord toggles back to relative"] = mode_back == "relative"

    # --- 9. ABSOLUTE robustness: missing / non-finite pos_base + recenter ------
    # (absolute is CROSS-mapped: the physical RIGHT controller drives the LEFT arm.)
    # (a) squeeze with NO pos_base (e.g. before the hologram is placed) -> the arm
    #     stays idle (well-defined), it does not silently hold a grab or move.
    n = make_node()
    n.control_mode = "absolute"
    n.target["left"] = [0.40, 0.0, 0.70]
    n.on_xr_frame({"controllers": {"right": ctrl([0, 0, 0], squeeze=True),
                                   "left": ctrl([0, 0, 0])}, "head": [0, 0, -1.0]})
    a_idle = (not n.clutch["left"]) and n.target["left"] == [0.40, 0.0, 0.70]
    # (b) non-finite pos_base (degenerate anchor inverse) -> ignored; the published
    #     target never becomes NaN/Inf (z is unbounded, so the clamp can't save it).
    n.on_xr_frame({"controllers": {"right": ctrl([0, 0, 0], squeeze=True,
                                                 pos_base=[float("inf"), 0.1, 0.8]),
                                   "left": ctrl([0, 0, 0])}, "head": [0, 0, -1.0]})
    b_finite = (not n.clutch["left"]) and all(np.isfinite(v) for v in n.target["left"])
    # (c) recenter (place) while engaged in FINE mode: the anchor must be refreshed
    #     to the new frame, else target = old_anchor + scale*(new-old) snaps. With
    #     the fix the grab re-snaps to the new hand exactly (no stale-frame jump).
    n2 = make_node()
    n2.control_mode = "absolute"
    n2.fine["left"] = True
    n2.on_xr_frame({"controllers": {"right": ctrl([0, 0, 0], squeeze=True, pos_base=[0.30, 0.10, 0.80]),
                                    "left": ctrl([0, 0, 0])}, "head": [0, 0, -1.0]})       # engage
    c_engaged = n2.clutch["left"]
    n2.on_xr_frame({"controllers": {"right": ctrl([0, 0, 0], squeeze=True, pos_base=[0.50, 0.20, 0.90]),
                                    "left": ctrl([0, 0, 0])}, "head": [0, 0, -1.0], "place": True})  # recenter
    c_resnap = max(abs(n2.target["left"][i] - v) for i, v in enumerate([0.50, 0.20, 0.90])) < 1e-6
    print(f"9. absolute robustness: no-anchor idle {a_idle}; non-finite ignored {b_finite}; "
          f"recenter re-anchors (no snap) {c_engaged and c_resnap}")
    gates["absolute: squeeze without an anchor stays idle"] = a_idle
    gates["absolute: non-finite pos_base ignored (target finite)"] = b_finite
    gates["absolute: recenter re-anchors (no stale-frame snap)"] = c_engaged and c_resnap

    print("\n---- GATES ----")
    npass = 0
    for k, v in gates.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        npass += int(v)
    print(f"\n{npass}/{len(gates)} gates passed")
    return 0 if npass == len(gates) else 1


if __name__ == "__main__":
    sys.exit(main())

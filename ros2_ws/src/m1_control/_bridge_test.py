"""Hardware-free unit tests for the three real-hardware bridge mappers.

Covers the PURE FUNCTIONS at the core of the new bridge nodes -- no ROS, no DDS,
no hardware -- the project's offline-gate idiom (cf. ``_ros_reach_check.py`` /
``_quest_position_test.py`` drive the real logic without a live graph):

  * joint_command_bridge.map_command   -- /m1/joint_command -> 19-vector pick,
                                           reorder / missing-name / drop-steer-wheel
  * base_bridge.select_motion_mode      -- AgileX PARALLEL / SPINNING /
                                           DUAL_ACKERMANN mode switch
  * ranger_state_shim.steer_wheel_to_jointstate -- 4 steer + 4 wheel ->
                                           8 base joints, names / order / signs

The bridge node CLASSES import ``rclpy`` at module top level; to keep this test
truly ROS-free we stub the ROS message/`rclpy` modules BEFORE importing, then
exercise only the pure functions + module constants. (If ROS is actually on the
path the stubs are simply unused for those imports.)

Run:
  PYTHONPATH=ros2_ws/src/m1_control /usr/bin/python3 _bridge_test.py
"""

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))


def _stub_ros():
    """Install minimal stub modules so the bridge nodes import with no ROS.

    Only the names the bridge modules reference at import time are provided;
    the node classes are never instantiated here, so the stubs need no behaviour.
    """
    def _mod(name):
        m = sys.modules.get(name)
        if m is None:
            m = types.ModuleType(name)
            sys.modules[name] = m
        return m

    rclpy = _mod("rclpy")
    rclpy.init = lambda *a, **k: None
    rclpy.spin = lambda *a, **k: None
    rclpy.ok = lambda *a, **k: False
    rclpy.shutdown = lambda *a, **k: None
    node_mod = _mod("rclpy.node")
    node_mod.Node = type("Node", (), {})
    rclpy.node = node_mod

    class _Msg:
        def __init__(self, *a, **k):
            for key, val in k.items():
                setattr(self, key, val)

    sensor = _mod("sensor_msgs.msg")
    sensor.JointState = _Msg
    _mod("sensor_msgs").msg = sensor
    geo = _mod("geometry_msgs.msg")
    geo.Twist = _Msg
    _mod("geometry_msgs").msg = geo
    std = _mod("std_msgs.msg")
    std.Float64MultiArray = _Msg
    std.Int8 = _Msg
    _mod("std_msgs").msg = std


# Use real rclpy if present (e.g. ROS sourced); otherwise stub it so the pure
# functions remain importable with no ROS install at all.
try:
    import rclpy  # noqa: F401
except Exception:  # noqa: BLE001
    _stub_ros()

from m1_control import base_bridge, joint_command_bridge, ranger_state_shim
from m1_control.joint_command_bridge import UPPER_BODY, map_command
from m1_control.base_bridge import select_motion_mode
from m1_control.ranger_state_shim import BASE_JOINTS, steer_wheel_to_jointstate
from m1_control.swerve import STEER_DIR, STEER_JOINTS, WHEEL_DIR, WHEEL_JOINTS

_results = []


def check(name, ok, detail=""):
    _results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


# --------------------------------------------------------------------------
# joint_command_bridge.map_command
# --------------------------------------------------------------------------
def test_map_command_canonical_passthrough():
    # Input already in the 17-order maps to identical positions.
    pos = [float(i) for i in range(len(UPPER_BODY))]
    out = map_command(list(UPPER_BODY), pos, UPPER_BODY)
    check("map_command: canonical pass-through preserves values",
          out == pos and len(out) == 17, f"len={len(out)}")


def test_map_command_reorders_by_name():
    # Shuffle the input order; output must still be in UPPER_BODY order,
    # carrying each joint's own value.
    want = {jn: float(i) + 0.5 for i, jn in enumerate(UPPER_BODY)}
    names = list(reversed(UPPER_BODY))
    pos = [want[n] for n in names]
    out = map_command(names, pos, UPPER_BODY)
    expect = [want[n] for n in UPPER_BODY]
    check("map_command: reorders out-of-order input by name", out == expect)


def test_map_command_missing_is_zero():
    # Drop two joints from the input -> they read 0.0, others preserved.
    present = [jn for jn in UPPER_BODY if jn not in
               ("openarm_left_joint4", "openarm_right_finger_joint1")]
    pos = [float(i) + 1.0 for i in range(len(present))]
    lookup = dict(zip(present, pos))
    out = map_command(present, pos, UPPER_BODY)
    ok = (len(out) == 17
          and out[UPPER_BODY.index("openarm_left_joint4")] == 0.0
          and out[UPPER_BODY.index("openarm_right_finger_joint1")] == 0.0
          and out[0] == lookup["lift_joint"])
    check("map_command: missing name -> 0.0, others preserved", ok)


def test_map_command_drops_steer_and_wheel():
    # A full 27-DOF /m1/joint_command (steer + lift + arms + fingers + wheels):
    # only the 17 commanded upper-body entries survive; steer/wheel AND the two
    # finger_joint2 mimics are dropped.
    extra = ["openarm_left_finger_joint2", "openarm_right_finger_joint2"]
    full_names = (list(STEER_JOINTS) + UPPER_BODY + extra + list(WHEEL_JOINTS))
    full_pos = []
    for jn in full_names:
        if jn in STEER_JOINTS or jn in WHEEL_JOINTS or jn in extra:
            full_pos.append(999.0)        # poison: must NOT appear in output
        else:
            full_pos.append(float(UPPER_BODY.index(jn)))
    out = map_command(full_names, full_pos, UPPER_BODY)
    expect = [float(i) for i in range(17)]
    check("map_command: drops steer/wheel/mimic, keeps the 17 commanded",
          out == expect and 999.0 not in out, f"len={len(out)}")


def test_upper_body_exact_order():
    expect = [
        "lift_joint",
        "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
        "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6",
        "openarm_left_joint7",
        "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
        "openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6",
        "openarm_right_joint7",
        "openarm_left_finger_joint1", "openarm_right_finger_joint1",
    ]
    check("UPPER_BODY: exact 17-commanded-joint canonical order",
          UPPER_BODY == expect, f"len={len(UPPER_BODY)}")


# --------------------------------------------------------------------------
# base_bridge.select_motion_mode
# --------------------------------------------------------------------------
def test_mode_parallel_on_strafe():
    # Significant vy -> PARALLEL, pass vx & vy, FORCE yaw=0 (even if yaw given).
    mode, lx, ly, az = select_motion_mode(0.3, 0.4, 0.9)
    check("select_motion_mode: strafe -> PARALLEL, yaw forced 0",
          mode == "PARALLEL" and lx == 0.3 and ly == 0.4 and az == 0.0,
          f"mode={mode} ({lx},{ly},{az})")


def test_mode_spinning_on_pure_yaw():
    # Only yaw (linear ~0) -> SPINNING, FORCE linear 0, pass yaw.
    mode, lx, ly, az = select_motion_mode(0.0, 0.0, 0.8)
    check("select_motion_mode: pure yaw -> SPINNING, linear forced 0",
          mode == "SPINNING" and lx == 0.0 and ly == 0.0 and az == 0.8,
          f"mode={mode} ({lx},{ly},{az})")


def test_mode_dual_ackermann_on_drive():
    # vx + small yaw, no strafe -> DUAL_ACKERMANN, pass vx & yaw, vy forced 0.
    mode, lx, ly, az = select_motion_mode(0.5, 0.0, 0.2)
    check("select_motion_mode: drive+turn -> DUAL_ACKERMANN, vy forced 0",
          mode == "DUAL_ACKERMANN" and lx == 0.5 and ly == 0.0 and az == 0.2,
          f"mode={mode} ({lx},{ly},{az})")


def test_mode_dual_ackermann_pure_drive():
    # Straight drive, no yaw -> DUAL_ACKERMANN.
    mode, lx, ly, az = select_motion_mode(0.6, 0.0, 0.0)
    check("select_motion_mode: straight drive -> DUAL_ACKERMANN",
          mode == "DUAL_ACKERMANN" and lx == 0.6 and ly == 0.0 and az == 0.0,
          f"mode={mode}")


def test_mode_strafe_dominates_yaw():
    # Strafe takes priority over yaw (mode-switched, can't do both): vy wins.
    mode, _lx, _ly, az = select_motion_mode(0.0, 0.5, 1.5)
    check("select_motion_mode: strafe priority over yaw (yaw dropped)",
          mode == "PARALLEL" and az == 0.0, f"mode={mode} az={az}")


def test_mode_epsilon_deadband():
    # Tiny components below epsilon read as zero -> still DUAL_ACKERMANN at rest.
    mode, lx, ly, az = select_motion_mode(1e-5, 1e-5, 1e-5)
    check("select_motion_mode: sub-epsilon -> DUAL_ACKERMANN (rest)",
          mode == "DUAL_ACKERMANN" and lx == 1e-5 and ly == 0.0 and az == 1e-5,
          f"mode={mode}")


# --------------------------------------------------------------------------
# ranger_state_shim.steer_wheel_to_jointstate
# --------------------------------------------------------------------------
def test_shim_names_and_order():
    expect = [
        "fl_steering_joint", "fr_steering_joint",
        "rr_steering_joint", "rl_steering_joint",
        "fl_wheel_joint", "fr_wheel_joint",
        "rr_wheel_joint", "rl_wheel_joint",
    ]
    names, pos, vel = steer_wheel_to_jointstate([0, 0, 0, 0], [0, 0, 0, 0])
    check("steer_wheel_to_jointstate: 8 base joints in canonical order",
          names == expect == BASE_JOINTS and len(pos) == 8 and len(vel) == 8,
          f"names={names}")


def test_shim_steer_to_position_wheel_to_velocity():
    # Steering -> position slots (first 4); wheel -> velocity slots (last 4);
    # the cross slots are zero.
    steer = [0.1, 0.2, 0.3, 0.4]
    wheel = [1.0, 2.0, 3.0, 4.0]
    names, pos, vel = steer_wheel_to_jointstate(steer, wheel)
    # Steering joints carry position, no velocity; wheels carry velocity, no pos.
    ok = (all(vel[i] == 0.0 for i in range(4))
          and all(pos[i] == 0.0 for i in range(4, 8)))
    check("steer_wheel_to_jointstate: steer->position, wheel->velocity, no crosstalk",
          ok, f"pos={pos} vel={vel}")


def test_shim_applies_swerve_sign_conventions():
    # The shim must apply swerve.py's STEER_DIR / WHEEL_DIR exactly. With unit
    # inputs, each slot equals that joint's direction sign.
    steer = [1.0, 1.0, 1.0, 1.0]
    wheel = [1.0, 1.0, 1.0, 1.0]
    names, pos, vel = steer_wheel_to_jointstate(steer, wheel)
    steer_ok = all(pos[k] == STEER_DIR[STEER_JOINTS[k]] for k in range(4))
    wheel_ok = all(vel[4 + k] == WHEEL_DIR[WHEEL_JOINTS[k]] for k in range(4))
    # Sanity: rr_steering_joint and rl... actually have a -1 / +1 mix and the
    # rl_wheel is the negated one -- assert the known-negated entries explicitly.
    known = (pos[BASE_JOINTS.index("rr_steering_joint")] == -1.0
             and vel[BASE_JOINTS.index("rl_wheel_joint")] == -1.0)
    check("steer_wheel_to_jointstate: applies STEER_DIR / WHEEL_DIR signs",
          steer_ok and wheel_ok and known,
          f"rr_steer={pos[BASE_JOINTS.index('rr_steering_joint')]} "
          f"rl_wheel={vel[BASE_JOINTS.index('rl_wheel_joint')]}")


def test_shim_rejects_wrong_length():
    ok = False
    try:
        steer_wheel_to_jointstate([0, 0, 0], [0, 0, 0, 0])
    except ValueError:
        ok = True
    check("steer_wheel_to_jointstate: rejects non-length-4 input", ok)


def main():
    tests = [
        test_map_command_canonical_passthrough,
        test_map_command_reorders_by_name,
        test_map_command_missing_is_zero,
        test_map_command_drops_steer_and_wheel,
        test_upper_body_exact_order,
        test_mode_parallel_on_strafe,
        test_mode_spinning_on_pure_yaw,
        test_mode_dual_ackermann_on_drive,
        test_mode_dual_ackermann_pure_drive,
        test_mode_strafe_dominates_yaw,
        test_mode_epsilon_deadband,
        test_shim_names_and_order,
        test_shim_steer_to_position_wheel_to_velocity,
        test_shim_applies_swerve_sign_conventions,
        test_shim_rejects_wrong_length,
    ]
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            check(t.__name__, False, f"raised {type(exc).__name__}: {exc}")

    npass = sum(_results)
    total = len(_results)
    print(f"\n==== {npass}/{total} gates passed ====", flush=True)
    sys.exit(0 if npass == total else 1)


if __name__ == "__main__":
    main()

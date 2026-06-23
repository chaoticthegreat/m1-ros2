"""Standalone swerve kinematics + odometry test (no ROS).

Exercises the swerve module the way the controller and the Quest viz use it and
reports PASS/FAIL gates so a change is easy to verify or catch a regression:

  A. IK <-> FK round trip   (module_states -> forward_kinematics recovers the
                             commanded body velocity to machine precision)
  B. Solver geometry        (drive / strafe / turn-in-place point the modules
                             the right way; the <=90 deg flip never over-swings)
  C. Desaturation           (an over-fast command is scaled uniformly: capped
                             magnitude, unchanged direction)
  D. Solver round trip      (decode the settled steer/spin commands back through
                             forward_kinematics -> recovers the body velocity)
  E. Odometry               (straight, strafe, turn-in-place, and a full circle
                             integrate to the analytic pose)

Run:
  /usr/bin/python3 _swerve_test.py [label]
Exit code is 0 only if every gate passes.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ros2_ws/src/m1_control"))

from m1_control import swerve  # noqa: E402
from m1_control.swerve import (  # noqa: E402
    MAX_WHEEL_SPEED,
    STEER_DIR,
    STEER_JOINTS,
    WHEEL_DIR,
    WHEEL_JOINTS,
    WHEEL_RADIUS,
    SwerveOdometry,
    SwerveSolver,
    desaturate,
    forward_kinematics,
    module_states,
    wrap_to_pi,
)


# --- A. IK <-> FK round trip -------------------------------------------------
def test_roundtrip():
    print("A. IK <-> FK ROUND TRIP")
    rng = np.random.default_rng(0)
    err = 0.0
    for _ in range(2000):
        vx = rng.uniform(-1.0, 1.0)
        vy = rng.uniform(-1.0, 1.0)
        yaw = rng.uniform(-2.0, 2.0)
        st = module_states(vx, vy, yaw)
        h = [s[0] for s in st]
        sp = [s[1] for s in st]
        rvx, rvy, ryaw = forward_kinematics(h, sp)
        err = max(err, abs(rvx - vx), abs(rvy - vy), abs(ryaw - yaw))
    print(f"   worst recovery error over 2000 random twists: {err:.2e}")
    return {"IK/FK round-trip < 1e-9": err < 1e-9}, {"roundtrip_err": err}


# --- B. solver geometry ------------------------------------------------------
def _applied_headings(solver):
    return list(solver.wheel_head)


def _settle(solver, vx, vy, yaw, ticks=300, dt=1.0 / 60.0):
    steer = wheel = None
    for _ in range(ticks):
        steer, wheel = solver.solve(vx, vy, yaw, dt)
    return steer, wheel


def test_geometry():
    print("B. SOLVER GEOMETRY")
    # Forward: every module points along +x (heading 0), wheels spin same way.
    s = SwerveSolver()
    _settle(s, 0.5, 0.0, 0.0)
    fwd_head_ok = all(abs(wrap_to_pi(h - 0.0)) < 1e-3 for h in _applied_headings(s))
    # Strafe left (+y): every module points along +y (heading +pi/2).
    s = SwerveSolver()
    _settle(s, 0.0, 0.4, 0.0)
    strafe_head_ok = all(abs(wrap_to_pi(h - math.pi / 2)) < 1e-3 for h in _applied_headings(s))
    # Turn in place: each module drives tangent to its radius, i.e. heading is
    # perpendicular to the (mx,my) vector. For +yaw the module velocity is
    # (-yaw*my, +yaw*mx); check the applied heading matches atan2 of that.
    s = SwerveSolver()
    _settle(s, 0.0, 0.0, 1.0)
    spin_ok = True
    for k, jn in enumerate(WHEEL_JOINTS):
        mx, my = swerve.MODULE_XY[jn.split("_")[0]]
        want = math.atan2(mx, -my)  # atan2(+yaw*mx, -yaw*my) with yaw>0
        # The applied heading may be the 180-flip equivalent (wheel reversed).
        h = s.wheel_head[k]
        d = min(abs(wrap_to_pi(h - want)), abs(wrap_to_pi(h - want - math.pi)))
        spin_ok = spin_ok and d < 1e-3
    print(f"   forward modules @0 rad: {fwd_head_ok} | strafe @pi/2: {strafe_head_ok} "
          f"| turn-in-place tangential: {spin_ok}")

    # The <=90 deg optimisation: reversing direction should flip+reverse, never
    # swing a module more than 90 deg in one settle. Drive +x, then -x.
    s = SwerveSolver()
    _settle(s, 0.5, 0.0, 0.0)
    h_before = list(s.wheel_head)
    # one tick of reversed command: heading should barely move (flip handles it)
    s.solve(-0.5, 0.0, 0.0, 1.0 / 60.0)
    max_swing = max(abs(wrap_to_pi(a - b)) for a, b in zip(s.wheel_head, h_before))
    print(f"   reverse command max module swing in 1 tick: {math.degrees(max_swing):.2f} deg")
    return {
        "forward points modules straight": fwd_head_ok,
        "strafe points modules sideways": strafe_head_ok,
        "turn-in-place is tangential": spin_ok,
        "reverse uses flip (swing < 90 deg)": max_swing < math.pi / 2,
    }, {"reverse_swing_deg": math.degrees(max_swing)}


# --- C. desaturation ---------------------------------------------------------
def test_desaturation():
    print("C. DESATURATION")
    # A huge spin-in-place asks every module for more than MAX_WHEEL_SPEED.
    raw = [50.0, -40.0, 30.0, -20.0]
    scaled, scale = desaturate(raw, MAX_WHEEL_SPEED)
    peak_ok = max(abs(x) for x in scaled) <= MAX_WHEEL_SPEED + 1e-9
    # Ratios (direction) preserved: scaled = scale * raw.
    ratio_ok = all(abs(sc - scale * r) < 1e-9 for sc, r in zip(scaled, raw))
    # Below the cap nothing changes.
    small, sc2 = desaturate([1.0, -2.0, 0.5, 1.5], MAX_WHEEL_SPEED)
    nochange_ok = (sc2 == 1.0) and small == [1.0, -2.0, 0.5, 1.5]
    # End to end: a command that saturates a module comes out capped but with
    # the SAME module headings as the un-capped command (direction preserved).
    big = SwerveSolver()
    small_s = SwerveSolver()
    _settle(big, 1.5, 0.0, 4.0)        # large translate + spin -> saturates
    _settle(small_s, 0.15, 0.0, 0.4)   # same direction, 10x smaller -> no cap
    head_match = all(abs(wrap_to_pi(a - b)) < 1e-3
                     for a, b in zip(big.wheel_head, small_s.wheel_head))
    _, wheel = big.solve(1.5, 0.0, 4.0, 1.0 / 60.0)
    peak_spin = max(abs(v) for v in wheel.values())
    print(f"   raw peak {max(abs(x) for x in raw):.0f} -> scaled peak "
          f"{max(abs(x) for x in scaled):.1f} (cap {MAX_WHEEL_SPEED:.0f}); scale {scale:.3f}")
    print(f"   saturating drive: peak wheel {peak_spin:.2f} rad/s | "
          f"direction preserved vs small cmd: {head_match}")
    return {
        "desaturate caps at MAX_WHEEL_SPEED": peak_ok,
        "desaturate preserves ratios": ratio_ok,
        "desaturate no-op below cap": nochange_ok,
        "saturated solve stays within cap": peak_spin <= MAX_WHEEL_SPEED + 1e-6,
        "saturation preserves direction": head_match,
    }, {"desat_scale": scale, "saturated_peak_spin": peak_spin}


# --- D. solver round trip ----------------------------------------------------
def test_solver_roundtrip():
    print("D. SOLVER ROUND TRIP (settled steer/spin -> body velocity)")
    err = 0.0
    cases = [(0.4, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, 0.0, 0.8),
             (0.3, -0.2, 0.5), (-0.25, 0.15, -0.6)]
    for vx, vy, yaw in cases:
        s = SwerveSolver()
        steer, wheel = _settle(s, vx, vy, yaw)
        # Undo the per-joint direction fixups to recover applied heading + spin.
        heads, signed_speeds = [], []
        for k, jn in enumerate(WHEEL_JOINTS):
            applied = steer[STEER_JOINTS[k]] / STEER_DIR[STEER_JOINTS[k]]
            spin = wheel[jn] / WHEEL_DIR[jn]
            heads.append(applied)
            signed_speeds.append(spin * WHEEL_RADIUS)  # signed: flips are encoded
        rvx, rvy, ryaw = forward_kinematics(heads, signed_speeds)
        e = max(abs(rvx - vx), abs(rvy - vy), abs(ryaw - yaw))
        err = max(err, e)
    print(f"   worst body-velocity recovery from settled commands: {err:.2e}")
    return {"solver round-trip < 1e-6": err < 1e-6}, {"solver_roundtrip_err": err}


# --- E. odometry -------------------------------------------------------------
def test_odometry():
    print("E. ODOMETRY")
    dt = 1.0 / 200.0
    # Straight 1 m/s for 2 s -> x=2.
    o = SwerveOdometry()
    for _ in range(400):
        o.update(1.0, 0.0, 0.0, dt)
    straight_ok = abs(o.x - 2.0) < 1e-6 and abs(o.y) < 1e-9 and abs(o.theta) < 1e-9
    # Strafe 0.5 m/s for 2 s -> y=1.
    o = SwerveOdometry()
    for _ in range(400):
        o.update(0.0, 0.5, 0.0, dt)
    strafe_ok = abs(o.y - 1.0) < 1e-6 and abs(o.x) < 1e-9
    # Turn in place 1 rad/s for 1 s -> theta=1, position fixed.
    o = SwerveOdometry()
    for _ in range(200):
        o.update(0.0, 0.0, 1.0, dt)
    spin_ok = abs(o.theta - 1.0) < 1e-6 and abs(o.x) < 1e-9 and abs(o.y) < 1e-9
    # Arc vx=0.6, yaw=0.8 for t=1.5 s vs analytic circle (R=vx/yaw).
    v, w, T = 0.6, 0.8, 1.5
    o = SwerveOdometry()
    n = int(T / dt)
    for _ in range(n):
        o.update(v, 0.0, w, dt)
    th = w * (n * dt)
    ax = (v / w) * math.sin(th)
    ay = (v / w) * (1.0 - math.cos(th))
    arc_err = max(abs(o.x - ax), abs(o.y - ay), abs(wrap_to_pi(o.theta - th)))
    # Full circle returns to the origin.
    o = SwerveOdometry()
    N = 20000
    for _ in range(N):
        o.update(1.0, 0.0, 1.0, (2 * math.pi) / N)
    circle_err = math.hypot(o.x, o.y)
    print(f"   straight x=2: {straight_ok} | strafe y=1: {strafe_ok} | "
          f"turn theta=1: {spin_ok}")
    print(f"   arc vs analytic err {arc_err:.2e} m | full-circle close err {circle_err:.2e} m")
    return {
        "straight integrates exactly": straight_ok,
        "strafe integrates exactly": strafe_ok,
        "turn-in-place keeps position": spin_ok,
        "arc matches analytic (<1e-3 m)": arc_err < 1e-3,
        "full circle closes (<1e-3 m)": circle_err < 1e-3,
    }, {"arc_err_m": arc_err, "circle_err_m": circle_err}


def main(label="current"):
    print(f"\n=================  SWERVE TEST: {label}  =================")
    all_gates, metrics = {}, {}
    for fn in (test_roundtrip, test_geometry, test_desaturation,
               test_solver_roundtrip, test_odometry):
        g, m = fn()
        all_gates.update(g)
        metrics.update(m)
        print()
    print("----------------  GATES  ----------------")
    npass = 0
    for name, ok in all_gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        npass += int(ok)
    print(f"\n{npass}/{len(all_gates)} gates passed")
    return 0 if npass == len(all_gates) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "current"))

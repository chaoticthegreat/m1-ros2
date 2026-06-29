"""Teleop-style solver stress test (no ROS).

The reachability benchmark (`_solver_bench.py`) only tests *cold, fixed*
targets. Real Quest teleop instead streams a target that moves a little every
tick. This harness reproduces that closed loop and measures the failure modes
the operator reported:

  1. "arm jumps to a random position instead of smoothly tracking"
     -> measured as a discontinuity in the solved goal (q_best) or a spike in
        the per-tick fingertip motion while the target itself moves smoothly.
  2. "moving one arm ruins the tracking of the other arm"
     -> hold one arm's target fixed, sweep the other, measure the held arm's
        fingertip drift and goal jumps.
  3. "it can be slow sometimes"
     -> per-tick solve_step wall time (60 Hz budget = 16.7 ms).

The simulated robot follows the commanded joint positions exactly (same
assumption as `_solver_bench` / `_e2e_check`'s FakeRobot).
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ros2_ws/src/m1_control"))

from m1_control.kinematics import (  # noqa: E402
    ARM_JOINTS,
    LIFT_JOINT,
    ReachController,
    UrdfModel,
)

URDF = "assets/ranger_air_description/urdf/ranger_air_description.urdf"


def load():
    with open(URDF) as fh:
        return ReachController(UrdfModel.from_string(fh.read()))


def _q0():
    q = {j: 0.0 for j in ARM_JOINTS["left"] + ARM_JOINTS["right"]}
    q[LIFT_JOINT] = 0.0
    return q


def _goal_vec(reach, arm):
    """The solver's current solved goal (q_best) for ``arm``'s 7 joints."""
    c = reach._cache
    if c is None or arm not in c["arms"]:
        return None
    order = []
    for a in c["arms"]:
        order += ARM_JOINTS[a]
    order.append(LIFT_JOINT)
    idx = order.index(ARM_JOINTS[arm][0])
    return np.array(c["q_best"][idx:idx + 7])


def reach_center(reach, arm, seed_lift=0.4):
    """A comfortably-reachable Cartesian point: FK of a mild posture."""
    q = _q0()
    q[LIFT_JOINT] = seed_lift
    q[ARM_JOINTS[arm][1]] = 0.5
    q[ARM_JOINTS[arm][3]] = 0.8
    return reach.fingertip(arm, q)


def _center_cfg(arm, lift=0.35):
    """A mild mid-range arm posture (dict) used as a trajectory center."""
    q = {LIFT_JOINT: lift}
    base = ARM_JOINTS[arm]
    vals = [0.0, 0.5, 0.0, 0.9, 0.0, 0.4, 0.0]
    for j, v in zip(base, vals):
        q[j] = v
    return q


def fk_traj(reach, arm, n=400, lift=0.35, amp=0.25):
    """Target = FK of a smoothly varying joint trajectory (lift held fixed).

    Reachable by construction, so any tracking error reflects solver lag/branch
    behaviour rather than the arm running out of workspace. The lift is held
    constant so a single shared-lift solution can serve a second, still arm.
    """
    cfg = _center_cfg(arm, lift)
    base = ARM_JOINTS[arm]
    freqs = [0.7, 0.9, 1.1, 0.8, 1.3, 1.0, 0.6]

    def traj(i):
        if i >= n:
            return None
        t = i / 60.0
        q = dict(cfg)
        for k, j in enumerate(base):
            q[j] = cfg[j] + amp * math.sin(freqs[k] * t + 0.4 * k)
        return reach.fingertip(arm, q)
    return traj


def run_traj(reach, traj, q, ticks_times, settle_first=True):
    """Drive a moving target through the closed loop, recording diagnostics.

    ``traj`` maps arm -> function(i)->xyz (called each tick). Returns a dict of
    arrays: per-tick tracking error, fingertip step, and solved-goal step, per
    arm.
    """
    arms = list(traj.keys())
    diag = {a: {"err": [], "fstep": [], "gstep": [], "res": []} for a in arms}
    prev_tip = {a: reach.fingertip(a, q) for a in arms}
    prev_goal = {a: None for a in arms}

    i = 0
    while True:
        targets = {}
        done = True
        for a in ("left", "right"):
            if a in traj:
                xyz = traj[a](i)
                if xyz is None:
                    targets[a] = None
                else:
                    targets[a] = np.asarray(xyz, float)
                    done = False
            else:
                targets[a] = None
        if done:
            break

        t0 = time.perf_counter()
        result = reach.solve_step(q, targets)
        ticks_times.append(time.perf_counter() - t0)
        res = result.get("_dist", {})
        for jn, val in result.items():
            if jn == "_dist":
                continue
            q[jn] = val

        for a in arms:
            if targets.get(a) is None:
                continue
            tip = reach.fingertip(a, q)
            diag[a]["err"].append(float(np.linalg.norm(targets[a] - tip)))
            diag[a]["res"].append(float(res.get(a, float("nan"))))
            diag[a]["fstep"].append(float(np.linalg.norm(tip - prev_tip[a])))
            prev_tip[a] = tip
            g = _goal_vec(reach, a)
            if g is not None and prev_goal[a] is not None:
                diag[a]["gstep"].append(float(np.linalg.norm(g - prev_goal[a])))
            prev_goal[a] = g
        i += 1
        if i > 5000:
            break
    return {a: {k: np.array(v) for k, v in d.items()} for a, d in diag.items()}


def summarize(label, diag, warmup=20):
    print(f"  [{label}]")
    for a, d in diag.items():
        err = d["err"][warmup:] * 1e3
        res = d["res"][warmup:] * 1e3
        fstep = d["fstep"][warmup:] * 1e3
        gstep = d["gstep"][max(0, warmup - 1):] if d["gstep"].size else np.array([0.0])
        print(f"    {a:5s}: track err mm mean={err.mean():6.2f} p95={np.percentile(err,95):6.2f} "
              f"max={err.max():6.2f} | solved resid mm mean={np.nanmean(res):5.2f} "
              f"max={np.nanmax(res):6.2f} | fingertip step mm max={fstep.max():5.1f} "
              f"| goal jump rad max={gstep.max():.3f}")
    return diag


def scenario_single(reach, ticks):
    """One arm tracking a smooth, fully-reachable Cartesian path."""
    return run_traj(reach, {"left": fk_traj(reach, "left")}, _q0(), ticks)


def scenario_reach_out(reach, ticks):
    """Drive the target straight out past the reachable boundary and back.

    The transiently-unreachable stretch is where a global restart would snap
    the arm to a far IK branch; here it should saturate smoothly and recover.
    """
    c = reach_center(reach, "left")
    out = c / np.linalg.norm(c)

    def traj(i):
        if i >= 360:
            return None
        s = math.sin(i / 360.0 * math.pi)  # 0 -> 1 -> 0
        return c + out * (0.9 * s)         # well past the arm's reach at the peak
    return run_traj(reach, {"left": traj}, _q0(), ticks)


def scenario_dual_coupling(reach, ticks):
    """Both arms active; RIGHT held perfectly still, LEFT swept (shared lift).

    Tests "moving one arm ruins the other": both trajectories share one lift
    height, so a shared-lift solution that holds the right arm exactly exists.
    The right arm's tracking error and goal should barely move while the left
    sweeps.
    """
    lift = 0.35
    left = fk_traj(reach, "left", lift=lift)
    cr = reach.fingertip("right", _center_cfg("right", lift))

    def right(i):
        return cr if i < 400 else None
    return run_traj(reach, {"left": left, "right": right}, _q0(), ticks)


def scenario_dual_jump(reach, ticks):
    """RIGHT held still; LEFT makes a big sudden jump (a large clutch move).

    A jump past the tracking threshold forces a cold re-solve of the *coupled*
    two-arm system -- the exact case that used to fling the still arm around.
    The right arm should stay put through it.
    """
    lift = 0.35
    la = reach.fingertip("left", _center_cfg("left", lift))
    lb = la + np.array([0.0, -0.30, 0.25])   # ~0.4 m jump, far past _IK_TRACK_JUMP
    cr = reach.fingertip("right", _center_cfg("right", lift))

    def left(i):
        if i >= 200:
            return None
        return la if i < 60 else lb         # settle, then jump at tick 60

    def right(i):
        return cr if i < 200 else None
    return run_traj(reach, {"left": left, "right": right}, _q0(), ticks)


def main(label="current"):
    print(f"\n==== TELEOP STRESS: {label} ====")
    ticks = []

    reach = load()
    d = scenario_single(reach, ticks)
    summarize("single-arm smooth reachable path", d)

    reach = load()
    d = scenario_reach_out(reach, ticks)
    summarize("single-arm reach past boundary & back", d)

    reach = load()
    d = scenario_dual_coupling(reach, ticks)
    summarize("dual-arm: RIGHT held still while LEFT sweeps", d)

    reach = load()
    d = scenario_dual_jump(reach, ticks)
    summarize("dual-arm: RIGHT held still while LEFT jumps far (cold re-solve)", d)

    tt = np.array(ticks) * 1e3
    over = int((tt > 16.7).sum())
    print(f"  solve_step ms: mean={tt.mean():.2f} p95={np.percentile(tt,95):.2f} "
          f"max={tt.max():.2f}  over-budget {over}/{len(tt)} ({100*over/len(tt):.1f}%)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "current")

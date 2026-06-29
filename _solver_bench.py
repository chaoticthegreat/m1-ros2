"""Standalone solver benchmark (no ROS).

Simulates the closed loop the controller runs: each tick the "robot" perfectly
follows the commanded joint positions (same assumption as _e2e_check's
FakeRobot), so we can measure how close the ReachController gets to a range of
Cartesian targets and how fast.
"""
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


TICK_TIMES = []  # solve_step wall-times (s), for real-time budgeting


def run_closed_loop(reach, targets, max_ticks=250, q0=None):
    """Run the solver in closed loop; return (q, history of per-arm dist)."""
    arms = [a for a in ("left", "right") if targets.get(a) is not None]
    q = {j: 0.0 for j in ARM_JOINTS["left"] + ARM_JOINTS["right"]}
    q[LIFT_JOINT] = 0.0
    if q0:
        q.update(q0)
    hist = []
    stable = 0
    for _ in range(max_ticks):
        t0 = time.perf_counter()
        result = reach.solve_step(q, targets)
        TICK_TIMES.append(time.perf_counter() - t0)
        for jn, val in result.items():
            if jn == "_dist":
                continue
            q[jn] = val
        d = {a: float(np.linalg.norm(np.asarray(targets[a]) - reach.fingertip(a, q)))
             for a in arms}
        hist.append(d)
        # Stop once the command holds steady (converged or saturated at limits).
        if len(result) <= 1:  # only "_dist" -> command held (deadband)
            stable += 1
            if stable >= 3:
                break
        else:
            stable = 0
    return q, hist


def reachable_targets(reach, share_lift=False):
    """Generate targets by sampling random joint configs and taking FK.

    Each target is the FK of a real config, so it is reachable and an ideal
    solver should drive the error to ~0. When ``share_lift`` is set both arms'
    sample configs use the *same* lift height, so a single shared-lift solution
    that satisfies both arms is guaranteed to exist (a fair dual-arm test).
    """
    model = reach.model
    rng = np.random.default_rng(0)
    out = []
    for _ in range(40):
        tgt = {}
        lift = rng.uniform(0.0, 0.85)
        for arm in ("left", "right"):
            q = {LIFT_JOINT: lift if share_lift else rng.uniform(0.0, 0.85)}
            for j in ARM_JOINTS[arm]:
                jt = model.joints[j]
                q[j] = rng.uniform(jt.lower, jt.upper)
            tgt[arm] = reach.fingertip(arm, q)
        out.append(tgt)
    return out


def bench(label):
    reach = load()
    print(f"\n==== {label} ====")

    # --- single-arm reachable targets ---
    single_errs = []
    single_ticks = []
    t0 = time.perf_counter()
    for tgt in reachable_targets(reach):
        one = {"left": tgt["left"], "right": None}
        q, hist = run_closed_loop(reach, one)
        final = hist[-1]["left"]
        single_errs.append(final)
        # ticks to settle under 5 mm
        settle = next((i for i, h in enumerate(hist) if h["left"] < 0.005), len(hist))
        single_ticks.append(settle)
    dt = time.perf_counter() - t0
    single_errs = np.array(single_errs)
    print(f"SINGLE-ARM reachable ({len(single_errs)} targets):")
    print(f"  final err mm: mean={single_errs.mean()*1e3:6.2f} "
          f"median={np.median(single_errs)*1e3:6.2f} "
          f"max={single_errs.max()*1e3:6.2f} "
          f"p90={np.percentile(single_errs,90)*1e3:6.2f}")
    print(f"  <5mm: {(single_errs<0.005).mean()*100:5.1f}%   "
          f"<1mm: {(single_errs<0.001).mean()*100:5.1f}%")
    print(f"  mean ticks to 5mm: {np.mean(single_ticks):6.1f}")

    # --- dual-arm reachable targets (shared lift compromise) ---
    dual_errs = []
    for tgt in reachable_targets(reach, share_lift=True):
        q, hist = run_closed_loop(reach, tgt)
        dual_errs.append(max(hist[-1]["left"], hist[-1]["right"]))
    dual_errs = np.array(dual_errs)
    print(f"DUAL-ARM reachable ({len(dual_errs)} targets, worst-arm err):")
    print(f"  final err mm: mean={dual_errs.mean()*1e3:6.2f} "
          f"median={np.median(dual_errs)*1e3:6.2f} "
          f"max={dual_errs.max()*1e3:6.2f}")
    print(f"  both <5mm: {(dual_errs<0.005).mean()*100:5.1f}%")

    print(f"  ({dt:.2f}s for single-arm sweep)")
    if TICK_TIMES:
        tt = np.array(TICK_TIMES) * 1e3
        over = int((tt > 16.7).sum())
        print(f"solve_step time ms: mean={tt.mean():.2f} p95={np.percentile(tt,95):.2f} "
              f"max={tt.max():.2f}  (60Hz budget=16.7ms)")
        print(f"  ticks over budget: {over}/{len(tt)} "
              f"({100.0*over/len(tt):.1f}%)  [these are target-change re-solves]")
    return single_errs, dual_errs


if __name__ == "__main__":
    bench(sys.argv[1] if len(sys.argv) > 1 else "current")

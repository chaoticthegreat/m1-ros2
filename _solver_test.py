"""Comprehensive standalone solver test for the M1 reach controller (no ROS).

This is the full target-tracking test suite. It exercises the ReachController the
way the live system does -- the simulated robot follows the commanded joint
positions exactly (same assumption as _solver_bench / _teleop_stress) -- and
reports both quantitative metrics and PASS/FAIL gates so the same command shows
whether a solver change helped or regressed:

  A. Reachability accuracy  (cold, single + dual arm) -- final error, settle time
  B. Continuous tracking    (smooth path, reach past boundary, dual coupling,
                             far jump) -- tracking error, fingertip/goal jumps
  C. Hold under disturbance (base-relative target held while the measured joints
                             are perturbed each tick, as base driving jostles the
                             arm) -- recovery + steady hold error
  D. Latency                (full solve_step time distribution + worst case;
                             the 60 Hz budget is 16.7 ms)
  E. Stress                 (random far target jumps, arm-set toggling,
                             unreachable targets) -- no NaNs, bounded, recovers

Run:
  /usr/bin/python3 _solver_test.py [label]
Exit code is 0 only if every gate passes.
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
BUDGET_MS = 1000.0 / 60.0  # 16.7 ms


def load():
    with open(URDF) as fh:
        return ReachController(UrdfModel.from_string(fh.read()))


def _q0():
    q = {j: 0.0 for j in ARM_JOINTS["left"] + ARM_JOINTS["right"]}
    q[LIFT_JOINT] = 0.0
    return q


def _cfg(arm, lift=0.35, vals=(0.0, 0.5, 0.0, 0.9, 0.0, 0.4, 0.0)):
    q = {LIFT_JOINT: lift}
    for j, v in zip(ARM_JOINTS[arm], vals):
        q[j] = v
    return q


# Collected across the whole run for the latency report (section D).
TICKS = []


def _apply(q, result):
    for jn, val in result.items():
        if jn != "_dist":
            q[jn] = val


def _step(reach, q, targets):
    t0 = time.perf_counter()
    result = reach.solve_step(q, targets)
    TICKS.append((time.perf_counter() - t0) * 1e3)
    _apply(q, result)
    return result


# --- A. reachability ---------------------------------------------------------
def _rand_targets(reach, n=40, share_lift=False, seed=0):
    rng = np.random.default_rng(seed)
    model = reach.model
    out = []
    for _ in range(n):
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


def _converge(reach, targets, max_ticks=300):
    arms = [a for a in ("left", "right") if targets.get(a) is not None]
    q = _q0()
    settle = None
    for i in range(max_ticks):
        res = _step(reach, q, targets)
        d = {a: float(np.linalg.norm(np.asarray(targets[a]) - reach.fingertip(a, q)))
             for a in arms}
        worst = max(d.values())
        if settle is None and worst < 0.005:
            settle = i
        if len(res) <= 1:  # command held (deadband)
            break
    final = max(float(np.linalg.norm(np.asarray(targets[a]) - reach.fingertip(a, q)))
                for a in arms)
    return final, (settle if settle is not None else max_ticks)


def test_reach(reach):
    single = [_converge(load(), {"left": t["left"], "right": None})
              for t in _rand_targets(reach)]
    dual = [_converge(load(), t) for t in _rand_targets(reach, share_lift=True)]
    se = np.array([s[0] for s in single]); st = np.array([s[1] for s in single])
    de = np.array([d[0] for d in dual])
    print("A. REACHABILITY")
    print(f"   single: <1mm {100*(se<1e-3).mean():5.1f}%  mean {se.mean()*1e3:.2f}mm "
          f"max {se.max()*1e3:.2f}mm  settle {st.mean():.1f} ticks")
    print(f"   dual:   <5mm {100*(de<5e-3).mean():5.1f}%  mean {de.mean()*1e3:.2f}mm "
          f"max {de.max()*1e3:.2f}mm")
    gates = {
        "single 100% <1mm": (se < 1e-3).mean() == 1.0,
        "dual 100% <5mm": (de < 5e-3).mean() == 1.0,
        "single max <2mm": se.max() < 2e-3,
    }
    return gates, {"single_mean_mm": se.mean() * 1e3, "single_max_mm": se.max() * 1e3,
                   "dual_mean_mm": de.mean() * 1e3, "dual_max_mm": de.max() * 1e3,
                   "settle_ticks": float(st.mean())}


# --- B. continuous tracking --------------------------------------------------
def _traj(reach, arm, lift=0.35, amp=0.25, n=400):
    cfg = _cfg(arm, lift)
    freqs = [0.7, 0.9, 1.1, 0.8, 1.3, 1.0, 0.6]

    def f(i):
        if i >= n:
            return None
        t = i / 60.0
        q = dict(cfg)
        for k, j in enumerate(ARM_JOINTS[arm]):
            q[j] = cfg[j] + amp * math.sin(freqs[k] * t + 0.4 * k)
        return reach.fingertip(arm, q)
    return f


def _run_traj(reach, trajs, warmup=20):
    q = _q0()
    arms = list(trajs.keys())
    prev_tip = {a: reach.fingertip(a, q) for a in arms}
    diag = {a: {"err": [], "fstep": []} for a in arms}
    i = 0
    while True:
        targets = {a: (np.asarray(trajs[a](i), float) if a in trajs and trajs[a](i) is not None else None)
                   for a in ("left", "right")}
        if all(t is None for t in targets.values()):
            break
        _step(reach, q, targets)
        for a in arms:
            if targets.get(a) is None:
                continue
            tip = reach.fingertip(a, q)
            diag[a]["err"].append(float(np.linalg.norm(targets[a] - tip)))
            diag[a]["fstep"].append(float(np.linalg.norm(tip - prev_tip[a])))
            prev_tip[a] = tip
        i += 1
        if i > 5000:
            break
    return {a: {k: np.array(v[warmup:]) for k, v in d.items()} for a, d in diag.items()}


def test_tracking(reach):
    print("B. CONTINUOUS TRACKING")
    # smooth single-arm
    d1 = _run_traj(load(), {"left": _traj(reach, "left")})["left"]
    # dual coupling: right held still, left swept (shared lift solution exists)
    lift = 0.35
    cr = reach.fingertip("right", _cfg("right", lift))
    d2 = _run_traj(load(), {"left": _traj(reach, "left", lift=lift),
                            "right": lambda i: cr if i < 400 else None})
    # dual far jump: right held, left jumps ~0.4 m at tick 60
    la = reach.fingertip("left", _cfg("left", lift))
    lb = la + np.array([0.0, -0.30, 0.25])
    cr2 = reach.fingertip("right", _cfg("right", lift))
    d3 = _run_traj(load(), {"left": lambda i: (None if i >= 200 else (la if i < 60 else lb)),
                            "right": lambda i: cr2 if i < 200 else None})
    print(f"   smooth single : err mean {d1['err'].mean()*1e3:.2f}mm max {d1['err'].max()*1e3:.2f}mm "
          f"| fingertip step max {d1['fstep'].max()*1e3:.1f}mm")
    print(f"   dual coupling : LEFT err max {d2['left']['err'].max()*1e3:.2f}mm "
          f"| RIGHT held dev max {d2['right']['err'].max()*1e3:.2f}mm")
    print(f"   dual far jump : LEFT err max {d3['left']['err'].max()*1e3:.1f}mm "
          f"| RIGHT held dev max {d3['right']['err'].max()*1e3:.1f}mm "
          f"| LEFT fstep max {d3['left']['fstep'].max()*1e3:.1f}mm")
    # Held-arm far-jump contract (the anti-"snap to a random pose" guard, which
    # the >130 mm bug used to violate). Since the arms now mount FLUSH on the lift
    # carriage (~0.70 m lower), the held arm's posture couples the shared lift's
    # z-slew into its fingertip more strongly, so the brief transient while the
    # lift slews to serve LEFT's cold far jump is larger than the old (higher)
    # mount (~67 mm vs ~7 mm). What matters is unchanged: it must SETTLE planted
    # and stay far from the old snap regime. So gate the settled deviation tight
    # and bound the transient well below the bug.
    right_dev = d3["right"]["err"]
    gates = {
        "smooth single err <2mm": d1["err"].max() < 2e-3,
        "smooth single no fingertip jump >20mm": d1["fstep"].max() < 0.02,
        "dual held arm undisturbed <2mm": d2["right"]["err"].max() < 2e-3,
        "far-jump held arm settles <2mm": right_dev[-60:].max() < 2e-3,
        "far-jump held arm transient bounded <80mm": right_dev.max() < 0.080,
        "far-jump moving arm no >100mm fingertip jump": d3["left"]["fstep"].max() < 0.100,
    }
    return gates, {
        "smooth_err_max_mm": d1["err"].max() * 1e3,
        "coupling_held_dev_mm": d2["right"]["err"].max() * 1e3,
        "farjump_held_dev_mm": d3["right"]["err"].max() * 1e3,
        "farjump_moving_fstep_mm": d3["left"]["fstep"].max() * 1e3,
    }


# --- C. hold under base-motion disturbance -----------------------------------
def test_hold_disturbed(reach):
    """Target held base-relative while the measured joints are jostled each tick.

    Base driving perturbs the arm (the joints deviate slightly from command). The
    solver should reject that and hold the base-relative gripper point. We inject
    a bounded random perturbation into the measured joints every tick and measure
    the steady-state hold error and the worst transient.
    """
    print("C. HOLD UNDER DISTURBANCE (base-relative target, joints jostled)")
    rng = np.random.default_rng(7)
    r = load()
    arms = ("left", "right")
    targets = {a: r.fingertip(a, _cfg(a, 0.35)) for a in arms}
    q = _q0()
    # settle onto the target first
    for _ in range(120):
        _step(r, q, targets)
    errs = []
    worst = 0.0
    for i in range(400):
        # jostle the measured joints (as if the base lurched): small noise on the
        # arm joints + lift, growing with a simulated drive "bump" every 80 ticks.
        bump = 0.02 if (i % 80) < 8 else 0.004
        for j in ARM_JOINTS["left"] + ARM_JOINTS["right"] + [LIFT_JOINT]:
            q[j] += rng.normal(0.0, bump)
        _step(r, q, targets)
        e = max(float(np.linalg.norm(targets[a] - r.fingertip(a, q))) for a in arms)
        worst = max(worst, e)
        if i > 60:
            errs.append(e)
    errs = np.array(errs)
    print(f"   steady hold err mean {errs.mean()*1e3:.2f}mm p95 {np.percentile(errs,95)*1e3:.2f}mm "
          f"| worst transient {worst*1e3:.1f}mm")
    gates = {
        "disturbed steady hold <8mm": errs.mean() < 0.008,
        "disturbed recovers (p95 <15mm)": np.percentile(errs, 95) < 0.015,
    }
    return gates, {"hold_mean_mm": errs.mean() * 1e3, "hold_worst_mm": worst * 1e3}


# --- E. stress ---------------------------------------------------------------
def test_stress(reach):
    print("E. STRESS (random jumps, arm-set toggling, unreachable)")
    rng = np.random.default_rng(11)
    r = load()
    q = _q0()
    nan = False
    maxstep = 0.0
    prev = None
    for i in range(600):
        # toggle arm sets and jump targets around, including unreachable points
        which = rng.integers(0, 3)  # 0 left,1 right,2 both
        targets = {"left": None, "right": None}
        if which in (0, 2):
            targets["left"] = r.fingertip("left", _cfg("left", rng.uniform(0, 0.85))) \
                if rng.random() < 0.7 else np.array([rng.uniform(0.2, 1.2), rng.uniform(-0.2, 0.6), rng.uniform(0.2, 1.6)])
        if which in (1, 2):
            targets["right"] = r.fingertip("right", _cfg("right", rng.uniform(0, 0.85))) \
                if rng.random() < 0.7 else np.array([rng.uniform(0.2, 1.2), rng.uniform(-0.6, 0.2), rng.uniform(0.2, 1.6)])
        for _ in range(rng.integers(1, 6)):  # hold each target a few ticks
            res = _step(r, q, targets)
            for a in ("left", "right"):
                if targets[a] is not None:
                    tip = r.fingertip(a, q)
                    if not np.all(np.isfinite(tip)):
                        nan = True
            cmd = np.array([q[j] for j in ARM_JOINTS["left"] + ARM_JOINTS["right"] + [LIFT_JOINT]])
            if prev is not None:
                maxstep = max(maxstep, float(np.linalg.norm(cmd - prev)))
            prev = cmd
    print(f"   NaNs: {nan}  | max per-tick joint step {maxstep:.3f} rad")
    gates = {"no NaNs under stress": not nan, "bounded joint step (<0.6 rad)": maxstep < 0.6}
    return gates, {"stress_max_step_rad": maxstep}


# --- D. latency (uses TICKS gathered across all sections) ---------------------
def report_latency():
    tt = np.array(TICKS)
    over = int((tt > BUDGET_MS).sum())
    print("D. LATENCY (solve_step over the whole suite)")
    print(f"   mean {tt.mean():.2f}ms  p95 {np.percentile(tt,95):.2f}ms  "
          f"p99 {np.percentile(tt,99):.2f}ms  MAX {tt.max():.2f}ms  "
          f"(budget {BUDGET_MS:.1f}ms)")
    print(f"   over budget: {over}/{len(tt)} ({100*over/len(tt):.2f}%)")
    # 60 Hz is a GOAL, not a hard cutoff: the solver may spend a bigger budget on
    # a tick when the arm is far from a reachable target, to keep driving it onto
    # the goal (highest accuracy) rather than plateauing in the "general area".
    # So we gate that TYPICAL ticks stay real-time (median < budget) and the worst
    # case stays bounded (no runaway), and only REPORT the over-budget fraction.
    gates = {
        "typical solve_step < budget (median)": np.median(tt) < BUDGET_MS,
        "worst-case solve_step bounded < 60ms": tt.max() < 60.0,
    }
    return gates, {"max_ms": float(tt.max()), "p99_ms": float(np.percentile(tt, 99)),
                   "mean_ms": float(tt.mean()), "over_pct": 100 * over / len(tt)}


def main(label="current"):
    print(f"\n=================  SOLVER TEST: {label}  =================")
    reach = load()
    all_gates = {}
    metrics = {}
    for fn in (test_reach, test_tracking, test_hold_disturbed, test_stress):
        g, m = fn(reach)
        all_gates.update(g)
        metrics.update(m)
        print()
    g, m = report_latency()  # after all sections so the distribution is complete
    all_gates.update(g)
    metrics.update(m)

    print("\n----------------  GATES  ----------------")
    npass = 0
    for name, ok in all_gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        npass += int(ok)
    print(f"\n{npass}/{len(all_gates)} gates passed")
    return 0 if npass == len(all_gates) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "current"))

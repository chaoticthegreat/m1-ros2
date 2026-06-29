#!/usr/bin/env /usr/bin/python3
"""Reproduce the live reach failures OFFLINE against the solver to find the cause.

Reads /tmp/m1_ros_log/reach_failures.jsonl (written by _solver_failure_logger.py
from the live stack) and, for each unique failing target, re-runs the SAME
ReachController.solve_step the brain uses -- but with perfect feedback (command
fed back as the measured config), so it isolates the SOLVER from sim tracking
(gravity/gains). For each failure we try three things and classify:

  DUAL   : both arms commanded together (exactly as live) from a neutral seed.
  SINGLE : only the failing arm commanded (other target cleared) -> does the
           shared lift then serve it? (separates a dual shared-lift COMPROMISE
           from a true unreachable / solver issue).
  RESEED : seeded from the LIVE measured joints the logger captured -> if DUAL
           from neutral converges but this stays stuck, the live solver was in a
           bad branch / local min.

Classification per failing arm:
  REACHABLE-SINGLE / DUAL-COMPROMISE : single reaches (<tol) but dual doesn't ->
       the shared lift can't serve both; clearing one arm fixes it (known coupling).
  UNREACHABLE : single ALSO fails and the lift saturates (0 or 0.85) or a joint
       pins -> geometric, the target is outside the reachable set.
  SOLVER-STUCK : single fails but lift/joints are NOT saturated -> the NLP settled
       in a local min (candidate solver bug worth fixing).
  SIM-TRACKING : offline reaches (<tol) -> the live gap is sim gain/gravity, not IK.

Run:  PYTHONPATH=ros2_ws/src/m1_control /usr/bin/python3 _reproduce_failures.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "ros2_ws/src/m1_control")
from m1_control.kinematics import (UrdfModel, ReachController, ARM_JOINTS,
                                    LIFT_JOINT)

FAILLOG = os.environ.get("ROS_LOG_DIR", "/tmp/m1_ros_log") + "/reach_failures.jsonl"
TOL_MM = 5.0       # solver "converged" tolerance
TICKS = 250        # solve_step ticks to let the amortized cold multi-start finish


def find_urdf():
    for c in ("ros2_ws/install/ranger_air_description/share/ranger_air_description/"
              "urdf/ranger_air_description.urdf",
              "assets/ranger_air_description/urdf/ranger_air_description.urdf"):
        if os.path.isfile(c):
            return c
    raise SystemExit("URDF not found")


def neutral(reach):
    q = {j: 0.0 for a in ("left", "right") for j in ARM_JOINTS[a]}
    q[LIFT_JOINT] = 0.0
    return q


def converge(reach, targets, q0, ticks=TICKS):
    q = dict(q0)
    last = None
    for _ in range(ticks):
        res = reach.solve_step(q, targets)
        for jn, v in res.items():
            if not jn.startswith("_"):
                q[jn] = v
        last = res.get("_dist")
    dist = {a: float(np.linalg.norm(np.asarray(targets[a]) - reach.fingertip(a, q)))
            for a in targets}
    return q, dist, last


def lift_state(reach, q):
    j = reach.model.joints[LIFT_JOINT]
    lift = q.get(LIFT_JOINT, 0.0)
    frac = (lift - j.lower) / (j.upper - j.lower)
    return lift, frac


def saturated(reach, q, arm):
    out = []
    for jj in ARM_JOINTS[arm] + [LIFT_JOINT]:
        jt = reach.model.joints[jj]
        f = (q.get(jj, 0.0) - jt.lower) / (jt.upper - jt.lower)
        if f <= 0.03 or f >= 0.97:
            out.append(f"{jj}={f:.2f}")
    return out


def classify(arm, dual_d, single_d, reseed_d, reach, q_dual, q_single):
    lift_d, fd = lift_state(reach, q_dual)
    lift_s, fs = lift_state(reach, q_single)
    sat_s = saturated(reach, q_single, arm)
    dual_ok = dual_d[arm] * 1e3 < TOL_MM
    single_ok = single_d[arm] * 1e3 < TOL_MM
    if dual_ok:
        return "SIM-TRACKING (offline dual reaches; live gap is sim gains/gravity)"
    if single_ok:
        return (f"DUAL-COMPROMISE (single reaches {single_d[arm]*1e3:.1f}mm @lift={lift_s:.2f}; "
                f"dual stuck {dual_d[arm]*1e3:.1f}mm @lift={lift_d:.2f} -- shared lift can't serve both)")
    # single also fails:
    if sat_s:
        return f"UNREACHABLE (single fails {single_d[arm]*1e3:.1f}mm, saturated {sat_s})"
    return (f"SOLVER-STUCK (single fails {single_d[arm]*1e3:.1f}mm, lift={lift_s:.2f} frac={fs:.2f}, "
            f"no joint saturated -> local min)")


def main():
    if not os.path.isfile(FAILLOG):
        raise SystemExit(f"no failure log at {FAILLOG} yet")
    rows = [json.loads(l) for l in open(FAILLOG) if l.strip()]
    # dedupe by (arm, target rounded 2cm)
    uniq = {}
    for r in rows:
        k = (r["arm"], round(r["target"][0], 2), round(r["target"][1], 2),
             round(r["target"][2], 2))
        uniq.setdefault(k, r)
    print(f"loaded {len(rows)} episodes, {len(uniq)} unique failing positions\n")

    reach = ReachController(UrdfModel.from_string(open(find_urdf()).read()))

    for (arm, *_), r in sorted(uniq.items()):
        tgt = r["target"]
        other = r.get("other_target")
        other_arm = "right" if arm == "left" else "left"
        # DUAL (as live): both targets if the other was active, else just this one
        dual_targets = {arm: np.array(tgt, float)}
        if r.get("dual_active") and other:
            dual_targets[other_arm] = np.array(other, float)
        qd, dd, _ = converge(reach, dual_targets, neutral(reach))
        # SINGLE (failing arm only)
        qs, sd, _ = converge(reach, {arm: np.array(tgt, float)}, neutral(reach))
        # RESEED from the live measured joints
        live_q = {j: d["q"] for j, d in r["joints"].items()}
        for j in (ARM_JOINTS["left"] + ARM_JOINTS["right"] + [LIFT_JOINT]):
            live_q.setdefault(j, 0.0)
        _, rd, _ = converge(reach, dual_targets, live_q)

        cls = classify(arm, dd, sd, rd, reach, qd, qs)
        print(f"=== {arm} target={tgt}  (live err {r['err_mm']}mm, dual={r['dual_active']}) ===")
        print(f"    offline DUAL   per-arm: " +
              ", ".join(f"{a}={dd[a]*1e3:.1f}mm" for a in dd) +
              f"  lift={qd.get(LIFT_JOINT):.3f}")
        print(f"    offline SINGLE {arm}: {sd[arm]*1e3:.1f}mm  lift={qs.get(LIFT_JOINT):.3f}")
        print(f"    offline RESEED(live q): {arm}={rd[arm]*1e3:.1f}mm")
        print(f"    --> {cls}\n")


if __name__ == "__main__":
    main()

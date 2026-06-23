"""Hard tracking / convergence stress for the M1 position-only reach solver (no ROS).

The solver is position-only: each tick it drives the gripper fingertip toward a
3D target *point* (orientation is ignored). Operators still sweep the gripper
across wide arcs continuously, so the solved configuration must cross internal
singularities and occasionally change IK branch -- all while each per-tick target
move stays small (so the solver stays in its cheap *in-branch tracking* regime
and never triggers a global re-solve). This harness reproduces that.

Every joint-FK target stream is the forward kinematics of a continuously varying
joint trajectory, so the point is reachable AT EVERY TICK by construction (a
perfect solver tracks it with ~0 error); any large *sustained* error is the
solver getting stuck. Scenarios:

  A. SMOOTH WIDE PATH   -- a large analytic Cartesian path (circle) swept smoothly
     across the workspace; the bread-and-butter teleop case.
  B. WIDE JOINT SWEEP   -- target = FK of a large joint trajectory, so the path
     drives the arm through internal singularities / branch changes.
  C. FULL LARGE SWEEP   -- an even larger multi-joint FK sweep.
  D. BOUNDARY + RECOVER -- target driven out past reach and brought back; the
     gripper must re-acquire (not stay stuck at the saturated config).
  E. COLD HARD POINTS   -- cold solves to FK of FULL-limit random configs
     (extreme poses), from a non-neutral start.
  F. DUAL TRACKING      -- both arms swept on a shared lift at once.
  G. COLD-THEN-NUDGE    -- a big cold jump immediately followed by small nudges
     (the cold job is still solving while the target is already moving).

Run:  /usr/bin/python3 _solver_test_tracking.py
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
BUDGET_MS = 1000.0 / 60.0
TICKS = []


def load():
    with open(URDF) as fh:
        return ReachController(UrdfModel.from_string(fh.read()))


def _q0(lift=0.4):
    q = {j: 0.0 for j in ARM_JOINTS["left"] + ARM_JOINTS["right"]}
    q[LIFT_JOINT] = lift
    return q


def _step(reach, q, targets):
    t0 = time.perf_counter()
    res = reach.solve_step(q, targets)
    TICKS.append((time.perf_counter() - t0) * 1e3)
    for jn, val in res.items():
        if not jn.startswith("_"):
            q[jn] = val
    return res


def _clip_cfg(reach, arm, vals, lift):
    """A joint dict for ``arm`` clipped to limits (lift shared)."""
    q = {LIFT_JOINT: float(np.clip(lift, 0.0, 0.85))}
    for j, v in zip(ARM_JOINTS[arm], vals):
        jt = reach.model.joints[j]
        q[j] = float(np.clip(v, jt.lower, jt.upper))
    return q


def _joint_traj(reach, arm, center, amps, freqs, lift, k):
    """Target point = fingertip FK of a smooth large-amplitude joint trajectory."""
    t = k / 60.0
    vals = [center[i] + amps[i] * math.sin(freqs[i] * t + 0.5 * i)
            for i in range(7)]
    return np.asarray(reach.fingertip(arm, _clip_cfg(reach, arm, vals, lift)))


def _track(reach, arm, pointfn, n, warmup=40, start_q=None):
    """Drive a single-arm point stream through the closed loop.

    Returns per-tick (pos_err, fingertip_step) arrays plus the worst per-tick
    target jump seen, so the caller can confirm we stayed in the tracking regime
    (small jumps) yet check whether the gripper kept up.
    """
    # Each scenario is an independent teleop session: clear any carried cache.
    reach._cache = None
    q = _q0() if start_q is None else dict(start_q)
    pe_list, fstep = [], []
    prev_tip = reach.fingertip(arm, q)
    prev_p = None
    max_jump = 0.0
    other = "right" if arm == "left" else "left"
    for k in range(n):
        p = pointfn(k)
        if prev_p is not None:
            max_jump = max(max_jump, float(np.linalg.norm(p - prev_p)))
        prev_p = p
        _step(reach, q, {arm: p, other: None})
        tip = reach.fingertip(arm, q)
        fstep.append(float(np.linalg.norm(tip - prev_tip)))
        prev_tip = tip
        pe_list.append(float(np.linalg.norm(p - tip)))
    sl = slice(warmup, None)
    return np.array(pe_list)[sl], np.array(fstep)[sl], max_jump


def _report(label, pe, fstep, max_jump):
    print(f"   {label}\n"
          f"     pos err mm: mean {pe.mean()*1e3:6.2f}  p95 {np.percentile(pe,95)*1e3:6.2f}  "
          f"max {pe.max()*1e3:7.2f}\n"
          f"     fingertip step max {fstep.max()*1e3:.1f}mm  | per-tick target jump max "
          f"{max_jump*1e3:.1f}mm (tracking regime if <60)")


# --- A. smooth wide sweep (well-conditioned, tracks sub-mm) -------------------
def test_smooth_path(reach):
    print("A. SMOOTH WIDE SWEEP (large reachable shoulder/elbow sweep)")
    # Big shoulder/elbow motion (a wide reachable position arc) with only tiny
    # wrist motion, so the path stays well-conditioned: the bread-and-butter
    # teleop case, which a position solver should track sub-mm throughout.
    center = [0.0, -0.7, 0.0, 1.1, 0.0, 0.2, 0.0]
    amps = [0.6, 0.5, 0.2, 0.6, 0.1, 0.15, 0.1]
    freqs = [0.5, 0.35, 0.6, 0.4, 0.7, 0.5, 0.6]
    pe, fstep, mj = _track(
        reach, "left",
        lambda k: _joint_traj(reach, "left", center, amps, freqs, 0.5, k), n=900)
    _report("single-arm smooth wide sweep", pe, fstep, mj)
    return {
        "smooth-sweep tracks <1mm": pe.max() < 1e-3,
        "smooth-sweep p95 <0.5mm": np.percentile(pe, 95) < 0.5e-3,
        "smooth-sweep no fingertip jump >20mm": fstep.max() < 0.020,
    }


# --- B. wide joint-FK sweep (mild challenge) ---------------------------------
def test_wide_sweep(reach):
    print("B. WIDE JOINT SWEEP (FK of a large multi-joint trajectory)")
    center = [-0.3, -0.8, 0.2, 1.2, 0.1, 0.1, 0.0]
    amps = [0.9, 0.6, 0.5, 0.8, 0.3, 0.3, 0.3]
    freqs = [0.45, 0.3, 0.55, 0.4, 0.85, 0.65, 1.0]
    pe, fstep, mj = _track(
        reach, "left",
        lambda k: _joint_traj(reach, "left", center, amps, freqs, 0.5, k), n=1200)
    _report("single-arm wide joint sweep", pe, fstep, mj)
    return {
        "wide-sweep tracks <8mm": pe.max() < 8e-3,
        "wide-sweep p95 <1mm": np.percentile(pe, 95) < 1e-3,
        "wide-sweep no fingertip jump >25mm": fstep.max() < 0.025,
    }


# --- C. full large multi-joint sweep (rides through internal singularities) ---
def test_full_sweep(reach):
    print("C. FULL LARGE SWEEP (aggressive sweep that crosses internal singularities)")
    # An aggressive full-amplitude multi-joint sweep (big wrist motion included)
    # drives the redundant arm through internal singularities. Position-only, the
    # orientation no longer pins the redundancy, so the solver may LAG for a few
    # ticks at a singular crossing (the bounded per-tick command can't supply the
    # joint-rate spike a singularity demands) and then recover -- it rides through,
    # it does NOT get stuck. So we gate the *typical* quality tightly (p95) and the
    # mean low (proving recovery), and only require the worst transient to stay
    # BOUNDED (not a teleport / not a sustained stall).
    center = [-0.3, -0.8, 0.2, 1.2, 0.1, 0.1, 0.0]
    amps = [1.1, 0.7, 1.0, 0.9, 1.2, 0.6, 1.3]
    freqs = [0.45, 0.3, 0.55, 0.4, 0.85, 0.65, 1.0]
    pe, fstep, mj = _track(
        reach, "left",
        lambda k: _joint_traj(reach, "left", center, amps, freqs, 0.5, k), n=1200)
    _report("single-arm full large sweep", pe, fstep, mj)
    return {
        "full-sweep typical tracking p95 <1mm": np.percentile(pe, 95) < 1e-3,
        "full-sweep recovers (mean <1mm)": pe.mean() < 1e-3,
        "full-sweep transient <3mm (was ~18mm before the accuracy work)": pe.max() < 3e-3,
        "full-sweep no fingertip teleport >40mm": fstep.max() < 0.040,
    }


# --- D. boundary excursion + recovery ----------------------------------------
def test_boundary_recover(reach):
    print("D. BOUNDARY + RECOVER (drive past reach, must re-acquire on return)")
    # A reachable center pushed straight out beyond the workspace and brought
    # back, then HELD home. While out, the arm saturates (fully extended at the
    # boundary singularity); on return it must re-acquire instead of staying stuck
    # at that saturated config (the original "gets stuck" failure left it ~150 mm
    # off forever). A transient slew onto the recovered branch is fine; what
    # matters is that it CONVERGES once the target is home.
    # NB: the excursion amplitude is 1.5 m (was 0.85 m). The center c sits ~1.5 m
    # from base_link, and the Drake solver tracks so well that 0.85 m out was still
    # WITHIN reach -- so it never saturated and the "recovered from a stuck state"
    # guard (return peak > 30 mm) couldn't fire. 1.5 m drives the target genuinely
    # past the workspace, so the arm truly saturates on the way out and must
    # re-acquire the reachable home point on return -- the behaviour this scenario
    # exists to verify.
    reach._cache = None
    q = _q0(0.45)
    q[ARM_JOINTS["left"][1]] = -0.6
    q[ARM_JOINTS["left"][3]] = 1.0
    c = np.asarray(reach.fingertip("left", q))
    outdir = c / np.linalg.norm(c)

    qd = _q0()
    excursion, settle = [], []
    for k in range(600):                          # out past reach (s: 0->1->0)
        p = c + outdir * (1.5 * math.sin(k / 600.0 * math.pi))
        _step(reach, qd, {"left": p, "right": None})
        excursion.append(float(np.linalg.norm(p - reach.fingertip("left", qd))))
    for _ in range(150):                          # hold home, let the flip settle
        _step(reach, qd, {"left": c, "right": None})
        settle.append(float(np.linalg.norm(c - reach.fingertip("left", qd))))
    settle = np.array(settle)
    final = settle[-40:]
    peak_return = max(excursion[300:])
    print(f"     peak err reaching out {max(excursion)*1e3:.0f}mm (expected, out of "
          f"reach) | return peak {peak_return*1e3:.0f}mm | settled home "
          f"mean {final.mean()*1e3:.2f}mm max {final.max()*1e3:.2f}mm")
    return {
        "boundary re-acquires (settles home) <2mm": final.max() < 2e-3,
        "boundary recovered from a stuck state (return peak >30mm)": peak_return > 0.03,
    }


# --- E. cold hard points from awkward starts ---------------------------------
def test_cold_hard(reach, n=60):
    print("E. COLD HARD POINTS (FK of FULL-limit configs, non-neutral start)")
    rng = np.random.default_rng(2024)
    pes, conv = [], 0
    for _ in range(n):
        lift = rng.uniform(0.05, 0.8)
        vals = [rng.uniform(reach.model.joints[j].lower, reach.model.joints[j].upper)
                for j in ARM_JOINTS["left"]]
        pos = np.asarray(reach.fingertip("left", _clip_cfg(reach, "left", vals, lift)))
        # Awkward, non-neutral start pose (NOT _q0): another random config.
        svals = [rng.uniform(reach.model.joints[j].lower, reach.model.joints[j].upper)
                 for j in ARM_JOINTS["left"]]
        q = _clip_cfg(reach, "left", svals, rng.uniform(0.05, 0.8))
        q.update({j: 0.0 for j in ARM_JOINTS["right"]})
        reach._cache = None   # independent cold solve per sample
        tgt = {"left": pos, "right": None}
        for _ in range(400):
            if len(_step(reach, q, tgt)) <= 1:
                break
        pe = float(np.linalg.norm(pos - reach.fingertip("left", q)))
        pes.append(pe)
        if pe < 2e-3:
            conv += 1
    pes = np.array(pes)
    print(f"     converged {conv}/{n} ({100*conv/n:.0f}%) | pos mean {pes.mean()*1e3:.2f}mm "
          f"max {pes.max()*1e3:.2f}mm")
    return {
        "cold-hard 100% converge <2mm": conv == n,
        "cold-hard max pos <1mm": pes.max() < 1e-3,
    }


# --- F. dual-arm tracking (both arms swept on the shared lift) ----------------
def test_dual_track(reach):
    print("F. DUAL TRACKING (both arms swept on a shared lift at once)")
    reach._cache = None
    lift = 0.5
    cenL = [0.0, -0.6, 0.0, 1.0, 0.0, 0.1, 0.0]
    cenR = [0.0, 0.6, 0.0, 1.0, 0.0, -0.1, 0.0]
    ampL = [0.5, 0.4, 1.0, 0.4, 1.2, 0.5, 1.3]
    ampR = [0.4, 0.4, 1.1, 0.5, 1.1, 0.5, 1.2]
    frqL = [0.5, 0.4, 0.7, 0.45, 0.9, 0.6, 1.0]
    frqR = [0.45, 0.35, 0.65, 0.5, 0.85, 0.55, 1.1]
    q = _q0()
    pes = []
    for k in range(60, 60 + 800):
        t = k / 60.0
        tgt = {}
        for arm, cen, amp, frq in (("left", cenL, ampL, frqL),
                                   ("right", cenR, ampR, frqR)):
            vals = [cen[i] + amp[i] * math.sin(frq[i] * t + 0.5 * i) for i in range(7)]
            tgt[arm] = np.asarray(reach.fingertip(arm, _clip_cfg(reach, arm, vals, lift)))
        _step(reach, q, tgt)
        if k > 100:
            for arm in ("left", "right"):
                pes.append(float(np.linalg.norm(tgt[arm] - reach.fingertip(arm, q))))
    pes = np.array(pes)
    print(f"     both arms: pos err p95 {np.percentile(pes,95)*1e3:.2f}mm max "
          f"{pes.max()*1e3:.2f}mm (one shared lift serves both -> a compromise)")
    # The two arms share ONE prismatic lift, so when both sweep widely the single
    # lift height that best serves both leaves each a little short at the extremes
    # -- a genuine compromise (documented limitation), NOT a stuck solve.
    return {
        "dual-track pos p95 <2mm": np.percentile(pes, 95) < 2e-3,
        "dual-track pos max <5mm": pes.max() < 5e-3,
    }


# --- G. cold jump then immediately nudge (job pending during tracking) --------
def test_cold_then_nudge(reach):
    print("G. COLD-THEN-NUDGE (nudge a target while its cold job is still solving)")
    # A big cold jump launches an amortized multi-seed search that takes a handful
    # of ticks; the operator then immediately starts nudging the target (< track
    # jump/tick) before it converges. The job is continued (not dropped) so the
    # re-acquire machinery can finish; the command leads by a bounded step, so any
    # branch refinement is a bounded slew, never a teleport, and once the job is
    # done steady tracking is branch-flip-free.
    reach._cache = None
    qg = _q0(0.55)
    qg[ARM_JOINTS["left"][0]] = -1.2
    qg[ARM_JOINTS["left"][1]] = -0.9
    qg[ARM_JOINTS["left"][3]] = 1.8
    c = np.asarray(reach.fingertip("left", qg))          # reachable hard target
    q = _q0()
    prev = reach.fingertip("left", q)
    slew_max = 0.0
    steady_step, errs = [], []
    SETTLE = 60
    for k in range(300):
        p = c + (np.zeros(3) if k == 0
                 else np.array([0.012 * math.sin(0.05 * k),
                                0.012 * math.cos(0.045 * k), 0.0]))
        _step(reach, q, {"left": p, "right": None})
        tip = reach.fingertip("left", q)
        step = float(np.linalg.norm(tip - prev))
        prev = tip
        slew_max = max(slew_max, step)
        if k >= SETTLE:
            steady_step.append(step)
            errs.append(float(np.linalg.norm(p - tip)))
    steady_step, errs = np.array(steady_step), np.array(errs)
    print(f"     transient slew max {slew_max*1e3:.0f}mm/tick (bounded by IK_MAX_DQ, not "
          f"teleport) | settled-track fingertip step max {steady_step.max()*1e3:.2f}mm "
          f"| settled track err max {errs.max()*1e3:.2f}mm")
    return {
        "cold-then-nudge transient slew bounded (<160mm/tick)": slew_max < 0.160,
        "cold-then-nudge settled tracking branch-flip-free (<3mm step)": steady_step.max() < 3e-3,
        "cold-then-nudge settles sub-mm": errs.max() < 1e-3,
    }


def main():
    print("\n=======  HARD POSITION TRACKING / CONVERGENCE STRESS  =======")
    reach = load()
    gates = {}
    for fn in (test_smooth_path, test_wide_sweep, test_full_sweep,
               test_boundary_recover, test_cold_hard,
               test_dual_track, test_cold_then_nudge):
        gates.update(fn(reach))
        print()
    tt = np.array(TICKS)
    over = int((tt > BUDGET_MS).sum())
    print(f"H. LATENCY: mean {tt.mean():.2f}ms p99 {np.percentile(tt,99):.2f}ms "
          f"max {tt.max():.2f}ms over-budget {over}/{len(tt)} ({100*over/len(tt):.2f}%)")
    # Smooth tracking should stay real-time at the p95 (the goal); a far-target
    # recovery may briefly exceed the 60 Hz budget (accuracy first), bounded.
    gates["tracking p95 solve_step < budget"] = np.percentile(tt, 95) < BUDGET_MS
    gates["worst-case solve_step bounded < 60ms"] = tt.max() < 60.0

    print("\n----------------  GATES  ----------------")
    npass = 0
    for name, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        npass += int(ok)
    print(f"\n{npass}/{len(gates)} gates passed")
    return 0 if npass == len(gates) else 1


if __name__ == "__main__":
    sys.exit(main())

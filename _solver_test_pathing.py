"""Point-to-point trajectory tests for the M1 reach solver + planner (no ROS).

The existing solver suites mostly exercise *cold* solves (jump straight to a
target) and continuous tracking of a streamed point. This suite adds what the
operator actually cares about for a deliberate motion: **going from one point to
another and landing on the second with good accuracy**, the *planned* version of
that (a collision-free Cartesian path A->B), and the "keep trying until it's as
close as the joints allow" behaviour. Sections:

  A. PLAN + TRACK  -- plan a collision-free Cartesian path A->B, then drive the
     controller (`solve_step`) along the planned fingertip points and check it
     lands on B accurately, follows the path, and never jumps.
  B. WARM SOLVE    -- settle the arm AT A, then command B and converge: a warm
     in-motion solve from a real posture (not a cold jump from neutral).
  C. PLANNER FREE  -- across many random reachable goals the planner returns a
     path that reaches the goal AND is self-collision-free (or honestly flags
     the rare task-coupled tight pose it cannot open).
  D. AVOIDANCE     -- the null-space + path-detour avoidance never trades the
     goal away and never makes clearance worse than ignoring collisions.
  E. PERSISTENCE   -- hard reachable targets are driven sub-mm (the solver keeps
     refining instead of plateauing a few mm short), while a genuinely
     unreachable target settles at the closest config without blowing the 60 Hz
     budget (the refinement gives up once it cannot get closer).

Run:  /usr/bin/python3 _solver_test_pathing.py
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
from m1_control.collision import CollisionModel  # noqa: E402
from m1_control.trajectory import TrajectoryPlanner  # noqa: E402

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
    q = {LIFT_JOINT: float(np.clip(lift, 0.0, 0.85))}
    for j, v in zip(ARM_JOINTS[arm], vals):
        jt = reach.model.joints[j]
        q[j] = float(np.clip(v, jt.lower, jt.upper))
    return q


def _reachable(reach, rng, arm, lift, scale=1.0):
    """A reachable target point for ``arm`` = FK of a random (clipped) config."""
    vals = [rng.uniform(scale * reach.model.joints[j].lower,
                        scale * reach.model.joints[j].upper)
            for j in ARM_JOINTS[arm]]
    q = _clip_cfg(reach, arm, vals, lift)
    q.update({j: 0.0 for j in ARM_JOINTS["right" if arm == "left" else "left"]})
    return np.asarray(reach.fingertip(arm, q)), q


def _smooth_goal(reach, rng, arm, q_start, lift, delta=0.5):
    """A goal point smoothly reachable from ``q_start`` (same IK branch).

    A straight Cartesian path can only track a goal in the SAME branch as the
    start -- a goal needing an elbow/branch flip mid-path isn't smoothly path-able
    (that's what the warm-solve / cold re-solve path is for). So we build B as the
    FK of ``q_start``'s arm config plus a bounded random joint perturbation, which
    is reachable from A without a branch change.
    """
    cur = [q_start[j] for j in ARM_JOINTS[arm]]
    vals = [c + rng.uniform(-delta, delta) for c in cur]
    q = _clip_cfg(reach, arm, vals, lift)
    return np.asarray(reach.fingertip(arm, q))


def _settle(reach, q, targets, ticks=400):
    """Drive ``solve_step`` until it reports held or ``ticks`` elapse."""
    for _ in range(ticks):
        if len(_step(reach, q, targets)) <= 1:
            break
    return {a: float(np.linalg.norm(np.asarray(targets[a]) - reach.fingertip(a, q)))
            for a in targets if targets[a] is not None}


# --- A. plan a path A->B, then TRACK it with the controller ------------------
def test_plan_and_track(reach, planner):
    print("A. PLAN + TRACK (plan collision-free path A->B, controller follows it)")
    rng = np.random.default_rng(101)
    end_errs, path_dev, max_step, free, reached = [], [], 0.0, 0, 0
    N = 24
    for _ in range(N):
        lift = rng.uniform(0.25, 0.6)
        # A is a settled reachable posture; B is smoothly reachable from A (same
        # branch) so the straight Cartesian path is trackable end-to-end.
        startA, qA = _reachable(reach, rng, "left", lift, scale=0.6)
        goalB = _smooth_goal(reach, rng, "left", qA, lift, delta=0.55)
        if np.linalg.norm(goalB - startA) < 0.12:
            continue                            # too small a move; skip
        traj = planner.plan(qA, {"left": goalB, "right": None})
        reached += int(traj.reached)
        free += int(traj.collision_free)
        # Track the planned fingertip path with the real controller and measure
        # how well it follows + where it lands.
        pts = traj.points_for("left")
        q = dict(qA)
        reach._cache = None
        prev = np.asarray(reach.fingertip("left", q))
        for p in pts:
            for _ in range(3):                  # a few ticks to track each sample
                _step(reach, q, {"left": p, "right": None})
            tip = np.asarray(reach.fingertip("left", q))
            max_step = max(max_step, float(np.linalg.norm(tip - prev)))
            prev = tip
            path_dev.append(float(np.linalg.norm(tip - p)))
        # End error = where the controller ACTUALLY ended after following the
        # path (NO trailing from-scratch settle -- that would converge onto the
        # reachable B from any state and mask a tracker that drifted off the
        # path; this number must depend on the tracking having stayed on B).
        end_errs.append(float(np.linalg.norm(goalB - reach.fingertip("left", q))))
    n = len(end_errs)
    end_errs = np.array(end_errs)
    pd = np.array(path_dev)
    print(f"     {n} moves: plans reached {reached}/{n}, collision-free {free}/{n} | "
          f"track end-err mean {end_errs.mean()*1e3:.2f} max {end_errs.max()*1e3:.2f}mm | "
          f"path follow dev p95 {np.percentile(pd,95)*1e3:.2f}mm | "
          f"max fingertip step {max_step*1e3:.1f}mm")
    return {
        "plan+track smooth plans reach goal (>=90%)": reached >= 0.9 * n,
        "plan+track plans collision-free (>=90%)": free >= 0.9 * n,
        "plan+track lands on B <2mm": end_errs.max() < 2e-3,
        "plan+track follows path (p95 dev <10mm)": np.percentile(pd, 95) < 0.010,
        "plan+track no fingertip jump >40mm": max_step < 0.040,
    }


# --- B. warm solve: settle at A, then move IN-MOTION to B --------------------
def test_warm_solve(reach):
    print("B. WARM SOLVE (settle at A, then move in-motion to B, land accurately)")
    rng = np.random.default_rng(202)
    errs, max_step = [], 0.0
    N = 30
    for _ in range(N):
        lift = rng.uniform(0.2, 0.6)
        A, qA = _reachable(reach, rng, "left", lift, scale=0.6)
        # B reachable from A's branch, and a real move away from A.
        B = _smooth_goal(reach, rng, "left", qA, lift, delta=0.6)
        if np.linalg.norm(B - A) < 0.12:
            B = A + np.array([0.10, 0.10, -0.10])
        q = dict(qA)
        reach._cache = None
        _settle(reach, q, {"left": A, "right": None}, ticks=200)   # settle AT A
        startA = np.asarray(reach.fingertip("left", q))
        # Stream the target A->B in small Cartesian steps: each per-tick goal move
        # is well under the track-jump gate, so the solver stays in the WARM
        # in-branch track (no cold teleport) -- the genuine "move between two
        # points" case the cold-solve tests do NOT cover.
        prev = startA
        steps = 60
        for s in range(1, steps + 1):
            p = startA + (B - startA) * (s / steps)
            _step(reach, q, {"left": p, "right": None})
            tip = np.asarray(reach.fingertip("left", q))
            max_step = max(max_step, float(np.linalg.norm(tip - prev)))
            prev = tip
        for _ in range(40):                       # hold B (still warm)
            _step(reach, q, {"left": B, "right": None})
        errs.append(float(np.linalg.norm(B - reach.fingertip("left", q))))
    errs = np.array(errs)
    print(f"     warm A->B end err: mean {errs.mean()*1e3:.2f}  max {errs.max()*1e3:.2f}mm  "
          f"<1mm {100*(errs<1e-3).mean():.0f}% | max fingertip step {max_step*1e3:.1f}mm")
    return {
        "warm-solve in-motion lands on B <2mm": errs.max() < 2e-3,
        "warm-solve >=90% <1mm": (errs < 1e-3).mean() >= 0.90,
        "warm-solve stayed in-branch (no >40mm jump)": max_step < 0.040,
    }


# --- C. planner keeps a path clear when the endpoints are clear --------------
def test_planner_collision_free(reach, planner):
    print("C. PLANNER COLLISION-FREE (clear endpoints -> the PATH stays clear)")
    # The honest test of "ensure no collisions": given a collision-free start AND
    # a collision-free goal config, the planner must keep the whole A->B path
    # self-collision-free (avoiding/detouring any mid-path contact). A goal pair
    # that is itself unavoidably colliding is out of scope (no path can fix it).
    cm = planner.collision
    rng = np.random.default_rng(303)
    free, reached, flagged_ok, n = 0, 0, 0, 0
    target = 24
    tries = 0
    while n < target and tries < 400:
        tries += 1
        lift = rng.uniform(0.25, 0.6)
        q0 = _q0(lift)
        if cm.clearance(q0)[0] <= planner.margin:
            continue                            # start must be clear
        gl = _smooth_goal(reach, rng, "left", q0, lift, delta=0.5)
        gr = _smooth_goal(reach, rng, "right", q0, lift, delta=0.5)
        # Build the goal joint config and require IT be collision-free.
        qg = _q0(lift)
        reach._cache = None
        _settle(reach, qg, {"left": gl, "right": gr}, ticks=250)
        if cm.clearance(qg)[0] <= planner.margin:
            continue                            # goal itself collides -> skip
        n += 1
        traj = planner.plan(q0, {"left": gl, "right": gr})
        reached += int(traj.reached)
        if traj.collision_free:
            free += 1
        # INDEPENDENT honesty check: recompute clearance from each waypoint's
        # stored joint config (which holds BOTH arms) and confirm the trajectory's
        # collision_free flag matches ground truth -- so the flag reflects the
        # ACHIEVED configs, not just the planner's internal bookkeeping. (This can
        # genuinely fail if the planner mislabels or drops a collision.)
        indep_collides = any(cm.clearance(w.q)[0] < planner.margin
                             for w in traj.waypoints)
        if traj.collision_free == (not indep_collides):
            flagged_ok += 1
    print(f"     clear-endpoint moves: {n} | reached {reached}/{n} | "
          f"collision-free path {free}/{n} | flag matches ground-truth {flagged_ok}/{n}")
    return {
        "planner keeps clear-endpoint paths collision-free (>=90%)": free >= 0.9 * n,
        "planner reaches clear dual goals (>=90%)": reached >= 0.9 * n,
        "collision_free flag matches independent recheck": flagged_ok == n,
    }


# --- D. avoidance is sound: exercised, never degrades reach or clearance ------
def test_avoidance_sound(reach, planner):
    print("D. AVOIDANCE SOUND (exercised on real collisions; never degrades)")
    # avoid=True must never end up with LESS clearance or break a reach that
    # avoid=False achieved. The comparison is exact (no slop that would swamp a
    # real regression). We also REQUIRE the avoidance path to actually be
    # exercised (some goals self-collide so phase-2/detour runs), and we count
    # reach-preservation ONLY over reachable baselines (so it is not satisfied
    # vacuously by goals that were unreachable to begin with).
    rng = np.random.default_rng(404)
    reach_base, ok_reach_base = 0, 0
    free_F, free_T, introduced, fixed, colliding = 0, 0, 0, 0, 0
    N = 30
    for _ in range(N):
        lift = rng.uniform(0.3, 0.6)
        q0 = _q0(lift)
        # Dual goals pulled toward each other / the centerline so some genuinely
        # self-collide (gives the avoidance work to do).
        gl = _smooth_goal(reach, rng, "left", q0, lift, delta=0.6)
        gr = _smooth_goal(reach, rng, "right", q0, lift, delta=0.6)
        trF = planner.plan(q0, {"left": gl, "right": gr}, avoid=False)
        trT = planner.plan(q0, {"left": gl, "right": gr}, avoid=True)
        free_F += int(trF.collision_free)
        free_T += int(trT.collision_free)
        if not trF.collision_free:
            colliding += 1
        # The load-bearing soundness property: avoidance must never turn a
        # collision-free path INTO a colliding one, and it should FIX some.
        if trF.collision_free and not trT.collision_free:
            introduced += 1
        if (not trF.collision_free) and trT.collision_free:
            fixed += 1
        if trF.reached:                                     # reachable baseline
            reach_base += 1
            ok_reach_base += int(trT.reached)
    print(f"     collision-free: no-avoid {free_F}/{N} -> avoid {free_T}/{N} | "
          f"introduced {introduced} fixed {fixed} | reach preserved "
          f"{ok_reach_base}/{reach_base} reachable | exercised on {colliding}/{N}")
    return {
        # Falsifiable: a detour/null-step that drove a clear path into contact fails.
        "avoidance never introduces a self-collision": introduced == 0,
        "avoidance improves the collision-free count": free_T >= free_F,
        "avoidance fixes >=1 real collision": fixed >= 1,
        "avoidance preserves every reachable reach": (ok_reach_base == reach_base
                                                      and reach_base >= 10),
        "avoidance actually exercised (>=3 colliding)": colliding >= 3,
    }


# --- E. persistent refinement + bounded give-up ------------------------------
def test_persistence(reach):
    print("E. PERSISTENCE (hard targets driven sub-mm; unreachable settles bounded)")
    rng = np.random.default_rng(2024)
    # Dual reachable targets generated together (shared lift): the worst few used
    # to plateau a few mm short; persistence keeps refining to sub-mm.
    errs = []
    N = 40
    for _ in range(N):
        lift = rng.uniform(0.05, 0.8)
        qg = _clip_cfg(reach, "left",
                       [rng.uniform(reach.model.joints[j].lower,
                                    reach.model.joints[j].upper)
                        for j in ARM_JOINTS["left"]], lift)
        qg.update({k: v for k, v in _clip_cfg(reach, "right",
                   [rng.uniform(reach.model.joints[j].lower,
                                reach.model.joints[j].upper)
                    for j in ARM_JOINTS["right"]], lift).items() if k != LIFT_JOINT})
        tL = np.asarray(reach.fingertip("left", qg))
        tR = np.asarray(reach.fingertip("right", qg))
        q = _q0(0.4)
        reach._cache = None
        d = _settle(reach, q, {"left": tL, "right": tR}, ticks=500)
        errs.append(max(d.values()))
    errs = np.array(errs)
    print(f"     hard dual targets: <1mm {100*(errs<1e-3).mean():.0f}%  "
          f"<2mm {100*(errs<2e-3).mean():.0f}%  max {errs.max()*1e3:.2f}mm")

    # A genuinely unreachable HELD target: the solver goes max-effort (may briefly
    # exceed the 60 Hz budget while it searches), but once it has GIVEN UP (found
    # the closest config the joints allow) it must STOP hammering the goal, so the
    # late-phase per-tick latency drops back under budget instead of staying high
    # forever. ONE solve_step per timing window (true per-tick latency).
    far = np.array([0.9, 0.0, 2.6])
    q = _q0(0.4)
    reach._cache = None
    t_un = []
    for _ in range(800):
        t0 = time.perf_counter()
        res = reach.solve_step(q, {"left": far, "right": None})
        t_un.append((time.perf_counter() - t0) * 1e3)
        for k, v in res.items():
            if not k.startswith("_"):
                q[k] = v
    t_un = np.array(t_un)
    settled = float(np.linalg.norm(far - reach.fingertip("left", q)))
    print(f"     unreachable target: settled {settled*1e3:.0f}mm (closest) | "
          f"early max {t_un[:60].max():.1f}ms -> late(>200) max {t_un[200:].max():.2f}ms "
          f"mean {t_un[200:].mean():.2f}ms (gives up, drops back under budget)")
    return {
        "persistence drives hard dual targets <2mm": errs.max() < 2e-3,
        "persistence hard dual >=95% <1mm": (errs < 1e-3).mean() >= 0.95,
        "unreachable gives up (late per-tick < budget)": t_un[200:].max() < BUDGET_MS,
        "unreachable worst-case bounded < 60ms": t_un.max() < 60.0,
        "unreachable settles (no NaN)": math.isfinite(settled),
    }


def main():
    print("\n=======  POINT-TO-POINT TRAJECTORY / PATHING TESTS  =======")
    reach = load()
    planner = TrajectoryPlanner(reach, CollisionModel(reach))
    gates = {}
    for fn, args in (
        (test_plan_and_track, (reach, planner)),
        (test_warm_solve, (reach,)),
        (test_planner_collision_free, (reach, planner)),
        (test_avoidance_sound, (reach, planner)),
        (test_persistence, (reach,)),
    ):
        gates.update(fn(*args))
        print()
    tt = np.array(TICKS)
    over = int((tt > BUDGET_MS).sum())
    print(f"F. LATENCY (solve_step over the suite): mean {tt.mean():.2f}ms "
          f"p99 {np.percentile(tt,99):.2f}ms max {tt.max():.2f}ms "
          f"over-budget {over}/{len(tt)} ({100*over/len(tt):.2f}%)")
    # 60 Hz is a goal, not a hard cutoff (far targets get a bigger budget for
    # accuracy); typical ticks real-time, worst-case bounded.
    gates["typical solve_step < budget (median)"] = np.median(tt) < BUDGET_MS
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

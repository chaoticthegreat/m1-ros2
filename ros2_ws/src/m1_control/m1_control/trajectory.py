"""Collision-free Cartesian path planning for the M1 dual-arm robot.

The live controller (:mod:`m1_control.controller_node` via
:class:`m1_control.kinematics.ReachController`) is a *reactive* solver: it drives
the fingertip toward a single target point each tick and knows nothing about the
robot's body. This module adds the missing offline piece -- given a start joint
configuration and a goal point (per arm), it produces a **collision-free
trajectory** of intermediate joint configurations the gripper can follow from A
to B.

It is intentionally *test + visualization only* (the proven real-time teleop path
is untouched). The planner is used by the trajectory test suite and by the Quest
node to draw a live path preview the operator can see before committing a motion.

How it plans:

  1. **Cartesian interpolation.** Take a straight line in task space from the
     start fingertip to the goal point (per active arm), sampled at ``n``
     waypoints. (A straight Cartesian path is the natural "go from here to there"
     motion and is what gets visualized.)
  2. **Warm-started IK per waypoint.** Solve each waypoint's joint configuration
     with the SAME damped-least-squares + adaptive-damping core the live solver
     uses (:meth:`ReachController._stack` / ``_dls``), warm-started from the
     previous waypoint -- so the path stays in one IK branch and is smooth, just
     like continuous teleop tracking.
  3. **Null-space self-collision avoidance.** The arm is redundant for a 3-DOF
     position task (7 joints + a shared lift), so when a waypoint's natural IK
     configuration would self-collide we push the joints *up the clearance
     gradient projected into the task null space* -- this slides the elbow / lift
     out of the collision WITHOUT moving the fingertip off its target. If a
     waypoint still cannot be made clear (the target point itself is
     unavoidably in collision), the planner reports the trajectory as NOT
     collision-free rather than pretending otherwise.

The returned :class:`Trajectory` carries, per waypoint, the joint configuration,
the per-arm Cartesian residual, the signed self-collision clearance, and a
colliding flag, plus path-level summaries (``collision_free``, ``reached``,
``end_error``) the tests gate on and the viz colours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from m1_control.collision import CollisionModel
from m1_control.kinematics import (
    ARM_JOINTS,
    IK_POS_TOL,
    LIFT_JOINT,
    ReachController,
    _IK_INT_MAX_DQ,
)

# --- planner tuning --------------------------------------------------------
PLAN_WAYPOINTS = 24        # samples along the Cartesian A->B line (inclusive)
PLAN_MARGIN = 0.01         # keep at least this signed clearance (m) -> collision-free
PLAN_IK_ITERS = 120        # max GN iterations to converge one waypoint (offline)
_AVOID_ITERS = 40          # max null-space clearance-improving nudges per waypoint
_AVOID_GAIN = 0.5          # null-space push along the clearance gradient
_AVOID_STEP_CAP = 0.08     # cap (rad) on one clearance nudge before re-converging
_RECONVERGE_ITERS = 8      # task cleanup iterations after each clearance nudge
_POSTURE_GAIN = 0.02       # gentle pull toward the warm-start (branch continuity)
_STEP_TOL = 1e-7           # stop a task solve when the step is this small
# Path detour: when an INTERMEDIATE waypoint still self-collides after null-space
# avoidance (a task-coupled collision the redundant DOFs can't open -- e.g. a
# gripper pinned near the body), bow the fingertip PATH around the obstacle by
# shifting that waypoint's target along the separating direction and re-solving.
# Endpoints (A, B) are never detoured -- they are the goals we must hit exactly.
_DETOUR_ITERS = 24         # max target shifts per colliding intermediate waypoint
_DETOUR_STEP = 0.02        # how far (m) to nudge the target each detour iteration
_DETOUR_MAX_DEV = 0.18     # cap (m) on how far a waypoint may leave the straight line


@dataclass
class Waypoint:
    """One sample along the planned path."""
    index: int
    points: dict                       # arm -> target point (3-vector) this sample
    q: dict                            # solved joint configuration (name -> value)
    tips: dict                         # arm -> achieved fingertip (3-vector)
    residual: dict                     # arm -> ||target - tip|| (m)
    clearance: float                   # signed self-collision clearance (m)
    colliding: bool                    # clearance < margin


@dataclass
class Trajectory:
    """A planned path A->B with per-waypoint state and path-level summaries."""
    arms: list
    waypoints: list = field(default_factory=list)
    collision_free: bool = True        # every waypoint clear of self-collision
    reached: bool = True               # final residual within tolerance (per arm)
    end_error: dict = field(default_factory=dict)   # arm -> final residual (m)
    min_clearance: float = float("inf")

    def points_for(self, arm: str) -> list:
        """Fingertip path of ``arm`` (list of 3-vectors) -- the viz polyline."""
        return [wp.tips[arm] for wp in self.waypoints if arm in wp.tips]

    def configs(self) -> list:
        """The collision-checked joint configurations, in order."""
        return [wp.q for wp in self.waypoints]


class TrajectoryPlanner:
    """Plans collision-free Cartesian paths using a :class:`ReachController`.

    Reuses the reach controller's FK / stacked Jacobian / damped-least-squares
    core, plus a :class:`CollisionModel` for the null-space self-collision
    avoidance. Holds no live state, so it is safe to call from a background
    thread (e.g. the Quest path-preview worker).
    """

    def __init__(self, reach: ReachController, collision: CollisionModel = None,
                 margin: float = PLAN_MARGIN):
        self.reach = reach
        self.collision = collision if collision is not None else CollisionModel(reach)
        self.margin = float(margin)

    # --- per-waypoint IK with null-space collision avoidance ---------------
    def _task_solve(self, q, arms, joint_order, lo, hi, pos, q_seed, max_iters):
        """Converge the fingertip task (warm-started DLS, no avoidance).

        Uses the live solver's stacked Jacobian + adaptive-damping step, plus a
        gentle null-space posture pull toward the warm-start so the path stays in
        one IK branch. Returns ``(q, dist)``.
        """
        seed = np.asarray(q_seed, dtype=np.float64)
        dist = {}
        for _ in range(max_iters):
            J, e, dist = self.reach._stack(q, joint_order, arms, pos)
            if all(d < IK_POS_TOL for d in dist.values()):
                break
            dq_task, N = self.reach._dls(J, e)
            dq = dq_task + N @ (_POSTURE_GAIN * (seed - q))
            nrm = float(np.linalg.norm(dq))
            if nrm < _STEP_TOL:
                break
            if nrm > _IK_INT_MAX_DQ:
                dq *= _IK_INT_MAX_DQ / nrm
            q = np.clip(q + dq, lo, hi)
        return q, dist

    def _solve_waypoint(self, arms, joint_order, lo, hi, pos, q_seed,
                        idx_all, max_iters, avoid, background=None):
        """Solve one waypoint with TASK PRIORITY, clearance best-effort.

        Phase 1 converges the fingertip onto ``pos`` (reaching the goal is the
        non-negotiable objective). Phase 2 -- only if that config self-collides --
        improves the clearance with null-space nudges along the clearance
        gradient, RE-CONVERGING the task after each nudge so the fingertip never
        drifts off target. If no task-satisfying clear config can be found
        (the target point itself is unavoidably in collision), the honest
        (reached-but-colliding) config is returned and the caller flags the path.

        Returns ``(q_vec, dist, clearance)``.
        """
        q = np.clip(np.asarray(q_seed, dtype=np.float64), lo, hi)
        # Phase 1: reach the target point.
        q, dist = self._task_solve(q, arms, joint_order, lo, hi, pos, q_seed,
                                   max_iters)
        clr = self.collision.clearance_of_vec(q, joint_order, background)
        if not avoid or clr >= self.margin:
            return q, dist, clr
        # Phase 2: push clearance up in the task null space, re-converging task.
        for _ in range(_AVOID_ITERS):
            grad = self.collision.clearance_gradient(
                q, joint_order, idx_all, background=background)
            J, e, _ = self.reach._stack(q, joint_order, arms, pos)
            _, N = self.reach._dls(J, e)
            step = N @ (_AVOID_GAIN * grad)
            n = float(np.linalg.norm(step))
            if n < 1e-9:
                break                          # plateau: gradient gives nothing
            if n > _AVOID_STEP_CAP:
                step *= _AVOID_STEP_CAP / n
            q_try = np.clip(q + step, lo, hi)
            # Re-converge the fingertip onto the target after the nudge.
            q_try, dist_try = self._task_solve(
                q_try, arms, joint_order, lo, hi, pos, q_try, _RECONVERGE_ITERS)
            reached = all(d < 5e-3 for d in dist_try.values())
            clr_try = self.collision.clearance_of_vec(q_try, joint_order, background)
            # Accept only if the task is still met and clearance genuinely
            # improved -- so avoidance can never trade the goal away.
            if reached and clr_try > clr + 1e-5:
                q, dist, clr = q_try, dist_try, clr_try
                if clr >= self.margin:
                    break
            else:
                break                          # no task-safe improvement available
        return q, dist, clr

    def _detour(self, arms, joint_order, lo, hi, line_pos, q, dist, clr,
                max_iters, background=None):
        """Route an intermediate waypoint around a task-coupled collision.

        Shifts the colliding arm's fingertip target along the separating
        direction (from :meth:`CollisionModel.clearance_detail`) and re-solves,
        bowing the path off the straight line until it clears the obstacle or the
        deviation cap is hit. ``line_pos`` is the original straight-line target
        per arm (the deviation is measured from it). Returns the updated
        ``(q, dist, clearance, achieved_targets)``.
        """
        cur = {a: np.array(line_pos[a], dtype=np.float64) for a in arms}
        for _ in range(max_iters):
            if clr >= self.margin:
                break
            q_dict = dict(background) if background else {}
            for k, jn in enumerate(joint_order):
                q_dict[jn] = float(q[k])
            _, _, push = self.collision.clearance_detail(q_dict)
            if not push:
                break
            moved = False
            for a in arms:
                if a not in push:
                    continue
                nt = cur[a] + _DETOUR_STEP * push[a]
                dev = nt - line_pos[a]
                dn = float(np.linalg.norm(dev))
                if dn > _DETOUR_MAX_DEV:       # clamp deviation from the line
                    nt = line_pos[a] + dev * (_DETOUR_MAX_DEV / dn)
                cur[a] = nt
                moved = True
            if not moved:
                break
            q_try, dist_try = self._task_solve(
                q.copy(), arms, joint_order, lo, hi, cur, q, _RECONVERGE_ITERS * 2)
            clr_try = self.collision.clearance_of_vec(q_try, joint_order, background)
            # Keep the detour only if it actually opened the gap (the fingertip
            # following the bowed target is the whole point, so we don't require
            # the original line residual -- ``cur`` IS the new target).
            if clr_try > clr + 1e-5:
                q, dist, clr = q_try, dist_try, clr_try
            else:
                break
        return q, dist, clr, cur

    # --- public planning entry point ---------------------------------------
    def plan(self, start_q: dict, goals: dict, n: int = PLAN_WAYPOINTS,
             avoid: bool = True, max_iters: int = PLAN_IK_ITERS) -> Trajectory:
        """Plan a collision-free Cartesian path from ``start_q`` to ``goals``.

        ``goals`` maps arm -> goal point (3-vector) or ``None``. Active arms are
        those with a goal. The fingertip of each active arm is interpolated
        linearly from its start (FK of ``start_q``) to its goal over ``n+1``
        samples; each sample is solved with collision-avoiding IK warm-started
        from the previous one.
        """
        arms = [a for a in ("left", "right") if goals.get(a) is not None]
        if not arms:
            return Trajectory(arms=[])

        joint_order = []
        for a in arms:
            joint_order += ARM_JOINTS[a]
        if LIFT_JOINT not in joint_order:
            joint_order.append(LIFT_JOINT)
        lo = np.array([self.reach.model.joints[j].lower for j in joint_order])
        hi = np.array([self.reach.model.joints[j].upper for j in joint_order])
        idx_all = list(range(len(joint_order)))

        starts = {a: np.asarray(self.reach.fingertip(a, start_q), dtype=np.float64)
                  for a in arms}
        ends = {a: np.asarray(goals[a], dtype=np.float64) for a in arms}

        # Background config for collision FK: the INACTIVE arm's MEASURED joints,
        # so a single-arm plan is checked against the other arm where it actually
        # is (its real folded pose), not a phantom straight-out (q=0) arm. The
        # inactive arm is held fixed -- the planner only moves ``joint_order``.
        background = {j: float(start_q.get(j, 0.0))
                      for a in ("left", "right") if a not in arms
                      for j in ARM_JOINTS[a]}

        traj = Trajectory(arms=list(arms))
        q_seed = np.array([start_q.get(j, 0.0) for j in joint_order], dtype=np.float64)
        for i in range(n + 1):
            s = i / n if n > 0 else 1.0
            line = {a: (1.0 - s) * starts[a] + s * ends[a] for a in arms}
            pos = {a: line[a].copy() for a in arms}
            q_vec, dist, clr = self._solve_waypoint(
                arms, joint_order, lo, hi, pos, q_seed, idx_all, max_iters, avoid,
                background=background)
            # Intermediate waypoint still colliding after null-space avoidance:
            # bow the PATH around the obstacle (endpoints A/B are never detoured,
            # they are the goals we must hit exactly).
            if avoid and clr < self.margin and 0 < i < n:
                q_vec, dist, clr, pos = self._detour(
                    arms, joint_order, lo, hi, line, q_vec, dist, clr,
                    _DETOUR_ITERS, background=background)
            q_seed = q_vec                       # warm-start the next waypoint
            q_dict = {jn: float(q_vec[k]) for k, jn in enumerate(joint_order)}
            q_dict.update(background)             # full config (for viz + re-checks)
            tips = {a: np.asarray(self.reach.fingertip(a, q_dict)) for a in arms}
            colliding = clr < self.margin
            traj.waypoints.append(Waypoint(
                index=i, points={a: pos[a] for a in arms}, q=q_dict, tips=tips,
                residual={a: float(dist.get(a, 0.0)) for a in arms},
                clearance=clr, colliding=colliding))
            traj.min_clearance = min(traj.min_clearance, clr)
            if colliding:
                traj.collision_free = False

        last = traj.waypoints[-1]
        traj.end_error = dict(last.residual)
        traj.reached = all(v < 5e-3 for v in last.residual.values())
        return traj


# ---------------------------------------------------------------------------
# Smoke test: plan single- and dual-arm paths and report. Run:
#   PYTHONPATH=ros2_ws/src/m1_control /usr/bin/python3 -m m1_control.trajectory
# ---------------------------------------------------------------------------
def _smoketest() -> int:
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    urdf = None
    for cand in (
        os.path.join(here, "..", "..", "..", "..", "assets",
                     "ranger_air_description", "urdf",
                     "ranger_air_description.urdf"),
        "assets/ranger_air_description/urdf/ranger_air_description.urdf",
    ):
        if os.path.isfile(cand):
            urdf = cand
            break
    if urdf is None:
        print("URDF not found; cannot run smoke test")
        return 1

    from m1_control.kinematics import UrdfModel

    reach = ReachController(UrdfModel.from_string(open(urdf).read()))
    cm = CollisionModel(reach)
    planner = TrajectoryPlanner(reach, cm)

    def cfg(lift=0.4):
        q = {j: 0.0 for j in ARM_JOINTS["left"] + ARM_JOINTS["right"]}
        q[LIFT_JOINT] = lift
        return q

    gates = {}

    # Single-arm: move the left fingertip to a reachable nearby point in OPEN
    # space. The arms now mount flush on the lift carriage (~0.70 m lower), so a
    # descending goal would pin the tip down by the carriage/body column (a
    # task-coupled self-collision the planner honestly flags); aim forward + out +
    # slightly up to keep this smoke path clear of the structure.
    q0 = cfg(0.4)
    start_tip = np.asarray(reach.fingertip("left", q0))
    goal = start_tip + np.array([0.12, 0.15, 0.05])
    tr = planner.plan(q0, {"left": goal, "right": None})
    print(f"single: reached={tr.reached} clear={tr.collision_free} "
          f"end_err={tr.end_error['left']*1e3:.2f}mm minclr={tr.min_clearance*1e3:.0f}mm "
          f"wpts={len(tr.waypoints)}")
    gates["single reaches goal"] = tr.reached
    gates["single collision-free"] = tr.collision_free

    # Dual-arm: move both fingertips forward + up so the arms come near each other
    # (a case the naive straight-line config brings them close -> avoidance), while
    # staying resolvable to collision-free. With the arms mounted flush+low on the
    # carriage, a downward inward goal pins the tips by the body (unavoidable task-
    # coupled collision), so aim forward/up where the planner can keep it clear.
    q0 = cfg(0.45)
    sl = np.asarray(reach.fingertip("left", q0))
    sr = np.asarray(reach.fingertip("right", q0))
    gl = sl + np.array([0.08, 0.05, 0.06])
    gr = sr + np.array([0.08, -0.05, 0.06])
    tr2 = planner.plan(q0, {"left": gl, "right": gr})
    print(f"dual:   reached={tr2.reached} clear={tr2.collision_free} "
          f"endL={tr2.end_error['left']*1e3:.2f} endR={tr2.end_error['right']*1e3:.2f}mm "
          f"minclr={tr2.min_clearance*1e3:.0f}mm")
    gates["dual reaches goal"] = tr2.reached
    gates["dual collision-free"] = tr2.collision_free

    print("\n----------------  TRAJECTORY GATES  ----------------")
    npass = 0
    for name, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        npass += int(ok)
    print(f"\n{npass}/{len(gates)} gates passed")
    return 0 if npass == len(gates) else 1


if __name__ == "__main__":
    import sys
    sys.exit(_smoketest())

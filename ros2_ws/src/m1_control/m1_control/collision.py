"""Dependency-free capsule self-collision model for the M1 dual-arm robot.

The reach controller in :mod:`m1_control.kinematics` is a reactive Cartesian
solver with no notion of the robot's *body* -- it drives a fingertip to a point
and never asks whether the arm passes through the other arm or the torso on the
way. The trajectory planner (:mod:`m1_control.trajectory`) needs exactly that
check, so this module supplies a cheap, conservative self-collision model built
from the SAME URDF forward kinematics (no meshes, no FCL, no new deps).

Each link is approximated by a **capsule** -- a line segment (its FK centerline)
swept by a sphere of a fixed radius. Two capsules collide when the shortest
distance between their center segments is less than the sum of their radii, so
the whole check reduces to segment-to-segment distance, which is a few lines of
numpy. The radii are deliberately a touch generous (the arm links are slimmer
than the capsule), so "clear" here means clear with margin -- a planner that
keeps the clearance non-negative produces motion that is collision-free on the
real geometry too.

What is checked (matches the project's scope -- self-collision only):

  * **arm <-> arm**  -- the left arm's moving links against the right arm's, the
    most likely real collision when both arms reach across each other.
  * **arm <-> body** -- each arm's moving links against the shared base / lift
    column / torso riser (the arms are mounted on top of it and can fold down
    into it).

Adjacent capsules that share a joint are never checked against each other (they
touch by construction), and the two arms' shared lower chain (base -> lift ->
mount) is treated as one *body* obstacle, not as part of either arm.

The model exposes a signed *clearance* (positive = gap, negative = penetration)
plus a finite-difference clearance gradient, which the planner projects into the
arm's task null space to slide a colliding-but-reachable waypoint to a
collision-free joint configuration WITHOUT moving the fingertip off its target.
"""

from __future__ import annotations

import numpy as np

from m1_control.kinematics import ARM_JOINTS, LIFT_JOINT

# --- Capsule radii (m). Conservative over-approximations of the real link
# thickness, so a non-negative clearance is collision-free with margin. Tunable.
ARM_RADIUS = 0.05          # slim arm links (real ~0.04; padded a little)
GRIPPER_RADIUS = 0.06      # the gripper/fingers are a little bulkier
BODY_RADIUS = 0.11         # lift column / torso riser
BASE_RADIUS = 0.19         # the wide base platform at the bottom
# Endpoints closer than this are treated as a shared joint, so the two capsules
# meeting there are adjacent and never checked against each other.
_ADJACENCY_EPS = 1e-3


def _seg_seg_witness(p1, q1, p2, q2):
    """Closest points between segment ``p1->q1`` and ``p2->q2``.

    Standard clamped-parameter closest-point-of-two-segments solution (Ericson,
    *Real-Time Collision Detection*). Pure numpy, robust to parallel / degenerate
    (zero-length) segments. Returns ``(distance, c1, c2)`` where ``c1`` is the
    closest point on segment 1 and ``c2`` on segment 2.
    """
    d1 = q1 - p1            # direction + length of segment 1
    d2 = q2 - p2            # direction + length of segment 2
    r = p1 - p2
    a = float(d1 @ d1)      # squared length of segment 1
    e = float(d2 @ d2)      # squared length of segment 2
    f = float(d2 @ r)
    EPS = 1e-12
    if a <= EPS and e <= EPS:           # both points
        return float(np.linalg.norm(p1 - p2)), p1, p2
    if a <= EPS:                        # segment 1 is a point
        s = 0.0
        t = min(max(f / e, 0.0), 1.0)
    else:
        c = float(d1 @ r)
        if e <= EPS:                    # segment 2 is a point
            t = 0.0
            s = min(max(-c / a, 0.0), 1.0)
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b       # always >= 0
            s = min(max((b * f - c * e) / denom, 0.0), 1.0) if denom > EPS else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = min(max(-c / a, 0.0), 1.0)
            elif t > 1.0:
                t = 1.0
                s = min(max((b - c) / a, 0.0), 1.0)
    c1 = p1 + d1 * s
    c2 = p2 + d2 * t
    return float(np.linalg.norm(c1 - c2)), c1, c2


def _seg_seg_distance(p1, q1, p2, q2):
    """Shortest distance between two segments (witness points discarded)."""
    return _seg_seg_witness(p1, q1, p2, q2)[0]


class Capsule:
    """A line segment ``(a, b)`` with radius ``r`` and a tag for diagnostics."""

    __slots__ = ("a", "b", "r", "tag")

    def __init__(self, a, b, r, tag):
        self.a = np.asarray(a, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.r = float(r)
        self.tag = tag


def _signed_distance(c1: Capsule, c2: Capsule) -> float:
    """Signed clearance between two capsules: positive = gap, negative = overlap."""
    return _seg_seg_distance(c1.a, c1.b, c2.a, c2.b) - (c1.r + c2.r)


class CollisionModel:
    """Capsule self-collision model derived from a :class:`ReachController`.

    Built once from the reach controller (which already holds the URDF FK
    chains). ``clearance(q)`` returns the worst-case signed gap across all
    checked capsule pairs; ``self_collision(q)`` thresholds it.
    """

    def __init__(self, reach, arm_radius=ARM_RADIUS, gripper_radius=GRIPPER_RADIUS,
                 body_radius=BODY_RADIUS, base_radius=BASE_RADIUS):
        self.reach = reach
        self.arm_radius = float(arm_radius)
        self.gripper_radius = float(gripper_radius)
        self.body_radius = float(body_radius)
        self.base_radius = float(base_radius)
        # Index the FK link_points into three zones, from the chain structure:
        #
        #   points [0 .. split-1]   shared column (base -> lift -> mount): BODY
        #   points [split-1 .. j1]  each arm's STATIC riser/bracket (rigid; moves
        #                           only with the lift) -> also treated as BODY
        #   points [j1 .. tip]      the MOVING arm (articulates with joints 1..7)
        #
        # ``split`` is the first link_points index where the two arms diverge
        # (their fixed mount bracket); ``j1`` is the index of the first ACTUATED
        # arm joint's frame -- only from there on do the links actually move with
        # the arm. Earlier code mistook the rigid risers for moving arm links, so
        # the two arms' fixed brackets read as a constant (pose-independent)
        # collision. Checking only moving-vs-moving (and moving-vs-body, where the
        # body includes the OTHER arm's static riser) fixes that while still
        # catching a wrist that swings into the other arm's riser.
        ch = reach.chains["left"].chain
        cr = reach.chains["right"].chain
        k = 0
        while k < min(len(ch), len(cr)) and ch[k] == cr[k]:
            k += 1
        self._split = k + 1                     # first divergent point index
        # First actuated arm joint (openarm_*_joint1) -> its point index.
        self._j1 = ch.index(ARM_JOINTS["left"][0]) + 1

    # --- capsule assembly --------------------------------------------------
    def _moving_caps(self, arm: str, pts: np.ndarray) -> list:
        """Capsules for one arm's MOVING links (joint1 frame -> fingertip)."""
        caps = []
        last = len(pts) - 1
        for i in range(self._j1, last):
            a, b = pts[i], pts[i + 1]
            if float(np.linalg.norm(b - a)) < _ADJACENCY_EPS:
                continue                       # zero-length link frame; skip
            r = self.gripper_radius if i + 1 == last else self.arm_radius
            caps.append(Capsule(a, b, r, f"{arm}[{i}]"))
        return caps

    def _riser_caps(self, arm: str, pts: np.ndarray) -> list:
        """Capsules for one arm's STATIC riser/bracket (mount -> joint1 frame)."""
        caps = []
        for i in range(self._split - 1, self._j1):
            a, b = pts[i], pts[i + 1]
            if float(np.linalg.norm(b - a)) < _ADJACENCY_EPS:
                continue
            caps.append(Capsule(a, b, self.arm_radius, f"riser_{arm}[{i}]"))
        return caps

    def _column_caps(self, col: np.ndarray) -> list:
        """Capsules for the shared body column (base -> lift -> mount)."""
        caps = []
        for i in range(self._split - 1):
            a, b = col[i], col[i + 1]
            if float(np.linalg.norm(b - a)) < _ADJACENCY_EPS:
                continue
            r = self.base_radius if i == 0 else self.body_radius
            caps.append(Capsule(a, b, r, f"body[{i}]"))
        return caps

    def capsules(self, q: dict) -> dict:
        """Capsule groups at joint config ``q``.

        Keys: ``lm``/``rm`` (left/right MOVING arm links), ``lr``/``rr``
        (left/right static risers), ``col`` (shared body column). The public
        ``left``/``right``/``body`` keys (moving arms + all static structure) are
        also provided for visualization.
        """
        lp = {a: np.asarray(self.reach.chains[a].link_points(q))
              for a in ("left", "right")}
        lm = self._moving_caps("left", lp["left"])
        rm = self._moving_caps("right", lp["right"])
        lr = self._riser_caps("left", lp["left"])
        rr = self._riser_caps("right", lp["right"])
        col = self._column_caps(lp["left"])
        return {"lm": lm, "rm": rm, "lr": lr, "rr": rr, "col": col,
                "left": lm, "right": rm, "body": col + lr + rr}

    # --- queries -----------------------------------------------------------
    def _pairs(self, caps: dict):
        """Capsule pairs to check. An arm's MOVING links are tested against:
          * the OTHER arm's moving links and static riser  (arm <-> arm), and
          * the shared body column                         (arm <-> body).
        An arm is never tested against its OWN structure (intra-arm self-collision
        is out of scope and handled by joint limits), so the rigid risers never
        read as a constant collision.
        """
        lm, rm, lr, rr, col = (caps["lm"], caps["rm"], caps["lr"],
                               caps["rr"], caps["col"])
        out = []
        for c1 in lm:                          # left moving vs right arm + column
            for c2 in rm:
                out.append((c1, c2))
            for c2 in rr:
                out.append((c1, c2))
            for c2 in col:
                out.append((c1, c2))
        for c1 in rm:                          # right moving vs left riser + column
            for c2 in lr:
                out.append((c1, c2))
            for c2 in col:
                out.append((c1, c2))
        return out

    def clearance(self, q: dict):
        """Worst-case signed clearance over all checked pairs.

        Returns ``(min_clearance, worst_pair)`` where ``min_clearance`` is the
        smallest signed gap (negative => penetration) and ``worst_pair`` is the
        ``(tagA, tagB)`` that produced it (``None`` if nothing to check).
        """
        caps = self.capsules(q)
        worst = float("inf")
        worst_pair = None
        for c1, c2 in self._pairs(caps):
            d = _signed_distance(c1, c2)
            if d < worst:
                worst = d
                worst_pair = (c1.tag, c2.tag)
        if worst_pair is None:
            return float("inf"), None
        return worst, worst_pair

    def self_collision(self, q: dict, margin: float = 0.0) -> bool:
        """True if any checked pair is closer than ``margin`` (default: touching)."""
        return self.clearance(q)[0] < margin

    def clearance_detail(self, q: dict):
        """Worst pair plus the task-space push that would separate it.

        Returns ``(min_clearance, pair_tags, push)`` where ``push`` maps each
        MOVING arm involved in the worst pair to a unit vector pointing the
        direction its links should travel to open the gap. The planner shifts the
        colliding arm's fingertip target along this to *route the path around* an
        obstacle the redundant DOFs alone cannot clear (a task-coupled collision,
        e.g. a gripper pinned near the body/other arm). ``push`` is empty if there
        is nothing to check.
        """
        caps = self.capsules(q)
        worst = float("inf")
        info = None
        for c1, c2 in self._pairs(caps):
            d, w1, w2 = _seg_seg_witness(c1.a, c1.b, c2.a, c2.b)
            sd = d - (c1.r + c2.r)
            if sd < worst:
                worst = sd
                info = (c1, c2, w1, w2)
        if info is None:
            return float("inf"), None, {}
        c1, c2, w1, w2 = info
        sep = w1 - w2                      # push c1's side away from c2's side
        n = float(np.linalg.norm(sep))
        if n < 1e-9:                       # coincident witnesses (deep penetration)
            sep = 0.5 * (c1.a + c1.b) - 0.5 * (c2.a + c2.b)
            n = float(np.linalg.norm(sep))
        sep = sep / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
        push = {}
        # c1 is always a moving arm capsule (tag "left[i]" / "right[i]").
        a1 = "left" if c1.tag.startswith("left[") else "right"
        push[a1] = sep
        # If the obstacle is the OTHER moving arm, push it the opposite way too.
        if c2.tag.startswith("left[") or c2.tag.startswith("right["):
            a2 = "left" if c2.tag.startswith("left[") else "right"
            push[a2] = -sep
        return worst, (c1.tag, c2.tag), push

    def clearance_of_vec(self, q_vec: np.ndarray, joint_order: list,
                         background: dict = None) -> float:
        """``clearance`` for a joint *vector* in ``joint_order`` (planner helper).

        ``background`` supplies joint values for DOFs NOT in ``joint_order`` --
        critically the other arm's MEASURED pose when only one arm is being
        planned. Without it those joints default to 0 (a phantom straight-out
        arm), so a path could be checked against a fictitious other arm. The
        ``joint_order`` values take precedence (they are what the planner moves).
        """
        q = dict(background) if background else {}
        for k, jn in enumerate(joint_order):
            q[jn] = float(q_vec[k])
        return self.clearance(q)[0]

    def clearance_gradient(self, q_vec: np.ndarray, joint_order: list,
                           idx: list, eps: float = 1e-4,
                           background: dict = None) -> np.ndarray:
        """Finite-difference gradient of the min-clearance wrt joints ``idx``.

        Central differences over the selected joint indices, so the planner can
        push a colliding waypoint *up the clearance gradient* (projected into the
        fingertip task's null space, so the target point is preserved). Cheap
        enough for offline planning: a couple of FK+distance evals per joint.
        ``background`` holds the fixed (e.g. other-arm measured) DOFs.
        """
        grad = np.zeros(len(idx), dtype=np.float64)
        for n, k in enumerate(idx):
            qp = q_vec.copy(); qp[k] += eps
            qm = q_vec.copy(); qm[k] -= eps
            grad[n] = (self.clearance_of_vec(qp, joint_order, background)
                       - self.clearance_of_vec(qm, joint_order, background)) / (2.0 * eps)
        return grad


# ---------------------------------------------------------------------------
# Self-test: validate the segment-segment distance math against closed-form
# cases, then exercise the model on the real URDF. Run:
#   /usr/bin/python3 -m m1_control.collision     (from ros2_ws/src/m1_control)
# or  /usr/bin/python3 ros2_ws/src/m1_control/m1_control/collision.py
# ---------------------------------------------------------------------------
def _selftest() -> int:
    import os
    import sys

    gates = {}

    # 1. Segment-segment distance: closed-form checks.
    f = _seg_seg_distance
    o = np.zeros(3)
    # parallel unit segments 1 apart along y
    gates["parallel segs"] = abs(
        f(np.array([0, 0, 0.]), np.array([1, 0, 0.]),
          np.array([0, 1, 0.]), np.array([1, 1, 0.])) - 1.0) < 1e-9
    # skew/crossing segments through origin and (0,0,1)->(0,0,2): closest 1
    gates["offset perpendicular"] = abs(
        f(np.array([-1, 0, 0.]), np.array([1, 0, 0.]),
          np.array([0, 0, 1.]), np.array([0, 0, 2.])) - 1.0) < 1e-9
    # crossing segments (X in a plane) -> distance 0
    gates["crossing segs"] = f(
        np.array([-1, -1, 0.]), np.array([1, 1, 0.]),
        np.array([-1, 1, 0.]), np.array([1, -1, 0.])) < 1e-9
    # point to segment
    gates["point-seg"] = abs(
        f(np.array([0, 2, 0.]), np.array([0, 2, 0.]),
          np.array([-1, 0, 0.]), np.array([1, 0, 0.])) - 2.0) < 1e-9
    # endpoint-clamped: colinear, gap 1
    gates["colinear gap"] = abs(
        f(np.array([0, 0, 0.]), np.array([1, 0, 0.]),
          np.array([2, 0, 0.]), np.array([3, 0, 0.])) - 1.0) < 1e-9

    # 2. Real-URDF model behavior.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    try:
        from m1_control.kinematics import ReachController, UrdfModel
    except ImportError:  # running as a file, fix path
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
        from m1_control.kinematics import ReachController, UrdfModel

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
        print("  (URDF not found; skipping model checks)")
    else:
        reach = ReachController(UrdfModel.from_string(open(urdf).read()))
        cm = CollisionModel(reach)
        rng = np.random.default_rng(7)

        def clip(arm, vals, lift):
            q = {LIFT_JOINT: float(np.clip(lift, 0.0, 0.85))}
            for j, v in zip(ARM_JOINTS[arm], vals):
                jt = reach.model.joints[j]
                q[j] = float(np.clip(v, jt.lower, jt.upper))
            return q

        # Statistical sanity: across many random *moderate* reachable configs the
        # model must read collision-free for the large majority -- otherwise the
        # capsules are so fat that ordinary dual-arm operation reads as a constant
        # collision and the planner is useless. (A minority genuinely cross.)
        clear = 0
        N = 80
        for _ in range(N):
            lift = rng.uniform(0.1, 0.7)
            vL = [rng.uniform(0.6 * reach.model.joints[j].lower,
                              0.6 * reach.model.joints[j].upper)
                  for j in ARM_JOINTS["left"]]
            vR = [rng.uniform(0.6 * reach.model.joints[j].lower,
                              0.6 * reach.model.joints[j].upper)
                  for j in ARM_JOINTS["right"]]
            q = clip("left", vL, lift)
            q.update({k: v for k, v in clip("right", vR, lift).items()
                      if k != LIFT_JOINT})
            if cm.clearance(q)[0] > 0.0:
                clear += 1
        print(f"  reachable-config clearance: {clear}/{N} collision-free "
              f"({100*clear/N:.0f}%)  split={cm._split} j1={cm._j1}")
        gates["most reachable poses collision-free (>=85%)"] = clear >= 0.85 * N

        # Explicitly swing both arms hard toward each other across the centerline
        # -> the moving links must cross and overlap (a hard self-collision).
        q = {j: 0.0 for j in ARM_JOINTS["left"] + ARM_JOINTS["right"]}
        q[LIFT_JOINT] = 0.5
        q[ARM_JOINTS["left"][0]] = 1.4
        q[ARM_JOINTS["right"][0]] = -1.4
        q[ARM_JOINTS["left"][1]] = 1.2
        q[ARM_JOINTS["right"][1]] = -1.2
        clr2, pair2 = cm.clearance(q)
        print(f"  crossed-arms clearance {clr2*1e3:.0f} mm  worst {pair2}")
        gates["overlapping arms collide"] = clr2 < 0.0

        # Gradient (used by the planner's null-space avoidance). Evaluate it in
        # the NEAR-CONTACT regime the planner actually operates in -- the
        # all-zeros / mid-lift pose hangs the gripper close to the column, a small
        # positive clearance. (Deep inside a penetration the segment distance pins
        # at 0 and the gradient plateaus; the planner avoids that by keeping
        # clearance >= margin, never letting a waypoint sink that far in.)
        jo = ARM_JOINTS["left"] + ARM_JOINTS["right"] + [LIFT_JOINT]
        qn = {j: 0.0 for j in ARM_JOINTS["left"] + ARM_JOINTS["right"]}
        qn[LIFT_JOINT] = 0.5
        qv = np.array([qn[j] for j in jo])
        g = cm.clearance_gradient(qv, jo, list(range(len(jo))))
        gates["gradient finite"] = bool(np.all(np.isfinite(g)))
        gates["gradient informative near contact"] = float(np.abs(g).max()) > 1e-6

    print("\n----------------  COLLISION GATES  ----------------")
    npass = 0
    for name, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        npass += int(ok)
    print(f"\n{npass}/{len(gates)} gates passed")
    return 0 if npass == len(gates) else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())

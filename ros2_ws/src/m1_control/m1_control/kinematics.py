"""Dependency-free URDF kinematics for the M1 robot.

This module parses a URDF string (no KDL / pinocchio needed) and provides
forward kinematics and a geometric Jacobian for arbitrary base->tip chains.
On top of that it implements a damped-least-squares (DLS) Cartesian reach
controller driven purely from the URDF, so it can run unchanged on the real
robot.

The solver treats each arm's 7 joints plus the single shared prismatic lift as
the actuated DOFs. Unlike a single-step Jacobian nudge, every call iterates a
full damped Gauss-Newton solve (against the URDF model) to the optimal joint
configuration for the requested target(s) -- with adaptive, singularity-aware
damping and multi-seed restarts to avoid local minima -- then leads the
measured pose toward that goal by a bounded step. On reachable targets this
drives the gripper to sub-millimetre error; on an unreachable one the joints
saturate at the closest configuration the limits allow.

When both arms reach at once they are solved together in one stacked system, so
the shared lift column is resolved as the least-squares compromise that best
serves both grippers (instead of the two arms fighting over the lift).

The solver distinguishes a *cold* target (first solve, a changed arm set, or a
big jump) from *tracking* (a teleop bridge nudging the goal a little each tick).
Only a cold target runs the heavy multi-seed search -- and even then it breaks
near-ties toward the pose closest to where the arm already is, so it never
teleports to a far IK branch. While tracking, the cached goal is an excellent
warm start, so each tick just refines a few in-branch iterations: smooth (no
random elbow/base flips), cheap, and decoupled (nudging one arm's target leaves
the other arm's solution put).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np


# --- Cartesian reach tuning ------------------------------------------------
# The reach is solved as a full damped-least-squares (Gauss-Newton) IK: every
# control tick we iterate the URDF model to the joint configuration that best
# reaches the target(s), then lead the *measured* pose toward that solution by a
# bounded step. Iterating to convergence (instead of taking a single linear
# nudge) means each command is anchored to the genuine optimum, so the solver
# reliably drives reachable targets to sub-millimetre error and settles an
# unreachable one at the closest configuration the joints allow.
#
# Damping is applied adaptively: it is zero while the Jacobian is well
# conditioned (exact, fast, unbiased tracking) and grows only as a singularity
# is approached, trading a little accuracy there for stability instead of
# over-damping everywhere like a fixed term would.
IK_SV_EPS = 0.04          # smallest singular value below which damping turns on
IK_DAMPING_MAX = 0.06     # peak DLS damping injected right at a singularity
IK_NULL_GAIN = 0.04       # null-space pull toward mid-range (posture quality)
IK_MAX_DQ = 0.22          # max joint motion (rad) the command leads per tick
IK_POS_TOL = 0.001        # internal-solve convergence tolerance (m)
IK_CMD_DEADBAND = 1e-4    # hold the command still once the solved step (rad) is tiny

# Internal Gauss-Newton iteration controls.
_IK_MAX_ITERS = 80        # max iterations for the primary solve
_IK_PROBE_ITERS = 40      # max iterations for each restart-seed probe
_IK_TRACK_ITERS = 24      # max iterations when refining a warm-started track
_IK_INT_MAX_DQ = 0.40     # cap on a single internal iteration's joint step (rad)
_IK_STEP_TOL = 1e-6       # stop iterating once the internal step is this small
_IK_RESTART_TOL = 0.005   # residual (m) above which we try alternate seeds
_IK_RANK_TOL = 1e-6       # singular values below this are treated as zero
_IK_TARGET_EPS = 1e-4     # target move (m) under which a cached solve is reused

# Continuous-tracking gate. A teleop bridge (Quest/web/keyboard) nudges the
# target a little every tick, so a move smaller than this is treated as
# *tracking*: we warm-start from the cached goal and refine in-branch instead of
# launching the global multi-seed restart search. That keeps the arm locked to
# the same elbow/shoulder solution (smooth, no random snaps) and cheap, while a
# genuinely new/cold target (a bigger jump) still gets the full search below.
_IK_TRACK_JUMP = 0.06     # target move (m) at/under which we stay in-branch
# When the cold search compares restart seeds, two solutions whose residuals are
# within this band are considered equally good, so we break the tie by joint-
# space proximity to the current pose -- never snapping to a far IK branch just
# because it shaves a fraction of a millimetre off an already-tiny residual.
_IK_CONTINUITY_BAND = 0.002

# End-effector link + fingertip offset (ee-local frame), from teleop.py.
EE_LINK_NAME = {
    "left": "openarm_left_ee_base_link",
    "right": "openarm_right_ee_base_link",
}
GRIPPER_TIP_OFFSET = np.array([0.0, 0.0, -0.145], dtype=np.float64)
ARM_JOINTS = {
    "left": [f"openarm_left_joint{i}" for i in range(1, 8)],
    "right": [f"openarm_right_joint{i}" for i in range(1, 8)],
}
LIFT_JOINT = "lift_joint"
BASE_LINK = "base_link"


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix of ``angle`` rad about a (normalized) ``axis``."""
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def _homogeneous(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T


@dataclass
class Joint:
    name: str
    jtype: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    lower: float = -math.pi
    upper: float = math.pi

    @property
    def origin_matrix(self) -> np.ndarray:
        return _homogeneous(_rpy_to_matrix(*self.origin_rpy), self.origin_xyz)

    @property
    def actuated(self) -> bool:
        return self.jtype in ("revolute", "prismatic", "continuous")


@dataclass
class UrdfModel:
    joints: dict = field(default_factory=dict)        # name -> Joint
    child_to_joint: dict = field(default_factory=dict)  # child link -> joint name

    @classmethod
    def from_string(cls, urdf_xml: str) -> "UrdfModel":
        root = ET.fromstring(urdf_xml)
        model = cls()
        for je in root.findall("joint"):
            name = je.attrib["name"]
            jtype = je.attrib.get("type", "fixed")
            parent = je.find("parent").attrib["link"]
            child = je.find("child").attrib["link"]

            origin = je.find("origin")
            xyz = np.zeros(3)
            rpy = np.zeros(3)
            if origin is not None:
                if "xyz" in origin.attrib:
                    xyz = np.array([float(v) for v in origin.attrib["xyz"].split()])
                if "rpy" in origin.attrib:
                    rpy = np.array([float(v) for v in origin.attrib["rpy"].split()])

            axis = np.array([1.0, 0.0, 0.0])
            ae = je.find("axis")
            if ae is not None and "xyz" in ae.attrib:
                axis = np.array([float(v) for v in ae.attrib["xyz"].split()])
                n = np.linalg.norm(axis)
                if n > 1e-9:
                    axis = axis / n

            lower, upper = -math.pi, math.pi
            le = je.find("limit")
            if le is not None:
                lower = float(le.attrib.get("lower", -math.pi))
                upper = float(le.attrib.get("upper", math.pi))

            model.joints[name] = Joint(
                name, jtype, parent, child, xyz, rpy, axis, lower, upper
            )
            model.child_to_joint[child] = name
        return model

    def chain(self, base: str, tip: str) -> list:
        """Ordered list of joint names along the path base -> tip."""
        chain = []
        link = tip
        while link != base:
            jname = self.child_to_joint.get(link)
            if jname is None:
                raise ValueError(f"No path from {base} to {tip}: stuck at {link}")
            chain.append(jname)
            link = self.joints[jname].parent
        chain.reverse()
        return chain


class ArmChain:
    """Forward kinematics + Jacobian for one base->ee chain (lift + 7 joints)."""

    def __init__(self, model: UrdfModel, arm: str):
        self.model = model
        self.arm = arm
        self.tip_link = EE_LINK_NAME[arm]
        self.chain = model.chain(BASE_LINK, self.tip_link)
        # Actuated joints in chain order (lift first, then the 7 arm joints).
        self.actuated = [j for j in self.chain if model.joints[j].actuated]

    def fk(self, q: dict):
        """Return (tip_pos, columns) where columns maps joint -> (axis_w, p_w, type).

        ``tip_pos`` is the world position of the gripper fingertip (link origin
        plus the rigid tip offset rotated into world).
        """
        T = np.eye(4, dtype=np.float64)
        cols = {}
        for jname in self.chain:
            joint = self.model.joints[jname]
            T = T @ joint.origin_matrix
            if joint.actuated:
                qj = float(q.get(jname, 0.0))
                axis_w = T[:3, :3] @ joint.axis
                p_w = T[:3, 3].copy()
                cols[jname] = (axis_w, p_w, joint.jtype)
                if joint.jtype == "prismatic":
                    T = T @ _homogeneous(np.eye(3), joint.axis * qj)
                else:  # revolute / continuous
                    T = T @ _homogeneous(_axis_rotation(joint.axis, qj), np.zeros(3))
        tip_pos = T[:3, 3] + T[:3, :3] @ GRIPPER_TIP_OFFSET
        return tip_pos, cols

    def link_points(self, q: dict):
        """Ordered world points tracing the chain base->tip for visualization.

        Returns a list of 3D positions: the origin of every joint frame along
        the chain (base_link, the lift, then each arm joint) followed by the
        gripper fingertip. Straight segments between consecutive points give a
        compact wireframe ("skeleton") of the arm at the supplied joint config,
        enough to see the pose and whether the fingertip reaches the target.
        """
        T = np.eye(4, dtype=np.float64)
        pts = [T[:3, 3].copy()]
        for jname in self.chain:
            joint = self.model.joints[jname]
            T = T @ joint.origin_matrix
            pts.append(T[:3, 3].copy())
            if joint.actuated:
                qj = float(q.get(jname, 0.0))
                if joint.jtype == "prismatic":
                    T = T @ _homogeneous(np.eye(3), joint.axis * qj)
                else:  # revolute / continuous
                    T = T @ _homogeneous(_axis_rotation(joint.axis, qj), np.zeros(3))
        tip_pos = T[:3, 3] + T[:3, :3] @ GRIPPER_TIP_OFFSET
        pts.append(tip_pos)
        return pts

    def position_jacobian(self, q: dict, joint_order: list):
        """3 x len(joint_order) linear Jacobian of the fingertip wrt joints.

        Joints in ``joint_order`` not part of this chain get zero columns (so a
        single stacked system can mix both arms + the shared lift).
        """
        tip_pos, cols = self.fk(q)
        J = np.zeros((3, len(joint_order)), dtype=np.float64)
        for k, jname in enumerate(joint_order):
            if jname not in cols:
                continue
            axis_w, p_w, jtype = cols[jname]
            if jtype == "prismatic":
                J[:, k] = axis_w
            else:
                J[:, k] = np.cross(axis_w, tip_pos - p_w)
        return tip_pos, J


class ReachController:
    """Converged DLS Cartesian reach for one or both arms + shared lift.

    Each :meth:`solve_step` runs a full damped Gauss-Newton IK (iterated to
    convergence against the URDF model, with adaptive damping and multi-seed
    restarts) to find the optimal joint configuration for the requested
    target(s), then leads the measured pose toward it by a bounded step. When
    both arms reach at once they are solved in one stacked system, so the shared
    lift column is resolved as the least-squares compromise that best serves
    both grippers.
    """

    def __init__(self, model: UrdfModel):
        self.model = model
        self.chains = {arm: ArmChain(model, arm) for arm in ("left", "right")}
        self._restart_rng = np.random.default_rng(0xC0FFEE)
        # Cache of the last fully-solved goal. The optimal joint configuration
        # depends only on the (fixed) target, not on where the arm currently is,
        # so while the target holds we reuse the solution and skip the heavy
        # iterate-and-restart search -- each tick then costs just a bounded step.
        self._cache = None

    def fingertip(self, arm: str, q: dict) -> np.ndarray:
        return self.chains[arm].fk(q)[0]

    # --- internal solve helpers -------------------------------------------
    def _bounds(self, joint_order: list):
        lo = np.array([self.model.joints[j].lower for j in joint_order], dtype=np.float64)
        hi = np.array([self.model.joints[j].upper for j in joint_order], dtype=np.float64)
        return lo, hi

    def _stack(self, q_vec, joint_order, arms, targets):
        """Stacked fingertip Jacobian + error for the current joint vector."""
        m = 3 * len(arms)
        J = np.zeros((m, len(joint_order)), dtype=np.float64)
        e = np.zeros(m, dtype=np.float64)
        dist = {}
        q = {jn: float(q_vec[k]) for k, jn in enumerate(joint_order)}
        for ai, a in enumerate(arms):
            tip_pos, Ja = self.chains[a].position_jacobian(q, joint_order)
            ev = np.asarray(targets[a], dtype=np.float64) - tip_pos
            dist[a] = float(np.linalg.norm(ev))
            J[3 * ai:3 * ai + 3, :] = Ja
            e[3 * ai:3 * ai + 3] = ev
        return J, e, dist

    @staticmethod
    def _dls(J, e):
        """Adaptively-damped least-squares step plus the null-space projector.

        Damping is zero while the smallest singular value stays above
        ``IK_SV_EPS`` (so the step is the exact, unbiased pseudo-inverse) and
        ramps to ``IK_DAMPING_MAX`` as that value collapses toward a
        singularity. Returns ``(dq, N)`` where ``N`` projects a secondary
        objective onto the task null space.
        """
        U, s, Vt = np.linalg.svd(J, full_matrices=False)
        s_min = s[-1] if s.size else 0.0
        if s_min >= IK_SV_EPS:
            lam2 = 0.0
        else:
            lam2 = (1.0 - (s_min / IK_SV_EPS) ** 2) * (IK_DAMPING_MAX ** 2)
        d = s / (s * s + lam2)
        dq = Vt.T @ (d * (U.T @ e))
        rank = s > _IK_RANK_TOL
        Vr = Vt[rank]
        N = np.eye(J.shape[1], dtype=np.float64) - Vr.T @ Vr
        return dq, N

    def _solve_from(self, seed, joint_order, arms, targets, lo, hi,
                    null_target, null_gain, max_iters=_IK_MAX_ITERS):
        """Iterate damped Gauss-Newton from ``seed`` to convergence.

        Joint limits are enforced by clamping every iterate, so an unreachable
        target naturally settles at the closest configuration the joints allow.
        The secondary (null-space) objective pulls each DOF toward
        ``null_target`` with per-DOF weight ``null_gain``; callers use this to
        keep the redundant DOFs well-behaved (arms toward mid-range on a cold
        solve, or the whole config toward the previous goal while tracking, so
        the shared lift cannot drift into a local minimum it can't escape).
        """
        q = np.clip(np.asarray(seed, dtype=np.float64), lo, hi)
        dist = {}
        for _ in range(max_iters):
            J, e, dist = self._stack(q, joint_order, arms, targets)
            if max(dist.values()) < IK_POS_TOL:
                break
            dq_task, N = self._dls(J, e)
            dq_null = null_gain * (null_target - q)
            dq = dq_task + N @ dq_null
            nrm = float(np.linalg.norm(dq))
            if nrm > _IK_INT_MAX_DQ:
                dq *= _IK_INT_MAX_DQ / nrm
            q = np.clip(q + dq, lo, hi)
            if nrm < _IK_STEP_TOL:
                _, _, dist = self._stack(q, joint_order, arms, targets)
                break
        return q, dist

    @staticmethod
    def _better(res_try, ref_try, res_best, ref_best):
        """Is the candidate a better cold-solve pick than the incumbent?

        Primary key is the Cartesian residual; but when two candidates reach
        within ``_IK_CONTINUITY_BAND`` of each other we treat them as equally
        good and prefer the one closest (joint space) to the current pose. This
        is what stops the search from snapping to a distant IK branch merely to
        trim a sub-millimetre off an already-converged residual.
        """
        if res_try < res_best - _IK_CONTINUITY_BAND:
            return True
        if res_try <= res_best + _IK_CONTINUITY_BAND:
            return ref_try < ref_best
        return False

    def _restart_seeds(self, lo, hi):
        """Diverse seeds used to escape a local minimum / poor start pose.

        Sweeping the shared lift (last entry) over its range with the arms at
        mid-range covers the dominant reachability factor (target height); a
        couple of random arm postures add coverage for awkward orientations.
        These only run on a target change whose primary solve fell short, so the
        list is kept short to bound the worst-case re-solve latency.
        """
        mid = 0.5 * (lo + hi)
        seeds = []
        for frac in (0.0, 0.35, 0.7, 1.0):
            s = mid.copy()
            s[-1] = lo[-1] + frac * (hi[-1] - lo[-1])
            seeds.append(s)
        for _ in range(2):
            seeds.append(lo + self._restart_rng.random(lo.shape[0]) * (hi - lo))
        return seeds

    def solve_step(self, q_meas: dict, targets: dict) -> dict:
        """Drive the command one bounded step toward the optimal reach solution.

        ``targets`` maps arm -> 3D world point (base frame). Returns a dict of
        joint name -> new commanded position plus ``"_dist"`` (per-arm residual
        of the solved configuration). ``q_meas`` is the measured joint dict.

        Two regimes share one code path:

        * **Tracking** -- the same arms are active and the target moved only a
          little (an operator bridge nudging the goal each tick): we warm-start
          from the cached goal and refine *in branch*, never launching the
          global restart search. This keeps teleop smooth (no random elbow/base
          flips) and cheap, and it isolates the arms -- nudging one arm's target
          leaves the other's solution where it was.
        * **Cold** -- first solve, the active arm set changed, or the target
          jumped far: run the full multi-seed search, but choose among seeds by
          residual *with a proximity tie-break*, so a distant IK branch is taken
          only when it genuinely reaches better, not to shave off a sub-mm.
        """
        arms = [a for a in ("left", "right") if targets.get(a) is not None]
        if not arms:
            return {}

        # Joint variable vector: each arm's 7 joints, then the shared lift last.
        joint_order = []
        for a in arms:
            joint_order += ARM_JOINTS[a]
        if LIFT_JOINT not in joint_order:
            joint_order.append(LIFT_JOINT)
        lo, hi = self._bounds(joint_order)
        mid_vec = 0.5 * (lo + hi)
        lift_idx = joint_order.index(LIFT_JOINT)
        # Cold solve: regularize the arm joints toward mid-range but leave the
        # lift free, so an extreme (e.g. very high) target can drive the lift to
        # its limit -- this is what keeps every reachable target solvable.
        cold_gain = np.full(len(joint_order), IK_NULL_GAIN, dtype=np.float64)
        cold_gain[lift_idx] = 0.0
        # Tracking: regularize the whole config (lift included) toward the
        # previous goal, damping redundant drift so the solution stays in-branch.
        track_gain = np.full(len(joint_order), IK_NULL_GAIN, dtype=np.float64)

        q_meas_vec = np.array([q_meas.get(j, 0.0) for j in joint_order], dtype=np.float64)
        tgt_vecs = [np.asarray(targets[a], dtype=np.float64) for a in arms]

        cache = self._cache
        cache_arms_match = cache is not None and cache["arms"] == tuple(arms)
        if cache_arms_match:
            arm_jump = {a: float(np.linalg.norm(c - t))
                        for a, c, t in zip(arms, cache["targets"], tgt_vecs)}
            jump = max(arm_jump.values())
        else:
            arm_jump = {a: float("inf") for a in arms}
            jump = float("inf")

        if cache_arms_match and jump < _IK_TARGET_EPS:
            # Target is steady (operator bridges republish it every tick): reuse
            # the goal we already solved instead of re-running the full IK.
            q_best, dist_best = cache["q_best"], cache["dist"]
        elif cache_arms_match and jump < _IK_TRACK_JUMP:
            # Continuous tracking: the target only nudged, so the previous goal
            # is an excellent warm start. Refine a few in-branch iterations and
            # DO NOT restart -- a global search here is what made the arm snap to
            # a random branch and made one moving arm disturb the other.
            q_best, dist_best = self._solve_from(
                cache["q_best"], joint_order, arms, targets, lo, hi,
                cache["q_best"], track_gain, max_iters=_IK_TRACK_ITERS)
            self._cache = {
                "arms": tuple(arms),
                "targets": tgt_vecs,
                "q_best": q_best,
                "dist": dist_best,
            }
        else:
            # Cold target (first solve / arm-set change / large jump). When the
            # same arms are active we *pin* any arm whose target barely moved to
            # its cached configuration and only re-search the arm(s) that jumped
            # (plus the shared lift). That stops a big move on one arm from
            # flinging the other one onto a different IK branch -- the held arm's
            # joints just compensate for the shared lift instead of teleporting.
            ref = cache["q_best"] if cache_arms_match else q_meas_vec
            null_target = mid_vec.copy()
            null_gain = cold_gain.copy()
            free = np.zeros(len(joint_order), dtype=bool)
            free[lift_idx] = True  # the shared lift is always free to re-search
            for a in arms:
                sl = slice(joint_order.index(ARM_JOINTS[a][0]),
                           joint_order.index(ARM_JOINTS[a][0]) + 7)
                if cache_arms_match and arm_jump[a] < _IK_TRACK_JUMP:
                    # Held arm: keep it on its current branch (pin + regularize
                    # toward the cached goal); it stays out of the restart shuffle.
                    null_target[sl] = cache["q_best"][sl]
                    null_gain[sl] = IK_NULL_GAIN
                else:
                    free[sl] = True

            base_seed = cache["q_best"].copy() if cache_arms_match else q_meas_vec
            q_best, dist_best = self._solve_from(
                base_seed, joint_order, arms, targets, lo, hi,
                null_target, null_gain)
            best_res = max(dist_best.values())
            best_ref = float(np.linalg.norm(q_best - ref))

            # Large residual means a poor basin (e.g. the singular zero pose or a
            # distant new target): search diverse seeds for the free DOFs only.
            # We prefer the lowest residual, but break near-ties toward the
            # configuration closest to the reference, for continuity.
            if best_res > _IK_RESTART_TOL:
                for raw in [q_meas_vec] + self._restart_seeds(lo, hi):
                    seed = base_seed.copy()
                    seed[free] = np.asarray(raw, dtype=np.float64)[free]
                    q_try, dist_try = self._solve_from(
                        seed, joint_order, arms, targets, lo, hi, null_target,
                        null_gain, max_iters=_IK_PROBE_ITERS)
                    res_try = max(dist_try.values())
                    ref_try = float(np.linalg.norm(q_try - ref))
                    if self._better(res_try, ref_try, best_res, best_ref):
                        q_best, dist_best, best_res, best_ref = (
                            q_try, dist_try, res_try, ref_try)
                        if best_res < IK_POS_TOL:
                            break

            self._cache = {
                "arms": tuple(arms),
                "targets": tgt_vecs,
                "q_best": q_best,
                "dist": dist_best,
            }

        # Command stepping: lead the measured pose toward the solved goal by a
        # bounded amount so the stiff drive supplies holding torque without the
        # command overshooting (the same contract the Isaac teleop relied on).
        dq = q_best - q_meas_vec
        nrm = float(np.linalg.norm(dq))
        if nrm < IK_CMD_DEADBAND:
            # Already at the solved configuration: hold to avoid command jitter.
            return {"_dist": dist_best}
        if nrm > IK_MAX_DQ:
            dq *= IK_MAX_DQ / nrm
        q_cmd = np.clip(q_meas_vec + dq, lo, hi)

        out = {jname: float(q_cmd[k]) for k, jname in enumerate(joint_order)}
        out["_dist"] = dist_best
        return out

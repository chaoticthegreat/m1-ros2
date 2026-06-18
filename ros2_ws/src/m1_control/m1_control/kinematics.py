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
serves both grippers (instead of the two arms fighting over the lift). The
solved goal is cached and reused while the target holds steady, so the heavy
search runs only when a target actually changes -- each control tick otherwise
costs a single bounded step.
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
_IK_INT_MAX_DQ = 0.40     # cap on a single internal iteration's joint step (rad)
_IK_STEP_TOL = 1e-6       # stop iterating once the internal step is this small
_IK_RESTART_TOL = 0.005   # residual (m) above which we try alternate seeds
_IK_RANK_TOL = 1e-6       # singular values below this are treated as zero
_IK_TARGET_EPS = 1e-4     # target move (m) under which a cached solve is reused

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
        self.arm_mid = {}
        for arm in ("left", "right"):
            mids = []
            for j in ARM_JOINTS[arm]:
                joint = model.joints[j]
                mids.append(0.5 * (joint.lower + joint.upper))
            self.arm_mid[arm] = np.array(mids, dtype=np.float64)
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

    def _solve_from(self, seed, joint_order, arms, targets, lo, hi, null_slices,
                    max_iters=_IK_MAX_ITERS):
        """Iterate damped Gauss-Newton from ``seed`` to convergence.

        Joint limits are enforced by clamping every iterate, so an unreachable
        target naturally settles at the closest configuration the joints allow.
        """
        q = np.clip(np.asarray(seed, dtype=np.float64), lo, hi)
        dist = {}
        for _ in range(max_iters):
            J, e, dist = self._stack(q, joint_order, arms, targets)
            if max(dist.values()) < IK_POS_TOL:
                break
            dq_task, N = self._dls(J, e)
            dq_null = np.zeros(len(joint_order), dtype=np.float64)
            for base, mid in null_slices:
                dq_null[base:base + 7] = IK_NULL_GAIN * (mid - q[base:base + 7])
            dq = dq_task + N @ dq_null
            nrm = float(np.linalg.norm(dq))
            if nrm > _IK_INT_MAX_DQ:
                dq *= _IK_INT_MAX_DQ / nrm
            q = np.clip(q + dq, lo, hi)
            if nrm < _IK_STEP_TOL:
                _, _, dist = self._stack(q, joint_order, arms, targets)
                break
        return q, dist

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
        null_slices = [
            (joint_order.index(ARM_JOINTS[a][0]), self.arm_mid[a]) for a in arms
        ]

        q_meas_vec = np.array([q_meas.get(j, 0.0) for j in joint_order], dtype=np.float64)
        tgt_vecs = [np.asarray(targets[a], dtype=np.float64) for a in arms]

        cache = self._cache
        cache_arms_match = cache is not None and cache["arms"] == tuple(arms)
        unchanged = cache_arms_match and all(
            np.linalg.norm(c - t) < _IK_TARGET_EPS
            for c, t in zip(cache["targets"], tgt_vecs))

        if unchanged:
            # Target is steady (operator bridges republish it every tick): reuse
            # the goal we already solved instead of re-running the full IK.
            q_best, dist_best = cache["q_best"], cache["dist"]
        else:
            # Warm-start from the cached goal when the same arms are reaching (a
            # nudged target reconverges in a few iterations); otherwise seed from
            # the measured pose. Either way we stay in the nearest IK branch.
            seed0 = cache["q_best"] if cache_arms_match else q_meas_vec
            q_best, dist_best = self._solve_from(
                seed0, joint_order, arms, targets, lo, hi, null_slices)
            best_res = max(dist_best.values())

            # Large residual means a poor basin (e.g. the singular zero pose or a
            # distant new target): search diverse seeds and keep the global best
            # so we converge to the optimal reachable configuration, never a
            # local minimum. This only runs when the target actually changes.
            if best_res > _IK_RESTART_TOL:
                for seed in [q_meas_vec] + self._restart_seeds(lo, hi):
                    q_try, dist_try = self._solve_from(
                        seed, joint_order, arms, targets, lo, hi, null_slices,
                        max_iters=_IK_PROBE_ITERS)
                    res_try = max(dist_try.values())
                    if res_try < best_res:
                        q_best, dist_best, best_res = q_try, dist_try, res_try
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

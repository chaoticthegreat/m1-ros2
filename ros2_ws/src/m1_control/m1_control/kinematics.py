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

The cold multi-seed search is *amortized* across control ticks: each tick spends
a bounded iteration budget and carries the unfinished primary/probe state in the
cache, advancing it on the following ticks while the command already leads toward
the best-so-far goal. So the worst-case ``solve_step`` stays well inside the
60 Hz budget (~7 ms, vs ~120 ms when the whole search ran in one tick) even as
the *total* search is more thorough than before -- which is what lifts hard,
near-workspace-boundary targets to sub-mm. Finally, every tick applies a small
per-arm Cartesian "hold" correction to the command (each arm's own joints,
shared lift fixed), so the two arms are decoupled through the lift: a held arm
keeps its fingertip planted while the lift slews to serve the other arm.
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
# When one arm holds a target while the other makes a big (cold) jump, the shared
# lift would otherwise swing across to serve the jumping arm and drag the held
# gripper with it (the held arm's joints can't recompensate the lift travel
# instantly, so its fingertip rides the lift through the transition). Anchoring
# the lift toward its cached height with this gain keeps that swing small, so the
# held arm barely moves; the jumping arm just leans more on its own 7 joints.
IK_LIFT_HOLD_GAIN = 0.12
IK_MAX_DQ = 0.22          # max joint motion (rad) the command leads per tick
# Per-arm Cartesian hold correction applied to the *command* each tick: with the
# shared lift fixed, a small capped damped step on each arm's own 7 joints pulls
# its gripper onto its target. This decouples the two arms through the shared
# lift -- a held arm keeps its fingertip planted while the lift slews to serve the
# other arm, instead of riding the lift through the transition. Capped small so it
# only trims the Cartesian error and never overrides the global IK branch choice.
_IK_ARM_HOLD_CAP = 0.06   # max joint motion (rad) of the per-arm hold correction
IK_POS_TOL = 0.001        # internal-solve convergence tolerance (m)
IK_CMD_DEADBAND = 1e-4    # hold the command still once the solved step (rad) is tiny

# --- Reach is position-only ------------------------------------------------
# The solver drives each gripper's fingertip to a target *point*; gripper
# orientation is NOT part of the task. A target is a 3-vector (or a dict
# carrying ``"pos"``); any orientation a caller still sends (a legacy pose
# dict's ``"quat"``/``"R"``) is ignored. The FK/Jacobian utilities below still
# expose orientation (``pose_jacobian``, ``gripper_pose``, ``_so3_log``,
# ``mat_to_quat``) because the visualization and the test harnesses use them to
# *report* the gripper's rotation -- but the reach solve never constrains it.

# Internal Gauss-Newton iteration controls.
_IK_MAX_ITERS = 80        # max iterations for the primary solve
_IK_PROBE_ITERS = 40      # max iterations for each restart-seed probe
# Max iterations refining a warm-started track. The line-search descent converges a
# warm (small-move) track in a couple of iterations, so this is mostly a worst-case
# cap for the catch-up right after a cold solve; kept modest because a dual 6-DOF
# track runs a 12x15 SVD per iteration, and burning the full cap there is what
# pushed the rare catch-up tick over the 60 Hz budget. The command leads toward the
# goal each tick, so a slightly-short refine just finishes on the next tick.
_IK_TRACK_ITERS = 12      # max iterations when refining a warm-started track
_IK_INT_MAX_DQ = 0.40     # cap on a single internal iteration's joint step (rad)
_IK_STEP_TOL = 1e-6       # stop iterating once the internal step is this small
_IK_RESTART_TOL = 0.005   # residual (m) above which we try alternate seeds
_IK_RANK_TOL = 1e-6       # singular values below this are treated as zero
_IK_TARGET_EPS = 1e-4     # target move (m) under which a cached solve is reused
# Backtracking line search inside each GN iteration: shrink the step until the
# task cost strictly decreases, so a step never lands in a worse configuration.
# This is what lets the in-branch solve ride a wrist singularity through instead
# of overshooting and getting stuck (the reported "can't keep tracking").
_IK_LS_MAX = 6            # max step halvings before declaring a local minimum
_IK_LS_SHRINK = 0.4       # step shrink factor per backtrack (0.4^5 ~ 1% floor)

# Posture (null-space) regularization toward a reference pose helps the redundant
# resolution converge cleanly on well-conditioned solves (notably the dual-arm
# shared-lift compromise) -- so the *primary* solve keeps it. But it drags an arm
# reaching for an *extreme* (near workspace-boundary, e.g. very high) target short
# of the goal: with the lift clamped at its limit the null space can no longer
# hold the gripper, so the pull toward mid-range leaks into the task and the arm
# settles ~18-22 mm short. Those targets are exactly the ones whose primary solve
# overshoots the restart tolerance, so the restart probes run as *pure task* (no
# posture pull) -- which reaches them sub-mm -- and the residual comparison keeps
# whichever is best. Normal/dual targets converge in the primary (posture kept);
# only a genuinely hard target falls through to the pure-task restart.

# Stall detection for the COLD multi-seed search (the fixed-step `_solve_from`
# path): a seed that stops making progress (a poor basin or an unreachable target
# at its closest config) bails instead of burning the whole iteration budget --
# this is what bounds the cold-solve worst-case latency.
_IK_STALL_ITERS = 6       # consecutive non-improving iters before bailing a solve
# Only a *flat/worsening* residual counts as a stall -- near a singularity a solve
# converges slowly but steadily, and bailing that slow progress leaves extreme
# targets short. So the bar for "progress" is tiny: anything still decreasing is
# kept; only a genuine plateau (an unreachable target at its closest config) bails.
_IK_STALL_REL = 1e-5      # relative residual improvement that still counts as progress

# Cold-solve latency bound (amortized restart). The multi-seed restart search is
# the only thing that ever blew the 60 Hz budget (~120 ms when it ran every seed
# in one tick). We instead spend at most this many internal iterations per tick
# and carry the unfinished restart seeds in the cache, advancing them on the
# following (still-steady) ticks -- the command meanwhile leads toward the
# best-so-far goal, so the arm starts moving immediately and the goal only
# sharpens over the next few ms. This makes every tick bounded while the *total*
# search done is actually larger (more seeds), which also lifts hard/extreme
# targets the old one-shot search left short. (The cold search uses the cheap
# fixed-step `_solve_from`, not the line-search path, so this budget is unchanged
# from before -- only warm tracking pays for the line search, and it is so close to
# its solution that it converges in a couple of cheap iterations.)
_IK_COLD_BUDGET = 18      # max internal GN iterations spent on a cold search per tick

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

# Tracking-loss recovery (re-acquire). While *tracking* (small per-tick target
# move) the solver only refines the cached solution in-branch and never restarts
# -- smooth, but it means a target that slowly walks the gripper into a wrist
# singularity, a needed elbow/branch flip, or out past the workspace and back can
# leave the in-branch solve stuck at a configuration it cannot iterate out of: the
# solved residual blows up and never recovers, because the small per-tick move
# never trips the cold path. (This is the operator-reported "it gets stuck and
# can't keep tracking the target".) So when the in-branch *solved* residual for an
# arm exceeds this, we treat its target as lost and kick off the same amortized
# multi-seed restart the cold path uses -- freeing only the stuck arm(s), pinning
# the healthy one(s) -- to re-acquire it. Set well above a healthy track (whose
# solved residual is sub-mm) so normal tracking never triggers it; the proximity
# tie-break plus the bounded per-tick command step keep the re-acquisition a
# smooth slew rather than a snap.
#
# A genuine loss -- stuck at a joint limit / wrong branch, or recovering from a
# boundary saturation -- shows up as a large *position* residual that persists. A
# brief workspace/wrist singularity crossing keeps the point reachable (small
# residual) and recovers in-branch, so gating on a sustained position residual
# re-acquires a true loss without snapping the arm to another branch for a
# transient.
_IK_REACQUIRE_POS = 0.02            # solved position residual (m) -> lost the point
# ...and only once it has *persisted* this many consecutive ticks, so even a long
# singularity crossing recovers in-branch before a global re-acquire is paid for.
# Kept short: firing sooner (while the stuck arm is only mildly off-branch) makes
# the re-acquired branch a CLOSER configuration, so the slew onto it is smaller and
# settles faster than letting the arm wind further into the boundary singularity
# first. As a bonus this also stops a genuinely-unreachable target from re-searching
# every tick (after a failed re-acquire the stuck counter restarts from zero).
_IK_REACQUIRE_TICKS = 3   # consecutive stuck ticks before a re-acquire fires

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


def mat_to_quat(R: np.ndarray) -> list:
    """Rotation matrix (3x3) -> unit quaternion ``[x, y, z, w]``."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return [x, y, z, w]


def quat_to_mat(q) -> np.ndarray:
    """Unit quaternion ``[x, y, z, w]`` -> 3x3 rotation matrix (inverse of
    :func:`mat_to_quat`). Normalizes defensively; a zero quaternion yields I."""
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle) in the world frame.

    This is the orientation error fed to the solver: ``_so3_log(R_target @
    R_current.T)`` is the world-frame angular displacement that rotates the
    current gripper orientation onto the target, matching the world-frame
    angular rows of the geometric Jacobian. Robust near 0 and pi.
    """
    c = (R[0, 0] + R[1, 1] + R[2, 2] - 1.0) * 0.5
    c = max(-1.0, min(1.0, c))
    angle = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
                 dtype=np.float64)
    if angle < 1e-6:
        return 0.5 * v  # small-angle: vee of the skew part
    if angle > math.pi - 1e-3:
        # Near pi sin(angle)->0 makes v unreliable; recover the axis from the
        # symmetric part (R+I)/2 = axis axis^T, pick its best-conditioned column.
        A = 0.5 * (R + np.eye(3, dtype=np.float64))
        i = int(np.argmax(np.diag(A)))
        axis = A[:, i] / math.sqrt(max(A[i, i], 1e-12))
        nrm = np.linalg.norm(axis)
        axis = axis / nrm if nrm > 1e-9 else np.array([1.0, 0.0, 0.0])
        # Sign at exactly pi is ambiguous; v carries the residual sign hint.
        if float(axis @ v) < 0:
            axis = -axis
        return angle * axis
    return (angle / (2.0 * math.sin(angle))) * v


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

    def link_transforms(self, q: dict) -> dict:
        """World 4x4 transform of every link given joint values ``q``.

        Walks the kinematic tree from the root link(s), applying each joint's
        fixed origin then its actuated motion. Used to pose every link mesh in
        the headset viz from a single measured joint state. Links whose joints
        aren't in ``q`` default to 0 (e.g. static base frames).
        """
        children: dict = {}
        for jn, j in self.joints.items():
            children.setdefault(j.parent, []).append(jn)
        child_links = {j.child for j in self.joints.values()}
        all_links = set(children) | child_links
        roots = [lk for lk in all_links if lk not in child_links]

        transforms = {lk: np.eye(4, dtype=np.float64) for lk in roots}
        stack = list(roots)
        while stack:
            link = stack.pop()
            T_parent = transforms[link]
            for jn in children.get(link, []):
                joint = self.joints[jn]
                T = T_parent @ joint.origin_matrix
                if joint.actuated:
                    qj = float(q.get(jn, 0.0))
                    if joint.jtype == "prismatic":
                        T = T @ _homogeneous(np.eye(3), joint.axis * qj)
                    else:  # revolute / continuous
                        T = T @ _homogeneous(_axis_rotation(joint.axis, qj), np.zeros(3))
                transforms[joint.child] = T
                stack.append(joint.child)
        return transforms

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

    def fk_pose(self, q: dict):
        """Return (tip_pos, R_tip, columns).

        ``tip_pos`` is the world position of the gripper fingertip (ee link
        origin plus the rigid tip offset rotated into world); ``R_tip`` is the
        gripper orientation as a 3x3 world rotation (the tip offset is a pure
        translation, so the gripper frame's orientation is the ee link's).
        ``columns`` maps each actuated joint -> (axis_w, p_w, type) for the
        geometric Jacobian.
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
        R_tip = T[:3, :3].copy()
        tip_pos = T[:3, 3] + R_tip @ GRIPPER_TIP_OFFSET
        return tip_pos, R_tip, cols

    def fk(self, q: dict):
        """Return (tip_pos, columns) -- position-only view of :meth:`fk_pose`."""
        tip_pos, _R, cols = self.fk_pose(q)
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

    def pose_jacobian(self, q: dict, joint_order: list):
        """6 x len(joint_order) geometric Jacobian (linear rows over angular).

        Returns ``(tip_pos, R_tip, J)``. Rows 0:3 are the fingertip linear
        Jacobian (identical to :meth:`position_jacobian`); rows 3:6 are the
        angular Jacobian -- a revolute joint contributes its world axis, a
        prismatic joint contributes nothing to orientation. Joints outside this
        chain get zero columns, so both arms + the shared lift stack cleanly.
        """
        tip_pos, R_tip, cols = self.fk_pose(q)
        J = np.zeros((6, len(joint_order)), dtype=np.float64)
        for k, jname in enumerate(joint_order):
            if jname not in cols:
                continue
            axis_w, p_w, jtype = cols[jname]
            if jtype == "prismatic":
                J[:3, k] = axis_w
            else:
                J[:3, k] = np.cross(axis_w, tip_pos - p_w)
                J[3:, k] = axis_w
        return tip_pos, R_tip, J


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

    def gripper_pose(self, arm: str, q: dict):
        """(tip_pos, R_tip) of one arm's gripper -- used to seed a 6-DOF target
        onto the live pose so the arm doesn't jump when orientation turns on."""
        tip_pos, R_tip, _ = self.chains[arm].fk_pose(q)
        return tip_pos, R_tip

    @staticmethod
    def _normalize_targets(targets: dict):
        """Extract the per-arm position target from the public ``targets`` map.

        ``targets[arm]`` is either a 3-vector (the position-only contract) or a
        dict carrying ``"pos"`` (a legacy 6-DOF pose dict is still accepted --
        any ``"quat"``/``"R"`` it carries is ignored, because the reach is
        position-only). Returns ``pos[arm]`` as a 3-vector; arms whose target is
        ``None`` are omitted.
        """
        pos = {}
        for a in ("left", "right"):
            t = targets.get(a)
            if t is None:
                continue
            if isinstance(t, dict):
                pos[a] = np.asarray(t["pos"], dtype=np.float64)
            else:
                pos[a] = np.asarray(t, dtype=np.float64)
        return pos

    # --- internal solve helpers -------------------------------------------
    def _bounds(self, joint_order: list):
        lo = np.array([self.model.joints[j].lower for j in joint_order], dtype=np.float64)
        hi = np.array([self.model.joints[j].upper for j in joint_order], dtype=np.float64)
        return lo, hi

    def _stack(self, q_vec, joint_order, arms, pos):
        """Stacked position Jacobian + error for the current joint vector.

        Each active arm contributes 3 rows: its fingertip linear Jacobian and
        the world-frame position error ``target - tip``. Returns ``(J, e, dist)``
        where ``dist`` is the per-arm position error (m, also used for reach
        colouring). Both arms + the shared lift stack into one system.
        """
        q = {jn: float(q_vec[k]) for k, jn in enumerate(joint_order)}
        Jblocks = []
        eblocks = []
        dist = {}
        for a in arms:
            tip_pos, Ja = self.chains[a].position_jacobian(q, joint_order)
            ep = np.asarray(pos[a], dtype=np.float64) - tip_pos
            dist[a] = float(np.linalg.norm(ep))
            Jblocks.append(Ja)
            eblocks.append(ep)
        J = np.vstack(Jblocks)
        e = np.concatenate(eblocks)
        return J, e, dist

    def _errs(self, q_vec, joint_order, arms, pos):
        """Stacked task error (no Jacobian) for the current joint vector.

        The error-only counterpart of :meth:`_stack`: it runs only forward
        kinematics (no Jacobian assembly), so it is the cheap evaluation the
        iteration's backtracking line search uses to test trial steps. Returns
        ``(e, dist)`` so ``e @ e`` is the identical task cost as :meth:`_stack`.
        """
        q = {jn: float(q_vec[k]) for k, jn in enumerate(joint_order)}
        eblocks = []
        dist = {}
        for a in arms:
            tip_pos, _ = self.chains[a].fk(q)
            ep = np.asarray(pos[a], dtype=np.float64) - tip_pos
            dist[a] = float(np.linalg.norm(ep))
            eblocks.append(ep)
        return np.concatenate(eblocks), dist

    @staticmethod
    def _resid(dist):
        """Scalar solve residual: the worst per-arm position error."""
        return max(dist.values())

    @staticmethod
    def _converged(dist):
        """All arms within position tolerance."""
        return all(d < IK_POS_TOL for d in dist.values())

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
        s_min = float(s[-1]) if s.size else 0.0
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

    def _solve_from(self, seed, joint_order, arms, pos, lo, hi,
                    null_target, null_gain, max_iters=_IK_MAX_ITERS,
                    line_search=True):
        """Iterate damped Gauss-Newton from ``seed`` toward convergence.

        Joint limits are enforced by clamping every iterate, so an unreachable
        target naturally settles at the closest configuration the joints allow.
        The secondary (null-space) objective pulls each DOF toward
        ``null_target`` with per-DOF weight ``null_gain``; callers use this to
        keep the redundant DOFs well-behaved (arms toward mid-range on a cold
        solve, or the whole config toward the previous goal while tracking, so
        the shared lift cannot drift into a local minimum it can't escape).

        Two descent strategies share this entry point, because warm tracking and
        the cold multi-seed search want opposite things near a singularity:

        * ``line_search=True`` (**tracking**) vets every step with a backtracking
          line search -- the damped step (task plus the null-space posture/pinning
          term, which is task-neutral to first order so it rides along without
          spoiling the descent) is shrunk until the task cost ``e^T e`` strictly
          decreases. A step can therefore never overshoot into a *worse*
          configuration, which is exactly how the in-branch solve used to "get
          stuck" at a wrist singularity (a full GN step overshooting); the monotone
          search rides the singularity through instead. If no positive step helps,
          we are at a local minimum and stop.
        * ``line_search=False`` (**cold search**) takes the full damped step each
          iteration and bails on a stall counter. Monotonicity is deliberately NOT
          imposed here: a seed often must step *over* a saddle -- e.g. the
          shared-lift compromise of a partially-unreachable dual target, where one
          arm's joints have to swing the lift the "wrong" way briefly so the held
          arm can recompense it -- which a strict line search would refuse, leaving
          the held arm short. The diverse seeds plus posture pinning supply
          robustness here instead of monotonicity.

        Returns ``(q, dist, iters)`` -- ``dist`` being the per-arm position
        error of the final iterate.
        """
        q = np.clip(np.asarray(seed, dtype=np.float64), lo, hi)
        if line_search:
            e, dist = self._errs(q, joint_order, arms, pos)
            cost = float(e @ e)
            iters = 0
            for _ in range(max_iters):
                iters += 1
                if self._converged(dist):
                    break
                J, _, dist = self._stack(q, joint_order, arms, pos)
                dq_task, N = self._dls(J, e)
                dq = dq_task + N @ (null_gain * (null_target - q))
                nrm = float(np.linalg.norm(dq))
                if nrm < _IK_STEP_TOL:
                    break
                if nrm > _IK_INT_MAX_DQ:
                    dq = dq * (_IK_INT_MAX_DQ / nrm)
                alpha = 1.0
                improved = False
                for _ls in range(_IK_LS_MAX):
                    q_try = np.clip(q + alpha * dq, lo, hi)
                    e2, dist2 = self._errs(q_try, joint_order, arms, pos)
                    cost2 = float(e2 @ e2)
                    if cost2 < cost:
                        q, e, dist, cost = q_try, e2, dist2, cost2
                        improved = True
                        break
                    alpha *= _IK_LS_SHRINK
                if not improved:
                    break
            return q, dist, iters

        dist = {}
        prev_res = float("inf")
        stall = 0
        iters = 0
        for _ in range(max_iters):
            iters += 1
            J, e, dist = self._stack(q, joint_order, arms, pos)
            res = self._resid(dist)
            if self._converged(dist):
                break
            # Bail out of a basin that has stopped improving (bounds latency).
            if res > prev_res * (1.0 - _IK_STALL_REL):
                stall += 1
                if stall >= _IK_STALL_ITERS:
                    break
            else:
                stall = 0
            prev_res = res
            dq_task, N = self._dls(J, e)
            dq = dq_task + N @ (null_gain * (null_target - q))
            nrm = float(np.linalg.norm(dq))
            if nrm > _IK_INT_MAX_DQ:
                dq *= _IK_INT_MAX_DQ / nrm
            q = np.clip(q + dq, lo, hi)
            if nrm < _IK_STEP_TOL:
                _, _, dist = self._stack(q, joint_order, arms, pos)
                break
        return q, dist, iters

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
        mid-range covers the dominant reachability factor (target height); a set
        of random arm postures adds coverage for awkward orientations and extreme
        (near workspace-boundary) targets a mid-range arm can't reach. The search
        is amortized across ticks (see ``_IK_COLD_BUDGET``), so we can afford a
        generous seed list -- the per-tick cost stays bounded regardless.
        """
        mid = 0.5 * (lo + hi)
        seeds = []
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            s = mid.copy()
            s[-1] = lo[-1] + frac * (hi[-1] - lo[-1])
            seeds.append(s)
        for _ in range(7):
            seeds.append(lo + self._restart_rng.random(lo.shape[0]) * (hi - lo))
        return seeds

    def _pump_restart(self, job, joint_order, arms, pos, lo, hi, budget):
        """Advance a pending cold restart search by up to ``budget`` GN iters.

        Finishes the (resumable) primary solve first, then probes the remaining
        diverse seeds -- pure task on the free DOFs -- keeping the best by
        residual with the proximity tie-break. Mutates ``job`` in place and sets
        ``job['done']`` once the seed list is exhausted or the best is already at
        tolerance. Because the work is capped per tick and carried in the cache,
        the heavy multi-seed search never blows the control-loop budget while
        still, in total, searching more thoroughly than the old one-shot pass.
        """
        used = 0

        def consider(q, dist):
            res = self._resid(dist)
            ref = float(np.linalg.norm(q - job["ref"]))
            if self._better(res, ref, job["best_res"], job["best_ref"]):
                job["q_best"], job["dist_best"] = q, dist
                job["best_res"], job["best_ref"] = res, ref

        def best_converged():
            return self._converged(job["dist_best"])

        # Resume the primary solve until it converges, stalls, or hits the cap.
        if not job["primary_done"]:
            b = min(budget - used, _IK_MAX_ITERS - job["primary_iters"])
            if b > 0:
                q, dist, it = self._solve_from(
                    job["q_primary"], joint_order, arms, pos, lo, hi,
                    job["null_target"], job["null_gain"], max_iters=b,
                    line_search=False)
                job["q_primary"], job["primary_iters"] = q, job["primary_iters"] + it
                used += it
                consider(q, dist)
                # Done if it converged, stalled early (it < b), or hit the cap.
                if (best_converged() or it < b
                        or job["primary_iters"] >= _IK_MAX_ITERS):
                    job["primary_done"] = True
                # A good-enough primary needs no restart probes at all.
                if best_converged() or job["best_res"] <= _IK_RESTART_TOL:
                    job["seeds"] = []

        # Probe the remaining seeds within the leftover budget. Each probe is
        # *resumable*: it advances by at most the per-tick budget and continues on
        # the next tick(s) until it converges/stalls, so a probe is never cut
        # mid-convergence (which would waste a seed) yet no single tick runs more
        # than ``budget`` iterations.
        while (job["primary_done"] and used < budget
               and not best_converged()):
            if job["probe_q"] is None:
                if not job["seeds"]:
                    break
                raw = np.asarray(job["seeds"].pop(0), dtype=np.float64)
                pq = job["base_seed"].copy()
                pq[job["free"]] = raw[job["free"]]
                job["probe_q"], job["probe_iters"] = pq, 0
            b = min(budget - used, _IK_PROBE_ITERS - job["probe_iters"])
            if b <= 0:
                break
            q, dist, it = self._solve_from(
                job["probe_q"], joint_order, arms, pos, lo, hi,
                job["null_target"], job["probe_gain"], max_iters=b,
                line_search=False)
            job["probe_q"], job["probe_iters"] = q, job["probe_iters"] + it
            used += it
            consider(q, dist)
            # Probe finished if it converged, stalled (it < b), or hit its cap.
            if (self._converged(dist) or it < b
                    or job["probe_iters"] >= _IK_PROBE_ITERS):
                job["probe_q"] = None

        job["done"] = job["primary_done"] and job["probe_q"] is None and (
            not job["seeds"] or best_converged())
        return job["q_best"], job["dist_best"]

    def _build_restart_job(self, joint_order, arms, free_arms, base_seed, ref,
                           pinned_q, lo, hi, q_meas_vec, mid_vec, lift_idx,
                           cold_gain):
        """Assemble a resumable multi-seed restart ``job`` for :meth:`_pump_restart`.

        ``free_arms`` is the set of arms to re-search (their 7 joints, plus the
        always-free shared lift); every other active arm is *pinned* to its
        ``pinned_q`` configuration (posture-regularized onto its current branch)
        and the lift is anchored toward ``pinned_q``'s height so it can't swing
        across and drag a held gripper. This one builder serves both callers:
        the **cold** path frees the arm(s) that jumped, and **tracking-loss
        recovery** frees the arm(s) whose in-branch solve got stuck -- identical
        machinery, only the choice of which arms are free differs.
        """
        null_target = mid_vec.copy()
        null_gain = cold_gain.copy()
        free = np.zeros(len(joint_order), dtype=bool)
        free[lift_idx] = True  # the shared lift is always free to re-search
        held_exists = False
        for a in arms:
            i0 = joint_order.index(ARM_JOINTS[a][0])
            sl = slice(i0, i0 + 7)
            if a in free_arms:
                free[sl] = True
            else:
                # Held arm: keep it on its current branch (pin + regularize toward
                # its solved goal); it stays out of the restart shuffle.
                null_target[sl] = pinned_q[sl]
                null_gain[sl] = IK_NULL_GAIN
                held_exists = True
        # Anchor the shared lift toward the held height (see IK_LIFT_HOLD_GAIN) so
        # it doesn't swing across to serve a re-searched arm and drag a held one.
        if held_exists:
            null_target[lift_idx] = pinned_q[lift_idx]
            null_gain[lift_idx] = IK_LIFT_HOLD_GAIN
        # The restart probes drop the posture pull on the *free* (re-searched) DOFs
        # -- pure task -- so an extreme near-boundary target the posture-regularized
        # primary settled short of is reached sub-mm; held arms keep their pinning
        # gain (and the lift anchor above). Amortized across ticks (_pump_restart).
        probe_gain = null_gain.copy()
        probe_gain[free] = 0.0
        if held_exists:
            probe_gain[lift_idx] = IK_LIFT_HOLD_GAIN
        return {
            "base_seed": base_seed, "q_primary": base_seed.copy(),
            "primary_iters": 0, "primary_done": False,
            "null_target": null_target, "null_gain": null_gain,
            "probe_gain": probe_gain, "free": free, "ref": ref,
            "seeds": [q_meas_vec.copy()] + self._restart_seeds(lo, hi),
            "probe_q": None, "probe_iters": 0,  # resumable in-flight probe
            "q_best": base_seed.copy(),
            "dist_best": {a: float("inf") for a in arms},
            "best_res": float("inf"), "best_ref": float("inf"), "done": False,
        }

    def solve_step(self, q_meas: dict, targets: dict) -> dict:
        """Drive the command one bounded step toward the optimal reach solution.

        ``targets`` maps arm -> a target *point* (base frame): a 3-vector, or a
        ``dict`` carrying ``"pos"`` (any ``"quat"``/``"R"`` it also carries is
        ignored -- the reach is position-only). Returns a dict of joint name ->
        new commanded position plus ``"_dist"`` (per-arm position residual of the
        solved configuration). ``q_meas`` is the measured joint dict.

        Three regimes share one code path:

        * **Tracking** -- the same arms are active and the target moved only a
          little (an operator bridge nudging the goal each tick): we warm-start
          from the cached goal and refine *in branch*, never launching the
          global restart search. This keeps teleop smooth (no random elbow/base
          flips) and cheap, and it isolates the arms -- nudging one arm's target
          leaves the other's solution where it was.
        * **Re-acquire** -- tracking refined but the *solved* residual for an arm
          is still large (the in-branch solve got stuck: a wrist singularity, a
          needed branch flip, or recovering from a boundary saturation). A small
          per-tick move never trips the cold path, so without this the gripper
          would stay stuck. We launch the same amortized multi-seed restart the
          cold path uses, freeing only the stuck arm(s) so it re-acquires the
          target while the healthy arm(s) stay pinned (see ``_IK_REACQUIRE_POS`` /
          ``_IK_REACQUIRE_TICKS``).
        * **Cold** -- first solve, the active arm set changed, or the target
          jumped far: run the full multi-seed search, but choose among seeds by
          residual *with a proximity tie-break*, so a distant IK branch is taken
          only when it genuinely reaches better, not to shave off a sub-mm.
        """
        pos = self._normalize_targets(targets)
        arms = [a for a in ("left", "right") if a in pos]
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
        pos_list = [pos[a] for a in arms]

        cache = self._cache
        cache_arms_match = cache is not None and cache["arms"] == tuple(arms)
        if cache_arms_match:
            arm_jump = {a: float(np.linalg.norm(cache["targets"][i] - pos_list[i]))
                        for i, a in enumerate(arms)}
            jump = max(arm_jump.values())
        else:
            arm_jump = {a: float("inf") for a in arms}
            jump = float("inf")

        job = cache.get("job") if cache_arms_match else None

        def _cache_result(q_best, dist_best, job=None, stuck=0):
            self._cache = {
                "arms": tuple(arms), "targets": pos_list,
                "q_best": q_best, "dist": dist_best,
                "job": None if (job is None or job["done"]) else job,
                "stuck": stuck,  # consecutive stuck-tick counter (re-acquire debounce)
            }
            return q_best, dist_best

        def _accept_or_recover(q_best, dist_best):
            """Cache a tracking/steady solution. If an arm's solved residual stays
            too large for several ticks the in-branch solve has genuinely lost its
            target (not just a transient singularity dip) -- re-acquire it with the
            amortized restart, freeing only the persistently-stuck arm(s)."""
            stuck_arms = {a for a in arms if dist_best[a] > _IK_REACQUIRE_POS}
            if not stuck_arms:
                return _cache_result(q_best, dist_best, stuck=0)
            # Large residual: debounce. Keep refining in-branch until it has
            # persisted, so a brief singularity undershoot recovers on its own.
            n = (cache.get("stuck", 0) if cache_arms_match else 0) + 1
            if n < _IK_REACQUIRE_TICKS:
                return _cache_result(q_best, dist_best, stuck=n)
            rjob = self._build_restart_job(
                joint_order, arms, stuck_arms, q_best.copy(), q_best.copy(), q_best,
                lo, hi, q_meas_vec, mid_vec, lift_idx, cold_gain)
            qb, db = self._pump_restart(
                rjob, joint_order, arms, pos, lo, hi, _IK_COLD_BUDGET)
            return _cache_result(qb, db, rjob, stuck=0)

        if cache_arms_match and jump < _IK_TRACK_JUMP and job is not None:
            # A multi-seed restart search (cold solve or re-acquire) is still
            # pending and the target is still close (steady or merely tracking):
            # advance the SAME search toward the current target rather than
            # dropping it. This is what lets the amortized re-acquire actually
            # FINISH while the operator keeps nudging the goal -- otherwise each
            # tick would restart it from scratch and it would never converge. The
            # command leads toward the best-so-far goal meanwhile, so the arm is
            # already moving while the goal sharpens over the next few ticks.
            q_best, dist_best = self._pump_restart(
                job, joint_order, arms, pos, lo, hi, _IK_COLD_BUDGET)
            _cache_result(q_best, dist_best, job)
        elif cache_arms_match and jump < _IK_TARGET_EPS:
            # Steady, nothing pending: reuse the goal we already solved -- unless
            # it is a stuck (large-residual) goal, in which case re-acquire it.
            q_best, dist_best = _accept_or_recover(
                cache["q_best"], cache["dist"])
        elif cache_arms_match and jump < _IK_TRACK_JUMP:
            # Continuous tracking: the target only nudged, so the previous goal
            # is an excellent warm start. Refine a few in-branch iterations and
            # DO NOT restart for a healthy track -- a global search here is what
            # made the arm snap to a random branch and made one moving arm disturb
            # the other. But if the refined solve is stuck (large residual), the
            # in-branch path can't recover on its own, so _accept_or_recover kicks
            # off a targeted re-acquire for just the stuck arm(s).
            q_best, dist_best, _ = self._solve_from(
                cache["q_best"], joint_order, arms, pos, lo, hi,
                cache["q_best"], track_gain, max_iters=_IK_TRACK_ITERS)
            q_best, dist_best = _accept_or_recover(q_best, dist_best)
        else:
            # Cold target (first solve / arm-set change / large jump). When the
            # same arms are active we *pin* any arm whose target barely moved to
            # its cached configuration and only re-search the arm(s) that jumped
            # (plus the shared lift). That stops a big move on one arm from
            # flinging the other one onto a different IK branch -- the held arm's
            # joints just compensate for the shared lift instead of teleporting.
            free_arms = {a for a in arms
                         if not (cache_arms_match and arm_jump[a] < _IK_TRACK_JUMP)}
            base = cache["q_best"] if cache_arms_match else q_meas_vec
            job = self._build_restart_job(
                joint_order, arms, free_arms, base.copy(), base.copy(),
                cache["q_best"] if cache_arms_match else None,
                lo, hi, q_meas_vec, mid_vec, lift_idx, cold_gain)
            q_best, dist_best = self._pump_restart(
                job, joint_order, arms, pos, lo, hi, _IK_COLD_BUDGET)
            _cache_result(q_best, dist_best, job)

        # Command stepping: lead the measured pose toward the solved goal by a
        # bounded amount so the stiff drive supplies holding torque without the
        # command overshooting (the same contract the Isaac teleop relied on).
        dq = q_best - q_meas_vec
        nrm = float(np.linalg.norm(dq))
        if nrm < IK_CMD_DEADBAND:
            # Already at the solved configuration: hold to avoid command jitter.
            # The return contract is the joint commands plus the single ``"_dist"``
            # meta key (callers use ``len(result) <= 1`` as the "command held"
            # sentinel).
            return {"_dist": dist_best}
        if nrm > IK_MAX_DQ:
            dq *= IK_MAX_DQ / nrm
        q_cmd = np.clip(q_meas_vec + dq, lo, hi)

        # Per-arm Cartesian hold (see _IK_ARM_HOLD_CAP): with the shared lift held
        # at its commanded height, trim each arm's own 7 joints so its gripper
        # stays on its target point. Keeps a held arm planted while the lift slews
        # for the other arm; on a steadily-tracking arm the error is tiny so this
        # is a negligible refinement.
        q_cmd_d = {jn: float(q_cmd[k]) for k, jn in enumerate(joint_order)}
        for a in arms:
            aj = ARM_JOINTS[a]
            tip, Ja = self.chains[a].position_jacobian(q_cmd_d, aj)
            err = pos[a] - tip
            if float(np.linalg.norm(err)) < IK_POS_TOL:
                continue
            dq_a, _ = self._dls(Ja, err)
            n = float(np.linalg.norm(dq_a))
            if n > _IK_ARM_HOLD_CAP:
                dq_a *= _IK_ARM_HOLD_CAP / n
            for k, j in enumerate(aj):
                gi = joint_order.index(j)
                q_cmd[gi] = min(hi[gi], max(lo[gi], q_cmd[gi] + float(dq_a[k])))

        out = {jname: float(q_cmd[k]) for k, jname in enumerate(joint_order)}
        out["_dist"] = dist_best
        return out

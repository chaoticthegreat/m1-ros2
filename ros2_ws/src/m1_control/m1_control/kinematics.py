"""URDF kinematics + Drake-backed Cartesian reach for the M1 robot.

This module parses a URDF string (dependency-free) and provides forward
kinematics and a geometric Jacobian for arbitrary base->tip chains. On top of
that it implements a **Drake**-backed position-only Cartesian reach controller:
the inverse kinematics is solved by Drake's ``MultibodyPlant`` +
``InverseKinematics`` (a nonlinear least-squares position-cost solve) wrapped in
an amortized multi-start, so a cold target reliably converges to the global
optimum instead of stalling in a local minimum (the failure the bespoke
multi-seed DLS solver this replaces used to hit: "stops solving early / doesn't
reach the optimal solution").

The solver treats each arm's 7 joints plus the single shared prismatic lift as
the actuated DOFs. Reachable targets converge to sub-millimetre error; an
unreachable one settles at the closest configuration the joint limits allow.
When both arms reach at once they are solved together (both fingertip costs in
one Drake program), so the shared lift is the least-squares compromise that best
serves both grippers.

The reach is POSITION-ONLY: a target is a 3D point (or a dict carrying ``"pos"``;
any ``"quat"``/``"R"`` is ignored). The FK/quaternion utilities
(``pose_jacobian``/``gripper_pose``/``_so3_log``/``mat_to_quat``) remain for the
viz + tests to *report* gripper rotation, but the solve never constrains it.

``solve_step`` distinguishes two regimes:

  * **Tracking** -- the target moved only a little (a teleop bridge nudging the
    goal each tick): warm-start one Drake solve from the cached solution, so the
    arm stays in branch (smooth, no random elbow/base snaps) and the solve is
    ~1-3 ms. If a warm solve stays stuck (large residual) for a few ticks the
    in-branch solve has lost the target, so it escalates to the cold multi-start.
  * **Cold** -- first solve / arm-set change / big jump: an amortized multi-start
    Drake search (a couple of seeds per tick, best-so-far carried in the cache
    while the command already leads toward it), so the worst-case ``solve_step``
    stays well inside the 60 Hz budget while the total search is thorough. Any arm
    whose target barely moved is pinned to its cached branch (and the shared lift
    anchored toward its cached height), so a big move on one arm never drags a
    held one.

Every tick also applies a small capped per-arm Cartesian hold/polish to the
command (each arm's own 7 joints, shared lift fixed), so the two arms are
decoupled through the lift and the command lands ON the target to sub-mm even
when the global solve settled a hair short.

Drake (``pydrake``) is imported lazily, only when the first solve runs, so
FK-only users (the viz nodes) never pay the import or the plant build.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np


# --- Cartesian reach tuning ------------------------------------------------
# Damped-least-squares parameters for the per-arm hold/polish step and for the
# clearance-gradient steps that trajectory.py reuses (via _stack / _dls).
IK_SV_EPS = 0.04          # smallest singular value below which DLS damping turns on
IK_DAMPING_MAX = 0.06     # peak DLS damping injected right at a singularity
_IK_RANK_TOL = 1e-6       # singular values below this are treated as zero
IK_POS_TOL = 2e-4         # task convergence tolerance (m) for the polish + planner

# Command stepping (shared by every regime). The Drake solve gives the optimal
# goal config; the command then LEADS the measured pose toward it by a bounded
# step so the stiff drive supplies holding torque without the command overshooting
# (the same contract the Isaac teleop relied on -- keeps per-tick joint velocity
# hardware-safe).
IK_MAX_DQ = 0.22          # max joint motion (rad) the command leads per tick
IK_CMD_DEADBAND = 1e-4    # hold the command still once the solved step (rad) is tiny
_IK_INT_MAX_DQ = 0.40     # cap on a single internal DLS step (rad); used by the
                          # per-arm polish here and imported by trajectory.py
# Per-arm Cartesian hold/polish applied to the command each tick: with the shared
# lift fixed, a few capped task-only Newton steps on each arm's own 7 joints land
# its gripper ON its target. Decouples the two arms through the shared lift -- a
# held arm keeps its fingertip planted while the lift slews for the other arm --
# and closes the last fraction of a mm the global solve may leave. The TOTAL
# displacement is hard-capped per arm so it never adds joint velocity.
_IK_ARM_HOLD_CAP = 0.06   # max joint motion (rad) of the per-arm hold correction
_IK_HOLD_ITERS = 8        # task-only Newton steps the per-arm hold/polish iterates

# --- Drake reach regimes ---------------------------------------------------
_IK_TRACK_JUMP = 0.06     # target move (m) at/under which we warm-track in-branch
_IK_TARGET_EPS = 1e-4     # target move (m) under which the cached solve is reused
_IK_REACQUIRE_POS = 0.002 # solved residual (m) above which a solve is "stuck"
                          # (gates the held-command sentinel + tracking re-acquire)
_IK_REACQUIRE_TICKS = 3   # consecutive stuck tracking ticks before a cold re-acquire
_IK_CONVERGED = 5.0e-4    # worst-arm residual (m) at/under which the multi-start stops
                          # seeding ("found the basin"). Tighter than _IK_REACQUIRE_POS
                          # because the per-arm polish CANNOT move the shared lift -- a
                          # dual solve stopped at a slightly-off lift height leaves both
                          # arms a mm or two short -- so the search keeps going until the
                          # lift is right. A single-arm target's first good seed already
                          # reaches microns, so it still stops in one tick.
_IK_SEEDS_PER_TICK = 2    # Drake multi-start seeds solved per tick (amortization);
                          # one Drake solve is ~2-10 ms, so this bounds the tick well
                          # under the 60 Hz budget while a cold target converges over
                          # a handful of ticks (the command leads toward best-so-far)
_IK_WPOS = 1.0e4          # position-cost weight (per arm) in the Drake program; with
                          # the small posture reg below this reaches reachable targets
                          # to microns and leaves the null space to the reg
_IK_REG_COLD = 1.0e-2     # light posture reg weight on a free arm's joints; each
                          # solve regularizes toward ITS OWN seed (not mid-range),
                          # which is what keeps the reach continuous (see _DrakeIK.solve)
_IK_REG_TRACK = 1.0e-2    # posture reg weight pulling the config toward the cached
                          # solution while tracking (in-branch continuity)
_IK_REG_HELD = 0.5        # strong reg holding a not-jumped arm on its cached branch
                          # while the other arm's target is re-searched (the shared
                          # lift is hard-pinned to cached in that case, not anchored)

# --- Reach is position-only ------------------------------------------------
# The solver drives each gripper's fingertip to a target *point*; gripper
# orientation is NOT part of the task. Any orientation a caller still sends (a
# legacy pose dict's "quat"/"R") is ignored (see _normalize_targets). The FK
# utilities below still expose orientation because the viz + tests use them to
# *report* the gripper's rotation -- but the reach solve never constrains it.


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

    def __post_init__(self):
        # The origin (parent->child fixed transform) and the actuated flag are
        # constant for the life of the joint, but FK reads ``origin_matrix`` once
        # per joint on *every* call -- and FK is the inner loop of both the IK
        # solver (a Jacobian/cost eval each Gauss-Newton iteration) and the viz
        # link walk. Recomputing ``_rpy_to_matrix`` (nine trig calls + two matmuls)
        # there every time is pure waste, so cache it at construction. Joint origins
        # never change after the URDF is parsed, so the cache is always valid.
        self._origin_matrix = _homogeneous(
            _rpy_to_matrix(*self.origin_rpy), self.origin_xyz)
        self._actuated = self.jtype in ("revolute", "prismatic", "continuous")

    @property
    def origin_matrix(self) -> np.ndarray:
        return self._origin_matrix

    @property
    def actuated(self) -> bool:
        return self._actuated


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
        model.urdf_xml = urdf_xml      # kept so the Drake IK backend can re-parse it
        return model

    def _walk_order(self) -> list:
        """Parent-before-child joint traversal order (topology, cached).

        The kinematic tree's shape is fixed once the URDF is parsed, so the
        per-call work of rebuilding the children map / finding the roots /
        ordering the walk is pure waste -- ``link_transforms`` runs every viz
        frame. Compute the traversal order once: a list of joint names such that
        a joint's parent link transform is always resolved before the joint is
        applied, plus the root links seeded at identity.
        """
        order = getattr(self, "_walk_order_cache", None)
        if order is not None:
            return order
        children: dict = {}
        for jn, j in self.joints.items():
            children.setdefault(j.parent, []).append(jn)
        child_links = {j.child for j in self.joints.values()}
        all_links = set(children) | child_links
        roots = [lk for lk in all_links if lk not in child_links]
        order = []  # joint names, parent-before-child (a joint's parent link transform is resolved before the joint is applied)
        stack = list(roots)
        while stack:
            link = stack.pop()
            for jn in children.get(link, []):
                order.append(jn)
                stack.append(self.joints[jn].child)
        cache = (roots, order)
        self._walk_order_cache = cache
        return cache

    def link_transforms(self, q: dict) -> dict:
        """World 4x4 transform of every link given joint values ``q``.

        Walks the kinematic tree from the root link(s), applying each joint's
        fixed origin then its actuated motion. Used to pose every link mesh in
        the headset viz from a single measured joint state. Links whose joints
        aren't in ``q`` default to 0 (e.g. static base frames).
        """
        roots, order = self._walk_order()
        transforms = {lk: np.eye(4, dtype=np.float64) for lk in roots}
        for jn in order:
            joint = self.joints[jn]
            T = transforms[joint.parent] @ joint.origin_matrix
            if joint.actuated:
                qj = float(q.get(jn, 0.0))
                if joint.jtype == "prismatic":
                    T = T @ _homogeneous(np.eye(3), joint.axis * qj)
                else:  # revolute / continuous
                    T = T @ _homogeneous(_axis_rotation(joint.axis, qj), np.zeros(3))
            transforms[joint.child] = T
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
    """Drake-backed Cartesian reach for one or both arms + shared lift.

    Each :meth:`solve_step` dispatches to a Drake-backed position-cost IK -- a
    warm in-branch tracking solve when the target moved only a little, or an
    amortized cold multi-start (a few Drake seeds per tick) for a first/big-jump
    solve -- to find the optimal joint configuration for the requested
    target(s), then leads the measured command toward it by a bounded step and
    applies a small capped per-arm Cartesian polish. When both arms reach at
    once they share one stacked Drake program, so the single shared lift is
    resolved as the least-squares compromise that best serves both grippers.
    """

    def __init__(self, model: UrdfModel):
        self.model = model
        self.chains = {arm: ArmChain(model, arm) for arm in ("left", "right")}
        # The seed RNG is fixed so a freshly constructed controller is
        # deterministic -- the position-only suite checks that a bare point and a
        # {pos,R} pose dict yield a bit-identical joint solution.
        self._seed_rng = np.random.default_rng(0xC0FFEE)
        # Build the Drake IK backend NOW (not lazily), so its one-time ~70 ms plant
        # build is never charged to a solve_step tick -- the 60 Hz latency gates
        # time solve_step, and the controller is constructed once at startup. The
        # plant is cached per URDF, so the test harnesses (which construct hundreds
        # of fresh controllers to force cold solves) build it only once. If pydrake
        # is unavailable we fall back to a lazy build so FK-only users (the viz
        # nodes) still work without Drake; solve_step then builds it on first use
        # and surfaces any import error there.
        self._dik = None
        try:
            self._ensure_dik()
        except ImportError:
            self._dik = None
        # Cache of the last solved goal + any pending amortized multi-start job.
        self._cache = None

    def fingertip(self, arm: str, q: dict) -> np.ndarray:
        return self.chains[arm].fk(q)[0]

    def gripper_pose(self, arm: str, q: dict):
        """(tip_pos, R_tip) of one arm's gripper from FK -- a position+orientation
        FK helper retained so the viz/tests can report gripper rotation. The
        reach itself is position-only and never constrains R_tip."""
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

    # --- Drake-backed inverse kinematics ----------------------------------
    def _ensure_dik(self):
        """Return the Drake IK backend, building it (once per URDF, process-wide)
        if needed. Called eagerly from __init__; the cache means the heavy plant
        build is shared across the many short-lived controllers the test harnesses
        create."""
        if self._dik is None:
            key = hash(self.model.urdf_xml)
            dik = _DIK_CACHE.get(key)
            if dik is None:
                dik = _DrakeIK(self.model.urdf_xml)
                _DIK_CACHE[key] = dik
            self._dik = dik
        return self._dik

    @staticmethod
    def _joint_order(arms):
        """Active reach DOFs: each arm's 7 joints, then the shared lift last."""
        order = []
        for a in arms:
            order += ARM_JOINTS[a]
        order.append(LIFT_JOINT)
        return order

    @staticmethod
    def _arm_of(j):
        if j in ARM_JOINTS["left"]:
            return "left"
        if j in ARM_JOINTS["right"]:
            return "right"
        return None

    def _active_idx(self, dik, joint_order):
        """Drake position indices of the actuated reach DOFs, in joint_order."""
        return [dik.jstart[j] for j in joint_order]

    def _dist(self, dik, arms, pos, qfull):
        tips = dik.fk(arms, qfull)
        return {a: float(np.linalg.norm(np.asarray(pos[a], float) - tips[a]))
                for a in arms}

    def _seed_overrides(self, dik, joint_order, base_full, free_local, vals):
        """Copy ``base_full`` and overwrite the free DOFs with ``vals`` (a vector
        in joint_order space)."""
        s = base_full.copy()
        for k in free_local:
            s[dik.jstart[joint_order[k]]] = vals[k]
        return s

    def _build_cold_job(self, dik, arms, pos, joint_order, lo, hi, q_meas, cache,
                        free_arms_override=None):
        """Assemble an amortized multi-start ``job``.

        Frees the arm(s) whose target jumped (re-search their joints + the shared
        lift); pins any not-jumped arm to its cached branch with a strong posture
        reg. When an arm IS held, the shared lift is HARD-pinned to its cached
        height (not just anchored): the lift is the only thing that couples the two
        arms, so letting it swing to better serve the jumping arm would drag the
        held gripper. With the lift fixed the held arm stays planted and the
        jumping arm reaches as far as its own 7 joints allow (the held arm is
        sacred; the jumping arm's reach is finished once its target is the only one
        active and the lift frees up again).
        """
        match = cache is not None and cache.get("arms") == tuple(arms)
        cached_vec = cache["q_best"] if match else None
        pos_list = [np.asarray(pos[a], float) for a in arms]

        if free_arms_override is not None:
            # re-acquire: free exactly the arm(s) flagged stuck (by residual), so a
            # healthy arm stays pinned and the shared lift stays put for it.
            free_arms = set(free_arms_override)
        else:
            # cold: free the arm(s) whose TARGET jumped.
            free_arms = set()
            for i, a in enumerate(arms):
                if not match:
                    free_arms.add(a)
                elif float(np.linalg.norm(cache["targets"][i] - pos_list[i])) >= _IK_TRACK_JUMP:
                    free_arms.add(a)
        if not free_arms:
            free_arms = set(arms)
        held_exists = any(a not in free_arms for a in arms)

        # base config the seeds perturb from: measured everywhere, cached on the
        # active DOFs (so held arms + lift start on their known-good branch). Any
        # DOF NOT in active_idx is pinned to its base value by the Drake solve.
        base_full = dik.full_q(q_meas)
        if cached_vec is not None:
            for k, j in enumerate(joint_order):
                base_full[dik.jstart[j]] = cached_vec[k]

        mid = 0.5 * (lo + hi)
        active_idx, reg_w, free_local = [], [], []
        for k, j in enumerate(joint_order):
            arm = self._arm_of(j)
            if j == LIFT_JOINT:
                if held_exists and cached_vec is not None:
                    continue                # hard-pin the lift to cached (not active)
                active_idx.append(dik.jstart[j])
                reg_w.append(0.0)           # lift free so extreme targets still solve
                free_local.append(k)
            elif arm in free_arms:
                active_idx.append(dik.jstart[j])
                reg_w.append(_IK_REG_COLD)  # light reg toward this seed's posture
                free_local.append(k)
            else:                           # held arm: strongly pinned toward its seed
                active_idx.append(dik.jstart[j])  # (= cached; held arms aren't varied)
                reg_w.append(_IK_REG_HELD)
        reg_w = np.asarray(reg_w, float)

        # The reg target for the EXPLORATION seeds: free arm joints toward mid-range
        # (held arms + a pinned lift keep their cached/base value), a consistent
        # well-conditioned posture so the multi-start's best doesn't jump to an
        # awkward random branch.
        reg_explore = base_full.copy()
        for k in free_local:
            j = joint_order[k]
            if j != LIFT_JOINT:
                reg_explore[dik.jstart[j]] = mid[k]

        # Each seed is paired with its reg target. Seed 0 is the base config itself
        # (measured / cached): solving from there finds the IK branch NEAREST the
        # current pose, so the command slews the short way and -- with the early-stop
        # in _pump_cold -- a reachable target converges in one tick without wandering
        # to a far branch. The diverse seeds (lift sweep at mid-range arms, then
        # random postures) are the fallback that escapes a local minimum for a hard
        # target. ALL seeds regularize toward mid-range (reg_explore): a consistent,
        # well-conditioned posture keeps the multi-start's best stable rather than an
        # awkward branch (the "current pose already reaches it" case is handled
        # earlier in solve_step by an explicit hold, so the cold path never needs to
        # reg toward the current pose). Only the free DOFs vary; a held arm (and a
        # pinned lift) stays at base.
        seeds = [(base_full.copy(), reg_explore)]
        lift_k = len(joint_order) - 1
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            vals = mid.copy()
            vals[lift_k] = lo[lift_k] + frac * (hi[lift_k] - lo[lift_k])
            seeds.append((self._seed_overrides(dik, joint_order, base_full, free_local, vals),
                          reg_explore))
        for _ in range(7):
            vals = lo + self._seed_rng.random(lo.shape[0]) * (hi - lo)
            seeds.append((self._seed_overrides(dik, joint_order, base_full, free_local, vals),
                          reg_explore))

        return {
            "active_idx": active_idx, "reg_w": reg_w,
            "seeds": seeds, "q_best_full": base_full.copy(),
            "best_cost": float("inf"),
            "dist_best": {a: float("inf") for a in arms},
            "done": False,
        }

    def _pump_cold(self, dik, arms, pos, job, n_seeds):
        """Advance the multi-start by up to ``n_seeds`` Drake solves; keep the
        best by total squared position error (the least-squares objective)."""
        for _ in range(n_seeds):
            if not job["seeds"]:
                break
            seed, reg_target = job["seeds"].pop(0)
            qf = dik.solve(arms, pos, seed, job["active_idx"],
                           reg_target, job["reg_w"], _IK_WPOS)
            dist = self._dist(dik, arms, pos, qf)
            cost = sum(d * d for d in dist.values())
            if cost < job["best_cost"]:
                job["best_cost"], job["q_best_full"], job["dist_best"] = cost, qf, dist
            # Early stop: once a seed reaches the target (worst arm converged), the
            # basin is found -- more seeds would only pick a different (equally good)
            # branch, so stop and let the command lead onto this one. A hard target
            # (or a dual whose shared lift isn't yet right) stays short here and
            # keeps drawing seeds. This is what keeps an easy/reachable target a
            # 1-tick, single-branch convergence (no wandering between branches).
            if max(job["dist_best"].values()) < _IK_CONVERGED:
                job["seeds"] = []
                break
        if not job["seeds"]:
            job["done"] = True
        return job["q_best_full"], job["dist_best"]

    def _solve_track(self, dik, arms, pos, joint_order, q_meas, cached_vec):
        """One warm Drake solve from the cached solution (in-branch continuity)."""
        active_idx = self._active_idx(dik, joint_order)
        seed = dik.full_q(q_meas)
        for k, j in enumerate(joint_order):
            seed[dik.jstart[j]] = cached_vec[k]
        # reg toward the seed (= cached config) keeps the warm solve in-branch
        reg_w = np.full(len(joint_order), _IK_REG_TRACK)
        qf = dik.solve(arms, pos, seed, active_idx, seed, reg_w, _IK_WPOS)
        return qf, self._dist(dik, arms, pos, qf)

    def solve_step(self, q_meas: dict, targets: dict) -> dict:
        """Drive the command one bounded step toward the optimal reach solution.

        ``targets`` maps arm -> a target *point* (base frame): a 3-vector, or a
        dict carrying ``"pos"`` (any ``"quat"``/``"R"`` is ignored). Returns a dict
        of joint name -> commanded position plus ``"_dist"`` (per-arm position
        residual of the solved configuration); when the command is already on the
        goal it returns only ``{"_dist": ...}`` -- the held sentinel callers use as
        "no new command". See the module docstring for the tracking/cold regimes.
        """
        pos = self._normalize_targets(targets)
        arms = [a for a in ("left", "right") if a in pos]
        if not arms:
            return {}
        dik = self._ensure_dik()
        joint_order = self._joint_order(arms)
        lo, hi = self._bounds(joint_order)
        q_meas_vec = np.array([q_meas.get(j, 0.0) for j in joint_order], dtype=np.float64)
        pos_list = [np.asarray(pos[a], float) for a in arms]

        cache = self._cache
        match = cache is not None and cache.get("arms") == tuple(arms)
        jump = (max(float(np.linalg.norm(cache["targets"][i] - pos_list[i]))
                    for i in range(len(arms))) if match else float("inf"))
        job = cache.get("job") if match else None
        stuck = cache.get("stuck", 0) if match else 0

        def vec(qf):
            return np.array([qf[dik.jstart[j]] for j in joint_order])

        def store(qbest_full, dist_best, job=None, stuck=0):
            self._cache = {
                "arms": tuple(arms), "targets": pos_list,
                "q_best": vec(qbest_full), "q_best_full": qbest_full,
                "dist": dist_best, "job": job, "stuck": stuck,
            }
            return self._cache["q_best"], dist_best

        if job is not None and jump < _IK_TRACK_JUMP:
            # A multi-start search is still pending and the target is still close:
            # keep advancing the SAME search toward the current target (amortized),
            # rather than dropping it and restarting from scratch each tick.
            qf, dist_best = self._pump_cold(dik, arms, pos, job, _IK_SEEDS_PER_TICK)
            q_best, dist_best = store(qf, dist_best, None if job["done"] else job)
        elif (match and jump < _IK_TARGET_EPS and job is None
              and max(cache["dist"].values()) <= _IK_REACQUIRE_POS):
            # Steady on an already-good solution: reuse it (cheap hold), refreshing
            # only the cached target snapshot.
            q_best, dist_best = store(cache["q_best_full"], cache["dist"])
        elif match and jump < _IK_TRACK_JUMP:
            # Continuous tracking: warm solve in-branch from the cached solution.
            qf, dist_best = self._solve_track(
                dik, arms, pos, joint_order, q_meas, cache["q_best"])
            if max(dist_best.values()) <= _IK_REACQUIRE_POS:
                q_best, dist_best = store(qf, dist_best, None, 0)
            elif stuck + 1 < _IK_REACQUIRE_TICKS:
                # large residual but maybe a transient singularity crossing: keep
                # refining in-branch a few ticks before paying for a global restart.
                q_best, dist_best = store(qf, dist_best, None, stuck + 1)
            else:
                # genuinely lost the target -> launch a FULL cold multi-start
                # (all active arms free, shared lift free). We deliberately do NOT
                # pin the lift or hold the "healthy" arm here: when a dual solve is
                # stuck it is usually a bad SHARED-LIFT compromise (one arm reaches
                # at a lift height the other can't use), and the only way out is to
                # let the lift + both arms re-search for the joint optimum. Pinning
                # the lift at the stuck compromise forecloses that and the arm stays
                # short forever (the live "right arm never reaches" bug). The
                # multi-start is seeded from the current pose (seed 0) so the
                # healthy arm only moves as much as the better shared solution needs.
                job = self._build_cold_job(dik, arms, pos, joint_order, lo, hi, q_meas,
                                           cache, free_arms_override=set(arms))
                qf, dist_best = self._pump_cold(dik, arms, pos, job, _IK_SEEDS_PER_TICK)
                q_best, dist_best = store(qf, dist_best, None if job["done"] else job)
        else:
            # Cold target: first solve / arm-set change / big jump.
            cur_dist = {a: float(np.linalg.norm(pos_list[i] - self.fingertip(a, q_meas)))
                        for i, a in enumerate(arms)}
            if max(cur_dist.values()) <= _IK_REACQUIRE_POS:
                # The CURRENT pose already reaches every target: there is nothing to
                # solve, so hold here. Cold-solving instead would resolve the
                # redundant joints toward mid-range -- a different IK branch reaching
                # the same point -- and the bounded command lead would swing the
                # fingertip off-target through that needless reconfiguration (the
                # "track a planned path whose first point is where the arm already
                # is" case). Cache the current config as the goal so subsequent
                # tracking warm-starts in this branch.
                q_best, dist_best = store(dik.full_q(q_meas), cur_dist, None, 0)
            else:
                job = self._build_cold_job(dik, arms, pos, joint_order, lo, hi, q_meas, cache)
                qf, dist_best = self._pump_cold(dik, arms, pos, job, _IK_SEEDS_PER_TICK)
                q_best, dist_best = store(qf, dist_best, None if job["done"] else job)

        # --- command stepping: bounded lead toward the goal -------------------
        dq = q_best - q_meas_vec
        nrm = float(np.linalg.norm(dq))
        solved_resid = max(dist_best.values()) if dist_best else 0.0
        # Only signal "held" when the command has arrived AND the solved config is
        # actually good; a solve still short keeps emitting commands (so the polish
        # below trims it and a re-acquire gets ticks) instead of freezing short.
        if nrm < IK_CMD_DEADBAND and solved_resid <= _IK_REACQUIRE_POS:
            return {"_dist": dist_best}
        if nrm > IK_MAX_DQ:
            dq *= IK_MAX_DQ / nrm
        q_cmd = np.clip(q_meas_vec + dq, lo, hi)

        # --- per-arm Cartesian hold/polish (shared lift fixed) ----------------
        for a in arms:
            aj = ARM_JOINTS[a]
            gidx = [joint_order.index(j) for j in aj]
            q_arm0 = q_cmd[gidx].copy()        # post-lead start (for the cap)
            lo_a, hi_a = lo[gidx], hi[gidx]
            for _polish in range(_IK_HOLD_ITERS):
                q_cmd_d = {jn: float(q_cmd[g]) for g, jn in enumerate(joint_order)}
                tip, Ja = self.chains[a].position_jacobian(q_cmd_d, aj)
                err = np.asarray(pos[a], float) - tip
                if float(np.linalg.norm(err)) < IK_POS_TOL:
                    break
                dq_a, _ = self._dls(Ja, err)
                n = float(np.linalg.norm(dq_a))
                if n > _IK_INT_MAX_DQ:
                    dq_a *= _IK_INT_MAX_DQ / n
                q_try = np.clip(q_cmd[gidx] + dq_a, lo_a, hi_a)
                disp = q_try - q_arm0
                dn = float(np.linalg.norm(disp))
                if dn > _IK_ARM_HOLD_CAP:          # hit the per-arm displacement cap
                    q_cmd[gidx] = np.clip(q_arm0 + disp * (_IK_ARM_HOLD_CAP / dn),
                                          lo_a, hi_a)
                    break
                q_cmd[gidx] = q_try

        out = {jname: float(q_cmd[k]) for k, jname in enumerate(joint_order)}
        out["_dist"] = dist_best
        return out


# Process-wide cache of built Drake backends, keyed by the URDF string, so the
# many short-lived ReachControllers the test harnesses create (one per cold
# solve) share a single ~70 ms plant build. Production builds exactly one.
_DIK_CACHE: dict = {}


class _DrakeIK:
    """Drake ``MultibodyPlant`` + ``InverseKinematics`` backend for the reach.

    Built once per URDF (cached process-wide via :data:`_DIK_CACHE`). The URDF
    is parsed with ``<visual>``/``<collision>`` stripped, so no ``package://`` mesh
    paths need resolving -- the IK only needs the kinematic tree + joint limits.
    ``base_link`` is welded to the world (arm targets are base-relative), and a
    fingertip frame is added per arm at :data:`GRIPPER_TIP_OFFSET` so the Drake
    solve constrains exactly the same point the custom FK reports (verified equal
    to machine precision). Each :meth:`solve` is one position-cost least-squares
    IK from a supplied seed; :class:`ReachController` runs the amortized
    multi-start over these.
    """

    def __init__(self, urdf_xml):
        # pydrake is imported here (lazily) so the module is usable for FK without
        # Drake installed, and so FK-only nodes never pay the import.
        from pydrake.multibody.plant import MultibodyPlant
        from pydrake.multibody.parsing import Parser
        from pydrake.multibody.tree import FixedOffsetFrame
        from pydrake.math import RigidTransform
        from pydrake.multibody.inverse_kinematics import InverseKinematics
        from pydrake.solvers import Solve, SolverOptions, SnoptSolver, IpoptSolver

        self._InverseKinematics = InverseKinematics
        self._Solve = Solve

        root = ET.fromstring(urdf_xml)
        for link in root.findall("link"):
            for tag in ("visual", "collision"):
                for e in list(link.findall(tag)):
                    link.remove(e)
        # Drop the gripper finger joints + links (and their URDF ``mimic``): they
        # hang off ``ee_base_link``, DOWNSTREAM of the fingertip frame, so they
        # never move the fingertip and play no part in the reach IK. Critically,
        # the live controller drives the fingers from the gripper command and the
        # URDF mimics finger2->finger1, so the measured finger pair generally does
        # NOT satisfy that mimic relation. If those joints are in the plant, the IK
        # pins each non-arm joint to its measured value, and a measured finger pair
        # that violates Drake's mimic *coupler constraint* makes the whole program
        # INFEASIBLE -- SNOPT then returns garbage and the arm "can't reach"
        # anything (the live-only failure the offline suites, which use zeroed/
        # consistent fingers, never exposed). Removing them from the IK plant makes
        # the solve robust to any finger state.
        drop = set()
        for j in list(root.findall("joint")):
            if "finger" in j.attrib.get("name", "") or j.find("mimic") is not None:
                child = j.find("child")
                if child is not None:
                    drop.add(child.attrib["link"])
                root.remove(j)
        for lk in list(root.findall("link")):
            if lk.attrib.get("name") in drop:
                root.remove(lk)
        stripped = ET.tostring(root, encoding="unicode")

        plant = MultibodyPlant(0.0)
        Parser(plant).AddModelsFromString(stripped, "urdf")
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName(BASE_LINK))
        self._tip = {}
        for a in ("left", "right"):
            ee = plant.GetFrameByName(EE_LINK_NAME[a])
            self._tip[a] = plant.AddFrame(FixedOffsetFrame(
                a + "_fingertip", ee, RigidTransform(GRIPPER_TIP_OFFSET)))
        plant.Finalize()

        self.plant = plant
        self.world = plant.world_frame()
        self.ctx = plant.CreateDefaultContext()
        self.nq = plant.num_positions()
        self.jstart = {plant.get_joint(j).name(): plant.get_joint(j).position_start()
                       for j in plant.GetJointIndices()
                       if plant.get_joint(j).num_positions() == 1}

        # Bound the worst-case solve time with a major-iteration cap (60 Hz is a
        # goal): prefer SNOPT (fast for IK), fall back to IPOPT, else Drake's
        # default. Tolerances stay at the (already micron-accurate) defaults.
        opts = SolverOptions()
        try:
            if SnoptSolver().available():
                opts.SetOption(SnoptSolver().solver_id(), "Major iterations limit", 300)
        except Exception:
            pass
        try:
            if IpoptSolver().available():
                opts.SetOption(IpoptSolver().solver_id(), "max_iter", 300)
        except Exception:
            pass
        self._opts = opts

    def full_q(self, q_meas):
        """Full Drake position vector seeded from a measured joint dict (joints
        not present default to 0; non-reach joints don't affect the fingertips)."""
        q = np.zeros(self.nq)
        for name, s in self.jstart.items():
            q[s] = float(q_meas.get(name, 0.0))
        return q

    def fk(self, arms, qfull):
        self.plant.SetPositions(self.ctx, qfull)
        return {a: np.asarray(self._tip[a].CalcPoseInWorld(self.ctx).translation())
                for a in arms}

    def solve(self, arms, pos, seed_full, active_idx, reg_target, reg_w, wpos):
        """One position-cost IK solve. Positions NOT in ``active_idx`` are pinned to
        ``seed_full`` (the other arm / base / wheels / fingers); active positions
        get joint limits + a quadratic posture reg pulling each toward
        ``reg_target[s]`` with per-DOF weight ``reg_w``. Each active arm contributes
        a position cost pulling its fingertip to its target.

        The reg TARGET (not just the seed) is what controls continuity vs. posture:
        the caller points the WARM seed's reg at itself (so a config already
        reaching the target stays put -- no needless elbow reconfiguration, the
        command has nothing to slew) and the diverse COLD exploration seeds' reg at
        mid-range (a consistent, well-conditioned posture, so the multi-start's best
        is stable rather than an awkward random branch). Returns the solved full-q
        vector (the last iterate even if the solver did not fully converge -- the
        multi-start keeps the best)."""
        ik = self._InverseKinematics(self.plant, with_joint_limits=True)
        q = ik.q()
        prog = ik.prog()
        active = set(active_idx)
        for s in range(self.nq):
            if s not in active:
                prog.AddBoundingBoxConstraint(seed_full[s], seed_full[s], q[s])
        wI = wpos * np.eye(3)
        for a in arms:
            ik.AddPositionCost(self.world, np.asarray(pos[a], float),
                               self._tip[a], np.zeros(3), wI)
        for k, s in enumerate(active_idx):
            w = float(reg_w[k])
            if w > 0.0:
                prog.AddQuadraticErrorCost(np.array([[w]]),
                                           np.array([float(reg_target[s])]), [q[s]])
        res = self._Solve(prog, seed_full, self._opts)
        return np.asarray(res.GetSolution(q))

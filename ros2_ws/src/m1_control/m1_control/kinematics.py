"""Dependency-free URDF kinematics for the M1 robot.

This module parses a URDF string (no KDL / pinocchio needed) and provides
forward kinematics and a geometric Jacobian for arbitrary base->tip chains.
On top of that it implements the same damped-least-squares (DLS) Cartesian
controller used by the Isaac teleop script, but driven purely from the URDF so
it can run unchanged on the real robot.

The solver treats each arm's 7 joints plus the single shared prismatic lift as
the actuated DOFs. When both arms reach at once they are solved together in one
system so the shared lift column is resolved as a compromise that helps both
grippers (instead of the two arms fighting over the lift), exactly like the
teleop solver.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np


# --- DLS / reach tuning (mirrors isaac/teleop.py) ---------------------------
IK_DAMPING = 0.06       # DLS damping (larger = safer near singularities)
IK_MAX_STEP = 0.12      # max Cartesian error (m) consumed per IK step
IK_GAIN = 0.85          # fraction of the solved step taken per tick
IK_MAX_DQ = 0.22        # max joint motion (rad) the command leads per tick
IK_NULL_GAIN = 0.03     # null-space pull toward mid-range
IK_POS_TOL = 0.012      # settle deadband (m): stop nudging within this

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
    """Stateless-ish DLS Cartesian reach for one or both arms + shared lift."""

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

    def fingertip(self, arm: str, q: dict) -> np.ndarray:
        return self.chains[arm].fk(q)[0]

    def solve_step(self, q_meas: dict, targets: dict) -> dict:
        """One DLS step. ``targets`` maps arm -> 3D world point (base frame).

        Returns a dict of joint name -> new commanded position (only joints that
        moved). ``q_meas`` is the measured joint position dict.
        """
        arms = [a for a in ("left", "right") if targets.get(a) is not None]
        if not arms:
            return {}

        # Build the joint variable vector: each arm's 7 joints + the shared lift.
        joint_order = []
        for a in arms:
            joint_order += ARM_JOINTS[a]
        if LIFT_JOINT not in joint_order:
            joint_order.append(LIFT_JOINT)

        m = 3 * len(arms)
        n = len(joint_order)
        big_j = np.zeros((m, n), dtype=np.float64)
        err = np.zeros(m, dtype=np.float64)
        dist = {}
        any_active = False
        for ai, a in enumerate(arms):
            tip_pos, J = self.chains[a].position_jacobian(q_meas, joint_order)
            e = np.asarray(targets[a], dtype=np.float64) - tip_pos
            d = float(np.linalg.norm(e))
            dist[a] = d
            if d < IK_POS_TOL:
                e = np.zeros(3)
            else:
                any_active = True
                if d > IK_MAX_STEP:
                    e = e * (IK_MAX_STEP / d)
            big_j[3 * ai:3 * ai + 3, :] = J
            err[3 * ai:3 * ai + 3] = e

        if not any_active:
            return {"_dist": dist}

        jjt = big_j @ big_j.T + (IK_DAMPING ** 2) * np.eye(m)
        try:
            j_pinv = big_j.T @ np.linalg.inv(jjt)
        except np.linalg.LinAlgError:
            return {"_dist": dist}

        q_vec = np.array([q_meas.get(j, 0.0) for j in joint_order], dtype=np.float64)
        dq = IK_GAIN * (j_pinv @ err)

        # Null-space: pull each arm's joints toward mid-range; lift gets no pull.
        dq_null = np.zeros(n, dtype=np.float64)
        for a in arms:
            base = joint_order.index(ARM_JOINTS[a][0])
            cur = q_vec[base:base + 7]
            dq_null[base:base + 7] = IK_NULL_GAIN * (self.arm_mid[a] - cur)
        dq = dq + (np.eye(n) - j_pinv @ big_j) @ dq_null

        dq_norm = float(np.linalg.norm(dq))
        if dq_norm > IK_MAX_DQ:
            dq = dq * (IK_MAX_DQ / dq_norm)

        out = {}
        for k, jname in enumerate(joint_order):
            joint = self.model.joints[jname]
            new_q = float(q_vec[k] + dq[k])
            new_q = max(joint.lower, min(joint.upper, new_q))
            out[jname] = new_q
        out["_dist"] = dist
        return out

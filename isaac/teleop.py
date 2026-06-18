"""Keyboard teleoperation for the Ranger Air robot in Isaac Sim.

Every one of the robot's 27 actuated DOFs is controllable from the keyboard.
The mobile base is driven as a swerve platform: you command a body velocity
(forward, strafe, yaw) and the four steer+wheel modules are solved for you, so
the robot can drive straight, crab sideways, turn in place, or arc while moving.

    * 4 wheels        -> velocity drive (swerve module spin)
    * 4 steering      -> position drive (swerve module heading)
    * 1 lift          -> position drive (raise / lower)
    * 2 x 7 arm joints-> position drive (jog the selected joint)
    * 4 finger joints -> position drive (open / close the active gripper)

This needs a display (it opens an Isaac Sim window and listens for keyboard
events), so do NOT run it headless. Launch with Isaac Sim's bundled Python:

    cd /home/jerry/Downloads/M1-visualizer
    /home/jerry/isaac-sim/python.sh isaac/teleop.py

If the USD asset does not exist yet, run isaac/convert_urdf_to_usd.py first.

-------------------------------------------------------------------------------
KEYBOARD MAP
-------------------------------------------------------------------------------
  Mobile base (swerve drive)
    W / S ............ drive forward / reverse
    A / D ............ turn left / right (in place, or arc while driving)
    Q / E ............ strafe (crab) left / right
    C ................ re-center wheels & stop the base
    SPACE ............ stop the base

  Lift
    R / V ............ raise / lower the lift column

  Arms (jog one joint at a time)
    TAB .............. switch active arm (LEFT <-> RIGHT)
    1 .. 7 ........... select which joint of the active arm to jog
    [ / ] ............ select previous / next joint
    UP / DOWN ........ jog the selected joint in + / - direction

  Grippers (active arm)
    O / K ............ open / close the active arm's gripper

  Reach to target (both arms)
    Two target spheres are created in the scene: a red one (/World/IKTarget_L)
    for the left gripper and a blue one (/World/IKTarget_R) for the right. Drag
    each with the viewport Move tool (TAB selects which sphere the gizmo grabs).
    While tracking is on, each arm reaches its gripper toward its own sphere
    using that arm's 7 joints plus the shared lift column; when both arms track,
    they are solved together so the single lift is shared sensibly. The lift
    raises / lowers the arms so high / low targets become reachable; the base is
    never moved, so an out-of-reach target just pulls the arm + lift to their
    limits, i.e. as close as possible.
    P ................ toggle reach-to-target tracking for BOTH arms on / off
    T ................ toggle tracking for the ACTIVE arm only (so you can have
                       a single arm reach: P to turn both off, TAB to pick the
                       arm, then T to enable just it)
    LEFT-CLICK ....... (re)enable tracking for the active arm after jogging it
    (UP/DOWN jogging pauses tracking for the jogged arm so manual control wins
     until you click or press T/P again; the other arm keeps reaching.)

  Global
    H ................ reset every joint to the default (zero) pose
    ESC .............. quit
-------------------------------------------------------------------------------
"""

import argparse
import math
import os
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_USD = os.path.join(REPO_ROOT, "assets", "usd", "ranger_air.usd")

# --- Base velocity command limits (swerve drive) -----------------------------
MAX_LINEAR_SPEED = 0.6        # forward / reverse speed (m/s)
MAX_STRAFE_SPEED = 0.45       # sideways crab speed (m/s)
MAX_YAW_RATE = 1.2            # turn-in-place / arc yaw rate (rad/s, +ve = left)
LINEAR_ACCEL = 1.5            # ramp for forward/strafe command (m/s^2)
YAW_ACCEL = 3.0               # ramp for yaw command (rad/s^2)
WHEEL_RADIUS = 0.055          # effective rolling radius (m); converts m/s -> rad/s
STEER_SMOOTH_TAU = 0.10       # low-pass time constant on module heading (s)

# --- Other control rates -----------------------------------------------------
LIFT_RATE = 0.2              # lift target change rate (m/s)
ARM_JOG_RATE = 0.8            # arm joint jog rate (rad/s)
GRIP_RATE = 1.5               # gripper open/close rate (rad/s)
PHYSICS_HZ = 120.0            # physics substeps per second

# --- Click-to-move inverse kinematics (arm + lift) ---------------------------
# Each arm has its own draggable target sphere; while tracking is on, every arm
# reaches its gripper toward its own sphere. Each physics frame we take one
# damped-least-squares (DLS) step. When both arms track at once we solve them
# together in a single system so the shared lift column is resolved as a
# compromise that helps both grippers, instead of the two arms fighting over it.
# Folding the lift in lets the gripper reach high / low targets the arm alone
# cannot. The base never moves, so an unreachable target simply pulls the arm +
# lift out to their joint limits, i.e. it gets "as close as possible".
#
# Convergence speed: with the stiff arm drive, the steady joint velocity is
# roughly ARM_KP * IK_MAX_DQ / ARM_KD, because the command only ever leads the
# measured pose by IK_MAX_DQ. So IK_MAX_DQ is the main throttle on how fast the
# gripper flies to the target; the task term still shrinks proportionally near
# the goal, so a bigger lead speeds the gross approach without overshooting.
IK_DAMPING = 0.06            # DLS damping (larger = safer near singularities)
IK_MAX_STEP = 0.12           # max Cartesian error (m) consumed per IK step
IK_GAIN = 0.85               # fraction of the solved step taken (damps approach)
IK_MAX_DQ = 0.22             # max joint motion (rad) the command leads per step
IK_NULL_GAIN = 0.03          # null-space pull toward mid-range (uses all joints)
IK_POS_TOL = 0.012           # settle deadband: stop nudging within this (m)
CLICK_NDC_THRESH = 0.012     # mouse travel (NDC) above which a press is a drag
# End-effector (gripper base) link of each arm, used as the IK tool frame.
EE_LINK_NAME = {
    "left": "openarm_left_ee_base_link",
    "right": "openarm_right_ee_base_link",
}
# Offset (m) from the ee_base_link origin to the closed gripper fingertip,
# expressed in the ee_base_link's local frame. The pinch-gripper fingers hang
# off the local -Z axis (the finger joints sit at z=-0.068 and the fingers
# extend ~0.077 m further), so the very tip is ~0.145 m down -Z. The IK drives
# this point onto the target, so the fingertip lands on the sphere centre.
GRIPPER_TIP_OFFSET = (0.0, 0.0, -0.145)
# One draggable target sphere per arm, color-coded so they are easy to tell
# apart (left = red, right = blue).
IK_TARGET_PRIM = {
    "left": "/World/IKTarget_L",
    "right": "/World/IKTarget_R",
}
IK_TARGET_COLOR = {
    "left": (1.0, 0.15, 0.15),
    "right": (0.2, 0.45, 1.0),
}

# --- Drive gains (stiffness kp, damping kd) per joint group ------------------
WHEEL_KP, WHEEL_KD = 0.0, 2500.0       # pure velocity drive (kp must stay 0)
STEER_KP, STEER_KD = 700.0, 180.0      # softer hold so steered wheels don't bind
LIFT_KP, LIFT_KD = 30000.0, 3000.0     # prismatic lift carries the arms
# Stiff arm drive: high stiffness so the arm holds its pose against gravity with
# only a tiny position error, and heavy damping so it settles without ringing.
ARM_KP, ARM_KD = 9000.0, 900.0
GRIP_KP, GRIP_KD = 400.0, 40.0

# --- Position limits (rad / m) used to clamp targets -------------------------
JOINT_LIMITS = {
    "lift_joint": (0.0, 0.85),
    "openarm_left_joint1": (-3.4907, 1.3963),
    "openarm_left_joint2": (-3.3161, 0.17453),
    "openarm_left_joint3": (-1.5708, 1.5708),
    "openarm_left_joint4": (0.0, 2.4435),
    "openarm_left_joint5": (-1.5708, 1.5708),
    "openarm_left_joint6": (-0.7854, 0.7854),
    "openarm_left_joint7": (-1.5708, 1.5708),
    "openarm_right_joint1": (-1.3963, 3.4907),
    "openarm_right_joint2": (-0.17453, 3.3161),
    "openarm_right_joint3": (-1.5708, 1.5708),
    "openarm_right_joint4": (0.0, 2.4435),
    "openarm_right_joint5": (-1.5708, 1.5708),
    "openarm_right_joint6": (-0.7854, 0.7854),
    "openarm_right_joint7": (-1.5708, 1.5708),
    "openarm_left_finger_joint1": (0.0, 0.7854),
    "openarm_left_finger_joint2": (0.0, 0.7854),
    "openarm_right_finger_joint1": (-0.7854, 0.0),
    "openarm_right_finger_joint2": (-0.7854, 0.0),
}

GRIPPER_OPEN = 0.7854  # max finger travel magnitude (rad)

WHEEL_JOINTS = ["fl_wheel_joint", "fr_wheel_joint", "rr_wheel_joint", "rl_wheel_joint"]
STEER_JOINTS = ["fl_steering_joint", "fr_steering_joint", "rr_steering_joint", "rl_steering_joint"]

# The URDF axes are not all aligned, so flip the odd ones out to make a positive
# command move every wheel/steer in the same physical direction. Flip a sign
# here if a wheel spins or a corner steers the wrong way.
WHEEL_DIR = {"fl_wheel_joint": 1.0, "fr_wheel_joint": 1.0, "rr_wheel_joint": 1.0, "rl_wheel_joint": -1.0}
STEER_DIR = {"fl_steering_joint": 1.0, "fr_steering_joint": 1.0, "rr_steering_joint": -1.0, "rl_steering_joint": 1.0}

# Swerve-module positions in the base frame (x forward, y left), in metres,
# taken from the steering-joint origins in the URDF. Keyed by corner prefix.
MODULE_XY = {
    "fl": (0.194, 0.169),
    "fr": (0.194, -0.169),
    "rr": (-0.194, -0.169),
    "rl": (-0.194, 0.169),
}


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def wrap_to_pi(angle):
    """Wrap an angle (rad) into (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def move_toward(current, target, max_delta):
    delta = target - current
    if abs(delta) <= max_delta:
        return target
    return current + math.copysign(max_delta, delta)


def _configure_drive_physics(physx_scene):
    """Higher physics rate and solver iterations for smoother wheel contact."""
    physx_scene.CreateTimeStepsPerSecondAttr().Set(PHYSICS_HZ)
    physx_scene.CreateMinPositionIterationCountAttr().Set(4)
    physx_scene.CreateMaxPositionIterationCountAttr().Set(8)
    physx_scene.CreateMinVelocityIterationCountAttr().Set(1)
    physx_scene.CreateMaxVelocityIterationCountAttr().Set(4)


def _configure_wheel_contacts(stage, robot_root):
    """Give wheel colliders consistent friction so they roll instead of snagging."""
    from pxr import PhysxSchema, Sdf, UsdPhysics, UsdShade

    mat_path = Sdf.Path("/World/PhysicsMaterials/wheel_material")
    mat_prim = stage.GetPrimAtPath(mat_path)
    if not mat_prim or not mat_prim.IsValid():
        mat = UsdShade.Material.Define(stage, mat_path)
        mat_prim = mat.GetPrim()
        usd_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
        usd_mat.CreateStaticFrictionAttr().Set(0.85)
        usd_mat.CreateDynamicFrictionAttr().Set(0.75)
        physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
        physx_mat.CreateFrictionCombineModeAttr().Set("average")
    mat_shade = UsdShade.Material(mat_prim)

    for prim in _iter_prims(stage, robot_root):
        if not prim.GetName().endswith("wheel_link"):
            continue
        UsdPhysics.CollisionAPI.Apply(prim)
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(mat_shade, UsdShade.Tokens.weakerThanDescendants, "physics")
        physx_col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        physx_col.CreateContactOffsetAttr(0.002)
        physx_col.CreateRestOffsetAttr(0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard teleop for the Ranger Air robot.")
    parser.add_argument("--usd", default=DEFAULT_USD, help="Path to the robot USD.")
    parser.add_argument(
        "--spawn-height",
        type=float,
        default=0.1,
        help="Height (m) to spawn the base above the ground plane.",
    )
    parser.add_argument(
        "--fix-base",
        action="store_true",
        help="Pin the base in place (disables driving but keeps the robot still for arm work).",
    )
    args = parser.parse_args()

    usd_path = os.path.abspath(args.usd)
    if not os.path.isfile(usd_path):
        raise FileNotFoundError(
            f"Robot USD not found: {usd_path}\n"
            f"Run: /home/jerry/isaac-sim/python.sh isaac/convert_urdf_to_usd.py"
        )

    # Isaac Sim redirects Python stdout, so mirror key status to a report file.
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_teleop_report.txt")
    report_lines = []

    def report(msg):
        print(msg)
        report_lines.append(str(msg))
        with open(report_path, "w") as fh:
            fh.write("\n".join(report_lines) + "\n")

    from isaacsim import SimulationApp

    # Teleop is always interactive (needs a window + keyboard focus).
    simulation_app = SimulationApp({"headless": False, "renderer": "RaytracedLighting"})

    import carb.input
    import carb.settings
    import numpy as np
    import omni.appwindow
    import omni.timeline
    import omni.usd
    from isaacsim.core.prims import Articulation, RigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics

    stage = omni.usd.get_context().get_stage()

    # --- Physics scene -----------------------------------------------------
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
    physx_scene = PhysxSchema.PhysxSceneAPI.Get(stage, "/physicsScene")
    physx_scene.CreateEnableCCDAttr(True)
    physx_scene.CreateEnableStabilizationAttr(True)
    physx_scene.CreateEnableGPUDynamicsAttr(False)
    physx_scene.CreateBroadphaseTypeAttr("MBP")
    physx_scene.CreateSolverTypeAttr("TGS")
    _configure_drive_physics(physx_scene)

    # --- Ground plane + light ---------------------------------------------
    PhysicsSchemaTools.addGroundPlane(
        stage, "/groundPlane", "Z", 100.0, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5)
    )
    distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
    distant_light.CreateIntensityAttr(2500)

    # --- Reference the robot asset -----------------------------------------
    robot_root = "/World/RangerAir"
    add_reference_to_stage(usd_path, robot_root)

    robot_prim = stage.GetPrimAtPath(robot_root)
    xform = UsdGeom.Xformable(robot_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(args.spawn_height)))

    # Find the prim carrying the articulation root (fall back to the robot root).
    articulation_path = robot_root
    for prim in _iter_prims(stage, robot_root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_path = prim.GetPath().pathString
            break

    if args.fix_base:
        _pin_base(stage, articulation_path, UsdPhysics)

    _configure_wheel_contacts(stage, robot_root)

    # --- Start the simulation ----------------------------------------------
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    simulation_app.update()

    robot = Articulation(articulation_path)
    robot.initialize()
    # A few warm-up updates so the physics handle is fully populated.
    for _ in range(3):
        simulation_app.update()

    if not robot.is_physics_handle_valid():
        report(f"[teleop] ERROR: {articulation_path} is not a valid articulation handle")
        simulation_app.close()
        return

    dof_names = list(robot.dof_names)
    num_dof = len(dof_names)
    name_to_idx = {name: i for i, name in enumerate(dof_names)}
    report(f"[teleop] articulation: {articulation_path}")
    report(f"[teleop] DOF count: {num_dof}")
    report(f"[teleop] DOF names: {dof_names}")

    def indices_for(names):
        return [name_to_idx[n] for n in names if n in name_to_idx]

    wheel_idx = indices_for(WHEEL_JOINTS)
    steer_idx = indices_for(STEER_JOINTS)
    lift_idx = name_to_idx.get("lift_joint")
    arm_joint_idx = {
        "left": [name_to_idx.get(f"openarm_left_joint{i}") for i in range(1, 8)],
        "right": [name_to_idx.get(f"openarm_right_joint{i}") for i in range(1, 8)],
    }
    finger_idx = {
        "left": indices_for(["openarm_left_finger_joint1", "openarm_left_finger_joint2"]),
        "right": indices_for(["openarm_right_finger_joint1", "openarm_right_finger_joint2"]),
    }

    # --- Click-to-move IK setup -------------------------------------------
    # We need, per arm: a handle to read the gripper's world pose, and the
    # row/column layout of the articulation Jacobian so we can pull out the
    # 3x7 linear block that maps the 7 arm-joint velocities to gripper motion.
    body_names = list(robot.body_names)

    def _ee_body_index(link_name):
        if link_name in body_names:
            return body_names.index(link_name)
        for i, b in enumerate(body_names):
            if b.endswith(link_name):
                return i
        return None

    ee_body_index = {arm: _ee_body_index(EE_LINK_NAME[arm]) for arm in ("left", "right")}
    ee_prim = {}
    for arm in ("left", "right"):
        ee_path = _find_link_prim_path(stage, robot_root, EE_LINK_NAME[arm], UsdPhysics)
        if ee_path is not None:
            try:
                ee_prim[arm] = RigidPrim(
                    ee_path,
                    name=f"ee_{arm}",
                    reset_xform_properties=False,
                    prepare_contact_sensors=False,
                )
            except Exception as exc:  # noqa: BLE001
                report(f"[teleop] WARN: could not wrap {arm} gripper link ({exc})")

    # Jacobian layout: fixed-base -> (num_bodies-1, 6, num_dof); floating-base ->
    # (num_bodies, 6, num_dof+6) with the 6 free-root columns coming first.
    ik_ready = False
    col_offset = 0
    body_row_offset = 0
    try:
        jac0 = np.asarray(robot.get_jacobians())[0]
        n_rows, _, n_cols = jac0.shape
        col_offset = n_cols - num_dof
        body_row_offset = robot.num_bodies - n_rows
        ik_ready = all(ee_body_index[a] is not None for a in ("left", "right")) and bool(ee_prim)
        report(f"[teleop] IK jacobian shape {jac0.shape}; col_offset={col_offset} row_offset={body_row_offset}")
    except Exception as exc:  # noqa: BLE001
        report(f"[teleop] WARN: jacobian unavailable, click-to-move disabled ({exc})")

    # Movable target objects: one sphere per arm. The user drags them around
    # with the viewport Move tool; while tracking is on each arm reaches its
    # gripper toward its own sphere. Each sphere starts at that arm's current
    # gripper pose so nothing jumps when we begin.
    target_prim = {}
    default_init = {"left": Gf.Vec3d(0.4, 0.25, 0.7), "right": Gf.Vec3d(0.4, -0.25, 0.7)}
    for arm in ("left", "right"):
        marker = UsdGeom.Sphere.Define(stage, Sdf.Path(IK_TARGET_PRIM[arm]))
        marker.CreateRadiusAttr(0.04)
        marker.CreateDisplayColorAttr([Gf.Vec3f(*IK_TARGET_COLOR[arm])])
        prim = marker.GetPrim()
        init_target = default_init[arm]
        try:
            if arm in ee_prim:
                _p = np.asarray(ee_prim[arm].get_world_poses()[0]).reshape(-1)[:3]
                init_target = Gf.Vec3d(float(_p[0]), float(_p[1]), float(_p[2]))
        except Exception:  # noqa: BLE001
            pass
        m_xform = UsdGeom.Xformable(prim)
        m_xform.ClearXformOpOrder()
        m_xform.AddTranslateOp().Set(init_target)
        target_prim[arm] = prim

    def select_target(arm):
        """Select the given arm's sphere so the Move gizmo drags that one."""
        try:
            omni.usd.get_context().get_selection().set_selected_prim_paths(
                [IK_TARGET_PRIM[arm]], True
            )
        except Exception:  # noqa: BLE001
            pass

    # Switch the viewport gizmo to translate and select the active arm's target
    # so it is immediately draggable with the mouse.
    try:
        carb.settings.get_settings().set("/app/transform/operation", "translate")
    except Exception:  # noqa: BLE001
        pass
    select_target("left")

    xform_cache = UsdGeom.XformCache()

    def get_target_world_pos(arm):
        xform_cache.Clear()
        translation = xform_cache.GetLocalToWorldTransform(target_prim[arm]).ExtractTranslation()
        return np.array([translation[0], translation[1], translation[2]], dtype=np.float32)

    # --- Configure per-DOF drive gains -------------------------------------
    kp = np.full(num_dof, ARM_KP, dtype=np.float32)
    kd = np.full(num_dof, ARM_KD, dtype=np.float32)
    for i in wheel_idx:
        kp[i], kd[i] = WHEEL_KP, WHEEL_KD
    for i in steer_idx:
        kp[i], kd[i] = STEER_KP, STEER_KD
    if lift_idx is not None:
        kp[lift_idx], kd[lift_idx] = LIFT_KP, LIFT_KD
    for arm in ("left", "right"):
        for i in finger_idx[arm]:
            kp[i], kd[i] = GRIP_KP, GRIP_KD
    robot.set_gains(kps=kp.reshape(1, -1), kds=kd.reshape(1, -1))

    # --- Command state -----------------------------------------------------
    pos_target = np.zeros(num_dof, dtype=np.float32)
    try:
        pos_target[:] = np.asarray(robot.get_joint_positions()).reshape(-1)
    except Exception:  # noqa: BLE001
        pass

    wheel_idx_arr = np.array(wheel_idx, dtype=np.int32)
    pos_ctrl_idx = np.array([i for i in range(num_dof) if i not in set(wheel_idx)], dtype=np.int32)

    state = {
        "vx": 0.0,        # forward velocity command (m/s)
        "vy": 0.0,        # strafe velocity command (m/s, +ve = left)
        "omega": 0.0,     # yaw rate command (rad/s, +ve = left/CCW)
        "wheel_head": [0.0, 0.0, 0.0, 0.0],  # applied module heading per wheel (rad)
        "lift": float(pos_target[lift_idx]) if lift_idx is not None else 0.0,
        "grip": {"left": 0.0, "right": 0.0},
        "active_arm": "left",
        "active_joint": 1,  # 1..7
        "ik_tracking": {"left": False, "right": False},  # per-arm reach on/off
        "ik_dist": {"left": 0.0, "right": 0.0},          # gripper-to-target dist (m)
        "quit": False,
    }

    def stop_base():
        state["vx"] = 0.0
        state["vy"] = 0.0
        state["omega"] = 0.0
        state["wheel_head"] = [0.0, 0.0, 0.0, 0.0]

    def reset_pose():
        pos_target[:] = 0.0
        stop_base()
        state["lift"] = 0.0
        state["grip"]["left"] = 0.0
        state["grip"]["right"] = 0.0
        state["ik_tracking"]["left"] = False
        state["ik_tracking"]["right"] = False
        report("[teleop] reset to default pose")

    arm_joint_mid = {
        arm: np.array(
            [0.5 * sum(JOINT_LIMITS.get(dof_names[i], (-math.pi, math.pi))) for i in arm_joint_idx[arm]],
            dtype=np.float64,
        )
        for arm in ("left", "right")
    }

    def _quat_rotate(quat, vec):
        """Rotate a 3-vector by an Isaac-Sim quaternion [w, x, y, z]."""
        w = float(quat[0])
        u = np.asarray(quat[1:4], dtype=np.float64)
        v = np.asarray(vec, dtype=np.float64)
        return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)

    def get_ee_tip_pos(arm):
        """World position of the gripper fingertip (the IK tool point)."""
        if arm not in ee_prim:
            return None
        try:
            positions, orientations = ee_prim[arm].get_world_poses()
        except Exception:  # noqa: BLE001
            return None
        pos = np.asarray(positions).reshape(-1)[:3].astype(np.float64)
        quat = np.asarray(orientations).reshape(-1)[:4].astype(np.float64)
        return pos + _quat_rotate(quat, GRIPPER_TIP_OFFSET)

    def solve_ik_step(arms):
        """One damped-least-squares step for every arm in ``arms`` + the lift.

        ``arms`` is the list of arms currently reaching (one or both). All of
        them are solved together in a single system whose joint vector is each
        arm's 7 joints followed by the single shared lift, e.g. for both arms it
        is [left 7, right 7, lift] = 15 DOF and the task is the two stacked 3-D
        fingertip errors. Solving jointly means the shared lift column is
        resolved as one compromise that helps both grippers, instead of the two
        arms each writing a different lift target and fighting frame to frame.

        The lift is folded in so the prismatic column raises / lowers the whole
        assembly in tandem with the joints, letting a gripper reach targets that
        are too high or low for the arm alone. A null-space term (which never
        disturbs the tip positions) nudges the redundant arm joints toward the
        middle of their range; the lift gets no null-space pull, so it only
        travels when the task actually needs the extra vertical reach.
        """
        if not ik_ready:
            return
        arms = [a for a in arms if a in ee_prim and all(i is not None for i in arm_joint_idx[a])]
        if not arms:
            return
        try:
            jac = np.asarray(robot.get_jacobians())[0]
            q_meas = np.asarray(robot.get_joint_positions()).reshape(-1)
        except Exception:  # noqa: BLE001
            return

        # Lay out the joint vector: each arm's 7 joints, then the shared lift.
        idxs = []
        arm_slot = {}
        for a in arms:
            arm_slot[a] = (len(idxs), len(idxs) + 7)
            idxs += list(arm_joint_idx[a])
        lift_in_ik = lift_idx is not None
        if lift_in_ik:
            lift_pos = len(idxs)
            idxs.append(lift_idx)
        n_ik = len(idxs)
        cols = [i + col_offset for i in idxs]

        m = 3 * len(arms)
        big_j = np.zeros((m, n_ik), dtype=np.float64)
        err_stack = np.zeros(m, dtype=np.float64)
        any_active = False
        for ai, a in enumerate(arms):
            try:
                positions, orientations = ee_prim[a].get_world_poses()
                ee_pos = np.asarray(positions).reshape(-1)[:3].astype(np.float64)
                ee_quat = np.asarray(orientations).reshape(-1)[:4].astype(np.float64)
            except Exception:  # noqa: BLE001
                return
            # Rigid offset from the link origin to the fingertip, in world frame.
            tip_offset = _quat_rotate(ee_quat, GRIPPER_TIP_OFFSET)
            tip_pos = ee_pos + tip_offset
            err = get_target_world_pos(a).astype(np.float64) - tip_pos
            dist = float(np.linalg.norm(err))
            state["ik_dist"][a] = dist
            # Settle deadband: zero this arm's task once it is close enough so it
            # holds steady; the other arm (and the lift) keep solving.
            if dist < IK_POS_TOL:
                err = np.zeros(3, dtype=np.float64)
            else:
                any_active = True
                # Cap the per-step error so the local linearization stays valid.
                if dist > IK_MAX_STEP:
                    err = err * (IK_MAX_STEP / dist)
            row = ee_body_index[a] - body_row_offset
            # Full Jacobian rows for this fingertip wrt all our IK columns. The
            # other arm's columns are ~0 here (its joints don't move this tip),
            # so the only coupling between arms is the shared lift column.
            j_v = jac[row, 0:3, :][:, cols].astype(np.float64)
            j_w = jac[row, 3:6, :][:, cols].astype(np.float64)
            # Tool-point Jacobian: the fingertip is rigidly offset from the link
            # origin by tip_offset, so its linear velocity picks up an omega x r
            # term. Without this the tip would settle ~14 cm past the target.
            rx, ry, rz = float(tip_offset[0]), float(tip_offset[1]), float(tip_offset[2])
            skew_r = np.array([[0.0, -rz, ry], [rz, 0.0, -rx], [-ry, rx, 0.0]], dtype=np.float64)
            big_j[3 * ai:3 * ai + 3, :] = j_v - skew_r @ j_w
            err_stack[3 * ai:3 * ai + 3] = err

        # Every reaching arm is already settled: nothing to command.
        if not any_active:
            return

        jjt = big_j @ big_j.T + (IK_DAMPING * IK_DAMPING) * np.eye(m)
        try:
            jjt_inv = np.linalg.inv(jjt)
        except np.linalg.LinAlgError:
            return
        j_pinv = big_j.T @ jjt_inv  # n_ik x m damped pseudo-inverse

        # Base the new command on the MEASURED joint angles, so the setpoint only
        # ever rides ~dq ahead of where the arm actually is. The stiff drive then
        # supplies the holding torque from that tiny lead (so it reaches under
        # gravity), while the bounded lead means it can't overshoot and ring.
        q_ik = q_meas[idxs].astype(np.float64)
        dq = IK_GAIN * (j_pinv @ err_stack)
        # Null-space resolution: pull each arm's 7 joints toward mid-range; the
        # lift's null target is its current position, so it never drifts on its
        # own and only moves when the task term needs the extra vertical reach.
        dq_null = np.zeros(n_ik, dtype=np.float64)
        for a in arms:
            lo_s, hi_s = arm_slot[a]
            dq_null[lo_s:hi_s] = IK_NULL_GAIN * (arm_joint_mid[a] - q_ik[lo_s:hi_s])
        dq = dq + (np.eye(n_ik) - j_pinv @ big_j) @ dq_null
        dq_norm = float(np.linalg.norm(dq))
        if dq_norm > IK_MAX_DQ:
            dq = dq * (IK_MAX_DQ / dq_norm)
        for k, di in enumerate(idxs):
            lo, hi = JOINT_LIMITS.get(dof_names[di], (-math.pi, math.pi))
            new_q = clamp(float(q_ik[k] + dq[k]), lo, hi)
            pos_target[di] = new_q
            # Keep the lift's command state in sync so the manual jog / HUD and
            # the per-frame lift write below agree with what the IK chose.
            if lift_in_ik and di == lift_idx:
                state["lift"] = new_q

    # --- Keyboard handling -------------------------------------------------
    KB = carb.input.KeyboardInput
    pressed = set()
    number_keys = {
        KB.KEY_1: 1, KB.KEY_2: 2, KB.KEY_3: 3, KB.KEY_4: 4,
        KB.KEY_5: 5, KB.KEY_6: 6, KB.KEY_7: 7,
    }

    def on_keyboard_event(event):
        et = event.type
        key = event.input
        if et == carb.input.KeyboardEventType.KEY_PRESS:
            pressed.add(key)
            # Discrete (one-shot) actions handled on the initial press.
            if key == KB.ESCAPE:
                state["quit"] = True
            elif key == KB.TAB:
                state["active_arm"] = "right" if state["active_arm"] == "left" else "left"
                # Select that arm's sphere so the Move gizmo drags the right one.
                select_target(state["active_arm"])
                report(f"[teleop] active arm: {state['active_arm'].upper()}")
            elif key in number_keys:
                state["active_joint"] = number_keys[key]
            elif key == KB.RIGHT_BRACKET:
                state["active_joint"] = state["active_joint"] % 7 + 1
            elif key == KB.LEFT_BRACKET:
                state["active_joint"] = (state["active_joint"] - 2) % 7 + 1
            elif key == KB.H:
                reset_pose()
            elif key == KB.C:
                stop_base()
            elif key == KB.P:
                trk = state["ik_tracking"]
                # Toggle both arms together: if either is reaching, turn both
                # off; otherwise turn both on.
                new_on = not (trk["left"] or trk["right"])
                trk["left"] = trk["right"] = new_on
                report(f"[teleop] reach-to-target {'ON' if new_on else 'OFF'} (both arms)")
            elif key == KB.T:
                # Toggle ONLY the active arm. Combined with P this lets you reach
                # with a single arm: P off (both off), TAB to the arm, T to enable
                # just it. The other arm holds its current pose.
                a = state["active_arm"]
                trk = state["ik_tracking"]
                trk[a] = not trk[a]
                report(f"[teleop] reach-to-target {a.upper()} {'ON' if trk[a] else 'OFF'}")
        elif et == carb.input.KeyboardEventType.KEY_RELEASE:
            pressed.discard(key)
        return True

    appwindow = omni.appwindow.get_default_app_window()
    keyboard = appwindow.get_keyboard()
    input_iface = carb.input.acquire_input_interface()
    kb_sub = input_iface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    # --- Viewport click -> (re)enable reach-to-target ----------------------
    # A transparent omni.ui.scene Screen sits over the viewport. A plain click
    # (not a drag, so it never fights the Move gizmo) re-enables tracking after
    # you have jogged the arm by hand. Dragging the target with the gizmo is a
    # drag and is ignored here, so the arm just keeps following the target.
    def on_viewport_click():
        if KB.LEFT_ALT in pressed or KB.RIGHT_ALT in pressed:
            return
        # Re-enable only the active arm, so a click does not clobber single-arm
        # mode. Use P to bring both arms back, or TAB then click for the other.
        a = state["active_arm"]
        if not state["ik_tracking"][a]:
            state["ik_tracking"][a] = True
            report(f"[teleop] reach-to-target {a.upper()} ON")

    keep_alive = []  # hold references so the scene/registration are not GC'd

    class _ClickToTrackScene:
        """Built by RegisterScene inside the viewport's omni.ui.scene SceneView."""

        def __init__(self, desc):
            from omni.ui import scene as sc

            # Attributes the viewport scene layer expects on a scene instance.
            self.visible = True
            self.name = "RangerAirClickToTrack"
            self.categories = ()
            self._press_ndc = None

            def on_began(sender):
                self._press_ndc = sender.gesture_payload.mouse

            def on_ended(sender):
                ndc = sender.gesture_payload.mouse
                # Only a near-stationary press/release counts as a click; a real
                # drag (gizmo move, box-select, camera) must not trigger a reach.
                if self._press_ndc is not None:
                    dx = ndc[0] - self._press_ndc[0]
                    dy = ndc[1] - self._press_ndc[1]
                    if (dx * dx + dy * dy) ** 0.5 > CLICK_NDC_THRESH:
                        return
                on_viewport_click()

            self._screen = sc.Screen(
                gestures=[sc.DragGesture(mouse_button=0, on_began_fn=on_began, on_ended_fn=on_ended)]
            )

        def destroy(self):
            self._screen = None

    try:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.kit.viewport.window")
        enable_extension("omni.kit.viewport.registry")
        from omni.kit.viewport.registry import RegisterScene

        keep_alive.append(RegisterScene(_ClickToTrackScene, "rangerair.clicktotrack"))
        report("[teleop] reach-to-target ready: move /World/IKTarget; P or click to track")
    except Exception as exc:  # noqa: BLE001
        report(f"[teleop] WARN: viewport click unavailable, use P to track ({exc})")

    # --- Optional on-screen HUD -------------------------------------------
    hud_label = None
    try:
        import omni.ui as ui

        hud_window = ui.Window("Ranger Air Teleop", width=360, height=500)
        with hud_window.frame:
            hud_label = ui.Label("", alignment=ui.Alignment.LEFT_TOP, word_wrap=True)
    except Exception as exc:  # noqa: BLE001
        report(f"[teleop] HUD unavailable ({exc}); using console only")

    help_text = (
        "RANGER AIR TELEOP\n"
        "Base:  W/S fwd/back  A/D turn L/R  Q/E strafe  C/SPACE stop\n"
        "Lift:  R raise  V lower\n"
        "Arms:  TAB switch arm  1-7 select joint  [ ] cycle  UP/DOWN jog\n"
        "Grip:  O open  K close\n"
        "Reach: drag red(L)/blue(R) spheres  P both  T active-arm  click re-track\n"
        "Misc:  H reset pose  ESC quit"
    )
    report(help_text)

    # --- Control loop ------------------------------------------------------
    def held(*keys):
        return any(k in pressed for k in keys)

    last_frame_time = time.monotonic()
    frame = 0
    while simulation_app.is_running() and not state["quit"]:
        now = time.monotonic()
        dt = clamp(now - last_frame_time, 1.0 / 240.0, 0.05)
        last_frame_time = now

        # Mobile base: build a body-velocity command (forward, strafe, yaw) and
        # ramp it for smoothness. Yaw with no forward speed -> turn in place;
        # yaw plus forward speed -> arc. This is solved per wheel below.
        vx_target = (MAX_LINEAR_SPEED if held(KB.W) else 0.0) - (MAX_LINEAR_SPEED if held(KB.S) else 0.0)
        vy_target = (MAX_STRAFE_SPEED if held(KB.Q) else 0.0) - (MAX_STRAFE_SPEED if held(KB.E) else 0.0)
        yaw_target = (MAX_YAW_RATE if held(KB.A) else 0.0) - (MAX_YAW_RATE if held(KB.D) else 0.0)
        if held(KB.SPACE):
            vx_target = vy_target = yaw_target = 0.0
        state["vx"] = move_toward(state["vx"], vx_target, LINEAR_ACCEL * dt)
        state["vy"] = move_toward(state["vy"], vy_target, LINEAR_ACCEL * dt)
        state["omega"] = move_toward(state["omega"], yaw_target, YAW_ACCEL * dt)

        # Lift: prismatic position target.
        if lift_idx is not None:
            if held(KB.R):
                state["lift"] += LIFT_RATE * dt
            if held(KB.V):
                state["lift"] -= LIFT_RATE * dt
            lo, hi = JOINT_LIMITS["lift_joint"]
            state["lift"] = clamp(state["lift"], lo, hi)

        # Active arm joint jog.
        arm = state["active_arm"]
        jidx = arm_joint_idx[arm][state["active_joint"] - 1]
        if jidx is not None:
            jname = dof_names[jidx]
            lo, hi = JOINT_LIMITS.get(jname, (-math.pi, math.pi))
            # Jogging takes over: pause tracking for the jogged arm so the
            # manual motion sticks until the user clicks / presses P to resume
            # reaching. The other arm keeps reaching its own target.
            if held(KB.UP):
                state["ik_tracking"][arm] = False
                pos_target[jidx] = clamp(pos_target[jidx] + ARM_JOG_RATE * dt, lo, hi)
            if held(KB.DOWN):
                state["ik_tracking"][arm] = False
                pos_target[jidx] = clamp(pos_target[jidx] - ARM_JOG_RATE * dt, lo, hi)

        # Gripper of the active arm.
        if held(KB.O):
            state["grip"][arm] = clamp(state["grip"][arm] + GRIP_RATE * dt, 0.0, GRIPPER_OPEN)
        if held(KB.K):
            state["grip"][arm] = clamp(state["grip"][arm] - GRIP_RATE * dt, 0.0, GRIPPER_OPEN)

        # Reach-to-target: drive each reaching arm's joints + the shared lift
        # toward its own target sphere. Runs after the manual jog so an active
        # reach overrides it; the base is untouched, so an unreachable target
        # just stretches the arm + lift toward it as far as the limits allow.
        reaching = [a for a in ("left", "right") if state["ik_tracking"][a]]
        if reaching:
            solve_ik_step(reaching)

        # --- Swerve kinematics: body velocity -> per-module heading + spin ----
        vx, vy, yaw = state["vx"], state["vy"], state["omega"]
        steer_blend = min(1.0, dt / STEER_SMOOTH_TAU)
        wheel_spin_cmd = [0.0, 0.0, 0.0, 0.0]
        for k, jn in enumerate(WHEEL_JOINTS):
            corner = jn.split("_")[0]
            mx, my = MODULE_XY[corner]
            # Velocity of this module's contact point in the base frame.
            vxi = vx - yaw * my
            vyi = vy + yaw * mx
            speed = math.hypot(vxi, vyi)
            applied = state["wheel_head"][k]
            if speed < 1e-4:
                # No motion for this module: hold heading, stop the wheel.
                spin = 0.0
            else:
                heading = math.atan2(vyi, vxi)
                spin = speed / WHEEL_RADIUS
                # Angle optimisation: never swing a module more than 90 deg;
                # flip it 180 deg and reverse the wheel spin instead.
                diff = wrap_to_pi(heading - applied)
                if abs(diff) > math.pi / 2.0:
                    heading = wrap_to_pi(heading + math.pi)
                    diff = wrap_to_pi(heading - applied)
                    spin = -spin
                applied = applied + diff * steer_blend
            state["wheel_head"][k] = applied
            pos_target[steer_idx[k]] = STEER_DIR[STEER_JOINTS[k]] * applied
            wheel_spin_cmd[k] = WHEEL_DIR[jn] * spin

        # --- Compose and apply joint commands --------------------------------
        # Wheels use velocity targets only; everything else uses position targets.
        # Sending both to all DOFs makes stale wheel position targets fight the drive.
        if lift_idx is not None:
            pos_target[lift_idx] = state["lift"]
        for i in finger_idx["left"]:
            pos_target[i] = state["grip"]["left"]
        for i in finger_idx["right"]:
            pos_target[i] = -state["grip"]["right"]

        if wheel_idx:
            wheel_vel = np.array(wheel_spin_cmd, dtype=np.float32).reshape(1, -1)
            robot.set_joint_velocity_targets(wheel_vel, joint_indices=wheel_idx_arr)

        pos_vals = pos_target[pos_ctrl_idx].reshape(1, -1)
        robot.set_joint_position_targets(pos_vals, joint_indices=pos_ctrl_idx)

        # --- HUD / status --------------------------------------------------
        if hud_label is not None and frame % 3 == 0:
            sel_name = dof_names[jidx] if jidx is not None else "n/a"
            sel_val = float(pos_target[jidx]) if jidx is not None else 0.0

            def _reach_line(a):
                tip = get_ee_tip_pos(a)
                state_txt = "track" if state["ik_tracking"][a] else "manual"
                if tip is not None:
                    e = get_target_world_pos(a).astype(np.float64) - tip
                    dist = float(np.linalg.norm(e))
                    return f"  {a.upper():5s} {state_txt:6s} {dist * 100.0:6.2f} cm"
                return f"  {a.upper():5s} {state_txt:6s}    n/a"

            reach_text = "Reach (err to sphere):\n" + "\n".join(
                _reach_line(a) for a in ("left", "right")
            )
            hud_label.text = (
                f"{help_text}\n\n"
                f"Active arm:   {arm.upper()}\n"
                f"Sel. joint:   {state['active_joint']}  ({sel_name})\n"
                f"  target:     {sel_val:+.3f} rad\n"
                f"{reach_text}\n"
                f"Base fwd:     {state['vx']:+.2f} m/s   strafe {state['vy']:+.2f}\n"
                f"Yaw rate:     {state['omega']:+.2f} rad/s\n"
                f"Lift:         {state['lift']:.3f} m\n"
                f"Grip L/R:     {state['grip']['left']:.2f} / {state['grip']['right']:.2f}"
            )

        simulation_app.update()
        frame += 1

    input_iface.unsubscribe_to_keyboard_events(keyboard, kb_sub)
    timeline.stop()
    simulation_app.close()
    report("[teleop] done.")


def _iter_prims(stage, root_path):
    """Yield the prim at root_path and all of its descendants."""
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return
    yield root
    for child in root.GetChildren():
        yield from _iter_prims(stage, child.GetPath().pathString)


def _find_link_prim_path(stage, root_path, link_name, UsdPhysics):
    """Return the prim path of the rigid body for a named link under root_path.

    The URDF importer nests prims (the link name is repeated), so prefer the
    prim that actually carries the Rigid Body API; fall back to any name match.
    """
    fallback = None
    for prim in _iter_prims(stage, root_path):
        if prim.GetName() != link_name:
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim.GetPath().pathString
        if fallback is None:
            fallback = prim.GetPath().pathString
    return fallback


def _pin_base(stage, articulation_path, UsdPhysics):
    """Add a fixed joint between the world and the articulation root link."""
    from pxr import Sdf, UsdPhysics as _UsdPhysics  # noqa: N813

    joint_path = Sdf.Path("/World/fixBaseJoint")
    fixed = _UsdPhysics.FixedJoint.Define(stage, joint_path)
    fixed.CreateBody1Rel().SetTargets([Sdf.Path(articulation_path)])


if __name__ == "__main__":
    main()

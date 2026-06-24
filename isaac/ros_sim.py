"""Run the Ranger Air robot in Isaac Sim as a ROS 2 "hardware" node.

This is the simulation stand-in for the real robot's driver. It loads the robot
USD, builds a physics scene, and wires up the Isaac Sim ROS 2 bridge so the rest
of the ROS 2 graph (the m1_control brain, robot_state_publisher, RViz, ...) can
talk to it over standard topics:

    Publishes    /clock                rosgraph_msgs/Clock
                 /joint_states         sensor_msgs/JointState
    Subscribes   /m1/joint_command     sensor_msgs/JointState
                     position -> steer / lift / arms / fingers
                     velocity -> wheels

The bridge runs as OmniGraph nodes inside Isaac Sim (Isaac ships its own ROS 2
libraries), so we do NOT import rclpy here -- that sidesteps the Python 3.11
(Isaac) vs 3.12 (Jazzy) mismatch. Drive gains are configured to match the teleop
script so the arms hold their pose under gravity while position targets arrive
over ROS.

Usage (Isaac Sim's bundled Python):

    /home/jerry/isaac-sim/python.sh isaac/ros_sim.py            # windowed
    /home/jerry/isaac-sim/python.sh isaac/ros_sim.py --headless # no GUI

Make sure ROS 2 Jazzy is sourced in the same shell so the bridge picks up the
right RMW, e.g.:

    source /opt/ros/jazzy/setup.bash
    /home/jerry/isaac-sim/python.sh isaac/ros_sim.py
"""

import argparse
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_USD = os.path.join(REPO_ROOT, "assets", "usd", "ranger_air.usd")

# Topic names (kept in sync with m1_control).
JOINT_STATES_TOPIC = "joint_states"
JOINT_COMMAND_TOPIC = "m1/joint_command"

# Drive gains per joint group (mirrors isaac/teleop.py).
WHEEL_KP, WHEEL_KD = 0.0, 2500.0       # velocity drive (kp must stay 0)
STEER_KP, STEER_KD = 700.0, 180.0
LIFT_KP, LIFT_KD = 30000.0, 3000.0
# Arm position-drive stiffness. Raised 9000->30000 (KD scaled with it): at 9000
# the simulated arm sagged ~3-4 cm under gravity at a near-max-reach (nearly fully
# extended) posture, so even a perfectly-correct joint command read amber/red in
# the reach viz. The controller command is right; this just lets the SIM arm
# actually hold it (the real robot's stiff drives track far better). The lift
# already uses 30000 and is stable at 120 Hz.
ARM_KP, ARM_KD = 30000.0, 2000.0
GRIP_KP, GRIP_KD = 400.0, 40.0
PHYSICS_HZ = 120.0

WHEEL_JOINTS = ["fl_wheel_joint", "fr_wheel_joint", "rr_wheel_joint", "rl_wheel_joint"]
STEER_JOINTS = ["fl_steering_joint", "fr_steering_joint", "rr_steering_joint", "rl_steering_joint"]
LIFT_JOINT = "lift_joint"
FINGER_JOINTS = [
    "openarm_left_finger_joint1", "openarm_left_finger_joint2",
    "openarm_right_finger_joint1", "openarm_right_finger_joint2",
]

# Startup posture. The arms mount flush+low on the lift carriage, so the all-zero
# pose folds them straight down INTO the drivebase. This "arms-up ready" config
# holds both grippers up/forward, clear of the base (verified: >0.4 m clear of the
# base box, self-clearance >24 mm). Set as the initial joint state AND the drive
# targets so the stiff arm drives hold it until the brain sends a command. Keep it
# in sync with the MoveIt SRDF "home" group_states (m1.srdf).
HOME_CONFIG = {
    "lift_joint": 0.4879,
    "openarm_left_joint1": 0.3925, "openarm_left_joint2": -0.8549,
    "openarm_left_joint3": 1.2295, "openarm_left_joint4": 1.4446,
    "openarm_left_joint5": -0.0023, "openarm_left_joint6": 0.0780,
    "openarm_left_joint7": -0.4366,
    "openarm_right_joint1": -0.3883, "openarm_right_joint2": 0.7599,
    "openarm_right_joint3": -1.2386, "openarm_right_joint4": 1.4257,
    "openarm_right_joint5": 0.0019, "openarm_right_joint6": -0.0691,
    "openarm_right_joint7": 0.4254,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ranger Air as a ROS 2 sim node.")
    parser.add_argument("--usd", default=DEFAULT_USD, help="Path to the robot USD.")
    parser.add_argument("--headless", action="store_true", help="Run without a GUI window.")
    parser.add_argument("--spawn-height", type=float, default=0.1,
                        help="Height (m) to spawn the base above the ground.")
    parser.add_argument("--fix-base", action="store_true",
                        help="Pin the base in place (for arm-only work).")
    args = parser.parse_args()

    usd_path = os.path.abspath(args.usd)
    if not os.path.isfile(usd_path):
        raise FileNotFoundError(
            f"Robot USD not found: {usd_path}\n"
            f"Run: /home/jerry/isaac-sim/python.sh isaac/convert_urdf_to_usd.py")

    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_ros_sim_report.txt")
    report_lines = []

    def report(msg):
        print(msg)
        report_lines.append(str(msg))
        with open(report_path, "w") as fh:
            fh.write("\n".join(report_lines) + "\n")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {"headless": args.headless, "renderer": "RaytracedLighting"})

    # Enable the ROS 2 bridge extension before importing graph helpers.
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()

    import numpy as np
    import omni.graph.core as og
    import omni.timeline
    import omni.usd
    from isaacsim.core.prims import Articulation
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
    physx_scene.CreateTimeStepsPerSecondAttr().Set(PHYSICS_HZ)

    # --- Ground plane + light ---------------------------------------------
    PhysicsSchemaTools.addGroundPlane(
        stage, "/groundPlane", "Z", 100.0, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5))
    UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight")).CreateIntensityAttr(2500)

    # --- Reference the robot asset -----------------------------------------
    robot_root = "/World/RangerAir"
    add_reference_to_stage(usd_path, robot_root)
    robot_prim = stage.GetPrimAtPath(robot_root)
    xform = UsdGeom.Xformable(robot_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(args.spawn_height)))

    articulation_path = robot_root
    for prim in _iter_prims(stage, robot_root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_path = prim.GetPath().pathString
            break
    report(f"[ros_sim] articulation: {articulation_path}")

    if args.fix_base:
        fixed = UsdPhysics.FixedJoint.Define(stage, Sdf.Path("/World/fixBaseJoint"))
        fixed.CreateBody1Rel().SetTargets([Sdf.Path(articulation_path)])

    # --- ROS 2 bridge OmniGraph -------------------------------------------
    _build_ros2_graph(og, stage, Sdf, articulation_path, report)

    # --- Start the simulation ----------------------------------------------
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    simulation_app.update()

    robot = Articulation(articulation_path)
    robot.initialize()
    for _ in range(3):
        simulation_app.update()

    if not robot.is_physics_handle_valid():
        report(f"[ros_sim] ERROR: {articulation_path} is not a valid articulation")
        simulation_app.close()
        return

    dof_names = list(robot.dof_names)
    report(f"[ros_sim] DOF count: {len(dof_names)}")
    report(f"[ros_sim] DOF names: {dof_names}")

    # --- Drive gains -------------------------------------------------------
    name_to_idx = {n: i for i, n in enumerate(dof_names)}
    kp = np.full(len(dof_names), ARM_KP, dtype=np.float32)
    kd = np.full(len(dof_names), ARM_KD, dtype=np.float32)
    for n in WHEEL_JOINTS:
        if n in name_to_idx:
            kp[name_to_idx[n]], kd[name_to_idx[n]] = WHEEL_KP, WHEEL_KD
    for n in STEER_JOINTS:
        if n in name_to_idx:
            kp[name_to_idx[n]], kd[name_to_idx[n]] = STEER_KP, STEER_KD
    if LIFT_JOINT in name_to_idx:
        kp[name_to_idx[LIFT_JOINT]], kd[name_to_idx[LIFT_JOINT]] = LIFT_KP, LIFT_KD
    for n in FINGER_JOINTS:
        if n in name_to_idx:
            kp[name_to_idx[n]], kd[name_to_idx[n]] = GRIP_KP, GRIP_KD
    robot.set_gains(kps=kp.reshape(1, -1), kds=kd.reshape(1, -1))
    report("[ros_sim] drive gains configured (arms stiff, wheels velocity-driven)")

    # --- Startup posture: hold the arms up, clear of the drivebase ----------
    # All-zero folds the (now flush+low) arms into the base, so seed an arms-up
    # ready pose. Set BOTH the physical state and the drive targets so the stiff
    # arm drives hold it until /m1/joint_command takes over.
    home = np.zeros(len(dof_names), dtype=np.float32)
    for n, v in HOME_CONFIG.items():
        if n in name_to_idx:
            home[name_to_idx[n]] = v
    for a in ("left", "right"):           # mimic: finger_joint2 follows finger_joint1
        f1, f2 = f"openarm_{a}_finger_joint1", f"openarm_{a}_finger_joint2"
        if f1 in name_to_idx and f2 in name_to_idx:
            home[name_to_idx[f2]] = home[name_to_idx[f1]]
    home2d = home.reshape(1, -1)
    robot.set_joint_positions(home2d)
    try:
        robot.set_joint_position_targets(home2d)
    except AttributeError:
        from isaacsim.core.utils.types import ArticulationAction
        robot.apply_action(ArticulationAction(joint_positions=home2d))
    for _ in range(3):
        simulation_app.update()
    report("[ros_sim] startup posture set (arms up, clear of base)")

    report(f"[ros_sim] publishing /{JOINT_STATES_TOPIC} and /clock")
    report(f"[ros_sim] subscribing /{JOINT_COMMAND_TOPIC}")
    report("[ros_sim] running. Ctrl-C in the terminal or close the window to stop.")

    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        pass
    finally:
        timeline.stop()
        simulation_app.close()
        report("[ros_sim] done.")


def _build_ros2_graph(og, stage, Sdf, articulation_path, report):
    """Create the action graph: clock + joint-state pub + joint-command sub."""
    keys = og.Controller.Keys
    try:
        og.Controller.edit(
            {"graph_path": "/M1RosGraph", "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnTick", "omni.graph.action.OnPlaybackTick"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                    ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                    ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                    ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ],
                keys.CONNECT: [
                    ("OnTick.outputs:tick", "PublishClock.inputs:execIn"),
                    ("OnTick.outputs:tick", "PublishJointState.inputs:execIn"),
                    ("OnTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                    ("OnTick.outputs:tick", "ArticulationController.inputs:execIn"),
                    ("Context.outputs:context", "PublishClock.inputs:context"),
                    ("Context.outputs:context", "PublishJointState.inputs:context"),
                    ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                    ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                    ("SubscribeJointState.outputs:jointNames",
                        "ArticulationController.inputs:jointNames"),
                    ("SubscribeJointState.outputs:positionCommand",
                        "ArticulationController.inputs:positionCommand"),
                    ("SubscribeJointState.outputs:velocityCommand",
                        "ArticulationController.inputs:velocityCommand"),
                    ("SubscribeJointState.outputs:effortCommand",
                        "ArticulationController.inputs:effortCommand"),
                ],
                keys.SET_VALUES: [
                    ("PublishClock.inputs:topicName", "clock"),
                    ("PublishJointState.inputs:topicName", JOINT_STATES_TOPIC),
                    ("SubscribeJointState.inputs:topicName", JOINT_COMMAND_TOPIC),
                ],
            },
        )
        # The joint-state publisher and articulation controller need their
        # targetPrim relationship pointed at the articulation root. The
        # og.Controller.set_target_prims helper is not present in every Isaac
        # build, so set the USD relationship directly (version-robust).
        for node in ("PublishJointState", "ArticulationController"):
            node_prim = stage.GetPrimAtPath(f"/M1RosGraph/{node}")
            rel = node_prim.GetRelationship("inputs:targetPrim")
            if not rel:
                rel = node_prim.CreateRelationship("inputs:targetPrim", custom=False)
            rel.SetTargets([Sdf.Path(articulation_path)])
        report("[ros_sim] ROS 2 bridge graph built at /M1RosGraph")
    except Exception as exc:  # noqa: BLE001
        report(f"[ros_sim] ERROR building ROS 2 graph: {exc}")
        raise


def _iter_prims(stage, root_path):
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return
    yield root
    for child in root.GetChildren():
        yield from _iter_prims(stage, child.GetPath().pathString)


if __name__ == "__main__":
    main()

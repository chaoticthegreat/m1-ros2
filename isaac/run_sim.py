"""Load the Ranger Air USD into an Isaac Sim physics scene and run a simulation.

Examples:

    # Headless smoke test (no window), 300 physics steps:
    /home/jerry/isaac-sim/python.sh isaac/run_sim.py --headless --steps 300

    # Interactive window with a simple wheel-drive demo:
    /home/jerry/isaac-sim/python.sh isaac/run_sim.py --demo

If the USD asset does not exist yet, run isaac/convert_urdf_to_usd.py first.
"""

import argparse
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_USD = os.path.join(REPO_ROOT, "assets", "usd", "ranger_air.usd")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate the Ranger Air robot.")
    parser.add_argument("--usd", default=DEFAULT_USD, help="Path to the robot USD.")
    parser.add_argument("--headless", action="store_true", help="Run without a GUI window.")
    parser.add_argument("--steps", type=int, default=600, help="Number of frames to simulate.")
    parser.add_argument(
        "--spawn-height",
        type=float,
        default=0.1,
        help="Height (m) to spawn the base above the ground plane.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Spin the wheels with a velocity drive to show the robot moving.",
    )
    args = parser.parse_args()

    usd_path = os.path.abspath(args.usd)
    if not os.path.isfile(usd_path):
        raise FileNotFoundError(
            f"Robot USD not found: {usd_path}\n"
            f"Run: /home/jerry/isaac-sim/python.sh isaac/convert_urdf_to_usd.py"
        )

    # Isaac Sim redirects Python stdout, so mirror key status to a report file.
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_run_report.txt")
    report_lines = []

    def report(msg):
        print(msg)
        report_lines.append(str(msg))
        with open(report_path, "w") as fh:
            fh.write("\n".join(report_lines) + "\n")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {"headless": args.headless, "renderer": "RaytracedLighting"}
    )

    import omni.timeline
    import omni.usd
    from isaacsim.core.prims import Articulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, UsdLux, UsdPhysics

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

    # --- Ground plane + light (local, no asset server needed) --------------
    PhysicsSchemaTools.addGroundPlane(
        stage, "/groundPlane", "Z", 100.0, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5)
    )
    distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
    distant_light.CreateIntensityAttr(2500)

    # --- Reference the robot asset -----------------------------------------
    robot_root = "/World/RangerAir"
    add_reference_to_stage(usd_path, robot_root)

    # Spawn the base slightly above the ground so wheels settle onto the plane.
    from pxr import UsdGeom

    robot_prim = stage.GetPrimAtPath(robot_root)
    xform = UsdGeom.Xformable(robot_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(args.spawn_height)))

    # Find the prim carrying the articulation root (fall back to the robot root).
    articulation_path = robot_root
    for prim in Usd_iter(stage, robot_root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_path = prim.GetPath().pathString
            break

    # --- Optional wheel-drive demo -----------------------------------------
    if args.demo:
        n_wheels = _setup_wheel_drive_demo(stage, robot_root, UsdPhysics)
        report(f"[sim] wheel-drive demo: configured {n_wheels} wheel joints")

    # --- Start the simulation ----------------------------------------------
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    simulation_app.update()

    robot = Articulation(articulation_path)
    robot.initialize()

    report(f"[sim] usd: {usd_path}")
    report(f"[sim] articulation path: {articulation_path}")
    if robot.is_physics_handle_valid():
        report("[sim] articulation handle: VALID")
        try:
            dof_names = list(robot.dof_names)
            report(f"[sim] DOF count: {len(dof_names)}")
            report(f"[sim] DOF names: {dof_names}")
        except Exception as exc:  # noqa: BLE001
            report(f"[sim] could not read DOF names: {exc}")
    else:
        report(f"[sim] WARNING: {articulation_path} is not a valid articulation handle")

    report(f"[sim] stepping {args.steps} frames (headless={args.headless})...")
    for _ in range(args.steps):
        simulation_app.update()
    report("[sim] stepping complete")

    timeline.stop()
    simulation_app.close()
    report("[sim] done.")


def Usd_iter(stage, root_path):
    """Yield the prim at root_path and all of its descendants."""
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return
    yield root
    for child in root.GetChildren():
        yield from Usd_iter(stage, child.GetPath().pathString)


def _setup_wheel_drive_demo(stage, robot_root, UsdPhysics):
    """Apply an angular velocity drive to the four wheel joints."""
    wheel_joint_names = [
        "fl_wheel_joint",
        "fr_wheel_joint",
        "rl_wheel_joint",
        "rr_wheel_joint",
    ]
    found = 0
    for prim in Usd_iter(stage, robot_root):
        name = prim.GetName()
        if name in wheel_joint_names:
            drive = UsdPhysics.DriveAPI.Get(prim, "angular")
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive.CreateTargetVelocityAttr().Set(200.0)  # deg/s
            drive.CreateDampingAttr().Set(2000.0)
            drive.CreateStiffnessAttr().Set(0.0)
            found += 1
    return found


if __name__ == "__main__":
    main()

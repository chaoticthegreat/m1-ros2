"""Convert the Ranger Air URDF into a USD asset using the Isaac Sim URDF importer.

Run with the Isaac Sim python environment, e.g.:

    cd /home/jerry/Downloads/M1-visualizer
    /home/jerry/isaac-sim/python.sh isaac/convert_urdf_to_usd.py

The resulting USD is written to assets/usd/ranger_air.usd by default.
"""

import argparse
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URDF = os.path.join(
    REPO_ROOT, "assets", "ranger_air_description", "urdf", "ranger_air_description.urdf"
)
DEFAULT_USD = os.path.join(REPO_ROOT, "assets", "usd", "ranger_air.usd")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Ranger Air URDF to USD.")
    parser.add_argument("--urdf", default=DEFAULT_URDF, help="Path to the input URDF.")
    parser.add_argument("--usd", default=DEFAULT_USD, help="Path to the output USD.")
    parser.add_argument(
        "--fix-base",
        action="store_true",
        help="Pin the base in place (default: mobile/free base).",
    )
    args = parser.parse_args()

    urdf_path = os.path.abspath(args.urdf)
    usd_path = os.path.abspath(args.usd)
    os.makedirs(os.path.dirname(usd_path), exist_ok=True)

    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    # SimulationApp must be created before importing any omni/isaacsim modules.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})

    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.asset.importer.urdf")
    simulation_app.update()

    import omni.kit.commands
    from pxr import Usd

    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    # Keep all links/frames (lidars, imu, camera mounts) instead of collapsing them.
    import_config.merge_fixed_joints = False
    # Mobile robot: base is free unless --fix-base is passed.
    import_config.fix_base = args.fix_base
    import_config.import_inertia_tensor = True
    import_config.distance_scale = 1.0
    import_config.make_default_prim = True
    # Don't bake a physics scene into the reusable robot asset; the sim script adds it.
    import_config.create_physics_scene = False
    # Honor the gripper <mimic> joints.
    import_config.parse_mimic = True
    import_config.self_collision = False
    import_config.convex_decomp = False

    print(f"[convert] importing {urdf_path}")
    print(f"[convert] writing  {usd_path}")

    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=import_config,
        dest_path=usd_path,
        get_articulation_root=True,
    )

    print(f"[convert] import status: {status}")
    print(f"[convert] articulation root prim: {prim_path}")

    # Sanity check: open the written stage and report the prim tree summary.
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open written USD: {usd_path}")
    default_prim = stage.GetDefaultPrim()
    n_prims = sum(1 for _ in stage.Traverse())
    print(f"[convert] default prim: {default_prim.GetPath() if default_prim else None}")
    print(f"[convert] total prims in stage: {n_prims}")

    simulation_app.close()
    print("[convert] done.")


if __name__ == "__main__":
    main()

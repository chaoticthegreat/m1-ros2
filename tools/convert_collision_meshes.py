#!/usr/bin/env python3
"""Convert the URDF COLLISION meshes Drake can't hull (.dae/.stl) to .obj, so the
reach IK plant can load real collision geometry (Drake's MakeConvexHull accepts
only .obj/.vtk/.gltf). Writes <name>.obj alongside each original. Run OFFLINE with
a throwaway trimesh venv (NOT the ROS interpreter):

  python3 -m venv /tmp/meshconv && /tmp/meshconv/bin/pip install trimesh pycollada
  /tmp/meshconv/bin/python tools/convert_collision_meshes.py

Only the links relevant to ARM self-collision are converted (arms + drivebase +
lift); wheels/steering/lidar/camera/imu collision is skipped (the IK strips it).
"""
import os
import xml.etree.ElementTree as ET

import trimesh

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(REPO, "assets/ranger_air_description/urdf/ranger_air_description.urdf")
SKIP = ("wheel", "steering", "lidar", "camera", "imu", "finger")


def resolve(pkg_uri):
    # package://PKG/REST -> assets/PKG/REST
    assert pkg_uri.startswith("package://"), pkg_uri
    rest = pkg_uri[len("package://"):]
    return os.path.join(REPO, "assets", rest)


def main():
    r = ET.fromstring(open(URDF).read())
    want = set()
    for lk in r.iter("link"):
        n = lk.get("name", "")
        for c in lk.findall("collision"):           # convert ALL collision meshes
            m = c.find("geometry/mesh")
            if m is not None:
                want.add(m.get("filename"))
    print(f"{len(want)} unique relevant collision meshes to convert")
    ok = 0
    for uri in sorted(want):
        src = resolve(uri)
        if not os.path.isfile(src):
            print(f"  MISSING {src}"); continue
        dst = os.path.splitext(src)[0] + ".obj"
        try:
            loaded = trimesh.load(src, force="mesh")
            if isinstance(loaded, trimesh.Scene):
                loaded = trimesh.util.concatenate(
                    [g for g in loaded.geometry.values()])
            loaded.export(dst)
            ok += 1
            print(f"  {os.path.basename(src):24s} -> {os.path.basename(dst)} "
                  f"({len(loaded.vertices)}v {len(loaded.faces)}f)")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {src}: {e}")
    print(f"converted {ok}/{len(want)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build-time converter: URDF visual meshes -> decimated glTF for the Quest viz.

This is an *offline* tool, not part of the robot runtime. It reads the combined
robot URDF, and for every link's ``<visual>`` mesh it:

  * loads the mesh (DAE/STL, via trimesh + pycollada),
  * bakes the URDF ``<origin>`` (link->visual transform) and the ``<mesh
    scale>`` into the geometry, so each output mesh is expressed directly in its
    *link frame* (the client then only needs the per-link FK transform),
  * flips the triangle winding when the scale mirrors (negative determinant,
    e.g. the right-arm ``1 -1 1``) so normals/back-face culling stay correct,
  * decimates to a triangle budget so the whole robot runs on the Quest GPU,
  * exports a self-contained ``.glb`` (geometry + normals embedded).

Identical (file, scale, origin) visuals are converted once and shared. A
``manifest.json`` maps each link instance to its ``.glb`` URL; the Quest page
loads those with three.js and poses each link every frame from the server's FK.

This only needs to be re-run if the robot meshes or URDF change. The output is
committed, so a normal checkout already has the meshes. To run it, make a
throwaway venv with the conversion deps (kept out of the repo):

    python3 -m venv .meshconv_venv
    .meshconv_venv/bin/pip install "trimesh>=4" pycollada fast-simplification numpy
    .meshconv_venv/bin/python tools/convert_meshes.py
    # then rebuild so colcon copies the new meshes into the package:
    (cd ros2_ws && colcon build --symlink-install --packages-select m1_control)

Outputs into ``ros2_ws/src/m1_control/m1_control/web_assets/`` (vendored so the
node serves them without a separate asset pipeline).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(
    REPO, "assets", "ranger_air_description", "urdf", "ranger_air_description.urdf"
)
ASSETS = os.path.join(REPO, "assets")
OUT_DIR = os.path.join(
    REPO, "ros2_ws", "src", "m1_control", "m1_control", "web_assets"
)
MESH_OUT = os.path.join(OUT_DIR, "meshes")

# Per-mesh triangle budget. The full-res visuals (e.g. ~8.7 MB steering DAEs)
# are far too heavy for a mobile XR GPU; this keeps each link light while still
# clearly readable as the real part.
FACE_BUDGET = 3500


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _resolve(filename: str) -> str:
    """package://<pkg>/<rel> -> assets/<pkg>/<rel> (also handles plain paths)."""
    if filename.startswith("package://"):
        rel = filename[len("package://"):]
        return os.path.join(ASSETS, rel)
    return filename


def _load_mesh(path: str) -> trimesh.Trimesh | None:
    """Load a mesh path into a single concatenated Trimesh (or None)."""
    try:
        loaded = trimesh.load(path, force="mesh", process=False)
    except Exception as exc:  # noqa: BLE001
        print(f"    ! load failed: {exc}")
        return None
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            return None
        loaded = trimesh.util.concatenate(geoms)
    if not isinstance(loaded, trimesh.Trimesh) or loaded.faces.shape[0] == 0:
        return None
    return loaded


def _bake(mesh: trimesh.Trimesh, scale: np.ndarray, xyz: np.ndarray,
          rpy: np.ndarray) -> trimesh.Trimesh:
    """Apply M = T(xyz) @ R(rpy) @ S(scale) to the geometry (link-frame bake)."""
    m = mesh.copy()
    M = np.eye(4)
    M[:3, :3] = _rpy(*rpy) @ np.diag(scale)
    M[:3, 3] = xyz
    m.apply_transform(M)
    # A mirroring transform (det<0) inverts triangle winding; flip it back so
    # outward normals and back-face culling remain correct.
    if np.linalg.det(M[:3, :3]) < 0:
        m.invert()
    return m


def _weld(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Merge coincident vertices so the mesh has real shared-edge topology.

    Meshes are loaded with ``process=False`` (so the DAE scene concatenation is
    left untouched), which leaves STL/DAE geometry as a *triangle soup*: every
    triangle owns its own copy of each vertex with no shared edges. Quadric
    decimation collapses edges, so on a triangle soup it has nothing to collapse
    coherently and instead emits giant degenerate slivers spanning the whole
    part (and holes) -- which is why heavily-decimated parts like the lift column
    rendered as invisible/shattered fragments. Welding first gives the decimator
    a connected surface so it simplifies cleanly.
    """
    m = mesh.copy()
    try:
        m.merge_vertices()
    except Exception as exc:  # noqa: BLE001
        print(f"    ! weld failed ({exc}); using unwelded mesh")
        return mesh
    return m


def _decimate(mesh: trimesh.Trimesh, budget: int) -> trimesh.Trimesh:
    # Weld first (see ``_weld``): decimating an unwelded triangle soup shatters
    # the part into spikes/holes -- the bug that left the lift column invisible.
    # On a welded mesh the decimator may floor above the budget (it preserves the
    # topology of multi-body CAD parts); that's fine -- the result is correct and
    # still far lighter than full res, which matters more than hitting the budget.
    mesh = _weld(mesh)
    if mesh.faces.shape[0] <= budget:
        return mesh
    try:
        out = mesh.simplify_quadric_decimation(face_count=budget)
    except Exception as exc:  # noqa: BLE001
        print(f"    ! decimate failed ({exc}); keeping welded full res")
        return mesh
    if out is None or out.faces.shape[0] == 0:
        print("    ! decimation produced empty mesh; keeping welded full res")
        return mesh
    return out


def main() -> int:
    if not os.path.isfile(URDF):
        print(f"URDF not found: {URDF}")
        return 1
    os.makedirs(MESH_OUT, exist_ok=True)

    root = ET.parse(URDF).getroot()
    manifest_links = []
    cache: dict[str, str] = {}        # dedup key -> mesh url
    converted = 0
    total_in = total_out = 0

    for link in root.findall("link"):
        lname = link.attrib["name"]
        for vis in link.findall("visual"):
            mesh_el = vis.find("geometry/mesh")
            if mesh_el is None:
                continue
            fname = mesh_el.attrib["filename"]
            scale = np.array(
                [float(v) for v in mesh_el.attrib.get("scale", "1 1 1").split()]
            )
            origin = vis.find("origin")
            xyz = np.zeros(3)
            rpy = np.zeros(3)
            if origin is not None:
                if "xyz" in origin.attrib:
                    xyz = np.array([float(v) for v in origin.attrib["xyz"].split()])
                if "rpy" in origin.attrib:
                    rpy = np.array([float(v) for v in origin.attrib["rpy"].split()])

            key = f"{fname}|{scale.tolist()}|{xyz.tolist()}|{rpy.tolist()}"
            url = cache.get(key)
            if url is None:
                src = _resolve(fname)
                if not os.path.isfile(src):
                    print(f"  - {lname}: missing source {src}")
                    continue
                print(f"  + {lname}: {os.path.basename(fname)}")
                mesh = _load_mesh(src)
                if mesh is None:
                    continue
                in_faces = mesh.faces.shape[0]
                mesh = _bake(mesh, scale, xyz, rpy)
                mesh = _decimate(mesh, FACE_BUDGET)
                _ = mesh.vertex_normals  # ensure normals are present in the GLB
                h = hashlib.sha1(key.encode()).hexdigest()[:10]
                out_name = f"{h}.glb"
                out_path = os.path.join(MESH_OUT, out_name)
                mesh.export(out_path, file_type="glb")
                url = f"/meshes/{out_name}"
                cache[key] = url
                converted += 1
                total_in += os.path.getsize(src)
                total_out += os.path.getsize(out_path)
                print(f"      {in_faces} -> {mesh.faces.shape[0]} faces, "
                      f"{os.path.getsize(out_path)/1024:.0f} KB")
            manifest_links.append({"link": lname, "mesh": url})

    manifest = {"root": "base_link", "links": manifest_links}
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)

    print(
        f"\nconverted {converted} unique meshes for {len(manifest_links)} link "
        f"instances\n  input {total_in/1048576:.1f} MB -> output "
        f"{total_out/1048576:.1f} MB\n  manifest: {os.path.join(OUT_DIR, 'manifest.json')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

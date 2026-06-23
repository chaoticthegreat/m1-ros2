#!/usr/bin/env python3
"""Build-time converter: URDF visual meshes -> decimated, *coloured* glTF for the
Quest viz.

This is an *offline* tool, not part of the robot runtime. It reads the combined
robot URDF, and for every link's ``<visual>`` mesh it:

  * loads the mesh (DAE/STL, via trimesh + pycollada) **keeping its per-solid
    materials** -- a CAD DAE like the Agilex base is an assembly of hundreds of
    separate solids (white body, black tyres/trim, red accents, grey hubs), and
    each solid carries its own diffuse colour,
  * bakes each solid's material colour onto its vertices (vertex colours), so the
    final single-mesh export still carries the real part colours instead of one
    flat grey -- this is what makes the base read as the actual robot,
  * bakes the URDF ``<origin>`` (link->visual transform) and the ``<mesh
    scale>`` into the geometry, so each output mesh is expressed directly in its
    *link frame* (the client then only needs the per-link FK transform),
  * flips the triangle winding when the scale mirrors (negative determinant,
    e.g. the right-arm ``1 -1 1``) so normals/back-face culling stay correct,
  * decimates to a triangle budget so the whole robot runs on the Quest GPU, then
    transfers the colours back onto the decimated vertices by nearest-neighbour
    (quadric decimation does not carry vertex attributes),
  * exports a self-contained ``.glb`` (geometry + normals + vertex colours).

Identical (file, scale, origin) visuals are converted once and shared. A
``manifest.json`` maps each link instance to its ``.glb`` URL; the Quest page
loads those with three.js and poses each link every frame from the server's FK.

This only needs to be re-run if the robot meshes or URDF change. The output is
committed, so a normal checkout already has the meshes. To run it, make a
throwaway venv with the conversion deps (kept out of the repo):

    python3 -m venv .meshconv_venv
    .meshconv_venv/bin/pip install "trimesh>=4" pycollada fast-simplification numpy scipy
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
from scipy.spatial import cKDTree

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

# Fallback colour for a solid with no usable material (RGBA, 0-255). A neutral
# light grey, matching what the old material-less pipeline rendered everything as.
DEFAULT_COLOR = np.array([200, 200, 200, 255], dtype=np.uint8)


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


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB display values (0..1) -> linear (0..1).

    The CAD/DAE material colours are authored as sRGB display colours, but glTF
    ``COLOR_0`` vertex colours are **linear** (three.js r161 has colour
    management on and encodes the lit result back to sRGB on output). Baking the
    sRGB value straight in would wash the darks out to grey; converting to linear
    here makes the headset reproduce the true part colours (proper black tyres).
    """
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _material_color(geom: trimesh.Trimesh) -> np.ndarray:
    """Best-effort **linear** RGBA (0-255 uint8) for one source solid's material.

    Handles both glTF-style PBR materials (``baseColorFactor``, 0..1 floats) and
    classic materials (``diffuse``, usually 0..255), and falls back to any
    vertex colours already on the solid, then to a neutral grey. The RGB is
    converted sRGB->linear (see :func:`_srgb_to_linear`) so the headset shows the
    real base colours (black tyres, red accents, white body) instead of a flat
    grey blob.
    """
    def _to_linear_u8(c: np.ndarray) -> np.ndarray:
        """sRGB RGBA in 0..1 -> linear RGBA uint8 (alpha left linear/unchanged)."""
        c = np.asarray(c, dtype=np.float64).reshape(-1)
        if c.size == 3:
            c = np.concatenate([c, [1.0]])
        rgb = _srgb_to_linear(c[:3])
        out = np.concatenate([rgb, [np.clip(c[3], 0.0, 1.0)]])
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)

    vis = getattr(geom, "visual", None)
    if vis is not None:
        mat = getattr(vis, "material", None)
        if mat is not None:
            for attr in ("baseColorFactor", "diffuse"):
                c = getattr(mat, attr, None)
                if c is None:
                    continue
                c = np.asarray(c, dtype=np.float64).reshape(-1)
                if c.size < 3:
                    continue
                if c.max() > 1.0 + 1e-6:  # 0..255 -> 0..1
                    c = c / 255.0
                return _to_linear_u8(c)
        # Already-coloured solids (rare for CAD DAEs): take the first colour.
        vc = getattr(vis, "vertex_colors", None)
        if vc is not None and len(vc):
            return _to_linear_u8(np.asarray(vc[0], dtype=np.float64) / 255.0)
    return _to_linear_u8(DEFAULT_COLOR.astype(np.float64) / 255.0)


def _load_colored(path: str) -> trimesh.Trimesh | None:
    """Load a mesh into one Trimesh, baking each solid's material as vertex colour.

    Loaded with ``process=False`` so the DAE scene's per-solid geometry and
    materials survive (force='mesh' would flatten the materials away). Each solid
    gets a uniform vertex colour from its material, then all solids are
    concatenated -- so the merged mesh keeps the multi-colour appearance of the
    real part.

    Each solid's scene **node transform is applied** (``Scene.graph``) before
    concatenating. A COLLADA/glTF file places its solids via node matrices that
    also carry the file's up-axis convention -- e.g. every OpenArm visual DAE is
    authored ``Y_UP`` with a +90deg-about-X node matrix that lifts it into the ROS
    ``Z_UP`` link frame. Iterating ``Scene.geometry`` alone takes each solid's raw
    geometry in its *local* frame and silently drops that matrix, which baked every
    such mesh rotated -90deg about X: in the headset the in-between arm links
    (link1..link6, the arm base) pointed the wrong way, while RViz -- which honours
    the node transform through assimp -- looked correct. Walking the graph lands
    each solid in the link frame exactly as RViz renders it; meshes whose graph is
    identity (or plain STLs with no scene) are unaffected.
    """
    try:
        loaded = trimesh.load(path, process=False)
    except Exception as exc:  # noqa: BLE001
        print(f"    ! load failed: {exc}")
        return None

    # (solid geometry, its world transform in the file's scene). For a bare STL
    # (no scene graph) the transform is identity.
    placed = []
    if isinstance(loaded, trimesh.Scene):
        for node in loaded.graph.nodes_geometry:
            world, gname = loaded.graph[node]
            g = loaded.geometry.get(gname)
            if isinstance(g, trimesh.Trimesh) and g.faces.shape[0]:
                placed.append((g, np.asarray(world, dtype=np.float64)))
    elif isinstance(loaded, trimesh.Trimesh) and loaded.faces.shape[0]:
        placed.append((loaded, np.eye(4)))
    if not placed:
        return None

    colored = []
    for g, world in placed:
        col = _material_color(g)        # material read in the solid's own frame
        g = g.copy()
        g.apply_transform(world)        # place it in the link frame (up-axis baked)
        g.visual = trimesh.visual.ColorVisuals(
            mesh=g, vertex_colors=np.tile(col, (len(g.vertices), 1)))
        colored.append(g)
    merged = trimesh.util.concatenate(colored)
    if not isinstance(merged, trimesh.Trimesh) or merged.faces.shape[0] == 0:
        return None
    return merged


def _bake(mesh: trimesh.Trimesh, scale: np.ndarray, xyz: np.ndarray,
          rpy: np.ndarray) -> trimesh.Trimesh:
    """Apply M = T(xyz) @ R(rpy) @ S(scale) to the geometry (link-frame bake)."""
    m = mesh.copy()
    M = np.eye(4)
    M[:3, :3] = _rpy(*rpy) @ np.diag(scale)
    M[:3, 3] = xyz
    m.apply_transform(M)
    # A mirroring transform (det<0) inverts triangle winding; flip it back so
    # outward normals and back-face culling remain correct. Vertex colours are
    # per-vertex and unaffected by the winding flip.
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
    a connected surface so it simplifies cleanly. ``merge_vertices`` keeps the
    per-solid vertex colours (it only merges vertices that also share a colour).
    """
    m = mesh.copy()
    try:
        m.merge_vertices()
    except Exception as exc:  # noqa: BLE001
        print(f"    ! weld failed ({exc}); using unwelded mesh")
        return mesh
    return m


def _vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    vis = getattr(mesh, "visual", None)
    vc = getattr(vis, "vertex_colors", None) if vis is not None else None
    if vc is None or len(vc) != len(mesh.vertices):
        return None
    return np.asarray(vc).copy()


def _decimate(mesh: trimesh.Trimesh, budget: int) -> trimesh.Trimesh:
    # Weld first (see ``_weld``): decimating an unwelded triangle soup shatters
    # the part into spikes/holes -- the bug that left the lift column invisible.
    # On a welded mesh the decimator may floor above the budget (it preserves the
    # topology of multi-body CAD parts); that's fine -- the result is correct and
    # still far lighter than full res, which matters more than hitting the budget.
    mesh = _weld(mesh)
    ref_pts = mesh.vertices.copy()
    ref_col = _vertex_colors(mesh)

    out = mesh
    if mesh.faces.shape[0] > budget:
        try:
            out = mesh.simplify_quadric_decimation(face_count=budget)
        except Exception as exc:  # noqa: BLE001
            print(f"    ! decimate failed ({exc}); keeping welded full res")
            out = mesh
        if out is None or out.faces.shape[0] == 0:
            print("    ! decimation produced empty mesh; keeping welded full res")
            out = mesh

    # Quadric decimation drops vertex attributes, so re-attach colours by
    # nearest source vertex. Colour regions are spatially coherent (a solid is
    # one colour), so nearest-neighbour transfer is exact except within a
    # triangle of a colour boundary -- imperceptible at viz scale.
    if ref_col is not None and len(out.vertices):
        if len(out.vertices) == len(ref_pts) and out is mesh:
            new_col = ref_col
        else:
            _, idx = cKDTree(ref_pts).query(out.vertices)
            new_col = ref_col[idx]
        out = out.copy()
        out.visual = trimesh.visual.ColorVisuals(mesh=out, vertex_colors=new_col)
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
                mesh = _load_colored(src)
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
                # Cache-bust on CONTENT: the .glb filename hashes the URDF key
                # (file/scale/origin), so re-baking the same link keeps the same
                # name -- a Quest browser that cached the old bytes would keep
                # serving them. Append a short content hash as a query param (the
                # node strips the query when serving the file), so a changed mesh
                # gets a fresh URL and reloads, while unchanged meshes stay cached.
                with open(out_path, "rb") as fh:
                    content_hash = hashlib.sha1(fh.read()).hexdigest()[:8]
                url = f"/meshes/{out_name}?v={content_hash}"
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

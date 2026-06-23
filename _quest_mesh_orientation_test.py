#!/usr/bin/env python3
"""Guard the Quest-viz robot-mesh ORIENTATION.

The headset viz poses each link's baked ``.glb`` at the link frame the runtime FK
(`UrdfModel.link_transforms`) computes, so every mesh must already sit in its link
frame -- exactly the orientation RViz shows. The baked meshes are produced offline
by ``tools/convert_meshes.py``.

The bug this guards: ``convert_meshes._load_colored`` used to iterate
``Scene.geometry`` and concatenate the raw per-solid geometry **without the COLLADA
scene-graph node transform**. Every OpenArm visual DAE is authored ``Y_UP`` with a
+90deg-about-X node matrix that lifts it into the ROS ``Z_UP`` link frame; dropping
that matrix baked every arm link (`link1..link6`, the arm base, both arms) rotated
-90deg about X. In the headset the in-between arm links pointed the wrong way while
RViz -- which honours the node transform through assimp -- looked correct. The fix
walks the scene graph and applies each solid's world transform before baking.

Two layers:

* **Part A (always runs, stdlib only)** -- parse every baked ``.glb`` and assert its
  axis-aligned bounds match a committed reference captured from a known-good build.
  A -90deg-about-X regression swaps the Y/Z extents of the elongated arm links by
  centimetres, far outside the millimetre tolerance, so a revert is caught even in a
  plain ROS env with no mesh-conversion deps.

* **Part B (runs only if ``trimesh`` is importable, e.g. the mesh-conversion venv)**
  -- re-derive each link's correct orientation straight from the source DAE the way
  assimp/RViz do (scene graph applied), then confirm the baked mesh matches that
  orientation *uniquely*: the identity (no extra rotation) is a strictly better fit
  than any 90/180/270deg rotation about any axis. This ties the baked meshes to the
  URDF source rather than to hard-coded numbers, and is sensitive to a 180deg flip
  that bounds alone cannot see.

Run:  ``python3 _quest_mesh_orientation_test.py``
      (Part B also needs trimesh:  ``.meshconv_venv/bin/python _quest_mesh_orientation_test.py``)
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys

def _mesh_file(url: str) -> str:
    """``/meshes/<hash>.glb?v=<content>`` -> ``<hash>.glb`` (strip dir + cache-bust)."""
    return os.path.basename(url.split("?", 1)[0])


REPO = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(REPO, "ros2_ws", "src", "m1_control", "m1_control", "web_assets")
MESH_DIR = os.path.join(WEB, "meshes")
MANIFEST = os.path.join(WEB, "manifest.json")
URDF = os.path.join(
    REPO, "assets", "ranger_air_description", "urdf", "ranger_air_description.urdf"
)
ASSETS = os.path.join(REPO, "assets")

# AABB [min xyz, max xyz] of every baked mesh, captured from a verified-correct
# build (link frame, metres). The arm links carry the Y_UP->Z_UP node transform, so
# a -90deg-about-X regression swaps their Y and Z extents well beyond TOL_A.
REF_BOUNDS = {
    "9829cd6fad.glb": [[-0.2755, -0.2495, 0.0431], [0.2745, 0.2533, 0.2648]],
    "116dc20fcb.glb": [[-0.006, -0.006, -0.0], [0.006, 0.006, 0.003]],
    "4e6ecf1819.glb": [[-0.0274, -0.027, -0.0309], [0.0278, 0.0276, 0.0132]],
    "359e389e7e.glb": [[-0.0274, -0.0281, -0.0306], [0.0281, 0.0292, 0.0119]],
    "d910ac203c.glb": [[-0.017, -0.0402, -0.0114], [0.005, 0.0402, 0.0125]],
    "dcb4ea42c6.glb": [[-0.0442, -0.0495, -0.095], [0.0438, 0.0759, 0.0959]],
    "76ce0cda02.glb": [[-0.0604, -0.0605, -0.068], [0.0604, 0.0605, 0.045]],
    "7232b420cd.glb": [[-0.0415, -0.0757, -0.0954], [0.0437, 0.0489, 0.0955]],
    "221b87946c.glb": [[-0.0604, -0.0605, -0.045], [0.0604, 0.0605, 0.068]],
    "78f85391ba.glb": [[-0.0435, -0.0757, -0.0954], [0.0445, 0.0497, 0.0955]],
    "54ae45e73e.glb": [[-0.0604, -0.0605, -0.045], [0.0604, 0.0605, 0.068]],
    "e6dfb5f91b.glb": [[-0.0445, -0.0497, -0.0954], [0.0435, 0.0757, 0.0955]],
    "316df2e68d.glb": [[-0.0604, -0.0605, -0.068], [0.0604, 0.0605, 0.045]],
    "3fba1bc6a0.glb": [[-0.2302, -0.26, -0.016], [0.228, 0.175, 1.244]],
    "b65fd926d3.glb": [[-0.0825, 0.0498, 0.2863], [0.0475, 0.2248, 0.4084]],
    "b6a19a0437.glb": [[-0.0847, -0.0815, 0.0], [0.0652, 0.085, 0.773]],
    "514d760237.glb": [[-0.0492, -0.0, -0.061], [0.0496, 0.0625, 0.061]],
    "d498603198.glb": [[-0.0461, -0.0044, -0.0499], [0.049, 0.1096, 0.0498]],
    "365fd9fefa.glb": [[-0.0484, -0.0491, -0.0677], [0.0351, 0.0492, 0.049]],
    "3dd77c62aa.glb": [[-0.0429, -0.0328, -0.183], [0.0356, 0.0411, 0.0007]],
    "88bb767263.glb": [[-0.0425, -0.0394, -0.0974], [0.0325, 0.0392, 0.0285]],
    "c9ca6b262f.glb": [[-0.0383, -0.0358, -0.1302], [0.0383, 0.0375, 0.0007]],
    "ed2e33bab8.glb": [[-0.038, -0.0375, -0.0286], [0.0281, 0.0375, 0.0287]],
    "7231cfe347.glb": [[-0.0325, -0.0477, -0.1106], [0.0457, 0.0527, 0.0178]],
    "caa8ee3386.glb": [[-0.0275, -0.0214, -0.1019], [0.0275, 0.018, 0.0155]],
    "41dd34a6a4.glb": [[-0.0275, -0.018, -0.1019], [0.0275, 0.0214, 0.0155]],
    "df568cb0e1.glb": [[-0.0492, -0.0625, -0.061], [0.0496, 0.0, 0.061]],
    "0a1aae2dd8.glb": [[-0.0461, -0.1096, -0.0499], [0.049, 0.0044, 0.0498]],
    "a4964de0ed.glb": [[-0.0484, -0.0492, -0.0677], [0.0351, 0.0491, 0.049]],
    "1ec7ab88b4.glb": [[-0.0383, -0.0375, -0.1302], [0.0383, 0.0358, 0.0007]],
    "9cdf2151c9.glb": [[-0.038, -0.0375, -0.0286], [0.0281, 0.0375, 0.0287]],
    "20297bff60.glb": [[-0.0325, -0.0527, -0.1106], [0.0457, 0.0477, 0.0178]],
    "efc3b77208.glb": [[-0.0275, -0.018, -0.1019], [0.0275, 0.0214, 0.0155]],
    "84b0277ccd.glb": [[-0.0275, -0.0214, -0.1019], [0.0275, 0.018, 0.0155]],
}

# Part A build-to-build decimation tolerance (m). A 90deg rotation of an elongated
# arm link swaps Y/Z extents by 4-18 cm, an order of magnitude above this. NB an
# axis-aligned box is symmetric under a 180deg rotation about its own axes, so Part A
# alone cannot see a 180deg flip of a near-symmetric link -- that case is covered by
# Part B (which includes 180deg candidates and requires a unique identity match).
TOL_A = 0.010

# Part B: max mean nearest-neighbour distance (m) from a baked (decimated) mesh's
# vertices to the source-DAE surface in the CORRECT orientation. Budget: quadric
# decimation (~0.5-2 mm) + bake/export rounding (<0.2 mm); the worst observed fit is
# ~0.8 mm. A wrong rotation lands vertices off-surface at cm scale, far above this.
TOL_B_IDENTITY = 0.003


# ---------------------------------------------------------------- GLB parsing (stdlib)
_JSON, _BIN = 0x4E4F534A, 0x004E4942


def _glb_chunks(path: str):
    with open(path, "rb") as fh:
        data = fh.read()
    magic, _ver, length = struct.unpack("<III", data[:12])
    if magic != 0x46546C67:
        raise ValueError(f"{path}: not a GLB")
    off, out = 12, {}
    while off < length:
        clen, ctype = struct.unpack("<II", data[off : off + 8])
        out[ctype] = data[off + 8 : off + 8 + clen]
        off += 8 + clen
    return json.loads(out[_JSON].decode("utf-8")), out.get(_BIN, b"")


def _glb_bounds(path):
    js, _ = _glb_chunks(path)
    acc = js["accessors"]
    lo = [1e9, 1e9, 1e9]
    hi = [-1e9, -1e9, -1e9]
    for m in js["meshes"]:
        for prim in m["primitives"]:
            a = acc[prim["attributes"]["POSITION"]]
            lo = [min(lo[i], a["min"][i]) for i in range(3)]
            hi = [max(hi[i], a["max"][i]) for i in range(3)]
    return lo, hi


def _glb_node_transforms_identity(js) -> bool:
    """The client poses each mesh by the link FK alone, so the GLB's own nodes must
    not carry a rotation/translation (vertices are pre-baked in the link frame)."""
    for n in js.get("nodes", []):
        if "matrix" in n:
            m = n["matrix"]
            ident = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
            if any(abs(m[i] - ident[i]) > 1e-6 for i in range(16)):
                return False
        if "rotation" in n and any(
            abs(v - r) > 1e-6 for v, r in zip(n["rotation"], [0, 0, 0, 1])
        ):
            return False
        if "translation" in n and any(abs(v) > 1e-6 for v in n["translation"]):
            return False
    return True


def _glb_verts(path):
    js, binc = _glb_chunks(path)
    bv, acc = js["bufferViews"], js["accessors"]
    out = []
    for m in js["meshes"]:
        for prim in m["primitives"]:
            a = acc[prim["attributes"]["POSITION"]]
            v = bv[a["bufferView"]]
            o = v.get("byteOffset", 0) + a.get("byteOffset", 0)
            n = a["count"]
            raw = binc[o : o + n * 12]
            out.extend(struct.unpack("<%df" % (n * 3), raw))
    pts = [out[i : i + 3] for i in range(0, len(out), 3)]
    return pts


# ------------------------------------------------------------------------- Part A
def part_a() -> int:
    fails = []
    if not os.path.isfile(MANIFEST):
        print(f"FAIL: manifest missing: {MANIFEST}")
        return 1
    man = json.load(open(MANIFEST))
    links = man.get("links", [])
    if not links:
        print("FAIL: manifest has no links")
        return 1

    seen = set()
    for e in links:
        mb = _mesh_file(e["mesh"])
        path = os.path.join(MESH_DIR, mb)
        if not os.path.isfile(path):
            fails.append(f"{e['link']}: baked mesh missing ({mb})")
            continue
        if mb in seen:
            continue
        seen.add(mb)
        try:
            js, _ = _glb_chunks(path)
            lo, hi = _glb_bounds(path)
        except Exception as exc:  # noqa: BLE001
            fails.append(f"{mb}: parse failed ({exc})")
            continue
        if not _glb_node_transforms_identity(js):
            fails.append(f"{mb}: GLB node carries a non-identity transform")
        ref = REF_BOUNDS.get(mb)
        if ref is None:
            fails.append(f"{mb}: no reference bounds (new mesh? regenerate REF_BOUNDS)")
            continue
        err = max(
            max(abs(lo[i] - ref[0][i]) for i in range(3)),
            max(abs(hi[i] - ref[1][i]) for i in range(3)),
        )
        if err > TOL_A:
            fails.append(
                f"{mb}: bounds off by {err*1000:.1f} mm > {TOL_A*1000:.0f} mm "
                f"(orientation regression?) got lo={[round(v,4) for v in lo]} "
                f"hi={[round(v,4) for v in hi]} want {ref}"
            )

    n = len(seen)
    if fails:
        print(f"Part A: FAILED ({len(fails)} issue(s) over {n} meshes):")
        for f in fails:
            print("  -", f)
        return 1
    print(f"Part A: OK -- {n} baked meshes match the reference orientation "
          f"(<= {TOL_A*1000:.0f} mm), all link nodes identity, every manifest link present.")
    return 0


# ------------------------------------------------------------------------- Part B
def _rotmat(axis, deg):
    import numpy as np

    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rpy(r, p, y):
    import numpy as np

    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def part_b() -> int:
    try:
        import numpy as np
        import trimesh
        from scipy.spatial import cKDTree
    except Exception as exc:  # noqa: BLE001
        print(f"Part B: SKIPPED -- needs trimesh/scipy ({exc}). "
              f"Run with the mesh-conversion venv to verify against source DAEs.")
        return 0

    import xml.etree.ElementTree as ET

    if not os.path.isfile(URDF):
        print(f"Part B: SKIPPED -- URDF not found ({URDF})")
        return 0

    def resolve(fn):
        return os.path.join(ASSETS, fn[len("package://"):]) if fn.startswith("package://") else fn

    man = json.load(open(MANIFEST))
    link2mesh = {e["link"]: _mesh_file(e["mesh"]) for e in man["links"]}
    cands = [("identity", np.eye(3))] + [
        (f"{ax}{d}", _rotmat(ax, d)) for ax in "xyz" for d in (90, 180, 270)
    ]
    root = ET.parse(URDF).getroot()
    fails = []
    worst = 0.0
    for link in root.findall("link"):
        ln = link.attrib["name"]
        if ln not in link2mesh:
            continue
        me = link.find("visual/geometry/mesh")
        if me is None:
            continue
        src = resolve(me.attrib["filename"])
        if not os.path.isfile(src):
            continue
        scale = np.array([float(v) for v in me.attrib.get("scale", "1 1 1").split()])
        o = link.find("visual/origin")
        xyz = np.array([float(v) for v in (o.attrib.get("xyz", "0 0 0").split() if o is not None else [0, 0, 0])])
        rp = np.array([float(v) for v in (o.attrib.get("rpy", "0 0 0").split() if o is not None else [0, 0, 0])])
        sc = trimesh.load(src, process=False)
        ref = sc.to_geometry() if isinstance(sc, trimesh.Scene) else sc.copy()
        M = np.eye(4)
        M[:3, :3] = _rpy(*rp) @ np.diag(scale)
        M[:3, 3] = xyz
        exp = ref.copy()
        exp.apply_transform(M)
        tree = cKDTree(exp.vertices)
        V = np.array(_glb_verts(os.path.join(MESH_DIR, link2mesh[ln])))
        c = exp.bounds.mean(0)
        dists = {}
        for nm, R in cands:
            Pr = (V - c) @ R + c
            d, _ = tree.query(Pr)
            dists[nm] = float(np.mean(d))
        order = sorted(dists.items(), key=lambda kv: kv[1])
        idd = dists["identity"]
        worst = max(worst, idd)
        best_wrong = min(v for k, v in dists.items() if k != "identity")
        # identity must be the strict best fit and clearly under any wrong rotation.
        if order[0][0] != "identity" or idd > TOL_B_IDENTITY or best_wrong <= idd:
            fails.append(
                f"{ln}: best={order[0][0]} id={idd*1000:.2f}mm best_wrong={best_wrong*1000:.2f}mm"
            )

    if fails:
        print(f"Part B: FAILED ({len(fails)} link(s) not uniquely matching RViz orientation):")
        for f in fails:
            print("  -", f)
        return 1
    print(f"Part B: OK -- every baked link mesh uniquely matches the source-DAE "
          f"(RViz) orientation; worst identity fit {worst*1000:.2f} mm, "
          f"strictly better than any 90/180/270deg rotation.")
    return 0


def main() -> int:
    print("== Quest-viz mesh orientation test ==")
    a = part_a()
    b = part_b()
    rc = a or b
    print("\nRESULT:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())

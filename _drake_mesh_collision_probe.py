#!/usr/bin/env /usr/bin/python3
"""Offline: load the REAL collision meshes into a Drake plant and report the
signed-distance pairs at (a) the live jammed MEASURED config and (b) the
unconstrained IK SOLUTION config for the same failing target.

Purpose: the repo capsule CollisionModel reports +24mm at the jam (it misses the
real contact), so before adding collision constraints to the live IK we must know
WHICH bodies actually collide and whether the IK's own solution config is the
colliding one (-> constraining the solution fixes it) or not. Also proves whether
mesh collision is viable in this Drake build (convexity/package resolution).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "ros2_ws/src/m1_control")

URDF = ("ros2_ws/install/ranger_air_description/share/ranger_air_description/"
        "urdf/ranger_air_description.urdf")
INSTALL_SHARE = "ros2_ws/install"
SRC_ASSETS = "assets"
BASE_LINK = "base_link"


import xml.etree.ElementTree as ET


def collision_urdf(urdf_xml):
    """Strip visual; repoint ALL collision mesh ext -> .obj; drop finger
    joints+links+mimic. (Includes every body so we can see what really collides.)"""
    root = ET.fromstring(urdf_xml)
    for link in root.findall("link"):
        n = link.get("name", "")
        for v in list(link.findall("visual")):
            link.remove(v)
        for c in list(link.findall("collision")):
            if "finger" in n:
                link.remove(c); continue
            m = c.find("geometry/mesh")
            if m is not None:
                fn = m.get("filename")
                m.set("filename", os.path.splitext(fn)[0] + ".obj")
    drop = set()
    for j in list(root.findall("joint")):
        if "finger" in j.attrib.get("name", "") or j.find("mimic") is not None:
            ch = j.find("child")
            if ch is not None:
                drop.add(ch.attrib["link"])
            root.remove(j)
    for lk in list(root.findall("link")):
        if lk.attrib.get("name") in drop:
            root.remove(lk)
    return ET.tostring(root, encoding="unicode")


def build_mesh_plant():
    from pydrake.systems.framework import DiagramBuilder
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
    from pydrake.multibody.parsing import Parser
    builder = DiagramBuilder()
    plant, sg = AddMultibodyPlantSceneGraph(builder, 0.0)
    parser = Parser(plant)
    pm = parser.package_map()
    for root in (SRC_ASSETS, INSTALL_SHARE):
        if os.path.isdir(root):
            try:
                pm.PopulateFromFolder(root)
            except Exception as e:  # noqa: BLE001
                print(f"  (package populate {root}: {e})")
    parser.AddModelsFromString(collision_urdf(open(URDF).read()), "urdf")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName(BASE_LINK))
    plant.Finalize()
    diagram = builder.Build()
    ctx = diagram.CreateDefaultContext()
    pctx = plant.GetMyContextFromRoot(ctx)
    return plant, sg, diagram, ctx, pctx


def set_q(plant, pctx, qdict):
    for ji in plant.GetJointIndices():
        j = plant.get_joint(ji)
        if j.num_positions() == 1:
            try:
                j.set_angle(pctx, float(qdict.get(j.name(), 0.0)))
            except Exception:
                # prismatic / generic 1-dof
                qv = plant.GetPositions(pctx).copy()
                qv[j.position_start()] = float(qdict.get(j.name(), 0.0))
                plant.SetPositions(pctx, qv)


def report_distances(plant, sg, diagram, ctx, pctx, qdict, label, topn=8):
    from pydrake.geometry import QueryObject
    set_q(plant, pctx, qdict)
    diagram.ForcedPublish(ctx)
    qo = plant.get_geometry_query_input_port().Eval(pctx)
    inspector = qo.inspector()
    pairs = qo.ComputeSignedDistancePairwiseClosestPoints(2.0)
    rows = []
    for p in pairs:
        nA = plant.GetBodyFromFrameId(inspector.GetFrameId(p.id_A)).name()
        nB = plant.GetBodyFromFrameId(inspector.GetFrameId(p.id_B)).name()
        rows.append((p.distance, nA, nB))
    rows.sort(key=lambda r: r[0])
    print(f"\n[{label}] closest {topn} body pairs (signed distance m; <0 = penetrating):")
    for d, a, b in rows[:topn]:
        flag = "  <-- COLLIDING" if d < 0 else ("  (touching)" if d < 0.005 else "")
        print(f"   {d:+.4f}  {a:32s} {b}{flag}")
    return rows


def main():
    # failing config(s) from the live log
    rows = [json.loads(l) for l in open("/tmp/m1_ros_log/reach_failures.jsonl") if l.strip()]
    if not rows:
        print("no failure log"); return
    by = {}
    for r in rows:
        by.setdefault(r["arm"], r)
    meas = {j: d["q"] for r in by.values() for j, d in r["joints"].items()}
    target = {a: np.array(by[a]["target"], float) for a in by}
    print("failing targets:", {a: list(t) for a, t in target.items()})

    print("\n=== building Drake plant WITH real collision meshes ===")
    try:
        plant, sg, diagram, ctx, pctx = build_mesh_plant()
        print(f"  OK: {plant.num_positions()} positions, mesh collision available")
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc()
        print(f"  MESH PLANT FAILED: {e}")
        return

    from m1_control.kinematics import UrdfModel, ARM_JOINTS, LIFT_JOINT
    # neutral reference config (arms out of the way, mid lift) -- a non-jamming pose
    neutral = {j: 0.0 for a in ("left", "right") for j in ARM_JOINTS[a]}
    neutral[LIFT_JOINT] = 0.4

    jam_rows = report_distances(plant, sg, diagram, ctx, pctx, meas,
                                "MEASURED (jammed) config")
    neu_rows = report_distances(plant, sg, diagram, ctx, pctx, neutral,
                                "NEUTRAL (zeros, lift 0.4) config")

    # config-DEPENDENT collisions: colliding at jam, NOT at neutral (or much worse)
    neu = {(a, b): d for d, a, b in neu_rows}
    neu.update({(b, a): d for d, a, b in neu_rows})
    print("\n=== CONFIG-DEPENDENT (colliding/near at jam, clear at neutral) ===")
    found = False
    for d, a, b in sorted(jam_rows):
        if d > 0.02:
            break
        nd = neu.get((a, b))
        if nd is None or nd > d + 0.02:   # meaningfully worse at the jam
            print(f"   jam={d:+.4f}  neutral={nd if nd is None else round(nd,4)}  "
                  f"{a} <-> {b}")
            found = True
    if not found:
        print("   NONE -- the only contacts are CONSTANT (mount overlap), present at"
              " neutral too. The jam is NOT an arm self-collision among these bodies.")


if __name__ == "__main__":
    main()

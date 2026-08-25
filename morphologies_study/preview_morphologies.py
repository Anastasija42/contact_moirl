"""preview_morphologies.py — show all 4 URDFs (human, neanderthal, naledi, chimp)
side-by-side in one MeshCat scene at their neutral pose.

Each model is offset by 0.6 m along X so they're visible next to each other.
Rotate / pan / zoom in the browser. Ctrl-C to stop.

Usage:
    python preview_morphologies.py
"""
from __future__ import annotations

import sys
import time
from copy import deepcopy
from pathlib import Path
import math

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

REPO = Path(__file__).resolve().parent

MORPHOLOGIES = [
    ("human",        "human_model/urdf/human.urdf",            0.0, [0.30, 0.55, 0.95, 0.95], 1.000, 1.000),
    ("neanderthal",  "human_model/urdf/homo_neanderthal.urdf", 0.8, [0.90, 0.35, 0.30, 0.95], 1.094, 1.193),
    ("naledi",       "human_model/urdf/homo_naledi.urdf",      1.6, [0.30, 0.80, 0.40, 0.95], 0.927, 0.909),
    ("chimp",        "human_model/urdf/chimp.urdf",            2.4, [0.70, 0.40, 0.85, 0.95], 0.972, 1.023),
]

SPREAD_AXIS = "y"

HIDE_KEYWORDS = [
    "upperleg", "lowerleg", "foot", "ankle",
    "pelvis", "abdomen", "hip",
]

def main():
    print("[preview] loading all 4 morphologies into one MeshCat scene...")
    primary_viz = None

    for i, (name, rel_path, dx, color, body_scale, torso_width_scale) in enumerate(MORPHOLOGIES):
        urdf = REPO / rel_path
        if not urdf.exists():
            print(f"  [{name}] SKIP: {urdf} not found")
            continue

        m       = pin.buildModelFromUrdf(str(urdf), pin.JointModelFreeFlyer())
        vis_m   = pin.buildGeomFromUrdf(m, str(urdf), pin.GeometryType.VISUAL)
        col_m   = pin.buildGeomFromUrdf(m, str(urdf), pin.GeometryType.COLLISION)

        if i == 0:
            print(f"     ALL visual geometry objects:")
            for g in vis_m.geometryObjects:
                print(f"       - {g.name}")

        to_remove = [g.name for g in vis_m.geometryObjects
                     if any(kw in g.name.lower() for kw in HIDE_KEYWORDS)]
        for gname in to_remove:
            try:
                vis_m.removeGeometryObject(gname)
            except Exception as e:
                print(f"     warn: could not remove {gname}: {e}")
        print(f"     removed {len(to_remove)} parts: "
              f"{[g for g in to_remove[:6]] + (['…'] if len(to_remove)>6 else [])}")

        for g in vis_m.geometryObjects:
            gname = g.name.lower()
            is_torso = ("torso" in gname or "thorax" in gname)
            try:
                if is_torso:
                    s = np.array([torso_width_scale, 1.0, torso_width_scale])
                    g.meshScale = g.meshScale * s
                else:
                    g.meshScale = g.meshScale * body_scale
            except Exception:
                pass
            try:
                g.meshColor = np.array(color, dtype=np.float64)
            except Exception:
                pass
        print(f"     applied body_scale={body_scale:.3f}, torso_width={torso_width_scale:.3f}")

        print(f"\n  ── [{name}] DEBUG ────────────────────────────────")
        print(f"     nq={m.nq}  nv={m.nv}  njoints={m.njoints}")
        print(f"     joints[0..3]: " + ", ".join(
            f"{m.names[j]}(nq={m.joints[j].nq}, idx_q={m.joints[j].idx_q})"
            for j in range(min(4, m.njoints))))
        if m.joints[1].nq == 7:
            print(f"     joint[1] is a FREE-FLYER (nq=7: x,y,z,qx,qy,qz,qw) — "
                  f"base is movable")
        elif m.joints[1].nq == 0:
            print(f"     joint[1] is FIXED — no movable base. q[0..6] are NOT base pose.")
        else:
            print(f"     joint[1] nq={m.joints[1].nq} — unusual base type")

        q = pin.neutral(m).astype(np.float64)
        print(f"     neutral q[:7] = {q[:7]}")

        if m.joints[1].nq == 7:
            axis_idx = {"x": 0, "y": 1, "z": 2}[SPREAD_AXIS]
            q[axis_idx] = dx
            q[2] = 1.0
            rotation_matrix = pin.utils.rpyToMatrix(math.pi / 2, 0, 0)
            quaternion = pin.Quaternion(rotation_matrix)
            q[3] = quaternion.x
            q[4] = quaternion.y
            q[5] = quaternion.z
            q[6] = quaternion.w
            print(f"     spread along {SPREAD_AXIS} at offset {dx} m")
            print(f"     after snippet, q[:7] = {q[:7]}")
        else:
            print(f"     skipping base pose setup (no free-flyer base)")

        try:
            data = m.createData()
            pin.forwardKinematics(m, data, q)
            pin.updateFramePlacements(m, data)
            first_joint = data.oMi[1].translation
            print(f"     joint[1] world pos after FK: {np.round(first_joint, 3)}")
            if m.njoints > 5:
                joint5 = data.oMi[min(5, m.njoints-1)].translation
                print(f"     joint[5] world pos after FK: {np.round(joint5, 3)}")
        except Exception as e:
            print(f"     FK error: {e}")

        for g in vis_m.geometryObjects:
            try:
                g.meshColor = np.array(color, dtype=np.float64)
            except Exception:
                pass

        col_np = np.array(color, dtype=np.float64)
        if i == 0:
            primary_viz = MeshcatVisualizer(m, col_m, vis_m)
            primary_viz.initViewer(open=True)
            try:
                primary_viz.loadViewerModel(rootNodeName=name,
                                            visual_color=col_np)
            except TypeError:
                primary_viz.loadViewerModel(rootNodeName=name,
                                            color=col_np)
            primary_viz.display(q)
            url = (primary_viz.viewer.url() if hasattr(primary_viz.viewer, 'url')
                   else 'see terminal output above')
            print(f"  [{name}] loaded at x={dx:.1f} m  (URL: {url})")
        else:
            viz = MeshcatVisualizer(m, col_m, vis_m)
            viz.initViewer(primary_viz.viewer)
            try:
                viz.loadViewerModel(rootNodeName=name, visual_color=col_np)
            except TypeError:
                viz.loadViewerModel(rootNodeName=name, color=col_np)
            viz.display(q)
            print(f"  [{name}] loaded at x={dx:.1f} m")

    if primary_viz is None:
        print("[preview] no URDFs loaded.")
        return 1

    print("\n[preview] all morphologies displayed. Rotate with mouse; Ctrl-C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[preview] stopped")
    return 0

if __name__ == "__main__":
    sys.exit(main())

"""
view_species_meshcat.py
=======================
Play back the solved forward-transfer trajectories (run_species_forward.py) in
MeshCat so you can SEE how the same human cost moves each morphology.

By default all requested species are loaded into ONE shared scene, overlaid at
the same origin in distinct colours, and animated together -- the clearest way
to watch where the strokes diverge. Use --spread to fan them out side-by-side
instead.

This reuses the exact reduced OCP model (arm + torso stub + rock) that produced
the trajectory, so what you see is what the solver optimised.

Run it yourself (it opens a browser and blocks):
    conda run -n unified_env python view_species_meshcat.py --task down_long
    conda run -n unified_env python view_species_meshcat.py --task down_long --spread 0.7
    conda run -n unified_env python view_species_meshcat.py --task up_long --species chimp human
Ctrl-C to stop.
"""
import argparse
import time

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from run_csqp_identifiability import load_geometry
from run_species_forward import build_species, load_wstar

SPECIES = ["human", "chimp", "homo_naledi", "homo_neanderthal"]
SCALE = {
    "human": (1.000, 1.000), "chimp": (0.972, 1.023),
    "homo_naledi": (0.927, 0.909), "homo_neanderthal": (1.094, 1.193),
}

HIDE_VIS = ("leg", "foot", "pelvis", "left_")

def style_visual(visual_model, rgba, species, hide_irrelevant=True):
    """Tint per species and rescale the torso width + limbs so the otherwise
    identical (shared-mesh) torso/head stop looking weird across taxa. Legs and
    the left arm are hidden by default (they are locked and misleading). The
    stick keeps a bright brown so the workpiece stays readable."""
    body_s, torso_s = SCALE.get(species, (1.0, 1.0))
    for g in visual_model.geometryObjects:
        nm = g.name.lower()
        if hide_irrelevant and any(h in nm for h in HIDE_VIS):
            g.meshColor = np.array([0.5, 0.5, 0.5, 0.0])
            continue
        try:
            if "torso" in nm or "thorax" in nm or "abdomen" in nm:
                g.meshScale = g.meshScale * np.array([torso_s, 1.0, torso_s])
            elif "stick" not in nm and "rock" not in nm:
                g.meshScale = g.meshScale * body_s
        except Exception:
            pass
        if "stick" in nm:
            g.meshColor = np.array([0.55, 0.36, 0.20, 0.95])
        else:
            g.meshColor = np.array(rgba, float)

def place_stick_at_tip(human, q0, tip_extra, surface_gap):
    """Move the stick visual so it meets the rock at its FAR working tip (the end
    away from the hand) instead of at its base. The away-from-hand axis is
    wrist->rock; the tip is `tip_extra` m beyond the rock origin along it, and the
    stick is set a hair (`surface_gap`) off the tip so surfaces touch, not
    interpenetrate. Cosmetic — does not change the contact constraint."""
    if tip_extra <= 0 and surface_gap <= 0:
        return
    m, d = human.pin_model, human.pin_model.createData()
    pin.framesForwardKinematics(m, d, np.asarray(q0, float))
    pin.updateFramePlacements(m, d)
    rock = d.oMf[m.getFrameId("rock_frame")].translation
    wrist = d.oMi[m.getJointId("right_wrist_X")].translation
    away = rock - wrist
    away = away / (np.linalg.norm(away) + 1e-12)
    tip = rock + tip_extra * away
    rail = human.p_end_world - human.p_start_world
    rail = rail / (np.linalg.norm(rail) + 1e-12)
    radial = away - np.dot(away, rail) * rail
    rn = np.linalg.norm(radial)
    radial = radial / rn if rn > 1e-9 else np.array([0.0, 0.0, -1.0])
    for g in human.visual_model.geometryObjects:
        if "stick" in g.name.lower():
            g.placement.translation = tip + surface_gap * radial

def set_stick_segment(human, p_start, p_end):
    """Place the stick visual as a cylinder spanning p_start->p_end in the world
    (used for the fixed/shared stick in Experiment A, so it stays put while the
    hand reaches toward it rather than being dragged onto the rock)."""
    p0, p1 = np.asarray(p_start, float), np.asarray(p_end, float)
    mid = 0.5 * (p0 + p1)
    z = p1 - p0
    z = z / (np.linalg.norm(z) + 1e-12)
    x = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y = np.cross(z, x); y /= np.linalg.norm(y) + 1e-12
    x = np.cross(y, z)
    R = np.column_stack([x, y, z])
    for g in human.visual_model.geometryObjects:
        if "stick" in g.name.lower():
            g.placement = pin.SE3(R, mid)

COLOR = {"human": [0.000, 0.447, 0.698, 0.90],
         "homo_naledi": [0.902, 0.624, 0.000, 0.90],
         "homo_neanderthal": [0.835, 0.333, 0.000, 0.90],
         "chimp": [0.000, 0.620, 0.451, 0.90],
         "bonobo": [0.800, 0.475, 0.655, 0.90],
         "australopithecus_prometheus": [0.337, 0.706, 0.914, 0.90],
         "australopithecus_sediba": [0.941, 0.894, 0.259, 0.90]}

MOVING_STICK = False

def load_model_and_traj(sp, geom_subject, task, cycle, w_run, outdir, inverse=False):
    import os
    q0_full, q_traj, dt = load_geometry(geom_subject, task, cycle)
    if inverse:
        sp_dir = os.path.join(outdir, sp)
        npz = os.path.join(sp_dir, "inverse.npz")
        if not os.path.exists(npz):
            npz = os.path.join(sp_dir, "authored.npz")
        if not os.path.exists(npz):
            raise FileNotFoundError(f"{sp_dir}/(inverse|authored).npz -- run "
                                    f"run_species_inverse.py first")
        data = np.load(npz, allow_pickle=True)
        q = np.asarray(data["q_traj"])
        nq = int(data["nq"])
        human = build_species(sp, geom_subject, task, q0_full, q_traj, dt, w_run)
        set_stick_segment(human, data["p_start"], data["p_end"])
        return human, q, nq, float(data["dt"]), False, True
    npz = os.path.join(outdir, f"{sp}__{task}", "forward.npz")
    if not os.path.exists(npz):
        raise FileNotFoundError(f"{npz} -- run run_species_forward.py for {task} first")
    data = np.load(npz, allow_pickle=True)
    xs = data["xs"]
    nq = int(data["nq"])
    if MOVING_STICK:
        human = build_species(sp, geom_subject, task, q0_full, q_traj, dt, w_run)
        return human, xs, nq, float(data["dt"]), False, True
    is_fixed = "fixed_stick" in outdir.lower()
    is_neutral = (bool(data["neutral_start"]) if "neutral_start" in data.files
                  else "neutral" in outdir.lower())
    if is_fixed:
        human = build_species(sp, geom_subject, task, q0_full, q_traj, dt, w_run)
        set_stick_segment(human, data["p_start"], data["p_end"])
        return human, xs, nq, float(data["dt"]), True, True
    if is_neutral:
        nq0 = (data["q0_used"] if "q0_used" in data.files
               else np.asarray(xs[0][:nq], float))
        st = (data["p_start"], data["p_end"]) if "p_start" in data.files else None
        human = build_species(sp, geom_subject, task, q0_full, q_traj, dt, w_run,
                              neutral_q0=nq0, stick=st)
    else:
        stick = (data["p_start"], data["p_end"]) if "p_start" in data.files else None
        human = build_species(sp, geom_subject, task, q0_full, q_traj, dt, w_run, stick=stick)
        if stick is not None:
            # so draw it there and mark placed -- do NOT let place_stick_at_tip
            set_stick_segment(human, data["p_start"], data["p_end"])
            return human, xs, nq, float(data["dt"]), True, True
    return human, xs, nq, float(data["dt"]), True, False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="down_long")
    ap.add_argument("--species", nargs="+", default=SPECIES)
    ap.add_argument("--geom_subject", default="S3")
    ap.add_argument("--cycle", type=int, default=5)
    ap.add_argument("--weights",
                    default="analysis/moirl/felix_dl/csqp_irl__nw1__windowed/weights.json")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--inverse", action="store_true",
                    help="play the hand-authored Experiment-B movements")
    ap.add_argument("--no_scale", action="store_true",
                    help="disable per-species torso/limb visual rescaling")
    ap.add_argument("--show_all", action="store_true",
                    help="keep legs + left arm (default: hidden as a false image)")
    ap.add_argument("--spread", type=float, default=0.0,
                    help="metres of offset between species (0 = overlay)")
    ap.add_argument("--spread_axis", choices=["x", "y", "z"], default="y",
                    help="axis to fan species along (x/y/z world). Pick whichever "
                         "looks side-by-side from your camera (default y)")
    ap.add_argument("--stick_tip", type=float, default=0.05,
                    help="place the stick this far past the rock origin toward its "
                         "far working tip (m)")
    ap.add_argument("--stick_offset", type=float, default=0.02,
                    help="extra surface gap so the stick touches, not penetrates (m)")
    ap.add_argument("--fps", type=float, default=0.0,
                    help="playback fps (0 = use the trajectory dt)")
    ap.add_argument("--generated", action="store_true",
                    help="load the 7-taxa generated URDFs (human_model/urdf/generated); "
                         "required for bonobo / A. sediba / A. prometheus")
    ap.add_argument("--moving_stick", action="store_true",
                    help="do not draw a static stick; let the tool/rock ride the solved "
                         "trajectory so the stick MOVES with the scrape")
    args = ap.parse_args()
    if args.generated:
        import run_species_forward as _rsf
        _rsf.URDF_DIR = _rsf.GEN_URDF_DIR
        print(f"[urdf] using generated set: {_rsf.URDF_DIR}")
    global MOVING_STICK
    MOVING_STICK = args.moving_stick
    if not args.outdir:
        args.outdir = ("analysis/species_experiments/inverse" if args.inverse
                       else "analysis/species_experiments")

    w_run, _ = load_wstar(args.weights)
    entries = []
    shared_viewer = None
    for i, sp in enumerate(args.species):
        try:
            human, xs, nq, dt, _, stick_placed = load_model_and_traj(
                sp, args.geom_subject, args.task, args.cycle, w_run, args.outdir,
                inverse=args.inverse)
        except Exception as e:
            print(f"[skip {sp}] {e}")
            continue
        if not stick_placed:
            place_stick_at_tip(human, xs[0, :nq], args.stick_tip, args.stick_offset)
        if not args.no_scale:
            style_visual(human.visual_model, COLOR.get(sp, [0.3, 0.3, 0.3, 0.85]), sp,
                         hide_irrelevant=not args.show_all)
        viz = MeshcatVisualizer(human.pin_model, human.collision_model, human.visual_model)
        if shared_viewer is None:
            viz.initViewer(open=True)
            shared_viewer = viz.viewer
        else:
            viz.initViewer(viewer=shared_viewer)
        if args.no_scale:
            try:
                viz.loadViewerModel(rootNodeName=sp, color=COLOR.get(sp))
            except TypeError:
                viz.loadViewerModel(rootNodeName=sp)
        else:
            viz.loadViewerModel(rootNodeName=sp)
        if args.spread:
            ax = {"x": 0, "y": 1, "z": 2}[args.spread_axis]
            T = np.eye(4); T[ax, 3] = i * args.spread
            try:
                viz.viewer[sp].set_transform(T)
            except Exception:
                pass
        entries.append((sp, viz, xs, nq, dt))
        print(f"[loaded] {sp}: {xs.shape[0]} frames  dt={dt:.4f}")

    if not entries:
        raise SystemExit("nothing to display")

    Tmax = max(e[2].shape[0] for e in entries)
    print(f"\n[viewer] open in browser. Looping {Tmax} frames "
          f"({len(entries)} species). Ctrl-C to stop.\n")
    try:
        while True:
            for k in range(Tmax):
                for sp, viz, xs, nq, dt in entries:
                    q = xs[min(k, xs.shape[0] - 1), :nq]
                    viz.display(np.asarray(q, float))
                step = (1.0 / args.fps) if args.fps > 0 else entries[0][4]
                time.sleep(step)
            time.sleep(0.4)
    except KeyboardInterrupt:
        print("\n[viewer] stopped.")

if __name__ == "__main__":
    main()

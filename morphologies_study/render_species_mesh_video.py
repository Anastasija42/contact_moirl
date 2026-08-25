"""
render_species_mesh_video.py
============================
Headless, photoreal **textured-mesh** videos of each morphology's solved shaving
motion, rendered with MuJoCo's offscreen renderer (EGL) -- no browser, no display.

MuJoCo is used purely as a *rasterizer*: we reuse the exact per-species Pinocchio
visual model that `view_species_meshcat.py` builds from `forward.npz` (correct
meshes, per-taxon torso/limb scaling, species colour, hidden legs+left-arm, stick
placement), compute every visible mesh's world pose per frame by Pinocchio forward
kinematics, and drive one MuJoCo **mocap body per mesh**. This sidesteps both the
species-URDF MuJoCo inertia incompatibility and the reduced-vs-full q mapping --
the geometry poses come straight from the solved reduced trajectory.

Run (repo root, unified_env; EGL is set automatically):
    python morphologies_study/render_species_mesh_video.py \
        --task down_long \
        --indir <run dir> \
        --weights data/weights/recovered_down_long.npz \
        --out papers/figures/morpho/videos

Solo per-species videos by default; --overlay renders all species in one scene.
--format mp4|gif  (mp4 needs imageio-ffmpeg; gif always works).
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
import argparse

sys.path.insert(0, "src")
sys.path.insert(0, "morphologies_study")

import numpy as np
import pinocchio as pin
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

# 7-taxa generated URDFs must be selected BEFORE importing the loader helpers
import run_species_forward as _rsf
_rsf.URDF_DIR = _rsf.GEN_URDF_DIR

import view_species_meshcat as V
from run_species_forward import load_wstar

SEVEN = ["human", "homo_neanderthal", "homo_naledi", "australopithecus_sediba",
         "australopithecus_prometheus", "chimp", "bonobo"]

PRETTY = {"human": "Modern human", "homo_neanderthal": "Neanderthal",
          "homo_naledi": "H. naledi", "australopithecus_sediba": "A. sediba",
          "australopithecus_prometheus": "A. prometheus (StW 573)",
          "chimp": "Chimpanzee", "bonobo": "Bonobo"}

def _label(frames, text, sub=""):
    """Draw a species/stroke caption in the lower-left of every frame."""
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
        subfont = ImageFont.truetype("DejaVuSans.ttf", 17)
    except Exception:
        font = ImageFont.load_default(); subfont = font
    out = []
    for fr in frames:
        im = Image.fromarray(fr)
        dr = ImageDraw.Draw(im)
        h = im.height
        dr.text((22, h - 58), text, fill=(30, 30, 35), font=font)
        if sub:
            dr.text((23, h - 28), sub, fill=(90, 90, 100), font=subfont)
        out.append(np.asarray(im))
    return out

def _quat_wxyz(R):
    """Rotation matrix -> MuJoCo (w,x,y,z) quaternion."""
    c = pin.Quaternion(np.asarray(R, float)).coeffs()
    return np.array([c[3], c[0], c[1], c[2]], float)

def collect_geoms(human):
    """Return the list of visible render primitives from the styled Pinocchio
    visual model: dicts with kind ('mesh'|'cyl'), geom index, and static params."""
    vm = human.visual_model
    prims = []
    for i, g in enumerate(vm.geometryObjects):
        rgba = np.asarray(g.meshColor, float)
        if rgba[3] <= 0.01:
            continue
        mp = str(g.meshPath) if g.meshPath else ""
        is_mesh = mp and os.path.isfile(mp) and mp.lower().endswith((".stl", ".obj", ".ply"))
        if is_mesh:
            prims.append(dict(kind="mesh", gi=i, name=g.name, path=os.path.abspath(mp),
                              scale=np.asarray(g.meshScale, float).copy(),
                              rgba=rgba.copy()))
        else:
            try:
                geo = g.geometry
                r = float(getattr(geo, "radius"))
                hl = float(getattr(geo, "halfLength", getattr(geo, "length", 0.1) / 2.0))
            except Exception:
                r, hl = 0.012, 0.12
            prims.append(dict(kind="cyl", gi=i, name=g.name, radius=r, halflen=hl,
                              rgba=rgba.copy()))
    return prims

def build_mjcf(all_prims, width, height, floor_z=None):
    """Assemble an MJCF: one mocap body (+geom) per render primitive, across all
    species passed in `all_prims` (list of (species_tag, prims))."""
    assets, bodies = [], []
    body_meta = []
    mid = 0
    for tag, prims in all_prims:
        for p in prims:
            bname = f"mb_{tag}_{p['gi']}"
            rgba = "{:.3f} {:.3f} {:.3f} {:.3f}".format(*p["rgba"])
            if p["kind"] == "mesh":
                mname = f"msh_{tag}_{p['gi']}"
                sx, sy, sz = p["scale"]
                assets.append(f'<mesh name="{mname}" file="{p["path"]}" '
                              f'scale="{sx:.6f} {sy:.6f} {sz:.6f}"/>')
                geom = f'<geom type="mesh" mesh="{mname}" rgba="{rgba}" ' \
                       f'contype="0" conaffinity="0"/>'
            else:
                geom = f'<geom type="cylinder" size="{p["radius"]:.4f} {p["halflen"]:.4f}" ' \
                       f'rgba="{rgba}" contype="0" conaffinity="0"/>'
            bodies.append(f'<body name="{bname}" mocap="true" pos="0 0 0">{geom}</body>')
            body_meta.append((tag, p["gi"]))
            mid += 1
    floor = ""
    if floor_z is not None:
        floor = (f'<geom name="floor" type="plane" pos="0 0 {floor_z:.3f}" size="3 3 0.1" '
                 f'material="floor"/>')
    mjcf = f"""
<mujoco model="species_render">
  <compiler angle="radian" autolimits="true"/>
  <visual>
    <global offwidth="{width}" offheight="{height}" azimuth="140" elevation="-18"/>
    <quality shadowsize="8192" offsamples="8"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55" specular="0.15 0.15 0.15"/>
    <map znear="0.01" zfar="30"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="flat" rgb1="0.94 0.95 0.97" rgb2="0.94 0.95 0.97"
             width="8" height="8"/>
    <texture name="floortex" type="2d" builtin="checker" rgb1="0.82 0.83 0.85"
             rgb2="0.88 0.89 0.91" width="512" height="512"/>
    <material name="floor" texture="floortex" texrepeat="6 6" reflectance="0.05" specular="0.1"/>
    {chr(10).join('    ' + a for a in assets)}
  </asset>
  <worldbody>
    <light pos="0.4 -1.2 2.4" dir="-0.2 0.5 -1" directional="true"
           diffuse="0.55 0.55 0.55" specular="0.2 0.2 0.2"/>
    {floor}
    {chr(10).join('    ' + b for b in bodies)}
  </worldbody>
</mujoco>
"""
    return mjcf, body_meta

def fk_all(human, q):
    m, vm = human.pin_model, human.visual_model
    data = m.createData()
    gdata = vm.createData()
    pin.forwardKinematics(m, data, np.asarray(q, float))
    pin.updateGeometryPlacements(m, data, vm, gdata)
    return {i: gdata.oMg[i] for i in range(len(vm.geometryObjects))}

def render(args):
    w_run, _ = load_wstar(args.weights)
    species = args.species

    loaded = []
    all_prims = []
    for sp in species:
        try:
            human, xs, nq, dt, _, _ = V.load_model_and_traj(
                sp, args.geom_subject, args.task, args.cycle, w_run, args.indir)
        except Exception as e:
            print(f"[skip {sp}] {e}")
            continue
        V.style_visual(human.visual_model, V.COLOR.get(sp, [0.4, 0.4, 0.4, 1.0]), sp,
                       hide_irrelevant=True)
        alpha = 0.55 if args.overlay else 1.0
        for g in human.visual_model.geometryObjects:
            if float(g.meshColor[3]) > 0.01 and "stick" not in g.name.lower():
                g.meshColor[3] = alpha
        prims = collect_geoms(human)
        tag = sp
        loaded.append((tag, human, xs, nq, prims))
        all_prims.append((tag, prims))
        print(f"[loaded] {sp}: {xs.shape[0]} frames, {len(prims)} render prims")

    if not loaded:
        raise SystemExit("nothing to render")

    os.makedirs(args.out, exist_ok=True)

    lookat, dist = _shared_camera(loaded, args)
    print(f"[camera] lookat={np.round(lookat,3)} distance={dist:.2f} az={args.cam_az} el={args.cam_el}")

    stroke = args.task.replace("_", " ")
    if args.overlay:
        _render_scene(loaded, all_prims, args, lookat, dist,
                      out_base=os.path.join(args.out, f"species_mesh_overlay__{args.task}"),
                      label="All morphologies", sub=f"{stroke} stroke  ·  shared human cost (A), force at capacity fraction")
    else:
        for tag, human, xs, nq, prims in loaded:
            _render_scene([(tag, human, xs, nq, prims)], [(tag, prims)], args, lookat, dist,
                          out_base=os.path.join(args.out, f"{tag}__{args.task}"),
                          label=PRETTY.get(tag, tag),
                          sub=f"{stroke} stroke  ·  shared human cost (A), force at capacity fraction")

def _shared_camera(loaded, args):
    """One lookat + absolute distance for ALL bodies. Frame on the union of every
    species' visible-geom point cloud over the whole stroke, then pad by a fixed
    mesh-volume margin so meshes (which extend beyond joint origins) don't crop."""
    pts = []
    for tag, human, xs, nq, prims in loaded:
        gis = [p["gi"] for p in prims]
        for k in range(0, xs.shape[0], max(1, xs.shape[0] // 10)):
            oMg = fk_all(human, xs[k, :nq])
            for gi in gis:
                pts.append(oMg[gi].translation.copy())
    pts = np.asarray(pts)
    lo, hi = pts.min(0), pts.max(0)
    center = 0.5 * (lo + hi)
    if args.cam_dist > 0:
        dist = args.cam_dist
    else:
        span = float((hi - lo).max()) + 0.45
        dist = 1.7 * span
    return center, dist

def _render_scene(loaded, all_prims, args, lookat, dist, out_base, label, sub=""):
    mjcf, body_meta = build_mjcf(all_prims, args.width, args.height,
                                 floor_z=(None if args.no_floor else None))
    m = mujoco.MjModel.from_xml_string(mjcf)
    d = mujoco.MjData(m)
    mocap_of = {}
    for (tag, gi) in body_meta:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"mb_{tag}_{gi}")
        mocap_of[(tag, gi)] = int(m.body_mocapid[bid])

    Tmax = max(e[2].shape[0] for e in loaded)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = dist
    cam.azimuth = args.cam_az
    cam.elevation = args.cam_el

    ren = mujoco.Renderer(m, height=args.height, width=args.width)

    frames = []
    for k in range(Tmax):
        for tag, human, xs, nq, prims in loaded:
            kk = min(k, xs.shape[0] - 1)
            oMg = fk_all(human, xs[kk, :nq])
            for p in prims:
                mi = mocap_of[(tag, p["gi"])]
                T = oMg[p["gi"]]
                d.mocap_pos[mi] = T.translation
                d.mocap_quat[mi] = _quat_wxyz(T.rotation)
        mujoco.mj_forward(m, d)
        ren.update_scene(d, cam)
        frames.append(ren.render().copy())

    if not args.no_label:
        frames = _label(frames, label, sub)
    _save(frames, out_base, args)
    print(f"[done] {out_base}.{ 'mp4' if args.format=='mp4' else 'gif'}  ({len(frames)} frames)")

def _save(frames, out_base, args):
    fps = args.fps
    if args.format == "mp4":
        out = out_base + ".mp4"
        try:
            imageio.mimsave(out, frames, fps=fps, quality=8, macro_block_size=None)
            return
        except Exception as e:
            print(f"[warn] mp4 failed ({e}); falling back to gif")
    out = out_base + ".gif"
    imageio.mimsave(out, frames, fps=fps, loop=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="down_long")
    ap.add_argument("--indir", required=True, help="directory of per-species forward.npz runs")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--species", nargs="+", default=SEVEN)
    ap.add_argument("--geom_subject", default="S3")
    ap.add_argument("--cycle", type=int, default=5)
    ap.add_argument("--out", default="papers/figures/morpho/videos")
    ap.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--overlay", action="store_true", help="all species in one scene")
    ap.add_argument("--no_floor", action="store_true")
    ap.add_argument("--no_label", action="store_true", help="omit the species caption")
    ap.add_argument("--cam_az", type=float, default=72.0)
    ap.add_argument("--cam_el", type=float, default=-10.0)
    ap.add_argument("--cam_dist", type=float, default=2.5,
                    help="absolute camera distance in metres (shared across bodies "
                         "so relative size shows); 0 = per-render auto-fit")
    args = ap.parse_args()
    render(args)

if __name__ == "__main__":
    main()

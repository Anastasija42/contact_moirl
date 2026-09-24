"""record_irl_meshcat.py — record an IRL run's meshcat ghost-overlay
animation as a GIF.

Mirrors the snippet in docs/results.md / MO_IRL_mocap.ipynb but uses
Playwright (headless Chromium) to capture frames programmatically.

Output: docs/assets/figures/irl_<combo>__video_<tag>.gif, where <tag>
is the run dir suffix without the `csqp_irl__` prefix
(e.g. nw3__basis, nw1__windowed, nw2__adaptive).

Usage:
    # One combo, all runs in it
    python docs/scripts/record_irl_meshcat.py --combo s2_dl

    # Several combos
    python docs/scripts/record_irl_meshcat.py --combo s3_dl s3_ul s2_dl s2_ul

    # Just one specific run
    python docs/scripts/record_irl_meshcat.py --combo s2_dl --tag csqp_irl__nw3__basis

Requirements (one-off):
    pip install playwright imageio pillow
    playwright install chromium
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import io as _io
import re
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pinocchio as pin

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Heavy imports kicked behind a quiet block so model-load chatter doesn't
# drown out the recorder log.
print("[record] importing Pinocchio + meshcat + Playwright...", flush=True)
from playwright.async_api import async_playwright          # noqa: E402
from PIL import Image                                       # noqa: E402
import imageio.v2 as imageio                                # noqa: E402
import meshcat.geometry as g                                # noqa: E402

from utils_slice_trajectories import remove_legs_from_visual_model  # noqa: E402
from moirl_config import COMBOS, DATE                       # noqa: E402

# Pull build_common_models from run_moirl_batch — same Human/HumanMPPI
# construction the IRL itself used.
import run_moirl_batch as _rmb                              # noqa: E402

OUT_DIR  = REPO / "docs/assets/figures"
ANALYSIS = REPO / "analysis/moirl"
TAG_RE   = re.compile(r"^csqp_irl__nw(\d+)__(windowed|basis|adaptive)(?:__.*)?$")


@contextlib.contextmanager
def _quiet():
    """Swallow stdout from heavy model-load chatter."""
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


def _setup_meshcat_with_ghost(human_mppi):
    """Two visualizers sharing one viewer: main = solid (IRL trajectory),
    ghost = semi-transparent grey (demo). Same as the snippet in
    MO_IRL_mocap.ipynb."""
    vis_main  = deepcopy(human_mppi.visual_model)
    vis_ghost = deepcopy(human_mppi.visual_model)
    for vm in (vis_main, vis_ghost):
        try:
            sid = vm.getGeometryId("stick_visual")
            if sid < vm.ngeoms:
                vm.removeGeometryObject("stick_visual")
        except Exception:
            pass
    # Hide legs on BOTH visualizers (main + ghost) so the recorded video
    # focuses on the arm + tool interaction, not the locked-body legs.
    vis_main  = remove_legs_from_visual_model(vis_main)
    vis_ghost = remove_legs_from_visual_model(vis_ghost)

    viz = pin.visualize.MeshcatVisualizer(
        human_mppi.pin_model, human_mppi.collision_model, vis_main)
    viz.initViewer(open=False)
    viz.loadViewerModel()

    viz_ghost = pin.visualize.MeshcatVisualizer(
        human_mppi.pin_model, human_mppi.collision_model, vis_ghost)
    viz_ghost.initViewer(viz.viewer)
    viz_ghost.loadViewerModel(rootNodeName="ghost",
                              color=np.array([0.6, 0.6, 0.6, 0.4]))

    # Standalone stick cylinder.
    axis = human_mppi.p_end_world - human_mppi.p_start_world
    axis_len = float(np.linalg.norm(axis))
    z = axis / axis_len
    x = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y = np.cross(z, x); y /= np.linalg.norm(y)
    x = np.cross(y, z)
    R_stick = np.column_stack([x, y, z])
    R_y_to_z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    R_stick_mc = R_stick @ R_y_to_z

    viz.viewer["stick"].set_object(
        g.Cylinder(axis_len, 0.02),
        g.MeshLambertMaterial(color=0x8B6640))
    M0 = np.eye(4)
    M0[:3, :3] = R_stick_mc
    M0[:3, 3]  = human_mppi.p_stick
    viz.viewer["stick"].set_transform(M0)

    return viz, viz_ghost, R_stick_mc


def _set_camera(viz, distance=1.6, azimuth_deg=30.0, elevation_deg=-15.0,
                lookat=None):
    """Aim meshcat's camera so the recorded GIF has a consistent view."""
    if lookat is None:
        lookat = [0.4, 0.0, 0.6]
    import meshcat.transformations as tf
    az  = np.deg2rad(azimuth_deg)
    el  = np.deg2rad(elevation_deg)
    eye = np.array([
        lookat[0] + distance * np.cos(el) * np.cos(az),
        lookat[1] + distance * np.cos(el) * np.sin(az),
        lookat[2] + distance * np.sin(el),
    ])
    cam_M = tf.translation_matrix(eye)
    viz.viewer["/Cameras/default"].set_transform(cam_M)
    viz.viewer["/Cameras/default/rotated/<object>"].set_property(
        "position", [0, 0, 0])


def _meshcat_url(viz):
    """Best-effort extraction of the meshcat-served URL."""
    # ZMQ visualizers expose .url(); the meshcat-python tornado server
    # uses the static_address. Fall back to the printed URL pattern.
    try:
        return viz.viewer.url()
    except Exception:
        try:
            return viz.viewer.window.web_url
        except Exception:
            pass
    raise RuntimeError("could not get meshcat URL from viz")


async def _capture(viz, viz_ghost, R_stick_mc, q_main, q_ghost, stick_traj,
                   p_stick, dt, out_path: Path, fps: int,
                   width: int, height: int, frame_delay_ms: int,
                   browser_args: list[str]):
    T = min(len(q_main), len(q_ghost))
    url = _meshcat_url(viz)
    print(f"[record] {out_path.name}  meshcat: {url}  T={T} dt={dt:.4f}",
          flush=True)

    # Prime first frame so the page has something to render.
    viz.display(q_main[0])
    viz_ghost.display(q_ghost[0])

    frames = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=browser_args)
        page = await browser.new_page(viewport={"width": width, "height": height})
        await page.goto(url, wait_until="networkidle")
        # Settle: meshcat client connects + first scene renders.
        await asyncio.sleep(2.0)

        for t in range(T):
            viz.display(q_main[t])
            viz_ghost.display(q_ghost[t])
            if stick_traj is not None and len(stick_traj) > 0:
                tc = min(t, len(stick_traj) - 1)
                disp = stick_traj[tc] - stick_traj[0]
                M = np.eye(4)
                M[:3, :3] = R_stick_mc
                M[:3, 3]  = p_stick + disp
                viz.viewer["stick"].set_transform(M)

            await asyncio.sleep(frame_delay_ms / 1000.0)
            png = await page.screenshot(type="png")
            frames.append(np.array(Image.open(_io.BytesIO(png)).convert("RGB")))

        await browser.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(f"[record]   saved {out_path}  ({len(frames)} frames @ {fps} fps)",
          flush=True)


def _runs_for(combo_name: str, tag_filter: str | None):
    base = ANALYSIS / combo_name
    if not base.exists():
        return []
    out = []
    for d in sorted(base.iterdir()):
        if not TAG_RE.match(d.name):
            continue
        if tag_filter and d.name != tag_filter:
            continue
        if not (d / "xs_irl.npz").exists():
            continue
        out.append((d.name, d))
    return out


async def _main_async(args):
    browser_args = ["--use-gl=swiftshader", "--enable-webgl", "--no-sandbox"]

    for combo_name in args.combo:
        combo = next((c for c in COMBOS if c["name"] == combo_name), None)
        if combo is None:
            print(f"[err] unknown combo {combo_name}"); continue
        runs = _runs_for(combo_name, args.tag)
        if not runs:
            print(f"[skip] no IRL runs in {ANALYSIS / combo_name}"); continue

        # Build models once per combo (heavy).
        print(f"\n[combo] {combo_name} — building models...", flush=True)
        with _quiet():
            human, human_mppi = _rmb.build_common_models(combo, None)
        _set_camera(
            (human_mppi if hasattr(human_mppi, "viewer") else human_mppi),
            distance=args.cam_distance,
            azimuth_deg=args.cam_azimuth,
            elevation_deg=args.cam_elevation,
        ) if False else None  # camera set after meshcat init below

        for tag, run_dir in runs:
            video_tag = tag.replace("csqp_irl__", "")
            out_path = OUT_DIR / f"irl_{combo_name}__video_{video_tag}.gif"
            if out_path.exists() and args.skip_existing:
                print(f"[skip] {out_path} exists"); continue

            xs_irl = np.load(run_dir / "xs_irl.npz")["xs"]
            xs_demo_path = run_dir / "xs_demo.npy"
            if xs_demo_path.exists():
                xs_demo = np.load(xs_demo_path)
            else:
                print(f"[warn] no xs_demo.npy at {run_dir.name} — "
                      "ghost will mirror the IRL trajectory")
                xs_demo = xs_irl

            nq = human_mppi.nq
            q_main  = np.asarray([x[:nq] for x in xs_irl])
            q_ghost = np.asarray([x[:nq] for x in xs_demo])

            with _quiet():
                viz, viz_ghost, R_stick_mc = _setup_meshcat_with_ghost(human_mppi)
            _set_camera(viz, distance=args.cam_distance,
                        azimuth_deg=args.cam_azimuth,
                        elevation_deg=args.cam_elevation)

            stick_traj = getattr(human_mppi, "stick_traj", None)
            p_stick    = human_mppi.p_stick

            await _capture(
                viz, viz_ghost, R_stick_mc, q_main, q_ghost,
                stick_traj, p_stick, dt=float(human_mppi.dt),
                out_path=out_path, fps=args.fps,
                width=args.width, height=args.height,
                frame_delay_ms=args.frame_delay_ms,
                browser_args=browser_args,
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--combo", nargs="+", required=True,
                    help="combo name(s) from moirl_config.COMBOS, e.g. s2_dl")
    ap.add_argument("--tag", default=None,
                    help="record only this run dir (e.g. csqp_irl__nw3__basis)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--frame_delay_ms", type=int, default=80,
                    help="ms to wait after viz.display before screenshot — "
                         "tune up if frames look stale")
    ap.add_argument("--cam_distance", type=float, default=1.6)
    ap.add_argument("--cam_azimuth",  type=float, default=30.0)
    ap.add_argument("--cam_elevation", type=float, default=-15.0)
    ap.add_argument("--skip_existing", action="store_true",
                    help="skip runs whose GIF already exists")
    args = ap.parse_args()

    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()

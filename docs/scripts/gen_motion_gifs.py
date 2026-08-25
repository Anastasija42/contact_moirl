"""gen_motion_gifs.py — render one GIF per (ID, DATE, TASK) cycle, dump them
into docs/assets/figures/motion_previews/, and rewrite
docs/motion_previews.md with a grid of all available previews.

Reuses the same patched XML + q-trajectory construction that
test_models_visual_mujoco.py uses (full-body IK replay). Rendering is
offscreen via mujoco.Renderer — no GUI required, so it's safe to run on
headless machines.

Run from the repo root:
    python docs/scripts/gen_motion_gifs.py                      # all combos
    python docs/scripts/gen_motion_gifs.py --ids S3 --dates 27_02
    python docs/scripts/gen_motion_gifs.py --cycle 5 --skip-existing
"""
from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
import re
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import pinocchio as pin

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import final_models.human_mppi as _fm_mppi
sys.modules["human_mppi"] = _fm_mppi
from irl_utils_setup import setup_experiment
from test_models_visual_mujoco import build_mujoco_scene, write_frame, save_gif

IDS      = ["S1", "S2", "S3"]
DATES    = ["13_02", "27_02"]
TASKS    = ["up_long", "up_short", "down_long", "down_short"]
TASK_DIR = {"up_long": "up", "up_short": "up",
            "down_long": "down", "down_short": "down"}

# Data sources to render. Each tuple (label, data_root) becomes one column
# in the preview grid. The folder names are swapped relative to what they
# actually contain: trajectories_from_mocap is the *with-sensor* take and
# trajectories_from_mocap_sensor is the *without-sensor* take.
DATA_SOURCES = [
    ("13_02",        "trajectories_from_mocap"),
    ("27_02_sensor", "trajectories_from_mocap"),         # with-sensor recording
    ("27_02",        "trajectories_from_mocap_sensor"),  # without-sensor recording
]
# Which underlying date each source uses.
SOURCE_DATE = {"13_02": "13_02", "27_02": "27_02", "27_02_sensor": "27_02"}

OUT_DIR  = REPO / "docs/assets/figures/motion_previews"
DOC_PAGE = REPO / "docs/motion_previews.md"


def _legacy_patch_view_xml_DEADCODE(mppi, pinned_xml: Path, urdf_path: Path, q_pin_full) -> Path:
    """Replicates test_models_visual_mujoco.py's XML-editing pipeline and
    returns the path to a patched, write-to-disk view.xml ready for
    mujoco.MjModel.from_xml_path."""
    tree = ET.parse(pinned_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    root_body = None
    for body in worldbody.findall("body"):
        if body.get("name") != "stick":
            root_body = body
            break
    if root_body is not None and root_body.find("freejoint") is None:
        fj = ET.SubElement(root_body, "freejoint")
        fj.set("name", "root_joint")

    # Joint axes ← URDF
    urdf_text = urdf_path.read_text()
    urdf_axis = {}
    for m in re.finditer(
        r'<joint name="([^"]+)"[^>]*>\s*<parent[^/]*/>\s*<child[^/]*/>\s*<origin[^/]*/>\s*<axis xyz="([^"]+)"',
        urdf_text, flags=re.DOTALL,
    ):
        urdf_axis[m.group(1)] = m.group(2)
    for joint in root.iter("joint"):
        jn = joint.get("name")
        if jn in urdf_axis and joint.get("type") == "hinge":
            new = urdf_axis[jn]
            if joint.get("axis") != new:
                joint.set("axis", new)

    # Hide legs + pelvis
    HIDE_KEYWORDS = ("upperleg", "lowerleg", "foot", "ankle", "hip", "pelvis")
    def _hide(body):
        nm = body.get("name", "").lower()
        if any(k in nm for k in HIDE_KEYWORDS):
            for g in body.findall("geom"):
                g.set("rgba", "0 0 0 0")
                g.set("contype", "0")
                g.set("conaffinity", "0")
        for child in body.findall("body"):
            _hide(child)
    for body in worldbody.findall("body"):
        _hide(body)

    # Full-body rock offset (requires _setup_full_stick having been called)
    rock_jid_full = mppi.full_pin_model.getJointId("rock_fixed_joint_full")
    rock_offset   = mppi.full_pin_model.jointPlacements[rock_jid_full].translation
    for body in root.iter("body"):
        if body.get("name") == "rock":
            body.set("pos", f"{rock_offset[0]:.6f} "
                            f"{rock_offset[1]:.6f} "
                            f"{rock_offset[2]:.6f}")
            break

    out = pinned_xml.with_name(pinned_xml.stem + "_gif.xml")
    tree.write(out)
    return out


def _pin_to_mj_q(q_pin):
    q_mj = np.empty_like(q_pin)
    q_mj[0:3] = q_pin[0:3]
    qx, qy, qz, qw = q_pin[3], q_pin[4], q_pin[5], q_pin[6]
    n = float(np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw))
    if n < 1e-6:
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    else:
        qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    q_mj[3] = qw; q_mj[4] = qx; q_mj[5] = qy; q_mj[6] = qz
    q_mj[7:] = q_pin[7:]
    return q_mj


def render_combo(ID: str, date: str, task: str, cycle_idx: int,
                 out_path: Path, data_root: str,
                 width=640, height=480, fps_cap=25,
                 cam_azimuth: float = 30.0,
                 cam_elevation: float = -15.0,
                 cam_distance: float = 1.8):
    """Render one GIF by reusing the EXACT scene + GIF code path the
    interactive viewer uses (`build_mujoco_scene` + `save_gif` from
    test_models_visual_mujoco.py). Camera defaults are the GIF preset
    (close-up framing of torso + rock); pass `--azimuth`/`--elevation`/
    `--distance` to override."""
    direction = TASK_DIR[task]
    combo_root = REPO / f"{data_root}/{date}/{ID}/{task}"
    cyc_dir    = combo_root / f"elaborated/{direction}"
    follow = sorted(
        int(p.stem.split("_")[1])
        for p in cyc_dir.glob("cycle_*.npz")
        if int(p.stem.split("_")[1]) > cycle_idx
    )
    try:
        scene = build_mujoco_scene(
            ID, task, date, cycle_idx, direction=direction,
            extra_cycles=follow, data_root=data_root,
        )
    except Exception as e:
        print(f"[err] {ID}/{date}/{task}/{data_root}: {e}")
        return False

    save_gif(scene, out_path,
             width=width, height=height, fps_cap=fps_cap,
             azimuth=cam_azimuth,
             elevation=cam_elevation,
             distance=cam_distance)
    return True


def build_page(gif_paths: list[Path]):
    """Write docs/motion_previews.md grouping GIFs into a grid by subject.

    Filename convention: {ID}__{source}__{task}.gif
    (double underscore separator so we can split cleanly even when source
    contains '_'.)
    """
    by_id: dict[str, list[tuple[str, str, Path]]] = {}
    for p in gif_paths:
        parts = p.stem.split("__")
        if len(parts) != 3:
            continue
        ID, source, task = parts
        by_id.setdefault(ID, []).append((source, task, p))

    source_labels = [s for s, _ in DATA_SOURCES]

    lines = ["---", "layout: default", "title: Motion previews", "---", "",
             "# Motion previews",
             "",
             "Auto-generated GIFs of one cycle per subject × source × task.",
             "`27_02_sensor` is the `trajectories_from_mocap_sensor/` folder "
             "(despite the name, these are the *without-sensor* takes).",
             "Regenerate with `python docs/scripts/gen_motion_gifs.py`.",
             ""]
    for ID in sorted(by_id):
        lines.append(f"## {ID}")
        lines.append("")
        header = "| Task | " + " | ".join(source_labels) + " |"
        sep    = "|------|" + "|".join(["-------"] * len(source_labels)) + "|"
        lines.append(header)
        lines.append(sep)
        tasks = sorted({t for _, t, _ in by_id[ID]})
        for task in tasks:
            cells = [f"**{task}**"]
            for src in source_labels:
                hit = [p for s, t, p in by_id[ID] if s == src and t == task]
                if hit:
                    rel = hit[0].relative_to(REPO / "docs")
                    cells.append(f'<img src="{rel}" width="200">')
                else:
                    cells.append("—")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    DOC_PAGE.write_text("\n".join(lines) + "\n")
    print(f"[page] wrote {DOC_PAGE.relative_to(REPO)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ids",   nargs="+", default=IDS,   choices=IDS)
    ap.add_argument("--dates", nargs="+", default=DATES, choices=DATES)
    ap.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    ap.add_argument("--cycle", type=int, default=3,
                    help="reference cycle — used for setup_experiment's rail "
                         "estimation AND as the first cycle in the animation "
                         "(earlier cycles have often not settled IK yet).")
    ap.add_argument("--width",  type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--azimuth",   type=float, default=30.0,
                    help="camera azimuth in degrees (default 30 = the GIF preset)")
    ap.add_argument("--elevation", type=float, default=-15.0,
                    help="camera elevation in degrees (default -15)")
    ap.add_argument("--distance",  type=float, default=1.8,
                    help="camera distance from lookat (default 1.8)")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    # Headless rendering for machines without a display.
    os.environ.setdefault("MUJOCO_GL", "egl")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Restrict to requested dates. Sources whose underlying date isn't
    # selected get dropped.
    sources = [(s, r) for s, r in DATA_SOURCES if SOURCE_DATE[s] in args.dates]

    produced = []
    for ID in args.ids:
        for source_label, data_root in sources:
            date = SOURCE_DATE[source_label]
            for task in args.tasks:
                out = OUT_DIR / f"{ID}__{source_label}__{task}.gif"
                if args.skip_existing and out.exists():
                    print(f"[keep] {out.relative_to(REPO)}")
                    produced.append(out)
                    continue

                direction = TASK_DIR[task]
                cyc_dir = REPO / f"{data_root}/{date}/{ID}/{task}/elaborated/{direction}"
                if not cyc_dir.exists():
                    print(f"[skip] {ID}/{source_label}/{task}: {cyc_dir} missing")
                    continue
                candidates = sorted(cyc_dir.glob("cycle_*.npz"))
                if not candidates:
                    continue
                preferred = cyc_dir / f"cycle_{args.cycle:02d}.npz"
                cycle_idx = args.cycle if preferred.exists() else \
                    int(candidates[0].stem.split("_")[1])

                ok = render_combo(ID, date, task, cycle_idx, out,
                                  data_root=data_root,
                                  width=args.width, height=args.height,
                                  cam_azimuth=args.azimuth,
                                  cam_elevation=args.elevation,
                                  cam_distance=args.distance)
                if ok:
                    produced.append(out)

    # Always rewrite the preview page against whatever GIFs exist on disk.
    all_gifs = sorted(OUT_DIR.glob("*.gif"))
    if all_gifs:
        build_page(all_gifs)
    else:
        print("[page] no GIFs on disk — page not written")


if __name__ == "__main__":
    main()

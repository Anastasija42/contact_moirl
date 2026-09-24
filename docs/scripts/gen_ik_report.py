"""
gen_ik_report.py
================
Scan trajectories_from_mocap/ and generate:
 - docs/ik_fit_table.md   (per-subject/session/task RMSE table)
 - docs/assets/ik_analysis/<subj>_<sess>_down_variability.png
 - docs/assets/ik_analysis/<subj>_between_session_<task>.png
 - docs/assets/ik_analysis/<subj>_<sess>_stick_rock.png
 - docs/assets/ik_analysis/<subj>_<sess>_rmse_bar.png

Usage:
    python docs/scripts/gen_ik_report.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT        = Path("trajectories_from_mocap")
SENSOR_ROOT = Path("trajectories_from_mocap_sensor")
ASSETS      = Path("docs/assets/ik_analysis")
ASSETS.mkdir(parents=True, exist_ok=True)
TABLE_PATH  = Path("docs/ik_fit_table.md")

SESSIONS = ["13_02", "27_02"]
SUBJECTS = ["s2", "s3", "s1"]
TASKS    = ["down_long", "down_short", "up_long", "up_short"]
ARM_MARKERS = ["RSHO", "RELB", "RWRI", "RMWRI", "RHM5"]


def load_rmse(session, subject, task, root=ROOT):
    base = root / session / subject / task
    for name in ("ik_rmse_full_body.csv", "ik_rmse_per_marker.csv"):
        p = base / name
        if p.exists():
            return pd.read_csv(p)
    return None


def load_rmse_split(session, subject, task, root=ROOT):
    """Return (full_body_df, arm_only_df) or (None, None)."""
    base = root / session / subject / task
    f_p = base / "ik_rmse_full_body.csv"
    a_p = base / "ik_rmse_arm_only.csv"
    if f_p.exists() and a_p.exists():
        return pd.read_csv(f_p), pd.read_csv(a_p)
    return None, None


def load_q(session, subject, task, root=ROOT):
    p = root / session / subject / task / "ik_joint_angles.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_markers(session, subject, task, root=ROOT):
    p = root / session / subject / task / "ik_marker_trajectories.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


# ──────────────────────────────────────────────────────────────────────────
# 1. RMSE summary table
# ──────────────────────────────────────────────────────────────────────────

rows = []
for sess in SESSIONS:
    for subj in SUBJECTS:
        for task in TASKS:
            df = load_rmse(sess, subj, task)
            if df is None:
                continue
            mean = df['rmse_m'].mean() * 100
            max_ = df['rmse_m'].max() * 100
            worst = df.loc[df['rmse_m'].idxmax(), 'marker']
            rows.append({
                'subject': subj, 'session': sess, 'task': task,
                'mean_cm': mean, 'max_cm': max_, 'worst_marker': worst,
                'n_markers': len(df),
            })

if rows:
    df_table = pd.DataFrame(rows)
    with open(TABLE_PATH, 'w') as f:
        f.write("---\nlayout: default\ntitle: IK Fit Table\n---\n\n")
        f.write("# Per-subject / session / task IK fit quality\n\n")
        f.write("Mean and max per-marker RMSE in **cm**. "
                "Columns are sortable if you open the raw CSV.\n\n")
        f.write("| Subject | Session | Task | Mean (cm) | Max (cm) | Worst marker | #Markers |\n")
        f.write("|---------|---------|------|-----------|----------|--------------|----------|\n")
        for _, r in df_table.sort_values(['subject', 'session', 'task']).iterrows():
            f.write(f"| {r.subject} | {r.session} | {r.task} | "
                    f"{r.mean_cm:.2f} | {r.max_cm:.2f} | {r.worst_marker} | {r.n_markers} |\n")
        f.write("\n[Back to IK analysis](ik_analysis)\n")
    print(f"[write] {TABLE_PATH}")
else:
    print("No RMSE data found — check folder structure")

# ──────────────────────────────────────────────────────────────────────────
# 2. Per-marker RMSE bar plots
# ──────────────────────────────────────────────────────────────────────────

for sess in SESSIONS:
    for subj in SUBJECTS:
        # Aggregate across tasks for a single subject-session
        aggregated = {}
        for task in TASKS:
            df = load_rmse(sess, subj, task)
            if df is None:
                continue
            for _, r in df.iterrows():
                aggregated.setdefault(r['marker'], []).append(r['rmse_m'])
        if not aggregated:
            continue
        markers = sorted(aggregated.keys())
        means   = [np.mean(aggregated[m]) * 100 for m in markers]
        stds    = [np.std(aggregated[m]) * 100 for m in markers]

        fig, ax = plt.subplots(figsize=(max(8, len(markers) * 0.3), 4))
        ax.bar(range(len(markers)), means, yerr=stds, capsize=3, color='steelblue')
        ax.set_xticks(range(len(markers)))
        ax.set_xticklabels(markers, rotation=70, ha='right', fontsize=8)
        ax.set_ylabel("RMSE (cm)")
        ax.set_title(f"{subj} — session {sess} — mean per-marker RMSE across tasks")
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        out = ASSETS / f"{subj}_{sess}_rmse_bar.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"[write] {out}")

# ──────────────────────────────────────────────────────────────────────────
# 3. Between-session elbow flexion comparison
# ──────────────────────────────────────────────────────────────────────────

ELBOW_KEY = "Right_Elbow_Flexion_Extension[rad]"

for subj in SUBJECTS:
    for task in TASKS:
        traces = {}
        for sess in SESSIONS:
            df = load_q(sess, subj, task)
            if df is None or ELBOW_KEY not in df.columns:
                continue
            traces[sess] = df[ELBOW_KEY].values
        if len(traces) < 2:
            continue

        fig, ax = plt.subplots(figsize=(10, 3))
        for sess, v in traces.items():
            t = np.arange(len(v)) * 0.0083  # raw mocap rate
            ax.plot(t, np.rad2deg(v), label=f"session {sess}", alpha=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Elbow flex (deg)")
        ax.set_title(f"{subj} — {task} — elbow flexion, between-session")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = ASSETS / f"{subj}_between_session_{task}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"[write] {out}")

# ──────────────────────────────────────────────────────────────────────────
# 4. Cycle variability (within session) from the elaborated meta
# ──────────────────────────────────────────────────────────────────────────

for sess in SESSIONS:
    for subj in SUBJECTS:
        for task in TASKS:
            meta_path = ROOT / sess / subj / task / "elaborated" / "meta.npz"
            if not meta_path.exists():
                continue
            try:
                m = np.load(meta_path, allow_pickle=True)
                q_down = m['q_down_all']   # (N, 100, J)
                joints = list(m['joint_names'])
                if 'right_elbow_Y' in joints:
                    j = joints.index('right_elbow_Y')
                elif 'right_elbow_Z' in joints:
                    j = joints.index('right_elbow_Z')
                else:
                    j = 3
                fig, ax = plt.subplots(figsize=(8, 3))
                t_pct = np.linspace(0, 100, q_down.shape[1])
                for i in range(q_down.shape[0]):
                    ax.plot(t_pct, np.rad2deg(q_down[i, :, j]), alpha=0.15, color='navy')
                ax.plot(t_pct, np.rad2deg(q_down.mean(0)[:, j]), color='red', linewidth=2, label='mean')
                ax.set_xlabel("% of stroke")
                ax.set_ylabel(f"{joints[j]} (deg)")
                ax.set_title(f"{subj} — {sess} — {task} — {q_down.shape[0]} cycles")
                ax.legend()
                ax.grid(alpha=0.3)
                fig.tight_layout()
                out = ASSETS / f"{subj}_{sess}_{task}_variability.png"
                fig.savefig(out, dpi=120)
                plt.close(fig)
                print(f"[write] {out}")
            except Exception as e:
                print(f"[skip] {meta_path}: {e}")

# ──────────────────────────────────────────────────────────────────────────
# 5. Stick/rock geometry per session
# ──────────────────────────────────────────────────────────────────────────

for sess in SESSIONS:
    for subj in SUBJECTS:
        for task in TASKS:
            src = ROOT / sess / subj / task / "stick_stone_analysis.png"
            if src.exists():
                dst = ASSETS / f"{subj}_{sess}_{task}_stick_rock.png"
                try:
                    import shutil
                    shutil.copyfile(src, dst)
                    print(f"[copy] {dst}")
                except Exception as e:
                    print(f"[skip] {src}: {e}")

# ──────────────────────────────────────────────────────────────────────────
# 6. Grouped RMSE heatmap — one figure, all subjects × sessions
# ──────────────────────────────────────────────────────────────────────────

def plot_rmse_heatmap():
    rows = []
    for sess in SESSIONS:
        for subj in SUBJECTS:
            agg = {}
            for task in TASKS:
                df = load_rmse(sess, subj, task)
                if df is None:
                    continue
                for _, r in df.iterrows():
                    if r['marker'] == 'GLOBAL':
                        continue
                    agg.setdefault(r['marker'], []).append(r['rmse_m'])
            for mk, vals in agg.items():
                rows.append({'subject': subj, 'session': sess,
                             'marker': mk, 'rmse_cm': np.mean(vals) * 100})
    if not rows:
        return
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index='marker', columns=['subject', 'session'],
                           values='rmse_cm', aggfunc='mean')

    fig, ax = plt.subplots(figsize=(1.2 * pivot.shape[1] + 3, 0.3 * pivot.shape[0] + 2))
    im = ax.imshow(pivot.values, aspect='auto', cmap='viridis_r')
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"{s}\n{ss}" for s, ss in pivot.columns], fontsize=9)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Mean per-marker RMSE across tasks (cm)")
    fig.colorbar(im, ax=ax, label='RMSE (cm)')
    mid = np.nanmean(pivot.values)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha='center', va='center', fontsize=7,
                        color='white' if v > mid else 'black')
    fig.tight_layout()
    out = ASSETS / "rmse_heatmap_all.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[write] {out}")


plot_rmse_heatmap()

# ──────────────────────────────────────────────────────────────────────────
# 7. Full-body vs arm-only (arm markers, per subject × session)
# ──────────────────────────────────────────────────────────────────────────

def plot_fullbody_vs_arm():
    for sess in SESSIONS:
        for subj in SUBJECTS:
            f_agg = {mk: [] for mk in ARM_MARKERS}
            a_agg = {mk: [] for mk in ARM_MARKERS}
            any_data = False
            for task in TASKS:
                fdf, adf = load_rmse_split(sess, subj, task)
                if fdf is None or adf is None:
                    continue
                any_data = True
                for _, r in fdf.iterrows():
                    if r['marker'] in ARM_MARKERS:
                        f_agg[r['marker']].append(r['rmse_m'])
                for _, r in adf.iterrows():
                    if r['marker'] in ARM_MARKERS:
                        a_agg[r['marker']].append(r['rmse_m'])
            if not any_data:
                continue

            fig, ax = plt.subplots(figsize=(7, 3.5))
            x = np.arange(len(ARM_MARKERS))
            w = 0.35
            f_vals = [np.mean(f_agg[m]) * 100 if f_agg[m] else np.nan for m in ARM_MARKERS]
            a_vals = [np.mean(a_agg[m]) * 100 if a_agg[m] else np.nan for m in ARM_MARKERS]
            ax.bar(x - w/2, f_vals, w, label='Full-body', color='steelblue')
            ax.bar(x + w/2, a_vals, w, label='Arm-only',  color='crimson')
            ax.set_xticks(x)
            ax.set_xticklabels(ARM_MARKERS)
            ax.set_ylabel("RMSE (cm)")
            ax.set_title(f"{subj} — session {sess} — full-body vs arm-only IK")
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            fig.tight_layout()
            out = ASSETS / f"{subj}_{sess}_fullbody_vs_arm.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            print(f"[write] {out}")


plot_fullbody_vs_arm()

# ──────────────────────────────────────────────────────────────────────────
# 8. Sensor vs no-sensor (27_02 take1 vs take2_sensor)
# ──────────────────────────────────────────────────────────────────────────

def plot_sensor_vs_nosensor():
    if not SENSOR_ROOT.exists():
        print(f"[skip] {SENSOR_ROOT} not found — run the take2_sensor batch first")
        return
    sess = "27_02"
    for subj in SUBJECTS:
        nosensor = {}
        sensor   = {}
        for task in TASKS:
            ns = load_rmse(sess, subj, task, root=ROOT)
            s  = load_rmse(sess, subj, task, root=SENSOR_ROOT)
            if ns is None or s is None:
                continue
            for _, r in ns.iterrows():
                if r['marker'] != 'GLOBAL':
                    nosensor.setdefault(r['marker'], []).append(r['rmse_m'])
            for _, r in s.iterrows():
                if r['marker'] != 'GLOBAL':
                    sensor.setdefault(r['marker'], []).append(r['rmse_m'])

        common = sorted(set(nosensor) & set(sensor))
        if not common:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(common) * 0.35), 4))
        x = np.arange(len(common))
        w = 0.35
        ns_vals = [np.mean(nosensor[m]) * 100 for m in common]
        s_vals  = [np.mean(sensor[m])  * 100 for m in common]
        ax.bar(x - w/2, ns_vals, w, label='No sensor (take1)',   color='steelblue')
        ax.bar(x + w/2, s_vals,  w, label='With sensor (take2)', color='darkorange')
        ax.set_xticks(x)
        ax.set_xticklabels(common, rotation=70, ha='right', fontsize=8)
        ax.set_ylabel("RMSE (cm)")
        ax.set_title(f"{subj} — 27_02 — with vs without sensor")
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        out = ASSETS / f"{subj}_27_02_sensor_vs_nosensor.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"[write] {out}")


plot_sensor_vs_nosensor()

# ──────────────────────────────────────────────────────────────────────────
# 9. Sensor vs no-sensor — cycle variability bands (27_02 only)
# ──────────────────────────────────────────────────────────────────────────

CYCLE_JOINTS = [
    "Right_Shoulder_Flexion_Extension[rad]",
    "Right_Elbow_Flexion_Extension[rad]",
    "Right_Wrist_Flexion_Extension[rad]",
    "Right_Wrist_Radial_Ulnar_Deviation[rad]",
]


def load_meta(root, session, subject, task):
    p = root / session / subject / task / "elaborated" / "meta.npz"
    return np.load(p, allow_pickle=True) if p.exists() else None


def plot_sensor_vs_nosensor_cycles():
    if not SENSOR_ROOT.exists():
        print(f"[skip] {SENSOR_ROOT} not found — run the take2_sensor batch first")
        return
    sess = "27_02"
    for subj in SUBJECTS:
        for task in TASKS:
            ns = load_meta(ROOT, sess, subj, task)
            se = load_meta(SENSOR_ROOT, sess, subj, task)
            if ns is None or se is None:
                continue

            joints = [j.item() if hasattr(j, "item") else str(j)
                      for j in ns["joint_names"]]
            joints_plot = [j for j in CYCLE_JOINTS if j in joints]
            if not joints_plot:
                continue

            fig, axes = plt.subplots(len(joints_plot), 2,
                                     figsize=(10, 2.3 * len(joints_plot)),
                                     sharex=True)
            if len(joints_plot) == 1:
                axes = np.array([axes])
            t = np.linspace(0, 100, ns["q_down_all"].shape[1])
            for row, jname in enumerate(joints_plot):
                idx = joints.index(jname)
                for col, phase in enumerate(("down", "up")):
                    ax = axes[row, col]
                    for data, color, label in (
                        (ns[f"q_{phase}_all"], "steelblue", "no sensor"),
                        (se[f"q_{phase}_all"], "darkorange", "sensor"),
                    ):
                        d = np.degrees(data[:, :, idx])
                        m, s = d.mean(axis=0), d.std(axis=0)
                        ax.plot(t, m, color=color, lw=1.8,
                                label=f"{label} (n={len(d)})")
                        ax.fill_between(t, m - s, m + s, alpha=0.25, color=color)
                    ax.grid(True, alpha=0.3)
                    ax.set_ylabel(jname.replace("[rad]", "").replace("_", " "),
                                  fontsize=7)
                    if row == 0:
                        ax.set_title(phase.upper(), fontsize=10)
                    if row == 0 and col == 1:
                        ax.legend(fontsize=7, loc="upper right")
            axes[-1, 0].set_xlabel("% of cycle")
            axes[-1, 1].set_xlabel("% of cycle")
            fig.suptitle(f"{subj} — 27_02 — {task} — sensor vs no-sensor (cycle bands)",
                         fontsize=11)
            fig.tight_layout()
            out = ASSETS / f"{subj}_27_02_{task}_cycles_sensor_vs_nosensor.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            print(f"[write] {out}")


plot_sensor_vs_nosensor_cycles()

# ──────────────────────────────────────────────────────────────────────────
# 9. Cycle strip — stroboscopic right-arm stick figure
# ──────────────────────────────────────────────────────────────────────────

CYCLE_CHAIN = ["RSHO", "RELB", "RWRI", "RHM5"]
N_POSES     = 6
SIDE_AXIS   = ("y", "z")   # plot y (vertical) vs z (side) of pelvis-centred coords

def plot_cycle_strip():
    task = "down_long"
    for sess in SESSIONS:
        for subj in SUBJECTS:
            df = load_markers(sess, subj, task)
            if df is None:
                continue
            # pick a mid-trajectory window (avoids startup transients)
            total = len(df)
            if total < 120:
                continue
            start = total // 3
            end   = min(start + 80, total)
            idxs  = np.linspace(start, end - 1, N_POSES).astype(int)

            fig, ax = plt.subplots(figsize=(5, 5))
            cmap = plt.cm.viridis(np.linspace(0.15, 0.95, N_POSES))

            missing = False
            for k, i in enumerate(idxs):
                pts = []
                for mk in CYCLE_CHAIN:
                    col_x = f"{mk}_x"; col_y = f"{mk}_y"; col_z = f"{mk}_z"
                    if col_x not in df.columns:
                        missing = True
                        break
                    pts.append([df[col_x].iloc[i], df[col_y].iloc[i], df[col_z].iloc[i]])
                if missing:
                    break
                pts = np.array(pts)
                # plot y (up) vs z (depth) — profile view
                ax.plot(pts[:, 2], pts[:, 1], '-o', color=cmap[k],
                        alpha=0.35 + 0.55 * k / (N_POSES - 1),
                        linewidth=1.8 + k * 0.2, markersize=5,
                        label=f"t={k/(N_POSES-1):.0%}" if k in (0, N_POSES // 2, N_POSES - 1) else None)
            if missing:
                plt.close(fig)
                continue
            ax.set_xlabel("z (m)  ← lateral →")
            ax.set_ylabel("y (m)  ← up →")
            ax.set_title(f"{subj} — {sess} — right arm, {N_POSES} poses across one stroke")
            ax.set_aspect('equal')
            ax.grid(alpha=0.3)
            ax.legend(loc='best', fontsize=8)
            fig.tight_layout()
            out = ASSETS / f"{subj}_{sess}_cycle_strip.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            print(f"[write] {out}")


plot_cycle_strip()

print("\nDone. Regenerate by re-running this script after new IK results land.")

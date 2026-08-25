"""Cycle slicing and scene helpers.

Detects individual shaving cycles in a recorded trajectory and slices them, and
holds the viewer-side utilities (static stick placement, leg removal, body
colouring) shared by the visualisations.
"""
import scipy.signal as signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from pathlib import Path
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import time
import json

def animate_with_joint_signals(motion_data, joints_to_use, viz,
                                joints_to_plot=None, dt=0.0083, playback_speed=1.0,
                                title="Motion"):
    """Play motion in Meshcat while live-plotting selected joint signals in matplotlib.

    Args:
        motion_data    : np.ndarray of shape (N_frames, N_joints)
        joints_to_use  : list of joint names matching columns of motion_data
        viz            : MeshcatVisualizer already initialized
        joints_to_plot : list of joint names to plot (default: wrist + elbow)
        dt             : timestep in seconds
        playback_speed : >1 faster, <1 slower
        title          : plot window title
    """
    if joints_to_plot is None:
        candidates = [
            "Right_Wrist_Radial_Ulnar_Deviation[rad]",
            "Right_Wrist_Flexion_Extension[rad]",
            "Right_Elbow_Flexion_Extension[rad]",
            "Right_Shoulder_Flexion_Extension[rad]",
            "Right_Shoulder_Abduction_Adduction[rad]",
        ]
        joints_to_plot = [j for j in candidates if j in joints_to_use][:4]

    if not joints_to_plot:
        raise ValueError("No valid joints to plot found in joints_to_use.")

    n_frames = len(motion_data)
    n_joints = len(joints_to_plot)
    indices  = [joints_to_use.index(j) for j in joints_to_plot]
    signals  = [motion_data[:, idx] for idx in indices]
    t_axis   = np.arange(n_frames) * dt

    plt.ion()
    fig, axes = plt.subplots(n_joints, 1, figsize=(10, 2.5 * n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=12)

    lines      = []
    vlines     = []
    short_name = lambda s: s.replace("[rad]", "").replace("_", " ").strip()

    for ax, sig, name in zip(axes, signals, joints_to_plot):
        ax.plot(t_axis, np.degrees(sig), color='lightgray', lw=1.2, label='full')
        line, = ax.plot([], [], color='royalblue', lw=2)
        vline = ax.axvline(0, color='red', lw=1.5, linestyle='--')
        ax.set_ylabel(f"{short_name(name)}\n[deg]", fontsize=8)
        ax.grid(True, alpha=0.4)
        lines.append(line)
        vlines.append(vline)

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()

    print(f"[INFO] Playing {n_frames} frames — {title}")
    for i, q in enumerate(motion_data):
        viz.display(q)

        t_now = i * dt
        for line, vline, sig in zip(lines, vlines, signals):
            line.set_data(t_axis[:i+1], np.degrees(sig[:i+1]))
            vline.set_xdata([t_now, t_now])

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(dt / playback_speed)

        if i % 50 == 0:
            print(f"  Frame {i}/{n_frames}")

    plt.ioff()
    print("[INFO] Done")
    return fig

def _enforce_amplitude_alternation(smoothed, peaks, troughs, min_swing):
    """Clean spurious mid-stroke extrema.

    ``find_peaks(prominence=...)`` alone keeps shallow peak-trough-peak wiggles at
    the top/bottom of a stroke — the arm hesitates near full extension and a tiny
    dip is detected as a whole cycle, slicing the *middle of a downstroke*. We
    enforce two things the raw detector does not:
      1. strict peak/trough ALTERNATION (collapse consecutive same-type extrema,
         keeping the most extreme — the true peak/trough), and
      2. a minimum peak-to-trough SWING: iteratively drop the shallowest reversal
         whose swing is < ``min_swing`` (auto-scaled to the stroke amplitude), then
         re-collapse as the freed neighbours merge.
    Returns cleaned (peaks, troughs) index arrays.
    """
    ev = sorted([(int(i), "P") for i in peaks] + [(int(i), "T") for i in troughs])
    if not ev:
        return np.asarray(peaks, dtype=int), np.asarray(troughs, dtype=int)

    def collapse(seq):
        out = []
        for idx, typ in seq:
            if out and out[-1][1] == typ:
                pidx = out[-1][0]
                more_extreme = (smoothed[idx] > smoothed[pidx]) if typ == "P" \
                    else (smoothed[idx] < smoothed[pidx])
                if more_extreme:
                    out[-1] = (idx, typ)
            else:
                out.append((idx, typ))
        return out

    ev = collapse(ev)
    changed = True
    while changed and len(ev) >= 2:
        changed = False
        swings = [abs(smoothed[ev[i][0]] - smoothed[ev[i + 1][0]])
                  for i in range(len(ev) - 1)]
        j = int(np.argmin(swings))
        if swings[j] < min_swing:
            del ev[j:j + 2]
            ev = collapse(ev)
            changed = True

    peaks_out   = np.asarray([i for i, t in ev if t == "P"], dtype=int)
    troughs_out = np.asarray([i for i, t in ev if t == "T"], dtype=int)
    return peaks_out, troughs_out

def detect_and_slice_cycles(df_angles,
                             signal_col="Right_Wrist_Radial_Ulnar_Deviation[rad]",
                             window_length=51, polyorder=3,
                             min_swing_frac=0.35,
                             peak_distance=100, peak_prominence=0.01,
                             trough_distance=100, trough_prominence=0.01,
                             plot=True, save_path=None):
    """Detect motion cycles from a joint angle signal and slice into UP/DOWN motions.

    Args:
        df_angles       : DataFrame from ik_joint_angles.csv
        signal_col      : column to use for cycle detection
        plot            : whether to show the verification plot

    Returns:
        all_up_motions   : list of np.ndarray, one per cycle
        all_down_motions : list of np.ndarray, one per cycle
        peaks            : peak frame indices
        troughs          : trough frame indices
    """
    EXCLUDE_COLS  = ["Relative_Time[s]", "Capture_Start_Time"]
    joints_to_use = [c for c in df_angles.columns if c not in EXCLUDE_COLS]
    data_full     = df_angles[joints_to_use].to_numpy()

    if signal_col not in joints_to_use:
        raise ValueError(f"signal_col '{signal_col}' not found in df_angles columns.")

    rwri_data = data_full[:, joints_to_use.index(signal_col)]
    smoothed  = savgol_filter(rwri_data, window_length=window_length, polyorder=polyorder)

    peaks,   _ = find_peaks( smoothed, distance=peak_distance,   prominence=peak_prominence)
    troughs, _ = find_peaks(-smoothed, distance=trough_distance, prominence=trough_prominence)

    if min_swing_frac and min_swing_frac > 0:
        amp = np.percentile(smoothed, 90) - np.percentile(smoothed, 10)
        n_before = (len(peaks), len(troughs))
        peaks, troughs = _enforce_amplitude_alternation(
            smoothed, peaks, troughs, min_swing=min_swing_frac * amp)
        print(f"[INFO] amplitude guard (min_swing={min_swing_frac * amp:.3f} rad): "
              f"peaks {n_before[0]}→{len(peaks)}, troughs {n_before[1]}→{len(troughs)}")

    if plot or save_path is not None:
        fig = plt.figure(figsize=(12, 4))
        plt.plot(smoothed, label='Smoothed Signal')
        plt.plot(peaks,   smoothed[peaks],   "rx", label='Peaks (Top)')
        plt.plot(troughs, smoothed[troughs], "go", label='Troughs (Bottom)')
        plt.title(f"Cycle Detection — {signal_col}")
        plt.legend(); plt.grid(True); plt.tight_layout()
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=120)
            plt.close(fig)
        elif plot:
            plt.show()

    all_up_motions   = []
    all_down_motions = []

    if len(peaks) > 0 and len(troughs) > 0 and peaks[0] < troughs[0]:
        peaks = peaks[1:]

    num_cycles = min(len(peaks), len(troughs) - 1)
    for i in range(num_cycles):
        all_up_motions.append(data_full[troughs[i] : peaks[i], :])
        all_down_motions.append(data_full[peaks[i] : troughs[i+1], :])
    print(f"\n[INFO] Total Cycles Found  : {len(all_up_motions)}")
    print(f"[INFO] Avg UP   length     : {np.mean([len(m) for m in all_up_motions]):.1f} frames")
    print(f"[INFO] Avg DOWN length     : {np.mean([len(m) for m in all_down_motions]):.1f} frames")
    for idx, m in enumerate(all_up_motions):
        print(f"  Cycle {idx}: UP={len(m):>4}  DOWN={len(all_down_motions[idx]):>4}")

    return all_up_motions, all_down_motions, peaks, troughs, joints_to_use

def get_stick_static_position(df_stick, frame_idx=None):
    """Return mean position of each stick marker. If frame_idx given, use that frame."""
    positions = {}
    for i in range(1, 4):
        x_col = f"stick:Marker{i}_X"
        y_col = f"stick:Marker{i}_Y"
        z_col = f"stick:Marker{i}_Z"
        if all(c in df_stick.columns for c in [x_col, y_col, z_col]):
            if frame_idx is not None:
                pos = df_stick[[x_col, y_col, z_col]].iloc[frame_idx].values.astype(float)
            else:
                pos = df_stick[[x_col, y_col, z_col]].mean().values.astype(float)
            positions[f"stick:Marker{i}"] = pos
    return positions

def get_stick_static_position(df_stick, frame_idx=None):
    """Return mean (or per-frame) position of each stick marker as dict.
    Also computes which two markers are farthest apart -> stick endpoints.
    
    Returns:
        positions   : dict {marker_name: np.array shape (3,)}
        stick_ends  : tuple (name_A, name_B) of the two farthest markers
    """
    positions = {}
    for i in range(1, 4):
        x_col, y_col, z_col = f"stick:Marker{i}_X", f"stick:Marker{i}_Y", f"stick:Marker{i}_Z"
        if all(c in df_stick.columns for c in [x_col, y_col, z_col]):
            if frame_idx is not None:
                pos = df_stick[[x_col, y_col, z_col]].iloc[frame_idx].values.astype(float)
            else:
                pos = df_stick[[x_col, y_col, z_col]].mean().values.astype(float)
            positions[f"stick:Marker{i}"] = pos

    stick_ends = None
    if len(positions) >= 2:
        names = list(positions.keys())
        max_dist  = -1
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                d = np.linalg.norm(positions[names[i]] - positions[names[j]])
                if d > max_dist:
                    max_dist  = d
                    stick_ends = (names[i], names[j])
        print(f"[INFO] Stick endpoints: {stick_ends[0]} ↔ {stick_ends[1]}  (dist={max_dist*1000:.1f} mm)")

    return positions, stick_ends

def add_static_stick_to_viewer(viewer, stick_static, stick_ends):
    """Add stick markers (spheres) and stick cylinder to a meshcat viewer.
    
    Args:
        viewer       : meshcat.Visualizer
        stick_static : dict {marker_name: np.array(3,)}  from get_stick_static_position
        stick_ends   : tuple (name_A, name_B) — the two farthest markers
    """
    import meshcat.geometry as g
    import meshcat.transformations as tf

    for name, pos in stick_static.items():
        if not np.isnan(pos).any():
            viewer[f"stick/{name}"].set_object(
                g.Sphere(0.012),
                g.MeshLambertMaterial(color=0xFFFF00, opacity=0.9)
            )
            viewer[f"stick/{name}"].set_transform(tf.translation_matrix(pos))

    if stick_ends is not None:
        m1 = stick_static[stick_ends[0]]
        m3 = stick_static[stick_ends[1]]
        mid    = (m1 + m3) / 2.0
        vec    = m3 - m1
        L      = np.linalg.norm(vec)

        y_axis    = vec / L
        ref       = np.array([1, 0, 0]) if abs(y_axis[0]) < 0.9 else np.array([0, 0, 1])
        x_axis    = np.cross(ref, y_axis);  x_axis /= np.linalg.norm(x_axis)
        z_axis    = np.cross(x_axis, y_axis)

        R         = np.column_stack([x_axis, y_axis, z_axis])

        T         = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = mid

        viewer["stick/cylinder"].set_object(
            g.Cylinder(L, 0.015),
            g.MeshLambertMaterial(color=0x8B4513, opacity=0.9)
        )
        viewer["stick/cylinder"].set_transform(T)
        print(f"[INFO] Stick: {stick_ends[0]} → {stick_ends[1]},  length={L*1000:.1f} mm")

def remove_legs_from_visual_model(human_visual_model):
    """Remove leg geometry objects from the visual model so they don't render.
    
    Works by setting leg geometry objects to invisible (zero alpha).
    """
    LEG_KEYWORDS = [
        "left_upperleg", "right_upperleg",
        "left_lowerleg", "right_lowerleg",
        "left_foot",     "right_foot",
        "left_hip",      "right_hip",
        "left_ankle",    "right_ankle",
        "middle_pelvis", "pelvis",
        "lank", "rank", "lkne", "rkne",
        "lmank", "rmank", "lasi", "rasi",
        "lpsi",  "rpsi",  "ltoe", "rtoe",
        "lhee",  "rhee",
    ]
    removed = []
    for go in human_visual_model.geometryObjects:
        name_lower = go.name.lower()
        if any(kw in name_lower for kw in LEG_KEYWORDS):
            go.overrideMaterial = True
            go.meshColor        = np.array([0.0, 0.0, 0.0, 0.0])  
            removed.append(go.name)

    return human_visual_model

def set_body_colors(human_visual_model):
    """Color geometry objects using original URDF color scheme.
    Left hand is extra transparent.
    """
    for go in human_visual_model.geometryObjects:
        if go.meshColor[3] == 0.0:
            continue

        go.overrideMaterial = True
        name_lower = go.name.lower()

        if "left_hand" in name_lower:
            go.meshColor = np.array([0.05, 0.8, 0.2, 0.15])
        elif "left" in name_lower:
            go.meshColor = np.array([0.05, 0.8, 0.2, 0.6])
        elif "right" in name_lower:
            go.meshColor = np.array([0.8, 0.05, 0.2, 0.6])
        else:
            go.meshColor = np.array([0.2, 0.05, 0.8, 0.3])

    return human_visual_model

def normalize_cycles(motions, target_length=100):
    """Resample all cycles to the same length using linear interpolation.
    
    Args:
        motions       : list of np.ndarray, each (N_i, N_joints) — variable length
        target_length : number of frames to resample to (default 100 = percentage)
    
    Returns:
        normalized    : np.ndarray of shape (N_cycles, target_length, N_joints)
    """
    from scipy.interpolate import interp1d

    normalized = []
    for m in motions:
        N = len(m)
        x_old = np.linspace(0, 1, N)
        x_new = np.linspace(0, 1, target_length)
        interp = interp1d(x_old, m, axis=0, kind='linear')
        normalized.append(interp(x_new))

    return np.stack(normalized)

def plot_cycle_variability(all_motions, joints_to_use, joints_to_plot=None,
                           target_length=100, label="Motion",
                           save_path=None, show=True):
    """Plot mean ± std across all normalized cycles for selected joints.
    
    Args:
        all_motions    : list of np.ndarray (variable length cycles)
        joints_to_use  : list of joint names
        joints_to_plot : subset to plot (default: wrist + elbow)
        target_length  : resample length
        label          : title prefix
    """
    if joints_to_plot is None:
        joints_to_plot = [
            "Right_Wrist_Radial_Ulnar_Deviation[rad]",
            "Right_Wrist_Flexion_Extension[rad]",
            "Right_Elbow_Flexion_Extension[rad]",
            "Right_Shoulder_Flexion_Extension[rad]",
        ]
    joints_to_plot = [j for j in joints_to_plot if j in joints_to_use]

    normalized = normalize_cycles(all_motions, target_length)
    n_cycles   = len(normalized)
    t_axis     = np.linspace(0, 100, target_length)
    short_name = lambda s: s.replace("[rad]","").replace("_"," ").strip()

    fig, axes = plt.subplots(len(joints_to_plot), 1,
                             figsize=(10, 2.8 * len(joints_to_plot)), sharex=True)
    if len(joints_to_plot) == 1:
        axes = [axes]
    fig.suptitle(f"{label}  —  {n_cycles} cycles  (mean ± 1 STD)", fontsize=12)

    for ax, jname in zip(axes, joints_to_plot):
        idx  = joints_to_use.index(jname)
        data = np.degrees(normalized[:, :, idx])

        mean = data.mean(axis=0)
        std  = data.std(axis=0)

        for i, cycle in enumerate(data):
            ax.plot(t_axis, cycle, color='steelblue', alpha=0.25, lw=0.8)

        ax.plot(t_axis, mean, color='navy', lw=2, label='Mean')
        ax.fill_between(t_axis, mean - std, mean + std,
                        alpha=0.3, color='steelblue', label='±1 STD')

        ax.set_ylabel(f"{short_name(jname)}\n[deg]", fontsize=8)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=7, loc='upper right')

    axes[-1].set_xlabel("% of motion cycle")
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
    elif show:
        plt.show()
    print(f"[INFO] Cycle STD summary:")
    for jname in joints_to_plot:
        idx  = joints_to_use.index(jname)
        data = np.degrees(normalized[:, :, idx])
        print(f"  {short_name(jname):45s}  mean_std={data.std(axis=0).mean():.2f} deg")

    return normalized, fig

JOINT_TO_CSV = {
    "middle_thoracic_X"      : "Thoracic_Flexion_Extension[rad]",
    "right_clavicle_joint_X" : "Right_Clavicle_Elevation_Depression[rad]",
    "right_shoulder_Z" : "Right_Shoulder_Flexion_Extension[rad]",
    "right_shoulder_X" : "Right_Shoulder_Abduction_Adduction[rad]",
    "right_shoulder_Y" : "Right_Shoulder_Internal_External_Rotation[rad]",
    "right_elbow_Z"    : "Right_Elbow_Flexion_Extension[rad]",
    "right_elbow_Y"    : "Right_Elbow_Pronation_Supination[rad]",
    "right_wrist_Z"    : "Right_Wrist_Flexion_Extension[rad]",
    "right_wrist_X"    : "Right_Wrist_Radial_Ulnar_Deviation[rad]",
}

ACTIVE_JOINTS = [
    "middle_thoracic_X",
    "right_clavicle_joint_X",
    "right_shoulder_Z",
    "right_shoulder_X",
    "right_shoulder_Y",
    "right_elbow_Z",
    "right_elbow_Y",
    "right_wrist_Z",
    "right_wrist_X",
]
def low_pass_filter_data(data, dt, cut_off_frequency, nbutter):
    """Filter from Prof Maxime Gautier (LS2N, Nantes, France)."""
    b, a = signal.butter(nbutter, dt * cut_off_frequency / 2, "low")
    data  = signal.filtfilt(b, a, data, axis=0,
                            padtype="odd", padlen=3 * (max(len(b), len(a)) - 1))
    nbord = 0
    data  = np.delete(data, np.s_[0:nbord], axis=0)
    data  = np.delete(data, np.s_[(data.shape[0] - nbord):data.shape[0]], axis=0)
    return data

def compute_qqdq_for_trajectory(motion_data, joints_to_use,
                                  dt=0.0083,
                                  joint_to_csv=None,
                                  active_joints=None,
                                  cut_off=0.2,
                                  nbutter=5):
    """Filter joint angles and compute dq, ddq via low-pass filter + finite differences.

    Args:
        motion_data   : np.ndarray (N_frames, N_joints)
        joints_to_use : list of CSV column names
        dt            : timestep in seconds
        joint_to_csv  : dict mapping URDF joint name -> CSV column name
        active_joints : list of URDF joint names to extract
        cut_off       : Butterworth cutoff frequency
        nbutter       : Butterworth filter order

    Returns:
        dict with:
            'q_matrix'   : (N,   n_joints)
            'dq_matrix'  : (N-1, n_joints)
            'ddq_matrix' : (N-2, n_joints)
            'joint_names': list of joint names
    """
    if joint_to_csv is None:
        joint_to_csv = JOINT_TO_CSV
    if active_joints is None:
        active_joints = ACTIVE_JOINTS

    N       = len(motion_data)
    n_joints = len(active_joints)

    q_mat   = np.empty((N,     n_joints))
    dq_mat  = np.empty((N - 1, n_joints))
    ddq_mat = np.empty((N - 2, n_joints))

    for i, jname in enumerate(active_joints):
        csv_col = joint_to_csv.get(jname)
        if csv_col is None or csv_col not in joints_to_use:
            print(f"[WARN] '{jname}' -> '{csv_col}' not found, filling zeros")
            q_mat[:, i]   = 0.0
            dq_mat[:, i]  = 0.0
            ddq_mat[:, i] = 0.0
            continue

        raw             = motion_data[:, joints_to_use.index(csv_col)]
        q_mat[:, i]     = low_pass_filter_data(raw, dt, cut_off, nbutter)
        dq_mat[:, i]    = np.diff(q_mat[:, i])   / dt
        ddq_mat[:, i]   = np.diff(dq_mat[:, i])  / dt

    q_mat   = q_mat[:-2, :]
    dq_mat  = dq_mat[:-1, :]

    return {
        'q_matrix'   : q_mat,
        'dq_matrix'  : dq_mat,
        'ddq_matrix' : ddq_mat,
        'joint_names': active_joints,
    }

def compute_qqdq_all_cycles(all_motions, joints_to_use, dt=0.0083,
                              joint_to_csv=None, active_joints=None,
                              cut_off=10, nbutter=5, label="cycles",
                              orig_lengths=None, target_length=None):
    """Apply compute_qqdq_for_trajectory to every cycle.

    For normalized (resampled) cycles, compute per-cycle dt so that
    the physical duration matches the original motion:
        dt_cycle = (orig_len / target_len) * raw_dt

    Parameters
    ----------
    orig_lengths : list[int] or None
        Original (pre-normalization) frame counts. If provided, per-cycle dt
        is used instead of the global dt.
    target_length : int or None
        Normalized length (e.g., 100). Required if orig_lengths is provided.
    """
    use_per_cycle = orig_lengths is not None and target_length is not None

    all_results = []
    for i, m in enumerate(all_motions):
        if use_per_cycle:
            dt_i = (orig_lengths[i] / target_length) * dt
        else:
            dt_i = dt

        res = compute_qqdq_for_trajectory(
            m, joints_to_use, dt=dt_i,
            joint_to_csv=joint_to_csv,
            active_joints=active_joints,
            cut_off=cut_off,
            nbutter=nbutter,
        )
        res['dt'] = float(dt_i)
        all_results.append(res)
        tag = f" dt={dt_i*1000:.2f}ms" if use_per_cycle else ""
        print(f"[INFO] {label} cycle {i:>2}: {len(m):>4} frames{tag}  "
              f"q={res['q_matrix'].shape}  dq={res['dq_matrix'].shape}  ddq={res['ddq_matrix'].shape}")
    return all_results

def save_elaborated_cycles(down_qdq, up_qdq, base_dir,
                           all_down_motions, all_up_motions,
                           df_angles, dt=0.0083, joints_to_use=None,
                           take=None):
    """Save per-cycle npz files and meta.

    If `take` is provided (e.g. "take1", "take2_sensor"), outputs go under
    `base_dir/elaborated/<take>/down|up/` so multiple takes of the same task
    don't collide.
    """
    base_dir = Path(base_dir)
    elab_dir = base_dir / "elaborated"
    if take:
        elab_dir = elab_dir / take
    down_dir = elab_dir / "down"
    up_dir   = elab_dir / "up"
    down_dir.mkdir(parents=True, exist_ok=True)
    up_dir.mkdir(parents=True, exist_ok=True)

    time_col = "Relative_Time[s]" if "Relative_Time[s]" in df_angles.columns else None
    time_arr = df_angles[time_col].to_numpy() if time_col else np.arange(len(df_angles)) * dt

    def get_real_times(motion_raw, joint_col_idx=0):
        """Find where this cycle starts in df_angles by matching joint values."""
        N       = len(motion_raw)
        col     = df_angles[joints_to_use[joint_col_idx]].to_numpy()
        target  = motion_raw[0, joint_col_idx]

        diffs      = np.abs(col - target)
        idx_start  = int(np.argmin(diffs))
        t_start    = float(time_arr[idx_start])
        t_end      = t_start + N * dt
        t_norm     = np.linspace(t_start, t_end, 100)
        return t_start, t_end, t_norm

    for i, (c, m_raw) in enumerate(zip(down_qdq, all_down_motions)):
        t_start, t_end, t_norm = get_real_times(m_raw)
        dt_cycle = c.get('dt', dt)
        path = down_dir / f"cycle_{i:02d}.npz"
        np.savez(path,
                 q_matrix      = c['q_matrix'],
                 dq_matrix     = c['dq_matrix'],
                 ddq_matrix    = c['ddq_matrix'],
                 joint_names   = np.array(c['joint_names']),
                 t_norm        = t_norm,
                 t_real_start  = t_start,
                 t_real_end    = t_end,
                 n_frames_orig = len(m_raw),
                 dt            = dt_cycle,
        )
        print(f"[INFO] Saved DOWN cycle {i:02d}  t=[{t_start:.2f}s - {t_end:.2f}s]  "
              f"orig={len(m_raw)} frames  dt={dt_cycle*1000:.2f}ms")

    for i, (c, m_raw) in enumerate(zip(up_qdq, all_up_motions)):
        t_start, t_end, t_norm = get_real_times(m_raw)
        dt_cycle = c.get('dt', dt)
        path = up_dir / f"cycle_{i:02d}.npz"
        np.savez(path,
                 q_matrix      = c['q_matrix'],
                 dq_matrix     = c['dq_matrix'],
                 ddq_matrix    = c['ddq_matrix'],
                 joint_names   = np.array(c['joint_names']),
                 t_norm        = t_norm,
                 t_real_start  = t_start,
                 t_real_end    = t_end,
                 n_frames_orig = len(m_raw),
                 dt            = dt_cycle,
        )
        print(f"[INFO] Saved UP   cycle {i:02d}  t=[{t_start:.2f}s - {t_end:.2f}s]  "
              f"orig={len(m_raw)} frames  dt={dt_cycle*1000:.2f}ms")

    np.savez(elab_dir / "meta.npz",
             dt            = dt,
             joint_names   = np.array(down_qdq[0]['joint_names']),
             n_down        = len(down_qdq),
             n_up          = len(up_qdq),
             q_down_all    = np.stack([c['q_matrix']   for c in down_qdq]),
             dq_down_all   = np.stack([c['dq_matrix']  for c in down_qdq]),
             ddq_down_all  = np.stack([c['ddq_matrix'] for c in down_qdq]),
             q_up_all      = np.stack([c['q_matrix']   for c in up_qdq]),
             dq_up_all     = np.stack([c['dq_matrix']  for c in up_qdq]),
             ddq_up_all    = np.stack([c['ddq_matrix'] for c in up_qdq]),
             n_frames_down = np.array([len(m) for m in all_down_motions]),
             n_frames_up   = np.array([len(m) for m in all_up_motions]),
    )
    print(f"\n[INFO] Saved meta -> {elab_dir / 'meta.npz'}")
    print(f"[INFO] DOWN: {len(down_qdq)} cycles,  UP: {len(up_qdq)} cycles")
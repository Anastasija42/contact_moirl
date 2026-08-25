"""
irl_utils_setup.py
==================
Shared setup utilities for IRL experiments with CPU MPPI.
Loads data, builds human_mppi, creates demo trajectories.

Usage:
    from irl_utils_setup import setup_experiment
    human_mppi, xs_opt_list, us_opt_list, human_ocp = setup_experiment()
"""

import os
import numpy as np
import pandas as pd
import pinocchio as pin
from pathlib import Path
from scipy.signal import savgol_filter, medfilt
from final_models import HumanMPPI
from utils_slice_trajectories import get_stick_static_position

RAW_FORCE_ROOT = Path(os.environ.get("TOOL_HANDLING_RAW_FORCE_ROOT", Path.home() / "Desktop"))
FORCE_OFFSET_SEC = 104.0

def _smooth_force(forces, dt, window_sec=0.5, polyorder=2,
                  median_sec=0.05):
    """Smooth a force profile to a stable, non-negative target.
    1) Median filter (kills sharp spikes / dropouts).
    2) Savitzky-Golay with a wider window + lower polyorder (no overshoot).
    3) Clamp at zero (negative force is sensor noise / savgol overshoot)."""
    arr = np.asarray(forces, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return arr

    m = max(3, int(round(median_sec / dt)) | 1)
    if m > n:
        m = n if n % 2 else n - 1
    if m >= 3:
        arr = medfilt(arr, kernel_size=m)

    win = int(round(window_sec / dt))
    if win % 2 == 0:
        win += 1
    if win > n:
        win = n if n % 2 == 1 else n - 1
    if win > polyorder:
        arr = savgol_filter(arr, win, polyorder)

    return np.maximum(arr, 0.0)

def _align_raw_force(raw_path, angles_csv, date, offset_sec=FORCE_OFFSET_SEC):
    """Mirror of slice_and_plot_force.align_force; returns df with mocap_ref_time."""
    f = pd.read_csv(raw_path)
    f["human_time"] = pd.to_datetime(f["human_time"], format="%H:%M:%S.%f")
    start_row = pd.read_csv(angles_csv, nrows=1)
    m_start = pd.to_datetime(start_row["Capture_Start_Time"].iloc[0],
                             format="%Y-%m-%d %I.%M.%S.%f %p")
    day_str, month_str = date.split("_")
    f["human_time"] = f["human_time"].apply(
        lambda t: t.replace(year=m_start.year,
                            month=int(month_str),
                            day=int(day_str)))
    f["mocap_ref_time"] = (
        (f["human_time"] - m_start).dt.total_seconds() + offset_sec)
    return f

def _resample_by_cycle_progress(ref_force, T):
    """Map a (N,) reference profile onto T samples by normalized cycle progress.
    Use when source and target cycles come from different recording sessions —
    absolute mocap_ref_time isn't comparable, but cycle phase 0..1 is."""
    N = len(ref_force)
    if N < 2 or T < 1:
        return None
    u_ref  = np.linspace(0.0, 1.0, N)
    u_curr = np.linspace(0.0, 1.0, T)
    return np.interp(u_curr, u_ref, ref_force)

def load_smoothed_force_profile(base_dir, id_, task, direction, cycle_idx,
                                 date, sliced_times, dt,
                                 data_root="trajectories_from_mocap",
                                 reference_date="27_02_sensor"):
    """Load measured normal force for one cycle, resampled onto sliced_times.

    Fallback chain:
      1) per-cycle slice CSV at this date
      2) raw recording at this date — on-the-fly align+slice
      3) reference cycle from `reference_date` (force is only recorded in the
         27_02_sensor take), resampled by normalized cycle progress (not abs time)

    Always Savitzky-Golay smoothed. Returns (forces[T], source) or (None, "missing").
    """
    T = len(sliced_times)

    force_csv = base_dir / f"elaborated/force_slices/{direction}/cycle_{cycle_idx:02d}.csv"
    if force_csv.exists():
        df = pd.read_csv(force_csv)
        forces = np.interp(sliced_times,
                           df["mocap_ref_time"].values,
                           -df["fz"].values)
        return _smooth_force(forces, dt), str(force_csv)

    raw_path = RAW_FORCE_ROOT / f"recording_{date}" / "force" / id_ / f"{task}.csv"
    angles_csv = base_dir / "ik_joint_angles.csv"
    if raw_path.exists() and angles_csv.exists():
        f = _align_raw_force(raw_path, angles_csv, date)
        mref = f["mocap_ref_time"].values
        forces = np.interp(sliced_times, mref, -f["fz"].values)
        n_inside = int(np.sum((sliced_times >= mref.min()) &
                              (sliced_times <= mref.max())))
        if n_inside >= 2 and float(np.ptp(forces)) > 1.0:
            return _smooth_force(forces, dt), f"raw:{raw_path}"

    if reference_date != date:
        ref_dir = (Path(data_root) / reference_date / id_ / task /
                   f"elaborated/force_slices/{direction}")
        ref_csv = ref_dir / f"cycle_{cycle_idx:02d}.csv"
        if not ref_csv.exists() and ref_dir.exists():
            cands = sorted(ref_dir.glob("cycle_*.csv"))
            if cands:
                ref_csv = cands[0]
        if ref_csv.exists():
            df = pd.read_csv(ref_csv)
            forces = _resample_by_cycle_progress(-df["fz"].values, T)
            if forces is not None:
                return _smooth_force(forces, dt), f"reference:{ref_csv}"

    return None, "missing"

DEFAULT_CONFIG = {
    'ID': 'S2',
    'TASK': 'down_long',
    'DIRECTION': 'down',
    'DATE': '13_02',
    'TAKE': None,
    'CYCLE_INDICES': [5],
    'SLICE_START': 30,
    'SLICE_END': -1,
    'DT': 0.0083,
    'MU': 0.3,
    'TARGET_FORCE': 60.0,
}

def get_model_tau(model, data, q, dq, ddq):
    """Compute inverse dynamics torques via Pinocchio RNEA."""
    tau = []
    for q_, dq_, ddq_ in zip(q, dq, ddq):
        pin.rnea(model, data, q_, dq_, ddq_)
        tau.append(data.tau.copy())
    return np.stack(tau)

def setup_experiment(config=None, build_mppi=True, build_ocp=True,
                     mppi_w_run=None):
    """
    Build everything needed for IRL experiments.

    Returns: (human_mppi, xs_opt_list, us_opt_list, human_ocp, config)
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    ID = cfg['ID']
    TASK = cfg['TASK']
    DIRECTION = cfg['DIRECTION']
    DATE = cfg['DATE']
    TAKE = cfg.get('TAKE', None)
    CYCLE_INDICES = cfg['CYCLE_INDICES']
    SLICE_START = cfg['SLICE_START']
    SLICE_END = cfg['SLICE_END']

    DATA_ROOT = cfg.get('DATA_ROOT', 'trajectories_from_mocap')
    base_dir = Path(f"{DATA_ROOT}/{DATE}/{ID}/{TASK}")
    elab_sub = f"elaborated/{TAKE}/{DIRECTION}" if TAKE else f"elaborated/{DIRECTION}"

    print(f"[setup] Subject={ID}, Task={TASK}, Direction={DIRECTION}")
    print(f"[setup] Cycles={CYCLE_INDICES}, Slice=[{SLICE_START}:{SLICE_END}]")

    joints_data_full = pd.read_csv(base_dir / "ik_joint_angles.csv")
    q_matrix_full = joints_data_full.to_numpy()
    q0_full = q_matrix_full[0][2:]

    ref_cycle = CYCLE_INDICES[0]
    cycle_data_ref = np.load(
        base_dir / f"{elab_sub}/cycle_{ref_cycle:02d}.npz",
        allow_pickle=True)
    q_ref = cycle_data_ref["q_matrix"].astype(np.float64)
    cycle_keys = list(cycle_data_ref["joint_names"]) \
        if "joint_names" in cycle_data_ref.files else []
    extras_needed = [j for j in ("middle_thoracic_X", "right_clavicle_joint_X")
                      if j not in cycle_keys]
    if extras_needed:
        from utils_slice_trajectories import JOINT_TO_CSV
        ik_csv = base_dir / "ik_joint_angles.csv"
        df_ik = pd.read_csv(ik_csv)
        t_all_csv = df_ik["Relative_Time[s]"].to_numpy(dtype=np.float64)
        t_start = float(cycle_data_ref["t_real_start"]) \
            if "t_real_start" in cycle_data_ref.files else 0.0
        t_end_cyc = float(cycle_data_ref["t_real_end"]) \
            if "t_real_end" in cycle_data_ref.files else 1.0
        t_real = np.linspace(t_start, t_end_cyc, len(q_ref))
        extra_q = np.zeros((len(q_ref), len(extras_needed)))
        for k, jname in enumerate(extras_needed):
            csv_col = JOINT_TO_CSV[jname]
            col_vals = df_ik[csv_col].to_numpy(dtype=np.float64)
            idxs = np.array([int(np.abs(t_all_csv - t).argmin()) for t in t_real])
            extra_q[:, k] = col_vals[idxs]
        q_ref = np.hstack([extra_q, q_ref])
    _end = SLICE_END if SLICE_END is not None else len(q_ref)
    q_ref_sliced = q_ref[SLICE_START:_end]
    q_traj = np.hstack([q_ref_sliced, np.zeros((len(q_ref_sliced), 1))])
    q0_reduced = q_traj[0]

    if 'dt' in cycle_data_ref.files:
        dt_cycle = float(cycle_data_ref['dt'])
    else:
        dt_cycle = cfg['DT']
        print(f"[setup] WARNING: no 'dt' in {ref_cycle:02d}.npz — re-run section_trajectories to fix velocity scaling")
    print(f"[setup] T={len(q_traj)-1}, nq={q_traj.shape[1]}, dt={dt_cycle*1000:.2f}ms")

    human_ocp = None
    if build_ocp:
        ocp_args = {
            "subject_id": ID.lower(),
            "date": DATE,
            "task": TASK,
            "mu": cfg['MU'],
            "target_force": cfg['TARGET_FORCE'],
            "contact": "automatic",
            "dt": dt_cycle,
            "w_run": {"Tau": 1e-6, "JA": 1e-6, "JV": 1e-6,
                      "JTC": 1e-6, "Geo": 1e-6, "Eng": 1e-6, "press_force": 1e-6},
            "w_term": {"JV": 1e-6, "Geo": 1e-6, "JTC": 1e-6, "JA": 1e-6},
            "solver_type": "CSQP",
            "q0_full": q0_full,
            "T": len(q_traj) - 1,
            "q_traj": q_traj,
            "q0": q0_reduced,
        }
        from final_models import HumanCrocoddyl as Human
        human_ocp = Human(args=ocp_args)
        print(f"[OCP] nq={human_ocp.nq}, T={human_ocp.T}")

    human_mppi = None
    if build_mppi:
        if mppi_w_run is None:
            from mppi_cpu import KEYS_RUN
            mppi_w_run = {k: 1e-4 for k in KEYS_RUN}

        mppi_args = {
            "subject_id": ID.lower(),
            "date": DATE,
            "task": TASK,
            "mu": cfg['MU'],
            "target_force": cfg['TARGET_FORCE'],
            "contact": "automatic",
            "dt": dt_cycle,
            "T": len(q_traj) - 1,
            "q0_full": q0_full,
            "q_traj": q_traj,
            "q0": q0_reduced,
            "use_gpu": False,
            "stick_static": False,
            "horizon": 20,
            "num_samples": 128,
            "noise_sigma": 20.0,
            "lambda_": 0.1,
            "w_run": mppi_w_run,
            "w_term": {},
            "primitive_rock": cfg.get('PRIMITIVE_ROCK', False),
            "rock_radius": cfg.get('ROCK_RADIUS', 0.03),
            "rock_penetration": cfg.get('ROCK_PENETRATION', 0.0),
        }
        human_mppi = HumanMPPI(mppi_args)
        print(f"[MPPI] nq={human_mppi.nq}, T={human_mppi.T}")

    xs_opt_list, us_opt_list = [], []
    for cidx in CYCLE_INDICES:
        cd = np.load(
            base_dir / f"{elab_sub}/cycle_{cidx:02d}.npz",
            allow_pickle=True)
        q = cd["q_matrix"].astype(np.float64)
        dq = cd["dq_matrix"].astype(np.float64)
        ddq = cd["ddq_matrix"].astype(np.float64)

        cycle_keys = list(cd["joint_names"]) if "joint_names" in cd.files else []
        extras_needed = [j for j in ("middle_thoracic_X", "right_clavicle_joint_X")
                          if j not in cycle_keys]
        if extras_needed:
            from utils_slice_trajectories import JOINT_TO_CSV
            ik_csv = base_dir / "ik_joint_angles.csv"
            df_ik = pd.read_csv(ik_csv)
            t_all_csv = df_ik["Relative_Time[s]"].to_numpy(dtype=np.float64)
            t_start = float(cd["t_real_start"]) if "t_real_start" in cd.files else 0.0
            t_end   = float(cd["t_real_end"])   if "t_real_end"   in cd.files else 1.0
            t_real  = np.linspace(t_start, t_end, len(q))
            extra_q   = np.zeros((len(q), len(extras_needed)))
            extra_dq  = np.zeros((len(q), len(extras_needed)))
            extra_ddq = np.zeros((len(q), len(extras_needed)))
            for k, jname in enumerate(extras_needed):
                csv_col = JOINT_TO_CSV[jname]
                col_vals = df_ik[csv_col].to_numpy(dtype=np.float64)
                idxs = np.array([int(np.abs(t_all_csv - t).argmin()) for t in t_real])
                extra_q[:, k] = col_vals[idxs]
                if len(q) > 1:
                    dt_cyc = (float(cd["dt"]) if "dt" in cd.files
                              else (t_end - t_start) / max(len(q) - 1, 1))
                    extra_dq[:, k]  = np.gradient(extra_q[:, k], dt_cyc)
                    extra_ddq[:, k] = np.gradient(extra_dq[:, k], dt_cyc)
            q   = np.hstack([extra_q,   q])
            dq  = np.hstack([extra_dq,  dq])
            ddq = np.hstack([extra_ddq, ddq])

        q_full = np.hstack([q, np.zeros((len(q), 1))])
        dq_full = np.hstack([dq, np.zeros((len(dq), 1))])
        ddq_full = np.hstack([ddq, np.zeros((len(ddq), 1))])

        _e = SLICE_END if SLICE_END is not None else len(q_full)
        xs = np.hstack([q_full, dq_full])[SLICE_START:_e]

        tau_model = human_ocp if human_ocp else human_mppi
        us = get_model_tau(tau_model.pin_model, tau_model.pin_data,
                           q_full, dq_full, ddq_full)[SLICE_START:_e]
        xs_opt_list.append(xs)
        us_opt_list.append(us)

    print(f"[demos] {len(xs_opt_list)} demos, lengths={[len(x) for x in xs_opt_list]}")

    if human_mppi is not None and xs_opt_list:
        nq = human_mppi.nq
        human_mppi.v0 = xs_opt_list[0][0, nq:].copy()
        human_mppi.x0 = xs_opt_list[0][0].copy()
        print(f"[setup] v0 from demo: |v0| = {np.linalg.norm(human_mppi.v0):.4f}")

    if human_mppi is not None:
        # Build the cycle-time grid the force profile must align to
        cd_ref = np.load(
            base_dir / f"{elab_sub}/cycle_{CYCLE_INDICES[0]:02d}.npz",
            allow_pickle=True)
        t_start = float(cd_ref['t_real_start'])
        t_end = float(cd_ref['t_real_end'])
        q_ref_full = cd_ref['q_matrix']
        motion_times = np.linspace(t_start, t_end, len(q_ref_full))
        sliced_times = motion_times[SLICE_START:(_e if SLICE_END is None else SLICE_END)]

        measured_forces, src = load_smoothed_force_profile(
            base_dir, ID, TASK, DIRECTION, CYCLE_INDICES[0],
            DATE, sliced_times, dt_cycle,
            data_root=DATA_ROOT,
            reference_date=cfg.get('FORCE_REFERENCE_DATE', '27_02_sensor'),
        )
        if measured_forces is not None:
            print(f"[force] Smoothed force from {src}: "
                  f"mean={measured_forces.mean():.1f}N, max={measured_forces.max():.1f}N")
        else:
            print(f"[force] No force data for {ID}/{DATE}/{TASK} — using Pinocchio torques")

        target_forces = None
        _ca_tau = cfg.get('CONTACT_AWARE_TAU', False)
        _consistent = cfg.get('CONSISTENT_DEMO', False)
        _fgain = float(cfg.get('DEMO_FORCE_GAIN', 0.6)) if _consistent else 0.25
        _freg = os.environ.get('FORCE_REGULATE', '0') != '0'
        _freg_kp = float(os.environ.get('FORCE_REG_KP', '0.4'))
        _freg_ki = float(os.environ.get('FORCE_REG_KI', '0.15'))
        if _freg:
            print(f"[force] FORCE_REGULATE ON — closed-loop demo press (kp={_freg_kp}, ki={_freg_ki})")
        for i, xs_i in enumerate(xs_opt_list):
            if _consistent:
                us_i, forces_i, qtr_i, vtr_i = human_mppi.get_tau_from_trajectory(
                    xs_i, force_profile=measured_forces, force_gain=_fgain,
                    record_contact_aware=False, return_qtrack=True,
                    force_regulate=_freg, force_reg_kp=_freg_kp, force_reg_ki=_freg_ki)
                nj_a = qtr_i.shape[1]; nqx = xs_i.shape[1] // 2
                xs_new = xs_i.copy()
                xs_new[1:len(qtr_i) + 1, :nj_a] = qtr_i
                xs_new[1:len(vtr_i) + 1, nqx:nqx + nj_a] = vtr_i
                xs_opt_list[i] = xs_new
                us_opt_list[i] = us_i
                if i == 0:
                    target_forces = forces_i
            else:
                if i == 0:
                    us_i, forces_i = human_mppi.get_tau_from_trajectory(
                        xs_i, force_profile=measured_forces, force_gain=0.25,
                        record_contact_aware=_ca_tau,
                        force_regulate=_freg, force_reg_kp=_freg_kp, force_reg_ki=_freg_ki)
                    target_forces = forces_i
                else:
                    us_i, _ = human_mppi.get_tau_from_trajectory(
                        xs_i, force_profile=measured_forces,
                        record_contact_aware=_ca_tau,
                        force_regulate=_freg, force_reg_kp=_freg_kp, force_reg_ki=_freg_ki)
                us_opt_list[i] = us_i
            print(f"  Demo {i}: us={us_opt_list[i].shape}, |tau| mean={np.linalg.norm(us_opt_list[i], axis=1).mean():.2f}"
                  + (f", track_q Δmocap={np.rad2deg(np.sqrt(((xs_opt_list[i][1:,:qtr_i.shape[1]]-xs_i[1:,:qtr_i.shape[1]])**2).mean())):.1f}°" if _consistent else ""))

        us_opt = us_opt_list[0]

        if target_forces is not None:
            human_mppi.set_demo_forces(target_forces, xs_opt_list[0])
            print(f"[force] MPPI demo forces (MuJoCo-recorded): "
                  f"mean={target_forces.mean():.1f}N, max={target_forces.max():.1f}N")

        if measured_forces is not None:
            pressing_mask = measured_forces > 5.0
            pressing = measured_forces[pressing_mask]
            T_total = len(measured_forces)
            FALLBACK_FRAC    = 0.30
            FALLBACK_TARGET  = 30.0
            if len(pressing) > 0:
                target_const  = float(np.mean(pressing))
                target_const  = max(target_const, 5.0)
                T_press_start = int(np.argmax(pressing_mask))
                src = "demo pressing-phase mean"
            else:
                target_const  = FALLBACK_TARGET
                T_press_start = int(FALLBACK_FRAC * T_total)
                src = (f"FALLBACK (no detected contact in profile, "
                       f"range {measured_forces.min():.1f}–{measured_forces.max():.1f}N)")
            human_mppi.target_force   = target_const
            human_mppi.T_press_start  = T_press_start
            if hasattr(human_mppi, 'set_force_target'):
                human_mppi.set_force_target(measured_forces)
            print(f"[force] press_force target = per-step smoothed profile "
                  f"(mean={float(np.mean(measured_forces)):.1f}N, "
                  f"max={float(np.max(measured_forces)):.1f}N, "
                  f"fallback constant {target_const:.1f}N from {src})")
            print(f"[force] T_press_start = {T_press_start}/{T_total} "
                  f"({100.0*T_press_start/max(T_total,1):.0f}% of cycle) — "
                  f"press_force feature gated off before this index")
            if config is not None and config.get('STICK_APPROACH', False) \
                    and getattr(human_mppi, 'stick_traj', None) is not None:
                _st = human_mppi.stick_traj
                _ps = int(min(max(T_press_start, 0), len(_st) - 1))
                if _ps > 0:
                    _st[:_ps] = _st[_ps]
                    _stc = getattr(human_mppi, '_stick_traj_ctrl', None)
                    if _stc is not None and len(_stc) > _ps:
                        _stc[:_ps] = _stc[_ps]
                    print(f"[stick-approach] froze stick at its t={_ps} contact position for "
                          f"frames 0..{_ps-1} → rock approaches a stationary stick (real gap + approach)")

    if human_ocp is not None and 'measured_forces' in locals() and measured_forces is not None:
        if hasattr(human_ocp, 'set_force_target_profile'):
            human_ocp.set_force_target_profile(measured_forces)
            print(f"[force] OCP press_force + actuation friction = smoothed profile (per-step)")

    print(f"[demos] Final: {len(xs_opt_list)} demos")

    return human_mppi, xs_opt_list, us_opt_list, human_ocp, cfg

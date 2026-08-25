"""Demonstration preparation for the CSQP driver.

Loading a measured or averaged press-force profile, resampling a recorded cycle
onto the solver's node count, projecting recorded motion onto controls the
dynamics can actually produce, and picking the most representative cycles per
subject. Separated from the driver because none of it depends on the run being
set up -- it is the step that turns recordings into demonstrations.
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import pinocchio as pin
from scipy.signal import savgol_filter

from irl_utils_setup import load_smoothed_force_profile
from repo_paths import REPO
from run_csqp_identifiability import (DATE, TASK_DIR, SLICE_START, SLICE_END,
                                      load_geometry)

def resample_q(q_traj, T_target):
    """Resample a (T+1, nq) trajectory to (T_target+1, nq) by normalized phase."""
    n = len(q_traj)
    if n == T_target + 1:
        return q_traj
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, T_target + 1)
    return np.column_stack([np.interp(dst, src, q_traj[:, j])
                            for j in range(q_traj.shape[1])])

def cosine(a, b, idx):
    a = np.asarray(a, float)[idx]; b = np.asarray(b, float)[idx]
    a = a / max(np.max(np.abs(a)), 1e-12); b = b / max(np.max(np.abs(b)), 1e-12)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def load_measured_force(subj, task, cyc, T_target):
    """Measured normal contact force for one cycle (Savitzky-Golay smoothed),
    resampled to T_target+1 nodes. Returns None if no force file. Mirrors the
    force loading in run_moirl_batch.run_combo."""
    from irl_utils_setup import load_smoothed_force_profile
    base = REPO / f"trajectories_from_mocap/{DATE}/{subj}/{task}"
    p = base / f"elaborated/{TASK_DIR[task]}/cycle_{cyc:02d}.npz"
    cd = np.load(p, allow_pickle=True)
    t_s = float(cd["t_real_start"]); t_e = float(cd["t_real_end"])
    qn = len(cd["q_matrix"])
    sliced_times = np.linspace(t_s, t_e, qn)[SLICE_START:SLICE_END]
    dt_cyc = float(cd["dt"]) if "dt" in cd.files else 0.0083
    forces, _ = load_smoothed_force_profile(base, subj, task, TASK_DIR[task], cyc,
                                            DATE, sliced_times, dt_cyc)
    if forces is None:
        return None
    return resample_q(np.asarray(forces, float)[:, None], T_target)[:, 0]

def load_avg_force_profile(subj, task, T_target, fill=False):
    """Subject/task-AVERAGED press force (= -fz, positive) over the stroke,
    resampled to T_target+1 nodes. This is the per-(subject,task) press_force
    reference: average the aligned force slices (one typical pressing trajectory
    per subject/stroke) instead of one noisy per-cycle trace. The CSQP contact
    window then selects where it applies. Force lives in the 27_02_sensor tree.
    Returns None if no slices.

    fill=True: sustained contact means no mid-stroke contact loss, so raise any
    interior VALLEY up to the lower flanking peak (cummax from both ends -> min),
    then lightly smooth. Peaks are preserved; only spurious dips move -- this
    cleans noisy light pressers (S1) and barely touches the clean averages."""
    cdir = (REPO / f"trajectories_from_mocap/27_02_sensor/{subj}/{task}/"
            f"elaborated/force_slices/{TASK_DIR[task]}")
    grid = np.linspace(0.0, 1.0, T_target + 1)
    profs = []
    for f in sorted(cdir.glob("cycle_*.csv")):
        df = pd.read_csv(f)
        if len(df) < 4:
            continue
        press = -df["fz"].to_numpy(float)
        profs.append(np.interp(grid, np.linspace(0, 1, len(press)), press))
    if not profs:
        return None
    prof = np.clip(np.mean(profs, axis=0), 0.0, None)
    if fill:
        prof = np.minimum(np.maximum.accumulate(prof),
                          np.maximum.accumulate(prof[::-1])[::-1])
        try:
            from scipy.signal import savgol_filter
            prof = savgol_filter(prof, 11, 3)
        except Exception:  # noqa: BLE001
            prof = np.convolve(prof, np.ones(5) / 5.0, mode="same")
        prof = np.clip(prof, 0.0, None)
    return prof

def consistent_us(human, q, dq, ddq, contact_aware_demo=False, bake_press=False,
                  contact_consistent_accel=False):
    """Demo torques consistent with the model's CURRENT per-step actuation.

    Default (contact_aware_demo=False): us[t] = u_RNEA[t] - tau_bias[t], where
    tau_bias is the model actuation at u=0 (so actuation(us) = u_RNEA exactly ->
    Tau = ||u_RNEA||^2, no shift after set_force_target_profile changed f_n). This
    cancels the press feedforward -> the demo is a NON-pressing fixed point (its
    contact reaction lambda ~ 0, press_force feature ~ (0-target)^2), so the IRL
    learns "don't press" regardless of the cost target.

    contact_aware_demo=True: us[t] = u_RNEA[t] (do NOT subtract tau_bias). The
    actuation feedforward then stays in the net torque; the quasi-static contact
    reaction cancels it in the EOM (Jc^T lambda = -tau_bias), so the SAME motion
    is reproduced BUT the demo now presses with lambda ~ target_force. The demo's
    press_force feature ~ 0 (at target) and its Tau/effort feature includes the
    pressing torque -> the IRL has a demo that genuinely presses and must explain
    it. This is the notebook's "force-bearing trajectory" behaviour. See
    project_irl_phi_contact_aware_tau.

    bake_press=True (press_normal_dual scheme):
    the demo's press is put DIRECTLY in the torque instead of via the actuation
    feedforward. us[t] = (u_RNEA - tau_bias) - Jc^T f_n, where Jc is the model's
    EXACT 1D contact row (R_surface.row(2)) and f_n the measured profile. Then the
    contact dual replays at f_n (press in the torque) AND tau_act = us + Jc^T
    f_fric = u_RNEA - Jc^T f_n cancels the friction feedforward -> the demo's
    effort feature is computed the SAME way the rollout's is (u_solved, tau_act =
    u + Jc^T f_fric). Symmetric by construction, no demo/rollout feature branch.
    Requires human.target_force_profile set (set_force_target_profile)."""
    import pinocchio as pin
    pin_data = human.pin_model.createData()
    nu, T = human.nu, len(q) - 1
    nv = human.pin_model.nv
    nvj = min(nu, nv)
    us = np.zeros((T, nu))
    profile = getattr(human, 'target_force_profile', None) if bake_press else None
    cmask = getattr(human, '_contact_mask', None)
    for t in range(T):
        _Jloc = pin.computeFrameJacobian(human.pin_model, pin_data, q[t],
                                         human.contact_frame_id, pin.LOCAL)
        R_t = (np.asarray(human._R_surface_tv[t], float)
               if getattr(human, '_R_surface_tv', None) is not None
               and t < len(human._R_surface_tv)
               else np.asarray(human.R_surface if human.R_surface is not None else np.eye(3), float))
        J_c = (R_t @ _Jloc[:3])[2]
        ddq_bake = ddq[t]
        _active = (cmask is None or (t < len(cmask) and cmask[t]))
        if contact_consistent_accel and _active:
            pin.forwardKinematics(human.pin_model, pin_data, q[t], dq[t], np.zeros(nv))
            pin.updateFramePlacements(human.pin_model, pin_data)
            _drift = float((R_t @ pin.getFrameClassicalAcceleration(
                human.pin_model, pin_data, human.contact_frame_id, pin.LOCAL).linear)[2])
            _aerr = float(J_c @ ddq[t]) + _drift
            ddq_bake = ddq[t] - J_c * (_aerr / (float(J_c @ J_c) + 1e-9))
        pin.rnea(human.pin_model, pin_data, q[t], dq[t], ddq_bake)
        u_rnea = np.asarray(pin_data.tau)[:nu].copy()
        if contact_aware_demo and not bake_press:
            us[t] = u_rnea
            continue
        rm = human.solver.problem.runningModels[t]
        rd = human.solver.problem.runningDatas[t]
        x = np.concatenate([q[t], dq[t]])
        rm.differential.actuation.calc(rd.differential.multibody.actuation, x, np.zeros(nu))
        tau_bias = np.asarray(rd.differential.multibody.actuation.tau)[:nu].copy()
        us[t] = u_rnea - tau_bias
        if bake_press and profile is not None and _active:
            fn = float(profile[t]) if t < len(profile) else float(profile[-1])
            us[t, :nvj] -= J_c[:nvj] * fn
    return us

def tv_wstar_windows(wstar, keys, T, ramp_keys, lo=0.15, center=0.55, width=0.10):
    """Time-relative ground-truth W*(t): exertion features (ramp_keys) grow from
    lo*base early to base late along a smooth reach->scrape sigmoid; everything
    else constant. Returns a (T, nr) per-node weight array ordered by `keys`."""
    t = np.linspace(0.0, 1.0, T)
    s = 1.0 / (1.0 + np.exp(-(t - center) / width))
    s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    prof = lo + (1.0 - lo) * s
    W = np.zeros((T, len(keys)))
    for j, k in enumerate(keys):
        base = float(wstar.get(k, 0.0))
        W[:, j] = base * prof if k in ramp_keys else base
    return W

def select_representative_cycles(subject, task, n, dur_band=(0.6, 1.6), L=50):
    """Pick the n most REPRESENTATIVE cycles for a subject: lowest joint-space
    RMS distance to that subject's MEDIAN cycle (robust centroid), after dropping
    duration outliers (dur outside dur_band x median duration — the S1 lesson:
    extreme-duration cycles mis-scale features under the shared common-T pooling).
    Returns a list of cycle ids, most-representative first."""
    import glob, os, re
    base = f"trajectories_from_mocap/{DATE}/{subject}/{task}/elaborated/{TASK_DIR[task]}"
    ids = sorted(int(re.search(r"cycle_(\d+)", f).group(1))
                 for f in glob.glob(os.path.join(base, "cycle_*.npz")))
    recs = []
    for c in ids:
        try:
            _, qt, dt = load_geometry(subject, task, c)
        except Exception:
            continue
        recs.append((c, resample_q(qt, L - 1), (len(qt) - 1) * dt))
    Q = np.stack([r[1] for r in recs])
    durs = np.array([r[2] for r in recs])
    med = np.median(Q, axis=0)
    dist = np.sqrt(((Q - med) ** 2).mean(axis=(1, 2)))
    med_dur = np.median(durs)
    ok = [(recs[i][0], dist[i]) for i in range(len(recs))
          if dur_band[0] * med_dur <= durs[i] <= dur_band[1] * med_dur]
    ok.sort(key=lambda x: x[1])
    chosen = [c for c, _ in ok[:n]]
    print(f"[rep] {subject}/{task}: median dur={med_dur*1e3:.0f}ms -> "
          f"representative cycles {chosen}")
    return chosen

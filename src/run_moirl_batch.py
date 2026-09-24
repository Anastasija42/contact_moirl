"""run_moirl_batch.py — automate MO-IRL across the combos defined in
moirl_config.py.

For every (combo, method) it:
  1. loads the demos listed in the combo (each demo: subject × task × cycle),
  2. builds a common Human (OCP, model_mocap) and HumanMPPI for
     `model_subject` so IRL has one reference kinematics/actuation pipeline,
  3. runs the requested method and dumps:
       analysis/moirl/{combo}/{method}/weights.json
       analysis/moirl/{combo}/{method}/xs_irl.npz
       analysis/moirl/{combo}/{method}/plot.png
  4. appends a row to analysis/moirl/summary.csv.

Methods available:
  ocp          – solve Crocoddyl OCP once per demo; no IRL.
  mppi_vp      – MPPI IRL learning ONLY 'progress_vel'.
  mppi_2phase  – run_phase1 + run_phase2 (task + style).
  mppi_3phase  – phase1 + phase2 + phase3 (phase3 = all features, warm-started).

Run:
  python run_moirl_batch.py                          # every combo × every method
  python run_moirl_batch.py --combos s3_dl
  python run_moirl_batch.py --methods ocp mppi_vp
  python run_moirl_batch.py --combos all_dl --methods mppi_3phase
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
import traceback
from pathlib import Path

@contextlib.contextmanager
def _quiet():
    """Suppress stdout — used to silence noisy URDF/IK/MuJoCo build chatter
    (Loading URDF, Locking joints, Syncing debug prints, etc.). Errors still
    propagate via exceptions; only print() output is swallowed."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pinocchio as pin

from final_models import HumanCrocoddyl as Human, HumanMPPI
from mppi_cpu import KinematicMPPI_CPU, KEYS_RUN as MPPI_KEYS

from mppi_mjx_kinematic import KinematicMPPI_MJX                                                                                                         
  
from MO_IRL import MO_IRL
from irl_phases import run_phase1, run_phase2, run_phase3
from moirl_config import COMBOS, METHODS, DATE

_BUILD_MU = 0.3
_BUILD_TARGET_FORCE = 60.0
_VERBOSE_IRL = False
_CONTACT_AWARE_COST = False
_CSQP_TAU_SPLIT = "none"
_TAU_GROUPS = ("thoracic", "clavicle", "shoulder", "elbow", "wrist")
_PROXIMAL_TAU = ("shoulder", "clavicle", "thoracic")

def _tau_run_keys():
    """Torque feature keys for the current tau_split mode (replaces global Tau)."""
    if _CSQP_TAU_SPLIT == "full":
        return [f"Tau_{g}" for g in _TAU_GROUPS]
    if _CSQP_TAU_SPLIT == "pruned":
        return [f"Tau_{g}" for g in _PROXIMAL_TAU]
    return ["Tau"]
_KIN_BACKEND = 'cpu'
_MPPI_K = 256
_MPPI_H = 40
_MPPI_NDIAL = 3
_MPPI_NOISE_SCALE = 1.0
_MPPI_QUIET_SOLVE = False
_MPPI_PRESS_EMERGENT  = False
_MPPI_PROGRESS_VEL_SQ = False
_MPPI_FORCE_MAX       = 80.0
_MPPI_P2_FEATURES = None
_MPPI_P1_DICT = None
_FORCE_TARGET_OVERRIDE = None
_FRICTION_AWARE_TAU = True
from irl_utils_setup import load_smoothed_force_profile

REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO / "analysis/moirl"
SLICE_START = 30
SLICE_END   = -1
TASK_DIR = {"up_long": "up", "up_short": "up",
            "down_long": "down", "down_short": "down"}

def _rnea_tau(pin_model, pin_data, q_full, dq_full, ddq_full):
    taus = []
    for q, dq, ddq in zip(q_full, dq_full, ddq_full):
        pin.rnea(pin_model, pin_data, q, dq, ddq)
        taus.append(pin_data.tau.copy())
    return np.stack(taus)

def _rnea_tau_with_contact(pin_model, pin_data, q_full, dq_full, ddq_full,
                            contact_frame_id, force_profile):
    """Demo τ with contact correction:  u = M·ddq + h − J_c^T · λ_measured.

    Plain RNEA returns τ = M·ddq + h, which implicitly assumes no external
    forces. Real shaving demos had a measured normal contact force at the
    rock; the muscle τ that produced the motion is the free-body τ minus
    the contact reaction projected through the contact Jacobian. Without
    this correction the demo's us is systematically too large by exactly
    the pressing component, and downstream CSQP back-solves a fake λ to
    close the dynamics → IRL signal on press_force is artifactual.

    `force_profile` is a per-step scalar normal-force trace (length T, N).
    The reaction is built as [0, 0, f_n] in the contact frame's LOCAL frame,
    matching Crocoddyl ContactModel1D's convention.
    """
    taus = []
    T_force = len(force_profile) if force_profile is not None else 0
    for t, (q, dq, ddq) in enumerate(zip(q_full, dq_full, ddq_full)):
        pin.rnea(pin_model, pin_data, q, dq, ddq)
        tau_free = pin_data.tau.copy()
        pin.computeJointJacobians(pin_model, pin_data, q)
        J_c = pin.getFrameJacobian(
            pin_model, pin_data, contact_frame_id,
            pin.ReferenceFrame.LOCAL)[:3]
        f_n = float(force_profile[t]) if t < T_force else 0.0
        lambda_local = np.array([0.0, 0.0, f_n])
        tau_muscle = tau_free - J_c.T @ lambda_local
        taus.append(tau_muscle)
    return np.stack(taus)

def _friction_aware_tau(human_ocp, q_full, dq_full, ddq_full):
    """Compute demo torques such that actuation(u_demo) = u_RNEA, so the
    OCP's dynamics map (which adds friction) produces ddq_demo when given
    u_demo. Without this correction, the demo's u (plain RNEA) doesn't
    match the OCP's actuation+friction → demo and OCP go through different
    dynamics → cost evaluation is asymmetric.

    Assumes the actuation is `actuation(u).tau = u + tau_bias(q, v)` (the
    typical friction_lib pattern). Probes the actuation at u=0 to extract
    tau_bias, then sets u_demo = u_RNEA − tau_bias.

    Falls back to plain RNEA if the actuation/friction_lib unavailable.
    """
    try:
        import crocoddyl
        sys.path.insert(0, str(REPO / "friction_lib/build"))
        import friction_lib
    except Exception as e:
        print(f"[friction-aware-tau] failed to import: {e} — falling back to plain RNEA")
        return _rnea_tau(human_ocp.pin_model, human_ocp.pin_data,
                          q_full, dq_full, ddq_full)

    state = crocoddyl.StateMultibody(human_ocp.pin_model)
    actuation = friction_lib.ActuationModelFriction(
        state, human_ocp.contact_frame_id, human_ocp.mu, human_ocp.target_force)
    a_data = actuation.createData()
    pin_data = human_ocp.pin_model.createData()
    nu = actuation.nu
    nv = human_ocp.nv

    taus = []
    max_resid = 0.0
    for q, dq, ddq in zip(q_full, dq_full, ddq_full):
        pin.rnea(human_ocp.pin_model, pin_data, q, dq, ddq)
        u_rnea = pin_data.tau[:nu].copy()

        x = np.concatenate([q, dq])
        actuation.calc(a_data, x, np.zeros(nu))
        tau_bias = a_data.tau[:nu].copy()

        u_fric = u_rnea - tau_bias

        actuation.calc(a_data, x, u_fric)
        pin.aba(human_ocp.pin_model, pin_data, q, dq, a_data.tau)
        resid = float(np.linalg.norm(pin_data.ddq - ddq[:nv]))
        max_resid = max(max_resid, resid)

        u_pad = np.zeros(len(pin_data.tau))
        u_pad[:nu] = u_fric
        taus.append(u_pad)

    print(f"[friction-aware-tau] max |aba(q,v,actuation(u_fric)) − ddq| = "
          f"{max_resid:.6f}  (small ⇒ linear inversion holds)")
    return np.stack(taus)

def load_demos(demo_specs, model_human):
    """demo_specs: [(subject, task, [cycle, ...]), ...]. Returns
    xs_opt_list, us_opt_list, meta (list of dicts).

    Each demo is sliced [SLICE_START:SLICE_END]. Torques are computed with
    the *common* model_human's Pinocchio model — consistent with IRL using
    one reference model for all demos.
    """
    xs_list, us_list, meta = [], [], []
    for subj, task, cycles in demo_specs:
        for cycle in cycles:
            p = REPO / (f"trajectories_from_mocap/{DATE}/{subj}/{task}/elaborated/"
                        f"{TASK_DIR[task]}/cycle_{cycle:02d}.npz")
            if not p.exists():
                print(f"[warn] skip {subj}/{task}/cycle{cycle:02d}: missing {p}")
                continue
            cd = np.load(p, allow_pickle=True)
            q   = cd["q_matrix"].astype(np.float64)
            dq  = cd["dq_matrix"].astype(np.float64)
            ddq = cd["ddq_matrix"].astype(np.float64)

            cycle_keys = list(cd["joint_names"]) if "joint_names" in cd.files else []
            extras_needed = [j for j in ("middle_thoracic_X", "right_clavicle_joint_X")
                              if j not in cycle_keys]
            if extras_needed:
                from utils_slice_trajectories import JOINT_TO_CSV
                ik_csv = REPO / f"trajectories_from_mocap/{DATE}/{subj}/{task}/ik_joint_angles.csv"
                df = pd.read_csv(ik_csv)
                t_all   = df["Relative_Time[s]"].to_numpy(dtype=np.float64)
                t_start = float(cd["t_real_start"]) if "t_real_start" in cd.files else 0.0
                t_end   = float(cd["t_real_end"])   if "t_real_end"   in cd.files else 1.0
                t_real  = np.linspace(t_start, t_end, len(q))
                extra_q  = np.zeros((len(q), len(extras_needed)))
                extra_dq = np.zeros((len(q), len(extras_needed)))
                extra_ddq = np.zeros((len(q), len(extras_needed)))
                for k, jname in enumerate(extras_needed):
                    csv_col = JOINT_TO_CSV[jname]
                    col_vals = df[csv_col].to_numpy(dtype=np.float64)
                    idxs = np.array([int(np.abs(t_all - t).argmin()) for t in t_real])
                    extra_q[:, k] = col_vals[idxs]
                    if len(q) > 1:
                        dt_cyc = (t_end - t_start) / max(len(q) - 1, 1)
                        extra_dq[:, k]  = np.gradient(extra_q[:, k], dt_cyc)
                        extra_ddq[:, k] = np.gradient(extra_dq[:, k], dt_cyc)
                q   = np.hstack([extra_q,  q])
                dq  = np.hstack([extra_dq, dq])
                ddq = np.hstack([extra_ddq, ddq])

            q_pad   = np.hstack([q,   np.zeros((len(q),   1))])
            dq_pad  = np.hstack([dq,  np.zeros((len(dq),  1))])
            ddq_pad = np.hstack([ddq, np.zeros((len(ddq), 1))])

            xs = np.hstack([q_pad, dq_pad])[SLICE_START:SLICE_END]
            if _FRICTION_AWARE_TAU:
                us_full = _friction_aware_tau(model_human, q_pad, dq_pad, ddq_pad)
            else:
                us_full = _rnea_tau(model_human.pin_model, model_human.pin_data,
                                     q_pad, dq_pad, ddq_pad)
            us = us_full[SLICE_START:SLICE_END]
            xs_list.append(xs)
            us_list.append(us)
            meta.append({"subject": subj, "task": task, "cycle": cycle,
                         "T": len(xs)})
    return xs_list, us_list, meta

def build_common_models(combo, first_xs):
    """Build a Human (model_mocap) and a HumanMPPI using the combo's
    model_subject URDF. first_xs is the first demo's (q, dq) trajectory,
    used for q0_full / q0 / T / q_traj."""
    model_subj = combo.get("model_subject") or combo["demos"][0][0]
    task = combo["demos"][0][1]
    cycle = combo["demos"][0][2][0]

    ik_csv = REPO / f"trajectories_from_mocap/{DATE}/{model_subj}/{task}/ik_joint_angles.csv"
    q0_full = pd.read_csv(ik_csv).to_numpy()[0][2:]

    cyc = np.load(REPO / (f"trajectories_from_mocap/{DATE}/{model_subj}/{task}/"
                          f"elaborated/{TASK_DIR[task]}/cycle_{cycle:02d}.npz"),
                  allow_pickle=True)
    q_ref  = cyc["q_matrix"].astype(np.float64)
    cycle_keys = list(cyc["joint_names"]) if "joint_names" in cyc.files else []
    extras_needed = [j for j in ("middle_thoracic_X", "right_clavicle_joint_X")
                      if j not in cycle_keys]
    if extras_needed:
        from utils_slice_trajectories import JOINT_TO_CSV
        df_ik = pd.read_csv(ik_csv)
        t_all_csv = df_ik["Relative_Time[s]"].to_numpy(dtype=np.float64)
        t_start = float(cyc["t_real_start"]) if "t_real_start" in cyc.files else 0.0
        t_end   = float(cyc["t_real_end"])   if "t_real_end"   in cyc.files else 1.0
        t_real  = np.linspace(t_start, t_end, len(q_ref))
        extra_q = np.zeros((len(q_ref), len(extras_needed)))
        for k, jname in enumerate(extras_needed):
            csv_col = JOINT_TO_CSV[jname]
            col_vals = df_ik[csv_col].to_numpy(dtype=np.float64)
            idxs = np.array([int(np.abs(t_all_csv - t).argmin()) for t in t_real])
            extra_q[:, k] = col_vals[idxs]
        q_ref = np.hstack([extra_q, q_ref])
    q_traj = np.hstack([q_ref, np.zeros((len(q_ref), 1))])[SLICE_START:SLICE_END]

    if "dt" in cyc.files:
        dt_cyc = float(cyc["dt"])
        print(f"[build] dt = {dt_cyc*1000:.3f} ms ({dt_cyc:.6f} s) — "
              f"from cycle_{cycle:02d}.npz")
    else:
        dt_cyc = 0.0083
        print(f"[build] WARNING: no 'dt' in cycle_{cycle:02d}.npz — "
              f"using fallback dt = {dt_cyc*1000:.3f} ms")

    args_common = dict(
        subject_id=model_subj.lower(), date=DATE, task=task,
        mu=_BUILD_MU, target_force=_BUILD_TARGET_FORCE, contact="automatic",
        dt=dt_cyc,
        q0_full=q0_full, q0=q_traj[0], q_traj=q_traj, T=len(q_traj) - 1,
        force_strict_slack_frac= 0.0,
    )
    if "_short" in task:
        args_common["tool_length"] = 0.0328
        print(f"[build] short-task detected → tool_length=0.0328m, "
              f"OCP will use two-phase contact/free split")
    

    human = Human(args=dict(args_common,
                            w_run={**{tk: 0.001 for tk in _tau_run_keys()},
                                   "JA": 0.001, "JV": 0.001,
                                   "JTC": 0.001, "Geo": 0.001,
                                   "Eng_thoracic": 0.001, "Eng_clavicle": 0.001,
                                   "Eng_shoulder": 0.001, "Eng_elbow": 0.001,
                                   "Eng_wrist": 0.001,

                                   "progress_vel": 1.0, "press_force": 1e-6,
                                   "rail_lat": 1e-6, "rock_ori": 1e-6},
                            w_term={},
                            solver_type="CSQP",
                            stick_static=False,
                            contact_aware_cost=_CONTACT_AWARE_COST))

    human_mppi = HumanMPPI(args=dict(args_common,
                                     use_gpu=False, stick_static=False,
                                     horizon=20, num_samples=1024,
                                     noise_sigma=20.0, lambda_=0.1,
                                     w_run={k: 1e-4 for k in MPPI_KEYS},
                                     w_term={}))
    return human, human_mppi

def _save_xs(out_dir, xs_arr):
    np.savez_compressed(out_dir / "xs_irl.npz", xs=xs_arr)

def _save_irl_xs(out_dir, IRL_obj):
    """Save the IRL-recovered trajectory. Prefer xs_best if the line
    search tracked an improvement; otherwise fall back to the last
    iteration's trajectory so downstream tooling (recorder, plot script)
    always finds an xs_irl.npz."""
    xs = None
    if hasattr(IRL_obj, "xs_best") and IRL_obj.xs_best is not None:
        xs = IRL_obj.xs_best
    elif hasattr(IRL_obj, "Xs") and len(IRL_obj.Xs) > 0:
        xs = IRL_obj.Xs[-1]
    if xs is not None:
        _save_xs(out_dir, np.asarray(xs))
    else:
        print("[warn] no IRL trajectory to save (xs_best=None and Xs empty)")

def _plot_convergence(out_dir, IRL_obj, title):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    q = np.asarray(IRL_obj.q_norm, dtype=float)
    ax.plot(q, lw=1.5)
    ax.set_xlabel("IRL iteration")
    ax.set_ylabel("q_norm")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / "plot.png", dpi=100); plt.close(fig)

def method_ocp(human, human_mppi, xs_list, us_list, out_dir):
    """Solve OCP on the first demo (CSQP) and dump xs_ocp."""
    x0 = human.x0
    xs_init = [x0.copy() for _ in range(human.T + 1)]
    nu = human.solver.problem.runningModels[0].differential.actuation.nu
    us_init = [np.zeros(nu) for _ in range(human.T)]
    xs_ocp, us_ocp = human.solve(xs_init=xs_init, us_init=us_init)
    _save_xs(out_dir, np.asarray(xs_ocp))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    nq = human.nq
    ax.plot(np.asarray(xs_list[0])[:, :nq], color="C0", alpha=0.3)
    ax.plot(np.asarray(xs_ocp)[:, :nq],     color="C3", alpha=0.5, linestyle="--")
    ax.set_title("OCP (dashed) vs first demo (solid)")
    fig.tight_layout(); fig.savefig(out_dir / "plot.png", dpi=100); plt.close(fig)
    return {"method": "ocp", "q_norm_final": None, "iters": None,
            "weights": human.args["w_run"]}

def _apply_mppi_force_parity(kin, human_mppi):
    """Set the CSQP-parity force-handling attributes on BOTH the rollout
    controller and the demo model (read via getattr in the feature code).
    Must run before the first solve() so the MJX JIT bakes the flags in."""
    for m in (kin, human_mppi):
        m.press_emergent  = _MPPI_PRESS_EMERGENT
        m.progress_vel_sq = _MPPI_PROGRESS_VEL_SQ
        m.force_max       = _MPPI_FORCE_MAX
    if hasattr(human_mppi, 'args') and isinstance(human_mppi.args, dict):
        human_mppi.args['force_max'] = _MPPI_FORCE_MAX
    if _MPPI_PRESS_EMERGENT or _MPPI_PROGRESS_VEL_SQ:
        print(f"[mppi] CSQP-parity: press_emergent={_MPPI_PRESS_EMERGENT} "
              f"progress_vel_sq={_MPPI_PROGRESS_VEL_SQ} force_max={_MPPI_FORCE_MAX}")

def _build_kin_cpu(human_mppi):
    kin = KinematicMPPI_CPU(human_mppi,
                            horizon=_MPPI_H, num_samples=_MPPI_K)
    kin.set_weights({k: 0.0 for k in MPPI_KEYS})
    kin.install_as_solver(human_mppi)
    _apply_mppi_force_parity(kin, human_mppi)
    profile = getattr(human_mppi, 'target_force_profile', None)
    if profile is not None and hasattr(kin, 'set_force_target_profile'):
        kin.set_force_target_profile(profile)
        print(f"[kin] press_force target = profile (len={len(profile)}, "
              f"min={float(np.min(profile)):.1f}N, "
              f"max={float(np.max(profile)):.1f}N)")
    return kin

def _build_kin_mjx(human_mppi):
    """Drop-in MJX replacement for _build_kin_cpu. Same install_as_solver
    contract — downstream MO_IRL._compute_dw reads self.model.solver
    opaquely so no changes are needed there. Validated 2026-05-24:
    ~17× per-rollout speedup at K=2048 on 9-DoF, noise tolerable for SGD."""
    kin = KinematicMPPI_MJX(human_mppi,
                            horizon=_MPPI_H, num_samples=_MPPI_K,
                            n_dial=_MPPI_NDIAL,
                            noise_scale=_MPPI_NOISE_SCALE)
    kin.verbose_solve = not _MPPI_QUIET_SOLVE
    kin.set_weights({k: 0.0 for k in MPPI_KEYS})
    kin.install_as_solver(human_mppi)
    _apply_mppi_force_parity(kin, human_mppi)
    profile = getattr(human_mppi, 'target_force_profile', None)
    if profile is not None and hasattr(kin, 'set_force_target_profile'):
        kin.set_force_target_profile(profile)
        print(f"[kin-mjx] press_force target = profile (len={len(profile)}, "
              f"min={float(np.min(profile)):.1f}N, "
              f"max={float(np.max(profile)):.1f}N)")
    return kin

def _build_kin(human_mppi):
    """Backend-dispatching builder. Reads module-global `_KIN_BACKEND`
    (set from --backend in main())."""
    if _KIN_BACKEND == 'mjx':
        return _build_kin_mjx(human_mppi)
    return _build_kin_cpu(human_mppi)

def method_mppi_vp(human_mppi, xs_list, us_list, out_dir, max_iter=15, n_w=1,
                   p1_lr=None, p1_steps=None, use_sgd=True):
    """MPPI IRL learning only 'progress_vel'."""
    kin = _build_kin(human_mppi)
    W_INIT = {k: 0.0 for k in MPPI_KEYS}
    W_INIT["progress_vel"] = 1.0
    kin.set_weights(W_INIT); kin.install_as_solver(human_mppi)

    from irl_phases import _build_mask, _build_mo_args
    mask = _build_mask(["progress_vel"], n_w=n_w)

    irl_args = {
        "model": human_mppi, "w_run": dict(W_INIT), "w_term": {},
        "xs_opt": xs_list, "us_opt": us_list,
        "irl_iter": 1000, "stopping": "opt", "tol": 1e-10,
        "max_iter": max_iter, "min_iter": 1,
        "compare_desired": False, "verbose": True, "n_w": n_w,
        "use_adam": not use_sgd,
    }
    lr_eff = p1_lr if p1_lr is not None else 50
    outer_opt = 'sgd' if use_sgd else 'lbfgs'
    # no-step value but never strictly improve on it ("No Step Found" forever
    mo_args = _build_mo_args(gradient_mask=mask, mppi_lr=lr_eff,
                             temperature=0.5, outer_optimizer=outer_opt,
                             line_search='opt')
    IRL_obj = MO_IRL(mo_args, irl_args)
    IRL_obj.steps = (np.asarray(p1_steps, dtype=float) if p1_steps is not None
                     else np.array([50, 20, 10, 5, 2, 1, 0.5]))
    IRL_obj.line_search.steps = IRL_obj.steps
    IRL_obj.solve()

    best = int(np.argmin(IRL_obj.q_norm))
    w_best = IRL_obj.ws[best][-1].copy()
    w_dict = {k: float(w_best[j]) for j, k in enumerate(IRL_obj.keys_run)}
    _plot_convergence(out_dir, IRL_obj, "mppi_vp (progress_vel only)")
    _save_irl_xs(out_dir, IRL_obj)
    return {"method": "mppi_vp",
            "q_norm_final": float(IRL_obj.q_norm[best]),
            "iters": best, "weights": w_dict}

def method_mppi_2phase(human_mppi, xs_list, us_list, out_dir, n_w=1,
                       p1_lr=None, p1_steps=None,
                       use_sgd=True, p1_init=1.0,
                       mode="windowed", basis_type="gaussian",
                       p1_dict=None, feature_scale=False):
    kin = _build_kin(human_mppi)
    outer_opt = 'sgd' if use_sgd else 'lbfgs'
    # asymptotically converges to the no-step value but never strictly
    line_search_crit = 'opt' if use_sgd else 'q_norm'
    sgd_lr_scale = 1.0
    p1_lr_eff = p1_lr if p1_lr is not None else 100 * sgd_lr_scale
    p2_lr_eff = 10 * sgd_lr_scale
    if p1_dict is not None:
        w1 = dict(p1_dict)
        print(f"[mppi_2phase] Phase 1 SKIPPED — using preloaded weights "
              f"({[k for k,v in w1.items() if abs(v) > 1e-6]})")
    else:
        p1_kwargs = dict(mppi_lr=p1_lr_eff,
                         max_iter=10, n_w=n_w,
                         use_adam=not use_sgd, init_weight=p1_init,
                         per_feature_scale=feature_scale,
                         outer_optimizer=outer_opt,
                         line_search=line_search_crit)
        if p1_steps is not None:
            p1_kwargs["steps"] = np.asarray(p1_steps, dtype=float)
        w1, _ = run_phase1(human_mppi, kin, xs_list, us_list, **p1_kwargs)
    w2, IRL2 = run_phase2(human_mppi, kin, xs_list, us_list,
                           w_phase1=w1, mppi_lr=p2_lr_eff, max_iter=20, n_w=n_w,
                           use_adam=not use_sgd,
                           weight_mode=mode, basis_type=basis_type,
                           per_feature_scale=feature_scale,
                           outer_optimizer=outer_opt,
                           line_search=line_search_crit,
                           features=_MPPI_P2_FEATURES)
    _plot_convergence(out_dir, IRL2,
                       f"mppi_2phase  n_w={n_w}  mode={mode}  (phase2 convergence)")
    _save_irl_xs(out_dir, IRL2)
    return {"method": "mppi_2phase", "n_w": n_w, "mode": mode,
            "q_norm_final": float(min(IRL2.q_norm)),
            "iters": int(np.argmin(IRL2.q_norm)),
            "weights": w2, "weights_phase1": w1,
            "phase1_skipped": p1_dict is not None}

def method_mppi_3phase(human_mppi, xs_list, us_list, out_dir, n_w=1,
                       p1_lr=None, p1_steps=None,
                       use_sgd=True, p1_init=1.0,
                       mode="windowed", basis_type="gaussian",
                       p1_dict=None, feature_scale=False):
    kin = _build_kin(human_mppi)
    outer_opt = 'sgd' if use_sgd else 'lbfgs'
    line_search_crit = 'opt' if use_sgd else 'q_norm'
    sgd_lr_scale = 1.0
    p1_lr_eff = p1_lr if p1_lr is not None else 100 * sgd_lr_scale
    p2_lr_eff = 10 * sgd_lr_scale
    p3_lr_eff = 5  * sgd_lr_scale
    if p1_dict is not None:
        w1 = dict(p1_dict)
        print(f"[mppi_3phase] Phase 1 SKIPPED — using preloaded weights "
              f"({[k for k,v in w1.items() if abs(v) > 1e-6]})")
    else:
        p1_kwargs = dict(mppi_lr=p1_lr_eff,
                         max_iter=10, n_w=n_w,
                         use_adam=not use_sgd, init_weight=p1_init,
                         per_feature_scale=feature_scale,
                         outer_optimizer=outer_opt,
                         line_search=line_search_crit)
        if p1_steps is not None:
            p1_kwargs["steps"] = np.asarray(p1_steps, dtype=float)
        w1, _ = run_phase1(human_mppi, kin, xs_list, us_list, **p1_kwargs)
    w2, _ = run_phase2(human_mppi, kin, xs_list, us_list,
                        w_phase1=w1, mppi_lr=p2_lr_eff, max_iter=20, n_w=n_w,
                        use_adam=not use_sgd,
                        weight_mode=mode, basis_type=basis_type,
                        per_feature_scale=feature_scale,
                        outer_optimizer=outer_opt,
                        line_search=line_search_crit,
                        features=_MPPI_P2_FEATURES)
    w3, IRL3 = run_phase3(human_mppi, kin, xs_list, us_list,
                           w_phase2=w2, mppi_lr=p3_lr_eff, max_iter=20, n_w=n_w,
                           use_adam=not use_sgd,
                           weight_mode=mode, basis_type=basis_type,
                           per_feature_scale=feature_scale,
                           outer_optimizer=outer_opt)
    _plot_convergence(out_dir, IRL3,
                       f"mppi_3phase  n_w={n_w}  mode={mode}  (phase3 convergence)")
    _save_irl_xs(out_dir, IRL3)
    return {"method": "mppi_3phase", "n_w": n_w, "mode": mode,
            "q_norm_final": float(min(IRL3.q_norm)),
            "iters": int(np.argmin(IRL3.q_norm)),
            "weights": w3, "weights_phase1": w1, "weights_phase2": w2}

def method_mppi_allfeat(human_mppi, xs_list, us_list, out_dir, n_w=1,
                        p1_lr=None, p1_steps=None,
                        use_sgd=True, p1_init=1.0,
                        mode="windowed", basis_type="gaussian",
                        p1_dict=None, feature_scale=False):
    """ALL features learned at once, warm-started ONLY by the progress_vel
    velocity term (phase 1). Skips phase 2's staged task-placement learning:
    once the motion travels the rail (good velocity warmstart), phase 3 opens
    up every weight (task + style + force) simultaneously. This is the
    'one good velocity warmstart, then learn everything' route — the MPPI analog
    of the CSQP all-feature run, without the staged freezing."""
    kin = _build_kin(human_mppi)
    outer_opt = 'sgd' if use_sgd else 'lbfgs'
    line_search_crit = 'opt' if use_sgd else 'q_norm'
    p1_lr_eff = p1_lr if p1_lr is not None else 100
    p3_lr_eff = 5
    if p1_dict is not None:
        w1 = dict(p1_dict)
        print(f"[mppi_allfeat] Phase 1 SKIPPED — using preloaded velocity warmstart "
              f"({[k for k,v in w1.items() if abs(v) > 1e-6]})")
    else:
        p1_kwargs = dict(mppi_lr=p1_lr_eff, max_iter=10, n_w=n_w,
                         use_adam=not use_sgd, init_weight=p1_init,
                         per_feature_scale=feature_scale,
                         outer_optimizer=outer_opt,
                         line_search=line_search_crit)
        if p1_steps is not None:
            p1_kwargs["steps"] = np.asarray(p1_steps, dtype=float)
        w1, _ = run_phase1(human_mppi, kin, xs_list, us_list, **p1_kwargs)
    w3, IRL3 = run_phase3(human_mppi, kin, xs_list, us_list,
                          w_phase2=w1, mppi_lr=p3_lr_eff, max_iter=20, n_w=n_w,
                          use_adam=not use_sgd,
                          weight_mode=mode, basis_type=basis_type,
                          per_feature_scale=feature_scale,
                          outer_optimizer=outer_opt)
    _plot_convergence(out_dir, IRL3,
                       f"mppi_allfeat  n_w={n_w}  mode={mode}  (all-feature convergence)")
    _save_irl_xs(out_dir, IRL3)
    return {"method": "mppi_allfeat", "n_w": n_w, "mode": mode,
            "q_norm_final": float(min(IRL3.q_norm)),
            "iters": int(np.argmin(IRL3.q_norm)),
            "weights": w3, "weights_phase1": w1}

def _build_csqp_mask(keys_run, keys_term, K, mask_out):
    """Build flat gradient mask for csqp_irl.

    Layout: [run_block_0..K-1, term_block_0..K-1], each block sorted by key.
    Features named in `mask_out` get 0.0 in every window of both run and
    term blocks (whichever they're present in); others stay at 1.0.
    """
    nr_run, nr_term = len(keys_run), len(keys_term)
    mask = np.ones((nr_run + nr_term) * K, dtype=float)
    if not mask_out:
        return mask.tolist()
    for feat in mask_out:
        if feat in keys_run:
            j = keys_run.index(feat)
            for k in range(K):
                mask[k * nr_run + j] = 0.0
        if feat in keys_term:
            j = keys_term.index(feat)
            offset = nr_run * K
            for k in range(K):
                mask[offset + k * nr_term + j] = 0.0
    print(f"[csqp_irl] gradient_mask: zeroed features {list(mask_out)} "
          f"(run+term, all {K} windows)")
    return mask.tolist()

def method_csqp_irl(human, xs_list, us_list, out_dir,
                    n_w=1, mode="windowed", K=None,
                    max_iter=1000, with_press_force=True,
                    refinement_max=8, refinement_residual_tol=1e-3,
                    refinement_ll_gain_tol=1e-3,
                    mask_out=None, task_seed=1e-6,
                    init_w_run=None, init_w_term=None,
                    feature_scale=False,
                    q_norm_force_weight=0.0,
                    line_search_steps=40,
                    normalize_w=False,
                    use_dq_norm=False,
                    dq_norm_tol=2.0,
                    use_accel_guard=False,
                    accel_tol=3.0,
                    select_best_feature=False,
                    line_search_base='q_norm'):
    """CSQP-based MO_IRL: model = HumanCrocoddyl, inner solves via CSQP.

    Mirrors the canonical irl_args / mo_args from MO_IRL_mocap.ipynb.

    mode :
      'windowed' — piecewise-constant W(t) with n_w blocks (legacy default).
      'basis'    — soft Gaussian basis, K = (K or n_w), auto sigma.
      'adaptive' — basis mode started with K = (K or n_w) and grown via
                   MO_IRL.solve_with_refinement (inserts new centers where
                   the residual gradient peaks). See 08_basis_weights.md.
    """
    from MO_IRL import MO_IRL
    import time as _t

    w_run = {
        'Eng_thoracic': 1e-6, 'Eng_clavicle': 1e-6, 'Eng_shoulder': 1e-6,
        'Eng_elbow':    1e-6, 'Eng_wrist':    1e-6,
        'Geo':  1e-6, 'JV':  1e-6,
        'JTC':  1e-6, 'JA':   1e-6,
        **{tk: 1e-6 for tk in _tau_run_keys()},
        'progress_vel': float(task_seed),
        'rail_lat':     float(task_seed),
        'rock_ori':     float(task_seed),
    }
    if with_press_force:
        w_run['press_force'] = float(task_seed)
    w_term = {}
    if task_seed != 1e-6:
        print(f"[csqp_irl] task-feature seed = {task_seed:g} "
              f"(progress_vel, rail_lat, rock_ori"
              f"{', press_force' if with_press_force else ''})")
    if init_w_run is not None:
        for k, v in init_w_run.items():
            if k in w_run:
                w_run[k] = float(v)
        print(f"[csqp_irl] init_w_run override: {len(init_w_run)} keys")
    if init_w_term is not None:
        for k, v in init_w_term.items():
            if k in w_term:
                w_term[k] = float(v)
        print(f"[csqp_irl] init_w_term override: {len(init_w_term)} keys")

    nr_run, nr_term = len(w_run), len(w_term)

    irl_args = {
        'model': human, 'w_run': w_run, 'w_term': w_term,
        'xs_opt': xs_list, 'us_opt': us_list,
        'irl_iter': 1000, 'stopping': 'q_norm', 'tol': 1e-10,
        'max_iter': max_iter, 'min_iter': 1,
        'compare_desired': False, 'verbose': True,
        'n_w': n_w,
    }
    if mode in ('basis', 'adaptive'):
        K_eff = int(K or n_w)
        irl_args['weight_mode']  = 'basis'
        irl_args['K']            = K_eff
        irl_args['basis_type']   = 'gaussian'
        irl_args['basis_sigma']  = None
    K_for_mask = irl_args.get('K', n_w)

    mo_args = {
        'with_temp_adjust': False,
        'sqp_iter': 50,
        'next_traj': 'worst',
        'line_search_steps': int(line_search_steps),
        'line_search_base': line_search_base,
        'use_jac': True, 'use_hess': False,
        'with_dmp': False, 'basis_num': 20,
        'alpha_x': 1.5, 'alpha_z': 50.0, 'beta_z': 10,
        'noise_f': 50.0, 'tau': 10.0, 'rollout_N': 5,
        'opt_vars': 'dw',
        'temperature': 1, 'l_type': 1, 'l_reg': 'elastic',
        'use_bad': False, 'normalize_w': bool(normalize_w),
        'use_dq_norm': bool(use_dq_norm), 'dq_norm_tol': float(dq_norm_tol),
        'use_accel_guard': bool(use_accel_guard), 'accel_tol': float(accel_tol),
        'normalize_features': False, 'normalizing_thrs': 1.0,
        'K_set': 1, 'N_samples': 50, 'scaled_sum': True,
        'use_best': False,
        'Lambda': 1e-5, 'Beta': 1e-8, 'dyn_reg': False,
        'kappa': 9., 'lambda_thrs': 1e-5,
        'lambda_init': 1., 'delta': 1.0,
        'stopping': 'q_norm', 'tol': 1e-4,
        'gradient_mask': _build_csqp_mask(
            sorted(w_run.keys()), sorted(w_term.keys()),
            K_for_mask, mask_out),
        'use_mppi_grad': False,
        'verbose_irl': _VERBOSE_IRL,
        'per_feature_scale': bool(feature_scale),
        'q_norm_force_weight': float(q_norm_force_weight),
    }
    if feature_scale:
        mo_args['Lambda'] = 0.0
        mo_args['Beta']   = 0.0
        print(f"[csqp_irl] per_feature_scale=True → Lambda/Beta forced to 0")

    print(f"[csqp_irl] n_w={n_w}  mode={mode}  "
          f"max_iter={max_iter}  features={nr_run}+{nr_term}  "
          f"press_force={with_press_force}", flush=True)

    IRL_obj = MO_IRL(mo_args, irl_args)
    t0 = _t.time()
    if mode == 'adaptive':
        refinement_history = IRL_obj.solve_with_refinement(
            max_refinements=refinement_max,
            residual_tol=refinement_residual_tol,
            ll_gain_tol=refinement_ll_gain_tol,
            sigma_new=None,
            verbose=True,
        )
        np.save(out_dir / "refinement_history.npy",
                np.array(refinement_history, dtype=object), allow_pickle=True)
    else:
        IRL_obj.solve()
    dur = _t.time() - t0
    print(f"[csqp_irl] duration: {dur:.1f}s  "
          f"(K_final={getattr(IRL_obj, 'K', n_w)})")

    keys_all = list(IRL_obj.keys_run) + list(IRL_obj.keys_term)
    print(f"\n[csqp_irl] feature evolution per iteration (integrated phi):")
    header = f"  {'iter':>6s}  " + "  ".join(f"{k[:10]:>10s}" for k in keys_all)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, phi in enumerate(IRL_obj.phis):
        if i == 0:    label = "demo"
        elif i == 1:  label = "init"
        else:         label = f"it{i-1}"
        cells = []
        for j in range(len(keys_all)):
            v = float(phi[j]) if j < len(phi) else float('nan')
            cells.append(f"{v:>10.3g}")
        print(f"  {label:>6s}  " + "  ".join(cells))
    if len(IRL_obj.phis) > 1:
        demo_phi = np.asarray(IRL_obj.phis[0])
        last_phi = np.asarray(IRL_obj.phis[-1])
        print("  " + "-" * (len(header) - 2))
        ratios = ["{:>10.3g}".format(last_phi[j] / max(abs(demo_phi[j]), 1e-12))
                  for j in range(len(keys_all))]
        print(f"  {'ratio':>6s}  " + "  ".join(ratios)
              + "   ← last/demo (1.0 = matched)")

    addr = -1
    if select_best_feature and len(getattr(IRL_obj, 'opt_div', [])) > 1:
        od = np.asarray(IRL_obj.opt_div, float)
        addr = 1 + int(np.argmin(od[1:]))
        lbl = "init" if addr == 0 else f"it{addr}"
        print(f"[csqp_irl] select_best_feature: reporting iterate {addr} ({lbl}) "
              f"with min feature-divergence opt_div={od[addr]:.4g} "
              f"(last was {od[-1]:.4g})")
    w_run_flat  = IRL_obj.ws[addr][0]
    w_term_flat = IRL_obj.ws[addr][1]
    n_blocks = (IRL_obj.K if irl_args.get('weight_mode') == 'basis'
                else IRL_obj.n_w)
    init_run_vals  = [w_run.get(k, 0.0)  for k in IRL_obj.keys_run]
    init_term_vals = [w_term.get(k, 0.0) for k in IRL_obj.keys_term]
    if n_blocks > 1:
        w_run_blk  = w_run_flat.reshape(n_blocks, IRL_obj.nr_run)
        w_term_blk = (w_term_flat.reshape(n_blocks, IRL_obj.nr_term)
                      if IRL_obj.nr_term > 0 else np.zeros((n_blocks, 0)))
        for k in range(n_blocks):
            wr_abs = w_run_blk[k]
            wt_abs = w_term_blk[k] if IRL_obj.nr_term > 0 else np.zeros(0)
            label = ("theta" if irl_args.get('weight_mode') == 'basis' else "window")
            print(f"\n  {label} {k+1}/{n_blocks}")
            print('  Running:')
            for i, key in enumerate(IRL_obj.keys_run):
                print(f"    {key:<14} init {init_run_vals[i]:.3e}  →  learned {wr_abs[i]:.3e}")
            if IRL_obj.nr_term > 0:
                print('  Terminal:')
                for i, key in enumerate(IRL_obj.keys_term):
                    print(f"    {key:<14} init {init_term_vals[i]:.3e}  →  learned {wt_abs[i]:.3e}")
    else:
        wr_abs = w_run_flat
        wt_abs = w_term_flat if IRL_obj.nr_term > 0 else np.zeros(0)
        print('\n  Running:')
        for i, key in enumerate(IRL_obj.keys_run):
            print(f"    {key:<14} init {init_run_vals[i]:.3e}  →  learned {wr_abs[i]:.3e}")
        if IRL_obj.nr_term > 0:
            print('  Terminal:')
            for i, key in enumerate(IRL_obj.keys_term):
                print(f"    {key:<14} init {init_term_vals[i]:.3e}  →  learned {wt_abs[i]:.3e}")

    _plot_convergence(out_dir, IRL_obj,
                      f"csqp_irl  n_w={n_w}  mode={mode}")
    _save_irl_xs(out_dir, IRL_obj)

    force_rmse_final = None
    joint_rmse_deg_final = None
    try:
        import matplotlib.pyplot as _plt
        if select_best_feature and len(getattr(IRL_obj, 'opt_div', [])) > 1:
            _od = np.asarray(IRL_obj.opt_div, float)
            _best = 1 + int(np.argmin(_od[1:]))
        else:
            _best = int(np.argmin(IRL_obj.q_norm)) if len(IRL_obj.q_norm) else -1
        xs_demo_arr = np.asarray(IRL_obj.Xs[0])
        xs_irl_arr  = np.asarray(IRL_obj.Xs[_best])
        _nq = human.nq
        T_cmp = min(len(xs_demo_arr), len(xs_irl_arr))
        q_demo = xs_demo_arr[:T_cmp, :_nq]
        q_irl  = xs_irl_arr[:T_cmp, :_nq]
        deg = 180.0 / np.pi
        joint_rmse_deg = np.sqrt(np.mean((q_demo - q_irl) ** 2, axis=0)) * deg
        joint_names = [human.pin_model.names[i + 1]
                       for i in range(min(_nq, len(human.pin_model.names) - 1))]

        force_target = np.asarray(
            human.target_force_profile
            if getattr(human, 'target_force_profile', None) is not None
            else np.full(T_cmp, float(getattr(human, 'target_force', 0.0)))
        )
        force_irl = np.zeros(T_cmp)
        try:
            for _t in range(min(T_cmp, len(human.solver.problem.runningDatas))):
                _d = human.solver.problem.runningDatas[_t]
                _contacts = _d.differential.multibody.contacts.contacts.todict()
                if _contacts:
                    _f = list(_contacts.values())[0].f.linear
                    force_irl[_t] = (float(_f[2])
                                     if not np.any(np.isnan(_f)) else 0.0)
        except Exception as _e:
            print(f"[warn] could not extract contact forces: {_e}")
        _T_f = min(len(force_irl), len(force_target))
        force_rmse_final = float(np.sqrt(
            np.mean((force_irl[:_T_f] - force_target[:_T_f]) ** 2)))
        joint_rmse_deg_final = joint_rmse_deg.tolist()

        n_cols = 3
        n_rows = int(np.ceil((_nq + 1) / n_cols))
        fig, axes = _plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows),
                                   squeeze=False)
        axes_flat = axes.ravel()
        for j in range(_nq):
            ax = axes_flat[j]
            ax.plot(q_demo[:, j] * deg, color='C0', lw=1.5, label='demo')
            ax.plot(q_irl[:, j]  * deg, color='C3', lw=1.5, label='IRL')
            name = joint_names[j] if j < len(joint_names) else f'q[{j}]'
            ax.set_title(f"{name}  rmse={joint_rmse_deg[j]:.2f}°", fontsize=9)
            ax.set_xlabel('step'); ax.set_ylabel('deg')
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.legend(fontsize=8)
        ax_f = axes_flat[_nq]
        ax_f.plot(force_target[:T_cmp], color='C0', lw=1.5, label='target')
        ax_f.plot(force_irl, color='C3', lw=1.5, label='IRL')
        ax_f.set_title(f"Contact force  rmse={force_rmse_final:.2f} N", fontsize=10)
        ax_f.set_xlabel('step'); ax_f.set_ylabel('N')
        ax_f.grid(True, alpha=0.3); ax_f.legend(fontsize=8)
        for _k in range(_nq + 1, len(axes_flat)):
            axes_flat[_k].axis('off')
        fig.suptitle(f"Motion + force compare  ({out_dir.name})", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_dir / "motion_force_compare.png", dpi=110)
        _plt.close(fig)

        np.savez(out_dir / "rmse_final.npz",
                 joint_rmse_deg=joint_rmse_deg,
                 joint_names=np.array(joint_names, dtype=object),
                 force_rmse=force_rmse_final,
                 force_target=force_target[:_T_f],
                 force_irl=force_irl[:_T_f])

        q_total = float(np.linalg.norm(joint_rmse_deg))
        print(f"\n[motion-force] per-joint RMSE (deg, final IRL vs demo):")
        for j in range(_nq):
            name = joint_names[j] if j < len(joint_names) else f'q[{j}]'
            print(f"    {name:<22}  {joint_rmse_deg[j]:>7.2f}°")
        print(f"[motion-force] joint_rmse_total = {q_total:.2f}°  "
              f"force_rmse = {force_rmse_final:.2f} N  "
              f"(target mean {float(np.mean(force_target[:_T_f])):.1f} N)")
    except Exception as _e:
        print(f"[warn] motion/force comparison failed: {_e}")

    try:
        from irl_viz import (plot_trajectory_morph,
                              plot_full_cost_decomposition)
        try:
            plot_trajectory_morph(
                IRL_obj, human, max_shown=6,
                save_path=str(out_dir / "trajectory_morph.png"))
        except Exception as _e:
            print(f"[warn] plot_trajectory_morph failed: {_e}")
        try:
            plot_full_cost_decomposition(
                IRL_obj, human,
                save_path=str(out_dir / "cost_decomposition.png"))
        except Exception as _e:
            print(f"[warn] plot_full_cost_decomposition failed: {_e}")
    except Exception as _e:
        print(f"[warn] irl_viz import failed: {_e}")

    np.save(out_dir / "q_norm.npy", np.asarray(IRL_obj.q_norm, dtype=float))
    try:
        ws_run  = np.stack([w_pair[0] for w_pair in IRL_obj.ws])
        ws_term = np.stack([w_pair[1] for w_pair in IRL_obj.ws])
        np.savez(out_dir / "ws_history.npz", ws_run=ws_run, ws_term=ws_term,
                 keys_run=np.array(IRL_obj.keys_run, dtype=object),
                 keys_term=np.array(IRL_obj.keys_term, dtype=object))
    except ValueError:
        ws_run_final  = np.asarray(IRL_obj.ws[-1][0], dtype=float)
        ws_term_final = np.asarray(IRL_obj.ws[-1][1], dtype=float)
        np.savez(out_dir / "ws_history.npz",
                 ws_run=ws_run_final[None, :],
                 ws_term=ws_term_final[None, :],
                 keys_run=np.array(IRL_obj.keys_run, dtype=object),
                 keys_term=np.array(IRL_obj.keys_term, dtype=object))
        print(f"[info] ws_history: shapes ragged (adaptive); saved final iter only")
    except Exception as _e:
        print(f"[warn] could not save ws_history: {_e}")
    np.save(out_dir / "force_target.npy",
            np.asarray(getattr(human, 'target_force_profile', None)
                       if getattr(human, 'target_force_profile', None) is not None
                       else np.array([])))

    try:
        phis = np.stack([np.asarray(p, dtype=float) for p in IRL_obj.phis])
        np.save(out_dir / "phi_demo.npy", phis)
    except Exception as _e:
        print(f"[warn] could not save phi_demo: {_e}")
    try:
        np.save(out_dir / "xs_demo.npy", np.asarray(xs_list[0]))
    except Exception as _e:
        print(f"[warn] could not save xs_demo: {_e}")

    try:
        import matplotlib.pyplot as _plt
        all_keys = list(IRL_obj.keys_run) + list(IRL_obj.keys_term)
        phi_demo_final = np.asarray(IRL_obj.phis[0],  dtype=float)
        phi_irl_final  = np.asarray(IRL_obj.phis[-1], dtype=float)
        diff = phi_demo_final - phi_irl_final
        fig, axes = _plt.subplots(2, 1, figsize=(11, 7),
                                   gridspec_kw={'height_ratios': [2, 1]})
        x = np.arange(len(all_keys))
        w = 0.4
        axes[0].bar(x - w/2, np.abs(phi_demo_final), w, label='|φ_demo|',
                     color='C0', alpha=0.8)
        axes[0].bar(x + w/2, np.abs(phi_irl_final),  w, label='|φ_IRL|',
                     color='C3', alpha=0.8)
        axes[0].set_yscale('log')
        axes[0].set_xticks(x); axes[0].set_xticklabels(all_keys, rotation=45,
                                                         ha='right', fontsize=9)
        axes[0].set_ylabel('|φ| (log scale)')
        axes[0].set_title(f'Feature values — demo vs IRL final  '
                           f'({out_dir.name})')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        colors = ['C2' if d >= 0 else 'C1' for d in diff]
        axes[1].bar(x, diff, color=colors, alpha=0.8)
        axes[1].axhline(0, color='k', lw=0.5)
        axes[1].set_xticks(x); axes[1].set_xticklabels(all_keys, rotation=45,
                                                         ha='right', fontsize=9)
        axes[1].set_ylabel('φ_demo − φ_IRL')
        axes[1].set_title('Feature residual (positive = IRL under-uses '
                           'feature; negative = over-uses)')
        axes[1].grid(True, alpha=0.3)
        _plt.tight_layout()
        _plt.savefig(out_dir / "feature_diff.png", dpi=110)
        _plt.close(fig)
        np.save(out_dir / "phi_diff_final.npy",
                np.array({'keys': all_keys,
                           'phi_demo': phi_demo_final,
                           'phi_irl':  phi_irl_final,
                           'diff':     diff}, dtype=object),
                allow_pickle=True)
        print("\n[feature-diff] |φ_demo| / |φ_IRL| / Δ (sorted by |Δ|):")
        order = np.argsort(-np.abs(diff))
        for j in order:
            print(f"    {all_keys[j]:<14}  "
                  f"demo={phi_demo_final[j]:>12.4g}  "
                  f"irl={phi_irl_final[j]:>12.4g}  "
                  f"Δ={diff[j]:>12.4g}")
    except Exception as _e:
        print(f"[warn] could not write feature_diff.png: {_e}")

    if mode in ("basis", "adaptive"):
        B_step   = getattr(IRL_obj, "B_step",   None)
        B_window = getattr(IRL_obj, "B_window", None)
        if B_step is not None:
            np.save(out_dir / "B_step.npy",   np.asarray(B_step))
        if B_window is not None:
            np.save(out_dir / "B_window.npy", np.asarray(B_window))

    best_idx = int(np.argmin(IRL_obj.q_norm)) if len(IRL_obj.q_norm) else 0
    n_blocks_ret = (IRL_obj.K if irl_args.get('weight_mode') == 'basis'
                    else IRL_obj.n_w)
    if n_blocks_ret > 1:
        w_run_final  = w_run_flat.reshape(n_blocks_ret, IRL_obj.nr_run).max(axis=0)
        w_term_final = (w_term_flat.reshape(n_blocks_ret, IRL_obj.nr_term).max(axis=0)
                        if IRL_obj.nr_term > 0 else np.zeros(0))
    else:
        w_run_final  = np.asarray(w_run_flat, dtype=float)
        w_term_final = np.asarray(w_term_flat, dtype=float) if IRL_obj.nr_term > 0 else np.zeros(0)
    weights_run_dict  = {k: float(w_run_final[i])  for i, k in enumerate(IRL_obj.keys_run)}
    weights_term_dict = {k: float(w_term_final[i]) for i, k in enumerate(IRL_obj.keys_term)}
    return {
        'method': 'csqp_irl', 'n_w': n_w, 'mode': mode,
        'K_final': int(getattr(IRL_obj, 'K', n_w)),
        'q_norm_final': float(min(IRL_obj.q_norm)) if len(IRL_obj.q_norm) else None,
        'joint_rmse_deg':    joint_rmse_deg_final,
        'force_rmse_final':  force_rmse_final,
        'iters': best_idx,
        'duration_s': round(dur, 1),
        'press_force_active': with_press_force,
        'weights_run':  weights_run_dict,
        'weights_term': weights_term_dict,
        'init_w_run':   {k: float(v) for k, v in w_run.items()},
        'init_w_term':  {k: float(v) for k, v in w_term.items()},
    }

def method_csqp_irl_progressive(human, xs_list, us_list, out_dir,
                                n_w=1, mode="windowed", K=None,
                                max_iter=1000, with_press_force=True,
                                task_seed=1e-6,
                                max_rounds=5, growth_threshold=100.0,
                                q_improve_tol=1e-3,
                                initial_mask_out=None,
                                feature_scale=False):
    """Progressive masking: run csqp_irl, freeze the bully, re-run, repeat.

    After each round, the feature whose final weight grew the most relative
    to its seed becomes "frozen" — added to the gradient mask for subsequent
    rounds. Each round warm-starts from the previous round's final weights.

    Stops when:
      - max_rounds reached, OR
      - no remaining feature grew >= growth_threshold * its seed, OR
      - q_norm did not improve by > q_improve_tol vs previous round.

    Each round writes its artifacts to out_dir/round_<i>/ and the final
    aggregate summary lands directly in out_dir/.
    """
    mask_out = list(initial_mask_out or [])
    rounds = []
    init_w_run  = None
    init_w_term = None
    best_q_norm = None

    print(f"\n{'='*70}")
    print(f"[csqp-progressive] starting (max_rounds={max_rounds}, "
          f"growth_thrs={growth_threshold}x, q_tol={q_improve_tol})")
    print(f"{'='*70}\n")

    for r in range(max_rounds):
        round_dir = out_dir / f"round_{r}"
        round_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'─'*70}\n[csqp-progressive] ROUND {r}  "
              f"mask_out={mask_out or '[]'}\n{'─'*70}")
        info = method_csqp_irl(
            human, xs_list, us_list, round_dir,
            n_w=n_w, mode=mode, K=K,
            max_iter=max_iter, with_press_force=with_press_force,
            mask_out=mask_out, task_seed=task_seed,
            init_w_run=init_w_run, init_w_term=init_w_term,
            feature_scale=feature_scale,
        )
        q_norm_round = info['q_norm_final']
        rounds.append({
            'round': r, 'mask_at_start': list(mask_out),
            'q_norm_final': q_norm_round,
            'weights_run':  info['weights_run'],
            'weights_term': info['weights_term'],
        })

        if best_q_norm is not None and q_norm_round >= best_q_norm - q_improve_tol:
            print(f"[csqp-progressive] q_norm did not improve "
                  f"({q_norm_round:.4f} vs prev best {best_q_norm:.4f}) → STOP")
            break
        best_q_norm = q_norm_round if best_q_norm is None else min(best_q_norm, q_norm_round)

        ratios = {}
        for src, weights, seeds in [
            ("run",  info['weights_run'],  info['init_w_run']),
            ("term", info['weights_term'], info['init_w_term']),
        ]:
            for k, w in weights.items():
                if k in mask_out:
                    continue
                seed = seeds.get(k, 1e-6)
                if seed <= 0:
                    continue
                ratio = abs(w) / seed
                ratios[k] = max(ratios.get(k, 0.0), ratio)

        sorted_feats = sorted(ratios.items(), key=lambda kv: kv[1], reverse=True)
        print(f"[csqp-progressive] feature growth ratios (final/seed):")
        for k, ratio in sorted_feats[:8]:
            print(f"    {k:<14}  {ratio:>10.2f}x")
        if not sorted_feats or sorted_feats[0][1] < growth_threshold:
            print(f"[csqp-progressive] no feature grew >= {growth_threshold}x → STOP")
            break
        bully = sorted_feats[0][0]
        bully_ratio = sorted_feats[0][1]
        print(f"[csqp-progressive] freezing bully: {bully} "
              f"(grew {bully_ratio:.0f}x its seed)")
        mask_out.append(bully)

        init_w_run  = dict(info['weights_run'])
        init_w_term = dict(info['weights_term'])

    final_round = rounds[-1] if rounds else {}
    print(f"\n{'='*70}")
    print(f"[csqp-progressive] DONE: {len(rounds)} rounds")
    for rr in rounds:
        print(f"  round {rr['round']}: q_norm={rr['q_norm_final']:.4f}  "
              f"masked={rr['mask_at_start']}")
    print(f"{'='*70}\n")
    try:
        import json as _json
        (out_dir / "progressive_summary.json").write_text(
            _json.dumps({'rounds': rounds, 'final_mask': mask_out},
                        indent=2, default=float))
    except Exception as _e:
        print(f"[warn] could not save progressive_summary: {_e}")

    if rounds:
        best_round = min(rounds, key=lambda r: r['q_norm_final'])
        best_dir = out_dir / f"round_{best_round['round']}"
        print(f"[csqp-progressive] mirroring BEST round ({best_round['round']}, "
              f"q_norm={best_round['q_norm_final']:.4f}) to {out_dir.name}/")
        for fname in ('plot.png', 'weights.json', 'q_norm.npy',
                      'phi_demo.npy', 'force_target.npy',
                      'xs_demo.npy', 'xs_irl.npz', 'ws_history.npz'):
            src = best_dir / fname
            if src.exists():
                try:
                    import shutil as _sh
                    _sh.copy2(src, out_dir / fname)
                except Exception:
                    pass

    return {
        'method': 'csqp_irl_progressive',
        'n_w': n_w, 'mode': mode,
        'q_norm_final': final_round.get('q_norm_final'),
        'rounds_run': len(rounds),
        'final_mask': mask_out,
        'rounds': rounds,
    }

METHOD_FNS = {
    "ocp":          method_ocp,
    "mppi_vp":      method_mppi_vp,
    "mppi_2phase":  method_mppi_2phase,
    "mppi_3phase":  method_mppi_3phase,
    "mppi_allfeat": method_mppi_allfeat,
    "csqp_irl":     method_csqp_irl,
    "csqp_irl_progressive": method_csqp_irl_progressive,
}

def run_combo(combo, methods, summary_rows):
    cname = combo["name"]
    print(f"\n{'=' * 70}\n[combo] {cname}  demos={combo['demos']}\n{'=' * 70}")
    print(f"[build] constructing Human + HumanMPPI (silenced)...", flush=True)
    with _quiet():
        human, human_mppi = build_common_models(combo, None)
        xs_list, us_list, meta = load_demos(combo["demos"], human)
    print(f"[build] done — nq={human.nq}, T={human.T}, "
          f"dt={human.dt*1000:.3f}ms ({human.dt:.6f}s), "
          f"demos={len(xs_list)}", flush=True)
    if not xs_list:
        print(f"[combo] {cname}: no demos loaded, skipping")
        return

    forces = None
    subj, task, cycles = combo["demos"][0]
    cidx = cycles[0]
    direction = TASK_DIR[task]
    base_dir = REPO / f"trajectories_from_mocap/{DATE}/{subj}/{task}"
    cyc_path = base_dir / f"elaborated/{direction}/cycle_{cidx:02d}.npz"
    if cyc_path.exists():
        cd = np.load(cyc_path, allow_pickle=True)
        t_s = float(cd["t_real_start"])
        t_e = float(cd["t_real_end"])
        q_full = cd["q_matrix"]
        motion_times = np.linspace(t_s, t_e, len(q_full))
        sliced_times = motion_times[SLICE_START:SLICE_END]
        dt_cyc = float(cd["dt"]) if "dt" in cd.files else 0.0083
        forces, src = load_smoothed_force_profile(
            base_dir, subj, task, direction, cidx, DATE,
            sliced_times, dt_cyc, data_root="trajectories_from_mocap",
        )
        if forces is not None:

            override_active = _FORCE_TARGET_OVERRIDE is not None
            if override_active:
                forces = np.full_like(forces, float(_FORCE_TARGET_OVERRIDE))
                print(f"[force-override] {cname}: per-step profile replaced "
                      f"with constant {_FORCE_TARGET_OVERRIDE:.1f}N "
                      f"(was demo's measured profile)")

            pressing_mask = forces > 5.0
            pressing = forces[pressing_mask]
            T_total = len(forces)
            FALLBACK_TARGET = 30.0
            FALLBACK_FRAC   = 0.30

            is_long_task = "_long" in task
            if override_active:
                target_const   = float(_FORCE_TARGET_OVERRIDE)
                T_press_start  = 0
                src_label      = f"OVERRIDE (--force_target_override={target_const:.1f}N)"
            elif len(pressing) > 0:
                target_const   = max(float(np.mean(pressing)), 5.0)
                T_press_start  = (0 if is_long_task
                                  else int(np.argmax(pressing_mask)))
                src_label      = ("demo pressing-phase mean (LONG task: no gating)"
                                  if is_long_task
                                  else "demo pressing-phase mean (SHORT task: gated)")
            else:
                target_const   = FALLBACK_TARGET
                T_press_start  = (0 if is_long_task
                                  else int(FALLBACK_FRAC * T_total))
                src_label      = (f"FALLBACK (no contact in profile, "
                                  f"range {forces.min():.1f}–{forces.max():.1f}N, "
                                  f"{'LONG: no gate' if is_long_task else 'SHORT: gated'})")
            print(f"[force] {cname}: profile mean={float(np.mean(forces)):.1f}N "
                  f"max={float(np.max(forces)):.1f}N "
                  f">5N for {int(pressing_mask.sum())}/{T_total} steps "
                  f"({100.0*pressing_mask.mean():.0f}%)")

            human.target_force            = target_const
            human.T_press_start           = T_press_start
            human_mppi.target_force       = target_const
            human_mppi.T_press_start      = T_press_start

            if hasattr(human, "set_force_target_profile"):
                human.set_force_target_profile(forces)

            if hasattr(human_mppi, "set_force_target"):
                human_mppi.set_force_target(None)
            print(f"[force] {cname}: target = {target_const:.1f}N constant ({src_label})")
            print(f"[force] {cname}: T_press_start = {T_press_start}/{T_total} "
                  f"({100.0*T_press_start/max(T_total,1):.0f}% of cycle)")

            _nq = human.nq
            for _i, _xs in enumerate(xs_list):
                _q = _xs[:, :_nq]
                us_corrected = us_list[_i].copy()
                for _t in range(len(us_corrected)):
                    pin.computeJointJacobians(human.pin_model, human.pin_data,
                                              _q[_t])
                    pin.updateFramePlacements(human.pin_model, human.pin_data)
                    J_c = pin.getFrameJacobian(
                        human.pin_model, human.pin_data,
                        human.contact_frame_id,
                        pin.ReferenceFrame.LOCAL)[:3]
                    f_n = float(forces[_t]) if _t < len(forces) else 0.0
                    lambda_local = np.array([0.0, 0.0, f_n])
                    us_corrected[_t] = us_corrected[_t] - J_c.T @ lambda_local
                us_list[_i] = us_corrected
            mean_norm = float(np.mean(
                [np.linalg.norm(u, axis=1).mean() for u in us_list]))
            print(f"[force] {cname}: us = friction-aware + contact correction "
                  f"(now reflects pressing effort); mean ‖u‖ ≈ {mean_norm:.2f} Nm")
        else:
            print(f"[force] {cname}: no force data found ({src}); using scalar target {human.target_force:.1f}N")

    us_list_mppi = list(us_list)
    if any(m.startswith("mppi_") for m in methods) and human_mppi is not None:
        for i, xs_i in enumerate(xs_list):
            if i == 0 and forces is not None:
                us_i, forces_i = human_mppi.get_tau_from_trajectory(
                    xs_i, force_profile=forces, force_gain=0.25)
                human_mppi.set_demo_forces(forces_i, xs_i)
                print(f"[mppi-tau] {cname} demo {i}: PD-tracked + force FF, "
                      f"|tau| mean={np.linalg.norm(us_i, axis=1).mean():.2f}, "
                      f"recorded force mean={forces_i.mean():.1f}N max={forces_i.max():.1f}N")
            else:
                us_i, _ = human_mppi.get_tau_from_trajectory(xs_i)
                print(f"[mppi-tau] {cname} demo {i}: PD-tracked, "
                      f"|tau| mean={np.linalg.norm(us_i, axis=1).mean():.2f}")
            us_list_mppi[i] = us_i

    csqp_n_ws    = combo.get("_csqp_n_ws",    [1])
    csqp_mode    = combo.get("_csqp_mode",    "windowed")
    csqp_max_iter = combo.get("_csqp_max_iter", 1000)
    csqp_press   = combo.get("_csqp_press",   True)
    csqp_mask_out = combo.get("_csqp_mask_out", None)
    csqp_task_seed = combo.get("_csqp_task_seed", 1e-6)
    feature_scale  = combo.get("_feature_scale", False)
    q_norm_force_weight = combo.get("_q_norm_force_weight", 0.0)
    csqp_line_search_steps = combo.get("_csqp_line_search_steps", 40)
    csqp_normalize_w = combo.get("_csqp_normalize_w", False)
    csqp_use_dq_norm = combo.get("_csqp_use_dq_norm", False)
    csqp_dq_norm_tol = combo.get("_csqp_dq_norm_tol", 2.0)
    csqp_use_accel_guard = combo.get("_csqp_use_accel_guard", False)
    csqp_accel_tol = combo.get("_csqp_accel_tol", 3.0)
    csqp_select_best_feature = combo.get("_csqp_select_best_feature", False)
    csqp_ls_base = combo.get("_csqp_ls_base", "q_norm")
    mppi_n_w     = combo.get("_mppi_n_w",     1)
    mppi_p1_lr    = combo.get("_mppi_p1_lr",    None)
    mppi_p1_steps = combo.get("_mppi_p1_steps", None)
    mppi_use_sgd  = combo.get("_mppi_use_sgd",  True)
    mppi_p1_init  = combo.get("_mppi_p1_init",  1.0)
    mppi_mode     = combo.get("_mppi_mode",     "windowed")
    mppi_basis_type = combo.get("_mppi_basis_type", "gaussian")
    mppi_p1_dict  = combo.get("_mppi_p1_dict",  None)
    run_id        = combo.get("_run_id",        "")

    for method in methods:
        is_csqp_like = method in ("csqp_irl", "csqp_irl_progressive")
        method_runs = ([(method, n_w) for n_w in csqp_n_ws]
                       if is_csqp_like else [(method, None)])

        for method_name, n_w_val in method_runs:
            if n_w_val is not None:
                tag = f"{method_name}__nw{n_w_val}__{csqp_mode}"
            elif method_name in ("mppi_2phase", "mppi_3phase", "mppi_allfeat"):
                tag = f"{method_name}__nw{mppi_n_w}__{mppi_mode}"
            else:
                tag = method_name
            if run_id:
                tag = f"{tag}__{run_id}"
            out_dir = OUT_ROOT / cname / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            try:
                if method_name == "ocp":
                    info = method_ocp(human, human_mppi, xs_list, us_list, out_dir)
                elif method_name == "mppi_vp":
                    info = method_mppi_vp(human_mppi, xs_list, us_list_mppi, out_dir,
                                          n_w=mppi_n_w,
                                          p1_lr=mppi_p1_lr, p1_steps=mppi_p1_steps,
                                          use_sgd=mppi_use_sgd)
                elif method_name == "mppi_2phase":
                    info = method_mppi_2phase(human_mppi, xs_list, us_list_mppi, out_dir,
                                              n_w=mppi_n_w,
                                              p1_lr=mppi_p1_lr, p1_steps=mppi_p1_steps,
                                              use_sgd=mppi_use_sgd, p1_init=mppi_p1_init,
                                              mode=mppi_mode, basis_type=mppi_basis_type,
                                              p1_dict=mppi_p1_dict,
                                              feature_scale=feature_scale)
                elif method_name == "mppi_3phase":
                    info = method_mppi_3phase(human_mppi, xs_list, us_list_mppi, out_dir,
                                              n_w=mppi_n_w,
                                              p1_lr=mppi_p1_lr, p1_steps=mppi_p1_steps,
                                              use_sgd=mppi_use_sgd, p1_init=mppi_p1_init,
                                              mode=mppi_mode, basis_type=mppi_basis_type,
                                              p1_dict=mppi_p1_dict,
                                              feature_scale=feature_scale)
                elif method_name == "mppi_allfeat":
                    info = method_mppi_allfeat(human_mppi, xs_list, us_list_mppi, out_dir,
                                               n_w=mppi_n_w,
                                               p1_lr=mppi_p1_lr, p1_steps=mppi_p1_steps,
                                               use_sgd=mppi_use_sgd, p1_init=mppi_p1_init,
                                               mode=mppi_mode, basis_type=mppi_basis_type,
                                               p1_dict=mppi_p1_dict,
                                               feature_scale=feature_scale)
                elif method_name == "csqp_irl":
                    info = method_csqp_irl(
                        human, xs_list, us_list, out_dir,
                        n_w=n_w_val,
                        mode=csqp_mode,
                        K=(n_w_val if csqp_mode in ("basis", "adaptive") else None),
                        max_iter=csqp_max_iter,
                        with_press_force=csqp_press,
                        mask_out=csqp_mask_out,
                        task_seed=csqp_task_seed,
                        feature_scale=feature_scale,
                        q_norm_force_weight=q_norm_force_weight,
                        line_search_steps=csqp_line_search_steps,
                        normalize_w=csqp_normalize_w,
                        use_dq_norm=csqp_use_dq_norm,
                        dq_norm_tol=csqp_dq_norm_tol,
                        use_accel_guard=csqp_use_accel_guard,
                        accel_tol=csqp_accel_tol,
                        select_best_feature=csqp_select_best_feature,
                        line_search_base=csqp_ls_base,
                    )
                elif method_name == "csqp_irl_progressive":
                    info = method_csqp_irl_progressive(
                        human, xs_list, us_list, out_dir,
                        n_w=n_w_val,
                        mode=csqp_mode,
                        K=(n_w_val if csqp_mode in ("basis", "adaptive") else None),
                        max_iter=csqp_max_iter,
                        with_press_force=csqp_press,
                        task_seed=csqp_task_seed,
                        max_rounds=combo.get("_csqp_progressive_rounds", 5),
                        growth_threshold=combo.get("_csqp_progressive_growth", 100.0),
                        q_improve_tol=combo.get("_csqp_progressive_qtol", 1e-3),
                        initial_mask_out=csqp_mask_out,
                        feature_scale=feature_scale,
                    )
                else:
                    print(f"[combo] unknown method: {method_name}"); continue
                (out_dir / "weights.json").write_text(json.dumps(info, indent=2, default=float))
                dt = time.time() - t0
                summary_rows.append({
                    "combo": cname, "method": tag, "runtime_s": round(dt, 1),
                    "q_norm_final": info.get("q_norm_final"),
                    "force_rmse_final": info.get("force_rmse_final"),
                    "iters": info.get("iters"),
                    "n_demos": len(xs_list),
                })
                print(f"[combo] {cname}/{tag} done in {dt:.1f}s")
            except Exception as e:
                print(f"[err] {cname}/{tag}: {e}")
                traceback.print_exc()
                summary_rows.append({
                    "combo": cname, "method": tag, "runtime_s": -1,
                    "q_norm_final": None, "iters": None, "n_demos": len(xs_list),
                    "error": str(e),
                })

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--combos",  nargs="+", default=None,
                    help="names of combos to run (default: all)")
    ap.add_argument("--methods", nargs="+", default=None,
                    help=f"methods to run (default: {METHODS})")
    ap.add_argument("--n_w",     nargs="+", type=int, default=[1],
                    help="csqp_irl: list of window counts to sweep, e.g. --n_w 1 2 3")
    ap.add_argument("--mode",    choices=("windowed", "basis", "adaptive"),
                    default="windowed",
                    help="csqp_irl weight parametrization: "
                         "windowed (legacy piecewise-constant), "
                         "basis (Gaussian, K=n_w), "
                         "adaptive (basis grown via solve_with_refinement, "
                         "starting K=n_w)")
    ap.add_argument("--max_iter", type=int, default=1000,
                    help="csqp_irl: max IRL iterations per (inner) solve")
    ap.add_argument("--no_press_force", action="store_true",
                    help="csqp_irl: disable press_force feature (default: on)")
    ap.add_argument("--csqp_mask_out", nargs="+", default=None,
                    help="csqp_irl: feature names to ZERO in the gradient "
                         "mask (frozen at their seed value, no learning). "
                         "Applied to both run and term blocks across all "
                         "windows. Example: --csqp_mask_out JA Tau")
    ap.add_argument("--csqp_task_seed", type=float, default=1e-6,
                    help="csqp_irl: initial weight for task features "
                         "(progress_vel, rail_lat, rock_ori, press_force). "
                         "Default 1e-6 (old behaviour, near-zero). Set to "
                         "e.g. 0.1 to start in a regime where task costs "
                         "are active and gradients carry signal from iter 0.")
    ap.add_argument("--csqp_progressive_rounds", type=int, default=5,
                    help="csqp_irl_progressive: max rounds (each round runs "
                         "the inner IRL to convergence then freezes the bully "
                         "feature). Default 5.")
    ap.add_argument("--csqp_progressive_growth", type=float, default=10000.0,
                    help="csqp_irl_progressive: a feature is considered the "
                         "bully if its final weight grew >= N x its seed. "
                         "Default 10000. Lower = more aggressive freezing. "
                         "Set higher to only freeze true bullies (e.g. JV "
                         "grew 96M× in early rounds).")
    ap.add_argument("--csqp_progressive_qtol", type=float, default=1e-3,
                    help="csqp_irl_progressive: stop if q_norm doesn't "
                         "improve by > tol vs prior round's best. Default 1e-3.")
    ap.add_argument("--q_norm_force_weight", type=float, default=0.0,
                    help="csqp_irl: include contact-force RMSE (N) in the "
                         "line-search criterion, scaled by this weight and "
                         "added to joint RMSE (deg). Default 0.0 (joint-only).")
    ap.add_argument("--csqp_normalize_w", action="store_true",
                    help="csqp_irl: renormalize the weight vector each step "
                         "(OCP-scale-invariant) to stop the late JA/Tau blow-up "
                         "on non-realizable recorded demos.")
    ap.add_argument("--csqp_ls_use_dq", action="store_true",
                    help="csqp_irl: line search also caps how much the velocity "
                         "error (q_dot) may grow — rejects jerky high-acceleration "
                         "steps that match positions but not velocities.")
    ap.add_argument("--csqp_ls_dq_tol", type=float, default=2.0,
                    help="csqp_irl: velocity-guard tolerance (accept if dq_norm < "
                         "tol x previous). >1 allows gradual growth, blocks the "
                         "explosion. Default 2.0.")
    ap.add_argument("--csqp_ls_use_accel", action="store_true",
                    help="csqp_irl: line search also rejects rollouts whose "
                         "acceleration magnitude exceeds accel_tol x the DEMO's "
                         "(catches the JA jitter the velocity guard misses).")
    ap.add_argument("--csqp_ls_accel_tol", type=float, default=3.0,
                    help="csqp_irl: acceleration-guard ceiling = tol x demo accel "
                         "magnitude. Default 3.0.")
    ap.add_argument("--csqp_select_best_feature", action="store_true",
                    help="csqp_irl: report the iterate whose recovered features "
                         "best match the demo (min opt_div), not the last iterate. "
                         "Avoids the q_norm-driven drift into the degenerate basin.")
    ap.add_argument("--csqp_tau_split", choices=["none", "full", "pruned"], default="none",
                    help="per-segment torque split: 'full'=5 Tau_<group>, 'pruned'="
                         "proximal (shoulder/clavicle/thoracic, the identifiable ones), "
                         "global Tau dropped. 'none'=aggregate Tau (default).")
    ap.add_argument("--csqp_ls_base", choices=["q_norm", "opt", "cost"], default="q_norm",
                    help="csqp_irl line-search acceptance criterion. 'q_norm' = joint "
                         "position match; 'opt' = feature divergence (use when q_norm is "
                         "at its floor on recorded data, so feature-improving steps get accepted).")
    ap.add_argument("--csqp_line_search_steps", type=int, default=40,
                    help="csqp_irl: number of line-search probes per outer "
                         "iteration (each is a CSQP re-solve). Default 40. "
                         "Lower (e.g. 8) for a much faster first pass, "
                         "especially with --contact_aware_cost.")
    ap.add_argument("--contact_aware_cost", action="store_true",
                    help="csqp: build the OCP with contact-aware Tau/Eng/JTC "
                         "costs (tau_act = u + Jc^T f_fric), so the solver cost "
                         "equals the IRL features. SLOWER (JTC residual uses "
                         "numerical contact-dynamics derivatives).")
    ap.add_argument("--feature_scale", action="store_true",
                    help="enable per-feature normalization in MO_IRL. Each "
                         "feature φ_k is divided by max(|φ_demo_k|, |φ_bad_k|, "
                         "1e-3) so all features end up O(1). Equalizes the "
                         "gradient share between regularizers (JA ~1e5) and "
                         "task features (rail_lat ~1e-4). Applies to both "
                         "csqp_irl[_progressive] and mppi_2phase / mppi_3phase. "
                         "csqp also drops Lambda/Beta to 0 to avoid L1 "
                         "saturation of small features.")
    ap.add_argument("--mppi_n_w", type=int, default=1,
                    help="mppi methods: number of time-varying weight windows "
                         "(passed through run_phase{1,2,3}). Default 1.")
    ap.add_argument("--mppi_mode", choices=("windowed", "basis"),
                    default="windowed",
                    help="mppi_2phase / mppi_3phase weight parametrization: "
                         "windowed (piecewise-constant) or basis (Gaussian, "
                         "K=mppi_n_w). Default windowed.")
    ap.add_argument("--mppi_basis_type", default="gaussian",
                    help="basis type when --mppi_mode=basis (default gaussian). "
                         "Forwarded to run_phase{2,3}.")
    ap.add_argument("--mppi_p1_npz", default=None,
                    help="path to a saved Phase 1 .npz (e.g. "
                         "experiments/irl_progress_vel_nw1.npz). When set, "
                         "Phase 1 is skipped and Phase 2 starts directly from "
                         "those weights — applies to mppi_2phase / mppi_3phase.")
    ap.add_argument("--mppi_p1_progress_vel", type=float, default=None,
                    help="skip Phase 1 entirely and start Phase 2 with the "
                         "given progress_vel weight (all other features at 0). "
                         "Mutually exclusive with --mppi_p1_npz.")
    ap.add_argument("--mppi_lr", type=float, default=None,
                    help="mppi: override Phase 1 mppi_lr (Adam learning rate). "
                         "Default 100. Try 1 to mirror test_single_feature.")
    ap.add_argument("--mppi_steps", nargs="+", type=float, default=None,
                    help="mppi: override Phase 1 line-search step list. "
                         "Default '50 20 10 5 2 1 0.5'. Try "
                         "'100 20 10 5 2 1 0.5 0.25 0.02' to mirror test_single_feature.")
    ap.add_argument("--mppi_use_sgd", action="store_true",
                    help="mppi: use plain SGD (dw = -lr * g) instead of Adam. "
                         "SGD is much more aggressive when gradients are small "
                         "since Adam normalizes per-coord step magnitude.")
    ap.add_argument("--backend", choices=("cpu", "mjx"), default="cpu",
                    help="Kinematic-MPPI rollout backend. cpu (default) = "
                         "deterministic, ~2.3 ms/rollout flat. mjx = "
                         "JAX/GPU, ~17× faster at K=2048 on the 9-DoF model "
                         "but with ~1e-3 relative nondeterminism. Pair with "
                         "--mppi_use_sgd; L-BFGS doesn't tolerate the noise.")
    ap.add_argument("--mppi_K", type=int, default=256,
                    help="Kinematic-MPPI rollout batch size. CPU default 256 "
                         "(any larger is slower 1:1). MJX scales near-flat "
                         "in K up to 2048+ — use --backend mjx --mppi_K 2048 "
                         "for the full ~17x speedup (validated 2026-05-24).")
    ap.add_argument("--mppi_H", type=int, default=40,
                    help="Kinematic-MPPI rollout horizon (steps). Default 40. "
                         "Larger H = longer JIT compile on MJX but otherwise "
                         "similar per-rollout cost.")
    ap.add_argument("--mppi_ndial", type=int, default=3,
                    help="MPPI DIAL iterations per outer step (default 3). "
                         "Each iter re-samples K noise vectors, so n_dial=3 "
                         "≈ 3× the GPU work per step. Drop to 1 for ~3× "
                         "wall-clock speedup at the cost of less-converged "
                         "MPPI controls per step (usually fine for IRL).")
    ap.add_argument("--mppi_noise_scale", type=float, default=1.0,
                    help="(mjx backend) multiplier on the MPPI noise "
                         "Cholesky factor — default 1.0 matches the CPU "
                         "noise level on paper but the MJX sample "
                         "distribution is empirically ~18%% narrower. "
                         "If features get stuck at their init weight "
                         "(g≈0 because rollouts don't excite that "
                         "feature), bump this to 1.5-3.0 to widen the "
                         "sample spread and unblock the gradient.")
    ap.add_argument("--mppi_quiet_solve", action="store_true",
                    help="Suppress the per-step [t=N] progress=... lines "
                         "inside each MPPI solve() call. Useful when "
                         "running many IRL iters and the diagnostic spam "
                         "swamps the optimizer convergence output.")
    ap.add_argument("--mppi_press_emergent", action="store_true",
                    help="CSQP parity: press_force = |f_n|/Fmax (emergent contact "
                         "reaction, ref 0) instead of the |f-target|/target tracking "
                         "residual. Applied to demo + CPU + MJX feature paths.")
    ap.add_argument("--mppi_progress_vel_sq", action="store_true",
                    help="CSQP parity: progress_vel = s_dot^2 (pace penalty) instead "
                         "of the linear -rail_vel travel reward. Demo + CPU + MJX.")
    ap.add_argument("--mppi_force_max", type=float, default=80.0,
                    help="Fmax denominator for --mppi_press_emergent (matches CSQP force_max).")
    ap.add_argument("--mppi_p2_features", nargs="+", default=None,
                    help="Phase 2 feature list override (default: approach "
                         "traveled rail_lat rock_ori). Pass e.g. "
                         "'--mppi_p2_features approach rail_lat rock_ori' "
                         "to drop traveled if it dominates the gradient and "
                         "marches the weight off to infinity.")
    ap.add_argument("--mppi_p1_init", type=float, default=1.0,
                    help="mppi: initial weight on progress_vel in Phase 1. "
                         "Default 1.0; bump to 1e4 or 1e6 to make MPPI move "
                         "immediately on iter 0 (matches the notebook style).")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show per-iter Phi|opt-cur|, Weights blocks, "
                         "L-BFGS-B raw dw, and iter-1 debug. Default off — "
                         "only the q_norm iteration table rows are printed.")
    ap.add_argument("--mu", type=float, default=0.3,
                    help="friction coefficient (default 0.3). Affects the OCP's "
                         "actuation friction model — sweep to find the value "
                         "that best matches the demo's force profile.")
    ap.add_argument("--target_force", type=float, default=60.0,
                    help="constructor target_force for OCP (default 60.0 N). "
                         "MPPI's target gets re-set from the demo's mean during "
                         "setup_experiment regardless of this.")
    ap.add_argument("--force_target_override", type=float, default=None,
                    help="if set, OVERRIDE the demo-derived per-step force "
                         "profile and constant target with this scalar value "
                         "(e.g. 5.0 for a 5N target). Bypasses pressing-phase "
                         "mean computation and pushes a constant profile to "
                         "the OCP. Useful for testing low-force scenarios.")
    ap.add_argument("--friction_aware_tau", action="store_true",
                    help="compute demo torques such that "
                         "actuation(u_demo) = u_RNEA, so the OCP's dynamics "
                         "(with friction) produce ddq_demo from u_demo. "
                         "Without this, demo's u (plain RNEA) doesn't match "
                         "the OCP's friction-aware dynamics → cost evaluation "
                         "is asymmetric (especially press_force, Tau, JTC).")
    ap.add_argument("--run_id", default="",
                    help="optional suffix appended to each output directory "
                         "name (e.g. 'sgd_p1e4', '20260428_1530', 'baseline'). "
                         "Empty (default) overwrites the canonical dir like "
                         "before. Pass any string to keep multiple runs "
                         "side-by-side: analysis/moirl/<combo>/<tag>__<run_id>/. "
                         "For an automatic timestamp, run with "
                         "--run_id $(date +%%Y%%m%%d_%%H%%M).")
    args = ap.parse_args()

    global _BUILD_MU, _BUILD_TARGET_FORCE, _VERBOSE_IRL, _MPPI_P1_DICT, _FORCE_TARGET_OVERRIDE, _FRICTION_AWARE_TAU, _KIN_BACKEND, _MPPI_K, _MPPI_H, _MPPI_NDIAL, _MPPI_NOISE_SCALE, _MPPI_QUIET_SOLVE, _MPPI_P2_FEATURES, _CONTACT_AWARE_COST, _CSQP_TAU_SPLIT, _MPPI_PRESS_EMERGENT, _MPPI_PROGRESS_VEL_SQ, _MPPI_FORCE_MAX
    _BUILD_MU           = float(args.mu)
    _BUILD_TARGET_FORCE = float(args.target_force)
    _VERBOSE_IRL        = bool(args.verbose)
    _CONTACT_AWARE_COST = bool(args.contact_aware_cost)
    _CSQP_TAU_SPLIT = str(args.csqp_tau_split)
    if _CSQP_TAU_SPLIT != "none":
        print(f"[batch] tau_split={_CSQP_TAU_SPLIT} — torque features: {_tau_run_keys()}")
    if _CONTACT_AWARE_COST:
        print("[batch] contact_aware_cost ENABLED — CSQP Tau/Eng/JTC == IRL "
              "features (slower JTC residual).")
    _MPPI_P1_DICT       = None
    _FORCE_TARGET_OVERRIDE = (None if args.force_target_override is None
                              else float(args.force_target_override))
    _FRICTION_AWARE_TAU = bool(args.friction_aware_tau)
    _KIN_BACKEND        = str(args.backend)
    _MPPI_K             = int(args.mppi_K)
    _MPPI_H             = int(args.mppi_H)
    _MPPI_NDIAL         = int(args.mppi_ndial)
    _MPPI_NOISE_SCALE   = float(args.mppi_noise_scale)
    _MPPI_QUIET_SOLVE   = bool(args.mppi_quiet_solve)
    _MPPI_PRESS_EMERGENT  = bool(args.mppi_press_emergent)
    _MPPI_PROGRESS_VEL_SQ = bool(args.mppi_progress_vel_sq)
    _MPPI_FORCE_MAX       = float(args.mppi_force_max)
    _MPPI_P2_FEATURES   = (list(args.mppi_p2_features)
                           if args.mppi_p2_features else None)
    if _KIN_BACKEND == 'mjx' and not args.mppi_use_sgd:
        print(f"[batch] --backend mjx + Adam: MJX's ~1e-3 noise can "
              f"compound through Adam's per-coord normalization. "
              f"Strongly consider adding --mppi_use_sgd. "
              f"(See memory [[feedback_no_adam]].)")
    if _KIN_BACKEND == 'cpu' and _MPPI_K >= 1024:
        print(f"[batch] WARNING: --mppi_K {_MPPI_K} on CPU is ~{_MPPI_K/256:.0f}× "
              f"slower than the CPU default K=256 (CPU per-rollout cost is "
              f"flat in K). Use --backend mjx to amortize the kernel cost.")
    print(f"[batch] kinematic-MPPI backend = {_KIN_BACKEND}  "
          f"K = {_MPPI_K}  H = {_MPPI_H}  n_dial = {_MPPI_NDIAL}  "
          f"noise_scale = {_MPPI_NOISE_SCALE:.2f}")
    if _FRICTION_AWARE_TAU:
        print(f"[batch] friction-aware demo torques ENABLED — "
              f"u_demo = u_RNEA - tau_bias(q,v)")
    if _FORCE_TARGET_OVERRIDE is not None:
        print(f"[batch] force_target OVERRIDE active: "
              f"all demos use constant {_FORCE_TARGET_OVERRIDE:.1f}N target "
              f"(demo mean ignored)")
    if args.mppi_p1_progress_vel is not None:
        from mppi_cpu import KEYS_RUN as _KEYS_RUN
        _MPPI_P1_DICT = {k: 0.0 for k in sorted(_KEYS_RUN)}
        _MPPI_P1_DICT["progress_vel"] = float(args.mppi_p1_progress_vel)
        print(f"[batch] Phase 1 SKIPPED — using hand-picked progress_vel="
              f"{args.mppi_p1_progress_vel}, all other features at 0")
    elif args.mppi_p1_npz:
        from mppi_cpu import KEYS_RUN as _KEYS_RUN
        _sorted = sorted(_KEYS_RUN)
        d = np.load(args.mppi_p1_npz, allow_pickle=True)
        w_flat = d['w_final']; n_w_p1 = int(d['n_w'])
        if 'keys_run' in d.files:
            loaded_keys = sorted(list(d['keys_run']))
        else:
            loaded_keys = _sorted
        nr_loaded = len(loaded_keys)
        w_tv = w_flat[:n_w_p1 * nr_loaded].reshape(n_w_p1, nr_loaded)
        w_avg_loaded = w_tv.mean(axis=0)
        loaded_dict = {k: float(w_avg_loaded[i]) for i, k in enumerate(loaded_keys)}
        _MPPI_P1_DICT = {k: float(loaded_dict.get(k, 0.0)) for k in _sorted}
        added = [k for k in _sorted if k not in loaded_keys]
        if added:
            print(f"[batch] Phase 1 npz has older key set; padding new keys "
                  f"with 0: {added}")
        print(f"[batch] Phase 1 LOADED from {args.mppi_p1_npz} "
              f"(n_w={n_w_p1}, averaged): "
              f"{ {k: round(v,4) for k,v in _MPPI_P1_DICT.items() if abs(v) > 1e-6} }")
    print(f"[batch] mu={_BUILD_MU}  target_force={_BUILD_TARGET_FORCE}N  "
          f"verbose={_VERBOSE_IRL}")

    combos  = [c for c in COMBOS if (not args.combos or c["name"] in args.combos)]
    methods = args.methods or METHODS
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for combo in combos:
        combo["_csqp_n_ws"]    = args.n_w
        combo["_csqp_mode"]    = args.mode
        combo["_csqp_max_iter"] = int(args.max_iter)
        combo["_csqp_press"]   = not args.no_press_force
        combo["_csqp_mask_out"] = (list(args.csqp_mask_out)
                                    if args.csqp_mask_out else None)
        combo["_csqp_task_seed"] = float(args.csqp_task_seed)
        combo["_csqp_progressive_rounds"] = int(args.csqp_progressive_rounds)
        combo["_csqp_progressive_growth"] = float(args.csqp_progressive_growth)
        combo["_csqp_progressive_qtol"]   = float(args.csqp_progressive_qtol)
        combo["_feature_scale"] = bool(args.feature_scale)
        combo["_q_norm_force_weight"] = float(args.q_norm_force_weight)
        combo["_csqp_line_search_steps"] = int(args.csqp_line_search_steps)
        combo["_csqp_normalize_w"] = bool(args.csqp_normalize_w)
        combo["_csqp_use_dq_norm"] = bool(args.csqp_ls_use_dq)
        combo["_csqp_dq_norm_tol"] = float(args.csqp_ls_dq_tol)
        combo["_csqp_use_accel_guard"] = bool(args.csqp_ls_use_accel)
        combo["_csqp_accel_tol"] = float(args.csqp_ls_accel_tol)
        combo["_csqp_select_best_feature"] = bool(args.csqp_select_best_feature)
        combo["_csqp_ls_base"] = str(args.csqp_ls_base)
        combo["_mppi_n_w"]     = int(args.mppi_n_w)
        combo["_mppi_p1_lr"]    = (None if args.mppi_lr is None
                                   else float(args.mppi_lr))
        combo["_mppi_p1_steps"] = (None if args.mppi_steps is None
                                   else list(args.mppi_steps))
        combo["_mppi_use_sgd"]  = bool(args.mppi_use_sgd)
        combo["_mppi_p1_init"]  = float(args.mppi_p1_init)
        combo["_mppi_mode"]     = args.mppi_mode
        combo["_mppi_basis_type"] = args.mppi_basis_type
        combo["_mppi_p1_dict"]  = _MPPI_P1_DICT
        rid = args.run_id.strip()
        if args.mu != 0.3:
            rid = f"mu{args.mu:.2f}" + (f"__{rid}" if rid else "")
        if _FORCE_TARGET_OVERRIDE is not None:
            ftag = f"f{_FORCE_TARGET_OVERRIDE:g}N"
            rid  = ftag + (f"__{rid}" if rid else "")
        if _FRICTION_AWARE_TAU:
            rid  = "fricaware" + (f"__{rid}" if rid else "")
        if args.csqp_mask_out:
            mtag = "no" + "+".join(args.csqp_mask_out)
            rid  = mtag + (f"__{rid}" if rid else "")
        if args.csqp_task_seed != 1e-6:
            stag = f"seed{args.csqp_task_seed:g}"
            rid  = stag + (f"__{rid}" if rid else "")
        if args.feature_scale:
            rid  = "fscale" + (f"__{rid}" if rid else "")
        if args.q_norm_force_weight > 0.0:
            rid  = f"qf{args.q_norm_force_weight:g}" + (f"__{rid}" if rid else "")
        combo["_run_id"]        = rid

    rows = []
    for combo in combos:
        run_combo(combo, methods, rows)

    if rows:
        pd.DataFrame(rows).to_csv(OUT_ROOT / "summary.csv", index=False)
        print(f"\n[done] wrote {OUT_ROOT / 'summary.csv'}")

if __name__ == "__main__":
    main()
"""
irl_phases.py
=============
Clean two-phase IRL with CPU MPPI.

Phase 1: Learn task weights (approach, traveled, rail_lat, rock_ori)
Phase 2: Freeze task weights, learn style weights (JA, JV, Geo, surface)

Usage (from notebook):
    from irl_phases import run_phase1, run_phase2

    w1, IRL1 = run_phase1(human_mppi, kin_cpu, xs_opt_list, us_opt_list,
                           mppi_lr=100, max_iter=10)

    w2, IRL2 = run_phase2(human_mppi, kin_cpu, xs_opt_list, us_opt_list,
                           w_phase1=w1, mppi_lr=10, max_iter=20)
"""

import numpy as np
from importlib import reload
import mppi_cpu; reload(mppi_cpu)
from mppi_cpu import KinematicMPPI_CPU, KEYS_RUN
import MO_IRL as MO_IRL_mod; reload(MO_IRL_mod)
from MO_IRL import MO_IRL
import IRL; reload(IRL)
import IRL_utils; reload(IRL_utils)

SORTED_KEYS = sorted(KEYS_RUN)

def _build_mask(features, n_w=1):
    """Build gradient mask: 1.0 for listed features, 0.0 for others.

    For n_w > 1 the per-feature mask is tiled across windows so each window
    has the same set of unlocked features. The MO_IRL parameter vector is
    laid out as [run_block_0, ..., run_block_{n_w-1}, term_block_0, ...],
    so for nr_term=0 (the MPPI keys) the tiled mask of length n_w * nr_run
    matches the parameter vector exactly.
    """
    mask = np.zeros(len(SORTED_KEYS))
    for feat in features:
        if isinstance(feat, tuple):
            name, scale = feat
        else:
            name, scale = feat, 1.0
        idx = SORTED_KEYS.index(name)
        mask[idx] = scale
    if n_w > 1:
        mask = np.tile(mask, n_w)
    return mask

def _build_mo_args(temperature=0.5, mppi_lr=100, line_search='q_norm',
                   gradient_mask=None, outer_optimizer='lbfgs',
                   lbfgs_history=5, lbfgs_lr=1.0, lbfgs_gamma_cap=10.0,
                   verbose_irl=False, per_feature_scale=False):
    """Standard mo_args bundle. Defaults match the test_phase2 settings:
    - q_norm line search (faster q_norm drops than 'opt' criterion).
    - LBFGS outer optimizer (curvature memory across outer iters).
    - γ cap = 10 prevents the early-iter explosion when y is small.
    Override any of these via the kwargs.
    """
    return {
        'use_mppi_grad': True,
        'line_search_base': line_search,
        'line_search_steps': 7,
        'sqp_iter': 30,
        'next_traj': 'worst',
        'K_set': 'all',
        'temperature': temperature,
        'mppi_lr': mppi_lr,
        'outer_optimizer': outer_optimizer,
        'lbfgs_history': lbfgs_history,
        'lbfgs_lr': lbfgs_lr,
        'lbfgs_gamma_cap': lbfgs_gamma_cap,
        'l_type': 1,
        'l_reg': 'elastic',
        'Lambda': 0.0,
        'Beta': 0.0,
        'dyn_reg': False,
        'use_jac': True,
        'use_hess': False,
        'use_best': False,
        'N_samples': 50,
        'scaled_sum': True,
        'normalize_w': False,
        'normalize_features': False,
        'normalizing_thrs': 1.0,
        'gradient_mask': gradient_mask,
        'opt_vars': 'all',
        'with_temp_adjust': False,
        'stopping': line_search,
        'tol': 1e-10,
        'with_dmp': False,
        'w_bounds': None,
        'verbose_irl': verbose_irl,
        'per_feature_scale': bool(per_feature_scale),
    }

def run_phase1(human_mppi, kin_cpu, xs_opt_list, us_opt_list,
               mppi_lr=100, temperature=0.5, max_iter=10,
               steps=None, n_w=1, use_adam=False, init_weight=1.0,
               per_feature_scale=False, outer_optimizer='lbfgs',
               line_search='q_norm'):
    """
    Phase 1: Learn velocity-tracking weight only.

    Single feature: progress_vel — gets the system moving along the rail
    before any positional / orientation constraints are introduced. Mirrors
    the single-feature warmstart pattern from test_single_feature.py.

    Returns: (best_weights_dict, IRL_object)
    """
    features = ['progress_vel']
    mask = _build_mask(features, n_w=n_w)

    W_INIT = {k: 0.0 for k in KEYS_RUN}
    W_INIT['progress_vel'] = float(init_weight)

    kin_cpu.set_weights(W_INIT)
    kin_cpu.install_as_solver(human_mppi)

    irl_args = {
        'model': human_mppi,
        'w_run': dict(W_INIT),
        'w_term': {},
        'xs_opt': xs_opt_list,
        'us_opt': us_opt_list,
        'irl_iter': 1000,
        'stopping': 'opt',
        'tol': 1e-10,
        'max_iter': max_iter,
        'min_iter': 1,
        'compare_desired': False,
        'verbose': True,
        'n_w': n_w,
        'use_adam': bool(use_adam),
    }

    mo_args = _build_mo_args(temperature=temperature, mppi_lr=mppi_lr,
                              line_search=line_search, gradient_mask=mask,
                              per_feature_scale=per_feature_scale,
                              outer_optimizer=outer_optimizer)

    if steps is None:
        steps = np.array([50, 20, 10, 5, 2, 1, 0.5])

    print("=" * 60)
    print("Phase 1: Velocity tracking (progress_vel only)")
    print(f"  Feature: progress_vel  init={float(init_weight):g}")
    print(f"  lr={mppi_lr}, temp={temperature}, max_iter={max_iter}, n_w={n_w}, "
          f"optimizer={'Adam' if use_adam else 'SGD'}")
    print("=" * 60)

    IRL_obj = MO_IRL(mo_args, irl_args)
    IRL_obj.steps = steps
    IRL_obj.line_search.steps = steps
    IRL_obj.solve()

    best_idx = np.argmin(IRL_obj.q_norm)
    w_best = IRL_obj.ws[best_idx][-1].copy()

    print(f"\nPhase 1 results:")
    print(f"  q_norm: {IRL_obj.q_norm[0]:.4f} → {IRL_obj.q_norm[best_idx]:.4f} (best @ iter {best_idx})")
    print(f"  Weights:")
    for j, k in enumerate(IRL_obj.keys_run):
        if w_best[j] > 0.01:
            print(f"    {k:<20} {w_best[j]:.4f}")

    w_dict = {k: float(w_best[j]) for j, k in enumerate(IRL_obj.keys_run)}
    return w_dict, IRL_obj

def run_phase2(human_mppi, kin_cpu, xs_opt_list, us_opt_list,
               w_phase1, mppi_lr=10, temperature=0.5, max_iter=20,
               steps=None, features=None, rock_ori_scale=1.0, n_w=1,
               use_adam=False, weight_mode='windowed',
               basis_type='gaussian', basis_sigma=None,
               per_feature_scale=False, outer_optimizer='lbfgs',
               line_search='q_norm'):
    """
    Phase 2: Learn task placement weights, warm-started from Phase 1's
    velocity-tracking weight.

    Default features: approach, traveled, rail_lat, rock_ori (the four
    placement / orientation tasks). rock_ori is amplified in the gradient
    mask by `rock_ori_scale` because its un-normalized magnitude is small
    relative to the others.

    Returns: (best_weights_dict, IRL_object)
    """
    if features is None:
        features = [
            'approach',
            'traveled',
            'rail_lat',
            ('rock_ori', rock_ori_scale),
        ]

    mask = _build_mask(features, n_w=n_w)

    W_INIT = {k: 0.0 for k in KEYS_RUN}
    for k, v in w_phase1.items():
        W_INIT[k] = v
    for feat in features:
        name = feat if isinstance(feat, str) else feat[0]
        if W_INIT[name] < 0.01:
            W_INIT[name] = 1.0

    kin_cpu.set_weights(W_INIT)
    kin_cpu.install_as_solver(human_mppi)

    irl_args = {
        'model': human_mppi,
        'w_run': dict(W_INIT),
        'w_term': {},
        'xs_opt': xs_opt_list,
        'us_opt': us_opt_list,
        'irl_iter': 1000,
        'stopping': 'opt',
        'tol': 1e-10,
        'max_iter': max_iter,
        'min_iter': 1,
        'compare_desired': False,
        'verbose': True,
        'n_w': n_w,
        'use_adam': bool(use_adam),
    }
    if weight_mode == 'basis':
        irl_args['weight_mode'] = 'basis'
        irl_args['K']           = n_w
        irl_args['basis_type']  = basis_type
        irl_args['basis_sigma'] = basis_sigma

    mo_args = _build_mo_args(temperature=temperature, mppi_lr=mppi_lr,
                              line_search=line_search, gradient_mask=mask,
                              per_feature_scale=per_feature_scale,
                              outer_optimizer=outer_optimizer)

    if steps is None:
        steps = np.array([1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005])

    print("=" * 60)
    print("Phase 2: Task placement weights")
    print(f"  Features (added on top of Phase 1): {features}")
    print(f"  Warm-started from Phase 1: {[k for k,v in w_phase1.items() if v > 0.01]}")
    print(f"  lr={mppi_lr}, temp={temperature}, max_iter={max_iter}, n_w={n_w}, "
          f"optimizer={'Adam' if use_adam else 'SGD'}, "
          f"weight_mode={weight_mode}"
          + (f" (basis_type={basis_type})" if weight_mode == 'basis' else ""))
    print("=" * 60)

    IRL_obj = MO_IRL(mo_args, irl_args)
    IRL_obj.steps = steps
    IRL_obj.line_search.steps = steps
    IRL_obj.solve()

    best_idx = np.argmin(IRL_obj.q_norm)
    w_best = IRL_obj.ws[best_idx][-1].copy()

    print(f"\nPhase 2 results:")
    print(f"  q_norm: {IRL_obj.q_norm[0]:.4f} → {IRL_obj.q_norm[best_idx]:.4f} (best @ iter {best_idx})")
    print(f"  All weights:")
    for j, k in enumerate(IRL_obj.keys_run):
        if w_best[j] > 0.01:
            print(f"    {k:<20} {w_best[j]:.4f}")

    w_dict = {k: float(w_best[j]) for j, k in enumerate(IRL_obj.keys_run)}
    return w_dict, IRL_obj

def run_phase3(human_mppi, kin_cpu, xs_opt_list, us_opt_list,
               w_phase2, mppi_lr=5, temperature=0.5, max_iter=20,
               steps=None, n_w=1, use_adam=False,
               weight_mode='windowed', basis_type='gaussian',
               basis_sigma=None, per_feature_scale=False,
               outer_optimizer='lbfgs'):
    """
    Phase 3: Open up ALL features (task + style + force) — warm-start from
    Phase 2 and let every weight learn. No gradient mask (learn everything).
    """
    W_INIT = {k: 0.0 for k in KEYS_RUN}
    for k, v in w_phase2.items():
        W_INIT[k] = v
    for k in KEYS_RUN:
        if W_INIT[k] < 0.01:
            W_INIT[k] = 1e-4

    kin_cpu.set_weights(W_INIT)
    kin_cpu.install_as_solver(human_mppi)

    irl_args = {
        'model': human_mppi,
        'w_run': dict(W_INIT),
        'w_term': {},
        'xs_opt': xs_opt_list,
        'us_opt': us_opt_list,
        'irl_iter': 1000,
        'stopping': 'opt',
        'tol': 1e-10,
        'max_iter': max_iter,
        'min_iter': 1,
        'compare_desired': False,
        'verbose': True,
        'n_w': n_w,
        'use_adam': bool(use_adam),
    }
    if weight_mode == 'basis':
        irl_args['weight_mode'] = 'basis'
        irl_args['K']           = n_w
        irl_args['basis_type']  = basis_type
        irl_args['basis_sigma'] = basis_sigma

    mo_args = _build_mo_args(temperature=temperature, mppi_lr=mppi_lr,
                              line_search='opt', gradient_mask=None,
                              per_feature_scale=per_feature_scale,
                              outer_optimizer=outer_optimizer)

    if steps is None:
        steps = np.array([1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005])

    print("=" * 60)
    print("Phase 3: All Features (task + style + force; no gradient mask)")
    print(f"  warm-start features: {[k for k,v in w_phase2.items() if v > 0.01]}")
    print(f"  lr={mppi_lr}, temp={temperature}, max_iter={max_iter}, n_w={n_w}, "
          f"optimizer={'Adam' if use_adam else 'SGD'}")
    print("=" * 60)

    IRL_obj = MO_IRL(mo_args, irl_args)
    IRL_obj.steps = steps
    IRL_obj.line_search.steps = steps
    IRL_obj.solve()

    best_idx = np.argmin(IRL_obj.q_norm)
    w_best = IRL_obj.ws[best_idx][-1].copy()

    print(f"\nPhase 3 results:")
    print(f"  q_norm: {IRL_obj.q_norm[0]:.4f} → {IRL_obj.q_norm[best_idx]:.4f} (best @ iter {best_idx})")
    for j, k in enumerate(IRL_obj.keys_run):
        if w_best[j] > 0.01:
            print(f"    {k:<20} {w_best[j]:.4f}")

    w_dict = {k: float(w_best[j]) for j, k in enumerate(IRL_obj.keys_run)}
    return w_dict, IRL_obj

def analyze_features(human_mppi, xs_traj, label="Trajectory"):
    """Print per-timestep features for a trajectory."""
    import mujoco

    model = human_mppi.mj_model
    rock_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rock_sphere")
    stick_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stick")
    contact_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rock_contact_point")
    rock_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rock")

    rail_vec = human_mppi.p_end_world - human_mppi.p_start_world
    rail_len = np.linalg.norm(rail_vec)
    rail_dir = rail_vec / rail_len

    kin = getattr(human_mppi, '_kin_mppi', None)
    q_ref = kin.rock_quat_ref if kin and hasattr(kin, 'rock_quat_ref') and kin.rock_quat_ref is not None else None

    print(f"\n{label}:")
    print(f"{'t':>4} {'progress_vel':>13} {'approach':>10} {'gap':>10} {'traveled':>10} {'rail_lat':>10} {'rock_ori':>10} {'force':>10}")
    print("-" * 85)

    for t in range(0, len(xs_traj), 5):
        x = xs_traj[t] if isinstance(xs_traj[t], np.ndarray) else np.array(xs_traj[t])
        q = x[:human_mppi.nq]
        v = x[human_mppi.nq:]
        human_mppi._sync_mujoco(q, v, np.zeros(model.nu), t=t)

        data = human_mppi.mj_data

        rock_pos = data.geom_xpos[rock_geom_id]
        stick_center = data.xpos[stick_body_id]
        stick_axis = data.xmat[stick_body_id].reshape(3, 3)[:, 2]
        t_proj = np.clip(np.dot(rock_pos - stick_center, stick_axis), -0.4, 0.4)
        nearest = stick_center + t_proj * stick_axis
        raw_dist = np.linalg.norm(rock_pos - nearest)
        gap = raw_dist - 0.04
        approach = raw_dist / 0.02

        site_pos = data.site_xpos[contact_site_id]
        traveled_dist = np.dot(site_pos - human_mppi.p_start_world, rail_dir)
        progress_pct = traveled_dist / rail_len * 100
        traveled = -(traveled_dist / rail_len)

        diff = site_pos - human_mppi.p_start_world
        lateral = diff - np.dot(diff, rail_dir) * rail_dir
        rail_lat = np.linalg.norm(lateral) / 0.02

        rock_ori = 0.0
        if q_ref is not None:
            q_cur = data.xquat[rock_body_id]
            dot = np.dot(q_cur, q_ref)
            rock_ori = 1.0 - dot ** 2

        f_contact = 0.0
        for ci in range(data.ncon):
            c = data.contact[ci]
            pair = {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]}
            if pair == {model.geom_bodyid[rock_geom_id] if rock_geom_id >= 0 else -1, stick_body_id}:
                fb = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, fb)
                f_contact = max(f_contact, abs(fb[0]))

        print(f"{t:4d} {progress_pct:9.1f}% {approach:10.2f} {gap:10.4f} {traveled:10.4f} {rail_lat:10.2f} {rock_ori:10.4f} {f_contact:9.1f}N")

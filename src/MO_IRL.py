"""Multi-objective MO-IRL solver.

Recovers non-negative, time-varying cost weights W(t) from a small set of
demonstrations. The outer loop proposes a weight update, an inner trajectory
generator (CSQP or MPPI) re-solves under it, and the two feature integrals are
compared; a line search accepts the step. W(t) is carried either as independent
windows or as coefficients on a Gaussian partition-of-unity basis.
"""
import os

from IPython.display import HTML
import numpy as np
from matplotlib import pyplot as plt
from IRL_utils import *
from Optimization_utils import *
from DMP import *
from IRL import IRL_Solver

class MO_IRL(IRL_Solver):
    def __init__(self, args, irl_args):
        super().__init__(irl_args)
        self.args = args
        self.gradient_mask = self.args['gradient_mask']
        self.K_set = self.args['K_set']
        self.N_samples = self.args['N_samples']
        self.l_type = self.args['l_type']
        self.opt_vars = self.args['opt_vars']
        self.slsqp_inner = bool(self.args.get('slsqp_inner', False))
        self.l_reg = self.args['l_reg']
        self.normalize_w = self.args['normalize_w']
        self.use_dq_norm = args.get('use_dq_norm', False)
        self.dq_norm_tol = float(args.get('dq_norm_tol', 2.0))
        self.use_accel_guard = args.get('use_accel_guard', False)
        self.accel_tol = float(args.get('accel_tol', 3.0))
        self.Lambda = args['Lambda']
        self.Beta = args['Beta']
        self.dyn_reg = args['dyn_reg']
        self.use_jac = args['use_jac']
        self.use_hess = args['use_hess']
        self.stopping = args['stopping']
        self.tol = args['tol']
        self.temperature = args['temperature']
        self.use_best = args['use_best']
        self.scaled_sum = args['scaled_sum']
        self.normalize_phis = args['normalize_features']; self.norm_thrs = args['normalizing_thrs']
        self.save_history_path = args.get('save_history_path', None)

        if self.args['with_temp_adjust']:
            self.lambda_thrs = self.args['lambda_thrs']
            self.lambda_t = self.args['lambda_init']
            self.lambda_set = [self.lambda_t]
            self.Delta = np.inf
            self.Deltas = [np.inf]
            self.delta = self.args['delta']
            self.kappa = self.args['kappa']
            self.temperature = 1/self.lambda_t
        self.line_search_steps = self.args['line_search_steps']
        self.line_search_base = self.args['line_search_base']
        if 'with_dmp' in self.args and self.args['with_dmp']:
            self.with_dmp = self.args['with_dmp']
            dmp_args = {
                'T': self.T,
                'dt': self.dt,
                'basis_num': self.args['basis_num'],
                'alpha_x': self.args['alpha_x'],
                'alpha_z': self.args['alpha_z'],
                'beta_z': self.args['beta_z'],
                'model': self.model,
                'noise_f': self.args['noise_f'],
                'tau': self.args['tau']
            }
            self.dmp = DMP(dmp_args)
            self.Xs, self.Us, self.phis, self.phis_set, self.phis_set_int = self.generate_trajectories()

        self.steps = 0.5**(2*(np.array(list(range(self.line_search_steps)))))

        # the model exposes target_force_profile. Must be installed BEFORE
        if bool(self.args.get('q_norm_meanjoint', False)):
            def _q_norm_meanjoint(xs1, xs2, to_deg=True):
                return float(np.mean(self.get_q_rmse(xs1, xs2, to_deg=to_deg)))
            self.get_q_norm = _q_norm_meanjoint
            print("[MO_IRL] base q_norm = MEAN per-joint RMSE(deg) [q_norm_meanjoint]")

        self.q_norm_force_weight = float(self.args.get('q_norm_force_weight', 0.0))
        if (self.q_norm_force_weight > 0.0
                and getattr(self.model, 'target_force_profile', None) is not None):
            _base_q_norm = self.get_q_norm
            _model       = self.model
            _w_force     = self.q_norm_force_weight
            def _q_norm_with_force(xs1, xs2, to_deg=True):
                joint = float(_base_q_norm(xs1, xs2, to_deg=to_deg))
                target = getattr(_model, 'target_force_profile', None)
                if target is None:
                    return joint
                fz = None
                try:
                    running = _model.solver.problem.runningDatas
                    fz_croc = np.zeros(len(running))
                    for _t, _d in enumerate(running):
                        _contacts = _d.differential.multibody.contacts.contacts.todict()
                        if not _contacts:
                            continue
                        _f = list(_contacts.values())[0].f.linear
                        if not np.any(np.isnan(_f)):
                            fz_croc[_t] = float(_f[2])
                    fz = fz_croc
                except (AttributeError, Exception):
                    pass
                if fz is None:
                    pf = getattr(_model, '_press_forces', None)
                    if pf is not None:
                        fz = np.asarray(pf, dtype=np.float64)
                if fz is None:
                    return joint
                _Tf = min(len(fz), len(target))
                if _Tf == 0:
                    return joint
                Tps = int(getattr(_model, 'T_press_start', 0) or 0)
                mask = np.arange(_Tf) >= Tps if Tps > 0 else np.ones(_Tf, dtype=bool)
                if not mask.any():
                    return joint
                f_diff = fz[:_Tf][mask] - np.asarray(target)[:_Tf][mask]
                f_rmse = float(np.sqrt(np.mean(f_diff ** 2)))
                return joint + _w_force * f_rmse
            self.get_q_norm = _q_norm_with_force
            print(f"[MO_IRL] q_norm criterion = joint_RMSE(deg) + "
                  f"{_w_force} * force_RMSE(N)  "
                  f"(MPPI fallback via _press_forces)")
        
        if 'xs_nopt' not in self.args or 'us_nopt' not in self.args:
            self.xs_nopt, self.us_nopt = self.generate_bad_trajectory()
        else:
            self.xs_nopt = np.stack(self.args['xs_nopt'])
            self.us_nopt = np.stack(self.args['us_nopt'])

        if self.models:
            phi_nopt, phis_nopt, _, phis_nopt_int = avg_features(self.models)
        else:
            phi_nopt, phis_nopt, _, phis_nopt_int = self.model.get_traj_features(self.xs_nopt, self.us_nopt)
        self.phis.append(phi_nopt); self.phis_set.append(phis_nopt); self.phis_set_int.append(phis_nopt_int)
        self.Xs.append(self.xs_nopt); self.Us.append(self.us_nopt)
        self.opt_div = [np.inf]
        self.cost_diffs = [np.inf]

        L = min(len(self.Xs[0]), len(self.xs_nopt))
        self.q_rmse = [self.get_q_rmse(self.Xs[0][:L], self.xs_nopt[:L])]
        self.q_norm = [float(self.get_q_norm(self.Xs[0][:L], self.xs_nopt[:L]))]
        self.dq_norm = [float(self.get_dq_norm(self.Xs[0][:L], self.xs_nopt[:L]))]

        if self.compare_desired:
            self.cost_diffs_des = [self.get_cost_diff(self.get_traj_costs(phis_nopt, self.w_run_star_v, self.w_term_star_v)[1], self.get_traj_costs(self.phis_opt, self.w_run_star_v, self.w_term_star_v)[1])]
        self.state_diffs = [np.inf]

        self.opt_div = [self.get_opt_div(phis_nopt_int, self.phis_opt_int)]
        self.cost_diffs = [self.get_cost_diff(self.get_traj_costs(phis_nopt, self.ws[0][0], self.ws[0][1])[1], self.get_traj_costs(self.phis_opt, self.ws[0][0], self.ws[0][1])[1])]
        if self.models:
            _chal = [np.stack(m.solver.xs.copy()) for m in self.models[:self.n_demos]]
            _Li = [min(len(self.Xs[i]), len(_chal[i])) for i in range(self.n_demos)]
            self.q_rmse = [np.mean([self.get_q_rmse(self.Xs[i][:_Li[i]], _chal[i][:_Li[i]])
                                    for i in range(self.n_demos)], axis=0)]
            self.q_norm = [float(np.mean([self.get_q_norm(self.Xs[i][:_Li[i]], _chal[i][:_Li[i]])
                                          for i in range(self.n_demos)]))]
            self.dq_norm = [float(np.mean([self.get_dq_norm(self.Xs[i][:_Li[i]], _chal[i][:_Li[i]])
                                           for i in range(self.n_demos)]))]
        else:
            self.q_rmse = [self.get_q_rmse(self.xs_opt, self.xs_nopt)]
            self.q_norm = [self.get_q_norm(self.xs_opt, self.xs_nopt)]
            self.dq_norm = [float(self.get_dq_norm(self.xs_opt, self.xs_nopt))]
        self.state_diffs = [self.get_state_diff(self.xs_nopt, self.Xs[0])]
        self.phi_diffs_history = []

 
        demo_signal_thresh = float(self.args.get('demo_signal_thresh', 1e-6))
        phi_demo_final =  np.abs(np.sum(np.asarray(self.phis_opt_int), axis=0))
        per_feature_signal = (phi_demo_final > demo_signal_thresh).astype(float)
        # demo never exercises this feature". Masking them freezes the weight at
        # Never-mask these; override the list via args['never_mask_features'].
        _never_mask = self.args.get('never_mask_features', ['press_force'])
        _keys_all = list(getattr(self, 'keys_run', [])) + list(getattr(self, 'keys_term', []))
        if len(_keys_all) == len(per_feature_signal):
            for _k in _never_mask:
                if _k in _keys_all:
                    per_feature_signal[_keys_all.index(_k)] = 1.0
        nr_r, nr_t = self.nr_run, self.nr_term

        if self.n_w > 1 and len(per_feature_signal) == (nr_r + nr_t):
            sig_run  = np.tile(per_feature_signal[:nr_r], self.n_w)
            sig_term = (np.tile(per_feature_signal[nr_r:nr_r + nr_t], self.n_w)
                        if nr_t > 0 else np.array([]))
            self.demo_signal_mask = np.concatenate([sig_run, sig_term])
        else:
            self.demo_signal_mask = per_feature_signal
        zeroed = [k for k, s in zip(
            list(self.keys_run) + list(self.keys_term), per_feature_signal)
            if s == 0.0]
        print(f"[MO_IRL] demo_signal_mask: phi_demo ≈ 0 → masking gradient for {zeroed} "
              f"(never_mask kept: {[k for k in _never_mask if k in _keys_all]})")

        if self.args.get('per_feature_scale', False):
            self._apply_per_feature_scale(rel_floor=self.args.get(
                'phi_scale_rel_floor', 0.01))
        else:
            self._feature_scale = np.ones(self.nr)

        self._mppi_phi_norm = os.environ.get(
            'MPPI_PHI_NORM', '').strip().lower() in ('1', 'true', 'on', 'yes')
        if self._mppi_phi_norm:
            _fs = getattr(self, '_feature_scale', None)
            if _fs is not None and not np.allclose(_fs, 1.0):
                _sigma_by_key = {k: float(_fs[i])
                                 for i, k in enumerate(self.keys_run)}
                _kins = []
                _k0 = getattr(self.model, '_kin_mppi', None)
                if _k0 is not None:
                    _kins.append(_k0)
                for _m in (getattr(self, 'models', None) or []):
                    _km = getattr(_m, '_kin_mppi', None)
                    if _km is not None and all(_km is not k for k in _kins):
                        _kins.append(_km)
                _pushed = 0
                for _kin in _kins:
                    if hasattr(_kin, 'set_phi_norm'):
                        _kin.set_phi_norm(_sigma_by_key)
                        _pushed += 1
                print(f"[MO_IRL] MPPI_PHI_NORM ON — pushed per-feature σ (max|φ|) "
                      f"to {_pushed} kin(s) by KEY; MPPI softmax now selects on "
                      f"Σ w·(φ/σ) (matches the σ-scaled IRL gradient)")
            else:
                print("[MO_IRL] MPPI_PHI_NORM ON but σ is all-ones "
                      "(per_feature_scale off?) — no σ pushed, MPPI stays raw")

        if self.args.get('demo_force_at_target', False) and 'press_force' in self.keys_run:
            pf = list(self.keys_run).index('press_force')
            def _zero_pf(arr):
                for t in range(len(arr)):
                    a = np.asarray(arr[t], float)
                    if pf < len(a):
                        a[pf] = 0.0
                    arr[t] = a
            nd = int(getattr(self, 'n_demos', 1))
            for i in range(min(nd, len(self.phis_set))):
                _zero_pf(self.phis_set[i])
                _zero_pf(self.phis_set_int[i])
                self.phis[i] = np.asarray(self.phis[i], float)
                if pf < len(self.phis[i]):
                    self.phis[i][pf] = 0.0
            _zero_pf(self.phis_opt)
            _zero_pf(self.phis_opt_int)
            self.phi_opt = np.asarray(self.phi_opt, float)
            if pf < len(self.phi_opt):
                self.phi_opt[pf] = 0.0
            print(f"[MO_IRL] demo_force_at_target: zeroed demo press_force feature "
                  f"(idx {pf}) in {nd} per-demo + mean arrays -> gradient drives pressing")

        if (self.args.get('demo_force_measured', False)
                and self.args.get('force_two_cost', False)
                and 'press_force' in self.keys_run
                and 'press_capacity' in self.keys_run):
            pf = list(self.keys_run).index('press_force')
            pc = list(self.keys_run).index('press_capacity')
            fmax = float(self.args.get('force_max', 80.0))
            sc = getattr(self, '_feature_scale', np.ones(self.nr))
            nd = int(getattr(self, 'n_demos', 1))
            models = getattr(self, 'models', None) or [self.model]
            def _set_pf(arr, vpf, vpc, integrate=False, dt=0.0):
                run_pf = run_pc = 0.0
                for t in range(len(arr)):
                    a = np.asarray(arr[t], float)
                    if integrate:
                        run_pf += vpf * dt; run_pc += vpc * dt
                        vf, vc = run_pf, run_pc
                    else:
                        vf, vc = vpf, vpc
                    if pf < len(a): a[pf] = vf
                    if pc < len(a): a[pc] = vc
                    arr[t] = a
            def _frac(f):
                vpf = ((f / fmax) ** 2) / (sc[pf] if pf < len(sc) else 1.0)
                vpc = (((f - fmax) / fmax) ** 2) / (sc[pc] if pc < len(sc) else 1.0)
                return vpf, vpc
            for i in range(min(nd, len(self.phis_set))):
                mdl = models[i] if i < len(models) else models[0]
                f_meas = float(getattr(mdl, 'target_force', 0.0) or 0.0)
                dt = float(getattr(mdl, 'dt', self.dt))
                vpf, vpc = _frac(f_meas)
                _set_pf(self.phis_set[i], vpf, vpc)
                _set_pf(self.phis_set_int[i], vpf, vpc, integrate=True, dt=dt)
                self.phis[i] = np.asarray(self.phis[i], float)
                if pf < len(self.phis[i]): self.phis[i][pf] = vpf
                if pc < len(self.phis[i]): self.phis[i][pc] = vpc
            f0 = float(getattr(models[0], 'target_force', 0.0) or 0.0)
            dt0 = float(getattr(models[0], 'dt', self.dt))
            vpf0, vpc0 = _frac(f0)
            _set_pf(self.phis_opt, vpf0, vpc0)
            _set_pf(self.phis_opt_int, vpf0, vpc0, integrate=True, dt=dt0)
            self.phi_opt = np.asarray(self.phi_opt, float)
            if pf < len(self.phi_opt): self.phi_opt[pf] = vpf0
            if pc < len(self.phi_opt): self.phi_opt[pc] = vpc0
            print(f"[MO_IRL] demo_force_measured: injected f_meas into demo "
                  f"press_force={vpf0:.4g}/press_capacity={vpc0:.4g} (scaled) for "
                  f"{nd} demos (Fmax={fmax:g}) -> demo presses at a fraction of capacity")

        # The expert reference is never in its own pool (Phis_d[0]=demo_d).
        self.pool_scope = self.args.get('pool_scope', 'averaged')
        if self.pool_scope not in ('averaged', 'same_demo', 'all_demos'):
            raise ValueError(
                f"pool_scope must be 'averaged', 'same_demo', or 'all_demos'; "
                f"got {self.pool_scope!r}")
        self._per_demo_pool = (self.pool_scope != 'averaged'
                               and bool(self.models) and len(self.models) > 1)
        self.pool_last_k = self.args.get('pool_last_k', None)
        self._pool_rollout_hist = None

        self.use_mppi_grad = self.args.get('use_mppi_grad', False)
        self.use_adam = self.args.get('use_adam', False)
        self.adam_beta1 = self.args.get('adam_beta1', 0.9)
        self.adam_beta2 = self.args.get('adam_beta2', 0.999)
        self.adam_eps   = self.args.get('adam_eps', 1e-8)
        self._adam_m = None
        self._adam_v = None
        self._adam_t = 0
        self.outer_optimizer = self.args.get('outer_optimizer', 'sgd')
        self.lbfgs_history   = int(self.args.get('lbfgs_history', 5))
        from collections import deque
        self._lbfgs_s = deque(maxlen=self.lbfgs_history)
        self._lbfgs_y = deque(maxlen=self.lbfgs_history)
        self._lbfgs_w_prev = None
        self._lbfgs_g_prev = None
        self.sqp_iter = self.args['sqp_iter']
        self.next_traj = self.args['next_traj']

        self._build_solvers()

        self.solver_args = {
            'hard_terminate': False,
            'iteration': 0,
            'KL_div': 1.0,
            'min_iter': self.min_iter,
            'max_iter': self.max_iter,
            'stopping': self.stopping,
            'tol': self.tol,
            'KL_div_star': None,
            'Opt_Dev': self.opt_div[0],
            'Q_RMSE': self.q_rmse[0],
            'Cost_Diff_Desired': np.inf,
            'Cost_Diff': self.cost_diffs[0],
            'State_Diff': np.inf,
            'Fcn_Val': 1.0,
            'Step': 1.0,
            'Step_Type': 'None',
            'message': '',
            'Phi_Diff': np.zeros(self.nr_run + self.nr_term),
        }

    def _apply_per_feature_scale(self, rel_floor=0.01):
        """Scale every stored phi array by 1/scale[k] so each feature
        contributes O(1) to the cost.

        Per-feature scale = `max(|φ|.max_t over BOTH demo and bad-trajectory,
        absolute_floor)`. Using both demo and bad trajectory captures features
        that are 0 on the demo (e.g. JA on a smooth motion) but non-zero on
        challengers — those features still need scaling. Absolute floor
        prevents the "max-feature drags everything up" pathology when one
        feature (typically press_force) has a much larger raw scale than
        the others.
        """
        nd = int(getattr(self, 'n_demos', 1))
        demo_sigmas = []
        for ph in self.phis_set[:max(nd, 1)]:
            try:
                arr = np.stack([np.asarray(p) for p in ph])
                demo_sigmas.append(np.abs(arr).max(axis=0))
            except Exception:
                pass
        if not demo_sigmas:
            self._feature_scale = np.ones(self.nr)
            return
        sigma_demo = np.maximum.reduce(demo_sigmas)
        sigmas = list(demo_sigmas)
        for ph in self.phis_set[max(nd, 1):]:
            try:
                arr = np.stack([np.asarray(p) for p in ph])
                sigmas.append(np.abs(arr).max(axis=0))
            except Exception:
                pass
        sigma = np.maximum.reduce(sigmas)
        if self.args.get('residual_scale', False):
            def _cum(ph):
                return np.stack([np.asarray(p) for p in ph]).sum(axis=0)
            demo_cums = [_cum(ph) for ph in self.phis_set[:max(nd, 1)]]
            chal_cums = [_cum(ph) for ph in self.phis_set[max(nd, 1):]]
            if demo_cums and chal_cums:
                resid = np.abs(np.mean(demo_cums, axis=0)
                               - np.mean(chal_cums, axis=0))
                rfloor = float(self.args.get('residual_scale_rel_floor', 0.05))
                sigma = np.maximum(resid, max(rfloor * float(resid.max()), 1e-6))
                print(f"[MO_IRL] residual_scale ON — scale = |Σφ_demo − Σφ_chal|"
                      f"  (rel_floor={rfloor})")
            else:
                print("[MO_IRL] residual_scale requested but no challenger in "
                      "phis_set — falling back to max|φ| scale")
        else:
            # oblivion so it never recovers (JA/JV win instead). Where the DEMO
            _CAP, _MEANINGFUL = 10.0, 1e-2
            sigma = np.where(sigma_demo > _MEANINGFUL,
                             np.minimum(sigma, _CAP * sigma_demo), sigma)
        ABSOLUTE_FLOOR = 1e-3
        scale = np.maximum(sigma, ABSOLUTE_FLOOR)
        scale = np.where(scale > 0, scale, 1.0)
        inv = 1.0 / scale
        self._feature_scale = scale

        def _scale_list(phi_list):
            return [np.asarray(p) * inv for p in phi_list]

        for i in range(len(self.phis_set)):
            self.phis_set[i]     = _scale_list(self.phis_set[i])
            self.phis_set_int[i] = _scale_list(self.phis_set_int[i])
        for i in range(len(self.phis)):
            self.phis[i] = np.asarray(self.phis[i]) * inv

        self.phi_opt          = self.phis[0]
        self.phis_opt         = self.phis_set[0]
        self.phis_opt_int     = self.phis_set_int[0]
        self.phis_opt_cum     = self.phis[0]
        if hasattr(self, 'w_run_star_v'):
            pass

        keys = list(self.keys_run) + list(self.keys_term)
        print(f"[MO_IRL] per-feature scale (max over demo+bad, abs_floor={ABSOLUTE_FLOOR}):")
        for k, s, sig in zip(keys, scale, sigma):
            print(f"  {k:18s} σ={sig:.4g}  scale={s:.4g}")

    def clear_traj(self):
        self.Xs = []; self.Us = []
        self.phis = []; self.phis_set = []; self.phis_set_int = []

    def make_feature_sets(self):
        for X, U in zip(self.Xs, self.Us):
            _, phis_, phis_set_, phis_set_int_ = self.model.get_traj_features(X, U)
            self.phis.append(phis_)
            self.phis_set.append(phis_set_)
            self.phis_set_int.append(phis_set_int_)
        self.opt_div = [np.inf]
        self.cost_diffs = [np.inf]

    def store_returns(self, returns):
        self.Xs.append(returns['new_x'].copy())
        self.Us.append(returns['new_u'].copy())
        self.ws.append(returns['new_w'])
        scale = getattr(self, '_feature_scale', None)
        if scale is not None and not np.allclose(scale, 1.0):
            inv = 1.0 / scale
            phi   = np.asarray(returns['phi']) * inv
            phis  = [np.asarray(p) * inv for p in returns['phis']]
            phis_int = [np.asarray(p) * inv for p in returns['phis_cum_int']]
        else:
            phi      = returns['phi']
            phis     = returns['phis']
            phis_int = returns['phis_cum_int']
        self.phis.append(phi)
        self.phis_set.append(phis)
        self.phis_set_int.append(phis_int)
        self.solver_args['Cost_Diff'] = returns['c_diff']; self.cost_diffs.append(self.solver_args['Cost_Diff'])
        self.solver_args['Opt_Dev'] = returns['opt_div']; self.opt_div.append(self.solver_args['Opt_Dev'])
        self.solver_args['State_Diff'] = returns['state_diff']; self.state_diffs.append(self.solver_args['State_Diff'])
        self.solver_args['Fcn_Val'] = returns['res'].fun; self.fcn_vals.append(self.solver_args['Fcn_Val'])

    def loop_termination_check(self):
        termination = False
        iteration = self.solver_args['iteration']
        hard_terminate = self.solver_args['hard_terminate']
        max_iter_termination = False
        min_iter_pass = False
        stopping_criteria = False

        if self.stopping == 'KL':
            stopping_criteria = self.solver_args['KL_div'] < self.tol
        elif self.stopping == 'cost':
            stopping_criteria = self.solver_args['Cost_Diff'] < self.tol
        elif self.stopping == 'opt':
            stopping_criteria = self.solver_args['Opt_Dev'] < self.tol
        
        
        if iteration >= self.max_iter:
            max_iter_termination = True
        if iteration >= self.min_iter:
            min_iter_pass = True
        
        if min_iter_pass and stopping_criteria:
            termination = True
            self.solver_args['message'] = 'Stopping criteria and minimum iteration termination criteria met.'
        elif max_iter_termination:
            termination = True
            self.solver_args['message'] = 'Maximum iteration termination criteria met.'
        else:
            termination = False
        
        nq = len(self.solver_args['Q_RMSE'])
        rmse_str = '  '.join('{:<9.4f}'.format(self.solver_args['Q_RMSE'][i]) for i in range(nq))
        info = '|| {:<5} || {}  {:<11.4f}  {:<11.4f}  {:<13.4f}  {:<11.4f}  {:<9.4f}  {:<4}  {:<5}'.format(
            self.solver_args['iteration']+1,
            rmse_str,
            self.solver_args['Opt_Dev'],
            self.solver_args['Cost_Diff'],
            self.solver_args['State_Diff'],
            self.solver_args['Q_NORM'],
            self.solver_args['Fcn_Val'],
            self.solver_args['Step'],
            self.solver_args['Step_Type'])

        verbose_irl = self.args.get('verbose_irl', True)

        phi_diff = self.solver_args.get('Phi_Diff', None)
        if phi_diff is not None and verbose_irl:
            all_keys = self.keys_run + self.keys_term
            phi_parts = '  |  '.join('{}: {:.5f}'.format(k, v) for k, v in zip(all_keys, phi_diff))
            info += '\n   Phi|opt-cur|: [ {} ]'.format(phi_parts)

        w_curr = self.ws[-1][-1]
        if self.K > 1:
            w_run_tv = w_curr[:self.nr_run * self.K].reshape(self.K, self.nr_run)
            label = 'theta' if self.weight_mode == 'basis' else 'w'
            for k_idx in range(self.K):
                w_parts = '  '.join('{}: {:.3e}'.format(k, w_run_tv[k_idx, j]) for j, k in enumerate(self.keys_run))
                info += '\n   Weights[{}{}]: [ {} ]'.format(label, k_idx, w_parts)
            if self.nr_term > 0:
                w_term_tv = w_curr[self.nr_run * self.K:
                                   (self.nr_run + self.nr_term) * self.K
                                  ].reshape(self.K, self.nr_term)
                for k_idx in range(self.K):
                    t_parts = '  '.join('{}: {:.3e}'.format(k, w_term_tv[k_idx, j])
                                        for j, k in enumerate(self.keys_term))
                    info += '\n   Terminal[{}{}]: [ {} ]'.format(label, k_idx, t_parts)
        else:
            w_parts = '  '.join('{}: {:.3e}'.format(k, w_curr[j]) for j, k in enumerate(self.keys_run))
            info += '\n   Weights: [ {} ]'.format(w_parts)
            if self.nr_term > 0:
                t_parts = '  '.join('{}: {:.3e}'.format(k, w_curr[self.nr_run + j])
                                    for j, k in enumerate(self.keys_term))
                info += '\n   Terminal: [ {} ]'.format(t_parts)

        if hard_terminate and min_iter_pass:
            termination = True
            if self.solver_args['message'] == '':
                self.solver_args['message'] = 'Hard termination criteria met.'
        return termination, info, self.solver_args['message']
    
    def solve(self):
        terminate = False
        w_loop = self.ws[-1][-1].copy()
        opt_inds = list(range(self.n_demos))
        nopt_inds = self.get_inds()
        if self.verbose:
            nq = self.nq
            q_header = '  '.join('q{:<2} RMSE'.format(i+1) for i in range(nq))
            print('-- iter --  {}  Opt Div      Cost Diff    State Diff     Joint Norm   Fcn Val    Step  Type'.format(q_header))
            rmse_str = '  '.join('{:<9.4f}'.format(self.q_rmse[0][i]) for i in range(nq))
            print('|| {:<5} || {}  {:<11.4f}  {:<11.4f}  {:<13.4f}  {:<11.4f}  {:<9}  {:<4}  {:<5}'.format(
                'Init.', rmse_str,
                self.opt_div[0], self.cost_diffs[0], self.state_diffs[0],
                self.q_norm[0], 'N/A', 'N/A', 'N/A'))
        while not terminate:
            w_prev = w_loop.copy()
            autoreg_returns = self.step(opt_inds, nopt_inds, w_prev, self.solver_args)
            w_loop = autoreg_returns['w_loop']
            self.Js.append(autoreg_returns['jac'])

            if not self.solver_args['hard_terminate']:
                self.iter += 1
                self.Xs.append(autoreg_returns['new_x'].copy())
                self.Us.append(autoreg_returns['new_u'].copy())
                self.phis.append(autoreg_returns['phi'])
                self.phis_set.append(autoreg_returns['phis'])
                self.phis_set_int.append(autoreg_returns['phis_cum_int'])
                self.solver_args['Cost_Diff'] = autoreg_returns['c_diff']; self.cost_diffs.append(self.solver_args['Cost_Diff'])
                self.solver_args['Opt_Dev'] = autoreg_returns['opt_div']; self.opt_div.append(self.solver_args['Opt_Dev'])
                self.solver_args['State_Diff'] = autoreg_returns['state_diff']; self.state_diffs.append(self.solver_args['State_Diff'])
                self.solver_args['Q_RMSE'] = autoreg_returns['q_rmse']; self.q_rmse.append(self.solver_args['Q_RMSE'])
                self.solver_args['Q_NORM'] = autoreg_returns['q_norm']; self.q_norm.append(self.solver_args['Q_NORM'])
                self.solver_args['DQ_NORM'] = autoreg_returns['dq_norm']; self.dq_norm.append(self.solver_args['DQ_NORM'])
                self.solver_args['Fcn_Val'] = autoreg_returns['res'].fun; self.fcn_vals.append(self.solver_args['Fcn_Val'])
                self.solver_args['Phi_Diff'] = autoreg_returns['phi_diff']
                self.phi_diffs_history.append(autoreg_returns['phi_diff'].copy())
                self.ws.append(autoreg_returns['new_w'])
                self.dw.append(autoreg_returns['dw'])
                self.Ps.append(self.get_Ps(self.phis_set, self.ws[-1][-1]))
                self.Ent.append(self.get_entropy(self.phis_set, self.ws[-1][-1]))
                self.norm_phis.append(self.get_norm_phi([self.phis_set_int[i] for i in nopt_inds]))

                if self.compare_desired:
                    _, cs_des = self.get_traj_costs(autoreg_returns['phis'], self.w_run_star_v, self.w_term_star_v)
                    _, cs_opt_des = self.get_traj_costs(self.phis_opt, self.w_run_star_v, self.w_term_star_v)
                    self.solver_args['Cost_Diff_Desired'] = self.get_cost_diff(cs_des, cs_opt_des); self.cost_diffs_des.append(self.solver_args['Cost_Diff_Desired'])      
            else:
                break
            terminate, info, message = self.loop_termination_check()
            if self.verbose and not terminate:
                print(info)
            if self.save_history_path is not None:
                self._dump_history()
            nopt_inds = self.get_inds()
            opt_inds = list(range(self.n_demos))
            self.solver_args['iteration'] = self.iter
        if self.verbose:
            print(self.solver_args['message'])

    def _dump_history(self):
        """Write the accumulated per-iteration IRL trajectory to an npz so the
        weight / gradient / phi-diff evolution can be plotted. Called every iter
        (overwrites) → survives a stall or kill. Fully defensive: never raises
        into the solve loop."""
        try:
            def _obj(seq):
                a = np.empty(len(seq), dtype=object)
                for i, v in enumerate(seq):
                    a[i] = np.asarray(v)
                return a
            weights = _obj([w[-1] for w in self.ws])
            grads   = _obj(self.Js)
            phidiff = _obj(self.phi_diffs_history)
            dw      = _obj(self.dw) if len(self.dw) else _obj([])
            np.savez(
                self.save_history_path,
                weights=weights, grads=grads, phi_diffs=phidiff, dw=dw,
                q_norm=np.asarray(self.q_norm, float),
                opt_div=np.asarray(self.opt_div, float),
                fcn_vals=np.asarray(self.fcn_vals, float) if hasattr(self, 'fcn_vals') else np.array([]),
                phi_demo=np.abs(np.sum(np.asarray(self.phis_opt_int), axis=0)),
                keys_run=np.asarray(self.keys_run, dtype=object),
                keys_term=np.asarray(self.keys_term, dtype=object),
                K=self.K, nr_run=self.nr_run, nr_term=self.nr_term,
                iter=self.iter,
                allow_pickle=True,
            )
        except Exception as _e:
            print('[MO_IRL._dump_history] skipped (%s)' % _e)

    def _per_demo_rollout_features(self):
        """Population per-demo pooling: read each subject model's CURRENT solved
        rollout features individually (NO averaging). Mirrors avg_features but
        keeps the per-subject lists separate, applying the same per-feature
        scaling that was applied to the stored demo features.

        Returns a list parallel to self.models: rollout_features[i] is the
        per-timestep feature list (length T+1) for subject i's current
        trajectory, tagged to demo i.
        """
        scale = getattr(self, '_feature_scale', None)
        apply_scale = scale is not None and not np.allclose(scale, 1.0)
        inv = (1.0 / scale) if apply_scale else None
        rollout_features = []
        for m in self.models:
            _, phis, _, _ = m.get_new_traj_features()
            if apply_scale:
                phis = [np.asarray(p) * inv for p in phis]
            rollout_features.append(phis)
        return rollout_features

    def _compute_per_demo_grad(self, w_prev):
        """Summed MaxEnt gradient over per-demo references (paper Eq. 4).

        For each demo d:
          Phis_d = [demo_features_d] + pool_d
        where demo_features_d = self.phis_set[d] (the per-demo, per-subject demo
        features, already per-feature-scaled by _apply_per_feature_scale) and
        pool_d is:
          - same_demo : [rollout_features[d]]            (subject d's own rollout)
          - all_demos : rollout_features[0..n_demos-1]   (all subjects' rollouts)
        Then (f_d, g_d) = log_likelihood_f(0, nopt_indices_d, ..., Phis_d) and
        f = Σ_d f_d, g = Σ_d g_d. Returns (f, g) with g a length-(nr_run*K +
        nr_term*K) vector. gradient_mask/demo_signal_mask are applied by caller.
        """
        nr_run_tv = self.nr_run * self.K
        w_run_v  = w_prev[:nr_run_tv]
        w_term_v = w_prev[nr_run_tv:]
        rollout_features = self._per_demo_rollout_features()
        nd = int(getattr(self, 'n_demos', len(self.models)))

        f_total = 0.0
        g_total = np.zeros(nr_run_tv + len(w_term_v))
        for d in range(nd):
            demo_feat = self.phis_set[d]
            if self.pool_scope == 'same_demo':
                pool = [rollout_features[d]]
            else:
                pool = list(rollout_features)
            Phis_d = [demo_feat] + pool
            nopt_indices_d = list(range(1, len(Phis_d)))
            f_d, g_d = self.log_likelihood.log_likelihood_f(
                0, nopt_indices_d, w_run_v, w_term_v, Phis_d, f_type='LJ_w')
            f_total += f_d
            g_total += g_d
        return float(f_total), g_total

    def _preserve_ref_g(self, phi):
        """Basis-projected time-integral of a demo's per-timestep features, in the
        SAME [K*nr_run, K*nr_term] layout as the weight vector w. Mirrors
        LogLikelihood.get_dldw's demo-side construction (B_step outer feature,
        dt-weighted run steps + un-weighted terminal). g_ref satisfies
        cost(w, demo) = <w, g_ref>, so <dw, g_ref>=0 preserves the demo's cost.
        Returned unit-normalized (scale-free for an equality-to-zero constraint,
        better conditioned for SLSQP)."""
        phi = np.asarray(phi)
        K = self.K
        nr = phi.shape[1]
        nr_r = self.nr_run
        nr_t = nr - nr_r
        B = self.log_likelihood.B_step
        T_snip = len(phi) - 1
        if B is None:
            g = np.zeros(nr)
            for step in range(T_snip):
                g[:nr_r] += phi[step][:nr_r] * self.dt
            g[:nr_r] += phi[-1][:nr_r]
            if nr_t > 0:
                g[nr_r:] += phi[-1][nr_r:]
            n = np.linalg.norm(g)
            return g / n if n > 1e-12 else g
        g = np.zeros(K * nr_r + K * nr_t)
        for step in range(T_snip):
            b = B[min(step, self.T)]
            c = phi[step] * self.dt
            g[:K * nr_r] += np.outer(b, c[:nr_r]).ravel()
            if nr_t > 0:
                g[K * nr_r:] += np.outer(b, c[nr_r:]).ravel()
        b_t = B[min(T_snip, self.T)]
        c_t = phi[-1]
        g[:K * nr_r] += np.outer(b_t, c_t[:nr_r]).ravel()
        if nr_t > 0:
            g[K * nr_r:] += np.outer(b_t, c_t[nr_r:]).ravel()
        n = np.linalg.norm(g)
        return g / n if n > 1e-12 else g

    def _compute_dw(self, opt_inds, nopt_inds, w_prev, dw, **kwargs):
        """Compute gradient direction — MPPI-MO-IRL (MPPI challenger + γ-scaling) or L-BFGS-B."""
        if getattr(self, '_per_demo_pool', False) and not self.use_mppi_grad:
            nd = int(getattr(self, 'n_demos', len(self.models)))
            cur = self._per_demo_rollout_features()
            if self._pool_rollout_hist is None:
                self._pool_rollout_hist = [[] for _ in range(nd)]
            for d in range(nd):
                self._pool_rollout_hist[d].append(cur[d])
                if (self.pool_last_k is not None
                        and len(self._pool_rollout_hist[d]) > self.pool_last_k):
                    self._pool_rollout_hist[d] = \
                        self._pool_rollout_hist[d][-self.pool_last_k:]

            phis_pd = [self.phis_set[d] for d in range(nd)]
            demo_roll_inds = []
            idx = nd
            for d in range(nd):
                inds_d = []
                for roll in self._pool_rollout_hist[d]:
                    phis_pd.append(roll)
                    inds_d.append(idx); idx += 1
                demo_roll_inds.append(inds_d)
            opt_inds_pd = list(range(nd))
            if self.pool_scope == 'same_demo':
                nopt_inds_pd = [list(demo_roll_inds[d]) for d in range(nd)]
            else:
                shared = [i for d in range(nd) for i in demo_roll_inds[d]]
                nopt_inds_pd = [list(shared) for _ in range(nd)]
            if self.slsqp_inner:
                self.optimizer.eq_refs = [
                    self._preserve_ref_g(self.phis_set[d]) for d in range(nd)]
            if self.args.get('verbose_irl', True):
                npool = sum(len(b) for b in demo_roll_inds)
                solver_name = 'SLSQP+preserve' if self.slsqp_inner else 'L-BFGS-B'
                print(f"  [pool_scope={self.pool_scope}] scipy {solver_name} over "
                      f"{nd} demo refs + accumulated pool ({npool} rollouts, "
                      f"last_k={self.pool_last_k})")
            return self.optimizer.min(opt_inds_pd, nopt_inds_pd, w_prev,
                                      phis_pd, dw, **kwargs)
        if self.use_mppi_grad:
            nr_run_tv = self.nr_run * self.K

            cache_hit = (hasattr(self, '_grad_w_cache')
                         and self._grad_w_cache is not None
                         and np.array_equal(self._grad_w_cache, w_prev)
                         and hasattr(self, '_grad_phis_cache')
                         and self._grad_phis_cache is not None)
            if cache_hit:
                phis_mppi = self._grad_phis_cache
                phis_each = getattr(self, '_grad_phis_each_cache', None)
                print(f"  [grad-cache] HIT — skipping MPPI rerun at unchanged w_prev")
            else:
                w_run_v  = w_prev[:nr_run_tv]
                w_term_v = w_prev[nr_run_tv:]
                if self.weight_mode == 'basis':
                    theta_run  = w_run_v.reshape(self.K, self.nr_run)
                    theta_term = w_term_v.reshape(self.K, self.nr_term) if self.nr_term > 0 else np.zeros((self.K, 0))
                    w_run_win  = self.B_window @ theta_run
                    w_term_win = self.B_window @ theta_term if self.nr_term > 0 else None
                    self.model.update_solver_weights_tv(w_run_win, w_term_win)
                elif self.n_w > 1:
                    w_run_win  = w_run_v.reshape(self.n_w, self.nr_run)
                    w_term_win = w_term_v.reshape(self.n_w, self.nr_term) if self.nr_term > 0 else None
                    self.model.update_solver_weights_tv(w_run_win, w_term_win)
                else:
                    w_run_dict = dict(zip(self.keys_run, w_run_v))
                    w_term_dict = dict(zip(self.keys_term, w_term_v)) if self.nr_term > 0 else {}
                    self.model.update_solver_weights(w_run_dict, w_term_dict)

                solver = self.model.solver
                _kin = getattr(self.model, '_kin_mppi', None)
                _U_before = (np.asarray(_kin.U).copy()
                             if _kin is not None else None)
                _do_reset = (self.args.get('mppi_reset_U_per_iter', True)
                             and hasattr(solver, 'reset'))
                print(f"  [grad-mppi] reset_enabled={self.args.get('mppi_reset_U_per_iter', True)}  "
                      f"shim_has_reset={hasattr(solver, 'reset')}  "
                      f"will_reset={_do_reset}")
                if _U_before is not None:
                    print(f"  [grad-mppi] U BEFORE: |U|={np.linalg.norm(_U_before):.4g}  "
                          f"max|U|={np.max(np.abs(_U_before)):.4g}")
                if _do_reset:
                    solver.reset()
                    if _kin is not None:
                        _U_after = np.asarray(_kin.U)
                        print(f"  [grad-mppi] U AFTER reset: |U|={np.linalg.norm(_U_after):.4g}  "
                              f"max|U|={np.max(np.abs(_U_after)):.4g}  "
                              f"(should be 0)")
                xs_init = list(self.xs_opt) if not isinstance(self.xs_opt, list) else self.xs_opt
                us_init = list(self.us_opt) if not isinstance(self.us_opt, list) else self.us_opt
                if _kin is not None and hasattr(_kin, 'w_run'):
                    _w_check = {k: float(_kin.w_run.get(k, 0.0))
                                for k in ['progress_vel', 'approach',
                                          'press_force', 'rock_ori']
                                if k in _kin.w_run}
                    print(f"  [grad-mppi] kin.w_run subset: {_w_check}")
                pool_models = (self.models if (self.models and len(self.models) > 1)
                               else [self.model])
                _pool_tv = (self.weight_mode == 'basis' or self.n_w > 1)
                phis_each = []
                for _mi, _m in enumerate(pool_models):
                    if len(pool_models) > 1:
                        if _pool_tv:
                            _m.update_solver_weights_tv(w_run_win, w_term_win)
                        else:
                            _m.update_solver_weights(
                                dict(zip(self.keys_run, w_run_v)),
                                dict(zip(self.keys_term, w_term_v)) if self.nr_term > 0 else {})
                        _msolv = _m.solver
                        if (self.args.get('mppi_reset_U_per_iter', True)
                                and hasattr(_msolv, 'reset')):
                            _msolv.reset()
                        _xs_i = list(self.Xs[_mi]) if _mi < len(self.Xs) else xs_init
                        _us_i = list(self.Us[_mi]) if _mi < len(self.Us) else us_init
                        _m.solver.solve(_xs_i, _us_i)
                    else:
                        _m.solver.solve(xs_init, us_init)
                    _, _phm, _, _ = _m.get_new_traj_features()
                    phis_each.append(_phm)
                if len(phis_each) == 1:
                    phis_mppi = phis_each[0]
                else:
                    phis_mppi = [np.mean([_pe[_t] for _pe in phis_each], axis=0)
                                 for _t in range(len(phis_each[0]))]
                self._grad_w_cache = w_prev.copy()
                self._grad_phis_cache = phis_mppi
                self._grad_phis_each_cache = phis_each

            kin_mppi = getattr(self.model, '_kin_mppi', None)
            use_scale = (kin_mppi is not None
                         and not np.allclose(np.array(kin_mppi._phi_scale), 1.0))
            if use_scale:
                scale_run = np.array(kin_mppi._phi_scale)[:self.nr_run]
                scale_full = np.ones(self.nr_run + self.nr_term)
                scale_full[:self.nr_run] = scale_run
                norm = lambda phis: [p / scale_full for p in phis]
            else:
                norm = lambda phis: phis

            use_buffer = self.args.get('use_buffer', False)
            _fs = getattr(self, '_feature_scale', None)
            _do_rs = (self.args.get('residual_scale', False) and _fs is not None
                      and not np.allclose(_fs, 1.0))
            _do_pn = (getattr(self, '_mppi_phi_norm', False) and not _do_rs
                      and _fs is not None and not np.allclose(_fs, 1.0))
            _scale_fresh = _do_rs or _do_pn
            _inv = (1.0 / np.asarray(_fs)) if _scale_fresh else None
            w_run_v  = w_prev[:nr_run_tv]
            w_term_v = w_prev[nr_run_tv:]
            _nd = int(getattr(self, 'n_demos', 1))

            if _nd > 1:
                def _scale_chal(_ph):
                    _ph = norm(_ph)
                    return [np.asarray(p) * _inv for p in _ph] if _scale_fresh else _ph
                if (getattr(self, '_pool_rollout_hist', None) is None
                        or len(self._pool_rollout_hist) != _nd):
                    self._pool_rollout_hist = [[] for _ in range(_nd)]
                _plk = getattr(self, 'pool_last_k', None)
                if not cache_hit:
                    for _d in range(_nd):
                        _pe = phis_each[_d] if _d < len(phis_each) else phis_each[-1]
                        self._pool_rollout_hist[_d].append(_scale_chal(_pe))
                        if not use_buffer:
                            self._pool_rollout_hist[_d] = self._pool_rollout_hist[_d][-1:]
                        elif _plk is not None and len(self._pool_rollout_hist[_d]) > _plk:
                            self._pool_rollout_hist[_d] = self._pool_rollout_hist[_d][-_plk:]
                _all_demos = (getattr(self, 'pool_scope', 'same_demo') == 'all_demos')
                f = 0.0
                g = np.zeros(nr_run_tv + len(w_term_v))
                Phis = None
                for _d in range(_nd):
                    _demo_d = norm(self.phis_set[_d])
                    if _all_demos:
                        _pool_d = [r for _dd in range(_nd)
                                   for r in self._pool_rollout_hist[_dd]]
                    else:
                        _pool_d = list(self._pool_rollout_hist[_d])
                    _Phis_d = [_demo_d] + _pool_d
                    _nopt_d = list(range(1, len(_Phis_d)))
                    _f_d, _g_d = self.log_likelihood.log_likelihood_f(
                        0, _nopt_d, w_run_v, w_term_v, _Phis_d, f_type='LJ_w')
                    f += _f_d
                    g = g + _g_d
                    if _d == 0:
                        Phis = [_Phis_d[0], _Phis_d[-1]]
                if self.args.get('verbose_irl', True):
                    _npool = sum(len(b) for b in self._pool_rollout_hist)
                    print(f"  [grad-mppi PER-DEMO] summed over {_nd} demos "
                          f"(pool_scope={'all_demos' if _all_demos else 'same_demo'}, "
                          f"{_npool} challengers, use_buffer={use_buffer})")
            else:
                if _scale_fresh:
                    phis_mppi = [np.asarray(p) * _inv for p in phis_mppi]
                Phis = [norm(self.phis_opt)]
                if use_buffer:
                    for ph in [self.phis_set[i] for i in nopt_inds]:
                        Phis.append(norm(ph))
                _samples = (self.model.get_sample_features()
                            if (self.args.get('sample_challengers', False)
                                and hasattr(self.model, 'get_sample_features')) else None)
                if _samples:
                    for _sp in _samples:
                        Phis.append(norm([np.asarray(p) * _inv for p in _sp] if _scale_fresh else _sp))
                    print(f"  [sample-chal] partition uses {len(_samples)} MPPI samples "
                          f"as challengers (optimum buffered)")
                else:
                    Phis.append(norm(phis_mppi))
                nopt_indices = list(range(1, len(Phis)))
                f, g = self.log_likelihood.log_likelihood_f(
                    0, nopt_indices, w_run_v, w_term_v, Phis, f_type='LJ_w')

            if self.args.get('debug_iter1', True) and self.iter == 0:
                for _name in ['press_force', 'traveled', 'progress_vel',
                              'approach', 'rail_lat']:
                    if _name not in self.keys_run:
                        continue
                    _i = list(self.keys_run).index(_name)
                    _d = np.array([np.asarray(p)[_i] for p in Phis[0]])
                    _c = np.array([np.asarray(p)[_i] for p in Phis[-1]])
                    print(f"  [DBG {_name:12s}] demo: Σ={_d.sum():8.4f} "
                          f"nz={int(np.count_nonzero(_d))}/{len(_d)} max={_d.max():7.4f} | "
                          f"chal: Σ={_c.sum():8.4f} "
                          f"nz={int(np.count_nonzero(_c))}/{len(_c)} max={_c.max():7.4f} | "
                          f"Σ(demo-chal)={(_d-_c).sum():8.4f}")

            if (self.args.get('debug_iter1', True) and self.iter == 0
                    and self.args.get('verbose_irl', True)):
                self._debug_iter1(w_prev, g, path='MPPI-grad',
                                  phi_demo=Phis[0], phi_chal=Phis[-1], f=f)

            lr = self.args.get('mppi_lr', 1.0)
            if self.l_reg in ('elastic', 'l2', 'l1') and (self.Beta > 0 or self.Lambda > 0):
                _w0 = np.asarray(self.ws[0][-1], float)
                if len(_w0) == len(g):
                    _dev = w_prev - _w0
                    g = g + 2.0 * self.Beta * _dev + self.Lambda * np.sign(_dev)
            if self.gradient_mask is not None and len(self.gradient_mask) == len(g):
                g = g * self.gradient_mask
            if (hasattr(self, 'demo_signal_mask')
                    and len(self.demo_signal_mask) == len(g)):
                g = g * self.demo_signal_mask
            if self.use_adam:
                if self._adam_m is None or self._adam_m.shape != g.shape:
                    self._adam_m = np.zeros_like(g)
                    self._adam_v = np.zeros_like(g)
                    self._adam_t = 0
                self._adam_t += 1
                b1, b2, eps = self.adam_beta1, self.adam_beta2, self.adam_eps
                self._adam_m = b1 * self._adam_m + (1.0 - b1) * g
                self._adam_v = b2 * self._adam_v + (1.0 - b2) * (g * g)
                m_hat = self._adam_m / (1.0 - b1 ** self._adam_t)
                v_hat = self._adam_v / (1.0 - b2 ** self._adam_t)
                dw_vec = -lr * m_hat / (np.sqrt(v_hat) + eps)
            elif self.outer_optimizer == 'lbfgs':
                if self._lbfgs_w_prev is not None and self._lbfgs_g_prev is not None:
                    s_new = w_prev - self._lbfgs_w_prev
                    y_new = g       - self._lbfgs_g_prev
                    if float(s_new @ y_new) > 1e-12:
                        self._lbfgs_s.append(s_new)
                        self._lbfgs_y.append(y_new)
                self._lbfgs_w_prev = w_prev.copy()
                self._lbfgs_g_prev = g.copy()

                if len(self._lbfgs_s) >= 1:
                    q = g.copy()
                    rhos   = [1.0 / (float(yi @ si) + 1e-12)
                              for si, yi in zip(self._lbfgs_s, self._lbfgs_y)]
                    alphas = []
                    for si, yi, rho in zip(reversed(self._lbfgs_s),
                                            reversed(self._lbfgs_y),
                                            reversed(rhos)):
                        a = rho * float(si @ q)
                        q = q - a * yi
                        alphas.append(a)
                    s_last, y_last = self._lbfgs_s[-1], self._lbfgs_y[-1]
                    gamma = float(s_last @ y_last) / (float(y_last @ y_last) + 1e-12)
                    gamma_cap = float(self.args.get('lbfgs_gamma_cap', 10.0))
                    gamma_used = min(max(gamma, 1e-3), gamma_cap)
                    r = gamma_used * q
                    for si, yi, rho, a in zip(self._lbfgs_s, self._lbfgs_y,
                                               rhos, reversed(alphas)):
                        b = rho * float(yi @ r)
                        r = r + (a - b) * si
                    lbfgs_lr = float(self.args.get('lbfgs_lr', 1.0))
                    dw_vec = -lbfgs_lr * r
                    print(f"  [LBFGS-outer] history={len(self._lbfgs_s)}  "
                          f"|dw|={np.linalg.norm(dw_vec):.4g}  "
                          f"max|dw|={np.max(np.abs(dw_vec)):.4g}  "
                          f"γ_raw={gamma:.4g} γ_used={gamma_used:.4g}")
                else:
                    lbfgs_lr = float(self.args.get('lbfgs_lr', 1.0))
                    dw_vec = -lbfgs_lr * g
                    print(f"  [LBFGS-outer] iter 0 — γ=1 (no history) "
                          f"|dw|={np.linalg.norm(dw_vec):.4g}")
            else:
                dw_vec = -g * lr
                gnorm = float(np.linalg.norm(g))
                dwnorm = float(np.linalg.norm(dw_vec))
                dwmax = float(np.max(np.abs(dw_vec))) if dw_vec.size else 0.0
                gmax = float(np.max(np.abs(g))) if g.size else 0.0
                print(f"  [SGD] lr={lr:.4g}  |g|={gnorm:.4g} "
                      f"max|g|={gmax:.4g}  |dw|={dwnorm:.4g} "
                      f"max|dw|={dwmax:.4g}")
            task_gain = float(self.args.get('task_feat_gain', 1.0))
            if task_gain != 1.0:
                task_feats = {'progress_vel', 'approach', 'press_force',
                              'rail_lat', 'rock_ori', 'traveled', 'surface'}
                gain = np.array([task_gain if k in task_feats else 1.0
                                 for k in self.keys_run])
                gain_run = np.tile(gain, self.K)
                gain_term = np.ones(len(dw_vec) - gain_run.size)
                full_gain = np.concatenate([gain_run, gain_term])
                task_idx = np.where(full_gain != 1.0)[0]
                eff_idx  = np.where(full_gain == 1.0)[0]
                print(f"  [task-gain] ×{task_gain:g} on {len(task_idx)} task coords | "
                      f"max|dw| task={np.max(np.abs(dw_vec[task_idx])):.4g} "
                      f"effort={np.max(np.abs(dw_vec[eff_idx])):.4g} (pre-gain)")
                dw_vec = dw_vec * full_gain

            if self.gradient_mask is not None and len(self.gradient_mask) == len(dw_vec):
                dw_vec = dw_vec * self.gradient_mask
            if (hasattr(self, 'demo_signal_mask')
                    and len(self.demo_signal_mask) == len(dw_vec)):
                dw_vec = dw_vec * self.demo_signal_mask

            class _R:
                x   = dw_vec
                fun = float(f)
                jac = dw_vec
            return _R()
        else:
            if (self.args.get('debug_iter1', True) and self.iter == 0
                    and self.args.get('verbose_irl', True)):
                nr_run_tv = self.nr_run * self.K
                Phis_dbg = [self.phis_opt] + [self.phis_set[i] for i in nopt_inds]
                _, g_dbg = self.log_likelihood.log_likelihood_f(
                    0, list(range(1, len(Phis_dbg))),
                    w_prev[:nr_run_tv], w_prev[nr_run_tv:], Phis_dbg, f_type='LJ_w')
                self._debug_iter1(w_prev, g_dbg, path='L-BFGS-B',
                                  phi_demo=Phis_dbg[0], phi_chal=Phis_dbg[-1])

            if self.slsqp_inner:
                self.optimizer.eq_refs = [self._preserve_ref_g(
                    self.phis_set[opt_inds[0]] if opt_inds else self.phis_opt)]
            res = self.optimizer.min(opt_inds, nopt_inds, w_prev, self.phis_set, dw, **kwargs)
            if (self.gradient_mask is not None
                    and len(self.gradient_mask) == len(res.x)):
                res.x = res.x * self.gradient_mask
            if self.args.get('verbose_irl', True):
                print(f"  [L-BFGS-B] raw dw: {np.round(res.x, 6)}")
                print(f"  [L-BFGS-B] |dw|={np.linalg.norm(res.x):.6g}, max={np.max(np.abs(res.x)):.6g}")

            if (self.args.get('debug_iter1', True) and self.iter == 0
                    and self.args.get('verbose_irl', True)):
                print(f"  [L-BFGS-B] nit={getattr(res, 'nit', '?')}  "
                      f"message={getattr(res, 'message', '?')}  "
                      f"fun={getattr(res, 'fun', '?')}")
                nz = np.sum(np.abs(res.x) > 1e-12)
                print(f"  [L-BFGS-B] non-zero coords in dw: {nz}/{len(res.x)}")
                if nz == 0:
                    print(f"  [L-BFGS-B] >>> dw is EXACTLY zero — L-BFGS-B terminated at x0.")
                    at_bound = (w_prev == 0)
                    wants_neg = (g_dbg > 0)
                    blocked   = at_bound & wants_neg
                    free_grad = np.where(blocked, 0.0, g_dbg)
                    print(f"  [L-BFGS-B] coords at bound w=0: {int(at_bound.sum())}/{len(w_prev)}")
                    print(f"  [L-BFGS-B] coords blocked (at bound & want w<0): {int(blocked.sum())}")
                    print(f"  [L-BFGS-B] |free gradient| (what L-BFGS-B would actually descend) = "
                          f"{np.linalg.norm(free_grad):.6g}")

            lr = self.args.get('mppi_lr', 1.0)
            if lr != 1.0:
                res.x = res.x * lr
                if self.args.get('verbose_irl', True):
                    print(f"  [L-BFGS-B] after lr={lr}: |dw|={np.linalg.norm(res.x):.6g}, max={np.max(np.abs(res.x)):.6g}")
            return res

    def _debug_iter1(self, w_prev, g, path, phi_demo=None, phi_chal=None, f=None):
        """Dump everything needed to diagnose first-iteration behaviour."""
        print(f"\n  [DEBUG iter 1 | {path}]")
        print(f"    w_prev       : {np.round(w_prev, 6)}")
        print(f"    |w_prev|     = {np.linalg.norm(w_prev):.6g}")
        print(f"    w==0 coords  : {np.where(w_prev == 0)[0].tolist()}")
        print(f"    g = ∇L(w_prev): {np.round(g, 6)}")
        print(f"    |g|          = {np.linalg.norm(g):.6g}    max|g| = {np.max(np.abs(g)):.6g}")
        print(f"    sign(g)      : {np.sign(g).astype(int).tolist()}")
        print(f"    descent -g   : {np.round(-g, 6)}")
        blocked = (w_prev == 0) & (g > 0)
        free_g  = np.where(blocked, 0.0, g)
        print(f"    blocked@bnd  : {int(blocked.sum())} coords (w=0 AND want dw<0)")
        print(f"    |proj grad|  = {np.linalg.norm(free_g):.6g}   "
              f"(this is what L-BFGS-B's pgtol is compared against)")
        if phi_demo is not None and phi_chal is not None:
            phi_d_last = phi_demo[-1] if hasattr(phi_demo, '__len__') else phi_demo
            phi_c_last = phi_chal[-1] if hasattr(phi_chal, '__len__') else phi_chal
            diff = np.asarray(phi_c_last) - np.asarray(phi_d_last)
            print(f"    φ_chal - φ_demo (terminal): {np.round(diff, 6)}")
            print(f"    |Δφ| = {np.linalg.norm(diff):.6g}")
        if f is not None:
            print(f"    f = -logP(demo|w_prev) = {f:.6g}")
        print()

    def step(self, opt_inds, nopt_inds, w_prev, solver_args):
        if self.normalize_w and len(self.ws) > 1:
            w_prev = self.normalize_vector(w_prev)
        dw = np.zeros_like(w_prev)
        if self.args['with_temp_adjust']:
            accept = False
            while not accept:
                res = self._compute_dw(opt_inds, nopt_inds, w_prev, dw, var=self.opt_vars, outer_loop_iter=self.iter, lambda_t=self.lambda_t)
                dw = res.x.copy()

                xs_init, us_init = self.initiate_trajectory()
                phis_set = [self.phis_set[i] for i in opt_inds+nopt_inds]
                w_loop, solver_args, _ = self.line_search.get_ls(
                    solver_args,
                    xs_init,
                    us_init,
                    w_prev,
                    dw,
                    self.phis_set_int[-1],
                    self.opt_div,
                    self.cost_diffs,
                    self.q_norm,
                    phis_set=phis_set,
                    dq_norm_list=self.dq_norm)
                
                if self.line_search_base == 'none':
                    new_x = np.stack(self.model.solver.xs.copy())
                    new_u = np.stack(self.model.solver.us.copy())
                    nr_run_tv = self.nr_run * self.K
                    new_w_run, new_w_term = self.vector_to_dict(w_loop[:nr_run_tv], w_loop[nr_run_tv:])
                    new_w = (w_loop[:nr_run_tv], w_loop[nr_run_tv:], w_loop)
                    phi, phis, _, phis_cum_int = self.model.get_new_traj_features()
                    w_run_v, w_term_v = w_loop[:self.nr_run * self.K], w_loop[self.nr_run * self.K:]
                    _, cs_opt = self.get_traj_costs(self.phis_opt, w_run_v, w_term_v)
                    _, cs = self.get_traj_costs(phis, w_run_v, w_term_v)
                    c_diff = self.get_cost_diff(cs, cs_opt)
                    if c_diff > self.cost_diffs[-1]:
                        self.lambda_t /= (1 + self.kappa)
                        print('Lambda: ', self.lambda_t)
                    else:
                        accept = True
                else: 
                    if solver_args['hard_terminate'] == True:
                        self.lambda_t /= (1 + self.kappa)
                        if self.K_set != 'all' and self.K_set < len(self.Xs):
                            self.K_set = self.K_set + 1
                            nopt_inds = self.get_inds()
                        print('Lambda: ', self.lambda_t)
                    else:
                        accept = True
                if self.lambda_t > self.lambda_thrs:
                    solver_args['hard_terminate'] = False
                else:
                    solver_args['hard_terminate'] = True
                    accept = True
        else:
            res = self._compute_dw(opt_inds, nopt_inds, w_prev, dw, var=self.opt_vars, outer_loop_iter=self.iter, lambda_t=1.0/self.temperature)
            dw = res.x.copy()
            xs_init, us_init = self.initiate_trajectory()
            phis_set = [self.phis_set[i] for i in opt_inds+nopt_inds]
            w_loop, solver_args, output_traj = self.line_search.get_ls(
                solver_args,
                xs_init,
                us_init,
                w_prev,
                dw,
                self.phis_set_int[-1],
                self.opt_div,
                self.cost_diffs,
                self.q_norm,
                phis_set=phis_set,
                dq_norm_list=self.dq_norm)
            
        new_x = np.stack(self.model.solver.xs.copy())
        new_u = np.stack(self.model.solver.us.copy())
        nr_run_tv = self.nr_run * self.K
        new_w_run, new_w_term = self.vector_to_dict(w_loop[:nr_run_tv], w_loop[nr_run_tv:])
        new_w = (w_loop[:nr_run_tv], w_loop[nr_run_tv:], w_loop)
        if self.models:
            phi, phis, _, phis_cum_int = avg_features(self.models)
        else:
            phi, phis, _, phis_cum_int = self.model.get_new_traj_features()
        # path (use_mppi_grad=False) never reads this cache (read is gated on
        if self.use_mppi_grad:
            self._grad_w_cache = None
            self._grad_phis_cache = None
        else:
            self._grad_w_cache = w_loop.copy()
            self._grad_phis_cache = phis
        w_run_v, w_term_v = w_loop[:self.nr_run * self.K], w_loop[self.nr_run * self.K:]
        _, cs_opt = self.get_traj_costs(self.phis_opt, w_run_v, w_term_v)
        _, cs = self.get_traj_costs(phis, w_run_v, w_term_v)
        c_diff = self.get_cost_diff(cs, cs_opt)
        opt_div = self.get_opt_div(phis_cum_int, self.phis_opt_int)

        if self.models:
            q_rmses, q_norms, dq_norms = [], [], []
            for i, m in enumerate(self.models):
                if self.args.get('ls_cold_start', False) and hasattr(m, '_ik_warmstart'):
                    _ls = getattr(self, 'line_search', None)
                    _cache = getattr(_ls, '_ik_warm_cache', None)
                    if _cache is None:
                        if not hasattr(self, '_step_ik_cache'):
                            self._step_ik_cache = {}
                        _cache = self._step_ik_cache
                    if i not in _cache:
                        _cache[i] = (m._neutral_warmstart()
                                     if getattr(m, '_warmstart_neutral', False)
                                     else m._ik_warmstart())
                    _xw, _uw = _cache[i]
                    m.solver.solve([np.asarray(x).copy() for x in _xw],
                                   [np.asarray(u).copy() for u in _uw],
                                   int(self.args.get('sqp_iter', 50)))
                xi = np.stack(m.solver.xs.copy())
                Li = min(len(self.Xs[i]), len(xi))
                q_rmses.append(self.get_q_rmse(self.Xs[i][:Li], xi[:Li]))
                q_norms.append(float(self.get_q_norm(self.Xs[i][:Li], xi[:Li])))
                dq_norms.append(float(self.get_dq_norm(self.Xs[i][:Li], xi[:Li])))
            q_rmse = np.mean(np.stack(q_rmses, axis=0), axis=0)
            q_norm = float(np.mean(q_norms))
            dq_norm = float(np.mean(dq_norms))
            x_diff = self.get_state_diff(new_x, self.Xs[0])
        else:
            L = min(len(self.Xs[0]), len(new_x))
            q_rmse = self.get_q_rmse(self.Xs[0][:L], new_x[:L])
            q_norm = float(self.get_q_norm(self.Xs[0][:L], new_x[:L]))
            dq_norm = float(self.get_dq_norm(self.Xs[0][:L], new_x[:L]))
            x_diff = self.get_state_diff(new_x, self.Xs[0])

        phi_diff = np.abs(phis_cum_int[0] - self.phis_opt_int[0])
        self.Ps_IRL.append(self.get_Ps([self.phis_set[i] for i in opt_inds+nopt_inds], new_w[-1]))
        self.Ent_IRL.append(self.get_entropy([self.phis_set[i] for i in opt_inds+nopt_inds], new_w[-1]))
        return_args = {
            'w_loop': w_loop,
            'dw': dw,
            'res': res,
            'new_x': new_x,
            'new_u': new_u,
            'new_w': new_w,
            'new_w_run': new_w_run,
            'new_w_term': new_w_term,
            'phi': phi,
            'phis': phis,
            'phis_cum_int': phis_cum_int,
            'c_diff': c_diff,
            'opt_div': opt_div,
            'state_diff': x_diff,
            'q_rmse': q_rmse,
            'q_norm': q_norm,
            'dq_norm': dq_norm,
            'jac': res.jac,
            'phi_diff': phi_diff
        }
        return return_args

    def _build_solvers(self):
        """(Re)build LogLikelihood, Optimizer, LineSearch from current self.K, B_step, B_window."""
        ll_args = {
            'T': self.T,
            'dt': self.dt,
            'temperature': self.temperature,
            'scaled_sum': self.scaled_sum,
            'traj_cost': self.get_traj_costs,
            'N_samples': self.N_samples,
            'l_type': self.l_type,
            'get_norm_phis': self.get_norm_phi,
            'normalize_phi': self.normalize_phis,
            'norm_thrs': self.norm_thrs,
            'gradient_mask': self.gradient_mask,
            'n_w': self.n_w,
            'nr': self.nr,
            'weight_mode': self.weight_mode,
            'K': self.K,
            'B_step': self.B_step,
            'nr_run_per_feature': self.nr_run,
            'nr_term_per_feature': self.nr_term,
            'partition_free_only': self.args.get('partition_free_only', False),
        }
        self.log_likelihood = LogLikelihood(ll_args)

        opt_args = {
            'irl_type': 'MO_IRL',
            'type': 'SLSQP' if self.slsqp_inner else 'L-BFGS-B',
            'slsqp_inner': self.slsqp_inner,
            'T': self.T,
            'dt': self.dt,
            'normalize_w': self.normalize_w,
            'Lambda': self.Lambda,
            'Beta': self.Beta,
            'dyn_reg': self.dyn_reg,
            'use_jac': self.use_jac,
            'use_hess': self.use_hess,
            'l_reg': self.l_reg,
            'iter': self.irl_iter,
            'verbose': self.verbose,
            'nr_run': self.nr_run * self.K,
            'nr_term': self.nr_term * self.K,
            'N_samples': self.N_samples,
            'll_fcn': self.log_likelihood.log_likelihood_f,
            'gradient_mask': self.gradient_mask,
        }
        self.optimizer = Optimizer(opt_args)

        ls_args = {
            'steps': self.steps,
            'type': self.line_search_base,
            'solver_args': {},
            'model': self.model,
            'models': self.models,
            'normalize_w': self.normalize_w,
            'normalize_vector': self.normalize_vector,
            'nr_run': self.nr_run * self.K,
            'nr_term': self.nr_term * self.K,
            'vector_to_dict': self.vector_to_dict,
            'dict_to_vector': self.dict_to_vector,
            'get_q_norm': self.get_q_norm,
            'get_dq_norm': self.get_dq_norm,
            'use_dq_norm': self.use_dq_norm,
            'dq_norm_tol': self.dq_norm_tol,
            'get_accel_mag': self.get_accel_mag,
            'use_accel_guard': self.use_accel_guard,
            'accel_tol': self.accel_tol,
            'get_opt_div': self.get_opt_div,
            'get_state_diff': self.get_state_diff,
            'get_traj_costs': self.get_traj_costs,
            'get_cost_diff': self.get_cost_diff,
            'get_entropy': self.get_entropy,
            'phis_opt': self.phis_opt,
            'phis_opt_int': self.phis_opt_int,
            'Xs_opt_list': self.Xs[:self.n_demos],
            'Us_opt_list': self.Us[:self.n_demos],
            'sqp_iter': self.sqp_iter,
            'n_w': self.n_w,
            'w_bounds': self.args.get('w_bounds', None),
            'weight_mode': self.weight_mode,
            'K': self.K,
            'B_window': self.B_window,
            'n_w_solver': self.n_w,
            'nr_run_per_feature': self.nr_run,
            'nr_term_per_feature': self.nr_term,
            'ls_cold_start': self.args.get('ls_cold_start', False),
            'challenger_press_warmstart': self.args.get('challenger_press_warmstart', False),
            'tau_norm_accept': self.args.get('tau_norm_accept', False),
            'pareto_qnorm_cap': self.args.get('pareto_qnorm_cap', float('inf')),
            'pc_col': (list(self.keys_run).index('press_capacity')
                       if 'press_capacity' in self.keys_run else None),
            'pf_col': (list(self.keys_run).index('press_force')
                       if 'press_force' in self.keys_run else None),
            'challenger_continuation': self.args.get('challenger_continuation', False),
            'cont_fracs': self.args.get('cont_fracs', (0.3, 0.6, 1.0)),
        }
        self.line_search = LineSearch(ls_args)

    def _compute_residual_signal(self):
        """Per-timestep mismatch signal (T+1, nr) at the current theta — used to pick where to add a basis."""
        theta_flat = self.ws[-1][-1]
        K_run = self.K * self.nr_run
        theta_run, theta_term = theta_flat[:K_run], theta_flat[K_run:]
        nopt_inds = self.get_inds()
        Phis = [self.phis_opt] + [self.phis_set[i] for i in nopt_inds]
        return self.log_likelihood.get_per_timestep_g(theta_run, theta_term, Phis, t_offset=0)

    def _extend_gradient_mask(self, K_new):
        """Re-tile the (separated-layout) gradient_mask after K changes."""
        if self.gradient_mask is None:
            return
        K_old = self.K
        nr_r, nr_t = self.nr_run, self.nr_term
        if len(self.gradient_mask) != K_old * (nr_r + nr_t):
            return
        mask_run_per_feat  = self.gradient_mask[:nr_r]
        mask_term_per_feat = self.gradient_mask[K_old * nr_r : K_old * nr_r + nr_t]
        self.gradient_mask = np.concatenate([
            np.tile(mask_run_per_feat, K_new),
            np.tile(mask_term_per_feat, K_new),
        ])

    def _insert_basis_center(self, t_star, sigma=None):
        """Augment B_step / B_window with a Gaussian centered at t_star, warm-start theta_k = W_old(t_star), rebuild solvers."""
        if self.weight_mode != 'basis':
            raise ValueError("_insert_basis_center requires weight_mode='basis'")
        if sigma is None:
            sigma = self.basis_sigma if self.basis_sigma is not None else max(self.T / max(self.K, 1), 1.0)

        t_step = np.arange(self.T + 1)
        new_col = np.exp(-((t_step - t_star) ** 2) / (2.0 * sigma * sigma))

        K_run = self.K * self.nr_run
        theta_flat = self.ws[-1][-1]
        theta_run_old  = theta_flat[:K_run].reshape(self.K, self.nr_run)
        theta_term_old = theta_flat[K_run:].reshape(self.K, self.nr_term) if self.nr_term > 0 else np.zeros((self.K, 0))
        W_run_at_tstar  = self.B_step[t_star] @ theta_run_old
        W_term_at_tstar = self.B_step[t_star] @ theta_term_old

        B_step_aug = np.concatenate([self.B_step, new_col[:, None]], axis=1)
        B_step_aug /= B_step_aug.sum(axis=1, keepdims=True) + 1e-12
        window_size = max(1, self.T // self.n_w)
        new_window_col = np.zeros(self.n_w)
        for k in range(self.n_w):
            lo = k * window_size
            hi = (k + 1) * window_size if k < self.n_w - 1 else self.T + 1
            new_window_col[k] = new_col[lo:hi].mean()
        B_window_aug = np.concatenate([self.B_window, new_window_col[:, None]], axis=1)
        B_window_aug /= B_window_aug.sum(axis=1, keepdims=True) + 1e-12

        theta_run_new  = np.concatenate([theta_run_old,  W_run_at_tstar[None, :]],  axis=0)
        theta_term_new = np.concatenate([theta_term_old, W_term_at_tstar[None, :]], axis=0)
        theta_run_new  = np.maximum(theta_run_new,  0.0)
        theta_term_new = np.maximum(theta_term_new, 0.0)
        new_flat = np.concatenate([theta_run_new.ravel(), theta_term_new.ravel()])

        self._extend_gradient_mask(self.K + 1)

        self.K += 1
        self.B_step = B_step_aug
        self.B_window = B_window_aug
        self.ws.append((theta_run_new.ravel(), theta_term_new.ravel(), new_flat))

        self._adam_m = None
        self._adam_v = None
        self._adam_t = 0

        self._build_solvers()

        return new_flat

    def solve_with_refinement(self, max_refinements=8, residual_tol=1e-3,
                              ll_gain_tol=1e-3, sigma_new=None, verbose=True):
        """Hierarchical basis refinement: alternate inner solve with greedy basis insertion."""
        if self.weight_mode != 'basis':
            raise ValueError("solve_with_refinement requires weight_mode='basis'")

        history = []
        for ref_iter in range(max_refinements):
            if verbose:
                print(f"\n=== Refinement iter {ref_iter} | K={self.K} ===")

            self.iter = 0
            self.solver_args['iteration'] = 0
            self.solver_args['hard_terminate'] = False
            self.solver_args['message'] = ''

            self.solve()

            last_fcn = self.fcn_vals[-1] if self.fcn_vals else float('inf')
            history.append({'ref_iter': ref_iter, 'K': self.K, 'fcn_val': float(last_fcn)})

            if ref_iter == max_refinements - 1:
                if verbose: print("Max refinements reached.")
                break

            residual = self._compute_residual_signal()
            residual_norm = np.linalg.norm(residual, axis=1)
            max_res = float(np.max(residual_norm))
            if verbose:
                print(f"max ||residual(t)||: {max_res:.4e}  (tol {residual_tol})")
            if max_res < residual_tol:
                if verbose: print("Residual below tolerance — stopping.")
                break

            if len(history) >= 2:
                gain = history[-2]['fcn_val'] - history[-1]['fcn_val']
                if verbose: print(f"LL gain since previous K: {gain:.4e}  (tol {ll_gain_tol})")
                if gain < ll_gain_tol:
                    if verbose: print("Marginal LL gain below tolerance — stopping.")
                    break

            t_star = int(np.argmax(residual_norm))
            if verbose: print(f"Inserting basis center at t*={t_star}")
            self._insert_basis_center(t_star, sigma=sigma_new)

        return history

    
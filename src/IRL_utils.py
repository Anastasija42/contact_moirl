"""Support objects for the IRL solvers.

The log-likelihood, the line search and its acceptance rules, feature averaging
across demonstrations, and the argument plumbing shared by the drivers.
"""
import numpy as np
import os
from scipy.optimize import Bounds
from itertools import combinations

class LogLikelihood():
    def __init__(self, args):
        self.gradient_mask = args['gradient_mask']
        self.T = args['T']
        self.n_w = args.get('n_w', 1)
        self.temperature = args['temperature']
        self.traj_cost = args['traj_cost']
        self.N_samples = args['N_samples']
        self.l_type = args['l_type']
        self.dt = args['dt']
        self.scaled_sum = args['scaled_sum']
        self.normalize_phi = args['normalize_phi']
        self.get_norm_phis = args['get_norm_phis']
        self.norm_thrs = args['norm_thrs']
        self.norm_phi = 1.0
        self.omega = np.zeros(self.T + 1)
        for i in range(self.T + 1):
            self.omega[i] = self.T + 1 - i
        self.omega /= np.sum(self.omega)
        self.n_w = args.get('n_w', 1)
        self.nr  = args.get('nr', None)
        self.weight_mode = args.get('weight_mode', 'single')
        self.K = args.get('K', max(1, self.n_w))
        self.B_step = args.get('B_step', None)
        self.nr_run_per_feature = args.get('nr_run_per_feature', None)
        self.nr_term_per_feature = args.get('nr_term_per_feature', None)
        self.partition_free_only = args.get('partition_free_only', False)

    def get_ti(self, phi_1, phi_2):
        nr = len(phi_1[0])
        ti = np.zeros(nr)
        for p1, p2 in zip(phi_1[:-1], phi_2[:-1]):
            ti += (p1 - p2)*self.dt
        ti += phi_1[-1] - phi_2[-1]
        return ti
    
    def get_likelihood(self, ci, t):
        if self.l_type == 1:
            val = np.exp(-t*ci)
        if self.l_type == 2:
            val = -np.exp(-t*ci)*ci
        return np.max([-1e+300, np.min([1e+300, val])])
    
    def get_dldw(self, w_run, w_term, Phis, t_offset=0):
        n_nopt = len(Phis) - 1
        phi_opt = Phis[0]
        t = self.temperature

        if self.weight_mode == 'basis':
            nr = len(phi_opt[0])
            nr_r = self.nr_run_per_feature
            nr_t = nr - nr_r
            T_snippet = len(phi_opt) - 1
            K = self.K
            dldw = np.zeros((n_nopt, K * nr))

            for i in range(n_nopt):
                phi_n = Phis[i + 1]
                for step in range(min(T_snippet, len(phi_opt) - 1)):
                    diff = -t * (phi_n[step] - phi_opt[step]) * self.dt
                    abs_t = min(t_offset + step, self.T)
                    b = self.B_step[abs_t]
                    dldw[i, :K * nr_r] += np.outer(b, diff[:nr_r]).ravel()
                    if nr_t > 0:
                        dldw[i, K * nr_r:] += np.outer(b, diff[nr_r:]).ravel()
                diff_term = -t * (phi_n[-1] - phi_opt[-1])
                abs_t = min(t_offset + T_snippet, self.T)
                b_term = self.B_step[abs_t]
                dldw[i, :K * nr_r] += np.outer(b_term, diff_term[:nr_r]).ravel()
                if nr_t > 0:
                    dldw[i, K * nr_r:] += np.outer(b_term, diff_term[nr_r:]).ravel()

            dldw /= self.norm_phi
            return dldw

        if self.n_w > 1:
            nr = len(phi_opt[0])
            nr_r = self.nr_run_per_feature if self.nr_run_per_feature is not None else nr
            nr_t = nr - nr_r
            T_snippet = len(phi_opt) - 1
            n_w = self.n_w
            full_T = self.T
            window_size = max(1, full_T // n_w)
            dldw = np.zeros((n_nopt, n_w * nr))
            term_base = n_w * nr_r

            for i in range(n_nopt):
                phi_n = Phis[i + 1]
                for step in range(min(T_snippet, len(phi_opt) - 1)):
                    k = min((t_offset + step) // window_size, n_w - 1)
                    diff = -t * (phi_n[step] - phi_opt[step]) * self.dt
                    dldw[i, k*nr_r:(k+1)*nr_r] += diff[:nr_r]
                    if nr_t > 0:
                        b = term_base + k * nr_t
                        dldw[i, b:b + nr_t] += diff[nr_r:]
                k_term = min((t_offset + T_snippet) // window_size, n_w - 1)
                diff_term = -t * (phi_n[-1] - phi_opt[-1])
                dldw[i, k_term*nr_r:(k_term+1)*nr_r] += diff_term[:nr_r]
                if nr_t > 0:
                    b = term_base + k_term * nr_t
                    dldw[i, b:b + nr_t] += diff_term[nr_r:]

            dldw /= self.norm_phi
            return dldw

        dldw = np.zeros(shape=(n_nopt, len(w_run)+len(w_term)))
        for i in range(n_nopt):
            dldw[i,:] = -t * self.get_ti(Phis[i+1], phi_opt)
        dldw /= self.norm_phi
        return dldw
    
    def get_P(self, Cs):
        t = self.temperature
        log_L = -t * np.asarray(Cs)
        log_L -= log_L.max()
        L = np.exp(log_L)
        return L, L/(np.sum(L) + 1e-50)
    
    def get_Cs(self, w_run, w_term, Phis, t_offset=0):
        if (self.partition_free_only and self.gradient_mask is not None
                and len(self.gradient_mask) == len(w_run) + len(w_term)):
            nrun = len(w_run)
            w_run = np.asarray(w_run) * self.gradient_mask[:nrun]
            w_term = np.asarray(w_term) * self.gradient_mask[nrun:]
        costs_set = []
        for phis in Phis:
            costs_set.append(self.traj_cost(phis, w_run, w_term, t_offset=t_offset)[0])
        Cs = np.stack(costs_set)
        Cs /= self.norm_phi
        return Cs
    
    def get_l(self, w_run, w_term, Phis, t_offset=0):
        Cs = self.get_Cs(w_run, w_term, Phis, t_offset=t_offset)
        P = self.get_P(Cs)[1]
        if self.l_type == 1:
            return -np.log(P[0] + 1e-50)
        elif self.l_type == 2:
            return -np.log(P[0] + 1e-50)

    def get_l_g_w(self, w_run, w_term, Phis, t_offset=0):
        Cs = self.get_Cs(w_run, w_term, Phis, t_offset=t_offset)
        L, P = self.get_P(Cs); P_nopt = np.reshape(P[1:], (len(P[1:]),1))
        Gw = np.squeeze(P_nopt.T @ self.get_dldw(w_run, w_term, Phis, t_offset=t_offset))
        return -np.log(P[0] + 1e-50), Gw

    def get_per_timestep_g(self, w_run, w_term, Phis, t_offset=0):
        """Per-timestep gradient signal aggregated over non-optimal trajectories.
        Returns shape (T_snippet+1, nr): row t = sum_i P_nopt[i] * (-temp) * (phi_n_i[t] - phi_opt[t]) * (dt for running, 1 for terminal).
        """
        Cs = self.get_Cs(w_run, w_term, Phis, t_offset=t_offset)
        _, P = self.get_P(Cs)
        P_nopt = P[1:]
        n_nopt = len(Phis) - 1
        phi_opt = Phis[0]
        nr = len(phi_opt[0])
        T_snippet = len(phi_opt) - 1
        t = self.temperature
        g = np.zeros((T_snippet + 1, nr))
        for i in range(n_nopt):
            phi_n = Phis[i + 1]
            for step in range(min(T_snippet, len(phi_opt) - 1)):
                g[step] += P_nopt[i] * (-t * (phi_n[step] - phi_opt[step]) * self.dt)
            g[T_snippet] += P_nopt[i] * (-t * (phi_n[-1] - phi_opt[-1]))
        g /= self.norm_phi
        return g

    def get_l_bretl(self, w_run, w_term, Phis, t_offset=0):
        Cs = self.get_Cs(w_run, w_term, Phis, t_offset=t_offset)
        P = self.get_P(Cs)[1]
        return -P[0]

    def get_l_g_bretl(self, w_run, w_term, Phis, t_offset=0):
        Cs = self.get_Cs(w_run, w_term, Phis, t_offset=t_offset)
        L, P = self.get_P(Cs); P_nopt = np.reshape(P[1:], (len(P[1:]),1))
        Z = np.sum(L)
        Gw = np.squeeze(P_nopt.T @ self.get_dldw(w_run, w_term, Phis, t_offset=t_offset))
        Gw = Gw * (-1/Z)
        return -P[0], Gw
    
    def log_likelihood_f(self, opt_ind, nopt_ind, w_run, w_term, Phis, f_type = 'L_w', temperature = None):
        if temperature is not None:
            self.temperature = temperature

        opt_inds = opt_ind if isinstance(opt_ind, list) else [opt_ind]

        samples = np.linspace(0, self.T-1, self.N_samples).astype(int)
        self.omega2 = np.zeros(self.N_samples)
        for i, s in enumerate(samples):
            self.omega2[i] = self.T + 1 - s
        self.omega2 /= (self.T + 1)

        if f_type == 'L_w':
            f = 0.0
            ll_f = self.get_l
        elif f_type == 'LJ_w':
            f = 0.0
            g = np.zeros(len(w_run)+len(w_term))
            ll_f = self.get_l_g_w

        per_opt_pools = (len(nopt_ind) == len(opt_inds) and len(nopt_ind) > 0
                         and all(isinstance(p, (list, tuple, np.ndarray))
                                 for p in nopt_ind))

        if self.normalize_phi:
            flat_nopt = ([i for p in nopt_ind for i in p] if per_opt_pools
                         else list(nopt_ind))
            all_inds = opt_inds + flat_nopt
            self.norm_phi = np.min([self.norm_thrs, np.mean(self.get_norm_phis([Phis[i] for i in all_inds[1:]]))])

        for _k, o_ind in enumerate(opt_inds):
            pool = list(nopt_ind[_k]) if per_opt_pools else list(nopt_ind)
            inds = [o_ind] + pool
            if f_type in ['L_w', 'L_w_b']:
                for i, o in zip(samples, self.omega2):
                    phis_set = [Phis[j][i:] for j in inds]
                    f += o*ll_f(w_run, w_term, phis_set, t_offset=int(i))
            elif f_type in ['LJ_w', 'LJ_w_b']:
                for i, o in zip(samples, self.omega2):
                    phis_set = [Phis[j][i:] for j in inds]
                    Z, J = ll_f(w_run, w_term, phis_set, t_offset=int(i))
                    if self.scaled_sum:
                        f += o*Z
                        g += o*J
                    else:
                        f += Z
                        g += J

        if f_type in ['L_w', 'L_w_b']:
            return f
        elif f_type in ['LJ_w', 'LJ_w_b']:
            if self.gradient_mask is not None:
                if len(self.gradient_mask) == len(g):
                    g = g * self.gradient_mask
                else:
                    print(f"Warning: Gradient mask len {len(self.gradient_mask)} != grad len {len(g)}")
            return f, g

def solve_all_avg(models, xinits, uinits, sqp_iter):
    """Population-IRL rollout: solve each subject model (weights already pushed)
    on its own warmstart, then average the per-timestep features across subjects
    (subjects must share a common horizon T). Returns
    (phi_mean, phis_mean, phis_cum_mean, phis_cum_int_mean, xs_list)."""
    phi_all, phis_all, phis_cum_all, phis_cum_int_all, xs_list = [], [], [], [], []
    for m, xi, ui in zip(models, xinits, uinits):
        m.solver.solve(list(xi), list(ui), sqp_iter)
        phi, phis, phis_cum, phis_cum_int = m.get_new_traj_features()
        phi_all.append(np.asarray(phi, dtype=float))
        phis_all.append(np.asarray(phis, dtype=float))
        phis_cum_all.append(np.asarray(phis_cum, dtype=float))
        phis_cum_int_all.append(np.asarray(phis_cum_int, dtype=float))
        xs_list.append(np.stack(m.solver.xs.copy()))
    mean = lambda L: np.mean(np.stack(L, axis=0), axis=0)
    return (mean(phi_all), mean(phis_all), mean(phis_cum_all),
            mean(phis_cum_int_all), xs_list)

def avg_features(models):
    """Average per-timestep features over subjects from their CURRENT (already
    solved) solver states — no re-solve. Returns
    (phi_mean, phis_mean, phis_cum_mean, phis_cum_int_mean)."""
    phi_all, phis_all, phis_cum_all, phis_cum_int_all = [], [], [], []
    for m in models:
        phi, phis, phis_cum, phis_cum_int = m.get_new_traj_features()
        phi_all.append(np.asarray(phi, dtype=float))
        phis_all.append(np.asarray(phis, dtype=float))
        phis_cum_all.append(np.asarray(phis_cum, dtype=float))
        phis_cum_int_all.append(np.asarray(phis_cum_int, dtype=float))
    mean = lambda L: np.mean(np.stack(L, axis=0), axis=0)
    return mean(phi_all), mean(phis_all), mean(phis_cum_all), mean(phis_cum_int_all)

class LineSearch():
    def __init__(self, args):
        self.steps = args['steps']
        self.type = args['type']
        self.solver_args = args['solver_args']
        self.model = args['model']
        self.models = args.get('models', None) or [self.model]
        self.normalize_vector = args['normalize_vector']
        self.normalize_w = args['normalize_w']
        self.nr_run = args['nr_run']
        self.nr_term = args['nr_term']
        self.vector_to_dict = args['vector_to_dict']
        self.dict_to_vector = args['dict_to_vector']
        self.get_q_norm  = args['get_q_norm']
        self.get_dq_norm = args.get('get_dq_norm', None)
        self.use_dq_norm = args.get('use_dq_norm', False)
        self.dq_norm_tol = float(args.get('dq_norm_tol', 2.0))
        self.get_accel_mag = args.get('get_accel_mag', None)
        self.use_accel_guard = args.get('use_accel_guard', False)
        self.accel_tol = float(args.get('accel_tol', 3.0))
        self._accel_ref = None
        self.get_opt_div = args['get_opt_div']
        self.get_traj_costs = args['get_traj_costs']
        self.get_cost_diff = args['get_cost_diff']
        self.get_state_diff = args['get_state_diff']
        self.get_entropy = args['get_entropy']
        self.phis_opt = args['phis_opt']
        self.phis_opt_int = args['phis_opt_int']
        self.Xs_opt_list = args['Xs_opt_list']
        self.Us_opt_list = args.get('Us_opt_list', None)
        self.challenger_press = bool(args.get('challenger_press_warmstart', False))
        self.tau_norm_accept = bool(args.get('tau_norm_accept', False))
        self.pc_col = args.get('pc_col', None)
        self.pf_col = args.get('pf_col', None)
        self.challenger_continuation = bool(args.get('challenger_continuation', False))
        self.cont_fracs = tuple(args.get('cont_fracs', (0.3, 0.6, 1.0)))
        self.sqp_iter = args['sqp_iter']
        # must resolve the arm redundancy (e.g. the ~25 deg elbow null-space) from
        self.ls_cold_start = bool(args.get('ls_cold_start', False))
        self._ik_warm_cache = {}
        self.n_w = args.get('n_w', 1)
        self.w_bounds = args.get('w_bounds', None)
        self.weight_mode = args.get('weight_mode', 'single')
        self.K = args.get('K', max(1, self.n_w))
        self.B_window = args.get('B_window', None)
        self.n_w_solver = args.get('n_w_solver', self.n_w)
        self.nr_run_per_feature = args.get('nr_run_per_feature', None)
        self.nr_term_per_feature = args.get('nr_term_per_feature', None)
        self._filter_history = None
        self._pareto_margin = float(args.get('pareto_margin', 1e-3))
        self._pareto_qnorm_cap = float(args.get('pareto_qnorm_cap', float('inf')))

    def _apply_bounds(self, w):
        w = np.maximum(0.0, w)
        if self.w_bounds is not None:
            w = np.minimum(self.w_bounds, w)
        return w

    def _push_weights_to_one(self, model, w_loop, pc_scale=1.0):
        """Map flat θ/W vector to ONE model's solver per-window weights.
        pc_scale<1 damps the press_cap column (continuation stage)."""
        if self.weight_mode == 'basis':
            K = self.K
            nr_r = self.nr_run_per_feature
            nr_t = self.nr_term_per_feature
            theta_run  = w_loop[:K * nr_r].reshape(K, nr_r)
            theta_term = w_loop[K * nr_r:].reshape(K, nr_t) if nr_t > 0 else np.zeros((K, 0))
            w_run_win  = self.B_window @ theta_run
            if pc_scale != 1.0:
                w_run_win = w_run_win.copy()
                for _c in (self.pc_col, self.pf_col):
                    if _c is not None and _c < w_run_win.shape[1]:
                        w_run_win[:, _c] *= float(pc_scale)
            w_term_win = self.B_window @ theta_term if nr_t > 0 else None
            model.update_solver_weights_tv(w_run_win, w_term_win)
            return

        if self.n_w > 1:
            nr_r = self.nr_run // self.n_w
            nr_t = self.nr_term // self.n_w
            model.update_solver_weights_tv(
                w_loop[:self.nr_run].reshape(self.n_w, nr_r),
                w_loop[self.nr_run:].reshape(self.n_w, nr_t))
        else:
            w_run_temp, w_term_temp = self.vector_to_dict(
                w_loop[:self.nr_run], w_loop[self.nr_run:])
            model.update_solver_weights(w_run_temp, w_term_temp)

    def _push_weights_to_solver(self, w_loop, pc_scale=1.0):
        """Push the shared weight vector to ALL subject models (one in the
        single-model case). pc_scale<1 damps the press_cap column (continuation)."""
        for m in self.models:
            self._push_weights_to_one(m, w_loop, pc_scale)
    
    def get_ti(self, phi_1, phi_2):
        nr = len(phi_1[0])
        ti = np.zeros(nr)
        for p1, p2 in zip(phi_1[:-1], phi_2[:-1]):
            ti += (p1 - p2)*self.model.dt
        ti += phi_1[-1] - phi_2[-1]
        return ti

    def _evaluate_at_alpha(self, step, w_temp, dw, xs_init, us_init):
        """Run one solver rollout at trial weights w = w_temp + step*dw and
        return the artifacts every LS-condition branch needs.

        Bundles the four things that every probe of the line search does:
        bound the weights, push them into the solver, solve, and extract
        trajectory features. Diagnostics (q_norm, opt_div) are computed here
        too so the print line and the condition branches share the same
        values. Keeping this as a single method is what lets Phase 2 fan
        these probes out across worker processes.
        """
        w_curr = self._apply_bounds(w_temp + step * dw)
        self._push_weights_to_solver(w_curr)

        if len(self.models) > 1:
            # or a cold rail-IK guess (--ls_cold_start, honest: the cost must earn
            if self.ls_cold_start:
                xinits, uinits = [], []
                for i, m in enumerate(self.models):
                    if i not in self._ik_warm_cache:
                        self._ik_warm_cache[i] = (m._neutral_warmstart()
                                                  if getattr(m, '_warmstart_neutral', False)
                                                  else m._ik_warmstart())
                    xw, uw = self._ik_warm_cache[i]
                    xinits.append([np.asarray(x).copy() for x in xw])
                    uinits.append([np.asarray(u).copy() for u in uw])
            else:
                xinits = [list(self.Xs_opt_list[i]) for i in range(len(self.models))]
                if self.challenger_press and self.Us_opt_list is not None:
                    uinits = [[np.asarray(u).copy() for u in self.Us_opt_list[i]]
                              for i in range(len(self.models))]
                else:
                    uinits = [[np.zeros(m.nu) for _ in range(len(xinits[i]) - 1)]
                              for i, m in enumerate(self.models)]
            if self.challenger_continuation and self.pc_col is not None:
                _xi, _ui = xinits, uinits
                for _frac in self.cont_fracs:
                    self._push_weights_to_solver(w_curr, pc_scale=_frac)
                    phi, phis, phis_cum, phis_cum_int, xs_list = solve_all_avg(
                        self.models, _xi, _ui, self.sqp_iter)
                    _xi = xs_list
                    _ui = [[np.asarray(u).copy() for u in m.solver.us] for m in self.models]
            else:
                phi, phis, phis_cum, phis_cum_int, xs_list = solve_all_avg(
                    self.models, xinits, uinits, self.sqp_iter)
            q_norm = float(np.mean([
                self.get_q_norm(np.asarray(self.Xs_opt_list[i]), xs_list[i])
                for i in range(len(self.models))]))
            dq_norm = (float(np.mean([
                self.get_dq_norm(np.asarray(self.Xs_opt_list[i]), xs_list[i])
                for i in range(len(self.models))]))
                if self.get_dq_norm is not None else None)
            accel_mag = (float(np.mean([self.get_accel_mag(xs_list[i])
                                        for i in range(len(self.models))]))
                         if self.get_accel_mag is not None else None)
            xs_curr = xs_list[0]
        else:
            if self.challenger_continuation and self.pc_col is not None:
                _xs, _us = list(xs_init), list(us_init)
                for _frac in self.cont_fracs:
                    self._push_weights_to_solver(w_curr, pc_scale=_frac)
                    self.model.solver.solve(list(_xs), list(_us), self.sqp_iter)
                    _xs = [np.asarray(x).copy() for x in self.model.solver.xs]
                    _us = [np.asarray(u).copy() for u in self.model.solver.us]
            else:
                self.model.solver.solve(xs_init, us_init, self.sqp_iter)
            phi, phis, phis_cum, phis_cum_int = self.model.get_new_traj_features()
            xs_curr = np.stack(self.model.solver.xs.copy())
            q_norms = [self.get_q_norm(xs_d, xs_curr) for xs_d in self.Xs_opt_list]
            q_norm = float(np.mean(q_norms))
            dq_norm = (float(np.mean([self.get_dq_norm(xs_d, xs_curr)
                                      for xs_d in self.Xs_opt_list]))
                       if self.get_dq_norm is not None else None)
            accel_mag = (self.get_accel_mag(xs_curr)
                         if self.get_accel_mag is not None else None)
        opt_div = self.get_opt_div(phis_cum_int, self.phis_opt_int)

        tau_norm = None
        if self.Us_opt_list is not None:
            _ms = self.models if len(self.models) > 1 else [self.model]
            _tn = []
            for i, m in enumerate(_ms):
                _ur = np.stack([np.asarray(u, float) for u in m.solver.us])
                _ud = np.asarray(self.Us_opt_list[i], float)
                _L = min(len(_ur), len(_ud)); _nu = min(_ur.shape[1], _ud.shape[1])
                _tn.append(float(np.sqrt(((_ur[:_L, :_nu] - _ud[:_L, :_nu]) ** 2).mean())))
            tau_norm = float(np.mean(_tn))

        return {
            "step": step,
            "w_curr": w_curr,
            "phi": phi,
            "phis": phis,
            "phis_cum": phis_cum,
            "phis_cum_int": phis_cum_int,
            "xs": xs_curr,
            "q_norm": q_norm,
            "dq_norm": dq_norm,
            "accel_mag": accel_mag,
            "opt_div": opt_div,
            "tau_norm": tau_norm,
        }
    
    def get_ls(self, solver_args, xs_init, us_init, w_prev, dw, prev_phis_int, opt_div_list, cost_diff_list, q_norm_list, phis_set, dq_norm_list=None):
        solver_args['hard_terminate'] = False
        output_traj = {}
        w_temp = w_prev.copy()
        chosen_step = False
        step_ind = 0
        prev_entropy = self.get_entropy(phis_set, w_prev)
        c1 = 0.0001
        c2 = 0.9

        # base is a lucky-low draw the noisy probes can never beat -- reseeding
        base_q_norm  = q_norm_list[-1]
        base_opt_div = opt_div_list[-1]
        base_tau_norm = None
        self._get_ls_n = getattr(self, '_get_ls_n', 0) + 1
        _skip_base = (
            self.type != 'none' and bool(self.models)
            and os.environ.get('IRL_SKIP_BASELINE_RESOLVE', '0').strip().lower()
                not in ('0', '', 'false', 'off', 'no')
            and self._get_ls_n > int(os.environ.get('IRL_BASELINE_RESOLVE_WARMUP', '2')))
        if self.type != 'none' and self.models and not _skip_base:
            _ev0 = self._evaluate_at_alpha(0.0, w_temp, dw, xs_init, us_init)
            base_q_norm  = _ev0["q_norm"]
            base_opt_div = _ev0["opt_div"]
            base_tau_norm = _ev0.get("tau_norm")
            _sq, _so = q_norm_list[-1], opt_div_list[-1]
            _hstr = ""
            try:
                _ms = self.models if len(self.models) > 1 else [self.model]
                _gaps = [float(getattr(m.solver, 'gap_norm', np.nan)) for m in _ms]
                _ffs = []
                for m in _ms:
                    try:
                        _ffs.append(float(np.linalg.norm(m.get_contact_forces(), axis=1).mean()))
                    except Exception:
                        pass
                _hstr = (f"  [HEALTH: chall gap={np.nanmean(_gaps):.2f}"
                         + (f" force={np.nanmean(_ffs):.1f}N" if _ffs else "") + "]")
            except Exception:
                pass
            _line = (f"    [LS baseline for accept] q_norm={base_q_norm:.4f}  "
                     f"opt_div={base_opt_div:.6f}{_hstr}")
            if abs(_sq - base_q_norm) > 1e-3 or abs(_so - base_opt_div) > 1e-6:
                _line += (f"   [WARN stored q={_sq:.4f} od={_so:.6f} != reseed -- "
                          f"warmstart/solver non-reproducible]")
            print(_line)
        elif _skip_base:
            print(f"    [LS baseline for accept] q_norm={base_q_norm:.4f}  "
                  f"opt_div={base_opt_div:.6f}   [reuse stored — IRL_SKIP_BASELINE_RESOLVE "
                  f"(deterministic challenger), saved 1 solve]")

        if self.type == 'none':
            w_loop = self._apply_bounds(w_prev + dw)
            chosen_step = True

            self._push_weights_to_solver(w_loop)
            self.model.solver.solve(xs_init, us_init, self.sqp_iter)

            solver_args['Step'] = 0

        while not chosen_step:
            step = self.steps[step_ind]
            ev = self._evaluate_at_alpha(step, w_temp, dw, xs_init, us_init)
            w_curr = ev["w_curr"]
            phis = ev["phis"]
            phis_cum_int = ev["phis_cum_int"]
            q_norm_ls = ev["q_norm"]
            opt_div_ls = ev["opt_div"]
            print(f"    [LS step {step_ind}] step={step:.4f}  q_norm={q_norm_ls:.4f} (base={base_q_norm:.4f})  opt_div={opt_div_ls:.6f} (base={base_opt_div:.6f})")

            if self.type == 'q_norm':
                q_norm = q_norm_ls
                condition_q = q_norm < base_q_norm
                if dq_norm_list is not None and self.get_dq_norm is not None:
                    dq_norm = ev.get("dq_norm")
                    if dq_norm is None:
                        dq_norm = float(np.mean([self.get_dq_norm(xs_d, ev["xs"])
                                                 for xs_d in self.Xs_opt_list]))
                    if self.use_dq_norm:
                        dq_ceil = self.dq_norm_tol * dq_norm_list[-1]
                        condition_q = condition_q and (dq_norm < dq_ceil)
                        print(f"    [LS step {step_ind}] dq_norm={dq_norm:.4f} "
                              f"(ceil={dq_ceil:.4f} = {self.dq_norm_tol}x prev)  "
                              f"velocity-guard {'OK' if dq_norm < dq_ceil else 'REJECT'}")
                if self.use_accel_guard and self.get_accel_mag is not None:
                    accel_mag = ev.get("accel_mag")
                    if accel_mag is not None and self._accel_ref is not None:
                        a_ceil = self.accel_tol * self._accel_ref
                        condition_q = condition_q and (accel_mag < a_ceil)
                        print(f"    [LS step {step_ind}] accel={accel_mag:.4f} "
                              f"(ceil={a_ceil:.4f} = {self.accel_tol}x last-accepted)  "
                              f"accel-guard {'OK' if accel_mag < a_ceil else 'REJECT'}")
            if self.type == 'opt':
                opt_div = self.get_opt_div(phis_cum_int, self.phis_opt_int)
                condition_opt = opt_div < base_opt_div
            if self.type == 'cost':
                phi_bar = phis_cum_int[0] - self.phis_opt_int[0]
                phi_bar_prev = prev_phis_int[0] - self.phis_opt_int[0]
                cost_diff = 0.5*(np.sum(w_curr*phis_cum_int[0]) - np.sum(w_curr*self.phis_opt_int[0]))**2
                dmdw_prev = np.sum(w_prev*phi_bar_prev)*phi_bar_prev
                dmdw = np.sum(w_curr*phi_bar)*phi_bar
                condition_cost_1 = cost_diff <= cost_diff_list[-1] + step*c1*np.sum(dw*dmdw_prev)
                condition_cost_2 = np.sum(dmdw*dw) >= c2*np.sum(dmdw_prev*dw)
                condition_cost = condition_cost_1 and condition_cost_2
            if self.type == 'filter_line_search':
                phi_bar = phis_cum_int[0] - self.phis_opt_int[0]
                phi_bar_prev = prev_phis_int[0] - self.phis_opt_int[0]
                opt_div = self.get_opt_div(phis_cum_int, self.phis_opt_int)

                condition_opt = opt_div < base_opt_div

                
                cost_diff = 0.5*(np.sum(w_curr*phis_cum_int[0]) - np.sum(w_curr*self.phis_opt_int[0]))**2
                dmdw_prev = np.sum(w_prev*phi_bar_prev)*phi_bar_prev
                dmdw = np.sum(w_curr*phi_bar)*phi_bar

                condition_cost_1 = cost_diff <= cost_diff_list[-1] + step*c1*np.sum(dw*dmdw_prev)
                condition_cost_2 = np.sum(dmdw*dw) >= c2*np.sum(dmdw_prev*dw)
                condition_cost = condition_cost_1 and condition_cost_2
                

            if self.type == 'entropy':
                phis_set_ls = phis_set.copy(); phis_set_ls.append(phis)
                prev_entropy = self.get_entropy(phis_set_ls, w_prev)
                entropy = self.get_entropy(phis_set_ls, w_curr)
                print('Current Entropy: ', entropy, 'Previous Entropy: ', prev_entropy)
                condition_entropy = entropy > prev_entropy

            if self.type == 'pareto':
                _m = self._pareto_margin
                _use_tau = (self.tau_norm_accept and ev.get("tau_norm") is not None
                            and base_tau_norm is not None)
                _cand = {'q': float(q_norm_ls), 'opt': float(opt_div_ls)}
                if _use_tau:
                    _cand['t'] = float(ev["tau_norm"])
                if self._filter_history is None:
                    _b = {'q': float(base_q_norm), 'opt': float(base_opt_div)}
                    if _use_tau:
                        _b['t'] = float(base_tau_norm)
                    self._filter_history = [_b]
                _improves = (_cand['q'] < base_q_norm * (1.0 - _m)
                             or _cand['opt'] < base_opt_div * (1.0 - _m)
                             or (_use_tau and _cand['t'] < base_tau_norm * (1.0 - _m)))
                def _dom(e):
                    d = (e['q'] <= _cand['q'] * (1.0 + _m)
                         and e['opt'] <= _cand['opt'] * (1.0 + _m))
                    if _use_tau:
                        d = d and (e.get('t', float('inf')) <= _cand['t'] * (1.0 + _m))
                    return d
                _dominated = any(_dom(e) for e in self._filter_history)
                _q_ok = (float(q_norm_ls) <= base_q_norm * (1.0 + self._pareto_qnorm_cap))
                condition_pareto = bool(_improves and not _dominated and _q_ok)
                _tstr = f" tau={_cand['t']:.4f} (base={base_tau_norm:.4f})" if _use_tau else ""
                _qstr = "" if self._pareto_qnorm_cap == float('inf') else f" q_ok={_q_ok}"
                print(f"    [LS step {step_ind}] pareto improves={_improves} "
                      f"dominated={_dominated}{_tstr}{_qstr} front={len(self._filter_history)} "
                      f"-> {'ACCEPT' if condition_pareto else 'reject'}")

            if self.type == 'opt':
                step_type = 'Opt'
                condition = condition_opt
            elif self.type == 'cost':
                step_type = 'Cost'
                condition = condition_cost
            elif self.type == 'entropy':
                step_type = 'Entropy'
                condition = condition_entropy
            elif self.type == 'q_norm':
                step_type = 'q Norm'
                condition = condition_q
            elif self.type == 'pareto':
                step_type = 'Pareto'
                condition = condition_pareto
            elif self.type == 'filter_line_search':
                is_worse_than_memory = False
                condition = condition_opt or condition_cost

                if condition:
                    for od, cd in zip(opt_div_list, cost_diff_list):
                        is_worse_than_memory = cost_diff > cd or opt_div > od
                    if is_worse_than_memory:
                        condition = False 
                    else:
                        if condition_opt and not condition_cost:
                            step_type = 'Opt'
                        elif condition_cost and not condition_opt:
                            step_type = 'Cost'
                        elif condition_opt and condition_cost:
                            step_type = 'Both' 
                    
        
            if condition:
                chosen_step = True
                solver_args['Step'] = step_ind
                solver_args['Step_Type'] = step_type
                if self.type == 'pareto':
                    _m = self._pareto_margin
                    def _cand_dominates(e):
                        d = (_cand['q'] <= e['q'] * (1.0 + _m)
                             and _cand['opt'] <= e['opt'] * (1.0 + _m))
                        if 't' in _cand:
                            d = d and (_cand['t'] <= e.get('t', float('inf')) * (1.0 + _m))
                        return d
                    self._filter_history = [
                        e for e in self._filter_history if not _cand_dominates(e)]
                    self._filter_history.append(_cand)
                w_loop = self._apply_bounds(w_temp + step*dw)
                if self.use_accel_guard and ev.get("accel_mag") is not None:
                    self._accel_ref = ev["accel_mag"]
            else:
                chosen_step = False
                step_ind += 1
                if step_ind == len(self.steps):
                    solver_args['Step'] = 'Stop'
                    chosen_step = True
                    solver_args['hard_terminate'] = True
                    solver_args['message'] = 'No Step Found'
                    w_loop = w_prev
                    break
        
        
        return w_loop, solver_args, output_traj

def optimization_args(opt_type, irl_iter):
    if opt_type == 'L-BFGS-B':
        options = {'maxiter': irl_iter, 'iprint': -1,'ftol': 1e-10 ,'gtol' : 1e-10, 'maxls': 100}
        tol = 1e-10
        lb = 0; ub = np.inf; bnds = Bounds(lb, ub)
    
    return options, tol, bnds

def subset_creator(nopt_inds):
    sets = []
    for i in range(1,len(nopt_inds)+1):
        sets.extend(list(combinations(nopt_inds, i)))
    return sets
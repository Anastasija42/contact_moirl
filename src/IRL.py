"""Single-objective IRL solver underlying MO_IRL.

Holds the MaxEnt log-likelihood gradient and the per-feature gradient mask that
keeps features spanning several orders of magnitude advancing at comparable
rates.
"""
import numpy as np
from Optimization_utils import *
from DMP import *

def make_gradient_mask(model, xs_demo, us_demo, n_w=1, rel_floor=0.01, K=None,
                       nr_run=None, nr_term=None):
    """
    Build a per-feature gradient normalization mask from demo feature magnitudes.

    Without normalization the IRL gradient component for feature i is O(|φ_i_demo|),
    so features with large integrated values dominate the update direction and converge
    much faster than small features, regardless of the line-search step size.

    This mask rescales each component by 1/|φ_i_demo| so that all features
    contribute comparably:

        g_masked[i] = g[i] / |φ_i_demo|   ~  O(1)  for all i

    The floor prevents extreme amplification of near-zero features:
        floor_i = rel_floor * max_j(|φ_j_demo|)
    so no feature gets more than 1/rel_floor times the weight of the largest one.

    Parameters
    ----------
    model    : Human / HumanMPPI instance — must implement get_traj_features()
    xs_demo  : (T+1, nx)  demo state trajectory
    us_demo  : (T, nu)    demo controls
    n_w      : number of time windows (>1 for TV weights); mask is tiled across windows
    rel_floor: relative floor as a fraction of the largest feature magnitude.
               Default 0.01 → smallest feature gets at most 100× the gradient of
               the largest.  Raise this if small terminal features are being over-amplified.

    Returns
    -------
    mask : (n_w * nr,) float array, where nr = nr_run + nr_term.
           Pass as mo_args['gradient_mask'].

    Usage
    -----
        from IRL import make_gradient_mask

        mask = make_gradient_mask(human, xs_irl, us_irl, n_w=irl_args['n_w'])
        mo_args['gradient_mask'] = mask
    """
    phi_demo, _, _, _ = model.get_traj_features(xs_demo, us_demo)
    sigma = np.abs(phi_demo)
    floor = rel_floor * np.max(sigma) if np.max(sigma) > 0 else 1e-6
    mask_single = 1.0 / np.maximum(sigma, floor)
    blocks = K if K is not None else n_w
    if nr_run is None:
        return np.tile(mask_single, blocks)
    nr_t = (len(mask_single) - nr_run) if nr_term is None else nr_term
    return np.concatenate([
        np.tile(mask_single[:nr_run],            blocks),
        np.tile(mask_single[nr_run:nr_run + nr_t], blocks),
    ])

class IRL_Solver():
    def __init__(self, args):
        self.xs_opt = args['xs_opt']
        self.us_opt = args['us_opt']
        self.multi_demo = isinstance(self.xs_opt, list)
        self.args = args
        self.model = args['model']
        # given it must be a list parallel to xs_opt/us_opt (demo i ↔ model i);
        self.models = args.get('models', None)
        if self.models:
            self.model = self.models[0]
        self.dt = self.model.dt
        self.max_iter = args['max_iter']
        self.min_iter = args['min_iter']
        self.solver = self.model.solver
        self.verbose = args['verbose']
        self.irl_iter = args['irl_iter']
        self.compare_desired = args['compare_desired']
        self.phi_opt = None
        self.cost_opt = None
        self.dw = []
        self.fcn_vals = []
        self.Xs = []; self.Us = []
        self.phis = []; self.phis_set = []; self.phis_set_int = []
        self.costs = []
        self.costs_set = []
        self.ws = []
        self.norm_phis = []
        self.Ps = []; self.Ps_IRL = []
        self.Ent = []; self.Ent_IRL = []
        self.w_star = None
        self.Js = []
        self.n_w = self.args.get('n_w', 1)

        self.weight_mode = self.args.get('weight_mode', 'single')
        if self.weight_mode == 'single' and self.n_w > 1:
            self.weight_mode = 'windowed'

        if self.weight_mode == 'basis':
            if 'K' not in self.args:
                raise ValueError("weight_mode='basis' requires args['K'] (number of basis functions)")
            self.K = int(self.args['K'])
            self.basis_type = self.args.get('basis_type', 'gaussian')
            self.basis_sigma = self.args.get('basis_sigma', None)
            if self.n_w < 2:
                self.n_w = self.K
        elif self.weight_mode == 'windowed':
            self.K = self.n_w
        else:
            self.K = 1

        self.B_step = None
        self.B_window = None

        self.init_params()
        if self.weight_mode == 'basis':
            self.B_step, self.B_window = self._build_basis()

    def init_params(self):
        self.iter = 0
        self.nq = self.model.nq
        self.nx = self.model.nx
        self.nv = self.model.nv
        self.nu = self.model.nu
        self.T = self.model.solver.problem.T
        self.model.solver.termination_tolerance = 1e-4
        self.model.solver.with_callbacks = False

        self.keys_run = sorted(list(self.args['w_run'].keys()))
        self.keys_term = sorted(list(self.args['w_term'].keys()))

        self.nr_run = len(self.keys_run)
        self.nr_term = len(self.keys_term)
        self.nr = self.nr_run + self.nr_term

        if self.multi_demo:
            xs_opt_list = self.xs_opt
            us_opt_list = self.us_opt
            self.n_demos = len(xs_opt_list)
            self.xs_opt = xs_opt_list[0]
            self.us_opt = us_opt_list[0]
        else:
            xs_opt_list = [self.xs_opt]
            us_opt_list = [self.us_opt]
            self.n_demos = 1

        for i, (xs, us) in enumerate(zip(xs_opt_list, us_opt_list)):
            m = self.models[i] if self.models else self.model
            phi_, phis_, phis_cum_, phis_int_ = m.get_traj_features(xs, us)
            self.Xs.append(xs); self.Us.append(us)
            self.phis.append(phi_)
            self.phis_set.append(phis_)
            self.phis_set_int.append(phis_int_)

        if self.models:
            self.phi_opt      = np.mean(np.stack(self.phis[:self.n_demos]), axis=0)
            self.phis_opt     = self._mean_phis_list(self.phis_set[:self.n_demos])
            self.phis_opt_int = self._mean_phis_list(self.phis_set_int[:self.n_demos])
            self.phis_opt_cum = self.phi_opt
        else:
            self.phi_opt = self.phis[0]
            self.phis_opt = self.phis_set[0]
            self.phis_opt_int = self.phis_set_int[0]
            self.phis_opt_cum = self.phis[0]

        if 'w_run' in self.args.keys() and 'w_term' in self.args.keys():
            self.w_run = self.args['w_run']
            self.w_term = self.args['w_term']
            self.prev_w_run = self.args['w_run']
            self.prev_w_term = self.args['w_term']
            self.ws.append(self.dict_to_vector(self.w_run, self.w_term))
        else:
            self.w_run, self.w_term = self.generate_zero_w()
            self.prev_w_run, self.prev_w_term = self.generate_zero_w()
            self.ws.append(self.dict_to_vector(self.w_run, self.w_term))

        if self.compare_desired:
            try:
                self.w_star_run, self.w_star_term = self.args['des_run'], self.args['des_term']
                self.w_star = self.dict_to_vector(self.w_star_run, self.w_star_term)
                w_run_star_v, w_term_star_v, _ = self.dict_to_vector(self.w_star_run, self.w_star_term)
                self.w_run_star_v = w_run_star_v; self.w_term_star_v = w_term_star_v
            except:
                print('Desired weights not provided.')
                self.compare_desired = False
        
        self.model.update_solver_weights(self.w_run, self.w_term)

    @staticmethod
    def _mean_phis_list(phis_lists):
        """Per-timestep mean of N per-subject phis (each a length-(T+1) sequence
        of (nr,) feature vectors). Returns a (T+1, nr) array. Population IRL
        requires a COMMON horizon T across subjects (resample demos)."""
        arrs = [np.asarray(pl, dtype=float) for pl in phis_lists]
        if len({a.shape for a in arrs}) != 1:
            raise ValueError(
                f"Population IRL needs a common horizon: per-subject phis have "
                f"shapes {[a.shape for a in arrs]}. Resample demos to one T.")
        return np.mean(np.stack(arrs, axis=0), axis=0)

    def generate_trajectories(self):
        Xs = self.Xs
        Us = self.Us
        phis = self.phis
        phis_set = self.phis_set
        phis_set_int = self.phis_set_int
        Xs_set, Us_set = self.dmp.generate_noisy_traj(Xs[-1], self.args['rollout_N'])
        for xs, us in zip(Xs_set, Us_set):
            Xs.append(xs)
            Us.append(us)
            phis_, phis_set_, _, phis_set_int_ = self.model.get_traj_features(xs, us)
            phis_set.append(phis_set_); phis_set_int.append(phis_set_int_)
            phis.append(phis_)
        
        return Xs, Us, phis, phis_set, phis_set_int

    def _push_init_weights_like_ls(self, m, w_run, w_term, pc_scale=1.0):
        """Push the init weighting the SAME way the line search does: in basis mode
        as the time-varying W(t)=B_window@theta (not a single constant via
        update_solver_weights). Otherwise the init-baseline challenger is solved in
        a different cost representation than every LS probe, so the Init-row q_norm
        disagrees with the LS reseed -> the [WARN stored != reseed]."""
        if self.weight_mode == 'basis' and self.B_window is not None:
            _, _, wv = self.dict_to_vector(w_run, w_term)
            nr_r, nr_t = len(self.keys_run), len(self.keys_term)
            theta_run = wv[:self.K * nr_r].reshape(self.K, nr_r)
            w_run_win = self.B_window @ theta_run
            if pc_scale != 1.0:
                w_run_win = w_run_win.copy()
                for _pk in ('press_capacity', 'press_force'):
                    if _pk in self.keys_run:
                        w_run_win[:, list(self.keys_run).index(_pk)] *= float(pc_scale)
            w_term_win = (self.B_window @ wv[self.K * nr_r:].reshape(self.K, nr_t)
                          if nr_t > 0 else None)
            m.update_solver_weights_tv(w_run_win, w_term_win)
        else:
            m.update_solver_weights(w_run, w_term)

    def generate_bad_trajectory(self):
        w_run = self.w_run.copy()
        w_term = self.w_term.copy()
        # Honest cold init: with ls_cold_start the INIT challenger must use the
        _cold = bool(self.args.get('ls_cold_start', False))
        if self.models:
            for i, m in enumerate(self.models):
                self._push_init_weights_like_ls(m, w_run, w_term)
                m.solver.termination_tolerance = 1e-4
                m.solver.with_callbacks = False
                if _cold:
                    xw, uw = m._ik_warmstart()
                    xi = [np.asarray(x).copy() for x in xw]
                    ui = [np.asarray(u).copy() for u in uw]
                else:
                    xi = list(self.Xs[i])
                    if self.args.get('challenger_press_warmstart', False):
                        # not the kinematics, so demo states + zero u never reaches high f.
                        ui = [np.asarray(u).copy() for u in self.Us[i]]
                    else:
                        ui = [np.zeros(m.nu) for _ in range(len(xi) - 1)]
                if self.args.get('challenger_continuation', False) and 'press_capacity' in self.keys_run:
                    _xi, _ui = xi, ui
                    for _frac in self.args.get('cont_fracs', (0.3, 0.6, 1.0)):
                        self._push_init_weights_like_ls(m, w_run, w_term, pc_scale=_frac)
                        m.solver.solve(list(_xi), list(_ui), self.args.get('sqp_iter', 50))
                        _xi = [np.asarray(x).copy() for x in m.solver.xs]
                        _ui = [np.asarray(u).copy() for u in m.solver.us]
                else:
                    m.solver.solve(xi, ui, self.args.get('sqp_iter', 50))
            m0 = self.models[0]
            return np.stack(m0.solver.xs.copy()), np.stack(m0.solver.us.copy())
        self._push_init_weights_like_ls(self.model, w_run, w_term)
        self.model.solver.termination_tolerance = 1e-4
        self.model.solver.with_callbacks = False
        if _cold:
            xw, uw = self.model._ik_warmstart()
            xs_init = [np.asarray(x).copy() for x in xw]
            us_init = [np.asarray(u).copy() for u in uw]
        elif self.args.get('challenger_press_warmstart', False):
            xs_init = list(self.Xs[0])
            us_init = [np.asarray(u).copy() for u in self.Us[0]]
        else:
            xs_init = [self.xs_opt[0] for i in range(self.T+1)]
            us_init = [self.us_opt[0] for i in range(self.T)]
        if self.args.get('challenger_continuation', False) and 'press_capacity' in self.keys_run:
            _xs, _us = list(xs_init), list(us_init)
            for _frac in self.args.get('cont_fracs', (0.3, 0.6, 1.0)):
                self._push_init_weights_like_ls(self.model, w_run, w_term, pc_scale=_frac)
                self.model.solver.solve(list(_xs), list(_us), self.args.get('sqp_iter', 50))
                _xs = [np.asarray(x).copy() for x in self.model.solver.xs]
                _us = [np.asarray(u).copy() for u in self.model.solver.us]
        else:
            self.model.solver.solve(xs_init, us_init, self.args.get('sqp_iter', 50))
        return np.stack(self.model.solver.xs.copy()), np.stack(self.model.solver.us.copy())
    
    def get_traj_costs(self, phis, w_run, w_term, t_offset=0):
        if self.weight_mode == 'basis':
            return self.get_traj_costs_basis(phis, w_run, w_term, t_offset=t_offset)
        if self.n_w > 1:
            w_run_tv = w_run.reshape(self.n_w, self.nr_run)
            w_term_tv = w_term.reshape(self.n_w, self.nr_term)
            return self.get_traj_costs_tv(phis, w_run_tv, w_term_tv, t_offset=t_offset)
        cost = 0.0
        T = len(phis)
        cost_set = np.zeros(T)
        w = np.concatenate((w_run, w_term))
        phi = phis[-1]
        cost = np.sum(phi*w)
        cost_set[-1] = cost
        for i in range(T-2,-1,-1):
            phi = phis[i]
            cost += np.sum(phi*w)*self.dt
            cost_set[i] = cost
        return cost, cost_set

    def _build_basis(self):
        T, K, n_w = self.T, self.K, self.n_w
        centers = np.linspace(0, T, K)
        sigma = self.basis_sigma if self.basis_sigma is not None else max(T / max(K - 1, 1), 1.0)
        t_step = np.arange(T + 1)

        if self.basis_type == 'gaussian':
            B_raw = np.exp(-((t_step[:, None] - centers[None, :]) ** 2) / (2.0 * sigma * sigma))
        elif self.basis_type == 'softrect':
            from scipy.special import expit
            edges = np.linspace(0, T, K + 1)
            B_raw = (expit((t_step[:, None] - edges[None, :-1]) / sigma)
                     - expit((t_step[:, None] - edges[None, 1:])  / sigma))
        elif self.basis_type == 'rectangular':
            window_size = max(1, T // K)
            bin_idx = np.minimum(t_step // window_size, K - 1)
            B_raw = np.zeros((T + 1, K))
            B_raw[np.arange(T + 1), bin_idx] = 1.0
        else:
            raise ValueError(f"Unknown basis_type: {self.basis_type}")

        B_step = B_raw / (B_raw.sum(axis=1, keepdims=True) + 1e-12)

        window_size = max(1, T // n_w)
        B_window = np.zeros((n_w, K))
        for k in range(n_w):
            lo = k * window_size
            hi = (k + 1) * window_size if k < n_w - 1 else T + 1
            B_window[k] = B_step[lo:hi].mean(axis=0)
        return B_step, B_window

    def get_traj_costs_basis(self, phis, theta_run_flat, theta_term_flat, t_offset=0):
        T = len(phis)
        K, nr_r, nr_t = self.K, self.nr_run, self.nr_term
        theta_run  = np.asarray(theta_run_flat).reshape(K, nr_r)
        theta_term = np.asarray(theta_term_flat).reshape(K, nr_t) if nr_t > 0 else np.zeros((K, 0))

        cost = 0.0
        cost_set = np.zeros(T)

        last_t = min(t_offset + T - 1, self.T)
        b_last = self.B_step[last_t]
        w_full_last = np.concatenate((b_last @ theta_run,
                                      b_last @ theta_term if nr_t > 0 else np.zeros(0)))
        cost = np.sum(phis[-1] * w_full_last)
        cost_set[-1] = cost

        for i in range(T - 2, -1, -1):
            abs_t = min(t_offset + i, self.T)
            b = self.B_step[abs_t]
            w_full = np.concatenate((b @ theta_run,
                                     b @ theta_term if nr_t > 0 else np.zeros(0)))
            cost += np.sum(phis[i] * w_full) * self.dt
            cost_set[i] = cost

        return cost, cost_set

    def get_traj_costs_tv(self, phis, w_run_tv, w_term_tv, t_offset=0):
        """
        w_run_tv:  (n_w, nr_run)  — one weight vector per window
        w_term_tv: (n_w, nr_term) — only last window applies to terminal
        t_offset:  absolute timestep of phis[0] in the full trajectory
        """
        T = len(phis)
        n_w = len(w_run_tv)
        full_T = self.T
        window_size = max(1, full_T // n_w)
        cost = 0.0
        cost_set = np.zeros(T)

        k_term = min((t_offset + T - 1) // window_size, n_w - 1)
        w_term = np.concatenate((w_run_tv[k_term], w_term_tv[k_term]))
        cost = np.sum(phis[-1] * w_term)
        cost_set[-1] = cost

        for i in range(T - 2, -1, -1):
            k = min((t_offset + i) // window_size, n_w - 1)
            w = np.concatenate((w_run_tv[k], w_term_tv[k]))
            cost += np.sum(phis[i] * w) * self.dt
            cost_set[i] = cost

        return cost, cost_set

    def get_q_rmse(self, xs1, xs2, to_deg = True):
        N = len(xs1)
        if to_deg:
            deg_conv = 180/np.pi
        else:
            deg_conv = 1.0
        if len(xs2) != N:
            print('Cannot compute RMSE on two differently shaped arrays')
        return np.sqrt((1/N)*np.sum((xs1[:,:self.nq]*deg_conv - xs2[:,:self.nq]*deg_conv)**2, axis=0))
    
    def get_dq_rmse(self, xs1, xs2, to_deg = True):
        N = len(xs1)
        if to_deg:
            deg_conv = 180/np.pi
        else:
            deg_conv = 1.0
        if len(xs2) != N:
            print('Cannot compute RMSE on two differently shaped arrays')
        return np.sqrt((1/N)*np.sum((xs1[:,self.nq:]*deg_conv - xs2[:,self.nq:]*deg_conv)**2, axis=0))
    
    def get_x_rmse(self, xs1, xs2):
        N = len(xs1)
        if len(xs2) != N:
            print('Cannot compute RMSE on two differently shaped arrays')
        return np.sqrt((1/N)*np.sum((xs1 - xs2)**2, axis=0))

    
    def get_traj_set_costs(self, phis_set, w_run, w_term):
        costs = []
        for phis in phis_set:
            cost, _ = self.get_traj_costs(phis, w_run, w_term)
            costs.append(cost)
        return costs
    
    def get_Ps(self, Phis, w):
        Ps = np.zeros(len(Phis))
        nr_run_total = self.K * self.nr_run
        w_run = w[:nr_run_total]
        w_term = w[nr_run_total:]
        for i, phi in enumerate(Phis):
            Ps[i] = self.get_traj_costs(phi, w_run, w_term)[0]
        Ps -= np.min(Ps)
        Ps = np.exp(-Ps) + 1e-20
        Ps /= np.sum(Ps)
        return Ps
    
    def get_entropy(self, Phis, w):
        Ps = self.get_Ps(Phis, w)
        return -np.sum(Ps*np.log(Ps + 1e-30))
    
    def initiate_trajectory(self):
        if self.next_traj == 'best':
            X, U = self.get_best_trajectory()
        elif self.next_traj == 'last':
            T = len(self.Us[-1])
            X = [self.Xs[-1][i] for i in range(T+1)]
            U = [self.Us[-1][i] for i in range(T)]
        elif self.next_traj == 'worst':
            T = len(self.Us[self.n_demos])
            X = [self.Xs[self.n_demos][i] for i in range(T+1)]
            U = [self.Us[self.n_demos][i] for i in range(T)]
        elif self.next_traj == 'optimal':
            T = len(self.Us[0])
            X = [self.Xs[0][i] for i in range(T+1)]
            U = [self.Us[0][i] for i in range(T)]
        return X, U  
    
    def get_inds(self, X_set = None):
        if X_set is None:
            X_set = self.Xs
        n_nopt = len(X_set) - self.n_demos
        if self.K_set == 'all':
            inds = list(range(self.n_demos, len(X_set)))
        else:
            if n_nopt > self.K_set:
                if self.use_best:
                    vals = np.array(range(self.n_demos, len(X_set)))
                    inds = list(np.argsort(vals)[:self.K_set] + self.n_demos)
                else:
                    inds = list(range(len(X_set)-self.K_set, len(X_set)))
            else:
                inds = list(range(self.n_demos, len(X_set)))
        return inds
    
    def get_opt_div(self, phis1_int, phis2_int, t = None):
        if t is None:
            t = self.T

        p1 = phis1_int[0]
        p2 = phis2_int[0]

        min_dim = min(len(p1), len(p2))
        p1_s = p1[:min_dim]
        p2_s = p2[:min_dim]

        ls_mask = getattr(self, 'opt_div_mask', None)
        if ls_mask is not None:
            m = np.asarray(ls_mask)[:min_dim].astype(bool)
            if np.any(m):
                return np.linalg.norm(p1_s[m] - p2_s[m]) / (t + 1)

        w_curr = self.ws[-1][-1] if self.ws else None
        if w_curr is not None:
            w_s = np.abs(w_curr[:min_dim])
            mask = w_s > 1e-10
            if np.any(mask):
                opt_div = np.linalg.norm(p1_s[mask] - p2_s[mask]) / (t + 1)
                return opt_div

        opt_div = np.linalg.norm(p1_s - p2_s) / (t + 1)
        return opt_div
        
    def get_cost_diff(self, cs1, cs2):
        cs1 = np.array(cs1)
        cs2 = np.array(cs2)
        cost_diff =0.5*(cs1[0] - cs2[0])**2
        cost_diff/=len(cs1)
        return cost_diff
    
    def get_state_diff(self, xs1, xs2):
        state_diff = 0.0
        for x1, x2 in zip(xs1, xs2):
            state_diff += np.linalg.norm(x1 - x2)
        state_diff /= len(xs1)
        state_diff = 0.5*state_diff**2
        return state_diff
    
    def _free_q_idx(self):
        """Column indices of the joints q_norm/dq_norm should measure. Excludes
        any PINNED DOF — a held trunk (lock_thorax) OR the fixed rock
        (rock_fixed_joint, always pinned) — because a pinned joint matches the
        demo trivially and dilutes the metric, flattering the fit on the joints
        actually recovered. Pinned = position bound range ~0 (auto-detects the
        rock and any lock without hardcoded indices); an explicit
        _locked_q_idx list is also honored."""
        m = getattr(self, 'model', None)
        pinned = set(getattr(m, '_locked_q_idx', None) or ())
        lb = getattr(m, 'x_lb', None)
        ub = getattr(m, 'x_ub', None)
        if lb is not None and ub is not None:
            rng = np.asarray(ub)[:self.nq] - np.asarray(lb)[:self.nq]
            pinned |= set(int(i) for i in np.where(rng < 0.02)[0])
        pm = getattr(m, 'pin_model', None)
        if pm is not None:
            for jid in range(pm.njoints):
                if 'rock' in pm.names[jid].lower():
                    j = pm.joints[jid]
                    if j.nq >= 1:
                        pinned |= set(range(int(j.idx_q), int(j.idx_q) + int(j.nq)))
        return [i for i in range(self.nq) if i not in pinned]

    def get_q_norm(self, xs1, xs2, to_deg = True):
        deg_conv = 180/np.pi if to_deg else 1.0
        idx = self._free_q_idx()
        d = (np.asarray(xs1)[:, idx] - np.asarray(xs2)[:, idx]) * deg_conv
        return np.linalg.norm(d)/len(xs1)

    def get_dq_norm(self, xs1, xs2, to_deg = True):
        deg_conv = 180/np.pi if to_deg else 1.0
        idx = [self.nq + i for i in self._free_q_idx()]
        d = (np.asarray(xs1)[:, idx] - np.asarray(xs2)[:, idx]) * deg_conv
        return np.linalg.norm(d)/len(xs1)
    
    def get_accel_mag(self, xs, to_deg = True):
        deg_conv = 180/np.pi if to_deg else 1.0
        v = np.asarray(xs)[:, self.nq:] * deg_conv
        if len(v) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(v, axis=0)) / len(v))

    def get_norm_phi(self, phis):
        norm_phi = []
        t = len(phis[0])
        for phi in phis:
            norm_phi.append(self.get_opt_div(phi, self.phis_opt_int, t))
        return norm_phi

    def dict_to_vector(self, w_run, w_term):
        wv_run = np.zeros(len(w_run))
        wv_term = np.zeros(len(w_term))
        wv = np.zeros(len(w_run)+len(w_term))
        for i, k in enumerate(self.keys_run):       
            wv[i] = w_run[k]
            wv_run[i] = w_run[k]
        for i, k in enumerate(self.keys_term):
            wv[i+len(w_run)] = w_term[k]
            wv_term[i] = w_term[k]
        if self.K > 1:
            wv_run = np.tile(wv_run, self.K)
            wv_term = np.tile(wv_term, self.K)
            wv = np.concatenate([wv_run, wv_term])
        return wv_run, wv_term, wv
    
    def vector_to_dict(self, wv_run, wv_term):
        w_run = {}
        w_term = {}
        if self.K > 1:
            wv_run = wv_run.reshape(self.K, self.nr_run).mean(axis=0)
            wv_term = wv_term.reshape(self.K, self.nr_term).mean(axis=0)
        for i, k in enumerate(self.keys_run):
            w_run[k] = wv_run[i]
        for i, k in enumerate(self.keys_term):
            w_term[k] = wv_term[i]
        return w_run, w_term
    
    def normalize_w_dict(self,w_run,w_term):
        A = np.float64(np.array(list(w_run.items()))[:,1])
        B = np.float64(np.array(list(w_term.items()))[:,1])
        M = np.max([np.max(A), np.max(B)])
        for key, value in zip(w_run.keys(), w_run.values()):
            w_run[key] = value/M
        for key, value in zip(w_term.keys(), w_term.values()):
            w_term[key] = value/M
        return w_run, w_term
    
    def normalize_w_vector(self,w_run,w_term):
        A = np.float64(w_run)
        B = np.float64(w_term) 
        if len(B) == 0:
            M = np.max(A)
        else:
            M = np.max([np.max(A), np.max(B)])
        if M != 0.0:
            w_run = w_run/M
            w_term = w_term/M
        return w_run, w_term

    def normalize_vector(self, wv):
        M = np.max(np.abs(wv))
        if M == 0:
            return wv
        return wv/M
    
    def generate_zero_w(self):
        w_run = {}
        w_term = {}
        for k in self.keys_run:
            w_run[k] = 0.0
        for k in self.keys_term:
            w_term[k] = 0.0
        return w_run, w_term

    def KL_divergence(self, likelihood_fcn, inds, w1, w2):
        KL = 0.0
        Px = np.zeros(len(inds))
        Qx = np.zeros(len(inds))
        kl_array = np.zeros(len(inds))
        for i in range(len(inds)):
            temp_inds = inds.copy(); temp_inds.pop(i)
            w_run_1 = w1[:self.nr_run]; w_term_1 = w1[self.nr_run:]
            w_run_2 = w2[:self.nr_run]; w_term_2 = w2[self.nr_run:]
            Px[i] = likelihood_fcn(inds[i], temp_inds, w_run_1, w_term_1, self.phis_set, f_type='L')
            Qx[i] = likelihood_fcn(inds[i], temp_inds, w_run_2, w_term_2, self.phis_set, f_type='L')
        
        # Avoiding Nan
        Px +=  1e-20
        Qx +=  1e-20

        if np.sum(Px) != 0:
            Px = Px/np.sum(Px)
        if np.sum(Qx) != 0:
            Qx = Qx/np.sum(Qx)
        if np.sum(Px) != 0:
            kl_array = Px*np.log(Px/Qx)
        KL = np.sum(kl_array)
        return KL

    def print_info(self):
        print('IRL Parameters:')
        print('Initial Running Weight: ', self.w_run)
        print('Initial Terminal Weight: ', self.w_term)
        print('Type: ', self.type)
        print('Set Size: ', self.K_set)
        print('Sample Size: ' , self.N_samples)
        print('Lambda: ', self.Lambda)
        print('SQP Iterations: ', self.sqp_iter)
        print('IRL Max Iteration: ', self.max_iter)
        print('Sample Time: ', self.dt)
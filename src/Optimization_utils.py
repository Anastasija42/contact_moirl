"""Optimizer wrapper used by the IRL outer loop.

Wraps the step schedules, L-BFGS history and line-search bookkeeping so the
solvers deal with one interface rather than the individual pieces.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds

class Optimizer():
    def __init__(self, args):
        self.type = args['type']
        self.irl_type = args['irl_type']
        self.T = args['T']
        self.dt = args['dt']
        self.Lambda = args['Lambda']
        self.Beta = args['Beta']
        self.l_reg = args['l_reg']
        self.normalize_w = args['normalize_w']
        self.use_jac = args['use_jac']
        self.use_hess = args['use_hess']
        self.iter = args['iter']
        self.verbose = args['verbose']
        self.nr_run = args['nr_run']
        self.nr_term = args['nr_term']
        self.dyn_reg = args['dyn_reg']
        self.nr = self.nr_run + self.nr_term
        self.N_samples = args['N_samples']
        self.ll_fcn = args['ll_fcn']
        self.gradient_mask = args.get('gradient_mask', None)
        self.slsqp_inner = args.get('slsqp_inner', False)
        self.eq_refs = None

    @staticmethod
    def f_w(x, args):
        phis = args['phis']
        opt_ind = args['opt_ind']
        nopt_ind = args['nopt_ind']
        nr_run = args['nr_run']
        nr_term = args['nr_term']
        ll_fcn = args['ll_fcn']
        l_reg = args['l_reg']
        prev_w_run = args['prev_w_run']
        prev_w_term = args['prev_w_term']
        var = args['var']
        temperature = args['temperature']
        l_type = args['l_type']
        f = 0.0
        l1_c = args['Lambda']; l2_c = args['Beta']

        if var == 'dw' or var == 'all':
            w_run = prev_w_run + x[:nr_run]; w_term = prev_w_term + x[nr_run:]
        elif var == 'w':
            w_run = x[:nr_run]; w_term = x[nr_run:]
        else:
            raise ValueError(
                f"f_w: unknown var={var!r}. Expected 'dw', 'w', or 'all'."
            )
        f = ll_fcn(opt_ind, nopt_ind, w_run, w_term, phis, l_type, temperature)
        if l_reg == 'lasso':
            f += l1_c*np.sum(np.abs(x))
        elif l_reg == 'ridge':
            f += 0.5*l2_c*np.linalg.norm(x)**2
        elif l_reg == 'elastic':
            f += l1_c*np.sum(np.abs(x)) + 0.5*l2_c*np.linalg.norm(x)**2
        return f
    
    @staticmethod
    def fg_w(x, args):
        f = 0.0
        g = np.zeros(len(x))
        phis = args['phis']
        opt_ind = args['opt_ind']
        nopt_ind = args['nopt_ind']
        nr_run = args['nr_run']
        nr_term = args['nr_term']
        ll_fcn = args['ll_fcn']
        l_reg = args['l_reg']
        prev_w_run = args['prev_w_run']
        prev_w_term = args['prev_w_term']
        dd_f = args['dd_f']
        var = args['var']
        prev_w = np.concatenate([prev_w_run, prev_w_term])
        temperature = args['temperature']
        l_type = args['l_type']
        f = 0.0

        if var == 'dw' or var == 'all':
            w_run = prev_w_run + x[:nr_run]; w_term = prev_w_term + x[nr_run:]
            f, g = ll_fcn(opt_ind, nopt_ind, w_run, w_term, phis, l_type, temperature)
        elif var == 'w':
            w_run = x[:nr_run]; w_term = x[nr_run:]
            f, g = ll_fcn(opt_ind, nopt_ind, w_run, w_term, phis, l_type, temperature)
        else:
            raise ValueError(
                f"fg_w: unknown var={var!r}. Expected 'dw', 'w', or 'all'."
            )

        l1_c = args['Lambda']; l2_c = args['Beta']

        if l_reg == 'lasso':
            f += l1_c*np.sum(np.abs(x))
            g += l1_c*dd_f(x, g)
        elif l_reg == 'ridge':
            f += 0.5*l2_c*np.linalg.norm(x)**2
            g += l2_c*x
        elif l_reg == 'elastic':
            f += l1_c*np.sum(np.abs(x)) + 0.5*l2_c*np.linalg.norm(x)**2
            g += l1_c*dd_f(x, g)  + l2_c*x
        return f, g
    
    @staticmethod
    def DD(X, dX):
        ddx = np.zeros(len(X))
        for i in range(len(X)):
            if X[i] < 0:
                ddx[i] = -1
            if X[i] > 0:
                ddx[i] = 1
            if X[i] == 0:
                ddx[i] = np.sign(dX[i])
        return ddx
    
    def min(self, opt_ind, nopt_ind, prev_w, phis, dw = None, var = 'w', outer_loop_iter = 100, lambda_t = None):
        if self.irl_type == 'MO_IRL' or self.irl_type == 'autoreg':
            if lambda_t is not None:
                temperature = 1/lambda_t
            else:
                temperature = 1
            decay_l1 = 1
            decay_l2 = 1
            alpha = 0.1
            if self.dyn_reg and outer_loop_iter > 3:
                decay_l1 = np.exp(-alpha*outer_loop_iter)
                decay_l2 = np.exp(alpha*outer_loop_iter)

            args = {
                'phis': phis,
                'opt_ind': opt_ind,
                'nopt_ind': nopt_ind,
                'nr_run': self.nr_run,
                'nr_term': self.nr_term,
                'Lambda': self.Lambda*decay_l1,
                'Beta': self.Beta*decay_l2,
                'll_fcn': self.ll_fcn,
                'l_reg': self.l_reg,
                'prev_w_run': prev_w[:self.nr_run],
                'prev_w_term': prev_w[self.nr_run:],
                'dd_f': self.DD,
                'var': var,
                'temperature': temperature
            }

            l_dw = len(prev_w)
            x0 = dw
            if self.use_jac:
                fcn = self.fg_w; jac = True; args['l_type'] = 'LJ_w'
            else:
                fcn = self.f_w; jac = None; args['l_type'] = 'L_w'
            
            lb_arr = list(-prev_w)
            ub_arr = [np.inf] * l_dw
            if self.gradient_mask is not None and len(self.gradient_mask) == l_dw:
                for i, m in enumerate(self.gradient_mask):
                    if m == 0:
                        lb_arr[i] = 0.0
                        ub_arr[i] = 0.0
            bnds = Bounds(lb_arr, ub_arr)
            iprint = -1
            
        
        elif self.irl_type == 'mrinal':
            args = {
                'phis': phis,
                'opt_ind': opt_ind,
                'nopt_ind': nopt_ind,
                'nr_run': self.nr_run,
                'nr_term': self.nr_term,
                'Lambda': self.Lambda,
                'Beta': self.Beta,
                'll_fcn': self.ll_fcn,
                'l_reg': self.l_reg,
                'prev_w_run': prev_w[:self.nr_run],
                'prev_w_term': prev_w[self.nr_run:],
                'dd_f': self.DD,
                'var': 'dw',
                'temperature': None
            }

            l_dw = len(prev_w)
            x0 = prev_w
            
            if self.use_jac:
                fcn = self.fg_w; jac = True; args['l_type'] = 'LJ_w'
            else:
                fcn = self.f_w; jac = None; args['l_type'] = 'L_w'
            
            if self.normalize_w:
                bnds = Bounds(list(np.zeros_like(prev_w)), [1.0]*l_dw)
            else:
                bnds = Bounds(list(np.zeros_like(prev_w)), [np.inf]*l_dw)

            iprint = 1

            
        elif self.irl_type == 'bretl' or self.irl_type == 'IS_IRL':
            if lambda_t is not None:
                temperature = 1/lambda_t
            args = {
                'phis': phis,
                'opt_ind': opt_ind,
                'nopt_ind': nopt_ind,
                'nr_run': self.nr_run,
                'nr_term': self.nr_term,
                'Lambda': self.Lambda,
                'Beta': self.Beta,
                'll_fcn': self.ll_fcn,
                'l_reg': self.l_reg,
                'prev_w_run': prev_w[:self.nr_run],
                'prev_w_term': prev_w[self.nr_run:],
                'dd_f': self.DD,
                'var': var,
                'temperature': temperature
            }

            l_dw = len(prev_w)
            x0 = prev_w
            
            if self.use_jac:
                fcn = self.fg_w; jac = True; args['l_type'] = 'LJ_w'
            else:
                fcn = self.f_w; jac = None; args['l_type'] = 'L_w'
            
            if self.normalize_w:
                bnds = Bounds(list(np.zeros_like(prev_w)), [1.0]*l_dw)
            else:
                bnds = Bounds(list(np.zeros_like(prev_w)), [np.inf]*l_dw)

            iprint = -1

            
        if self.type == 'SLSQP':
            cons = []
            for g_ref in (self.eq_refs or []):
                cons.append({
                    'type': 'eq',
                    'fun': (lambda x, gg=g_ref: float(np.dot(x, gg))),
                    'jac': (lambda x, gg=g_ref: gg),
                })
            res = minimize(fun=fcn,
                        x0=x0,
                        args=args,
                        jac=jac,
                        bounds=bnds,
                        method='SLSQP',
                        constraints=cons,
                        options={'maxiter': min(self.iter, 200), 'ftol': 1e-12})
        else:
            res = minimize(fun=fcn,
                        x0=x0,
                        args=args,
                        jac=jac,
                        bounds=bnds,
                        method=self.type,
                        options = {'maxiter': self.iter, 'iprint': iprint,'ftol': 1e-30 ,'gtol' : 1e-30, 'maxls': 50})

        return res

"""The OSS transfer driver, unchanged, with the friction actuation's corrected state derivative (friction_lib_dx).
WEIGHT_SIGMA_NPZ=<population_recovery.npz with sigma/keys_run>: push the weights as w / (sigma * c) by feature name, the way the fit's own
solver and replay apply them since CSQP fix 2 (6b7c2a2a, sigma-push); c = 0.5 Fmax^2 for the two-cost press terms (phi = cost / c), else 1.
Without it the weights go in raw, as the published transfer (and the pre-fix-2 fits) did.
usage: PYTHONPATH=<oss>/src python fwd_dx.py <run_species_forward.py args>"""
import os, sys, runpy
import numpy as np
import crocoddyl  # noqa: F401  (friction_lib binds against crocoddyl's types)
sys.path.insert(0, "os.environ.get('FRICTION_LIB_DX','friction_lib/build')")
import friction_lib
import run_csqp_identifiability
from final_models import human_crocoddyl as HC
print("[dx] friction_lib =", friction_lib.__file__, "| src =", run_csqp_identifiability.__file__, flush=True)
assert "friction_lib_dx" in friction_lib.__file__ and "/th_oss_transfer/" in run_csqp_identifiability.__file__
if os.environ.get("WEIGHT_SIGMA_NPZ"):
    _z = np.load(os.environ["WEIGHT_SIGMA_NPZ"], allow_pickle=True)
    SIG = dict(zip([str(k) for k in _z["keys_run"]], np.asarray(_z["sigma"], float)))
    def _div(self):
        c = lambda k: (0.5 * float(self.args.get('force_max', 80.0)) ** 2 + 1e-12
                       if k == 'press_peak' or (k in ('press_force', 'press_capacity') and self.args.get('force_two_cost', False)) else 1.0)
        miss = [k for k in self.keys_run if k not in SIG and not k.endswith('_thoracic')]
        if miss: raise ValueError("[sigma-div] no sigma for %s" % miss)
        return np.array([SIG.get(k, 1.0) * c(k) for k in self.keys_run])
    _tv, _const = HC.HumanCrocoddyl.update_solver_weights_tv, HC.HumanCrocoddyl.update_solver_weights
    def update_solver_weights_tv(self, w_run_windows, w_term_windows):
        return _tv(self, np.asarray(w_run_windows, float) / _div(self)[None, :], w_term_windows)
    def update_solver_weights(self, w_run, w_term):
        d = dict(zip(self.keys_run, _div(self)))
        return _const(self, {k: v / d.get(k, 1.0) for k, v in dict(w_run).items()}, w_term)
    HC.HumanCrocoddyl.update_solver_weights_tv = update_solver_weights_tv
    HC.HumanCrocoddyl.update_solver_weights = update_solver_weights
    print("[sigma-div] weights pushed as w/(sigma c) from", os.environ["WEIGHT_SIGMA_NPZ"], {k: round(v, 4) for k, v in SIG.items()}, flush=True)
sys.argv = ["morphologies_study/run_species_forward.py"] + sys.argv[1:]
runpy.run_path("${REPO_ROOT:-$(pwd)}/morphologies_study/run_species_forward.py", run_name="__main__")

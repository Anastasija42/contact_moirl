"""Time-varying recovery for the docs, 3-ACT cost where ALL THREE weights matter
in turn (so effort is genuinely identifiable, not ~0):
  progress high early (glide) -> force peaks mid (press) -> effort high late (ease off).
Recover with a windowed cost and a FINER Gaussian basis; plot recovered vs true."""
import os, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'src')
from toy_box_slide_irl import ToyBoxSlide, ema_forces
from MO_IRL import MO_IRL
from run_tv_weights import base_mo_args, recovered_Wt, compare

T, ITERS = 70, 15


def wstar_3act(T, keys, sharp=9.0):
    x = np.arange(T) / T
    s_mid = 1.0 / (1.0 + np.exp(-sharp * (x - 0.35)))   # rises ~35%
    s_late = 1.0 / (1.0 + np.exp(-sharp * (x - 0.65)))  # rises ~65%
    W = np.zeros((T, len(keys)))
    ei, fi, pi = keys.index('effort'), keys.index('force'), keys.index('progress')
    W[:, pi] = 0.3 + 1.9 * (1 - s_mid)     # progress: high early, decays after ~35%
    W[:, fi] = 0.1 + 2.0 * (s_mid - s_late)  # force: a BUMP peaking mid-stroke
    W[:, ei] = 0.1 + 1.6 * s_late          # effort: rises late (ease off)
    return W


toy = ToyBoxSlide(T=T, solref_time=0.1)
keys = toy.keys_run; nr = len(keys)
W_star = wstar_3act(T, keys)

toy.update_solver_weights_tv(W_star); toy._kin.reset()
xs, us = toy._kin.solve()
f = ema_forces(toy._kin._last_forces); toy.target_force_profile = f.copy()
print(f'3-act TV demo: x {xs[0,0]:.2f}->{xs[-1,0]:.2f} m, N {f[:T//3].mean():.0f}/{f[T//3:2*T//3].mean():.0f}/{f[2*T//3:].mean():.0f} N (early/mid/late)')


def recover(name, weight_mode, n_w, K=None):
    w0 = {k: 0.2 for k in keys}
    ia = {'model': toy, 'w_run': w0, 'w_term': {}, 'xs_opt': [xs], 'us_opt': [us],
          'irl_iter': ITERS, 'stopping': 'opt', 'tol': 1e-6, 'max_iter': ITERS,
          'min_iter': 1, 'compare_desired': False, 'verbose': False, 'n_w': n_w,
          'weight_mode': weight_mode}
    if weight_mode == 'basis':
        ia['K'] = K; ia['basis_type'] = 'gaussian'
    irl = MO_IRL(base_mo_args(toy, xs, us, n_w), ia); irl.solve()
    W_hat = recovered_Wt(irl, T, nr)
    rel, mcos, a = compare(W_hat, W_star)
    print(f'  {name}: rel_norm={rel:.3f}  mean_cos={mcos:.3f}  a={a:.3f}')
    return a * W_hat, mcos


res = {}
res['windowed (8 windows)'] = recover('windowed_n8', 'windowed', 8)
res['Gaussian basis (K=20)'] = recover('basis_K20', 'basis', 20, K=20)

t = np.arange(T)
C = {'effort': '#333333', 'force': '#D55E00', 'progress': '#009E73'}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
for ax, (label, (W_hat, mcos)) in zip(axes, res.items()):
    for j, k in enumerate(keys):
        ax.plot(t, W_star[:, j], color=C[k], lw=3.0, ls='--', alpha=0.55)
        ax.plot(t, W_hat[:, j], color=C[k], lw=2.2, label=k)
    ax.set_title(f'{label} — mean cos {mcos:.2f}', fontsize=11)
    ax.set_xlabel('time step'); ax.grid(alpha=0.3); ax.set_ylim(-0.3, 2.7)
axes[0].set_ylabel('cost weight (scaled to true)'); axes[0].legend(fontsize=9, loc='upper center', ncol=3)
fig.suptitle('Recovering a 3-act time-varying cost: progress → force → effort  (dashed = true, solid = recovered)', fontsize=12)
fig.tight_layout()
fig.savefig('docs/assets/figures/toy_box/toy_tv_recovery.png', dpi=150)
print('saved docs/assets/figures/toy_box/toy_tv_recovery.png')

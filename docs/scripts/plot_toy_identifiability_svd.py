"""IDENTIFIABILITY via SVD: the toy has 2 controls but 3 features, so the feature
counts live on a ~2D manifold — one weight combination is unobservable. Sample the
exploration around the demo, form the feature-count covariance (the IRL's Fisher
information / local posterior Hessian), and read off the near-null direction."""
import numpy as np, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'src')
from toy_box_slide_irl import ToyBoxSlide, DEMO_WSTARS

toy = ToyBoxSlide(T=70, solref_time=0.1)
keys = toy.keys_run                              # ['effort','force','progress']
toy.update_solver_weights(DEMO_WSTARS['mixed']); toy._kin.reset()
xs_demo, us_demo = toy._kin.solve()
U_demo = np.asarray(us_demo)

# sample the exploration: perturb the demo control (like MPPI noise) and roll out
K, SIGMA = 300, 10.0
rng = np.random.default_rng(0)
Phi = []
for k in range(K):
    U = U_demo + rng.normal(0, SIGMA, U_demo.shape)
    _, xs, us, _ = toy._kin._rollout(U, record_forces=True)
    Phi.append(np.asarray(toy.get_traj_features(xs, us)[0]))   # [0] = summed feature counts (nr,)
Phi = np.asarray(Phi)                            # (K, 3)

# standardise each feature (identifiability = collinearity of DIRECTIONS, not units)
Z = (Phi - Phi.mean(0)) / (Phi.std(0) + 1e-12)
C = Z.T @ Z / K                                  # = correlation matrix here
evals, evecs = np.linalg.eigh(C)                 # ascending

print('feature order:', keys)
print('correlation matrix:')
print(np.round(C, 3))
print(f'\neigenvalues (var explained along each principal weight axis): {np.round(evals,4)}')
print(f'condition number (lambda_max / lambda_min): {evals[-1]/max(evals[0],1e-12):.1f}')
u_dir = evecs[:, 0]
print('\nUNOBSERVABLE direction (smallest eigenvalue) — weights that trade off with ~no effect:')
for k, v in zip(keys, u_dir):
    print(f'   {k:9s}: {v:+.3f}')

# ── figure ──
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax[0].bar(range(3), evals[::-1], color=['#0072B2', '#0072B2', '#D55E00'])
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(['PC1\n(identifiable)', 'PC2', 'PC3\n(unobservable)'])
ax[0].set_ylabel('variance along axis (eigenvalue)')
ax[0].set_title(f'Feature-count spectrum — condition number {evals[-1]/max(evals[0],1e-12):.0f}')
ax[0].grid(axis='y', alpha=0.3)
# the unobservable combination as a bar of weights
ax[1].bar(range(3), u_dir, color=['#333333', '#D55E00', '#009E73'])
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(keys)
ax[1].axhline(0, color='k', lw=0.8)
ax[1].set_ylabel('weight in the unobservable combination')
ax[1].set_title('The direction the demo CANNOT pin down')
ax[1].grid(axis='y', alpha=0.3)
fig.tight_layout(); fig.savefig('docs/assets/figures/toy_box/toy_identifiability_svd.png', dpi=150)
print('\nsaved docs/assets/figures/toy_box/toy_identifiability_svd.png')

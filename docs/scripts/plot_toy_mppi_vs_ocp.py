"""Side-by-side recovery of the three force treatments, MPPI (sampling) vs the
exact-gradient OCP (CSQP). Numbers come from:
  MPPI: experiments/run_toy_ablation.py      --only 08_force_tracked,09_force_recovered,10_force_imposed
  OCP : experiments/run_toy_csqp_ablation.py --only 08_force_tracked,09_force_recovered,10_force_imposed
"""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

modes = ['tracked', 'recovered', 'imposed']
mppi = [0.99, 0.92, 0.81]      # cosine to true w*  (run_toy_ablation, 15 iters)
ocp  = [1.00, 0.57, 1.00]      # cosine to true w*  (run_toy_csqp_ablation)
x = np.arange(3); w = 0.36
fig, ax = plt.subplots(figsize=(8.2, 4.6))
b1 = ax.bar(x - w/2, mppi, w, label='MPPI (sampling)', color='#0072B2', ec='k')
b2 = ax.bar(x + w/2, ocp,  w, label='OCP / CSQP (exact gradient)', color='#E69F00', ec='k')
for b, v in list(zip(b1, mppi)) + list(zip(b2, ocp)):
    ax.text(b.get_x() + b.get_width()/2, v + 0.015, f'{v:.2f}', ha='center', fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(['TRACKED\n|f-f_ref|^2', 'RECOVERED\nf^2 + (f-f_cap)^2', 'IMPOSED\nforce known, fix it'])
ax.set_ylabel('recovery cosine to true w*'); ax.set_ylim(0, 1.12)
ax.axhline(1.0, color='gray', ls=':', lw=1)
ax.set_title('Force treatment x solver - toy recovery (MPPI vs exact OCP)')
ax.legend(loc='lower left', fontsize=9); ax.grid(axis='y', alpha=0.3)
fig.tight_layout(); fig.savefig('docs/assets/figures/toy_box/toy_mppi_vs_ocp.png', dpi=150)
print('saved docs/assets/figures/toy_box/toy_mppi_vs_ocp.png')

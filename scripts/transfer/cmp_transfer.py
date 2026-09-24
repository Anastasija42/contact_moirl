"""Transfer rerun vs the published run, per body: joint RMSE between the two solutions, KKT, contact force, and the realised cost
|w.phi| in the paper's groups (effort = torque + energy + joint velocity; smoothness = JA + JTC (+Geo); contact = press terms; task).
usage: cmp_transfer.py <stroke> <new root (runs_dx/<tag>)> [published root]"""
import os, sys, os, numpy as np
S = sys.argv[1]; NEW = sys.argv[2]
PUB = sys.argv[3] if len(sys.argv) > 3 else os.environ.get('PUBLISHED_TRANSFER_DIR', '')
ORDER = ['bonobo', 'chimp', 'australopithecus_prometheus', 'australopithecus_sediba', 'homo_naledi', 'homo_neanderthal', 'human']
G = [('effort', ('Tau_', 'Eng_', 'JV')), ('smooth', ('JA', 'JTC', 'Geo')), ('contact', ('press_', 'force')), ('task', ('progress', 'rail', 'rock'))]
def load(root, sp):
    f = os.path.join(root, sp, '%s__%s' % (sp, S), 'forward.npz')
    return np.load(f, allow_pickle=True) if os.path.exists(f) else None
def groups(z):
    c = dict(zip([str(k) for k in z['keys']], np.asarray(z['cost_contrib'], float))); t = sum(abs(v) for v in c.values())
    return [100 * sum(abs(v) for k, v in c.items() if any(k.startswith(p) for p in ps)) / t for _, ps in G], t
hp, hn = load(PUB, 'human'), load(NEW, 'human')
div = lambda z, h: np.degrees(np.sqrt(np.mean((z['xs'][:, 1:int(z['nq'])] - h['xs'][:, 1:int(z['nq'])]) ** 2))) if h is not None else float('nan')
print('%-28s %8s %8s | %5s %5s | %6s | %13s | %-22s %-22s | %s' % ('body', 'KKT pub', 'KKT new', 'F pub', 'F new', 'dq deg', 'vs human p->n', 'pub eff/smo/con/task', 'new eff/smo/con/task', 'total |w.phi| pub -> new'))
for sp in ORDER:
    a, b = load(PUB, sp), load(NEW, sp)
    if a is None or b is None: continue
    nq = int(a['nq']); dq = np.degrees(np.sqrt(np.mean((a['xs'][:, 1:nq] - b['xs'][:, 1:nq]) ** 2)))   # thorax (q0) static
    fa, fb = np.linalg.norm(a['f_contact'], axis=1), np.linalg.norm(b['f_contact'], axis=1)
    (ga, ta), (gb, tb) = groups(a), groups(b)
    print('%-28s %8.2f %8.2f | %5.1f %5.1f | %6.2f | %5.1f -> %5.1f | %-22s %-22s | %.0f -> %.0f' % (sp, float(a['kkt']), float(b['kkt']), fa[fa > 1].mean(), fb[fb > 1].mean(), dq, div(a, hp), div(b, hn),
          '/'.join('%.0f' % x for x in ga), '/'.join('%.0f' % x for x in gb), ta, tb))

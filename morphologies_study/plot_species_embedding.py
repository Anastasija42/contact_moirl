#!/usr/bin/env python
"""
Cross-morphology embedding + cost-contribution plots for the morphology paper.

Reads the per-species forward.npz produced by run_species_forward.py
(fields: keys, w, phi, cost_contrib, xs, nq, kkt) and produces:

  1) species_embedding.png  -- two 2D embeddings, the human centred at (0,0):
        (A) COST-CONTRIBUTION space  (tactic: where each body PAYS the fixed cost)
        (B) KINEMATIC space          (morphology: how each body MOVES vs human)
     Distance from the human = closeness. Down/up strokes overlaid (filled/open),
     linked per body. The point: a body can sit near in one space and far in the
     other -> the two axes (morphology vs tactic) are distinct.

  2) species_cost_contrib.png -- per-(group) realised-cost heatmap, body vs human,
     showing WHICH objectives each morphology pays differently (the explanation).

Usage:
  python plot_species_embedding.py --indir analysis/species_experiments \
      --tasks down_long up_long --method mds --out figures/morpho
"""
import os, argparse, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLADE_COLOR = {
    "human": "#000000", "homo_neanderthal": "#009E73", "homo_naledi": "#0072B2",
    "australopithecus_sediba": "#E69F00", "australopithecus_prometheus": "#56B4E9",
    "stw573": "#56B4E9", "chimp": "#D55E00", "bonobo": "#CC79A7",
}
PRETTY = {
    "human": "Modern human", "homo_neanderthal": "Neanderthal", "homo_naledi": "H. naledi",
    "australopithecus_sediba": "A. sediba", "australopithecus_prometheus": "A. prometheus",
    "chimp": "Chimpanzee", "bonobo": "Bonobo",
}
ORDER = ["bonobo", "chimp", "australopithecus_prometheus", "australopithecus_sediba",
         "homo_naledi", "homo_neanderthal", "human"]
def feat_group(k):
    if k.startswith("Tau") or k.startswith("Eng"): return "effort"
    if k in ("JV", "JA", "JTC", "Geo"):            return "smoothness"
    if k in ("press_force", "surface", "approach", "rock_ori", "rail_lat"): return "contact"
    if k in ("progress_vel", "traveled"):          return "progress"
    return "other"

def load_species(indir, task, ref="human", drop_diverged=True):
    """Return {species: forward.npz dict} for one task. Rollouts whose OCP did not
    converge (huge KKT) carry a non-physical contact force that would dominate the
    cost-contribution embedding, so they are dropped (the human anchor is kept with
    a warning, since the whole space is centred on it)."""
    out = {}
    for p in glob.glob(os.path.join(indir, f"*__{task}", "forward.npz")):
        sp = os.path.basename(os.path.dirname(p)).split("__")[0]
        d = np.load(p, allow_pickle=True)
        rec = {k: d[k] for k in d.files}
        tf = float(rec["target_force"]) if "target_force" in rec else 40.0
        pk = float(np.linalg.norm(rec["f_contact"], axis=1).max()) if "f_contact" in rec else 0.0
        kkt = float(rec["kkt"]) if "kkt" in rec else float("nan")
        if drop_diverged and (pk > 10 * tf or (np.isfinite(kkt) and kkt > 1e8)):
            if sp == ref:
                print(f"[warn] REFERENCE '{ref}' diverged for {task} "
                      f"(peakF={pk:.0f}N); this task's embedding is unreliable")
            else:
                print(f"[drop] {sp} diverged for {task} (peakF={pk:.0f}N)"); continue
        out[sp] = rec
    return out

def embed(M, method="mds", seed=0):
    """M: (n_species, n_dims) feature matrix -> (n_species, 2) coords."""
    Mz = (M - M.mean(0)) / (M.std(0) + 1e-9)
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(Mz)
    from sklearn.manifold import MDS
    return MDS(n_components=2, dissimilarity="euclidean",
               random_state=seed, normalized_stress="auto").fit_transform(Mz)

def per_joint_rmse(xs_sp, xs_hu, nq):
    """Per-joint RMSE (deg) of a species' joint trajectory vs the human's."""
    n = min(int(nq), xs_sp.shape[1], xs_hu.shape[1])
    T = min(xs_sp.shape[0], xs_hu.shape[0])
    d = np.rad2deg(xs_sp[:T, :n] - xs_hu[:T, :n])
    return np.sqrt((d ** 2).mean(0))

def build_matrices(data, ref="human"):
    """From {species: npz} build cost-contribution and kinematic-divergence
    matrices, aligned to the COMMON feature keys across all species (older runs
    may have dropped features for some bodies)."""
    species = [s for s in data if s != ref]
    species = ([ref] if ref in data else []) + sorted(species)
    keysets = [set(str(k) for k in data[s]["keys"]) for s in species]
    common = sorted(set.intersection(*keysets))
    def vec(s):
        kmap = {str(k): c for k, c in zip(data[s]["keys"],
                                          np.asarray(data[s]["cost_contrib"], float))}
        return np.array([abs(kmap[k]) for k in common])
    C = np.array([vec(s) for s in species])
    xs_hu = np.asarray(data[ref]["xs"], float)
    nq = int(np.asarray(data[ref]["nq"]))
    K = np.array([per_joint_rmse(np.asarray(data[s]["xs"], float), xs_hu, nq)
                  for s in species])
    return species, common, C, K

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="analysis/species_experiments")
    ap.add_argument("--tasks", nargs="+", default=["down_long", "up_long"])
    ap.add_argument("--method", default="pca", choices=["pca", "mds"])
    ap.add_argument("--ref", default="human")
    ap.add_argument("--out", default="figures/morpho")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    markers = {args.tasks[0]: ("o", "down"), }
    if len(args.tasks) > 1:
        markers[args.tasks[1]] = ("D", "up")
    coords_by_task = {}
    for task in args.tasks:
        data = load_species(args.indir, task)
        if args.ref not in data:
            print(f"[warn] no '{args.ref}' for task {task}; skipping"); continue
        species, keys, C, K = build_matrices(data, args.ref)
        emC = embed(C, args.method); emK = embed(K, args.method)
        emC -= emC[species.index(args.ref)]; emK -= emK[species.index(args.ref)]
        coords_by_task[task] = (species, emC, emK)
        mk = markers.get(task, ("o", task))[0]
        for em, ax, ttl in [(emC, axes[0], "Cost-contribution space (tactic)"),
                            (emK, axes[1], "Kinematic space (morphology)")]:
            for i, s in enumerate(species):
                c = CLADE_COLOR.get(s, "#888888")
                star = (s == args.ref)
                ax.scatter(*em[i], s=320 if star else 170, color=c,
                           marker="*" if star else mk,
                           edgecolor="k", linewidth=0.7, zorder=3, label=None)
                ax.annotate(PRETTY.get(s, s), em[i], fontsize=8.5,
                            xytext=(5, 5), textcoords="offset points")
            ax.set_title(ttl, fontsize=12, pad=8)
            ax.axhline(0, color="grey", lw=0.6, alpha=0.4)
            ax.axvline(0, color="grey", lw=0.6, alpha=0.4)
            ax.set_xlabel("dim 1", fontsize=10); ax.set_ylabel("dim 2", fontsize=10)
            ax.margins(0.18)
    if len(coords_by_task) == 2:
        (sp0, c0, k0), (sp1, c1, k1) = coords_by_task.values()
        common = [s for s in sp0 if s in sp1]
        for em_a, em_b, ax in [(c0, c1, axes[0]), (k0, k1, axes[1])]:
            for s in common:
                a, b = em_a[sp0.index(s)], em_b[sp1.index(s)]
                ax.plot([a[0], b[0]], [a[1], b[1]], color=CLADE_COLOR.get(s, "#888"),
                        lw=0.8, alpha=0.5, zorder=1)
    from matplotlib.lines import Line2D
    taxa = [s for s in ORDER if any(s in coords_by_task[t][0] for t in coords_by_task)]
    tax_handles = [Line2D([0], [0], marker="*" if s == args.ref else "o", color="w",
                          markerfacecolor=CLADE_COLOR.get(s, "#888"),
                          markeredgecolor="k", markersize=13 if s == args.ref else 10,
                          label=PRETTY.get(s, s)) for s in taxa]
    stroke_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
               markeredgecolor="k", markersize=10, label=f"{args.tasks[0]} (down)")]
    if len(args.tasks) > 1:
        stroke_handles.append(
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#555",
                   markeredgecolor="k", markersize=9, label=f"{args.tasks[1]} (up)"))
    leg1 = fig.legend(handles=tax_handles, loc="lower center", ncol=len(taxa),
                      fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.add_artist(leg1)
    fig.legend(handles=stroke_handles, loc="lower center", ncol=2, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Morphology vs tactic — distance from the human (★) is closeness; "
                 f"{args.method.upper()} embedding, strokes linked per taxon",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    p1 = os.path.join(args.out, "species_embedding.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight"); print("saved", p1)

    groups = ["effort", "smoothness", "contact", "progress"]
    for task in args.tasks:
        data = load_species(args.indir, task)
        if args.ref not in data:
            print(f"[fig2] no '{args.ref}' for {task}; skip"); continue
        sp_raw, keys, C, _ = build_matrices(data, args.ref)
        G = np.zeros((len(sp_raw), len(groups)))
        for j, k in enumerate(keys):
            g = feat_group(k)
            if g in groups:
                G[:, groups.index(g)] += C[:, j]
        G = G / (G.sum(1, keepdims=True) + 1e-9)
        D = G - G[sp_raw.index(args.ref)]
        order = [sp_raw.index(s) for s in ORDER if s in sp_raw]
        D = D[order]; rows = [sp_raw[i] for i in order]
        vmax = np.abs(D).max() or 1e-9
        fig2, ax = plt.subplots(figsize=(7.5, 0.7 * len(rows) + 1.6))
        im = ax.imshow(D, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g.capitalize() for g in groups], fontsize=10)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([PRETTY.get(s, s) for s in rows], fontsize=10)
        for i in range(len(rows)):
            for j in range(len(groups)):
                val = D[i, j]
                txtc = "white" if abs(val) > 0.6 * vmax else "k"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=9, color=txtc)
        ax.set_title(f"Where each body pays the human cost differently — {task}\n"
                     "red = pays MORE than the human here, blue = less",
                     fontsize=11.5, pad=8)
        fig2.colorbar(im, label="Δ cost share", fraction=0.046, pad=0.04)
        fig2.tight_layout()
        p2 = os.path.join(args.out, f"species_cost_contrib__{task}.png")
        fig2.savefig(p2, dpi=150, bbox_inches="tight"); print("saved", p2)
        plt.close(fig2)

if __name__ == "__main__":
    main()

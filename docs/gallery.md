---
layout: default
title: Gallery
---

# Gallery

The parts of this project that don't fit in the paper: **interactive explorers** you
open and drive in the browser, and **looping clips** of the toy contact problem, the
cross-morphology transfer, and the force sweeps. Static figures and per-subject GIFs
live on the [Results](results), [Cross-Morphology](morphology), and
[Motion Previews](motion_previews) pages.

<style>
.vgrid { display: flex; flex-wrap: wrap; gap: 14px; justify-content: center; margin: 14px 0; }
.vcard { width: 300px; }
.vcard video { width: 100%; border-radius: 6px; background: #111; }
.vcard .cap { font-size: 0.82em; color: #555; margin-top: 4px; text-align: center; }
.vwide { max-width: 680px; margin: 0 auto; }
.vwide video { width: 100%; border-radius: 6px; background: #111; }

/* interactive-explorer cards — the artifacts a paper can't hold */
.xgrid { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:16px; margin:18px 0 6px; }
.xcard { display:block; text-decoration:none !important; background:#fff; border:1px solid #e2e8f0;
  border-radius:12px; padding:18px; transition:transform .12s ease, box-shadow .12s ease, border-color .12s ease; }
.xcard:hover { transform:translateY(-2px); box-shadow:0 10px 26px rgba(37,99,235,.10);
  border-color:#c7d6f5; text-decoration:none !important; }
.xcard .eyebrow { font-family:'Fira Code',monospace; font-size:11px; letter-spacing:.05em;
  text-transform:uppercase; color:#2563eb; }
.xcard h3 { margin:.35rem 0 .35rem !important; font-size:1.1rem; color:#1e293b !important;
  border:none !important; padding:0 !important; }
.xcard p { margin:0; font-size:.9rem; line-height:1.55; color:#475569; }
.xcard .go { display:inline-block; margin-top:.75rem; font-size:.82rem; font-weight:600; color:#2563eb; }
</style>

---

## Interactive explorers

The parts of this project that don't fit on a page — open, drag, and step through
them in the browser. Each is a self-contained instrument panel.

<div class="xgrid">
  <a class="xcard" href="{{ site.baseurl }}/arm_inspector.html">
    <div class="eyebrow">MPPI · sampling solver</div>
    <h3>Arm Inspector</h3>
    <p>Step the MPPI-IRL solver through its iterations on the seated-arm recovery — the sampled trajectory cloud, the challenger, and the weights converging.</p>
    <span class="go">Open explorer →</span>
  </a>
  <a class="xcard" href="{{ site.baseurl }}/ocp_inspector.html">
    <div class="eyebrow">Crocoddyl · analytical solver</div>
    <h3>OCP Arm Inspector</h3>
    <p>The same recovery driven by the analytical CSQP inner solver — the clean-instrument counterpart for comparing the two optimizers side by side.</p>
    <span class="go">Open explorer →</span>
  </a>
  <a class="xcard" href="{{ site.baseurl }}/reach_planner.html">
    <div class="eyebrow">Toy · planner-dependence</div>
    <h3>Whose preferences?</h3>
    <p>A force-free reaching toy: one demonstration, several planners, and recovered preferences that come out different for each — planner-dependence made tangible.</p>
    <span class="go">Open explorer →</span>
  </a>
</div>

<div class="xgrid" style="grid-template-columns:1fr;margin-top:14px">
  <a class="xcard" href="{{ site.baseurl }}/redundancy_case_study.html">
    <div class="eyebrow">Case study · identifiability</div>
    <h3>Why won't the elbow match?</h3>
    <p>A redundant arm's elbow deploys 6° off the human's — is the cost wrong, or can the controller just not express the posture on-rail? Reading the rollout cloud tells the two apart, and traces the gap to a geometric rail leak with a clean fix.</p>
    <span class="go">Read the case study →</span>
  </a>
</div>

---

## Toy contact problem

A small, fully-known box-slide MPPI-IRL problem we use to validate the pipeline
against ground truth before trusting it on real demonstrations. The same box is
asked to **glide**, **press**, and do a **mixed** glide-then-press — three
qualitatively different optimal behaviours that the recovered weights have to tell
apart. See the [Identifiability](identifiability) page for the box-slide ablations.

<div class="vgrid">
  <div class="vcard"><video src="assets/videos/toy/glide.mp4" autoplay loop muted playsinline></video><div class="cap"><b>Glide</b> — slide to target, no normal force.</div></div>
  <div class="vcard"><video src="assets/videos/toy/press.mp4" autoplay loop muted playsinline></video><div class="cap"><b>Press</b> — hold position, build normal force.</div></div>
  <div class="vcard"><video src="assets/videos/toy/mixed.mp4" autoplay loop muted playsinline></video><div class="cap"><b>Mixed</b> — glide to target, then press.</div></div>
</div>

---

## Seven hominins, one cost

The headline cross-morphology result: a **single human-recovered cost**, re-solved
on seven upper-limb body plans performing the same long down-stroke. Identical
preferences, different bodies — the motion diverges by morphology alone.

<div class="vwide">
  <video src="assets/videos/transfer/grid.mp4" autoplay loop muted playsinline></video>
  <div class="cap" style="text-align:center;font-size:0.82em;color:#555;">Human, Neanderthal, <em>H. naledi</em>, <em>A. sediba</em>, <em>A. prometheus</em>, chimpanzee, bonobo — one fixed cost, long down-stroke.</div>
</div>

Per-body, against a moving stick all seven hold the demonstration's contact force
(mean ≈ 37 N, peak ≈ 40 N) and essentially close the reach (reach-gap < 1 cm),
so the divergence shows up in *how* each body organises the stroke rather than in
whether it can do the task at all.

<div class="vgrid">
  <div class="vcard"><video src="assets/videos/transfer/human.mp4" autoplay loop muted playsinline></video><div class="cap">Modern human</div></div>
  <div class="vcard"><video src="assets/videos/transfer/homo_neanderthal.mp4" autoplay loop muted playsinline></video><div class="cap">Neanderthal</div></div>
  <div class="vcard"><video src="assets/videos/transfer/homo_naledi.mp4" autoplay loop muted playsinline></video><div class="cap"><em>H. naledi</em></div></div>
  <div class="vcard"><video src="assets/videos/transfer/australopithecus_sediba.mp4" autoplay loop muted playsinline></video><div class="cap"><em>A. sediba</em></div></div>
  <div class="vcard"><video src="assets/videos/transfer/australopithecus_prometheus.mp4" autoplay loop muted playsinline></video><div class="cap"><em>A. prometheus</em></div></div>
  <div class="vcard"><video src="assets/videos/transfer/chimp.mp4" autoplay loop muted playsinline></video><div class="cap">Chimpanzee</div></div>
  <div class="vcard"><video src="assets/videos/transfer/bonobo.mp4" autoplay loop muted playsinline></video><div class="cap">Bonobo</div></div>
</div>

---

## Press-force sweep

The same transfer, but with the press-force target swept from 40 N up to 100 N.
Most bodies track the requested force across the whole range — but the sweep also
exposes where morphology *limits* the task. At 60 N and above, **<em>A. prometheus</em>
drops the press entirely** (mean force collapses to ≈ 5 N), and at 80 N the
**Neanderthal** can no longer hold a clean press (mean ≈ 35 N against an 80 N
request). High force is where the weaker / less favourably-levered bodies fail
first.

<div class="vgrid">
  <div class="vcard"><video src="assets/videos/press_sweep/grid_40N.mp4" autoplay loop muted playsinline></video><div class="cap"><b>40 N</b> — all seven hold the press.</div></div>
  <div class="vcard"><video src="assets/videos/press_sweep/grid_60N.mp4" autoplay loop muted playsinline></video><div class="cap"><b>60 N</b> — <em>A. prometheus</em> drops out.</div></div>
  <div class="vcard"><video src="assets/videos/press_sweep/grid_80N.mp4" autoplay loop muted playsinline></video><div class="cap"><b>80 N</b> — Neanderthal press weakens too.</div></div>
  <div class="vcard"><video src="assets/videos/press_sweep/grid_100N.mp4" autoplay loop muted playsinline></video><div class="cap"><b>100 N</b> — survivors still track ~100 N.</div></div>
</div>

---

## Fixed vs. moving stick

Whether the workpiece is **pinned** or **held in the other hand and free to move**
changes the difficulty completely. Against a *fixed* stick the non-human bodies
leave large reach gaps (Neanderthal ≈ 21 cm, *H. naledi* ≈ 36 cm) — they simply
can't get the rock to the stick. Letting the stick move (the realistic two-handed
case) lets every body close the contact (reach-gap < 1 cm), which is the condition
under which the cross-morphology comparison is fair.

<div class="vgrid">
  <div class="vcard"><video src="assets/videos/stick/grid_fixed.mp4" autoplay loop muted playsinline></video><div class="cap"><b>Fixed stick</b> — apes/archaics fall short of contact.</div></div>
  <div class="vcard"><video src="assets/videos/stick/grid_moving.mp4" autoplay loop muted playsinline></video><div class="cap"><b>Moving stick</b> — all seven reach and press.</div></div>
</div>

---

[Home](.) | [Method](method) | [Identifiability](identifiability) | [Cross-Morphology](morphology) | [Results](results) | [Code](code)

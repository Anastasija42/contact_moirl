---
layout: default
title: Gallery
---

# Gallery

A collection of the project's most illustrative clips — the toy contact problem we
debug on, the headline cross-morphology transfer, and the sweeps that stress-test
how the recovered cost behaves on different bodies and at different force levels.
Every clip loops; the static figures and per-subject GIFs live on the
[Results](results), [Cross-Morphology](morphology), and [Motion Previews](motion_previews)
pages.

<style>
.vgrid { display: flex; flex-wrap: wrap; gap: 14px; justify-content: center; margin: 14px 0; }
.vcard { width: 300px; }
.vcard video { width: 100%; border-radius: 6px; background: #111; }
.vcard .cap { font-size: 0.82em; color: #555; margin-top: 4px; text-align: center; }
.vwide { max-width: 680px; margin: 0 auto; }
.vwide video { width: 100%; border-radius: 6px; background: #111; }
</style>

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

[Home](index) | [Method](method) | [Identifiability](identifiability) | [Cross-Morphology](morphology) | [Results](results) | [Motion Previews](motion_previews) | [Code](code)

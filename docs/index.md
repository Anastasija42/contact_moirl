---
layout: default
title: "Learning Tool Handling from Human Demonstrations"
show_tagline: true
---

## What this is

The code and the cross-morphology viewer for two studies of Palaeolithic tool
use, carried out with the
[Anthrotopography Lab (Prof. Radu Iovita)](https://wp.nyu.edu/csho/research/laboratories/anthrotopography_laboratory/).

A person holds a stone and shaves a wooden stick: a sustained-contact movement
that combines a controlled pressing force with a directed stroke. From a few
recorded cycles we infer the motor cost that best explains the observed
movement, capturing the trade-off between effort, smoothness, contact force and
pace. We then hold that cost fixed and re-solve the same task on a different
body, so that what changes in the resulting movement is attributable to
morphology alone.

Three things are published here:

- [**Code**](code.html) is the released repository and what each of its entry points
  reproduces.
- [**Contact it was never told about**](sampler_inspector.html) shows the
  result the method rests on: one recovered objective deployed cold on cycles it
  never saw, with the contact it produces set against the contact that was
  recorded. Nothing about when the tool touches the stone was supplied.
- [**Seven bodies, one cost**](species_inspector.html) is the interactive
  viewer: the recovered human cost re-solved on seven hominin upper limbs
  (modern human, Neanderthal, *H. naledi*, *A. sediba*, *A. prometheus*,
  chimpanzee, bonobo), with the reach, the joint paths and the cost composition
  each body produces.

The recordings themselves are not released. What the repository ships is what
the studies are built on: the joint-angle trajectories fitted to the
recordings, the per-cycle force-sensor slices, the recovered costs, and the
models. Participants appear only as S1, S2 and S3.

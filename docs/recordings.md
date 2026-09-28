---
layout: default
title: Recordings
---

# Recordings

Three right-handed participants shaved a wooden stick with a hafted stone, recorded with optical
motion capture and, in one session, an instrumented handle. Every cycle was fitted to that
participant's scaled body model; those joint-angle trajectories are what the repository ships and what
every result is measured against. Participants appear only as S1, S2 and S3.

The panels below are the eight arm joints across all cycles of one participant and stroke: the mean in
solid blue, one standard deviation shaded, and each individual cycle as a faint line. They are the
reason the method has to work from few demonstrations and tolerate spread, and they show where that
spread actually is.

Two things are visible in all of them. The proximal joints repeat tightly, a degree or less across a
whole session, while the distal ones do not: elbow pronation and both wrist axes spread five to ten
degrees at the same instant of the stroke. And the spread is largest at the start of the cycle, where
the tool is being placed, and closes as the stroke proceeds.

## One cycle each

The recorded cycle replayed on that participant's scaled body model, down stroke, session `13_02`.

| S1 | S2 | S3 |
|---|---|---|
| ![S1]({{ site.baseurl }}/assets/motion/S1_down.gif) | ![S2]({{ site.baseurl }}/assets/motion/S2_down.gif) | ![S3]({{ site.baseurl }}/assets/motion/S3_down.gif) |

## How well the model fits the markers

The trajectories come from inverse kinematics on the marker set, in two passes: a full-body fit, then
an arm-only pass that recovers arm accuracy the full-body fit trades away for the torso and head. A
good fit is a mean per-marker RMSE under 2 cm, and the capture system's own precision is about 0.5 cm,
so 1 to 2 cm after IK is what one should expect. Hand and finger markers are the worst, often above
5 cm, from skin slip and occlusion rather than registration error. Between sessions the absolute joint
angles differ by a constant offset, because the calibration pose differs, while stroke shape and
amplitude agree.

## Variability, down stroke

| | |
|---|---|
| **S1** | ![S1 down]({{ site.baseurl }}/assets/recordings/S1_down.png) |
| **S2** | ![S2 down]({{ site.baseurl }}/assets/recordings/S2_down.png) |
| **S3** | ![S3 down]({{ site.baseurl }}/assets/recordings/S3_down.png) |

## Variability, up stroke

| | |
|---|---|
| **S1** | ![S1 up]({{ site.baseurl }}/assets/recordings/S1_up.png) |
| **S2** | ![S2 up]({{ site.baseurl }}/assets/recordings/S2_up.png) |
| **S3** | ![S3 up]({{ site.baseurl }}/assets/recordings/S3_up.png) |

The up strokes are the shorter, faster half of the motion and repeat less tightly. S1's are
unusually quick, a median of 407 ms against 1266 and 1618 ms for the other two, and against 1112 ms
for her own down strokes.

These plots ship with the data, one pair per participant, session and stroke, under
`trajectories_from_mocap/<session>/<subject>/<stroke>/plots/`.

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

## Down stroke

| | |
|---|---|
| **S1** | ![S1 down]({{ site.baseurl }}/assets/recordings/S1_down.png) |
| **S2** | ![S2 down]({{ site.baseurl }}/assets/recordings/S2_down.png) |
| **S3** | ![S3 down]({{ site.baseurl }}/assets/recordings/S3_down.png) |

## Up stroke

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

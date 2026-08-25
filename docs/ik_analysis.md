---
layout: default
title: IK & Marker Registration
---

# Mocap → Model Pipeline

This page documents the inverse-kinematics fit of the subject-scaled Pinocchio
models to the OptiTrack mocap data. Each subject was recorded on **two
separate sessions** (13 Feb and 27 Feb), and both sessions were registered
independently — so per-subject results appear twice, and the comparison is
itself informative.

## Pipeline overview

```
anthropometry JSON ──► scale URDF ──► seed q_calib upright ──► slider-pose to relaxed stance ──►
    derive marker offsets (p_local = oMᵢ⁻¹ · p_world) ──► save markers.json ──►
    two-pass IK (full-body, then arm-only → merged) ──► q_traj + per-pass RMSE
```

1. **Measure anthropometry once per subject.** A relaxed-stance mocap frame is used to extract per-subject segment lengths (torso, upper arms, forearms, shoulder/pelvis widths) and saved to `config/subject_anthropometry.json`. Later runs read this file rather than re-measuring from whatever frame happens to be first — avoids picking up hunched/moving frames.

2. **Scale the URDF** from the anthropometry entry. Each subject's URDF + mesh scales are written to `config/scaled_registered_models/<date>/<subject>.urdf`.

3. **Derive marker offsets from a relaxed stance frame.** The model is seeded upright with the pelvis at the mocap pelvis centre, optionally slider-tuned so the model pose matches the subject's stance in the chosen frame, then each marker's local offset is computed as `p_local = oMᵢ⁻¹ · p_world`. Saved to `<subject>_markers.json`. Implemented in `calibrate_from_stance.py`.

4. **IK per frame — two passes.** `run_ik_trajectories.py` runs IK twice:
   - **Full-body pass**: 16 markers (pelvis, torso, head, arms) with a QP solver, warm-started per frame from the previous frame.
   - **Arm-only pass**: second IK using only the five right-arm markers, warm-started per frame from the full-body result.

   In **merged** mode (default) the arm-only DOFs overwrite the full-body arm DOFs — the arms get the tighter arm-only fit while the rest of the body keeps the full-body fit.

5. **Optional marker refinement** (`--refine-iters N`). Alternates IK and marker-offset updates: each iteration averages the per-frame residual in each marker's local frame, applies 90% of that delta as a correction, and reruns IK. Stops at 2 mm mean delta. Writes refined offsets to `<subject>_markers_refined_<task>.json` when `--save-refined-markers` is set.

6. **Batch driver** (`batch_run_ik.py`) runs every subject × date × task combination. CSV-prefix quirks (e.g. `S2` instead of `S2` in 27_02 files) are handled via a small overrides dict at the top of the script.

Output layout per subject + session + task:

```
trajectories_from_mocap/<DATE>/<SUBJECT>/<TASK>/
  ik_joint_angles.csv           # q(t), one row per mocap frame
  ik_marker_trajectories.csv    # model-side marker positions (x,y,z per marker)
  ik_rmse_full_body.csv         # fit error per marker from the full-body pass (cm)
  ik_rmse_arm_only.csv          # fit error per marker from the arm-only pass (cm)
  stick_stone_trajectories.csv  # rigid-body stick + rock positions
  stick_stone_analysis.png      # contact-geometry time series
  elaborated/
    down/cycle_NN.npz           # per-stroke slices (q, dq, ddq, dt, times)
    up/cycle_NN.npz
    meta.npz
```

A parallel tree `trajectories_from_mocap_sensor/` holds the `take2_sensor` runs
(27 Feb only, same layout) — same pipeline, recordings with the force sensor
mounted.

## IK fit quality — full table

See [ik_fit_table.md](ik_fit_table.md) for mean/max per-marker RMSE (cm)
across all subjects × sessions × tasks.

## Tracking quality at a glance

Mean per-marker RMSE across all tasks, all subjects, both sessions:

![RMSE heatmap all]({{ '/assets/ik_analysis/rmse_heatmap_all.png' | relative_url }})

Darker = worse. Hand/finger markers dominate; pelvis and head are tight.

## Full-body vs arm-only IK

The pipeline runs IK twice: first on all 16 markers, then a second pass
tracking only the five right-arm markers. For arm markers, the arm-only
pass should be tighter (the solver isn't trading arm fit for torso/head fit).

- S2: [13_02]({{ '/assets/ik_analysis/S2_13_02_fullbody_vs_arm.png' | relative_url }}) · [27_02]({{ '/assets/ik_analysis/S2_27_02_fullbody_vs_arm.png' | relative_url }})
- S3: [13_02]({{ '/assets/ik_analysis/S3_13_02_fullbody_vs_arm.png' | relative_url }}) · [27_02]({{ '/assets/ik_analysis/S3_27_02_fullbody_vs_arm.png' | relative_url }})
- S1: [13_02]({{ '/assets/ik_analysis/S1_13_02_fullbody_vs_arm.png' | relative_url }}) · [27_02]({{ '/assets/ik_analysis/S1_27_02_fullbody_vs_arm.png' | relative_url }})

## With vs without sensor (27 Feb only)

27 Feb has two takes per task — `take1` without the sensor, `take2_sensor` with it.
Comparison shows whether the sensor perturbs marker tracking.

- S2: ![S2 sensor]({{ '/assets/ik_analysis/S2_27_02_sensor_vs_nosensor.png' | relative_url }})
- S3: ![S3 sensor]({{ '/assets/ik_analysis/S3_27_02_sensor_vs_nosensor.png' | relative_url }})
- S1: ![S1 sensor]({{ '/assets/ik_analysis/S1_27_02_sensor_vs_nosensor.png' | relative_url }})

### Cycle variability bands — sensor vs no-sensor

Mean ± 1 SD across detected cycles for the four right-arm DOFs, split by DOWN
and UP phase. Blue = no sensor, orange = sensor. A tight overlap means the
sensor didn't perturb the movement; a shifted band means it did.

- S2 — [down_long]({{ '/assets/ik_analysis/S2_27_02_down_long_cycles_sensor_vs_nosensor.png' | relative_url }}) · [down_short]({{ '/assets/ik_analysis/S2_27_02_down_short_cycles_sensor_vs_nosensor.png' | relative_url }}) · [up_long]({{ '/assets/ik_analysis/S2_27_02_up_long_cycles_sensor_vs_nosensor.png' | relative_url }}) · [up_short]({{ '/assets/ik_analysis/S2_27_02_up_short_cycles_sensor_vs_nosensor.png' | relative_url }})
- S3 — [down_long]({{ '/assets/ik_analysis/S3_27_02_down_long_cycles_sensor_vs_nosensor.png' | relative_url }}) · [down_short]({{ '/assets/ik_analysis/S3_27_02_down_short_cycles_sensor_vs_nosensor.png' | relative_url }}) · [up_long]({{ '/assets/ik_analysis/S3_27_02_up_long_cycles_sensor_vs_nosensor.png' | relative_url }}) · [up_short]({{ '/assets/ik_analysis/S3_27_02_up_short_cycles_sensor_vs_nosensor.png' | relative_url }})
- S1 — [down_long]({{ '/assets/ik_analysis/S1_27_02_down_long_cycles_sensor_vs_nosensor.png' | relative_url }}) · [down_short]({{ '/assets/ik_analysis/S1_27_02_down_short_cycles_sensor_vs_nosensor.png' | relative_url }}) · [up_long]({{ '/assets/ik_analysis/S1_27_02_up_long_cycles_sensor_vs_nosensor.png' | relative_url }}) · [up_short]({{ '/assets/ik_analysis/S1_27_02_up_short_cycles_sensor_vs_nosensor.png' | relative_url }})

## Per-subject per-session RMSE (detail)

Per-marker bars with SD across the four tasks, one chart per subject × session:

- S2 [13_02]({{ '/assets/ik_analysis/S2_13_02_rmse_bar.png' | relative_url }}) · [27_02]({{ '/assets/ik_analysis/S2_27_02_rmse_bar.png' | relative_url }})
- S3 [13_02]({{ '/assets/ik_analysis/S3_13_02_rmse_bar.png' | relative_url }}) · [27_02]({{ '/assets/ik_analysis/S3_27_02_rmse_bar.png' | relative_url }})
- S1 [13_02]({{ '/assets/ik_analysis/S1_13_02_rmse_bar.png' | relative_url }}) · [27_02]({{ '/assets/ik_analysis/S1_27_02_rmse_bar.png' | relative_url }})

## Right-arm stroke profile (cycle strip)

Six evenly-spaced poses from one mid-trajectory down-stroke, overlaid with
fading opacity. Shows motion amplitude and path shape from a still image.

- S2: [13_02]({{ '/assets/ik_analysis/S2_13_02_cycle_strip.png' | relative_url }}) · [27_02]({{ '/assets/ik_analysis/S2_27_02_cycle_strip.png' | relative_url }})
- S3: [13_02]({{ '/assets/ik_analysis/S3_13_02_cycle_strip.png' | relative_url }}) · [27_02]({{ '/assets/ik_analysis/S3_27_02_cycle_strip.png' | relative_url }})
- S1: [13_02]({{ '/assets/ik_analysis/S1_13_02_cycle_strip.png' | relative_url }}) · [27_02]({{ '/assets/ik_analysis/S1_27_02_cycle_strip.png' | relative_url }})

## Between-session elbow flexion (same subject, same task)

The absolute joint angles depend on the `q_calib` offset, so a constant bias is
expected. What matters is the **shape** and **amplitude** of each stroke.

- S2 — down_long: ![S2 between-session down_long]({{ '/assets/ik_analysis/S2_between_session_down_long.png' | relative_url }})
- S3 — down_long: ![S3 between-session down_long]({{ '/assets/ik_analysis/S3_between_session_down_long.png' | relative_url }})
- S1 — down_long: ![S1 between-session down_long]({{ '/assets/ik_analysis/S1_between_session_down_long.png' | relative_url }})

## Within-session cycle variability

Each subject performs ~60 strokes per session. Each trace is one normalised
down-stroke (elbow flexion vs stroke progress). The red line is the mean.

### 27 Feb session

- S2 down_long: ![S2 27 Feb down_long variability]({{ '/assets/ik_analysis/S2_27_02_down_long_variability.png' | relative_url }})
- S3 down_long: ![S3 27 Feb down_long variability]({{ '/assets/ik_analysis/S3_27_02_down_long_variability.png' | relative_url }})
- S1 down_long: ![S1 27 Feb down_long variability]({{ '/assets/ik_analysis/S1_27_02_down_long_variability.png' | relative_url }})

### 13 Feb session

- S2 down_long: ![S2 13 Feb down_long variability]({{ '/assets/ik_analysis/S2_13_02_down_long_variability.png' | relative_url }})
- S3 down_long: ![S3 13 Feb down_long variability]({{ '/assets/ik_analysis/S3_13_02_down_long_variability.png' | relative_url }})
- S1 down_long: ![S1 13 Feb down_long variability]({{ '/assets/ik_analysis/S1_13_02_down_long_variability.png' | relative_url }})

## Stick + rock geometry

Rigid-body positions of the stick and rock over the full recording.

- S2 13 Feb: ![S2 13 Feb stick+rock]({{ '/assets/ik_analysis/S2_13_02_down_long_stick_rock.png' | relative_url }})
- S3 13 Feb: ![S3 13 Feb stick+rock]({{ '/assets/ik_analysis/S3_13_02_down_long_stick_rock.png' | relative_url }})
- S1 13 Feb: ![S1 13 Feb stick+rock]({{ '/assets/ik_analysis/S1_13_02_down_long_stick_rock.png' | relative_url }})

## What this tells us about the registration

- A **good fit** should have mean RMSE < 2 cm across markers — OptiTrack's own precision is ~0.5 cm, so 1–2 cm is expected after IK.
- **Bad markers** (hand/finger) often exceed 5 cm — these are dominated by skin slip, not registration error.
- **Between-session** traces should overlap in shape even if offset in absolute value — a systematic offset means `q_calib` differs but the kinematics are identical.
- **Arm-only RMSE < full-body RMSE** for arm markers confirms the two-pass approach is worth the extra compute: the full-body IK is trading some arm fit for better torso/head fit, and the second pass claws that back.
- **Sensor vs no-sensor** should look comparable on the non-arm markers; a large jump on arm markers only would signal sensor-induced occlusion or noise.

## Why the take difference does not hurt the recovery

The two sessions differ ~6× in marker-fit precision — the OptiTrack take (13_02)
fits to **0.2–0.7 cm** on most strokes (rising to ~1.3 cm only on the long
down-stroke), while the Evercoast take (27_02) fits to **~2–3.5 cm**. Yet the
**recovered cost agrees across both takes** (same proximal-load-plus-smoothness
structure; see [Results](results#recovery-is-stable-across-recording-sessions)).
The recovery is therefore **not limited by kinematic precision** at this level —
the same cost emerges whether the joints are reconstructed to 0.5 cm or to 3 cm.
The largest residuals fall on the **wrist and shoulder markers** (the ends of the
actuated chain), consistent with soft-tissue artefact and occlusion during the
press.

The takes embody a measurement trade-off: OptiTrack reconstructs kinematics to
sub-centimetre accuracy but carries **no contact information**, while Evercoast
trades marker precision for **volumetric contact localisation + the ATI force
sensor**. Precision matters not for the cost but for the **contact phase** —
knowing *when and where* the tool presses — which is exactly where the short
strokes fail (their brief engagement is hard to localise optically). Kinematic
precision is already sufficient for the cost; **contact precision is the
bottleneck.** (Methodology paper, Appendix C.)

---

*Regenerate this page with:* `python docs/scripts/gen_ik_report.py`

*Rerun the IK pipeline with:*
```bash
python batch_run_ik.py                                    # take1, all subjects/dates/tasks
python batch_run_ik.py --dates 27_02 --take take2_sensor \
    --out-dir trajectories_from_mocap_sensor              # take2, sensor recordings
```

[Home](.) | [Method](method) | [Results](results) | [Notes](notes)

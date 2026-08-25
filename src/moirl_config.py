"""moirl_config.py — combos + method list consumed by run_moirl_batch.py.

A combo is a named bundle of demos. Each demo is (subject, task, cycle).
`model_subject` (optional) names which subject's URDF to use when building
the common Human / HumanMPPI that IRL compares against — useful for
cross-subject combos. Defaults to the first demo's subject.

METHODS:
    ocp           — solve Crocoddyl OCP per demo (no IRL).
    mppi_vp       — MPPI IRL learning ONLY 'progress_vel'.
    mppi_2phase   — irl_phases.run_phase1 then run_phase2.
    mppi_3phase   — phase1 + phase2 + phase3 (all features, warm-started).
"""
from __future__ import annotations

DATE = "27_02"

COMBOS = [
    {"name": "s3_dl",        "demos": [("S3", "down_long", [5])],  "model_subject": "S3"},
    {"name": "s2_dl",        "demos": [("S2", "down_long", [10])],  "model_subject": "S2"},
    {"name": "s1_dl",        "demos": [("S1", "down_long", [10])],  "model_subject": "S1"},

    {"name": "s3_ul",        "demos": [("S3", "up_long",   [6])],  "model_subject": "S3"},
    {"name": "s2_ul",        "demos": [("S2", "up_long",   [5])],  "model_subject": "S2"},
    {"name": "s1_ul",        "demos": [("S1", "up_long",   [5])],  "model_subject": "S1"},

    {"name": "s3_ds",        "demos": [("S3", "down_short", [5])],  "model_subject": "S3"},
    {"name": "s2_ds",        "demos": [("S2", "down_short", [5])],  "model_subject": "S2"},
    {"name": "s1_ds",        "demos": [("S1", "down_short", [5])],  "model_subject": "S1"},

    {"name": "s3_us",        "demos": [("S3", "up_short",   [5])],  "model_subject": "S3"},
    {"name": "s2_us",        "demos": [("S2", "up_short",   [5])],  "model_subject": "S2"},
    {"name": "s1_us",        "demos": [("S1", "up_short",   [5])],  "model_subject": "S1"},

    {"name": "s3_dl_multi",  "demos": [("S3", "down_long", [5,6,7,8])],
                                "model_subject": "S3"},
    {"name": "s2_dl_multi",  "demos": [("S2", "down_long", [5,6,7,8])],
                                "model_subject": "S2"},
    {"name": "s1_dl_multi",  "demos": [("S1", "down_long", [10,11,12,13])],
                                "model_subject": "S1"},
    {"name": "s3_ul_multi",  "demos": [("S3", "up_long",   [5,6,7,8])],
                                "model_subject": "S3"},
    {"name": "s2_ul_multi",  "demos": [("S2", "up_long",   [3,4,5,6,7,8,9,10,11,12])],
                                "model_subject": "S2"},
    {"name": "s1_ul_multi",  "demos": [("S1", "up_long",   [3,4,5,6,7,8,9,10,11,12])],
                                "model_subject": "S1"},

    {"name": "all_dl",          "demos": [("S3", "down_long", [5,6,7,8]),
                                           ("S2", "down_long", [5,6,7,8]),
                                           ("S1", "down_long", [5,6,7,8])],
                                "model_subject": "S3"},
    {"name": "all_ul",          "demos": [("S3", "up_long",   [5,6,7,8]),
                                           ("S2", "up_long",   [5,6,7,8]),
                                           ("S1", "up_long",   [5,6,7,8])],
                                "model_subject": "S3"},
    {"name": "all_ds",          "demos": [("S3", "down_short", [5,6,7,8]),
                                           ("S2", "down_short", [5,6,7,8]),
                                           ("S1", "down_short", [5,6,7,8])],
                                "model_subject": "S3"},
]

METHODS = ["ocp", "mppi_vp", "mppi_2phase", "mppi_3phase"]
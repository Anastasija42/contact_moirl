"""fwd_dx.py plus a per-body press capacity for the transfer, WITHOUT touching the recovered weights (2026-09-22 up-stroke test 2).
The recovery fitted each subject against its own capacity (S1 14 / S2 28 / S3 47 N); the transfer gives every body one FORCE_MAX (47 N).
Here the human gets FMAX_REF (default 47 N, the geometry subject S3's fitted capacity) and every other body FMAX_REF * (arm mass /
human arm mass)^(2/3), the transfer's own isometric scaling (build_fmax_map). press_capacity keeps its recovered weight W(t).
usage: PYTHONPATH=<oss>/src FMAX_REF=47 python fwd_fmax.py <run_species_forward.py args incl. --task --species>"""
import os, sys
import crocoddyl  # noqa: F401
sys.path.insert(0, "os.environ.get('FRICTION_LIB_DX','friction_lib/build')")
import friction_lib
sys.path.insert(0, "${REPO_ROOT:-$(pwd)}/morphologies_study")
import run_species_forward as rsf
assert "friction_lib_dx" in friction_lib.__file__ and "/th_oss_transfer/" in rsf.__file__
argv = sys.argv[1:]
task = argv[argv.index("--task") + 1]; species = argv[argv.index("--species") + 1]
td = rsf.TASK_DEFAULTS[task]
w_path = argv[argv.index("--weights") + 1] if "--weights" in argv else td["weights"]
rsf.URDF_DIR = rsf.GEN_URDF_DIR
rsf.FMAX_HUMAN = float(os.environ.get("FMAX_REF", 47.0))
w_run, _ = rsf.load_wstar(w_path)
fmap, am = rsf.build_fmax_map(sorted({"human", species}), td["geom_subject"], task, td["cycle"], w_run)
rsf.FMAX_MAP = fmap
print("[fmax] per-body capacity (recovered weights untouched):", {k: round(v, 1) for k, v in fmap.items()},
      "arm mass", {k: round(v, 2) for k, v in am.items()}, flush=True)
sys.argv = ["run_species_forward.py"] + argv
rsf.main()

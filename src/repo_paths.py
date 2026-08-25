"""Portable repo-root resolution.

The library is installed editable (``pip install -e .``) with ``src/`` as the
import root, so ``from repo_paths import REPO`` works from any module, including
the ``final_models`` package. ``REPO`` is resolved from this file's own location
(not the cwd and not a hardcoded home directory), so the repo runs from any clone
path. Build a data/model path as ``REPO / "human_model" / "urdf" / "human.urdf"``.
"""
from pathlib import Path

def _find_repo() -> Path:
    here = Path(__file__).resolve()
    for cand in (here.parent.parent, *here.parents):
        if (cand / "human_model").is_dir():
            return cand
    return here.parent.parent

REPO = _find_repo()

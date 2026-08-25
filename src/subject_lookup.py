"""Subject-name resolution.

Subjects are identified as S1, S2, S3 (ordered by press level: 14, 28, 47 N).
These are the only names used in code, on the command line, and in results.

On-disk data may still use the original recording directory names. To support
that, an optional JSON alias file maps each subject id to the directory name:

    {"S1": "<dirname>", "S2": "<dirname>", "S3": "<dirname>"}

Its location defaults to config/subject_aliases.json and can be overridden with
the SUBJECT_ALIASES environment variable. When the file is absent, directories
are expected to be named S1/S2/S3 directly.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

CANONICAL_SUBJECTS = ["S1", "S2", "S3"]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ALIAS_FILE = _REPO_ROOT / "config" / "subject_aliases.json"


@lru_cache(maxsize=1)
def _aliases() -> dict:
    """Map subject id -> on-disk directory name. Empty when no alias file."""
    path = Path(os.environ.get("SUBJECT_ALIASES", _DEFAULT_ALIAS_FILE))
    if not path.is_file():
        return {}
    with open(path) as fh:
        raw = json.load(fh)
    return {k.upper(): str(v) for k, v in raw.items() if k.upper() in CANONICAL_SUBJECTS}


def canonical_subject(name: str) -> str:
    """Return the canonical subject id for `name`, accepting any casing and any
    configured on-disk alias. Raises ValueError if unknown."""
    if name is None:
        return None
    target = str(name).lower()
    for c in CANONICAL_SUBJECTS:
        if c.lower() == target:
            return c
    for sid, disk in _aliases().items():
        if disk.lower() == target:
            return sid
    raise ValueError(f"Unknown subject '{name}'. Known: {CANONICAL_SUBJECTS}")


def subject_dir_names(name: str) -> list:
    """Directory names to try for `name`, most specific first."""
    sid = canonical_subject(name)
    alias = _aliases().get(sid)
    return [sid] if alias is None else [alias, sid]


def disk_name(name: str, lower: bool = True) -> str:
    """On-disk name for `name`: the configured alias when one exists, else the
    subject id itself. Use this to build model/calibration file paths so they
    resolve both before and after the data files are renamed to S1/S2/S3."""
    sid = canonical_subject(name)
    out = _aliases().get(sid, sid)
    return out.lower() if lower else out


def find_subject_dir(parent: Path, name: str) -> Path:
    """Find the child directory of `parent` holding subject `name`."""
    parent = Path(parent)
    if not parent.exists():
        raise FileNotFoundError(parent)
    children = {c.name.lower(): c for c in parent.iterdir()}
    for candidate in subject_dir_names(name):
        hit = children.get(candidate.lower())
        if hit is not None:
            return hit
    raise FileNotFoundError(f"No subject dir matching '{name}' under {parent}")


def find_subject_file(parent: Path, pattern_prefix: str, ext: str = "") -> Path:
    """Find a file under `parent` whose name starts with `pattern_prefix`
    (case-insensitive), optionally filtered by extension."""
    parent = Path(parent)
    if not parent.exists():
        raise FileNotFoundError(parent)
    target = pattern_prefix.lower()
    for child in parent.iterdir():
        if child.name.lower().startswith(target) and child.name.lower().endswith(ext.lower()):
            return child
    raise FileNotFoundError(f"No file matching '{pattern_prefix}*{ext}' under {parent}")


def normalize_marker_names(df, canonical=None):
    """Rewrite 'PREFIX:MARKER' mocap columns so the subject prefix becomes its
    canonical id (or `canonical`, when given).

    Returns (df, substitutions).
    """
    known = {c.lower(): c for c in CANONICAL_SUBJECTS}
    known.update({v.lower(): k for k, v in _aliases().items()})
    subs = []
    new_cols = {}
    for col in df.columns:
        if ':' not in col:
            continue
        prefix, rest = col.split(':', 1)
        match = known.get(prefix.lower())
        if match is None:
            continue
        target = canonical if canonical else match
        if prefix != target:
            new_cols[col] = f"{target}:{rest}"
            subs.append((col, new_cols[col]))
    if new_cols:
        df.rename(columns=new_cols, inplace=True)
    return df, subs

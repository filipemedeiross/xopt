from __future__ import annotations

import sys

from pathlib import Path
from typing  import Iterable


DEFAULT_PROJECT_MARKERS  = "instances", "pymedian"
DEFAULT_INSTANCE_PATTERN = "pmed[0-9]*.txt"


def find_project_root(
    start   : str | Path | None = None,
    markers : Iterable[str]     = DEFAULT_PROJECT_MARKERS,
) -> Path:
    current = (
        Path.cwd()
        if   start is None
        else Path(start)
    ).resolve()

    marker_names = tuple(markers)

    for candidate in (current, *current.parents):
        if all(
            (candidate / marker).exists()
            for marker in marker_names
        ):
            return candidate

    markers_text = ", ".join(
        repr(marker) for marker in marker_names
    )

    raise FileNotFoundError(
        f"Could not find project root containing: {markers_text}."
    )


def ensure_project_on_path(
    root: str | Path | None = None
) -> Path:
    project_root      = (
        find_project_root()
        if   root is None
        else Path(root).resolve()
    )

    project_root_text = str(project_root)

    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)

    return project_root


def instances_dir(
    root: str | Path | None = None
) -> Path:
    project_root = (
        find_project_root()
        if   root is None
        else Path(root).resolve()
    )

    path = project_root / "instances"
    if not path.exists():
        raise FileNotFoundError(f"Instances folder not found: {path}")

    return path


def list_pmedian_instances(
    root    : str | Path | None = None,
    pattern : str               = DEFAULT_INSTANCE_PATTERN,
) -> list[Path]:
    return sorted(instances_dir(root).glob(pattern))

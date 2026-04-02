import re

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()

    for candidate in [start, *start.parents]:
        if (candidate / 'instances').exists() and \
           (candidate / 'pymedian' ).exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate the project root containing 'instances' and 'pymedian'."
    )


def instance_sort_key(pathlike: str | Path) -> tuple[int, str]:
    stem  = Path     (pathlike ).stem
    match = re.search(r'(\d+)$', stem)

    if match is None:
        return (10**9, stem)

    return (int(match.group(1)), stem)


def canonical_instance_name(instance_name: str) -> str:
    instance_name = instance_name.strip()

    if instance_name.endswith('.txt'):
        return instance_name

    return f'{instance_name}.txt'

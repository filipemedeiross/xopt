import re

import pandas as pd

from pathlib import Path
from .       import instance_sort_key


def list_orlibrary_instances(instances_dir: Path) -> list[str]:
    return sorted(
        [
            path.name
            for path in instances_dir.glob('pmed*.txt')
            if  path.name != 'pmedopt.txt'
        ],
        key=instance_sort_key,
    )


def apply_instance_selection(
    instance_names   : list[str],
    pattern          : str | None = None,
    limit            : int | None = None,
) -> list[str]:
    selected = list(instance_names)

    if pattern:
        regex    = re.compile(pattern)

        selected = [
            name
            for name in selected
            if  regex.search(name)
        ]

    if limit is not None:
        selected = selected[:limit]

    return selected


def read_instance_metadata(instance_path: Path) -> dict[str, int]:
    header = instance_path.read_text().splitlines()[0].split()

    if len(header) < 3:
        raise ValueError(
            f'Could not parse instance header: {instance_path}'
        )

    return {
        'n': int(header[0]),
        'p': int(header[2]),
    }


def load_best_known_costs(pmedopt_path: Path) -> pd.DataFrame:
    rows = []

    for raw_line in pmedopt_path.read_text().splitlines()[1:]:
        line = raw_line.strip()

        if not line:
            continue

        parts = line.split()

        rows.append(
            {
                'instance_id'     : parts[0].strip(),
                'best_known_cost' : float(parts[1] ),
            }
        )

    df = pd.DataFrame(rows)

    df['instance_order'] = df['instance_id'].map(
        lambda value: instance_sort_key(value)[0]
    )

    return df.sort_values(['instance_order', 'instance_id']) \
             .drop       (columns='instance_order'         ) \
             .reset_index(drop   =True)


def load_best_known_costs_to_dict(
    pmedopt_path: Path
) -> dict[str, int]:
    best_known_costs: dict[str, int] = {}

    for raw_line in pmedopt_path.read_text().splitlines()[1:]:
        line = raw_line.strip()

        if not line:
            continue

        instance_id, value = line.split()[:2]

        best_known_costs[f'{instance_id}.txt'] = int(value)

    return best_known_costs


def load_best_known_costs_to_dict_id(
    pmedopt_path: Path
) -> dict[str, float]:
    rows = {}

    for raw_line in pmedopt_path.read_text().splitlines()[1:]:
        line = raw_line.strip()

        if not line:
            continue

        instance_id, value = line.split()[:2]
        rows[instance_id]  = float(value)

    return rows

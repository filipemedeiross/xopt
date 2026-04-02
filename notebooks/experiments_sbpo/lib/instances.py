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

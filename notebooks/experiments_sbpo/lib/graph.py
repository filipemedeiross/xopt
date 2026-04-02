import heapq
import numpy as np

from pathlib import Path


def read_orlibrary_graph(instance_path: Path) -> dict[str, object]:
    with instance_path.open() as file:
        header = file.readline().split()

        if len(header) < 3:
            raise ValueError(
                f'Could not parse instance header: {instance_path}'
            )

        n, m, p = map(int, header[:3])

        adjacency_map: list[dict[int, int]] = [dict() for _ in range(n)]
        raw_edge_count         = 0
        duplicate_edge_updates = 0

        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            u, v, cost = map(int, line.split())
            u -= 1
            v -= 1

            if v in adjacency_map[u]:
                duplicate_edge_updates += 1

            adjacency_map[u][v] = cost
            adjacency_map[v][u] = cost
            raw_edge_count += 1

    if raw_edge_count != m:
        raise ValueError(
            f'Expected {m} edges in {instance_path.name}, but found {raw_edge_count}.'
        )

    adjacency = [
        list(neighbors.items())
        for neighbors in adjacency_map
    ]

    return {
        'n': n,
        'm': m,
        'p': p,

        'adjacency'              : adjacency             ,
        'duplicate_edge_updates' : duplicate_edge_updates,
    }


def all_pairs_shortest_paths(
    adjacency: list[list[tuple[int, int]]]
) -> np.ndarray:
    n         = len(adjacency)
    distances = np.full((n, n), np.inf, dtype=np.float64)

    for source in range(n):
        row         = distances[source]
        row[source] = 0.0

        heap: list[tuple[float, int]] = [(0.0, source)]

        while heap:
            dist_u, u = heapq.heappop(heap)
            if dist_u > row[u]:
                continue

            for v, weight in adjacency[u]:
                candidate = dist_u + weight

                if candidate < row[v]:
                    row[v] = candidate

                    heapq.heappush(heap, (candidate, v))

        if np.isinf(row).any():
            raise ValueError(
                'Instance graph is disconnected. Could not compute all-pairs distances.'
            )

    return distances.astype(np.int64)

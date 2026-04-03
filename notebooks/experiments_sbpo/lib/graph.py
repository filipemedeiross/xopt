import heapq

import numpy    as np
import networkx as nx

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


def build_top_ltm(
    long_term_memory : list[dict[str, object]],
    top_fraction     : float                  ,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    if not long_term_memory:
        raise ValueError("long_term_memory is empty.")

    top_solution_count = max(
        1, int(np.ceil(len(long_term_memory) * top_fraction))
    )

    analysis_ltm = sorted(
        long_term_memory, key=lambda sol: float(sol["cost"])
    )[:top_solution_count]

    matrix = np.vstack(
        [
            np.asarray(
                sol["facilities"], dtype=np.int8
            )
            for sol in analysis_ltm
        ]
    )

    costs = np.asarray(
        [
            float(sol["cost"])
            for sol in analysis_ltm
        ],
        dtype=float,
    )

    return analysis_ltm, matrix, costs


def build_cooccurrence_matrix(matrix: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(
        matrix.T @ matrix, dtype=np.int64
    )

    np.fill_diagonal(adjacency, 0)

    return adjacency


def build_weighted_graph(adjacency: np.ndarray) -> nx.Graph:
    n = adjacency.shape[0]

    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    rows, cols = np.where(np.triu(adjacency, k=1) > 0)

    for i, j in zip(rows.tolist(), cols.tolist()):
        graph.add_edge(
            int(i), int(j), weight=float(adjacency[i, j])
        )

    return graph


def build_unweighted_graph(adjacency: np.ndarray) -> nx.Graph:
    n = adjacency.shape[0]

    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    rows, cols = np.where(np.triu(adjacency, k=1) > 0)

    graph.add_edges_from(
        (int(i), int(j))
        for i, j in zip(rows.tolist(), cols.tolist())
    )

    return graph


def total_edge_weight(adjacency: np.ndarray) -> float:
    return float(np.triu(adjacency, k=1).sum())


def total_edge_count(adjacency: np.ndarray) -> int:
    return int(np.count_nonzero(np.triu(adjacency, k=1) > 0))

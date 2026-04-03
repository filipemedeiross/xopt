import numpy as np


def densest_subgraph_greedy(
    adjacency : np.ndarray,
    min_size  : int = 3   ,
) -> tuple[set[int], float]:
    n = adjacency.shape[0]

    remaining     = set(range(n))
    best_subgraph = set(range(n)) if n and n < min_size else set()
    best_density  = -np.inf

    while len(remaining) >= max(1, min_size):
        nodes    = sorted   (remaining)
        subgraph = adjacency[np.ix_(nodes, nodes)]

        current_weight  = float(np.triu(subgraph, k=1).sum())
        current_density = current_weight / max(1, len(remaining))

        if current_density > best_density:
            best_density  = current_density
            best_subgraph = remaining.copy()

        min_pos = int(
            np.argmin(subgraph.sum(axis=1))
        )
        remaining.remove(nodes[min_pos])

    if not best_subgraph and n:
        best_subgraph = set(range(min(n, max(1, min_size))))
        best_density  = 0.0

    return best_subgraph, max(0.0, float(best_density))

import math

import numpy as np

from itertools import combinations


def _random_labels(
    n   : int                ,
    p   : int                ,
    rng : np.random.Generator,
) -> np.ndarray:
    labels = rng.integers(0, p, size=n)

    if n >= p:
        permutation             = rng.permutation(n)
        labels[permutation[:p]] = np .arange     (p)

    return labels


def _group_sums(
    A      : np.ndarray,
    labels : np.ndarray,
    p      : int       ,
) -> np.ndarray:
    sums = np.zeros((A.shape[0], p), dtype=float)

    for group_id in range(p):
        members = labels == group_id

        if np.any(members):
            sums[:, group_id] = A[:, members].sum(axis=1)

    return sums


def max_p_cut_local_search(
    A: np.ndarray,
    p: int       ,
    n_restarts: int = 30  ,
    max_iter  : int = 2000,
    seed      : int = 42  ,
) -> tuple[np.ndarray, float, float]:
    A = np.asarray(A, dtype=float)

    n = A.shape[0]
    p = max(1, min(int(p), n))

    total_weight = float(np.triu(A, 1).sum())

    if total_weight <= 1e-12:
        labels = np.arange(n, dtype=int) % p

        return labels, 0.0, 0.0

    rng = np.random.default_rng(seed)

    best_cut      = -np.inf
    best_internal =  np.inf
    best_labels   = None

    for _ in range(max(1, int(n_restarts))):
        labels = _random_labels(n, p, rng)
        sums   = _group_sums   (A, labels, p)

        internal = float(
            np.triu(
                A * (labels[:, None] == labels[None, :]), 1
            ).sum()
        )
        cut      = total_weight - internal

        for _ in range(max(1, int(max_iter))):
            improved = False

            for node in rng.permutation(n):
                old_group = labels[node]

                gains = sums[node, old_group] - sums[node, :]
                gains[old_group] = -np.inf

                new_group = int  (np.argmax(gains))
                gain      = float(gains[new_group])

                if gain > 1e-12:
                    labels[node] = new_group
                    cut += gain

                    sums[:, old_group] -= A[:, node]
                    sums[:, new_group] += A[:, node]

                    improved = True

            if not improved:
                break

        internal = total_weight - cut

        if cut > best_cut + 1e-12:
            best_cut      = float(cut     )
            best_internal = float(internal)
            best_labels   = labels.copy()

    return best_labels, best_cut, best_internal


def best_facility_separation(
    labels   : np.ndarray,
    best_set : set[int]  ,
) -> float:
    best_nodes = sorted(best_set)

    if len(best_nodes) <= 1:
        return 1.0

    separated_pairs = sum(
        labels[i] != labels[j]
        for i, j in combinations(best_nodes, 2)
    )

    return separated_pairs / math.comb(len(best_nodes), 2)

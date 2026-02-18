import heapq
import numpy as np

from itertools    import count
from scipy.sparse import csr_matrix
from collections  import defaultdict

from .cluster import Cluster , UnionFind
from .tools   import distance, list2matrix


def agnes(bigraph, thresh=None):
    entry_counter = count()

    n        = bigraph.N
    l        = bigraph.L
    adj_list = bigraph.adj_list

    adj_matrix = list2matrix(adj_list, n, l)

    C = {
        v : Cluster(
            {v}, csr_matrix(adj)
        )
        for v, adj in enumerate(adj_matrix)
    }

    D = []
    for i in range(n):
        for j in range(i + 1, n):
            heapq.heappush(
                D,
                (
                    distance(C[i], C[j], l),
                    next    (entry_counter),
                    i,
                    j,
                )
            )

    F = UnionFind(2 * n - 1)

    L = []
    for v in range(n, 2 * n - 1):
        while D:
            d, _, k, m = heapq.heappop(D)

            if k in C and m in C:
                break
        else:
            break

        if thresh and thresh > (l - d):
            break

        C_k = C[k]
        C_m = C[m]
        C_v = C_k | C_m

        L.append([k, m, d, len(C_v.v)])

        del C[k], C[m]

        for u, C_u in C.items():
            heapq.heappush(
                D,
                (
                    distance(C_u, C_v, l  ),
                    next    (entry_counter),
                    u,
                    v,
                )
            )

        C[v] = C_v

        F.union(v, k)
        F.union(v, m)

    clusters = defaultdict(list)

    for v, step in enumerate(L, start=n):
        root = F.find(v)

        clusters[root].append(step)

    return [
        np.asarray(steps, dtype=np.float64)
        for _, steps in sorted(clusters.items())
    ]

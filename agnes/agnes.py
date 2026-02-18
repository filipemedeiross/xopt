import heapq
import numpy as np

from itertools    import count
from scipy.sparse import csr_matrix

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

    roots = []
    for v in range(n, 2 * n - 1):
        root_v = F.find(v)

        if v != root_v:
            roots.append(root_v)

    Ls = []
    for v in np.unique(roots):
        cluster = []

        for pos, step in enumerate(L):
            if roots[pos] == v:
                cluster.append(step)

        Ls.append(np.array(cluster, dtype='float64'))

    return Ls

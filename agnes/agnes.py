import numpy as np

from scipy.sparse import csr_matrix

from .cluster import Cluster , UnionFind
from .tools   import distance, list2matrix


def agnes(adj_list, n, l, thresh=None):
    adj_matrix = list2matrix(adj_list, n, l)

    C = {
        v : Cluster(
            {v}, csr_matrix(adj)
        )
        for v, adj in enumerate(adj_matrix)
    }
    D = [
        (
            i, j, distance(C[i], C[j], l)
        )
        for i in range(n)
        for j in range(i + 1, n)
    ]
    F = UnionFind(2 * n - 1)

    L = []
    for v in range(n, 2 * n - 1):
        k, m, d = min(D, key=lambda x: x[2])

        if thresh and thresh > (l - d):
            break

        C_k = C[k]
        C_m = C[m]
        C_v = C_k | C_m

        L.append([k, m, d, len(C_v.v)])

        del C[k], C[m]

        D = [
            candidate
            for candidate in D
            if not any(c in (k, m) for c in candidate[:2])
        ]

        for u, C_u in C.items():
            D.append((u, v, distance(C_u, C_v, l)))

        C[v] = C_v

        F.union(k, m)
        F.union(k, v)

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

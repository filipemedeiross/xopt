import numpy as np

from scipy.sparse import csr_matrix

from .cluster import Cluster


def distance(a: Cluster,
             b: Cluster,
             size: int ) -> int:
    return int(size - a.similarity(b))


def list2matrix(adj_list, n, l):
    rows = []
    cols = []
    data = []

    for u, adj in enumerate(adj_list):
        for v in adj:
            rows.append(u)
            cols.append(v)
            data.append(1)

    matrix = (data, (rows, cols))

    return csr_matrix(matrix, shape=(n, l), dtype='uint32')


def normalize_linkage(L):
    canonical = L.copy()
    children  = L[:, :2]

    flat = children.ravel()

    argsort = np.argsort(flat   )
    indices = np.argsort(argsort)

    canonical[:, :2] = indices.reshape(children.shape)

    return canonical

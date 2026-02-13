from typing      import List
from dataclasses import dataclass

from scipy.sparse import csr_matrix


@dataclass(slots=True, eq=False)
class Cluster:
    id: int
    v : List[int]   # vertices in the cluster
    l : csr_matrix  # shared-resource intersection


class UnionFind:
    def __init__(self, n):
        self.v = list(range(n))

    def find(self, u):
        while u != self.v[u]:
            self.v[u] = self.v[self.v[u]]
            u         = self.v[u]

        return u

    def union(self, u, v):
        self.v[self.find(v)] = self.find(u)

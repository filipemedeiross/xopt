from __future__ import annotations

from dataclasses  import dataclass
from scipy.sparse import csr_matrix


@dataclass(slots=True, eq=False)
class Cluster:
    id: int
    v : set[int]    # vertices in the cluster
    l : csr_matrix  # shared-resource intersection

    def __post_init__(self) -> None:
        if not isinstance(self.l, csr_matrix):
            raise TypeError("l must be a csr_matrix")

        if self.l.shape[0] != 1:
            raise ValueError("l must have shape (1, L)")

    def __add__(self, other: Cluster) -> set[int]:
        return self.v | other.v

    def __and__(self, other: Cluster) -> csr_matrix:
        return self.l.multiply(other.l)

    def __matmul__(self, other: Cluster) -> int:
        if not isinstance(other, Cluster):
            return NotImplemented

        if self.l.shape[1] != other.l.shape[1]:
            raise ValueError("Clusters must share the same universe size L")

        return int((self.l @ other.l.T)[0, 0])

    def similarity(self, other: Cluster) -> int:
        return self @ other


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

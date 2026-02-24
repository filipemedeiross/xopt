from __future__ import annotations

from typing import Literal, TypedDict


DetailsFormat = Literal[
    "binary" ,
    "indices",
]


class Summary(TypedDict):
    instance     : str
    n            : int
    p            : int

    long_term_mem     : int
    kmedoids_cost     : float
    tspmed_cost       : float
    medoids           : list[int]
    tspmed_facilities : list[int]


class MedoidsDetail(TypedDict):
    restart : int
    cost    : float
    medoids : list[int]


class TspmedDetail(TypedDict):
    restart    : int
    cost       : float
    facilities : list[int]


class LongTermMemoryDetail(TypedDict):
    cost       : float
    facilities : list[int]


class Details(TypedDict):
    long_term_memory : list[LongTermMemoryDetail]
    medoids          : list[MedoidsDetail]
    tspmed           : list[TspmedDetail ]

from __future__ import annotations

import numpy as np

from numpy.typing    import NDArray
from dataclasses     import dataclass
from typing          import Any     , Literal
from collections.abc import Iterable, Mapping, Sequence


FacilitiesFormat = Literal[
    "auto"   ,
    "binary" ,
    "indices",
]


@dataclass(frozen=True)
class SolutionArrays:
    """Binary solution matrix, cost vector, and original row order."""

    X     : NDArray[np.integer ]
    costs : NDArray[np.floating]
    order : NDArray[np.integer ]


@dataclass(frozen=True)
class BestSolution:
    """Best long-term-memory solution by cost."""

    index      : int
    cost       : float
    facilities : tuple  [int, ...]
    record     : Mapping[str, Any]


def facilities_to_binary(
    facilities   : Sequence[int],
    n_facilities : int          ,
    *,
    details_format : FacilitiesFormat = "auto" ,
    dtype          : type[np.integer] = np.int8,
) -> NDArray[np.integer]:
    if n_facilities < 0:
        raise ValueError("n_facilities must be non-negative.")

    values = np.asarray(facilities, dtype=int)
    if values.ndim != 1:
        raise ValueError("facilities must be a one-dimensional sequence.")

    current_format = _infer_facilities_format(
        values, n_facilities, details_format
    )

    if current_format == "binary":
        _validate_binary_facilities(values, n_facilities)

        return values.astype(dtype, copy=True)

    _validate_index_facilities(values, n_facilities)

    vector = np.zeros(
        n_facilities, dtype=dtype
    )
    vector[values] = 1

    return vector


def long_term_memory_to_arrays(
    memory       : Iterable[Mapping[str, Any]],
    n_facilities : int                        ,
    *,
    details_format : FacilitiesFormat = "auto" ,
    sort_by_cost   : bool             = True   ,
    matrix_dtype   : type[np.integer] = np.int8,
) -> SolutionArrays:
    records = tuple(memory)

    if not records:
        raise ValueError("long-term memory is empty.")

    X = np.vstack(
        [
            facilities_to_binary(
                record["facilities"],
                n_facilities        ,
                details_format=details_format,
                dtype         =matrix_dtype  ,
            )
            for record in records
        ]
    )
    costs = np.asarray(
        [
            float(record["cost"])
            for record in records
        ],
        dtype=float,
    )

    if sort_by_cost:
        return sort_solution_arrays(X, costs)

    return SolutionArrays(
        X    =X    ,
        costs=costs,
        order=np.arange(len(records), dtype=int),
    )


def sort_solution_arrays(
    X     : NDArray[np.integer ],
    costs : NDArray[np.floating],
) -> SolutionArrays:
    X_array    = np.asarray(X)
    cost_array = np.asarray(costs, dtype=float)

    if X_array.ndim != 2:
        raise ValueError("X must be a two-dimensional matrix."           )
    if cost_array.ndim != 1:
        raise ValueError("costs must be a one-dimensional vector."       )
    if X_array.shape[0] != cost_array.shape[0]:
        raise ValueError("X and costs must have the same number of rows.")

    order = np.argsort(cost_array, kind="stable")

    return SolutionArrays(
        X    =X_array   [order],
        costs=cost_array[order],
        order=order,
    )


def best_solution(
    memory : Iterable[Mapping[str, Any]]
) -> BestSolution:
    records = tuple(memory)

    if not records:
        raise ValueError("long-term memory is empty.")

    index, record = min(
        enumerate(records),
        key=lambda item: float(item[1]["cost"]),
    )

    return BestSolution(
        index     =index,
        cost      =float(record["cost"]                              ),
        facilities=tuple(int(value) for value in record["facilities"]),
        record    =record,
    )


def best_facilities_from_summary(
    summary : Mapping[str, Any],
    *,
    key        : str  = "tspmed_facilities",
    zero_based : bool = True               ,
) -> set[int]:
    offset = 1 if zero_based else 0

    return {
        int(facility) - offset
        for facility in summary[key]
    }


def selected_facilities(
    solution  : Sequence[int],
    *,
    one_based : bool = False ,
) -> NDArray[np.integer]:
    indices = np.where(
        np.asarray(solution, dtype=int) == 1
    )[0]

    if one_based:
        indices = indices + 1

    return indices


def _infer_facilities_format(
    values         : NDArray[np.integer],
    n_facilities   : int                ,
    details_format : FacilitiesFormat   ,
) -> Literal["binary", "indices"]:
    if details_format == "binary" or details_format == "indices":
        return details_format

    if details_format != "auto":
        raise ValueError(
            "details_format must be 'auto', 'binary', or 'indices'."
        )

    if values.size == n_facilities and np.isin(values, (0, 1)).all():
        return "binary"

    return "indices"


def _validate_binary_facilities(
    values       : NDArray[np.integer],
    n_facilities : int                ,
) -> None:
    if values.size != n_facilities:
        raise ValueError(
            "binary facilities must have length equal to n_facilities."
        )

    if not np.isin(values, (0, 1)).all():
        raise ValueError("binary facilities must contain only 0 and 1.")


def _validate_index_facilities(
    values       : NDArray[np.integer],
    n_facilities : int                ,
) -> None:
    if values.size == 0:
        return

    invalid = values[
        (values < 0) | (values >= n_facilities)
    ]

    if invalid.size:
        raise IndexError(
            f"facility index out of bounds for n_facilities={n_facilities}: {int(invalid[0])}"
        )

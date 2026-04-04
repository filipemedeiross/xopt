import numpy  as np
import pandas as pd

from .utils import finite_or_none, as_sorted_tuple


def compute_solution_cost(
    distances      : np.ndarray                 ,
    open_facilities: tuple[int, ...] | list[int],
) -> int | None:
    open_facilities = list(as_sorted_tuple(open_facilities))

    if not open_facilities:
        return None

    return int(
        distances[:, open_facilities].min(axis=1).sum()
    )


def compute_gap_percent(
    cost            : float | None,
    best_known_cost : float | None,
) -> float:
    if cost is None or pd.isna(cost):
        return np.nan

    if best_known_cost is None  or \
       pd.isna(best_known_cost) or \
       float  (best_known_cost) == 0.0:
        return np.nan

    return 100.0 * (float(cost) - float(best_known_cost)) / float(best_known_cost)


def gap_to_reference_percent(
    value     : float | int | None,
    reference : float | int | None,
) -> float | None:
    value     = finite_or_none(value    )
    reference = finite_or_none(reference)

    if value     is None or \
       reference is None or \
       reference == 0:
        return None

    return 100.0 * (value - reference) / reference


def improvement_percent(
    baseline  : float | int | None,
    candidate : float | int | None,
) -> float:
    baseline  = finite_or_none(baseline )
    candidate = finite_or_none(candidate)

    if baseline  is None or \
       candidate is None or \
       baseline == 0.0:
        return np.nan

    return 100.0 * (baseline - candidate) / baseline


def speedup_factor(
    baseline_seconds  : float | int | None,
    candidate_seconds : float | int | None,
) -> float | None:
    baseline_seconds  = finite_or_none(baseline_seconds )
    candidate_seconds = finite_or_none(candidate_seconds)

    if baseline_seconds  is None or \
       candidate_seconds is None or \
       candidate_seconds <= 0:
        return None

    return baseline_seconds / candidate_seconds

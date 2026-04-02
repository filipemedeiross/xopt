import numpy  as np
import pandas as pd


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

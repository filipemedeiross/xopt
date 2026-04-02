import numpy as np


def extract_open_facilities(
    y         : list       ,
    threshold : float = 0.5,
) -> list[int]:
    return [
        index
        for index, variable in enumerate(y)
        if  variable.x is not None and variable.x >= threshold
    ]


def compute_solution_cost(
    distances       : np.ndarray,
    open_facilities : list[int] ,
) -> int | None:
    if not open_facilities:
        return None

    return int(
        distances[:, open_facilities].min(axis=1).sum()
    )


def compute_assignments(
    distances       : np.ndarray,
    open_facilities : list[int] ,
) -> list[int]:
    if not open_facilities:
        return []

    closest_open_facility = np.argmin(
        distances[:, open_facilities], axis=1
    )

    return [
        open_facilities[index]
        for index in closest_open_facility.tolist()
    ]

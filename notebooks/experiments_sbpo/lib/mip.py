import numpy as np

from mip import xsum, minimize, Model, BINARY


def extract_open_facilities(
    y         : list       ,
    threshold : float = 0.5,
) -> list[int]:
    return [
        index
        for index, variable in enumerate(y)
        if  variable.x is not None and variable.x >= threshold
    ]


def extract_open_facilities_candidates(
    candidate_facilities: list[int],
    y                   : list     ,
    *,
    threshold: float = 0.5,
) -> list[int]:
    return [
        facility
        for facility, variable in zip(candidate_facilities, y)
        if  variable.x is not None and variable.x >= threshold
    ]


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


def build_pmedian_model(
    distances : np.ndarray,
    p         : int       ,
):
    if distances.ndim != 2:
        raise ValueError('Distances must be a 2D array.')

    n_rows, n_cols = distances.shape
    if n_rows != n_cols:
        raise ValueError(
            'This implementation assumes a square distance matrix.'
        )

    n = n_rows
    if not (1 <= p <= n):
        raise ValueError('P must satisfy 1 <= p <= n.')

    if np.any(distances < 0):
        raise ValueError('Distances must be nonnegative.')

    model = Model(solver_name='CBC')
    model.verbose = 0

    y = [
        model.add_var(
            var_type=BINARY, name=f'y_{j}'
        )
        for j in range(n)
    ]

    x: list[list]  = []
    row_cost_terms = []

    for i in range(n):
        x_row = [
            model.add_var(
                var_type=BINARY, name=f'x_{i}_{j}'
            )
            for j in range(n)
        ]
        x.append(x_row)

        model.add_constr(
            xsum(x_row[j] for j in range(n)) == 1,
            name=f'assign_{i}',
        )

        for j in range(n):
            model.add_constr(
                x_row[j] <= y[j], name=f'link_{i}_{j}'
            )

        row_cost_terms.append(
            xsum(float(distances[i, j]) * x_row[j] for j in range(n))
        )

    model.add_constr(
        xsum(y[j] for j in range(n)) == p,
        name='select_p'
    )

    model.objective = minimize(xsum(row_cost_terms))

    return model, x, y

from __future__ import annotations

from pathlib import Path

import xopt

from .encoding import to_one_based, \
                      encode_facilities
from .types    import Summary, \
                      Details, \
                      DetailsFormat


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")


def solve_pmedian(
    instance_path: str | Path,
    *,
    restarts       : int = 10,
    max_iter       : int = 20,
    factor         : int = 2 ,
    details_format : DetailsFormat = "binary",
) -> tuple[Summary, Details]:

    _validate_positive("restarts", restarts)
    _validate_positive("max_iter", max_iter)
    _validate_positive("factor"  , factor  )

    path = Path(instance_path)
    if not path.exists():
        raise FileNotFoundError(f"Instance file not found: {path}")

    instance = xopt.Instance(str(path))
    n = instance.get_n()
    p = instance.get_p()

    medoids_runs = xopt.kmedoids(
        instance,
        max_iter=max_iter,
        restarts=restarts,
    )

    long_term = xopt.SolutionTrie(n, p)

    ts_results = []
    for medoids in medoids_runs:
        result = xopt.tspmed(
            instance       ,
            medoids.medoids,
            iter_factor=factor   ,
            long_term  =long_term,
        ).best

        ts_results.append(result)

    best_kmedoids = min(medoids_runs, key=lambda run: run.cost)
    best_tspmed   = min(ts_results  , key=lambda run: run.cost)

    long_term_mem = long_term.get_all_solutions(instance)

    summary: Summary = {
        "instance" : str(path),
        "n"        : n,
        "p"        : p,

        "kmedoids_cost"     : best_kmedoids.cost,
        "tspmed_cost"       : best_tspmed  .cost,
        "long_term_mem"     : len(long_term_mem),
        "medoids"           : to_one_based(best_kmedoids.medoids   ),
        "tspmed_facilities" : to_one_based(best_tspmed  .facilities),
    }

    details: Details = {
        "long_term_memory": [
            {
                "cost"       : sol.cost,
                "facilities" : encode_facilities(
                    sol.facilities, n, details_format
                ),
            }
            for sol in long_term_mem
        ],
        "medoids": [
            {
                "restart" : idx     ,
                "cost"    : run.cost,
                "medoids" : encode_facilities(
                    run.medoids, n, details_format
                ),
            }
            for idx, run in enumerate(medoids_runs)
        ],
        "tspmed": [
            {
                "restart"    : idx        ,
                "cost"       : result.cost,
                "facilities" : encode_facilities(
                    result.facilities, n, details_format
                ),
            }
            for idx, result in enumerate(ts_results)
        ],
    }

    return summary, details

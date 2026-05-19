from collections.abc import Callable

import numpy    as np
import networkx as nx


def normalize_solution(
    facilities: tuple[int, ...] | list[int] | np.ndarray,
) -> tuple[int, ...]:
    return tuple(
        sorted(
            int(facility)
            for facility in facilities
        )
    )


def binary_facility_vector(
    n          : int,
    facilities : tuple[int, ...] | list[int] | np.ndarray,
) -> np.ndarray:
    n        = int(n)
    solution = normalize_solution(facilities)

    if n < 0:
        raise ValueError("n must be non-negative.")
    if len(solution) != len(set(solution)):
        raise ValueError("Facilities must not contain duplicates.")
    if any(facility < 0 or facility >= n for facility in solution):
        raise IndexError(f"Facility index out of bounds for n={n}.")

    vector = np.zeros(n, dtype=np.int8)

    vector[list(solution)] = 1

    return vector


def facilities_from_binary_vector(values: object) -> tuple[int, ...]:
    vector = np.asarray(values, dtype=np.int8)

    if vector.ndim != 1:
        raise ValueError("facility vector must be one-dimensional.")
    if not np.isin(vector, [0, 1]).all():
        raise ValueError("facility vector must be binary.")

    return tuple(
        np.flatnonzero(vector).astype(int).tolist()
    )


def solution_swap_distance(
    left  : tuple[int, ...] | list[int] | np.ndarray,
    right : tuple[int, ...] | list[int] | np.ndarray,
    p     : int | None = None,
) -> int:
    left_solution  = normalize_solution(left )
    right_solution = normalize_solution(right)

    if p is None:
        p = len(left_solution)

    if len(left_solution) != int(p) or len(right_solution) != int(p):
        raise ValueError("Solutions must have p selected facilities.")

    return int(
        int(p) - len(set(left_solution) & set(right_solution))
    )


def pmedian_solution_cost(
    instance   : object,
    facilities : tuple[int, ...] | list[int] | np.ndarray,
) -> float:
    solution = normalize_solution(facilities)

    if not solution:
        raise ValueError("At least one facility is required.")

    distances = None

    if isinstance(instance, np.ndarray):
        distances = instance
    elif isinstance(instance, dict):
        for key in ("distances", "distance_matrix", "costs", "cost_matrix"):
            if key in instance:
                distances = instance[key]
                break
    else:
        for key in ("distances", "distance_matrix", "costs", "cost_matrix"):
            if hasattr(instance, key):
                distances = getattr(instance, key)
                break

    if distances is None:
        raise ValueError(
            "Could not infer a distance matrix from instance. "
            "Pass a cost_fn for this instance type."
        )

    matrix = np.asarray(distances, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance matrix must be square.")
    if any(facility < 0 or facility >= matrix.shape[1] for facility in solution):
        raise IndexError(f"Facility index out of bounds for n={matrix.shape[1]}.")

    return float(
        matrix[:, list(solution)].min(axis=1).sum()
    )


def build_solution_swap_graph(
    solutions : list[tuple[int, ...]],
    costs     : list[float] | np.ndarray | None = None,
) -> nx.Graph:
    graph = nx.Graph()

    if costs is not None and len(costs) != len(solutions):
        raise ValueError("Costs must have one value per solution.")

    for index, solution in enumerate(solutions):
        attributes: dict[str, object] = {
            "solution": solution       ,
            "label"   : f"S{index + 1}",
        }

        if costs is not None:
            attributes["cost"] = float(costs[index])

        graph.add_node(index, **attributes)

    for left in range(len(solutions)):
        for right in range(left + 1, len(solutions)):
            distance = solution_swap_distance(
                solutions[left ],
                solutions[right],
            )

            if distance <= 1:
                graph.add_edge(
                    left ,
                    right,
                    weight  =distance,
                    distance=distance,
                )

    return graph


def build_swap_component_graph(
    swap_graph : nx.Graph,
    solutions  : list[tuple[int, ...]],
) -> tuple[nx.Graph, list[list[int]]]:
    components = [
        sorted(component)
        for component in nx.connected_components(swap_graph)
    ]

    components = sorted(
        components, key=lambda component: (component[0], len(component))
    )

    component_graph = nx.Graph()

    for component_index, component_nodes in enumerate(components):
        component_graph.add_node(
            component_index,
            solution_nodes=    component_nodes ,
            size          =len(component_nodes),
        )

    for left_component in range(len(components)):
        for right_component in range(left_component + 1, len(components)):
            best_pair = None

            for left_solution in components[left_component]:
                for right_solution in components[right_component]:
                    distance = solution_swap_distance(
                        solutions[left_solution ],
                        solutions[right_solution],
                    )

                    candidate_pair = (
                        distance,
                        left_solution ,
                        right_solution,
                    )

                    if best_pair is None or candidate_pair < best_pair:
                        best_pair = candidate_pair

            if best_pair is None:
                continue

            distance, left_solution, right_solution = best_pair

            component_graph.add_edge(
                left_component ,
                right_component,
                weight           =distance      ,
                source_solution  =left_solution ,
                target_solution  =right_solution,
                source_facilities=solutions[left_solution ],
                target_facilities=solutions[right_solution],
                intersection=tuple(
                    sorted(
                        set(solutions[left_solution]) & set(solutions[right_solution])
                    )
                ),
            )

    return component_graph, components


def greedy_swap_path_between_solutions(
    start   : tuple[int, ...] | list[int] | np.ndarray,
    target  : tuple[int, ...] | list[int] | np.ndarray,
    *,
    cost_fn : Callable[[tuple[int, ...]], float] | None = None,
) -> tuple[list[tuple[int, ...]], list[dict[str, object]]]:
    current = normalize_solution(start )
    target  = normalize_solution(target)

    path = [current]
    rows = []

    while current != target:
        current_set = set(current)
        target_set  = set(target )

        remove_options = sorted(current_set - target_set )
        add_options    = sorted(target_set  - current_set)

        candidates = []
        for remove_facility in remove_options:
            for add_facility in add_options:
                candidate = normalize_solution(
                    list((current_set - {remove_facility}) | {add_facility})
                )

                cost = (
                    float(cost_fn(candidate))
                    if   cost_fn is not None
                    else 0.0
                )

                candidates.append(
                    (
                        cost           ,
                        remove_facility,
                        add_facility   ,
                        candidate      ,
                    )
                )

        if not candidates:
            break

        cost, remove_facility, add_facility, next_solution = min(candidates)

        rows.append(
            {
                "step"              : len(rows)       + 1,
                "removed"           : remove_facility + 1,
                "added"             : add_facility    + 1,
                "cost"              : cost,
                "swap_distance_left": solution_swap_distance(next_solution, target),
                "solution"          : [node + 1 for node in next_solution]         ,
            }
        )

        current = next_solution

        path.append(current)

    return path, rows


def _infer_instance_size(instance: object) -> int | None:
    if isinstance(instance, np.ndarray):
        return int(instance.shape[0])
    if isinstance(instance, dict):
        if "n" in instance:
            return int(instance["n"])
        if "points" in instance:
            return len(instance["points"])
    if hasattr(instance, "get_n"):
        return int(instance.get_n())
    if hasattr(instance, "n"):
        return int(getattr(instance, "n"))
    if hasattr(instance, "points"):
        return len(getattr(instance, "points"))

    return None


def _record_cost(
    record   : dict[str, object],
    solution : tuple[int, ...  ],
    cost_fn  : Callable[[tuple[int, ...]], float],
) -> float:
    if "cost" in record:
        return float(record["cost"])

    return float(cost_fn(solution))


def complete_long_term_memory_with_swap_trajectories(
    instance         : object,
    long_term_memory : list[dict[str, object]],
    *,
    facilities_key   : str = "facilities",
    cost_fn          : Callable[[tuple[int, ...]], float] | None = None,
) -> dict[str, object]:
    if not long_term_memory:
        raise ValueError("Long term memory is empty.")

    n = _infer_instance_size(instance)

    first_vector = np.asarray(long_term_memory[0][facilities_key], dtype=np.int8)
    if n is None:
        n = int(first_vector.size)

    if n != int(first_vector.size):
        raise ValueError("facility vector length does not match instance size.")

    effective_cost_fn = (
        cost_fn
        if   cost_fn is not None
        else lambda solution: pmedian_solution_cost(instance, solution)
    )

    original_records = [
        dict(record)
        for record in long_term_memory
    ]

    for record in original_records:
        vector = np.asarray(record[facilities_key], dtype=np.int8)

        if vector.ndim != 1 or int(vector.size) != n:
            raise ValueError(
                "all facility vectors must be one-dimensional and match instance size."
            )

    original_solutions = [
        facilities_from_binary_vector(record[facilities_key])
        for record in original_records
    ]
    original_costs = [
        _record_cost(record, solution, effective_cost_fn)
        for record, solution in zip(original_records, original_solutions)
    ]
    for record, cost in zip(original_records, original_costs):
        record.setdefault("cost", float(cost))

    original_swap_graph = build_solution_swap_graph(
        original_solutions,
        original_costs    ,
    )
    original_is_connected = (
        original_swap_graph.number_of_nodes() <= 1
        or nx.is_connected(original_swap_graph)
    )

    component_graph, components = build_swap_component_graph(
        original_swap_graph,
        original_solutions ,
    )
    component_mst = nx.minimum_spanning_tree(
        component_graph, weight="weight"
    )

    completed_records   = list(original_records  )
    completed_solutions = list(original_solutions)
    solution_origin     = ["original"] * len(completed_solutions)
    solution_to_index   = {
        solution: index
        for index, solution in enumerate(completed_solutions)
    }
    trajectory_rows = []

    if not original_is_connected:
        for _, _, edge_data in component_mst.edges(data=True):
            source_solution_index = int(edge_data["source_solution"])
            target_solution_index = int(edge_data["target_solution"])

            path, path_rows = greedy_swap_path_between_solutions(
                original_solutions[source_solution_index],
                original_solutions[target_solution_index],
                cost_fn=effective_cost_fn,
            )

            previous_solution = path[0]
            previous_index    = solution_to_index[previous_solution]

            for path_position, next_solution in enumerate(path[1:], start=1):
                if next_solution not in solution_to_index:
                    solution_to_index[next_solution] = len(completed_solutions)

                    completed_solutions.append(next_solution)
                    completed_records  .append(
                        {
                            "cost"       : float(effective_cost_fn(next_solution)),
                            facilities_key: binary_facility_vector(n, next_solution).tolist(),
                        }
                    )
                    solution_origin.append("trajectory")

                next_index = solution_to_index[next_solution    ]
                step_data  = path_rows        [path_position - 1]

                trajectory_rows.append(
                    {
                        "from_solution"        : previous_index + 1,
                        "to_solution"          : next_index     + 1,
                        "removed"              : step_data["removed"           ],
                        "added"                : step_data["added"             ],
                        "cost"                 : step_data["cost"              ],
                        "swap_distance_to_goal": step_data["swap_distance_left"],
                        "solution"             : step_data["solution"          ],
                    }
                )

                previous_solution = next_solution
                previous_index    = next_index

    completed_costs = [
        _record_cost(record, solution, effective_cost_fn)
        for record, solution in zip(completed_records, completed_solutions)
    ]
    completed_swap_graph = build_solution_swap_graph(
        completed_solutions,
        completed_costs    ,
    )
    completed_is_connected = (
        completed_swap_graph.number_of_nodes() <= 1
        or nx.is_connected(completed_swap_graph)
    )

    return {
        "long_term_memory"         : completed_records  ,
        "solutions"                : completed_solutions,
        "solution_origin"          : solution_origin,
        "trajectory_rows"          : trajectory_rows,

        "original_swap_graph"      : original_swap_graph ,
        "completed_swap_graph"     : completed_swap_graph,
        "component_graph"          : component_graph,
        "component_mst"            : component_mst  ,
        "components"               : components     ,

        "original_is_connected"    : original_is_connected ,
        "completed_is_connected"   : completed_is_connected,
        "original_component_count" : len(components),
        "completed_component_count": nx.number_connected_components(completed_swap_graph),
        "original_solution_count"  : len(original_records ),
        "completed_solution_count" : len(completed_records),
        "added_solution_count"     : len(completed_records) - len(original_records),
        "movement_count"           : len(trajectory_rows  ),
    }

import numpy    as np
import networkx as nx

from .graph  import build_top_ltm            , \
                    build_cooccurrence_matrix, \
                    build_unweighted_graph   , \
                    total_edge_count         , \
                    total_edge_weight

from .maxcut import max_p_cut_local_search  , \
                    best_facility_separation


def densest_subgraph_greedy(
    adjacency : np.ndarray,
    min_size  : int = 3   ,
) -> tuple[set[int], float]:
    n = adjacency.shape[0]

    remaining     = set(range(n))
    best_subgraph = set(range(n)) if n and n < min_size else set()
    best_density  = -np.inf

    while len(remaining) >= max(1, min_size):
        nodes    = sorted   (remaining)
        subgraph = adjacency[np.ix_(nodes, nodes)]

        current_weight  = float(np.triu(subgraph, k=1).sum())
        current_density = current_weight / max(1, len(remaining))

        if current_density > best_density:
            best_density  = current_density
            best_subgraph = remaining.copy()

        min_pos = int(
            np.argmin(subgraph.sum(axis=1))
        )
        remaining.remove(nodes[min_pos])

    if not best_subgraph and n:
        best_subgraph = set(range(min(n, max(1, min_size))))
        best_density  = 0.0

    return best_subgraph, max(0.0, float(best_density))


def extract_partition_groups(
    labels : np.ndarray,
    p      : int       ,
) -> list[tuple[int, ...]]:
    return [
        tuple(
            sorted(
                np.flatnonzero(labels == group_id).tolist()
            )
        )
        for group_id in range(p)
    ]


def extract_highest_k_core_nodes(
    graph : nx.Graph
) -> tuple[dict[int, int], int, set[int]]:
    if graph.number_of_edges() == 0:
        core_numbers = {node: 0 for node in graph.nodes}
    else:
        core_numbers = nx.core_number(graph)

    max_core_level = max(core_numbers.values()) if core_numbers else 0

    highest_nodes  = {
        node
        for node, level in core_numbers.items()
        if  level == max_core_level
    }

    return core_numbers, int(max_core_level), highest_nodes


def extract_structure_insights(
    summary : dict[str, object],
    details : dict[str, object],
    *,
    top_fraction       : float,
    max_p_cut_restarts : int  ,
    max_p_cut_max_iter : int  ,
    global_seed        : int  ,
) -> dict[str, object]:
    long_term_memory = details['long_term_memory']

    if not long_term_memory:
        raise ValueError('long_term_memory is empty.')

    analysis_ltm, matrix, costs = build_top_ltm            (long_term_memory, top_fraction)
    adjacency                   = build_cooccurrence_matrix(matrix   )
    unweighted_graph            = build_unweighted_graph   (adjacency)

    best_facilities = tuple(
        sorted(
            int(value) - 1
            for value in summary['tspmed_facilities']
        )
    )
    best_set = set(best_facilities)

    p = int(summary['p'])
    n = int(summary['n'])

    labels_maxcut, cut_weight, internal_weight = max_p_cut_local_search(
        adjacency,
        p        ,
        n_restarts=max_p_cut_restarts,
        max_iter  =max_p_cut_max_iter,
        seed      =global_seed       ,
    )
    max_p_cut_groups = extract_partition_groups(labels_maxcut, p)

    _, max_core_level, highest_core_nodes = extract_highest_k_core_nodes(unweighted_graph)
    densest_nodes    , densest_density    = densest_subgraph_greedy     (
        adjacency, min_size=max(3, p)
    )

    return {
        'best_facilities'           : best_facilities,
        'best_cost'                 : float(summary['tspmed_cost']),
        'memory_size'               : len  (long_term_memory      ),
        'top_solution_count'        : len  (analysis_ltm          ),
        'top_cost_cutoff'           : float(costs.max()           ),
        'cooccurrence_edges'        : total_edge_count (adjacency ),
        'cooccurrence_total_weight' : total_edge_weight(adjacency ),

        'max_p_cut_groups'                   : max_p_cut_groups,
        'max_p_cut_fraction_cut'             : cut_weight / max(1e-12, cut_weight + internal_weight),
        'max_p_cut_best_facility_separation' : best_facility_separation(labels_maxcut, best_set    ),

        'k_core_max_level'          : max_core_level    ,
        'highest_k_core_nodes'      : highest_core_nodes,
        'k_core_candidate_count'    : len(highest_core_nodes)            ,
        'k_core_candidate_fraction' : len(highest_core_nodes) / max(1, n),
        'k_core_best_set_recall'    : len(best_set.intersection(highest_core_nodes)) / max(1, len(best_set)),

        'densest_nodes'                       : densest_nodes  ,
        'densest_subgraph_density'            : densest_density,
        'densest_subgraph_candidate_count'    : len(densest_nodes)            ,
        'densest_subgraph_candidate_fraction' : len(densest_nodes) / max(1, n),
        'densest_subgraph_best_set_recall'    : len(best_set.intersection(densest_nodes)) / max(1, len(best_set)),
    }

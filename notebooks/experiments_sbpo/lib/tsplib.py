from __future__ import annotations

import re
import gzip
import math

from pathlib     import Path
from dataclasses import dataclass
from time        import perf_counter

from urllib.parse   import urljoin
from urllib.request import Request, urlopen

import numpy    as np
import pandas   as pd
import networkx as nx

from .graph   import build_top_ltm
from .maxcut  import max_p_cut_local_search
from .explain import densest_subgraph_greedy, extract_highest_k_core_nodes


TSPLIB_PAGE_URL = "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp.html"
USER_AGENT      = "xopt-tsplib-geometric/1.0"

GEOMETRIC_EDGE_WEIGHT_TYPES = frozenset(
    {
        "ATT"    ,
        "CEIL_2D",
        "EUC_2D" ,
        "GEO"    ,
    }
)

REQUIRED_POINT_COLUMNS = {
    "node"   ,
    "label"  ,
    "lat"    ,
    "long"   ,
    "coord_x",
    "coord_y",
}


@dataclass(frozen=True)
class TSPLIBInstanceSpec:
    name             : str
    dimension        : int
    edge_weight_type : str
    p                : int
    url              : str
    raw_path         : Path


@dataclass(frozen=True)
class GeometricInstance:
    name             : str
    points           : pd.DataFrame
    p                : int
    source           : str
    instance_path    : Path
    edge_weight_type : str
    metadata         : dict[str, object]
    edge_table       : pd.DataFrame | None = None


def fetch_bytes(url: str, *, timeout: int = 60) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})

    with urlopen(request, timeout=timeout) as response:
        return response.read()


def decode_tsp_payload(url: str, payload: bytes) -> str:
    if url.endswith(".gz"):
        payload = gzip.decompress(payload)

    return payload.decode("latin-1")


def fetch_text(url: str, *, timeout: int = 60) -> str:
    return decode_tsp_payload(url, fetch_bytes(url, timeout=timeout))


def discover_tsp_urls(page_url: str = TSPLIB_PAGE_URL) -> list[str]:
    html = fetch_bytes(page_url).decode("latin-1")

    urls_by_name: dict[str, str] = {}
    for href in re.findall(r'href=["\']([^"\']+\.tsp(?:\.gz)?)["\']', html, re.I):
        url = urljoin(page_url, href)

        urls_by_name.setdefault(Path(url).name, url)

    return [
        urls_by_name[name]
        for name in sorted(urls_by_name)
    ]


def parse_tsplib_metadata(text: str) -> dict[str, object]:
    metadata: dict[str, object] = {}

    has_node_coords = False

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        upper = line.upper()

        if upper == "NODE_COORD_SECTION":
            has_node_coords = True
            break

        if upper.endswith("_SECTION") or upper == "EOF":
            break

        if ":" in line:
            key, value = line.split(":", 1)
        else:
            parts = line.split(maxsplit=1)

            if len(parts) != 2:
                continue

            key, value = parts

        metadata[key.strip().upper()] = value.strip()

    metadata["HAS_NODE_COORD_SECTION"] = has_node_coords

    if "DIMENSION" in metadata:
        metadata["DIMENSION"] = int(str(metadata["DIMENSION"]))

    return metadata


def default_p(dimension: int) -> int:
    return max(2, min(5, int(round(math.sqrt(int(dimension))))))


def spec_from_manifest_row(row: dict[str, str], project_root: Path) -> TSPLIBInstanceSpec:
    raw_path = Path(row["raw_path"])

    if not raw_path.is_absolute():
        raw_path = project_root / raw_path

    return TSPLIBInstanceSpec(
        name            =row["instance"],
        url             =row["url"     ],
        raw_path        =raw_path,

        dimension       =int(row["dimension"       ]),
        p               =int(row["p"               ]),
        edge_weight_type=str(row["edge_weight_type"]).upper(),
    )


def spec_to_manifest_row(
    spec         : TSPLIBInstanceSpec,
    project_root : Path,
    notebook_path: Path,
) -> dict[str, object]:
    raw_path      = relative_to_root(spec.raw_path, project_root)
    notebook_path = relative_to_root(notebook_path, project_root)

    return {
        "instance"         : spec.name,
        "url"              : spec.url ,
        "raw_path"         : raw_path ,

        "p"                : int(spec.p          ),
        "dimension"        : int(spec.dimension  ),
        "edge_weight_type" : spec.edge_weight_type,

        "notebook"         : notebook_path,
        "section"          : section_slug(spec.name),
    }


def relative_to_root(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def section_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()

    return slug or "instance"


def collect_geometric_tsplib_specs(
    raw_dir : Path,
    *,
    page_url: str = TSPLIB_PAGE_URL,
) -> tuple[list[TSPLIBInstanceSpec], list[dict[str, str]]]:
    raw_dir.mkdir(parents=True, exist_ok=True)

    specs    : list[TSPLIBInstanceSpec] = []
    failures : list[dict[str, str]    ] = []

    for url in discover_tsp_urls(page_url):
        try:
            tsp_text = fetch_text(url)
        except Exception as exc:
            failures.append(
                {
                    "url"   : url,
                    "error" : f"{type(exc).__name__}: {exc}",
                }
            )

            continue

        metadata         = parse_tsplib_metadata(tsp_text)
        edge_weight_type = str(
            metadata.get("EDGE_WEIGHT_TYPE", "")
        ).upper()

        if str(metadata.get("TYPE", "")).upper() != "TSP":
            continue
        if edge_weight_type not in GEOMETRIC_EDGE_WEIGHT_TYPES:
            continue
        if not metadata.get("HAS_NODE_COORD_SECTION"):
            continue

        raw_path = raw_dir / Path(url).name.removesuffix(".gz")

        raw_path.write_text(tsp_text, encoding="latin-1")

        dimension = int(metadata["DIMENSION"])

        specs.append(
            TSPLIBInstanceSpec(
                name            =str(metadata["NAME"]),
                dimension       =dimension            ,
                edge_weight_type=edge_weight_type     ,
                p               =default_p(dimension ),

                url     =url     ,
                raw_path=raw_path,
            )
        )

    return sorted(specs, key=lambda spec: spec.name.lower()), failures


def _parse_header_line(line: str) -> tuple[str, str] | None:
    if ":" in line:
        key, value = line.split(":", 1)

        return key.strip().upper(), value.strip()

    parts = line.split(maxsplit=1)

    if len(parts) == 2:
        return parts[0].strip().upper(), parts[1].strip()

    return None


def read_tsplib_geometric_tsp(
    path: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    metadata   : dict[str, object             ] = {}
    coord_rows : list[tuple[int, float, float]] = []

    in_coord_section = False

    for raw_line in path.read_text(encoding="latin-1").splitlines():
        line = raw_line.strip()

        if not line:
            continue

        upper = line.upper()

        if upper == "EOF":
            break

        if upper == "NODE_COORD_SECTION":
            in_coord_section = True
            continue

        if upper.endswith("_SECTION"):
            in_coord_section = False
            continue

        if in_coord_section:
            parts = line.split()

            if len(parts) < 3:
                continue

            coord_rows.append(
                (
                    int  (parts[0]) - 1,
                    float(parts[1])    ,
                    float(parts[2])    ,
                )
            )

            continue

        parsed = _parse_header_line(line)

        if parsed is not None:
            key, value    = parsed
            metadata[key] = value

    if not coord_rows:
        raise ValueError(f"No NODE_COORD_SECTION found in {path.name}.")

    edge_weight_type = str(metadata.get("EDGE_WEIGHT_TYPE", "")).upper()

    points = (
        pd.DataFrame(
            coord_rows,
            columns=[
                "node"   ,
                "coord_x",
                "coord_y",
            ],
        )
        .sort_values("node"   )
        .reset_index(drop=True)
    )

    expected = np.arange(len(points), dtype=int)

    if not np.array_equal(points["node"].to_numpy(dtype=int), expected):
        raise ValueError("TSPLIB node ids must be one-based and contiguous.")

    if edge_weight_type == "GEO":
        points["lat" ] = points["coord_x"].map(tsplib_geo_to_degrees)
        points["long"] = points["coord_y"].map(tsplib_geo_to_degrees)
    else:
        points["lat" ] = points["coord_y"].astype(float)
        points["long"] = points["coord_x"].astype(float)

    points["label"] = (points["node"] + 1).astype(str)

    columns = [
        "node"   ,
        "label"  ,
        "lat"    ,
        "long"   ,
        "coord_x",
        "coord_y",
    ]

    return points[columns], metadata


def validate_points(points: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_POINT_COLUMNS.difference(points.columns)

    if missing:
        raise ValueError(f"Missing point columns: {sorted(missing)}")

    result = points.copy()

    result["node"] = result["node"].astype(int)
    result         = (
        result
        .sort_values("node"   )
        .reset_index(drop=True)
    )

    expected = np.arange(len(result)   )
    observed = result["node"].to_numpy()

    if not np.array_equal(observed, expected):
        raise ValueError("Point node ids must be zero-based and contiguous.")

    return result


def load_tsplib_instance(
    spec: TSPLIBInstanceSpec
) -> GeometricInstance:
    points, metadata = read_tsplib_geometric_tsp(spec.raw_path)
    points           = validate_points          (points)

    edge_weight_type = str(metadata.get("EDGE_WEIGHT_TYPE", spec.edge_weight_type)).upper()

    if edge_weight_type != spec.edge_weight_type:
        raise ValueError(
            f"Expected EDGE_WEIGHT_TYPE={spec.edge_weight_type}, found {edge_weight_type}."
        )

    return GeometricInstance(
        name             =f"tsplib_{spec.name}_p{spec.p}",
        points           =points,
        p                =spec.p,

        source           =f"TSPLIB95 TSP geometric instance ({edge_weight_type})",
        instance_path    =spec.raw_path   ,
        edge_weight_type =edge_weight_type,
        metadata         =metadata        ,
    )


def tsplib_geo_to_degrees(value: float) -> float:
    degrees = int(value)
    minutes = value - degrees

    return degrees + minutes * 100.0 / 60.0


def tsplib_geo_to_radians(values: np.ndarray) -> np.ndarray:
    values  = np.asarray(values, dtype=float)
    degrees = np.floor  (values)
    minutes = values - degrees

    return np.pi * (degrees + 5.0 * minutes / 3.0) / 180.0


def tsplib_distance_to_facility(
    instance : GeometricInstance,
    facility : int              ,
) -> np.ndarray:
    points = instance.points
    ewtype = instance.edge_weight_type.upper()

    x  = points["coord_x"].to_numpy(dtype=float)
    y  = points["coord_y"].to_numpy(dtype=float)
    fx = float(x[facility])
    fy = float(y[facility])

    if   ewtype == "EUC_2D" :
        distances = np.floor(np.hypot(x - fx, y - fy) + 0.5)
    elif ewtype == "CEIL_2D":
        distances = np.ceil (np.hypot(x - fx, y - fy))
    elif ewtype == "ATT":
        rij       = np.sqrt (((x - fx) ** 2 + (y - fy) ** 2) / 10.0)
        tij       = np.floor(rij + 0.5)
        distances = tij + (tij < rij)
    elif ewtype == "GEO":
        lat = tsplib_geo_to_radians(x)
        lon = tsplib_geo_to_radians(y)

        q1  = np.cos(lon - lon[facility])
        q2  = np.cos(lat - lat[facility])
        q3  = np.cos(lat + lat[facility])
        arg = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)

        distances = np.floor(
            6378.388 * np.arccos(np.clip(arg, -1.0, 1.0)) + 1.0
        )
    else:
        raise ValueError(f"Unsupported TSPLIB EDGE_WEIGHT_TYPE: {ewtype}")

    distances = distances.astype(np.float64, copy=False)

    distances[int(facility)] = 0.0

    return distances


def assignment_profile(
    instance   : GeometricInstance          ,
    facilities : tuple[int, ...] | list[int],
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    facilities = tuple(int(value) for value in facilities)

    if not facilities:
        raise ValueError("At least one facility is required.")

    distance_matrix = np.column_stack(
        [
            tsplib_distance_to_facility(
                instance, facility
            )
            for facility in facilities
        ]
    )

    best_pos  = np.argmin(distance_matrix, axis=1)
    best_dist = distance_matrix[np.arange(distance_matrix.shape[0]), best_pos]

    if len(facilities) == 1:
        second_dist = np.full_like(best_dist, np.inf, dtype=float)
    else:
        second_pos  = np.argpartition(distance_matrix, kth=1, axis=1)[:, 1]
        second_dist = distance_matrix[np.arange(distance_matrix.shape[0]), second_pos]

    assignments = np.asarray(facilities, dtype=int)[best_pos]

    return float(best_dist.sum()), assignments, best_dist, second_dist, best_pos


def encode_facilities_binary(
    facilities: tuple[int, ...] | list[int], n: int
) -> list[int]:
    values = [0] * int(n)

    for facility in facilities:
        values[int(facility)] = 1

    return values


def _add_solution_to_memory(
    memory     : list [dict [str, object]],
    seen       : set  [tuple[int, ...   ]],
    facilities : tuple[int, ...          ],
    cost       : float,
    n          : int  ,
) -> None:
    key = tuple(sorted(int(value) for value in facilities))

    if key in seen:
        return

    seen  .add   (key)
    memory.append(
        {
            "cost"       : float(cost),
            "facilities" : encode_facilities_binary(key, n),
        }
    )


def farthest_first_initialization(
    instance : GeometricInstance  ,
    p        : int                ,
    rng      : np.random.Generator,
) -> tuple[int, ...]:
    n = len(instance.points  )
    p = max(1, min(int(p), n))

    facilities = [int(rng.integers(0, n))]
    nearest    = tsplib_distance_to_facility(instance, facilities[0])

    while len(facilities) < p:
        candidate = int(np.argmax(nearest))

        if candidate in facilities:
            remaining = np.setdiff1d(
                np.arange (n         , dtype=int),
                np.asarray(facilities, dtype=int),
            )

            candidate = int(rng.choice(remaining))

        facilities.append(candidate)

        nearest = np.minimum(
            nearest,
            tsplib_distance_to_facility(instance, candidate),
        )

    return tuple(sorted(facilities))


def local_search_tsplib_pmedian(
    instance : GeometricInstance,
    initial  : tuple[int, ...  ],
    *,
    max_iter     : int,
    swap_samples : int,
    rng          : np.random.Generator    ,
    memory       : list[dict[str, object]],
    seen         : set[tuple[int, ...]]   ,
) -> tuple[tuple[int, ...], float]:
    n          = len (instance.points)
    facilities = list(initial)

    cost, _, best_dist, second_dist, best_pos = assignment_profile(instance, facilities)

    _add_solution_to_memory(memory, seen, tuple(facilities), cost, n)

    all_nodes = np.arange(n, dtype=int)

    for _ in range(max(1, int(max_iter))):
        closed = np.setdiff1d(
            all_nodes,
            np.asarray(facilities, dtype=int),
            assume_unique=False,
        )

        if closed.size == 0:
            break

        sample_size = min(int(swap_samples), int(closed.size))
        candidates  = rng.choice(closed, size=sample_size, replace=False)
        open_pos    = rng.integers(0, len(facilities), size=sample_size )

        best_candidate_delta = 0.0
        best_candidate_pos   = None
        best_candidate_node  = None

        for remove_pos, candidate in zip(open_pos.tolist(), candidates.tolist()):
            candidate_dist = tsplib_distance_to_facility(instance, int(candidate))

            fallback = np.where  (best_pos == remove_pos, second_dist, best_dist)
            new_dist = np.minimum(fallback, candidate_dist)
            delta    = float     (new_dist.sum() - cost   )

            if delta < best_candidate_delta:
                best_candidate_delta = delta
                best_candidate_pos   = int(remove_pos)
                best_candidate_node  = int(candidate )

        if best_candidate_pos is None:
            break

        facilities[best_candidate_pos] = best_candidate_node
        facilities = sorted(facilities)

        cost, _, best_dist, second_dist, best_pos = assignment_profile(instance, facilities)

        _add_solution_to_memory(memory, seen, tuple(facilities), cost, n)

    return tuple(sorted(facilities)), float(cost)


def solve_tsplib_geometric_pmedian(
    instance : GeometricInstance,
    *,
    restarts       : int,
    max_iter       : int,
    swap_samples   : int,
    details_format : str = "binary",
    seed           : int = 42      ,
) -> tuple[dict[str, object], dict[str, object]]:
    if details_format != "binary":
        raise ValueError("TSPLIB geometric analysis uses binary long-term-memory records.")

    rng = np.random.default_rng(seed)

    n = len(instance.points)
    p = max(1, min(int(instance.p), n))

    memory: list[dict[str, object]] = []
    seen  : set [tuple[int, ...  ]] = set()

    medoids_details = []
    tspmed_details  = []

    best_initial_cost = float("inf")
    best_cost         = float("inf")
    best_initial      = None
    best_facilities   = None

    for restart in range(max(1, int(restarts))):
        initial                  = farthest_first_initialization(instance, p, rng )
        initial_cost, _, _, _, _ = assignment_profile           (instance, initial)

        medoids_details.append(
            {
                "restart" : restart            ,
                "cost"    : float(initial_cost),
                "medoids" : encode_facilities_binary(initial, n),
            }
        )

        if initial_cost < best_initial_cost:
            best_initial_cost = float(initial_cost)
            best_initial      = initial

        facilities, cost = local_search_tsplib_pmedian(
            instance,
            initial ,
            max_iter    =max_iter    ,
            swap_samples=swap_samples,
            rng         =rng   ,
            memory      =memory,
            seen        =seen  ,
        )

        tspmed_details.append(
            {
                "restart"    : restart    ,
                "cost"       : float(cost),
                "facilities" : encode_facilities_binary(facilities, n),
            }
        )

        if cost < best_cost:
            best_cost       = float(cost)
            best_facilities = facilities

    if best_initial is None or best_facilities is None:
        raise RuntimeError("TSPLIB p-median solver did not produce a solution.")

    summary = {
        "instance" : str(instance.instance_path),
        "n"        : n,
        "p"        : p,

        "kmedoids_cost"     : best_initial_cost,
        "tspmed_cost"       : best_cost        ,
        "long_term_mem"     : len(memory)      ,
        "medoids"           : [int(value) + 1 for value in best_initial   ],
        "tspmed_facilities" : [int(value) + 1 for value in best_facilities],
    }

    details = {
        "long_term_memory" : memory         ,
        "medoids"          : medoids_details,
        "tspmed"           : tspmed_details ,
    }

    return summary, details


def nearest_selected_nodes(
    instance : GeometricInstance,
    selected : tuple[int, ...]  ,
) -> np.ndarray:
    _, assignments, _, _, _ = assignment_profile(instance, selected)

    return assignments


def solve_geometric_instance(
    instance      : GeometricInstance,
    solver_params : dict[str, object],
) -> dict[str, object]:
    start = perf_counter()

    summary, details = solve_tsplib_geometric_pmedian(
        instance, **solver_params
    )

    runtime = perf_counter() - start

    selected = tuple(
        sorted(
            int(node) - 1
            for node in summary["tspmed_facilities"]
        )
    )

    assignments = nearest_selected_nodes(instance, selected)

    return {
        "instance"    : instance   ,
        "summary"     : summary    ,
        "details"     : details    ,
        "selected"    : selected   ,
        "assignments" : assignments,
        "runtime"     : runtime    ,
    }


def active_cooccurrence_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    active_nodes = np.flatnonzero(matrix.sum(axis=0) > 0).astype(int)

    if active_nodes.size == 0:
        return active_nodes, np.zeros((0, 0), dtype=np.int64)

    active_matrix = matrix[:, active_nodes]
    adjacency     = np.asarray(active_matrix.T @ active_matrix, dtype=np.int64)

    np.fill_diagonal(adjacency, 0)

    return active_nodes, adjacency


def build_active_unweighted_graph(
    active_nodes : np.ndarray,
    adjacency    : np.ndarray,
) -> nx.Graph:
    graph = nx.Graph()

    graph.add_nodes_from(
        active_nodes.astype(int).tolist()
    )

    rows, cols = np.where(np.triu(adjacency, k=1) > 0)

    graph.add_edges_from (
        (
            int(active_nodes[row]),
            int(active_nodes[col]),
        )
        for row, col in zip(rows.tolist(), cols.tolist())
    )

    return graph


def max_k_cut_connected_partition(
    active_nodes       : np.ndarray,
    adjacency          : np.ndarray,
    p                  : int,
    n                  : int,
    *,
    max_cut_restarts   : int = 40  ,
    max_cut_max_iter   : int = 2000,
    seed               : int = 42  ,
) -> tuple[np.ndarray, list[tuple[int, ...]], float, float, np.ndarray]:
    labels = np.full(n, -1, dtype=int)

    if adjacency.size == 0 or active_nodes.size == 0:
        return labels, [], 0.0, 0.0, np.arange(n, dtype=int)

    connected_positions = np.flatnonzero(adjacency.sum(axis=1) > 0).astype(int)

    if connected_positions.size == 0:
        return labels, [], 0.0, 0.0, np.arange(n, dtype=int)

    connected_nodes      = active_nodes[connected_positions]
    connected_adjacency  = adjacency   [np.ix_(connected_positions, connected_positions)]
    active_p             = max(1, min(int(p), int(connected_positions.size)))
    effective_restarts   = min(
        int(max_cut_restarts),
        max(5, 2400 // max(1, int(connected_positions.size))),
    )
    effective_iterations = min(
        int(max_cut_max_iter),
        800 if connected_positions.size > 400 else int(max_cut_max_iter),
    )

    active_labels, cut_weight, internal_weight = max_p_cut_local_search(
        connected_adjacency,
        active_p           ,
        n_restarts=effective_restarts  ,
        max_iter  =effective_iterations,
        seed      =seed,
    )

    labels[connected_nodes] = active_labels

    groups = [
        tuple(
            sorted(
                connected_nodes[
                    np.flatnonzero(
                        active_labels == group_id
                    )
                ]
                .astype(int)
                .tolist()
            )
        )
        for group_id in range(active_p)
    ]

    ignored_nodes = np.flatnonzero(labels < 0).astype(int)

    return labels, groups, cut_weight, internal_weight, ignored_nodes


def extract_geometric_structures(
    result            : dict[str, object],
    *,
    top_fraction      : float = 1.0 ,
    max_cut_restarts  : int   = 40  ,
    max_cut_max_iter  : int   = 2000,
    seed              : int   = 42  ,
) -> dict[str, object]:
    instance = result["instance"]
    details  = result["details" ]
    n        = len(instance.points)

    top_ltm, matrix, costs  = build_top_ltm                (details["long_term_memory"], top_fraction)
    active_nodes, adjacency = active_cooccurrence_matrix   (matrix                 )
    graph                   = build_active_unweighted_graph(active_nodes, adjacency)

    labels_maxcut, maxcut_groups, cut_weight, internal_weight, maxcut_ignored_nodes = (
        max_k_cut_connected_partition(
            active_nodes,
            adjacency   ,
            instance.p  ,
            n           ,
            max_cut_restarts=max_cut_restarts,
            max_cut_max_iter=max_cut_max_iter,
            seed=seed,
        )
    )

    core_numbers, max_core_level, highest_core_nodes = extract_highest_k_core_nodes(graph)
    densest_positions, densest_density               = densest_subgraph_greedy     (
        adjacency,
        min_size=max(3, min(instance.p, max(1, int(active_nodes.size)))),
    )
    densest_nodes = {
        int(active_nodes[position])
        for position in densest_positions
        if 0 <= int(position) < len(active_nodes)
    }

    return {
        "top_ltm"      : top_ltm     ,
        "matrix"       : matrix      ,
        "costs"        : costs       ,
        "active_nodes" : active_nodes,
        "adjacency"    : adjacency   ,
        "graph"        : graph       ,

        "labels_maxcut"          : labels_maxcut       ,
        "maxcut_groups"          : maxcut_groups       ,
        "maxcut_ignored_nodes"   : maxcut_ignored_nodes,
        "maxcut_cut_weight"      : cut_weight          ,
        "maxcut_internal_weight" : internal_weight     ,

        "core_numbers"       : core_numbers      ,
        "max_core_level"     : max_core_level    ,
        "highest_core_nodes" : highest_core_nodes,
        "densest_nodes"      : densest_nodes     ,
        "densest_density"    : densest_density   ,
    }


def run_tsplib_instance_analysis(
    spec               : TSPLIBInstanceSpec,
    *,
    solver_params      : dict[str, object],
    top_fraction       : float = 1.0 ,
    max_cut_restarts   : int   = 40  ,
    max_cut_max_iter   : int   = 2000,
    seed               : int   = 42  ,
) -> dict[str, object]:
    instance = load_tsplib_instance    (spec)
    result   = solve_geometric_instance(instance, solver_params)

    result["structures"] = extract_geometric_structures(
        result,
        top_fraction    =top_fraction    ,
        max_cut_restarts=max_cut_restarts,
        max_cut_max_iter=max_cut_max_iter,
        seed            =seed            ,
    )

    return result


def labels_for_nodes(
    points : pd.DataFrame,
    nodes  : set[int] | tuple[int, ...] | list[int],
) -> list[str]:
    return [
        str(points.loc[node, "label"])
        for node in sorted(nodes)
    ]


def solution_table_for_result(result: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "instance"        : result["instance"].name  ,
                "source"          : result["instance"].source,

                "n"               : result["summary"]["n"            ],
                "p"               : result["summary"]["p"            ],
                "tspmed_cost"     : result["summary"]["tspmed_cost"  ],
                "long_term_mem"   : result["summary"]["long_term_mem"],
                "runtime_seconds" : result["runtime"],

                "selected_labels" : [
                    result["instance"].points.loc[node, "label"]
                    for node in result["selected"]
                ],
            }
        ]
    )


def structure_table_for_result(result: dict[str, object]) -> pd.DataFrame:
    structures = result["structures"]

    return pd.DataFrame(
        [
            {
                "instance"           : result["instance"].name,

                "top_solution_count" : len  (structures["top_ltm"]),
                "top_cost_min"       : float(structures["costs"  ].min()),
                "top_cost_max"       : float(structures["costs"  ].max()),

                "cooccurrence_edges" : int(
                    np.count_nonzero(np.triu(structures["adjacency"], 1) > 0)
                ),
                "max_core_level"       :     structures["max_core_level"      ] ,
                "highest_core_size"    : len(structures["highest_core_nodes"  ]),
                "densest_size"         : len(structures["densest_nodes"       ]),
                "densest_density"      :     structures["densest_density"     ] ,
                "maxcut_ignored_nodes" : len(structures["maxcut_ignored_nodes"]),
                "maxcut_fraction_cut"  : (
                    structures["maxcut_cut_weight"] /
                    max(
                        1e-12,
                        structures["maxcut_cut_weight"     ] +
                        structures["maxcut_internal_weight"],
                    )
                ),
            }
        ]
    )


def structure_nodes_table_for_result(result: dict[str, object]) -> pd.DataFrame:
    instance   = result["instance"  ]
    structures = result["structures"]

    return pd.DataFrame(
        [
            {
                "instance" : instance.name,

                "selected_medians"     : labels_for_nodes(instance.points, result    ["selected"          ]),
                "densest_nodes"        : labels_for_nodes(instance.points, structures["densest_nodes"     ]),
                "highest_k_core_nodes" : labels_for_nodes(instance.points, structures["highest_core_nodes"]),
                "max_k_cut_groups"     : [
                    labels_for_nodes(instance.points, group)
                    for group in structures["maxcut_groups"]
                ],
            }
        ]
    )

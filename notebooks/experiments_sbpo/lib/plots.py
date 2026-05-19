from __future__ import annotations

import io
import math

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from IPython.display  import Image  , display
from urllib.request   import Request, urlopen

from .tsplib import GeometricInstance


MAP_TILE_ZOOM_RANGE = (5, 13)
MAP_TILE_MAX_TILES  = 48
MAP_TILE_TIMEOUT    = 8
MAP_TILE_ALPHA      = 0.52
MAP_ATTRIBUTION     = "Map data (C) OpenStreetMap contributors"
MAP_TILE_URL        = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

SHOW_MAP_BACKGROUND = True
HEATMAP_MAX_POINTS  = 1200
ASSIGNMENT_LINE_MAX = 1200
IGNORED_NODE_MAX    = 6000

_BASEMAP_TILE_CACHE: dict[tuple[int, int, int], np.ndarray] = {}
_BASEMAP_WARNED                                             = False


def _axis_limits(
    points       : pd.DataFrame,
    pad_fraction : float = 0.08,
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min, x_max = points["long"].min(), points["long"].max()
    y_min, y_max = points["lat" ].min(), points["lat" ].max()

    x_pad = max(1e-6, (x_max - x_min) * pad_fraction)
    y_pad = max(1e-6, (y_max - y_min) * pad_fraction)

    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def _lonlat_to_tile(
    longitude : float,
    latitude  : float,
    zoom      : int  ,
) -> tuple[int, int]:
    latitude = float(np.clip(latitude, -85.05112878, 85.05112878))

    n_tiles = 2 ** int(zoom)
    x_tile  = int((longitude + 180.0) / 360.0 * n_tiles)
    lat_rad = math.radians(latitude)
    y_tile  = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n_tiles)

    return max(0, min(n_tiles - 1, x_tile)), max(0, min(n_tiles - 1, y_tile))


def _tile_to_lonlat(x_tile: int, y_tile: int, zoom: int) -> tuple[float, float]:
    n_tiles = 2 ** int(zoom)
    lon     = x_tile / n_tiles * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y_tile / n_tiles)))

    return lon, math.degrees(lat_rad)


def _tile_lonlat_bounds(
    x_tile : int,
    y_tile : int,
    zoom   : int,
) -> tuple[float, float, float, float]:
    west, north = _tile_to_lonlat(x_tile    , y_tile    , zoom)
    east, south = _tile_to_lonlat(x_tile + 1, y_tile + 1, zoom)

    return west, east, south, north


def _tile_ranges_for_view(
    xlim : tuple[float, float],
    ylim : tuple[float, float],
    zoom : int,
) -> tuple[range, range]:
    lon_min, lon_max = sorted(xlim)
    lat_min, lat_max = sorted(ylim)

    x_min, y_min = _lonlat_to_tile(lon_min, lat_max, zoom)
    x_max, y_max = _lonlat_to_tile(lon_max, lat_min, zoom)

    return range(x_min, x_max + 1), range(y_min, y_max + 1)


def _choose_tile_zoom(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[int, range, range]:
    min_zoom, max_zoom = MAP_TILE_ZOOM_RANGE

    for zoom in range(max_zoom, min_zoom - 1, -1):
        x_tiles, y_tiles = _tile_ranges_for_view(xlim, ylim, zoom)

        if len(x_tiles) * len(y_tiles) <= MAP_TILE_MAX_TILES:
            return zoom, x_tiles, y_tiles

    x_tiles, y_tiles = _tile_ranges_for_view(xlim, ylim, min_zoom)

    return min_zoom, x_tiles, y_tiles


def _load_map_tile(zoom: int, x_tile: int, y_tile: int) -> np.ndarray:
    key = (int(zoom), int(x_tile), int(y_tile))

    if key not in _BASEMAP_TILE_CACHE:
        url     = MAP_TILE_URL.format(z=zoom, x=x_tile, y=y_tile)
        request = Request(url, headers={"User-Agent": "xopt-geometric-pmedian-notebook/1.0"})

        with urlopen(request, timeout=MAP_TILE_TIMEOUT) as response:
            _BASEMAP_TILE_CACHE[key] = plt.imread(io.BytesIO(response.read()), format="png")

    return _BASEMAP_TILE_CACHE[key]


def _points_look_geographic(points: pd.DataFrame) -> bool:
    if points.empty:
        return False

    longitudes = points["long"].to_numpy(dtype=float)
    latitudes  = points["lat" ].to_numpy(dtype=float)

    return bool(
        np.isfinite(longitudes).all()
        and np.isfinite(latitudes).all()
        and (longitudes >= -180.0).all()
        and (longitudes <=  180.0).all()
        and (latitudes  >=  -90.0).all()
        and (latitudes  <=   90.0).all()
    )


def _resolve_show_map_background(
    instance,
    points: pd.DataFrame,
    show_map_background: bool | None,
) -> bool:
    if show_map_background is not None:
        return bool(show_map_background)

    edge_weight_type = getattr(instance, "edge_weight_type", None)

    if edge_weight_type is not None:
        return SHOW_MAP_BACKGROUND and str(edge_weight_type).upper() == "GEO"

    return SHOW_MAP_BACKGROUND and _points_look_geographic(points)


def _add_map_background(
    ax,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    show_map_background: bool,
) -> None:
    global _BASEMAP_WARNED

    if not show_map_background:
        return

    try:
        zoom, x_tiles, y_tiles = _choose_tile_zoom(xlim, ylim)

        for x_tile in x_tiles:
            for y_tile in y_tiles:
                ax.imshow(
                    _load_map_tile(zoom, x_tile, y_tile),
                    extent       =_tile_lonlat_bounds(x_tile, y_tile, zoom),
                    origin       ="upper"       ,
                    interpolation="bilinear"    ,
                    alpha        =MAP_TILE_ALPHA,
                    zorder       =0             ,
                )

        ax.text(
            0.99,
            0.01,
            MAP_ATTRIBUTION,
            transform=ax.transAxes,
            ha       ="right"     ,
            va       ="bottom"    ,
            fontsize =6           ,
            color    ="#333333" ,
            alpha    =0.72        ,
            zorder   =10          ,
        )
    except Exception as exc:
        if not _BASEMAP_WARNED:
            print(f"Map background unavailable: {exc}")

            _BASEMAP_WARNED = True


def _setup_axis(
    ax,
    points: pd.DataFrame,
    title : str         ,
    *,
    show_map_background: bool,
) -> None:
    xlim, ylim = _axis_limits(points)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    _add_map_background(ax, xlim, ylim, show_map_background=show_map_background)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title (title)
    ax.grid      (alpha=0.12, linewidth=0.6)
    ax.set_aspect("equal", adjustable="box")


def _plot_network_edges(
    ax,
    instance: GeometricInstance,
    *,
    color     : str   = "#c8c8c8",
    alpha     : float = 0.45,
    linewidth : float = 1.0 ,
) -> None:
    if instance.edge_table is None:
        return

    points = instance.points.set_index("node")

    for row in instance.edge_table.itertuples(index=False):
        p1 = points.loc[int(row.u)]
        p2 = points.loc[int(row.v)]

        ax.plot(
            [p1["long"], p2["long"]],
            [p1["lat" ], p2["lat" ]],
            color    =color    ,
            alpha    =alpha    ,
            linewidth=linewidth,
            zorder   =1        ,
        )


def sampled_node_ids(n: int, limit: int, seed: int = 42) -> np.ndarray:
    if n <= limit:
        return np.arange(n, dtype=int)

    rng = np.random.default_rng(seed)

    return np.sort(rng.choice(np.arange(n, dtype=int), size=int(limit), replace=False))


def solution_color_by_facility(
    result: dict[str, object]
) -> dict[int, object]:
    selected_sorted = sorted(
        int(node)
        for node in result["selected"]
    )
    palette = plt.colormaps["tab20"].resampled(max(1, len(selected_sorted)))

    return {
        facility: palette(idx)
        for idx, facility in enumerate(selected_sorted)
    }


def plot_geometric_instance_map(
    instance: GeometricInstance,
    ax=None,
    *,
    show_map_background: bool | None = None,
) -> None:
    points = instance.points

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    show_map_background = _resolve_show_map_background(
        instance,
        points,
        show_map_background,
    )

    _setup_axis(
        ax,
        points,
        f"{instance.name}",
        show_map_background=show_map_background,
    )
    _plot_network_edges(ax, instance, color="#404040", alpha=0.48, linewidth=1.15)

    if "kind" in points.columns:
        colors = (
            points["kind"]
            .map(
                {
                    "current"        : "#2b8cbe",
                    "new_unweighted" : "#f03b20",
                }
            )
            .fillna("#2c7fb8")
        )
    else:
        colors = "#2c7fb8"

    point_size = 48 if len(points) <= 30 else 20

    ax.scatter(
        points["long"],
        points["lat" ],
        s        =point_size,
        c        =colors,
        edgecolor="white",
        linewidth=0.35,
        zorder   =3,
    )

    if len(points) <= 30 and "label" in points.columns:
        for row in points.itertuples(index=False):
            ax.annotate(
                str(row.label),
                (row.long, row.lat),
                xytext    =(4, 4),
                textcoords="offset points",
                fontsize  =7,
                zorder    =4,
            )


def plot_solution_map(
    result: dict[str, object],
    ax=None,
    *,
    show_map_background: bool | None = None,
    seed               : int         = 42  ,
) -> None:
    instance = result["instance"]

    points      = instance.points
    selected    = set(int(node) for node in result["selected"])
    assignments = result["assignments"]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    show_map_background = _resolve_show_map_background(
        instance,
        points,
        show_map_background,
    )

    _setup_axis(
        ax    ,
        points,
        f"{instance.name}",
        show_map_background=show_map_background,
    )
    _plot_network_edges(ax, instance)

    color_by_facility = solution_color_by_facility(result)
    points_by_node    = points.set_index("node", drop=False)

    line_nodes = sampled_node_ids(len(points), ASSIGNMENT_LINE_MAX, seed=seed)

    for node in line_nodes.tolist():
        facility = int(assignments[node])

        if node == facility:
            continue

        p1 = points_by_node.loc[node    ]
        p2 = points_by_node.loc[facility]

        ax.plot(
            [p1["long"], p2["long"]],
            [p1["lat" ], p2["lat" ]],
            color    =color_by_facility.get(facility, "#999999"),
            linewidth=0.45 if len(points) > 1200 else 0.8,
            alpha    =0.30 if len(points) > 1200 else 1.0,
            zorder   =2,
        )

    node_colors = [
        color_by_facility.get(int(assignments[node]), "#999999")
        for node in points["node"]
    ]
    selected_points = points[points["node"].isin(selected)]
    point_size      = 7 if len(points) > 3000 else 14 if len(points) > 800 else 26

    ax.scatter(
        points["long"],
        points["lat" ],
        s        =point_size ,
        c        =node_colors,
        alpha    =0.62 if len(points) > 800 else 1.0,
        edgecolor="none" if len(points) > 800 else "white",
        linewidth=0.0 if len(points) > 800 else 0.3,
        zorder   =3,
    )
    ax.scatter(
        selected_points["long"],
        selected_points["lat" ],
        s        =155,
        c        ="#111111",
        marker   ="*"    ,
        edgecolor="white",
        linewidth=0.8,
        zorder   =4  ,
    )


def plot_densest_subgraph_map(
    result: dict[str, object],
    ax=None,
    *,
    show_map_background: bool | None = None,
) -> None:
    instance   = result["instance"  ]
    structures = result["structures"]
    points     = instance.points
    densest    = set(int(node) for node in structures["densest_nodes"])
    selected   = set(int(node) for node in result["selected"])

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    show_map_background = _resolve_show_map_background(
        instance,
        points,
        show_map_background,
    )

    _setup_axis(
        ax,
        points,
        f"{instance.name}",
        show_map_background=show_map_background,
    )
    _plot_network_edges(ax, instance)

    densest_unselected = densest - selected

    if densest_unselected:
        highlighted = points[points["node"].isin(densest_unselected)]

        ax.scatter(
            highlighted["long"],
            highlighted["lat" ],
            s        =90,
            c        ="#e66101",
            edgecolor="#4d2600",
            linewidth=0.7,
            zorder   =4,
        )

    selected_points = points[points["node"].isin(selected)]

    if not selected_points.empty:
        ax.scatter(
            selected_points["long"],
            selected_points["lat" ],
            s        =120,
            marker   ="^",
            c        ="#e61100",
            edgecolor="#4d2600",
            linewidth=0.8,
            zorder   =5,
        )


def plot_k_number_heatmap(
    result: dict[str, object],
    ax=None,
    *,
    show_map_background: bool | None = None,
) -> None:
    instance     = result["instance"  ]
    core_numbers = result["structures"]["core_numbers"]
    points       = instance.points
    selected     = set(int(node) for node in result["selected"])

    values = np.asarray(
        [
            core_numbers.get(int(node), 0)
            for node in points["node"]
        ],
        dtype=float,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    show_map_background = _resolve_show_map_background(
        instance,
        points,
        show_map_background,
    )

    _setup_axis(
        ax,
        points,
        f"{instance.name}",
        show_map_background=show_map_background,
    )
    _plot_network_edges(ax, instance, alpha=0.22)

    x = points["long"].to_numpy(dtype=float)
    y = points["lat" ].to_numpy(dtype=float)

    if len(points) <= HEATMAP_MAX_POINTS:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        grid_x = np.linspace(xlim[0], xlim[1], 160)
        grid_y = np.linspace(ylim[0], ylim[1], 160)
        gx, gy = np.meshgrid(grid_x, grid_y)

        dx = gx[..., None] - x[None, None, :]
        dy = gy[..., None] - y[None, None, :]
        squared_distance = dx * dx + dy * dy

        bandwidth = max(
            (xlim[1] - xlim[0]) ** 2 + (ylim[1] - ylim[0]) ** 2,
            1e-12,
        ) / 250.0

        weights = 1.0 / np.maximum(squared_distance, bandwidth)
        heat    = (weights * values[None, None, :]).sum(axis=2) / weights.sum(axis=2)

        ax.contourf(
            gx,
            gy,
            heat,
            levels=18,
            alpha =0.78,
            zorder=1,
            cmap  ="inferno",
        )

    selected_mask       = points["node"].isin(selected).to_numpy()
    non_selected_points = points.loc[~selected_mask]
    non_selected_values = values[~selected_mask]
    selected_points     = points.loc[selected_mask]
    selected_values     = values[selected_mask]
    color_min           = float(values.min()) if values.size else 0.0
    color_max           = float(values.max()) if values.size else 1.0

    if not non_selected_points.empty:
        point_sizes = np.where(
            non_selected_values == 0,
            6  if len(points) > 3000 else 17,
            28 if len(points) > 3000 else 34,
        )

        ax.scatter(
            non_selected_points["long"],
            non_selected_points["lat" ],
            c        =non_selected_values,
            s        =point_sizes,
            linewidth=0.35 if len(points) <= 3000 else 0.0,
            zorder   =3,
            cmap     ="inferno",
            vmin     =color_min,
            vmax     =color_max,
            edgecolor="white" if len(points) <= 3000 else "none",
            alpha    =0.72 if len(points) > 3000 else 1.0,
        )

    if not selected_points.empty:
        ax.scatter(
            selected_points["long"],
            selected_points["lat" ],
            c        =selected_values,
            s        =95,
            marker   ="^",
            linewidth=0.7,
            zorder   =4,
            cmap     ="inferno",
            vmin     =color_min,
            vmax     =color_max,
            edgecolor="#111111",
        )


def max_k_cut_color_by_selected_facility(
    result: dict[str, object]
) -> dict[int, object]:
    labels            = result["structures"]["labels_maxcut"]
    color_by_facility = solution_color_by_facility(result)
    color_by_cut      = {}

    for facility, color in color_by_facility.items():
        if 0 <= facility < len(labels):
            cut = int(labels[facility])

            if cut >= 0 and cut not in color_by_cut:
                color_by_cut[cut] = color

    return color_by_cut


def plot_max_k_cut_clusters(
    result: dict[str, object],
    ax=None,
    *,
    show_map_background: bool | None = None,
    seed               : int         = 42,
    group_colors       : dict[int, object] | None = None,
) -> None:
    instance = result["instance"  ]
    labels   = result["structures"]["labels_maxcut"]
    points   = instance.points
    selected = set(int(node) for node in result["selected"])

    active_nodes  = np.flatnonzero(labels >= 0)
    ignored_nodes = np.asarray(
        result["structures"].get("maxcut_ignored_nodes", np.flatnonzero(labels < 0)),
        dtype=int,
    )
    group_colors = {
        int(group): color
        for group, color in (group_colors or {}).items()
    }

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    show_map_background = _resolve_show_map_background(
        instance,
        points,
        show_map_background,
    )

    _setup_axis(
        ax,
        points,
        f"{instance.name}",
        show_map_background=show_map_background,
    )
    _plot_network_edges(ax, instance, alpha=0.25)

    ignored_set = set(int(node) for node in ignored_nodes)

    ignored_unselected_nodes = np.asarray(
        sorted(ignored_set - selected),
        dtype=int,
    )
    selected_ignored_nodes = np.asarray(
        sorted(ignored_set & selected),
        dtype=int,
    )

    if ignored_unselected_nodes.size > IGNORED_NODE_MAX:
        rng = np.random.default_rng(seed)
        ignored_unselected_nodes = np.sort(
            rng.choice(
                ignored_unselected_nodes,
                size=IGNORED_NODE_MAX,
                replace=False,
            )
        )

    points_by_node = points.set_index("node", drop=False)

    if ignored_unselected_nodes.size:
        ignored_points = points_by_node.loc[ignored_unselected_nodes]

        ax.scatter(
            ignored_points["long"],
            ignored_points["lat" ],
            s        =5 if len(points) > 3000 else 29,
            c        ="#c7c7c7",
            alpha    =0.35 if len(points) > 3000 else 0.45,
            edgecolor="none" if len(points) > 3000 else "white",
            linewidth=0.0 if len(points) > 3000 else 0.2,
            zorder   =2,
        )

    if active_nodes.size:
        active_labels = labels[active_nodes]
        palette       = plt.colormaps["tab20"].resampled(
            max(1, int(active_labels.max()) + 1)
        )

        def color_for_cut(cut: int) -> object:
            return group_colors.get(int(cut), palette(int(cut)))

        active_unselected_nodes = np.asarray(
            [
                int(node)
                for node in active_nodes
                if int(node) not in selected
            ],
            dtype=int,
        )
        selected_active_nodes = np.asarray(
            [
                int(node)
                for node in active_nodes
                if int(node) in selected
            ],
            dtype=int,
        )

        if active_unselected_nodes.size:
            active_points = points_by_node.loc[active_unselected_nodes]
            colors        = [
                color_for_cut(int(labels[node]))
                for node in active_unselected_nodes
            ]

            ax.scatter(
                active_points["long"],
                active_points["lat" ],
                s        =42 if len(points) > 3000 else 58,
                c        =colors,
                edgecolor="white",
                linewidth=0.35,
                zorder   =3,
            )

        if selected_active_nodes.size:
            selected_points = points_by_node.loc[selected_active_nodes]
            selected_colors = [
                color_for_cut(int(labels[node]))
                for node in selected_active_nodes
            ]

            ax.scatter(
                selected_points["long"],
                selected_points["lat" ],
                s        =125,
                marker   ="^",
                c        =selected_colors,
                edgecolor="#111111",
                linewidth=0.75,
                zorder   =4,
            )

        handles = [
            Line2D(
                [0],
                [0],
                marker          ="o",
                linestyle       ="",
                markerfacecolor =color_for_cut(group),
                markeredgecolor ="white",
                label           =f"Grupo {group}",
            )
            for group in sorted(np.unique(active_labels).tolist())
        ]

        ax.legend(handles=handles, loc="best", fontsize=8, frameon=True)

    if selected_ignored_nodes.size:
        selected_ignored_points = points_by_node.loc[selected_ignored_nodes]

        ax.scatter(
            selected_ignored_points["long"],
            selected_ignored_points["lat" ],
            s        =125,
            marker   ="^",
            c        ="#8c8c8c",
            edgecolor="#111111",
            linewidth=0.75,
            zorder   =4,
        )


def plot_instance_geometric_summary(
    result: dict[str, object],
    *,
    figsize             : tuple[float, float] = (10, 18),
    show_map_background : bool | None = None,
    seed                : int         = 42,
    display_output      : bool        = True,
    dpi                 : int         = 160,
):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=4,
        figsize           =figsize,
        constrained_layout=True,
    )

    plot_solution_map(
        result,
        ax=axes[0],
        show_map_background=show_map_background,
        seed=seed,
    )
    plot_max_k_cut_clusters(
        result,
        ax=axes[1],
        show_map_background=show_map_background,
        seed=seed,
        group_colors=max_k_cut_color_by_selected_facility(result),
    )
    plot_densest_subgraph_map(
        result,
        ax=axes[2],
        show_map_background=show_map_background,
    )
    plot_k_number_heatmap(
        result,
        ax=axes[3],
        show_map_background=show_map_background,
    )

    axes[0].set_title("Melhor solução")
    axes[1].set_title("Max-k-cut"     )
    axes[2].set_title("Subgrafo Denso")
    axes[3].set_title("Número K"      )

    if not display_output:
        return fig

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    display(Image(data=buffer.getvalue()))
    plt.close(fig)

    return None


def plot_instance_geometric_summary2(
    result: dict[str, object],
    *,
    figsize             : tuple[float, float] = (10, 18),
    show_map_background : bool | None = None,
    seed                : int         = 42,
    display_output      : bool        = True,
    dpi                 : int         = 160,
):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize           =figsize,
        constrained_layout=True   ,
    )

    plot_solution_map(
        result,
        ax=axes[0, 0],
        show_map_background=show_map_background,
        seed=seed,
    )
    plot_max_k_cut_clusters(
        result,
        ax=axes[0, 1],
        show_map_background=show_map_background,
        seed=seed,
        group_colors=max_k_cut_color_by_selected_facility(result),
    )
    plot_densest_subgraph_map(
        result,
        ax=axes[1, 0],
        show_map_background=show_map_background,
    )
    plot_k_number_heatmap(
        result,
        ax=axes[1, 1],
        show_map_background=show_map_background,
    )

    axes[0, 0].set_title("Melhor solução")
    axes[0, 1].set_title("Max-k-cut"     )
    axes[1, 0].set_title("Subgrafo Denso")
    axes[1, 1].set_title("Número K"      )

    if not display_output:
        return fig

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    display(Image(data=buffer.getvalue()))
    plt.close(fig)

    return None

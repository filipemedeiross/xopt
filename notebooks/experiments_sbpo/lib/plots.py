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
        request = Request(url, headers={"User-Agent": "xopt-tsplib-geometric-plots/1.0"})

        with urlopen(request, timeout=MAP_TILE_TIMEOUT) as response:
            _BASEMAP_TILE_CACHE[key] = plt.imread(io.BytesIO(response.read()), format="png")

    return _BASEMAP_TILE_CACHE[key]


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


def plot_solution_map(
    result: dict[str, object],
    ax=None,
    *,
    show_map_background: bool = False,
    seed               : int  = 42   ,
) -> None:
    instance = result["instance"]

    points      = instance.points
    selected    = set(result["selected"])
    assignments = result["assignments"]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    _setup_axis(
        ax    ,
        points,
        f"{instance.name}",
        show_map_background=show_map_background,
    )
    _plot_network_edges(ax, instance)

    palette = plt.colormaps["tab20"].resampled(max(1, len(selected)))

    selected_sorted   = sorted(selected)
    color_by_facility = {
        facility : palette(idx)
        for idx, facility in enumerate(selected_sorted)
    }

    line_nodes = sampled_node_ids(len(points), 1200, seed=seed)

    for node in line_nodes.tolist():
        facility = int(assignments[node])

        if node == facility:
            continue

        p1 = points.loc[node    ]
        p2 = points.loc[facility]

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
        alpha    =0.62  ,
        edgecolor="none",
        zorder   =3     ,
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
    show_map_background: bool = False,
) -> None:
    instance   = result["instance"  ]
    structures = result["structures"]
    points     = instance.points
    densest    = set(structures["densest_nodes"])

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    _setup_axis(
        ax,
        points,
        f"{instance.name}",
        show_map_background=show_map_background,
    )
    _plot_network_edges(ax, instance)

    point_size = 5 if len(points) > 3000 else 10 if len(points) > 800 else 20

    ax.scatter(
        points["long"],
        points["lat" ],
        s        =point_size,
        c        ="#bdbdbd",
        alpha    =0.34,
        edgecolor="none",
        zorder   =2,
    )

    if densest:
        highlighted = points[points["node"].isin(densest)]

        ax.scatter(
            highlighted["long"],
            highlighted["lat" ],
            s        =90,
            c        ="#e66101",
            edgecolor="#4d2600",
            linewidth=0.7,
            zorder   =4,
        )

    selected_points = points[points["node"].isin(result["selected"])]

    ax.scatter(
        selected_points["long"],
        selected_points["lat" ],
        s        =90,
        c        ="#e61100",
        edgecolor="#4d2600",
        linewidth=0.7,
        zorder   =5,
    )


def plot_k_number_heatmap(
    result: dict[str, object],
    ax=None,
    *,
    show_map_background: bool = False,
) -> None:
    instance     = result["instance"  ]
    core_numbers = result["structures"]["core_numbers"]
    points       = instance.points

    values = np.asarray(
        [
            core_numbers.get(int(node), 0)
            for node in points["node"]
        ],
        dtype=float,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    _setup_axis(
        ax,
        points,
        f"{instance.name}",
        show_map_background=show_map_background,
    )
    _plot_network_edges(ax, instance, alpha=0.22)

    x = points["long"].to_numpy(dtype=float)
    y = points["lat" ].to_numpy(dtype=float)

    if len(points) <= 1200:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        grid_x = np.linspace(xlim[0], xlim[1], 140)
        grid_y = np.linspace(ylim[0], ylim[1], 140)
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

    point_sizes = np.where(
        values == 0,
        6  if len(points) > 3000 else 12,
        28 if len(points) > 3000 else 34,
    )

    ax.scatter(
        x,
        y,
        c        =values,
        s        =point_sizes,
        linewidth=0.20,
        zorder   =3,
        cmap     ="inferno",
        edgecolor="white" if len(points) <= 3000 else "none",
        alpha    =0.72,
    )

    selected_points = points[points["node"].isin(result["selected"])]

    ax.scatter(
        selected_points["long"],
        selected_points["lat" ],
        s        =40,
        c        ="#ff0000",
        linewidth=0.5,
        zorder   =4,
        edgecolor="#111111",
    )


def plot_max_k_cut_clusters(
    result: dict[str, object],
    ax=None,
    *,
    show_map_background: bool = False,
    seed               : int = 42,
) -> None:
    instance = result["instance"  ]
    labels   = result["structures"]["labels_maxcut"]
    points   = instance.points

    active_nodes  = np.flatnonzero(labels >= 0)
    ignored_nodes = np.asarray(
        result["structures"].get("maxcut_ignored_nodes", np.flatnonzero(labels < 0)),
        dtype=int,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    _setup_axis(
        ax,
        points,
        f"{instance.name}",
        show_map_background=show_map_background,
    )
    _plot_network_edges(ax, instance, alpha=0.25)

    ignored_sample = ignored_nodes
    if ignored_sample.size > 6000:
        rng = np.random.default_rng(seed)
        ignored_sample = np.sort(rng.choice(ignored_sample, size=6000, replace=False))

    if ignored_sample.size:
        ignored_points = points[points["node"].isin(ignored_sample)]

        ax.scatter(
            ignored_points["long"],
            ignored_points["lat" ],
            s        =5 if len(points) > 3000 else 12,
            c        ="#c7c7c7",
            alpha    =0.35,
            edgecolor="none",
            zorder   =2,
        )

    if active_nodes.size:
        active_labels = labels[active_nodes]
        palette       = plt.colormaps["tab20"].resampled(max(1, int(active_labels.max()) + 1))
        colors        = [palette(int(label)) for label in active_labels]
        active_points = points.set_index("node").loc[active_nodes]

        ax.scatter(
            active_points["long"],
            active_points["lat" ],
            s        =42 if len(points) > 3000 else 58,
            c        =colors,
            edgecolor="white",
            linewidth=0.35,
            zorder   =3,
        )

    selected_points = points[points["node"].isin(result["selected"])]

    ax.scatter(
        selected_points["long"],
        selected_points["lat" ],
        s        =50,
        marker   ="+",
        c        ="#000000",
        linewidth=1,
        zorder   =4,
    )

    if active_nodes.size and int(active_labels.max()) < 20:
        handles = [
            Line2D(
                [0],
                [0],
                marker          ="o",
                linestyle       ="",
                markerfacecolor =palette(group),
                markeredgecolor ="white",
                label           =f"Cut {group}",
            )
            for group in sorted(np.unique(active_labels).tolist())
        ]

        ax.legend(handles=handles, loc="best", fontsize=8, frameon=True)


def plot_instance_geometric_summary(
    result: dict[str, object],
    *,
    figsize             : tuple[float, float] = (10, 6),
    show_map_background : bool = False,
    seed                : int = 42,
    display_output      : bool = True,
    dpi                 : int = 160,
):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=figsize,
        constrained_layout=True,
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

    axes[0, 0].set_title("Solution")
    axes[0, 1].set_title("Max-k-cut")
    axes[1, 0].set_title("Densest subgraph")
    axes[1, 1].set_title("K number")

    if not display_output:
        return fig

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    display(Image(data=buffer.getvalue()))
    plt.close(fig)

    return None

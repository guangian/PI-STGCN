"""Delaunay–Voronoi physical control-volume graph used by PI-STGCN."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay, Voronoi

R_EARTH = 6371000.0


def lonlat_to_xy(coords: np.ndarray) -> np.ndarray:
    """Project longitude/latitude to a local metric coordinate system."""
    lon, lat = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    lat0 = float(lat.mean())
    return np.stack(
        [R_EARTH * (lon - lon.mean()) * np.cos(lat0),
         R_EARTH * (lat - lat.mean())],
        axis=1,
    )


@dataclass
class PhysicalGraph:
    """Fixed physical graph and control-volume geometry.

    Each undirected edge is stored once with ``phys_i < phys_j``. Cross-aquifer
    edges are removed before the Darcy and CVFD operators are evaluated.
    """

    phys_i: np.ndarray
    phys_j: np.ndarray
    phys_dist: np.ndarray
    phys_width: np.ndarray
    cell_area: np.ndarray
    interior_mask: np.ndarray
    stats: dict


def _voronoi_geometry(
    xy: np.ndarray, clip_q: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], float]]:
    vor = Voronoi(xy)
    n = len(xy)
    area = np.full(n, np.nan)
    interior = np.zeros(n, dtype=bool)
    for point, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not region or -1 in region:
            continue
        polygon = vor.vertices[region]
        x, y = polygon[:, 0], polygon[:, 1]
        area[point] = 0.5 * abs(
            np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
        )
        interior[point] = True

    finite_area = area[interior]
    if finite_area.size == 0:
        raise ValueError("Voronoi graph has no finite interior control volume")
    lo, hi = np.quantile(finite_area, clip_q)
    area = np.where(
        np.isnan(area), float(np.median(finite_area)), np.clip(area, lo, hi)
    )

    ridge_width: dict[tuple[int, int], float] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        key = (min(p1, p2), max(p1, p2))
        ridge_width[key] = (
            np.nan
            if v1 == -1 or v2 == -1
            else float(np.linalg.norm(vor.vertices[v1] - vor.vertices[v2]))
        )
    return area, interior, ridge_width


def build_physical_graph(
    coords: np.ndarray,
    aquifer_onehot: np.ndarray,
    min_dist: float,
    area_clip_q: tuple[float, float] = (0.05, 0.95),
    max_wd_ratio: float = 20.0,
) -> PhysicalGraph:
    """Build the fixed Delaunay–Voronoi graph described in the manuscript."""
    xy = lonlat_to_xy(coords)
    aquifer = aquifer_onehot.argmax(axis=1)
    triangulation = Delaunay(xy)

    edges: set[tuple[int, int]] = set()
    cross_aquifer = 0
    for simplex in triangulation.simplices:
        for edge_idx in range(3):
            i = int(simplex[edge_idx])
            j = int(simplex[(edge_idx + 1) % 3])
            if aquifer[i] != aquifer[j]:
                cross_aquifer += 1
                continue
            edges.add((min(i, j), max(i, j)))

    area, interior, ridge_width = _voronoi_geometry(xy, area_clip_q)
    finite_widths = [w for w in ridge_width.values() if np.isfinite(w)]
    default_width = float(np.median(finite_widths)) if finite_widths else 100.0

    phys_i, phys_j, phys_dist, phys_width = [], [], [], []
    n_clipped = 0
    for i, j in sorted(edges):
        distance = max(float(np.linalg.norm(xy[i] - xy[j])), min_dist)
        width = ridge_width.get((i, j), np.nan)
        width = default_width if not np.isfinite(width) else width
        if width / distance > max_wd_ratio:
            width = max_wd_ratio * distance
            n_clipped += 1
        phys_i.append(i)
        phys_j.append(j)
        phys_dist.append(distance)
        phys_width.append(width)

    stats = {
        "n_phys_edges": len(phys_i),
        "n_cross_aquifer_edges_removed": cross_aquifer // 2,
        "n_wd_clipped": n_clipped,
        "n_interior": int(interior.sum()),
    }
    return PhysicalGraph(
        phys_i=np.asarray(phys_i, dtype=np.int64),
        phys_j=np.asarray(phys_j, dtype=np.int64),
        phys_dist=np.asarray(phys_dist, dtype=np.float32),
        phys_width=np.asarray(phys_width, dtype=np.float32),
        cell_area=area.astype(np.float32),
        interior_mask=interior,
        stats=stats,
    )

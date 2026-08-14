"""双图构建：
- 信息传播图：分含水层类型独立 kNN（保证层内度数），供 GNN 消息传递；
- 物理控制体图：Delaunay 邻接【同含水层类型才保留边】+ Voronoi 面积/界面宽，
  w/d 比值截断防共点伪高传导度；
- 仅使用坐标与静态属性，绝不使用任何时序数据（无图泄漏）。
坐标投影到米制局部平面（等距圆柱近似），A_i/w_ij/d_ij 均为米制。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay, Voronoi, cKDTree

R_EARTH = 6371000.0


def lonlat_to_xy(coords: np.ndarray) -> np.ndarray:
    lon, lat = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    lat0 = float(lat.mean())
    return np.stack([R_EARTH * (lon - lon.mean()) * np.cos(lat0),
                     R_EARTH * (lat - lat.mean())], axis=1)


@dataclass
class DualGraphs:
    info_src: np.ndarray
    info_dst: np.ndarray
    info_dist: np.ndarray        # [E_info] m
    info_edge_attr: np.ndarray   # [E_info, 4] 可学习边函数输入：Δx/Δy/ΔDEM（标准化）+ log 距离(km)
    phys_i: np.ndarray           # 物理图（无向，i<j 存一次，全部同含水层）
    phys_j: np.ndarray
    phys_dist: np.ndarray        # d_ij (m)
    phys_width: np.ndarray       # Voronoi 界面宽 w_ij (m，已按 max_wd_ratio 截断)
    cell_area: np.ndarray        # [N] A_i (m²)
    interior_mask: np.ndarray    # [N] 有限 Voronoi 单元
    stats: dict


def _voronoi_geometry(xy: np.ndarray, clip_q) -> tuple[np.ndarray, np.ndarray, dict]:
    vor = Voronoi(xy)
    n = len(xy)
    area = np.full(n, np.nan)
    interior = np.zeros(n, dtype=bool)
    for p, ridx in enumerate(vor.point_region):
        region = vor.regions[ridx]
        if len(region) == 0 or -1 in region:
            continue
        poly = vor.vertices[region]
        x, y = poly[:, 0], poly[:, 1]
        area[p] = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        interior[p] = True
    fin = area[interior]
    lo, hi = np.quantile(fin, clip_q[0]), np.quantile(fin, clip_q[1])
    area = np.where(np.isnan(area), float(np.median(fin)), np.clip(area, lo, hi))
    ridge_w: dict[tuple[int, int], float] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        key = (min(p1, p2), max(p1, p2))
        ridge_w[key] = (np.nan if (v1 == -1 or v2 == -1)
                        else float(np.linalg.norm(vor.vertices[v1] - vor.vertices[v2])))
    return area, interior, ridge_w


def build_dual_graphs(coords: np.ndarray, aquifer_onehot: np.ndarray,
                      knn_k: int, min_dist: float,
                      area_clip_q=(0.05, 0.95), max_wd_ratio: float = 20.0,
                      dem: np.ndarray | None = None) -> DualGraphs:
    xy = lonlat_to_xy(coords)
    n = len(xy)
    aq = aquifer_onehot.argmax(axis=1)
    if dem is None:
        dem = np.zeros(n, dtype=np.float64)
    dem_n = (dem - dem.mean()) / (dem.std() + 1e-8)
    xy_scale = xy.std() + 1e-8

    # ---------- 信息图：分含水层独立 kNN ----------
    src, dst, dd = [], [], []
    for a in np.unique(aq):
        idx = np.where(aq == a)[0]
        if len(idx) < 2:
            continue
        tree = cKDTree(xy[idx])
        k = min(knn_k + 1, len(idx))
        dist, nbr = tree.query(xy[idx], k=k)
        for r, gi in enumerate(idx):
            for d, c in zip(dist[r, 1:], nbr[r, 1:]):
                src.append(int(gi)); dst.append(int(idx[c])); dd.append(max(float(d), min_dist))
    edge_set = set(zip(src, dst))
    for i, j, d in list(zip(src, dst, dd)):     # 对称闭包
        if (j, i) not in edge_set:
            src.append(j); dst.append(i); dd.append(d)
            edge_set.add((j, i))
    # 可学习边函数的边属性（有向，与 src/dst 对齐）：方向性几何 + log 距离
    src_a, dst_a = np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)
    edge_attr = np.stack([
        (xy[src_a, 0] - xy[dst_a, 0]) / xy_scale,
        (xy[src_a, 1] - xy[dst_a, 1]) / xy_scale,
        (dem_n[src_a] - dem_n[dst_a]),
        np.log(np.asarray(dd) / 1000.0 + 1e-3)], axis=1).astype(np.float32)

    # ---------- 物理图：Delaunay + 同层过滤 + w/d 截断 ----------
    tri = Delaunay(xy)
    pe = set()
    for simplex in tri.simplices:
        for a2 in range(3):
            i, j = int(simplex[a2]), int(simplex[(a2 + 1) % 3])
            if aq[i] == aq[j]:                              # 跨含水层排边
                pe.add((min(i, j), max(i, j)))
    area, interior, ridge_w = _voronoi_geometry(xy, area_clip_q)
    wfin = [w for w in ridge_w.values() if np.isfinite(w)]
    wmed = float(np.median(wfin)) if wfin else 100.0
    pi, pj, pdist, pwidth = [], [], [], []
    n_wd_clipped = 0
    for (i, j) in sorted(pe):
        d = max(float(np.linalg.norm(xy[i] - xy[j])), min_dist)
        w = ridge_w.get((i, j), np.nan)
        w = wmed if not np.isfinite(w) else w
        if w / d > max_wd_ratio:                            # 防共点伪边高传导度
            w = max_wd_ratio * d
            n_wd_clipped += 1
        pi.append(i); pj.append(j); pdist.append(d); pwidth.append(w)

    stats = {"n_info_edges": len(src), "n_phys_edges": len(pi),
             "n_cross_aquifer_edges_removed": int(sum(
                 1 for s in tri.simplices for a2 in range(3)
                 if aq[int(s[a2])] != aq[int(s[(a2 + 1) % 3])]) // 2),
             "n_wd_clipped": n_wd_clipped,
             "n_interior": int(interior.sum())}
    return DualGraphs(
        info_src=src_a, info_dst=dst_a,
        info_dist=np.asarray(dd, dtype=np.float32), info_edge_attr=edge_attr,
        phys_i=np.asarray(pi, dtype=np.int64), phys_j=np.asarray(pj, dtype=np.int64),
        phys_dist=np.asarray(pdist, dtype=np.float32), phys_width=np.asarray(pwidth, dtype=np.float32),
        cell_area=area.astype(np.float32), interior_mask=interior, stats=stats)

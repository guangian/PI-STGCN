"""统一评估协议（站点中位值主口径）：
- 一切指标在米制、仅真实观测点（Mask=1）上计算；
- **主对比口径 = 站点中位值**：逐井 RMSE 中位数、逐井 NSE 中位数、
  逐井 persistence skill 中位数——不被少数灾难井或井间高程差污染；
- pooled 指标保留作辅助；禁用逐井均值聚合；
- 本模块是唯一评估入口。
"""
from __future__ import annotations

import numpy as np

MIN_OBS_PER_WELL = 5


def _pooled(y_true, y_pred, m):
    err = (y_pred - y_true)[m]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mu = float(y_true[m].mean())
    var = float(np.mean((y_true[m] - mu) ** 2))
    return rmse, mae, float(1.0 - np.mean(err ** 2) / (var + 1e-12))


def _per_well_arrays(y_true, y_pred, y_base, m):
    """逐井 RMSE / NSE / persistence-RMSE（观测点不足的井置 NaN）. y [W, N]."""
    N = y_true.shape[1]
    rmse = np.full(N, np.nan)
    nse = np.full(N, np.nan)
    rmse_b = np.full(N, np.nan)
    for i in range(N):
        mi = m[:, i]
        if mi.sum() < MIN_OBS_PER_WELL:
            continue
        t, p, b = y_true[mi, i], y_pred[mi, i], y_base[mi, i]
        rmse[i] = np.sqrt(np.mean((p - t) ** 2))
        rmse_b[i] = np.sqrt(np.mean((b - t) ** 2))
        var = np.mean((t - t.mean()) ** 2)
        if var > 1e-10:
            nse[i] = 1.0 - np.mean((p - t) ** 2) / var
    return rmse, nse, rmse_b


def _median_block(y_true, y_pred, y_base, m):
    rmse_w, nse_w, rmse_bw = _per_well_arrays(y_true, y_pred, y_base, m)
    ok = np.isfinite(rmse_w) & np.isfinite(rmse_bw)
    skill_w = np.where(ok, 1.0 - rmse_w / (rmse_bw + 1e-12), np.nan)
    return {
        "rmse_well_median": float(np.nanmedian(rmse_w)),
        "nse_well_median": float(np.nanmedian(nse_w)),
        "skill_well_median": float(np.nanmedian(skill_w)),
        "persistence_rmse_well_median": float(np.nanmedian(rmse_bw)),
        "n_wells_scored": int(np.isfinite(rmse_w).sum())}


def evaluate_meters(h_true_m: np.ndarray, h_pred_m: np.ndarray,
                    h_persist_m: np.ndarray, obs_mask: np.ndarray) -> dict:
    """输入 [W, N, P] 米制；obs_mask [W, N, P]（1=真实观测）。逐步长 + 汇总.

    主口径 = 站点中位值（rmse_well_median / nse_well_median / skill_well_median）。
    """
    P = h_true_m.shape[2]
    out: dict = {"per_horizon": {},
                 "protocol": "meters, real-observation mask, station-median primary"}
    for k in range(P):
        t, p, b = h_true_m[:, :, k], h_pred_m[:, :, k], h_persist_m[:, :, k]
        m = obs_mask[:, :, k] > 0
        med = _median_block(t, p, b, m)
        rmse, mae, r2p = _pooled(t, p, m)
        rmse_b, _, _ = _pooled(t, b, m)
        med.update({"rmse_m_pooled": rmse, "mae_m_pooled": mae, "r2_pooled_aux": r2p,
                    "persistence_rmse_m_pooled": rmse_b,
                    "skill_pooled": float(1.0 - rmse / (rmse_b + 1e-12)),
                    "n_obs_points": int(m.sum())})
        out["per_horizon"][f"T+{k+1}"] = med

    # overall：井×步长联合（每井把 P 个步长的观测点并在一起算逐井指标）
    W, N, _ = h_true_m.shape
    t2 = h_true_m.transpose(0, 2, 1).reshape(W * P, N)
    p2 = h_pred_m.transpose(0, 2, 1).reshape(W * P, N)
    b2 = h_persist_m.transpose(0, 2, 1).reshape(W * P, N)
    m2 = (obs_mask > 0).transpose(0, 2, 1).reshape(W * P, N)
    med = _median_block(t2, p2, b2, m2)
    rmse, mae, r2p = _pooled(h_true_m, h_pred_m, obs_mask > 0)
    rmse_b, _, _ = _pooled(h_true_m, h_persist_m, obs_mask > 0)
    med.update({"rmse_m_pooled": rmse, "mae_m_pooled": mae, "r2_pooled_aux": r2p,
                "persistence_rmse_m_pooled": rmse_b,
                "skill_pooled": float(1.0 - rmse / (rmse_b + 1e-12))})
    out["overall"] = med
    return out


# ---------------------------------------------------------------- 站点级明细
# 逐井原始指标值（箱线图/雷达图等分布类图形的直接数据源）。
# evaluate_meters 的 rmse_well_median 等主口径数字 = 本明细按步长取井中位数，可逐位对账。
PER_WELL_FIELDS = ("well_id", "horizon", "n_obs", "rmse_m", "mae_m", "nse", "r2",
                   "skill_vs_persistence")


def _per_well_full(y_true, y_pred, y_base, m):
    """逐井 RMSE/MAE/NSE/R²/skill（输入 2D [S, N]）；观测不足或方差退化置 NaN.

    R² = 逐井 Pearson 相关系数的平方（clip 0..1）；NSE 同 _per_well_arrays 定义；
    skill = 1 − RMSE_model/RMSE_persistence（逐井）。
    """
    N = y_true.shape[1]
    out = {k: np.full(N, np.nan) for k in ("rmse", "mae", "nse", "r2", "skill")}
    n_obs = np.zeros(N, dtype=int)
    for i in range(N):
        mi = m[:, i]
        n_obs[i] = int(mi.sum())
        if n_obs[i] < MIN_OBS_PER_WELL:
            continue
        t, p, b = y_true[mi, i], y_pred[mi, i], y_base[mi, i]
        mse = float(np.mean((p - t) ** 2))
        out["rmse"][i] = np.sqrt(mse)
        out["mae"][i] = float(np.mean(np.abs(p - t)))
        var = float(np.mean((t - t.mean()) ** 2))
        if var > 1e-10:
            out["nse"][i] = 1.0 - mse / var
            if float(np.std(p)) > 1e-10:
                r = float(np.corrcoef(p, t)[0, 1])
                out["r2"][i] = min(max(r * r, 0.0), 1.0)
        rmse_b = float(np.sqrt(np.mean((b - t) ** 2)))
        out["skill"][i] = 1.0 - out["rmse"][i] / (rmse_b + 1e-12)
    return out, n_obs


def per_well_records(h_true_m: np.ndarray, h_pred_m: np.ndarray, h_persist_m: np.ndarray,
                     obs_mask: np.ndarray, well_ids: list) -> list[dict]:
    """[W, N, P] 米制 → 长格式站点级明细：每行 = (井, 步长)，另含 overall 行
    （该井 P 个步长的观测点合并后计算，与 evaluate_meters 的 overall 同口径）。
    """
    W, N, P = h_true_m.shape
    assert N == len(well_ids), f"数组井数 {N} ≠ 井号数 {len(well_ids)}"
    rows: list[dict] = []

    def _fmt(x):
        return "" if not np.isfinite(x) else round(float(x), 6)

    def block(label, t, p, b, m):
        met, n_obs = _per_well_full(t, p, b, m)
        for i, wid in enumerate(well_ids):
            if n_obs[i] == 0:
                continue
            rows.append({"well_id": wid, "horizon": label, "n_obs": int(n_obs[i]),
                         "rmse_m": _fmt(met["rmse"][i]), "mae_m": _fmt(met["mae"][i]),
                         "nse": _fmt(met["nse"][i]), "r2": _fmt(met["r2"][i]),
                         "skill_vs_persistence": _fmt(met["skill"][i])})

    for k in range(P):
        block(f"T+{k+1}", h_true_m[:, :, k], h_pred_m[:, :, k], h_persist_m[:, :, k],
              obs_mask[:, :, k] > 0)
    tr = (lambda a: a.transpose(0, 2, 1).reshape(W * P, N))
    block("overall", tr(h_true_m), tr(h_pred_m), tr(h_persist_m), tr(obs_mask) > 0)
    return rows


def write_per_well_csv(rows: list[dict], path, extra_fields: tuple = ()) -> None:
    """站点级明细落盘（UTF-8 CSV）；extra_fields 置于 well_id 之前（如 model/target 列）."""
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[*extra_fields, *PER_WELL_FIELDS])
        w.writeheader()
        w.writerows(rows)

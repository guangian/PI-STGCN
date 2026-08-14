"""数据管线：

- 从单井文件 processed_wells/<id>/water_level.csv 读取【原始观测 + Mask】，
  杜绝上游全期双向线性插值的未来信息：
  * 输入特征用因果前向填充（ffill）；序列头部（首个观测之前）缺测用该井【训练期观测均值】
    填充（与逐井归一化统计同一信息类别——仅训练期聚合量、无逐点未来值；禁止 bfill）；
  * 训练损失与评估指标仅在 Mask=1（真实观测）处计算；
- 客观 QC 仅使用【训练期】信息；
- 强迫单位按单井文件列名核定：precip/ET 为 mm/day（×1e-3→m/day），WaterUse_m3d 已是 m³/day；
- 强迫按日期严格 reindex 并断言对齐；
- 逐井 z-score 统计仅来自训练期真实观测并落盘版本化。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def _read_wells_summary(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={0: str})
    df.columns = ["well_id", "lon", "lat", "elev", "aquifer_type", "province_code"][: len(df.columns)]
    return df.set_index("well_id")


@dataclass
class DataBundle:
    dates: pd.DatetimeIndex
    well_ids: list[str]
    H_fill: np.ndarray       # [T, N] 因果填充水位 (m)：特征/锚定用
    H_obs: np.ndarray        # [T, N] 原始观测（缺测为 NaN）：监督/评估用
    mask: np.ndarray         # [T, N] 1=真实观测
    coords: np.ndarray       # [N, 2] lon/lat（重合井已微抖动）
    dem: np.ndarray          # [N]
    aquifer_onehot: np.ndarray  # [N, 2] 潜水 / 承压
    precip: np.ndarray       # [T] m/day
    et: np.ndarray           # [T] m/day
    wu: np.ndarray           # [T, N] m³/day
    train_end_idx: int
    val_end_idx: int
    qc_report: dict = field(default_factory=dict)
    z_bot_ref: np.ndarray | None = None   # [N] 浅层系统底板标高 (m)，钻孔地层几何插值；
    #   缺省 None（跨流域目标域无此资料时模型自动回退 dem−b₀+Δz 先验几何）
    L0_geo: np.ndarray | None = None      # [N] 井到排泄网络（河道/海岸）的地理距离 (m)，
    #   DEM 汇流分析派生（make_drainage_distance.py）；None 时 τ_b 回退全局可学习 L₀


def _causal_fill(col: np.ndarray, head_value: float) -> np.ndarray:
    """因果填充：ffill + 序列头部（首个观测之前）用 head_value（该井训练期观测均值）.

    头部一律不用未来逐点观测（无 bfill）。head_value 是训练期聚合统计，
    与逐井 z-score 归一化统计同一信息类别。
    """
    s = pd.Series(col).ffill()
    return s.fillna(float(head_value)).to_numpy()


def load_bundle(cfg: dict, coder_dir: Path) -> DataBundle:
    root = (coder_dir / cfg["paths"]["data_root"]).resolve()
    wells_dir = (coder_dir / cfg["paths"]["wells_dir"]).resolve()
    dcfg, fcfg, qc = cfg["data"], cfg["forcing"], cfg["data"]["qc"]

    summary = _read_wells_summary((coder_dir / cfg["paths"]["wells_summary"]).resolve())
    dem_df = pd.read_csv(root / "geometry" / "dem_at_wells.csv", dtype={"well_id": str}).set_index("well_id")

    # ---------------- 逐井读取原始观测 + Mask + 抽水 ----------------
    dates: pd.DatetimeIndex | None = None
    obs_cols, mask_cols, wu_cols, ids = [], [], [], []
    for wdir in sorted(wells_dir.iterdir()):
        if not wdir.is_dir() or wdir.name not in summary.index:
            continue
        f = wdir / "water_level.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["Date"]).set_index("Date")
        if dates is None:
            dates = df.index
        elif not df.index.equals(dates):
            df = df.reindex(dates)
        ids.append(wdir.name)
        obs_cols.append(df["WaterLevel"].to_numpy(dtype=np.float32))
        mask_cols.append(df["Mask"].fillna(0).to_numpy(dtype=np.float32))
        wu_cols.append(df["WaterUse_m3d"].ffill().fillna(0.0).to_numpy(dtype=np.float32))
    assert dates is not None and len(ids) > 0, "未读到任何井文件"
    H_obs_all = np.stack(obs_cols, axis=1)          # [T, n]
    M_all = np.stack(mask_cols, axis=1)
    WU_all = np.stack(wu_cols, axis=1)
    M_all = M_all * np.isfinite(H_obs_all)          # Mask=1 但值缺失的防御
    # Mask 是观测有效性的唯一事实源。即使替换数据意外在 Mask=0 行保留了
    # WaterLevel 数值，也不得让它进入因果前向填充或训练期统计。
    H_obs_all[M_all <= 0] = np.nan
    assert not np.isfinite(H_obs_all[M_all <= 0]).any(), "Mask=0 水位必须不可见"

    train_end_idx = int(np.searchsorted(dates.values, np.datetime64(dcfg["train_end"]), side="right")) - 1
    val_end_idx = int(np.searchsorted(dates.values, np.datetime64(dcfg["val_end"]), side="right")) - 1
    tr = slice(0, train_end_idx + 1)

    # ---------------- 客观 QC：仅训练期信息 ----------------
    report: dict = {}
    keep_idx, dropped = [], {"missing_rate": [], "extreme_level": [], "constant": [], "jump": []}
    for j, w in enumerate(ids):
        m_tr = M_all[tr, j].astype(bool)
        h_tr = H_obs_all[tr, j][m_tr]
        if 1.0 - float(m_tr.mean()) > qc["max_missing_rate"] or len(h_tr) < 10:
            dropped["missing_rate"].append(w); continue
        if np.max(np.abs(h_tr)) > qc["max_abs_level"]:
            dropped["extreme_level"].append(w); continue
        if np.std(h_tr) < qc["min_std"]:
            dropped["constant"].append(w); continue
        if len(h_tr) > 1 and np.max(np.abs(np.diff(h_tr))) > qc["max_step_jump"]:
            dropped["jump"].append(w); continue
        keep_idx.append(j)
    keep = [ids[j] for j in keep_idx]
    report["dropped"] = {k: sorted(v) for k, v in dropped.items()}
    report["n_candidates"], report["n_kept"] = len(ids), len(keep)
    report["qc_scope"] = f"train period only (idx 0..{train_end_idx})"

    H_obs = H_obs_all[:, keep_idx]
    mask = M_all[:, keep_idx]
    wu = WU_all[:, keep_idx] * float(fcfg["wu_unit_to_m3_per_day"])
    # 头部填充值 = 该井训练期真实观测均值（QC 保证训练期观测 ≥10 个）
    h_tr_obs = np.where(mask[tr] > 0, H_obs[tr], np.nan)
    train_mean = np.nanmean(h_tr_obs, axis=0)                        # [N]
    H_fill = np.stack([_causal_fill(H_obs[:, j], train_mean[j])
                       for j in range(H_obs.shape[1])], axis=1).astype(np.float32)
    head_steps = [int(np.argmax(np.isfinite(H_obs[:, j]))) for j in range(H_obs.shape[1])]
    report["headfill_steps_total"] = int(sum(head_steps))
    report["headfill_wells"] = int(sum(1 for s_ in head_steps if s_ > 0))
    report["headfill_policy"] = "per-well train-period observed mean (no bfill)"

    coords = summary.loc[keep, ["lon", "lat"]].to_numpy(dtype=np.float64)
    dem = dem_df.reindex(keep)["elevation"].to_numpy(dtype=np.float32)
    aq = summary.loc[keep, "aquifer_type"].astype(str)
    aquifer_onehot = np.stack(
        [(aq.str.contains("潜")).to_numpy(dtype=np.float32),
         (~aq.str.contains("潜")).to_numpy(dtype=np.float32)], axis=1)
    report["n_phreatic"], report["n_confined"] = int(aquifer_onehot[:, 0].sum()), int(aquifer_onehot[:, 1].sum())

    # 重合坐标微抖动（Delaunay/kNN 需要；~2 m，记录在案）
    jit = float(cfg["graph"]["dup_jitter_deg"])
    rng = np.random.default_rng(0)
    seen: dict[tuple, int] = {}
    n_jit = 0
    for i in range(len(coords)):
        key = (round(coords[i, 0], 8), round(coords[i, 1], 8))
        if key in seen:
            coords[i] += rng.normal(0.0, jit, size=2); n_jit += 1
        else:
            seen[key] = i
    report["n_jittered_duplicates"] = n_jit

    # ---------------- 区域强迫：按日期 reindex + 断言 ----------------
    pr_df = pd.read_csv(root / "forcing" / "precipitation.csv", index_col=0, parse_dates=True).reindex(dates)
    et_df = pd.read_csv(root / "forcing" / "evapotranspiration.csv", index_col=0, parse_dates=True).reindex(dates)
    assert not pr_df.iloc[:, 0].isna().any() and not et_df.iloc[:, 0].isna().any(), "强迫序列与水位日期未对齐"
    precip = pr_df.iloc[:, 0].to_numpy(dtype=np.float32) * float(fcfg["precip_unit_to_m_per_day"])
    et = et_df.iloc[:, 0].to_numpy(dtype=np.float32) * float(fcfg["et_unit_to_m_per_day"])

    # 浅层系统底板标高（真实含水层几何，make_bottom_elevation.py 产出；仅源域可用）
    z_bot_ref = None
    zf = root / "geometry" / "z_bot_at_wells.csv"
    if zf.exists():
        zdf = pd.read_csv(zf, dtype={"well_id": str}, encoding="utf-8-sig").set_index("well_id")
        z_bot_ref = zdf.reindex(keep)["z_bot_m"].to_numpy(dtype=np.float32)
        assert np.isfinite(z_bot_ref).all(), "z_bot_at_wells.csv 覆盖不全：存在 QC 保留井缺底板值"
    # 排泄网络距离（make_drainage_distance.py 产出；仅源域可用）
    L0_geo = None
    lf = root / "geometry" / "L0_at_wells.csv"
    if lf.exists():
        ldf = pd.read_csv(lf, dtype={"well_id": str}, encoding="utf-8-sig").set_index("well_id")
        L0_geo = ldf.reindex(keep)["L0_m"].to_numpy(dtype=np.float32)
        assert np.isfinite(L0_geo).all(), "L0_at_wells.csv 覆盖不全：存在 QC 保留井缺排泄距离"

    return DataBundle(dates=dates, well_ids=keep, H_fill=H_fill, H_obs=H_obs, mask=mask,
                      coords=coords, dem=dem, aquifer_onehot=aquifer_onehot,
                      precip=precip, et=et, wu=wu,
                      train_end_idx=train_end_idx, val_end_idx=val_end_idx, qc_report=report,
                      z_bot_ref=z_bot_ref, L0_geo=L0_geo)


# ---------------------------------------------------------------- normalizer

class PerWellNormalizer:
    """逐井 z-score：统计量仅来自训练期【真实观测】，落盘版本化."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean, self.std = mean.astype(np.float32), np.maximum(std, 0.05).astype(np.float32)

    @classmethod
    def fit(cls, H_obs: np.ndarray, mask: np.ndarray, train_end_idx: int) -> "PerWellNormalizer":
        h = np.where(mask[: train_end_idx + 1] > 0, H_obs[: train_end_idx + 1], np.nan)
        return cls(np.nanmean(h, axis=0), np.nanstd(h, axis=0))

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"mean": self.mean.tolist(), "std": self.std.tolist(),
                   "fit_scope": "train-period real observations only"},
                  open(path, "w", encoding="utf-8"))


# ---------------------------------------------------------------- windows

@dataclass
class WindowSet:
    x_idx: np.ndarray   # [W, L]
    y_idx: np.ndarray   # [W, P]


def build_windows(T: int, input_len: int, pred_len: int,
                  train_end: int, val_end: int) -> dict[str, WindowSet]:
    """目标块整体归属划分；跨界窗口丢弃（无目标泄漏）."""
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for t0 in range(input_len - 1, T - pred_len):
        tgt_first, tgt_last = t0 + 1, t0 + pred_len
        if tgt_last <= train_end:
            splits["train"].append(t0)
        elif tgt_first > train_end and tgt_last <= val_end:
            splits["val"].append(t0)
        elif tgt_first > val_end:
            splits["test"].append(t0)
    out = {}
    for k, t0s_list in splits.items():
        t0s = np.asarray(t0s_list, dtype=np.int64)
        out[k] = WindowSet(
            x_idx=np.stack([t0s - input_len + 1 + i for i in range(input_len)], axis=1),
            y_idx=np.stack([t0s + 1 + i for i in range(pred_len)], axis=1))
    return out


def make_features(bundle: DataBundle, normalizer: PerWellNormalizer) -> dict:
    """时序特征 [T, N, 7]（h_norm/观测标志/降水/蒸散/抽水/年周期 sin/cos）+ 静态协变量 [N, 7].

    强迫的全局 z-score 统计仅用训练期；静态协变量 = 域内标准化经纬度 + DEM + 含水层 one-hot
    + 埋深（dem − 训练期均值水位）+ log 训练期水位波动尺度。后两项是训练期观测统计量
    （与逐井 z-score 同信息类别、因果可得、跨域可算），给 ParamNet 提供表达
    「包气带厚度 / 动态强度」空间结构的词汇——纯 lon/lat 平滑场表达不了岩性梯度（Phase-2）。
    """
    T, N = bundle.H_fill.shape
    tr = slice(0, bundle.train_end_idx + 1)

    def gz(x):
        m, s = float(x[tr].mean()), float(x[tr].std()) + 1e-8
        return ((x - m) / s).astype(np.float32)

    h_norm = (bundle.H_fill - normalizer.mean) / normalizer.std
    soy = np.minimum((bundle.dates.dayofyear - 1) // 5, 72).to_numpy().astype(np.float32) / 73.0
    feats = np.zeros((T, N, 7), dtype=np.float32)
    feats[..., 0] = h_norm
    feats[..., 1] = bundle.mask                     # 观测/填充标志（因果可得）
    feats[..., 2] = gz(bundle.precip)[:, None]
    feats[..., 3] = gz(bundle.et)[:, None]
    feats[..., 4] = gz(bundle.wu)
    feats[..., 5] = np.sin(2 * np.pi * soy)[:, None]
    feats[..., 6] = np.cos(2 * np.pi * soy)[:, None]

    zs = lambda x: (x - x.mean()) / (x.std() + 1e-8)
    depth = np.clip(bundle.dem - normalizer.mean, 0.0, None)             # [N] 埋深 (m)
    log_amp = np.log(normalizer.std + 1e-3)                              # [N] 波动尺度
    static = np.concatenate([
        (bundle.coords - bundle.coords.mean(0)) / (bundle.coords.std(0) + 1e-8),
        ((bundle.dem - bundle.dem.mean()) / (bundle.dem.std() + 1e-8))[:, None],
        bundle.aquifer_onehot,
        zs(depth)[:, None], zs(log_amp)[:, None]], axis=1).astype(np.float32)   # [N, 7]
    return {"feats": feats, "static": static}

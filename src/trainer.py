"""训练器：物理 rollout 结构锚 + 掩码监督 + 扰动形式 CVFD 残差.

损失：L = w_h · maskMSE(ĥ, h)_z + w_dh · maskMSE(Δĥ, Δh)_z + λ_PDE · mean((r/r0)²)
- _z 为逐井标准化空间；监督仅真实观测点；Δ 序列以 h_last 为首项前插；
- 物理锚 FluxRollout 用未来强迫显式积分（scenario 假设，正文声明；no_future 消融给退化幅度），
  轨迹对 ParamNet 可微；
- PDE 残差每 batch 抽样 pde_windows_per_batch 窗，仅潜水×内部节点，强迫取区间右端点标记。
"""
from __future__ import annotations

import collections
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data_pipeline import DataBundle, PerWellNormalizer, WindowSet
from .evaluation import evaluate_meters
from .model import PISTGCNv2
from .physics import CVFDResidual, FluxRollout, ParamNet, vertical_source_m_per_day


def set_seed(seed: int, deterministic: bool = False):
    """全链路种子（P1-6）：NumPy/Torch/CUDA；benchmark=False 固定算法选择。

    deterministic=True 时额外启用 cudnn.deterministic（膨胀因果卷积的确定性反向核
    在部分 GPU 上慢 >100×，故默认关闭）；GPU 卷积的微小非确定性 << 种子间方差，
    完全逐位复现可用该开关或 CPU（run_meta 记录本次设置）。
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)


def load_gamma_facies_bounds(cfg: dict, well_ids) -> "np.ndarray | None":
    """γ 相带分带区间 [N,2]（gamma_facies_band 臂）：手册降水入渗系数按沉积相分带。

    数据源：param_reference/wells_param_zones.csv 的 facies 列（越流分区沉积相图，输入数据）
    + config physics.gamma_facies_bands 的公开手册区间常数。不含任何 K/Sy 检验目标值，
    零校准红线不破；缺相带的井回退全局 gamma_range。
    """
    import pandas as pd
    csv = Path(__file__).parents[1] / "data" / "param_reference" / "wells_param_zones.csv"
    if not csv.exists():
        return None
    bands = cfg["physics"].get("gamma_facies_bands") or {}
    g_lo, g_hi = (float(v) for v in cfg["physics"]["gamma_range"])
    fac_map = pd.read_csv(csv, dtype={"well_id": str}).set_index("well_id")["facies"]
    out = np.empty((len(well_ids), 2), dtype=np.float32)
    n_hit = 0
    for i, wid in enumerate(well_ids):
        fac = fac_map.get(str(wid))
        band = bands.get(str(fac)) if isinstance(fac, str) else None
        if band is None:
            out[i] = (g_lo, g_hi)
        else:
            out[i] = (float(band[0]), float(band[1]))
            n_hit += 1
    print(f"[gamma_band] 相带区间命中 {n_hit}/{len(well_ids)} 井（其余回退全局区间）", flush=True)
    return out


class Trainer:
    def __init__(self, cfg: dict, bundle: DataBundle, feats: dict, windows: dict[str, WindowSet],
                 normalizer: PerWellNormalizer, graphs, device: str, ablation: dict, out_dir: Path):
        self.cfg, self.bundle, self.windows = cfg, bundle, windows
        self.device = torch.device(device)
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        tr, p, f = cfg["train"], cfg["physics"], cfg["forcing"]
        self.w_h, self.w_dh = float(tr["w_h"]), float(tr["w_dh"])
        self.lambda_pde = float(ablation.get("lambda_pde", tr["lambda_pde"]))
        self.pde_loss_mode = str(ablation.get("pde_loss_mode", tr.get("pde_loss_mode", "trajectory")))
        if self.pde_loss_mode not in {"trajectory", "correction_defect"}:
            raise ValueError(f"未知 pde_loss_mode={self.pde_loss_mode}")
        self.pde_defect_beta = float(ablation.get("pde_defect_beta", tr.get("pde_defect_beta", 0.0)))
        self.pde_operator_params = str(ablation.get(
            "pde_operator_params", tr.get("pde_operator_params", "learned_detached")))
        if self.pde_operator_params not in {"learned_detached", "prior"}:
            raise ValueError(f"未知 pde_operator_params={self.pde_operator_params}")
        self.pde_weight_mode = str(ablation.get("pde_weight_mode", tr.get("pde_weight_mode", "fixed")))
        if self.pde_weight_mode not in {"fixed", "gradnorm"}:
            raise ValueError(f"未知 pde_weight_mode={self.pde_weight_mode}")
        self.pde_ramp_epochs = float(ablation.get("pde_ramp_epochs", tr.get("pde_ramp_epochs", 0.0)))
        self.pde_grad_target_ratio = float(tr.get("pde_grad_target_ratio", 0.05))
        self.pde_weight_min = float(tr.get("pde_weight_min", 0.0))
        self.pde_weight_max = float(tr.get("pde_weight_max", max(self.lambda_pde, 0.2)))
        self.pde_weight_ema = float(tr.get("pde_weight_ema", 0.9))
        self.pde_conflict_gate = bool(tr.get("pde_conflict_gate", True))
        self._adaptive_lambda = None
        self._epoch_progress = 1.0
        self.n_pde_win = int(tr.get("pde_windows_per_batch", 2))
        self.dt_days = float(cfg["data"]["step_days"])
        # Phase-3：观测序列水量平衡残差（正逆问题联立求解的反演锚）。
        # trajectory-PDE 的残差算在预测轨迹上，参数可通过"改预测"而非"改参数"满足它；
        # 本项把 CVFD 残差直接算在【训练期观测水位增量】上（仅两端都有真实观测的
        # 内部潜水节点计分），Sy·Δh_obs 对真实开采/滞后补给回归——Sy/γ 由此获得
        # 绝对尺度约束（wu 为逐井实测系列）。0 = 关闭（历史协议不变）。
        self.pde_obs_weight = float(ablation.get("pde_obs_weight", p.get("pde_obs_weight", 0.0)))
        # 观测平衡的时间口径：False=逐步增量（噪声敏感）；True=窗口累积（季节尺度闭合）
        self.pde_obs_cumulative = bool(ablation.get("pde_obs_cumulative",
                                                    p.get("pde_obs_cumulative", False)))

        # NN 输入通道（0 h_norm,1 mask,2 precip,3 et,4 wu,5 sin,6 cos）；
        # nn_input_forcing=false（no_nnforcing 消融）时强迫只走物理路径
        self.nn_forcing = bool(ablation.get("nn_input_forcing", tr.get("nn_input_forcing", True)))
        self.nn_feat_idx = [0, 1, 2, 3, 4, 5, 6] if self.nn_forcing else [0, 1, 5, 6]
        # lean 头多尺度滞后（no_multiscale 消融在 lean 模式下退化为全 lag-1，维度不变）
        self.lean_lags = tuple(ablation.get("lean_lags", self.LEAN_LAGS))
        assert len(self.lean_lags) == len(self.LEAN_LAGS), "lean_lags 维度必须恒定（参数量对照公平）"
        # masked-node 训练：随机遮蔽井输入历史，逼图消息传递承载空间信息
        self.node_mask_frac = float(ablation.get("node_mask_frac", tr.get("node_mask_frac", 0.15)))
        self.mask_eval_weight = float(tr.get("mask_eval_weight", 1.0))

        mdl = cfg["model"]
        self.model = PISTGCNv2(
            cfg, n_feats=len(self.nn_feat_idx), n_static=feats["static"].shape[-1],
            graphs=graphs, device=self.device,
            use_phys_flux=bool(ablation.get("use_phys_flux", True)),
            signed_flux=bool(ablation.get("signed_flux", True)),
            use_paramnet=bool(ablation.get("use_paramnet", True)),
            use_anchor=bool(ablation.get("use_anchor", True)),
            cumsum_head=bool(ablation.get("cumsum_head", False)),
            use_cond_gate=bool(ablation.get("use_gate", True)),
            learnable_edges=bool(ablation.get("learnable_edges", True)),
            use_leddam=bool(ablation.get("use_leddam", mdl.get("use_leddam", False))),
            use_star=bool(ablation.get("use_star", mdl.get("use_star", False))),
            detach_phys_features=bool(ablation.get(
                "detach_phys_features", self.pde_loss_mode == "correction_defect")),
            use_darcy_attn=bool(ablation.get("use_darcy_attn", mdl.get("use_darcy_attn", True))),
            darcy_learnable_exponents=bool(ablation.get("darcy_learnable_exponents", True)),
            darcy_dynamic=bool(ablation.get("darcy_dynamic", True)),
            darcy_signed=bool(ablation.get("darcy_signed", True)),
            darcy_uniform=bool(ablation.get("darcy_uniform", False)),
            darcy_magnitude=bool(ablation.get("darcy_magnitude", True)),
            head_mode=str(ablation.get("head_mode", mdl.get("head_mode", "lean"))),
            n_lean=self.N_LEAN,
            derive_tau=bool(ablation.get("derive_tau", mdl.get("derive_tau", True))),
            darcy_content=bool(ablation.get("darcy_content", mdl.get("darcy_content", True))),
            n_fut=self.N_FUT,
            use_ref_bottom=bool(ablation.get("use_ref_bottom", mdl.get("use_ref_bottom", False))),
            derive_tau_r=bool(ablation.get("derive_tau_r", mdl.get("derive_tau_r", False))),
            derive_L0_geo=bool(ablation.get("derive_L0_geo", mdl.get("derive_L0_geo", False))))
        # Phase-2b：逐井排泄距离注入 + log_s 解析初始化（中位数对齐全局解析 L₀，
        # 保证起点 τ 分布与全局 L₀ 版本一致；缺资料时 L0_geo_t=None 自动回退）
        if self.model.derive_L0_geo and bundle.L0_geo is not None:
            self.model.L0_geo_t = torch.as_tensor(bundle.L0_geo, dtype=torch.float32,
                                                  device=self.device)
            with torch.no_grad():
                l0_med = float(np.median(bundle.L0_geo))
                self.model.resp.log_s.data = (self.model.resp.log_L0.detach()
                                              - torch.log(torch.tensor(l0_med)))
        # γ 相带分带（gamma_facies_band 臂）：手册入渗系数区间逐井注入 ParamNet。
        # 纠正 γ 空间模式倒挂（实测冲洪积 0.08 / 海积 0.17，手册应为 0.15–0.30 / 0.05–0.15），
        # 并经 PDE 储量项把 Sy 推回真实量级（γ/Sy 振幅简并的物理侧约束）。
        if bool(ablation.get("gamma_facies_band", False)):
            gb = load_gamma_facies_bounds(cfg, bundle.well_ids)
            if gb is None:
                raise FileNotFoundError(
                    "gamma_facies_band 需要 data/param_reference/wells_param_zones.csv（make_param_reference.py 生成）")
            self.model.gamma_bounds_t = torch.as_tensor(gb, dtype=torch.float32,
                                                        device=self.device)
        self.head_mode = self.model.head_mode
        if self.pde_loss_mode == "correction_defect" and not self.model.use_anchor:
            raise ValueError("correction_defect 要求 use_anchor=true，以唯一分解 physics baseline 与 NN correction")
        if self.pde_loss_mode == "correction_defect" and not self.model.detach_phys_features:
            raise ValueError("correction_defect 强制要求 detach_phys_features=true，禁止 PDE 梯度泄漏到物理通路")
        self.future_forcing = bool(ablation.get("future_forcing", True))
        # 伪物理对照臂（物理特异性）：PDE 残差的强迫序列被随机置换（破坏物理内容、保留正则强度）。
        # 置换严格限制在【训练期索引 0..train_end_idx】内部（训练期封闭），
        # 训练期之外恒等映射——伪 PDE 只在训练窗上计算，绝不读取验证/测试期强迫。
        self.pde_shuffle = bool(ablation.get("pde_shuffle_forcing", False))
        if self.pde_loss_mode == "correction_defect" and self.pde_shuffle:
            raise ValueError("beta=0 缺陷差会抵消外部强迫，correction_defect 禁止使用 pde_shuffle_forcing")
        if self.pde_shuffle:
            rng_s = np.random.default_rng(int(tr["seed"]) + 99)
            perm = np.arange(bundle.H_fill.shape[0])
            tr_idx = np.arange(bundle.train_end_idx + 1)
            perm[tr_idx] = rng_s.permutation(tr_idx)
            self.forcing_perm = torch.as_tensor(perm, dtype=torch.long, device=self.device)
        # EMA 单检查点权重平均（小样本降种子方差；单模型合法）
        self.ema_decay = float(ablation.get("ema_decay", tr.get("ema_decay", 0.0)))
        self.ema_state = None
        self.rollout = FluxRollout(f["et_extinction_depth_m"],
                                   self.dt_days, float(p["rollout_dh_clip_m"]))
        self.pde = CVFDResidual(f["et_extinction_depth_m"],
                                self.dt_days, float(p["pde_char_scale_m_per_day"]))

        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=self.device)
        self.feats_t = t(feats["feats"])                  # [T, N, F]
        self.static_t = t(feats["static"])                # [N, S]
        self.bundle_train_end = int(bundle.train_end_idx) + 1
        self.H_fill = t(bundle.H_fill)                    # [T, N] 因果填充（特征/锚定）
        self.H_obs = t(np.nan_to_num(bundle.H_obs, nan=0.0))
        self.mask_t = t(bundle.mask)                      # [T, N]
        self.h_mean = t(normalizer.mean)                  # [N] 训练期观测均值（兼作扰动基准 h̄）
        self.h_std = t(normalizer.std)
        self.dem = t(bundle.dem)
        # Phase-2：真实底板几何（目标域缺资料时为 None → hydro_params 自动回退）与埋深绑定变量
        self.z_bot_ref_t = t(bundle.z_bot_ref) if bundle.z_bot_ref is not None else None
        self.depth_ref_t = torch.clamp(self.dem - self.h_mean, min=0.0)
        self.precip, self.et = t(bundle.precip), t(bundle.et)
        self.wu = t(bundle.wu)
        self.pde_mask = t(graphs.interior_mask.astype(np.float32) * bundle.aquifer_onehot[:, 0])

        # 训练期强迫均值（准稳态背景 W̄ 在损失中用当前 γ 动态计算，保持可微一致）
        tr_sl = slice(0, bundle.train_end_idx + 1)
        self.p_bar = self.precip[tr_sl].mean()
        self.et_bar = self.et[tr_sl].mean()
        self.wu_bar = self.wu[tr_sl].mean(0)                                     # [N]
        self.d_ext = float(f["et_extinction_depth_m"])

        # 训练期气候态强迫（no_future 消融：rollout 不用真实未来强迫）
        soy = np.minimum((bundle.dates.dayofyear - 1) // 5, 72).to_numpy()
        self.soy = torch.as_tensor(soy, dtype=torch.long, device=self.device)    # [T]
        n_soy = 73
        clim_p = np.zeros(n_soy, dtype=np.float32)
        clim_e = np.zeros(n_soy, dtype=np.float32)
        clim_w = np.zeros((n_soy, bundle.wu.shape[1]), dtype=np.float32)
        soy_tr = soy[: bundle.train_end_idx + 1]
        for si in range(n_soy):
            sel = soy_tr == si
            if sel.any():
                clim_p[si] = bundle.precip[: bundle.train_end_idx + 1][sel].mean()
                clim_e[si] = bundle.et[: bundle.train_end_idx + 1][sel].mean()
                clim_w[si] = bundle.wu[: bundle.train_end_idx + 1][sel].mean(0)
        self.clim_p, self.clim_e, self.clim_w = t(clim_p), t(clim_e), t(clim_w)

        # 逐步长损失均衡权重（horizon_balance）：w_k ∝ 1/Var_train[偏移_k]，均值归一
        # ——T+1 的目标偏移幅度远小于 T+6，不均衡会导致短步长欠拟合、打不过 persistence
        self.horizon_w = None
        if bool(tr.get("horizon_balance", False)):
            w_tr = windows["train"]
            t_last = w_tr.x_idx[:, -1]
            hz = (bundle.H_fill - normalizer.mean) / normalizer.std          # [T, N] z 空间
            var_k = []
            for k in range(int(cfg["data"]["pred_len"])):
                off = hz[w_tr.y_idx[:, k]] - hz[t_last]                      # [W, N]
                mk = bundle.mask[w_tr.y_idx[:, k]] > 0
                var_k.append(float(np.var(off[mk])) + 1e-4)
            w = 1.0 / np.asarray(var_k)
            self.horizon_w = t(w / w.mean())                                 # [P]

        self.win_year = bundle.dates.year.to_numpy()      # [T] 每步年份（分层评估用）

        # 物理锚可靠性系数（跨域迁移用）：None=不缩放；[N] 或标量，见 calibrate_phys_alpha
        self.phys_alpha = None

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=float(tr["lr"]),
                                     weight_decay=float(tr["weight_decay"]))
        # 学习率调度（warmup + 余弦退火）。117 个训练窗口下每轮仅 ~15 步，恒定大步长会在
        # 3 轮内冲进过拟合区；warmup+cosine 让优化过程用更多小步逼近平坦解。
        self.select_metric = str(ablation.get("select_metric", tr.get("select_metric", "rmse")))
        # 验证 skill 的逐轮抖动（±0.005）与消融臂之间的效应量（~0.002）同量级，单点 argmax
        # 选出的检查点由噪声主导。两种去噪选型（都只用验证集决策，测试集不参与）：
        #   select_smooth_w>0：按 ±w 轮中心滑动平均的验证分选中心轮的权重（主口径）；
        #   swa_top_k>1     ：对验证最优 k 轮做权重平均，平均后变差则回退（备选）。
        self.lambda_prior = float(ablation.get("lambda_prior", p.get("lambda_prior", 0.0)))
        self.sel_smooth_w = int(ablation.get("select_smooth_w", tr.get("select_smooth_w", 0)))
        self.swa_top_k = int(ablation.get("swa_top_k", tr.get("swa_top_k", 0)))
        self.lr_schedule = str(ablation.get("lr_schedule", tr.get("lr_schedule", "none")))
        self.warmup_steps = int(tr.get("warmup_steps", 0))
        self._step_count = 0
        self._total_steps = None
        self._pde_rng = np.random.default_rng(int(tr["seed"]) + 7)

    # -------------------------------------------------- lean 特征
    LEAN_LAGS = (1, 2, 3, 6, 12)
    N_LEAN = len(LEAN_LAGS) + 6            # 多尺度增量 + 强迫 4 项 + 季节 2 项
    # v4 融合头的已知未来强迫特征（scenario 假设下唯一外生信息源）：
    # p_fu / e_fu / w_fu / sin / cos —— 与 lean 头同口径，使主干路径不再仅经
    # 物理 rollout 间接感知未来强迫（今晨定位的 v4 技巧分缺口）
    N_FUT = 5

    def _fut_features(self, yi, t_last, dtype):
        """已知未来强迫距平 [B,N,N_FUT]。no_future 用训练期气候态；
        no_nnforcing 时强迫三项置零（只保留季节相位，与 nn_feat_idx 口径一致）。"""
        if self.future_forcing:
            p_fu = (self.precip[yi].mean(1) - self.p_bar) / (self.p_bar + 1e-8)      # [B]
            e_fu = (self.et[yi].mean(1) - self.et_bar) / (self.et_bar + 1e-8)        # [B]
            w_fu = (self.wu[yi].mean(1) - self.wu_bar) / (self.wu_bar.abs().mean() + 1e-8)
        else:
            soy_y = self.soy[yi]                                                     # [B,P]
            p_fu = (self.clim_p[soy_y].mean(1) - self.p_bar) / (self.p_bar + 1e-8)
            e_fu = (self.clim_e[soy_y].mean(1) - self.et_bar) / (self.et_bar + 1e-8)
            w_fu = (self.clim_w[soy_y].mean(1) - self.wu_bar) / (self.wu_bar.abs().mean() + 1e-8)
        B = p_fu.shape[0]
        N = self.h_mean.shape[0]
        ang = 2.0 * math.pi * self.soy[t_last].to(dtype) / 73.0
        bc = lambda v: v.view(B, 1).expand(B, N)
        if not self.nn_forcing:      # no_nnforcing：强迫只走物理路径
            z = torch.zeros(B, N, device=self.device, dtype=dtype)
            cols = [z, z, z, bc(torch.sin(ang)), bc(torch.cos(ang))]
        else:
            cols = [bc(p_fu), bc(e_fu), w_fu, bc(torch.sin(ang)), bc(torch.cos(ang))]
        return torch.stack(cols, dim=-1)

    def _lean_features(self, x_idx, yi, t_last):
        """低维物理特征 [B,N,D]：自身多尺度增量 + 已知未来强迫 + 季节相位。

        特征族与信号上限探针一致（该探针显示可学信号低维且几乎全部来自已知未来强迫）；
        没有自由的高维时序编码器，因而没有逐井记忆的通道。
        """
        hz = (self.H_fill - self.h_mean) / self.h_std                  # [T,N]
        cur = hz[t_last]                                               # [B,N]
        lags = [cur - hz[torch.clamp(t_last - lg, min=0)] for lg in self.lean_lags]
        xi = torch.as_tensor(x_idx, device=self.device)
        B, N = cur.shape
        if self.future_forcing:
            p_fu = (self.precip[yi].mean(1) - self.p_bar) / (self.p_bar + 1e-8)
            e_fu = (self.et[yi].mean(1) - self.et_bar) / (self.et_bar + 1e-8)
            w_fu = (self.wu[yi].mean(1) - self.wu_bar) / (self.wu_bar.abs().mean() + 1e-8)
        else:                        # no_future：NN 头也只能看训练期气候态（与 rollout 同口径）
            soy_y = self.soy[yi]
            p_fu = (self.clim_p[soy_y].mean(1) - self.p_bar) / (self.p_bar + 1e-8)
            e_fu = (self.clim_e[soy_y].mean(1) - self.et_bar) / (self.et_bar + 1e-8)
            w_fu = (self.clim_w[soy_y].mean(1) - self.wu_bar) / (self.wu_bar.abs().mean() + 1e-8)
        p_in = (self.precip[xi].mean(1) - self.p_bar) / (self.p_bar + 1e-8)
        ang = 2.0 * math.pi * self.soy[t_last].to(cur.dtype) / 73.0
        bcast = lambda v: v.view(B, 1).expand(B, N)
        if not self.nn_forcing:      # no_nnforcing：强迫只走物理路径，NN 头置零（保留季节相位）
            z = torch.zeros_like(cur)
            f_cols = [z, z, z, z]
        else:
            f_cols = [bcast(p_in), bcast(p_fu), bcast(e_fu), w_fu]
        cols = lags + f_cols + [bcast(torch.sin(ang)), bcast(torch.cos(ang))]
        return torch.stack(cols, dim=-1)                               # [B,N,D]

    # -------------------------------------------------- core
    def _forward_windows(self, x_idx: np.ndarray, y_idx: np.ndarray, node_mask=None):
        xb = self.feats_t[torch.as_tensor(x_idx, device=self.device)]      # [B, L, N, Fall]
        xb = xb.permute(0, 2, 1, 3)                                        # [B, N, L, Fall]
        if self.head_mode != "lean":
            # v4 增量化输入：水位通道换成逐步差分。绝对水位序列 + 全域标量强迫会让
            # 高容量时间编码器"认出窗口再记忆答案"（v3 主干过拟合、被迫退到 lean 头的根因）。
            # 差分后同一井在不同年份的相似动态共享表示，样本量实际是 窗口×井 而非 窗口。
            xb = xb.clone()
            h = xb[..., 0]                                                 # [B,N,L] 逐井 z 空间
            xb[..., 0] = torch.cat([torch.zeros_like(h[..., :1]),
                                    h[..., 1:] - h[..., :-1]], dim=-1)
        if node_mask is not None:      # masked-node：遮蔽井的历史水位置零、观测标志置 0
            keep = (~node_mask).to(xb.dtype).view(1, -1, 1)               # [1, N, 1]
            xb = xb.clone()
            xb[..., 0] = xb[..., 0] * keep      # h 通道历史清零
            xb[..., 1] = xb[..., 1] * keep      # 观测标志清零 → 模型据此知道该井需靠邻居重建
        xb = xb[..., self.nn_feat_idx]                                     # 选择 NN 输入通道
        t_last = torch.as_tensor(x_idx[:, -1], device=self.device)
        yi = torch.as_tensor(y_idx, device=self.device)
        h_last_m = self.H_fill[t_last]                                     # [B, N]
        h_last_z = (h_last_m - self.h_mean) / self.h_std
        hp = self.model.hydro_params(self.static_t, self.dem, self.z_bot_ref_t, self.depth_ref_t)
        K, Sy, gamma, z_bot = hp["K"], hp["Sy"], hp["gamma"], hp["z_bot"]
        tau_r = hp["tau_r"]
        # v4：退水常数由 (T̄, Sy) 导出而非自由输出 —— 这条绑定让 K/Sy 直接决定
        # 每口井的退水时序与补给响应幅值，从观测退水曲线上可辨识（v3 中 K 只进侧向
        # 通量项，梯度比垂向项小三个量级，反演实测为空转）。
        tau_b, T_bar = self.model.recession(hp, h_last_m)

        if self.model.use_phys_flux:
            if self.future_forcing:      # scenario 假设：未来强迫已知
                p_f, e_f, w_f = self.precip[yi], self.et[yi], self.wu[yi]
            else:                        # no_future 消融：训练期气候态替代
                soy_y = self.soy[yi]
                p_f, e_f, w_f = self.clim_p[soy_y], self.clim_e[soy_y], self.clim_w[soy_y]
            xi_t = torch.as_tensor(x_idx, device=self.device)
            precip_init = self.precip[xi_t].mean(dim=1)                    # [B] 输入窗平均降水
            # 补给湿度门控（gate 的正确位置）：前期降水距平 → γ 乘子 (0,2)，作用于主导垂向补给项
            if self.model.use_recharge_gate:
                anom = ((precip_init - self.p_bar) / (self.p_bar + 1e-6)).unsqueeze(-1)  # [B,1]
                mult = 2.0 * torch.sigmoid(self.model.recharge_gate(anom))               # [B,1]
                gamma_eff = gamma.unsqueeze(0) * mult                                     # [B,N]
            else:
                gamma_eff = gamma.unsqueeze(0).expand(len(x_idx), -1)                     # [B,N]
            # 物理锚：CVFD 显式积分 + 包气带滞后 + 线性退水（signed 由消融开关控制）
            h_phys_m = self.rollout(
                h_last_m, self.h_mean, K, Sy, gamma_eff, z_bot, self.dem, self.model.area,
                self.model.ei, self.model.ej, self.model.e_w, self.model.e_d,
                p_f, e_f, w_f,
                self.model.cond, signed=self.model.signed_flux,
                tau_r=tau_r, tau_b=tau_b, precip_init=precip_init)               # [B, N, P]
            phys_delta_z = (h_phys_m - self.h_mean.unsqueeze(-1)) / self.h_std.unsqueeze(-1) \
                - h_last_z.unsqueeze(-1)
            if self.phys_alpha is not None:    # 跨域：目标域历史上闭式再标定的锚可靠性系数
                phys_delta_z = phys_delta_z * self.phys_alpha
        else:
            phys_delta_z = torch.zeros(
                *h_last_z.shape, self.model.pred_len, device=self.device)

        # 达西注意力上下文：米制水头状态 + ParamNet 物理参数（K/z_bot 经此进入主干路径，
        # 使参数场对预测承重，而非仅供物理旁路消费）
        darcy_ctx = {"h_m": h_last_m, "h_anom_m": h_last_m - self.h_mean,
                     "K": K, "z_bot": z_bot}
        model_x = self._lean_features(x_idx, yi, t_last) if self.head_mode == "lean" else xb
        fut = None if self.head_mode == "lean" else self._fut_features(yi, t_last, xb.dtype)
        pred_parts = self.model(
            model_x, self.static_t, h_last_z, phys_delta_z, return_components=True,
            darcy_ctx=darcy_ctx, fut_feats=fut)
        h_pred_z = pred_parts["prediction"]
        h_true_m = self.H_obs[yi].permute(0, 2, 1)                         # [B, N, P]
        y_mask = self.mask_t[yi].permute(0, 2, 1)
        h_true_z = (h_true_m - self.h_mean.unsqueeze(-1)) / self.h_std.unsqueeze(-1)
        h_pred_m = h_pred_z * self.h_std.unsqueeze(-1) + self.h_mean.unsqueeze(-1)
        # 精确分解：h_pred = (h_last + phys_delta) + correction。
        # PDE 分支使用 detached phys_delta 构造 correction，因而只能更新 NN。
        physics_base_z = h_last_z.unsqueeze(-1) + phys_delta_z
        correction_z = (pred_parts["offset_z"]
                        + (pred_parts["gate"] - 1.0) * phys_delta_z.detach())
        pde_pred_z = physics_base_z.detach() + correction_z
        physics_base_m = physics_base_z * self.h_std.unsqueeze(-1) + self.h_mean.unsqueeze(-1)
        pde_pred_m = pde_pred_z * self.h_std.unsqueeze(-1) + self.h_mean.unsqueeze(-1)
        return {"h_pred_z": h_pred_z, "h_true_z": h_true_z, "h_pred_m": h_pred_m,
                "h_true_m": h_true_m, "h_last_m": h_last_m, "h_last_z": h_last_z,
                "y_mask": y_mask, "K": K, "Sy": Sy, "gamma": gamma, "z_bot": z_bot,
                "tau_b": tau_b, "T_bar": T_bar,
                "phys_delta_z": phys_delta_z, "physics_base_m": physics_base_m,
                "pde_pred_m": pde_pred_m, "correction_z": correction_z,
                "gate": pred_parts["gate"], "offset_z": pred_parts["offset_z"],
                "darcy_diag": pred_parts.get("darcy_diag")}

    def _shrink_feats(self) -> torch.Tensor:
        """逐井可预报性描述子 [N, N_SHRINK_FEATS]，只用训练期数据构造（无泄漏）。

        低信噪比井（自身波动小、增量小、观测稀）应当把 NN 修正量收回锚点；
        这些量在任何流域都能从历史序列直接算出，故收缩头可随权重迁移到新流域。
        """
        te = self.bundle_train_end
        h = self.H_fill[:te]                                            # [T,N] 米制
        m = self.mask_t[:te]
        sd = lambda v: torch.log(v.clamp(min=1e-3))
        cols = [sd(h.std(dim=0))]
        for lag in (1, 3, 6):
            d = h[lag:] - h[:-lag]
            cols.append(sd(d.std(dim=0)))
        cols.append(m.mean(dim=0))                                      # 观测覆盖率
        nb = torch.zeros_like(cols[0]).index_add_(
            0, self.model.da_dst, h.std(dim=0)[self.model.da_src])
        deg = torch.zeros_like(cols[0]).index_add_(
            0, self.model.da_dst, torch.ones_like(self.model.da_src, dtype=h.dtype))
        cols.append(sd(nb / deg.clamp(min=1)))                          # 邻域波动尺度
        x = torch.stack(cols, dim=-1)
        return (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)

    def enable_shrink_transfer(self) -> None:
        """跨域/重载场景启用收缩头：权重随 checkpoint 迁移，描述子用**本域**历史重算。

        描述子是闭式统计量（波动尺度、观测覆盖率、邻域尺度），不需要标签也不需要训练，
        因此 zero-shot 迁移时同样可得——这正是收缩头做成参数化而非逐井自由参数的原因。
        """
        self.model.shrink_feats = self._shrink_feats().detach()
        self.model.static_feats_buf = self.static_t.detach()
        self.model.use_shrink = True

    def fit_shrink(self, epochs: int = 300, lr: float = 0.05) -> dict:
        """在【验证期】标定可靠性收缩头，主干全程冻结。

        为什么不能在训练期标定：训练期模型对低信噪比井拟合良好，过度修正不显现，
        梯度不会推动 λ 离开 1。只有在留出期才能观测到"修正量帮倒忙"，这与后验逐井
        最小二乘标定同源，区别是这里用与井数无关的参数化，从而可迁移到新流域。
        """
        self.enable_shrink_transfer()
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.model.shrink_head.parameters():
            p.requires_grad_(True)

        # 冻结主干后，验证期的 raw 预测与锚是常量，先缓存再拟合 λ（快且无梯度污染）
        self.model.use_shrink = False
        w = self.windows["val"]
        with torch.no_grad():
            o = self._forward_windows(w.x_idx, w.y_idx)
            raw = o["h_pred_z"].detach()
            base = o["h_last_z"].unsqueeze(-1).detach()
            true_z = o["h_true_z"].detach()
            msk = o["y_mask"].detach()
        corr, targ = raw - base, true_z - base
        # 目标函数必须与评价口径一致：主口径是【逐井】NSE/skill 中位——每口井等权、
        # 且各自以自身方差归一。若直接最小化合并 MSE，梯度会被高方差井主导，
        # 安静井（正是需要收缩的那批）几乎不产生梯度，λ 就收不下来。
        cnt = msk.sum(dim=(0, 2))                                        # [N]
        mu = (true_z * msk).sum(dim=(0, 2)) / cnt.clamp(min=1)
        var = (((true_z - mu.view(1, -1, 1)) ** 2) * msk).sum(dim=(0, 2)) / cnt.clamp(min=1)
        wsel = cnt >= 5
        wnorm = torch.where(wsel, 1.0 / var.clamp(min=1e-4), torch.zeros_like(var))

        def nse_loss(lam):
            se = ((lam * corr - targ) ** 2 * msk).sum(dim=(0, 2))        # [N]
            return ((se / cnt.clamp(min=1)) * wnorm)[wsel].mean()

        opt = torch.optim.Adam(self.model.shrink_head.parameters(), lr=lr)
        one = torch.ones(1, 1, 1, device=raw.device)
        base_mse = float(nse_loss(one).detach())
        for _ in range(epochs):
            opt.zero_grad()
            lam = self.model.shrink_lambda().unsqueeze(0)
            loss = nse_loss(lam)
            loss.backward()
            opt.step()
        with torch.no_grad():
            lam = self.model.shrink_lambda()
            fit_mse = float(loss.detach())
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.use_shrink = True
        return {"shrink_val_mse_before": base_mse, "shrink_val_mse_after": fit_mse,
                "shrink_lambda_median": float(lam.median()),
                "shrink_lambda_p10": float(lam.quantile(0.10)),
                "shrink_lambda_p90": float(lam.quantile(0.90)),
                "shrink_frac_below_half": float((lam < 0.5).float().mean())}

    @torch.no_grad()
    def calibrate_phys_alpha(self, split: str = "train") -> float:
        """跨域物理锚闭式再标定（迁移协议内合法）：在目标域【训练期历史】上按预测步长
        最小二乘拟合锚可靠性系数 α_k = argmin‖d_obs,k − α_k·d_phys,k‖²，逐步长截断到 [0,1]。

        与逐井归一化统计同级别：只用目标域因果可得历史观测，无权重训练、无测试标签。
        源域标定的 γ/τ 等参数跨域幅值失配时，α_k<1 自动降低物理锚权重（保留形状信息）；
        逐步长拟合允许"短步长锚可靠、长步长失配"的差异化加权（预报混合的标准做法）。
        """
        if not self.model.use_phys_flux:
            self.phys_alpha = None
            return 1.0
        self.phys_alpha = None                     # 拟合期间取消缩放
        self.model.eval()
        w = self.windows[split]
        P = self.model.pred_len
        num = torch.zeros(P, device=self.device)
        den = torch.zeros(P, device=self.device)
        for s in range(0, len(w.x_idx), 16):
            o = self._forward_windows(w.x_idx[s:s + 16], w.y_idx[s:s + 16])
            d_obs = o["h_true_z"] - o["h_last_z"].unsqueeze(-1)
            d_phy = o["phys_delta_z"]
            m = o["y_mask"]
            num += (m * d_obs * d_phy).sum(dim=(0, 1))
            den += (m * d_phy * d_phy).sum(dim=(0, 1))
        alpha = (num / (den + 1e-8)).clamp(0.0, 1.0)               # [P]
        self.phys_alpha = alpha.view(1, 1, P)
        return float(alpha.mean())

    @staticmethod
    def _masked_mse(a, b, m):
        return (((a - b) ** 2) * m).sum() / (m.sum() + 1e-8)

    def _pde_effective_weight(self, data_loss, pde_loss):
        """返回 detached PDE 有效权重及梯度诊断。

        fixed 为主实验路径。可选 gradnorm 仅比较离输出最近的共享 NN 参数
        （trunk/offset/gate），不会用 ParamNet 梯度控制权重；若两项方向冲突，
        当前 batch 的 PDE 权重置零。首个 ramp 周期内线性升权，避免 epoch-1
        checkpoint 实际从未受到 PDE 训练。
        """
        if self.lambda_pde <= 0:
            return 0.0, {}
        if self.pde_ramp_epochs > 0:
            ramp = min(1.0, max(0.0, self._epoch_progress / self.pde_ramp_epochs))
        else:
            ramp = 1.0
        if self.pde_weight_mode == "fixed":
            return float(self.lambda_pde * ramp), {"pde_ramp": ramp}

        shared = (list(self.model.trunk.parameters())
                  + list(self.model.offset_head.parameters())
                  + list(self.model.gate_head.parameters()))
        gd = torch.autograd.grad(data_loss, shared, retain_graph=True, allow_unused=True)
        gp = torch.autograd.grad(pde_loss, shared, retain_graph=True, allow_unused=True)
        gd2 = torch.zeros((), device=self.device)
        gp2 = torch.zeros((), device=self.device)
        dot = torch.zeros((), device=self.device)
        for d, p_ in zip(gd, gp):
            if d is not None:
                gd2 = gd2 + (d.detach() ** 2).sum()
            if p_ is not None:
                gp2 = gp2 + (p_.detach() ** 2).sum()
            if d is not None and p_ is not None:
                dot = dot + (d.detach() * p_.detach()).sum()
        g_data = float(torch.sqrt(gd2 + 1e-24))
        g_pde = float(torch.sqrt(gp2 + 1e-24))
        cosine = float(dot / (torch.sqrt(gd2 * gp2) + 1e-12))
        raw = self.pde_grad_target_ratio * g_data / (g_pde + 1e-12)
        raw = min(self.pde_weight_max, max(self.pde_weight_min, raw))
        if self._adaptive_lambda is None:
            self._adaptive_lambda = raw
        else:
            a = self.pde_weight_ema
            self._adaptive_lambda = a * self._adaptive_lambda + (1.0 - a) * raw
        conflict = self.pde_conflict_gate and cosine <= 0.0
        effective = 0.0 if conflict else self._adaptive_lambda * ramp
        return float(effective), {
            "pde_ramp": ramp,
            "pde_grad_data_norm": g_data,
            "pde_grad_norm": g_pde,
            "pde_grad_cosine": cosine,
            "pde_grad_conflict": float(conflict),
        }

    def _loss(self, batch_x, batch_y):
        # masked-node：训练时随机遮蔽一部分井的输入历史，逼图消息传递重建（给边函数梯度）。
        # "弱遮蔽"口径：仅遮蔽输入历史特征（h_norm/mask 通道），锚 h_last 与物理初值仍可见
        # ——模型需靠邻居修正动态而非重建绝对水位；h 项对被遮蔽井 ×2，Δh 项经 m·m_prev 为 ×4。
        node_mask = None
        if self.node_mask_frac > 0 and self.model.learnable_edges:
            nm = torch.rand(self.model.n_nodes, device=self.device) < self.node_mask_frac
            node_mask = nm
        o = self._forward_windows(batch_x, batch_y, node_mask=node_mask)
        m = o["y_mask"]
        if node_mask is not None:
            wextra = 1.0 + self.mask_eval_weight * node_mask.to(m.dtype).view(1, -1, 1)
            m = m * wextra
        if self.horizon_w is not None:
            err2 = ((o["h_pred_z"] - o["h_true_z"]) ** 2) * m                # [B,N,P]
            per_k = err2.sum(dim=(0, 1)) / (m.sum(dim=(0, 1)) + 1e-8)        # [P]
            loss_h = (per_k * self.horizon_w).mean()
        else:
            loss_h = self._masked_mse(o["h_pred_z"], o["h_true_z"], m)

        anchor = o["h_last_z"].unsqueeze(-1)
        d_pred = torch.diff(torch.cat([anchor, o["h_pred_z"]], dim=-1), dim=-1)
        d_true = torch.diff(torch.cat([anchor, o["h_true_z"]], dim=-1), dim=-1)
        m_prev = torch.cat([torch.ones_like(m[..., :1]), m[..., :-1]], dim=-1)
        loss_dh = self._masked_mse(d_pred, d_true, m * m_prev)

        data_loss = self.w_h * loss_h + self.w_dh * loss_dh
        loss = data_loss
        comps = {"h": float(loss_h.detach()), "dh": float(loss_dh.detach()),
                 "data": float(data_loss.detach()), "pde": 0.0, "pde_weight": 0.0,
                 "pde_weighted": 0.0, "smooth": 0.0,
                 "smooth_weight": float(self.cfg["physics"]["smooth_reg_weight"]),
                 "smooth_weighted": 0.0,
                 "phys_delta_abs_z": float(o["phys_delta_z"].detach().abs().mean()),
                 "correction_abs_z": float(o["correction_z"].detach().abs().mean()),
                 "gate_mean": float(o["gate"].detach().mean())}

        if self.lambda_pde > 0:
            # 旧 trajectory 模式保持物理参数可微以复现历史结果；新版 correction_defect
            # 显式 detach 全部物理量，使 PDE 只更新 NN correction。
            if self.pde_loss_mode == "trajectory":
                K_pde, Sy_pde = o["K"], o["Sy"]
                gamma_pde, z_bot_pde = o["gamma"], o["z_bot"]
            elif self.pde_operator_params == "prior":
                # 严格析因：PDE-only 与 PDE+rollout 使用同一套固定文献先验算子，
                # 避免 R0/R1 中 ParamNet 训练状态不同而污染 PDE 主效应。
                K_pde = self.model.const_K.expand_as(o["K"]).detach()
                Sy_pde = self.model.const_Sy.expand_as(o["Sy"]).detach()
                gamma_pde = self.model.const_gamma.expand_as(o["gamma"]).detach()
                z_bot_pde = (self.z_bot_ref_t if self.z_bot_ref_t is not None
                             else self.dem - self.model.thickness_prior).detach()
            else:
                K_pde, Sy_pde = o["K"].detach(), o["Sy"].detach()
                gamma_pde, z_bot_pde = o["gamma"].detach(), o["z_bot"].detach()
            w_bar = vertical_source_m_per_day(
                self.h_mean, self.dem, self.p_bar, self.et_bar,
                self.wu_bar, self.model.area, gamma_pde, self.d_ext)
            pde_terms = []
            pde_diags = []
            picks = self._pde_rng.choice(len(batch_x), size=min(self.n_pde_win, len(batch_x)), replace=False)
            for b in picks:
                tidx = np.concatenate([[batch_x[b, -1]], batch_y[b]])
                ti = torch.as_tensor(tidx, device=self.device)
                if self.pde_shuffle:      # 伪物理对照：强迫时间置换（同 λ 同结构，物理内容被破坏）
                    ti = self.forcing_perm[ti]
                if self.pde_loss_mode == "trajectory":
                    seq_m = torch.cat(
                        [o["h_last_m"][b:b + 1], o["h_pred_m"][b].permute(1, 0)], dim=0)
                    pde_terms.append(self.pde(
                        seq_m, self.h_mean, w_bar, o["K"], o["Sy"], o["gamma"], o["z_bot"],
                        self.dem, self.model.area, self.pde_mask,
                        self.model.ei, self.model.ej, self.model.e_w, self.model.e_d,
                        self.precip[ti], self.et[ti], self.wu[ti], self.model.cond,
                        signed=self.model.signed_flux))
                else:
                    pred_seq_m = torch.cat(
                        [o["h_last_m"][b:b + 1], o["pde_pred_m"][b].permute(1, 0)], dim=0)
                    ref_seq_m = torch.cat(
                        [o["h_last_m"][b:b + 1], o["physics_base_m"][b].permute(1, 0)], dim=0).detach()
                    term, diag = self.pde.defect_difference_loss(
                        pred_seq_m, ref_seq_m, self.h_mean, w_bar.detach(),
                        K_pde, Sy_pde, gamma_pde, z_bot_pde,
                        self.dem, self.model.area, self.pde_mask,
                        self.model.ei, self.model.ej, self.model.e_w, self.model.e_d,
                        self.precip[ti], self.et[ti], self.wu[ti], self.model.cond,
                        signed=self.model.signed_flux, beta=self.pde_defect_beta)
                    pde_terms.append(term)
                    pde_diags.append(diag)
            pde = torch.stack(pde_terms).mean()
            effective_weight, weight_diag = self._pde_effective_weight(data_loss, pde)
            loss = loss + effective_weight * pde
            comps["pde"] = float(pde.detach())
            comps["pde_weight"] = effective_weight
            comps["pde_weighted"] = effective_weight * float(pde.detach())
            comps.update(weight_diag)
            if pde_diags:
                for key in pde_diags[0]:
                    comps[f"pde_{key}"] = float(torch.stack([d[key] for d in pde_diags]).mean().detach())

        if self.pde_obs_weight > 0:
            # 观测平衡残差（Phase-3 反演锚）：CVFD 残差算在训练期真实观测增量上。
            # 参数全程可微（K 经传导度与侧向通量、Sy 经储量项、γ 经源汇项收梯度）；
            # 只在两端都有真实观测的内部潜水节点计分（填充值不参与）。
            w_bar_obs = vertical_source_m_per_day(
                self.h_mean, self.dem, self.p_bar, self.et_bar,
                self.wu_bar, self.model.area, o["gamma"], self.d_ext)
            obs_terms = []
            picks_o = self._pde_rng.choice(len(batch_x), size=min(self.n_pde_win, len(batch_x)),
                                           replace=False)
            for b in picks_o:
                tidx = np.concatenate([[batch_x[b, -1]], batch_y[b]])
                ti_h = torch.as_tensor(tidx, device=self.device)
                ti_f = self.forcing_perm[ti_h] if self.pde_shuffle else ti_h
                raw = self.pde.raw_residual(
                    self.H_fill[ti_h], self.h_mean, w_bar_obs,
                    o["K"], o["Sy"], o["gamma"], o["z_bot"],
                    self.dem, self.model.area,
                    self.model.ei, self.model.ej, self.model.e_w, self.model.e_d,
                    self.precip[ti_f], self.et[ti_f], self.wu[ti_f], self.model.cond,
                    signed=self.model.signed_flux)                      # [S-1, N]
                m_obs = self.mask_t[ti_h]
                vm = m_obs[:-1] * m_obs[1:] * self.pde_mask.unsqueeze(0)  # [S-1, N]
                if self.pde_obs_cumulative:
                    # 窗口累积口径：残差先沿时间取有效步均值再平方（水量平衡在窗口
                    # 尺度闭合）。逐步增量的 Δh_obs 被观测噪声主导，逐步平方会迫使
                    # 优化器压缩 Sy 匹配噪声（实测 Sy→0.014、带内 6%）；累积后随机
                    # 噪声对消，Sy 对齐的是窗口尺度的真实蓄变量。
                    r_win = (raw * vm).sum(0) / vm.sum(0).clamp(min=1.0)   # [N]
                    has = (vm.sum(0) > 0).float()
                    r = r_win / self.pde.r0
                    obs_terms.append((r ** 2 * has).sum() / has.sum().clamp(min=1.0))
                else:
                    r = raw / self.pde.r0
                    obs_terms.append((r ** 2 * vm).sum() / vm.sum().clamp(min=1.0))
            pde_obs = torch.stack(obs_terms).mean()
            loss = loss + self.pde_obs_weight * pde_obs
            comps["pde_obs"] = float(pde_obs.detach())
            comps["pde_obs_weighted"] = self.pde_obs_weight * float(pde_obs.detach())

        if self.model.use_paramnet:
            sm = float(self.cfg["physics"]["smooth_reg_weight"])
            smooth = (ParamNet.smoothness_penalty(torch.log(o["K"]), self.model.ei, self.model.ej, self.model.e_d)
                      + ParamNet.smoothness_penalty(o["Sy"], self.model.ei, self.model.ej, self.model.e_d)
                      + 0.01 * ((o["z_bot"] - (self.dem - self.model.thickness_prior)) ** 2).mean())
            loss = loss + sm * smooth
            comps["smooth"] = float(smooth.detach())
            comps["smooth_weighted"] = sm * float(smooth.detach())
            if self.lambda_prior > 0:      # MAP 先验锚：欠定方向拉回文献先验，避免拟合噪声
                pr = self.model.param_net.prior_penalty(self.static_t)
                loss = loss + self.lambda_prior * pr
                comps["prior"] = float(pr.detach())
                comps["prior_weighted"] = self.lambda_prior * float(pr.detach())
        return loss, comps

    # -------------------------------------------------- loops
    def _ema_update(self):
        with torch.no_grad():
            sd = self.model.state_dict()
            if self.ema_state is None:
                self.ema_state = {k: v.detach().clone() for k, v in sd.items()}
                return
            d = self.ema_decay
            for k, v in sd.items():
                if v.dtype.is_floating_point:
                    self.ema_state[k].mul_(d).add_(v, alpha=1.0 - d)
                else:
                    self.ema_state[k].copy_(v)

    def _swap_in_ema(self):
        """临时把 EMA 权重换进模型（返回原权重备份）."""
        bak = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.ema_state, strict=True)
        return bak

    def _swa_finalize(self, pool: list, best_score: float) -> dict:
        """对验证最优的 k 个检查点做权重平均，并在验证集上与单点最优对比后二选一。

        pool 为 [(select_score, epoch, state_dict)]，score 越小越好。整型缓冲取最优轮的值。
        接受与否只看验证分数，测试集不参与；若平均后变差则回退单点最优（记录在 train_log）。
        """
        info = {"swa_top_k": self.swa_top_k, "swa_epochs": [], "swa_used": False,
                "swa_val_score": None, "single_best_val_score": best_score}
        if self.swa_top_k <= 1 or len(pool) < 2:
            return info
        top = sorted(pool, key=lambda r: r[0])[: self.swa_top_k]
        info["swa_epochs"] = [ep for _, ep, _ in top]
        avg = {}
        for k, v in top[0][2].items():
            if v.dtype.is_floating_point:
                avg[k] = torch.stack([s[k].float() for _, _, s in top]).mean(0).to(v.dtype)
            else:
                avg[k] = v.clone()
        bak = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(avg, strict=True)
        vo = self.evaluate("val")["overall"]
        score = (-vo["skill_well_median"] if self.select_metric == "skill"
                 else vo["rmse_well_median"])
        info["swa_val_score"] = score
        if score < best_score:
            info["swa_used"] = True
            info["best_val_rmse_well_median"] = vo["rmse_well_median"]
            torch.save(self.model.state_dict(), self.out_dir / "best_model.pt")
        self.model.load_state_dict(bak, strict=True)
        return info

    @staticmethod
    def _epoch_log_fields(agg: dict, n_win: int) -> dict:
        """损失与机制诊断分开命名，避免把 gate/λ 等误标为 loss。"""
        loss_keys = {"h", "dh", "data", "pde", "pde_weighted",
                     "smooth", "smooth_weighted", "total"}
        out = {}
        for key, value in agg.items():
            prefix = "loss" if key in loss_keys else "diag"
            out[f"{prefix}_{key}"] = value / n_win
        return out

    def _apply_lr_schedule(self):
        """warmup + 余弦退火；lr_schedule='none' 时保持原恒定学习率（历史结果可复现）。"""
        if self.lr_schedule == "none":
            return
        self._step_count += 1
        base = float(self.cfg["train"]["lr"])
        if self._step_count <= self.warmup_steps:
            scale = self._step_count / max(self.warmup_steps, 1)
        else:
            done = (self._step_count - self.warmup_steps) / max(
                (self._total_steps or self._step_count) - self.warmup_steps, 1)
            scale = 0.5 * (1.0 + math.cos(math.pi * min(done, 1.0)))
        for gp in self.opt.param_groups:
            gp["lr"] = base * scale

    def train(self, epochs: int | None = None):
        tr = self.cfg["train"]
        epochs = epochs or int(tr["epochs"])
        bs, patience = int(tr["batch_windows"]), int(tr["patience"])
        w = self.windows["train"]
        n_win = len(w.x_idx)
        self._total_steps = epochs * ((n_win + bs - 1) // bs)
        best_val, bad, history = float("inf"), 0, []
        best_epoch, lambda_at_best, best_rmse = None, 0.0, float("inf")
        swa_pool: list = []                       # [(select_score, epoch, state_dict)]
        ring = collections.deque(maxlen=2 * self.sel_smooth_w + 1)   # 平滑选型的中心窗口
        rng = np.random.default_rng(int(tr["seed"]))
        t0 = time.time()
        for ep in range(1, epochs + 1):
            self.model.train()
            order = rng.permutation(n_win)
            agg = {"total": 0.0}
            for s in range(0, n_win, bs):
                idx = order[s: s + bs]
                self._epoch_progress = (ep - 1) + min(1.0, (s + len(idx)) / max(n_win, 1))
                self.opt.zero_grad()
                loss, comps = self._loss(w.x_idx[idx], w.y_idx[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(tr["grad_clip"]))
                self._apply_lr_schedule()
                self.opt.step()
                if self.ema_decay > 0:
                    self._ema_update()
                for k, v in comps.items():
                    agg[k] = agg.get(k, 0.0) + v * len(idx)
                agg["total"] += float(loss.detach()) * len(idx)

            # 验证：EMA 开启时用 EMA 权重计分并保存（单检查点权重平均，非集成）
            if self.ema_decay > 0 and self.ema_state is not None:
                bak = self._swap_in_ema()
                val = self.evaluate("val")["overall"]["rmse_well_median"]
                if val < best_val - 1e-5:
                    best_val, bad = val, 0
                    best_epoch = ep
                    lambda_at_best = agg.get("pde_weight", 0.0) / n_win
                    torch.save(self.model.state_dict(), self.out_dir / "best_model.pt")
                else:
                    bad += 1
                self.model.load_state_dict(bak, strict=True)
                history.append({"epoch": ep, "val_rmse_well_median": val,
                                **self._epoch_log_fields(agg, n_win)})
                if bad >= patience:
                    break
                continue

            vo = self.evaluate("val")["overall"]
            # 选型口径与论文主口径对齐：正文以逐井配对（skill / 胜率）为主口径，
            # 而 rmse_well_median 是非配对量，二者在本数据上会给出相反的选择。
            val = (-vo["skill_well_median"] if self.select_metric == "skill"
                   else vo["rmse_well_median"])
            history.append({"epoch": ep, "val_rmse_well_median": vo["rmse_well_median"],
                            "val_rmse_m_pooled": vo["rmse_m_pooled"],
                            "val_skill_well_median": vo["skill_well_median"],
                            "val_nse_well_median": vo["nse_well_median"],
                            **self._epoch_log_fields(agg, n_win)})
            if self.swa_top_k > 1:
                swa_pool.append((val, ep, {k: v.detach().clone().cpu()
                                           for k, v in self.model.state_dict().items()}))
                swa_pool = sorted(swa_pool, key=lambda r: r[0])[: self.swa_top_k]

            if self.sel_smooth_w > 0:
                # 平滑选型：把最近 2w+1 轮的验证分取中心滑动平均，选平均最优那一段的中心轮权重。
                # 单点尖峰不再能决定选型，早停也改由平滑分驱动。
                ring.append((ep, val, vo["rmse_well_median"],
                             {k: v.detach().clone().cpu()
                              for k, v in self.model.state_dict().items()}))
                if len(ring) == ring.maxlen:
                    sm = sum(r[1] for r in ring) / len(ring)
                    c_ep, _, c_rmse, c_sd = ring[self.sel_smooth_w]
                    if sm < best_val - 1e-6:
                        best_val, bad = sm, 0
                        best_epoch, best_rmse = c_ep, c_rmse
                        lambda_at_best = agg.get("pde_weight", 0.0) / n_win
                        torch.save(c_sd, self.out_dir / "best_model.pt")
                    else:
                        bad += 1
                        if bad >= patience:
                            break
                continue

            if val < best_val - 1e-5:
                best_val, bad = val, 0
                best_epoch, best_rmse = ep, vo["rmse_well_median"]
                lambda_at_best = agg.get("pde_weight", 0.0) / n_win
                torch.save(self.model.state_dict(), self.out_dir / "best_model.pt")
            else:
                bad += 1
                if bad >= patience:
                    break
        swa_info = self._swa_finalize(swa_pool, best_val)
        best_rmse = swa_info.pop("best_val_rmse_well_median", best_rmse)
        swa_info["select_smooth_w"] = self.sel_smooth_w
        json.dump({"history": history, "best_val_rmse_well_median": best_rmse,
                   "select_metric": self.select_metric, "best_select_score": best_val,
                   "best_epoch": best_epoch, "lambda_at_best": lambda_at_best,
                   "pde_loss_mode": self.pde_loss_mode,
                   "pde_weight_mode": self.pde_weight_mode, **swa_info,
                   "wall_time_s": time.time() - t0, "epochs_run": len(history)},
                  open(self.out_dir / "train_log.json", "w"), indent=1)
        if not (self.out_dir / "best_model.pt").exists():
            # 训练轮数不足以填满平滑选型窗（±select_smooth_w，如 3-epoch 冒烟测试）时
            # 从未触发过检查点保存：回退保存最终权重，保证短跑也能走完评估链路。
            torch.save(self.model.state_dict(), self.out_dir / "best_model.pt")
        self.model.load_state_dict(
            torch.load(self.out_dir / "best_model.pt", weights_only=True), strict=True)
        return best_rmse

    @torch.no_grad()
    def evaluate(self, split: str, return_arrays: bool = False):
        self.model.eval()
        w = self.windows[split]
        preds, trues, persist, masks = [], [], [], []
        bs = 16
        for s in range(0, len(w.x_idx), bs):
            xi, yi = w.x_idx[s: s + bs], w.y_idx[s: s + bs]
            o = self._forward_windows(xi, yi)
            preds.append(o["h_pred_m"].cpu().numpy())
            trues.append(o["h_true_m"].cpu().numpy())
            masks.append(o["y_mask"].cpu().numpy())
            persist.append(np.repeat(o["h_last_m"].cpu().numpy()[:, :, None],
                                     o["h_true_m"].shape[-1], axis=2))
        pred, true = np.concatenate(preds), np.concatenate(trues)
        pers, mask = np.concatenate(persist), np.concatenate(masks)
        metrics = evaluate_meters(true, pred, pers, mask)
        if not return_arrays:
            return metrics
        # 每个窗口目标块首步的年份（分层评估用）
        w_year = self.win_year[np.asarray([yi0 for yi0 in w.y_idx[:, 0]])]
        arr = {"pred": pred.astype(np.float32), "true": true.astype(np.float32),
               "persist": pers.astype(np.float32), "mask": mask.astype(np.uint8),
               "window_year": w_year.astype(np.int16)}
        return metrics, arr

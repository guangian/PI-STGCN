"""物理模块：
- ParamNet（dPL 化）：物理协变量 → K / Sy / Δz_bot，先验有界 + 同层距离加权平滑正则 + dz 幅度正则；
- 反对称达西通量：F_ij = T_ij·(h_i − h_j)，F_ij = −F_ji 架构级守恒，符号保留流向；
- CVFD 控制体水量平衡残差（MODFLOW-USG 式）：
  * 仅在【潜水 × 内部 Voronoi】节点计算（Boussinesq 适用域）；
  * 强迫与水位变化区间对齐：h_t→h_{t+1} 用标记 t+1（覆盖 (t, t+1]）的强迫；
  * 残差按特征尺度 r0 无量纲化后平方，与数据损失同 O(1) 量级。

单位表：h/z [m]；K [m/d]；Sy [-]；A [m²]；w,d [m]；T_ij [m²/d]；
F_ij [m³/d]；R(补给) [m/d]；Q(抽水) [m³/d]；残差 r_i [m/d]，r0 [m/d]。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ParamNet(nn.Module):
    """Static well descriptors -> bounded effective K, Sy, gamma and free tau_b.

    The first three outputs are the effective parameter fields described in
    the manuscript. ``tau_b`` is retained only for the reported free-recession
    ablation; the full model derives recession from Sy and effective T.
    """

    def __init__(self, in_dim: int, hidden: int, k_range, sy_range, gamma_range,
                 tau_b_range,
                 k_prior: float = 10.0, sy_prior: float = 0.10, gamma_prior: float = 0.15,
                 tau_b_prior: float = 365.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 4))
        self.register_buffer("logk_lo", torch.log(torch.tensor(float(k_range[0]))))
        self.register_buffer("logk_hi", torch.log(torch.tensor(float(k_range[1]))))
        self.sy_lo, self.sy_hi = float(sy_range[0]), float(sy_range[1])
        self.g_lo, self.g_hi = float(gamma_range[0]), float(gamma_range[1])
        self.register_buffer("logtb_lo", torch.log(torch.tensor(float(tau_b_range[0]))))
        self.register_buffer("logtb_hi", torch.log(torch.tensor(float(tau_b_range[1]))))

        import math

        def logit(x):
            x = min(max(x, 1e-4), 1 - 1e-4)
            return math.log(x / (1 - x))

        def frac_log(v, lo, hi):
            return (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))

        def frac_lin(v, lo, hi):
            return (v - lo) / (hi - lo)

        prior_bias = [
            logit(frac_log(k_prior, float(k_range[0]), float(k_range[1]))),
            logit(frac_lin(sy_prior, self.sy_lo, self.sy_hi)),
            logit(frac_lin(gamma_prior, self.g_lo, self.g_hi)),
            logit(frac_log(tau_b_prior, float(tau_b_range[0]), float(tau_b_range[1]))),
        ]
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor(prior_bias, dtype=torch.float32))
    def forward(self, static_feats: torch.Tensor):
        s = torch.sigmoid(self.net(static_feats))
        K = torch.exp(self.logk_lo + s[:, 0] * (self.logk_hi - self.logk_lo))
        Sy = self.sy_lo + s[:, 1] * (self.sy_hi - self.sy_lo)
        gamma = self.g_lo + s[:, 2] * (self.g_hi - self.g_lo)
        tau_b = torch.exp(self.logtb_lo + s[:, 3] * (self.logtb_hi - self.logtb_lo))
        return {"K": K, "Sy": Sy, "gamma": gamma, "tau_b": tau_b}

    @staticmethod
    def smoothness_penalty(param: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                           dist: torch.Tensor) -> torch.Tensor:
        """同层图边上的距离加权平滑：近边差异惩罚重、远边轻."""
        w = 1.0 / (dist / 1000.0 + 1.0)                   # km 尺度衰减
        return (w * (param[src] - param[dst]) ** 2).sum() / (w.sum() + 1e-8)


def segment_softmax(logits: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """按目标节点分组的稀疏 softmax：logits [..., E]，index [E] ∈ [0,n)。"""
    lead = logits.shape[:-1]
    idx = index.view(*([1] * len(lead)), -1).expand(*lead, -1)
    mx = torch.full((*lead, n), -1e30, device=logits.device, dtype=logits.dtype)
    mx = mx.scatter_reduce(-1, idx, logits, reduce="amax", include_self=True)
    ex = torch.exp(logits - mx.gather(-1, idx))
    den = torch.zeros((*lead, n), device=logits.device, dtype=logits.dtype)
    den.scatter_add_(-1, idx, ex)
    return ex / (den.gather(-1, idx) + 1e-12)


def segment_logsumexp(logits: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """按目标节点分组的 logsumexp：logits [..., E] → [..., n]，无邻居的节点为 0。"""
    lead = logits.shape[:-1]
    idx = index.view(*([1] * len(lead)), -1).expand(*lead, -1)
    mx = torch.full((*lead, n), -1e30, device=logits.device, dtype=logits.dtype)
    mx = mx.scatter_reduce(-1, idx, logits, reduce="amax", include_self=True)
    ex = torch.exp(logits - mx.gather(-1, idx))
    den = torch.zeros((*lead, n), device=logits.device, dtype=logits.dtype)
    den.scatter_add_(-1, idx, ex)
    has = den > 0
    return torch.where(has, mx + torch.log(den.clamp(min=1e-30)), torch.zeros_like(den))


class RecessionTimeScale(nn.Module):
    """Derive tau = Sy * L0^2 / (4 * T_bar) for the physical rollout."""

    def __init__(self, tau_range, tau_prior: float, k_prior: float, sy_prior: float,
                 b_prior: float, learnable_scale: bool = True):
        super().__init__()
        import math

        T0 = k_prior * b_prior
        L0 = math.sqrt(4.0 * tau_prior * T0 / max(sy_prior, 1e-6))
        self.log_L0 = nn.Parameter(torch.tensor(math.log(L0)), requires_grad=learnable_scale)
        self.tau_lo, self.tau_hi = float(tau_range[0]), float(tau_range[1])

    def tau_days(self, T_bar: torch.Tensor, Sy: torch.Tensor) -> torch.Tensor:
        L0 = torch.exp(self.log_L0)
        tau = Sy * L0 * L0 / (4.0 * T_bar.clamp(min=1e-3))
        return tau.clamp(self.tau_lo, self.tau_hi)


class DarcyAttention(nn.Module):
    """达西注意力：把达西定律本身写成注意力打分函数（对标 Gravityformer 的引力注意力）。

    引力注意力  A_ij ∝ G · m_i^{λ1} · m_j^{λ2} · d_ij^{-λ3}
    达西注意力  A_ij ∝ G · K_j^{λ1} · K_i^{λ2} · b̄_ij^{λb} · (w_ij/d_ij)^{λg} · |Δh_ij|^{λh}

    λ 全取 1、G=1 时打分恰好等于达西交换通量的大小 |Q_ij| = T_ij·|h_i − h_j|，
    因此 softmax 之后的 α_ij 就是"井 i 的侧向水量交换在各邻居上的份额"——注意力权重
    自带物理量纲解释。λ 经 softplus 保证非负，使"K 越大/距离越远 → 权重越大/越小"的
    单调性在整个假设空间内恒成立；可学习的只是达西型幂律族内的指数，不是任意打分函数。

    与引力注意力的本质差别是达西流有方向：水自高水头流向低水头。故消息聚合取反对称
    形式 s_ij = tanh((h'_j − h'_i)/ε)，i 从 j 收到的消息与 j 从 i 收到的严格反号，
    与 F_ij = −F_ji 的守恒结构一致；引力/质量型注意力无法表达这一点。

    **份额与量级分离**：softmax 只保留邻居间的相对份额，会把每口井的侧向交换总量
    强行归一到 1，从而抹掉传导度的绝对大小；同时使任何仅依赖目标井 i 的因子
    （λ2·logK_i 与 logG）在组内为常数、被归一化消去而不可辨识。故量级由独立通道承载：
        τ_i = (Σ_j T_ij / median_i Σ_j T_ij) ** γ_T
    γ_T=0 退化为纯份额注意力（与不含该通道的版本逐位相同），γ_T=1 为完整达西量级缩放。

    参数量：6 个达西幂律标量 + γ_T + ε，全部与井数 N 无关。
    """

    def __init__(self, min_b: float = 1.0, eps_h_init: float = 0.1,
                 learnable_exponents: bool = True, dynamic: bool = True, signed: bool = True,
                 uniform: bool = False, magnitude: bool = True):
        super().__init__()
        self.min_b = float(min_b)
        self.uniform = uniform          # True = 打分恒等 → 退化为均匀 kNN 平均（对照臂）
        self.dynamic = dynamic          # False = 去掉 |Δh| 项（退化为静态传导度核）
        self.signed = signed            # False = 丢弃流向，只保留交换强度
        self.magnitude = magnitude      # False = 关闭量级通道（只保留归一化后的相对份额）
        # softplus 反函数：softplus(0.5413)=1.0 → 初始指数恰为达西定律的 1
        init = 0.541324854612918
        mk = lambda: nn.Parameter(torch.tensor(init), requires_grad=learnable_exponents)
        self.raw_lam_src = mk()         # λ1：源井（邻居 j）渗透系数指数
        self.raw_lam_dst = mk()         # λ2：目标井（i）渗透系数指数
        self.raw_lam_b = mk()           # λb：界面饱和厚度指数
        self.raw_lam_g = mk()           # λg：几何因子 (w/d) 指数
        self.raw_lam_h = mk()           # λh：水力梯度指数
        self.log_G = nn.Parameter(torch.zeros(()), requires_grad=learnable_exponents)
        # γ_T：量级通道指数。τ_i = (ΣT_ij / median) ** γ_T，γ_T=0 起步（τ≡1，与旧行为一致），
        # γ_T=1 即达西的全量级缩放。没有它，softmax 会把每口井的侧向交换总量强行归一到 1，
        # 传导度大小的信息被完全抹掉——这正是 λ2/G 不可辨识、且均匀权重与达西权重等效的根源。
        self.gamma_T = nn.Parameter(torch.zeros(()), requires_grad=learnable_exponents)
        self.raw_eps_h = nn.Parameter(
            torch.tensor(float(torch.log(torch.expm1(torch.tensor(eps_h_init))))))

    def exponents(self) -> dict:
        sp = nn.functional.softplus
        return {"lam_src": sp(self.raw_lam_src), "lam_dst": sp(self.raw_lam_dst),
                "lam_b": sp(self.raw_lam_b), "lam_g": sp(self.raw_lam_g),
                "lam_h": sp(self.raw_lam_h), "G": torch.exp(self.log_G),
                "eps_h": sp(self.raw_eps_h)}

    def log_conductance(self, K, z_bot, h_m, src, dst, width, dist):
        """边传导度的对数 log T_ij = log(K̄^λ · b̄^λb · (w/d)^λg)（含可学习指数与 G）。"""
        e = self.exponents()
        logK = torch.log(K.clamp(min=1e-6))
        b_i = torch.clamp(h_m - z_bot, min=self.min_b)                       # [B,N]
        log_b = torch.log(0.5 * (b_i[..., src] + b_i[..., dst]))             # [B,E]
        log_geo = torch.log(width.clamp(min=1e-6)) - torch.log(dist.clamp(min=1e-6))
        return (self.log_G + e["lam_src"] * logK[src] + e["lam_dst"] * logK[dst]
                + e["lam_b"] * log_b + e["lam_g"] * log_geo)                 # [B,E]

    def effective_T(self, K, z_bot, h_m, src, dst, width, dist, n):
        """邻域有效导水系数 T̄_i (m²/d)：达西注意力权重下的邻边面导水系数几何平均。

        物理动机：单井对面状补给的排泄响应由**流向排泄边界那条路径**的导水能力决定，
        而不是井点上的 K。故 τ 中的 T 取邻域达西加权量，而非局部值——这正是达西注意力
        在本模型里承重的地方（关掉它，τ 退化为逐井独立量，退水时序失去空间一致性）。

        量纲纪律（2026-08-02 修复）：被聚合的量必须是量纲固定的**面导水系数**
        T_ij = K̄_ij·b̄_ij（K̄ 取几何平均，m²/d）；可学习指数只进入注意力权重 α
        （决定"哪条路径承重"），不进入被聚合量本身。此前版本聚合的是打分
        G·K^{λ1+λ2}·b̄^{λb}·(w/d)^{λg}·d——λ1+λ2≈2 把 K 平方、×w 引入界面宽度量纲，
        T̄ 实测膨胀 ~1.6e5 倍，τ = Sy·L₀²/(4T̄) 全体压到 30 d 下限：退水时序失去
        逐井差异，Sy 沦为自由振幅旋钮被压到区间下缘，K/Sy 可辨识性通道整体失效。
        修复后 gain = τ/Sy = L₀²/(4T̄) 只随 T̄ 变（振幅辨识 K），decay = exp(−Δt/τ)
        随 Sy/T̄ 变（时序辨识 Sy），两条通道解耦。

        返回 [B,N]。darcy_uniform 对照臂下权重同步退化为均匀平均（口径一致）。
        """
        logK = torch.log(K.clamp(min=1e-6))
        b_i = torch.clamp(h_m - z_bot, min=self.min_b)                       # [B,N]
        log_T_areal = (0.5 * (logK[src] + logK[dst])
                       + torch.log(0.5 * (b_i[..., src] + b_i[..., dst])))   # [B,E] m²/d
        if self.uniform:
            score = torch.zeros_like(log_T_areal)                            # 均匀 kNN 对照
        else:
            score = self.log_conductance(K, z_bot, h_m, src, dst, width, dist)
        alpha = segment_softmax(score.clamp(-30.0, 30.0), dst, n)
        contrib = alpha * log_T_areal
        log_Tbar = torch.zeros(*contrib.shape[:-1], n, device=contrib.device,
                               dtype=contrib.dtype).index_add_(-1, dst, contrib)
        return torch.exp(log_Tbar.clamp(-20.0, 20.0))

    def attention(self, h_anom, K, z_bot, h_m, src, dst, width, dist, n):
        """返回 α [B,E]（按目标井 dst 归一化）、有向符号 s [B,E] 与量级因子 τ [B,N]。

        α 只携带"份额"信息（softmax 后每个目标井的邻居权重和恒为 1），任何仅依赖目标井
        的因子在归一化中被消掉。达西定律里侧向交换的**总量**同样是物理量，由 τ 承载：
            τ_i = (Σ_j T_ij / median_i Σ_j T_ij) ** γ_T
        γ_T=0 时 τ≡1（退化为纯份额注意力），γ_T=1 时恢复完整的达西量级缩放。
        """
        e = self.exponents()
        dh = h_anom[..., src] - h_anom[..., dst]                             # [B,E] >0 → j 为上游
        if self.uniform:
            log_T = torch.zeros_like(dh)                                     # 均匀 kNN 对照
            logit = log_T
        else:
            log_T = self.log_conductance(
                K, z_bot, h_m, src, dst, width, dist).expand_as(dh)
            logit = log_T
            if self.dynamic:
                logit = logit + e["lam_h"] * torch.log(dh.abs() + 1e-4)
        alpha = segment_softmax(logit.clamp(-30.0, 30.0), dst, n)            # [B,E]
        sign = torch.tanh(dh / e["eps_h"]) if self.signed else torch.ones_like(dh)
        if self.magnitude and not self.uniform:
            log_Ti = segment_logsumexp(log_T.clamp(-30.0, 30.0), dst, n)     # [B,N]
            z = log_Ti - log_Ti.median(dim=-1, keepdim=True).values
            tau = torch.exp((self.gamma_T * z).clamp(-1.4, 1.4))             # τ ∈ [0.25, 4]
        else:
            tau = torch.ones_like(h_anom)
        return alpha, sign, tau

    def smooth_response(self, r, h_anom, K, z_bot, h_m, src, dst, width, dist,
                        beta_iso, beta_dir, return_diag: bool = False):
        """在【强迫响应场】上做达西加权再分配（本模块在主干上的承重形式）。

        降水/蒸散在本数据中是全流域标量，逐井响应差异必须由模型生成；而相邻控制体
        之间的响应差会被侧向流拉平——这正是达西算子的作用。故对响应场 r [B,N,P]：

            r̃_i = r_i + β_iso · Σ_j α_ij (r_j − r_i) + β_dir · Σ_j α_ij s_ij r_j

        第一项是达西加权的拉普拉斯平滑（对称、保均值），第二项是沿水力梯度的有向
        再分配（反对称、体现流向）。β 经 sigmoid 有界，全部参数量 = 2P，与 N 无关。
        """
        n = r.shape[1]
        alpha, sign, tau = self.attention(h_anom, K, z_bot, h_m, src, dst, width, dist, n)
        a = alpha.unsqueeze(-1)                                              # [B,E,1]
        t = tau.unsqueeze(-1)                                                # [B,N,1]
        nb = torch.zeros_like(r).index_add_(1, dst, a * r[:, src])           # Σ α_ij r_j
        deg = torch.zeros_like(r).index_add_(1, dst, a.expand_as(a * r[:, src]))
        out = r + beta_iso * t * (nb - deg * r)
        if self.signed:
            dirn = torch.zeros_like(r).index_add_(1, dst, a * sign.unsqueeze(-1) * r[:, src])
            out = out + beta_dir * t * dirn
        if not return_diag:
            return out
        with torch.no_grad():
            ent = -(alpha * torch.log(alpha + 1e-12))
            ent_node = torch.zeros(r.shape[0], n, device=r.device).index_add_(1, dst, ent)
            diag = {f"darcy_{k}": float(v.detach()) for k, v in self.exponents().items()}
            diag["darcy_eff_neighbors"] = float(torch.exp(ent_node).mean())
            diag["darcy_beta_iso"] = float(beta_iso.detach().mean())
            diag["darcy_beta_dir"] = float(beta_dir.detach().mean())
            diag["darcy_gamma_T"] = float(self.gamma_T.detach())
            diag["darcy_tau_p10"] = float(tau.detach().quantile(0.10))
            diag["darcy_tau_p90"] = float(tau.detach().quantile(0.90))
            diag["darcy_smooth_frac"] = float(
                ((out - r).abs().mean() / (r.abs().mean() + 1e-9)).detach())
        return out, diag

class EdgeConductance(nn.Module):
    """边传导度 T_ij = K̄_ij · b̄_ij · w_ij / d_ij（K̄ 谐波平均，b̄ 界面平均饱和厚度）."""

    def __init__(self, min_b: float):
        super().__init__()
        self.min_b = float(min_b)

    def forward(self, K, z_bot, h_m, ei, ej, width, dist):
        K_h = 2.0 * K[ei] * K[ej] / (K[ei] + K[ej] + 1e-8)
        b_i = torch.clamp(h_m[..., ei] - z_bot[ei], min=self.min_b)
        b_j = torch.clamp(h_m[..., ej] - z_bot[ej], min=self.min_b)
        return K_h * 0.5 * (b_i + b_j) * (width / dist)


def node_net_inflow(F, ei, ej, n):
    """节点净流入 = Σ_入 F − Σ_出 F（m³/d）；对每条边 (i,j)：j 收 +F_ij、i 失 −F_ij."""
    out = torch.zeros(*F.shape[:-1], n, device=F.device, dtype=F.dtype)
    out.index_add_(-1, ej, F)
    out.index_add_(-1, ei, -F)
    return out


def node_strength(F_abs, ei, ej, n):
    """无方向强度聚合 Σ|F|（no_sign 消融专用：完全丢弃方向，复刻旧 |Δh| 设计缺陷）."""
    out = torch.zeros(*F_abs.shape[:-1], n, device=F_abs.device, dtype=F_abs.dtype)
    out.index_add_(-1, ej, F_abs)
    out.index_add_(-1, ei, F_abs)
    return out


def vertical_source_m_per_day(h_m, dem, precip, et, wu, area,
                              gamma, d_ext: float):
    """垂向源汇 W = γ·P − ET_g(埋深衰减) − Q/A（m/d）.

    h_m [..., N]；precip/et 标量或 [...]（广播）；wu [..., N] m³/d；
    gamma 标量或 [N]（ParamNet 逐井输出）。
    5 天步长、~20 km 井距下，垂向源汇是水位变化的主导物理项
    （γP/Sy 雨季可达 0.1 m/步，而井间侧向通量仅 ~1e-4 m/步）。
    """
    depth = torch.clamp(dem - h_m, min=0.0)
    et_g = et * torch.clamp(1.0 - depth / d_ext, min=0.0)
    return gamma * precip - et_g - wu / area


class FluxRollout(nn.Module):
    """物理先验轨迹（结构锚）：

        h_phys(k) = h_phys(k−1) + Δt/Sy · [ W_k + net'(T, h'−h̄)/A ]，k = 1..P

    - W_k：垂向源汇（已知未来强迫，scenario-based forecasting 假设）；
    - net'：扰动形式侧向通量 F'_ij = T_ij·(h'_i − h'_j)，h' = h − h̄（训练期均值），
      线性化 Boussinesq——准稳态背景流被消去，静态水头差不再污染动态；
    - signed=False 时（no_sign 消融）侧向通量改 Σ|F'| 强度聚合：所有节点只进不出，
      演示"非负权重不守恒"缺陷的实际后果。
    完全可微：ParamNet 的 K/Sy 经此路径直接接收预测损失梯度。
    """

    def __init__(self, d_ext: float, dt_days: float, dh_clip_m: float):
        super().__init__()
        self.d_ext = float(d_ext)
        self.dt, self.clip = float(dt_days), float(dh_clip_m)

    def forward(self, h_last_m, h_mean_m, K, Sy, gamma, z_bot, dem, area,
                ei, ej, width, dist, precip_fut, et_fut, wu_fut,
                cond: EdgeConductance, signed: bool = True,
                tau_r=None, tau_b=None, precip_init=None):
        """CVFD 显式积分 + 包气带补给滞后 + 线性退水.

        h_last_m [B,N]；precip_fut/et_fut [B,P]；wu_fut [B,P,N]；
        tau_r/tau_b [N] 时间常数（天）；precip_init [B] 输入窗平均降水（初始化补给储库）。
        补给经一阶储库 S_r 释放：dS = γP·Δt − (S/τ_r)·Δt，实际到水位补给 = S/τ_r；
        退水项 −(h − h_mean)/τ_b·Δt 提供均值回归。
        """
        n = h_last_m.shape[-1]
        P = precip_fut.shape[-1]
        h = h_last_m
        # 补给储库初始化到与输入窗平均降水的稳态：S_r0 = γ·P_init·τ_r
        if tau_r is not None and precip_init is not None:
            S_r = gamma * precip_init.unsqueeze(-1) * tau_r.unsqueeze(0)   # [B, N]
        else:
            S_r = None
        traj = []
        for k in range(P):
            T = cond(K, z_bot, h, ei, ej, width, dist)                   # [B, E]
            h_anom = h - h_mean_m
            if signed:
                F = T * (h_anom[..., ei] - h_anom[..., ej])
                lat = node_net_inflow(F, ei, ej, n)                       # [B, N] m³/d
            else:
                F_abs = (T * (h_anom[..., ei] - h_anom[..., ej])).abs()
                lat = node_strength(F_abs, ei, ej, n)

            depth = torch.clamp(dem - h, min=0.0)
            et_g = et_fut[..., k:k + 1] * torch.clamp(1.0 - depth / self.d_ext, min=0.0)
            if S_r is not None:      # 包气带滞后：补给先入储库再释放
                recharge = S_r / tau_r.unsqueeze(0)                       # [B, N] m/d
                S_r = S_r + (gamma * precip_fut[..., k:k + 1] - recharge) * self.dt
            else:
                recharge = gamma * precip_fut[..., k:k + 1]
            W = recharge - et_g - wu_fut[..., k, :] / area                # [B, N] m/d
            dh = (W + lat / area) / Sy * self.dt
            if tau_b is not None:    # 线性退水（均值回归，治积分漂移）
                # τ_b 可为 [N]（ParamNet 自由输出）或 [B,N]（v4 由 T̄/Sy 导出，随状态变化）
                tb = tau_b if tau_b.dim() == h.dim() else tau_b.unsqueeze(0)
                dh = dh - (h - h_mean_m) / tb * self.dt
            h = h + dh.clamp(-self.clip, self.clip)
            traj.append(h)
        return torch.stack(traj, dim=-1)                                  # [B, N, P]


class CVFDResidual(nn.Module):
    """Boussinesq 控制体水量平衡残差（扰动/准稳态背景形式）。

        r_i = Sy_i·(h_i^{t+1} − h_i^t)/Δt − [net'(h')_i/A_i + (W_i(t+1) − W̄_i)]   [m/d]
        loss = mean_valid( (r_i / r0)² )                                            [无量纲]

    - 侧向通量用扰动水头 h' = h − h̄（消除静态水头差主导，v2.0 的系统性偏置来源）；
    - 源汇取距平 W − W̄（W̄ 为训练期平均源汇，吸收未知准稳态背景不平衡）；
    - valid = 潜水 × 内部节点；强迫取 t+1 标记（覆盖 (t, t+1] 区间）。

    The public release keeps the trajectory residual used by the manuscript.
    """

    def __init__(self, extinction_depth: float, dt_days: float,
                 char_scale_m_per_day: float):
        super().__init__()
        self.d_ext = float(extinction_depth)
        self.dt = float(dt_days)
        self.r0 = float(char_scale_m_per_day)

    def raw_residual(self, h_seq_m, h_mean_m, w_bar, K, Sy, gamma, z_bot, dem, area,
                     ei, ej, width, dist, precip, et, wu, cond: EdgeConductance,
                     signed: bool = True):
        """返回每个区间/节点的未归一化控制体残差 ``[S-1,N]``，单位 m/d。

        h_seq_m [S,N]；w_bar [N]；precip/et [S]；wu [S,N]。该函数不隐式
        detach；需要梯度隔离的调用方必须显式传入 detached 基线和物理参数。
        """
        h0, h1 = h_seq_m[:-1], h_seq_m[1:]                       # [S-1, N]
        T_edge = cond(K, z_bot, h0, ei, ej, width, dist)
        h_anom = h0 - h_mean_m
        F = T_edge * (h_anom[..., ei] - h_anom[..., ej])         # 扰动通量
        if signed:
            net_in = node_net_inflow(F, ei, ej, h0.shape[-1])    # [S-1, N] m³/d
        else:
            net_in = node_strength(F.abs(), ei, ej, h0.shape[-1])

        W = vertical_source_m_per_day(
            h0, dem.unsqueeze(0), precip[1:].unsqueeze(1), et[1:].unsqueeze(1),
            wu[1:], area.unsqueeze(0), gamma, self.d_ext)        # [S-1, N] m/d（t+1 标记）

        lhs = Sy.unsqueeze(0) * (h1 - h0) / self.dt
        rhs = net_in / area.unsqueeze(0) + (W - w_bar.unsqueeze(0))
        return lhs - rhs                                           # [S-1,N] m/d

    def _masked_square(self, raw_r, valid_mask):
        r = raw_r / self.r0                                      # 无量纲
        r = r * valid_mask.unsqueeze(0)
        return (r ** 2).sum() / (valid_mask.sum() * r.shape[0] + 1e-8)

    def forward(self, h_seq_m, h_mean_m, w_bar, K, Sy, gamma, z_bot, dem, area, valid_mask,
                ei, ej, width, dist, precip, et, wu, cond: EdgeConductance,
                signed: bool = True):
        """Return the masked, dimensionless trajectory CVFD loss."""
        raw = self.raw_residual(
            h_seq_m, h_mean_m, w_bar, K, Sy, gamma, z_bot, dem, area,
            ei, ej, width, dist, precip, et, wu, cond, signed=signed)
        return self._masked_square(raw, valid_mask)

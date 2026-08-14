"""PI-STGCN 主干（锚定式直接多步 + 结构化物理锚）：

输入 [B, N, L, F]（节点维显式保留）
 ├─ ParamNet(dPL)：静态协变量 → K/Sy/Δz/γ/τ（先验有界，节点数无关）
 ├─ 时间流：通道共享膨胀因果 TCN
 ├─ 空间流：可学习边函数邻接上的残差图卷积
 ├─ 物理锚：FluxRollout 显式积分轨迹 h_phys(1..P)（由 trainer 计算传入，
 │   可微 → ParamNet 从预测损失收梯度）
 └─ 输出融合：ĥ_z(k) = h_last_z + g_k ⊙ Δ_phys_z(k) + offset_k，
     g_k ∈ (0,1) 节点级门控（sigmoid 有界）；offset 为 NN 残差。

全部消融开关经构造函数单因素注入；图 buffer persistent=False（checkpoint 与节点数解耦，
跨流域直接加载）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F_nn

from .physics import DarcyAttention, EdgeConductance, ParamNet, ResponseKernel

# 收缩头的逐井可预报性描述子数量（见 Trainer._shrink_feats）：
# log 自身水位标准差、log 各尺度增量标准差(3)、观测覆盖率、log 邻域水位标准差
N_SHRINK_FEATS = 6


class TemporalBlock(nn.Module):
    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.norm = nn.GroupNorm(4, ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                       # [BN, C, L]
        y = self.conv(x)[..., : x.shape[-1]]    # 因果裁剪
        return x + self.drop(F_nn.silu(self.norm(y)))


class PISTGCNv2(nn.Module):
    def __init__(self, cfg: dict, n_feats: int, n_static: int, graphs, device,
                 use_phys_flux: bool = True, signed_flux: bool = True, use_paramnet: bool = True,
                 use_anchor: bool = True, cumsum_head: bool = False, use_cond_gate: bool = True,
                 learnable_edges: bool = True, use_leddam: bool = False, use_star: bool = False,
                 detach_phys_features: bool = False, use_darcy_attn: bool = True,
                 darcy_learnable_exponents: bool = True, darcy_dynamic: bool = True,
                 darcy_signed: bool = True, head_mode: str = "lean", n_lean: int = 11,
                 darcy_uniform: bool = False, darcy_magnitude: bool = True,
                 derive_tau: bool = True, darcy_content: bool = True, n_fut: int = 0,
                 use_ref_bottom: bool = False, derive_tau_r: bool = False,
                 derive_L0_geo: bool = False):
        super().__init__()
        m, p, g = cfg["model"], cfg["physics"], cfg["graph"]
        h = int(m["hidden"])
        self.pred_len = int(cfg["data"]["pred_len"])
        self.use_phys_flux, self.signed_flux, self.use_paramnet = use_phys_flux, signed_flux, use_paramnet
        self.use_anchor = use_anchor          # False = 直接回归绝对水位（no_anchor 消融：检验增量学习）
        self.cumsum_head = cumsum_head        # True = NN 偏移改逐步增量 cumsum（dh_cumsum 消融，旧机制对照）
        self.learnable_edges = learnable_edges  # False = 退回固定高斯核（no_graph 消融）
        self.use_leddam = use_leddam          # Leddam 式可学习因果趋势分解（Yu et al., ICML 2024）
        self.use_star = use_star              # SOFTS 式掩码全局核心（Han et al., NeurIPS 2024）
        # 新版 residual-only PDE：物理增量作为 NN 特征时切断梯度，避免 PDE 经融合干路
        # 回传到 rollout/ParamNet；监督预测中的 gate*phys_delta 仍保持可微。
        self.detach_phys_features = detach_phys_features
        self.thickness_prior = float(p["aquifer_thickness_prior_m"])
        # Phase-2 正逆问题升级：
        # use_ref_bottom —— 底板用钻孔地层几何插值（z_bot_ref），Δz 不再学习；目标域
        #   无该资料时（hydro_params 收到 None）自动回退 dem−b₀+Δz 先验几何。
        # derive_tau_r —— 包气带补给滞后绑定埋深：τ_r = clamp(c_r·depth, 区间)，c_r 全局
        #   可学习标量。把「埋深→响应慢」的空间梯度显式交给补给滞后通道，防止它经
        #   退水时序被错误吸收进 K/T̄（河北外部检验实测的负相关混杂来源，2026-08-02）。
        self.use_ref_bottom = use_ref_bottom
        self.derive_tau_r = derive_tau_r
        self.log_c_r = nn.Parameter(torch.zeros(()))     # c_r=1 d/m 起步（中位埋深≈20 m→τ_r≈20 d）
        self.tau_r_lo, self.tau_r_hi = (float(v) for v in p["tau_recharge_days"])
        # derive_L0_geo —— τ_b 的排泄半间距用逐井地理距离（DEM 汇流派生）×全局改正标量，
        #   而非全局单一 L₀；L0_geo_t 由 Trainer 注入（跨域缺资料时为 None 自动回退）。
        self.derive_L0_geo = derive_L0_geo
        self.L0_geo_t = None                             # [N]，Trainer 注入（不进 state_dict）

        # ---- 图张量：persistent=False（与节点数解耦） ----
        self.register_buffer("ei", torch.as_tensor(graphs.phys_i), persistent=False)
        self.register_buffer("ej", torch.as_tensor(graphs.phys_j), persistent=False)
        self.register_buffer("e_w", torch.as_tensor(graphs.phys_width), persistent=False)
        self.register_buffer("e_d", torch.as_tensor(graphs.phys_dist), persistent=False)
        self.register_buffer("area", torch.as_tensor(graphs.cell_area), persistent=False)
        n = graphs.cell_area.shape[0]
        self.n_nodes = n
        # 信息图（有向边列表 + 边属性），供可学习边函数在主路径上生成注意力权重
        self.register_buffer("info_src", torch.as_tensor(graphs.info_src), persistent=False)
        self.register_buffer("info_dst", torch.as_tensor(graphs.info_dst), persistent=False)
        self.register_buffer("info_attr", torch.as_tensor(graphs.info_edge_attr), persistent=False)
        # 固定高斯核权重（no_graph 消融退回此，也作可学习边函数的先验偏置）
        kern_m = float(g.get("info_kernel_km", 20.0)) * 1000.0
        self.register_buffer("info_w_prior",
                             torch.exp(-(torch.as_tensor(graphs.info_dist) / kern_m) ** 2), persistent=False)
        # 可学习边函数 g_φ(Δx,Δy,ΔDEM,logd) → 标量边权（参数量与 N 无关，跨域自洽）
        self.edge_fn = nn.Sequential(
            nn.Linear(graphs.info_edge_attr.shape[1], 32), nn.SiLU(), nn.Linear(32, 1))
        # 达西注意力跑在【物理控制体图】的有向闭包上（Delaunay 同层邻接 + Voronoi 界面宽），
        # 即 CVFD 通量面拓扑本身——注意力权重因此有 m³/d 的量纲解释
        di = torch.as_tensor(graphs.phys_i)
        dj = torch.as_tensor(graphs.phys_j)
        self.register_buffer("da_src", torch.cat([dj, di]), persistent=False)
        self.register_buffer("da_dst", torch.cat([di, dj]), persistent=False)
        self.register_buffer("da_w", torch.as_tensor(graphs.phys_width).repeat(2), persistent=False)
        self.register_buffer("da_d", torch.as_tensor(graphs.phys_dist).repeat(2), persistent=False)

        # ---- 模块 ----（ParamNet 先验中心 = no_paramnet 常数，保证消融对照公平）
        self.param_net = ParamNet(n_static, 32, p["k_range_m_per_day"], p["sy_range"],
                                  p["dz_range_m"], p["gamma_range"],
                                  p["tau_recharge_days"], p["tau_baseflow_days"],
                                  k_prior=float(p.get("k_const", 10.0)),
                                  sy_prior=float(p.get("sy_const", 0.10)),
                                  gamma_prior=float(p.get("gamma_const", 0.15)),
                                  tau_r_prior=float(p.get("tau_recharge_const", 20.0)),
                                  tau_b_prior=float(p.get("tau_baseflow_const", 365.0)))
        # γ 相带分带区间 [N,2]（gamma_facies_band 臂由 Trainer 注入；None = 全局标量区间）
        self.gamma_bounds_t: torch.Tensor | None = None
        self.register_buffer("const_K", torch.tensor(float(p.get("k_const", 10.0))))
        self.register_buffer("const_Sy", torch.tensor(float(p.get("sy_const", 0.10))))
        self.register_buffer("const_gamma", torch.tensor(float(p.get("gamma_const", 0.15))))
        self.register_buffer("const_tau_r", torch.tensor(float(p.get("tau_recharge_const", 20.0))))
        self.register_buffer("const_tau_b", torch.tensor(float(p.get("tau_baseflow_const", 365.0))))
        self.cond = EdgeConductance(p["min_sat_thickness_m"])
        # 物理响应核：把 τ 绑到 (T̄, Sy) 上，使反演参数直接决定退水时序与补给幅值。
        # derive_tau=False（tau_free 对照臂）时退回 ParamNet 的自由 τ_b，用于证明
        # "可辨识性来自这条绑定"而不是来自额外容量。
        self.derive_tau = derive_tau
        self.resp = ResponseKernel(
            p["tau_baseflow_days"], tau_prior=float(p.get("tau_baseflow_const", 365.0)),
            k_prior=float(p.get("k_const", 10.0)), sy_prior=float(p.get("sy_const", 0.10)),
            b_prior=float(p["aquifer_thickness_prior_m"]))
        # 补给湿度门控：γ_eff = γ·2σ(f(前期降水距平))，作用于主导的垂向补给项
        self.use_recharge_gate = use_cond_gate
        self.recharge_gate = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.recharge_gate[-1].weight)
        nn.init.zeros_(self.recharge_gate[-1].bias)      # 初始中性（乘子=1）

        # Leddam 式可学习因果分解（作用于水位通道，通道 0）：softmax 卷积核提取趋势，
        # 残差=去趋势扰动；[趋势, 扰动] 作为附加输入通道进 TCN（参数与 N 无关）
        if self.use_leddam:
            self.dec_kernel = nn.Parameter(torch.zeros(7))     # softmax 后≈均匀滑动平均起步
            n_feats_in = n_feats + 2
        else:
            n_feats_in = n_feats
        self.in_proj = nn.Linear(n_feats_in, h)
        self.static_proj = nn.Linear(n_static, h)
        self.tcn = nn.Sequential(*[TemporalBlock(h, m["tcn_kernel"], d_, m["dropout"])
                                   for d_ in m["tcn_dilations"]])
        self.gc = nn.ModuleList([nn.Linear(h, h) for _ in range(int(m["gc_layers"]))])
        # 达西注意力（始终实例化，开关只决定是否启用 → 参数量恒定）
        self.use_darcy_attn = use_darcy_attn
        self.darcy_uniform = darcy_uniform   # True = 打分退化为均匀 kNN 平均（对照臂）
        self.darcy_content = darcy_content   # False = 只留纯物理打分（去 Gravityformer 内容项）
        self.darcy = DarcyAttention(
            h, min_b=float(p["min_sat_thickness_m"]),
            learnable_exponents=darcy_learnable_exponents,
            dynamic=darcy_dynamic, signed=darcy_signed, uniform=darcy_uniform,
            magnitude=darcy_magnitude)
        # SOFTS/STAR 式掩码全局核心：全网聚合 → 残差门控分发（盆地尺度共同信号；与 N 无关）
        if self.use_star:
            self.star_enc = nn.Sequential(nn.Linear(h, h), nn.SiLU())
            self.star_gate = nn.Linear(3 * h, h)
            self.star_out = nn.Linear(2 * h, h)
            nn.init.zeros_(self.star_out.weight)
            nn.init.zeros_(self.star_out.bias)                 # 残差零初始化：中性起步
        # ---- lean 头（v3）：只吃低维物理特征，自由度从 ~9e4 压到 ~2e3 ----
        # 依据：16 系数岭回归在验证期已优于 90k 参数主干（skill +0.0559 对 +0.0497），
        # 说明可学信号低维且几乎全部来自已知未来强迫；大容量自由干只带来记忆而非泛化。
        self.head_mode = head_mode
        lh = int(m.get("lean_hidden", 32))
        self.lean_head = nn.Sequential(
            nn.Linear(n_lean + n_static + self.pred_len, lh), nn.SiLU(),
            nn.Dropout(m["dropout"]), nn.Linear(lh, lh), nn.SiLU())
        self.lean_offset = nn.Linear(lh, self.pred_len)
        self.lean_gate = nn.Linear(lh, self.pred_len)
        nn.init.zeros_(self.lean_offset.weight); nn.init.zeros_(self.lean_offset.bias)
        nn.init.constant_(self.lean_gate.bias, 0.0)
        # ---- 可靠性收缩头：ĥ = h_last + λ⊙(ĥ_raw − h_last)，λ = sigmoid(·) ∈ (0,1) ----
        # 训练期模型对低信噪比井拟合得很好，过度修正只在留出期显现，故本头**不参与主训练**，
        # 而在主干冻结后于验证期单独标定（见 Trainer.fit_shrink）。输入是逐井可预报性描述子
        # （自身波动尺度、观测密度、静态协变量），与井数无关，可直接前向迁移到新流域。
        self.shrink_head = nn.Sequential(
            nn.Linear(N_SHRINK_FEATS + n_static, 16), nn.SiLU(),
            nn.Linear(16, self.pred_len))
        nn.init.zeros_(self.shrink_head[-1].weight)
        nn.init.constant_(self.shrink_head[-1].bias, 4.0)    # sigmoid(4)=0.982 ≈ 不收缩
        self.use_shrink = False              # 标定完成后由 Trainer 置 True
        self.shrink_feats = None             # [N, N_SHRINK_FEATS]，由 Trainer 注入
        self.static_feats_buf = None         # [N, n_static]，收缩头用的静态协变量
        # 达西再分配强度（响应场上）：两个门的起点都必须落在零，
        # 使 full 与 no_darcy 在第 0 步逐位相同 —— 任何差异只能来自学习本身。
        # 门函数不同则原始参数的零点不同：sigmoid 用 -4（→0.018），tanh 用 0（→0）。
        self.beta_iso_raw = nn.Parameter(torch.full((self.pred_len,), -4.0))
        self.beta_dir_raw = nn.Parameter(torch.zeros(self.pred_len))

        # 融合头：共享干 + offset / gate 双输出（gate sigmoid 有界）
        # n_fut：已知未来强迫的距平（scenario 假设下唯一的外生信息源）。v3 的 lean 头
        # 把它作为显式特征，主干路径却只能经物理 rollout 间接看到——这是主干版技巧分
        # 落后的直接原因，v4 把它并回融合头。
        self.n_fut = int(n_fut)
        self.trunk = nn.Sequential(
            nn.Linear(3 * h + self.pred_len + self.n_fut, h), nn.SiLU(), nn.Dropout(m["dropout"]))
        self.offset_head = nn.Linear(h, self.pred_len)
        nn.init.zeros_(self.offset_head.weight)          # 零初始化：从"persistence+½物理"出发，稳 T+1
        nn.init.zeros_(self.offset_head.bias)
        self.gate_head = nn.Linear(h, self.pred_len)
        nn.init.constant_(self.gate_head.bias, 0.0)     # 初始 g=0.5：物理与 NN 均衡起步
        self.to(device)

    # ------------------------------------------------ 信息图消息传递（可学习边函数在主路径）
    def _graph_adj(self):
        """由可学习边函数 g_φ 生成对称归一化邻接（每次前向重算，参数与 N 无关）.

        no_graph 消融时退回固定高斯核先验；两者都做行归一化 + 自环。
        """
        n = self.n_nodes
        if self.learnable_edges:
            w = F_nn.softplus(self.edge_fn(self.info_attr).squeeze(-1)) * self.info_w_prior
        else:
            w = self.info_w_prior
        A = torch.zeros(n, n, device=w.device, dtype=w.dtype)
        A[self.info_src, self.info_dst] = w
        A = A + torch.eye(n, device=w.device)
        d = A.sum(1).clamp(min=1e-6).pow(-0.5)
        return d[:, None] * A * d[None, :]

    # ------------------------------------------------ physics params
    def hydro_params(self, static_feats, dem, z_bot_ref=None, depth_ref=None):
        """K/Sy/γ/τ 直接输出 + 含水层几何。

        z_bot_ref [N]：钻孔地层几何插值的浅层系统底板（use_ref_bottom 时优先；None 回退先验）。
        depth_ref [N]：埋深 = clamp(dem − h̄_train, 0)（derive_tau_r 的绑定变量；训练期统计量）。
        """
        n = static_feats.shape[0]
        if self.use_paramnet:
            pp = self.param_net(static_feats, gamma_bounds=self.gamma_bounds_t)
            K, Sy, dz, gamma, tau_r, tau_b = (pp["K"], pp["Sy"], pp["dz"], pp["gamma"],
                                              pp["tau_r"], pp["tau_b"])
        else:
            K = self.const_K.expand(n)
            Sy = self.const_Sy.expand(n)
            gamma = self.const_gamma.expand(n)
            dz = torch.zeros(n, device=static_feats.device)
            tau_r = self.const_tau_r.expand(n)
            tau_b = self.const_tau_b.expand(n)
        if self.use_ref_bottom and z_bot_ref is not None:
            z_bot = z_bot_ref                                # 真实几何：Δz 不参与
        else:
            z_bot = dem - self.thickness_prior + dz
        if self.derive_tau_r and depth_ref is not None:
            tau_r = torch.clamp(torch.exp(self.log_c_r) * depth_ref,
                                self.tau_r_lo, self.tau_r_hi)
        return {"K": K, "Sy": Sy, "gamma": gamma, "z_bot": z_bot, "tau_r": tau_r, "tau_b": tau_b}

    def recession(self, hp: dict, h_m):
        """由反演参数导出退水时间常数 τ_b 与邻域有效导水系数 T̄（v4 可辨识性的关键）。

        T̄ 走达西注意力（邻域加权），τ = Sy·L₀²/(4T̄)。达西关闭或 derive_tau=False 时
        分别退化为局部 T 与 ParamNet 的自由 τ_b —— 两者都是消融臂，用来分离
        "邻域加权" 与 "τ 绑定物理参数" 各自的贡献。返回 (tau_b [B,N] 或 [N], T̄)。
        """
        K, Sy, z_bot = hp["K"], hp["Sy"], hp["z_bot"]
        b = torch.clamp(h_m - z_bot, min=self.cond.min_b)                    # [B,N]
        if self.use_darcy_attn:
            T_bar = self.darcy.effective_T(K, z_bot, h_m, self.da_src, self.da_dst,
                                           self.da_w, self.da_d, self.n_nodes)
        else:
            T_bar = K * b                                                    # 局部导水系数
        if not self.derive_tau:
            return hp["tau_b"], T_bar
        L0_geo = self.L0_geo_t if self.derive_L0_geo else None
        return self.resp.tau_days(T_bar, Sy, L0_geo), T_bar

    # ------------------------------------------------ lean 前向（v3 主路径）
    def _forward_lean(self, lean_x, static_feats, h_last_z, phys_delta_z,
                      return_components: bool, darcy_ctx):
        """低容量前向：ĥ(k) = h_last + g_k⊙Δ_phys(k) + 达西再分配后的 NN 响应修正。

        lean_x [B,N,D] 为低维物理特征（自身多尺度增量 / 已知未来强迫 / 季节相位），
        由调用方构造——与信号上限探针中被证明可泛化的特征族一致。
        """
        B, N, _ = lean_x.shape
        st = static_feats.unsqueeze(0).expand(B, -1, -1)
        feat = self.lean_head(torch.cat([lean_x, st, phys_delta_z], dim=-1))
        offset = self.lean_offset(feat)                                  # [B,N,P] 响应场
        gate = torch.sigmoid(self.lean_gate(feat))

        darcy_diag = None
        if self.use_darcy_attn and darcy_ctx is not None:
            # 达西注意力作用在【响应场】上：全流域标量强迫 → 逐井响应的空间结构
            # 由物理连通性正则化，而非由自由参数逐井记忆
            beta_iso = torch.sigmoid(self.beta_iso_raw)
            beta_dir = torch.tanh(self.beta_dir_raw)
            out = self.darcy.smooth_response(
                offset, darcy_ctx["h_anom_m"], darcy_ctx["K"], darcy_ctx["z_bot"],
                darcy_ctx["h_m"], self.da_src, self.da_dst, self.da_w, self.da_d,
                beta_iso, beta_dir, return_diag=return_components)
            offset, darcy_diag = out if return_components else (out, None)

        base = h_last_z.unsqueeze(-1) if self.use_anchor else torch.zeros_like(offset)
        pred = base + gate * phys_delta_z + offset
        if self.use_shrink and self.shrink_feats is not None:
            # 可靠性收缩：低信噪比井把整个修正量按 λ 收回锚点（λ 在验证期标定）
            lam = self.shrink_lambda()                                   # [N,P]
            pred = base + lam.unsqueeze(0) * (pred - base)
        if return_components:
            return {"prediction": pred, "base_z": base, "gate": gate, "offset_z": offset,
                    "darcy_diag": darcy_diag}
        return pred

    def shrink_lambda(self):
        """逐井 × 逐预见期的收缩系数 λ ∈ (0,1)，由可预报性描述子 + 静态协变量生成。"""
        x = torch.cat([self.shrink_feats, self.static_feats_buf], dim=-1)
        return torch.sigmoid(self.shrink_head(x))                        # [N,P]

    # ------------------------------------------------ forward（唯一路径）
    def forward(self, x, static_feats, h_last_z, phys_delta_z, return_components: bool = False,
                darcy_ctx: dict | None = None, fut_feats: torch.Tensor | None = None):
        """x [B,N,L,F]；h_last_z [B,N]；phys_delta_z [B,N,P] = h_phys_z − h_last_z
        （no_physflux 时由调用方传全零）。默认返回 h_pred_z [B,N,P]；
        return_components=True 时同时返回 gate/offset/base，供 residual-only PDE 构造修正量。
        darcy_ctx 提供达西注意力所需的米制状态与物理参数：h_m/h_anom_m [B,N]、K/z_bot [N]。"""
        if self.head_mode == "lean":
            return self._forward_lean(x, static_feats, h_last_z, phys_delta_z,
                                      return_components, darcy_ctx)
        B, N, L, _ = x.shape
        if self.use_leddam:      # 可学习因果趋势分解：h 通道 → [趋势, 扰动] 附加通道
            k = torch.softmax(self.dec_kernel, dim=0).view(1, 1, -1)
            hch = x[..., 0].reshape(B * N, 1, L)
            trend = F_nn.conv1d(F_nn.pad(hch, (k.shape[-1] - 1, 0), mode="replicate"), k)
            trend = trend.reshape(B, N, L)
            x = torch.cat([x, trend.unsqueeze(-1), (x[..., 0] - trend).unsqueeze(-1)], dim=-1)
        z = self.in_proj(x)
        z = z.permute(0, 1, 3, 2).reshape(B * N, -1, L)
        z = self.tcn(z)[..., -1].reshape(B, N, -1)

        A_norm = self._graph_adj()                        # 可学习邻接（主路径）
        s = A_norm @ z
        for lin in self.gc:
            s = F_nn.silu(lin(A_norm @ s)) + s

        darcy_diag = None
        if self.use_darcy_attn and darcy_ctx is not None:
            out = self.darcy(s, darcy_ctx["h_anom_m"], darcy_ctx["K"], darcy_ctx["z_bot"],
                             darcy_ctx["h_m"], self.da_src, self.da_dst, self.da_w, self.da_d,
                             return_diag=return_components, use_content=self.darcy_content)
            s, darcy_diag = out if return_components else (out, None)

        st = self.static_proj(static_feats).unsqueeze(0).expand(B, -1, -1)
        if self.use_star:        # 掩码全局核心：全网聚合 → 残差门控分发
            core = self.star_enc(s).mean(dim=1, keepdim=True).expand(-1, N, -1)  # [B,N,h]
            g_star = torch.sigmoid(self.star_gate(torch.cat([s, core, st], dim=-1)))
            s = s + g_star * self.star_out(torch.cat([s, core], dim=-1))
        phys_feat = phys_delta_z.detach() if self.detach_phys_features else phys_delta_z
        parts = [z, s, st, phys_feat]
        if self.n_fut:
            parts.append(fut_feats if fut_feats is not None
                         else torch.zeros(B, N, self.n_fut, device=z.device, dtype=z.dtype))
        feat = self.trunk(torch.cat(parts, dim=-1))
        offset = self.offset_head(feat)                          # [B,N,P]
        if self.cumsum_head:                                     # dh_cumsum 消融：逐步增量累加（旧机制）
            offset = torch.cumsum(offset, dim=-1)
        gate = torch.sigmoid(self.gate_head(feat))               # (0,1)
        base = h_last_z.unsqueeze(-1) if self.use_anchor else torch.zeros_like(offset)
        pred = base + gate * phys_delta_z + offset
        if self.use_shrink and self.shrink_feats is not None:
            pred = base + self.shrink_lambda().unsqueeze(0) * (pred - base)
        if return_components:
            return {"prediction": pred, "base_z": base, "gate": gate, "offset_z": offset,
                    "darcy_diag": darcy_diag}
        return pred

"""PI-STGCN forecast backbone described in the manuscript.

The released model has one executable forecast path:

1. bounded hydrogeological parameters from static well descriptors;
2. a node-shared MLP over multi-scale increments, forcing and physical rollout;
3. DarcyAttention redistribution on the fixed physical graph;
4. anchored parallel multi-horizon fusion and validation-only shrinkage.

Legacy TCN, generic graph-convolution and content-attention branches were
removed because they were not used by the manuscript's final model.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .physics import DarcyAttention, EdgeConductance, ParamNet, RecessionTimeScale

N_SHRINK_FEATS = 6


class PISTGCN(nn.Module):
    """Physics-informed, anchored six-horizon groundwater forecaster."""

    def __init__(
        self,
        cfg: dict,
        n_static: int,
        graph,
        ablation: dict | None = None,
    ):
        super().__init__()
        ablation = ablation or {}
        model_cfg = cfg["model"]
        physics_cfg = cfg["physics"]

        self.pred_len = int(cfg["data"]["pred_len"])
        self.n_nodes = int(graph.cell_area.shape[0])
        self.thickness_prior = float(physics_cfg["aquifer_thickness_prior_m"])

        # The retained switches correspond to the single-factor ablations
        # reported in the manuscript.
        self.use_phys_flux = bool(ablation.get("use_phys_flux", True))
        self.signed_flux = bool(ablation.get("signed_flux", True))
        self.use_paramnet = bool(ablation.get("use_paramnet", True))
        self.use_anchor = bool(ablation.get("use_anchor", True))
        self.use_recharge_gate = bool(ablation.get("use_gate", True))
        self.use_darcy_attn = bool(
            ablation.get("use_darcy_attn", model_cfg["use_darcy_attn"])
        )
        self.derive_tau = bool(ablation.get("derive_tau", model_cfg["derive_tau"]))
        self.derive_tau_r = bool(model_cfg.get("derive_tau_r", True))

        self.register_buffer("ei", torch.as_tensor(graph.phys_i), persistent=False)
        self.register_buffer("ej", torch.as_tensor(graph.phys_j), persistent=False)
        self.register_buffer("e_w", torch.as_tensor(graph.phys_width), persistent=False)
        self.register_buffer("e_d", torch.as_tensor(graph.phys_dist), persistent=False)
        self.register_buffer("area", torch.as_tensor(graph.cell_area), persistent=False)
        self.register_buffer(
            "da_src", torch.cat([torch.as_tensor(graph.phys_j), torch.as_tensor(graph.phys_i)]),
            persistent=False,
        )
        self.register_buffer(
            "da_dst", torch.cat([torch.as_tensor(graph.phys_i), torch.as_tensor(graph.phys_j)]),
            persistent=False,
        )
        self.register_buffer(
            "da_w", torch.as_tensor(graph.phys_width).repeat(2), persistent=False
        )
        self.register_buffer(
            "da_d", torch.as_tensor(graph.phys_dist).repeat(2), persistent=False
        )

        self.param_net = ParamNet(
            n_static,
            hidden=32,
            k_range=physics_cfg["k_range_m_per_day"],
            sy_range=physics_cfg["sy_range"],
            gamma_range=physics_cfg["gamma_range"],
            tau_b_range=physics_cfg["tau_baseflow_days"],
            k_prior=float(physics_cfg["k_const"]),
            sy_prior=float(physics_cfg["sy_const"]),
            gamma_prior=float(physics_cfg["gamma_const"]),
            tau_b_prior=float(physics_cfg["tau_baseflow_const"]),
        )
        self.register_buffer("const_K", torch.tensor(float(physics_cfg["k_const"])))
        self.register_buffer("const_Sy", torch.tensor(float(physics_cfg["sy_const"])))
        self.register_buffer("const_gamma", torch.tensor(float(physics_cfg["gamma_const"])))
        self.register_buffer(
            "const_tau_r", torch.tensor(float(physics_cfg["tau_recharge_const"]))
        )
        self.register_buffer(
            "const_tau_b", torch.tensor(float(physics_cfg["tau_baseflow_const"]))
        )

        self.cond = EdgeConductance(physics_cfg["min_sat_thickness_m"])
        self.recession_scale = RecessionTimeScale(
            physics_cfg["tau_baseflow_days"],
            tau_prior=float(physics_cfg["tau_baseflow_const"]),
            k_prior=float(physics_cfg["k_const"]),
            sy_prior=float(physics_cfg["sy_const"]),
            b_prior=self.thickness_prior,
        )

        self.tau_r_lo, self.tau_r_hi = (
            float(v) for v in physics_cfg["tau_recharge_days"]
        )
        self.log_c_r = nn.Parameter(torch.zeros(()))
        self.recharge_gate = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 1)
        )
        nn.init.zeros_(self.recharge_gate[-1].weight)
        nn.init.zeros_(self.recharge_gate[-1].bias)

        self.darcy = DarcyAttention(
            min_b=float(physics_cfg["min_sat_thickness_m"]),
            learnable_exponents=bool(ablation.get("darcy_learnable_exponents", True)),
            dynamic=bool(ablation.get("darcy_dynamic", True)),
            signed=bool(ablation.get("darcy_signed", True)),
            uniform=bool(ablation.get("darcy_uniform", False)),
            magnitude=bool(ablation.get("darcy_magnitude", True)),
        )

        n_temporal = 11
        hidden = int(model_cfg.get("response_hidden", 32))
        self.response_mlp = nn.Sequential(
            nn.Linear(n_temporal + n_static + self.pred_len, hidden),
            nn.SiLU(),
            nn.Dropout(float(model_cfg["dropout"])),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.response_head = nn.Linear(hidden, self.pred_len)
        self.physics_gate = nn.Linear(hidden, self.pred_len)
        nn.init.zeros_(self.response_head.weight)
        nn.init.zeros_(self.response_head.bias)
        nn.init.zeros_(self.physics_gate.bias)

        self.beta_iso_raw = nn.Parameter(torch.full((self.pred_len,), -4.0))
        self.beta_dir_raw = nn.Parameter(torch.zeros(self.pred_len))

        self.shrink_head = nn.Sequential(
            nn.Linear(N_SHRINK_FEATS + n_static, 16),
            nn.SiLU(),
            nn.Linear(16, self.pred_len),
        )
        nn.init.zeros_(self.shrink_head[-1].weight)
        nn.init.constant_(self.shrink_head[-1].bias, 4.0)
        self.use_shrink = False
        self.shrink_feats: torch.Tensor | None = None
        self.static_feats_buf: torch.Tensor | None = None

    def hydro_params(
        self,
        static_feats: torch.Tensor,
        dem: torch.Tensor,
        z_bot_ref: torch.Tensor | None,
        depth_ref: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return bounded K, Sy, gamma and the geometry/time-scale terms."""
        n = static_feats.shape[0]
        if self.use_paramnet:
            params = self.param_net(static_feats)
            K = params["K"]
            Sy = params["Sy"]
            gamma = params["gamma"]
            tau_b = params["tau_b"]
        else:
            K = self.const_K.expand(n)
            Sy = self.const_Sy.expand(n)
            gamma = self.const_gamma.expand(n)
            tau_b = self.const_tau_b.expand(n)

        z_bot = z_bot_ref if z_bot_ref is not None else dem - self.thickness_prior
        if self.derive_tau_r:
            tau_r = torch.clamp(
                torch.exp(self.log_c_r) * depth_ref, self.tau_r_lo, self.tau_r_hi
            )
        else:
            tau_r = self.const_tau_r.expand(n)
        return {
            "K": K,
            "Sy": Sy,
            "gamma": gamma,
            "z_bot": z_bot,
            "tau_r": tau_r,
            "tau_b": tau_b,
        }

    def recession(
        self, params: dict[str, torch.Tensor], h_m: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Derive the recession time scale from Sy and Darcy-weighted T."""
        K, Sy, z_bot = params["K"], params["Sy"], params["z_bot"]
        saturated = torch.clamp(h_m - z_bot, min=self.cond.min_b)
        if self.use_darcy_attn:
            T_bar = self.darcy.effective_T(
                K,
                z_bot,
                h_m,
                self.da_src,
                self.da_dst,
                self.da_w,
                self.da_d,
                self.n_nodes,
            )
        else:
            T_bar = K * saturated
        if not self.derive_tau:
            return params["tau_b"], T_bar
        return self.recession_scale.tau_days(T_bar, Sy), T_bar

    def shrink_lambda(self) -> torch.Tensor:
        if self.shrink_feats is None or self.static_feats_buf is None:
            raise RuntimeError("Shrinkage descriptors have not been initialized")
        features = torch.cat([self.shrink_feats, self.static_feats_buf], dim=-1)
        return torch.sigmoid(self.shrink_head(features))

    def forward(
        self,
        temporal_features: torch.Tensor,
        static_feats: torch.Tensor,
        h_last_z: torch.Tensor,
        phys_delta_z: torch.Tensor,
        darcy_ctx: dict[str, torch.Tensor] | None = None,
        return_components: bool = False,
    ):
        """Generate all forecast horizons in one anchored forward pass."""
        batch_size = temporal_features.shape[0]
        static = static_feats.unsqueeze(0).expand(batch_size, -1, -1)
        encoded = self.response_mlp(
            torch.cat([temporal_features, static, phys_delta_z], dim=-1)
        )
        response = self.response_head(encoded)
        gate = torch.sigmoid(self.physics_gate(encoded))

        darcy_diag = None
        if self.use_darcy_attn and darcy_ctx is not None:
            beta_iso = torch.sigmoid(self.beta_iso_raw)
            beta_dir = torch.tanh(self.beta_dir_raw)
            redistributed = self.darcy.smooth_response(
                response,
                darcy_ctx["h_anom_m"],
                darcy_ctx["K"],
                darcy_ctx["z_bot"],
                darcy_ctx["h_m"],
                self.da_src,
                self.da_dst,
                self.da_w,
                self.da_d,
                beta_iso,
                beta_dir,
                return_diag=return_components,
            )
            if return_components:
                response, darcy_diag = redistributed
            else:
                response = redistributed

        base = h_last_z.unsqueeze(-1) if self.use_anchor else torch.zeros_like(response)
        prediction = base + gate * phys_delta_z + response
        if self.use_shrink:
            prediction = base + self.shrink_lambda().unsqueeze(0) * (prediction - base)

        if return_components:
            return {
                "prediction": prediction,
                "base_z": base,
                "gate": gate,
                "offset_z": response,
                "darcy_diag": darcy_diag,
            }
        return prediction

"""Training, validation selection, shrinkage calibration and evaluation."""
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
from .model import PISTGCN
from .physics import CVFDResidual, FluxRollout, ParamNet, vertical_source_m_per_day


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed NumPy, Torch and CUDA; recordable deterministic mode is optional."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)


class Trainer:
    LEAN_LAGS = (1, 2, 3, 6, 12)

    def __init__(
        self,
        cfg: dict,
        bundle: DataBundle,
        features: dict,
        windows: dict[str, WindowSet],
        normalizer: PerWellNormalizer,
        graph,
        device: str,
        ablation: dict,
        out_dir: Path,
    ):
        self.cfg = cfg
        self.bundle = bundle
        self.windows = windows
        self.device = torch.device(device)
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = cfg["train"]
        physics_cfg = cfg["physics"]
        forcing_cfg = cfg["forcing"]
        self.w_h = float(train_cfg["w_h"])
        self.w_dh = float(train_cfg["w_dh"])
        self.lambda_pde = float(ablation.get("lambda_pde", train_cfg["lambda_pde"]))
        self.n_pde_win = int(train_cfg["pde_windows_per_batch"])
        self.dt_days = float(cfg["data"]["step_days"])
        self.future_forcing = bool(ablation.get("future_forcing", True))
        self.nn_forcing = bool(
            ablation.get("nn_input_forcing", train_cfg["nn_input_forcing"])
        )
        self.lean_lags = tuple(ablation.get("lean_lags", self.LEAN_LAGS))
        if len(self.lean_lags) != len(self.LEAN_LAGS):
            raise ValueError("lean_lags must keep five entries for a fair ablation")

        self.model = PISTGCN(
            cfg,
            n_static=features["static"].shape[-1],
            graph=graph,
            ablation=ablation,
        ).to(self.device)
        self.rollout = FluxRollout(
            forcing_cfg["et_extinction_depth_m"],
            self.dt_days,
            float(physics_cfg["rollout_dh_clip_m"]),
        )
        self.pde = CVFDResidual(
            forcing_cfg["et_extinction_depth_m"],
            self.dt_days,
            float(physics_cfg["pde_char_scale_m_per_day"]),
        )

        def tensor(values, dtype=torch.float32):
            return torch.as_tensor(np.asarray(values), dtype=dtype, device=self.device)

        self.static_t = tensor(features["static"])
        self.bundle_train_end = int(bundle.train_end_idx) + 1
        self.H_fill = tensor(bundle.H_fill)
        self.H_obs = tensor(np.nan_to_num(bundle.H_obs, nan=0.0))
        self.mask_t = tensor(bundle.mask)
        self.h_mean = tensor(normalizer.mean)
        self.h_std = tensor(normalizer.std)
        self.dem = tensor(bundle.dem)
        self.z_bot_ref_t = tensor(bundle.z_bot_ref) if bundle.z_bot_ref is not None else None
        self.depth_ref_t = torch.clamp(self.dem - self.h_mean, min=0.0)
        self.precip = tensor(bundle.precip)
        self.et = tensor(bundle.et)
        self.wu = tensor(bundle.wu)
        self.pde_mask = tensor(
            graph.interior_mask.astype(np.float32) * bundle.aquifer_onehot[:, 0]
        )

        train_slice = slice(0, bundle.train_end_idx + 1)
        self.p_bar = self.precip[train_slice].mean()
        self.et_bar = self.et[train_slice].mean()
        self.wu_bar = self.wu[train_slice].mean(0)
        self.d_ext = float(forcing_cfg["et_extinction_depth_m"])

        season = np.minimum((bundle.dates.dayofyear - 1) // 5, 72).to_numpy()
        self.soy = torch.as_tensor(season, dtype=torch.long, device=self.device)
        climatology_p = np.zeros(73, dtype=np.float32)
        climatology_e = np.zeros(73, dtype=np.float32)
        climatology_w = np.zeros((73, bundle.wu.shape[1]), dtype=np.float32)
        train_season = season[: bundle.train_end_idx + 1]
        for season_idx in range(73):
            selected = train_season == season_idx
            if selected.any():
                climatology_p[season_idx] = bundle.precip[train_slice][selected].mean()
                climatology_e[season_idx] = bundle.et[train_slice][selected].mean()
                climatology_w[season_idx] = bundle.wu[train_slice][selected].mean(0)
        self.clim_p = tensor(climatology_p)
        self.clim_e = tensor(climatology_e)
        self.clim_w = tensor(climatology_w)

        self.horizon_w = None
        if bool(train_cfg["horizon_balance"]):
            train_windows = windows["train"]
            t_last = train_windows.x_idx[:, -1]
            h_z = (bundle.H_fill - normalizer.mean) / normalizer.std
            variance = []
            for horizon in range(int(cfg["data"]["pred_len"])):
                offset = h_z[train_windows.y_idx[:, horizon]] - h_z[t_last]
                observed = bundle.mask[train_windows.y_idx[:, horizon]] > 0
                variance.append(float(np.var(offset[observed])) + 1e-4)
            weights = 1.0 / np.asarray(variance)
            self.horizon_w = tensor(weights / weights.mean())

        self.win_year = bundle.dates.year.to_numpy()
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        self.select_metric = str(train_cfg["select_metric"])
        self.sel_smooth_w = int(train_cfg["select_smooth_w"])
        self._pde_rng = np.random.default_rng(int(train_cfg["seed"]) + 7)

    def _temporal_features(self, x_idx, y_idx, t_last) -> torch.Tensor:
        """Five level increments, four forcing terms and two seasonal terms."""
        h_z = (self.H_fill - self.h_mean) / self.h_std
        current = h_z[t_last]
        increments = [
            current - h_z[torch.clamp(t_last - lag, min=0)] for lag in self.lean_lags
        ]
        input_idx = torch.as_tensor(x_idx, device=self.device)
        batch_size, n_nodes = current.shape

        if self.future_forcing:
            p_future = (
                self.precip[y_idx].mean(1) - self.p_bar
            ) / (self.p_bar + 1e-8)
            e_future = (
                self.et[y_idx].mean(1) - self.et_bar
            ) / (self.et_bar + 1e-8)
            w_future = (
                self.wu[y_idx].mean(1) - self.wu_bar
            ) / (self.wu_bar.abs().mean() + 1e-8)
        else:
            season_y = self.soy[y_idx]
            p_future = (
                self.clim_p[season_y].mean(1) - self.p_bar
            ) / (self.p_bar + 1e-8)
            e_future = (
                self.clim_e[season_y].mean(1) - self.et_bar
            ) / (self.et_bar + 1e-8)
            w_future = (
                self.clim_w[season_y].mean(1) - self.wu_bar
            ) / (self.wu_bar.abs().mean() + 1e-8)

        p_history = (
            self.precip[input_idx].mean(1) - self.p_bar
        ) / (self.p_bar + 1e-8)
        angle = 2.0 * math.pi * self.soy[t_last].to(current.dtype) / 73.0

        def broadcast(values):
            return values.view(batch_size, 1).expand(batch_size, n_nodes)

        if self.nn_forcing:
            forcing = [
                broadcast(p_history),
                broadcast(p_future),
                broadcast(e_future),
                w_future,
            ]
        else:
            forcing = [torch.zeros_like(current) for _ in range(4)]
        columns = increments + forcing + [broadcast(torch.sin(angle)), broadcast(torch.cos(angle))]
        return torch.stack(columns, dim=-1)

    def _forward_windows(self, x_idx: np.ndarray, y_idx: np.ndarray) -> dict:
        t_last = torch.as_tensor(x_idx[:, -1], device=self.device)
        target_idx = torch.as_tensor(y_idx, device=self.device)
        h_last_m = self.H_fill[t_last]
        h_last_z = (h_last_m - self.h_mean) / self.h_std

        params = self.model.hydro_params(
            self.static_t, self.dem, self.z_bot_ref_t, self.depth_ref_t
        )
        K, Sy = params["K"], params["Sy"]
        gamma, z_bot = params["gamma"], params["z_bot"]
        tau_b, T_bar = self.model.recession(params, h_last_m)

        if self.model.use_phys_flux:
            if self.future_forcing:
                p_future = self.precip[target_idx]
                e_future = self.et[target_idx]
                w_future = self.wu[target_idx]
            else:
                season_y = self.soy[target_idx]
                p_future = self.clim_p[season_y]
                e_future = self.clim_e[season_y]
                w_future = self.clim_w[season_y]
            input_idx = torch.as_tensor(x_idx, device=self.device)
            precip_init = self.precip[input_idx].mean(dim=1)
            if self.model.use_recharge_gate:
                anomaly = (
                    (precip_init - self.p_bar) / (self.p_bar + 1e-6)
                ).unsqueeze(-1)
                multiplier = 2.0 * torch.sigmoid(self.model.recharge_gate(anomaly))
                gamma_effective = gamma.unsqueeze(0) * multiplier
            else:
                gamma_effective = gamma.unsqueeze(0).expand(len(x_idx), -1)
            h_phys_m = self.rollout(
                h_last_m,
                self.h_mean,
                K,
                Sy,
                gamma_effective,
                z_bot,
                self.dem,
                self.model.area,
                self.model.ei,
                self.model.ej,
                self.model.e_w,
                self.model.e_d,
                p_future,
                e_future,
                w_future,
                self.model.cond,
                signed=self.model.signed_flux,
                tau_r=params["tau_r"],
                tau_b=tau_b,
                precip_init=precip_init,
            )
            phys_delta_z = (
                (h_phys_m - self.h_mean.unsqueeze(-1)) / self.h_std.unsqueeze(-1)
                - h_last_z.unsqueeze(-1)
            )
        else:
            phys_delta_z = torch.zeros(
                *h_last_z.shape, self.model.pred_len, device=self.device
            )

        darcy_ctx = {
            "h_m": h_last_m,
            "h_anom_m": h_last_m - self.h_mean,
            "K": K,
            "z_bot": z_bot,
        }
        model_parts = self.model(
            self._temporal_features(x_idx, target_idx, t_last),
            self.static_t,
            h_last_z,
            phys_delta_z,
            darcy_ctx=darcy_ctx,
            return_components=True,
        )
        h_pred_z = model_parts["prediction"]
        h_true_m = self.H_obs[target_idx].permute(0, 2, 1)
        y_mask = self.mask_t[target_idx].permute(0, 2, 1)
        h_true_z = (
            h_true_m - self.h_mean.unsqueeze(-1)
        ) / self.h_std.unsqueeze(-1)
        h_pred_m = h_pred_z * self.h_std.unsqueeze(-1) + self.h_mean.unsqueeze(-1)
        correction_z = model_parts["offset_z"] + (
            model_parts["gate"] - 1.0
        ) * phys_delta_z.detach()
        return {
            "h_pred_z": h_pred_z,
            "h_true_z": h_true_z,
            "h_pred_m": h_pred_m,
            "h_true_m": h_true_m,
            "h_last_m": h_last_m,
            "h_last_z": h_last_z,
            "y_mask": y_mask,
            "K": K,
            "Sy": Sy,
            "gamma": gamma,
            "z_bot": z_bot,
            "tau_b": tau_b,
            "T_bar": T_bar,
            "phys_delta_z": phys_delta_z,
            "correction_z": correction_z,
            "gate": model_parts["gate"],
            "offset_z": model_parts["offset_z"],
            "darcy_diag": model_parts["darcy_diag"],
        }

    def _shrink_feats(self) -> torch.Tensor:
        """Training-only station descriptors for reliability shrinkage."""
        h = self.H_fill[: self.bundle_train_end]
        mask = self.mask_t[: self.bundle_train_end]

        def safe_log(values):
            return torch.log(values.clamp(min=1e-3))

        columns = [safe_log(h.std(dim=0))]
        for lag in (1, 3, 6):
            columns.append(safe_log((h[lag:] - h[:-lag]).std(dim=0)))
        columns.append(mask.mean(dim=0))
        neighbour_scale = torch.zeros_like(columns[0]).index_add_(
            0, self.model.da_dst, h.std(dim=0)[self.model.da_src]
        )
        degree = torch.zeros_like(columns[0]).index_add_(
            0,
            self.model.da_dst,
            torch.ones_like(self.model.da_src, dtype=h.dtype),
        )
        columns.append(safe_log(neighbour_scale / degree.clamp(min=1)))
        features = torch.stack(columns, dim=-1)
        return (features - features.mean(0, keepdim=True)) / (
            features.std(0, keepdim=True) + 1e-6
        )

    def enable_shrink_transfer(self) -> None:
        """Recompute training-only descriptors and activate loaded shrink weights."""
        self.model.shrink_feats = self._shrink_feats().detach()
        self.model.static_feats_buf = self.static_t.detach()
        self.model.use_shrink = True

    def fit_shrink(self, epochs: int = 300, lr: float = 0.05) -> dict:
        """Fit the parameterized station/horizon shrinkage head on validation only."""
        self.enable_shrink_transfer()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.model.shrink_head.parameters():
            parameter.requires_grad_(True)

        self.model.use_shrink = False
        windows = self.windows["val"]
        with torch.no_grad():
            output = self._forward_windows(windows.x_idx, windows.y_idx)
            raw = output["h_pred_z"].detach()
            base = (
                output["h_last_z"].unsqueeze(-1).detach()
                if self.model.use_anchor
                else torch.zeros_like(raw)
            )
            true_z = output["h_true_z"].detach()
            mask = output["y_mask"].detach()
        correction, target = raw - base, true_z - base
        count = mask.sum(dim=(0, 2))
        mean = (true_z * mask).sum(dim=(0, 2)) / count.clamp(min=1)
        variance = (
            ((true_z - mean.view(1, -1, 1)) ** 2 * mask).sum(dim=(0, 2))
            / count.clamp(min=1)
        )
        selected = count >= 5
        normalizer = torch.where(
            selected, 1.0 / variance.clamp(min=1e-4), torch.zeros_like(variance)
        )

        def validation_loss(shrink):
            squared_error = ((shrink * correction - target) ** 2 * mask).sum(
                dim=(0, 2)
            )
            return ((squared_error / count.clamp(min=1)) * normalizer)[selected].mean()

        optimizer = torch.optim.Adam(self.model.shrink_head.parameters(), lr=lr)
        base_loss = float(
            validation_loss(torch.ones(1, 1, 1, device=raw.device)).detach()
        )
        loss = None
        for _ in range(epochs):
            optimizer.zero_grad()
            shrink = self.model.shrink_lambda().unsqueeze(0)
            loss = validation_loss(shrink)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            shrink = self.model.shrink_lambda()
            fitted_loss = float(loss.detach())
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        self.model.use_shrink = True
        return {
            "shrink_val_loss_before": base_loss,
            "shrink_val_loss_after": fitted_loss,
            "shrink_lambda_median": float(shrink.median()),
            "shrink_lambda_p10": float(shrink.quantile(0.10)),
            "shrink_lambda_p90": float(shrink.quantile(0.90)),
            "shrink_frac_below_half": float((shrink < 0.5).float().mean()),
        }

    @staticmethod
    def _masked_mse(a, b, mask):
        return (((a - b) ** 2) * mask).sum() / (mask.sum() + 1e-8)

    def _loss(self, batch_x, batch_y):
        output = self._forward_windows(batch_x, batch_y)
        mask = output["y_mask"]
        if self.horizon_w is not None:
            error = ((output["h_pred_z"] - output["h_true_z"]) ** 2) * mask
            per_horizon = error.sum(dim=(0, 1)) / (mask.sum(dim=(0, 1)) + 1e-8)
            loss_h = (per_horizon * self.horizon_w).mean()
        else:
            loss_h = self._masked_mse(
                output["h_pred_z"], output["h_true_z"], mask
            )

        anchor = output["h_last_z"].unsqueeze(-1)
        delta_pred = torch.diff(
            torch.cat([anchor, output["h_pred_z"]], dim=-1), dim=-1
        )
        delta_true = torch.diff(
            torch.cat([anchor, output["h_true_z"]], dim=-1), dim=-1
        )
        previous_mask = torch.cat(
            [torch.ones_like(mask[..., :1]), mask[..., :-1]], dim=-1
        )
        loss_dh = self._masked_mse(
            delta_pred, delta_true, mask * previous_mask
        )

        loss = self.w_h * loss_h + self.w_dh * loss_dh
        components = {
            "h": float(loss_h.detach()),
            "dh": float(loss_dh.detach()),
            "pde": 0.0,
            "pde_weighted": 0.0,
            "smooth": 0.0,
            "smooth_weighted": 0.0,
            "phys_delta_abs_z": float(output["phys_delta_z"].detach().abs().mean()),
            "correction_abs_z": float(output["correction_z"].detach().abs().mean()),
            "gate_mean": float(output["gate"].detach().mean()),
        }

        if self.lambda_pde > 0:
            w_bar = vertical_source_m_per_day(
                self.h_mean,
                self.dem,
                self.p_bar,
                self.et_bar,
                self.wu_bar,
                self.model.area,
                output["gamma"],
                self.d_ext,
            )
            terms = []
            selected = self._pde_rng.choice(
                len(batch_x),
                size=min(self.n_pde_win, len(batch_x)),
                replace=False,
            )
            for batch_idx in selected:
                indices = np.concatenate(
                    [[batch_x[batch_idx, -1]], batch_y[batch_idx]]
                )
                forcing_idx = torch.as_tensor(indices, device=self.device)
                sequence_m = torch.cat(
                    [
                        output["h_last_m"][batch_idx : batch_idx + 1],
                        output["h_pred_m"][batch_idx].permute(1, 0),
                    ],
                    dim=0,
                )
                terms.append(
                    self.pde(
                        sequence_m,
                        self.h_mean,
                        w_bar,
                        output["K"],
                        output["Sy"],
                        output["gamma"],
                        output["z_bot"],
                        self.dem,
                        self.model.area,
                        self.pde_mask,
                        self.model.ei,
                        self.model.ej,
                        self.model.e_w,
                        self.model.e_d,
                        self.precip[forcing_idx],
                        self.et[forcing_idx],
                        self.wu[forcing_idx],
                        self.model.cond,
                        signed=self.model.signed_flux,
                    )
                )
            pde_loss = torch.stack(terms).mean()
            loss = loss + self.lambda_pde * pde_loss
            components["pde"] = float(pde_loss.detach())
            components["pde_weighted"] = self.lambda_pde * float(pde_loss.detach())

        if self.model.use_paramnet:
            smooth_weight = float(self.cfg["physics"]["smooth_reg_weight"])
            smooth_loss = ParamNet.smoothness_penalty(
                torch.log(output["K"]),
                self.model.ei,
                self.model.ej,
                self.model.e_d,
            ) + ParamNet.smoothness_penalty(
                output["Sy"], self.model.ei, self.model.ej, self.model.e_d
            )
            loss = loss + smooth_weight * smooth_loss
            components["smooth"] = float(smooth_loss.detach())
            components["smooth_weighted"] = smooth_weight * float(
                smooth_loss.detach()
            )
        return loss, components

    @staticmethod
    def _epoch_log_fields(aggregate: dict, n_windows: int) -> dict:
        loss_keys = {"h", "dh", "pde", "pde_weighted", "smooth", "smooth_weighted", "total"}
        return {
            f"{'loss' if key in loss_keys else 'diag'}_{key}": value / n_windows
            for key, value in aggregate.items()
        }

    def train(self, epochs: int | None = None) -> float:
        train_cfg = self.cfg["train"]
        epochs = epochs or int(train_cfg["epochs"])
        batch_size = int(train_cfg["batch_windows"])
        patience = int(train_cfg["patience"])
        windows = self.windows["train"]
        n_windows = len(windows.x_idx)
        best_score = float("inf")
        best_rmse = float("inf")
        best_epoch = None
        bad_epochs = 0
        checkpoint_saved = False
        history = []
        ring = collections.deque(maxlen=2 * self.sel_smooth_w + 1)
        rng = np.random.default_rng(int(train_cfg["seed"]))
        started = time.time()

        for epoch in range(1, epochs + 1):
            self.model.train()
            order = rng.permutation(n_windows)
            aggregate = {"total": 0.0}
            for start in range(0, n_windows, batch_size):
                batch = order[start : start + batch_size]
                self.opt.zero_grad()
                loss, components = self._loss(
                    windows.x_idx[batch], windows.y_idx[batch]
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(train_cfg["grad_clip"])
                )
                self.opt.step()
                for key, value in components.items():
                    aggregate[key] = aggregate.get(key, 0.0) + value * len(batch)
                aggregate["total"] += float(loss.detach()) * len(batch)

            validation = self.evaluate("val")["overall"]
            score = (
                -validation["skill_well_median"]
                if self.select_metric == "skill"
                else validation["rmse_well_median"]
            )
            history.append(
                {
                    "epoch": epoch,
                    "val_rmse_well_median": validation["rmse_well_median"],
                    "val_rmse_m_pooled": validation["rmse_m_pooled"],
                    "val_skill_well_median": validation["skill_well_median"],
                    "val_nse_well_median": validation["nse_well_median"],
                    **self._epoch_log_fields(aggregate, n_windows),
                }
            )

            state = {
                key: value.detach().clone().cpu()
                for key, value in self.model.state_dict().items()
            }
            if self.sel_smooth_w > 0:
                ring.append(
                    (epoch, score, validation["rmse_well_median"], state)
                )
                if len(ring) < ring.maxlen:
                    continue
                smoothed_score = sum(item[1] for item in ring) / len(ring)
                center_epoch, _, center_rmse, center_state = ring[self.sel_smooth_w]
                if smoothed_score < best_score - 1e-6:
                    best_score = smoothed_score
                    best_rmse = center_rmse
                    best_epoch = center_epoch
                    bad_epochs = 0
                    torch.save(center_state, self.out_dir / "best_model.pt")
                    checkpoint_saved = True
                else:
                    bad_epochs += 1
            elif score < best_score - 1e-6:
                best_score = score
                best_rmse = validation["rmse_well_median"]
                best_epoch = epoch
                bad_epochs = 0
                torch.save(state, self.out_dir / "best_model.pt")
                checkpoint_saved = True
            else:
                bad_epochs += 1

            if bad_epochs >= patience:
                break

        checkpoint = self.out_dir / "best_model.pt"
        if not checkpoint_saved:
            torch.save(self.model.state_dict(), checkpoint)
            validation = self.evaluate("val")["overall"]
            best_rmse = validation["rmse_well_median"]
            best_score = (
                -validation["skill_well_median"]
                if self.select_metric == "skill"
                else best_rmse
            )
            best_epoch = len(history)

        json.dump(
            {
                "history": history,
                "best_val_rmse_well_median": best_rmse,
                "select_metric": self.select_metric,
                "best_select_score": best_score,
                "best_epoch": best_epoch,
                "select_smooth_w": self.sel_smooth_w,
                "wall_time_s": time.time() - started,
                "epochs_run": len(history),
            },
            open(self.out_dir / "train_log.json", "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=1,
        )
        self.model.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
        return float(best_rmse)

    @torch.no_grad()
    def evaluate(self, split: str, return_arrays: bool = False):
        self.model.eval()
        windows = self.windows[split]
        predictions, truths, persistence, masks = [], [], [], []
        for start in range(0, len(windows.x_idx), 16):
            output = self._forward_windows(
                windows.x_idx[start : start + 16],
                windows.y_idx[start : start + 16],
            )
            predictions.append(output["h_pred_m"].cpu().numpy())
            truths.append(output["h_true_m"].cpu().numpy())
            masks.append(output["y_mask"].cpu().numpy())
            persistence.append(
                np.repeat(
                    output["h_last_m"].cpu().numpy()[:, :, None],
                    output["h_true_m"].shape[-1],
                    axis=2,
                )
            )
        prediction = np.concatenate(predictions)
        truth = np.concatenate(truths)
        baseline = np.concatenate(persistence)
        mask = np.concatenate(masks)
        metrics = evaluate_meters(truth, prediction, baseline, mask)
        if not return_arrays:
            return metrics
        years = self.win_year[windows.y_idx[:, 0]]
        arrays = {
            "pred": prediction.astype(np.float32),
            "true": truth.astype(np.float32),
            "persist": baseline.astype(np.float32),
            "mask": mask.astype(np.uint8),
            "window_year": years.astype(np.int16),
        }
        return metrics, arrays

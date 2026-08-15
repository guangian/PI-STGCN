"""Train, select, evaluate and export the PI-STGCN release artifacts."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from src.data_pipeline import (  # noqa: E402
    PerWellNormalizer,
    build_windows,
    load_bundle,
    make_static_features,
    set_data_root,
)
from src.evaluation import per_well_records, write_rmse_nse_tables  # noqa: E402
from src.graphs import build_physical_graph  # noqa: E402
from src.trainer import Trainer, set_seed  # noqa: E402


ABLATIONS = {
    "none": {},
    "no_anchor": {"use_anchor": False},
    "no_shrinkage": {"disable_shrink": True},
    "no_cvfd": {"lambda_pde": 0.0},
    "free_recession": {"derive_tau": False},
    "no_recharge_gate": {"use_gate": False},
    "no_darcy": {"use_darcy_attn": False},
    "static_darcy": {"darcy_dynamic": False},
    "no_direction": {"darcy_signed": False},
    "no_future_forcing": {"future_forcing": False},
    "no_nn_forcing": {"nn_input_forcing": False},
    "single_scale": {"lean_lags": (1, 1, 1, 1, 1)},
}


def run(cfg: dict, args: argparse.Namespace) -> dict:
    set_data_root(cfg, args.data_root)
    ablation = dict(ABLATIONS[args.ablation])
    experiment = "full" if args.ablation == "none" else f"ablation_{args.ablation}"
    if args.seed is not None:
        cfg["train"]["seed"] = int(args.seed)
        experiment = f"{experiment}_seed{args.seed}"
    if args.smoke:
        experiment = f"{experiment}_smoke"

    output_dir = (CODE_DIR / cfg["paths"]["results_root"]).resolve() / experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg["train"]["seed"]))

    bundle = load_bundle(cfg, CODE_DIR)
    with open(output_dir / "qc_report.json", "w", encoding="utf-8") as handle:
        json.dump(bundle.qc_report, handle, ensure_ascii=False, indent=1)

    graph = build_physical_graph(
        bundle.coords,
        bundle.aquifer_onehot,
        float(cfg["graph"]["min_edge_dist_m"]),
        tuple(cfg["graph"]["voronoi_area_clip_q"]),
        max_wd_ratio=float(cfg["graph"]["max_wd_ratio"]),
    )
    normalizer = PerWellNormalizer.fit(
        bundle.H_obs, bundle.mask, bundle.train_end_idx
    )
    normalizer.save(output_dir / "normalizer.json")
    features = make_static_features(bundle, normalizer)
    windows = build_windows(
        bundle.H_fill.shape[0],
        int(cfg["data"]["input_len"]),
        int(cfg["data"]["pred_len"]),
        bundle.train_end_idx,
        bundle.val_end_idx,
    )

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    trainer = Trainer(
        cfg,
        bundle,
        features,
        windows,
        normalizer,
        graph,
        device,
        ablation,
        output_dir,
    )
    n_params = sum(
        parameter.numel()
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    )
    metadata = {
        "experiment": experiment,
        "ablation": args.ablation,
        "device": str(device),
        "n_wells": len(bundle.well_ids),
        "n_params": int(n_params),
        "input_steps": int(cfg["data"]["input_len"]),
        "forecast_steps": int(cfg["data"]["pred_len"]),
        "n_windows": {key: int(len(value.x_idx)) for key, value in windows.items()},
        "graph_stats": graph.stats,
        "n_pde_nodes": int(trainer.pde_mask.sum().item()),
        "torch": torch.__version__,
        "seed": int(cfg["train"]["seed"]),
        "known_future_forcing": trainer.future_forcing,
        "primary_metric": "station-median RMSE/NSE",
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "run_meta.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=1)
    print(
        f"[{experiment}] wells={metadata['n_wells']} params={n_params} "
        f"windows={metadata['n_windows']} device={device}",
        flush=True,
    )

    epochs = 3 if args.smoke else args.epochs
    best_validation_rmse = trainer.train(epochs=epochs)

    use_shrinkage = bool(cfg["train"]["use_shrink"]) and not bool(
        ablation.get("disable_shrink", False)
    )
    if use_shrinkage:
        metadata["shrinkage"] = trainer.fit_shrink()
        shrinkage = metadata["shrinkage"]
        print(
            f"[{experiment}] shrink lambda_med={shrinkage['shrink_lambda_median']:.3f} "
            f"lambda<0.5={shrinkage['shrink_frac_below_half']:.1%} "
            f"val_loss {shrinkage['shrink_val_loss_before']:.5f}"
            f"->{shrinkage['shrink_val_loss_after']:.5f}",
            flush=True,
        )
        torch.save(trainer.model.state_dict(), output_dir / "best_model.pt")

    validation_metrics = trainer.evaluate("val")
    with open(output_dir / "metrics_val.json", "w", encoding="utf-8") as handle:
        json.dump(validation_metrics, handle, ensure_ascii=False, indent=1)

    test_metrics, arrays = trainer.evaluate("test", return_arrays=True)
    with open(output_dir / "metrics_test.json", "w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, ensure_ascii=False, indent=1)
    rows = per_well_records(
        arrays["true"],
        arrays["pred"],
        arrays["persist"],
        arrays["mask"],
        bundle.well_ids,
    )
    write_rmse_nse_tables(
        rows,
        output_dir / "per_well_RMSE_NSE.csv",
        output_dir / "per_well_RMSE_NSE_pivot.csv",
    )
    if args.save_arrays:
        np.savez_compressed(output_dir / "test_arrays.npz", **arrays)

    metadata["completed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(output_dir / "run_meta.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=1)
    overall = test_metrics["overall"]
    print(
        f"[{experiment}] DONE val_rmse_med={best_validation_rmse:.4f} "
        f"test_rmse_med={overall['rmse_well_median']:.4f} "
        f"test_nse_med={overall['nse_well_median']:.4f}",
        flush=True,
    )
    return test_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PI-STGCN six-step groundwater-level forecasting"
    )
    parser.add_argument("--config", type=Path, default=CODE_DIR / "config.yaml")
    parser.add_argument("--data-root", help="Private data directory; never copied")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--ablation", choices=tuple(ABLATIONS), default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run(cfg, args)


if __name__ == "__main__":
    main()

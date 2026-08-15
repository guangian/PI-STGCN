"""Generate one observed-vs-predicted time-series plot for every well.

The checkpoint produces overlapping T+1..T+6 forecasts. For a single
continuous prediction curve, all forecasts targeting the same date are
averaged. Train, validation, and test periods share one axis and are separated
by dashed vertical lines.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from src.data_pipeline import (  # noqa: E402
    PerWellNormalizer,
    build_windows,
    load_bundle,
    make_static_features,
    set_data_root,
)
from src.graphs import build_physical_graph  # noqa: E402
from src.trainer import Trainer, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-well observed/predicted train-val-test plots."
    )
    parser.add_argument("--config", type=Path, default=CODE_DIR / "config.yaml")
    parser.add_argument("--data-root", help="Private data directory; never copied")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CODE_DIR / "results" / "full" / "best_model.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CODE_DIR / "results" / "full" / "well_prediction_plots",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of wells for a quick visual smoke test.",
    )
    return parser.parse_args()


def rebuild_trainer(cfg: dict, device: str, checkpoint: Path):
    set_seed(int(cfg["train"]["seed"]))
    bundle = load_bundle(cfg, CODE_DIR)
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
    features = make_static_features(bundle, normalizer)
    windows = build_windows(
        bundle.H_fill.shape[0],
        int(cfg["data"]["input_len"]),
        int(cfg["data"]["pred_len"]),
        bundle.train_end_idx,
        bundle.val_end_idx,
    )
    trainer = Trainer(
        cfg,
        bundle,
        features,
        windows,
        normalizer,
        graph,
        device,
        {},
        checkpoint.parent,
    )
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    trainer.model.load_state_dict(state, strict=True)
    if bool(cfg["train"]["use_shrink"]):
        trainer.enable_shrink_transfer()
    trainer.model.eval()
    return bundle, windows, trainer


def aggregate_predictions(bundle, windows, trainer) -> tuple[np.ndarray, dict]:
    """Average all T+1..T+6 predictions that target each date."""
    n_dates, n_wells = bundle.H_obs.shape
    pred_sum = np.zeros((n_dates, n_wells), dtype=np.float64)
    pred_count = np.zeros(n_dates, dtype=np.int16)
    split_summary = {}

    for split in ("train", "val", "test"):
        _, arrays = trainer.evaluate(split, return_arrays=True)
        pred = arrays["pred"]
        target_indices = windows[split].y_idx
        if pred.shape[:1] + pred.shape[2:] != (
            target_indices.shape[0],
            target_indices.shape[1],
        ):
            raise RuntimeError(
                f"Prediction/window mismatch for {split}: "
                f"{pred.shape} vs {target_indices.shape}"
            )
        for horizon_idx in range(target_indices.shape[1]):
            date_indices = target_indices[:, horizon_idx]
            np.add.at(pred_sum, date_indices, pred[:, :, horizon_idx])
            np.add.at(pred_count, date_indices, 1)
        split_summary[split] = {
            "windows": int(pred.shape[0]),
            "first_target_index": int(target_indices.min()),
            "last_target_index": int(target_indices.max()),
        }

    prediction = np.full((n_dates, n_wells), np.nan, dtype=np.float32)
    available = pred_count > 0
    prediction[available] = (
        pred_sum[available] / pred_count[available, None]
    ).astype(np.float32)
    split_summary["prediction_count_per_date"] = {
        "min_nonzero": int(pred_count[available].min()),
        "max": int(pred_count.max()),
        "dates_with_predictions": int(available.sum()),
    }
    return prediction, split_summary


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.size": 9,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )


def plot_well(
    output_path: Path,
    dates,
    observed: np.ndarray,
    predicted: np.ndarray,
    well_id: str,
    train_boundary,
    val_boundary,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    ax.plot(
        dates,
        observed,
        color="#333333",
        linewidth=0.9,
        label="Observed",
        zorder=2,
    )
    ax.plot(
        dates,
        predicted,
        color="#0072B2",
        linewidth=1.1,
        label="Predicted (mean T+1–T+6)",
        zorder=3,
    )
    for boundary in (train_boundary, val_boundary):
        ax.axvline(
            boundary,
            color="#666666",
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )

    segment_edges = (dates[0], train_boundary, val_boundary, dates[-1])
    for left, right, label in zip(
        segment_edges[:-1],
        segment_edges[1:],
        ("Train", "Validation", "Test"),
    ):
        midpoint = left + (right - left) / 2
        ax.text(
            midpoint,
            0.98,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            color="#555555",
            fontsize=8,
        )

    ax.set_title(f"Well {well_id}", loc="left", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Groundwater level (m)")
    ax.set_xlim(dates[0], dates[-1])
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.margins(y=0.08)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.05,
        pil_kwargs={"optimize": True},
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    set_data_root(cfg, args.data_root)
    device = (
        args.device
        if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    configure_style()
    bundle, windows, trainer = rebuild_trainer(cfg, device, args.checkpoint)
    prediction, split_summary = aggregate_predictions(bundle, windows, trainer)
    observed = np.where(bundle.mask > 0, bundle.H_obs, np.nan)
    dates = bundle.dates.to_pydatetime()
    train_boundary = dates[bundle.train_end_idx]
    val_boundary = dates[bundle.val_end_idx]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_wells = len(bundle.well_ids)
    limit = n_wells if args.limit is None else min(args.limit, n_wells)
    for well_idx in range(limit):
        well_id = str(bundle.well_ids[well_idx])
        plot_well(
            args.output_dir / f"well_{well_id}.png",
            dates,
            observed[:, well_idx],
            prediction[:, well_idx],
            well_id,
            train_boundary,
            val_boundary,
            args.dpi,
        )
        if (well_idx + 1) % 50 == 0 or well_idx + 1 == limit:
            print(f"plots={well_idx + 1}/{limit}", flush=True)

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "plots": limit,
                "wells_total": n_wells,
                "device": device,
                "torch": torch.__version__,
                "matplotlib": matplotlib.__version__,
                "aggregation": "mean of all T+1..T+6 forecasts per target date",
                "boundaries": {
                    "train_end": str(bundle.dates[bundle.train_end_idx].date()),
                    "validation_end": str(
                        bundle.dates[bundle.val_end_idx].date()
                    ),
                },
                "splits": split_summary,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

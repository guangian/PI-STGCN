"""PI-STGCN 主入口（唯一调用点）。

一次调用完成主实验：训练（2018–2019）→ 验证选型（2020）→ 测试评估（2021–2022），
六步预测 T+1..T+6（5 天步长，共 30 天）。

用法（在仓库根目录下）：
  python main.py                       # 主实验（T+1..T+6 一次产出）
  python main.py --smoke               # 冒烟测试（3 epoch，写入 results/full_smoke/）
  python main.py --seed 43             # 覆盖随机种子（结果目录加后缀）

结果写入 results/full/：
  qc_report.json / normalizer.json / train_log.json / best_model.pt /
  metrics_val.json / metrics_test.json / metrics_test.csv /
  per_well_metrics_test.csv（站点级明细：逐井×步长 RMSE/MAE/NSE/R²/skill，箱线图数据源）/
  run_meta.json
"""
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

from src.data_pipeline import PerWellNormalizer, build_windows, load_bundle, make_features  # noqa: E402
from src.evaluation import per_well_records, write_per_well_csv  # noqa: E402
from src.graphs import build_dual_graphs  # noqa: E402
from src.trainer import Trainer, set_seed  # noqa: E402


def run(cfg: dict, args) -> dict:
    ablation: dict = {}                            # 主实验：全模块开启（无消融开关）
    exp = "full"
    if args.seed is not None:                      # 多种子支持
        cfg["train"]["seed"] = int(args.seed)
        exp = f"{exp}_seed{args.seed}"
    if args.smoke:                                 # 冒烟测试独立目录，避免覆盖正式结果
        exp = f"{exp}_smoke"
    out_dir = (CODE_DIR / cfg["paths"]["results_root"]).resolve() / exp
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg["train"]["seed"]))

    bundle = load_bundle(cfg, CODE_DIR)
    json.dump(bundle.qc_report, open(out_dir / "qc_report.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    graphs = build_dual_graphs(bundle.coords, bundle.aquifer_onehot,
                               int(cfg["graph"]["knn_k"]), float(cfg["graph"]["min_edge_dist_m"]),
                               tuple(cfg["graph"]["voronoi_area_clip_q"]),
                               max_wd_ratio=float(cfg["graph"]["max_wd_ratio"]),
                               dem=bundle.dem)
    normalizer = PerWellNormalizer.fit(bundle.H_obs, bundle.mask, bundle.train_end_idx)
    normalizer.save(out_dir / "normalizer.json")
    feats = make_features(bundle, normalizer)
    windows = build_windows(bundle.H_fill.shape[0], int(cfg["data"]["input_len"]),
                            int(cfg["data"]["pred_len"]), bundle.train_end_idx, bundle.val_end_idx)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    trainer = Trainer(cfg, bundle, feats, windows, normalizer, graphs, device, ablation, out_dir)

    n_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    meta = {"exp": exp, "device": str(device),
            "n_wells": len(bundle.well_ids), "n_params": int(n_params),
            "n_windows": {k: int(len(v.x_idx)) for k, v in windows.items()},
            "graph_stats": graphs.stats,
            "n_pde_nodes(phreatic&interior)": int(trainer.pde_mask.sum().item()),
            "train_end_idx": bundle.train_end_idx, "val_end_idx": bundle.val_end_idx,
            "torch": torch.__version__, "seed": int(cfg["train"]["seed"]),
            "cudnn": {"deterministic": bool(torch.backends.cudnn.deterministic),
                      "benchmark": bool(torch.backends.cudnn.benchmark)},
            "assumption": "known-future-forcing rollout (scenario-based forecasting, 正文声明)",
            "primary_metric": "station-median (rmse/nse/skill well-median)",
            "pde_protocol": {"loss_mode": trainer.pde_loss_mode,
                             "weight_mode": trainer.pde_weight_mode,
                             "operator_params": trainer.pde_operator_params,
                             "lambda_configured": trainer.lambda_pde,
                             "defect_beta": trainer.pde_defect_beta,
                             "detach_phys_features": trainer.model.detach_phys_features,
                             "ramp_epochs": trainer.pde_ramp_epochs,
                             "grad_target_ratio": trainer.pde_grad_target_ratio,
                             "weight_bounds": [trainer.pde_weight_min, trainer.pde_weight_max],
                             "weight_ema": trainer.pde_weight_ema,
                             "conflict_gate": trainer.pde_conflict_gate},
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open(out_dir / "run_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[{exp}] wells={meta['n_wells']} params={n_params} windows={meta['n_windows']} device={device}", flush=True)

    epochs = 3 if args.smoke else (args.epochs or None)
    best_val = trainer.train(epochs=epochs)

    # 可靠性收缩标定：主干冻结，只在验证期拟合收缩头（测试集不参与）
    if bool(cfg["train"].get("use_shrink", False)):
        meta["shrink"] = trainer.fit_shrink()
        print(f"[{exp}] shrink λ_med={meta['shrink']['shrink_lambda_median']:.3f} "
              f"λ<0.5={meta['shrink']['shrink_frac_below_half']:.1%} "
              f"val_mse {meta['shrink']['shrink_val_mse_before']:.5f}"
              f"→{meta['shrink']['shrink_val_mse_after']:.5f}", flush=True)
        json.dump(meta, open(out_dir / "run_meta.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        torch.save(trainer.model.state_dict(), out_dir / "best_model.pt")

    m_val = trainer.evaluate("val")
    json.dump(m_val, open(out_dir / "metrics_val.json", "w"), indent=1)

    mt, arr = trainer.evaluate("test", return_arrays=True)
    json.dump(mt, open(out_dir / "metrics_test.json", "w"), indent=1)
    # 站点级明细（逐井 × 步长 RMSE/MAE/NSE/R²/skill）：箱线图等分布图的原始数据，
    # 站点中位值主口径可由本表逐位复算
    write_per_well_csv(per_well_records(arr["true"], arr["pred"], arr["persist"],
                                        arr["mask"], bundle.well_ids),
                       out_dir / "per_well_metrics_test.csv")
    if args.save_arrays:               # 保存逐井测试数组，供分层评估与配对统计
        np.savez_compressed(out_dir / "test_arrays.npz", **arr)
    # CSV 摘要（站点中位值为主口径；论文表格由此自动生成）
    lines = ["horizon,rmse_well_median,nse_well_median,skill_well_median,"
             "persistence_rmse_well_median,rmse_m_pooled,mae_m_pooled,r2_pooled_aux,"
             "n_wells_scored,n_obs_points"]
    for k in range(int(cfg["data"]["pred_len"])):
        h = mt["per_horizon"][f"T+{k+1}"]
        lines.append(f"T+{k+1},{h['rmse_well_median']:.4f},{h['nse_well_median']:.4f},"
                     f"{h['skill_well_median']:.4f},{h['persistence_rmse_well_median']:.4f},"
                     f"{h['rmse_m_pooled']:.4f},{h['mae_m_pooled']:.4f},{h['r2_pooled_aux']:.4f},"
                     f"{h['n_wells_scored']},{h['n_obs_points']}")
    (out_dir / "metrics_test.csv").write_text("\n".join(lines), encoding="utf-8")

    meta["completed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(meta, open(out_dir / "run_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    o = mt["overall"]
    print(f"[{exp}] DONE val_rmse_med={best_val:.4f} test_rmse_med={o['rmse_well_median']:.4f} "
          f"test_nse_med={o['nse_well_median']:.4f} skill_med={o['skill_well_median']:.4f}", flush=True)
    return mt


def main():
    ap = argparse.ArgumentParser(description="PI-STGCN 主实验：六步地下水位预测 T+1..T+6")
    ap.add_argument("--config", default=str(CODE_DIR / "config.yaml"))
    ap.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数（默认用 config）")
    ap.add_argument("--smoke", action="store_true", help="3-epoch 冒烟测试")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=None, help="覆盖 config 种子（结果目录加后缀）")
    ap.add_argument("--save_arrays", action="store_true", help="保存逐井测试预测供统计分析")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    run(cfg, args)


if __name__ == "__main__":
    main()

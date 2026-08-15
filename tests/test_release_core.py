from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from src.evaluation import write_rmse_nse_tables
from src.model import PISTGCN
from src.physics import EdgeConductance, node_net_inflow


ROOT = Path(__file__).resolve().parents[1]


def small_graph():
    return SimpleNamespace(
        phys_i=np.asarray([0, 1, 2, 0], dtype=np.int64),
        phys_j=np.asarray([1, 2, 3, 3], dtype=np.int64),
        phys_dist=np.full(4, 1000.0, dtype=np.float32),
        phys_width=np.full(4, 200.0, dtype=np.float32),
        cell_area=np.full(4, 1.0e6, dtype=np.float32),
    )


class ReleaseCoreTests(unittest.TestCase):
    def test_public_backbone_shape_and_parameter_count(self):
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        model = PISTGCN(cfg, n_static=7, graph=small_graph()).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4093)

        batch, nodes, horizons = 2, 4, 6
        temporal = torch.randn(batch, nodes, 11)
        static = torch.randn(nodes, 7)
        h_last_z = torch.randn(batch, nodes)
        physical_delta = torch.randn(batch, nodes, horizons) * 0.1
        h_m = torch.full((batch, nodes), 25.0)
        context = {
            "h_m": h_m,
            "h_anom_m": h_m - h_m.mean(0),
            "K": torch.full((nodes,), 10.0),
            "z_bot": torch.zeros(nodes),
        }
        output = model(
            temporal,
            static,
            h_last_z,
            physical_delta,
            darcy_ctx=context,
            return_components=True,
        )
        self.assertEqual(output["prediction"].shape, (batch, nodes, horizons))
        self.assertTrue(torch.isfinite(output["prediction"]).all())
        self.assertTrue(((output["gate"] > 0) & (output["gate"] < 1)).all())

    def test_edge_flux_is_conservative(self):
        graph = small_graph()
        cond = EdgeConductance(min_b=1.0)
        K = torch.full((4,), 10.0)
        z_bot = torch.zeros(4)
        head = torch.tensor([[24.0, 23.0, 22.0, 21.0]])
        i = torch.as_tensor(graph.phys_i)
        j = torch.as_tensor(graph.phys_j)
        conductance = cond(
            K,
            z_bot,
            head,
            i,
            j,
            torch.as_tensor(graph.phys_width),
            torch.as_tensor(graph.phys_dist),
        )
        flux = conductance * (head[..., i] - head[..., j])
        net = node_net_inflow(flux, i, j, n=4)
        self.assertAlmostEqual(float(net.sum()), 0.0, places=5)

    def test_compact_result_exports(self):
        rows = [
            {"well_id": "A", "horizon": "T+1", "rmse_m": 1.0, "nse": 0.5},
            {"well_id": "B", "horizon": "T+1", "rmse_m": 2.0, "nse": 0.4},
            {"well_id": "A", "horizon": "overall", "rmse_m": 1.5, "nse": 0.3},
            {"well_id": "B", "horizon": "overall", "rmse_m": 2.5, "nse": 0.2},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            long_path = Path(temp_dir) / "long.csv"
            pivot_path = Path(temp_dir) / "pivot.csv"
            write_rmse_nse_tables(rows, long_path, pivot_path)
            with long_path.open(newline="", encoding="utf-8") as handle:
                long_rows = list(csv.DictReader(handle))
            with pivot_path.open(newline="", encoding="utf-8") as handle:
                pivot_rows = list(csv.DictReader(handle))
        self.assertEqual(len(long_rows), 4)
        self.assertEqual(len(pivot_rows), 2)
        self.assertEqual(pivot_rows[0]["overall_NSE"], "0.300000")


if __name__ == "__main__":
    unittest.main()

# Methodology-to-code alignment

This file maps the released PI-STGCN implementation to the final forecasting
path described in the manuscript. The public code intentionally exposes one
main architecture rather than retaining superseded experimental branches.

## End-to-end path

| Method component | Public implementation |
|---|---|
| Training-only normalization and causal filling | `src.data_pipeline.PerWellNormalizer`, `load_bundle` |
| Seven static well descriptors | `src.data_pipeline.make_static_features` |
| Fixed Delaunay–Voronoi physical graph | `src.graphs.build_physical_graph` |
| Bounded effective K, Sy and recharge coefficient | `src.physics.ParamNet` |
| Harmonic-mean interface conductance | `src.physics.EdgeConductance` |
| Dynamic Darcy score, sign and magnitude channels | `src.physics.DarcyAttention.attention` |
| Darcy redistribution of the neural response | `src.physics.DarcyAttention.smooth_response` |
| Differentiable physical trajectory | `src.physics.FluxRollout` |
| Five multi-scale level increments and forcing | `src.trainer.Trainer._temporal_features` |
| Node-shared response MLP and physical gate | `src.model.PISTGCN.forward` |
| Anchored parallel T+1...T+6 forecast | `src.model.PISTGCN.forward` |
| Level, change, CVFD and parameter-smoothness losses | `src.trainer.Trainer._loss` |
| Validation-only station/horizon shrinkage | `src.trainer.Trainer.fit_shrink` |
| Observed-point-only RMSE and NSE | `src.evaluation` |

The forecast equation implemented by `PISTGCN.forward` is

```text
h_pred_z = h_last_z + gate * physical_delta_z + darcy_response_z
```

All six forecast horizons are generated in one forward pass. DarcyAttention is
evaluated only on fixed physical edges; there is no learnable N-by-N adjacency
matrix.

## Temporal inputs

The response MLP receives exactly the interpretable feature family used in the
method section:

- standardized level increments at lags 1, 2, 3, 6 and 12 (5–60 days);
- historical-window precipitation anomaly;
- forecast-window precipitation and evapotranspiration anomalies;
- forecast-window groundwater-abstraction anomaly;
- sine and cosine seasonal terms;
- the seven static well descriptors; and
- the six-horizon standardized physical increment.

The tracked release configuration uses a 24-step history window and a six-step
forecast horizon, with one step equal to five days.

## Loss and selection protocol

The main loss is

```text
L = 1.0 * L_level + 0.5 * L_change
    + 0.2 * L_CVFD + 0.001 * L_smooth
```

Only real observations (`Mask = 1`) contribute to supervised losses and
metrics. CVFD is evaluated only for interior phreatic control volumes. Model
selection uses validation station-median persistence skill with a centered
±3-epoch moving average. The test split is not used for checkpoint selection
or shrinkage calibration.

## Reproducing reported ablations

The public entry point keeps only the single-factor switches discussed in the
manuscript:

| Command suffix | Removed or changed component |
|---|---|
| `--ablation no_anchor` | current-level anchor |
| `--ablation no_shrinkage` | validation reliability shrinkage |
| `--ablation no_cvfd` | CVFD residual |
| `--ablation free_recession` | Sy/T-derived recession binding |
| `--ablation no_recharge_gate` | antecedent-precipitation recharge gate |
| `--ablation no_darcy` | Darcy response redistribution |
| `--ablation static_darcy` | dynamic head-difference term |
| `--ablation no_direction` | signed hydraulic-gradient channel |
| `--ablation no_future_forcing` | known future forcing |
| `--ablation no_nn_forcing` | forcing in the neural response path |
| `--ablation single_scale` | five distinct level-increment lags |

Example:

```bash
python main.py --data-root /path/to/private/data --ablation no_cvfd
```

Each ablation writes to its own `results/ablation_<name>/` directory.

## Released result artifacts

A full run creates the two files distributed through GitHub:

- `results/full/per_well_RMSE_NSE.csv`: 561 wells × seven horizon labels;
- `results/full/per_well_RMSE_NSE_pivot.csv`: one wide row per well.

The optional plotting command reconstructs shrinkage descriptors from the
training period before inference, ensuring that plotted predictions and final
metrics use the same model state.

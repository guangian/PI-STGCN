# PI-STGCN

Official code and released results for **“PI-STGCN: An Interpretable
Spatio-Temporal Graph Neural Network Based on Darcy's Law and the Boussinesq
Model for Multiscale Groundwater Level Prediction.”**

PI-STGCN generates T+1...T+6 groundwater-level forecasts in one forward pass
(five days per step; 30 days total). The final release path combines:

- five interpretable groundwater-level increments from 5 to 60 days;
- a node-shared response MLP with known future forcing;
- bounded effective hydrogeological parameters from ParamNet;
- a fixed Delaunay–Voronoi physical control-volume graph;
- dynamic, signed DarcyAttention response redistribution;
- a differentiable physical trajectory and CVFD water-balance residual;
- current-level anchoring and validation-only reliability shrinkage.

The implementation has 4,093 trainable parameters and contains no learnable
N-by-N adjacency matrix. See [METHODOLOGY_ALIGNMENT.md](METHODOLOGY_ALIGNMENT.md)
for the equation-to-code map and the supported manuscript ablations.

> **Data availability:** groundwater observations and water-use records are
> not included because redistribution permission has not been finalized. See
> [DATA.md](DATA.md) for the input schemas, units and provenance. The `data/`
> directory is ignored by Git.

## Repository structure

```text
.
├── main.py                         # train -> validate -> test -> export
├── generate_well_prediction_plots.py
├── config.yaml                     # released experiment settings
├── requirements.txt
├── DATA.md
├── METHODOLOGY_ALIGNMENT.md
├── LICENSE
├── src/
│   ├── data_pipeline.py            # loading, causal filling, QC, normalization
│   ├── graphs.py                   # fixed Delaunay–Voronoi physical graph
│   ├── model.py                    # node-shared MLP, anchor, gate, shrinkage
│   ├── physics.py                  # ParamNet, DarcyAttention, rollout, CVFD
│   ├── trainer.py                  # losses, selection, shrinkage, evaluation
│   └── evaluation.py               # per-well RMSE/NSE and exports
└── results/full/
    ├── per_well_RMSE_NSE.csv
    └── per_well_RMSE_NSE_pivot.csv
```

Checkpoints, logs, raw arrays, private data and all per-well PNG files remain
local through `.gitignore`.

## Released results

The released seed-42 run evaluates 561 wells on the independent 2021–2022
test period. Its overall station-median scores are:

| Metric | Value |
|---|---:|
| RMSE | 0.5198 m |
| NSE | 0.7603 |

- [`per_well_RMSE_NSE.csv`](results/full/per_well_RMSE_NSE.csv) is the long
  table with `well_id`, `horizon`, `RMSE_m` and `NSE` for T+1...T+6 and
  `overall` (3,927 rows).
- [`per_well_RMSE_NSE_pivot.csv`](results/full/per_well_RMSE_NSE_pivot.csv)
  is the final wide table with one row per well (561 rows).

Both files are generated directly by `main.py`; no manual spreadsheet step is
required.

## Environment

The tested environment is Python 3.12, PyTorch 2.5.1 and CUDA 12.1. Create an
environment and install the dependencies:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Install the CPU or CUDA build appropriate for the local machine when CUDA 12.1
is unavailable.

## Data preparation

Prepare the private dataset with the structure documented in [DATA.md](DATA.md).
It can remain anywhere on the local machine; `--data-root` overrides all three
tracked data paths without editing `config.yaml`.

The released temporal protocol is:

| Split | Target period | Purpose |
|---|---|---|
| Train | 2018-01-05 to 2019-12-31 | optimization and training-only statistics |
| Validation | 2020-01-05 to 2020-12-30 | checkpoint selection and shrinkage |
| Test | 2021-01-04 to 2022-12-30 | final evaluation only |

The input history contains 24 five-day steps and the target contains six
five-day steps. Windows whose target block crosses a split boundary are
discarded.

## Run

Smoke test:

```bash
python main.py --data-root "/path/to/private/data" --smoke --device cuda
```

Full experiment:

```bash
python main.py --data-root "/path/to/private/data" --device cuda
```

The run automatically writes the two released RMSE/NSE tables. Use
`--device cpu` on a CPU-only machine, `--seed` for another seed and
`--save-arrays` for
local downstream analysis.

The verified seed-42 run selected epoch 21, stopped after 64 epochs and used
approximately 145 seconds for model training on the local CUDA environment.

The public entry point also reproduces every single-factor ablation retained
in the manuscript, for example:

```bash
python main.py --data-root "/path/to/private/data" --ablation no_cvfd
```

See [METHODOLOGY_ALIGNMENT.md](METHODOLOGY_ALIGNMENT.md) for all choices.

## Per-well prediction figures

After a full run, generate one observed-versus-predicted figure per well:

```bash
python generate_well_prediction_plots.py \
  --data-root "/path/to/private/data" \
  --device cuda \
  --dpi 300
```

Each figure places train, validation and test on one timeline and separates
them with dashed vertical lines. Overlapping T+1...T+6 forecasts targeting the
same date are averaged. The script restores the validation-calibrated
shrinkage state before inference, so plots and final metrics use the same
prediction protocol. The 561 generated PNG files are intentionally excluded
from GitHub.

## Evaluation protocol

- Quality control, normalization and causal filling use training data only.
- No backward filling is used.
- Supervised losses and metrics use real observations only (`Mask = 1`).
- Model selection uses validation station-median persistence skill with a
  centered ±3-epoch moving average.
- The test split is never used for training, checkpoint selection or shrinkage.
- Forecast forcing is known, consistent with a scenario-based protocol.

Per-well metrics are:

```text
RMSE = sqrt(mean((prediction - observation)^2))
NSE  = 1 - sum((observation - prediction)^2)
           / sum((observation - mean(observation))^2)
```

## License and citation

Code is released under the [MIT License](LICENSE). The license does not cover
third-party data. A formal citation will be added when the associated paper is
published.

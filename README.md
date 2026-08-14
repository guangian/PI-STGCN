# PI-STGCN

Official code and released results for **“PI-STGCN: An Interpretable
Spatio-Temporal Graph Neural Network Based on Darcy's Law and the Boussinesq
Model for Multiscale Groundwater Level Prediction.”**

PI-STGCN performs six-step groundwater-level forecasting (T+1 to T+6, a
5-day step and a 30-day forecast horizon) on a dual spatial graph. The model
combines a data-driven spatio-temporal backbone with Darcy-aware attention and
a finite-volume groundwater-balance constraint.

> **Data availability:** the groundwater observations and water-use records
> are not distributed in this public repository because redistribution
> permission has not yet been finalized. See [DATA.md](DATA.md) for the exact
> input layout, schemas, units, and provenance. The code and released result
> artifacts are public.

## Repository structure

```text
.
├── main.py                         # train -> validate -> test
├── generate_well_prediction_plots.py
├── config.yaml                     # data, graph, physics, model, and training settings
├── requirements.txt                # tested Python dependencies
├── DATA.md                         # data availability and required schemas
├── LICENSE                         # MIT License for code
├── src/
│   ├── data_pipeline.py            # loading, causal filling, QC, normalization, windows
│   ├── graphs.py                   # physical Delaunay/Voronoi and informational kNN graphs
│   ├── model.py                    # PI-STGCN network
│   ├── physics.py                  # Darcy attention and physics rollout/residual
│   ├── trainer.py                  # optimization, model selection, shrinkage, evaluation
│   └── evaluation.py               # per-well and aggregate metrics
└── results/full/
    ├── per_well_RMSE_NSE.csv       # long table: 561 wells x 7 horizons
    └── per_well_RMSE_NSE_pivot.csv # GitHub-ready wide pivot table
```

The private `data/` directory, Python environment, checkpoints, logs, raw
prediction arrays, per-well PNG figures, and intermediate reports are excluded
through `.gitignore`.

## Released results

The released test artifacts were produced with seed 42 on 561 monitoring
wells. The verified overall station-median test scores are:

| Metric | Value |
|---|---:|
| RMSE | 0.5252 m |
| NSE | 0.7519 |

Result files:

- [`per_well_RMSE_NSE.csv`](results/full/per_well_RMSE_NSE.csv) contains
  `well_id`, `horizon`, `RMSE_m`, and `NSE` for T+1 through T+6 and `overall`
  (3,927 rows).
- [`per_well_RMSE_NSE_pivot.csv`](results/full/per_well_RMSE_NSE_pivot.csv)
  contains one row per well and paired RMSE/NSE columns for all horizons
  (561 rows).
The repository includes the script for generating 561 per-well PNG figures,
but the generated images themselves are intentionally not distributed. Each
figure overlays observed and predicted groundwater levels on one timeline;
dashed vertical lines separate train, validation, and test. The continuous
predicted curve is the mean of all overlapping T+1...T+6 forecasts that target
the same date.

RMSE and NSE are calculated per well using observed points only (`Mask = 1`):

```text
RMSE = sqrt(mean((prediction - observation)^2))
NSE  = 1 - sum((observation - prediction)^2)
           / sum((observation - mean(observation))^2)
```

## Environment

The released run was tested with:

- Python 3.12.6
- PyTorch 2.5.1+cu121
- CUDA 12.1
- NVIDIA GeForce RTX 4070

Create an environment and install the dependencies:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

For GPU execution, install the PyTorch build that matches the local CUDA
runtime. For example, the released experiment used:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data preparation

Place the locally obtained data under `data/` using the exact structure and
column schemas documented in [DATA.md](DATA.md). The default paths are defined
in `config.yaml`; they may be changed for another local dataset.

The published split contains 365 five-day timestamps:

| Split | Target period | Purpose |
|---|---|---|
| Train | 2018-01-05 to 2019-12-31 | optimization and training-only statistics |
| Validation | 2020-01-05 to 2020-12-30 | checkpoint selection and shrinkage calibration |
| Test | 2021-01-04 to 2022-12-30 | final evaluation only |

Windows whose target block crosses a split boundary are discarded.

## Run the experiment

Smoke test:

```bash
python main.py --smoke --device cuda
```

Full experiment:

```bash
python main.py --device cuda
```

Use `--device cpu` on a CPU-only machine. The seed can be overridden with
`--seed`; `--save_arrays` additionally stores the test arrays for downstream
analysis. On the tested RTX 4070, the released seed-42 run selected epoch 21,
stopped after 64 epochs, and took approximately 133 seconds.

After training has produced `results/full/best_model.pt`, regenerate the 561
per-well figures with:

```bash
python generate_well_prediction_plots.py --device cuda --dpi 300
```

Use `python main.py --help` and
`python generate_well_prediction_plots.py --help` for all options.

## Evaluation protocol

- Quality control, normalization, and causal head filling use training-period
  information only.
- Input gaps are forward-filled; no backward filling is used.
- Losses and metrics use real observations only (`Mask = 1`).
- Model selection uses validation station-median persistence skill.
- The test split is never used for training, early stopping, checkpoint
  selection, or shrinkage calibration.
- Forecasts use known future forcing, consistent with a scenario-based
  forecasting protocol.

## License and citation

The source code is released under the [MIT License](LICENSE). This license does
not cover third-party data. Data provenance and redistribution constraints are
documented in [DATA.md](DATA.md).

The formal citation will be added when the associated paper is published.

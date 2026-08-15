# Data availability and input specification

## Availability

The dataset used for the released PI-STGCN experiment is **not included** in
this public repository. Redistribution permission for the groundwater-level
observations and water-use records has not yet been finalized. Until written
authorization or an official public-access route is available, the complete
`data/` directory must remain private.

The repository's MIT License applies to source code only and does not grant
rights to any third-party dataset.

## Expected directory layout

The default paths in `config.yaml` expect the following local structure:

```text
data/
├── processed_wells/
│   ├── wells_summary_north_china_plain.csv
│   └── <well_id>/
│       └── water_level.csv
├── forcing/
│   ├── precipitation.csv
│   └── evapotranspiration.csv
└── geometry/
    ├── dem_at_wells.csv
    └── z_bot_at_wells.csv      # optional
```

All time-dependent files must use the same five-day timestamps. The released
experiment contains 365 timestamps from 2018-01-05 through 2022-12-30 and 561
quality-controlled wells.

## Required schemas

### Well summary

`data/processed_wells/wells_summary_north_china_plain.csv`

| Column position | Meaning | Unit/type |
|---:|---|---|
| 1 | well ID | string |
| 2 | longitude | decimal degrees |
| 3 | latitude | decimal degrees |
| 4 | ground elevation | m |
| 5 | aquifer type | category |
| 6 | province code | string/category |

The loader assigns the internal names `well_id`, `lon`, `lat`, `elev`,
`aquifer_type`, and `province_code` by column order. Each well ID must have a
matching directory and geometry record.

### Per-well observations

`data/processed_wells/<well_id>/water_level.csv`

| Column | Meaning | Unit/type |
|---|---|---|
| `Date` | timestamp | ISO date |
| `WaterLevel` | groundwater-level elevation | m |
| `Mask` | real-observation indicator | 0 or 1 |
| `Precipitation_mm` | optional redundant local forcing | mm/day |
| `Evapotranspiration_mm` | optional redundant local forcing | mm/day |
| `WaterUse_m3d` | groundwater abstraction | m^3/day |

The training pipeline reads `WaterLevel`, `Mask`, and `WaterUse_m3d` from each
well file. `Mask = 0` makes a water-level value unavailable to supervision and
evaluation even if a numeric value is present.

### Regional forcing

| File | Header | Unit |
|---|---|---|
| `forcing/precipitation.csv` | `date,precip` | mm/day, five-day-window mean |
| `forcing/evapotranspiration.csv` | `date,et` | mm/day, five-day-window mean |

Both series are converted to m/day using the factors in `config.yaml`.

### Geometry

| File | Required columns | Unit |
|---|---|---|
| `geometry/dem_at_wells.csv` | `well_id,elevation` | m |
| `geometry/z_bot_at_wells.csv` | `well_id,z_bot_m` | m |

DEM is required. `z_bot_at_wells.csv` is optional; when it is absent, the
model subtracts the configured aquifer-thickness prior from DEM.

## Quality control and missing values

The default pipeline applies the following rules using the training period
only:

1. real-observation rate of at least 60%;
2. no training-period single-step change greater than 30 m per five days;
3. absolute groundwater level no greater than 3,000 m;
4. training-period standard deviation of at least 0.001 m.

Input gaps are filled causally by forward filling. Missing values before the
first observation use the well's training-period observed mean. No backward
filling is used, and filled points never contribute to loss or metrics.

## Provenance of auxiliary sources

- Meteorological forcing was derived from ERA5-Land. When redistributed under
  an applicable authorization, retain the attribution: “Contains modified
  Copernicus Climate Change Service information [2018–2022].” See
  Muñoz-Sabater et al. (2021), *Earth System Science Data*,
  https://doi.org/10.5194/essd-13-4349-2021.
- Well elevations were sampled from NASA SRTM. See Farr et al. (2007),
  *Reviews of Geophysics*, https://doi.org/10.1029/2005RG000183.
- Groundwater-level and water-use provenance, access instructions, and a
  formal citation must be supplied by the data provider before those records
  can be publicly redistributed.

## Using another dataset

To run PI-STGCN on an authorized dataset, reproduce the directory layout and
schemas above and pass its root through `python main.py --data-root <path>`.
Alternatively, update the three entries under `paths` in `config.yaml`. Adjust
the date range, split boundaries, units and quality-control thresholds in the
same configuration file. The loader checks timestamp alignment and well-level
coverage before training.

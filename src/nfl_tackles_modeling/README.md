# NFL Tackles Modeling

Predicts `tackles_combined` (solo + assist) vs market line for NFL defensive player props.
Strategy: bet Over or Under based on a hybrid probabilistic estimator.

## Model

Two models trained jointly on the labeled dataset:

| Model | Purpose |
|---|---|
| OLS (`market_L16_game_ctx_pos_overprob`, 9 features) | Point prediction; source for Gaussian + Bootstrap + Hetero P(over) |
| NegBin NB2 (same 9 features) | Probabilistic P(over) via count distribution |

Walk-forward eval: trained on 2024 season, tested on 2025 season.

## Intermediate files

All intermediate outputs land in `~/Downloads/tmp/`:

| File | Produced by | Consumed by |
|---|---|---|
| `nfl_tackles_historical_spine.parquet` | `build_historical_spine.py` | `build_labeled_dataset.py` |
| `nfl_tackles_labeled.parquet` | `build_labeled_dataset.py` | `train.py`, `line_calibration.py` |
| `nfl_tackles_artifacts/` | `train.py` | `infer.py`, `generate_sweep_report.py`, Lambda |
| `nfl_tackles_inference.parquet` | `infer.py` | manual review |
| `nfl_tackles_sweep_report.html` | `generate_sweep_report.py` | browser |

Artifacts directory contains: `ols_pipeline.joblib`, `residuals.npy`, `nb_coefs.npy`, `nb_alpha.npy`, `meta.json`.

## Full pipeline

All scripts run from the `scripts/` directory:

```sh
cd /Users/thomasmyles/dev/betting/src/nfl_tackles_modeling/scripts
```

### Step 1 — Build labeled dataset

Skip if `nfl_tackles_labeled.parquet` is current.

```sh
python build_historical_spine.py   # assemble rolling spine features from nfl_data_py
python build_labeled_dataset.py    # join spine + Odds API props → labeled parquet
```

Tackle actuals from PFR weekly defensive stats via `nfl_data_py` (solo + assist).
Snap counts joined for snap% features.

### Step 2 — Train models + serialize artifacts

```sh
python train.py
# python train.py --upload-s3  # push artifacts to S3 for Lambda
```

### Step 3 — Sanity check inference (optional)

Shows in-sample hit rates by season / direction / bucket.

```sh
python infer.py
# python infer.py --edge-threshold 0.05
```

### Step 4 — Generate sweep report

135-combo grid search across edge thresholds, directions, and line buckets.
Color-coded HTML with sortable table, units won, max drawdown.

```sh
python generate_sweep_report.py
open ~/Downloads/nfl_tackles_sweep_report.html
```

### Optional — Calibration check

Run after retraining to verify the hybrid estimator is still well-calibrated
across line buckets.

```sh
python line_calibration.py
```

Produces 3 tables: per-line, snapped-to-.5, and 3-tackle-band buckets.

---

### Full one-liner (Steps 2–4, assuming data is current)

```sh
python train.py && python generate_sweep_report.py && open ~/Downloads/nfl_tackles_sweep_report.html
```

---

### When to re-run what

| Event | Steps needed |
|---|---|
| New season data added | 1 → 2 → 4 |
| Retrain only (same data) | 2 → 4 |
| Just refresh the report | 4 only |
| Upload new model to Lambda | `train.py --upload-s3` |

## Scripts reference

| Script | Purpose |
|---|---|
| `build_historical_spine.py` | Fetch PFR tackle + snap data via `nfl_data_py`, build rolling features |
| `build_labeled_dataset.py` | Join spine with Odds API prop lines → labeled parquet |
| `train.py` | Fit OLS + NegBin, serialize artifacts (+ optional S3 upload) |
| `infer.py` | Score full dataset, filter to bets, write inference parquets |
| `generate_sweep_report.py` | 135-combo strategy sweep → HTML report |
| `line_calibration.py` | Calibration tables: actual vs implied vs model by line bucket |
| `build_name_map.py` | Build player name map for Odds API ↔ PFR name reconciliation |
| `validate_tackle_counts.py` | Spot-check tackle counts against raw data |
| `audit_historical_spine.py` | Audit spine coverage (seasons, players, missing weeks) |
| `eda_labeled_dataset.py` | Exploratory analysis on labeled parquet |
| `param_sweep.py` | Hyperparameter sweep (model-level) |
| `spec_sweep.py` | Feature spec sweep (which feature combos to evaluate) |

## Config

`config/tackles_model_specs.yaml` — named feature combos and model types for the spec sweep.
Baseline spec is the reference row for `delta_mae_vs_baseline` comparisons.

"""
Model spec sweep for NFL tackles.

Iterates over every spec in tackles_model_specs.yaml × configured models,
evaluating each with:
  - Walk-forward OOS  : train on all seasons except the latest, test on latest
  - 5-fold CV         : on full filtered dataset (sanity check / small-n guard)

Baseline row is declared in the config (baseline.spec + baseline.model).
  is_baseline        = True for that one row
  delta_mae_vs_bl    = oos_mae − baseline_oos_mae  (negative = beats baseline)

Derived features added at load time (usable in any spec):
  position_group        LB | CB | S | DL
  pos_LB/CB/S/DL        binary dummies for position_group
  line_deviation        tackle_rate_L10 − offered_line
  consensus_over_prob   de-vigged implied over probability (mean across books)

Per-spec options (set in config under a spec entry):
  models: [ols]       override global model list for this spec only
  stratify_by: col    run spec separately per unique value of col;
                      results get a 'stratum' column (default 'all')

Models:
  ols   — OLS with StandardScaler
  ridge — Ridge regression (alpha=1.0) with StandardScaler
  mlp   — MLP regressor (64→32) with StandardScaler
  xgb   — XGBoost (moderate params)
  xgb2  — XGBoost (heavier regularization, shallower trees; better for small n)

Run:
  python src/nfl_tackles_modeling/scripts/spec_sweep.py
  python src/nfl_tackles_modeling/scripts/spec_sweep.py --config path/to/other.yaml
  python src/nfl_tackles_modeling/scripts/spec_sweep.py --sort cv_mae
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

LABELED_PATH = Path.home() / "Downloads" / "tmp" / "nfl_tackles_labeled.parquet"
OUT_PATH     = Path.home() / "Downloads" / "tmp" / "nfl_tackles_spec_sweep.parquet"
DEFAULT_CFG  = Path("src/nfl_tackles_modeling/config/tackles_model_specs.yaml")

TARGET = "tackles_combined"

POS_GROUP_MAP = {
    "LB": "LB",
    "CB": "CB", "DB": "CB",
    "S":  "S",  "FS": "S",  "SS": "S",
    "DE": "DL", "DT": "DL", "DL": "DL", "NT": "DL",
}


# ── Model factory ──────────────────────────────────────────────────────────────

def make_model(model_type: str):
    if model_type == "ols":
        return Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    if model_type == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    if model_type == "mlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(64, 32), activation="relu",
                max_iter=1000, random_state=42, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=20,
            )),
        ])
    if model_type == "xgb":
        # Moderate params — same as initial sweep
        return XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            verbosity=0, random_state=42,
        )
    if model_type == "xgb2":
        # Heavier regularization for small-n regime:
        # shallower trees, more iterations at lower LR,
        # L1+L2 reg, higher min_child_weight to prevent splitting on tiny groups
        return XGBRegressor(
            n_estimators=400, max_depth=2, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=2.0, min_child_weight=10,
            verbosity=0, random_state=42,
        )
    raise ValueError(f"Unknown model_type: {model_type!r}  (supported: ols, ridge, mlp, xgb, xgb2)")


# ── Derived features ───────────────────────────────────────────────────────────

def _american_to_implied(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.where(s.isna(), np.where(s < 0, -s / (-s + 100), 100 / (s + 100)))


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Position group
    df["position_group"] = df["position"].map(POS_GROUP_MAP)

    # Position dummies (0/1 int so they work with both OLS and XGB)
    for grp in ["LB", "CB", "S", "DL"]:
        df[f"pos_{grp}"] = (df["position_group"] == grp).astype(int)

    # Line deviation: how much does recent form disagree with the market? (uses L8 window)
    if "tackle_rate_L8" in df.columns and "offered_line" in df.columns:
        df["line_deviation"] = df["tackle_rate_L8"] - df["offered_line"]

    # Consensus de-vigged over probability: mean across books of (over_imp / (over_imp + under_imp))
    over_cols  = [c for c in df.columns if c.endswith("_over_price")]
    under_cols = [c for c in df.columns if c.endswith("_under_price")]
    if over_cols and under_cols:
        over_imp  = np.stack([_american_to_implied(df[c]).values for c in over_cols],  axis=1)
        under_imp = np.stack([_american_to_implied(df[c]).values for c in under_cols], axis=1)
        total     = over_imp + under_imp
        devig     = np.where(total > 0, over_imp / total, np.nan)
        df["consensus_over_prob"] = np.nanmean(devig, axis=1)

    return df


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_spec(
    df: pd.DataFrame,
    features: list[str],
    model_type: str,
    train_seasons: list[int],
    test_season: int,
    cv_splits: int,
) -> dict:
    sub = df[features + [TARGET, "season"]].dropna()

    train_df = sub[sub["season"].isin(train_seasons)]
    test_df  = sub[sub["season"] == test_season]

    result: dict = {"n_train": len(train_df), "n_test": len(test_df)}

    # ── Walk-forward OOS ──────────────────────────────────────────────────────
    if len(train_df) >= 20 and len(test_df) >= 20:
        m = make_model(model_type)
        m.fit(train_df[features].values, train_df[TARGET].values)
        preds = m.predict(test_df[features].values)
        y = test_df[TARGET].values
        result.update({
            "oos_mae":  mean_absolute_error(y, preds),
            "oos_rmse": float(np.sqrt(mean_squared_error(y, preds))),
            "oos_r2":   r2_score(y, preds),
        })
    else:
        result.update({"oos_mae": np.nan, "oos_rmse": np.nan, "oos_r2": np.nan})

    # ── K-fold CV (full data) ─────────────────────────────────────────────────
    X = sub[features].values
    y = sub[TARGET].values
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    maes, rmses, r2s = [], [], []
    for tr_idx, val_idx in kf.split(X):
        m = make_model(model_type)
        m.fit(X[tr_idx], y[tr_idx])
        p = m.predict(X[val_idx])
        maes.append(mean_absolute_error(y[val_idx], p))
        rmses.append(float(np.sqrt(mean_squared_error(y[val_idx], p))))
        r2s.append(r2_score(y[val_idx], p))
    result.update({
        "cv_mae":  float(np.mean(maes)),
        "cv_rmse": float(np.mean(rmses)),
        "cv_r2":   float(np.mean(r2s)),
    })
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--sort",   default="oos_mae")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    # ── Load + filter + derive ────────────────────────────────────────────────
    df = pd.read_parquet(LABELED_PATH)
    drop_pos = set(cfg.get("position_filter", {}).get("drop", []))
    before = len(df)
    df = df[df["position"].notna() & ~df["position"].isin(drop_pos)]
    df = add_derived_features(df)

    print(f"\nLabeled dataset: {before:,} → {len(df):,} rows after position filter")
    print(f"Seasons: {sorted(df['season'].unique())}  |  target mean: {df[TARGET].mean():.2f}")
    print(f"Position groups:\n{df['position_group'].value_counts().to_string()}\n")

    seasons       = sorted(df["season"].unique())
    test_season   = seasons[-1]
    train_seasons = seasons[:-1]
    print(f"Walk-forward split: train={train_seasons}  test=[{test_season}]")

    global_models = cfg.get("models", ["ols", "xgb"])
    cv_splits     = cfg.get("cv_splits", 5)
    specs         = cfg["specs"]
    baseline_cfg  = cfg.get("baseline", {})
    bl_spec       = baseline_cfg.get("spec", "market")
    bl_model      = baseline_cfg.get("model", "ols")

    print(f"Specs: {len(specs)}  |  Global models: {global_models}  |  CV splits: {cv_splits}")
    print(f"Baseline: spec='{bl_spec}'  model='{bl_model}'\n")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    rows = []
    for spec in specs:
        spec_name    = spec["name"]
        features     = spec["features"]
        spec_models  = spec.get("models", global_models)   # per-spec override
        stratify_by  = spec.get("stratify_by")             # optional column name

        if stratify_by:
            # Run separately for each stratum
            strata = sorted(df[stratify_by].dropna().unique())
            for stratum in strata:
                sub = df[df[stratify_by] == stratum]
                for model_type in spec_models:
                    metrics = evaluate_spec(
                        sub, features, model_type, train_seasons, test_season, cv_splits
                    )
                    rows.append({
                        "spec": spec_name, "model": model_type,
                        "stratum": str(stratum),
                        "n_features": len(features),
                        "feature_list": ",".join(features),
                        "is_baseline": False,
                        **metrics,
                    })
        else:
            for model_type in spec_models:
                metrics = evaluate_spec(
                    df, features, model_type, train_seasons, test_season, cv_splits
                )
                rows.append({
                    "spec": spec_name, "model": model_type,
                    "stratum": "all",
                    "n_features": len(features),
                    "feature_list": ",".join(features),
                    "is_baseline": (spec_name == bl_spec and model_type == bl_model),
                    **metrics,
                })

    results = pd.DataFrame(rows)

    # ── Delta vs baseline ─────────────────────────────────────────────────────
    bl_row = results[
        (results["spec"] == bl_spec) & (results["model"] == bl_model) &
        (results["stratum"] == "all")
    ]
    bl_oos_mae = bl_row["oos_mae"].iloc[0] if not bl_row.empty else np.nan
    results["delta_mae_vs_bl"] = results["oos_mae"] - bl_oos_mae

    # ── Sort + display ────────────────────────────────────────────────────────
    sort_col = args.sort if args.sort in results.columns else "oos_mae"

    # Non-stratified first (sorted), then stratified section (sorted by spec then stratum)
    non_strat = results[results["stratum"] == "all"].sort_values(sort_col, na_position="last")
    strat     = results[results["stratum"] != "all"].sort_values(["spec", "stratum"])
    results   = pd.concat([non_strat, strat], ignore_index=True)

    display_cols = [
        "spec", "model", "stratum", "is_baseline", "n_features", "feature_list",
        "n_train", "n_test",
        "oos_mae", "oos_rmse", "oos_r2",
        "cv_mae",  "cv_rmse",  "cv_r2",
        "delta_mae_vs_bl",
    ]
    disp = results[display_cols].copy()
    for col in ["oos_mae","oos_rmse","oos_r2","cv_mae","cv_rmse","cv_r2","delta_mae_vs_bl"]:
        disp[col] = disp[col].round(4)

    print(f"\n{'='*140}")
    print(f"  SPEC SWEEP RESULTS  (sorted by {sort_col}; stratified specs appended below)")
    print(f"  Baseline: {bl_spec} / {bl_model}  →  OOS MAE = {bl_oos_mae:.4f}")
    print(f"  delta_mae_vs_bl < 0  means beats the market baseline")
    print(f"{'='*140}\n")
    print(disp.to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(OUT_PATH, index=False)
    print(f"\n{'='*140}")
    print(f"  Saved → {OUT_PATH}  ({len(results)} rows × {len(results.columns)} cols)")
    print(f"{'='*140}\n")


if __name__ == "__main__":
    main()

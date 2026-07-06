"""
MLB Pitcher Strikeouts — Step 3: Model Training
================================================
Predicts pitcher strikeout count (continuous) via OLS + XGBoost regression,
OOF walk-forward split by season.

3a: Individual predictors — OLS (RMSE, MAE, R²)
3b: Individual predictors — XGBoost regressor
3c: Combo models; best model saved as artifact

OOF design: temporal walk-forward
  Fold 1: train 2024,       test 2025
  Fold 2: train 2024+2025,  test 2026

Artifacts saved:
  models/mlb_strikeouts_model.joblib  — best model pipeline
  models/mlb_strikeouts_residuals.npy — training residuals (for bootstrap P(over) calc)
  models/mlb_strikeouts_meta.json     — feature list, target, metrics

Outputs (local):
  ~/Downloads/tmp/mlb_strikeouts/step3a_individual_ols.csv
  ~/Downloads/tmp/mlb_strikeouts/step3b_individual_xgb.csv
  ~/Downloads/tmp/mlb_strikeouts/step3c_combos.csv
  ~/Downloads/tmp/mlb_strikeouts/step3_oof_predictions.parquet

Usage:
  python src/mlb_strikeouts_modeling/scripts/v3_model.py
  python src/mlb_strikeouts_modeling/scripts/v3_model.py --rebuild-spine
"""
from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

REPO_ROOT  = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET  = "the-odds-api-mt"
SPINE_KEY    = "mlb/strikeouts_model/spine/mlb_strikeouts_spine.parquet"
LABELED_KEY  = "mlb/strikeouts_model/labeled/mlb_strikeouts_labeled.parquet"
OUT_DIR    = Path.home() / "Downloads/tmp/mlb_strikeouts"
MODELS_DIR = REPO_ROOT / "models"
SPOT_CHECK = "paul skenes"

SEASONS    = [2024, 2025, 2026]
OOF_FOLDS  = [
    (2025, [2024]),
    (2026, [2024, 2025]),
]

TARGET = "strikeouts"

# All candidate features — evaluated individually in 3a/3b
ALL_FEATURES = [
    # Season-scoped rolling K averages (reset each season)
    "k_roll_s1", "k_roll_s3", "k_roll_s5", "k_roll_s10", "k_roll_s20",
    "k_roll_season",
    # Career-scoped rolling K averages (carry across seasons)
    "k_roll_c1", "k_roll_c3", "k_roll_c5", "k_roll_c10", "k_roll_c20",
    "k_roll_career",
    # Opponent and game context
    "opp_k_against_season",
    "ip_roll_season",
    "is_home",
    "days_rest",
    "game_month",
    # Market — consensus only (player-level, book-independent)
    "consensus_line",
    # New v5 features — consensus line odds bins (player-game level, book-independent)
    # Column names match build_labeled_dataset.py output exactly (v5 canonical names)
    "over_price_bucket", "under_price_bucket",
    "over_price_bucket_fine", "under_price_bucket_fine",
    # New v5 features — line and implied prob range (player-game level, book-independent)
    "min_line", "max_line",
    "min_over_prob", "max_over_prob",
    "min_under_prob", "max_under_prob",
]


def load_spine() -> pd.DataFrame:
    """Load the labeled dataset (has all rolling + v5 features) and deduplicate
    to one row per (player_key, game_date) for regression.
    Rolling features are identical across books for the same player-game;
    the new v5 features are also player-game level.
    """
    s3   = boto3.client("s3")
    body = s3.get_object(Bucket=S3_BUCKET, Key=LABELED_KEY)["Body"].read()
    lbl  = pd.read_parquet(BytesIO(body))
    # The market data also has a 'season' column, so after merging we get
    # season_x (market) and season_y (spine). Use spine's season (season_y).
    if "season" not in lbl.columns and "season_y" in lbl.columns:
        lbl["season"] = lbl["season_y"]
    elif "season" not in lbl.columns and "season_x" in lbl.columns:
        lbl["season"] = lbl["season_x"]
    # Deduplicate to one row per player-game — keep first (all book-independent
    # features are identical across books for the same player-game)
    df = lbl.drop_duplicates(subset=["player_key", "game_date"], keep="first").copy()
    return df


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"n_samples": len(y_true), "rmse": rmse, "mae": mae, "r2": r2}


def run_ols_oof(df: pd.DataFrame, features: str | list[str]) -> tuple[dict, pd.DataFrame, np.ndarray]:
    feats     = [features] if isinstance(features, str) else list(features)
    fold_mets = []
    all_preds = []
    residuals = []

    for test_season, train_seasons in OOF_FOLDS:
        train = df[df["season"].isin(train_seasons)].dropna(subset=feats + [TARGET])
        test  = df[df["season"] == test_season].dropna(subset=feats + [TARGET])
        if len(train) < 100 or len(test) < 50:
            continue

        pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
        pipe.fit(train[feats], train[TARGET])

        residuals.extend((train[TARGET].values - pipe.predict(train[feats])).tolist())

        yhat = pipe.predict(test[feats])
        fold_mets.append(regression_metrics(test[TARGET].values, yhat))
        all_preds.append(pd.DataFrame({
            "player_key": test["player_key"].values,
            "game_date":  test["game_date"].values,
            "season":     test["season"].values,
            TARGET:       test[TARGET].values,
            "yhat":       yhat,
            "fold":       test_season,
            "features":   str(feats),
        }))

    if not fold_mets:
        empty = {"n_samples": 0, "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
        return empty, pd.DataFrame(), np.array([])

    avg   = {k: float(np.mean([m[k] for m in fold_mets])) for k in fold_mets[0]}
    preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return avg, preds, np.array(residuals)


def run_xgb_oof(df: pd.DataFrame, features: str | list[str]) -> tuple[dict, pd.DataFrame, np.ndarray]:
    feats     = [features] if isinstance(features, str) else list(features)
    fold_mets = []
    all_preds = []
    residuals = []

    for test_season, train_seasons in OOF_FOLDS:
        train = df[df["season"].isin(train_seasons)].dropna(subset=feats + [TARGET])
        test  = df[df["season"] == test_season].dropna(subset=feats + [TARGET])
        if len(train) < 100 or len(test) < 50:
            continue

        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=3.0, reg_alpha=0.5,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        model.fit(train[feats], train[TARGET])

        residuals.extend((train[TARGET].values - model.predict(train[feats])).tolist())

        yhat = model.predict(test[feats])
        fold_mets.append(regression_metrics(test[TARGET].values, yhat))
        all_preds.append(pd.DataFrame({
            "player_key": test["player_key"].values,
            "game_date":  test["game_date"].values,
            "season":     test["season"].values,
            TARGET:       test[TARGET].values,
            "yhat":       yhat,
            "fold":       test_season,
            "features":   str(feats),
        }))

    if not fold_mets:
        empty = {"n_samples": 0, "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
        return empty, pd.DataFrame(), np.array([])

    avg   = {k: float(np.mean([m[k] for m in fold_mets])) for k in fold_mets[0]}
    preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return avg, preds, np.array(residuals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-spine", action="store_true",
                        help="Rebuild spine + labeled dataset before training")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.rebuild_spine:
        import subprocess
        print("Rebuilding spine...", flush=True)
        subprocess.run([sys.executable, str(REPO_ROOT / "src/mlb_strikeouts_modeling/scripts/build_spine.py")], check=True)
        print("Rebuilding labeled dataset...", flush=True)
        subprocess.run([sys.executable, str(REPO_ROOT / "src/mlb_strikeouts_modeling/scripts/build_labeled_dataset.py")], check=True)

    print("Loading spine from S3...", flush=True)
    df = load_spine()
    df["season"] = df["season"].astype(int)
    settled = df[df[TARGET].notna()].copy()
    print(f"  Total rows: {len(settled):,}  |  Pitchers: {settled['player_id'].nunique()}")
    for s in SEASONS:
        sub = settled[settled["season"] == s]
        print(f"  {s}: {len(sub):,} starts, {sub['player_id'].nunique()} pitchers, "
              f"avg K={sub[TARGET].mean():.2f}")

    # ── Spot-check: Paul Skenes ───────────────────────────────────────────────
    skenes = settled[settled["player_key"] == SPOT_CHECK].sort_values("game_date")
    if len(skenes) > 0:
        print(f"\n── Spot-check: {SPOT_CHECK} ({len(skenes)} starts) ──")
        print(skenes[["game_date", "season", "strikeouts", "k_roll_s5", "k_roll_career",
                       "consensus_line"]].tail(10).to_string(index=False))
    else:
        print(f"  WARNING: {SPOT_CHECK} not found in spine")

    # ── Step 3a: Individual OLS ───────────────────────────────────────────────
    print("\n── STEP 3a: Individual OLS (regression) ──", flush=True)
    rows_3a = []
    for feat in ALL_FEATURES:
        result, _, _ = run_ols_oof(settled, feat)
        rows_3a.append({"feature": feat, "model_type": "ols", **result})
        print(f"  {feat:<25}: RMSE={result['rmse']:.4f}  MAE={result['mae']:.4f}  R²={result['r2']:.4f}  n={result['n_samples']:,}")

    df_3a = pd.DataFrame(rows_3a).sort_values("rmse")
    df_3a.to_csv(OUT_DIR / "step3a_individual_ols.csv", index=False)
    print(f"\nTop 5 by RMSE:\n{df_3a.head(5)[['feature','rmse','mae','r2']].to_string(index=False)}")
    print(f"Saved: {OUT_DIR}/step3a_individual_ols.csv")

    # ── Step 3b: Individual XGBoost ───────────────────────────────────────────
    print("\n── STEP 3b: Individual XGBoost (regression) ──", flush=True)
    rows_3b = []
    for feat in ALL_FEATURES:
        result, _, _ = run_xgb_oof(settled, feat)
        rows_3b.append({"feature": feat, "model_type": "xgboost", **result})
        print(f"  {feat:<25}: RMSE={result['rmse']:.4f}  MAE={result['mae']:.4f}  R²={result['r2']:.4f}")

    df_3b = pd.DataFrame(rows_3b).sort_values("rmse")
    df_3b.to_csv(OUT_DIR / "step3b_individual_xgb.csv", index=False)
    print(f"\nTop 5 by RMSE:\n{df_3b.head(5)[['feature','rmse','mae','r2']].to_string(index=False)}")
    print(f"Saved: {OUT_DIR}/step3b_individual_xgb.csv")

    print("\n── XGBoost vs OLS RMSE delta (negative = XGB better) ──")
    merged_ab = df_3a[["feature", "rmse"]].merge(df_3b[["feature", "rmse"]], on="feature", suffixes=("_ols", "_xgb"))
    merged_ab["delta_rmse"] = merged_ab["rmse_xgb"] - merged_ab["rmse_ols"]
    print(merged_ab.sort_values("delta_rmse").to_string(index=False))

    # ── Step 3c: Combo models ─────────────────────────────────────────────────
    print("\n── STEP 3c: Combo models ──", flush=True)

    top_ols = df_3a.dropna(subset=["rmse"]).head(6)["feature"].tolist()
    top_xgb = df_3b.dropna(subset=["rmse"]).head(6)["feature"].tolist()
    top_all = list(dict.fromkeys(top_ols + top_xgb))  # deduplicated, order preserved
    print(f"  Top OLS features: {top_ols}")
    print(f"  Top XGB features: {top_xgb}")

    non_market = [f for f in top_all if f != "consensus_line"]

    V4_BASELINE = ["k_roll_career", "k_roll_c5", "opp_k_against_season", "is_home", "consensus_line"]

    combos = [
        # Best non-market features only (no consensus_line)
        non_market[:5],
        # Best non-market + consensus_line
        non_market[:5] + ["consensus_line"],
        # All top features (no consensus_line)
        non_market,
        # All top features (with consensus_line)
        top_all,
        # Rolling window sweep: season window only
        ["k_roll_s5", "k_roll_career", "opp_k_against_season", "is_home", "days_rest"],
        ["k_roll_s5", "k_roll_career", "opp_k_against_season", "is_home", "days_rest", "game_month"],
        ["k_roll_s10", "k_roll_career", "opp_k_against_season", "is_home", "days_rest", "game_month"],
        # Old winner minus novig_prob_over (season + career window)
        ["k_roll_s5", "k_roll_c5", "k_roll_career", "opp_k_against_season", "is_home", "days_rest", "game_month"],
        # Old winner minus novig_prob_over + consensus_line
        ["k_roll_s5", "k_roll_c5", "k_roll_career", "opp_k_against_season", "is_home",
         "consensus_line"],
        # Broader combo: days_rest, game_month, ip_roll_season
        ["k_roll_s5", "k_roll_c5", "k_roll_career", "opp_k_against_season", "is_home",
         "days_rest", "game_month", "ip_roll_season"],
        # + consensus_line
        ["k_roll_s5", "k_roll_c5", "k_roll_career", "opp_k_against_season", "is_home",
         "days_rest", "game_month", "ip_roll_season", "consensus_line"],
        # Lean on consensus_line heavily + minimal rolling
        ["k_roll_s5", "k_roll_career", "opp_k_against_season", "is_home", "consensus_line"],
        # v4 baseline (5 features — current production model)
        V4_BASELINE,
        # Medium combo with season + career + context
        ["k_roll_s3", "k_roll_s10", "k_roll_c5", "k_roll_career", "opp_k_against_season",
         "is_home", "days_rest", "consensus_line"],
        # ── v5 new feature combos ──────────────────────────────────────────────
        # Group A: simple odds buckets added to v4 baseline
        ["k_roll_career", "k_roll_c5", "opp_k_against_season", "is_home", "consensus_line",
         "over_price_bucket", "under_price_bucket"],
        # Group B: granular odds buckets added to v4 baseline
        ["k_roll_career", "k_roll_c5", "opp_k_against_season", "is_home", "consensus_line",
         "over_price_bucket_fine", "under_price_bucket_fine"],
        # Group C: line + prob range added to v4 baseline
        ["k_roll_career", "k_roll_c5", "opp_k_against_season", "is_home", "consensus_line",
         "min_line", "max_line", "min_over_prob", "max_over_prob"],
        # Group D: all v5 features
        ["k_roll_career", "k_roll_c5", "opp_k_against_season", "is_home", "consensus_line",
         "over_price_bucket", "under_price_bucket", "min_line", "max_line",
         "min_over_prob", "max_over_prob", "min_under_prob", "max_under_prob"],
    ]

    rows_3c     = []
    best_rmse   = float("inf")
    best_fn     = None
    best_feats  = None
    best_preds  = None
    best_resid  = None

    for feats in combos:
        feats = [f for f in feats if f in ALL_FEATURES]  # guard against typos
        if not feats:
            continue
        for label, fn in [("ols", run_ols_oof), ("xgboost", run_xgb_oof)]:
            result, preds, resid = fn(settled, feats)
            rationale = (f"{label} on {len(feats)} features; "
                         f"{'consensus_line included' if 'consensus_line' in feats else 'no market features'}")
            rows_3c.append({
                "features_included": str(feats),
                "n_features":        len(feats),
                "model_type":        label,
                **result,
                "rationale":         rationale,
            })
            print(f"  [{label:<7}] {len(feats)} feats: RMSE={result['rmse']:.4f}  "
                  f"MAE={result['mae']:.4f}  R²={result['r2']:.4f}  "
                  f"{'[mkt]' if 'consensus_line' in feats else '     '}")

            # always replace on strictly better RMSE; also replace when within
            # +0.005 RMSE of the best but simpler (fewer features)
            is_better = not np.isnan(result["rmse"]) and (
                result["rmse"] < best_rmse
                or (result["rmse"] <= best_rmse + 0.005
                    and len(feats) < len(best_feats or []))
            )
            if is_better:
                best_rmse  = result["rmse"]
                best_fn    = fn
                best_feats = feats
                best_preds = preds
                best_resid = resid

    df_3c = pd.DataFrame(rows_3c).sort_values("rmse")
    df_3c.to_csv(OUT_DIR / "step3c_combos.csv", index=False)
    print(f"\nBest combo: RMSE={best_rmse:.4f}  features={best_feats}")
    print(f"Saved: {OUT_DIR}/step3c_combos.csv")

    # ── v4 baseline vs v5 best comparison ────────────────────────────────────
    v4_rows = df_3c[df_3c["features_included"] == str(V4_BASELINE)]
    if len(v4_rows) > 0:
        v4_best_rmse = v4_rows["rmse"].min()
        print(f"\n── v4 baseline RMSE: {v4_best_rmse:.4f}")
        print(f"── v5 best RMSE:     {best_rmse:.4f}")
        delta = best_rmse - v4_best_rmse
        if delta < -0.001:
            print(f"── v5 IMPROVES over v4 by {abs(delta):.4f} RMSE units")
        elif delta > 0.001:
            print(f"── v4 baseline WINS — v5 is {delta:.4f} RMSE units worse; keep v4 model")
        else:
            print(f"── v4 and v5 are essentially tied (delta={delta:.4f}); prefer simpler v4 model")
    else:
        print(f"\n── v4 baseline not found in combos; best RMSE={best_rmse:.4f}")

    # ── CRITICAL ASSERT: yhat must be book-independent ─────────────────────
    # Since we deduplicate to one row per player-game before regression,
    # OOF predictions are already one row per player-game — this assert
    # always passes here and documents the invariant.
    if best_preds is not None and len(best_preds) > 0:
        yhat_check = best_preds.groupby(["player_key", "game_date"])["yhat"].nunique()
        n_inconsistent = (yhat_check > 1).sum()
        assert n_inconsistent == 0, (
            f"BOOK-DEPENDENCE BUG: {n_inconsistent} player-games have different yhat values "
            f"across rows. A book-dependent feature is in the model inputs — check ALL_FEATURES "
            f"for anything that varies per book (e.g. novig_prob_over). Fix before proceeding."
        )
        print(f"  [assert] yhat book-independence confirmed: all {len(yhat_check):,} player-games have consistent yhat")

    # ── Train best model on ALL settled data, save artifacts ─────────────────
    print("\nTraining best model on all settled data...", flush=True)
    if best_feats:
        full = settled.dropna(subset=best_feats + [TARGET])

        model_type_label = "xgboost" if best_fn == run_xgb_oof else "ols"
        if best_fn == run_xgb_oof:
            final_model = xgb.XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=3.0, reg_alpha=0.5,
                random_state=42, n_jobs=-1, verbosity=0,
            )
            final_model.fit(full[best_feats], full[TARGET])
        else:
            final_model = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
            final_model.fit(full[best_feats], full[TARGET])

        yhat_full      = final_model.predict(full[best_feats])
        final_residuals = full[TARGET].values - yhat_full

        joblib.dump(final_model, MODELS_DIR / "mlb_strikeouts_model.joblib")
        np.save(MODELS_DIR / "mlb_strikeouts_residuals.npy", final_residuals)

        meta = {
            "features":       best_feats,
            "target":         TARGET,
            "model_type":     model_type_label,
            "n_rows_train":   int(len(full)),
            "oof_rmse":       round(best_rmse, 4),
            "residual_std":   round(float(final_residuals.std()), 4),
            "residual_mean":  round(float(final_residuals.mean()), 4),
            "seasons_trained": [int(s) for s in full["season"].unique().tolist()],
        }
        (MODELS_DIR / "mlb_strikeouts_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  Saved model → {MODELS_DIR}/mlb_strikeouts_model.joblib")
        print(f"  Saved residuals → {MODELS_DIR}/mlb_strikeouts_residuals.npy  ({len(final_residuals):,} values)")
        print(f"  Residual σ={final_residuals.std():.4f}  mean={final_residuals.mean():.4f}")

    # ── Save OOF predictions + residuals ─────────────────────────────────────
    if best_preds is not None and len(best_preds) > 0:
        best_preds.to_parquet(OUT_DIR / "step3_oof_predictions.parquet", index=False)
        np.save(OUT_DIR / "step3_oof_residuals.npy", best_resid)
        print(f"\n  OOF predictions: {len(best_preds):,} rows → {OUT_DIR}/step3_oof_predictions.parquet")
        print(f"  OOF residuals:   {len(best_resid):,} values  σ={best_resid.std():.4f} → {OUT_DIR}/step3_oof_residuals.npy")
        for fold_s in best_preds["fold"].unique():
            sub = best_preds[best_preds["fold"] == fold_s]
            mets = regression_metrics(sub[TARGET].values, sub["yhat"].values)
            print(f"    {fold_s}: n={mets['n_samples']:,}  RMSE={mets['rmse']:.4f}  MAE={mets['mae']:.4f}  R²={mets['r2']:.4f}")

        # Spot-check Paul Skenes
        sc = best_preds[best_preds["player_key"] == SPOT_CHECK]
        if len(sc) > 0:
            print(f"\n── Spot-check {SPOT_CHECK} OOF predictions ──")
            print(sc[["game_date", "season", TARGET, "yhat"]].to_string(index=False))
            print(f"  mean yhat={sc['yhat'].mean():.2f}  mean actual={sc[TARGET].mean():.2f}")
        else:
            print(f"\n  NOTE: {SPOT_CHECK} not in OOF folds (may only appear in 2026 if 2024 is the sole training season)")

    print("\nDone.")


if __name__ == "__main__":
    main()

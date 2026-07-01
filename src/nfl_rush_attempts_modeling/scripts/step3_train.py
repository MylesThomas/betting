"""
Step 3 — Model Training for NFL rush attempts.

Variable/numeric line market — predict actual carries count (regression).
P(over) is derived in Step 4 by comparing oof_carries against each book line.

Step 3a: n=1 individual predictors (Ridge regression, OOF CV)
Step 3b: n=1 individual predictors (XGBRegressor, OOF CV)
Step 3c: Combo models (best features, Ridge + XGBoost, OOF CV)

Target: carries (actual rush attempts, continuous)
Training unit: one row per player-game (deduped from per-book training set)
OOF design: strict temporal folds (sorted by season, week — no future in any fold)

NaN handling: fill all rolling features with 0 (first-game players have no
history; games_played=0 tells the model how much to trust rolling features).
over_rate_L* filled with 0.5 (neutral prior for players with no line history).

Engineered features (computed in engineer_features(), replicated at inference):
  carry_bucket_L8_*  — one-hot role tier from carry_rate_L8 (4 dummies, ref = <5)
  line_bucket_*      — one-hot line tier from consensus_point (4 dummies, ref = <5)
  line_deviation     — consensus_point - carry_rate_L8 (market vs recent form)
  carry_trend        — carry_rate_L3 - carry_rate_L5 (short-term momentum)
  bell_cow_flag      — carry_rate_Lcareer >= 12 (long-run bell-cow identity)
  line_x_pos_RB      — consensus_point × pos_RB (high line = very different for RBs)
  line_x_bell_cow    — consensus_point × bell_cow_flag (bell-cow + high line signal)

Output:
  ~/Downloads/tmp/rush_attempts/step3a_results.csv
  ~/Downloads/tmp/rush_attempts/step3b_results.csv
  ~/Downloads/tmp/rush_attempts/step3c_results.csv
  ~/Downloads/tmp/rush_attempts/oof_predictions.parquet
  models/nfl_rush_attempts/best_model.pkl
"""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

TRAIN_PATH = Path.home() / "Downloads" / "tmp" / "rush_attempts" / "training.parquet"
OUT_DIR    = Path.home() / "Downloads" / "tmp" / "rush_attempts"
MODEL_DIR  = REPO_ROOT / "models" / "nfl_rush_attempts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5

# ── Feature definitions ────────────────────────────────────────────────────────

ROLLING_CARRY   = ["carry_rate_L1", "carry_rate_L3", "carry_rate_L5",
                   "carry_rate_L8", "carry_rate_L16", "carry_rate_Lcareer"]
ROLLING_YARDS   = ["rush_yards_L1", "rush_yards_L3", "rush_yards_L5",
                   "rush_yards_L8", "rush_yards_L16", "rush_yards_Lcareer"]
OVER_RATE       = ["over_rate_L3", "over_rate_L5", "over_rate_L8",
                   "over_rate_L16", "over_rate_Lcareer"]
OPP_DEF         = ["opp_carry_allowed_L8", "opp_carry_allowed_L16",
                   "opp_carry_allowed_Lcareer"]
MARKET_FEATURES = ["consensus_point"]
CONTEXT         = ["is_home", "game_total", "is_playoff", "games_played"]
POSITION        = ["pos_RB", "pos_QB"]

# Engineered features added by engineer_features()
CARRY_BUCKETS = [
    "carry_bucket_L8_5to9", "carry_bucket_L8_10to14",
    "carry_bucket_L8_15to19", "carry_bucket_L8_20plus",
]
LINE_BUCKETS = [
    "line_bucket_5to9", "line_bucket_10to14",
    "line_bucket_15to19", "line_bucket_20plus",
]
ENGINEERED = (
    CARRY_BUCKETS + LINE_BUCKETS +
    ["line_deviation", "carry_trend", "bell_cow_flag",
     "line_x_pos_RB", "line_x_bell_cow"]
)

ALL_CANDIDATE_FEATURES = (
    ROLLING_CARRY + ROLLING_YARDS + OVER_RATE + OPP_DEF +
    MARKET_FEATURES + CONTEXT + POSITION + ENGINEERED
)


# ── Feature engineering (replicated at inference time in scoring step) ────────

CARRY_BINS = [0, 5, 10, 15, 20, np.inf]
CARRY_LABELS = ["lt5", "5to9", "10to14", "15to19", "20plus"]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features derived from existing spine columns.
    Called at training time and must be replicated identically at inference.
    Reference category for bucket dummies is <5 (omitted).
    """
    df = df.copy()

    # Carry rate tier dummies (reference = <5 carries/game avg over L8)
    carry_tier = pd.cut(df["carry_rate_L8"], bins=CARRY_BINS,
                        labels=CARRY_LABELS, right=False)
    for label in CARRY_LABELS[1:]:   # drop "lt5" as reference
        col = f"carry_bucket_L8_{label}"
        df[col] = (carry_tier == label).astype(int)

    # Consensus line tier dummies (reference = <5 — QBs and deep backups)
    line_tier = pd.cut(df["consensus_point"], bins=CARRY_BINS,
                       labels=CARRY_LABELS, right=False)
    for label in CARRY_LABELS[1:]:
        col = f"line_bucket_{label}"
        df[col] = (line_tier == label).astype(int)

    # Market deviation from recent form: positive = market projects usage uptick
    df["line_deviation"] = df["consensus_point"] - df["carry_rate_L8"]

    # Short-term momentum: positive = hot streak over last 3 vs last 5
    df["carry_trend"] = df["carry_rate_L3"] - df["carry_rate_L5"]

    # Long-run bell-cow identity (career avg >= 12 carries/game)
    df["bell_cow_flag"] = (df["carry_rate_Lcareer"] >= 12).astype(int)

    # Interaction: high market line means more for RBs than QBs
    df["line_x_pos_RB"] = df["consensus_point"] * df["pos_RB"]

    # Interaction: bell-cow getting a high line is a strong anchoring signal
    df["line_x_bell_cow"] = df["consensus_point"] * df["bell_cow_flag"]

    return df


# ── Data loading + preprocessing ──────────────────────────────────────────────

def load_training() -> pd.DataFrame:
    df = pd.read_parquet(TRAIN_PATH)

    for col in ROLLING_CARRY + ROLLING_YARDS + OPP_DEF:
        df[col] = df[col].fillna(0)
    for col in OVER_RATE:
        df[col] = df[col].fillna(0.5)

    # Dedup to one row per player-game
    df_pg = (
        df.sort_values("n_books", ascending=False)
          .drop_duplicates(subset=["nfl_game_id", "player_name_norm"])
          .reset_index(drop=True)
    )

    df_pg = engineer_features(df_pg)

    print(f"Training set: {len(df_pg):,} unique player-games")
    print(f"Target (carries) — mean: {df_pg['carries'].mean():.2f}, "
          f"std: {df_pg['carries'].std():.2f}, "
          f"median: {df_pg['carries'].median():.0f}")
    print(f"Seasons: {sorted(df_pg['season'].unique())}")
    return df_pg


# ── OOF cross-validation (strict temporal) ────────────────────────────────────

def temporal_oof_cv(df: pd.DataFrame, features: list[str], model_type: str,
                    n_folds: int = N_FOLDS) -> np.ndarray:
    """
    Returns oof_carries (predicted rush attempt count) sorted to match df row order.
    Folds are created by sorting on (season, week) — all training data precedes
    validation data in time.
    """
    df = df.reset_index(drop=True)
    sorted_idx = df.sort_values(["season", "week"]).index.tolist()
    fold_size  = len(sorted_idx) // n_folds

    oof_carries = np.full(len(df), np.nan)

    X = df[features].values
    y = df["carries"].values

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end   = (fold + 1) * fold_size if fold < n_folds - 1 else len(sorted_idx)

        val_idx   = sorted_idx[val_start:val_end]
        train_idx = sorted_idx[:val_start]

        if len(train_idx) < 50:
            continue

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val       = X[val_idx]

        if model_type == "ridge":
            scaler   = StandardScaler()
            X_tr_sc  = scaler.fit_transform(X_tr)
            X_val_sc = scaler.transform(X_val)
            model    = Ridge(alpha=1.0)
            model.fit(X_tr_sc, y_tr)
            oof_carries[val_idx] = model.predict(X_val_sc)

        elif model_type == "xgboost":
            model = XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                verbosity=0, random_state=42, n_jobs=-1,
            )
            model.fit(X_tr, y_tr)
            oof_carries[val_idx] = model.predict(X_val)

    return oof_carries


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    feature_name: str, model_type: str,
                    coef: float | None = None) -> dict:
    mask   = ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]

    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae  = float(mean_absolute_error(yt, yp))
    try:
        r2 = float(r2_score(yt, yp))
    except Exception:
        r2 = np.nan

    return {
        "feature":     feature_name,
        "model_type":  model_type,
        "n_samples":   int(mask.sum()),
        "rmse":        round(rmse, 4),
        "mae":         round(mae, 4),
        "r2":          round(r2, 4),
        "coefficient": round(float(coef), 4) if coef is not None else None,
    }


# ── Step 3a: n=1 Ridge regression ─────────────────────────────────────────────

def run_3a(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 3a — n=1 individual predictors (Ridge regression, OOF)")
    print("="*60)

    results = []
    y = df["carries"].values

    for feat in ALL_CANDIDATE_FEATURES:
        if feat not in df.columns:
            continue
        feat_df = df[[feat, "season", "week", "carries"]].copy()

        oof = temporal_oof_cv(feat_df, [feat], "ridge")

        # Full-data Ridge for coefficient sign check
        Xf     = df[[feat]].values
        scaler = StandardScaler()
        Xf_sc  = scaler.fit_transform(Xf)
        ridge  = Ridge(alpha=1.0).fit(Xf_sc, y)
        coef   = ridge.coef_[0]

        metrics = compute_metrics(y, oof, feat, "ridge", coef)
        results.append(metrics)
        print(f"  {feat:<35} RMSE={metrics['rmse']:.4f}  R²={metrics['r2']:+.4f}  coef={coef:+.3f}")

    results_df = pd.DataFrame(results).sort_values("rmse", ascending=True)
    return results_df


# ── Step 3b: n=1 XGBRegressor ─────────────────────────────────────────────────

def run_3b(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 3b — n=1 individual predictors (XGBRegressor, OOF)")
    print("="*60)

    results = []
    y = df["carries"].values

    for feat in ALL_CANDIDATE_FEATURES:
        if feat not in df.columns:
            continue
        feat_df = df[[feat, "season", "week", "carries"]].copy()
        oof = temporal_oof_cv(feat_df, [feat], "xgboost")
        metrics = compute_metrics(y, oof, feat, "xgboost")
        results.append(metrics)
        print(f"  {feat:<35} RMSE={metrics['rmse']:.4f}  R²={metrics['r2']:+.4f}")

    results_df = pd.DataFrame(results).sort_values("rmse", ascending=True)
    return results_df


# ── Step 3c: Combo models ──────────────────────────────────────────────────────

def run_3c(df: pd.DataFrame, top_features_3a: list[str],
           top_features_3b: list[str]) -> tuple[pd.DataFrame, np.ndarray, object]:
    print("\n" + "="*60)
    print("STEP 3c — Combo models (best features, Ridge + XGBoost, OOF)")
    print("="*60)

    y = df["carries"].values
    results = []

    combos = [
        # Market baseline
        ("market_only", ["consensus_point"]),
        # Rolling carry + market (previous best)
        ("carry_L8_market",        ["carry_rate_L8", "consensus_point"]),
        # Carry tier buckets + market (address extremes)
        ("carry_buckets_market",
         CARRY_BUCKETS + ["consensus_point"]),
        # Line tier buckets + market (non-linear response to line value)
        ("line_buckets_market",
         LINE_BUCKETS + ["consensus_point"]),
        # Both bucket sets + market
        ("carry_line_buckets_market",
         CARRY_BUCKETS + LINE_BUCKETS + ["consensus_point"]),
        # Interactions: line × RB, line × bell-cow
        ("interactions_market",
         ["line_x_pos_RB", "line_x_bell_cow", "consensus_point"]),
        # Deviation + trend + market
        ("deviation_trend_market",
         ["line_deviation", "carry_trend", "bell_cow_flag", "consensus_point"]),
        # Best engineered parsimonious: carry buckets + interactions + market
        ("engineered_parsimonious",
         CARRY_BUCKETS + ["line_x_pos_RB", "line_x_bell_cow",
                          "line_deviation", "consensus_point"]),
        # Full engineered set
        ("engineered_full",
         CARRY_BUCKETS + LINE_BUCKETS +
         ["line_deviation", "carry_trend", "bell_cow_flag",
          "line_x_pos_RB", "line_x_bell_cow",
          "carry_rate_L8", "consensus_point"]),
        # Full kitchen sink (all original + all engineered)
        ("full_kitchen_sink",
         ["carry_rate_L5", "carry_rate_L8", "carry_rate_Lcareer",
          "rush_yards_L5", "rush_yards_L8",
          "is_home", "game_total", "is_playoff",
          "pos_RB", "pos_QB", "games_played",
          "consensus_point"] + ENGINEERED),
        # Parsimonious: top 3 from 3a + market
        ("parsimonious_top3_market",
         list(dict.fromkeys([top_features_3a[0], top_features_3a[1],
                             top_features_3a[2], "consensus_point"]))),
    ]

    best_rmse  = np.inf
    best_oof   = None
    best_model = None
    best_name  = None
    best_feats = None

    for name, feats in combos:
        feats = list(dict.fromkeys(f for f in feats if f in df.columns))
        if len(feats) < 1:
            continue

        combo_df = df[feats + ["season", "week", "carries"]].copy()

        for mtype in ["ridge", "xgboost"]:
            oof     = temporal_oof_cv(combo_df, feats, mtype)
            metrics = compute_metrics(y, oof, name, mtype)
            metrics["features_included"] = str(feats)
            metrics["n_features"]        = len(feats)
            metrics["rationale"]         = _rationale(name, mtype)
            results.append(metrics)

            combo_key = f"{name}_{mtype}"
            print(f"  {combo_key:<55} RMSE={metrics['rmse']:.4f}  R²={metrics['r2']:+.4f}")

            if metrics["rmse"] < best_rmse:
                best_rmse  = metrics["rmse"]
                best_oof   = oof
                best_name  = combo_key
                best_feats = feats
                best_model = mtype

    results_df = pd.DataFrame(results).sort_values("rmse", ascending=True)
    print(f"\nBest combo: {best_name}  RMSE={best_rmse:.4f}")
    print(f"  Features: {best_feats}")

    # Retrain best model on full data
    print(f"\nRetraining best model ({best_model}) on full data...")
    X_full = df[best_feats].values
    if best_model == "ridge":
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X_full)
        prod_model = Ridge(alpha=1.0)
        prod_model.fit(X_sc, y)
        prod_artifact = {"model": prod_model, "scaler": scaler,
                         "features": best_feats, "model_type": best_model}
    else:
        prod_model = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            verbosity=0, random_state=42, n_jobs=-1,
        )
        prod_model.fit(X_full, y)
        prod_artifact = {"model": prod_model, "scaler": None,
                         "features": best_feats, "model_type": best_model}

    model_path = MODEL_DIR / "best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(prod_artifact, f)
    print(f"Saved model to: {model_path}")

    return results_df, best_oof, prod_artifact


def _rationale(name: str, mtype: str) -> str:
    rationales = {
        "market_only":
            "Market consensus line alone. Regression baseline.",
        "carry_L8_market":
            "8-game carry average + market line. Previous best from v1 run.",
        "carry_buckets_market":
            "Carry tier dummies + market. Explicit role segments let Ridge fit differently for bell-cows vs backups.",
        "line_buckets_market":
            "Line tier dummies + market. Non-linear response to QB vs RB carry line ranges.",
        "carry_line_buckets_market":
            "Both bucket sets + market. Segment the prediction space by player role AND line tier.",
        "interactions_market":
            "Line × RB and line × bell-cow interactions + market. High line is a much stronger signal for an RB bell-cow.",
        "deviation_trend_market":
            "Market deviation from recent form + momentum + bell-cow flag + market. Captures usage spikes and streaks.",
        "engineered_parsimonious":
            "Carry tier buckets + interactions + market deviation + market. Addresses regression-toward-mean at extremes.",
        "engineered_full":
            "All engineered features + carry_rate_L8 + market. Full non-linear feature set.",
        "full_kitchen_sink":
            "All original + all engineered features. Overfitting expected but useful as RMSE floor.",
        "parsimonious_top3_market":
            "Top 3 individual predictors from 3a + market line.",
    }
    return rationales.get(name, "")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_training()
    y  = df["carries"].values

    # 3a — Ridge regression
    results_3a = run_3a(df)
    results_3a.to_csv(OUT_DIR / "step3a_results.csv", index=False)
    print(f"\nTop 10 (3a — sorted ascending RMSE):\n"
          f"{results_3a[['feature','rmse','mae','r2','coefficient']].head(10).to_string(index=False)}")

    # 3b — XGBRegressor
    results_3b = run_3b(df)
    results_3b.to_csv(OUT_DIR / "step3b_results.csv", index=False)
    print(f"\nTop 10 (3b — sorted ascending RMSE):\n"
          f"{results_3b[['feature','rmse','mae','r2']].head(10).to_string(index=False)}")

    # 3c — Combos (select best by lowest RMSE)
    top3a = results_3a["feature"].head(5).tolist()
    top3b = results_3b["feature"].head(5).tolist()
    results_3c, best_oof, prod_artifact = run_3c(df, top3a, top3b)
    results_3c.to_csv(OUT_DIR / "step3c_results.csv", index=False)
    print(f"\nTop 10 (3c — sorted ascending RMSE):\n"
          f"{results_3c[['feature','model_type','rmse','r2','n_features']].head(10).to_string(index=False)}")

    # Save OOF predictions (oof_carries = predicted rush attempt count)
    df["oof_carries"] = best_oof
    df.to_parquet(OUT_DIR / "oof_predictions.parquet", index=False)
    print(f"\nSaved OOF predictions to: {OUT_DIR / 'oof_predictions.parquet'}")
    print("\n=== Step 3 complete ===")

"""
Step 3c — Combo Models + Consensus-Line Calibration Method.

Compares:
  1. Top-K logistic regression (best features from Step 3a)
  2. Consensus-line calibration: use empirical hit rate per line as P(over/under)
     (the "consensus_line" prediction method from the grid search config)
  3. XGB with top-K features only

The consensus_line method is most interesting for efficient markets:
it asks "does the market systematically misprice certain line buckets?"
No feature engineering needed — just the calibration table.

Usage:
  python src/mlb_game_totals_modeling/scripts/20260711_step3c_combo_models.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_DIR  = Path.home() / "Downloads/tmp/mlb_game_totals"
LOCAL_PATH = LOCAL_DIR / "game_totals_spine.parquet"

N_FOLDS = 5
SEED    = 42


def load_spine() -> pd.DataFrame:
    df = pd.read_parquet(LOCAL_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["hit_over"]  = df["hit_over"].astype(int)
    df["hit_under"] = df["hit_under"].astype(int)
    return df


def oof_logistic_multi(df: pd.DataFrame, feature_cols: list[str], target: str,
                        label: str) -> tuple[float, np.ndarray, pd.DataFrame]:
    """OOF logistic regression with multiple features."""
    valid = df[feature_cols + [target, "game_pk", "game_date"]].dropna(
        subset=feature_cols + [target]
    )
    valid = valid.sort_values("game_date").reset_index(drop=True)
    X = valid[feature_cols].values.astype(float)
    y = valid[target].values
    groups = valid["game_pk"].values

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_proba = np.zeros(len(valid))

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)
        clf = LogisticRegression(max_iter=500)
        clf.fit(X_tr_sc, y_tr)
        oof_proba[test_idx] = clf.predict_proba(X_te_sc)[:, 1]

    auc   = roc_auc_score(y, oof_proba)
    preds = (oof_proba > 0.5).astype(int)
    prec  = precision_score(y, preds, zero_division=0)
    rec   = recall_score(y, preds, zero_division=0)
    print(f"  {label}: AUC={auc:.4f}  prec={prec:.4f}  rec={rec:.4f}")
    return auc, oof_proba, valid


def oof_consensus_line(df: pd.DataFrame, target: str) -> tuple[float, np.ndarray, pd.DataFrame]:
    """
    Consensus-line calibration method.
    In each training fold, compute historical hit rate per line.
    Apply to test fold: P(target | line=X) = training hit rate at line X.
    Falls back to global mean if line unseen in training.
    """
    valid = df[["line", target, "game_pk", "game_date"]].dropna()
    valid = valid.sort_values("game_date").reset_index(drop=True)
    y = valid[target].values
    groups = valid["game_pk"].values

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_proba = np.zeros(len(valid))

    for train_idx, test_idx in gkf.split(valid.values, y, groups):
        tr_data = valid.iloc[train_idx]
        te_data = valid.iloc[test_idx]

        global_mean = tr_data[target].mean()
        cal_table = tr_data.groupby("line")[target].mean()

        oof_proba[test_idx] = te_data["line"].map(cal_table).fillna(global_mean).values

    auc   = roc_auc_score(y, oof_proba)
    preds = (oof_proba > 0.5).astype(int)
    prec  = precision_score(y, preds, zero_division=0)
    rec   = recall_score(y, preds, zero_division=0)
    print(f"  Consensus-line ({target}): AUC={auc:.4f}  prec={prec:.4f}  rec={rec:.4f}")
    return auc, oof_proba, valid


def oof_xgb_topk(df: pd.DataFrame, feature_cols: list[str], target: str,
                  label: str) -> float:
    """XGB with top-K features."""
    valid = df[feature_cols + [target, "game_pk", "game_date"]].dropna(
        subset=feature_cols + [target]
    )
    valid = valid.sort_values("game_date").reset_index(drop=True)
    X = valid[feature_cols].values.astype(float)
    y = valid[target].values
    groups = valid["game_pk"].values

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_proba = np.zeros(len(valid))

    for train_idx, test_idx in gkf.split(X, y, groups):
        clf = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            eval_metric="logloss", random_state=SEED, verbosity=0,
        )
        clf.fit(X[train_idx], y[train_idx])
        oof_proba[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

    auc   = roc_auc_score(y, oof_proba)
    preds = (oof_proba > 0.5).astype(int)
    prec  = precision_score(y, preds, zero_division=0)
    rec   = recall_score(y, preds, zero_division=0)
    print(f"  {label}: AUC={auc:.4f}  prec={prec:.4f}  rec={rec:.4f}")
    return auc


def main() -> None:
    print("Loading spine...")
    df = load_spine()

    # Derived features
    df["combined_ra_L5"]  = df["home_ra_L5"]  + df["away_ra_L5"]
    df["combined_ra_L10"] = df["home_ra_L10"] + df["away_ra_L10"]
    print(f"  {len(df)} rows, {df['game_pk'].nunique()} games\n")

    # ── Top features from Step 3a ────────────────────────────────────────────
    # auc_over rank: line_is_integer (0.5159), home_ra_career (0.5122), home_ra_L20 (0.5102),
    #                home_rs_L10 (0.5082), line (0.5079), min_line (0.5073), min_raw_implied_prob_under (0.5069)
    top5_over  = ["line_is_integer", "home_ra_career", "home_ra_L20", "home_rs_L10", "line"]
    top7_over  = top5_over + ["min_line", "min_raw_implied_prob_under"]
    # auc_under rank: line_is_integer (0.5156), min_raw_implied_prob_under (0.5137), home_ra_L20 (0.5116),
    #                 combined_ra_L5 (0.5115), combined_ra_L10 (0.5107)
    top5_under = ["line_is_integer", "min_raw_implied_prob_under", "home_ra_L20",
                   "combined_ra_L5", "combined_ra_L10"]
    # Mins/max + bins
    minmax_features = [
        "min_line", "max_line",
        "min_raw_implied_prob_over", "max_raw_implied_prob_over",
        "min_raw_implied_prob_under", "max_raw_implied_prob_under",
    ]
    top_plus_minmax = list(set(top7_over + minmax_features))

    print("=== HIT_OVER MODELS ===")
    oof_logistic_multi(df, top5_over,       "hit_over", "Logistic top-5 (over)")
    oof_logistic_multi(df, top7_over,       "hit_over", "Logistic top-7 (over)")
    oof_logistic_multi(df, top_plus_minmax, "hit_over", "Logistic top-7 + min/max features")
    oof_xgb_topk(df, top5_over,            "hit_over", "XGB top-5 (over)")
    auc_cal_over, _, _ = oof_consensus_line(df, "hit_over")

    print("\n=== HIT_UNDER MODELS ===")
    oof_logistic_multi(df, top5_under,      "hit_under", "Logistic top-5 (under)")
    oof_logistic_multi(df, top7_over,       "hit_under", "Logistic top-7 (over features vs under target)")
    oof_logistic_multi(df, top_plus_minmax, "hit_under", "Logistic top-7 + min/max")
    oof_xgb_topk(df, top5_under,           "hit_under", "XGB top-5 (under)")
    auc_cal_under, _, _ = oof_consensus_line(df, "hit_under")

    # ── Calibration table on full dataset ───────────────────────────────────
    print("\n=== CALIBRATION TABLE (full dataset, for reference) ===")
    cal = (
        df.groupby("line")
        .agg(
            n           = ("hit_over",        "count"),
            over_rate   = ("hit_over",         "mean"),
            under_rate  = ("hit_under",        "mean"),
            push_rate   = ("hit_push",         "mean"),
            novig_over  = ("novig_prob_over",  "mean"),
            novig_under = ("novig_prob_under", "mean"),
        )
        .reset_index()
    )
    cal["cal_gap_over"]  = (cal["over_rate"]  - cal["novig_over"]).round(3)
    cal["cal_gap_under"] = (cal["under_rate"] - cal["novig_under"]).round(3)
    for col in ["over_rate", "under_rate", "push_rate", "novig_over", "novig_under"]:
        cal[col] = cal[col].round(3)
    print(cal.to_string(index=False))

    print("\n=== MODEL SELECTION RECOMMENDATION ===")
    print("""
  Individual features have AUC ~0.50-0.516 — near-random for a highly efficient market.
  XGB with all features: AUC 0.505 (worse than best single feature).
  Consensus-line calibration method is the strongest signal:
    - Line 9.5 under: actual 54.8%, market-implied 50.7% (+4.1pp gap)
    - Line 8.0 under: actual 48.0%, push 7.6% — integer line inflation
    - Line 9.0 under: actual 46.7%, push 10.4%

  DECISION: Use consensus_line prediction method in grid search.
    - p_model = empirical hit rate at that line bucket (from training window)
    - edge = p_model - novig_prob_over (or under)
    - The 9.5 under is the primary bet target

  Logistic regression with top-5 features adds marginal AUC vs random,
  but not enough to justify the complexity over a simple calibration lookup.
    """)


if __name__ == "__main__":
    main()

"""
Step 4 — Method C Per-Line Calibration for MLB Game Totals.

Maps OOF y_hat (predicted total_runs) → P(hit_over | line) using per-line
logistic regression. This is the "Method C" calibration used in total_bases
and pitcher_outs pipelines.

For each distinct line bucket (where n_games >= 50):
  - Fit logistic(y_hat) → P(hit_over=1)
  - Brier score: mean((p_cal - hit_over)²) — lower is better
  - Compare to novig baseline: Brier(novig_prob_over)

p_model must be book-invariant: same value per (game_pk, line) across all books.
This is guaranteed because y_hat is game-level and calibration is by line bucket only.

Edge formula (per skill spec):
  edge_under = p_model_under - raw_prob_under   (raw = vig-inclusive)

Output:
  ~/Downloads/tmp/mlb_game_totals/step4_spine_calibrated.parquet  — full spine + p_model
  Prints calibration table per line

Usage:
  python src/mlb_game_totals_modeling/scripts/20260711_step4_calibration.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_DIR      = Path.home() / "Downloads/tmp/mlb_game_totals"
SPINE_PATH     = LOCAL_DIR / "game_totals_spine.parquet"
YHAT_PATH      = LOCAL_DIR / "step3_oof_yhat.parquet"
OUT_PATH       = LOCAL_DIR / "step4_spine_calibrated.parquet"

MIN_GAMES_PER_LINE = 50  # minimum to fit per-line calibration


def load_data() -> pd.DataFrame:
    """Load spine and broadcast game-level y_hat to all book rows."""
    spine = pd.read_parquet(SPINE_PATH)
    yhat  = pd.read_parquet(YHAT_PATH)[["game_pk", "y_hat"]]

    df = spine.merge(yhat, on="game_pk", how="left")
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def fit_calibration_in_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    In-sample Method C calibration: for each line bucket with >= MIN_GAMES_PER_LINE,
    fit logistic(y_hat) → P(hit_over). Falls back to global logistic for rare lines.

    Returns df with added columns:
      p_model_over, p_model_under, edge_over, edge_under
    """
    df = df.copy()
    df["p_model_over"]  = np.nan
    df["p_model_under"] = np.nan

    valid = df.dropna(subset=["y_hat", "hit_over", "line", "raw_prob_over", "raw_prob_under"])

    # Global fallback calibration (fitted on all valid rows)
    sc_global = StandardScaler()
    X_global  = sc_global.fit_transform(valid[["y_hat"]].values.astype(float))
    clf_global = LogisticRegression(max_iter=500, C=1.0)
    clf_global.fit(X_global, valid["hit_over"].values.astype(int))
    global_preds = clf_global.predict_proba(X_global)[:, 1]

    # Per-line calibration
    for line_val in sorted(valid["line"].unique()):
        mask = valid["line"] == line_val
        n    = mask.sum()

        if n < MIN_GAMES_PER_LINE:
            # Use global logistic fallback
            sub_X = sc_global.transform(valid.loc[mask, ["y_hat"]].values.astype(float))
            preds = clf_global.predict_proba(sub_X)[:, 1]
        else:
            sub  = valid[mask]
            X    = sub[["y_hat"]].values.astype(float)
            y    = sub["hit_over"].values.astype(int)
            sc   = StandardScaler()
            X_sc = sc.fit_transform(X)
            clf  = LogisticRegression(max_iter=500, C=0.5)
            clf.fit(X_sc, y)
            preds = clf.predict_proba(X_sc)[:, 1]

        df.loc[valid[mask].index, "p_model_over"]  = preds
        df.loc[valid[mask].index, "p_model_under"] = 1.0 - preds

    # Edge: p_model - raw_prob (per skill spec, vig-inclusive)
    df["edge_over"]  = df["p_model_over"]  - df["raw_prob_over"]
    df["edge_under"] = df["p_model_under"] - df["raw_prob_under"]

    return df


def calibration_table(df: pd.DataFrame) -> pd.DataFrame:
    """Calibration analysis per line bucket."""
    valid = df.dropna(subset=["p_model_over", "hit_over"])
    rows = []

    for line_val in sorted(valid["line"].unique()):
        sub = valid[valid["line"] == line_val]
        n   = sub["game_pk"].nunique()  # unique games
        if n < 10:
            continue

        actual_over   = sub["hit_over"].mean()
        actual_under  = sub["hit_under"].mean()
        p_cal_over    = sub["p_model_over"].mean()
        novig_over    = sub["novig_prob_over"].mean()
        raw_over      = sub["raw_prob_over"].mean()

        brier_model  = ((sub["p_model_over"] - sub["hit_over"].astype(float)) ** 2).mean()
        brier_novig  = ((sub["novig_prob_over"] - sub["hit_over"].astype(float)) ** 2).mean()

        rows.append({
            "line":         line_val,
            "n_games":      n,
            "actual_over":  round(actual_over, 3),
            "actual_under": round(actual_under, 3),
            "p_cal_over":   round(p_cal_over, 3),
            "novig_over":   round(novig_over, 3),
            "cal_gap":      round(actual_over - p_cal_over, 3),
            "novig_gap":    round(actual_over - novig_over, 3),
            "brier_model":  round(brier_model, 4),
            "brier_novig":  round(brier_novig, 4),
            "brier_delta":  round(brier_novig - brier_model, 4),
        })

    return pd.DataFrame(rows)


def main() -> None:
    print("Loading spine + OOF y_hat...")
    df = load_data()
    print(f"  {len(df)} rows, {df['game_pk'].nunique()} games")
    print(f"  Rows with y_hat: {df['y_hat'].notna().sum()}\n")

    print("Fitting Method C per-line calibration (in-sample)...")
    df = fit_calibration_in_sample(df)
    print(f"  Rows with p_model_over: {df['p_model_over'].notna().sum()}\n")

    # Calibration table
    print("=== CALIBRATION TABLE (in-sample) ===")
    cal = calibration_table(df)
    print(cal.to_string(index=False))

    # Summary: is Model better than novig?
    n_better = (cal["brier_delta"] > 0).sum()
    print(f"\n  Model better than novig (brier_delta > 0): {n_better} / {len(cal)} line buckets")

    # Focus on 9.5 under
    row_95 = cal[cal["line"] == 9.5]
    if len(row_95):
        r = row_95.iloc[0]
        print(f"\n=== SPOTLIGHT: LINE 9.5 ===")
        print(f"  n_games:      {r['n_games']}")
        print(f"  actual_under: {1 - r['actual_over']:.3f}")
        print(f"  p_cal_under:  {1 - r['p_cal_over']:.3f}  (model-calibrated)")
        print(f"  novig_under:  {1 - r['novig_over']:.3f}  (market fair price)")
        print(f"  model edge:   {(1 - r['p_cal_over']) - (df[df['line']==9.5]['raw_prob_under'].mean()):.3f}  (vs raw_prob)")

    # Edge distribution for 9.5 under bets
    bets_95 = df[(df["line"] == 9.5) & df["edge_under"].notna()].copy()
    if len(bets_95) > 0:
        print(f"\n=== EDGE DISTRIBUTION: line=9.5, under ===")
        print(f"  mean edge: {bets_95['edge_under'].mean():.4f}")
        print(f"  fraction with edge>0: {(bets_95['edge_under'] > 0).mean():.3f}")
        print(f"  fraction with edge>2pp: {(bets_95['edge_under'] > 0.02).mean():.3f}")
        print(f"  fraction with edge>5pp: {(bets_95['edge_under'] > 0.05).mean():.3f}")
        hits_pos_edge = bets_95[bets_95["edge_under"] > 0]["hit_under"].mean()
        hits_neg_edge = bets_95[bets_95["edge_under"] <= 0]["hit_under"].mean()
        print(f"  hit rate (edge>0):  {hits_pos_edge:.3f}  n={int((bets_95['edge_under'] > 0).sum())}")
        print(f"  hit rate (edge<=0): {hits_neg_edge:.3f}  n={int((bets_95['edge_under'] <= 0).sum())}")

    # Save
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved calibrated spine → {OUT_PATH}")
    print(f"  Columns added: p_model_over, p_model_under, edge_over, edge_under")


if __name__ == "__main__":
    main()

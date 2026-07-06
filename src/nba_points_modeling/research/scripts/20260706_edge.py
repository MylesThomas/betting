"""
NBA Player Points — Step 4: Bootstrap P(under/over) → Edge
============================================================
Uses OOF yhat from Step 3. For each player-game, bootstraps P(under) by
drawing N=10,000 samples from yhat + resample(OOF training residuals).
P(under) = fraction of draws <= offered_line.

Edge = p_model_under - p_market_under  (positive → bet UNDER)
Edge = p_model_over  - p_market_over   (positive → bet OVER)

Matches approach used in nfl_rec_yards_modeling/scripts/infer.py.

Outputs:
  ~/Downloads/tmp/points_eda/step4_edge.parquet
  ~/Downloads/tmp/points_eda/step4_calibration.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR    = Path.home() / "Downloads/tmp/points_eda"
SPOT_CHECK = "stephen curry"
N_BOOT     = 10_000
RNG        = np.random.default_rng(42)


def bootstrap_p_under(yhat: np.ndarray, line: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """P(pts_actual <= line) via bootstrap resampling of OOF residuals."""
    samples = RNG.choice(residuals, size=(len(yhat), N_BOOT), replace=True)
    sims    = yhat[:, None] + samples            # shape: (n_rows, N_BOOT)
    return (sims <= line[:, None]).mean(axis=1)  # fraction of draws <= line


def main():
    print("Loading OOF predictions...", flush=True)
    oof = pd.read_parquet(OUT_DIR / "step3_oof_predictions.parquet")
    print(f"  OOF rows: {len(oof):,}")

    print("Loading OOF residuals...", flush=True)
    residuals = np.load(OUT_DIR / "step3_oof_residuals.npy")
    print(f"  Residuals: {len(residuals):,}  σ={residuals.std():.4f}  mean={residuals.mean():.4f}")

    print("Loading spine for market data...", flush=True)
    spine = pd.read_parquet(OUT_DIR / "points_spine.parquet")
    spine_settled = spine[spine["pts_actual"].notna()].copy()

    # Join OOF predictions to spine for offered_line, novig_prob_over, is_over, etc.
    # Drop pts_actual from OOF to avoid column conflict (spine's value is authoritative)
    oof_join = oof.drop(columns=["pts_actual"], errors="ignore")
    df = oof_join.merge(
        spine_settled[[
            "player_key", "game_date", "offered_line", "novig_prob_over",
            "is_over", "pts_actual", "min_line", "max_line", "n_books",
            "is_home", "days_rest", "opp_pts_allowed_L10",
        ]],
        on=["player_key", "game_date"],
        how="inner",
    )
    print(f"  After join: {len(df):,} rows")
    print(f"  novig_prob_over nulls: {df['novig_prob_over'].isna().sum()}")

    # Drop rows missing line or market prob
    df = df.dropna(subset=["offered_line", "novig_prob_over", "yhat"])
    print(f"  After dropping nulls: {len(df):,} rows")

    # ── Bootstrap P(under/over) ───────────────────────────────────────────────
    print(f"\nBootstrapping P(under/over)  [N={N_BOOT:,} draws per row]...", flush=True)
    yhat_arr = df["yhat"].values
    line_arr = df["offered_line"].values

    p_model_under = bootstrap_p_under(yhat_arr, line_arr, residuals)
    p_model_over  = 1.0 - p_model_under

    df["p_model_under"] = p_model_under
    df["p_model_over"]  = p_model_over

    # ── Market probabilities (novig) ──────────────────────────────────────────
    # novig_prob_over = P(over) from market; 1 - that = P(under)
    df["p_market_over"]  = df["novig_prob_over"]
    df["p_market_under"] = 1.0 - df["novig_prob_over"]

    # ── Edge ──────────────────────────────────────────────────────────────────
    df["edge_under"] = df["p_model_under"] - df["p_market_under"]  # positive = bet UNDER
    df["edge_over"]  = df["p_model_over"]  - df["p_market_over"]   # positive = bet OVER

    print(f"\nP(model under) range: [{p_model_under.min():.4f}, {p_model_under.max():.4f}]")
    print(f"avg p_model_under: {p_model_under.mean():.4f}  avg p_market_under: {df['p_market_under'].mean():.4f}")
    print(f"\nEdge distribution:")
    print(f"  edge_under: mean={df['edge_under'].mean():.4f}  std={df['edge_under'].std():.4f}")
    print(f"  edge_over:  mean={df['edge_over'].mean():.4f}   std={df['edge_over'].std():.4f}")
    print(f"  % with edge_under > 0.02: {(df['edge_under']>0.02).mean():.1%}")
    print(f"  % with edge_under > 0.05: {(df['edge_under']>0.05).mean():.1%}")
    print(f"  % with edge_over  > 0.02: {(df['edge_over']>0.02).mean():.1%}")
    print(f"  % with edge_over  > 0.05: {(df['edge_over']>0.05).mean():.1%}")

    # ── Calibration: p_model_under deciles vs actual under rate ───────────────
    df["p_under_decile"] = pd.qcut(df["p_model_under"], 10, labels=False, duplicates="drop")
    df["is_under"] = (df["is_over"] == 0).astype(float)

    calib = (
        df.groupby("p_under_decile", observed=True)
        .agg(
            n=("is_under", "count"),
            avg_p_model=("p_model_under", "mean"),
            actual_under_rate=("is_under", "mean"),
        )
        .reset_index()
    )
    calib["calib_gap"] = calib["actual_under_rate"] - calib["avg_p_model"]
    calib["calib_gap_pct"] = (calib["calib_gap"] * 100).round(1)
    calib["flag"] = calib["calib_gap"].abs() > 0.15

    print(f"\nCalibration by p_model_under decile:")
    print(calib[["p_under_decile", "n", "avg_p_model", "actual_under_rate", "calib_gap_pct", "flag"]].to_string(index=False))

    flagged = calib[calib["flag"]]
    if len(flagged) > 0:
        print(f"\n  ⚠ {len(flagged)} deciles with |calib_gap| > 15pp — review for systematic bias")
    else:
        print(f"\n  All deciles within |calib_gap| ≤ 15pp ✓")

    calib.to_csv(OUT_DIR / "step4_calibration.csv", index=False)

    # ── Edge by line bucket ───────────────────────────────────────────────────
    df["line_bucket"] = pd.cut(
        df["offered_line"],
        bins=[0, 14.5, 19.5, 24.5, 29.5, 100],
        labels=["≤14.5", "15–19.5", "20–24.5", "25–29.5", "≥30"],
    )
    print(f"\nEdge by line bucket:")
    line_edge = (
        df.groupby("line_bucket", observed=True)
        .agg(
            n=("edge_under", "count"),
            avg_yhat=("yhat", "mean"),
            avg_line=("offered_line", "mean"),
            avg_edge_under=("edge_under", "mean"),
            avg_edge_over=("edge_over", "mean"),
            actual_under_rate=("is_under", "mean"),
            avg_p_model_under=("p_model_under", "mean"),
            avg_p_market_under=("p_market_under", "mean"),
        )
        .reset_index()
    )
    print(line_edge.to_string(index=False))

    # ── Spot-check: Curry ─────────────────────────────────────────────────────
    print(f"\n── Spot-check: {SPOT_CHECK} ──")
    curry = df[df["player_key"] == SPOT_CHECK].sort_values("game_date")
    print(f"  OOF rows: {len(curry)}")
    if len(curry) > 0:
        cols = ["game_date", "season", "pts_actual", "yhat", "offered_line",
                "p_model_under", "p_market_under", "edge_under", "edge_over", "is_under"]
        print(curry[cols].tail(10).to_string(index=False))
        print(f"\n  avg yhat={curry['yhat'].mean():.2f}  avg pts_actual={curry['pts_actual'].mean():.2f}")
        print(f"  avg p_model_under={curry['p_model_under'].mean():.4f}  avg p_market_under={curry['p_market_under'].mean():.4f}")
        print(f"  avg edge_under={curry['edge_under'].mean():.4f}  avg edge_over={curry['edge_over'].mean():.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_parquet(OUT_DIR / "step4_edge.parquet", index=False)
    print(f"\nSaved → {OUT_DIR}/step4_edge.parquet  ({len(df):,} rows)")
    print(f"  Rows with edge_under > 0.05: {(df['edge_under']>0.05).sum():,}")
    print(f"  Rows with edge_over  > 0.05: {(df['edge_over']>0.05).sum():,}")

    print("\nDone.")


if __name__ == "__main__":
    main()

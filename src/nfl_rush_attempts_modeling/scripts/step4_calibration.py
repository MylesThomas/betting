"""
Step 4 — Outcome Distribution → Binary Bet Signal

For each player-game-book row:
  p_model  = P(actual_carries > book_line) derived from the OOF residual distribution
  p_market = no-vig book_over_prob (per-book, already computed in spine)
  edge     = p_model - p_market

P(over) approach:
  actual_carries = oof_carries + residual
  P(over) = P(residual > book_line - oof_carries) = P(residual > shortfall)

  Residuals are stratified by predicted carry bucket because the regression
  model has a known bias at extremes (under-predicts 20+ carry players,
  over-predicts 0-4 carry players). Using a global residual distribution
  would propagate that bias directly into P(over) estimates.

  Within each predicted carry bucket, the residual distribution is estimated
  via an empirical CDF (smoothed with Gaussian KDE for robustness on small buckets).

Output:
  ~/Downloads/tmp/rush_attempts/step4_bets.parquet
  ~/Downloads/tmp/rush_attempts/step4_calibration.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
OOF_PATH   = Path.home() / "Downloads" / "tmp" / "rush_attempts" / "oof_predictions.parquet"
TRAIN_PATH = Path.home() / "Downloads" / "tmp" / "rush_attempts" / "training.parquet"
OUT_DIR    = Path.home() / "Downloads" / "tmp" / "rush_attempts"

PRED_BINS   = [0, 5, 10, 15, 20, np.inf]
PRED_LABELS = ["lt5", "5to9", "10to14", "15to19", "20plus"]


# ── Residual distribution per predicted-carry bucket ─────────────────────────

def build_residual_cdfs(oof: pd.DataFrame) -> dict[str, callable]:
    """
    For each predicted carry bucket, fit a smoothed empirical CDF over OOF residuals.
    Returns a dict: bucket_label → callable cdf(x) = P(residual <= x).
    """
    df = oof[oof["oof_carries"].notna()].copy()
    df["residual"]    = df["carries"] - df["oof_carries"]
    df["pred_bucket"] = pd.cut(df["oof_carries"], bins=PRED_BINS,
                                labels=PRED_LABELS, right=False)

    cdfs = {}
    print("\nResidual distribution by predicted carry bucket:")
    print(f"  {'bucket':<10} {'n':>6}  {'mean_resid':>11}  {'std_resid':>10}  {'p10':>6}  {'p90':>6}")

    for label in PRED_LABELS:
        r = df[df["pred_bucket"] == label]["residual"].dropna().values
        if len(r) < 20:
            # Fall back to global distribution if bucket too sparse
            r = df["residual"].dropna().values

        mean_r, std_r = r.mean(), r.std()
        p10, p90 = np.percentile(r, 10), np.percentile(r, 90)
        print(f"  {label:<10} {len(r):>6}  {mean_r:>+11.3f}  {std_r:>10.3f}  {p10:>6.1f}  {p90:>6.1f}")

        # Smoothed ECDF via Gaussian KDE
        kde = stats.gaussian_kde(r, bw_method="scott")

        # Evaluate CDF via numerical integration on a fine grid
        x_min = r.min() - 3 * std_r
        x_max = r.max() + 3 * std_r
        xs = np.linspace(x_min, x_max, 500)
        pdf_vals = kde(xs)
        cdf_vals = np.cumsum(pdf_vals) * (xs[1] - xs[0])
        cdf_vals = np.clip(cdf_vals / cdf_vals[-1], 0, 1)  # normalize to [0,1]

        # Interpolated CDF callable — clips at [0,1] for out-of-range inputs
        cdf_fn = interp1d(xs, cdf_vals, kind="linear",
                          bounds_error=False, fill_value=(0.0, 1.0))
        cdfs[label] = cdf_fn

    return cdfs


def p_over_from_residual(shortfall: float, bucket: str,
                          cdfs: dict[str, callable]) -> float:
    """P(actual > book_line) = P(residual > shortfall) = 1 - CDF(shortfall)."""
    return float(1.0 - cdfs[bucket](shortfall))


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    # Load data
    oof   = pd.read_parquet(OOF_PATH)
    train = pd.read_parquet(TRAIN_PATH)

    print(f"OOF rows: {len(oof):,}  (with predictions: {oof['oof_carries'].notna().sum():,})")
    print(f"Training rows (per-book): {len(train):,}")

    # Build residual CDFs
    cdfs = build_residual_cdfs(oof)

    # Join oof_carries to per-book training set
    oof_slim = oof[["nfl_game_id", "player_name_norm", "oof_carries"]].copy()
    df = train.merge(oof_slim, on=["nfl_game_id", "player_name_norm"], how="inner")
    df = df[df["oof_carries"].notna()].reset_index(drop=True)
    print(f"\nAfter joining oof_carries: {len(df):,} per-book rows "
          f"({df[['nfl_game_id','player_name_norm']].drop_duplicates().shape[0]:,} unique player-games)")

    # Predicted carry bucket for each row
    df["pred_bucket"] = pd.cut(df["oof_carries"], bins=PRED_BINS,
                                labels=PRED_LABELS, right=False).astype(str)

    # Shortfall = how far the book line is above the model's prediction
    # Positive shortfall = model predicts fewer carries than the line → lean under
    df["shortfall"] = df["book_line"] - df["oof_carries"]

    # P(over) via stratified residual CDF
    print("\nComputing P(over) via stratified residual CDF...")
    p_model_vals = np.empty(len(df))
    for bucket in PRED_LABELS:
        mask = df["pred_bucket"] == bucket
        if mask.sum() == 0:
            continue
        shortfalls = df.loc[mask, "shortfall"].values
        p_model_vals[mask] = np.array([
            p_over_from_residual(s, bucket, cdfs) for s in shortfalls
        ])
    df["p_model"] = p_model_vals

    # Market implied probability (no-vig, per-book, already in spine)
    df["p_market"] = df["book_over_prob"]

    # Edge
    df["edge"] = df["p_model"] - df["p_market"]

    # ── Calibration check ────────────────────────────────────────────────────
    df["p_model_decile"] = pd.qcut(df["p_model"], q=10, labels=False, duplicates="drop")

    cal = (
        df.groupby("p_model_decile")
          .agg(
              avg_p_model  = ("p_model",  "mean"),
              actual_over  = ("is_over",  "mean"),
              n_rows       = ("is_over",  "count"),
          )
          .reset_index()
    )
    cal["calib_error"] = (cal["avg_p_model"] - cal["actual_over"]).abs()
    cal["pass"] = cal["calib_error"] < 0.15

    print("\nCalibration by P(over) decile:")
    print(f"  {'decile':>7}  {'avg_p_model':>12}  {'actual_over':>12}  "
          f"{'calib_err':>10}  {'n':>6}  {'pass':>6}")
    for _, row in cal.iterrows():
        flag = "PASS" if row["pass"] else "FAIL"
        print(f"  {int(row['p_model_decile']):>7}  {row['avg_p_model']:>12.4f}  "
              f"{row['actual_over']:>12.4f}  {row['calib_error']:>10.4f}  "
              f"{int(row['n_rows']):>6}  {flag:>6}")

    # ── Summary stats ─────────────────────────────────────────────────────────
    print(f"\nP(over) stats:")
    print(f"  min={df['p_model'].min():.3f}  max={df['p_model'].max():.3f}  "
          f"mean={df['p_model'].mean():.3f}  std={df['p_model'].std():.3f}")
    print(f"\nEdge stats:")
    print(f"  min={df['edge'].min():.3f}  max={df['edge'].max():.3f}  "
          f"mean={df['edge'].mean():.3f}  std={df['edge'].std():.3f}")
    print(f"\nP(market) stats:")
    print(f"  min={df['p_market'].min():.3f}  max={df['p_market'].max():.3f}  "
          f"mean={df['p_market'].mean():.3f}")

    # ── Distribution of edge by direction ─────────────────────────────────────
    print(f"\nEdge distribution:")
    for lo, hi in [(-1,-0.10),(-0.10,-0.05),(-0.05,0),(0,0.05),(0.05,0.10),(0.10,1)]:
        n = ((df['edge'] >= lo) & (df['edge'] < hi)).sum()
        print(f"  [{lo:+.2f}, {hi:+.2f}): {n:>5} rows ({n/len(df)*100:.1f}%)")

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_cols = [
        "nfl_game_id", "player_name_norm", "player_display_name",
        "bookmaker", "season", "week", "is_playoff",
        "position", "carries", "is_over",
        "book_line", "book_over_price", "book_under_price",
        "consensus_point", "n_books",
        "oof_carries", "shortfall", "pred_bucket",
        "p_model", "p_market", "edge",
    ]
    out = df[out_cols].copy()
    out.to_parquet(OUT_DIR / "step4_bets.parquet", index=False)
    cal.to_csv(OUT_DIR / "step4_calibration.csv", index=False)
    print(f"\nSaved {len(out):,} rows to step4_bets.parquet")
    print("=== Step 4 complete ===")


if __name__ == "__main__":
    run()

"""
MLB Pitcher Strikeouts — Step 4: Bootstrap P(over) → Edge
==========================================================
Uses OOF yhat from Step 3. For each player-game, bootstraps P(over) by
drawing N=10,000 samples from yhat + resample(OOF training residuals).
P(over) = fraction of draws > line.

Since this is a variable-line market, different lines for the same game
produce different P(over) estimates — each (player, game, book, line) row
is a genuinely distinct bet.

Edge = p_model_over - raw_p_over  (positive → bet OVER; raw_p = actual breakeven from posted odds)
Edge = p_model_under - raw_p_under (positive → bet UNDER)

Outputs:
  ~/Downloads/tmp/mlb_strikeouts/step4_edge.parquet
  ~/Downloads/tmp/mlb_strikeouts/step4_calibration.csv

Usage:
  python src/mlb_strikeouts_modeling/scripts/v4_edge.py
"""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

REPO_ROOT  = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR    = Path.home() / "Downloads/tmp/mlb_strikeouts"
S3_BUCKET  = "the-odds-api-mt"
LABELED_KEY = "mlb/strikeouts_model/labeled/mlb_strikeouts_labeled.parquet"
SPOT_CHECK = "paul skenes"
N_BOOT     = 10_000
RNG        = np.random.default_rng(42)


def bootstrap_p_over(yhat: np.ndarray, lines: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """P(strikeouts > line) for each (yhat, line) pair via bootstrap."""
    samples = RNG.choice(residuals, size=(len(yhat), N_BOOT), replace=True)
    sims    = yhat[:, None] + samples            # (n_rows, N_BOOT)
    return (sims > lines[:, None]).mean(axis=1)  # fraction where sim > line


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading OOF predictions...", flush=True)
    oof = pd.read_parquet(OUT_DIR / "step3_oof_predictions.parquet")
    print(f"  OOF rows: {len(oof):,}  |  seasons: {sorted(oof['fold'].unique())}")

    print("Loading OOF residuals...", flush=True)
    residuals = np.load(OUT_DIR / "step3_oof_residuals.npy")
    print(f"  Residuals: {len(residuals):,}  σ={residuals.std():.4f}  mean={residuals.mean():.4f}")

    # Clip residuals — anything > 5σ is a likely data artifact
    sigma = residuals.std()
    n_before = len(residuals)
    residuals = np.clip(residuals, -5 * sigma, 5 * sigma)
    print(f"  Clipped to +/-5sigma ({5*sigma:.1f}K): {n_before - len(residuals)} rows affected")

    print("Loading labeled dataset from S3...", flush=True)
    s3   = boto3.client("s3")
    body = s3.get_object(Bucket=S3_BUCKET, Key=LABELED_KEY)["Body"].read()
    lbl  = pd.read_parquet(BytesIO(body))
    print(f"  Labeled rows: {len(lbl):,}  |  events: {lbl['event_id'].nunique():,}  |  players: {lbl['player_id'].nunique()}")

    # Join OOF predictions to labeled rows on (player_key, game_date)
    oof_slim = oof[["player_key", "game_date", "yhat"]].drop_duplicates(subset=["player_key", "game_date"])
    lbl["game_date"] = lbl["game_date"].astype(str)
    oof_slim["game_date"] = oof_slim["game_date"].astype(str)
    df = lbl.merge(oof_slim, on=["player_key", "game_date"], how="inner")
    print(f"  After join with OOF: {len(df):,} rows  ({len(df)/len(lbl):.1%} of labeled)")

    # Drop rows missing market probability or yhat
    df = df.dropna(subset=["novig_over", "raw_p_over", "raw_p_under", "line", "yhat"])
    print(f"  After dropping nulls: {len(df):,} rows")

    # ── Bootstrap P(over) for each unique (player, game, line) ────────────────
    # IMPORTANT: deduplicate before bootstrapping so that p_model_over is
    # identical for all books on the same (player_key, game_date, line).
    # The bootstrap RNG is position-dependent; running it on per-book rows
    # would assign different random samples to each book → book-dependent
    # p_model_over, which would fail the book-independence invariant.
    unique_pgl = df.drop_duplicates(subset=["player_key", "game_date", "line"]).copy()
    print(f"\nBootstrapping P(over)  [N={N_BOOT:,} draws, {len(unique_pgl):,} unique (player,game,line) rows]...", flush=True)
    p_model_over_unique  = bootstrap_p_over(unique_pgl["yhat"].values, unique_pgl["line"].values, residuals)
    p_model_under_unique = 1.0 - p_model_over_unique

    unique_pgl["p_model_over"]  = np.clip(p_model_over_unique, 0.01, 0.99)
    unique_pgl["p_model_under"] = np.clip(p_model_under_unique, 0.01, 0.99)

    # Join p_model_over back to full df (all books) — ensures book independence
    df = df.merge(
        unique_pgl[["player_key", "game_date", "line", "p_model_over", "p_model_under"]],
        on=["player_key", "game_date", "line"],
        how="left",
    )
    p_model_over  = df["p_model_over"].values
    p_model_under = df["p_model_under"].values

    n_clip_high = (p_model_over >= 0.99).sum()
    n_clip_low  = (p_model_over <= 0.01).sum()
    print(f"  Clipping: {n_clip_high} rows at 0.99, {n_clip_low} rows at 0.01 "
          f"({(n_clip_high+n_clip_low)/len(df):.2%} of total)")

    # ── Market probabilities ───────────────────────────────────────────────────
    # raw_p = actual breakeven (1/decimal_odds) — this is what determines payout
    # novig_over retained for calibration and fav/dog classification only
    df["p_market_over"]  = df["raw_p_over"]
    df["p_market_under"] = df["raw_p_under"]

    # ── Edge ──────────────────────────────────────────────────────────────────
    df["edge_over"]  = df["p_model_over"]  - df["p_market_over"]
    df["edge_under"] = df["p_model_under"] - df["p_market_under"]

    print(f"\nP(model over) range:  [{df['p_model_over'].min():.4f}, {df['p_model_over'].max():.4f}]")
    print(f"avg p_model_over:  {df['p_model_over'].mean():.4f}  avg p_market_over (raw):  {df['p_market_over'].mean():.4f}")
    print(f"avg p_model_under: {df['p_model_under'].mean():.4f}  avg p_market_under (raw): {df['p_market_under'].mean():.4f}")
    print(f"avg novig_over:    {df['novig_over'].mean():.4f}  (retained for fav/dog filter)")
    print(f"\nEdge distribution:")
    print(f"  edge_over:  mean={df['edge_over'].mean():.4f}  std={df['edge_over'].std():.4f}")
    print(f"  edge_under: mean={df['edge_under'].mean():.4f}  std={df['edge_under'].std():.4f}")
    print(f"  % with edge_over  > 0.02: {(df['edge_over']>0.02).mean():.1%}")
    print(f"  % with edge_over  > 0.05: {(df['edge_over']>0.05).mean():.1%}")
    print(f"  % with edge_under > 0.02: {(df['edge_under']>0.02).mean():.1%}")
    print(f"  % with edge_under > 0.05: {(df['edge_under']>0.05).mean():.1%}")

    # ── Calibration: p_model_over deciles vs actual over rate ─────────────────
    df["p_over_decile"] = pd.qcut(df["p_model_over"], 10, labels=False, duplicates="drop")
    calib = (
        df.groupby("p_over_decile", observed=True)
        .agg(
            n=("is_over", "count"),
            avg_p_model=("p_model_over", "mean"),
            actual_over_rate=("is_over", "mean"),
        )
        .reset_index()
    )
    calib["calib_gap"]     = calib["actual_over_rate"] - calib["avg_p_model"]
    calib["calib_gap_pct"] = (calib["calib_gap"] * 100).round(1)
    calib["flag"]          = calib["calib_gap"].abs() > 0.15

    print(f"\nCalibration by p_model_over decile:")
    print(calib[["p_over_decile", "n", "avg_p_model", "actual_over_rate", "calib_gap_pct", "flag"]].to_string(index=False))

    flagged = calib[calib["flag"]]
    if len(flagged) > 0:
        print(f"\n  ⚠ {len(flagged)} deciles with |calib_gap| > 15pp — review for systematic bias")
    else:
        print(f"\n  All deciles within |calib_gap| ≤ 15pp ✓")

    calib.to_csv(OUT_DIR / "step4_calibration.csv", index=False)

    # ── Edge by line bucket ────────────────────────────────────────────────────
    df["line_bucket"] = pd.cut(
        df["line"],
        bins=[0, 3.5, 4.5, 5.5, 6.5, 7.5, 20],
        labels=["≤3.5", "4.5", "5.5", "6.5", "7.5", "≥8.5"],
    )
    print(f"\nEdge by consensus line bucket:")
    line_edge = (
        df.groupby("line_bucket", observed=True)
        .agg(
            n=("edge_over", "count"),
            avg_yhat=("yhat", "mean"),
            avg_line=("line", "mean"),
            avg_edge_over=("edge_over", "mean"),
            avg_edge_under=("edge_under", "mean"),
            actual_over_rate=("is_over", "mean"),
            avg_p_model_over=("p_model_over", "mean"),
            avg_p_market_over=("p_market_over", "mean"),
        )
        .reset_index()
    )
    print(line_edge.to_string(index=False))

    # ── Spot-check: Paul Skenes ───────────────────────────────────────────────
    print(f"\n── Spot-check: {SPOT_CHECK} ──")
    sc = df[df["player_key"] == SPOT_CHECK].sort_values("game_date")
    print(f"  OOF rows: {len(sc)}")
    if len(sc) > 0:
        cols = [c for c in ["game_date", "season", "strikeouts", "yhat", "line",
                "p_model_over", "p_market_over", "edge_over", "is_over"] if c in sc.columns]
        print(sc[cols].drop_duplicates(subset=["game_date", "line"]).head(15).to_string(index=False))
        print(f"\n  avg yhat={sc['yhat'].mean():.2f}  avg actual K={sc['strikeouts'].mean():.2f}")
        print(f"  avg p_model_over={sc['p_model_over'].mean():.4f}  avg p_market_over={sc['p_market_over'].mean():.4f}")
    else:
        print(f"  {SPOT_CHECK} not in OOF (probably only in 2026 test fold with 2024-only training)")

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_parquet(OUT_DIR / "step4_edge.parquet", index=False)
    print(f"\nSaved → {OUT_DIR}/step4_edge.parquet  ({len(df):,} rows)")
    print(f"  Rows with edge_over  > 0.05: {(df['edge_over']>0.05).sum():,}")
    print(f"  Rows with edge_under > 0.05: {(df['edge_under']>0.05).sum():,}")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
MLB Total Bases — Deduped Grid Search
======================================
Answers two questions:
  1. How does the strategy perform if we dedupe to one bet per (player, game, line)?
     Uses best available book (highest under_price) as the kept row.
  2. Does book-count concentration predict edge? Split bets by 1 / 2-4 / 5+ books
     at the canonical strategy (edge≥5pp, dogs, 1.5_only).

IS  = 2024 + 2025
OOS = 2026

Usage:
  python src/mlb_total_bases_modeling/scripts/20260724_grid_search_deduped.py
  python src/mlb_total_bases_modeling/scripts/20260724_grid_search_deduped.py --no-upload
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET    = "the-odds-api-mt"
SPINE_KEY    = "mlb/total_bases_model/regression/mlb_tb_reg_spine.parquet"
MODEL_KEY    = "mlb/total_bases_model/model/mlb_tb_regression_v2.joblib"
IS_OUT_KEY   = "mlb/total_bases_model/backtest/mlb_tb_v2_is_grid_deduped.csv"
OOS_OUT_KEY  = "mlb/total_bases_model/backtest/mlb_tb_v2_oos_grid_deduped.csv"
OUT_DIR      = Path.home() / "Downloads/tmp/mlb_total_bases"

TARGET      = "total_bases"
IS_SEASONS  = [2024, 2025]
OOS_SEASON  = 2026

EDGE_THRESHOLDS = [0.00, 0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
ODDS_BUCKETS    = ["all", "dogs (+odds)", "favs (-odds)"]
LINE_BUCKETS    = ["all_lines", "0.5_only", "1.5_only"]

# Canonical strategy used for concentration split
CANONICAL = dict(edge=0.05, odds="dogs (+odds)", lines="1.5_only")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, dict]:
    s3 = boto3.client("s3")

    print("Loading regression spine from S3 ...")
    body  = s3.get_object(Bucket=S3_BUCKET, Key=SPINE_KEY)["Body"].read()
    spine = pd.read_parquet(BytesIO(body))
    print(f"  {len(spine):,} rows")

    print("Loading model bundle from S3 ...")
    body   = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)["Body"].read()
    bundle = joblib.load(BytesIO(body))
    print(f"  Model: {bundle.get('combo_name','?')} | features: {bundle['features_numeric']}")
    return spine, bundle


# ── Feature engineering ───────────────────────────────────────────────────────

def prepare_spine(spine: pd.DataFrame) -> pd.DataFrame:
    df = spine.copy()
    df["season"]         = pd.to_datetime(df["game_date"]).dt.year
    df["raw_prob_under"] = 1.0 / df["under_price"]
    df["raw_prob_over"]  = 1.0 / df["over_price"]

    line_range = (
        df.groupby(["name_norm", "game_date"])["line"]
        .agg(min_line="min", max_line="max")
        .reset_index()
    )
    df = df.merge(line_range, on=["name_norm", "game_date"], how="left")
    return df


def score_spine(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    model        = bundle["model"]
    scaler       = bundle["scaler"]
    features     = bundle["features_numeric"]
    calib_models = bundle.get("calib_models", {})

    unique_pg = df[["name_norm", "game_date", "line"] + features].drop_duplicates(
        subset=["name_norm", "game_date", "line"]
    )
    unique_pg = unique_pg.dropna(subset=features)

    X     = unique_pg[features].values.astype(float)
    X_sc  = scaler.transform(X)
    y_hat = model.predict(X_sc).astype(float)
    unique_pg = unique_pg[["name_norm", "game_date", "line"]].copy()
    unique_pg["y_hat"] = y_hat

    p_model_rows = []
    for line, calib in calib_models.items():
        sub = unique_pg[unique_pg["line"] == line].copy()
        if sub.empty:
            continue
        proba = calib.predict_proba(sub["y_hat"].values.reshape(-1, 1))[:, 1]
        sub["p_model_over"]  = np.clip(proba, 0.01, 0.99)
        sub["p_model_under"] = 1.0 - sub["p_model_over"]
        p_model_rows.append(sub)

    if not p_model_rows:
        raise RuntimeError("No calibration models matched any line in spine")
    p_model_df = pd.concat(p_model_rows, ignore_index=True)

    df = df.merge(p_model_df[["name_norm", "game_date", "line", "y_hat", "p_model_under"]],
                  on=["name_norm", "game_date", "line"], how="inner")
    df["edge_under_raw"]   = df["p_model_under"] - df["raw_prob_under"]
    df["edge_under_novig"] = df["p_model_under"] - df["novig_prob_under"]
    return df


# ── Book-count / dedup ────────────────────────────────────────────────────────

def dedupe_best_book(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per (player, game, line) — highest under_price (best odds available)."""
    return (
        df.sort_values("under_price", ascending=False)
        .drop_duplicates(subset=["name_norm", "game_date", "line"], keep="first")
        .reset_index(drop=True)
    )


# ── Grid search ───────────────────────────────────────────────────────────────

def max_drawdown(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    cum  = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    return float(np.max(peak - cum))


def run_grid(df: pd.DataFrame, label: str) -> pd.DataFrame:
    records = []
    for edge_thresh in EDGE_THRESHOLDS:
        for odds_bucket in ODDS_BUCKETS:
            for line_bucket in LINE_BUCKETS:
                if line_bucket == "0.5_only":
                    sub = df[df["line"] == 0.5]
                elif line_bucket == "1.5_only":
                    sub = df[df["line"] == 1.5]
                else:
                    sub = df

                sub = sub[sub["edge_under_raw"] >= edge_thresh]

                if odds_bucket == "dogs (+odds)":
                    sub = sub[sub["raw_prob_under"] < 0.50]
                elif odds_bucket == "favs (-odds)":
                    sub = sub[sub["raw_prob_under"] >= 0.50]

                if sub.empty:
                    continue

                won    = (sub[TARGET] < sub["line"]).values.astype(float)
                payout = sub["under_price"].values - 1.0
                pnl    = np.where(won, payout, -1.0)

                records.append({
                    "edge_threshold":     edge_thresh,
                    "direction":          "UNDER",
                    "odds_bucket":        odds_bucket,
                    "line_bucket":        line_bucket,
                    "n_bets":             len(pnl),
                    "win_rate":           round(float(won.mean()), 4),
                    "units_won":          round(float(pnl.sum()), 2),
                    "roi":                round(float(pnl.mean()), 4),
                    "avg_raw_prob_under": round(float(sub["raw_prob_under"].mean()), 4),
                    "avg_edge_raw":       round(float(sub["edge_under_raw"].mean()), 4),
                    "max_drawdown":       round(max_drawdown(pnl), 2),
                })

    result = pd.DataFrame(records).sort_values("units_won", ascending=False).reset_index(drop=True)
    print(f"\n{label} grid — top 10 by units:")
    print(result.head(10)[
        ["edge_threshold", "odds_bucket", "line_bucket", "n_bets", "win_rate", "units_won", "roi", "avg_edge_raw"]
    ].to_string(index=False))
    return result


# ── Side-by-side comparison ───────────────────────────────────────────────────

def compare_raw_vs_deduped(raw_grid: pd.DataFrame, deduped_grid: pd.DataFrame, label: str) -> None:
    print(f"\n── {label}: raw vs deduped (dogs, 1.5_only) ──")
    print(f"{'edge':>6}  {'raw_n':>8}  {'raw_u':>8}  {'raw_roi':>8}  │  {'ded_n':>8}  {'ded_u':>8}  {'ded_roi':>8}")
    print("─" * 72)
    for edge in EDGE_THRESHOLDS:
        def _get(grid, edge):
            mask = (
                (grid["edge_threshold"] == edge) &
                (grid["odds_bucket"]    == "dogs (+odds)") &
                (grid["line_bucket"]    == "1.5_only")
            )
            row = grid[mask]
            if row.empty:
                return "—", "—", "—"
            r = row.iloc[0]
            return f"{int(r['n_bets']):,}", f"{r['units_won']:+.2f}", f"{r['roi']:+.4f}"

        rn, ru, rr = _get(raw_grid, edge)
        dn, du, dr = _get(deduped_grid, edge)
        print(f"{edge:>6.2f}  {rn:>8}  {ru:>8}  {rr:>8}  │  {dn:>8}  {du:>8}  {dr:>8}")


# ── Concentration split ───────────────────────────────────────────────────────

def concentration_split(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    At canonical strategy params, split bets by n_books bucket and report stats.
    Returns a DataFrame with the split results.
    """
    sub = df[df["line"] == 1.5].copy()
    sub = sub[sub["edge_under_raw"] >= CANONICAL["edge"]]
    sub = sub[sub["raw_prob_under"] < 0.50]

    if sub.empty:
        print(f"\n{label}: no bets at canonical params")
        return pd.DataFrame()

    def _bucket(n):
        if n == 1:
            return "1 book"
        elif n <= 4:
            return "2–4 books"
        else:
            return "5+ books"

    sub = sub.copy()
    sub["bucket"] = sub["n_books"].map(_bucket)

    bucket_order = ["1 book", "2–4 books", "5+ books", "All"]
    records = []
    for bucket in bucket_order[:-1]:
        b = sub[sub["bucket"] == bucket]
        if b.empty:
            continue
        won    = (b[TARGET] < b["line"]).values.astype(float)
        payout = b["under_price"].values - 1.0
        pnl    = np.where(won, payout, -1.0)
        records.append({
            "bucket":   bucket,
            "n_bets":   len(pnl),
            "win_rate": round(float(won.mean()), 4),
            "units":    round(float(pnl.sum()), 2),
            "roi":      round(float(pnl.mean()), 4),
            "avg_n_books": round(float(b["n_books"].mean()), 1),
        })

    # All bucket
    won    = (sub[TARGET] < sub["line"]).values.astype(float)
    payout = sub["under_price"].values - 1.0
    pnl    = np.where(won, payout, -1.0)
    records.append({
        "bucket":   "All",
        "n_bets":   len(pnl),
        "win_rate": round(float(won.mean()), 4),
        "units":    round(float(pnl.sum()), 2),
        "roi":      round(float(pnl.mean()), 4),
        "avg_n_books": round(float(sub["n_books"].mean()), 1),
    })

    result = pd.DataFrame(records)
    print(f"\n── {label}: concentration split (edge≥5pp, dogs, 1.5_only) ──")
    print(result.to_string(index=False))
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    spine, bundle = load_data()
    df = prepare_spine(spine)
    print(f"\nSeason distribution: {df['season'].value_counts().sort_index().to_dict()}")

    print("\nScoring spine ...")
    df = score_spine(df, bundle)
    print(f"  Scored {len(df):,} rows")

    print("\nBook count stats (from spine) ...")
    print(f"  avg n_books per row: {df['n_books'].mean():.2f}")
    print(f"  n_books distribution:\n{df['n_books'].value_counts().sort_index().head(10)}")

    df_deduped = dedupe_best_book(df)
    print(f"\n  Raw rows: {len(df):,} → Deduped rows: {len(df_deduped):,} ({len(df_deduped)/len(df)*100:.1f}%)")

    is_raw      = df[df["season"].isin(IS_SEASONS)].copy()
    oos_raw     = df[df["season"] == OOS_SEASON].copy()
    is_deduped  = df_deduped[df_deduped["season"].isin(IS_SEASONS)].copy()
    oos_deduped = df_deduped[df_deduped["season"] == OOS_SEASON].copy()

    print(f"\n  IS  raw: {len(is_raw):,}  deduped: {len(is_deduped):,}")
    print(f"  OOS raw: {len(oos_raw):,}  deduped: {len(oos_deduped):,}")

    # Run grids
    is_raw_grid      = run_grid(is_raw,      f"IS  raw ({', '.join(str(s) for s in IS_SEASONS)})")
    is_deduped_grid  = run_grid(is_deduped,  f"IS  deduped ({', '.join(str(s) for s in IS_SEASONS)})")
    oos_raw_grid     = run_grid(oos_raw,     f"OOS raw ({OOS_SEASON})")
    oos_deduped_grid = run_grid(oos_deduped, f"OOS deduped ({OOS_SEASON})")

    # Side-by-side comparisons
    compare_raw_vs_deduped(is_raw_grid,  is_deduped_grid,  "IS")
    compare_raw_vs_deduped(oos_raw_grid, oos_deduped_grid, "OOS")

    # Concentration split (on raw — n_books is meaningful here)
    is_conc  = concentration_split(is_raw,  "IS")
    oos_conc = concentration_split(oos_raw, "OOS")

    # Save CSVs locally
    is_path  = OUT_DIR / "mlb_tb_v2_is_grid_deduped.csv"
    oos_path = OUT_DIR / "mlb_tb_v2_oos_grid_deduped.csv"
    is_deduped_grid.to_csv(is_path, index=False)
    oos_deduped_grid.to_csv(oos_path, index=False)

    is_conc_path  = OUT_DIR / "mlb_tb_v2_is_concentration_split.csv"
    oos_conc_path = OUT_DIR / "mlb_tb_v2_oos_concentration_split.csv"
    if not is_conc.empty:
        is_conc.to_csv(is_conc_path, index=False)
    if not oos_conc.empty:
        oos_conc.to_csv(oos_conc_path, index=False)

    print(f"\nSaved locally → {OUT_DIR}/")

    if not args.no_upload:
        s3 = boto3.client("s3")
        s3.put_object(Bucket=S3_BUCKET, Key=IS_OUT_KEY,  Body=is_deduped_grid.to_csv(index=False).encode())
        s3.put_object(Bucket=S3_BUCKET, Key=OOS_OUT_KEY, Body=oos_deduped_grid.to_csv(index=False).encode())
        print(f"Uploaded to s3://{S3_BUCKET}/mlb/total_bases_model/backtest/")

    # Return results dict for HTML generation
    return {
        "is_raw_grid":      is_raw_grid,
        "oos_raw_grid":     oos_raw_grid,
        "is_deduped_grid":  is_deduped_grid,
        "oos_deduped_grid": oos_deduped_grid,
        "is_conc":          is_conc,
        "oos_conc":         oos_conc,
        "raw_rows":         len(df),
        "deduped_rows":     len(df_deduped),
        "is_raw_rows":      len(is_raw),
        "oos_raw_rows":     len(oos_raw),
        "is_deduped_rows":  len(is_deduped),
        "oos_deduped_rows": len(oos_deduped),
    }


if __name__ == "__main__":
    main()

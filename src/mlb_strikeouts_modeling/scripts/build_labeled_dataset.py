"""
Build labeled dataset for pitcher strikeouts model training.

Joins the rolling-feature spine with market data to produce one row per
pitcher-start-book-line, labeled with over/under outcome.

For variable-line markets: each (player, game, book, line) is a distinct bet
with its own implied probability. These are kept as separate rows.

Output schema (key columns):
  player_id, player_name, season, game_date, game_pk,
  strikeouts (actual), consensus_line, line (per-book), bookmaker,
  over_price, under_price, novig_prob_over,
  outcome (over/under), is_over (binary),
  + all rolling feature columns from spine

Output paths:
  Local: ~/Downloads/tmp/mlb_strikeouts_labeled.parquet
  S3:    s3://the-odds-api-mt/mlb/strikeouts_model/labeled/mlb_strikeouts_labeled.parquet

Usage:
  python src/mlb_strikeouts_modeling/scripts/build_labeled_dataset.py
"""
from __future__ import annotations

import re
import sys
import unicodedata
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

SPINE_KEY     = "mlb/strikeouts_model/spine/mlb_strikeouts_spine.parquet"
MARKET_BUCKET = "the-odds-api-mt"
MARKET_PREFIX = "mlb/strikeouts_model/market_raw"
OUT_KEY       = "mlb/strikeouts_model/labeled/mlb_strikeouts_labeled.parquet"
LOCAL_OUT     = Path.home() / "Downloads/tmp/mlb_strikeouts_labeled.parquet"

SEASONS       = [2024, 2025, 2026]

NAME_MAP = {
    "louie varland": "louis varland",
}


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", name.strip().lower())
    name = re.sub(r"\s+", " ", name).strip()
    return NAME_MAP.get(name, name)


def load_spine() -> pd.DataFrame:
    s3 = boto3.client("s3")
    body = s3.get_object(Bucket=MARKET_BUCKET, Key=SPINE_KEY)["Body"].read()
    return pd.read_parquet(BytesIO(body))


def load_market_flat() -> pd.DataFrame:
    """Load all per-book market rows (not aggregated to consensus)."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    frames = []
    for season in SEASONS:
        prefix = f"{MARKET_PREFIX}/{season}/"
        for page in paginator.paginate(Bucket=MARKET_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                body = s3.get_object(Bucket=MARKET_BUCKET, Key=obj["Key"])["Body"].read()
                frames.append(pd.read_parquet(BytesIO(body)))
    return pd.concat(frames, ignore_index=True)


def main():
    print("Loading spine...")
    spine = load_spine()
    print(f"  {len(spine):,} rows, join rate: {spine['consensus_line'].notna().mean():.1%}")

    print("Loading market data (flat, per-book)...")
    mkt = load_market_flat()
    print(f"  {len(mkt):,} rows, {mkt['event_id'].nunique()} events")

    # Use main market only for labeled dataset (alt lines lack under prices)
    mkt = mkt[mkt["market_key"] == "pitcher_strikeouts"].copy()
    mkt = mkt[mkt["over_price"].notna() & mkt["under_price"].notna()].copy()

    # Compute per-book no-vig probability.
    # Prices from the Odds API are decimal odds (e.g. 1.91 = -110 American).
    # Use directly — do NOT run through american_to_decimal.
    mkt["player_key"]  = mkt["player_name"].apply(normalize_name)
    mkt["dec_over"]    = mkt["over_price"]   # already decimal
    mkt["dec_under"]   = mkt["under_price"]  # already decimal
    mkt["raw_p_over"]  = 1 / mkt["dec_over"]
    mkt["raw_p_under"] = 1 / mkt["dec_under"]
    mkt["novig_over"]  = mkt["raw_p_over"] / (mkt["raw_p_over"] + mkt["raw_p_under"])

    # ── v5 New Features: consensus line odds bins + line/prob range ───────────
    # Decimal → American conversion
    def dec_to_am(dec: float) -> float:
        return (dec - 1) * 100 if dec >= 2.0 else -100.0 / (dec - 1)

    # Step 1: consensus line odds (only rows where line == modal line)
    modal_line = (
        mkt.groupby(["player_key", "game_date"])["line"]
        .agg(lambda x: x.mode().iloc[0])
        .rename("consensus_line_computed")
        .reset_index()
    )
    cl_mkt = mkt.merge(modal_line, on=["player_key", "game_date"])
    cl_mkt = cl_mkt[cl_mkt["line"] == cl_mkt["consensus_line_computed"]].copy()
    cl_mkt["am_over"]  = cl_mkt["dec_over"].apply(dec_to_am)
    cl_mkt["am_under"] = cl_mkt["dec_under"].apply(dec_to_am)

    cl_agg = (
        cl_mkt.groupby(["player_key", "game_date"])
        .agg(
            avg_over_odds_am=("am_over", "mean"),
            avg_under_odds_am=("am_under", "mean"),
        )
        .reset_index()
    )

    def simple_bin(odds: float) -> int:
        if odds < -100:
            return 0   # favorite
        if odds <= 100:
            return 1   # pick'em
        return 2       # underdog

    def granular_bin(odds: float) -> int:
        if odds < -300:  return 0
        if odds < -200:  return 1
        if odds < -110:  return 2
        if odds < -100:  return 3
        if odds <= 100:  return 4
        if odds <= 200:  return 5
        if odds <= 300:  return 6
        if odds <= 500:  return 7
        return 8

    cl_agg["over_price_bucket"]       = cl_agg["avg_over_odds_am"].apply(simple_bin)
    cl_agg["under_price_bucket"]      = cl_agg["avg_under_odds_am"].apply(simple_bin)
    cl_agg["over_price_bucket_fine"]  = cl_agg["avg_over_odds_am"].apply(granular_bin)
    cl_agg["under_price_bucket_fine"] = cl_agg["avg_under_odds_am"].apply(granular_bin)

    # Step 2: line range and raw implied prob range (all lines/books)
    range_agg = (
        mkt.groupby(["player_key", "game_date"])
        .agg(
            min_line=("line", "min"),
            max_line=("line", "max"),
            min_over_prob=("raw_p_over", "min"),
            max_over_prob=("raw_p_over", "max"),
            min_under_prob=("raw_p_under", "min"),
            max_under_prob=("raw_p_under", "max"),
        )
        .reset_index()
    )

    # ── Join v5 features into mkt before the main spine merge ────────────────
    mkt = mkt.merge(
        cl_agg[["player_key", "game_date",
                "over_price_bucket", "under_price_bucket",
                "over_price_bucket_fine", "under_price_bucket_fine"]],
        on=["player_key", "game_date"],
        how="left",
    )
    mkt = mkt.merge(range_agg, on=["player_key", "game_date"], how="left")

    # Join spine rolling features to market rows
    spine_cols = [
        "player_key", "game_date", "strikeouts",
        "k_roll_s1", "k_roll_s3", "k_roll_s5", "k_roll_s10", "k_roll_s20",
        "k_roll_season", "k_roll_career",
        "k_roll_c1", "k_roll_c3", "k_roll_c5", "k_roll_c10", "k_roll_c20",
        "ip_roll_season", "opp_k_against_season",
        "is_home", "days_rest", "game_month", "season", "player_id",
        "player_name", "is_short_start", "consensus_line",
    ]
    spine_slim = spine[spine_cols].copy()

    labeled = mkt.merge(
        spine_slim,
        on=["player_key", "game_date"],
        how="inner",
    )

    print(f"  Joined rows: {len(labeled):,}")

    # Label outcomes
    labeled["outcome"] = np.where(
        labeled["strikeouts"] > labeled["line"], "over",
        np.where(labeled["strikeouts"] == labeled["line"], "push", "under")
    )
    labeled["is_over"] = (labeled["outcome"] == "over").astype(int)

    # Filter out pushes (lines are half-integers so this should be 0 rows)
    n_push = (labeled["outcome"] == "push").sum()
    print(f"  Pushes: {n_push} (expected 0 — all lines are .5)")
    labeled = labeled[labeled["outcome"] != "push"].copy()

    print(f"  Final rows: {len(labeled):,}")
    print(f"  Over rate: {labeled['is_over'].mean():.1%}")
    print(f"  Players: {labeled['player_id'].nunique()}")
    print(f"  Date range: {labeled['game_date'].min()} → {labeled['game_date'].max()}")

    # ── v5 Feature Summary ────────────────────────────────────────────────────
    v5_cols = [
        "over_price_bucket", "under_price_bucket",
        "over_price_bucket_fine", "under_price_bucket_fine",
        "min_line", "max_line",
        "min_over_prob", "max_over_prob",
        "min_under_prob", "max_under_prob",
    ]
    print("\n── v5 Feature Null Summary ──")
    for col in v5_cols:
        if col in labeled.columns:
            n_null = labeled[col].isna().sum()
            print(f"  {col:<30}: {n_null:,} nulls ({n_null/len(labeled):.2%})")
        else:
            print(f"  {col:<30}: NOT FOUND in labeled dataset")

    print("\n── over_price_bucket distribution (0=fav, 1=pick'em, 2=dog) ──")
    if "over_price_bucket" in labeled.columns:
        print(labeled["over_price_bucket"].value_counts().sort_index().to_string())

    print("\n── under_price_bucket distribution ──")
    if "under_price_bucket" in labeled.columns:
        print(labeled["under_price_bucket"].value_counts().sort_index().to_string())

    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(LOCAL_OUT, index=False)
    print(f"\nSaved locally → {LOCAL_OUT}")

    s3 = boto3.client("s3")
    buf = BytesIO()
    labeled.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=MARKET_BUCKET, Key=OUT_KEY, Body=buf.getvalue())
    print(f"Uploaded → s3://{MARKET_BUCKET}/{OUT_KEY}")


if __name__ == "__main__":
    main()

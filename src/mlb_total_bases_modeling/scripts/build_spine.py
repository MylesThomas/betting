"""
Build the training spine for the MLB batter total bases model.

For each player-game in the market data, computes rolling features
(strictly prior to that game), joins to market lines/odds, and
produces one row per player-game-line.

Output schema (key columns):
  player_name, game_date, season, line, team, opponent, is_home,
  total_bases, hit_over,
  novig_prob_over, novig_prob_under, avg_raw_prob_over, avg_raw_prob_under, n_books,
  tb_L1, tb_L3, tb_L5, tb_L10, tb_L20, tb_Lseason, tb_Lcareer,
  hr_L5, hr_L10, hr_Lcareer,
  ab_L5, ab_L10, ab_Lcareer,
  hits_L5, hits_L10,
  days_rest, games_played_career

Usage:
  python src/mlb_total_bases_modeling/scripts/build_spine.py
  python src/mlb_total_bases_modeling/scripts/build_spine.py --no-s3
"""
from __future__ import annotations

import argparse
import os
import sys
import unicodedata
import re
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET   = "the-odds-api-mt"
S3_KEY_ACT  = "mlb/total_bases_model/actuals/mlb_batting_statcast.parquet"
S3_KEY_MKT  = "mlb/total_bases_model/market_raw/mlb_total_bases_market_raw.parquet"
S3_KEY_OUT  = "mlb/total_bases_model/spine/mlb_total_bases_spine.parquet"
LOCAL_ACT   = Path.home() / "Downloads/tmp/mlb_batting_statcast.parquet"
LOCAL_MKT   = Path.home() / "Downloads/tmp/mlb_total_bases_market_raw.parquet"
LOCAL_OUT   = Path.home() / "Downloads/tmp/mlb_total_bases_spine.parquet"

MANUAL_MAP = {
    "daniel vogelbach":    "Dan Vogelbach",
    "michael a taylor":    "Michael Taylor",
    "max muncy (2002)":    "Max Muncy",
    "diego a castillo":    "Diego Castillo",
    "james jarvis":        "Jim Jarvis",
    "donnie walton":       "Donovan Walton",
    "josh kuroda-grauer":  "Joshua Kuroda-Grauer",
}

ROLLING_WINDOWS = [1, 3, 5, 10, 20]


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[.,'\-]", "", name)
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", name)
    name = re.sub(r"\s+", "", name)
    return name.strip()


def build_rolling_features(actuals: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-player rolling features, strictly prior to each game.
    One row per player-game.
    """
    actuals = actuals.copy()
    actuals["game_date"] = pd.to_datetime(actuals["game_date"])
    actuals = actuals.sort_values(["name_norm", "game_date"]).reset_index(drop=True)

    frames = []
    for player, grp in actuals.groupby("name_norm", sort=False):
        grp = grp.sort_values("game_date").reset_index(drop=True)
        grp["games_played_career"] = range(len(grp))  # 0-indexed: games BEFORE this row

        # Window rolling (shift(1) so current game not included)
        for w in ROLLING_WINDOWS:
            grp[f"tb_L{w}"]   = grp["total_bases"].shift(1).rolling(w, min_periods=1).mean()
            grp[f"hr_L{w}"]   = grp["home_runs"].shift(1).rolling(w, min_periods=1).mean()
            grp[f"ab_L{w}"]   = grp["ab"].shift(1).rolling(w, min_periods=1).mean()
            grp[f"hits_L{w}"] = grp["hits"].shift(1).rolling(w, min_periods=1).mean()

        # Career rolling (all prior games)
        grp["tb_Lcareer"]   = grp["total_bases"].shift(1).expanding().mean()
        grp["hr_Lcareer"]   = grp["home_runs"].shift(1).expanding().mean()
        grp["ab_Lcareer"]   = grp["ab"].shift(1).expanding().mean()

        # Season rolling (reset each season)
        grp["tb_Lseason"] = (
            grp.groupby("season")["total_bases"]
            .transform(lambda s: s.shift(1).expanding().mean())
        )
        grp["hr_Lseason"] = (
            grp.groupby("season")["home_runs"]
            .transform(lambda s: s.shift(1).expanding().mean())
        )

        # Days rest
        grp["days_rest"] = grp["game_date"].diff().dt.days.fillna(0).astype(int)

        frames.append(grp)

    return pd.concat(frames, ignore_index=True)


def build_market_consensus(market: pd.DataFrame) -> pd.DataFrame:
    """
    From the standard market (batter_total_bases) with both over+under prices,
    compute per-player-game-line consensus novig probabilities.
    """
    mkt = market[
        (market["market_key"] == "batter_total_bases")
        & market["over_price"].notna()
        & market["under_price"].notna()
        & (market["over_price"] > 1.0)
        & (market["under_price"] > 1.0)
    ].copy()

    mkt["raw_prob_over"]  = 1.0 / mkt["over_price"]
    mkt["raw_prob_under"] = 1.0 / mkt["under_price"]
    mkt["total_prob"]     = mkt["raw_prob_over"] + mkt["raw_prob_under"]
    mkt["novig_over"]     = mkt["raw_prob_over"]  / mkt["total_prob"]
    mkt["novig_under"]    = mkt["raw_prob_under"] / mkt["total_prob"]

    consensus = (
        mkt.groupby(["name_norm", "game_date", "line"])
        .agg(
            avg_raw_prob_over  = ("raw_prob_over",  "mean"),
            avg_raw_prob_under = ("raw_prob_under", "mean"),
            novig_prob_over    = ("novig_over",     "mean"),
            novig_prob_under   = ("novig_under",    "mean"),
            n_books            = ("bookmaker",      "nunique"),
            event_id           = ("event_id",       "first"),
            home_team          = ("home_team",       "first"),
            away_team          = ("away_team",       "first"),
        )
        .reset_index()
    )
    return consensus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-s3", action="store_true")
    args = parser.parse_args()

    print("Loading actuals …")
    actuals = pd.read_parquet(LOCAL_ACT)
    actuals = actuals[actuals["ab"] >= 1].copy()  # exclude DNP rows

    # Normalize + manual map
    manual_norm = {normalize_name(k): normalize_name(v) for k, v in MANUAL_MAP.items()}
    actuals["name_norm"] = actuals["player_name"].map(normalize_name).map(
        lambda n: manual_norm.get(n, n)
    )

    print(f"  {len(actuals):,} batter-games (AB≥1), {actuals['name_norm'].nunique():,} players")

    # Aggregate doubleheaders: market props cover the full day, not per-game.
    # Sum counting stats; count game_pks to flag doubleheaders.
    sum_cols  = ["total_bases", "home_runs", "singles", "doubles", "triples", "ab", "hits"]
    meta_cols = ["player_name", "season", "team", "opponent"]  # name_norm is a groupby key
    actuals = (
        actuals.groupby(["name_norm", "game_date"], sort=False)
        .agg(
            **{c: (c, "sum") for c in sum_cols if c in actuals.columns},
            **{c: (c, "first") for c in meta_cols if c in actuals.columns},
            is_doubleheader=("game_pk", "count"),
        )
        .reset_index()
    )
    actuals["is_doubleheader"] = (actuals["is_doubleheader"] > 1).astype(int)
    dh_count = actuals["is_doubleheader"].sum()
    print(f"  After doubleheader aggregation: {len(actuals):,} player-dates ({dh_count:,} doubleheader days)")

    print("Building rolling features …")
    feat = build_rolling_features(actuals)
    print(f"  {len(feat):,} rows with rolling features")

    print("Loading market data …")
    market = pd.read_parquet(LOCAL_MKT)
    market["game_date"] = market["game_date"].astype(str)
    market["name_norm"] = market["player_name"].map(normalize_name).map(
        lambda n: manual_norm.get(n, n)
    )

    print("Computing market consensus per player-game-line …")
    consensus = build_market_consensus(market)
    print(f"  {len(consensus):,} unique player-game-line obs")

    # min_line/max_line per player-game (v2 model features)
    line_stats = (
        consensus.groupby(["name_norm", "game_date"])["line"]
        .agg(min_line="min", max_line="max")
        .reset_index()
    )
    consensus = consensus.merge(line_stats, on=["name_norm", "game_date"], how="left")

    # Join features to consensus
    feat["game_date"] = feat["game_date"].dt.strftime("%Y-%m-%d")
    feat_cols = (
        ["name_norm", "game_date", "player_name", "season", "team", "opponent",
         "total_bases", "home_runs", "ab", "hits",
         "games_played_career", "days_rest", "is_doubleheader",
         "tb_Lcareer", "hr_Lcareer", "ab_Lcareer", "tb_Lseason", "hr_Lseason"]
        + [f"tb_L{w}" for w in ROLLING_WINDOWS]
        + [f"hr_L{w}" for w in ROLLING_WINDOWS]
        + [f"ab_L{w}" for w in ROLLING_WINDOWS]
        + [f"hits_L{w}" for w in ROLLING_WINDOWS]
    )
    spine = consensus.merge(feat[feat_cols], on=["name_norm", "game_date"], how="inner")

    spine["is_home"] = (spine["team"] == spine["home_team"]).astype(int)

    # Targets
    spine["actual_total_bases"] = spine["total_bases"]
    spine["hit_over"]           = (spine["total_bases"] > spine["line"]).astype(int)
    spine["hit_under"]          = (spine["total_bases"] < spine["line"]).astype(int)

    # Drop doubleheader rows from training spine.
    # The market posts separate single-game props for each DH game, but actuals are
    # summed across both games — comparing single-game prices vs two-game totals
    # inflates the apparent over rate by ~28pp. Model trains on single-game data only.
    # is_doubleheader is kept as a column so inference code can skip DH days.
    dh_rows = spine["is_doubleheader"].sum()
    spine_clean = spine[spine["is_doubleheader"] == 0].copy()
    print(f"  Dropped {dh_rows:,} doubleheader rows (single-game market vs two-game actuals mismatch)")

    # Drop the raw cols not needed downstream
    spine_clean = spine_clean.drop(columns=["home_team", "away_team"])
    spine = spine_clean

    print(f"\nSpine built: {len(spine):,} rows")
    print(f"Unique players: {spine['player_name'].nunique():,}")
    print(f"Unique events: {spine['event_id'].nunique():,}")
    print(f"Lines: {sorted(spine['line'].unique())}")
    print(f"Date range: {spine['game_date'].min()} → {spine['game_date'].max()}")
    print(f"Null rates in key feature cols:")
    for col in ["tb_L5", "tb_L10", "tb_Lcareer", "tb_Lseason", "days_rest", "novig_prob_over"]:
        n = spine[col].isna().sum()
        print(f"  {col}: {n:,} ({100*n/len(spine):.1f}%)")

    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    spine.to_parquet(LOCAL_OUT, index=False)
    print(f"\nSaved locally → {LOCAL_OUT}")

    if not args.no_s3:
        buf = BytesIO()
        spine.to_parquet(buf, index=False)
        boto3.client("s3").put_object(Bucket=S3_BUCKET, Key=S3_KEY_OUT, Body=buf.getvalue())
        print(f"Saved to S3 → s3://{S3_BUCKET}/{S3_KEY_OUT}")


if __name__ == "__main__":
    main()

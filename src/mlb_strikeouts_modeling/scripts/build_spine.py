"""
Build rolling feature spine for MLB pitcher strikeouts model.

For each pitcher-game, computes rolling K totals and rates at multiple windows,
opponent team K-against rate, home/away flag, days rest, and joins with market
implied probability from Odds API data.

No lookahead: rolling features at game G use strictly prior starts.

Inputs:
  Game logs:  s3://the-odds-api-mt/mlb/strikeouts_model/pitcher_gamelogs/{season}.parquet
  Market:     s3://the-odds-api-mt/mlb/strikeouts_model/market_raw/{season}/{event_id}.parquet

Outputs:
  S3:   s3://the-odds-api-mt/mlb/strikeouts_model/spine/mlb_strikeouts_spine.parquet
  Local: ~/Downloads/tmp/mlb_strikeouts_spine.parquet

Usage:
  python src/mlb_strikeouts_modeling/scripts/build_spine.py
  python src/mlb_strikeouts_modeling/scripts/build_spine.py --verify  # print summary only, no upload
"""
from __future__ import annotations

import argparse
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

GAMELOG_BUCKET = "the-odds-api-mt"
GAMELOG_PREFIX = "mlb/strikeouts_model/pitcher_gamelogs"
MARKET_BUCKET  = "the-odds-api-mt"
MARKET_PREFIX  = "mlb/strikeouts_model/market_raw"
SPINE_BUCKET   = "the-odds-api-mt"
SPINE_KEY      = "mlb/strikeouts_model/spine/mlb_strikeouts_spine.parquet"
LOCAL_OUT      = Path.home() / "Downloads/tmp/mlb_strikeouts_spine.parquet"

SEASONS        = [2024, 2025, 2026]

# Rolling windows in number of prior starts
ROLL_WINDOWS   = [1, 3, 5, 10, 20]

# Static name map: Odds API name → MLB Stats API name (after normalize)
NAME_MAP = {
    "louie varland": "louis varland",
}

# Team name normalization (Oakland moved/renamed)
TEAM_MAP = {
    "athletics": "oakland athletics",
}


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", name.strip().lower())
    name = re.sub(r"\s+", " ", name).strip()
    return NAME_MAP.get(name, name)


def normalize_team(name: str) -> str:
    n = str(name).strip().lower()
    return TEAM_MAP.get(n, n)


def load_gamelogs() -> pd.DataFrame:
    s3 = boto3.client("s3")
    frames = []
    for season in SEASONS:
        key = f"{GAMELOG_PREFIX}/{season}.parquet"
        body = s3.get_object(Bucket=GAMELOG_BUCKET, Key=key)["Body"].read()
        df = pd.read_parquet(BytesIO(body))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_market() -> pd.DataFrame:
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


def build_market_consensus(df_mkt: pd.DataFrame) -> pd.DataFrame:
    """
    For each (player_name_key, game_date), compute:
    - consensus_line: modal line across books (main market only)
    - novig_prob_over: avg no-vig P(over) using books with both sides
    - n_books: number of books contributing to consensus
    """
    main = df_mkt[df_mkt["market_key"] == "pitcher_strikeouts"].copy()
    main = main[main["over_price"].notna() & main["under_price"].notna()].copy()

    if main.empty:
        return pd.DataFrame()

    # Prices from the Odds API are decimal odds — use directly.
    main["player_key"] = main["player_name"].apply(normalize_name)
    main["dec_over"]   = main["over_price"]   # already decimal
    main["dec_under"]  = main["under_price"]  # already decimal
    main["raw_p_over"] = 1 / main["dec_over"]
    main["raw_p_under"]= 1 / main["dec_under"]
    main["novig_over"] = main["raw_p_over"] / (main["raw_p_over"] + main["raw_p_under"])

    # Consensus line: mode across books for each player-game
    # Use only the standard lines (not alt) for the consensus
    line_mode = (
        main.groupby(["player_key", "game_date"])["line"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.median())
        .reset_index()
        .rename(columns={"line": "consensus_line"})
    )

    # Avg novig P(over) across books at consensus line
    # Filter to rows at (or near) the consensus line per player-game
    main2 = main.merge(line_mode, on=["player_key", "game_date"])
    at_line = main2[main2["line"] == main2["consensus_line"]]

    market_agg = (
        at_line.groupby(["player_key", "game_date"]).agg(
            consensus_line=("consensus_line", "first"),
            novig_prob_over=("novig_over", "mean"),
            min_line=("line", "min"),
            max_line=("line", "max"),
            n_books=("bookmaker", "nunique"),
        ).reset_index()
    )
    return market_agg


def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling K features at pitcher level (season-scoped and career-scoped).
    Shift(1) ensures no lookahead.
    """
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    def ip_to_decimal(ip):
        try:
            ip = float(ip)
            whole = int(ip)
            thirds = round((ip - whole) * 10)
            return whole + thirds / 3
        except Exception:
            return np.nan

    df["ip_decimal"] = df["innings_pitched"].apply(ip_to_decimal)
    df["is_short_start"] = (df["ip_decimal"] < 3).astype(int)

    # --- Season-scoped rolling (resets each season) ---
    grp_season = df.groupby(["player_id", "season"], sort=False)
    for w in ROLL_WINDOWS:
        df[f"k_roll_s{w}"] = (
            grp_season["strikeouts"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )

    df["k_roll_season"] = (
        grp_season["strikeouts"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df["ip_roll_season"] = (
        grp_season["ip_decimal"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df["start_num_season"] = (
        grp_season["strikeouts"]
        .transform(lambda x: x.shift(1).expanding().count())
    )

    # --- Career-scoped rolling ---
    grp_career = df.groupby("player_id", sort=False)
    for w in ROLL_WINDOWS:
        df[f"k_roll_c{w}"] = (
            grp_career["strikeouts"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
    df["k_roll_career"] = (
        grp_career["strikeouts"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # --- Days rest since last start ---
    df["prev_date"] = grp_career["game_date"].transform(lambda x: x.shift(1))
    df["days_rest"] = (df["game_date"] - df["prev_date"]).dt.days.clip(0, 99)

    # --- Month of season (seasonality proxy) ---
    df["game_month"] = df["game_date"].dt.month

    return df


def build_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling season K-against rate by opponent team (how many Ks opposing pitchers
    rack up against this lineup).
    Uses only past games to avoid leakage.
    """
    df = df.copy()
    df["opp_key"] = df["opponent_name"].apply(normalize_team)

    # For each game, opponent_ks = K that the PITCHING team achieved.
    # So the opponent "faced" that many Ks. This is what we want:
    # opponent K-against rate = how many Ks they're allowing to opposing pitchers.
    # Sort by date to compute rolling.
    df = df.sort_values("game_date").reset_index(drop=True)

    # Season-level rolling opponent K-against rate
    df["season_year"] = df["game_date"].dt.year

    def opp_season_avg(group):
        return group["strikeouts"].shift(1).expanding().mean()

    opp_agg = df.groupby(["opp_key", "season_year"]).apply(
        opp_season_avg, include_groups=False
    ).reset_index(level=[0, 1], drop=True).rename("opp_k_against_season")

    df["opp_k_against_season"] = opp_agg.values
    return df


def build_spine(df_logs: pd.DataFrame, mkt_consensus: pd.DataFrame) -> pd.DataFrame:
    df = build_rolling_features(df_logs)
    df = build_opponent_features(df)

    df["player_key"] = df["player_name"].apply(normalize_name)
    df["game_date"]  = df["game_date"].astype(str)

    if not mkt_consensus.empty:
        df = df.merge(
            mkt_consensus,
            on=["player_key", "game_date"],
            how="left",
        )
    else:
        df["consensus_line"]   = np.nan
        df["novig_prob_over"]  = np.nan
        df["min_line"]         = np.nan
        df["max_line"]         = np.nan
        df["n_books"]          = np.nan

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    print("Loading game logs...")
    df_logs = load_gamelogs()
    print(f"  {len(df_logs):,} rows, {df_logs['player_id'].nunique()} pitchers")

    print("Loading market data...")
    df_mkt = load_market()
    print(f"  {len(df_mkt):,} rows, {df_mkt['event_id'].nunique()} events")

    print("Building market consensus...")
    mkt_consensus = build_market_consensus(df_mkt)
    print(f"  {len(mkt_consensus):,} player-game market rows")

    print("Building spine...")
    spine = build_spine(df_logs, mkt_consensus)
    print(f"  {len(spine):,} rows, {spine.columns.tolist()}")

    # Quick quality checks
    mkt_join_rate = spine["consensus_line"].notna().mean()
    print(f"\nJoin rate (has market data): {mkt_join_rate:.1%}")

    if args.verify:
        print("\n[VERIFY MODE — no upload]")
        print(spine.describe().to_string())
        return

    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    spine.to_parquet(LOCAL_OUT, index=False)
    print(f"\nSaved locally → {LOCAL_OUT}")

    s3 = boto3.client("s3")
    buf = BytesIO()
    spine.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=SPINE_BUCKET, Key=SPINE_KEY, Body=buf.getvalue())
    print(f"Uploaded → s3://{SPINE_BUCKET}/{SPINE_KEY}")


if __name__ == "__main__":
    main()

"""
Refresh MLB pitcher strikeouts spine from MLB Stats API + S3 market data.

Steps:
  1. Fetch current-season (2026) starter game logs from MLB Stats API
  2. Upload refreshed season parquet to S3
  3. Load all season game logs from S3
  4. Rebuild rolling-feature spine
  5. Upload spine to S3

Run:
    python src/mlb_strikeouts_modeling/scripts/update_spine.py
    python src/mlb_strikeouts_modeling/scripts/update_spine.py --verify
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import unicodedata
from io import BytesIO
from pathlib import Path

import boto3
import botocore.exceptions
import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

MLB_API_BASE    = "https://statsapi.mlb.com/api/v1"
SLEEP_S         = 0.05

S3_BUCKET       = "the-odds-api-mt"
GAMELOG_PREFIX  = "mlb/strikeouts_model/pitcher_gamelogs"
MARKET_PREFIX   = "mlb/strikeouts_model/market_raw"
SPINE_KEY       = "mlb/strikeouts_model/spine/mlb_strikeouts_spine.parquet"

SEASONS         = [2024, 2025, 2026]
REFRESH_SEASON  = 2026        # only re-fetch this season; others are historical
ROLL_WINDOWS    = [1, 3, 5, 10, 20]

NAME_MAP = {
    "louie varland": "louis varland",
}
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


def _s3():
    return boto3.client("s3")


# ── MLB Stats API helpers ──────────────────────────────────────────────────────

def get_starter_player_ids(season: int) -> list[tuple[int, str]]:
    r = requests.get(
        f"{MLB_API_BASE}/stats",
        params={
            "stats": "season", "group": "pitching", "season": season,
            "gameType": "R", "playerPool": "All", "limit": 2000, "offset": 0, "sportId": 1,
        },
        timeout=30,
    )
    r.raise_for_status()
    splits = r.json().get("stats", [{}])[0].get("splits", [])
    return [
        (s["player"]["id"], s["player"]["fullName"])
        for s in splits
        if s.get("stat", {}).get("gamesStarted", 0) >= 1
    ]


def get_pitcher_gamelogs(player_id: int, season: int) -> list[dict]:
    r = requests.get(
        f"{MLB_API_BASE}/people/{player_id}/stats",
        params={"stats": "gameLog", "season": season, "group": "pitching"},
        timeout=30,
    )
    if r.status_code != 200:
        return []
    stats_list = r.json().get("stats", [])
    if not stats_list:
        return []
    rows = []
    for s in stats_list[0].get("splits", []):
        stat = s.get("stat", {})
        if stat.get("gamesStarted", 0) < 1:
            continue
        rows.append({
            "player_id":       player_id,
            "season":          season,
            "game_date":       s.get("date", ""),
            "game_pk":         s.get("game", {}).get("gamePk"),
            "strikeouts":      stat.get("strikeOuts", 0),
            "innings_pitched": stat.get("inningsPitched", "0.0"),
            "hits":            stat.get("hits", 0),
            "earned_runs":     stat.get("earnedRuns", 0),
            "walks":           stat.get("baseOnBalls", 0),
            "batters_faced":   stat.get("battersFaced", 0),
            "pitches":         stat.get("numberOfPitches", 0),
            "opponent_id":     s.get("opponent", {}).get("id"),
            "opponent_name":   s.get("opponent", {}).get("name", ""),
            "is_home":         int(s.get("isHome", False)),
            "is_win":          int(s.get("isWin", False)),
            "games_started":   stat.get("gamesStarted", 0),
        })
    return rows


def fetch_season_gamelogs(season: int) -> pd.DataFrame:
    players = get_starter_player_ids(season)
    print(f"  Season {season}: {len(players)} starters")
    all_rows = []
    for i, (pid, name) in enumerate(players, 1):
        rows = get_pitcher_gamelogs(pid, season)
        for row in rows:
            row["player_name"] = name
        all_rows.extend(rows)
        time.sleep(SLEEP_S)
        if i % 50 == 0:
            print(f"    [{i}/{len(players)}] {len(all_rows)} rows")
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["innings_pitched"] = pd.to_numeric(df["innings_pitched"], errors="coerce")
    df = df.drop_duplicates(subset=["player_id", "game_pk"]).reset_index(drop=True)
    print(f"  {len(df):,} starts  |  dates: {df['game_date'].min()} → {df['game_date'].max()}")
    return df


def upload_gamelogs(s3c, df: pd.DataFrame, season: int):
    key = f"{GAMELOG_PREFIX}/{season}.parquet"
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3c.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
    print(f"  Uploaded → s3://{S3_BUCKET}/{key}")


def load_all_gamelogs() -> pd.DataFrame:
    s3c = _s3()
    frames = []
    for season in SEASONS:
        key = f"{GAMELOG_PREFIX}/{season}.parquet"
        body = s3c.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
        frames.append(pd.read_parquet(BytesIO(body)))
    return pd.concat(frames, ignore_index=True)


# ── Market data ────────────────────────────────────────────────────────────────

def load_market() -> pd.DataFrame:
    s3c = _s3()
    paginator = s3c.get_paginator("list_objects_v2")
    frames = []
    for season in SEASONS:
        prefix = f"{MARKET_PREFIX}/{season}/"
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                body = s3c.get_object(Bucket=S3_BUCKET, Key=obj["Key"])["Body"].read()
                frames.append(pd.read_parquet(BytesIO(body)))
    return pd.concat(frames, ignore_index=True)


def build_market_consensus(df_mkt: pd.DataFrame) -> pd.DataFrame:
    main = df_mkt[df_mkt["market_key"] == "pitcher_strikeouts"].copy()
    main = main[main["over_price"].notna() & main["under_price"].notna()].copy()
    if main.empty:
        return pd.DataFrame()

    def american_to_decimal(price: float) -> float:
        if pd.isna(price):
            return np.nan
        return price / 100 + 1 if price > 0 else 100 / abs(price) + 1

    main["player_key"] = main["player_name"].apply(normalize_name)
    main["dec_over"]   = main["over_price"].apply(american_to_decimal)
    main["dec_under"]  = main["under_price"].apply(american_to_decimal)
    main["raw_p_over"] = 1 / main["dec_over"]
    main["raw_p_under"]= 1 / main["dec_under"]
    main["novig_over"] = main["raw_p_over"] / (main["raw_p_over"] + main["raw_p_under"])

    line_mode = (
        main.groupby(["player_key", "game_date"])["line"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.median())
        .reset_index()
        .rename(columns={"line": "consensus_line"})
    )
    main2  = main.merge(line_mode, on=["player_key", "game_date"])
    at_line = main2[main2["line"] == main2["consensus_line"]]
    market_agg = (
        at_line.groupby(["player_key", "game_date"]).agg(
            consensus_line  = ("consensus_line", "first"),
            novig_prob_over = ("novig_over", "mean"),
            min_line        = ("line", "min"),
            max_line        = ("line", "max"),
            n_books         = ("bookmaker", "nunique"),
        ).reset_index()
    )
    return market_agg


# ── Spine build ────────────────────────────────────────────────────────────────

def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["game_date"]  = pd.to_datetime(df["game_date"])
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    def ip_to_decimal(ip):
        try:
            ip = float(ip)
            whole = int(ip)
            return whole + round((ip - whole) * 10) / 3
        except Exception:
            return np.nan

    df["ip_decimal"]    = df["innings_pitched"].apply(ip_to_decimal)
    df["is_short_start"] = (df["ip_decimal"] < 3).astype(int)

    grp_season = df.groupby(["player_id", "season"], sort=False)
    for w in ROLL_WINDOWS:
        df[f"k_roll_s{w}"] = grp_season["strikeouts"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
    df["k_roll_season"] = grp_season["strikeouts"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["ip_roll_season"] = grp_season["ip_decimal"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["start_num_season"] = grp_season["strikeouts"].transform(
        lambda x: x.shift(1).expanding().count()
    )

    grp_career = df.groupby("player_id", sort=False)
    for w in ROLL_WINDOWS:
        df[f"k_roll_c{w}"] = grp_career["strikeouts"].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
    df["k_roll_career"] = grp_career["strikeouts"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    df["prev_date"] = grp_career["game_date"].transform(lambda x: x.shift(1))
    df["days_rest"] = (df["game_date"] - df["prev_date"]).dt.days.clip(0, 99)
    df["game_month"] = df["game_date"].dt.month
    return df


def build_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["opp_key"]     = df["opponent_name"].apply(normalize_team)
    df["season_year"] = df["game_date"].dt.year
    df = df.sort_values("game_date").reset_index(drop=True)

    opp_agg = df.groupby(["opp_key", "season_year"]).apply(
        lambda g: g["strikeouts"].shift(1).expanding().mean(),
        include_groups=False,
    ).reset_index(level=[0, 1], drop=True).rename("opp_k_against_season")
    df["opp_k_against_season"] = opp_agg.values
    return df


def build_spine(df_logs: pd.DataFrame, mkt_consensus: pd.DataFrame) -> pd.DataFrame:
    df = build_rolling_features(df_logs)
    df = build_opponent_features(df)
    df["player_key"] = df["player_name"].apply(normalize_name)
    df["game_date"]  = df["game_date"].astype(str)
    if not mkt_consensus.empty:
        df = df.merge(mkt_consensus, on=["player_key", "game_date"], how="left")
    else:
        for col in ("consensus_line", "novig_prob_over", "min_line", "max_line", "n_books"):
            df[col] = np.nan
    return df


def upload_spine(spine: pd.DataFrame) -> None:
    buf = BytesIO()
    spine.to_parquet(buf, index=False)
    buf.seek(0)
    _s3().put_object(Bucket=S3_BUCKET, Key=SPINE_KEY, Body=buf.getvalue())
    print(f"Uploaded → s3://{S3_BUCKET}/{SPINE_KEY}  ({len(spine):,} rows)")


def load_s3_spine() -> pd.DataFrame | None:
    try:
        body = _s3().get_object(Bucket=S3_BUCKET, Key=SPINE_KEY)["Body"].read()
        return pd.read_parquet(BytesIO(body))
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="Rebuild locally and compare vs S3; do not upload")
    args = parser.parse_args()

    print(f"MLB Strikeouts Spine Update", flush=True)

    # Step 1: Refresh current season game logs from MLB Stats API
    print(f"\nFetching {REFRESH_SEASON} game logs from MLB Stats API...", flush=True)
    fresh_df = fetch_season_gamelogs(REFRESH_SEASON)
    if fresh_df.empty:
        raise RuntimeError(f"No game log data returned for {REFRESH_SEASON}")

    s3c = _s3()
    if not args.verify:
        upload_gamelogs(s3c, fresh_df, REFRESH_SEASON)

    # Step 2: Load all seasons from S3
    print("\nLoading all season game logs from S3...", flush=True)
    df_logs = load_all_gamelogs()
    print(f"  {len(df_logs):,} total rows  |  {df_logs['player_id'].nunique()} pitchers")

    # Step 3: Load market data
    print("Loading market data...", flush=True)
    df_mkt = load_market()
    print(f"  {len(df_mkt):,} market rows  |  {df_mkt['event_id'].nunique()} events")

    mkt_consensus = build_market_consensus(df_mkt)
    print(f"  {len(mkt_consensus):,} player-game consensus rows")

    # Step 4: Build spine
    print("Building spine...", flush=True)
    spine = build_spine(df_logs, mkt_consensus)
    print(f"  {len(spine):,} rows  |  {spine['player_key'].nunique()} pitchers")
    print(f"  Seasons: {spine['season'].value_counts().sort_index().to_dict()}")
    print(f"  k_roll_s5 null rate: {spine['k_roll_s5'].isna().mean()*100:.1f}%")
    print(f"  consensus_line join rate: {spine['consensus_line'].notna().mean()*100:.1f}%")

    if args.verify:
        existing = load_s3_spine()
        if existing is not None:
            print(f"\nS3 spine: {len(existing):,} rows  |  Local: {len(spine):,} rows  |  Diff: {len(spine)-len(existing):+,}")
        return

    # Step 5: Upload spine
    upload_spine(spine)
    print("Done.")


if __name__ == "__main__":
    main()

"""
Build rolling feature spine for MLB pitcher outs recorded model.

Spine grain: (player_key, game_date, bookmaker, line) — one row per book per line.
Rolling features (at player-game level) are computed once and broadcast to all book rows
for the same player-game. Per-book columns (novig_prob_over, over_price, under_price)
vary by book row.

Date join: historical S3 market files use UTC-derived game_dates; gamelogs use ET dates.
For West Coast games starting after midnight UTC the market date is 1 day ahead of the
gamelog date. The spine join tries exact game_date first, then game_date−1 as fallback.

Target: outs_recorded = int(innings_pitched)*3 + fractional digit
  e.g. 5.1 IP → 16 outs, 6.0 IP → 18 outs

Inputs:
  Gamelogs: s3://the-odds-api-mt/mlb/strikeouts_model/pitcher_gamelogs/{season}.parquet
  Market:   s3://the-odds-api-mt/mlb/pitcher_outs_model/market_raw/{season}/{event_id}.parquet

Outputs:
  S3:    s3://the-odds-api-mt/mlb/pitcher_outs_model/spine/mlb_pitcher_outs_spine.parquet
  Local: ~/Downloads/tmp/mlb_pitcher_outs_spine.parquet

Usage:
  python src/mlb_pitcher_outs_modeling/scripts/build_spine.py
  python src/mlb_pitcher_outs_modeling/scripts/build_spine.py --verify  # no upload
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from datetime import date, timedelta
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
MARKET_PREFIX  = "mlb/pitcher_outs_model/market_raw"
SPINE_BUCKET   = "the-odds-api-mt"
SPINE_KEY      = "mlb/pitcher_outs_model/spine/mlb_pitcher_outs_spine.parquet"
LOCAL_OUT      = Path.home() / "Downloads/tmp/mlb_pitcher_outs_spine.parquet"

SEASONS        = [2024, 2025, 2026]

# Rolling windows in prior starts (career-scoped and season-scoped)
ROLL_WINDOWS   = [1, 3, 5, 10, 20]

# Player name corrections: Odds API name (normalized) → gamelog name (normalized)
NAME_MAP = {
    "louie varland":      "louis varland",
    "luis l ortiz":       "luis ortiz",
    # Note: carlos f rodriguez, connor seabold, daniel davis, jake latz, kyle nelson,
    # lucas erceg, samuel aldegheri appear to be genuine relief pitchers — they should
    # have no gamelog entries and will produce no joined rows.
}

TEAM_MAP = {
    "athletics": "oakland athletics",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def ip_to_outs(ip) -> int:
    """5.1 → 16, 6.0 → 18, 7.2 → 23."""
    s = str(float(ip))
    full, frac = s.split(".")
    return int(full) * 3 + int(frac[0])


def decimal_to_american(d: float) -> float:
    """Convert decimal odds to American odds."""
    if d >= 2.0:
        return round((d - 1) * 100, 1)
    elif d > 1.0:
        return round(-100 / (d - 1), 1)
    return float("nan")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_gamelogs() -> pd.DataFrame:
    s3 = boto3.client("s3")
    frames = []
    for season in SEASONS:
        key = f"{GAMELOG_PREFIX}/{season}.parquet"
        body = s3.get_object(Bucket=GAMELOG_BUCKET, Key=key)["Body"].read()
        frames.append(pd.read_parquet(BytesIO(body)))
    df = pd.concat(frames, ignore_index=True)
    df["outs_recorded"] = df["innings_pitched"].apply(ip_to_outs)
    return df


LOCAL_MARKET = Path.home() / "Downloads/tmp/mlb_pitcher_outs_market_raw.parquet"


def load_market(use_local: bool = False) -> pd.DataFrame:
    """Load market data. Use local merged parquet if available and --local flag set."""
    if use_local and LOCAL_MARKET.exists():
        print(f"  Using local market file: {LOCAL_MARKET}")
        return pd.read_parquet(LOCAL_MARKET)

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    frames = []
    n = 0
    for season in SEASONS:
        prefix = f"{MARKET_PREFIX}/{season}/"
        for page in paginator.paginate(Bucket=MARKET_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                body = s3.get_object(Bucket=MARKET_BUCKET, Key=obj["Key"])["Body"].read()
                df = pd.read_parquet(BytesIO(body))
                if len(df) > 0:
                    frames.append(df)
                n += 1
                if n % 500 == 0:
                    print(f"  Loaded {n} event files…")
    print(f"  Loaded {n} total event files")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Rolling feature engineering
# ---------------------------------------------------------------------------

def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling outs features at pitcher level. Shift(1) = no lookahead."""
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

    # --- Season-scoped rolling (resets each season) ---
    grp_season = df.groupby(["player_id", "season"], sort=False)

    for w in ROLL_WINDOWS:
        df[f"outs_roll_s{w}"] = (
            grp_season["outs_recorded"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )

    df["outs_roll_season"] = (
        grp_season["outs_recorded"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df["ip_roll_season"] = (
        grp_season["ip_decimal"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df["start_num_season"] = (
        grp_season["outs_recorded"]
        .transform(lambda x: x.shift(1).expanding().count())
    )

    # --- Career-scoped rolling ---
    grp_career = df.groupby("player_id", sort=False)

    for w in ROLL_WINDOWS:
        df[f"outs_roll_c{w}"] = (
            grp_career["outs_recorded"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )

    df["outs_roll_career"] = (
        grp_career["outs_recorded"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # --- Reuse K features from strikeouts pipeline (K rate = proxy for stuff) ---
    for w in ROLL_WINDOWS:
        df[f"k_roll_c{w}"] = (
            grp_career["strikeouts"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
    df["k_roll_career"] = (
        grp_career["strikeouts"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df["k_roll_season"] = (
        grp_season["strikeouts"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # --- Days rest since last start ---
    df["prev_date"] = grp_career["game_date"].transform(lambda x: x.shift(1))
    df["days_rest"] = (df["game_date"] - df["prev_date"]).dt.days.clip(0, 99)

    # --- Seasonality ---
    df["game_month"] = df["game_date"].dt.month

    return df


def build_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling season K-against rate by opponent team (how many Ks the opponent
    allows to pitchers = proxy for how deep/K-prone the lineup is)."""
    df = df.copy()
    df["opp_key"] = df["opponent_name"].apply(normalize_team)
    df = df.sort_values("game_date").reset_index(drop=True)
    df["season_year"] = df["game_date"].dt.year

    def opp_season_avg(group):
        return group["strikeouts"].shift(1).expanding().mean()

    opp_agg = df.groupby(["opp_key", "season_year"]).apply(
        opp_season_avg, include_groups=False
    ).reset_index(level=[0, 1], drop=True).rename("opp_k_against_season")

    df["opp_k_against_season"] = opp_agg.values
    return df


# ---------------------------------------------------------------------------
# Market feature engineering
# ---------------------------------------------------------------------------

def build_market_features(df_mkt: pd.DataFrame) -> pd.DataFrame:
    """Compute per-book and per-player-game market features.

    Per-book (vary by row):
      - novig_prob_over: book's own no-vig P(over)
      - novig_prob_under: book's own no-vig P(under)

    Player-game level (broadcast to all books for same player-game):
      - consensus_line: mode across books
      - over_price_bucket_fine / under_price_bucket_fine (from consensus prices)
      - min_line / max_line
      - min/max raw implied prob over/under
      - team_run_line_point: pitcher's team run line point (resolved in spine join)
      - team_moneyline_odds: pitcher's team American moneyline (resolved in spine join)
    """
    df = df_mkt[df_mkt["market_key"] == "pitcher_outs"].copy()
    df = df[df["over_price"].notna() & df["under_price"].notna()].copy()

    # Deduplicate: Odds API occasionally returns duplicate player-book-line rows
    df = df.drop_duplicates(subset=["event_id","player_name","bookmaker","line"])

    if df.empty:
        return pd.DataFrame()

    df["player_key"] = df["player_name"].apply(normalize_name)

    # Per-book novig
    df["raw_p_over"]     = 1 / df["over_price"]
    df["raw_p_under"]    = 1 / df["under_price"]
    df["novig_prob_over"]  = df["raw_p_over"] / (df["raw_p_over"] + df["raw_p_under"])
    df["novig_prob_under"] = df["raw_p_under"] / (df["raw_p_over"] + df["raw_p_under"])

    # --- Consensus line (mode across books per player-game) ---
    line_mode = (
        df.groupby(["player_key", "game_date"])["line"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.median())
        .reset_index()
        .rename(columns={"line": "consensus_line"})
    )
    df = df.merge(line_mode, on=["player_key", "game_date"], how="left")

    # --- Min/max line and raw implied probs per player-game ---
    pg_agg = (
        df.groupby(["player_key", "game_date"]).agg(
            min_line=("line", "min"),
            max_line=("line", "max"),
            min_raw_prob_over=("raw_p_over", "min"),
            max_raw_prob_over=("raw_p_over", "max"),
            min_raw_prob_under=("raw_p_under", "min"),
            max_raw_prob_under=("raw_p_under", "max"),
            n_books=("bookmaker", "nunique"),
        ).reset_index()
    )
    df = df.merge(pg_agg, on=["player_key", "game_date"], how="left")

    # --- Odds bin features (based on avg consensus price at consensus line) ---
    at_consensus = df[df["line"] == df["consensus_line"]].copy()
    consensus_avg = (
        at_consensus.groupby(["player_key", "game_date"]).agg(
            avg_over_price=("over_price", "mean"),
            avg_under_price=("under_price", "mean"),
        ).reset_index()
    )

    def decimal_to_american_safe(d):
        try:
            if d >= 2.0:
                return (d - 1) * 100
            elif d > 1.0:
                return -100 / (d - 1)
            return float("nan")
        except Exception:
            return float("nan")

    consensus_avg["avg_over_american"]  = consensus_avg["avg_over_price"].apply(decimal_to_american_safe)
    consensus_avg["avg_under_american"] = consensus_avg["avg_under_price"].apply(decimal_to_american_safe)

    def odds_bucket_coarse(american: float) -> str:
        if pd.isna(american):
            return "unknown"
        elif american > 5:
            return "plus_odds"
        elif american < -5:
            return "minus_odds"
        else:
            return "even"

    def odds_bucket_fine(american: float) -> str:
        if pd.isna(american):
            return "unknown"
        elif american <= -300:
            return "-500_to_-300"
        elif american <= -200:
            return "-300_to_-200"
        elif american <= -110:
            return "-200_to_-110"
        elif american <= 5:
            return "-110_to_even"
        elif american <= 110:
            return "even_to_+110"
        elif american <= 200:
            return "+110_to_+200"
        elif american <= 300:
            return "+200_to_+300"
        else:
            return "+300_plus"

    consensus_avg["over_price_bucket_coarse"] = consensus_avg["avg_over_american"].apply(odds_bucket_coarse)
    consensus_avg["over_price_bucket_fine"]   = consensus_avg["avg_over_american"].apply(odds_bucket_fine)
    consensus_avg["under_price_bucket_coarse"]= consensus_avg["avg_under_american"].apply(odds_bucket_coarse)
    consensus_avg["under_price_bucket_fine"]  = consensus_avg["avg_under_american"].apply(odds_bucket_fine)

    df = df.merge(
        consensus_avg[["player_key","game_date",
                        "over_price_bucket_coarse","over_price_bucket_fine",
                        "under_price_bucket_coarse","under_price_bucket_fine"]],
        on=["player_key","game_date"], how="left"
    )

    return df


# ---------------------------------------------------------------------------
# Spine join
# ---------------------------------------------------------------------------

def build_spine(df_logs: pd.DataFrame, df_mkt_features: pd.DataFrame) -> pd.DataFrame:
    """Join rolling features onto each market row.

    Handles UTC→ET date offset: tries exact game_date first, then game_date−1 day
    to recover West Coast game rows where the market file used UTC date.
    """
    logs = build_rolling_features(df_logs)
    logs = build_opponent_features(logs)
    logs = logs.reset_index(drop=True)

    logs["player_key"] = logs["player_name"].apply(normalize_name)
    logs["game_date_dt"] = pd.to_datetime(logs["game_date"])
    logs["game_date"]    = logs["game_date_dt"].dt.strftime("%Y-%m-%d")
    # Also create game_date_plus1 for joining against UTC-dated market rows
    logs["game_date_p1"] = (logs["game_date_dt"] + timedelta(days=1)).dt.strftime("%Y-%m-%d")

    if df_mkt_features.empty:
        return pd.DataFrame()

    mkt = df_mkt_features.copy()
    mkt["game_date_str"] = mkt["game_date"].astype(str)

    # --- Build gamelog lookup by (player_key, game_date) ---
    _exclude = {"player_name","game_date","game_date_dt","game_date_p1",
                "player_id","game_pk","innings_pitched","ip_decimal","prev_date",
                "player_key"}  # player_key added back explicitly below to avoid dup columns
    log_feature_cols = [c for c in logs.columns if c not in _exclude]

    # Use composite key for clean merging (avoids multi-column join naming issues)
    mkt["_join_exact"]    = mkt["player_key"] + "|" + mkt["game_date_str"]
    mkt["_join_fallback"] = mkt["player_key"] + "|" + mkt["game_date_str"]

    logs_exact = logs[log_feature_cols + ["player_key","game_date"]].copy().reset_index(drop=True)
    logs_exact["_join_key"] = logs_exact["player_key"].astype(str) + "|" + logs_exact["game_date"].astype(str)
    logs_exact = logs_exact.drop(columns=["player_key","game_date"])

    logs_fallback = logs[log_feature_cols + ["player_key","game_date_p1"]].copy().reset_index(drop=True)
    logs_fallback["_join_key"] = logs_fallback["player_key"].astype(str) + "|" + logs_fallback["game_date_p1"].astype(str)
    logs_fallback = logs_fallback.drop(columns=["player_key","game_date_p1"])

    # Rename overlapping log columns before merging (season appears in both mkt and logs)
    log_overlap_cols = [c for c in logs_exact.columns if c != "_join_key" and c in mkt.columns]
    logs_exact = logs_exact.rename(columns={c: f"{c}_log" for c in log_overlap_cols})
    logs_fallback = logs_fallback.rename(columns={c: f"{c}_log" for c in log_overlap_cols if c in logs_fallback.columns})

    # Step 1: exact match
    attempt = mkt.merge(
        logs_exact,
        left_on="_join_exact",
        right_on="_join_key",
        how="left",
        indicator=True,
    )
    attempt = attempt.drop(columns=["_join_key"], errors="ignore")
    exact = attempt[attempt["_merge"] == "both"].drop(columns=["_merge"])
    exact["date_match"] = "exact"

    # Step 2: fallback for unmatched (UTC date was 1 day ahead of ET date)
    unmatched_mask = attempt["_merge"] == "left_only"
    mkt_cols_in_attempt = [c for c in mkt.columns if c in attempt.columns]
    mkt_unmatched = attempt[unmatched_mask][mkt_cols_in_attempt].copy()

    fallback = mkt_unmatched.merge(
        logs_fallback,
        left_on="_join_fallback",
        right_on="_join_key",
        how="inner",
    )
    fallback = fallback.drop(columns=["_join_key"], errors="ignore")
    fallback["date_match"] = "date_minus1"

    spine = pd.concat([exact, fallback], ignore_index=True)
    spine = spine.drop(columns=["_join_exact","_join_fallback"], errors="ignore")

    # --- Resolve pitcher's team betting features ---
    # is_home from gamelogs tells us whether the pitcher was home team
    # consensus_home_moneyline / home_run_line_point come from market data
    def resolve_team_features(row):
        if row.get("is_home") == 1:
            ml_dec  = row.get("consensus_home_moneyline")
            rl_pt   = row.get("home_run_line_point")
            rl_odds = row.get("home_run_line_odds")
        else:
            ml_dec  = row.get("consensus_away_moneyline")
            rl_pt   = row.get("away_run_line_point")
            rl_odds = row.get("away_run_line_odds")
        return pd.Series({
            "team_moneyline_dec":  ml_dec,
            "team_run_line_point": rl_pt,
            "team_run_line_odds":  rl_odds,
        })

    team_feats = spine.apply(resolve_team_features, axis=1)
    spine["team_moneyline_dec"]  = team_feats["team_moneyline_dec"]
    spine["team_run_line_point"] = team_feats["team_run_line_point"]
    spine["team_run_line_odds"]  = team_feats["team_run_line_odds"]

    # Convert team moneyline from decimal to American
    spine["team_moneyline_odds"] = spine["team_moneyline_dec"].apply(
        lambda d: decimal_to_american(d) if pd.notna(d) else np.nan
    )

    # --- Final column selection ---
    keep_cols = [
        # Identifiers
        "player_key", "game_date_str", "season", "bookmaker", "line",
        "home_team", "away_team", "event_id",
        # Target
        "outs_recorded",
        # Market per-book features
        "novig_prob_over", "novig_prob_under", "over_price", "under_price",
        # Market player-game features
        "consensus_line", "n_books",
        "min_line", "max_line",
        "min_raw_prob_over", "max_raw_prob_over",
        "min_raw_prob_under", "max_raw_prob_under",
        "over_price_bucket_coarse", "over_price_bucket_fine",
        "under_price_bucket_coarse", "under_price_bucket_fine",
        # Team betting
        "team_run_line_point", "team_moneyline_odds",
        # Rolling outs features
        "outs_roll_career", "outs_roll_season", "outs_roll_s1",
        "outs_roll_s3", "outs_roll_s5", "outs_roll_s10", "outs_roll_s20",
        "outs_roll_c1", "outs_roll_c3", "outs_roll_c5",
        "outs_roll_c10", "outs_roll_c20",
        # Rolling K features (stuff proxy)
        "k_roll_career", "k_roll_season",
        "k_roll_c1", "k_roll_c3", "k_roll_c5", "k_roll_c10", "k_roll_c20",
        # Opponent quality
        "opp_k_against_season",
        # Context
        "is_home", "days_rest", "game_month", "start_num_season", "ip_roll_season",
        # Join diagnostics
        "date_match",
    ]
    # Keep only columns that exist
    keep_cols = [c for c in keep_cols if c in spine.columns]
    spine = spine[keep_cols].copy()
    spine = spine.rename(columns={"game_date_str": "game_date"})

    # Final dedup: same physical game can appear under two event_ids (rescheduled games).
    # Keep the earlier event_id (original schedule) per (player_key, game_date, bookmaker, line).
    before = len(spine)
    spine = spine.sort_values("event_id").drop_duplicates(
        subset=["player_key", "game_date", "bookmaker", "line"], keep="first"
    )
    if len(spine) < before:
        print(f"  Cross-event dedup: removed {before - len(spine)} rows")

    return spine


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="Print summary but skip S3 upload")
    parser.add_argument("--local", action="store_true",
                        help="Load market data from local merged parquet (faster for dev)")
    args = parser.parse_args()

    print("Loading game logs...")
    df_logs = load_gamelogs()
    print(f"  {len(df_logs):,} rows, {df_logs['player_id'].nunique()} pitchers, "
          f"seasons: {sorted(df_logs['season'].unique())}")

    print("Loading market data...")
    df_mkt = load_market(use_local=args.local)
    print(f"  {len(df_mkt):,} rows, {df_mkt['event_id'].nunique()} events")

    print("Building market features...")
    df_mkt_features = build_market_features(df_mkt)
    print(f"  {len(df_mkt_features):,} rows after filtering to pitcher_outs with both sides")

    print("Building spine (rolling features + join)...")
    spine = build_spine(df_logs, df_mkt_features)
    print(f"  {len(spine):,} rows in spine")

    if len(spine) == 0:
        print("ERROR: empty spine!")
        return

    # Quality checks
    total_mkt_rows = len(df_mkt_features[df_mkt_features["market_key"] == "pitcher_outs"])
    join_rate = len(spine) / total_mkt_rows if total_mkt_rows > 0 else 0
    exact_pct = (spine["date_match"] == "exact").mean()
    fallback_pct = (spine["date_match"] == "date_minus1").mean()

    print(f"\nJoin rate: {join_rate:.1%} ({len(spine):,} / {total_mkt_rows:,} market rows)")
    print(f"Date match: exact={exact_pct:.1%}, date-1 fallback={fallback_pct:.1%}")
    print(f"\nNull rates for key features:")
    for col in ["outs_recorded","novig_prob_over","consensus_line","outs_roll_career",
                "outs_roll_season","opp_k_against_season","team_run_line_point","team_moneyline_odds"]:
        if col in spine.columns:
            null_rate = spine[col].isna().mean()
            print(f"  {col}: {null_rate:.1%}")

    print(f"\nColumns: {list(spine.columns)}")

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

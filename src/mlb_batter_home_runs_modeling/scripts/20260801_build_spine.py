"""
Step 2 — Build feature spine at (player, game_date, bookmaker, offered_line) grain.

Rolling features (L1, L3, L5, L10, L20, season, career):
  hr_roll_L1/L3/L5/L10/L20/season/career
  ab_roll_L5/career (proxy for playing time + lineup position)

Opponent (team-level HR-allowed rate):
  opp_hr_rate_career, opp_hr_rate_L20

Game context:
  is_home (1=home, 0=away)
  games_played_career (player experience)

Market consensus features (book-invariant per player-game-line):
  consensus_over_odds_bin, consensus_over_odds_bin_granular
  consensus_under_odds_bin, consensus_under_odds_bin_granular
  consensus_line, min_line, max_line
  min/max_raw_implied_prob_over, min/max_raw_implied_prob_under

Per-book cols (edge calc only — NOT model inputs):
  novig_prob_over, raw_implied_prob_over, raw_implied_prob_under

Target:
  hr_actual, hr_over_0_5 (1 if home_runs >= 1 else 0)

Output:
  S3:    s3://the-odds-api-mt/mlb/batter_home_runs_model/spine/mlb_batter_hr_spine.parquet
  Local: ~/Downloads/tmp/mlb_batter_hr_spine.parquet

Usage:
  python src/mlb_batter_home_runs_modeling/scripts/20260801_build_spine.py
  python src/mlb_batter_home_runs_modeling/scripts/20260801_build_spine.py --no-s3
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
import warnings
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT    = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET    = "the-odds-api-mt"
S3_ACTUALS   = "mlb/total_bases_model/actuals/mlb_batting_statcast.parquet"
S3_SPINE_OUT = "mlb/batter_home_runs_model/spine/mlb_batter_hr_spine.parquet"
LOCAL_MARKET = Path.home() / "Downloads/tmp/mlb_batter_home_runs_market_raw.parquet"
LOCAL_OUT    = Path.home() / "Downloads/tmp/mlb_batter_hr_spine.parquet"

# Known Odds API ↔ Statcast name mismatches not caught by normalize_name().
NAME_MAP: dict[str, str] = {
    "daniel vogelbach":   "dan vogelbach",
    "donnie walton":      "donovan walton",
    "eddy alvarez":       "francisco alvarez",
    "josh kuroda grauer": "joshua kuroda grauer",
}


# ─── Utilities ────────────────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"\s*\(\d{4}\)", "", name)
    name = re.sub(r"[''`]", "", name)
    name = re.sub(r"[-]", " ", name)
    name = re.sub(r"\.", "", name)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", name)
    name = re.sub(r"\b([a-z]) ([a-z])\b", r"\1\2", name)
    name = re.sub(r"\b([a-z]) ([a-z])\b", r"\1\2", name)
    name = re.sub(r"(?<=\s)[a-z](?=\s)", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return NAME_MAP.get(name, name)


def decimal_to_american(d: float) -> float:
    if pd.isna(d) or d <= 1.0:
        return np.nan
    if d >= 2.0:
        return (d - 1) * 100
    return -100 / (d - 1)


def odds_bin_coarse(american_odds: float) -> str:
    if pd.isna(american_odds):
        return "unknown"
    if american_odds > 0:
        return "plus_odds"
    elif american_odds < 0:
        return "minus_odds"
    return "even"


def odds_bin_granular(american_odds: float) -> str:
    if pd.isna(american_odds):
        return "unknown"
    if american_odds <= -300:
        return "-500_to_-300"
    elif american_odds <= -200:
        return "-300_to_-200"
    elif american_odds <= -110:
        return "-200_to_-110"
    elif american_odds < 0:
        return "-110_to_even"
    elif american_odds == 0:
        return "even"
    elif american_odds <= 110:
        return "even_to_+110"
    elif american_odds <= 200:
        return "+110_to_+200"
    elif american_odds <= 300:
        return "+200_to_+300"
    return "+300_plus"


# ─── A: Load market ───────────────────────────────────────────────────────────

def load_market() -> pd.DataFrame:
    if LOCAL_MARKET.exists():
        print(f"  Loading market from local: {LOCAL_MARKET}")
        market = pd.read_parquet(LOCAL_MARKET)
    else:
        print("  Local market not found — loading from S3...")
        s3c = boto3.client("s3")
        frames = []
        paginator = s3c.get_paginator("list_objects_v2")
        for season in [2024, 2025, 2026]:
            prefix = f"mlb/batter_home_runs_model/market_raw/{season}/"
            for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
                for obj in page.get("Contents", []):
                    if obj["Size"] == 0:
                        continue
                    resp = s3c.get_object(Bucket=S3_BUCKET, Key=obj["Key"])
                    df = pd.read_parquet(BytesIO(resp["Body"].read()))
                    if len(df) > 0:
                        frames.append(df)
        market = pd.concat(frames, ignore_index=True)
        LOCAL_MARKET.parent.mkdir(parents=True, exist_ok=True)
        market.to_parquet(LOCAL_MARKET, index=False)

    market["game_date"] = market["game_date"].astype(str)
    print(f"  Market loaded: {len(market):,} rows, {market['player_name'].nunique():,} players")
    return market


def dedup_market(market: pd.DataFrame) -> pd.DataFrame:
    """Dedup to (player_name, game_date, bookmaker, line) grain."""
    has_under = market[market["under_price"].notna()]
    no_under  = market[market["under_price"].isna()]

    key = ["player_name", "game_date", "bookmaker", "line"]
    has_under_dedup = has_under.sort_values("over_price").drop_duplicates(subset=key, keep="first")
    covered = set(has_under_dedup[key].apply(tuple, axis=1))
    no_under_filt = no_under[~no_under[key].apply(tuple, axis=1).isin(covered)]
    no_under_dedup = no_under_filt.drop_duplicates(subset=key, keep="first")

    deduped = pd.concat([has_under_dedup, no_under_dedup], ignore_index=True)
    print(f"  After dedup: {len(deduped):,} rows (was {len(market):,})")
    return deduped


# ─── B: Load actuals ──────────────────────────────────────────────────────────

def load_actuals() -> pd.DataFrame:
    s3c = boto3.client("s3")
    obj = s3c.get_object(Bucket=S3_BUCKET, Key=S3_ACTUALS)
    df  = pd.read_parquet(BytesIO(obj["Body"].read()))
    df["game_date"] = df["game_date"].astype(str)
    df["name_norm"] = df["player_name"].apply(normalize_name)
    df["is_home"]   = (df["team"] == df["home_team"]).astype(int)

    # Dedup doubleheaders: keep highest-AB game per (player, date)
    df = df.sort_values("ab", ascending=False).drop_duplicates(subset=["game_date", "name_norm"])
    print(f"  Actuals: {len(df):,} batter-games, {df['name_norm'].nunique():,} players")
    print(f"  Date range: {df['game_date'].min()} → {df['game_date'].max()}")
    return df


# ─── C: Build rolling batter features ────────────────────────────────────────

def add_batter_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling HR features. shift(1) prevents lookahead."""
    df = df.sort_values(["name_norm", "game_date"]).reset_index(drop=True)

    for window, label in [(1, "L1"), (3, "L3"), (5, "L5"), (10, "L10"), (20, "L20")]:
        df[f"hr_roll_{label}"] = df.groupby("name_norm")["home_runs"].transform(
            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
        )

    df["hr_roll_season"] = df.groupby(["name_norm", "season"])["home_runs"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["hr_roll_career"] = df.groupby("name_norm")["home_runs"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    df["ab_roll_L5"] = df.groupby("name_norm")["ab"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["ab_roll_career"] = df.groupby("name_norm")["ab"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    df["games_played_career"] = df.groupby("name_norm")["home_runs"].transform(
        lambda x: x.shift(1).expanding().count()
    )

    return df


# ─── D: Build opponent HR-allowed features ───────────────────────────────────

def add_opponent_hr_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rolling HR-allowed rate per pitching team (opponent).
    For each batter-game, opp_hr_rate_career = avg HRs per batter-game by the opponent's
    pitching staff across all prior batter-game rows involving that team as the opponent.
    """
    # Team-game level: total HRs allowed + total batter-games faced
    df_sorted = df.sort_values(["opponent", "game_date"]).reset_index(drop=True)

    # One row per (opponent, game_date, batter-game)
    opp_hr_career = df_sorted.groupby("opponent")["home_runs"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    opp_hr_L20 = df_sorted.groupby("opponent")["home_runs"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=1).mean()
    )

    df["opp_hr_rate_career"] = opp_hr_career.values
    df["opp_hr_rate_L20"]    = opp_hr_L20.values

    return df


# ─── E: Add market consensus + per-book features ─────────────────────────────

def add_market_features(market: pd.DataFrame) -> pd.DataFrame:
    market = market.copy()
    market["raw_implied_prob_over"]  = 1.0 / market["over_price"]
    market["raw_implied_prob_under"] = 1.0 / market["under_price"]
    novig_total = market["raw_implied_prob_over"] + market["raw_implied_prob_under"]
    market["novig_prob_over"]  = market["raw_implied_prob_over"]  / novig_total
    market["novig_prob_under"] = market["raw_implied_prob_under"] / novig_total

    # Consensus decimal odds at (player_key, game_date, line) across all books
    cons = (
        market.groupby(["player_key", "game_date", "line"])
        .agg(
            consensus_decimal_over=("over_price", "mean"),
            consensus_decimal_under=("under_price", "mean"),
        )
        .reset_index()
    )
    cons["consensus_american_over"]  = cons["consensus_decimal_over"].apply(decimal_to_american)
    cons["consensus_american_under"] = cons["consensus_decimal_under"].apply(decimal_to_american)
    cons["consensus_over_odds_bin"]           = cons["consensus_american_over"].apply(odds_bin_coarse)
    cons["consensus_over_odds_bin_granular"]  = cons["consensus_american_over"].apply(odds_bin_granular)
    cons["consensus_under_odds_bin"]          = cons["consensus_american_under"].apply(odds_bin_coarse)
    cons["consensus_under_odds_bin_granular"] = cons["consensus_american_under"].apply(odds_bin_granular)

    # Min/max features at (player_key, game_date)
    minmax = (
        market.groupby(["player_key", "game_date"])
        .agg(
            min_line=("line", "min"),
            max_line=("line", "max"),
            min_raw_implied_prob_over=("raw_implied_prob_over", "min"),
            max_raw_implied_prob_over=("raw_implied_prob_over", "max"),
            min_raw_implied_prob_under=("raw_implied_prob_under", "min"),
            max_raw_implied_prob_under=("raw_implied_prob_under", "max"),
        )
        .reset_index()
    )

    # Consensus line — avg offered line across all books for this player-game
    cons_line = (
        market.groupby(["player_key", "game_date"])["line"]
        .mean()
        .reset_index()
        .rename(columns={"line": "consensus_line"})
    )

    market = market.merge(cons,      on=["player_key", "game_date", "line"], how="left")
    market = market.merge(minmax,    on=["player_key", "game_date"],         how="left")
    market = market.merge(cons_line, on=["player_key", "game_date"],         how="left")
    return market


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-s3", action="store_true")
    args = parser.parse_args()

    print("=== A: Load market ===")
    market = load_market()

    print("\n=== B: Dedup market to (player, game_date, bookmaker, line) grain ===")
    market = dedup_market(market)

    print("\n=== C: Normalize names + add per-book features ===")
    market["name_norm"]  = market["player_name"].apply(normalize_name)
    market["player_key"] = market["name_norm"]
    market = add_market_features(market)
    market = market.rename(columns={"line": "offered_line"})
    print(f"  Market ready: {len(market):,} rows")

    print("\n=== D: Load actuals ===")
    actuals = load_actuals()

    print("\n=== E: Build rolling batter features ===")
    actuals = add_batter_rolling_features(actuals)
    print(f"  Rolling features added. Shape: {actuals.shape}")

    print("\n=== F: Build opponent HR-allowed features ===")
    actuals = add_opponent_hr_features(actuals)
    print(f"  Opp HR rate null rate: {actuals['opp_hr_rate_career'].isna().mean():.1%}")

    print("\n=== G: Join features to market spine ===")
    feature_cols = [
        "name_norm", "game_date",
        "home_runs", "ab", "is_home",
        "hr_roll_L1", "hr_roll_L3", "hr_roll_L5", "hr_roll_L10", "hr_roll_L20",
        "hr_roll_season", "hr_roll_career",
        "ab_roll_L5", "ab_roll_career",
        "games_played_career",
        "opp_hr_rate_career", "opp_hr_rate_L20",
    ]
    spine = market.merge(
        actuals[feature_cols].rename(columns={"name_norm": "player_key"}),
        on=["player_key", "game_date"],
        how="left",
    )

    spine["hr_actual"] = spine["home_runs"]
    spine["hr_over_0_5"] = np.where(
        spine["hr_actual"].isna(),
        np.nan,
        (spine["hr_actual"] >= 1).astype(float),
    )
    spine["season"] = spine["season"].fillna(spine["game_date"].str[:4].astype(float))

    # Final dedup: trades can produce 2 event rows for same player+date.
    before_dedup = len(spine)
    spine = (
        spine.sort_values("over_price")
        .drop_duplicates(subset=["player_key", "game_date", "bookmaker", "offered_line"])
        .reset_index(drop=True)
    )
    if len(spine) < before_dedup:
        print(f"  Final dedup removed {before_dedup - len(spine):,} trade-duplicate rows")

    match_rate = spine["hr_actual"].notna().mean()
    print(f"  Spine rows: {len(spine):,}")
    print(f"  Match rate (market rows with actuals): {match_rate:.1%}")

    unmatched = (
        spine[spine["hr_actual"].isna()]["name_norm"]
        .value_counts()
        .head(20)
    )
    if len(unmatched):
        print("  Top unmatched players:")
        for name, cnt in unmatched.items():
            print(f"    '{name}': {cnt} rows")

    print("\n=== H: Spot-check — Aaron Judge ===")
    judge_key = normalize_name("Aaron Judge")
    judge = spine[spine["player_key"] == judge_key].sort_values(["game_date", "bookmaker"])
    print(f"  Aaron Judge rows: {len(judge):,}")
    show_cols = [
        "game_date", "bookmaker", "offered_line", "hr_actual", "hr_over_0_5",
        "hr_roll_L5", "hr_roll_L10", "hr_roll_career",
        "opp_hr_rate_career", "novig_prob_over", "is_home",
    ]
    print(judge[show_cols].tail(10).to_string(index=False))

    # Leakage check
    judge_dedup = judge.drop_duplicates(subset=["game_date"]).sort_values("game_date")
    print("\n  Leakage check (hr_roll_L5 should NOT include same-game HRs):")
    for _, row in judge_dedup.tail(5).iterrows():
        print(f"    {row['game_date']}: actual={row['hr_actual']}, hr_roll_L5={row['hr_roll_L5']:.4f}")

    print("\n=== I: Season-start check ===")
    for season in [2024, 2025, 2026]:
        first_rows = spine[spine["season"] == season].sort_values("game_date")
        if len(first_rows):
            r = first_rows.iloc[0]
            print(f"  Season {season} first row: game_date={r['game_date']}, "
                  f"hr_roll_season={r['hr_roll_season']}, hr_roll_career={r['hr_roll_career']}")

    print("\n=== J: Save ===")
    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    spine.to_parquet(LOCAL_OUT, index=False)
    print(f"  Saved locally → {LOCAL_OUT}  ({len(spine):,} rows)")
    print(f"  Columns: {spine.columns.tolist()}")

    if not args.no_s3:
        buf = BytesIO()
        spine.to_parquet(buf, index=False)
        boto3.client("s3").put_object(Bucket=S3_BUCKET, Key=S3_SPINE_OUT, Body=buf.getvalue())
        print(f"  Saved to S3 → s3://{S3_BUCKET}/{S3_SPINE_OUT}")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
Step 2 — Build feature matrix at (player, game_date, bookmaker, line) grain.

Batter rolling features (window sizes: L1, L5, L10, L20, season, career):
  k_roll_L1, k_roll_L5, k_roll_L10, k_roll_L20, k_roll_season, k_roll_career
  pa_roll_career, pa_roll_L5, k_rate_career, k_rate_L5

Pitcher rolling features:
  opp_k_rate_career, opp_k_rate_L5

Game flags:
  is_home, stand (batter handedness: L/R/S)

Market features:
  offered_line
  consensus_over_odds_bin, consensus_over_odds_bin_granular
  consensus_under_odds_bin, consensus_under_odds_bin_granular
  min_line, max_line
  min_raw_implied_prob_over, max_raw_implied_prob_over
  min_raw_implied_prob_under, max_raw_implied_prob_under

Per-book (not model inputs — edge calculation only):
  novig_prob_over, raw_implied_prob_over, raw_implied_prob_under

Target:
  over_flag = 1 if actual strikeouts > line

Saves to S3 and ~/Downloads/tmp/mlb_batter_strikeouts_features.parquet
"""
from __future__ import annotations

import sys
import unicodedata
import re
import warnings
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import pybaseball

warnings.filterwarnings("ignore")
pybaseball.cache.enable()

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET   = "the-odds-api-mt"
S3_FEATURES = "mlb/batter_strikeouts_model/features/mlb_batter_strikeouts_features.parquet"
LOCAL_SPINE  = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_spine.parquet"
LOCAL_MARKET = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_market_raw.parquet"
LOCAL_OUT    = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_features.parquet"

SEASON_DATES = {
    2024: ("2024-03-20", "2024-10-01"),
    2025: ("2025-03-18", "2025-10-01"),
    2026: ("2026-03-25", "2026-07-11"),
}

NAME_MAP = {
    "daniel vogelbach": "dan vogelbach",
    "donnie walton":    "donovan walton",
}

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


# ─── Step A: pull inning_topbot from Statcast (cached) ──────────────────────

def build_pa_lookup() -> pd.DataFrame:
    """Pull (batter, pitcher, game_pk, game_date, home_team, away_team, inning_topbot)
    from Statcast for all seasons. Used to assign home/away per batter and identify
    opposing starting pitcher."""
    frames = []
    keep = ["batter", "pitcher", "game_pk", "game_date",
            "home_team", "away_team", "inning", "inning_topbot", "events", "stand"]
    for season, (start, end) in SEASON_DATES.items():
        print(f"  Pulling Statcast {season} (cached)...")
        df = pybaseball.statcast(start, end)
        pa = df[df["events"].notna()][keep].copy()
        pa["season"] = season
        frames.append(pa)
        print(f"    {season}: {len(pa):,} PAs")
    return pd.concat(frames, ignore_index=True)


# ─── Step B: build pitcher game logs ────────────────────────────────────────

def build_pitcher_game_logs(pa: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PA data to pitcher-game level: k_count, batters_faced."""
    pa["is_k"] = pa["events"].isin({"strikeout", "strikeout_double_play"}).astype(int)
    pgl = (
        pa.groupby(["pitcher", "game_pk", "game_date", "season"])
        .agg(k_count=("is_k", "sum"), batters_faced=("events", "count"))
        .reset_index()
    )
    pgl["game_date"] = pd.to_datetime(pgl["game_date"]).dt.strftime("%Y-%m-%d")
    return pgl


# ─── Step C: identify starting pitchers per (game_pk, side) ─────────────────

def identify_starting_pitchers(pa: pd.DataFrame) -> pd.DataFrame:
    """For each game_pk, identify home_starter and away_starter.

    Top of inning 1 (away batters face home pitcher) → home starter
    Bottom of inning 1 (home batters face away pitcher) → away starter
    """
    inn1 = pa[pa["inning"] == 1].sort_values(["game_pk", "inning_topbot"])
    home_starters = (
        inn1[inn1["inning_topbot"] == "Top"]
        .groupby("game_pk")["pitcher"].first()
        .reset_index()
        .rename(columns={"pitcher": "home_starter_id"})
    )
    away_starters = (
        inn1[inn1["inning_topbot"] == "Bot"]
        .groupby("game_pk")["pitcher"].first()
        .reset_index()
        .rename(columns={"pitcher": "away_starter_id"})
    )
    starters = home_starters.merge(away_starters, on="game_pk", how="outer")
    return starters


# ─── Step D: assign batter team and opposing pitcher ────────────────────────

def assign_batter_team(pa: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame:
    """Add is_home and opposing_pitcher_id to each batter-PA."""
    pa = pa.merge(starters, on="game_pk", how="left")
    # Top of inning → away batter (is_home=0) → opposing pitcher = home_starter
    # Bot of inning → home batter (is_home=1) → opposing pitcher = away_starter
    pa["is_home"] = (pa["inning_topbot"] == "Bot").astype(int)
    pa["opposing_pitcher_id"] = np.where(
        pa["is_home"] == 1, pa["away_starter_id"], pa["home_starter_id"]
    )
    return pa


# ─── Step E: build batter game log with is_home + opp pitcher ───────────────

def build_batter_game_log(pa: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PA data to (batter, game_date, game_pk) with is_home + opp pitcher."""
    pa["is_k"] = pa["events"].isin({"strikeout", "strikeout_double_play"}).astype(int)
    bgl = (
        pa.groupby(["batter", "game_pk", "game_date", "home_team", "away_team",
                    "season", "is_home", "opposing_pitcher_id"])
        .agg(
            strikeouts=("is_k", "sum"),
            plate_appearances=("events", "count"),
        )
        .reset_index()
    )
    bgl["game_date"] = pd.to_datetime(bgl["game_date"]).dt.strftime("%Y-%m-%d")
    return bgl


# ─── Step F: compute rolling batter features ────────────────────────────────

def rolling_mean_no_future(series: pd.Series, window: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).mean()

def season_mean_no_future(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby(["batter", "season"])[col].transform(
        lambda x: x.shift(1).expanding().mean()
    )

def career_mean_no_future(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("batter")[col].transform(
        lambda x: x.shift(1).expanding().mean()
    )


def add_batter_rolling_features(bgl: pd.DataFrame) -> pd.DataFrame:
    bgl = bgl.sort_values(["batter", "game_date"]).reset_index(drop=True)

    for window, label in [(1, "L1"), (5, "L5"), (10, "L10"), (20, "L20")]:
        bgl[f"k_roll_{label}"] = bgl.groupby("batter")["strikeouts"].transform(
            lambda x: rolling_mean_no_future(x, window)
        )
        if label == "L5":
            bgl["pa_roll_L5"] = bgl.groupby("batter")["plate_appearances"].transform(
                lambda x: rolling_mean_no_future(x, window)
            )

    bgl["k_roll_season"] = season_mean_no_future(bgl, "strikeouts")
    bgl["k_roll_career"] = career_mean_no_future(bgl, "strikeouts")
    bgl["pa_roll_career"] = career_mean_no_future(bgl, "plate_appearances")

    # K rate = K per PA (rolling)
    bgl["k_rate_career"] = (
        bgl.groupby("batter")["strikeouts"].transform(lambda x: x.shift(1).expanding().sum()) /
        bgl.groupby("batter")["plate_appearances"].transform(lambda x: x.shift(1).expanding().sum())
    )
    bgl["k_rate_L5"] = (
        bgl.groupby("batter")["strikeouts"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()) /
        bgl.groupby("batter")["plate_appearances"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    )
    return bgl


# ─── Step G: compute rolling pitcher features ───────────────────────────────

def add_pitcher_rolling_features(pgl: pd.DataFrame) -> pd.DataFrame:
    pgl = pgl.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
    pgl["opp_k_rate_career"] = (
        pgl.groupby("pitcher")["k_count"].transform(lambda x: x.shift(1).expanding().sum()) /
        pgl.groupby("pitcher")["batters_faced"].transform(lambda x: x.shift(1).expanding().sum())
    )
    pgl["opp_k_rate_L5"] = (
        pgl.groupby("pitcher")["k_count"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()) /
        pgl.groupby("pitcher")["batters_faced"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    )
    return pgl[["pitcher", "game_date", "opp_k_rate_career", "opp_k_rate_L5"]].rename(
        columns={"pitcher": "opposing_pitcher_id"}
    )


# ─── Step H: compute market consensus features ──────────────────────────────

def american_to_decimal(american: float) -> float:
    if pd.isna(american):
        return np.nan
    if american > 0:
        return american / 100 + 1
    else:
        return 100 / abs(american) + 1


def odds_bin_coarse(american_odds: float) -> str:
    if pd.isna(american_odds):
        return "unknown"
    if american_odds > 0:
        return "plus_odds"
    elif american_odds < 0:
        return "minus_odds"
    else:
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
    else:
        return "+300_plus"


def add_market_features(market: pd.DataFrame) -> pd.DataFrame:
    """Add consensus odds bins and min/max features at (player_key, game_date, line) level."""
    market = market.copy()
    market["decimal_over"]  = market["over_price"]
    market["decimal_under"] = market["under_price"]
    market["raw_implied_prob_over"]  = 1.0 / market["decimal_over"]
    market["raw_implied_prob_under"] = 1.0 / market["decimal_under"]
    market["novig_total"] = market["raw_implied_prob_over"] + market["raw_implied_prob_under"]
    market["novig_prob_over"]  = market["raw_implied_prob_over"]  / market["novig_total"]
    market["novig_prob_under"] = market["raw_implied_prob_under"] / market["novig_total"]

    # Consensus: average decimal over/under across all books at (player_key, game_date, line)
    cons = (
        market.groupby(["player_key", "game_date", "line"])
        .agg(
            consensus_decimal_over=("decimal_over", "mean"),
            consensus_decimal_under=("decimal_under", "mean"),
            min_line=("line", "min"),
            max_line=("line", "max"),
            min_raw_implied_prob_over=("raw_implied_prob_over", "min"),
            max_raw_implied_prob_over=("raw_implied_prob_over", "max"),
            min_raw_implied_prob_under=("raw_implied_prob_under", "min"),
            max_raw_implied_prob_under=("raw_implied_prob_under", "max"),
        )
        .reset_index()
    )

    # Convert consensus decimal to American for binning
    def decimal_to_american(d):
        if pd.isna(d):
            return np.nan
        if d >= 2.0:
            return (d - 1) * 100
        else:
            return -100 / (d - 1)

    cons["consensus_american_over"]  = cons["consensus_decimal_over"].apply(decimal_to_american)
    cons["consensus_american_under"] = cons["consensus_decimal_under"].apply(decimal_to_american)
    cons["consensus_over_odds_bin"]           = cons["consensus_american_over"].apply(odds_bin_coarse)
    cons["consensus_over_odds_bin_granular"]  = cons["consensus_american_over"].apply(odds_bin_granular)
    cons["consensus_under_odds_bin"]          = cons["consensus_american_under"].apply(odds_bin_coarse)
    cons["consensus_under_odds_bin_granular"] = cons["consensus_american_under"].apply(odds_bin_granular)

    market = market.merge(cons, on=["player_key", "game_date", "line"], how="left")
    return market


# ─── Step I: add stand (batter handedness) from spine ───────────────────────

def add_stand(bgl: pd.DataFrame, pa_full: pd.DataFrame) -> pd.DataFrame:
    """Add stand (L/R/S) to batter game log from PA data."""
    stand_lookup = (
        pa_full.groupby(["batter", "game_pk"])["stand"].first().reset_index()
    )
    return bgl.merge(stand_lookup, on=["batter", "game_pk"], how="left")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=== Step A: Pull Statcast PA data (cached) ===")
    pa_full = build_pa_lookup()

    print("\n=== Step B: Build pitcher game logs ===")
    pgl = build_pitcher_game_logs(pa_full.copy())
    print(f"  Pitcher game logs: {len(pgl):,} rows")

    print("\n=== Step C: Identify starting pitchers ===")
    starters = identify_starting_pitchers(pa_full)
    print(f"  Starting pitcher pairs: {len(starters):,} games")

    print("\n=== Step D: Assign batter team + opposing pitcher ===")
    pa_assigned = assign_batter_team(pa_full.copy(), starters)

    print("\n=== Step E: Build batter game log ===")
    bgl = build_batter_game_log(pa_assigned)
    print(f"  Batter game logs: {len(bgl):,} rows")

    print("\n=== Step F: Batter rolling features ===")
    bgl = add_batter_rolling_features(bgl)

    print("\n=== Step G: Pitcher rolling features ===")
    pitcher_features = add_pitcher_rolling_features(pgl)
    bgl = bgl.merge(pitcher_features, on=["opposing_pitcher_id", "game_date"], how="left")
    print(f"  opp_k_rate_career null rate: {bgl['opp_k_rate_career'].isna().mean():.1%}")

    print("\n=== Step H: Load spine for batter names ===")
    spine = pd.read_parquet(LOCAL_SPINE)
    spine["game_date"] = spine["game_date"].dt.strftime("%Y-%m-%d")
    spine["player_key"] = spine["batter_name"].map(normalize_name)

    # Merge player_key onto bgl
    name_lookup = spine[["batter", "player_key"]].drop_duplicates("batter")
    bgl = bgl.merge(name_lookup.rename(columns={"batter": "batter_id"}),
                    left_on="batter", right_on="batter_id", how="left")

    print("\n=== Step I: Add stand from PA data ===")
    bgl = add_stand(bgl, pa_full)

    print("\n=== Step J: Load and prep market data ===")
    market = pd.read_parquet(LOCAL_MARKET)
    market = market[market["line"].isin([0.5, 1.5])].copy()
    market["player_key"] = market["player_name"].map(normalize_name)
    market = add_market_features(market)

    print("\n=== Step K: Join batter features + market ===")
    # Batter features are at (player_key, game_date) — broadcast to all book rows
    batter_feature_cols = [
        "player_key", "game_date",
        "strikeouts", "plate_appearances",
        "k_roll_L1", "k_roll_L5", "k_roll_L10", "k_roll_L20",
        "k_roll_season", "k_roll_career",
        "pa_roll_career", "pa_roll_L5",
        "k_rate_career", "k_rate_L5",
        "opp_k_rate_career", "opp_k_rate_L5",
        "is_home", "stand",
    ]
    batter_features = bgl[batter_feature_cols].drop_duplicates(["player_key", "game_date"])

    spine_df = market.merge(batter_features, on=["player_key", "game_date"], how="inner")
    print(f"  Spine rows after join: {len(spine_df):,}")

    # Target
    spine_df["over_flag"] = (spine_df["strikeouts"] > spine_df["line"]).astype(int)
    spine_df["offered_line"] = spine_df["line"]

    # Final column order
    final_cols = [
        "player_key", "player_name", "game_date", "season",
        "bookmaker", "offered_line",
        "over_price", "under_price",
        "raw_implied_prob_over", "raw_implied_prob_under",
        "novig_prob_over", "novig_prob_under",
        "strikeouts", "plate_appearances", "over_flag",
        "k_roll_L1", "k_roll_L5", "k_roll_L10", "k_roll_L20",
        "k_roll_season", "k_roll_career",
        "pa_roll_career", "pa_roll_L5",
        "k_rate_career", "k_rate_L5",
        "opp_k_rate_career", "opp_k_rate_L5",
        "is_home", "stand",
        "consensus_over_odds_bin", "consensus_over_odds_bin_granular",
        "consensus_under_odds_bin", "consensus_under_odds_bin_granular",
        "min_line", "max_line",
        "min_raw_implied_prob_over", "max_raw_implied_prob_over",
        "min_raw_implied_prob_under", "max_raw_implied_prob_under",
        "event_id", "home_team", "away_team",
    ]
    final_cols = [c for c in final_cols if c in spine_df.columns]
    spine_df = spine_df[final_cols]

    # Verify grain: (player_key, game_date, bookmaker, line) is unique
    dup_check = spine_df.duplicated(subset=["player_key", "game_date", "bookmaker", "offered_line"])
    if dup_check.any():
        print(f"  WARNING: {dup_check.sum()} duplicate (player, game, book, line) rows — dropping")
        spine_df = spine_df[~dup_check]

    print(f"\nFinal spine: {len(spine_df):,} rows")
    print(f"Unique players: {spine_df['player_key'].nunique():,}")
    print(f"Date range: {spine_df['game_date'].min()} → {spine_df['game_date'].max()}")
    print(f"\nNull rates:")
    null_rates = spine_df[["k_roll_career","k_roll_L5","opp_k_rate_career","opp_k_rate_L5","is_home","stand"]].isna().mean().round(3)
    print(null_rates.to_string())

    # Aaron Judge spot-check
    judge = spine_df[spine_df["player_key"] == "aaron judge"].sort_values("game_date")
    if not judge.empty:
        print(f"\nAaron Judge — first 5 rows:")
        show_cols = ["game_date","bookmaker","offered_line","strikeouts","over_flag",
                     "k_roll_L5","k_roll_career","k_rate_career","opp_k_rate_career","is_home","stand"]
        print(judge[[c for c in show_cols if c in judge.columns]].head(5).to_string())

    # Save
    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    spine_df.to_parquet(LOCAL_OUT, index=False)
    print(f"\nSaved local: {LOCAL_OUT}")

    s3c = boto3.client("s3")
    buf = BytesIO()
    spine_df.to_parquet(buf, index=False)
    s3c.put_object(Bucket=S3_BUCKET, Key=S3_FEATURES, Body=buf.getvalue())
    print(f"Saved S3:    s3://{S3_BUCKET}/{S3_FEATURES}")


if __name__ == "__main__":
    main()

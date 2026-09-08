"""
Step 2 — Build feature spine at (player, game_date, bookmaker, offered_line) grain.

Batter rolling features (L1, L5, L10, L20, season, career):
  hits_roll_L1/L5/L10/L20/season/career
  ba_roll_L5/career, ab_roll_career

Opposing pitcher features:
  opp_h_rate_career, opp_h_rate_L5

Game context:
  is_home (1=home, 0=away), stand (L/R/S)

Market consensus features (book-invariant per player-game-line):
  consensus_over_odds_bin, consensus_over_odds_bin_granular
  consensus_under_odds_bin, consensus_under_odds_bin_granular
  consensus_line, min_line, max_line
  min/max_raw_implied_prob_over, min/max_raw_implied_prob_under

Per-book cols (edge calc only — NOT model inputs):
  novig_prob_over, raw_implied_prob_over, raw_implied_prob_under

Target:
  hits_actual, over_flag

Output:
  S3:    s3://the-odds-api-mt/mlb/batter_hits_model/spine/mlb_batter_hits_spine.parquet
  Local: ~/Downloads/tmp/mlb_batter_hits_spine.parquet

Usage:
  python src/mlb_batter_hits_modeling/scripts/20260727_build_spine.py
  python src/mlb_batter_hits_modeling/scripts/20260727_build_spine.py --no-s3
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
import pybaseball

warnings.filterwarnings("ignore")
pybaseball.cache.enable()

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET    = "the-odds-api-mt"
S3_ACTUALS   = "mlb/total_bases_model/actuals/mlb_batting_statcast.parquet"
S3_SPINE_OUT = "mlb/batter_hits_model/spine/mlb_batter_hits_spine.parquet"
LOCAL_MARKET  = Path.home() / "Downloads/tmp/mlb_batter_hits_market_raw.parquet"
LOCAL_OUT     = Path.home() / "Downloads/tmp/mlb_batter_hits_spine.parquet"

SEASON_DATES = {
    2024: ("2024-03-20", "2024-10-01"),
    2025: ("2025-03-18", "2025-10-01"),
    2026: ("2026-03-25", "2026-07-27"),
}

# Known Odds API ↔ Statcast name mismatches not caught by normalize_name().
NAME_MAP: dict[str, str] = {
    "daniel vogelbach":  "dan vogelbach",
    "donnie walton":     "donovan walton",
    "eddy alvarez":      "francisco alvarez",
    "josh kuroda grauer": "joshua kuroda grauer",  # Odds API uses Josh; Statcast uses Joshua
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
    # Collapse "C. J." → "cj" style pairs of single initials
    name = re.sub(r"\b([a-z]) ([a-z])\b", r"\1\2", name)
    name = re.sub(r"\b([a-z]) ([a-z])\b", r"\1\2", name)
    # Remove lone middle initial between spaces (e.g. "michael a taylor" → "michael taylor")
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
    """Load market from local cache (fast) or rebuild from S3."""
    if LOCAL_MARKET.exists():
        print(f"  Loading market from local: {LOCAL_MARKET}")
        market = pd.read_parquet(LOCAL_MARKET)
    else:
        print("  Local market not found — loading from S3 (slow)...")
        s3c = boto3.client("s3")
        frames = []
        paginator = s3c.get_paginator("list_objects_v2")
        for season in [2024, 2025, 2026]:
            prefix = f"mlb/batter_hits_model/market_raw/{season}/"
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
    """
    Dedup to (player_name, game_date, bookmaker, line) grain.
    batter_hits and batter_hits_alternate can produce duplicate rows per book.
    Keep the row with under_price (two-sided) when duplicated; otherwise keep first.
    """
    has_under = market[market["under_price"].notna()]
    no_under  = market[market["under_price"].isna()]

    key = ["player_name", "game_date", "bookmaker", "line"]
    # Among two-sided rows, keep one per key
    has_under_dedup = has_under.sort_values("over_price").drop_duplicates(subset=key, keep="first")
    # Among over-only rows, keep keys not already covered by two-sided
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


# ─── C: Pull Statcast PA for pitcher features + stand ────────────────────────

def build_pa_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      pitcher_features: (opposing_pitcher_id, game_date) → opp_h_rate_career/L5
      batter_context:   (batter, game_date) → stand, is_home, opposing_pitcher_id
    """
    frames = []
    keep   = ["batter", "pitcher", "game_pk", "game_date",
              "home_team", "away_team", "inning", "inning_topbot", "events", "stand"]
    for season, (start, end) in SEASON_DATES.items():
        print(f"  Pulling Statcast {season} (cached) ...")
        df = pybaseball.statcast(start, end)
        pa = df[df["events"].notna()][keep].copy()
        pa["season"] = season
        frames.append(pa)
        print(f"    {season}: {len(pa):,} PAs")
    pa_all = pd.concat(frames, ignore_index=True)
    pa_all["game_date"] = pd.to_datetime(pa_all["game_date"]).dt.strftime("%Y-%m-%d")

    # --- Starting pitchers ---
    inn1 = pa_all[pa_all["inning"] == 1].sort_values(["game_pk", "inning_topbot"])
    home_st = (
        inn1[inn1["inning_topbot"] == "Top"]
        .groupby("game_pk")["pitcher"].first()
        .reset_index().rename(columns={"pitcher": "home_starter"})
    )
    away_st = (
        inn1[inn1["inning_topbot"] == "Bot"]
        .groupby("game_pk")["pitcher"].first()
        .reset_index().rename(columns={"pitcher": "away_starter"})
    )
    starters = home_st.merge(away_st, on="game_pk", how="outer")
    pa_all = pa_all.merge(starters, on="game_pk", how="left")
    pa_all["is_home_batter"]     = (pa_all["inning_topbot"] == "Bot").astype(int)
    pa_all["opposing_pitcher_id"] = np.where(
        pa_all["is_home_batter"] == 1, pa_all["away_starter"], pa_all["home_starter"]
    )

    # --- Pitcher game logs ---
    hit_events = {"single", "double", "triple", "home_run"}
    pa_all["is_hit"] = pa_all["events"].isin(hit_events).astype(int)
    pgl = (
        pa_all.groupby(["pitcher", "game_date"])
        .agg(hits_allowed=("is_hit", "sum"), batters_faced=("is_hit", "count"))
        .reset_index()
    )
    pgl = pgl.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
    hits_c = pgl.groupby("pitcher")["hits_allowed"].transform(lambda x: x.shift(1).expanding().sum())
    bf_c   = pgl.groupby("pitcher")["batters_faced"].transform(lambda x: x.shift(1).expanding().sum())
    pgl["opp_h_rate_career"] = hits_c / bf_c.replace(0, np.nan)
    hits_l5 = pgl.groupby("pitcher")["hits_allowed"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    bf_l5   = pgl.groupby("pitcher")["batters_faced"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    pgl["opp_h_rate_L5"] = hits_l5 / bf_l5.replace(0, np.nan)
    pitcher_features = pgl[["pitcher", "game_date", "opp_h_rate_career", "opp_h_rate_L5"]].rename(
        columns={"pitcher": "opposing_pitcher_id"}
    )

    # --- Batter context: stand + opposing pitcher per (batter, game_date) ---
    batter_ctx = (
        pa_all.groupby(["batter", "game_date"])
        .agg(
            stand=("stand", "first"),
            opposing_pitcher_id=("opposing_pitcher_id", "first"),
        )
        .reset_index()
    )

    print(f"  Pitcher features: {len(pitcher_features):,} rows")
    print(f"  Batter context: {len(batter_ctx):,} rows")
    return pitcher_features, batter_ctx


# ─── D: Build rolling batter features ────────────────────────────────────────

def add_batter_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling hit features. Groups by batter MLBAM ID. shift(1) prevents lookahead."""
    df = df.sort_values(["batter", "game_date"]).reset_index(drop=True)

    for window, label in [(1, "L1"), (5, "L5"), (10, "L10"), (20, "L20")]:
        df[f"hits_roll_{label}"] = df.groupby("batter")["hits"].transform(
            lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()
        )

    df["hits_roll_season"] = df.groupby(["batter", "season"])["hits"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["hits_roll_career"] = df.groupby("batter")["hits"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["ab_roll_career"] = df.groupby("batter")["ab"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    hits_L5_s = df.groupby("batter")["hits"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    ab_L5_s   = df.groupby("batter")["ab"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    df["ba_roll_L5"] = hits_L5_s / ab_L5_s.replace(0, np.nan)

    hits_c_s = df.groupby("batter")["hits"].transform(lambda x: x.shift(1).expanding().sum())
    ab_c_s   = df.groupby("batter")["ab"].transform(lambda x: x.shift(1).expanding().sum())
    df["ba_roll_career"] = hits_c_s / ab_c_s.replace(0, np.nan)

    return df


# ─── E: Add market consensus + per-book features ─────────────────────────────

def add_market_features(market: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-book and consensus features.
    Must be called BEFORE renaming 'line' to 'offered_line'.
    """
    market = market.copy()
    market["raw_implied_prob_over"]  = 1.0 / market["over_price"]
    market["raw_implied_prob_under"] = 1.0 / market["under_price"]
    novig_total = market["raw_implied_prob_over"] + market["raw_implied_prob_under"]
    market["novig_prob_over"]  = market["raw_implied_prob_over"]  / novig_total
    market["novig_prob_under"] = market["raw_implied_prob_under"] / novig_total

    # Consensus at (player_key, game_date, line) — average decimal odds across books
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

    # Min/max features at (player_key, game_date) — across ALL lines and books
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
    market = add_market_features(market)  # called before renaming 'line'
    market = market.rename(columns={"line": "offered_line"})
    print(f"  Market ready: {len(market):,} rows")

    print("\n=== D: Load actuals ===")
    actuals = load_actuals()

    print("\n=== E: Pull PA data for pitcher features + stand ===")
    pitcher_features, batter_ctx = build_pa_features()

    # Join batter context (stand + opposing pitcher) into actuals
    actuals = actuals.merge(
        batter_ctx[["batter", "game_date", "stand", "opposing_pitcher_id"]],
        on=["batter", "game_date"],
        how="left",
    )
    print(f"  Opp pitcher null rate: {actuals['opposing_pitcher_id'].isna().mean():.1%}")
    print(f"  Stand null rate:       {actuals['stand'].isna().mean():.1%}")

    print("\n=== F: Build batter rolling features ===")
    actuals = add_batter_rolling_features(actuals)
    print(f"  Rolling features added. Shape: {actuals.shape}")

    print("\n=== G: Join pitcher rolling features ===")
    actuals = actuals.merge(
        pitcher_features,
        on=["opposing_pitcher_id", "game_date"],
        how="left",
    )
    print(f"  Opp h_rate null rate: {actuals['opp_h_rate_career'].isna().mean():.1%}")

    print("\n=== H: Join features to market spine ===")
    feature_cols = [
        "name_norm", "game_date",
        "hits", "ab", "is_home", "stand",
        "opposing_pitcher_id",
        "hits_roll_L1", "hits_roll_L5", "hits_roll_L10", "hits_roll_L20",
        "hits_roll_season", "hits_roll_career",
        "ba_roll_L5", "ba_roll_career", "ab_roll_career",
        "opp_h_rate_career", "opp_h_rate_L5",
        # omit 'season' — market already has it; including it causes _x/_y suffix collision
    ]
    spine = market.merge(
        actuals[feature_cols].rename(columns={"name_norm": "player_key"}),
        on=["player_key", "game_date"],
        how="left",
    )

    spine["hits_actual"] = spine["hits"]
    spine["over_flag"] = np.where(
        spine["hits_actual"].isna(),
        np.nan,
        (spine["hits_actual"] > spine["offered_line"]).astype(float),
    )
    spine["season"] = spine["season"].fillna(spine["game_date"].str[:4].astype(float))

    # Final dedup: mid-season trades can produce 2 event rows for same player+date.
    # Both get the same actuals attached. Keep the row with the lowest over_price
    # (sharper line = primary market; fallback to first).
    before_dedup = len(spine)
    spine = (
        spine.sort_values("over_price")
        .drop_duplicates(subset=["player_key", "game_date", "bookmaker", "offered_line"])
        .reset_index(drop=True)
    )
    if len(spine) < before_dedup:
        print(f"  Final dedup removed {before_dedup - len(spine):,} trade-duplicate rows")

    match_rate = spine["hits_actual"].notna().mean()
    print(f"  Spine rows: {len(spine):,}")
    print(f"  Match rate (market rows with actuals): {match_rate:.1%}")

    unmatched = (
        spine[spine["hits_actual"].isna()]["name_norm"]
        .value_counts()
        .head(20)
    )
    if len(unmatched):
        print("  Top unmatched players (Odds API → not in Statcast):")
        for name, cnt in unmatched.items():
            print(f"    '{name}': {cnt} rows")

    print("\n=== I: Spot-check — Freddie Freeman ===")
    ff_key = normalize_name("Freddie Freeman")
    ff = spine[spine["player_key"] == ff_key].sort_values(["game_date", "bookmaker", "offered_line"])
    print(f"  Freeman rows: {len(ff):,}")
    show_cols = [
        "game_date", "bookmaker", "offered_line", "hits_actual", "over_flag",
        "hits_roll_L5", "hits_roll_career", "ba_roll_career",
        "opp_h_rate_career", "novig_prob_over", "stand", "is_home",
    ]
    print(ff[show_cols].tail(10).to_string(index=False))

    # Leakage check: for a spot-check game, hits_roll_L5 must not include that game's actual
    ff_dedup = ff.drop_duplicates(subset=["game_date"]).sort_values("game_date")
    print("\n  Leakage check (hits_roll_L5 should NOT include same-game hits):")
    for _, row in ff_dedup.tail(5).iterrows():
        print(f"    {row['game_date']}: actual={row['hits_actual']}, hits_roll_L5={row['hits_roll_L5']:.3f}")

    print("\n=== J: Season-start check — first game of each season ===")
    # Season-level rolling should start at NaN for the first game of the season
    for season in [2024, 2025, 2026]:
        first_game = spine[spine["season"] == season].sort_values("game_date").iloc[0]
        print(f"  Season {season} first row: game_date={first_game['game_date']}, "
              f"hits_roll_season={first_game['hits_roll_season']}, "
              f"hits_roll_career={first_game['hits_roll_career']}")

    print("\n=== K: Save ===")
    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    spine.to_parquet(LOCAL_OUT, index=False)
    print(f"  Saved locally → {LOCAL_OUT}  ({len(spine):,} rows)")

    if not args.no_s3:
        buf = BytesIO()
        spine.to_parquet(buf, index=False)
        boto3.client("s3").put_object(Bucket=S3_BUCKET, Key=S3_SPINE_OUT, Body=buf.getvalue())
        print(f"  Saved to S3 → s3://{S3_BUCKET}/{S3_SPINE_OUT}")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
Step 2 — Feature Engineering / Spine Builder

Builds the labeled feature matrix at (player, game_date, bookmaker, line) grain.
No lookahead: all rolling features use strictly prior games (shift(1)).

Features built:
  Rolling walks (all windows): L1, L3, L5, L10, season, career, cross-season c5
  Related pitcher stats: strikeouts rolling L5/career, innings_pitched rolling L5/career
  Opponent walks drawn: opp_walks_against_season (how many walks this lineup draws per game)
  Game context: is_home, days_rest, games_into_season
  Consensus market: consensus_line, min_line, max_line
  Odds bins (player-game level, book-invariant):
    consensus_over_odds_bin, consensus_over_odds_bin_granular
    consensus_under_odds_bin, consensus_under_odds_bin_granular
    over_price_bucket_fine, under_price_bucket_fine
  Min/max raw implied probs: min/max of 1/decimal_odds across books
  Team context: team_run_line_point, team_moneyline_odds
  Per-book: novig_prob_over, novig_prob_under

Output:
  S3:    s3://the-odds-api-mt/mlb/pitcher_walks_model/spine/mlb_pitcher_walks_spine.parquet
  Local: ~/Downloads/tmp/mlb_pitcher_walks_spine.parquet

Usage:
  python src/mlb_pitcher_walks_modeling/scripts/20260706_build_spine.py
  python src/mlb_pitcher_walks_modeling/scripts/20260706_build_spine.py --dry-run
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

S3_BUCKET      = "the-odds-api-mt"
GAMELOG_PREFIX = "mlb/strikeouts_model/pitcher_gamelogs"
SPINE_KEY      = "mlb/pitcher_walks_model/spine/mlb_pitcher_walks_spine.parquet"

LOCAL_MARKET = Path.home() / "Downloads/tmp/mlb_pitcher_walks_market_raw.parquet"
LOCAL_SPINE  = Path.home() / "Downloads/tmp/mlb_pitcher_walks_spine.parquet"

SEASONS = [2024, 2025, 2026]


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    if not name:
        return ""
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = name.lower()
    name = re.sub(r"[^a-z ]", "", name)
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", name)
    return name.strip()


# ---------------------------------------------------------------------------
# American odds helpers
# ---------------------------------------------------------------------------

def american_to_decimal(american: float) -> float | None:
    if pd.isna(american):
        return None
    if american > 0:
        return american / 100 + 1
    if american < 0:
        return 100 / abs(american) + 1
    return None


def american_profit(american: float) -> float | None:
    """Profit per unit wagered (for ROI calculation)."""
    if pd.isna(american):
        return None
    if american > 0:
        return american / 100
    if american < 0:
        return 100 / abs(american)
    return None


def raw_implied_prob(american: float) -> float | None:
    dec = american_to_decimal(american)
    if dec is None or dec == 0:
        return None
    return 1.0 / dec


def novig_probs(over_american: float, under_american: float) -> tuple[float, float] | tuple[None, None]:
    rpo = raw_implied_prob(over_american)
    rpu = raw_implied_prob(under_american)
    if rpo is None or rpu is None or (rpo + rpu) == 0:
        return None, None
    denom = rpo + rpu
    return rpo / denom, rpu / denom


# ---------------------------------------------------------------------------
# Odds bucket helpers (mirroring strikeouts v5 fine bins)
# ---------------------------------------------------------------------------

def over_price_bucket_fine(american: float) -> str:
    """9-tier granular bin for American over odds (player-game level, book-invariant)."""
    if pd.isna(american):
        return "unknown"
    if american <= -300:
        return "hvy_fav_300plus"
    if american <= -200:
        return "hvy_fav_200_300"
    if american <= -110:
        return "fav_110_200"
    if american < 0:
        return "sl_fav_0_110"
    if american == 0:
        return "even"
    if american <= 110:
        return "sl_dog_0_110"
    if american <= 200:
        return "dog_110_200"
    if american <= 300:
        return "dog_200_300"
    return "hvy_dog_300plus"


def under_price_bucket_fine(american: float) -> str:
    """Same 9-tier bin for American under odds."""
    return over_price_bucket_fine(american)


def odds_bin_coarse(american: float) -> str:
    """Coarse 3-value bin: plus / minus / even."""
    if pd.isna(american):
        return "unknown"
    if american > 0:
        return "plus"
    if american < 0:
        return "minus"
    return "even"


def odds_bin_granular(american: float) -> str:
    """8-bucket granular bin."""
    if pd.isna(american):
        return "unknown"
    if american <= -300:
        return "-500_to_-300"
    if american <= -200:
        return "-300_to_-200"
    if american <= -110:
        return "-200_to_-110"
    if american < 0:
        return "-110_to_even"
    if american == 0:
        return "even_to_+110"
    if american <= 110:
        return "even_to_+110"
    if american <= 200:
        return "+110_to_+200"
    if american <= 300:
        return "+200_to_+300"
    return "+300_plus"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_gamelogs() -> pd.DataFrame:
    s3 = boto3.client("s3")
    frames = []
    for yr in SEASONS:
        resp = s3.get_object(Bucket=S3_BUCKET, Key=f"{GAMELOG_PREFIX}/{yr}.parquet")
        frames.append(pd.read_parquet(BytesIO(resp["Body"].read())))
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    return df


def load_market() -> pd.DataFrame:
    df = pd.read_parquet(LOCAL_MARKET)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df[df["market_key"] == "pitcher_walks"].copy()


# ---------------------------------------------------------------------------
# Rolling features (no lookahead — shift(1) within each group)
# ---------------------------------------------------------------------------

def rolling_mean_shift(series: pd.Series, window: int) -> pd.Series:
    """Rolling mean over `window` games, shifted by 1 to exclude current game."""
    return series.shift(1).rolling(window, min_periods=1).mean()


def build_gamelog_features(logs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all player-game level rolling features from gamelogs.
    Returns one row per (player_id, game_date) — the player-game level features
    that will be broadcast to all book rows in the spine.
    """
    logs = logs.copy()
    logs["game_date"] = pd.to_datetime(logs["game_date"])
    logs = logs.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    # ── Rolling walks (all windows) ──────────────────────────────────────────
    grp = logs.groupby("player_id")["walks"]
    logs["walks_roll_L1"]     = grp.shift(1)                                       # last game only
    logs["walks_roll_L3"]     = rolling_mean_shift(logs.groupby("player_id")["walks"].transform("cumcount"), 3)  # placeholder — recalc below
    logs["walks_roll_L3"]     = grp.apply(lambda s: s.shift(1).rolling(3, min_periods=1).mean()).reset_index(level=0, drop=True)
    logs["walks_roll_L5"]     = grp.apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).reset_index(level=0, drop=True)
    logs["walks_roll_L10"]    = grp.apply(lambda s: s.shift(1).rolling(10, min_periods=1).mean()).reset_index(level=0, drop=True)
    logs["walks_roll_career"] = grp.apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)

    # Season-scoped rolling (reset at season boundary)
    grp_szn = logs.groupby(["player_id", "season"])["walks"]
    logs["walks_roll_season"] = grp_szn.apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)

    # Cross-season 5-game rolling (spans season boundaries — uses player_id grouping only)
    logs["walks_roll_c5"] = grp.apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).reset_index(level=0, drop=True)

    # ── Related pitcher stats (K, IP, pitches) ───────────────────────────────
    for stat, col in [("strikeouts", "strikeouts"), ("innings_pitched", "innings_pitched"), ("pitches", "pitches")]:
        g = logs.groupby("player_id")[stat]
        logs[f"{col}_roll_L5"]     = g.apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).reset_index(level=0, drop=True)
        logs[f"{col}_roll_career"] = g.apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)

    # ── Days since last start ─────────────────────────────────────────────────
    logs["prev_game_date"] = logs.groupby("player_id")["game_date"].shift(1)
    logs["days_rest"] = (logs["game_date"] - logs["prev_game_date"]).dt.days

    # ── Games into season (1-indexed, shift so current game is not counted) ───
    logs["games_into_season"] = logs.groupby(["player_id", "season"]).cumcount()  # 0-indexed before shift

    return logs


def build_opponent_features(logs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling opponent walks-drawn rate for each team × game_date.
    opp_walks_against_season: how many walks per game this opponent team draws this season
    from opposing pitchers (rolling season avg, shifted to exclude current game).
    """
    logs = logs.copy()
    logs["game_date"] = pd.to_datetime(logs["game_date"])
    logs = logs.sort_values(["opponent_name", "game_date"])

    # For each pitcher start, the opponent receives walks_this_game.
    # Group by opponent_name + season, compute rolling mean of walks (shifted).
    opp_grp = logs.groupby(["opponent_name", "season"])["walks"]
    logs["opp_walks_against_season"] = opp_grp.apply(
        lambda s: s.shift(1).expanding().mean()
    ).reset_index(level=[0, 1], drop=True)

    return logs[["player_id", "game_date", "opponent_name", "opp_walks_against_season"]]


# ---------------------------------------------------------------------------
# Market-level features (player-game level, book-invariant)
# ---------------------------------------------------------------------------

def build_market_features(market: pd.DataFrame) -> pd.DataFrame:
    """
    Compute consensus and min/max features at (player_name_norm, game_date) level.
    These are the same for every book row of the same player-game.
    """
    market = market.copy()
    market["name_norm"] = market["player_name"].apply(normalize_name)

    # Convert American odds to raw implied probs
    market["raw_prob_over"]  = market["over_price"].apply(raw_implied_prob)
    market["raw_prob_under"] = market["under_price"].apply(raw_implied_prob)

    # Consensus American odds per player-game: simple mean across books at the consensus line
    # Consensus line = modal line across books for this player-game
    pg = market.groupby(["name_norm", "game_date"])

    # Modal (most common) line
    consensus_line = pg["line"].agg(lambda x: x.mode()[0]).rename("consensus_line")

    # Mean over/under American odds across all books (at any line — captures overall market view)
    consensus_over_am  = pg["over_price"].mean().rename("consensus_over_american")
    consensus_under_am = pg["under_price"].mean().rename("consensus_under_american")

    # Min/max line
    min_line = pg["line"].min().rename("min_line")
    max_line = pg["line"].max().rename("max_line")

    # Min/max raw implied probs
    min_rpo = pg["raw_prob_over"].min().rename("min_raw_implied_prob_over")
    max_rpo = pg["raw_prob_over"].max().rename("max_raw_implied_prob_over")
    min_rpu = pg["raw_prob_under"].min().rename("min_raw_implied_prob_under")
    max_rpu = pg["raw_prob_under"].max().rename("max_raw_implied_prob_under")

    mkt_features = pd.concat([
        consensus_line, consensus_over_am, consensus_under_am,
        min_line, max_line,
        min_rpo, max_rpo, min_rpu, max_rpu,
    ], axis=1).reset_index()

    # Odds bins (from consensus American odds)
    mkt_features["consensus_over_odds_bin"]          = mkt_features["consensus_over_american"].apply(odds_bin_coarse)
    mkt_features["consensus_over_odds_bin_granular"] = mkt_features["consensus_over_american"].apply(odds_bin_granular)
    mkt_features["consensus_under_odds_bin"]         = mkt_features["consensus_under_american"].apply(odds_bin_coarse)
    mkt_features["consensus_under_odds_bin_granular"] = mkt_features["consensus_under_american"].apply(odds_bin_granular)

    return mkt_features


# ---------------------------------------------------------------------------
# Team context (pitcher's team h2h / run line)
# ---------------------------------------------------------------------------

def add_team_context(spine: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve pitcher's team moneyline and run line point from the market's game-level columns.
    Uses is_home from gamelogs to pick home vs away consensus odds.
    """
    spine = spine.copy()
    # is_home from gamelogs (already in spine via gamelog join)
    # home pitcher → team_moneyline = consensus_home_moneyline
    # away pitcher → team_moneyline = consensus_away_moneyline
    spine["team_moneyline_odds"] = np.where(
        spine["is_home"] == 1,
        spine["consensus_home_moneyline"],
        spine["consensus_away_moneyline"],
    )
    spine["team_run_line_point"] = np.where(
        spine["is_home"] == 1,
        spine["home_run_line_point"],
        spine["away_run_line_point"],
    )
    return spine


# ---------------------------------------------------------------------------
# Per-book features (novig, price buckets)
# ---------------------------------------------------------------------------

def add_per_book_features(spine: pd.DataFrame) -> pd.DataFrame:
    spine = spine.copy()

    # Per-book novig using that book's own over/under price
    novig = spine.apply(
        lambda r: novig_probs(r["over_price"], r["under_price"]), axis=1, result_type="expand"
    )
    spine["novig_prob_over"]  = novig[0]
    spine["novig_prob_under"] = novig[1]

    # Per-book price bucket (fine 9-tier) — uses THIS book's price at THIS line
    spine["over_price_bucket_fine"]  = spine["over_price"].apply(over_price_bucket_fine)
    spine["under_price_bucket_fine"] = spine["under_price"].apply(under_price_bucket_fine)

    # Per-book profit multiplier (for ROI calculation in backtest)
    spine["profit_if_over_wins"]  = spine["over_price"].apply(american_profit)
    spine["profit_if_under_wins"] = spine["under_price"].apply(american_profit)

    return spine


# ---------------------------------------------------------------------------
# Label outcomes
# ---------------------------------------------------------------------------

def label_outcomes(spine: pd.DataFrame) -> pd.DataFrame:
    spine = spine.copy()
    spine["outcome"] = np.where(
        spine["walks"] > spine["line"], "over",
        np.where(spine["walks"] < spine["line"], "under", "push")
    )
    spine["target_over"] = (spine["outcome"] == "over").astype(int)
    return spine


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> None:
    print("Loading gamelogs from S3...")
    logs = load_gamelogs()
    print(f"  Gamelogs: {len(logs):,} rows, {logs['player_name'].nunique():,} pitchers, seasons {sorted(logs['season'].unique())}")

    print("Building gamelog features...")
    logs_feat = build_gamelog_features(logs)

    print("Building opponent features...")
    opp_feat = build_opponent_features(logs)

    print("Loading market data...")
    market = load_market()
    market["name_norm"] = market["player_name"].apply(normalize_name)
    print(f"  Market: {len(market):,} rows, {market['player_name'].nunique():,} pitchers")

    print("Building market-level features...")
    mkt_feat = build_market_features(market)

    # ── Join gamelogs → market ───────────────────────────────────────────────
    print("Joining gamelogs to market...")
    logs_feat["name_norm"] = logs_feat["player_name"].apply(normalize_name)
    logs_feat["game_date"] = pd.to_datetime(logs_feat["game_date"])
    market["game_date"]    = pd.to_datetime(market["game_date"])

    gamelog_cols = [
        "name_norm", "game_date", "season", "player_name", "player_id",
        "walks", "strikeouts", "innings_pitched", "pitches", "is_home",
        "opponent_name", "games_into_season", "days_rest",
        "walks_roll_L1", "walks_roll_L3", "walks_roll_L5", "walks_roll_L10",
        "walks_roll_career", "walks_roll_season", "walks_roll_c5",
        "strikeouts_roll_L5", "strikeouts_roll_career",
        "innings_pitched_roll_L5", "innings_pitched_roll_career",
        "pitches_roll_L5", "pitches_roll_career",
    ]
    logs_slim = logs_feat[gamelog_cols].drop_duplicates(subset=["name_norm", "game_date"])

    # Market → gamelog join
    spine = market.merge(
        logs_slim,
        on=["name_norm", "game_date"],
        how="left",
        suffixes=("", "_log"),
    )

    # Join opponent features
    opp_feat["game_date"] = pd.to_datetime(opp_feat["game_date"])
    opp_feat["name_norm"] = opp_feat.apply(
        lambda r: normalize_name(r["opponent_name"]) if "opponent_name" in r else "", axis=1
    )
    spine = spine.merge(
        opp_feat[["player_id", "game_date", "opp_walks_against_season"]].drop_duplicates(),
        on=["player_id", "game_date"],
        how="left",
    )

    # Join market-level features
    mkt_feat["game_date"] = pd.to_datetime(mkt_feat["game_date"])
    spine = spine.merge(mkt_feat, on=["name_norm", "game_date"], how="left")

    # ── Add team context ─────────────────────────────────────────────────────
    spine = add_team_context(spine, logs_feat)

    # ── Add per-book features ────────────────────────────────────────────────
    print("Computing per-book features (novig, price buckets)...")
    spine = add_per_book_features(spine)

    # ── Label outcomes ───────────────────────────────────────────────────────
    spine = label_outcomes(spine)

    # ── Add player_key (normalized name + player_id) ─────────────────────────
    spine["player_key"] = spine["player_id"].astype(str) + "_" + spine["name_norm"].str.replace(" ", "_")

    # ── Drop rows with no gamelog match (walks is null) ──────────────────────
    matched = spine["walks"].notna()
    print(f"\nJoin quality:")
    print(f"  Total rows:   {len(spine):,}")
    print(f"  Matched:      {matched.sum():,} ({matched.mean():.1%})")
    print(f"  Unmatched:    {(~matched).sum():,} ({(~matched).mean():.1%})")

    spine_matched = spine[matched].copy()

    # ── Null rates by feature ────────────────────────────────────────────────
    feature_cols = [
        "walks_roll_L1", "walks_roll_L3", "walks_roll_L5", "walks_roll_L10",
        "walks_roll_career", "walks_roll_season", "walks_roll_c5",
        "strikeouts_roll_L5", "strikeouts_roll_career",
        "innings_pitched_roll_L5", "innings_pitched_roll_career",
        "pitches_roll_L5", "pitches_roll_career",
        "opp_walks_against_season", "days_rest", "games_into_season",
        "consensus_line", "min_line", "max_line",
        "consensus_over_american", "consensus_under_american",
        "min_raw_implied_prob_over", "max_raw_implied_prob_over",
        "min_raw_implied_prob_under", "max_raw_implied_prob_under",
        "novig_prob_over", "novig_prob_under",
        "team_moneyline_odds", "team_run_line_point",
        "over_price_bucket_fine", "under_price_bucket_fine",
        "consensus_over_odds_bin", "consensus_over_odds_bin_granular",
        "consensus_under_odds_bin", "consensus_under_odds_bin_granular",
    ]
    print("\nNull rates by feature column:")
    for col in feature_cols:
        if col in spine_matched.columns:
            null_rate = spine_matched[col].isna().mean()
            flag = " ⚠" if null_rate > 0.10 else ""
            print(f"  {col:<45} {null_rate:.1%}{flag}")

    # ── Spot-check: Freddy Peralta ───────────────────────────────────────────
    print("\n=== Spot-check: Freddy Peralta ===")
    fp = spine_matched[spine_matched["player_name"] == "Freddy Peralta"].sort_values("game_date")
    show_cols = [
        "game_date", "season", "bookmaker", "line", "walks",
        "walks_roll_L5", "walks_roll_season", "walks_roll_career",
        "strikeouts_roll_L5", "days_rest", "games_into_season",
        "novig_prob_over", "novig_prob_under", "outcome",
    ]
    show_cols = [c for c in show_cols if c in fp.columns]
    print(fp[show_cols].tail(12).to_string(index=False))

    # ── Leakage check ────────────────────────────────────────────────────────
    print("\n=== Leakage check (Freddy Peralta walks_roll_L5 vs actual) ===")
    fp_check = fp[["game_date", "walks", "walks_roll_L5"]].drop_duplicates("game_date").sort_values("game_date")
    print(fp_check.tail(8).to_string(index=False))

    if dry_run:
        print("\n[DRY RUN] Spine not saved.")
        return

    # ── Save ─────────────────────────────────────────────────────────────────
    LOCAL_SPINE.parent.mkdir(parents=True, exist_ok=True)
    spine_matched.to_parquet(LOCAL_SPINE, index=False)
    print(f"\nSaved locally → {LOCAL_SPINE} ({len(spine_matched):,} rows)")

    buf = BytesIO()
    spine_matched.to_parquet(buf, index=False)
    boto3.client("s3").put_object(Bucket=S3_BUCKET, Key=SPINE_KEY, Body=buf.getvalue())
    print(f"Saved to S3   → s3://{S3_BUCKET}/{SPINE_KEY}")

    print(f"\nSpine summary:")
    print(f"  Grain: (player_key, game_date, bookmaker, line)")
    print(f"  Total rows: {len(spine_matched):,}")
    print(f"  Unique player-games: {spine_matched.groupby(['player_key','game_date']).ngroups:,}")
    print(f"  Seasons: {sorted(spine_matched['season'].unique())}")
    print(f"  Books: {sorted(spine_matched['bookmaker'].unique())}")
    print(f"  Lines: {sorted(spine_matched['line'].unique())}")
    print(f"  Outcome dist: {spine_matched['outcome'].value_counts().to_dict()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)

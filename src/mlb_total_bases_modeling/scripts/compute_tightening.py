"""
Compute tightening watch for MLB total bases props.

Identifies players whose market conditions have drifted adversely vs. their
season-to-date baseline — indicating books are reacting to sharp action.

Uses only first-seen snapshot rows (one data point per player+game_date) to
compute rolling averages without double-counting.

Usage:
  uv run python src/mlb_total_bases_modeling/scripts/compute_tightening.py
  uv run python src/mlb_total_bases_modeling/scripts/compute_tightening.py --gameday 2026-09-05
  uv run python src/mlb_total_bases_modeling/scripts/compute_tightening.py --gameday 2026-09-05 --players "Aaron Judge" "Shohei Ohtani"

Output schema (one row per player, filtered to players arg):
  player_name, n_games_30d, n_games_season,
  avg_line_30d, avg_line_season, avg_under_odds_30d, avg_under_odds_season,
  today_modal_line, today_avg_under_odds,
  line_drift, odds_drift_cents, tightening_flag
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional

import boto3
import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

S3_BUCKET = "the-odds-api-mt"
S3_PREFIX = "mlb/total_bases_model/prop_snapshots"

ROLLING_DAYS = 30


def _tightening_flag(worst_cents: Optional[float]) -> str:
    if worst_cents is None:
        return "unknown"
    magnitude = abs(worst_cents)
    if magnitude < 5:
        return "ok"
    if magnitude < 10:
        return "mild"
    if magnitude < 20:
        return "moderate"
    if magnitude < 30:
        return "strong"
    return "severe"


def _load_season_first_seen(s3_client, season: int) -> pd.DataFrame:
    """
    Load all first-seen snapshot rows for the season from S3.
    Only loads rows where binary_player_game_first_seen=True to avoid
    double-counting within a day.
    """
    prefix = f"{S3_PREFIX}/{season}/"
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        keys = [
            obj["Key"]
            for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix)
            for obj in page.get("Contents", [])
            if obj["Key"].endswith(".parquet")
        ]
    except Exception:
        return pd.DataFrame()

    if not keys:
        return pd.DataFrame()

    frames = []
    for key in sorted(keys):
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            df = pd.read_parquet(BytesIO(obj["Body"].read()))
            # Keep only first-seen rows
            if "binary_player_game_first_seen" in df.columns:
                df = df[df["binary_player_game_first_seen"].astype(bool)]
            if not df.empty:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _modal_line(lines: pd.Series) -> Optional[float]:
    """Most common line value in a series."""
    counts = lines.dropna().value_counts()
    if counts.empty:
        return None
    return float(counts.index[0])


def _median_odds(odds: pd.Series) -> Optional[float]:
    """Median of numeric odds values."""
    valid = odds.dropna()
    if valid.empty:
        return None
    return float(valid.median())


def _per_game_agg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate snapshot rows to one row per (player_name, game_date):
      - modal_line: most common line across all books for this player+game
      - avg_under_odds: median under odds across all books
    """
    rows = []
    for (player, gd), grp in df.groupby(["player_name", "game_date_et"]):
        rows.append({
            "player_name":    player,
            "game_date":      gd,
            "modal_line":     _modal_line(grp["under_line"]),
            "avg_under_odds": _median_odds(grp["under_american_odds"]),
        })
    return pd.DataFrame(rows)


def compute_tightening(
    game_date: str,
    season: int,
    players: Optional[list[str]] = None,
    s3_client=None,
    strategy_only: bool = True,
    markets: Optional[list[str]] = None,
    lines: Optional[list[float]] = None,
) -> pd.DataFrame:
    """
    Compute tightening watch for the given game_date.

    Args:
        game_date:      YYYY-MM-DD
        season:         calendar year
        players:        if provided, filter output to these players only
        s3_client:      optional boto3 S3 client (injected for testing)
        strategy_only:  when True, filter to markets + lines before aggregation
        markets:        list of market_key values (from config strategy.markets)
        lines:          list of line values (from config strategy.lines)

    Returns DataFrame with one row per player, or empty DataFrame if no data.
    """
    if s3_client is None:
        s3_client = boto3.client("s3")

    all_df = _load_season_first_seen(s3_client, season)
    if all_df.empty:
        return pd.DataFrame()

    if strategy_only and markets is not None and lines is not None:
        all_df = all_df[
            all_df["market_key"].isin(markets)
            & all_df["under_line"].isin(lines)
            & all_df["under_american_odds"].notna()
        ].copy()
        if all_df.empty:
            return pd.DataFrame()

    # Per-game aggregates
    per_game = _per_game_agg(all_df)
    if per_game.empty:
        return pd.DataFrame()

    gd_date = date.fromisoformat(game_date)
    cutoff_30d = (gd_date - timedelta(days=ROLLING_DAYS)).isoformat()

    # Today's data
    today_df = per_game[per_game["game_date"] == game_date]
    if today_df.empty:
        return pd.DataFrame()

    results = []
    target_players = players if players else today_df["player_name"].unique().tolist()

    for player in target_players:
        today_row = today_df[today_df["player_name"] == player]
        if today_row.empty:
            continue

        today_modal_line     = today_row["modal_line"].iloc[0]
        today_avg_under_odds = today_row["avg_under_odds"].iloc[0]

        # Season-to-date: all prior game_dates (exclude today)
        season_mask = (
            (per_game["player_name"] == player)
            & (per_game["game_date"] < game_date)
        )
        season_df = per_game[season_mask]

        # 30-day: prior game_dates within rolling window
        rolling_mask = season_mask & (per_game["game_date"] >= cutoff_30d)
        rolling_df = per_game[rolling_mask]

        n_season = len(season_df)
        n_30d    = len(rolling_df)

        avg_line_season    = float(season_df["modal_line"].mean()) if n_season > 0 else None
        avg_line_30d       = float(rolling_df["modal_line"].mean()) if n_30d > 0 else None
        avg_odds_season    = float(season_df["avg_under_odds"].mean()) if n_season > 0 else None
        avg_odds_30d       = float(rolling_df["avg_under_odds"].mean()) if n_30d > 0 else None

        # Drift vs season baseline (positive = adverse for UNDER bettors: line went up or odds shortened)
        line_drift = (
            (float(today_modal_line) - avg_line_season)
            if today_modal_line is not None and avg_line_season is not None
            else None
        )
        odds_drift = (
            (float(today_avg_under_odds) - avg_odds_season)
            if today_avg_under_odds is not None and avg_odds_season is not None
            else None
        )

        # Flag based on absolute worst of line-drift-in-odds-equivalent or odds_drift
        # Use odds_drift as the primary signal; line_drift is reported separately
        flag = _tightening_flag(odds_drift)

        results.append({
            "player_name":          player,
            "n_games_30d":          n_30d,
            "n_games_season":       n_season,
            "avg_line_30d":         avg_line_30d,
            "avg_line_season":      avg_line_season,
            "avg_under_odds_30d":   avg_odds_30d,
            "avg_under_odds_season": avg_odds_season,
            "today_modal_line":     today_modal_line,
            "today_avg_under_odds": today_avg_under_odds,
            "line_drift":           line_drift,
            "odds_drift_cents":     odds_drift,
            "tightening_flag":      flag,
        })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values("odds_drift_cents", ascending=True).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Compute tightening watch for MLB total bases.")
    parser.add_argument(
        "--gameday",
        default=date.today().isoformat(),
        help="Game date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--players",
        nargs="+",
        default=None,
        help="Player names to filter to (default: all players with data today).",
    )
    args = parser.parse_args()
    game_date = args.gameday
    season = int(game_date[:4])

    print(f"Computing tightening watch for {game_date} (season {season})...")
    df = compute_tightening(game_date, season, players=args.players)

    if df.empty:
        print("No tightening data — no snapshot data found for this date/season.")
        return

    print(f"\n{len(df):,} players analysed.\n")

    flagged = df[df["tightening_flag"] != "ok"]
    if flagged.empty:
        print("No tightening signals (all players at 'ok' level).")
    else:
        print(f"Tightening signals ({len(flagged)} players):")
        print(
            flagged[["player_name", "n_games_season", "avg_under_odds_season",
                      "today_avg_under_odds", "odds_drift_cents", "tightening_flag"]]
            .to_string(index=False)
        )

    print("\nAll players:")
    print(
        df[["player_name", "n_games_season", "today_modal_line",
            "today_avg_under_odds", "line_drift", "odds_drift_cents", "tightening_flag"]]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

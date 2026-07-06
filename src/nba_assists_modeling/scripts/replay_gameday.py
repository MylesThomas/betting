"""
Replay the assists pipeline for a past gameday using historical props from a local parquet.

Usage:
    python src/nba_assists_modeling/scripts/replay_gameday.py \
        --gameday 2026-06-13 \
        --props ~/Downloads/tmp/assists_eda/assists_props_raw.parquet

The local parquet must have columns: player, player_key, bookmaker, prop_line,
over_odds, under_odds, game_date.

Reads spine as-of gameday from S3 (no lookahead — uses last available row per
player with game_date <= gameday, where features are already shift(1)-lagged).

Saves recommendations to S3 and sends email, exactly like the live pipeline.
Then automatically runs settle for the same gameday.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.nba_assists_modeling.scripts.run_pipeline import (
    S3_BUCKET, S3_PREFIX, SPINE_KEY,
    EDGE_THRESHOLD_PRIMARY, EDGE_THRESHOLD_SHOW,
    american_profit, build_bet_rows, build_html, filter_bets,
    load_spine_latest, s3_get_parquet, s3_put_csv, score,
    send_email, publish_sns,
    current_nba_season,
)


def load_spine_as_of(spine: pd.DataFrame, gameday: str) -> pd.DataFrame:
    """Last row per player with game_date <= gameday (features are already lag-shifted)."""
    spine["game_date"] = pd.to_datetime(spine["game_date"]).dt.date
    cutoff = pd.Timestamp(gameday).date()
    sub = spine[spine["game_date"] <= cutoff].copy()
    sub = sub.sort_values("game_date")
    latest = (
        sub.dropna(subset=["ast_roll_20"])
        .groupby("player_key", as_index=False)
        .last()
    )
    return latest[["player_key", "game_date", "ast_roll_20"]].rename(
        columns={"game_date": "last_game_date"}
    )


def load_props_from_parquet(path: str, gameday: str) -> pd.DataFrame:
    """
    Read historical props parquet and pivot to long format expected by build_bet_rows:
    columns: player, player_key, bookmaker, game_date, side, prop_line, odds, home_team, away_team
    """
    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    df = df[df["game_date"] == gameday].copy()
    if df.empty:
        raise ValueError(f"No props found for {gameday} in {path}")

    overs = df[["player", "player_key", "bookmaker", "game_date", "prop_line", "over_odds"]].copy()
    overs["side"] = "over"
    overs = overs.rename(columns={"over_odds": "odds"})

    unders = df[["player", "player_key", "bookmaker", "game_date", "prop_line", "under_odds"]].copy()
    unders["side"] = "under"
    unders = unders.rename(columns={"under_odds": "odds"})

    long = pd.concat([overs, unders], ignore_index=True)
    long = long.dropna(subset=["odds"])
    long["home_team"] = ""
    long["away_team"] = ""
    return long


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", required=True)
    parser.add_argument("--props", default=str(Path.home() / "Downloads/tmp/assists_eda/assists_props_raw.parquet"))
    args = parser.parse_args()
    gameday = args.gameday

    print(f"\nNBA Assists Replay | gameday={gameday}", flush=True)

    print("Loading spine from S3 (as-of gameday)...", flush=True)
    spine = s3_get_parquet(SPINE_KEY)
    spine_latest = load_spine_as_of(spine, gameday)
    print(f"  Spine players available: {len(spine_latest):,}")

    print(f"Loading props from {args.props}...", flush=True)
    props = load_props_from_parquet(args.props, gameday)
    print(f"  Raw prop rows: {len(props):,} ({props['player_key'].nunique()} players, {props['bookmaker'].nunique()} books)")

    df   = build_bet_rows(props, spine_latest)
    df   = score(df)
    n_scored = len(df)
    print(f"  Players scored: {n_scored}")

    bets = filter_bets(df)
    print(f"  Qualifying bets (edge>={EDGE_THRESHOLD_SHOW}): {len(bets)}")
    print(f"  Primary bets   (edge>={EDGE_THRESHOLD_PRIMARY}): {bets['is_primary'].sum()}")

    if not bets.empty:
        print("\nTop bets:")
        print(bets[["player_key", "consensus_line", "best_over_odds", "p_over_model", "p_over_market", "edge", "is_primary", "ast_roll_20"]].head(10).to_string(index=False))

    # Save to S3
    rec_key = f"{S3_PREFIX}/daily_runs/{gameday}/recommendations.csv"
    save_cols = [
        "player", "player_key", "game_date", "season",
        "consensus_line", "min_line", "max_line", "n_books",
        "best_over_odds", "best_over_book", "avg_over_profit", "avg_under_profit",
        "ast_roll_20", "yhat", "p_over_model", "p_over_market", "edge", "is_primary",
    ]
    recs_to_save = bets[[c for c in save_cols if c in bets.columns]]
    s3_put_csv(rec_key, recs_to_save)
    print(f"\n  Saved → s3://{S3_BUCKET}/{rec_key}")

    # Send pipeline email
    n_primary = int(bets["is_primary"].sum()) if not bets.empty else 0
    subject   = f"[REPLAY] NBA Assists {gameday} — {n_primary} primary · {len(bets)} total bets"
    html_body = build_html(bets, gameday, n_scored)
    send_email(subject, html_body)

    # Auto-run settle
    print(f"\nRunning settle for {gameday}...", flush=True)
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src/nba_assists_modeling/scripts/settle_assists.py"),
         "--gameday", gameday],
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        print("  Settle exited with non-zero status — check output above")

    print("\nReplay complete.")


if __name__ == "__main__":
    main()

"""
Score a historical game day using the trained model and real spine odds.

Loads the spine from S3, filters to the given gameday, scores through the
production model, applies the production filter (QB, line >= 6.5, edge >= 3pp),
and uploads the recommendations CSV to S3.

Usage:
  python src/nfl_rush_attempts_modeling/scripts/score_historical.py --gameday 2025-10-05
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import boto3
import nfl_data_py as nfl
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPTS_DIR.parents[2]
sys.path.insert(0, str(SCRIPTS_DIR))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from run_pipeline import (
    S3_BUCKET, S3_PREFIX,
    load_s3_pkl, load_s3_parquet,
    engineer_features, score, filter_bets,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", required=True, help="YYYY-MM-DD")
    args    = parser.parse_args()
    gameday = args.gameday
    season  = int(gameday[:4])

    print(f"\nNFL Rush Attempts — score historical — gameday={gameday}")
    print("=" * 60)

    # ── Load spine ────────────────────────────────────────────────────────────
    print("Loading spine from S3...")
    spine = load_s3_parquet(f"{S3_PREFIX}/spine/nfl_rush_attempts_spine.parquet")
    if spine is None or spine.empty:
        sys.exit("  ERROR: spine not found — run update_spine.py first")
    if "player_name_norm" in spine.columns and "player_norm" not in spine.columns:
        spine = spine.rename(columns={"player_name_norm": "player_norm"})
    print(f"  Spine rows: {len(spine):,}")

    # ── Load model artifacts ──────────────────────────────────────────────────
    print("Loading model artifacts from S3...")
    model_artifact = load_s3_pkl(f"{S3_PREFIX}/artifacts/best_model.pkl")
    cdfs           = load_s3_pkl(f"{S3_PREFIX}/artifacts/residual_cdfs.pkl")
    print(f"  Model: {model_artifact['model_type']}  features: {model_artifact['features']}")

    # ── Map gameday → NFL week ────────────────────────────────────────────────
    print(f"Mapping {gameday} to NFL week...")
    sched = nfl.import_schedules([season])
    sched = sched[sched["game_type"] == "REG"].copy()
    sched["gameday_str"] = pd.to_datetime(sched["gameday"]).dt.strftime("%Y-%m-%d")
    weeks = sched[sched["gameday_str"] == gameday]["week"].unique().tolist()
    if not weeks:
        sys.exit(f"  No schedule entries for {gameday}")
    print(f"  {gameday} = Week(s) {weeks}")

    # ── Filter to rows with real Odds API lines ───────────────────────────────
    day_rows = spine[
        (spine["season"] == season) &
        (spine["week"].isin(weeks)) &
        (spine["consensus_point"].notna())
    ].copy()

    pos_counts = day_rows["position"].value_counts().to_dict() if "position" in day_rows.columns else {}
    print(f"  Rows with real lines: {len(day_rows)} {pos_counts}")

    if day_rows.empty:
        print(f"  No rows with odds for {gameday} — nothing to score")
        _upload_empty(gameday)
        return

    # ── Prep for scoring ──────────────────────────────────────────────────────
    day_rows["line"]           = day_rows["consensus_point"]
    day_rows["book_over_prob"] = day_rows["consensus_over_prob"]
    day_rows["book"]           = "consensus"
    day_rows["event_id"]       = f"hist_{gameday}"

    # Fill NaN in rolling feature cols (same as join_spine_features at live runtime)
    for col in [c for c in day_rows.columns
                if c.startswith(("carry_rate_", "rush_yards_", "opp_carry_"))]:
        day_rows[col] = day_rows[col].fillna(0)
    for col in [c for c in day_rows.columns if c.startswith("over_rate_")]:
        day_rows[col] = day_rows[col].fillna(0.5)
    for col in ["pos_RB", "pos_QB", "is_home", "games_played", "game_total", "is_playoff"]:
        if col in day_rows.columns:
            day_rows[col] = day_rows[col].fillna(0)

    # ── Score ─────────────────────────────────────────────────────────────────
    print("Scoring...")
    scored = score(day_rows, model_artifact, cdfs)
    print(f"  Scored {len(scored)} rows  "
          f"p_model range [{scored['p_model'].min():.3f}, {scored['p_model'].max():.3f}]")

    # Compute American under price from consensus no-vig probability
    p_under = 1 - scored["book_over_prob"]
    scored["under_price"] = np.where(
        p_under >= 0.5,
        -(p_under / (1 - p_under) * 100).round(),
        ((1 - p_under) / p_under * 100).round(),
    ).clip(-500, 500).astype(int)

    # ── Apply production filter ───────────────────────────────────────────────
    bets = filter_bets(scored)
    print(f"  Qualifying bets (QB UNDER, line≥6.5, edge≥3pp): {len(bets)}")

    if not bets.empty:
        print(f"\n  {'Player':<26} {'Pos':<4} {'Team':<5} {'Line':>5}  "
              f"{'p_model':>8}  {'p_mkt':>7}  {'edge':>7}  {'odds':>6}")
        team_col = "team" if "team" in bets.columns else "recent_team"
        for _, r in bets.iterrows():
            print(f"  {r['player_display_name']:<26} {r['position']:<4} "
                  f"{r.get(team_col, '?'):<5} {r['line']:>5.1f}  "
                  f"{r['p_model']*100:>7.1f}%  {r['p_market']*100:>6.1f}%  "
                  f"{r['edge']*100:>+6.1f}pp  {int(r['offered_price']):>+d}")

    # ── Upload to S3 ──────────────────────────────────────────────────────────
    team_col = "team" if "team" in bets.columns else "recent_team"
    recs = bets[["player_norm", "player_display_name", team_col, "position",
                 "line", "p_model", "p_market", "edge",
                 "book", "event_id", "direction", "offered_price"]].copy()
    if team_col != "team":
        recs = recs.rename(columns={team_col: "team"})

    key = f"{S3_PREFIX}/daily_runs/{gameday}/recommendations.csv"
    boto3.client("s3").put_object(
        Bucket=S3_BUCKET, Key=key, Body=recs.to_csv(index=False).encode()
    )
    print(f"\n  Uploaded {len(recs)} recs → s3://{S3_BUCKET}/{key}")


def _upload_empty(gameday: str) -> None:
    cols = ["player_norm", "player_display_name", "team", "position",
            "line", "p_model", "p_market", "edge",
            "book", "event_id", "direction", "offered_price"]
    key = f"{S3_PREFIX}/daily_runs/{gameday}/recommendations.csv"
    boto3.client("s3").put_object(
        Bucket=S3_BUCKET, Key=key,
        Body=pd.DataFrame(columns=cols).to_csv(index=False).encode()
    )
    print(f"  Uploaded empty recs → s3://{S3_BUCKET}/{key}")


if __name__ == "__main__":
    main()

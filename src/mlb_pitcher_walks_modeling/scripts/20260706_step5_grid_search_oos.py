"""
Step 5 — OOS grid search for MLB pitcher_walks model.

Sweeps: shrinkage × edge_threshold × direction × odds_bucket
OOS period: 2026 test set (train 2025, no future data used)

p_final = (1 - shrinkage) * p_model + shrinkage * novig_prob_over

Usage:
  python src/mlb_pitcher_walks_modeling/scripts/20260706_step5_grid_search_oos.py
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

OOF_PREDS = Path.home() / "Downloads/tmp/mlb_pitcher_walks_oof_preds.parquet"
SPINE     = Path.home() / "Downloads/tmp/mlb_pitcher_walks_spine.parquet"
OUT       = Path.home() / "Downloads/tmp/mlb_pitcher_walks_step5_oos.csv"

SHRINKAGE       = [0.0, 0.25, 0.50, 0.75]
EDGE_THRESHOLDS = [0.0, 0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
DIRECTIONS      = ["over", "under", "both"]
ODDS_BUCKETS    = ["all", "plus_odds", "minus_odds"]
MIN_BOOKS       = 2


def kelly_bet(p: float, american_odds: float) -> float:
    """Fractional Kelly stake given American odds and win probability."""
    if american_odds >= 0:
        b = american_odds / 100
    else:
        b = 100 / abs(american_odds)
    return max((b * p - (1 - p)) / b, 0.0)


def main() -> None:
    print("Loading data...")
    oof   = pd.read_parquet(OOF_PREDS)   # player-game level
    spine = pd.read_parquet(SPINE)
    spine = spine[spine["home_run_line_point"].abs() <= 2.0].copy()
    spine_2026 = spine[spine["season"] == 2026].copy()

    # Merge p_model onto per-book spine
    merged = spine_2026.merge(
        oof[["player_key", "game_date", "p_model"]],
        on=["player_key", "game_date"],
        how="inner",
    )
    print(f"OOS rows: {len(merged):,}")

    # Deduplicate to best line per player-game-book
    # (keep only the standard lines: 0.5, 1.5, 2.5, 3.5)
    merged = merged[merged["line"].isin([0.5, 1.5, 2.5, 3.5, 4.5])].copy()

    # Filter books with sufficient coverage
    book_counts = merged.groupby("bookmaker")["event_id"].nunique()
    valid_books = book_counts[book_counts >= 10].index
    merged = merged[merged["bookmaker"].isin(valid_books)].copy()
    print(f"After line/book filter: {len(merged):,}")

    rows = []
    for shrink, edge_thr, direction, odds_bucket in product(
        SHRINKAGE, EDGE_THRESHOLDS, DIRECTIONS, ODDS_BUCKETS
    ):
        df = merged.copy()

        # Apply shrinkage
        df["p_final"] = (1 - shrink) * df["p_model"] + shrink * df["novig_prob_over"]

        # Compute edges
        df["edge_over"]  = df["p_final"] - df["novig_prob_over"]
        df["edge_under"] = (1 - df["p_final"]) - df["novig_prob_under"]

        # Apply odds bucket filter
        if odds_bucket == "plus_odds":
            over_mask  = df["over_price"]  > 0
            under_mask = df["under_price"] > 0
        elif odds_bucket == "minus_odds":
            over_mask  = df["over_price"]  < 0
            under_mask = df["under_price"] < 0
        else:
            over_mask  = pd.Series(True, index=df.index)
            under_mask = pd.Series(True, index=df.index)

        bets = []

        if direction in ("over", "both"):
            over_bets = df[df["edge_over"] >= edge_thr].copy()
            if odds_bucket != "all":
                over_bets = over_bets[over_mask.reindex(over_bets.index)]
            over_bets["bet_dir"] = "over"
            over_bets["edge"]    = over_bets["edge_over"]
            over_bets["hit"]     = over_bets["target_over"].astype(int)
            over_bets["odds"]    = over_bets["over_price"]
            bets.append(over_bets)

        if direction in ("under", "both"):
            under_bets = df[df["edge_under"] >= edge_thr].copy()
            if odds_bucket != "all":
                under_bets = under_bets[under_mask.reindex(under_bets.index)]
            under_bets["bet_dir"] = "under"
            under_bets["edge"]    = under_bets["edge_under"]
            under_bets["hit"]     = (1 - under_bets["target_over"]).astype(int)
            under_bets["odds"]    = under_bets["under_price"]
            bets.append(under_bets)

        if not bets:
            continue
        bets_df = pd.concat(bets, ignore_index=True)
        if len(bets_df) == 0:
            continue

        # Require min_books per player-game (deduplicate to unique player-game-line first)
        pg_books = bets_df.groupby(["player_key", "game_date", "line"])["bookmaker"].nunique()
        valid_pg = pg_books[pg_books >= MIN_BOOKS].reset_index()[["player_key","game_date","line"]]
        bets_df = bets_df.merge(valid_pg, on=["player_key","game_date","line"])

        if len(bets_df) < 20:
            continue

        # PnL: $1 flat bet per qualifying row
        def pnl_from_american(odds, hit):
            if odds >= 0:
                return (odds / 100) * hit - (1 - hit)
            else:
                return (100 / abs(odds)) * hit - (1 - hit)

        bets_df["pnl"] = bets_df.apply(lambda r: pnl_from_american(r["odds"], r["hit"]), axis=1)
        n_bets   = len(bets_df)
        hit_rate = bets_df["hit"].mean()
        roi      = bets_df["pnl"].sum() / n_bets
        total_pu = bets_df["pnl"].sum()

        rows.append({
            "shrinkage":      shrink,
            "edge_threshold": edge_thr,
            "direction":      direction,
            "odds_bucket":    odds_bucket,
            "n_bets":         n_bets,
            "hit_rate":       round(hit_rate, 4),
            "roi":            round(roi, 4),
            "total_pu":       round(total_pu, 2),
        })

    results = pd.DataFrame(rows).sort_values("roi", ascending=False)
    results.to_csv(OUT, index=False)

    print(f"\n{'='*80}")
    print("TOP 30 COMBOS BY OOS ROI (2026)")
    print(f"{'='*80}")
    print(results.head(30).to_string(index=False))

    print(f"\n{'='*40}")
    print("TOP 10 — UNDER ONLY")
    print(results[results["direction"] == "under"].head(10).to_string(index=False))

    print(f"\n{'='*40}")
    print("TOP 10 — OVER ONLY")
    print(results[results["direction"] == "over"].head(10).to_string(index=False))

    print(f"\nFull results saved → {OUT}")


if __name__ == "__main__":
    main()

"""
Step 5 — OOS grid search.

Uses OOF predictions from Step 4 (no IS leakage).
Strategy: UNDER, evaluated at different edge_under thresholds and line filters.

Units: flat 1u per bet. Profit = (under_price - 1) * 1u if win, -1u if loss.
Edge = p_model_under - raw_implied_prob_under (vig-inclusive).

Grid:
  edge_under_min: [0.00, 0.03, 0.05, 0.07, 0.10]
  line_filter:    [all, 0.5_only, 1.5_only]
  min_books:      [1, 2, 3]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

SCORED_PATH = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_step4_scored.parquet"
OUT_PATH    = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_step5_grid.csv"


def compute_roi(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"n_bets": 0, "wins": 0, "losses": 0, "units": 0.0, "roi": 0.0, "win_pct": 0.0}
    under_hit = (df["over_flag"] == 0).astype(int)
    profit = np.where(under_hit == 1, df["under_price"] - 1.0, -1.0)
    return {
        "n_bets":  len(df),
        "wins":    int(under_hit.sum()),
        "losses":  int((under_hit == 0).sum()),
        "units":   round(float(profit.sum()), 3),
        "roi":     round(float(profit.sum()) / len(df) * 100, 3),
        "win_pct": round(float(under_hit.mean()), 4),
    }


def main():
    print("Loading scored parquet...")
    df = pd.read_parquet(SCORED_PATH)

    # OOF-only rows with valid predictions and outcomes
    df_oof = df[df["p_model_under"].notna() & df["over_flag"].notna()].copy()
    print(f"  {len(df_oof):,} OOF rows with predictions + outcomes")
    print(f"  Lines: {sorted(df_oof['offered_line'].unique())}")
    print(f"  Books: {df_oof['bookmaker'].nunique()} unique")
    print(f"  Date range: {df_oof['game_date'].min()} → {df_oof['game_date'].max()}")

    # ── n_books per player-game: how many books posted the line ──────────
    n_books = (df_oof.groupby(["player_key", "game_date", "offered_line"])
                     .size().reset_index(name="n_books_posting"))
    df_oof = df_oof.merge(n_books, on=["player_key", "game_date", "offered_line"], how="left")

    # ── Grid search ───────────────────────────────────────────────────────
    edge_thresholds = [0.00, 0.03, 0.05, 0.07, 0.10, 0.15]
    line_filters = {
        "all":      None,
        "0.5_only": [0.5],
        "1.5_only": [1.5],
    }
    min_books_opts = [1, 2, 3]

    results = []
    for edge_min in edge_thresholds:
        for line_name, lines in line_filters.items():
            for min_b in min_books_opts:
                mask = df_oof["edge_under"] >= edge_min
                if lines:
                    mask &= df_oof["offered_line"].isin(lines)
                mask &= df_oof["n_books_posting"] >= min_b

                m = compute_roi(df_oof[mask])
                results.append({
                    "edge_min": edge_min,
                    "line_filter": line_name,
                    "min_books": min_b,
                    **m,
                })

    grid = pd.DataFrame(results)
    print(f"\n{'Edge':>6} {'Lines':<12} {'MinBks':>6} {'n':>7} {'W':>5} {'L':>5} {'Units':>8} {'ROI%':>8} {'WinPct':>8}")
    print("-" * 75)
    for _, r in grid.iterrows():
        print(f"{r['edge_min']:>6.2f} {r['line_filter']:<12} {r['min_books']:>6} "
              f"{r['n_bets']:>7,} {r['wins']:>5} {r['losses']:>5} "
              f"{r['units']:>8.2f} {r['roi']:>8.3f} {r['win_pct']:>8.4f}")

    grid.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")

    # ── Highlight best configurations ─────────────────────────────────────
    print("\n=== Top 10 by ROI% (n_bets >= 50) ===")
    top = grid[grid["n_bets"] >= 50].sort_values("roi", ascending=False).head(10)
    print(top[["edge_min","line_filter","min_books","n_bets","wins","losses","units","roi","win_pct"]].to_string(index=False))

    # ── Detailed breakdown by line for best single edge threshold ─────────
    best_edge = grid[(grid["n_bets"] >= 50) & (grid["line_filter"] == "all")].sort_values("roi").iloc[-1]["edge_min"]
    print(f"\n=== Line breakdown at edge >= {best_edge:.2f} ===")
    for line_val in sorted(df_oof["offered_line"].unique()):
        mask = (df_oof["edge_under"] >= best_edge) & (df_oof["offered_line"] == line_val)
        m = compute_roi(df_oof[mask])
        print(f"  line={line_val}: n={m['n_bets']:,}  W={m['wins']}  L={m['losses']}  "
              f"units={m['units']:+.2f}  ROI={m['roi']:+.3f}%  winpct={m['win_pct']:.4f}")

    # ── Quarterly breakdown for chosen strategy ────────────────────────────
    print(f"\n=== Quarterly breakdown (edge >= {best_edge:.2f}, all lines, min_books=1) ===")
    quarters = {
        "Q1": ("2024-03-01", "2024-04-30"),
        "Q2": ("2024-05-01", "2024-06-30"),
        "Q3": ("2024-07-01", "2024-08-31"),
        "Q4": ("2024-09-01", "2024-10-31"),
    }
    for q_name, (q_start, q_end) in quarters.items():
        mask = (
            (df_oof["edge_under"] >= best_edge)
            & (df_oof["game_date"] >= q_start)
            & (df_oof["game_date"] <= q_end)
        )
        m = compute_roi(df_oof[mask])
        if m["n_bets"] == 0:
            print(f"  {q_name}: 0 bets")
        else:
            print(f"  {q_name}: n={m['n_bets']:,}  W={m['wins']}  L={m['losses']}  "
                  f"units={m['units']:+.2f}  ROI={m['roi']:+.3f}%")

    # ── Break-even check ──────────────────────────────────────────────────
    print(f"\n=== Break-even win% needed vs actual (edge >= {best_edge:.2f}) ===")
    chosen = df_oof[df_oof["edge_under"] >= best_edge].copy()
    if len(chosen) > 0:
        break_even = chosen["raw_implied_prob_under"].mean()
        actual_win = (chosen["over_flag"] == 0).mean()
        print(f"  Mean break-even (raw_implied_prob_under): {break_even:.4f}")
        print(f"  Actual under hit rate:                    {actual_win:.4f}")
        print(f"  Excess over break-even:                   {actual_win - break_even:+.4f}")


if __name__ == "__main__":
    main()

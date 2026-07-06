"""
MLB Pitcher Strikeouts — Step 4.5: Monotonicity + Book Independence Checks
===========================================================================
Loads step4_edge.parquet (output of v4_edge.py) and runs three checks:

  Check A: Book independence
    For the same (player_key, game_date, line), p_model_over must be identical
    across all books. If this fails, a book-dependent feature has leaked in.

  Check B: Line monotonicity
    For the same (player_key, game_date) with multiple lines, P(over) must
    decrease as line increases. Inversions indicate a modeling or data bug.

  Check C: Intuitive example
    Shows a concrete table for one pitcher across multiple lines on the same
    game date — confirms that as the line rises, p_model_over drops.

Usage:
  python src/mlb_strikeouts_modeling/scripts/v45_checks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = Path.home() / "Downloads/tmp/mlb_strikeouts"


def main():
    edge_path = OUT_DIR / "step4_edge.parquet"
    print(f"Loading {edge_path}...", flush=True)
    df = pd.read_parquet(edge_path)
    print(f"  Rows: {len(df):,}  Players: {df['player_key'].nunique():,}")

    # ── Check A: Book independence ────────────────────────────────────────────
    print("\nCheck A — Book independence:")
    check_a = df.groupby(["player_key", "game_date", "line"])["p_model_over"].nunique()
    n_consistent = (check_a == 1).sum()
    n_inconsistent = (check_a > 1).sum()
    print(f"  {n_consistent:,} player-game-lines have consistent p_model_over across books")
    print(f"  {n_inconsistent:,} player-game-lines have INCONSISTENT p_model_over  <- BUG if > 0")
    assert n_inconsistent == 0, (
        f"Book-dependence found in p_model_over — {n_inconsistent} player-game-lines "
        f"have differing p_model_over values across books. Check model features for "
        f"anything that varies per book."
    )
    print("  [PASS] Book independence confirmed: p_model_over is identical across books for every player-game-line")

    # ── Check B: Line monotonicity ────────────────────────────────────────────
    print("\nCheck B — Line monotonicity (P(over) must decrease as line increases):")

    multi_line = df.groupby(["player_key", "game_date"]).filter(
        lambda g: g["line"].nunique() > 1
    )
    multi_line = multi_line.drop_duplicates(subset=["player_key", "game_date", "line"])
    n_multi = multi_line["player_key"].nunique()
    print(f"  Player-games with multiple lines: {n_multi:,}")

    inversions = []
    for (pk, gd), grp in multi_line.groupby(["player_key", "game_date"]):
        grp_sorted = grp.sort_values("line")
        p_vals = grp_sorted["p_model_over"].values
        lines  = grp_sorted["line"].values
        for i in range(len(p_vals) - 1):
            if p_vals[i + 1] > p_vals[i] + 0.001:  # allow tiny float tolerance
                inversions.append({
                    "player_key":    pk,
                    "game_date":     gd,
                    "line_low":      lines[i],
                    "line_high":     lines[i + 1],
                    "p_over_at_low":  round(float(p_vals[i]), 4),
                    "p_over_at_high": round(float(p_vals[i + 1]), 4),
                })

    print(f"  Inversions found: {len(inversions):,}")
    if inversions:
        inv_df = pd.DataFrame(inversions)
        n_ml   = multi_line["player_key"].nunique()
        print(f"  Inversion rate: {len(inversions) / len(multi_line):.2%} of multi-line rows")
        print("\n  First 10 inversions:")
        print(inv_df.head(10).to_string(index=False))
    else:
        print("  [PASS] All multi-line player-games are monotonically ordered")

    # ── Check C: Intuitive example ────────────────────────────────────────────
    print("\nCheck C — Intuitive example (same pitcher, multiple lines):")
    example_pool = multi_line.groupby(["player_key", "game_date"]).filter(lambda g: len(g) >= 3)
    if len(example_pool) == 0:
        print("  No player-game with >= 3 lines found; skipping example.")
    else:
        ex = (
            example_pool.groupby(["player_key", "game_date"])
            .first()
            .reset_index()
            .iloc[0]
        )
        ex_rows = multi_line[
            (multi_line["player_key"] == ex["player_key"]) &
            (multi_line["game_date"] == ex["game_date"])
        ].sort_values("line")

        print(f"\n  Pitcher: {ex['player_key']}  |  Game date: {ex['game_date']}")
        cols = [
            c for c in [
                "line", "p_model_over", "p_model_under",
                "p_market_over", "edge_over", "edge_under",
            ]
            if c in ex_rows.columns
        ]
        print(ex_rows[cols].to_string(index=False))
        print("  (As line increases, p_model_over decreases and p_model_under increases — expected)")

    print("\nAll checks complete.")


if __name__ == "__main__":
    main()

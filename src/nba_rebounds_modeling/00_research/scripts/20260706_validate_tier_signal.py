"""
Validate whether the LLM-generated recognition_tier has predictive signal
for NBA rebounds over/under outcomes.

Three tables produced:
  Table 1 — tier → raw hit rates (over / under / push count)
  Table 2 — tier × odds_direction (fav/pick/dog) → hit rates
  Table 3 — tier × season → hit rates

Saves each table as a CSV and prints to stdout.

Usage:
    python src/nba_rebounds_modeling/00_research/scripts/20260706_validate_tier_signal.py \
        --props ~/Downloads/tmp/rebounds_props.parquet \
        --tiers ~/Downloads/tmp/rebounds_player_tiers.parquet \
        --out-dir ~/Downloads/tmp/tier_validation/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TIERS_ORDER = ["superstar", "known_starter", "fringe", "unknown"]

# Under odds: fav = implied >52.4% (roughly -110 or better for under)
# Dog = implied <47.6% (roughly +110 or worse for under)
# Pick = in between
FAV_THRESHOLD = -110.0
DOG_THRESHOLD = 110.0


def american_to_implied(american: float) -> float:
    if american < 0:
        return (-american) / (-american + 100.0)
    return 100.0 / (american + 100.0)


def odds_direction(under_odds: float) -> str:
    if under_odds <= FAV_THRESHOLD:
        return "fav"
    if under_odds >= DOG_THRESHOLD:
        return "dog"
    return "pick"


def hit_rate_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Compute over/under hit rates and counts for a grouped DataFrame."""
    agg = (
        df.groupby(group_cols, observed=True)
        .agg(
            n_rows=("hit_under", "count"),
            n_push=("is_push", "sum"),
            n_under_win=("hit_under", "sum"),
            n_over_win=("hit_over", "sum"),
        )
        .reset_index()
    )
    agg["n_bets"] = agg["n_rows"] - agg["n_push"]
    agg["hit_rate_under"] = agg["n_under_win"] / agg["n_bets"]
    agg["hit_rate_over"] = agg["n_over_win"] / agg["n_bets"]
    # baseline injected by caller after computing population rate
    return agg


def add_baseline_diff(agg: pd.DataFrame, baseline: float) -> pd.DataFrame:
    agg["baseline_diff_under_pp"] = (agg["hit_rate_under"] - baseline) * 100
    return agg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--props", default="~/Downloads/tmp/rebounds_props.parquet")
    p.add_argument("--tiers", default="~/Downloads/tmp/rebounds_player_tiers.parquet")
    p.add_argument("--out-dir", default="~/Downloads/tmp/tier_validation/")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    props_path = Path(args.props).expanduser()
    tiers_path = Path(args.tiers).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    props = pd.read_parquet(props_path)
    tiers = pd.read_parquet(tiers_path)

    print(f"props rows:  {len(props):,}")
    print(f"tiers rows:  {len(tiers):,}")

    df = props.merge(tiers, on="player_normalized", how="left")
    n_unmapped = df["recognition_tier"].isna().sum()
    if n_unmapped:
        print(f"[WARN] {n_unmapped} rows have no tier — filling with 'unknown'")
        df["recognition_tier"] = df["recognition_tier"].fillna("unknown")

    df["recognition_tier"] = pd.Categorical(
        df["recognition_tier"], categories=TIERS_ORDER, ordered=True
    )

    # Outcome columns.
    df["is_push"] = df["REB"] == df["line"]
    df["hit_under"] = (df["REB"] < df["line"]) & ~df["is_push"]
    df["hit_over"] = (df["REB"] > df["line"]) & ~df["is_push"]

    # Odds direction on the under side.
    df["odds_direction"] = df["under_odds"].apply(odds_direction)

    total_bets = (~df["is_push"]).sum()
    overall_under = df.loc[~df["is_push"], "hit_under"].mean()
    print(f"\nBaseline (population under hit rate, excl push): n={total_bets:,}  under_hit={overall_under:.4f} ({overall_under*100:.2f}%)")
    print("baseline_diff_under_pp = tier hit rate minus this population rate, in percentage points\n")

    # --- Table 1: tier only ---
    t1 = add_baseline_diff(hit_rate_table(df, ["recognition_tier"]), overall_under)
    t1_path = out_dir / "table1_tier.csv"
    t1.to_csv(t1_path, index=False)
    print("=== Table 1: tier → hit rates ===")
    print(t1.to_string(index=False))

    # --- Table 2: tier × odds_direction ---
    t2 = add_baseline_diff(hit_rate_table(df, ["recognition_tier", "odds_direction"]), overall_under)
    t2_path = out_dir / "table2_tier_x_odds.csv"
    t2.to_csv(t2_path, index=False)
    print("\n=== Table 2: tier × odds_direction → hit rates ===")
    print(t2.to_string(index=False))

    # --- Table 3: tier × season ---
    t3 = add_baseline_diff(hit_rate_table(df, ["recognition_tier", "season"]), overall_under)
    t3_path = out_dir / "table3_tier_x_season.csv"
    t3.to_csv(t3_path, index=False)
    print("\n=== Table 3: tier × season → hit rates ===")
    print(t3.to_string(index=False))

    print(f"\nCSVs written to {out_dir}")

    # Quick signal check: flag any tier with >2pp deviation from population baseline.
    deviations = t1[t1["baseline_diff_under_pp"].abs() > 2.0]
    if len(deviations):
        print(f"\n[SIGNAL] tiers with >2pp deviation from {overall_under*100:.2f}% baseline (UNDER):")
        print(deviations[["recognition_tier", "n_bets", "hit_rate_under", "baseline_diff_under_pp"]].to_string(index=False))
    else:
        print(f"\n[NO SIGNAL] no tier exceeds ±2pp deviation from {overall_under*100:.2f}% baseline.")


if __name__ == "__main__":
    main()

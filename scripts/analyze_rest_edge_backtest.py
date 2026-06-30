"""
Backtest: does preseason net rest edge predict team win total performance?

Joins historical net_rest (2015-2025) with preseason win total lines to test
whether teams with favorable rest schedules outperform their preseason consensus.

Usage:
    python scripts/analyze_rest_edge_backtest.py
    python scripts/analyze_rest_edge_backtest.py --min-season 2019
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

TMP_DIR = Path.home() / "Downloads" / "tmp"


def load_data(min_season: int, max_season: int) -> pd.DataFrame:
    rest = pd.read_csv(TMP_DIR / "net_rest_all.csv")
    wins = pd.read_csv(TMP_DIR / "win_totals_all.csv")

    # wins has 2025 season but the season isn't done from a betting perspective
    # (2025 = Sept 2025 - Feb 2026, so it is complete by now June 2026)
    df = rest.merge(wins[["season", "team", "win_total_line", "actual_wins", "ou_result"]],
                    on=["season", "team"], how="inner")

    df = df[(df["season"] >= min_season) & (df["season"] <= max_season)]
    df = df[df["ou_result"].isin(["Over", "Under"])]  # exclude pushes

    df["wins_vs_line"] = df["actual_wins"] - df["win_total_line"]
    df["is_over"] = (df["ou_result"] == "Over").astype(int)

    return df.reset_index(drop=True)


def bucket_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Group teams by net_rest quintile and show O-U performance."""
    df = df.copy()
    df["rest_bucket"] = pd.qcut(df["net_rest"], q=5,
                                labels=["Q1 (worst)", "Q2", "Q3", "Q4", "Q5 (best)"])
    agg = df.groupby("rest_bucket", observed=True).agg(
        n=("is_over", "count"),
        overs=("is_over", "sum"),
        avg_net_rest=("net_rest", "mean"),
        avg_wins_vs_line=("wins_vs_line", "mean"),
    ).assign(
        over_pct=lambda x: (x["overs"] / x["n"] * 100).round(1),
    )
    return agg


def threshold_analysis(df: pd.DataFrame, thresholds: list[int]) -> pd.DataFrame:
    """For each net_rest threshold, show O-U record of teams above/below."""
    rows = []
    for thresh in thresholds:
        above = df[df["net_rest"] >= thresh]
        below = df[df["net_rest"] < thresh]
        if len(above) < 5 or len(below) < 5:
            continue
        rows.append({
            "threshold": f">= {thresh:+d}",
            "n_teams": len(above),
            "overs": above["is_over"].sum(),
            "over_pct": round(above["is_over"].mean() * 100, 1),
            "avg_wins_vs_line": round(above["wins_vs_line"].mean(), 2),
        })
        rows.append({
            "threshold": f"< {thresh:+d}",
            "n_teams": len(below),
            "overs": below["is_over"].sum(),
            "over_pct": round(below["is_over"].mean() * 100, 1),
            "avg_wins_vs_line": round(below["wins_vs_line"].mean(), 2),
        })
    return pd.DataFrame(rows)


def yearly_best_worst(df: pd.DataFrame) -> pd.DataFrame:
    """For each season: did the team with the best net rest go over?"""
    rows = []
    for season, grp in df.groupby("season"):
        grp = grp.sort_values("net_rest", ascending=False)
        best = grp.iloc[0]
        worst = grp.iloc[-1]
        rows.append({
            "season": season,
            "best_team": best["team"],
            "best_net_rest": int(best["net_rest"]),
            "best_actual_wins": best["actual_wins"],
            "best_line": best["win_total_line"],
            "best_wins_vs_line": round(best["wins_vs_line"], 1),
            "best_ou": best["ou_result"],
            "worst_team": worst["team"],
            "worst_net_rest": int(worst["net_rest"]),
            "worst_actual_wins": worst["actual_wins"],
            "worst_line": worst["win_total_line"],
            "worst_wins_vs_line": round(worst["wins_vs_line"], 1),
            "worst_ou": worst["ou_result"],
        })
    return pd.DataFrame(rows)


def correlation_analysis(df: pd.DataFrame) -> None:
    from scipy import stats  # optional; fall back to numpy if missing

    corr_wins = df["net_rest"].corr(df["actual_wins"])
    corr_margin = df["net_rest"].corr(df["wins_vs_line"])
    corr_is_over = df["net_rest"].corr(df["is_over"])

    try:
        r_wins, p_wins = stats.pearsonr(df["net_rest"], df["actual_wins"])
        r_margin, p_margin = stats.pearsonr(df["net_rest"], df["wins_vs_line"])
        r_over, p_over = stats.pearsonr(df["net_rest"], df["is_over"])
        print(f"  net_rest vs actual_wins:    r={r_wins:+.3f}  p={p_wins:.3f}")
        print(f"  net_rest vs wins_vs_line:   r={r_margin:+.3f}  p={p_margin:.3f}")
        print(f"  net_rest vs is_over (0/1):  r={r_over:+.3f}  p={p_over:.3f}")
    except ImportError:
        print(f"  net_rest vs actual_wins:    r={corr_wins:+.3f}")
        print(f"  net_rest vs wins_vs_line:   r={corr_margin:+.3f}")
        print(f"  net_rest vs is_over (0/1):  r={corr_is_over:+.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-season", type=int, default=2015)
    parser.add_argument("--max-season", type=int, default=2025)
    args = parser.parse_args()

    df = load_data(args.min_season, args.max_season)
    seasons = sorted(df["season"].unique())
    print(f"Dataset: {len(df)} team-seasons, {len(seasons)} seasons ({seasons[0]}–{seasons[-1]})")
    print(f"  Pushes excluded: {df['ou_result'].value_counts().to_dict()}")

    # 1. Overall O-U rate
    total_over_pct = df["is_over"].mean() * 100
    avg_wins_vs_line = df["wins_vs_line"].mean()
    print(f"\nOverall: {df['is_over'].sum()}/{len(df)} Over = {total_over_pct:.1f}%")
    print(f"  Avg wins vs line: {avg_wins_vs_line:+.2f}")

    # 2. Correlation
    print("\nCorrelations (Pearson r):")
    correlation_analysis(df)

    # 3. Quintile buckets
    print("\nQuintile Analysis (net_rest):")
    buckets = bucket_analysis(df)
    print(buckets[["n", "avg_net_rest", "over_pct", "avg_wins_vs_line"]].to_string())

    # 4. Threshold analysis
    print("\nThreshold Analysis:")
    thresholds = threshold_analysis(df, thresholds=[10, 8, 5, 0, -5, -8, -10])
    print(thresholds.to_string(index=False))

    # 5. Yearly best vs worst
    print("\nYearly Best vs Worst Net Rest:")
    bw = yearly_best_worst(df)
    print(bw[["season", "best_team", "best_net_rest", "best_wins_vs_line", "best_ou",
               "worst_team", "worst_net_rest", "worst_wins_vs_line", "worst_ou"]].to_string(index=False))

    # 6. Extreme rest teams (|net_rest| >= 10)
    extreme = df[df["net_rest"].abs() >= 10].copy()
    print(f"\nExtreme rest teams (|net_rest| >= 10): {len(extreme)} team-seasons")
    ext_over = extreme.groupby(extreme["net_rest"] > 0)["is_over"].agg(["count","mean"])
    ext_over.index = ["rest_disadvantaged (neg)", "rest_advantaged (pos)"]
    ext_over["over_pct"] = (ext_over["mean"] * 100).round(1)
    print(ext_over[["count","over_pct"]].to_string())


if __name__ == "__main__":
    main()

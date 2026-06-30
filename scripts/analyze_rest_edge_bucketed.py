"""
Steps 2 & 3: Rest edge vs win total O-U analysis (2010-2025).

Step 2: Over% for each individual net_rest value (histogram view)
Step 3: Bucketed table in 5-unit bands with sample size flags
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

TMP_DIR = Path.home() / "Downloads" / "tmp"
MIN_N = 10  # flag buckets below this


def load_joined() -> pd.DataFrame:
    rest = pd.read_csv(TMP_DIR / "net_rest_all.csv")
    wins = pd.read_csv(TMP_DIR / "win_totals_all.csv")
    df = rest.merge(
        wins[["season", "team", "win_total_line", "actual_wins", "ou_result"]],
        on=["season", "team"], how="inner"
    )
    df = df[df["ou_result"].isin(["Over", "Under"])].copy()
    df["wins_vs_line"] = df["actual_wins"] - df["win_total_line"]
    df["is_over"] = (df["ou_result"] == "Over").astype(int)
    return df.reset_index(drop=True)


def step2_individual(df: pd.DataFrame) -> None:
    print("=" * 72)
    print("STEP 2 — Over% by Individual net_rest Value (2010–2025)")
    print("=" * 72)
    print(f"{'net_rest':>9}  {'n':>4}  {'Over':>5}  {'Under':>6}  {'Over%':>6}  {'Avg W vs Line':>14}  bar")
    print("-" * 72)

    grp = (
        df.groupby("net_rest")
        .agg(n=("is_over", "count"), overs=("is_over", "sum"),
             avg_wvl=("wins_vs_line", "mean"))
        .reset_index()
        .sort_values("net_rest")
    )
    grp["unders"] = grp["n"] - grp["overs"]
    grp["over_pct"] = grp["overs"] / grp["n"] * 100

    for _, row in grp.iterrows():
        flag = "  *" if row["n"] < MIN_N else ""
        bar_len = int(row["over_pct"] / 2)  # 50% = 25 chars
        bar = "█" * bar_len
        marker = "│" if abs(row["over_pct"] - 50) < 1 else ("▶" if row["over_pct"] > 50 else "◀")
        print(
            f"{int(row['net_rest']):>9}  {int(row['n']):>4}  "
            f"{int(row['overs']):>5}  {int(row['unders']):>6}  "
            f"{row['over_pct']:>5.1f}%  {row['avg_wvl']:>+14.2f}  "
            f"{bar:<25} {marker}{flag}"
        )

    print(f"\n* = n < {MIN_N}, treat as noise")


def step3_bucketed(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("STEP 3 — Over% by 5-Unit net_rest Bucket (2010–2025)")
    print("=" * 72)

    bins   = [-999, -21, -16, -11, -6, -1, 4, 9, 14, 19, 999]
    labels = ["≤-21", "-20→-16", "-15→-11", "-10→-6", "-5→-1",
              "0→+4", "+5→+9", "+10→+14", "+15→+19", "≥+20"]

    df = df.copy()
    df["bucket"] = pd.cut(df["net_rest"], bins=bins, labels=labels, right=True)

    agg = (
        df.groupby("bucket", observed=True)
        .agg(
            n=("is_over", "count"),
            overs=("is_over", "sum"),
            avg_net_rest=("net_rest", "mean"),
            avg_wvl=("wins_vs_line", "mean"),
        )
        .reset_index()
    )
    agg["unders"] = agg["n"] - agg["overs"]
    agg["over_pct"] = agg["overs"] / agg["n"] * 100

    print(f"{'Bucket':>10}  {'n':>4}  {'Over':>5}  {'Under':>6}  {'Over%':>6}  "
          f"{'Avg Rest':>9}  {'Avg W vs Line':>14}  {'Flag':>6}")
    print("-" * 72)

    for _, row in agg.iterrows():
        flag = "SMALL" if row["n"] < MIN_N else ""
        signal = ""
        if row["n"] >= MIN_N:
            if row["over_pct"] >= 55:
                signal = " ◀ OVER lean"
            elif row["over_pct"] <= 45:
                signal = " ◀ UNDER lean"
        print(
            f"{str(row['bucket']):>10}  {int(row['n']):>4}  "
            f"{int(row['overs']):>5}  {int(row['unders']):>6}  "
            f"{row['over_pct']:>5.1f}%  "
            f"{row['avg_net_rest']:>+9.1f}  {row['avg_wvl']:>+14.2f}  "
            f"{flag:<6}{signal}"
        )

    print(f"\nFLAG = n < {MIN_N}")

    # Summary: overall dataset stats for context
    total = len(df)
    base_over_pct = df["is_over"].mean() * 100
    print(f"\nBaseline: {df['is_over'].sum()}/{total} Over = {base_over_pct:.1f}% across all {total} team-seasons")


def main():
    df = load_joined()
    print(f"Dataset: {len(df)} team-seasons, seasons {df['season'].min()}–{df['season'].max()}\n")
    step2_individual(df)
    step3_bucketed(df)


if __name__ == "__main__":
    main()

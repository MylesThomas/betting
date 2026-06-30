"""
Build NFL rest edge analysis for the 2026 season.

Replicates Warren Sharp's 2026 NFL Preview rest edge analysis (pp.25-40).
Validates computed values against Sharp's published numbers.

Usage:
    python scripts/build_rest_edge_2026.py
    python scripts/build_rest_edge_2026.py --no-cache   # force re-fetch
    python scripts/build_rest_edge_2026.py --skip-validate
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import boto3
import pandas as pd

S3_BUCKET = "the-odds-api-mt"
S3_PREFIX = "nfl/rest_edge"
TMP_DIR = Path.home() / "Downloads" / "tmp"

from nfl_rest_edge.fetch_schedule import fetch_season_schedule
from nfl_rest_edge.compute_rest import compute_rest_metrics, compute_team_summary, build_weekly_rest_edge_table
from nfl_rest_edge.validate import validate_2026


def fmt_rest(v) -> str:
    if pd.isna(v):
        return "  "
    v = int(v)
    if v == 0:
        return " 0"
    return f"{v:+d}"


def print_net_rest_table(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 50)
    print("2026 NFL NET REST RANKINGS")
    print(f"Source: Warren Sharp 2026 Football Preview")
    print("=" * 50)
    print(f"{'Team':<6} {'Net Rest':>8}  {'Adv':>4}  {'Disadv':>6}")
    print("-" * 30)
    for _, row in summary.iterrows():
        net = int(row["net_rest"])
        sign = "+" if net > 0 else ""
        print(
            f"{row['team']:<6} {sign}{net:>7}  {int(row['rest_adv_games']):>4}  {int(row['rest_disadv_games']):>6}"
        )
    best = summary.iloc[0]
    worst = summary.iloc[-1]
    swing = int(best["net_rest"]) - int(worst["net_rest"])
    print(f"\nSwing: {swing} days  ({best['team']} +{int(best['net_rest'])} to {worst['team']} {int(worst['net_rest'])})")


def print_situation_flags(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("2026 SITUATIONAL FLAGS")
    print("=" * 80)
    print(
        f"{'Team':<6} {'SWRoad':>6}  {'PostPrime':>9}  {'OppPrep':>7}  {'NegBye':>6}  {'3in10':>5}  {'4in17':>5}"
    )
    print("-" * 60)
    for _, row in summary.iterrows():
        print(
            f"{row['team']:<6}"
            f" {int(row['short_week_road']):>6}"
            f"  {int(row['post_road_prime']):>9}"
            f"  {int(row['opp_extra_prep']):>7}"
            f"  {int(row['negated_bye']):>6}"
            f"  {'Y' if row['in_3_in_10'] else '-':>5}"
            f"  {'Y' if row['in_4_in_17'] else '-':>5}"
        )

    print("\nLegend:")
    print("  SWRoad   = short-week road games (days_rest < 6, away)")
    print("  PostPrime = games after a road SNF or MNF")
    print("  OppPrep  = games where opponent had >6 days to prepare")
    print("  NegBye   = negated bye weeks (off bye but opponent also on extra rest)")
    print("  3in10    = any 3-game stretch within 10 calendar days")
    print("  4in17    = any 4-game stretch within 17 calendar days")


def print_weekly_table(weekly: pd.DataFrame) -> None:
    print("\n" + "=" * 120)
    print("2026 NFL WEEKLY REST EDGE (+ = team has more rest, - = less rest than opponent)")
    print("=" * 120)
    # Print header
    col_header = f"{'Team':<6} {'Net':>4}  " + "  ".join(f"{c:>4}" for c in weekly.columns if c != "Net")
    print(col_header)
    print("-" * len(col_header))
    for team, row in weekly.iterrows():
        net = int(row["Net"])
        sign = "+" if net > 0 else ""
        week_vals = "  ".join(
            fmt_rest(row[c]) for c in weekly.columns if c != "Net"
        )
        print(f"{team:<6} {sign}{net:>3}  {week_vals}")


def print_bills_spotlight(team_games: pd.DataFrame) -> None:
    buf = team_games[team_games["team"] == "BUF"].sort_values("game_date")
    print("\n" + "=" * 60)
    print("BUFFALO BILLS — SCHEDULE SPOTLIGHT")
    print("=" * 60)
    print(f"{'Wk':>3}  {'Date':<12} {'vs':^3} {'Opp':<6} {'Type':<6} {'Rest':>5} {'Edge':>5} {'Flag'}")
    print("-" * 60)
    for _, row in buf.iterrows():
        ha = "vs" if row["is_home"] else "@"
        flags = []
        if row["short_week_road"]:
            flags.append("SHORT_ROAD")
        if row["in_4_in_17"]:
            flags.append("4in17")
        if row["post_road_prime"]:
            flags.append("post-prime")
        flag_str = ", ".join(flags)
        print(
            f"{int(row['week']):>3}  {str(row['game_date']):<12} {ha:^3} {row['opponent']:<6} "
            f"{row['game_type']:<6} {int(row['days_rest']):>5} {int(row['rest_edge']):>+5}  {flag_str}"
        )


def _save_outputs(
    team_games: pd.DataFrame,
    summary: pd.DataFrame,
    weekly: pd.DataFrame,
    season: int,
) -> None:
    s3 = boto3.client("s3")
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        f"team_games_{season}.csv": (team_games, False),   # (df, include_index)
        f"summary_{season}.csv":    (summary,    False),
        f"weekly_{season}.csv":     (weekly,     True),
    }

    print("\nSaving outputs...")
    for filename, (df, idx) in outputs.items():
        csv_bytes = df.to_csv(index=idx).encode()
        s3_key = f"{S3_PREFIX}/{season}/{filename}"

        # S3
        s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=csv_bytes)
        print(f"  s3://{S3_BUCKET}/{s3_key}")

        # ~/Downloads/tmp
        local_path = TMP_DIR / filename
        local_path.write_bytes(csv_bytes)
        print(f"  {local_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    print("Fetching 2026 NFL schedule...")
    schedule = fetch_season_schedule(season=2026, use_cache=not args.no_cache)
    print(f"  {len(schedule)} games loaded")

    print("\nComputing rest metrics...")
    team_games = compute_rest_metrics(schedule)
    summary = compute_team_summary(team_games)
    weekly = build_weekly_rest_edge_table(team_games)

    print_net_rest_table(summary)
    print_situation_flags(summary)
    print_weekly_table(weekly)
    print_bills_spotlight(team_games)

    if not args.skip_validate:
        print("\nRunning validation against Warren Sharp's published values...")
        passed = validate_2026(team_games, summary)
        if not passed:
            print("\nValidation FAILED — check output above for mismatches.")
            sys.exit(1)
        else:
            print("\nAll checks passed. Pipeline matches Sharp's published 2026 analysis.")

    _save_outputs(team_games, summary, weekly, season=2026)


if __name__ == "__main__":
    main()

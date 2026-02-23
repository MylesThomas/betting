"""
One-off: find Morgan State @ South Carolina State in outcomes and lines (any date).
Run from repo root: python tmp/find_morgan_state_south_carolina_state_game.py
"""
import sys
from pathlib import Path

# project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from io import BytesIO
import boto3
import pandas as pd

BUCKET = "ncaab-betting-mt"
OUTCOMES_PREFIX = "data/01_input/historical_game_results/"
LINES_PREFIX = "data/01_input/the-odds-api/ncaab/game_lines/"

# Search a range of dates (postponed from 2026-02-14, might be on another day)
DATES = [f"2026-02-{d:02d}" for d in range(14, 22)]


def main():
    s3 = boto3.client("s3")

    print("=== OUTCOMES (Morgan State / South Carolina State) ===\n")
    for date in DATES:
        key = f"{OUTCOMES_PREFIX}{date}.csv"
        try:
            r = s3.get_object(Bucket=BUCKET, Key=key)
            df = pd.read_csv(BytesIO(r["Body"].read()))
        except Exception as e:
            print(f"  {date}: skip ({e})")
            continue
        home = df.get("HOME_TEAM", pd.Series(dtype=object)).astype(str)
        away = df.get("AWAY_TEAM", pd.Series(dtype=object)).astype(str)
        mask = (
            home.str.contains("Morgan State", case=False, na=False)
            | away.str.contains("Morgan State", case=False, na=False)
            | home.str.contains("South Carolina State", case=False, na=False)
            | away.str.contains("South Carolina State", case=False, na=False)
        )
        if mask.any():
            sub = df.loc[mask]
            for _, row in sub.iterrows():
                print(f"  {date}: HOME={row.get('HOME_TEAM')}  AWAY={row.get('AWAY_TEAM')}")
        else:
            print(f"  {date}: (none)")

    print("\n=== LINES (Morgan State / South Carolina State) ===\n")
    for date in DATES:
        key = f"{LINES_PREFIX}{date}.csv"
        try:
            r = s3.get_object(Bucket=BUCKET, Key=key)
            df = pd.read_csv(BytesIO(r["Body"].read()))
        except Exception as e:
            print(f"  {date}: skip ({e})")
            continue
        home = df.get("home_team", pd.Series(dtype=object)).astype(str)
        away = df.get("away_team", pd.Series(dtype=object)).astype(str)
        mask = (
            home.str.contains("Morgan", case=False, na=False)
            | away.str.contains("Morgan", case=False, na=False)
            | home.str.contains("South Carolina State", case=False, na=False)
            | away.str.contains("South Carolina State", case=False, na=False)
        )
        if mask.any():
            sub = df.loc[mask]
            for _, row in sub.iterrows():
                print(f"  {date}: home={row.get('home_team')}  away={row.get('away_team')}  spread={row.get('consensus_spread')}")
        else:
            print(f"  {date}: (none)")


if __name__ == "__main__":
    main()

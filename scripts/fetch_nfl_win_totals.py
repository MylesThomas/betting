"""
Fetch historical NFL preseason win total lines from sportsoddshistory.com.

Saves one CSV per season to S3 and ~/Downloads/tmp.

Usage:
    python scripts/fetch_nfl_win_totals.py                  # 2015-2025
    python scripts/fetch_nfl_win_totals.py --seasons 2020 2021 2022
    python scripts/fetch_nfl_win_totals.py --seasons 2024 --no-cache
"""

import argparse
import sys
import time
from io import StringIO
from pathlib import Path

import boto3
import pandas as pd
import requests
import urllib3
from botocore.exceptions import ClientError
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from nfl_team_utils import full_name_to_abbr

urllib3.disable_warnings()

S3_BUCKET = "the-odds-api-mt"
S3_PREFIX = "nfl/win_totals"
TMP_DIR = Path.home() / "Downloads" / "tmp"
BASE_URL = "https://www.sportsoddshistory.com/nfl-win/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# Aliases not in nfl_team_utils (name variants found on this site)
EXTRA_ALIASES = {
    "Washington Football Team": "WAS",
    "Washington Redskins": "WAS",
    "Oakland Raiders": "LV",
    "San Diego Chargers": "LAC",
    "St. Louis Rams": "LAR",
    "St Louis Rams": "LAR",
}


def _abbr(name: str) -> str:
    name = name.strip()
    abbr = full_name_to_abbr(name) or EXTRA_ALIASES.get(name)
    if abbr is None:
        raise ValueError(f"Unknown team name: {name!r}")
    return abbr


def _s3_key(season: int) -> str:
    return f"{S3_PREFIX}/{season}/win_totals_{season}.csv"


def _read_from_s3(season: int) -> pd.DataFrame | None:
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=_s3_key(season))
        return pd.read_csv(StringIO(obj["Body"].read().decode()))
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def _write_to_s3(df: pd.DataFrame, season: int) -> str:
    s3 = boto3.client("s3")
    key = _s3_key(season)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=df.to_csv(index=False).encode())
    return f"s3://{S3_BUCKET}/{key}"


def fetch_season(season: int) -> pd.DataFrame:
    """Scrape win total lines for one NFL season."""
    url = f"{BASE_URL}?y={season}&sa=nfl&t=win&o=t"
    session = requests.Session()
    session.max_redirects = 3
    for attempt in range(3):
        try:
            resp = session.get(url, headers=HEADERS, verify=False, timeout=30, allow_redirects=True)
            if "sportsoddshistory" not in resp.url:
                raise ValueError(f"Redirected away from sportsoddshistory to {resp.url}")
            resp.raise_for_status()
            break
        except (requests.Timeout, requests.ConnectionError, ValueError) as e:
            if attempt == 2:
                raise
            print(f"\n    retry {attempt+1}/3 after error: {e}", end=" ", flush=True)
            time.sleep(3 + attempt * 2)

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 7 or not tr.find("a"):
            continue
        team_name = tds[0].text.strip()
        try:
            abbr = _abbr(team_name)
        except ValueError as e:
            print(f"  WARNING: {e} — skipping row")
            continue

        line_raw = tds[1].text.strip()
        try:
            line = float(line_raw)
        except ValueError:
            continue

        over_odds = tds[2].text.strip()
        under_odds = tds[3].text.strip()
        settled_week = tds[4].text.strip()
        actual_wins_raw = tds[5].text.strip()
        ou_result = tds[6].text.strip()  # "Over", "Under", or "Push"

        try:
            actual_wins = int(actual_wins_raw)
        except ValueError:
            actual_wins = None

        rows.append({
            "season": season,
            "team": abbr,
            "team_name": team_name,
            "win_total_line": line,
            "over_odds": over_odds,
            "under_odds": under_odds,
            "settled_week": settled_week,
            "actual_wins": actual_wins,
            "ou_result": ou_result,
        })

    return pd.DataFrame(rows)


def fetch_and_save(season: int, use_cache: bool = True) -> pd.DataFrame:
    if use_cache:
        cached = _read_from_s3(season)
        if cached is not None:
            print(f"  {season}: loaded from S3 ({len(cached)} teams)")
            return cached

    print(f"  {season}: fetching from sportsoddshistory.com...", end=" ", flush=True)
    df = fetch_season(season)
    print(f"{len(df)} teams")

    uri = _write_to_s3(df, season)
    print(f"    -> {uri}")

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    local = TMP_DIR / f"win_totals_{season}.csv"
    local.write_bytes(df.to_csv(index=False).encode())
    print(f"    -> {local}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=list(range(2015, 2026)))
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    print(f"Fetching NFL preseason win totals: seasons {args.seasons[0]}–{args.seasons[-1]}")
    all_dfs = []
    for season in args.seasons:
        df = fetch_and_save(season, use_cache=not args.no_cache)
        all_dfs.append(df)
        time.sleep(1.5)

    combined = pd.concat(all_dfs, ignore_index=True)

    # Save combined
    combined_key = f"{S3_PREFIX}/win_totals_all.csv"
    s3 = boto3.client("s3")
    s3.put_object(Bucket=S3_BUCKET, Key=combined_key, Body=combined.to_csv(index=False).encode())
    local_all = TMP_DIR / "win_totals_all.csv"
    local_all.write_bytes(combined.to_csv(index=False).encode())

    print(f"\nCombined: {len(combined)} rows ({combined['season'].nunique()} seasons)")
    print(f"  s3://{S3_BUCKET}/{combined_key}")
    print(f"  {local_all}")

    # Quick sanity check
    print("\nSample (2024):")
    sample = combined[combined["season"] == 2024].sort_values("win_total_line", ascending=False)
    print(sample[["team", "win_total_line", "actual_wins", "ou_result"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

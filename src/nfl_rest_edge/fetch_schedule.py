"""
Fetch NFL schedule from ESPN CDN API.

Usage:
    from nfl_rest_edge.fetch_schedule import fetch_season_schedule
    df = fetch_season_schedule(season=2026)
"""

import time
from datetime import datetime, date
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import pandas as pd
import requests
from botocore.exceptions import ClientError

ET = ZoneInfo("America/New_York")

# ESPN uses WSH; our canonical code is WAS
ESPN_ABBR_MAP = {
    "WSH": "WAS",
}

S3_BUCKET = "the-odds-api-mt"
S3_SCHEDULE_KEY = "nfl/rest_edge/{season}/schedule_{season}.csv"


def _utc_str_to_et_date(utc_str: str) -> date:
    dt_utc = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
    return dt_utc.astimezone(ET).date()


def _classify_game_type(et_date: date, broadcast: str) -> str:
    dow = et_date.weekday()  # Mon=0 ... Sun=6
    if dow == 3:
        return "TNF"
    if dow == 0:
        return "MNF"
    if dow == 2:
        return "WED"
    if dow == 5:
        return "SAT"
    if dow == 4:
        return "FRI"
    if dow == 6:
        return "SNF" if "NBC" in broadcast else "REG"
    return "OTHER"


def _normalize_abbr(abbr: str) -> str:
    return ESPN_ABBR_MAP.get(abbr, abbr)


def fetch_week(season: int, week: int) -> list[dict]:
    url = "https://cdn.espn.com/core/nfl/schedule"
    params = {"xhr": "1", "season": season, "week": week}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    schedule = data.get("content", {}).get("schedule", {})
    games = []
    for day in schedule.values():
        for g in day.get("games", []):
            comp = g.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            home = next((c for c in competitors if c["homeAway"] == "home"), {})
            away = next((c for c in competitors if c["homeAway"] == "away"), {})

            broadcasts = comp.get("broadcasts", [])
            broadcast = ",".join(broadcasts[0]["names"]) if broadcasts else ""

            et_date = _utc_str_to_et_date(g["date"])

            games.append(
                {
                    "week": week,
                    "game_date": et_date,
                    "game_type": _classify_game_type(et_date, broadcast),
                    "home_team": _normalize_abbr(
                        home.get("team", {}).get("abbreviation", "")
                    ),
                    "away_team": _normalize_abbr(
                        away.get("team", {}).get("abbreviation", "")
                    ),
                    "broadcast": broadcast,
                }
            )
    return games


def _s3_key(season: int) -> str:
    return S3_SCHEDULE_KEY.format(season=season)


def _read_schedule_from_s3(season: int) -> pd.DataFrame | None:
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=_s3_key(season))
        df = pd.read_csv(StringIO(obj["Body"].read().decode()), parse_dates=["game_date"])
        df["game_date"] = df["game_date"].dt.date
        return df
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return None
        raise


def _write_schedule_to_s3(df: pd.DataFrame, season: int) -> str:
    s3 = boto3.client("s3")
    key = _s3_key(season)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=df.to_csv(index=False).encode())
    return f"s3://{S3_BUCKET}/{key}"


# nfl_data_py uses different abbreviations for relocated teams.
# Map to our canonical codes (matching sportsoddshistory win-total abbrs).
_NFL_DATA_ABBR_MAP = {
    "LA": "LAR",   # Los Angeles Rams (2016+)
    "STL": "LAR",  # St. Louis Rams (≤2015)
    "OAK": "LV",   # Oakland Raiders (≤2019)
    "SD": "LAC",   # San Diego Chargers (≤2016)
}


def fetch_historical_schedule(season: int, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch regular-season schedule for a historical NFL season (≤2025) via nfl_data_py.

    Returns a DataFrame with one row per game (columns match fetch_season_schedule):
        week, game_date, game_type, home_team, away_team, broadcast
    """
    if use_cache:
        cached = _read_schedule_from_s3(season)
        if cached is not None:
            print(f"Loaded schedule from s3://{S3_BUCKET}/{_s3_key(season)}")
            return cached

    import nfl_data_py as nfl  # lazy import

    raw = nfl.import_schedules([season])
    reg = raw[raw["game_type"] == "REG"].copy()

    reg["game_date"] = pd.to_datetime(reg["gameday"]).dt.date
    reg["home_team"] = reg["home_team"].map(lambda t: _NFL_DATA_ABBR_MAP.get(t, t))
    reg["away_team"] = reg["away_team"].map(lambda t: _NFL_DATA_ABBR_MAP.get(t, t))
    reg["broadcast"] = ""  # broadcast not available in nfl_data_py

    # Classify game type by weekday (same logic as _classify_game_type)
    def _classify(row) -> str:
        dow = row["game_date"].weekday()
        if dow == 3:
            return "TNF"
        if dow == 0:
            return "MNF"
        if dow == 2:
            return "WED"
        if dow == 5:
            return "SAT"
        if dow == 4:
            return "FRI"
        return "REG"  # Sunday — no broadcast info so can't distinguish SNF

    reg["game_type"] = reg.apply(_classify, axis=1)

    cols = ["week", "game_date", "game_type", "home_team", "away_team", "broadcast"]
    df = reg[cols].sort_values(["week", "game_date", "home_team"]).reset_index(drop=True)

    uri = _write_schedule_to_s3(df, season)
    print(f"Fetched {len(df)} games for {season}, saved to {uri}")
    return df


def fetch_season_schedule(
    season: int = 2026,
    num_weeks: int = 18,
    use_cache: bool = True,
    delay: float = 0.3,
) -> pd.DataFrame:
    """
    Fetch all regular-season games for a given NFL season.

    Checks S3 cache first. Falls back to ESPN CDN API and writes result to S3.

    Returns a DataFrame with one row per game (272 rows for a 17-game season):
        week, game_date, game_type, home_team, away_team, broadcast
    """
    if use_cache:
        cached = _read_schedule_from_s3(season)
        if cached is not None:
            print(f"Loaded schedule from s3://{S3_BUCKET}/{_s3_key(season)}")
            return cached

    all_games = []
    for week in range(1, num_weeks + 1):
        games = fetch_week(season, week)
        all_games.extend(games)
        print(f"  Week {week:2d}: {len(games)} games")
        time.sleep(delay)

    df = pd.DataFrame(all_games)
    df = df.sort_values(["week", "game_date", "home_team"]).reset_index(drop=True)

    uri = _write_schedule_to_s3(df, season)
    print(f"Saved schedule to {uri}")

    return df

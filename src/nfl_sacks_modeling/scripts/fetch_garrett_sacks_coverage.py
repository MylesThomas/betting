"""
Phase 0: Myles Garrett sacks coverage POC.

Fetches player_sacks lines for all 17 CLE REG 2025 games using all available
US bookmakers (regions=us,us2). Joins game metadata onto every prop row.

Output:
  ~/Downloads/tmp/garrett_sacks_2025.parquet  — queryable
  ~/Downloads/tmp/garrett_sacks_2025.csv      — human-readable

Run:
  python nfl_sacks_modeling/scripts/fetch_garrett_sacks_coverage.py
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "americanfootball_nfl"
REGIONS       = "us,us2"
MARKET        = "player_sacks"
SLEEP_S       = 0.15

ET  = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

OUT_DIR     = Path.home() / "Downloads" / "tmp"
OUT_PARQUET = OUT_DIR / "garrett_sacks_2025.parquet"
OUT_CSV     = OUT_DIR / "garrett_sacks_2025.csv"


def snapshot_utc(gameday: str, gametime_et: str, offset_min: int = -30) -> str:
    dt_et = datetime.strptime(f"{gameday} {gametime_et}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    dt_utc = dt_et.astimezone(UTC) + timedelta(minutes=offset_min)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_sacks(event_id: str, snapshot: str) -> tuple[list[dict], int]:
    resp = requests.get(
        f"{ODDS_API_BASE}/historical/sports/{SPORT}/events/{event_id}/odds",
        params={
            "apiKey":     ODDS_API_KEY,
            "markets":    MARKET,
            "regions":    REGIONS,
            "oddsFormat": "american",
            "dateFormat": "iso",
            "date":       snapshot,
        },
        timeout=60,
    )
    time.sleep(SLEEP_S)
    remaining = int(resp.headers.get("x-requests-remaining", 999_999))

    if resp.status_code in (404, 422):
        return [], remaining
    resp.raise_for_status()

    data = resp.json().get("data", {})
    rows = []
    if data:
        for book in data.get("bookmakers", []):
            for mkt in book.get("markets", []):
                for outcome in mkt.get("outcomes", []):
                    rows.append({
                        "market":       mkt["key"],
                        "bookmaker":    book["key"],
                        "last_update":  book.get("last_update", ""),
                        "outcome_name": outcome.get("name", ""),
                        "outcome_desc": outcome.get("description", ""),
                        "price":        outcome.get("price"),
                        "point":        outcome.get("point"),
                        "snapshot":     snapshot,
                    })
    return rows, remaining


def main():
    if not ODDS_API_KEY:
        sys.exit("ODDS_API_KEY not set — add it to .env")

    map_path = REPO_ROOT / "data" / "nfl" / "event_id_map_2025.csv"
    games    = pd.read_csv(map_path)
    cle_reg  = games[
        (games["game_type"] == "REG") &
        (games["nfl_game_id"].str.contains("CLE"))
    ].sort_values(["gameday", "gametime"]).reset_index(drop=True)

    assert len(cle_reg) == 17, f"Expected 17 CLE REG games, got {len(cle_reg)}"
    print(f"CLE REG games: {len(cle_reg)}")
    print(f"Markets: {MARKET}  |  Regions: {REGIONS}\n")

    all_rows = []

    for i, row in cle_reg.iterrows():
        game_id  = row["nfl_game_id"]
        event_id = str(row["odds_api_event_id"])
        gameday  = row["gameday"]
        gametime = str(row.get("gametime", "13:00"))
        week     = int(row["week"])
        home     = row["home_team"]
        away     = row["away_team"]
        is_london = int(gametime.split(":")[0]) < 10

        snap = snapshot_utc(gameday, gametime, offset_min=-30)
        rows, remaining = fetch_sacks(event_id, snap)

        # London game: retry at -2h if -30min returned nothing
        if is_london and not rows:
            snap2 = snapshot_utc(gameday, gametime, offset_min=-120)
            rows2, remaining = fetch_sacks(event_id, snap2)
            if rows2:
                print(f"  [{i+1:>2}/17] wk{week:>2}  {game_id}  LONDON retry -2h → {len(rows2)} rows  remaining={remaining}")
                rows = rows2
            else:
                print(f"  [{i+1:>2}/17] wk{week:>2}  {game_id}  LONDON no lines at -30m or -2h  remaining={remaining}")

        # Attach game metadata to every row
        for r in rows:
            r.update({
                "nfl_game_id":  game_id,
                "week":         week,
                "gameday":      gameday,
                "gametime_et":  gametime,
                "home_team":    home,
                "away_team":    away,
                "is_london":    is_london,
            })

        garrett_rows = [r for r in rows if "garrett" in str(r.get("outcome_desc", "")).lower()]
        books        = sorted({r["bookmaker"] for r in rows})
        line_vals    = sorted({r["point"] for r in garrett_rows if r.get("point") is not None})
        has_garrett  = "MG=YES" if garrett_rows else "MG=NO "
        line_str     = str(line_vals[0]) if line_vals else "?"

        print(f"  [{i+1:>2}/17] wk{week:>2}  {game_id:<25}  {has_garrett}  line={line_str:<4}  books={books}  remaining={remaining}")
        all_rows.extend(rows)

    if not all_rows:
        print("\nNo rows returned — check ODDS_API_KEY and bookmaker availability.")
        return

    df = pd.DataFrame(all_rows)

    col_order = [
        "nfl_game_id", "week", "gameday", "gametime_et", "home_team", "away_team",
        "is_london", "market", "bookmaker", "outcome_name", "outcome_desc",
        "point", "price", "last_update", "snapshot",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df = df.sort_values(["week", "bookmaker", "outcome_desc", "outcome_name"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)

    # Summary
    garrett_df    = df[df["outcome_desc"].str.lower().str.contains("garrett", na=False)]
    covered_games = garrett_df["nfl_game_id"].nunique()
    all_books     = sorted(df["bookmaker"].unique())

    print(f"\n{'='*60}")
    print(f"  COVERAGE SUMMARY")
    print(f"{'='*60}")
    print(f"  Garrett games covered : {covered_games}/17")
    print(f"  Total sacks prop rows : {len(df)}")
    print(f"  Books in dataset      : {all_books}")
    print()
    print(f"  {'Wk':<4} {'Game':<27} {'Line':<6} {'Books with MG line'}")
    print(f"  {'-'*65}")
    for _, gdf in garrett_df.groupby("nfl_game_id", sort=False):
        wk   = int(gdf["week"].iloc[0])
        gid  = gdf["nfl_game_id"].iloc[0]
        bks  = sorted(gdf["bookmaker"].unique())
        pts  = sorted(gdf["point"].dropna().unique())
        line = str(pts[0]) if pts else "?"
        print(f"  {wk:<4} {gid:<27} {line:<6} {bks}")

    if covered_games < 17:
        missing = set(cle_reg["nfl_game_id"]) - set(garrett_df["nfl_game_id"])
        print(f"\n  MISSING ({17 - covered_games} games): {sorted(missing)}")

    print(f"\n  → {OUT_PARQUET}")
    print(f"  → {OUT_CSV}")


if __name__ == "__main__":
    main()

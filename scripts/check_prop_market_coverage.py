"""
Check Odds API coverage for 4 NFL prop markets — week 1, seasons 2021-2025.

For each market × season reports:
  - % of games with ≥1 book posting
  - Which bookmakers appeared
  - Avg player lines per game (when posted)
  - Avg books per game (when posted)

Credit cost estimate: ~5 seasons × ~16 games × 4 markets = ~320 calls
At 4 credits/call (player props) ≈ 1,280 credits total.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import nfl_data_py as nfl
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "americanfootball_nfl"
REGIONS       = "us,us2"
SLEEP_S       = 0.25

SEASONS = [2021, 2022, 2023, 2024, 2025]

TARGET_MARKETS = {
    "player_tackles_assists":  "LB/DB Tackles",
    "player_reception_yds":    "WR/TE Rec Yards",
    "player_rush_attempts":    "RB Rush Attempts",
    "player_pass_completions": "QB Completions",
}

ET  = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

TEAM_NAME_MAP = {
    "ARI": "Arizona Cardinals",   "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",   "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",  "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",      "DEN": "Denver Broncos",
    "DET": "Detroit Lions",       "GB":  "Green Bay Packers",
    "HOU": "Houston Texans",      "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars","KC":  "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers","LA":  "Los Angeles Rams",
    "LV":  "Las Vegas Raiders",   "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",   "NE":  "New England Patriots",
    "NO":  "New Orleans Saints",  "NYG": "New York Giants",
    "NYJ": "New York Jets",       "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers", "SF":  "San Francisco 49ers",
    "SEA": "Seattle Seahawks",    "TB":  "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",    "WAS": "Washington Commanders",
}


def snapshot_10am_et(gameday: str) -> str:
    dt_et  = datetime.strptime(f"{gameday} 10:00", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    dt_utc = dt_et.astimezone(UTC)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_event_ids_for_gameday(gameday: str) -> dict[tuple[str, str], str]:
    snapshot = snapshot_10am_et(gameday)
    resp = requests.get(
        f"{ODDS_API_BASE}/historical/sports/{SPORT}/odds",
        params={
            "apiKey":      ODDS_API_KEY,
            "markets":     "h2h",
            "regions":     "us",
            "oddsFormat":  "american",
            "dateFormat":  "iso",
            "date":        snapshot,
        },
        timeout=60,
    )
    time.sleep(SLEEP_S)
    if resp.status_code in (404, 422):
        return {}
    resp.raise_for_status()
    return {(e["home_team"], e["away_team"]): e["id"]
            for e in resp.json().get("data", [])}


def get_week1_games(season: int) -> pd.DataFrame:
    sched = nfl.import_schedules([season])
    week1 = sched[(sched["week"] == 1) & (sched["game_type"] == "REG")].copy()
    week1["home_full"] = week1["home_team"].map(TEAM_NAME_MAP)
    week1["away_full"] = week1["away_team"].map(TEAM_NAME_MAP)

    cache_path = REPO_ROOT / "data" / "nfl" / f"event_id_map_{season}.csv"
    if cache_path.exists():
        cache = pd.read_csv(cache_path)
        cache_w1 = cache[cache["week"] == 1][["nfl_game_id", "odds_api_event_id"]]
        week1 = week1.merge(
            cache_w1.rename(columns={"nfl_game_id": "game_id"}),
            on="game_id", how="left",
        )
        print(f"  event IDs: loaded {cache_w1['odds_api_event_id'].notna().sum()} from cache")
    else:
        all_ids: dict[tuple, str] = {}
        for gameday in sorted(week1["gameday"].unique()):
            print(f"    fetching event IDs for {gameday}...")
            all_ids.update(fetch_event_ids_for_gameday(gameday))
        week1["odds_api_event_id"] = week1.apply(
            lambda r: all_ids.get((r["home_full"], r["away_full"])), axis=1
        )

    return week1[week1["odds_api_event_id"].notna()].reset_index(drop=True)


def fetch_market(event_id: str, market: str, snapshot: str) -> tuple[list[dict], int]:
    resp = requests.get(
        f"{ODDS_API_BASE}/historical/sports/{SPORT}/events/{event_id}/odds",
        params={
            "apiKey":     ODDS_API_KEY,
            "markets":    market,
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

    rows = []
    data = resp.json().get("data", {})
    for book in data.get("bookmakers", []):
        for mkt in book.get("markets", []):
            for outcome in mkt.get("outcomes", []):
                rows.append({
                    "bookmaker": book["key"],
                    "player":    outcome.get("description", ""),
                    "side":      outcome.get("name", ""),
                    "point":     outcome.get("point"),
                    "price":     outcome.get("price"),
                })
    return rows, remaining


def run():
    if not ODDS_API_KEY:
        sys.exit("ODDS_API_KEY not set")

    # results[market][season] = dict of stats
    results: dict[str, dict[int, dict]] = {m: {} for m in TARGET_MARKETS}

    for season in SEASONS:
        print(f"\n{'='*60}")
        print(f"  SEASON {season}")
        print(f"{'='*60}")

        games = get_week1_games(season)
        n_total = len(games)
        print(f"  {n_total} week-1 games found\n")

        gameday_snap = {gd: snapshot_10am_et(gd) for gd in games["gameday"].unique()}

        for market in TARGET_MARKETS:
            stats = {
                "games_total":   n_total,
                "games_covered": 0,
                "books_seen":    set(),
                "lines_list":    [],
                "books_list":    [],
            }

            for _, row in games.iterrows():
                event_id = str(row["odds_api_event_id"])
                snap     = gameday_snap[row["gameday"]]
                rows, remaining = fetch_market(event_id, market, snap)

                if rows:
                    stats["games_covered"] += 1
                    books   = {r["bookmaker"] for r in rows}
                    players = {r["player"] for r in rows if r["side"] == "Over"}
                    stats["books_seen"].update(books)
                    stats["lines_list"].append(len(players))
                    stats["books_list"].append(len(books))
                    status = f"covered  books={len(books)}  lines={len(players)}"
                else:
                    status = "no data"

                print(f"    {row['away_team']} @ {row['home_team']:<4}  "
                      f"{market:<28}  {status:<40}  credits={remaining}")

            results[market][season] = stats

    # ── Report ────────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 90)
    print("  NFL PROP MARKET COVERAGE — WEEK 1, 2021–2025")
    print("=" * 90)

    for market, label in TARGET_MARKETS.items():
        print(f"\n{'─' * 90}")
        print(f"  {label}  ({market})")
        print(f"{'─' * 90}")
        hdr = f"  {'Season':<8} {'Games':<7} {'Cov':<6} {'Cov%':<7} {'AvgLines':<10} {'AvgBooks':<10}  Books"
        print(hdr)
        print(f"  {'-'*7} {'-'*6} {'-'*5} {'-'*6} {'-'*9} {'-'*9}  {'-'*40}")

        for season in SEASONS:
            s = results[market].get(season, {})
            if not s:
                print(f"  {season}")
                continue
            n_tot = s["games_total"]
            n_cov = s["games_covered"]
            pct   = f"{100*n_cov/n_tot:.0f}%" if n_tot else "n/a"
            avg_l = f"{sum(s['lines_list'])/len(s['lines_list']):.1f}" if s["lines_list"] else "0"
            avg_b = f"{sum(s['books_list'])/len(s['books_list']):.1f}" if s["books_list"] else "0"
            books = ", ".join(sorted(s["books_seen"])) or "none"
            print(f"  {season:<8} {n_tot:<7} {n_cov:<6} {pct:<7} {avg_l:<10} {avg_b:<10}  {books}")

    print()


if __name__ == "__main__":
    run()

"""
Barebones: fetch NCAAB from Odds API - h2h moneylines (no scores; scores come from ESPN).
Default: yesterday + today (today split into on-date vs upcoming). With --date: fetch one date only.
Saves CSVs to ~/Downloads/tmp. Team names normalized to ESPN via src.ncaab_team_name_mapping.
Uses ET for dates. Reads ODDS_API_KEY from .env (via load_dotenv) or environment.
  python tmp/fetch_ncaab_odds_api_h2h_and_scores.py
  python tmp/fetch_ncaab_odds_api_h2h_and_scores.py --date 2026-02-18
  python tmp/fetch_ncaab_odds_api_h2h_and_scores.py --date 2026-02-18 --limit-calls-per-day 1
"""
import argparse
import csv
import logging
import os
import sys
import time
import urllib3
import requests
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# Repo root so "from src..." works when run as python tmp/script.py or from notebook
def _find_repo_root():
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / ".gitignore").exists():
            return d
        d = d.parent
    raise RuntimeError("Could not find repo root (no .gitignore)")
sys.path.insert(0, str(_find_repo_root()))

from src.ncaab_team_name_mapping import normalize_ncaab_team_name
from src.odds_utils import odds_to_implied_probability, probability_to_american_odds

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

ET = ZoneInfo("America/New_York")
OUT_DIR = os.path.expanduser("~/Downloads/tmp")

API_KEY = os.environ.get("ODDS_API_KEY")
BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_ncaab"


def _commence_date_et(commence_time_str):
    """Return the game start date in ET, or None if unparseable."""
    if not commence_time_str:
        return None
    s = commence_time_str.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(ET).date()
    except (ValueError, TypeError):
        return None


def _normalize_team_columns(rows):
    """Overwrite home_team and away_team in place with ESPN-aligned names."""
    for row in rows:
        if row.get("home_team") is not None:
            row["home_team"] = normalize_ncaab_team_name(row["home_team"])
        if row.get("away_team") is not None:
            row["away_team"] = normalize_ncaab_team_name(row["away_team"])


def _consensus_american(home_prices, away_prices):
    """Consensus = median implied prob per side, convert back to American. No normalizing."""
    if not home_prices or not away_prices:
        return None, None
    home_probs = sorted(odds_to_implied_probability(p) for p in home_prices)
    away_probs = sorted(odds_to_implied_probability(p) for p in away_prices)
    n_h, n_a = len(home_probs), len(away_probs)
    med_home = home_probs[n_h // 2] if n_h % 2 else (home_probs[n_h // 2 - 1] + home_probs[n_h // 2]) / 2
    med_away = away_probs[n_a // 2] if n_a % 2 else (away_probs[n_a // 2 - 1] + away_probs[n_a // 2]) / 2
    home_american = probability_to_american_odds(med_home * 100)
    away_american = probability_to_american_odds(med_away * 100)
    return int(round(away_american)), int(round(home_american))


def _fair_american(home_prices, away_prices):
    """No-vig fair lines: normalize avg implied probs to sum to 1, convert to American. For optional logging only."""
    if not home_prices or not away_prices:
        return None, None
    avg_home_prob = sum(odds_to_implied_probability(p) for p in home_prices) / len(home_prices)
    avg_away_prob = sum(odds_to_implied_probability(p) for p in away_prices) / len(away_prices)
    total = avg_home_prob + avg_away_prob
    if total <= 0:
        return None, None
    fair_home = avg_home_prob / total
    fair_away = avg_away_prob / total
    home_american = probability_to_american_odds(fair_home * 100)
    away_american = probability_to_american_odds(fair_away * 100)
    return int(round(away_american)), int(round(home_american))


def _fmt_ml(price):
    """Format American ML for logging (e.g. +200, -250)."""
    if price is None:
        return "—"
    return f"+{int(price)}" if price >= 0 else str(int(price))


def _log_matchup_odds(away_team, home_team, book_lines, consensus_away, consensus_home, fair_away=None, fair_home=None):
    """Log matchup header, each book's odds (away, home), consensus (avg of books), then optional fair (no-vig)."""
    log.info("%s @ %s", away_team, home_team)
    for book_name, a_ml, h_ml in book_lines:
        log.info("  - %s: %s %s, %s %s", book_name, away_team, _fmt_ml(a_ml), home_team, _fmt_ml(h_ml))
    log.info("  - consensus: %s %s, %s %s", away_team, _fmt_ml(consensus_away), home_team, _fmt_ml(consensus_home))
    if fair_away is not None and fair_home is not None:
        log.info("  - fair (no-vig): %s %s, %s %s", away_team, _fmt_ml(fair_away), home_team, _fmt_ml(fair_home))


def _fetch_events_and_odds_for_date(date_et, date_str, ts_noon_iso, limit_odds_calls=None):
    """
    Fetch historical events for date_et (noon ET), then h2h odds per event.
    If limit_odds_calls is set, only fetch odds for the first N events (for testing).
    Returns list of rows with fetch_date, game_date, event_id, home_team, away_team,
    commence_time, home_ml_odds, away_ml_odds, num_books, and optionally error.
    """
    url_events = f"{BASE}/historical/sports/{SPORT}/events"
    params_events = {"apiKey": API_KEY, "date": ts_noon_iso, "dateFormat": "iso"}
    log.info("  API: GET %s (historical events for date=%s)", url_events, ts_noon_iso)
    r_ev = requests.get(url_events, params=params_events, verify=False)
    log.info("  Status: %s", r_ev.status_code)
    log.info("  x-requests-remaining: %s", r_ev.headers.get("x-requests-remaining", "N/A"))
    if r_ev.status_code != 200:
        log.error("  Body: %s", r_ev.text[:500])
        return None
    events_list = r_ev.json().get("data", [])
    total_events = len(events_list)
    log.info("  Events for date (noon ET): %s", total_events)

    if limit_odds_calls is not None:
        events_list = events_list[:limit_odds_calls]
        log.info("  Limiting to first %s events for odds (--limit-calls-per-day=%s); %s events skipped", len(events_list), limit_odds_calls, total_events - len(events_list))
    log.info("  API: GET .../events/{event_id}/odds (historical h2h, %s calls)", len(events_list))
    rows = []
    for i, ev in enumerate(events_list, 1):
        eid = ev.get("id")
        home = ev.get("home_team")
        away = ev.get("away_team")
        commence = ev.get("commence_time")
        game_date_et = _commence_date_et(commence)
        url_odds = f"{BASE}/historical/sports/{SPORT}/events/{eid}/odds"
        params_odds = {
            "apiKey": API_KEY,
            "date": ts_noon_iso,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        time.sleep(0.1)
        r_o = requests.get(url_odds, params=params_odds, verify=False)
        if r_o.status_code != 200:
            log.warning("    [%s/%s] %s @ %s: error %s", i, len(events_list), away, home, r_o.status_code)
            rows.append({
                "fetch_date": date_str,
                "game_date": game_date_et.isoformat() if game_date_et else None,
                "event_id": eid,
                "home_team": home,
                "away_team": away,
                "commence_time": commence,
                "error": r_o.status_code,
            })
            continue
        od = r_o.json().get("data", {})
        bookmakers = od.get("bookmakers", [])
        home_prices = []
        away_prices = []
        book_lines = []
        for book in bookmakers:
            for m in book.get("markets", []):
                if m.get("key") != "h2h":
                    continue
                book_name = book.get("key", book.get("title", "?"))
                book_home, book_away = None, None
                for out in m.get("outcomes", []):
                    p = out.get("price")
                    if p is None:
                        continue
                    if out.get("name") == home:
                        book_home = p
                    elif out.get("name") == away:
                        book_away = p
                if book_home is not None or book_away is not None:
                    book_lines.append((book_name, book_away, book_home))
                    if book_home is not None:
                        home_prices.append(book_home)
                    if book_away is not None:
                        away_prices.append(book_away)
                break
        away_ml, home_ml = _consensus_american(home_prices, away_prices)
        fair_away, fair_home = _fair_american(home_prices, away_prices)
        _log_matchup_odds(away, home, book_lines, away_ml, home_ml, fair_away=fair_away, fair_home=fair_home)
        rows.append({
            "fetch_date": date_str,
            "game_date": game_date_et.isoformat() if game_date_et else None,
            "event_id": eid,
            "home_team": home,
            "away_team": away,
            "commence_time": commence,
            "home_ml_odds": home_ml,
            "away_ml_odds": away_ml,
            "num_books": len(home_prices),
        })
    return rows


def _parse_args():
    p = argparse.ArgumentParser(description="Fetch NCAAB h2h moneylines from Odds API.")
    p.add_argument("--date", type=str, default=None, metavar="YYYY-MM-DD", help="Fetch only this date (ET). If omitted, fetch yesterday and today.")
    p.add_argument("--limit-calls-per-day", type=int, default=None, metavar="N", help="Max odds API calls per date (for testing). Events call always runs; only first N events get odds.")
    return p.parse_args()


def main():
    if not API_KEY:
        log.error("ODDS_API_KEY not set")
        sys.exit(1)

    args = _parse_args()
    now_et = datetime.now(ET)
    today_et = now_et.date()

    if args.date:
        try:
            requested = date.fromisoformat(args.date)
        except ValueError:
            log.error("Invalid --date %s; use YYYY-MM-DD", args.date)
            sys.exit(1)
        if requested > today_et:
            log.error("--date %s is in the future (today ET is %s)", args.date, today_et.isoformat())
            sys.exit(1)
        date_str = requested.isoformat()
        log.info("Single-date mode: %s (ET)", date_str)
        os.makedirs(OUT_DIR, exist_ok=True)
        ts_noon = datetime(requested.year, requested.month, requested.day, 12, 0, 0, tzinfo=ET)
        ts_noon_iso = ts_noon.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        log.info("--- Historical events (for h2h) ---")
        rows = _fetch_events_and_odds_for_date(requested, date_str, ts_noon_iso, limit_odds_calls=args.limit_calls_per_day)
        if rows is None:
            sys.exit(1)
        on_date_rows = [r for r in rows if r.get("game_date") == date_str]
        upcoming_rows = [r for r in rows if r.get("game_date") != date_str]
        log.info("  Split by commence date (ET): on-date (%s) = %s, upcoming = %s", date_str, len(on_date_rows), len(upcoming_rows))
        h2h_fieldnames = ["fetch_date", "game_date", "event_id", "home_team", "away_team", "commence_time", "home_ml_odds", "away_ml_odds", "num_books", "error"]
        _normalize_team_columns(on_date_rows)
        out_on = os.path.join(OUT_DIR, f"ncaab_h2h_{date_str}.csv")
        with open(out_on, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=h2h_fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(on_date_rows)
        log.info("  Saved: %s (%s games)", out_on, len(on_date_rows))
        _normalize_team_columns(upcoming_rows)
        out_upcoming = os.path.join(OUT_DIR, f"ncaab_h2h_upcoming_{date_str}.csv")
        with open(out_upcoming, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=h2h_fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(upcoming_rows)
        log.info("  Saved: %s (%s games)", out_upcoming, len(upcoming_rows))
        log.info("Done. ncaab_h2h_%s.csv (%s on-date) + ncaab_h2h_upcoming_%s.csv (%s upcoming).", date_str, len(on_date_rows), date_str, len(upcoming_rows))
        return

    # Default: yesterday + today
    os.makedirs(OUT_DIR, exist_ok=True)

    yesterday_et = today_et - timedelta(days=1)
    today_str = today_et.isoformat()
    yesterday_str = yesterday_et.isoformat()
    log.info("ET now: %s", now_et)
    log.info("Today (ET): %s, Yesterday (ET): %s", today_str, yesterday_str)

    # --- 1) Yesterday's games + ML (historical API; scores come from ESPN)
    ts_yesterday_noon = datetime(yesterday_et.year, yesterday_et.month, yesterday_et.day, 12, 0, 0, tzinfo=ET)
    ts_yesterday_iso = ts_yesterday_noon.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    url_historical_events = f"{BASE}/historical/sports/{SPORT}/events"
    params_yesterday = {"apiKey": API_KEY, "date": ts_yesterday_iso, "dateFormat": "iso"}
    log.info("--- Yesterday's games + ML (historical API) ---")
    log.info("  API: GET %s (historical events for date=%s)", url_historical_events, ts_yesterday_iso)
    r_yesterday = requests.get(url_historical_events, params=params_yesterday, verify=False)
    log.info("  Status: %s", r_yesterday.status_code)
    log.info("  x-requests-remaining: %s", r_yesterday.headers.get("x-requests-remaining", "N/A"))
    if r_yesterday.status_code != 200:
        log.error("  Body: %s", r_yesterday.text[:500])
        sys.exit(1)
    data_yesterday = r_yesterday.json()
    events_yesterday = data_yesterday.get("data", [])
    if not isinstance(events_yesterday, list):
        events_yesterday = []
    if args.limit_calls_per_day is not None:
        events_yesterday = events_yesterday[: args.limit_calls_per_day]
        log.info("  Limiting to first %s events (--limit-calls-per-day=%s)", len(events_yesterday), args.limit_calls_per_day)
    yesterday_rows = []
    for ev in events_yesterday:
        ct = ev.get("commence_time")
        if not ct:
            continue
        yesterday_rows.append({
            "game_date": yesterday_str,
            "event_id": ev.get("id"),
            "home_team": ev.get("home_team"),
            "away_team": ev.get("away_team"),
            "commence_time": ct,
        })
    log.info("  Historical events for yesterday (%s): %s", yesterday_str, len(yesterday_rows))

    # Fetch historical h2h ML for yesterday's games (pre-game odds snapshot)
    log.info("  API: GET %s/historical/sports/%s/events/{event_id}/odds (historical h2h for yesterday, %s calls)", BASE, SPORT, len(yesterday_rows))
    for i, row in enumerate(yesterday_rows, 1):
        eid = row["event_id"]
        home, away = row["home_team"], row["away_team"]
        url_odds = f"{BASE}/historical/sports/{SPORT}/events/{eid}/odds"
        params_odds = {
            "apiKey": API_KEY,
            "date": ts_yesterday_iso,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        time.sleep(0.1)
        r_o = requests.get(url_odds, params=params_odds, verify=False)
        if r_o.status_code != 200:
            log.warning("    [%s/%s] %s @ %s: error %s", i, len(yesterday_rows), away, home, r_o.status_code)
            row["home_ml_odds"] = None
            row["away_ml_odds"] = None
            row["num_books"] = 0
            continue
        od = r_o.json().get("data", {})
        bookmakers = od.get("bookmakers", [])
        home_prices = []
        away_prices = []
        book_lines = []
        for book in bookmakers:
            for m in book.get("markets", []):
                if m.get("key") != "h2h":
                    continue
                book_name = book.get("key", book.get("title", "?"))
                book_home, book_away = None, None
                for out in m.get("outcomes", []):
                    p = out.get("price")
                    if p is None:
                        continue
                    if out.get("name") == home:
                        book_home = p
                    elif out.get("name") == away:
                        book_away = p
                if book_home is not None or book_away is not None:
                    book_lines.append((book_name, book_away, book_home))
                    if book_home is not None:
                        home_prices.append(book_home)
                    if book_away is not None:
                        away_prices.append(book_away)
                break
        away_ml, home_ml = _consensus_american(home_prices, away_prices)
        row["home_ml_odds"] = home_ml
        row["away_ml_odds"] = away_ml
        row["num_books"] = len(home_prices)
        fair_away, fair_home = _fair_american(home_prices, away_prices)
        _log_matchup_odds(away, home, book_lines, away_ml, home_ml, fair_away=fair_away, fair_home=fair_home)

    _normalize_team_columns(yesterday_rows)
    out_yesterday = os.path.join(OUT_DIR, f"ncaab_h2h_{yesterday_str}.csv")
    yesterday_fieldnames = ["game_date", "event_id", "home_team", "away_team", "commence_time", "home_ml_odds", "away_ml_odds", "num_books"]
    with open(out_yesterday, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=yesterday_fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(yesterday_rows)
    log.info("  Saved: %s", out_yesterday)

    # --- 2) H2H moneylines - historical events for today, then split by actual commence date (ET)
    ts_today_noon = datetime(today_et.year, today_et.month, today_et.day, 12, 0, 0, tzinfo=ET)
    ts_iso = ts_today_noon.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    log.info("--- Historical events (for h2h) ---")
    rows = _fetch_events_and_odds_for_date(today_et, today_str, ts_iso, limit_odds_calls=args.limit_calls_per_day)
    if rows is None:
        sys.exit(1)
    today_rows = [r for r in rows if r.get("game_date") == today_str]
    upcoming_rows = [r for r in rows if r.get("game_date") != today_str]
    log.info("  Split by commence date (ET): today (%s) = %s, upcoming = %s", today_str, len(today_rows), len(upcoming_rows))

    h2h_fieldnames = ["fetch_date", "game_date", "event_id", "home_team", "away_team", "commence_time", "home_ml_odds", "away_ml_odds", "num_books", "error"]
    _normalize_team_columns(today_rows)
    out_today = os.path.join(OUT_DIR, f"ncaab_h2h_{today_str}.csv")
    with open(out_today, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=h2h_fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(today_rows)
    log.info("  Saved: %s (%s games)", out_today, len(today_rows))

    _normalize_team_columns(upcoming_rows)
    out_upcoming = os.path.join(OUT_DIR, f"ncaab_h2h_upcoming_{today_str}.csv")
    with open(out_upcoming, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=h2h_fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(upcoming_rows)
    log.info("  Saved: %s (%s games)", out_upcoming, len(upcoming_rows))
    log.info("Done.")


if __name__ == "__main__":
    main()

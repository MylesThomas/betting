#!/usr/bin/env python3
"""
Compare The Odds API historical vs live NBA feeds for one ET calendar date.

Use when ingest says "0 games" but you expect a slate (e.g. playoffs, same-day run).

What it does:
  1) GET historical/sports/basketball_nba/events at UTC noon for the date (matches fetch_nba_player_props).
  2) GET sports/basketball_nba/events (live upcoming list).
  3) Filters by tipoff on the target date in America/New_York (calendar day).
  4) Also shows the old 6:00–23:59 ET window count (legacy ingest behavior).
  5) Optional: --probe-odds hits one event with historical vs live player_props to see 422/empty.

Usage (repo root):
  export ODDS_API_KEY=...
  python scripts/tmp_compare_historical_vs_live_nba_events.py
  python scripts/tmp_compare_historical_vs_live_nba_events.py --date 2026-04-18
  python scripts/tmp_compare_historical_vs_live_nba_events.py --date 2026-04-18 --probe-odds
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ET = ZoneInfo("America/New_York")
BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

# Same single-market probe as a cheap sanity check (ingest pulls many markets).
DEFAULT_PROBE_MARKETS = "player_points"
REGIONS = "us"
ODDS_FORMAT = "american"
DATE_FORMAT = "iso"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--date",
        type=str,
        default=None,
        help="YYYY-MM-DD (default: today ET)",
    )
    p.add_argument(
        "--probe-odds",
        action="store_true",
        help="After listing events, fetch historical vs live odds for one game (extra API credits).",
    )
    p.add_argument(
        "--markets",
        type=str,
        default=DEFAULT_PROBE_MARKETS,
        help=f"Markets for --probe-odds only (default: {DEFAULT_PROBE_MARKETS})",
    )
    p.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification (matches some repo scripts on strict corporate/Mac setups).",
    )
    return p.parse_args()


def _req(
    method: str,
    url: str,
    *,
    params: dict | None = None,
    verify: bool,
) -> tuple[requests.Response, dict]:
    r = requests.request(method, url, params=params, timeout=45, verify=verify)
    hdr = {
        "x-requests-last": r.headers.get("x-requests-last"),
        "x-requests-used": r.headers.get("x-requests-used"),
        "x-requests-remaining": r.headers.get("x-requests-remaining"),
    }
    return r, hdr


def historical_events_snapshot(
    api_key: str, day: date, *, verify: bool
) -> tuple[list[dict], str, dict]:
    """Same UTC snapshot as fetch_nba_player_props.get_historical_events (hour=12, Z suffix)."""
    ts = datetime.combine(day, datetime.min.time()).replace(hour=12).isoformat() + "Z"
    url = f"{BASE}/historical/sports/{SPORT}/events"
    params = {
        "api_key": api_key,
        "date": ts,
        "dateFormat": DATE_FORMAT,
    }
    r, hdr = _req("GET", url, params=params, verify=verify)
    r.raise_for_status()
    data = r.json()
    ev = data.get("data", data if isinstance(data, list) else [])
    lst = ev if isinstance(ev, list) else []
    return lst, ts, hdr


def live_events_all(api_key: str, *, verify: bool) -> tuple[list[dict], dict]:
    url = f"{BASE}/sports/{SPORT}/events"
    params = {"apiKey": api_key}
    r, hdr = _req("GET", url, params=params, verify=verify)
    r.raise_for_status()
    ev = r.json()
    lst = ev if isinstance(ev, list) else []
    return lst, hdr


def filter_et_calendar_day(events: list[dict], day: date) -> list[dict]:
    out = []
    for e in events:
        ct = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")).astimezone(ET)
        if ct.date() == day:
            out.append(e)
    return out


def filter_old_six_am_window(events: list[dict], day: date) -> list[dict]:
    """Legacy ingest: tips between 06:00 and 23:59 ET on that calendar date."""
    start = datetime(day.year, day.month, day.day, 6, 0, 0, tzinfo=ET)
    end = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=ET)
    out = []
    for e in events:
        ct = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")).astimezone(ET)
        if start <= ct <= end:
            out.append(e)
    return out


def _summarize_odds_json(payload: dict) -> dict:
    """Bookmaker and market counts for debug (handles historical {'data': ...} or live shape)."""
    root = payload.get("data", payload) if isinstance(payload, dict) else {}
    if not isinstance(root, dict):
        return {"error": "unexpected_payload_shape"}
    bms = root.get("bookmakers") or []
    n_markets = 0
    for bm in bms:
        for _m in bm.get("markets") or []:
            n_markets += 1
    return {
        "away": root.get("away_team"),
        "home": root.get("home_team"),
        "bookmakers": len(bms),
        "markets_total": n_markets,
    }


def probe_event_odds(
    api_key: str,
    event_id: str,
    day: date,
    markets: str,
    *,
    verify: bool,
) -> None:
    """Compare one event: historical snapshot odds vs live odds."""
    odds_ts = datetime.combine(day, datetime.min.time()).replace(hour=15).isoformat() + "Z"
    hist_url = f"{BASE}/historical/sports/{SPORT}/events/{event_id}/odds"
    hist_params = {
        "api_key": api_key,
        "date": odds_ts,
        "regions": REGIONS,
        "markets": markets,
        "oddsFormat": ODDS_FORMAT,
        "dateFormat": DATE_FORMAT,
    }
    live_url = f"{BASE}/sports/{SPORT}/events/{event_id}/odds"
    live_params = {
        "apiKey": api_key,
        "regions": REGIONS,
        "markets": markets,
        "oddsFormat": ODDS_FORMAT,
        "dateFormat": DATE_FORMAT,
    }

    print()
    print("--- PROBE ODDS (extra API calls) ---")
    print(f"event_id={event_id}")
    print(f"historical date param (matches ingest odds snapshot hour=15 UTC): {odds_ts}")
    print()

    r1, h1 = _req("GET", hist_url, params=hist_params, verify=verify)
    print(f"GET historical .../events/{{id}}/odds  -> HTTP {r1.status_code}  credits: {h1}")
    if r1.ok:
        try:
            sj = _summarize_odds_json(r1.json())
            print(f"  summary: {json.dumps(sj)}")
        except Exception as exc:
            print(f"  (parse error: {exc})")
    else:
        print(f"  body: {r1.text[:500]!r}")

    r2, h2 = _req("GET", live_url, params=live_params, verify=verify)
    print(f"GET live .../events/{{id}}/odds       -> HTTP {r2.status_code}  credits: {h2}")
    if r2.ok:
        try:
            sj = _summarize_odds_json(r2.json())
            print(f"  summary: {json.dumps(sj)}")
        except Exception as exc:
            print(f"  (parse error: {exc})")
    else:
        print(f"  body: {r2.text[:500]!r}")


def _print_event_lines(title: str, evs: list[dict]) -> None:
    print(title)
    if not evs:
        print("  (none)")
        return
    for e in sorted(evs, key=lambda x: x.get("commence_time", "")):
        ct = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")).astimezone(ET)
        print(
            f"  {e.get('away_team')} @ {e.get('home_team')}  |  {ct.isoformat()}  |  id={e.get('id')}"
        )


def main() -> int:
    _load_dotenv()
    args = _parse_args()
    verify_tls = not args.insecure

    api_key = os.environ.get("ODDS_API_KEY") or os.environ.get("THE_ODDS_API_KEY")
    if not api_key:
        print("Set ODDS_API_KEY (or THE_ODDS_API_KEY), or add it to .env at repo root.", file=sys.stderr)
        return 1

    if args.date:
        target = date.fromisoformat(args.date)
    else:
        target = datetime.now(ET).date()

    today_et = datetime.now(ET).date()

    hist_raw, hist_ts, hist_hdr = historical_events_snapshot(api_key, target, verify=verify_tls)
    live_raw, live_hdr = live_events_all(api_key, verify=verify_tls)

    hist_cal = filter_et_calendar_day(hist_raw, target)
    hist_win = filter_old_six_am_window(hist_raw, target)
    live_cal = filter_et_calendar_day(live_raw, target)

    ids_hist = {e.get("id") for e in hist_cal if e.get("id")}
    ids_live = {e.get("id") for e in live_cal if e.get("id")}
    only_hist = ids_hist - ids_live
    only_live = ids_live - ids_hist
    both = ids_hist & ids_live

    print("=== The Odds API: historical vs live (NBA) ===")
    print(f"Target ET calendar date: {target}   |   today ET: {today_et}")
    print()
    print("Historical events request (ingest-aligned):")
    print(f"  GET {BASE}/historical/sports/{SPORT}/events")
    print(f"  date={hist_ts}   (UTC noon for that calendar day)")
    print(f"  response: {len(hist_raw)} event(s) unfiltered   |   API credits: {hist_hdr}")
    print()
    print("Live events request:")
    print(f"  GET {BASE}/sports/{SPORT}/events")
    print(f"  response: {len(live_raw)} event(s) unfiltered   |   API credits: {live_hdr}")
    print()
    print("--- After filters (tipoff on target date, America/New_York) ---")
    print(f"  Calendar-day (recommended):  historical={len(hist_cal)}   live={len(live_cal)}")
    print(f"  Legacy 6:00–23:59 ET only: historical={len(hist_win)}")
    print()
    print("Event id sets (calendar-day lists):")
    print(f"  in both feeds: {len(both)}")
    print(f"  only in historical: {len(only_hist)}")
    print(f"  only in live:       {len(only_live)}")
    if only_hist:
        print(f"    ids only historical: {sorted(only_hist)[:12]}{' ...' if len(only_hist) > 12 else ''}")
    if only_live:
        print(f"    ids only live:       {sorted(only_live)[:12]}{' ...' if len(only_live) > 12 else ''}")
    print()

    _print_event_lines("Historical → calendar day:", hist_cal)
    print()
    _print_event_lines("Live → calendar day:", live_cal)
    print()
    _print_event_lines("Historical → legacy 6am window:", hist_win)

    print()
    print("--- Read ---")
    if not hist_cal and live_cal:
        print(
            "Historical returned no games for this ET date after filtering, but live does. "
            "Typical for same-morning runs: use the live events + live event odds for “today” ET."
        )
    elif not hist_cal and not live_cal:
        print(
            "Neither feed has a game on this ET date (after filter). "
            "Double-check the schedule, or try again later if lines are not posted yet."
        )
    elif len(hist_cal) != len(hist_win):
        print(
            "Calendar-day count != 6am-window count: very early ET tips (midnight–5:59am) "
            "count for the calendar day but fall outside the legacy window."
        )
    else:
        print("Counts line up for the checks above; if ingest still fails, look at S3 skip/force or script logs.")

    if args.probe_odds:
        pick = None
        for pool, label in (
            (live_cal, "live calendar-day"),
            (hist_cal, "historical calendar-day"),
        ):
            if pool:
                pick = pool[0]
                print(f"\n--probe-odds: using first event from {label} --")
                break
        if pick and pick.get("id"):
            probe_event_odds(
                api_key,
                str(pick["id"]),
                target,
                args.markets,
                verify=verify_tls,
            )
        else:
            print("\n--probe-odds: no event to probe (no games in filtered lists).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

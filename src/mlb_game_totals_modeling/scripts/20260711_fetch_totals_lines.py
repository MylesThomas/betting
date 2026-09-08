"""
Fetch historical totals (over/under) lines for all MLB game dates.

Uses the date-level historical odds endpoint (one call per date, all events returned)
rather than per-event calls — much faster (~460 calls vs 4,565).

Snapshot: 13:00 UTC (9 AM ET) — before any MLB games start. Captures the full
pre-game market for every game scheduled that day.

If a game from our known event list is not captured at the 9 AM snapshot (e.g.,
doubleheaders that had lines posted the previous day), we fall back to 16:00 UTC.

Cost: ~460 dates × 20 credits ≈ 9,200 credits total.

Output schema (one row per event_id × bookmaker × line):
  event_id | game_date | home_team | away_team |
  bookmaker | line | over_price | under_price |
  dec_over | dec_under | raw_prob_over | raw_prob_under |
  snapshot | n_books_total

Output paths:
  S3:    s3://the-odds-api-mt/mlb/game_totals_model/lines/mlb_game_totals_lines.parquet
  Local: ~/Downloads/tmp/mlb_game_totals/totals_lines.parquet

Usage:
  python src/mlb_game_totals_modeling/scripts/20260711_fetch_totals_lines.py
  python src/mlb_game_totals_modeling/scripts/20260711_fetch_totals_lines.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "baseball_mlb"
REGIONS       = "us"
SLEEP_S       = 0.2

S3_BUCKET        = "the-odds-api-mt"
SHARED_LINES_KEY = "mlb/game_lines/mlb_game_lines.parquet"
TOTALS_LINES_KEY = "mlb/game_totals_model/lines/mlb_game_totals_lines.parquet"
LOCAL_OUT        = Path.home() / "Downloads/tmp/mlb_game_totals/totals_lines.parquet"

# Snapshot times to try per date (UTC): 9 AM ET, then noon ET as fallback
SNAPSHOT_HOURS_UTC = [13, 16]


def _american_to_decimal(odds: float) -> float:
    if odds >= 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)


def _fetch_date_odds(date_str: str, snapshot_hour_utc: int) -> tuple[list[dict], int]:
    """Fetch all MLB totals events at a given date + hour snapshot."""
    snapshot = f"{date_str}T{snapshot_hour_utc:02d}:00:00Z"
    for attempt in range(3):
        try:
            r = requests.get(
                f"{ODDS_API_BASE}/historical/sports/{SPORT}/odds",
                params={
                    "apiKey":     ODDS_API_KEY,
                    "markets":    "totals",
                    "regions":    REGIONS,
                    "oddsFormat": "american",
                    "date":       snapshot,
                },
                timeout=30,
            )
            break
        except requests.RequestException:
            if attempt == 2:
                return [], -1
            time.sleep(2 ** (attempt + 1))

    remaining = int(r.headers.get("x-requests-remaining", 999_999))
    if r.status_code != 200:
        return [], remaining

    data = r.json().get("data", [])
    # Filter to only events for this game_date
    events_for_date = [
        e for e in data
        if e.get("commence_time", "")[:10] == date_str
    ]
    return events_for_date, remaining


def _parse_event_rows(event: dict, snapshot: str) -> list[dict]:
    """Extract one row per (bookmaker, line) from a single event."""
    event_id  = event["id"]
    home_team = event["home_team"]
    away_team = event["away_team"]
    game_date = event["commence_time"][:10]

    bookmakers = event.get("bookmakers", [])
    n_books_with_totals = sum(
        1 for bk in bookmakers
        if any(m["key"] == "totals" for m in bk.get("markets", []))
    )

    rows = []
    for bk in bookmakers:
        book_name = bk["key"]
        for mkt in bk.get("markets", []):
            if mkt["key"] != "totals":
                continue
            outcomes = mkt.get("outcomes", [])
            over_out  = next((o for o in outcomes if o["name"] == "Over"),  None)
            under_out = next((o for o in outcomes if o["name"] == "Under"), None)
            if not over_out or not under_out:
                continue
            line = over_out.get("point") or under_out.get("point")
            if line is None:
                continue
            over_price  = float(over_out["price"])
            under_price = float(under_out["price"])
            dec_over    = _american_to_decimal(over_price)
            dec_under   = _american_to_decimal(under_price)
            rows.append({
                "event_id":       event_id,
                "game_date":      game_date,
                "home_team":      home_team,
                "away_team":      away_team,
                "bookmaker":      book_name,
                "line":           float(line),
                "over_price":     over_price,
                "under_price":    under_price,
                "dec_over":       dec_over,
                "dec_under":      dec_under,
                "raw_prob_over":  1.0 / dec_over,
                "raw_prob_under": 1.0 / dec_under,
                "snapshot":       snapshot,
                "n_books_total":  n_books_with_totals,
            })
    return rows


def load_events() -> pd.DataFrame:
    """Load shared game_lines parquet to get all known event_ids."""
    s3   = boto3.client("s3")
    body = s3.get_object(Bucket=S3_BUCKET, Key=SHARED_LINES_KEY)["Body"].read()
    df   = pd.read_parquet(BytesIO(body))
    return (
        df[["event_id", "game_date", "home_team", "away_team"]]
        .drop_duplicates("event_id")
        .reset_index(drop=True)
    )


def load_existing() -> pd.DataFrame:
    try:
        s3   = boto3.client("s3")
        body = s3.get_object(Bucket=S3_BUCKET, Key=TOTALS_LINES_KEY)["Body"].read()
        return pd.read_parquet(BytesIO(body))
    except Exception:
        return pd.DataFrame()


def save(df: pd.DataFrame) -> None:
    s3  = boto3.client("s3")
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=TOTALS_LINES_KEY, Body=buf.getvalue())
    print(f"  Saved {len(df)} rows → s3://{S3_BUCKET}/{TOTALS_LINES_KEY}")
    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LOCAL_OUT, index=False)
    print(f"  Saved → {LOCAL_OUT}")


def main(dry_run: bool = False) -> None:
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY environment variable not set")

    # All known events (from shared game_lines)
    events       = load_events()
    known_ids    = set(events["event_id"])
    existing_df  = load_existing()
    done_ids     = set(existing_df["event_id"].unique()) if not existing_df.empty else set()

    need_ids  = known_ids - done_ids
    need_df   = events[events["event_id"].isin(need_ids)]
    all_dates = sorted(need_df["game_date"].unique())

    print(f"Known events: {len(known_ids)} | Already fetched: {len(done_ids)} | Remaining: {len(need_ids)}")
    print(f"Unique dates to fetch: {len(all_dates)}")

    if dry_run:
        print("Dry-run — no API calls.")
        return

    all_rows: list[dict] = []
    covered_ids: set[str] = set()
    credits_remaining = None

    for i, date_str in enumerate(all_dates):
        # Get event_ids we still need for this date
        date_need = set(need_df[need_df["game_date"] == date_str]["event_id"])

        events_found: list[dict] = []
        for hour_utc in SNAPSHOT_HOURS_UTC:
            snapshot   = f"{date_str}T{hour_utc:02d}:00:00Z"
            day_events, credits_remaining = _fetch_date_odds(date_str, hour_utc)
            # Only keep events we need
            relevant = [e for e in day_events if e["id"] in date_need]
            for e in relevant:
                if e["id"] not in covered_ids:
                    events_found.append(e)
            time.sleep(SLEEP_S)

            # If we've covered all needed events for this date, stop trying snapshots
            covered_now = {e["id"] for e in events_found}
            if date_need <= covered_now:
                break

        for event in events_found:
            snapshot_used = f"{date_str}T{SNAPSHOT_HOURS_UTC[0]:02d}:00:00Z"
            rows = _parse_event_rows(event, snapshot_used)
            all_rows.extend(rows)
            covered_ids.add(event["id"])

        missed = date_need - covered_ids
        if missed and len(missed) < 5:
            print(f"  {date_str}: covered {len(events_found)} events, missed {len(missed)}: {missed}")
        elif missed:
            print(f"  {date_str}: covered {len(events_found)} events, missed {len(missed)}")

        if (i + 1) % 50 == 0 or (i + 1) == len(all_dates):
            print(f"  [{i+1}/{len(all_dates)}] {date_str} | Credits remaining: {credits_remaining}")
            if all_rows:
                new_df = pd.DataFrame(all_rows)
                if not existing_df.empty:
                    combined = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    combined = new_df
                save(combined)
                existing_df = combined
                all_rows = []

    print(f"\nDone. Credits remaining: {credits_remaining}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)

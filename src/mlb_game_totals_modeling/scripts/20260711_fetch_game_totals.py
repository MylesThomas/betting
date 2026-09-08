"""
Fetch historical game totals (over/under) odds for all MLB events in the
shared game_lines parquet (2024-03-28 → present).

One row per (event_id, bookmaker, line) — captures every book's posted line and
over/under odds at the pre-game snapshot. Different books may post different lines
for the same game (e.g., DraftKings at 8.5, BetMGM at 9.0).

Cost estimate: ~20 credits/event × 4,565 events ≈ 91,300 credits.
Idempotent — skips event_ids already in the output parquet.

Output schema:
  event_id   | game_date | home_team | away_team |
  bookmaker  | line      | over_price | under_price

Output:
  S3:    s3://the-odds-api-mt/mlb/game_totals_model/odds/mlb_game_totals_odds.parquet
  Local: ~/Downloads/tmp/mlb_game_totals/game_totals_odds.parquet

Usage:
  python src/mlb_game_totals_modeling/scripts/20260711_fetch_game_totals.py
  python src/mlb_game_totals_modeling/scripts/20260711_fetch_game_totals.py --dry-run
  python src/mlb_game_totals_modeling/scripts/20260711_fetch_game_totals.py --limit 50
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
SLEEP_S       = 0.3
CREDIT_STOP   = 50_000

S3_BUCKET      = "the-odds-api-mt"
GAME_LINES_KEY = "mlb/game_lines/mlb_game_lines.parquet"
OUTPUT_KEY     = "mlb/game_totals_model/odds/mlb_game_totals_odds.parquet"
LOCAL_OUT      = Path.home() / "Downloads/tmp/mlb_game_totals/game_totals_odds.parquet"


class CreditExhausted(Exception):
    pass


def _snapshot_times(game_date: str) -> list[str]:
    """Return candidate UTC snapshot times for a game_date (11am, 1pm, 3pm ET = 15, 17, 19 UTC)."""
    times = []
    for hour_et in (11, 13, 15):
        dt_utc = datetime(
            int(game_date[:4]), int(game_date[5:7]), int(game_date[8:]),
            hour_et + 4, 0, 0, tzinfo=timezone.utc,
        )
        times.append(dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return times


def _fetch_totals(event_id: str, snapshot: str) -> tuple[list[dict], int]:
    """Fetch totals market for one event at a given snapshot. Returns (bookmakers, credits_remaining)."""
    for attempt in range(3):
        try:
            r = requests.get(
                f"{ODDS_API_BASE}/historical/sports/{SPORT}/events/{event_id}/odds",
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

    data = r.json().get("data") or {}
    return data.get("bookmakers", []), remaining


def _parse_event(bookmakers: list[dict]) -> list[dict]:
    """Extract per-book, per-line over/under odds from bookmakers list."""
    rows = []
    for bk in bookmakers:
        book_key = bk.get("key", "")
        for mkt in bk.get("markets", []):
            if mkt["key"] != "totals":
                continue
            outcomes = mkt.get("outcomes", [])
            over_out  = next((o for o in outcomes if o["name"] == "Over"),  None)
            under_out = next((o for o in outcomes if o["name"] == "Under"), None)
            if over_out and under_out and over_out.get("point") == under_out.get("point"):
                rows.append({
                    "bookmaker":   book_key,
                    "line":        float(over_out["point"]),
                    "over_price":  int(over_out["price"]),
                    "under_price": int(under_out["price"]),
                })
    return rows


def load_events() -> pd.DataFrame:
    s3   = boto3.client("s3")
    body = s3.get_object(Bucket=S3_BUCKET, Key=GAME_LINES_KEY)["Body"].read()
    df   = pd.read_parquet(BytesIO(body))
    return (
        df[["event_id", "game_date", "home_team", "away_team"]]
        .drop_duplicates("event_id")
        .reset_index(drop=True)
    )


def load_existing() -> pd.DataFrame:
    try:
        s3   = boto3.client("s3")
        body = s3.get_object(Bucket=S3_BUCKET, Key=OUTPUT_KEY)["Body"].read()
        return pd.read_parquet(BytesIO(body))
    except Exception:
        return pd.DataFrame()


def save_output(df: pd.DataFrame) -> None:
    s3  = boto3.client("s3")
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=OUTPUT_KEY, Body=buf.getvalue())
    print(f"  Saved → s3://{S3_BUCKET}/{OUTPUT_KEY}  ({len(df)} rows)")

    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LOCAL_OUT, index=False)
    print(f"  Saved → {LOCAL_OUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit",   type=int, default=None)
    args = parser.parse_args()

    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY not set")

    print("Loading events from game_lines parquet...")
    events   = load_events()
    existing = load_existing()

    already_done = set(existing["event_id"].tolist()) if not existing.empty else set()
    todo = events[~events["event_id"].isin(already_done)].reset_index(drop=True)

    if args.limit:
        todo = todo.head(args.limit)

    print(f"  Total events:    {len(events)}")
    print(f"  Already fetched: {len(already_done)}")
    print(f"  To fetch:        {len(todo)}")
    print(f"  Est. cost:       ~{len(todo) * 20:,} credits (20/event)")

    if args.dry_run:
        print("\nDry run — exiting.")
        return

    new_rows: list[dict] = []
    failed:   list[str]  = []
    checkpoint_every = 200

    for i, row in todo.iterrows():
        event_id  = row["event_id"]
        game_date = str(row["game_date"])[:10]

        parsed_rows = []
        for snapshot in _snapshot_times(game_date):
            bks, remaining = _fetch_totals(event_id, snapshot)
            if remaining >= 0 and remaining < CREDIT_STOP:
                raise CreditExhausted(
                    f"Credits remaining ({remaining}) below safety threshold {CREDIT_STOP}"
                )
            if bks:
                parsed_rows = _parse_event(bks)
                if parsed_rows:
                    break
            time.sleep(SLEEP_S)

        if parsed_rows:
            for pr in parsed_rows:
                new_rows.append({
                    "event_id":  event_id,
                    "game_date": game_date,
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    **pr,
                })
        else:
            failed.append(event_id)

        if (i + 1) % 100 == 0:
            pct = (i + 1) / len(todo) * 100
            print(f"  [{i+1}/{len(todo)}] {pct:.0f}% · rows={len(new_rows)} · failed={len(failed)}")

        # Checkpoint every N events
        if new_rows and (i + 1) % checkpoint_every == 0:
            combined = pd.concat(
                [existing, pd.DataFrame(new_rows)],
                ignore_index=True,
            ).drop_duplicates(["event_id", "bookmaker", "line"])
            save_output(combined)
            print(f"  [checkpoint] saved {len(combined)} total rows")

    print(f"\nDone. New rows: {len(new_rows)} · Events with no data: {len(failed)}")

    if new_rows:
        combined = pd.concat(
            [existing, pd.DataFrame(new_rows)],
            ignore_index=True,
        ).drop_duplicates(["event_id", "bookmaker", "line"])
        save_output(combined)

    if failed:
        print(f"\nEvents with no totals data ({len(failed)}):")
        for eid in failed[:20]:
            print(f"  {eid}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")


if __name__ == "__main__":
    main()

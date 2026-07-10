"""
Fetch historical h2h + spreads (run line) for all events in the MLB strikeouts
labeled dataset. Saves consensus no-vig probabilities per event to S3.

Cost estimate: ~20 credits per event × 4,565 events ≈ 91,300 credits total.
Idempotent — skips events already in the output parquet.

Output schema (one row per event_id):
  event_id | game_date | home_team | away_team |
  home_ml_prob | away_ml_prob | home_rl_prob | away_rl_prob |
  n_h2h_books | n_spread_books

  home_ml_prob  — no-vig win probability for home team (from h2h median across books)
  home_rl_prob  — no-vig prob home team covers −1.5 (from spreads median across books)

Output:
  S3:    s3://the-odds-api-mt/mlb/strikeouts_model/game_lines/mlb_game_lines.parquet
  Local: ~/Downloads/tmp/mlb_strikeouts/game_lines.parquet

Usage:
  python src/mlb_strikeouts_modeling/scripts/20260709_fetch_game_lines.py
  python src/mlb_strikeouts_modeling/scripts/20260709_fetch_game_lines.py --dry-run
  python src/mlb_strikeouts_modeling/scripts/20260709_fetch_game_lines.py --limit 50
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
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

S3_BUCKET    = "the-odds-api-mt"
LABELED_KEY  = "mlb/strikeouts_model/labeled/mlb_strikeouts_labeled.parquet"
LINES_KEY    = "mlb/strikeouts_model/game_lines/mlb_game_lines.parquet"
LOCAL_OUT    = Path.home() / "Downloads/tmp/mlb_strikeouts/game_lines.parquet"


class CreditExhausted(Exception):
    pass


def _american_to_decimal(odds: float) -> float:
    if odds >= 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)


def _novig_pair(price_a: float, price_b: float) -> tuple[float, float]:
    """Return (novig_a, novig_b) from two American odds values."""
    raw_a = 1 / _american_to_decimal(price_a)
    raw_b = 1 / _american_to_decimal(price_b)
    total = raw_a + raw_b
    return raw_a / total, raw_b / total


def _snapshot_times(game_date: str) -> list[str]:
    """Return candidate UTC snapshot times to try for a given game_date.
    Uses 11am, 1pm, 3pm ET — pre-game window covering all game start times.
    """
    times = []
    for hour_et in (11, 13, 15):
        # ET offset: -4 in summer (EDT)
        dt_utc = datetime(
            int(game_date[:4]), int(game_date[5:7]), int(game_date[8:]),
            hour_et + 4, 0, 0, tzinfo=timezone.utc,
        )
        times.append(dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return times


def _fetch_odds(event_id: str, snapshot: str) -> tuple[list[dict], int]:
    """Fetch h2h + spreads for one event at a given snapshot time.
    Returns (bookmakers_list, credits_remaining).
    """
    for attempt in range(3):
        try:
            r = requests.get(
                f"{ODDS_API_BASE}/historical/sports/{SPORT}/events/{event_id}/odds",
                params={
                    "apiKey":  ODDS_API_KEY,
                    "markets": "h2h,spreads",
                    "regions": REGIONS,
                    "oddsFormat": "american",
                    "date":    snapshot,
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


def _parse_event(bookmakers: list[dict], home_team: str) -> dict | None:
    """Extract consensus (median across books) no-vig probs from bookmakers list.
    Returns None if insufficient data (<2 books for either market).
    """
    h2h_home_probs:    list[float] = []
    spread_home_probs: list[float] = []

    for bk in bookmakers:
        for mkt in bk.get("markets", []):
            outcomes = mkt.get("outcomes", [])
            if mkt["key"] == "h2h" and len(outcomes) == 2:
                home_out = next((o for o in outcomes if o["name"] == home_team), None)
                away_out = next((o for o in outcomes if o["name"] != home_team), None)
                if home_out and away_out:
                    try:
                        p_home, _ = _novig_pair(home_out["price"], away_out["price"])
                        h2h_home_probs.append(p_home)
                    except Exception:
                        pass

            elif mkt["key"] == "spreads":
                # Home team covers −1.5
                home_rl = next(
                    (o for o in outcomes if o["name"] == home_team and o.get("point", 0) < 0), None
                )
                away_rl = next(
                    (o for o in outcomes if o["name"] != home_team and o.get("point", 0) > 0), None
                )
                if home_rl and away_rl:
                    try:
                        p_home_rl, _ = _novig_pair(home_rl["price"], away_rl["price"])
                        spread_home_probs.append(p_home_rl)
                    except Exception:
                        pass

    if not h2h_home_probs:
        return None

    home_ml = float(pd.Series(h2h_home_probs).median())
    home_rl = float(pd.Series(spread_home_probs).median()) if spread_home_probs else float("nan")

    return {
        "home_ml_prob":    home_ml,
        "away_ml_prob":    1.0 - home_ml,
        "home_rl_prob":    home_rl,
        "away_rl_prob":    (1.0 - home_rl) if not pd.isna(home_rl) else float("nan"),
        "n_h2h_books":     len(h2h_home_probs),
        "n_spread_books":  len(spread_home_probs),
    }


def load_labeled_events() -> pd.DataFrame:
    s3   = boto3.client("s3")
    body = s3.get_object(Bucket=S3_BUCKET, Key=LABELED_KEY)["Body"].read()
    df   = pd.read_parquet(BytesIO(body))
    return (
        df[["event_id", "game_date", "home_team", "away_team"]]
        .drop_duplicates("event_id")
        .reset_index(drop=True)
    )


def load_existing_lines() -> pd.DataFrame:
    try:
        s3   = boto3.client("s3")
        body = s3.get_object(Bucket=S3_BUCKET, Key=LINES_KEY)["Body"].read()
        return pd.read_parquet(BytesIO(body))
    except Exception:
        return pd.DataFrame()


def save_lines(df: pd.DataFrame) -> None:
    s3  = boto3.client("s3")
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=LINES_KEY, Body=buf.getvalue())
    print(f"  Saved → s3://{S3_BUCKET}/{LINES_KEY}  ({len(df)} rows)")

    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LOCAL_OUT, index=False)
    print(f"  Saved → {LOCAL_OUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print plan, skip API calls")
    parser.add_argument("--limit",   type=int, default=None, help="Max events to fetch (for testing)")
    args = parser.parse_args()

    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY not set")

    print("Loading labeled dataset events...")
    events   = load_labeled_events()
    existing = load_existing_lines()

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

    for i, row in todo.iterrows():
        event_id  = row["event_id"]
        game_date = row["game_date"]
        home_team = row["home_team"]

        parsed = None
        for snapshot in _snapshot_times(game_date):
            bks, remaining = _fetch_odds(event_id, snapshot)
            if remaining >= 0 and remaining < CREDIT_STOP:
                raise CreditExhausted(f"Credits remaining ({remaining}) below safety threshold {CREDIT_STOP}")
            if bks:
                parsed = _parse_event(bks, home_team)
                if parsed:
                    break
            time.sleep(SLEEP_S)

        if parsed:
            new_rows.append({
                "event_id":  event_id,
                "game_date": game_date,
                "home_team": home_team,
                "away_team": row["away_team"],
                **parsed,
            })
        else:
            failed.append(event_id)

        if (i + 1) % 100 == 0:
            pct = (i + 1) / len(todo) * 100
            print(f"  [{i+1}/{len(todo)}] {pct:.0f}% · found={len(new_rows)} · failed={len(failed)}")

    print(f"\nDone. Fetched: {len(new_rows)} · No data: {len(failed)}")

    if new_rows:
        combined = pd.concat(
            [existing, pd.DataFrame(new_rows)],
            ignore_index=True,
        ).drop_duplicates("event_id")
        save_lines(combined)
    else:
        print("No new rows to save.")

    if failed:
        print(f"\nEvents with no h2h data ({len(failed)}):")
        for eid in failed[:20]:
            print(f"  {eid}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")


if __name__ == "__main__":
    main()

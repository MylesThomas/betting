"""
Fetch live batter_total_bases prop snapshots from Odds API and write to S3.

Called by lambda_snapshot.py and runnable as a standalone script:
  uv run python src/mlb_total_bases_modeling/scripts/snapshot_props.py

Output schema (one row per player × bookmaker × market_key × snapshot):
  snapshot_ts_utc, snapshot_ts_et, season, game_date_et, game_date_utc, event_id,
  home_team, away_team, commence_time_utc, commence_time_et,
  bookmaker, market_key, player_name,
  over_line, over_american_odds, under_line, under_american_odds,
  binary_player_game_first_seen, last_odds_player_game,
  credits_before, credits_after
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd
import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

SPORT         = "baseball_mlb"
MARKETS       = "batter_total_bases,batter_total_bases_alternate"
REGIONS       = "us,us2"
S3_BUCKET     = "the-odds-api-mt"
S3_PREFIX     = "mlb/total_bases_model/prop_snapshots"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"


def _api_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "ODDS_API_KEY is not set. Add it to your .env file or environment."
        )
    return key


def parse_remaining(headers: dict) -> int:
    try:
        return int(headers.get("x-requests-remaining", 999_999))
    except (ValueError, TypeError):
        return 999_999


def _get_events(api_key: str) -> tuple[list[dict], int]:
    """Fetch today's events list. Returns (events, credits_remaining)."""
    r = requests.get(
        f"{ODDS_API_BASE}/sports/{SPORT}/events",
        params={"apiKey": api_key},
        timeout=30,
    )
    r.raise_for_status()
    remaining = parse_remaining(r.headers)
    return r.json(), remaining


def _fetch_event_odds(event_id: str, api_key: str) -> tuple[list[dict], int]:
    """
    Fetch prop odds for one event.
    Returns (rows, credits_remaining).
    One row per player × bookmaker × market_key (over/under merged into one row).
    """
    r = requests.get(
        f"{ODDS_API_BASE}/sports/{SPORT}/events/{event_id}/odds",
        params={
            "apiKey":     api_key,
            "markets":    MARKETS,
            "regions":    REGIONS,
            "oddsFormat": "american",
        },
        timeout=30,
    )
    remaining = parse_remaining(r.headers)
    if r.status_code != 200:
        return [], remaining

    data = r.json()
    rows: list[dict] = []
    for bm in data.get("bookmakers", []):
        book = bm["key"]
        for mkt in bm.get("markets", []):
            mkt_key = mkt["key"]
            outcomes = mkt.get("outcomes", [])
            over_outcomes  = [o for o in outcomes if o["name"] == "Over"]
            under_outcomes = [o for o in outcomes if o["name"] == "Under"]

            # Pair by player description + point value
            for o in over_outcomes:
                pt     = o.get("point")
                player = o.get("description", "")
                under  = next(
                    (u for u in under_outcomes
                     if u.get("description") == player and u.get("point") == pt),
                    None,
                )
                rows.append({
                    "bookmaker":           book,
                    "market_key":          mkt_key,
                    "player_name":         player,
                    "over_line":           float(pt) if pt is not None else None,
                    "over_american_odds":  int(o["price"]) if o.get("price") is not None else None,
                    "under_line":          float(pt) if pt is not None else None,
                    "under_american_odds": int(under["price"]) if under and under.get("price") is not None else None,
                })

            # Handle unmatched unders (no corresponding over at same line)
            matched_under_players_pts = {
                (o.get("description", ""), o.get("point"))
                for o in over_outcomes
            }
            for u in under_outcomes:
                key = (u.get("description", ""), u.get("point"))
                if key not in matched_under_players_pts:
                    pt     = u.get("point")
                    player = u.get("description", "")
                    rows.append({
                        "bookmaker":           book,
                        "market_key":          mkt_key,
                        "player_name":         player,
                        "over_line":           float(pt) if pt is not None else None,
                        "over_american_odds":  None,
                        "under_line":          float(pt) if pt is not None else None,
                        "under_american_odds": int(u["price"]) if u.get("price") is not None else None,
                    })

    return rows, remaining


def _load_existing_snapshots(s3_client, season: int, game_date: str) -> pd.DataFrame:
    """
    Load all existing snapshot parquets for a given season/game_date from S3.
    Returns empty DataFrame if none exist.
    """
    prefix = f"{S3_PREFIX}/{season}/{game_date}/"
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix)
        keys = [
            obj["Key"]
            for page in pages
            for obj in page.get("Contents", [])
            if obj["Key"].endswith(".parquet")
        ]
    except Exception:
        return pd.DataFrame()

    if not keys:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for key in keys:
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            frames.append(pd.read_parquet(BytesIO(obj["Body"].read())))
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _build_last_odds_json(row: pd.Series) -> str | None:
    """Build the last_odds_player_game JSON string from a historical row."""
    def _int(v):
        try:
            return None if pd.isna(v) else int(v)
        except (TypeError, ValueError):
            return None

    def _float(v):
        try:
            return None if pd.isna(v) else float(v)
        except (TypeError, ValueError):
            return None

    return json.dumps({
        "over":  {"line_value": _float(row["over_line"]),  "american_odds": _int(row["over_american_odds"])},
        "under": {"line_value": _float(row["under_line"]), "american_odds": _int(row["under_american_odds"])},
    })


def main() -> dict:
    """
    Fetch live prop snapshots and write to S3.

    Returns:
        {"rows_written": int, "credits_used": int, "credits_after": int}
    """
    api_key = _api_key()
    s3 = boto3.client("s3")

    # Step 1: record credits_before + get today's events
    events, credits_before = _get_events(api_key)
    print(f"Events fetched: {len(events)}, credits_before={credits_before:,}")

    if not events:
        print("No events returned from Odds API.")
        return {"rows_written": 0, "credits_used": 0, "credits_after": credits_before}

    snapshot_ts_utc = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snapshot_ts_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _et = ZoneInfo("America/New_York")
    snapshot_ts_et  = datetime.now(timezone.utc).astimezone(_et).strftime("%Y-%m-%d %I:%M %p ET")

    # Step 2: fetch odds per event
    all_rows: list[dict] = []
    credits_after = credits_before  # will be updated on each call

    for ev in events:
        event_id     = ev["id"]
        home_team    = ev.get("home_team", "")
        away_team    = ev.get("away_team", "")
        commence_time = ev.get("commence_time", "")
        # game_date = ET calendar date of game start (Odds API commence_time is UTC;
        # west coast evening games cross midnight UTC so [:10] would give wrong date)
        if commence_time:
            _ct_et = (
                datetime.fromisoformat(commence_time.rstrip("Z"))
                .replace(tzinfo=timezone.utc)
                .astimezone(_et)
            )
            game_date_et     = _ct_et.strftime("%Y-%m-%d")
            game_date_utc    = commence_time[:10]
            commence_time_et = _ct_et.strftime("%Y-%m-%d %I:%M %p ET")
        else:
            game_date_et     = datetime.now(_et).strftime("%Y-%m-%d")
            game_date_utc    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            commence_time_et = ""
        season = int(game_date_et[:4])

        odds_rows, credits_after = _fetch_event_odds(event_id, api_key)
        print(
            f"  {game_date_et}  {away_team[:15]:15} @ {home_team[:15]:15}  "
            f"{len(odds_rows):4} rows  credits={credits_after:,}"
        )

        for row in odds_rows:
            all_rows.append({
                "snapshot_ts_utc":   snapshot_ts_iso,
                "snapshot_ts_et":    snapshot_ts_et,
                "season":            season,
                "game_date_et":      game_date_et,
                "game_date_utc":     game_date_utc,
                "event_id":          event_id,
                "home_team":         home_team,
                "away_team":         away_team,
                "commence_time_utc": commence_time,
                "commence_time_et":  commence_time_et,
                **row,
                # placeholders filled in below
                "binary_player_game_first_seen": None,
                "last_odds_player_game":         None,
                "credits_before":                credits_before,
                "credits_after":                 credits_after,
            })

    if not all_rows:
        print("No prop rows fetched.")
        return {
            "rows_written": 0,
            "credits_used": credits_before - credits_after,
            "credits_after": credits_after,
        }

    df = pd.DataFrame(all_rows)

    # Step 3: first-seen detection per game_date_et
    for game_date_et, grp_idx in df.groupby("game_date_et").groups.items():
        grp    = df.loc[grp_idx]
        season = int(grp["season"].iloc[0])

        existing = _load_existing_snapshots(s3, season, game_date_et)

        if existing.empty:
            already_seen: set[tuple] = set()
        else:
            already_seen = set(
                zip(existing["player_name"], existing["event_id"])
            )

        for idx in grp_idx:
            player_name = df.at[idx, "player_name"]
            event_id    = df.at[idx, "event_id"]
            pair        = (player_name, event_id)

            if pair not in already_seen:
                df.at[idx, "binary_player_game_first_seen"] = True
                df.at[idx, "last_odds_player_game"]         = None
            else:
                df.at[idx, "binary_player_game_first_seen"] = False
                # Look up most recent prior row for this player × event × bookmaker × market_key
                bookmaker  = df.at[idx, "bookmaker"]
                market_key = df.at[idx, "market_key"]
                if not existing.empty:
                    mask = (
                        (existing["player_name"] == player_name)
                        & (existing["event_id"]   == event_id)
                        & (existing["bookmaker"]  == bookmaker)
                        & (existing["market_key"] == market_key)
                    )
                    prior = existing.loc[mask]
                    if not prior.empty:
                        # Most recent by snapshot_ts_utc
                        prior_sorted = prior.sort_values("snapshot_ts_utc", ascending=False)
                        df.at[idx, "last_odds_player_game"] = _build_last_odds_json(
                            prior_sorted.iloc[0]
                        )

    # Step 4: enforce dtypes
    df["binary_player_game_first_seen"] = df["binary_player_game_first_seen"].astype(bool)
    df["last_odds_player_game"] = df["last_odds_player_game"].astype(object)  # always VARCHAR, never NULL-typed
    df["credits_after"] = credits_after  # update with final value from last API call

    # Step 5: write one parquet per game_date_et
    total_rows = 0
    for game_date_et, grp_idx in df.groupby("game_date_et").groups.items():
        grp    = df.loc[grp_idx].copy()
        season = int(grp["season"].iloc[0])
        s3_key = f"{S3_PREFIX}/{season}/{game_date_et}/snapshot_{snapshot_ts_utc}.parquet"

        buf = BytesIO()
        grp.to_parquet(buf, index=False)
        s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=buf.getvalue())
        print(f"  Written → s3://{S3_BUCKET}/{s3_key}  ({len(grp):,} rows)")
        total_rows += len(grp)

    credits_used = credits_before - credits_after
    print(f"Done. rows_written={total_rows:,}  credits_used={credits_used}  credits_after={credits_after:,}")
    return {
        "rows_written":  total_rows,
        "credits_used":  credits_used,
        "credits_after": credits_after,
    }


if __name__ == "__main__":
    main()

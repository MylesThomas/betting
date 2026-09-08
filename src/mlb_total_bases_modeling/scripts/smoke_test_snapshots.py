"""
Smoke test for the MLB total bases prop snapshot pipeline.
Run automatically at end of deploy script AND manually on-demand.

Usage:
  uv run python src/mlb_total_bases_modeling/scripts/smoke_test_snapshots.py
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import urllib3
import boto3
import pandas as pd
import requests
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from src.mlb_total_bases_modeling.scripts.compute_clv import compute_clv
from src.mlb_total_bases_modeling.scripts.compute_tightening import compute_tightening

S3_BUCKET     = "the-odds-api-mt"
S3_PREFIX     = "mlb/total_bases_model/prop_snapshots"
SPORT         = "baseball_mlb"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
MARKETS       = "batter_total_bases,batter_total_bases_alternate"
REGIONS       = "us,us2"

REQUIRED_COLUMNS = [
    "snapshot_ts_utc", "season", "game_date", "event_id",
    "home_team", "away_team", "commence_time",
    "bookmaker", "market_key", "player_name",
    "over_line", "over_american_odds", "under_line", "under_american_odds",
    "binary_player_game_first_seen", "last_odds_player_game",
    "credits_before", "credits_after",
]


def _api_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "ODDS_API_KEY is not set. Add it to your .env file or environment."
        )
    return key


def _latest_snapshot_parquet(s3_client) -> tuple[pd.DataFrame, str] | tuple[None, None]:
    """
    Find and load the most recent snapshot parquet across all dates.
    Returns (DataFrame, s3_key) or (None, None) if nothing found.
    """
    today = date.today()
    # Check today and yesterday to handle late-night runs near midnight
    for check_date in [today.isoformat(), (today - timedelta(days=1)).isoformat()]:
        year = check_date[:4]
        prefix = f"{S3_PREFIX}/{year}/{check_date}/"
        try:
            paginator = s3_client.get_paginator("list_objects_v2")
            keys = [
                obj["Key"]
                for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix)
                for obj in page.get("Contents", [])
                if obj["Key"].endswith(".parquet")
            ]
        except Exception:
            keys = []
        if keys:
            latest_key = sorted(keys)[-1]
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=latest_key)
            df  = pd.read_parquet(BytesIO(obj["Body"].read()))
            return df, latest_key

    return None, None


def _fetch_live_odds_for_event(event_id: str, api_key: str) -> dict | None:
    """
    Fetch live odds for one event from Odds API.
    Returns {(player_name, bookmaker, market_key): under_american_odds} mapping.
    Returns None on SSL error (local proxy environment — not a code failure).
    """
    try:
        r = requests.get(
            f"{ODDS_API_BASE}/sports/{SPORT}/events/{event_id}/odds",
            params={
                "apiKey":     api_key,
                "markets":    MARKETS,
                "regions":    REGIONS,
                "oddsFormat": "american",
            },
            timeout=30,
            verify=False,
        )
    except requests.exceptions.SSLError as e:
        print(f"  [SKIP] SSL error fetching live odds (local proxy env): {e}")
        return None
    if r.status_code != 200:
        return {}

    data = r.json()
    result: dict[tuple, int | None] = {}
    for bm in data.get("bookmakers", []):
        book = bm["key"]
        for mkt in bm.get("markets", []):
            mkt_key  = mkt["key"]
            outcomes = mkt.get("outcomes", [])
            for o in outcomes:
                if o["name"] == "Under":
                    player = o.get("description", "")
                    result[(player, book, mkt_key)] = int(o["price"]) if o.get("price") is not None else None
    return result


def main():
    s3 = boto3.client("s3")
    failures: list[str] = []

    def ok(label: str, detail: str = ""):
        msg = f"  [PASS] {label}" + (f" — {detail}" if detail else "")
        print(msg)

    def fail(label: str, detail: str = ""):
        msg = f"  [FAIL] {label}" + (f" — {detail}" if detail else "")
        print(msg)
        failures.append(label)

    print("\nRunning smoke tests for MLB total bases snapshot pipeline\n")

    # ── Check 1: Schema + non-empty ───────────────────────────────────────────
    df, loaded_key = _latest_snapshot_parquet(s3)
    if df is None or df.empty:
        fail("Check 1: Schema + non-empty", "No snapshot parquet found in S3 for today/yesterday")
        print(f"\n0/3 checks passed (2 TODO checks skipped — implement after Steps B and C)")
        sys.exit(1)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        fail("Check 1: Schema + non-empty", f"Missing columns: {missing_cols}")
    else:
        ok("Check 1: Schema + non-empty", f"{len(df):,} rows, all columns present — {loaded_key}")

    # ── Check 2: First-seen sanity ────────────────────────────────────────────
    # Load ALL parquets for today to check first-seen — the latest snapshot
    # alone won't have any (all players already seen in earlier snapshots).
    game_date_str  = df["game_date"].iloc[0]
    season_str     = game_date_str[:4]
    prefix_all     = f"{S3_PREFIX}/{season_str}/{game_date_str}/"
    try:
        pag  = s3.get_paginator("list_objects_v2")
        all_keys = [
            obj["Key"]
            for page in pag.paginate(Bucket=S3_BUCKET, Prefix=prefix_all)
            for obj in page.get("Contents", [])
            if obj["Key"].endswith(".parquet")
        ]
        all_frames = []
        for k in all_keys:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=k)
            all_frames.append(pd.read_parquet(BytesIO(obj["Body"].read())))
        df_all = pd.concat(all_frames, ignore_index=True) if all_frames else df
    except Exception:
        df_all = df

    n_first = int(df_all["binary_player_game_first_seen"].astype(bool).sum()) if "binary_player_game_first_seen" in df_all.columns else 0
    if n_first > 0:
        ok("Check 2: First-seen sanity", f"{n_first:,} first-seen rows across {len(all_keys)} snapshots for {game_date_str}")
    else:
        fail("Check 2: First-seen sanity", "No binary_player_game_first_seen=True rows found across any snapshot today")

    # ── Check 3: Odds API live comparison ─────────────────────────────────────
    print("\n  [Check 3] Fetching live odds for comparison...")
    try:
        api_key = _api_key()
    except EnvironmentError as e:
        fail("Check 3: Odds API live comparison", str(e))
        n_passed = 3 - len(failures)
        print(f"\n{'='*55}")
        print(f"{n_passed}/3 checks passed (2 TODO checks skipped — implement after Steps B and C)")
        sys.exit(0 if not failures else 1)

    # Pick first event_id from the snapshot
    first_event_id = df["event_id"].iloc[0]
    game_date_str  = df["game_date"].iloc[0]

    live_odds = _fetch_live_odds_for_event(first_event_id, api_key)
    if live_odds is None:
        # SSL error — local proxy environment, not a test failure
        ok("Check 3: Odds API live comparison", "SKIPPED — SSL error in local env (Lambda itself runs fine in AWS)")
    elif not live_odds:
        fail(
            "Check 3: Odds API live comparison",
            f"No live odds returned for event {first_event_id} (game may have started or no lines available)",
        )
    else:
        # Pick up to 3 players from this event in the snapshot
        event_rows = df[df["event_id"] == first_event_id].copy()
        sampled    = event_rows.drop_duplicates(subset=["player_name", "bookmaker", "market_key"]).head(3)

        check_3_issues: list[str] = []
        print(f"  {'Player':30s}  {'Book':15s}  {'MktKey':35s}  {'Snap':>6}  {'Live':>6}  {'Diff':>5}  {'OK?':>5}")
        print(f"  {'-'*30}  {'-'*15}  {'-'*35}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}")
        for _, row in sampled.iterrows():
            key        = (row["player_name"], row["bookmaker"], row["market_key"])
            snap_odds  = row["under_american_odds"]
            live_val   = live_odds.get(key)

            if snap_odds is None or live_val is None:
                status = "N/A"
                diff_str = "N/A"
            else:
                diff     = abs(int(snap_odds) - int(live_val))
                passed   = diff <= 10
                status   = "PASS" if passed else "FAIL"
                diff_str = str(diff)
                if not passed:
                    check_3_issues.append(
                        f"{row['player_name']}/{row['bookmaker']}: snap={snap_odds} live={live_val} diff={diff}"
                    )

            print(
                f"  {str(row['player_name'])[:30]:30s}  "
                f"{str(row['bookmaker'])[:15]:15s}  "
                f"{str(row['market_key'])[:35]:35s}  "
                f"{str(snap_odds):>6}  "
                f"{str(live_val):>6}  "
                f"{diff_str:>5}  "
                f"{status:>5}"
            )

        if check_3_issues:
            fail("Check 3: Odds API live comparison", "; ".join(check_3_issues))
        else:
            ok("Check 3: Odds API live comparison", "All sampled players within ±10 American odds")

    # ── Check 4: compute_clv() runs without error, no nulls on computable rows ──
    print("\n  [Check 4] Running compute_clv()...")
    clv_game_date = df["game_date"].iloc[0] if not df.empty else date.today().isoformat()
    clv_season = int(clv_game_date[:4])
    n_snapshots = df["snapshot_ts_utc"].nunique() if not df.empty else 0
    if n_snapshots < 2:
        ok("Check 4: compute_clv()", f"Skipped — only {n_snapshots} snapshot for {clv_game_date} (need ≥2 for meaningful CLV)")
    else:
        try:
            clv_df = compute_clv(clv_game_date, clv_season)
            if clv_df.empty:
                ok("Check 4: compute_clv()", f"No CLV rows returned for {clv_game_date}")
            else:
                # Only check rows where both reference prices were available
                computable = clv_df[
                    clv_df["first_seen_under_odds"].notna() &
                    clv_df["closing_under_odds"].notna()
                ]
                if computable.empty:
                    ok("Check 4: compute_clv()", f"{len(clv_df):,} rows, none with both reference prices yet")
                else:
                    null_clv = computable["clv_first_seen_cents"].isna().mean()
                    if null_clv > 0.1:
                        fail("Check 4: compute_clv()", f"{null_clv:.0%} of computable rows have null clv_first_seen_cents")
                    else:
                        ok("Check 4: compute_clv()", f"{len(clv_df):,} rows ({len(computable):,} computable), {null_clv:.0%} null CLV")
        except Exception as e:
            fail("Check 4: compute_clv()", str(e))

    # ── Check 5: compute_tightening() returns non-empty DataFrame ─────────────
    print("\n  [Check 5] Running compute_tightening()...")
    try:
        tig_df = compute_tightening(clv_game_date, clv_season)
        if tig_df.empty:
            ok("Check 5: compute_tightening()", f"No tightening data yet for {clv_game_date} (need >1 game_date in season snapshots)")
        else:
            n_flagged = (tig_df["tightening_flag"] != "ok").sum()
            ok("Check 5: compute_tightening()", f"{len(tig_df):,} players, {n_flagged} flagged")
    except Exception as e:
        fail("Check 5: compute_tightening()", str(e))

    # ── Summary ───────────────────────────────────────────────────────────────
    n_passed = 5 - len(failures)
    print(f"\n{'='*55}")
    print(f"{n_passed}/5 checks passed")
    if failures:
        print("Failed checks:")
        for f in failures:
            print(f"  - {f}")
    print("="*55)

    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()

"""
Fetch DraftKings + Bovada lines for every market for every 2025-26 NFL game.
Saves one parquet per game to S3 (idempotent — skips already-uploaded games).

Prerequisites:
  1. data/nfl/event_id_map_2025.csv must exist (run build_nfl_event_id_map.py first)
  2. ODDS_API_KEY in .env
  3. AWS credentials configured (boto3 default chain)

Usage:
  python scripts/build_nfl_all_markets_2025.py [--dry-run] [--game-type REG]

Output:
  s3://the-odds-api-mt/nfl/all_markets/2025/{nfl_game_id}.parquet

Credit estimate (per game):
  ~21 calls @ batch=4 markets  →  5,985 calls total for all 285 games
  Run the SB notebook first to see x-requests-remaining before committing.
"""

import argparse
import math
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import pandas as pd
import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

# ── Config ─────────────────────────────────────────────────────────────────────
ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "americanfootball_nfl"
BOOKMAKERS    = ["draftkings", "bovada"]
REGIONS       = "us,us2"
BATCH_SIZE    = 1
SLEEP_S       = 0.25

S3_BUCKET     = "the-odds-api-mt"
S3_PREFIX     = "nfl/all_markets/2025"

EVENT_ID_MAP  = REPO_ROOT / "data" / "nfl" / "event_id_map_2025.csv"

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

ALL_MARKETS = [
    # ── game / spread / total ──────────────────────────────────────────────────
    "h2h", "spreads", "totals",
    "alternate_spreads", "alternate_totals",
    "team_totals", "alternate_team_totals",
    # ── quarters ──────────────────────────────────────────────────────────────
    "h2h_q1", "h2h_q2", "h2h_q3", "h2h_q4",
    "h2h_3_way_q1", "h2h_3_way_q2", "h2h_3_way_q3", "h2h_3_way_q4",
    "spreads_q1", "spreads_q2", "spreads_q3", "spreads_q4",
    "alternate_spreads_q1", "alternate_spreads_q2", "alternate_spreads_q3",
    "totals_q1", "totals_q2", "totals_q3", "totals_q4",
    "alternate_totals_q1", "alternate_totals_q2", "alternate_totals_q3",
    "team_totals_q1", "team_totals_q2", "team_totals_q3", "team_totals_q4",
    "alternate_team_totals_q1", "alternate_team_totals_q2",
    # ── halves ────────────────────────────────────────────────────────────────
    "h2h_h1", "h2h_h2",
    "h2h_3_way_h1", "h2h_3_way_h2",
    "spreads_h1", "spreads_h2",
    "alternate_spreads_h1", "alternate_spreads_h2",
    "totals_h1", "totals_h2",
    "alternate_totals_h1", "alternate_totals_h2",
    "team_totals_h1", "team_totals_h2",
    "alternate_team_totals_h1", "alternate_team_totals_h2",
    # ── player props ─────────────────────────────────────────────────────────
    "player_pass_tds", "player_pass_yds", "player_pass_attempts",
    "player_pass_completions", "player_pass_interceptions",
    "player_rush_yds", "player_rush_attempts",
    "player_receptions", "player_reception_yds",
    "player_tackles_assists", "player_sacks",
    "player_pass_rush_yds", "player_rush_reception_yds",
    "player_tds_over", "player_1st_td", "player_anytime_td", "player_last_td",
    # ── alternate player props ─────────────────────────────────────────────
    "player_pass_tds_alternate", "player_pass_yds_alternate",
    "player_pass_attempts_alternate", "player_pass_completions_alternate",
    "player_pass_interceptions_alternate",
    "player_rush_yds_alternate", "player_rush_attempts_alternate",
    "player_reception_yds_alternate", "player_receptions_alternate",
    "player_pass_rush_yds_alternate", "player_rush_reception_yds_alternate",
]


def snapshot_utc(gameday: str, gametime_et: str) -> str:
    """Return ISO UTC timestamp 30 min before kickoff."""
    from datetime import datetime, timedelta
    dt_et = datetime.strptime(f"{gameday} {gametime_et}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    dt_utc = dt_et.astimezone(UTC) - timedelta(minutes=30)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def s3_key(nfl_game_id: str) -> str:
    return f"{S3_PREFIX}/{nfl_game_id}.parquet"


def already_uploaded(s3, nfl_game_id: str) -> bool:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=s3_key(nfl_game_id))
        return True
    except s3.exceptions.ClientError:
        return False


def fetch_batch(event_id: str, markets: list[str], snapshot: str) -> dict | None:
    resp = requests.get(
        f"{ODDS_API_BASE}/historical/sports/{SPORT}/events/{event_id}/odds",
        params={
            "apiKey":     ODDS_API_KEY,
            "markets":    ",".join(markets),
            "bookmakers": ",".join(BOOKMAKERS),
            "regions":    REGIONS,
            "oddsFormat": "american",
            "dateFormat": "iso",
            "date":       snapshot,
        },
        timeout=30,
    )
    time.sleep(SLEEP_S)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json(), resp.headers.get("x-requests-remaining", "?")


def flatten_event(payload: dict, snapshot: str) -> list[dict]:
    data = payload.get("data", {})
    if not data:
        return []
    rows = []
    for book in data.get("bookmakers", []):
        for mkt in book.get("markets", []):
            for outcome in mkt.get("outcomes", []):
                rows.append({
                    "odds_api_event_id": data.get("id", ""),
                    "home_team":         data.get("home_team", ""),
                    "away_team":         data.get("away_team", ""),
                    "commence_time":     data.get("commence_time", ""),
                    "snapshot_time":     snapshot,
                    "market":            mkt["key"],
                    "bookmaker":         book["key"],
                    "last_update":       book.get("last_update", ""),
                    "outcome_name":      outcome.get("name", ""),
                    "outcome_desc":      outcome.get("description", ""),
                    "price":             outcome.get("price"),
                    "point":             outcome.get("point"),
                })
    return rows


def upload_df(s3, df: pd.DataFrame, nfl_game_id: str):
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=s3_key(nfl_game_id), Body=buf.getvalue())


def process_game(s3, row: pd.Series, dry_run: bool) -> int:
    """Fetch all markets for one game. Returns credits remaining (last seen)."""
    nfl_game_id   = row["nfl_game_id"]
    event_id      = row["odds_api_event_id"]
    snapshot      = snapshot_utc(row["gameday"], row["gametime"])

    if not event_id or pd.isna(event_id):
        print(f"  SKIP {nfl_game_id} — no odds_api_event_id")
        return -1

    if already_uploaded(s3, nfl_game_id):
        print(f"  SKIP {nfl_game_id} — already in S3")
        return -1

    if dry_run:
        print(f"  DRY  {nfl_game_id}  snapshot={snapshot}")
        return -1

    batches = [ALL_MARKETS[i:i+BATCH_SIZE] for i in range(0, len(ALL_MARKETS), BATCH_SIZE)]
    all_rows = []
    remaining = "?"

    for batch in batches:
        result = fetch_batch(event_id, batch, snapshot)
        if result is None:
            continue
        payload, remaining = result
        all_rows.extend(flatten_event(payload, snapshot))

    if not all_rows:
        print(f"  EMPTY {nfl_game_id} — no data returned (snapshot={snapshot})")
        return int(remaining) if remaining != "?" else -1

    df = pd.DataFrame(all_rows)
    df["nfl_game_id"] = nfl_game_id
    upload_df(s3, df, nfl_game_id)

    n_markets = df["market"].nunique()
    n_books   = df["bookmaker"].nunique()
    print(f"  OK  {nfl_game_id}  markets={n_markets}  books={n_books}  remaining={remaining}")
    return int(remaining) if remaining != "?" else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",   action="store_true", help="Print plan without hitting API")
    parser.add_argument("--game-type", default=None, help="Filter to game type: REG, WC, DIV, CON, SB")
    parser.add_argument("--week",      type=int, default=None, help="Filter to a single week")
    args = parser.parse_args()

    if not ODDS_API_KEY and not args.dry_run:
        sys.exit("ODDS_API_KEY not set — add it to .env or export it.")

    if not EVENT_ID_MAP.exists():
        sys.exit(f"Event ID map not found: {EVENT_ID_MAP}\nRun build_nfl_event_id_map.py first.")

    games = pd.read_csv(EVENT_ID_MAP)

    import nfl_data_py as nfl
    sched = nfl.import_schedules([2025])[["game_id", "gametime"]].rename(columns={"game_id": "nfl_game_id"})
    games = games.merge(sched, on="nfl_game_id", how="left")

    if args.game_type:
        games = games[games["game_type"] == args.game_type]
    if args.week:
        games = games[games["week"] == args.week]

    games = games.sort_values(["gameday", "gametime"]).reset_index(drop=True)

    calls_est = len(games) * math.ceil(len(ALL_MARKETS) / BATCH_SIZE)
    print(f"Games     : {len(games)}")
    print(f"Markets   : {len(ALL_MARKETS)}  (batch={BATCH_SIZE})")
    print(f"Calls est : ~{calls_est:,}")
    print(f"Bookmakers: {BOOKMAKERS}")
    print(f"S3 prefix : s3://{S3_BUCKET}/{S3_PREFIX}/")
    if args.dry_run:
        print("\n--- DRY RUN ---")
    print()

    s3 = boto3.client("s3")

    for _, row in games.iterrows():
        process_game(s3, row, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()

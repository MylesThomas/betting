"""Fetch all available market odds for Super Bowl LX (Feb 8 2026) from The Odds API."""
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv("/Users/thomasmyles/dev/betting/.env")

API_KEY  = os.environ["ODDS_API_KEY"]
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT    = "americanfootball_nfl"
REGIONS  = "us,us2"
SNAPSHOT = "2026-02-08T23:00:00Z"
OUT_DIR  = Path("/Users/thomasmyles/dev/betting/notebooks/data/nfl/sb_lx_markets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_NFL_MARKETS = [
    "h2h", "spreads", "totals",
    "alternate_spreads", "alternate_totals", "team_totals", "alternate_team_totals",
    "h2h_q1", "h2h_q2", "h2h_q3", "h2h_q4",
    "h2h_3_way_q1", "h2h_3_way_q2", "h2h_3_way_q3", "h2h_3_way_q4",
    "spreads_q1", "spreads_q2", "spreads_q3", "spreads_q4",
    "alternate_spreads_q1", "alternate_spreads_q2", "alternate_spreads_q3", "alternate_spreads_q4",
    "totals_q1", "totals_q2", "totals_q3", "totals_q4",
    "alternate_totals_q1", "alternate_totals_q2", "alternate_totals_q3", "alternate_totals_q4",
    "team_totals_q1", "team_totals_q2", "team_totals_q3", "team_totals_q4",
    "alternate_team_totals_q1", "alternate_team_totals_q2", "alternate_team_totals_q3", "alternate_team_totals_q4",
    "h2h_h1", "h2h_h2", "h2h_3_way_h1", "h2h_3_way_h2",
    "spreads_h1", "spreads_h2", "alternate_spreads_h1", "alternate_spreads_h2",
    "totals_h1", "totals_h2", "alternate_totals_h1", "alternate_totals_h2",
    "team_totals_h1", "team_totals_h2", "alternate_team_totals_h1", "alternate_team_totals_h2",
    "player_pass_tds", "player_pass_yds", "player_pass_yds_q1",
    "player_pass_attempts", "player_pass_completions", "player_pass_interceptions",
    "player_pass_longest_completion",
    "player_rush_tds", "player_rush_yds", "player_rush_attempts", "player_rush_longest",
    "player_receptions", "player_reception_tds", "player_reception_yds", "player_reception_longest",
    "player_tackles_assists", "player_solo_tackles", "player_sacks", "player_defensive_interceptions",
    "player_field_goals", "player_kicking_points", "player_pats",
    "player_pass_rush_yds", "player_rush_reception_yds",
    "player_pass_rush_reception_yds", "player_rush_reception_tds", "player_pass_rush_reception_tds",
    "player_tds_over", "player_1st_td", "player_anytime_td", "player_last_td",
    "player_pass_tds_alternate", "player_pass_yds_alternate",
    "player_pass_attempts_alternate", "player_pass_completions_alternate",
    "player_pass_interceptions_alternate", "player_pass_longest_completion_alternate",
    "player_rush_tds_alternate", "player_rush_yds_alternate",
    "player_rush_attempts_alternate", "player_rush_longest_alternate",
    "player_reception_tds_alternate", "player_reception_yds_alternate",
    "player_receptions_alternate", "player_reception_longest_alternate",
    "player_tackles_assists_alternate", "player_solo_tackles_alternate",
    "player_sacks_alternate", "player_field_goals_alternate",
    "player_kicking_points_alternate", "player_pats_alternate",
    "player_pass_rush_yds_alternate", "player_rush_reception_yds_alternate",
    "player_pass_rush_reception_yds_alternate", "player_rush_reception_tds_alternate",
    "player_pass_rush_reception_tds_alternate",
]


def remaining(resp):
    return resp.headers.get("x-requests-remaining", "?")


def get(url, params):
    resp = requests.get(url, params=params, timeout=30)
    time.sleep(0.2)
    return resp


# ── Step 1: find event ────────────────────────────────────────────────────────
print("Step 1: finding Super Bowl LX event...")
resp = get(
    f"{BASE_URL}/historical/sports/{SPORT}/odds",
    {"apiKey": API_KEY, "markets": "h2h", "regions": REGIONS,
     "oddsFormat": "american", "dateFormat": "iso", "date": SNAPSHOT},
)
print(f"  status={resp.status_code}  remaining={remaining(resp)}")
if resp.status_code != 200:
    print(f"  ERROR: {resp.text}")
    raise SystemExit(1)

events = resp.json().get("data", [])
print(f"  {len(events)} event(s) in snapshot:")
for e in events:
    print(f"    {e['home_team']} vs {e['away_team']}  |  {e['commence_time']}")

sb = next(
    (e for e in events
     if "Seattle" in (e["home_team"] + e["away_team"])
     or "New England" in (e["home_team"] + e["away_team"])),
    None,
)
if not sb:
    print("ERROR: Super Bowl not found in snapshot")
    raise SystemExit(1)

EVENT_ID = sb["id"]
print(f"  Found: {sb['home_team']} vs {sb['away_team']}  id={EVENT_ID}\n")


# ── Step 2: probe available markets ──────────────────────────────────────────
def fetch_event(event_id, markets):
    return get(
        f"{BASE_URL}/historical/sports/{SPORT}/events/{event_id}/odds",
        {"apiKey": API_KEY, "markets": ",".join(markets), "regions": REGIONS,
         "oddsFormat": "american", "dateFormat": "iso", "date": SNAPSHOT},
    )


print(f"Step 2: probing {len(ALL_NFL_MARKETS)} markets in batches of 4...")
available = []
for i in range(0, len(ALL_NFL_MARKETS), 4):
    batch = ALL_NFL_MARKETS[i:i+4]
    resp = fetch_event(EVENT_ID, batch)
    label = ",".join(batch[:2]) + ",..."
    print(f"  [{label}] status={resp.status_code}  remaining={remaining(resp)}")
    if resp.status_code != 200:
        print(f"    ERROR: {resp.text[:200]}")
        continue
    event_data = resp.json().get("data", {})
    found = {mkt["key"] for bk in event_data.get("bookmakers", []) for mkt in bk.get("markets", [])}
    for k in batch:
        if k in found:
            available.append(k)

print(f"\nAvailable markets ({len(available)}):")
for m in available:
    print(f"  {m}")
print()


# ── Step 3: full fetch, one per market ────────────────────────────────────────
print(f"Step 3: fetching {len(available)} markets individually...")
fetched = {}
for i, market in enumerate(available, 1):
    out_path = OUT_DIR / f"{market}.json"
    if out_path.exists():
        print(f"  [{i}/{len(available)}] {market} — cached")
        fetched[market] = json.loads(out_path.read_text())
        continue
    resp = fetch_event(EVENT_ID, [market])
    print(f"  [{i}/{len(available)}] {market} → {resp.status_code}  remaining={remaining(resp)}")
    if resp.status_code != 200:
        print(f"    ERROR: {resp.text[:200]}")
        continue
    payload = resp.json()
    out_path.write_text(json.dumps(payload, indent=2))
    fetched[market] = payload

print(f"\nSaved {len(fetched)} market files to {OUT_DIR}")


# ── Step 4: flatten to parquet ────────────────────────────────────────────────
print("\nStep 4: flattening to parquet...")
import pandas as pd

rows = []
for market, payload in fetched.items():
    event_data = payload.get("data", {})
    if not event_data:
        continue
    home_team   = event_data.get("home_team", "")
    away_team   = event_data.get("away_team", "")
    commence_ts = event_data.get("commence_time", "")
    snapshot_ts = payload.get("timestamp", SNAPSHOT)
    for bookmaker in event_data.get("bookmakers", []):
        book_key   = bookmaker["key"]
        book_title = bookmaker["title"]
        for mkt in bookmaker.get("markets", []):
            for outcome in mkt.get("outcomes", []):
                rows.append({
                    "event_id":        event_data.get("id", ""),
                    "home_team":       home_team,
                    "away_team":       away_team,
                    "commence_time":   commence_ts,
                    "snapshot_time":   snapshot_ts,
                    "market":          mkt["key"],
                    "bookmaker":       book_key,
                    "bookmaker_title": book_title,
                    "market_update":   mkt.get("last_update", ""),
                    "outcome_name":    outcome.get("name", ""),
                    "outcome_desc":    outcome.get("description", ""),
                    "price":           outcome.get("price"),
                    "point":           outcome.get("point"),
                    "sid":             outcome.get("sid", ""),
                })

df = pd.DataFrame(rows)
out_parquet = OUT_DIR / "sb_lx_all_markets.parquet"
df.to_parquet(out_parquet, index=False)

print(f"Rows      : {len(df):,}")
print(f"Markets   : {df['market'].nunique()}")
print(f"Bookmakers: {df['bookmaker'].nunique()}")
print(f"Saved     : {out_parquet}")
print()
print(df.groupby("market")[["bookmaker"]].nunique().rename(columns={"bookmaker": "n_books"}).sort_values("n_books", ascending=False).to_string())

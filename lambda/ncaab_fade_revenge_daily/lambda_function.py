"""
NCAAB Fade Revenge Spot – daily Lambda (9am ET).

Run time: ~9am ET.
1. Run fetch scripts for yesterday only (lines + game results to S3).
2. Load outcomes + lines from S3 (season start through yesterday), join, normalize.
3. Find rematch spots (team 0-N vs opponent); focal = rematch team we bet on (when focal is away only).
4. Get today's games from ESPN scoreboard; optionally enrich with Odds API spread; write plays to S3; send SNS.

Plays CSV and email include conference for each team (home_conference, away_conference).

Env: ODDS_API_KEY (optional, for spread in email/CSV), SNS_TOPIC_ARN (optional). IAM: S3 read/write, SNS Publish.
"""

import os
import sys
import subprocess
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from io import StringIO

import boto3
import pandas as pd
import requests

# Package root = directory containing this file (Lambda zip layout)
PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

try:
    from src.ncaab_team_name_mapping import ODDS_API_TO_ESPN_NCAAB
except ImportError:
    ODDS_API_TO_ESPN_NCAAB = {}

try:
    from src.ncaab_conference_data import NCAAB_CONFERENCE_MAPPING_2025_26
except ImportError:
    NCAAB_CONFERENCE_MAPPING_2025_26 = {}

ET = ZoneInfo('America/New_York')
LOG = logging.getLogger(__name__)

# S3
BUCKET = 'ncaab-betting-mt'
OUTCOMES_PREFIX = 'data/01_input/historical_game_results/'
LINES_PREFIX = 'data/01_input/the-odds-api/ncaab/game_lines/'
PLAYS_PREFIX = 'data/04_output/plays/fade-revenge-spot/'

# Season
def _get_current_ncaab_season() -> str:
    now = datetime.now(ET)
    if now.month >= 11:
        return f"{now.year}-{str(now.year + 1)[-2:]}"
    return f"{now.year - 1}-{str(now.year)[-2:]}"


def _season_start(season: str) -> str:
    starts = {
        '2025-26': '2025-11-03',
        '2024-25': '2024-11-03',
        '2023-24': '2023-11-06',
        '2022-23': '2022-11-07',
        '2021-22': '2021-11-09',
        '2020-21': '2020-11-25',
    }
    return starts.get(season, '2025-11-03')


ODDS_TO_ESPN_LOWER = {k.lower(): v for k, v in ODDS_API_TO_ESPN_NCAAB.items()}


def _odds_to_espn(name: str) -> str:
    if pd.isna(name):
        return ""
    key = str(name).lower().strip()
    return ODDS_TO_ESPN_LOWER.get(key, name)


def _conference(team_name: str) -> str:
    t = str(team_name).strip()
    return NCAAB_CONFERENCE_MAPPING_2025_26.get(t, "")


# -----------------------------------------------------------------------------
# Step 1: Run fetch scripts for yesterday (outcomes + lines) and today (lines only)
# -----------------------------------------------------------------------------

def run_fetch_scripts(yesterday: str, today: str, season: str) -> tuple[bool, str]:
    """
    Run fetch scripts so S3 has the data we need:
    - Yesterday: game lines + game results (ESPN). Used for backtest/joined data.
    - Today: game lines only. Written so load_today_lines_from_s3 finds today's file.
    """
    env = os.environ.copy()
    path_parts = [str(PACKAGE_ROOT), '/opt/python']
    env['PYTHONPATH'] = os.pathsep.join(path_parts)
    api_key = os.environ.get('ODDS_API_KEY', '')
    if api_key:
        env['ODDS_API_KEY'] = api_key
    else:
        LOG.warning("ODDS_API_KEY not set; fetch scripts will fail")
        return False, "ODDS_API_KEY not set"
    scripts_dir = PACKAGE_ROOT / 'scripts'
    lines_script = scripts_dir / 'fetch_historical_ncaab_season_lines.py'
    espn_script = scripts_dir / 'fetch_historical_game_results_espn_api.py'
    if not lines_script.exists() or not espn_script.exists():
        return False, f"Scripts not found: {scripts_dir}"

    all_ok = True

    # Lines for yesterday (historical closing lines)
    cmd_lines_yesterday = [
        sys.executable, str(lines_script),
        '--s3', '--season', season,
        '--start-date', yesterday, '--end-date', yesterday,
    ]
    LOG.info("Running: %s", ' '.join(cmd_lines_yesterday))
    r1 = subprocess.run(cmd_lines_yesterday, cwd=str(PACKAGE_ROOT), env=env, capture_output=True, text=True, timeout=300)
    if r1.returncode != 0:
        LOG.warning("Lines fetch (yesterday) failed: %s", r1.stderr[-500:] if r1.stderr else r1.stdout[-500:])
        all_ok = False

    # Lines for today (so load_today_lines_from_s3 finds s3://.../game_lines/{today}.csv)
    cmd_lines_today = [
        sys.executable, str(lines_script),
        '--s3', '--season', season,
        '--start-date', today, '--end-date', today,
    ]
    LOG.info("Running: %s", ' '.join(cmd_lines_today))
    r_today = subprocess.run(cmd_lines_today, cwd=str(PACKAGE_ROOT), env=env, capture_output=True, text=True, timeout=300)
    if r_today.returncode != 0:
        LOG.warning("Lines fetch (today) failed: %s", r_today.stderr[-500:] if r_today.stderr else r_today.stdout[-500:])
        all_ok = False

    # ESPN: yesterday's game results only (today's games not played yet)
    cmd_espn = [
        sys.executable, str(espn_script),
        '--s3', '--season', season,
        '--start-date', yesterday, '--end-date', yesterday,
        '--sport', 'ncaab',
    ]
    LOG.info("Running: %s", ' '.join(cmd_espn))
    r2 = subprocess.run(cmd_espn, cwd=str(PACKAGE_ROOT), env=env, capture_output=True, text=True, timeout=120)
    if r2.returncode != 0:
        LOG.warning("ESPN fetch failed: %s", r2.stderr[-500:] if r2.stderr else r2.stdout[-500:])
        all_ok = False

    return all_ok, ""


# -----------------------------------------------------------------------------
# Data load from S3
# -----------------------------------------------------------------------------

def _list_s3_keys(client, bucket: str, prefix: str) -> list:
    out = []
    pag = client.get_paginator('list_objects_v2')
    for page in pag.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            out.append(obj['Key'])
    return out


def _read_s3_csv(client, bucket: str, key: str) -> pd.DataFrame | None:
    try:
        r = client.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(StringIO(r['Body'].read().decode('utf-8')))
    except Exception as e:
        LOG.warning("Failed to read s3://%s/%s: %s", bucket, key, e)
        return None


def load_outcomes(s3, start_date: str, end_date: str) -> pd.DataFrame:
    keys = _list_s3_keys(s3, BUCKET, OUTCOMES_PREFIX)
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    dfs = []
    for key in keys:
        try:
            fn = key.split('/')[-1].replace('.csv', '')
            d = pd.to_datetime(fn).date()
            if start <= d <= end:
                df = _read_s3_csv(s3, BUCKET, key)
                if df is not None and not df.empty:
                    dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    out['GAME_DATE'] = pd.to_datetime(out['GAME_DATE']).dt.date
    out = out.drop_duplicates(subset=['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM'])
    return out


def load_lines(s3, start_date: str, end_date: str) -> pd.DataFrame:
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    dates = pd.date_range(start=start, end=end, freq='D')
    dfs = []
    for d in dates:
        date_str = d.strftime('%Y-%m-%d')
        key = f"{LINES_PREFIX}{date_str}.csv"
        df = _read_s3_csv(s3, BUCKET, key)
        if df is not None and not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    if 'date' in out.columns:
        out['date'] = pd.to_datetime(out['date']).dt.date
    out = out.drop_duplicates(subset=['date', 'home_team', 'away_team'])
    return out


# -----------------------------------------------------------------------------
# Join outcomes + lines
# -----------------------------------------------------------------------------

def join_outcomes_lines(outcomes: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty:
        return outcomes
    outcomes = outcomes.copy()
    outcomes['date_key'] = outcomes['GAME_DATE']
    outcomes['home_key'] = outcomes['HOME_TEAM'].astype(str).str.strip()
    outcomes['away_key'] = outcomes['AWAY_TEAM'].astype(str).str.strip()

    if lines.empty:
        outcomes['consensus_spread'] = float('nan')
        return outcomes

    lines = lines.copy()
    lines['date_key'] = pd.to_datetime(lines['date']).dt.date
    lines['home_key'] = lines['home_team'].apply(lambda t: _odds_to_espn(t).strip())
    lines['away_key'] = lines['away_team'].apply(lambda t: _odds_to_espn(t).strip())
    lines = lines[['date_key', 'home_key', 'away_key', 'consensus_spread']].drop_duplicates(
        subset=['date_key', 'home_key', 'away_key']
    )

    joined = outcomes.merge(
        lines,
        on=['date_key', 'home_key', 'away_key'],
        how='left',
        suffixes=('', '_line'),
    )
    if '_line' in joined.columns:
        joined = joined[[c for c in joined.columns if not c.endswith('_line')]]
    return joined


# -----------------------------------------------------------------------------
# Pair history and rematch spots (focal = winless team); key by sorted pair
# -----------------------------------------------------------------------------

def _winner(row) -> str:
    if row['HOME_SCORE'] > row['AWAY_SCORE']:
        return row['HOME_TEAM']
    return row['AWAY_TEAM']


def build_rematch_spots(joined: pd.DataFrame) -> dict:
    """
    For each pair (team_a, team_b) that has played, compute wins per team.
    If one team has 0 wins, the next meeting is a rematch spot: focal = that team (we bet on focal when focal is away).
    Return dict: tuple(sorted([h, a])) -> focal_team.
    """
    if joined.empty or 'HOME_TEAM' not in joined.columns:
        return {}
    df = joined.copy()
    df['pair'] = df.apply(
        lambda r: tuple(sorted([str(r['HOME_TEAM']).strip(), str(r['AWAY_TEAM']).strip()])),
        axis=1,
    )
    df = df.sort_values(['pair', 'GAME_DATE']).reset_index(drop=True)

    rematch = {}
    for pair, grp in df.groupby('pair', sort=False):
        grp = grp.sort_values('GAME_DATE')
        wins = {}
        for _, row in grp.iterrows():
            w = _winner(row)
            wins[w] = wins.get(w, 0) + 1
        # After all games in our data: focal for *next* meeting = team with 0 wins
        teams = list(pair)
        focal = None
        for t in teams:
            if wins.get(t, 0) == 0:
                focal = t
                break
        if focal is not None:
            rematch[pair] = focal
    return rematch


def build_pair_stats(joined: pd.DataFrame) -> dict:
    """
    For each pair (team_a, team_b) that has played, return meetings count and wins per team.
    Return dict: tuple(sorted([h, a])) -> {"meetings": int, "wins": {team: count}}.
    """
    if joined.empty or 'HOME_TEAM' not in joined.columns:
        return {}
    df = joined.copy()
    df['pair'] = df.apply(
        lambda r: tuple(sorted([str(r['HOME_TEAM']).strip(), str(r['AWAY_TEAM']).strip()])),
        axis=1,
    )
    df = df.sort_values(['pair', 'GAME_DATE']).reset_index(drop=True)
    out = {}
    for pair, grp in df.groupby('pair', sort=False):
        grp = grp.sort_values('GAME_DATE')
        wins = {}
        for _, row in grp.iterrows():
            w = _winner(row)
            wins[w] = wins.get(w, 0) + 1
        out[pair] = {"meetings": len(grp), "wins": wins}
    return out


def build_pair_prior_games(joined: pd.DataFrame) -> dict:
    """
    For each pair, return list of prior games (date, home, away, home_score, away_score)
    for debugging / showing why we fade. Key: tuple(sorted([h, a])).
    """
    if joined.empty or "HOME_TEAM" not in joined.columns:
        return {}
    df = joined.copy()
    df["pair"] = df.apply(
        lambda r: tuple(sorted([str(r["HOME_TEAM"]).strip(), str(r["AWAY_TEAM"]).strip()])),
        axis=1,
    )
    df = df.sort_values(["pair", "GAME_DATE"]).reset_index(drop=True)
    out = {}
    for pair, grp in df.groupby("pair", sort=False):
        grp = grp.sort_values("GAME_DATE")
        games = []
        for _, row in grp.iterrows():
            d = row["GAME_DATE"]
            date_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
            h = str(row["HOME_TEAM"]).strip()
            a = str(row["AWAY_TEAM"]).strip()
            hs = int(row["HOME_SCORE"]) if pd.notna(row["HOME_SCORE"]) else 0
            aws = int(row["AWAY_SCORE"]) if pd.notna(row["AWAY_SCORE"]) else 0
            games.append({"date": date_str, "home": h, "away": a, "home_score": hs, "away_score": aws})
        out[pair] = games
    return out


def _format_prior_meetings(prior_games: list[dict]) -> str:
    """Format prior games as e.g. '2026-01-15: Away 72 @ Home 68; 2026-02-01: ...'."""
    if not prior_games:
        return ""
    parts = []
    for g in prior_games:
        parts.append(f"{g['date']}: {g['away']} {g['away_score']} @ {g['home']} {g['home_score']}")
    return "; ".join(parts)


# -----------------------------------------------------------------------------
# Today's games from ESPN scoreboard; Odds API used only to enrich with spread
# -----------------------------------------------------------------------------

ESPN_NCAAB_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
)


def fetch_today_events_espn(target_date_et: str) -> list[dict]:
    """
    Fetch today's NCAAB games from ESPN scoreboard (groups=50 for all D1).
    Returns only events whose start time in ET falls on target_date_et (YYYY-MM-DD).
    Each dict: home_team, away_team (ESPN display names).
    """
    date_str = target_date_et.replace("-", "")
    params = {"dates": date_str, "limit": 500, "groups": "50"}
    try:
        resp = requests.get(ESPN_NCAAB_SCOREBOARD_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        LOG.warning("ESPN scoreboard request failed: %s", e)
        return []
    events = data.get("events") or []
    target = datetime.strptime(target_date_et, "%Y-%m-%d").date()
    out = []
    for event in events:
        date_iso = event.get("date") or ""
        if date_iso:
            try:
                dt_utc = datetime.fromisoformat(date_iso.replace("Z", "+00:00"))
                dt_et = dt_utc.astimezone(ET)
                if dt_et.date() != target:
                    continue
            except Exception:
                continue
        comps = (event.get("competitions") or [{}])[0].get("competitors") or []
        if len(comps) != 2:
            continue
        home_comp = next((c for c in comps if c.get("homeAway") == "home"), None)
        away_comp = next((c for c in comps if c.get("homeAway") == "away"), None)
        if not home_comp or not away_comp:
            continue
        home_name = (home_comp.get("team") or {}).get("displayName") or ""
        away_name = (away_comp.get("team") or {}).get("displayName") or ""
        if not home_name or not away_name:
            continue
        out.append({"home_team": home_name.strip(), "away_team": away_name.strip()})
    return out


def load_today_lines_from_s3(s3, today_et: str) -> tuple[dict, pd.DataFrame]:
    """
    Load today's game lines from S3 (data/01_input/the-odds-api/ncaab/game_lines/{date}.csv).
    Returns (lookup, df): lookup = (home_espn, away_espn) -> consensus_spread; df has per-book *_spread columns.
    If file missing or empty, returns ({}, empty DataFrame).
    """
    key = f"{LINES_PREFIX}{today_et}.csv"
    df = _read_s3_csv(s3, BUCKET, key)
    if df is None or df.empty or "home_team" not in df.columns or "away_team" not in df.columns:
        return {}, pd.DataFrame()
    lookup = {}
    for _, row in df.iterrows():
        home_espn = _odds_to_espn(row.get("home_team") or "").strip()
        away_espn = _odds_to_espn(row.get("away_team") or "").strip()
        if not home_espn or not away_espn:
            continue
        val = row.get("consensus_spread")
        lookup[(home_espn, away_espn)] = None if pd.isna(val) else float(val)
    return lookup, df


def _log_per_book_from_game_lines_row(row: pd.Series) -> None:
    """Print one line per book spread for a game (from game_lines CSV row)."""
    away = (row.get("away_team") or "").strip()
    home = (row.get("home_team") or "").strip()
    cons = row.get("consensus_spread")
    cons_str = f"{float(cons):.2f}" if cons is not None and not pd.isna(cons) else "N/A"
    print(f"  {away} @ {home} (consensus_spread_home {cons_str})")
    spread_cols = [c for c in row.index if str(c).endswith("_spread") and c != "consensus_spread"]
    for col in sorted(spread_cols):
        v = row.get(col)
        if pd.isna(v):
            print(f"    {col}: NULL")
        else:
            print(f"    {col}: {float(v):.2f}")


def fetch_odds_api_spreads(api_key: str) -> dict:
    """
    GET /v4/sports/basketball_ncaab/odds (US). Build lookup (home_espn, away_espn) -> spread_home.
    spread_home = consensus (median) of all books' home spread; rounded to nearest 0.5.
    Fallback when today's game_lines are not in S3.
    """
    import json
    import statistics
    from urllib.parse import urlencode
    import urllib.request
    params = {"regions": "us", "markets": "spreads", "apiKey": api_key}
    url = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds?" + urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = json.loads(resp.read().decode())
    except Exception as e:
        LOG.warning("Odds API request failed: %s", e)
        return {}
    lookup = {}
    print("Odds API per-book home spreads (away @ home; then each book):")
    for event in raw:
        home_odds = (event.get("home_team") or "").strip()
        away_odds = (event.get("away_team") or "").strip()
        if not home_odds or not away_odds:
            continue
        home_espn = _odds_to_espn(home_odds).strip()
        away_espn = _odds_to_espn(away_odds).strip()
        book_spreads = []
        for book in event.get("bookmakers") or []:
            book_key = book.get("key") or book.get("title") or "?"
            for m in book.get("markets") or []:
                if m.get("key") != "spreads":
                    continue
                outcomes = m.get("outcomes") or []
                home_pt = None
                for o in outcomes:
                    if (o.get("name") or "").strip() == home_odds and "point" in o:
                        try:
                            home_pt = float(o["point"])
                            break
                        except (TypeError, ValueError):
                            pass
                if home_pt is None and len(outcomes) == 2 and all("point" in o for o in outcomes):
                    try:
                        home_pt = float(outcomes[0]["point"])
                    except (TypeError, ValueError, KeyError):
                        pass
                if home_pt is not None:
                    book_spreads.append((book_key, home_pt))
                break
        if not book_spreads:
            spread_home = None
        else:
            spreads_home = [pt for _, pt in book_spreads]
            med = statistics.median(spreads_home)
            spread_home = round(med * 2) / 2.0
        lookup[(home_espn, away_espn)] = spread_home
        if book_spreads:
            print(f"  {away_espn} @ {home_espn} (consensus home spread {spread_home})")
            for book_key, pt in book_spreads:
                print(f"    {book_key}: {pt:.1f}")
    return lookup


def log_today_games(
    today_events: list[dict],
    pair_stats: dict,
    rematch_spots: dict,
    odds_lookup: dict | None = None,
) -> None:
    """
    Log one line per today game: meetings, record, line (from Odds API when available), REMATCH or skip.
    today_events from ESPN: each dict has home_team, away_team (ESPN names).
    """
    import sys
    odds_lookup = odds_lookup or {}
    if not today_events:
        print("Today's games: 0 (no events)")
        sys.stdout.flush()
        return
    print(f"Today's games ({len(today_events)}): each line = game | meetings | record | line | REMATCH or skip")
    for i, ev in enumerate(today_events, 1):
        home_espn = (ev.get("home_team") or "").strip()
        away_espn = (ev.get("away_team") or "").strip()
        pair = tuple(sorted([home_espn, away_espn]))
        stats = pair_stats.get(pair)
        if stats is None:
            meetings_str = "0 (first meeting)"
            record_str = "—"
        else:
            n = stats["meetings"]
            wins = stats["wins"]
            wh = wins.get(home_espn, 0)
            wa = wins.get(away_espn, 0)
            meetings_str = str(n)
            record_str = f"{home_espn} {wh}-{wa} {away_espn}"
        lookup_key = (home_espn, away_espn)
        spread = odds_lookup.get(lookup_key)
        line_str = str(spread) if spread is not None else "N/A"
        if i <= 3:
            print(f"  [lookup] key={lookup_key!r} -> line={line_str}")
        focal = rematch_spots.get(pair)
        if focal is not None:
            if focal == away_espn:
                decision = f"REMATCH -> Bet {focal} (focal=revenge team, away)"
            else:
                decision = "REMATCH -> skip (focal home; we only bet focal away)"
        else:
            decision = "NOT A REMATCH -> skip"
        print(
            f"  [{i}/{len(today_events)}] {away_espn} @ {home_espn} | Meetings: {meetings_str} | "
            f"Record: {record_str} | Line: {line_str} | {decision}"
        )
    sys.stdout.flush()


# -----------------------------------------------------------------------------
# Plays: one row per today game; rematch games have focal_team/bet_team filled (filter WHERE bet_team IS NOT NULL)
# -----------------------------------------------------------------------------

def build_today_plays(
    rematch_spots: dict,
    today_events: list[dict],
    pair_stats: dict,
    pair_prior_games: dict | None = None,
    odds_lookup: dict | None = None,
) -> pd.DataFrame:
    pair_prior_games = pair_prior_games or {}
    odds_lookup = odds_lookup or {}
    today_str = datetime.now(ET).strftime("%Y-%m-%d")
    rows = []
    for ev in today_events:
        home_espn = (ev.get("home_team") or "").strip()
        away_espn = (ev.get("away_team") or "").strip()
        pair = tuple(sorted([home_espn, away_espn]))
        stats = pair_stats.get(pair)
        if stats is None:
            meetings_count = 0
            record = ""
        else:
            meetings_count = stats["meetings"]
            wins = stats["wins"]
            wh = wins.get(home_espn, 0)
            wa = wins.get(away_espn, 0)
            record = f"{home_espn} {wh}-{wa} {away_espn}"
        consensus_spread_home = odds_lookup.get((home_espn, away_espn))
        focal = rematch_spots.get(pair)
        if focal is None:
            bet_team = None
            side = None
            prior_meetings_str = None
        else:
            # Strategy filter: only include when focal is away (52.6% ATS segment)
            if focal == away_espn:
                bet_team = focal
                side = "bet revenge (rematch – lost first meeting)"
            else:
                bet_team = None
                side = None
            prior_games = pair_prior_games.get(pair, [])
            prior_meetings_str = _format_prior_meetings(prior_games) if focal is not None else None
        rows.append({
            "game_date": today_str,
            "home_team": home_espn,
            "away_team": away_espn,
            "home_conference": _conference(home_espn) or "",
            "away_conference": _conference(away_espn) or "",
            "meetings_count": meetings_count,
            "record": record,
            "consensus_spread_home": consensus_spread_home if consensus_spread_home is not None else None,
            "focal_team": focal,
            "bet_team": bet_team,
            "side": side,
            "prior_meetings": prior_meetings_str,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Yesterday's results
# -----------------------------------------------------------------------------

def load_yesterday_results(s3, yesterday_et: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = f"{PLAYS_PREFIX}{yesterday_et}.csv"
    try:
        r = s3.get_object(Bucket=BUCKET, Key=key)
        plays = pd.read_csv(StringIO(r['Body'].read().decode('utf-8')))
    except Exception:
        plays = pd.DataFrame()
    outcomes_key = f"{OUTCOMES_PREFIX}{yesterday_et}.csv"
    try:
        r = s3.get_object(Bucket=BUCKET, Key=outcomes_key)
        outcomes = pd.read_csv(StringIO(r['Body'].read().decode('utf-8')))
    except Exception:
        outcomes = pd.DataFrame()
    return plays, outcomes


def evaluate_yesterday_plays(plays: pd.DataFrame, outcomes: pd.DataFrame) -> list[dict]:
    if plays.empty or outcomes.empty:
        return []
    results = []
    for _, row in plays.iterrows():
        home = str(row.get('home_team', '')).strip()
        away = str(row.get('away_team', '')).strip()
        bet_team = row.get('bet_team')
        mask = (
            (outcomes['HOME_TEAM'].astype(str).str.strip() == home) &
            (outcomes['AWAY_TEAM'].astype(str).str.strip() == away)
        )
        if not mask.any():
            results.append({"game": f"{away} @ {home}", "bet_team": bet_team, "result": "no_result"})
            continue
        oc = outcomes.loc[mask].iloc[0]
        hs, aws = int(oc['HOME_SCORE']), int(oc['AWAY_SCORE'])
        winner = (oc['HOME_TEAM'] if hs > aws else oc['AWAY_TEAM'])
        su_win = str(winner).strip() == str(bet_team).strip()
        results.append({
            "game": f"{away} @ {home}",
            "bet_team": bet_team,
            "score": f"{hs}-{aws}",
            "su_win": su_win,
        })
    return results


# -----------------------------------------------------------------------------
# S3 write plays; SNS email
# -----------------------------------------------------------------------------

def write_plays_s3(s3, today_et: str, plays_df: pd.DataFrame) -> str:
    key = f"{PLAYS_PREFIX}{today_et}.csv"
    buf = StringIO()
    columns = [
        "game_date", "home_team", "away_team", "home_conference", "away_conference",
        "meetings_count", "record", "consensus_spread_home",
        "focal_team", "bet_team", "side", "prior_meetings",
    ]
    if plays_df.empty:
        pd.DataFrame(columns=columns).to_csv(buf, index=False)
    else:
        plays_df.to_csv(buf, index=False, columns=columns, na_rep="")
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue(), ContentType="text/csv")
    return f"s3://{BUCKET}/{key}"


def send_sns(sns, topic_arn: str, subject: str, body: str) -> bool:
    if not topic_arn:
        LOG.info("SNS_TOPIC_ARN not set; skipping email")
        return False
    try:
        sns.publish(TopicArn=topic_arn, Subject=subject[:100], Message=body)
        LOG.info("SNS sent: %s", subject[:50])
        return True
    except Exception as e:
        LOG.warning("SNS failed: %s", e)
        return False


# -----------------------------------------------------------------------------
# Lambda handler
# -----------------------------------------------------------------------------

def lambda_handler(event=None, context=None):
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    now = datetime.now(ET)
    today_et = now.strftime("%Y-%m-%d")
    yesterday_et = (now - timedelta(days=1)).strftime("%Y-%m-%d")

    season = _get_current_ncaab_season()
    start_date = _season_start(season)
    print(f"NCAAB Fade Revenge run: season={season} start={start_date} yesterday={yesterday_et} today={today_et}")
    sys.stdout.flush()

    s3 = boto3.client('s3')
    api_key = os.environ.get('ODDS_API_KEY', '')
    sns_topic = os.environ.get('SNS_TOPIC_ARN', '')
    sns_client = boto3.client('sns') if sns_topic else None

    # Step 1: Run fetch scripts for yesterday (outcomes + lines) and today (lines only)
    ok, err = run_fetch_scripts(yesterday_et, today_et, season)
    if not ok and err:
        LOG.warning("Fetch step had issues: %s", err)
    LOG.info("Fetch scripts completed (ok=%s)", ok)

    # Load outcomes and lines (season start through yesterday)
    outcomes = load_outcomes(s3, start_date, yesterday_et)
    lines = load_lines(s3, start_date, yesterday_et)
    print(f"Outcomes: {len(outcomes)} rows; Lines: {len(lines)} rows")
    sys.stdout.flush()

    if outcomes.empty:
        LOG.warning("No outcomes in S3 for %s–%s", start_date, yesterday_et)
        body = f"NCAAB Fade Revenge Spot – {today_et}\n\nNo outcome data for season to date."
        if sns_client and sns_topic:
            send_sns(sns_client, sns_topic, f"NCAAB Fade Revenge – {today_et} (no data)", body)
        return {"status": "no_outcomes", "today_et": today_et}

    joined = join_outcomes_lines(outcomes, lines)
    matched = joined['consensus_spread'].notna().sum() if 'consensus_spread' in joined.columns else 0
    LOG.info("Joined: %s games, %s with spread", len(joined), matched)

    rematch_spots = build_rematch_spots(joined)
    today_events = fetch_today_events_espn(today_et)
    odds_lookup, today_lines_df = load_today_lines_from_s3(s3, today_et)
    if not odds_lookup or today_lines_df.empty:
        key = f"{LINES_PREFIX}{today_et}.csv"
        body = (
            f"NCAAB Fade Revenge Spot – {today_et}\n\n"
            "Something is broken: today's game lines are missing or empty.\n\n"
            f"Expected: s3://{BUCKET}/{key}\n\n"
            "The Lambda fetches today's lines at the start of the run (fetch_historical_ncaab_season_lines.py for today). "
            "If this file is still missing, the fetch step may have failed (check ODDS_API_KEY, API credits, or script errors in logs)."
        )
        if sns_client and sns_topic:
            send_sns(sns_client, sns_topic, f"NCAAB Fade Revenge – {today_et} (broken: no lines)", body)
        return {"status": "no_lines", "today_et": today_et}
    print(f"Rematch spots: {len(rematch_spots)} pairs with a winless team; Today's events (ESPN): {len(today_events)}; Lines: {len(odds_lookup)}")
    print("Game lines from S3 (consensus_spread_home; per-book):")
    for _, row in today_lines_df.iterrows():
        _log_per_book_from_game_lines_row(row)
    sys.stdout.flush()

    pair_stats = build_pair_stats(joined)
    log_today_games(today_events, pair_stats, rematch_spots, odds_lookup)

    pair_prior_games = build_pair_prior_games(joined)
    plays_df = build_today_plays(rematch_spots, today_events, pair_stats, pair_prior_games, odds_lookup)
    plays_path = write_plays_s3(s3, today_et, plays_df)
    rematch_count = (
        (plays_df["bet_team"].notna() & (plays_df["bet_team"].astype(str).str.strip() != "")).sum()
        if not plays_df.empty else 0
    )
    print(f"Wrote plays: {plays_path} ({len(plays_df)} games, {rematch_count} rematch plays; filter WHERE bet_team IS NOT NULL)")
    if not plays_df.empty and rematch_count > 0:
        rematch_rows = plays_df[plays_df["bet_team"].notna() & (plays_df["bet_team"].astype(str).str.strip() != "")]
        with_spread = rematch_rows["consensus_spread_home"].notna().sum()
        print(f"Rematch plays with consensus_spread_home: {with_spread}/{len(rematch_rows)}")
        for _, row in rematch_rows.head(5).iterrows():
            print(f"  {row['away_team']} @ {row['home_team']} -> consensus_spread_home={row.get('consensus_spread_home')}")
    sys.stdout.flush()

    yesterday_plays, yesterday_outcomes = load_yesterday_results(s3, yesterday_et)
    yesterday_results = evaluate_yesterday_plays(yesterday_plays, yesterday_outcomes)

    # Email
    lines_email = [f"NCAAB Fade Revenge Spot – {today_et}", ""]
    if yesterday_results:
        lines_email.append("Yesterday's results:")
        resolved = [r for r in yesterday_results if r.get("result") != "no_result"]
        if resolved:
            wins = sum(1 for r in resolved if r.get("su_win"))
            losses = len(resolved) - wins
            lines_email.append(f"  {yesterday_et}: {wins}-{losses}")
        for r in yesterday_results:
            if r.get("result") == "no_result":
                lines_email.append(f"  {r['game']}: no result")
            else:
                w = "W" if r.get("su_win") else "L"
                lines_email.append(f"  {w} {r['game']} (bet {r['bet_team']}) {r.get('score', '')}")
        lines_email.append("")
    else:
        lines_email.append("Yesterday's results: None.")

    # Only list rows that are actual plays (bet_team not null/empty), sorted A-Z by bet_team
    plays_only = (
        plays_df[plays_df["bet_team"].notna() & (plays_df["bet_team"].astype(str).str.strip() != "")]
        if not plays_df.empty else pd.DataFrame()
    )
    if not plays_only.empty:
        plays_only = plays_only.sort_values("bet_team").reset_index(drop=True)

    # Add divide
    lines_email.append("")
    lines_email.append("--------------")
    lines_email.append("")

    # Continue with today
    lines_email.append("Today's plays (bet revenge – focal away only):")
    if plays_only.empty:
        lines_email.append("  None.")
    else:
        for _, row in plays_only.iterrows():
            hc = row.get("home_conference", "") or ""
            ac = row.get("away_conference", "") or ""
            conf = f" ({ac} @ {hc})" if (hc or ac) else ""
            prior = row.get("prior_meetings", "") or ""
            spread = row.get("consensus_spread_home", None)
            if spread is None or pd.isna(spread):
                line_str = "line NA"
            else:
                bet_line = spread if row["bet_team"] == row["home_team"] else -spread
                line_str = f"{bet_line:+.1f}" if isinstance(bet_line, (int, float)) else str(bet_line)
            lines_email.append(f"  Bet {row['bet_team']} {line_str} (rematch – lost first meeting) – {row['away_team']} @ {row['home_team']}{conf}")
            if prior:
                lines_email.append(f"    Prior: {prior}")
    lines_email.append("")
    lines_email.append(f"Plays CSV: {plays_path}")

    body = "\n".join(lines_email)
    if sns_client and sns_topic:
        send_sns(sns_client, sns_topic, f"NCAAB Bet Revenge (focal away) – {today_et}", body)

    return {
        "status": "ok",
        "today_et": today_et,
        "yesterday_et": yesterday_et,
        "games_count": len(plays_df),
        "plays_count": int(rematch_count),
        "plays_path": plays_path,
        "yesterday_results_count": len(yesterday_results),
    }

"""
Compare NCAAB API coverage: ESPN default vs ESPN groups=50 (all D1).

Context:
- The live odds tracker (lambda_function_track_live_odds.py) calls the ESPN NCAAB
  scoreboard with NO query params. It's unclear if that returns only "top 25" /
  featured games or a broader set.
- The historical game results script (fetch_historical_game_results_espn_api.py)
  uses groups=50 for NCAAB to get "ALL games (not just featured)" / all D1.

This script calls both ESPN variants (and The Odds API for reference) for a given
date and prints game counts so we can see if the live tracker is missing games.

APIs compared:
  1. ESPN NCAAB scoreboard – default (dates only, no groups) – may be top 25 / featured.
  2. ESPN NCAAB scoreboard – groups=50 – all D1 (same as historical fetch).
  3. The Odds API basketball_ncaab/odds – reference count (no top25 vs all param).

Usage (from repo root):
  python lambda/track_live_odds/tmp/compare_ncaab_api_coverage.py
  python lambda/track_live_odds/tmp/compare_ncaab_api_coverage.py --date 20260215
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Project root (find via .gitignore, no sys.path hacks per cursor rules)
def find_project_root():
    current = Path.cwd()
    while current != current.parent:
        if (current / '.gitignore').exists():
            return current
        current = current.parent
    return Path.cwd()

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Suppress SSL warnings if present
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

ESPN_NCAAB_BASE = 'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard'
ODDS_API_BASE = 'https://api.the-odds-api.com/v4'


def get_date_str(days_ago: int = 0) -> str:
    """Return YYYYMMDD in ET for comparison (ESPN uses this format in params)."""
    et = ZoneInfo('America/New_York')
    dt = datetime.now(et) - timedelta(days=days_ago)
    return dt.strftime('%Y%m%d')


def events_to_game_list(data: dict) -> list[tuple[str, str]]:
    """Extract (away_team, home_team) from ESPN scoreboard response."""
    out = []
    for event in data.get('events', []):
        comps = event.get('competitions', [{}])
        if not comps:
            continue
        competitors = comps[0].get('competitors', [])
        away = next((c for c in competitors if c.get('homeAway') == 'away'), None)
        home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
        if away and home:
            away_name = away.get('team', {}).get('displayName', '?')
            home_name = home.get('team', {}).get('displayName', '?')
            out.append((away_name, home_name))
    return out


def call_espn_ncaab_scoreboard(date_str: str, groups: str | None) -> tuple[dict, int]:
    """
    Call ESPN NCAAB scoreboard for one day.
    Returns (raw response dict, number of events).
    """
    params = {'dates': date_str, 'limit': 500}
    if groups is not None:
        params['groups'] = groups
    response = requests.get(ESPN_NCAAB_BASE, params=params, timeout=15, verify=False)
    response.raise_for_status()
    data = response.json()
    events = data.get('events', [])
    return data, len(events)


def call_odds_api_ncaab() -> tuple[list, int]:
    """
    Call The Odds API basketball_ncaab/odds (no date filter; returns current/upcoming).
    Returns (list of games, count).
    """
    api_key = os.getenv('ODDS_API_KEY')
    if not api_key or api_key == 'your_api_key_here':
        return [], 0
    url = f"{ODDS_API_BASE}/sports/basketball_ncaab/odds"
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': 'spreads,h2h',
        'oddsFormat': 'american',
    }
    response = requests.get(url, params=params, timeout=15, verify=False)
    response.raise_for_status()
    games = response.json()
    return games, len(games) if isinstance(games, list) else 0


def main():
    parser = argparse.ArgumentParser(
        description='Compare NCAAB coverage: ESPN default vs ESPN groups=50 (all D1).'
    )
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Date YYYYMMDD (default: today ET)',
    )
    args = parser.parse_args()
    date_str = args.date or get_date_str()

    print('=' * 80)
    print('NCAAB API COVERAGE: Top 25 / featured vs All D1')
    print('=' * 80)
    print(f"Date (ESPN): {date_str}")
    print()

    # 1) ESPN – default (no groups) – what live odds tracker effectively uses with "today"
    print('1. ESPN NCAAB scoreboard – DEFAULT (no groups)')
    print('   (Same style as live odds tracker: no groups param)')
    try:
        data_default, count_default = call_espn_ncaab_scoreboard(date_str, groups=None)
        print(f'   Games returned: {count_default}')
        games_default = events_to_game_list(data_default)
        for i, (away, home) in enumerate(games_default, 1):
            print(f'      {i}. {away} @ {home}')
    except Exception as e:
        print(f'   Error: {e}')
        count_default = None
        games_default = []
    print()

    # 2) ESPN – groups=50 (all D1)
    print('2. ESPN NCAAB scoreboard – groups=50 (all D1)')
    print('   (Same as fetch_historical_game_results_espn_api.py for NCAAB)')
    try:
        data_50, count_50 = call_espn_ncaab_scoreboard(date_str, groups='50')
        print(f'   Games returned: {count_50}')
        games_50 = events_to_game_list(data_50)
        for i, (away, home) in enumerate(games_50, 1):
            print(f'      {i}. {away} @ {home}')
    except Exception as e:
        print(f'   Error: {e}')
        count_50 = None
        games_50 = []
    print()

    # 3) The Odds API – reference
    print('3. The Odds API – basketball_ncaab/odds')
    print('   (No top25 vs all D1 param; current/upcoming games only)')
    try:
        odds_games, count_odds = call_odds_api_ncaab()
        if count_odds == 0 and not os.getenv('ODDS_API_KEY'):
            print('   Skipped (ODDS_API_KEY not set)')
        else:
            print(f'   Games returned: {count_odds}')
            for i, g in enumerate(odds_games if isinstance(odds_games, list) else [], 1):
                away = g.get('away_team', '?')
                home = g.get('home_team', '?')
                print(f'      {i}. {away} @ {home}')
    except Exception as e:
        print(f'   Error: {e}')
        count_odds = None
    print()

    # Summary
    print('=' * 80)
    print('Summary')
    print('=' * 80)
    if count_default is not None and count_50 is not None:
        if count_default < count_50:
            print(
                f"ESPN default returned FEWER games ({count_default}) than groups=50 ({count_50})."
            )
            print(
                "Live odds tracker uses no groups → may only be getting featured/top-25."
            )
        elif count_default == count_50:
            print(
                f"ESPN default and groups=50 both returned {count_default} games."
            )
        else:
            print(
                f"ESPN default ({count_default}) returned more than groups=50 ({count_50})."
            )
    print()


if __name__ == '__main__':
    main()

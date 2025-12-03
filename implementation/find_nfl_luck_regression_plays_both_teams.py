"""
Find NFL Regression Betting Opportunities: BOTH TEAMS LUCK Approach

Based on analysis from 20251202_nfl_both_teams_luck_analysis.py

Core Strategy:
When a LUCKY team (overperformed last week) plays an UNLUCKY team 
(underperformed last week), bet on the UNLUCKY team.

Why it works:
- Lucky teams are due to regress DOWN (their luck won't continue)
- Unlucky teams are due to regress UP (their bad luck won't continue)
- The spread doesn't account for this regression

Luck = score - adj_score (from Unexpected Points data)
- Positive luck = overperformed (scored more than expected)
- Negative luck = underperformed (scored less than expected)

Luck Categories (based on --threshold, default from config.py):
- Lucky: luck >= +threshold
- Neutral: -threshold < luck < +threshold  
- Unlucky: luck <= -threshold

Usage:
    # Find plays for current week (asks before API calls)
    python implementation/find_nfl_luck_regression_plays_both_teams.py --current-week
    python implementation/find_nfl_luck_regression_plays_both_teams.py --current-week --verbose-mode
    
    # Production mode (no prompts)
    python implementation/find_nfl_luck_regression_plays_both_teams.py --current-week --no-safe-mode
    
    # Custom threshold
    python implementation/find_nfl_luck_regression_plays_both_teams.py --current-week --threshold 5
    
    # Test mode (simulates 2025-12-04 with 1 TNF game, no API calls)
    python implementation/find_nfl_luck_regression_plays_both_teams.py --current-week --test
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
import glob
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import ssl
import urllib3

# Load environment variables
load_dotenv()

# Fix SSL certificate issues (for API calls)
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Add src to path by finding project root (look for .gitignore)
def find_project_root() -> Path:
    """Find project root by looking for .gitignore file."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / '.gitignore').exists():
            return parent
    raise FileNotFoundError("Could not find project root (.gitignore not found)")

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from nfl_team_utils import add_team_abbr_columns
from nfl_luck_utils import (
    categorize_luck,
    categorize_spread,
    load_unexpected_points_data,
    build_prior_luck_lookup,
    get_prior_week_luck,
    NFL_LINES_UPCOMING_DIR,
)
from config import NFL_LUCK_THRESHOLD_DEFAULT, EMOJI

# Parse arguments
parser = argparse.ArgumentParser(description='Find NFL regression betting plays using both teams luck')
parser.add_argument('--current-week', action='store_true',
                   help='Find plays for current week (games in next 7 days). Will fetch live lines if needed.')
parser.add_argument('--week', type=int,
                   help='Backtest a specific week (e.g., --week 13). Uses saved data, no API calls.')
parser.add_argument('--threshold', type=float, default=NFL_LUCK_THRESHOLD_DEFAULT,
                   help=f'Luck threshold for categorization (default: {NFL_LUCK_THRESHOLD_DEFAULT})')
parser.add_argument('--verbose-mode', action='store_true',
                   help='Print detailed reasoning for each game')
parser.add_argument('--safe-mode', action='store_true', default=True,
                   help='Ask before making API calls (default: True). Use --no-safe-mode for production.')
parser.add_argument('--no-safe-mode', dest='safe_mode', action='store_false',
                   help='Skip confirmation prompts for API calls (for production use)')
parser.add_argument('--test', action='store_true',
                   help='Test mode: simulate 2025-12-04 with 1 TNF game (DAL @ DET), no API calls')
args = parser.parse_args()

# Must specify either --current-week or --week (not both)
if not args.current_week and args.week is None:
    parser.error("Must specify either --current-week or --week N")
if args.current_week and args.week is not None:
    parser.error("Cannot use both --current-week and --week")

# =============================================================================
# BACKTEST MODE (--week N)
# =============================================================================
if args.week is not None:
    from config import DATA_ROOT
    
    print("=" * 100)
    print(f"BACKTEST: Week {args.week}")
    print("=" * 100)
    
    backtest_file = DATA_ROOT / f"03_intermediate/nfl_both_teams_luck_analysis_threshold_{int(args.threshold)}.csv"
    
    if not backtest_file.exists():
        print(f"\n{EMOJI['error']} File not found: {backtest_file}")
        print(f"   Run: python backtesting/20251202_nfl_both_teams_luck_analysis.py --threshold {int(args.threshold)}")
        sys.exit(1)
    
    df = pd.read_csv(backtest_file)
    df['game_time'] = pd.to_datetime(df['game_time'])
    df_week = df[df['week'] == args.week].copy()
    
    # Add game_time_et column
    from zoneinfo import ZoneInfo
    df_week['game_time_et'] = df_week['game_time'].dt.tz_convert('America/New_York').dt.strftime('%a %Y-%m-%d %H:%M ET')
    
    if len(df_week) == 0:
        print(f"\n{EMOJI['error']} No games for Week {args.week}")
        print(f"   Available weeks: {sorted(df['week'].unique())}")
        sys.exit(1)
    
    print(f"\n{EMOJI['success']} {len(df_week)} games | Threshold ±{args.threshold}")
    
    # Find Lucky vs Unlucky matchups
    df_week['is_lu'] = (
        ((df_week['away_luck_cat'] == 'Lucky') & (df_week['home_luck_cat'] == 'Unlucky')) |
        ((df_week['away_luck_cat'] == 'Unlucky') & (df_week['home_luck_cat'] == 'Lucky'))
    )
    
    # Add bet columns
    df_week['bet_team'] = 'n/a'
    df_week['bet_spread'] = 'n/a'
    df_week['bet_reason'] = 'n/a'
    
    for idx, row in df_week.iterrows():
        if row['is_lu']:
            if row['away_luck_cat'] == 'Unlucky':
                df_week.at[idx, 'bet_team'] = row['away_abbr']
                df_week.at[idx, 'bet_spread'] = row['consensus_spread']
                df_week.at[idx, 'bet_reason'] = f"{row['away_abbr']} unlucky ({row['away_prior_luck']:+.1f}), {row['home_abbr']} lucky ({row['home_prior_luck']:+.1f})"
            else:
                df_week.at[idx, 'bet_team'] = row['home_abbr']
                df_week.at[idx, 'bet_spread'] = -row['consensus_spread']
                df_week.at[idx, 'bet_reason'] = f"{row['home_abbr']} unlucky ({row['home_prior_luck']:+.1f}), {row['away_abbr']} lucky ({row['away_prior_luck']:+.1f})"
    
    print(f"\n{EMOJI['chart']} Games:")
    for _, g in df_week.iterrows():
        lu = EMOJI['star'] if g['is_lu'] else "  "
        print(f"  {lu} {g['away_abbr']:>4} ({g['away_prior_luck']:+.1f}) @ {g['home_abbr']:<4} ({g['home_prior_luck']:+.1f}) | {g['matchup_type']}")
    
    df_plays = df_week[df_week['is_lu']]
    print(f"\n{EMOJI['target']} Lucky vs Unlucky: {len(df_plays)}")
    
    if len(df_plays) > 0:
        print("\nBET UNLUCKY TEAM:")
        wins = 0
        for _, p in df_plays.iterrows():
            if p['away_luck_cat'] == 'Unlucky':
                bet, spread, covered = p['away_abbr'], p['consensus_spread'], p['away_covered']
            else:
                bet, spread, covered = p['home_abbr'], -p['consensus_spread'], p['home_covered']
            
            result = EMOJI['success'] if covered else EMOJI['error']
            if covered:
                wins += 1
            print(f"  {result} {bet} {spread:+.1f}")
        
        print(f"\nRecord: {wins}-{len(df_plays)-wins}")
    
    # Save backtest results
    output_dir = Path(f"/Users/thomasmyles/dev/betting/data/04_output/nfl/2025/plays/week_{args.week}")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    threshold_str = f"threshold_{int(args.threshold)}"
    
    # Save all games
    all_games_path = output_dir / f"all_games_{threshold_str}_{timestamp}.csv"
    df_week.to_csv(all_games_path, index=False)
    print(f"\n{EMOJI['save']} Saved all {len(df_week)} games to: {all_games_path}")
    
    # Save plays only
    if len(df_plays) > 0:
        plays_path = output_dir / f"plays_{threshold_str}_{timestamp}.csv"
        df_plays.to_csv(plays_path, index=False)
        print(f"{EMOJI['save']} Saved {len(df_plays)} plays to: {plays_path}")
    
    sys.exit(0)

# Test mode configuration
TEST_MODE = args.test
if TEST_MODE:
    # Simulate running on Thursday 2025-12-04 at 6pm ET (only 1 TNF game to save API credits)
    TEST_DATE = datetime(2025, 12, 4, 23, 0, 0, tzinfo=timezone.utc)  # 6pm ET = 11pm UTC

threshold = args.threshold

print("=" * 100)
print("NFL REGRESSION BETTING PLAYS: BOTH TEAMS LUCK APPROACH")
print("=" * 100)
print(f"\nLuck Threshold: ±{threshold}")
print(f"  Lucky: luck >= +{threshold}")
print(f"  Unlucky: luck <= -{threshold}")
print(f"  Neutral: between")
if TEST_MODE:
    print(f"\n{EMOJI['test']} TEST MODE: Simulating 2025-12-04 (1 TNF game to save API credits)")
print(f"\nSafe Mode: {'ON (will ask before API calls)' if args.safe_mode else 'OFF (production mode)'}")

# =============================================================================
# STEP 1: Load Unexpected Points data (for prior week luck)
# =============================================================================
print("\n" + "=" * 100)
print("STEP 1: Loading Unexpected Points data")
print("=" * 100)

try:
    df_up = load_unexpected_points_data()
    print(f"{EMOJI['success']} Loaded Unexpected Points data: {len(df_up)} rows")
    print(f"   Weeks: {df_up['week'].min()} to {df_up['week'].max()}")
    max_week = df_up['week'].max()
except FileNotFoundError as e:
    print(f"\n{EMOJI['error']} ERROR: {e}")
    print("   Download the latest Unexpected Points data from:")
    print("   https://docs.google.com/spreadsheets/d/1ktlf_ekms7aI6r0tF_HeX0zaxps-bHWYsgglUReC558/edit")
    sys.exit(1)

# Build prior luck lookup (handles bye weeks)
luck_lookup = build_prior_luck_lookup(df_up)
print(f"   Teams tracked: {len(luck_lookup['weeks_played'])}")

# =============================================================================
# STEP 2: Get upcoming games (next 7 days)
# =============================================================================
print("\n" + "=" * 100)
print("STEP 2: Checking for upcoming games (next 7 days)")
print("=" * 100)

from zoneinfo import ZoneInfo
import requests
import os

# Use test date if in test mode
if TEST_MODE:
    today = TEST_DATE
    print(f"\n{EMOJI['test']} Using simulated date: {today.strftime('%Y-%m-%d %H:%M UTC')}")
else:
    today = datetime.now(timezone.utc)
next_week_7d = today + timedelta(days=7)

# Convert to ET for display
today_et = today.astimezone(ZoneInfo('America/New_York'))
next_week_7d_et = next_week_7d.astimezone(ZoneInfo('America/New_York'))

print(f"\nSearching for games:")
print(f"  UTC: {today.strftime('%Y-%m-%d %H:%M')} to {next_week_7d.strftime('%Y-%m-%d %H:%M')}")
print(f"  ET:  {today_et.strftime('%Y-%m-%d %H:%M')} to {next_week_7d_et.strftime('%Y-%m-%d %H:%M')}")

# Check upcoming directory for already fetched future games
upcoming_dir = Path(NFL_LINES_UPCOMING_DIR)
upcoming_dir.mkdir(parents=True, exist_ok=True)

# Find existing files in upcoming directory
upcoming_files = sorted(glob.glob(str(upcoming_dir / "nfl_game_lines_*.csv")))

# Load existing upcoming game data
existing_lines = pd.DataFrame()
fetch_fresh = False
df_current_week_lines = None

if upcoming_files:
    print(f"\nFound {len(upcoming_files)} file(s) in upcoming directory")
    dfs = []
    for csv_file in upcoming_files:
        df_temp = pd.read_csv(csv_file)
        df_temp['game_time'] = pd.to_datetime(df_temp['game_time'])
        if df_temp['game_time'].dt.tz is None:
            df_temp['game_time'] = df_temp['game_time'].dt.tz_localize('UTC')
        dfs.append(df_temp)
    
    existing_lines = pd.concat(dfs, ignore_index=True)
    
    # Find games in next 7 days
    upcoming_games = existing_lines[
        (existing_lines['game_time'] >= today) &
        (existing_lines['game_time'] <= next_week_7d)
    ]
    
    if len(upcoming_games) > 0:
        unique_games = upcoming_games['game_id'].nunique()
        
        # Show what games we have
        print(f"{EMOJI['success']} Found {unique_games} game(s) in next 7 days from existing data")
        for game_id in upcoming_games['game_id'].unique()[:5]:
            game = upcoming_games[upcoming_games['game_id'] == game_id].iloc[0]
            game_time_et = pd.to_datetime(game['game_time']).astimezone(ZoneInfo('America/New_York'))
            print(f"  • {game['away_team']} @ {game['home_team']} on {game_time_et.strftime('%a %Y-%m-%d %H:%M ET')}")
        if unique_games > 5:
            print(f"  ... and {unique_games - 5} more")
        
        # Ask if we want to use existing or fetch fresh
        if args.safe_mode:
            print(f"\n{EMOJI['warning']}  Use existing data or fetch fresh lines?")
            choice = input("   Use existing? [Y/n]: ").strip().lower()
            
            if choice in ['', 'y', 'yes', '1']:
                print(f"   {EMOJI['success']} Using existing data")
                df_current_week_lines = upcoming_games
                fetch_fresh = False
            else:
                print(f"   {EMOJI['refresh']} Will fetch fresh lines from API...")
                fetch_fresh = True
        else:
            # In non-safe mode, use existing data if available
            print(f"   {EMOJI['info']}  Using existing data (safe-mode off)")
            df_current_week_lines = upcoming_games
            fetch_fresh = False
    else:
        print(f"   No games found in next 7 days in existing files")
        fetch_fresh = True
else:
    print(f"\nNo files in upcoming directory")
    fetch_fresh = True

# Fetch from API if needed
if fetch_fresh or df_current_week_lines is None:
    print(f"\n{'='*100}")
    print("Fetching fresh lines from The Odds API")
    print(f"{'='*100}")
    
    # Check for API key
    api_key = os.getenv('ODDS_API_KEY')
    if not api_key:
        print(f"\n{EMOJI['error']} ERROR: ODDS_API_KEY environment variable not set")
        print("   Set it with: export ODDS_API_KEY='your_key_here'")
        sys.exit(1)
    
    print(f"Using API key: {api_key[:8]}...")
    
    # Final confirmation in safe mode
    if args.safe_mode:
        confirm = input(f"\n{EMOJI['warning']}  This will use API credits. Continue? [y/N]: ").strip().lower()
        if confirm not in ['y', 'yes']:
            print(f"   {EMOJI['error']} Aborted by user")
            sys.exit(0)
    
    # Monkey-patch requests to disable SSL verification
    original_request = requests.Session.request
    def patched_request(self, *args, **kwargs):
        kwargs['verify'] = False
        return original_request(self, *args, **kwargs)
    requests.Session.request = patched_request
    
    # Fetch from odds API
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
    
    # Format timestamps for API
    commence_from = today.strftime('%Y-%m-%dT%H:%M:%SZ')
    commence_to = next_week_7d.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': 'spreads',
        'oddsFormat': 'american',
        'commenceTimeFrom': commence_from,
        'commenceTimeTo': commence_to,
    }
    
    print(f"\n{EMOJI['refresh']} Fetching NFL spreads from The Odds API (next 7 days)...")
    print(f"   Date range: {commence_from} to {commence_to}")
    
    fetched_at = datetime.now(timezone.utc).isoformat()
    
    response = requests.get(url, params=params, verify=False)
    
    if response.status_code != 200:
        print(f"\n{EMOJI['error']} ERROR: API request failed with status {response.status_code}")
        print(f"   Response: {response.text}")
        sys.exit(1)
    
    data = response.json()
    print(f"{EMOJI['success']} Received {len(data)} games from API")
    
    # Filter to games in our 7-day window
    games_in_range = []
    for game in data:
        game_time_utc = pd.to_datetime(game['commence_time'])
        if game_time_utc.tz is None:
            game_time_utc = game_time_utc.tz_localize('UTC')
        
        if today <= game_time_utc <= next_week_7d:
            games_in_range.append(game)
    
    print(f"\n{EMOJI['calendar']} Games in 7-day window: {len(games_in_range)}")
    
    # Parse and save the data
    lines_to_save = []
    for game in games_in_range:
        game_id = game['id']
        game_time = game['commence_time']
        away_team = game['away_team']
        home_team = game['home_team']
        
        for bookmaker in game['bookmakers']:
            bookmaker_key = bookmaker['key']
            bookmaker_last_update = bookmaker['last_update']
            
            for market in bookmaker['markets']:
                if market['key'] == 'spreads':
                    for outcome in market['outcomes']:
                        if outcome['name'] == away_team:
                            lines_to_save.append({
                                'game_id': game_id,
                                'game_time': game_time,
                                'away_team': away_team,
                                'home_team': home_team,
                                'bookmaker': bookmaker_key,
                                'fetched_at': fetched_at,
                                'last_update': bookmaker_last_update,
                                'away_spread': outcome['point'],
                                'away_price': outcome['price'],
                            })
    
    if lines_to_save:
        df_new_lines = pd.DataFrame(lines_to_save)
        
        # Save to upcoming directory
        today_str = today.strftime('%Y-%m-%d')
        output_path = upcoming_dir / f"nfl_game_lines_{today_str}.csv"
        df_new_lines.to_csv(output_path, index=False)
        
        print(f"{EMOJI['save']} Saved {len(df_new_lines)} line records to {output_path}")
        
        # Use fresh data
        df_new_lines['game_time'] = pd.to_datetime(df_new_lines['game_time'])
        if df_new_lines['game_time'].dt.tz is None:
            df_new_lines['game_time'] = df_new_lines['game_time'].dt.tz_localize('UTC')
        
        df_current_week_lines = df_new_lines
    else:
        print(f"\n{EMOJI['error']} ERROR: No games returned from API")
        sys.exit(1)

print(f"\n{'='*100}")
print(f"{EMOJI['success']} Using {len(df_current_week_lines)} line records for {df_current_week_lines['game_id'].nunique()} upcoming game(s)")
print(f"{'='*100}")

# =============================================================================
# STEP 3: Calculate consensus spreads and add team abbreviations
# =============================================================================
print("\n" + "=" * 100)
print("STEP 3: Calculating consensus spreads")
print("=" * 100)

# Add team abbreviations
df_lines = add_team_abbr_columns(df_current_week_lines)

# Calculate consensus spread for each game
consensus_lines = []

for game_id, game_group in df_lines.groupby('game_id'):
    spreads = game_group['away_spread'].dropna()
    
    if len(spreads) == 0:
        continue
    
    consensus_lines.append({
        'game_id': game_id,
        'game_time': game_group['game_time'].iloc[0],
        'away_team': game_group['away_team'].iloc[0],
        'away_abbr': game_group['away_abbr'].iloc[0],
        'home_team': game_group['home_team'].iloc[0],
        'home_abbr': game_group['home_abbr'].iloc[0],
        'consensus_spread': spreads.median(),
        'num_books': len(spreads),
    })

df_consensus = pd.DataFrame(consensus_lines)
print(f"{EMOJI['success']} Calculated consensus spreads for {len(df_consensus)} games")

# Determine target week
target_week = max_week + 1
print(f"\nTarget Week: {target_week} (based on UP data through week {max_week})")

# =============================================================================
# STEP 4: Get prior week luck for both teams in each game
# =============================================================================
print("\n" + "=" * 100)
print("STEP 4: Getting prior week luck for both teams in each game")
print("=" * 100)

games_with_luck = []

for _, game in df_consensus.iterrows():
    away = game['away_abbr']
    home = game['home_abbr']
    
    # Get prior luck for both teams (week before target_week = max_week)
    away_prior_luck = get_prior_week_luck(luck_lookup, away, target_week)
    home_prior_luck = get_prior_week_luck(luck_lookup, home, target_week)
    
    # Categorize luck
    away_luck_cat = categorize_luck(away_prior_luck, threshold) if away_prior_luck is not None else 'Unknown'
    home_luck_cat = categorize_luck(home_prior_luck, threshold) if home_prior_luck is not None else 'Unknown'
    
    # Spread info
    abs_spread = abs(game['consensus_spread'])
    spread_cat = categorize_spread(game['consensus_spread'])
    away_is_favorite = game['consensus_spread'] < 0
    
    # Determine if this is a Lucky vs Unlucky matchup
    is_lucky_vs_unlucky = (
        (away_luck_cat == 'Lucky' and home_luck_cat == 'Unlucky') or
        (away_luck_cat == 'Unlucky' and home_luck_cat == 'Lucky')
    )
    
    # If Lucky vs Unlucky, which team should we bet?
    # BET THE UNLUCKY TEAM (regression to mean)
    bet_team = None
    bet_spread = None
    bet_reason = None
    
    if is_lucky_vs_unlucky:
        if away_luck_cat == 'Unlucky':
            bet_team = away
            bet_spread = game['consensus_spread']
            bet_reason = f"{away} was unlucky ({away_prior_luck:+.1f}), {home} was lucky ({home_prior_luck:+.1f})"
        else:
            bet_team = home
            bet_spread = -game['consensus_spread']
            bet_reason = f"{home} was unlucky ({home_prior_luck:+.1f}), {away} was lucky ({away_prior_luck:+.1f})"
    
    game_time_et = game['game_time'].astimezone(ZoneInfo('America/New_York'))
    
    games_with_luck.append({
        'game_id': game['game_id'],
        'game_time': game['game_time'],
        'game_time_et': game_time_et.strftime('%a %Y-%m-%d %H:%M ET'),
        'week': target_week,
        'away_abbr': away,
        'home_abbr': home,
        'consensus_spread': game['consensus_spread'],
        'abs_spread': abs_spread,
        'spread_cat': spread_cat,
        'away_is_favorite': away_is_favorite,
        'home_is_favorite': not away_is_favorite,
        'away_score': 'n/a',
        'home_score': 'n/a',
        'actual_margin': 'n/a',
        'away_covered': 'n/a',
        'home_covered': 'n/a',
        'away_luck': 'n/a',
        'home_luck': 'n/a',
        'away_prior_luck': away_prior_luck,
        'home_prior_luck': home_prior_luck,
        'away_luck_cat': away_luck_cat,
        'home_luck_cat': home_luck_cat,
        'matchup_type': f"{away_luck_cat} vs {home_luck_cat}",
        'is_lu': is_lucky_vs_unlucky,
        'bet_team': bet_team,
        'bet_spread': bet_spread,
        'bet_reason': bet_reason,
    })

df_games = pd.DataFrame(games_with_luck)

print(f"\n{EMOJI['chart']} Prior Week Luck Summary (Week {max_week}):")
for _, g in df_games.iterrows():
    away_luck_str = f"{g['away_prior_luck']:+.1f}" if g['away_prior_luck'] is not None else "N/A"
    home_luck_str = f"{g['home_prior_luck']:+.1f}" if g['home_prior_luck'] is not None else "N/A"
    matchup = f"{EMOJI['star']} {g['matchup_type']}" if g['is_lu'] else g['matchup_type']
    
    print(f"  {g['away_abbr']:>4s} ({away_luck_str:>6s}) @ {g['home_abbr']:<4s} ({home_luck_str:>6s}) | {matchup}")

# =============================================================================
# STEP 5: Verbose mode - explain reasoning for each game
# =============================================================================
if args.verbose_mode:
    print("\n" + "=" * 100)
    print("VERBOSE MODE: Detailed analysis for each game")
    print("=" * 100)
    
    for _, game in df_games.iterrows():
        print(f"\n{'='*100}")
        print(f"GAME: {game['away_abbr']} @ {game['home_abbr']}")
        print(f"Time: {game['game_time_et']}")
        print(f"Spread: {game['away_abbr']} {game['consensus_spread']:+.1f}")
        print(f"{'='*100}")
        
        print(f"\n{EMOJI['chart']} Prior Week (Week {max_week}) Luck:")
        
        # Away team
        if game['away_prior_luck'] is not None:
            away_status = ""
            if game['away_luck_cat'] == 'Lucky':
                away_status = f"{EMOJI['lucky']} LUCKY (>= +{threshold})"
            elif game['away_luck_cat'] == 'Unlucky':
                away_status = f"{EMOJI['unlucky']} UNLUCKY (<= -{threshold})"
            else:
                away_status = f"{EMOJI['neutral']} NEUTRAL"
            print(f"   {game['away_abbr']}: {game['away_prior_luck']:+.1f} → {away_status}")
        else:
            print(f"   {game['away_abbr']}: No prior data (Week 1 or missing)")
        
        # Home team
        if game['home_prior_luck'] is not None:
            home_status = ""
            if game['home_luck_cat'] == 'Lucky':
                home_status = f"{EMOJI['lucky']} LUCKY (>= +{threshold})"
            elif game['home_luck_cat'] == 'Unlucky':
                home_status = f"{EMOJI['unlucky']} UNLUCKY (<= -{threshold})"
            else:
                home_status = f"{EMOJI['neutral']} NEUTRAL"
            print(f"   {game['home_abbr']}: {game['home_prior_luck']:+.1f} → {home_status}")
        else:
            print(f"   {game['home_abbr']}: No prior data (Week 1 or missing)")
        
        print(f"\n{EMOJI['target']} Matchup Analysis:")
        print(f"   Type: {game['matchup_type']}")
        
        if game['is_lu']:
            print(f"\n   {EMOJI['success']} LUCKY vs UNLUCKY MATCHUP DETECTED!")
            print(f"   {EMOJI['up']} Regression Strategy: Bet the UNLUCKY team")
            print(f"   Reason: {game['bet_reason']}")
            print(f"\n   {EMOJI['money']} BET: {game['bet_team']} {game['bet_spread']:+.1f}")
        else:
            print(f"\n   {EMOJI['error']} Not a Lucky vs Unlucky matchup - NO BET")

# =============================================================================
# STEP 6: Display betting recommendations
# =============================================================================
print("\n" + "=" * 100)
print(f"BETTING OPPORTUNITIES FOR WEEK {target_week}")
print("=" * 100)

# Filter to only Lucky vs Unlucky matchups
df_plays = df_games[df_games['is_lu']].copy()

if len(df_plays) == 0:
    print(f"\n{EMOJI['error']} No Lucky vs Unlucky matchups found for this week")
    print(f"\nBreakdown of {len(df_games)} games:")
    
    # Count matchup types
    matchup_counts = df_games['matchup_type'].value_counts()
    for matchup, count in matchup_counts.items():
        print(f"   {matchup}: {count}")
else:
    print(f"\n{EMOJI['success']} Found {len(df_plays)} Lucky vs Unlucky matchup(s)\n")
    print(f"{EMOJI['target']} STRATEGY: Bet the UNLUCKY team (regression to mean)")
    print("-" * 100)
    
    for _, play in df_plays.iterrows():
        print(f"\n  {EMOJI['success']} BET: {play['bet_team']} {play['bet_spread']:+.1f}")
        print(f"     Game: {play['away_abbr']} @ {play['home_abbr']}")
        print(f"     Time: {play['game_time_et']}")
        print(f"     Spread Category: {play['spread_cat']}")
        print(f"     Reason: {play['bet_reason']}")
        
        # Show which team is fav/dog
        unlucky_is_fav = (play['bet_spread'] < 0)
        role = "FAVORITE" if unlucky_is_fav else "UNDERDOG"
        print(f"     Unlucky team is: {role}")

# =============================================================================
# STEP 7: Save outputs
# =============================================================================
output_dir = Path(f"/Users/thomasmyles/dev/betting/data/04_output/nfl/2025/plays/week_{target_week}")
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
threshold_str = f"threshold_{int(threshold)}"

# Save all games analysis
all_games_path = output_dir / f"all_games_{threshold_str}_{timestamp}.csv"
df_games.to_csv(all_games_path, index=False)

print(f"\n{'=' * 100}")
print(f"{EMOJI['success']} Saved all {len(df_games)} games to: {all_games_path}")

# Save plays only
if len(df_plays) > 0:
    plays_path = output_dir / f"plays_{threshold_str}_{timestamp}.csv"
    df_plays.to_csv(plays_path, index=False)
    print(f"{EMOJI['save']} Saved {len(df_plays)} plays to: {plays_path}")

print("=" * 100)

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{EMOJI['chart']} Summary:")
print(f"   Games analyzed: {len(df_games)}")
print(f"   Lucky vs Unlucky matchups: {len(df_plays)}")
print(f"   Threshold used: ±{threshold}")
print(f"   Data through: Week {max_week}")

if len(df_plays) > 0:
    print(f"\n{EMOJI['target']} Plays Found:")
    for _, play in df_plays.iterrows():
        print(f"   • {play['bet_team']} {play['bet_spread']:+.1f} vs {play['away_abbr'] if play['bet_team'] == play['home_abbr'] else play['home_abbr']}")

print(f"\n{EMOJI['success']} COMPLETE")
print("=" * 100)

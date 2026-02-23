"""
Find NFL Regression Betting Opportunities for Upcoming Week

Based on analysis from 20251126_nfl_spread_covering_vs_score_differential.py

Two proven strategies (thru Week 12, 2025):
1. PRIMARY: Back unlucky favorites (after -7 luck, favorites with spread ≤7)
   - 3.5-7 favorites: 72.7% ATS, +38.8% ROI (11 games)
   - ≤3 favorites: 66.7% ATS, +27.3% ROI (15 games)

2. SECONDARY: Back lucky big underdogs (after +7 luck, underdogs with spread ≥7)
   - 71.4% ATS, +36.4% ROI (7 games)

Usage:
    # Backtest on historical week
    python implementation/find_nfl_regression_plays.py --week 10
    
    # Find plays for current week (asks before API calls)
    python implementation/find_nfl_regression_plays.py --current-week
    python implementation/find_nfl_regression_plays.py --current-week --verbose-mode
    
    # Production mode (no prompts)
    python implementation/find_nfl_regression_plays.py --current-week --no-safe-mode
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
import glob
from dotenv import load_dotenv
import ssl
import urllib3

# Load environment variables
load_dotenv()

# Fix SSL certificate issues (for API calls)
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from nfl_team_utils import add_team_abbr_columns

# Hardcoded strategy parameters based on analysis
LUCK_THRESHOLD = 7.0
PRIMARY_MAX_SPREAD = 7.0
SECONDARY_MIN_SPREAD = 7.0

print("=" * 100)
print("NFL REGRESSION BETTING PLAYS FINDER")
print("=" * 100)

# Parse arguments
parser = argparse.ArgumentParser(description='Find NFL regression betting plays')
parser.add_argument('--week', type=int, required=False,
                   help='Week number to find plays for (required unless --current-week is used)')
parser.add_argument('--current-week', action='store_true',
                   help='Find plays for current week (games in next 7 days). Will fetch live lines if needed.')
parser.add_argument('--verbose-mode', action='store_true',
                   help='Print detailed reasoning for each team')
parser.add_argument('--safe-mode', action='store_true', default=True,
                   help='Ask before making API calls (default: True). Use --no-safe-mode for production.')
parser.add_argument('--no-safe-mode', dest='safe_mode', action='store_false',
                   help='Skip confirmation prompts for API calls (for production use)')
args = parser.parse_args()

if not args.week and not args.current_week:
    parser.error("Either --week or --current-week must be specified")

if args.week and args.current_week:
    parser.error("Cannot specify both --week and --current-week")

# Determine target week
if args.current_week:
    target_week = None  # Will be determined after checking games
    prev_week = None
else:
    target_week = args.week
    prev_week = target_week - 1

if not args.current_week:
    print(f"\nTarget Week: {target_week}")
    print(f"Using luck from: Week {prev_week}")
else:
    print(f"\nMode: CURRENT WEEK (games in next 7 days)")
    print(f"Safe Mode: {'ON (will ask before API calls)' if args.safe_mode else 'OFF (production mode)'}")

print(f"Strategies: PRIMARY (unlucky favs ≤7) | SECONDARY (lucky dogs ≥7)")

# =============================================================================
# STEP 0: If current week mode, check for games and fetch if needed
# =============================================================================
if args.current_week:
    from datetime import datetime, timedelta, timezone
    import requests
    import os
    
    print("\n" + "=" * 100)
    print("STEP 0: Checking for upcoming games (next 7 days)")
    print("=" * 100)
    
    from zoneinfo import ZoneInfo
    
    today = datetime.now(timezone.utc)
    next_week_7d = today + timedelta(days=7)
    
    # Convert to ET for display
    today_et = today.astimezone(ZoneInfo('America/New_York'))
    next_week_7d_et = next_week_7d.astimezone(ZoneInfo('America/New_York'))
    
    print(f"\nSearching for games in next 7 days:")
    print(f"  UTC: {today.strftime('%Y-%m-%d %H:%M')} to {next_week_7d.strftime('%Y-%m-%d %H:%M')}")
    print(f"  ET:  {today_et.strftime('%Y-%m-%d %H:%M')} to {next_week_7d_et.strftime('%Y-%m-%d %H:%M')}")
    
    # Check upcoming directory for already fetched future games
    upcoming_dir = Path("/Users/thomasmyles/dev/betting/data/01_input/the-odds-api/nfl/game_lines/upcoming")
    upcoming_dir.mkdir(parents=True, exist_ok=True)
    
    # Find existing files in upcoming directory
    upcoming_files = sorted(glob.glob(str(upcoming_dir / "nfl_game_lines_*.csv")))
    
    # Load existing upcoming game data
    existing_lines = pd.DataFrame()
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
            print(f"✅ Found {unique_games} game(s) in next 7 days from existing data")
            for game_id in upcoming_games['game_id'].unique()[:5]:
                game = upcoming_games[upcoming_games['game_id'] == game_id].iloc[0]
                game_time_et = pd.to_datetime(game['game_time']).astimezone(ZoneInfo('America/New_York'))
                print(f"  • {game['away_team']} @ {game['home_team']} on {game_time_et.strftime('%a %Y-%m-%d %H:%M ET')}")
            if unique_games > 5:
                print(f"  ... and {unique_games - 5} more")
            
            # Ask if we want to use existing or fetch fresh
            if args.safe_mode:
                print(f"\n⚠️  Use existing data or fetch fresh lines?")
                print(f"   1. Use existing (FREE)")
                print(f"   2. Fetch fresh from API (costs credits)")
                
                choice = input("\n   Use existing? [Y/n]: ").strip().lower()
                
                if choice in ['', 'y', 'yes']:
                    print("   ✅ Using existing data")
                    df_current_week_lines = upcoming_games
                    fetch_fresh = False
                else:
                    print("   🔄 Will fetch fresh lines from API...")
                    fetch_fresh = True
            else:
                # In non-safe mode, use existing data if available
                print("   ℹ️  Using existing data (safe-mode off)")
                df_current_week_lines = upcoming_games
                fetch_fresh = False
        else:
            print(f"   No games found in next 7 days in existing files")
            fetch_fresh = True
    else:
        print(f"\nNo files in upcoming directory")
        fetch_fresh = True
    
    # Fetch from API if needed
    if 'fetch_fresh' not in locals():
        fetch_fresh = False
    
    if fetch_fresh or len(existing_lines) == 0:
        print(f"\n{'='*100}")
        print("Fetching fresh lines from The Odds API")
        print(f"{'='*100}")
        
        # Check for API key
        api_key = os.getenv('ODDS_API_KEY')
        if not api_key:
            print("\n❌ ERROR: ODDS_API_KEY environment variable not set")
            print("   Set it with: export ODDS_API_KEY='your_key_here'")
            print("   Or fetch data manually with:")
            print("   python scripts/fetch_nfl_season_lines.py")
            sys.exit(1)
        
        print(f"Using API key: {api_key[:8]}...")
        
        # Final confirmation in safe mode
        if args.safe_mode:
            confirm = input("\n⚠️  This will use API credits. Continue? [y/N]: ").strip().lower()
            if confirm not in ['y', 'yes']:
                print("   ❌ Aborted by user")
                sys.exit(0)
        
        # Monkey-patch requests to disable SSL verification
        original_request = requests.Session.request
        def patched_request(self, *args, **kwargs):
            kwargs['verify'] = False
            return original_request(self, *args, **kwargs)
        requests.Session.request = patched_request
        
        # Fetch from odds API
        url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
        
        # Format timestamps for API (must be YYYY-MM-DDTHH:MM:SSZ format with 'Z' suffix)
        commence_from = today.strftime('%Y-%m-%dT%H:%M:%SZ')
        commence_to = next_week_7d.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        params = {
            'apiKey': api_key,
            'regions': 'us',
            'markets': 'spreads',
            'oddsFormat': 'american',
            'commenceTimeFrom': commence_from,  # Filter: only games starting from now
            'commenceTimeTo': commence_to,  # Filter: only games in next 7 days
        }
        
        print(f"\n🔄 Fetching NFL spreads from The Odds API (next 7 days)...")
        print(f"   Date range: {commence_from} to {commence_to}")
        
        # Record when we're fetching (UTC)
        fetched_at = datetime.now(timezone.utc).isoformat()
        
        response = requests.get(url, params=params, verify=False)
        
        if response.status_code != 200:
            print(f"\n❌ ERROR: API request failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            sys.exit(1)
        
        data = response.json()
        print(f"✅ Received {len(data)} games from API (already filtered to 7-day window)")
        
        # Double-check filtering (safety check - API should already filter)
        games_in_range = []
        games_out_of_range = []
        
        for game in data:
            game_time_utc = pd.to_datetime(game['commence_time'])
            if game_time_utc.tz is None:
                game_time_utc = game_time_utc.tz_localize('UTC')
            
            if today <= game_time_utc <= next_week_7d:
                games_in_range.append(game)
            else:
                game_time_et = game_time_utc.astimezone(ZoneInfo('America/New_York'))
                games_out_of_range.append(f"  {game['away_team']} @ {game['home_team']} on {game_time_et.strftime('%Y-%m-%d %H:%M ET')}")
        
        print(f"\n📅 Date filter applied: {len(games_in_range)} games in 7-day window")
        
        if games_out_of_range:
            print(f"   ⚠️  Warning: API returned {len(games_out_of_range)} games outside requested range:")
            for g in games_out_of_range[:5]:  # Show first 5
                print(g)
            if len(games_out_of_range) > 5:
                print(f"   ... and {len(games_out_of_range) - 5} more")
            print(f"   This is unexpected - API should have filtered by commenceTimeFrom/To")
        
        # Sanity check: NFL has 32 teams, max 16 games per week
        if len(games_in_range) > 32:
            print(f"\n❌ ERROR: Found {len(games_in_range)} games in next 7 days")
            print(f"   This is >32 which suggests multiple weeks are being included!")
            print(f"   Aborting to prevent incorrect analysis.")
            sys.exit(1)
        
        # Show games we're using with ET times
        print(f"\n✅ Games to analyze:")
        for game in games_in_range[:10]:  # Show first 10
            game_time_utc = pd.to_datetime(game['commence_time'])
            if game_time_utc.tz is None:
                game_time_utc = game_time_utc.tz_localize('UTC')
            game_time_et = game_time_utc.astimezone(ZoneInfo('America/New_York'))
            print(f"  {game['away_team']} @ {game['home_team']} on {game_time_et.strftime('%a %Y-%m-%d %H:%M ET')}")
        if len(games_in_range) > 10:
            print(f"  ... and {len(games_in_range) - 10} more")
        
        # Parse and save the data (only games in range)
        lines_to_save = []
        for game in games_in_range:
            game_id = game['id']
            game_time = game['commence_time']
            away_team = game['away_team']
            home_team = game['home_team']
            
            # Extract spreads from bookmakers
            for bookmaker in game.get('bookmakers', []):
                bookmaker_key = bookmaker['key']
                bookmaker_last_update = bookmaker.get('last_update')  # When this bookmaker last updated their odds
                
                for market in bookmaker.get('markets', []):
                    if market['key'] == 'spreads':
                        for outcome in market['outcomes']:
                            if outcome['name'] == away_team:
                                lines_to_save.append({
                                    'game_id': game_id,
                                    'game_time': game_time,
                                    'away_team': away_team,
                                    'home_team': home_team,
                                    'bookmaker': bookmaker_key,
                                    'fetched_at': fetched_at,  # When WE fetched from API
                                    'last_update': bookmaker_last_update,  # When bookmaker last updated this line
                                    'away_spread': outcome['point'],
                                    'away_price': outcome['price'],
                                })
        
        if lines_to_save:
            df_new_lines = pd.DataFrame(lines_to_save)
            
            # Save to upcoming directory
            upcoming_dir = Path("/Users/thomasmyles/dev/betting/data/01_input/the-odds-api/nfl/game_lines/upcoming")
            upcoming_dir.mkdir(parents=True, exist_ok=True)
            
            today_str = today.strftime('%Y-%m-%d')
            output_path = upcoming_dir / f"nfl_game_lines_{today_str}.csv"
            df_new_lines.to_csv(output_path, index=False)
            
            print(f"💾 Saved {len(df_new_lines)} line records to {output_path}")
            print(f"   ℹ️  Data includes: fetched_at (when we called API) and last_update (when bookmaker updated)")
            
            # Use fresh data
            df_new_lines['game_time'] = pd.to_datetime(df_new_lines['game_time'])
            if df_new_lines['game_time'].dt.tz is None:
                df_new_lines['game_time'] = df_new_lines['game_time'].dt.tz_localize('UTC')
            
            df_current_week_lines = df_new_lines
            print(f"✅ Ready to analyze {len(df_current_week_lines['game_id'].unique())} upcoming games")
        else:
            print("\n❌ ERROR: No games returned from API")
            sys.exit(1)
    
    print(f"\n{'='*100}")
    print(f"✅ Using {len(df_current_week_lines)} line records for {df_current_week_lines['game_id'].nunique()} upcoming game(s)")
    print(f"{'='*100}")

# =============================================================================
# STEP 1: Load historical tracking data to get each team's most recent luck
# =============================================================================
print("\n" + "=" * 100)
if args.current_week:
    print(f"STEP 1: Loading most recent team luck")
else:
    print(f"STEP 1: Loading team luck from Week {prev_week}")
print("=" * 100)

tracking_path = Path(f"/Users/thomasmyles/dev/betting/data/03_intermediate/nfl_game_by_game_tracking_threshold_{int(LUCK_THRESHOLD)}.csv")

if not tracking_path.exists():
    print(f"\n❌ ERROR: Tracking file not found at {tracking_path}")
    print("\n   Run this first:")
    print("   python backtesting/20251126_nfl_spread_covering_vs_score_differential.py --observe-trends --team all")
    sys.exit(1)

df_tracking = pd.read_csv(tracking_path)

print(f"Loaded {len(df_tracking)} rows from tracking file")
print(f"Weeks available: {df_tracking['week'].min()} to {df_tracking['week'].max()}")

# Determine which week to use for luck data
if args.current_week:
    # Use most recent week in tracking data
    lookup_week = df_tracking['week'].max()
    print(f"\nUsing most recent week: {lookup_week}")
    prev_week = lookup_week
else:
    lookup_week = prev_week

# Get previous week data
prev_week_data = df_tracking[df_tracking['week'] == lookup_week].copy()

if len(prev_week_data) == 0:
    print(f"\n❌ ERROR: No data found for Week {lookup_week}")
    print(f"   Available weeks: {sorted(df_tracking['week'].unique())}")
    sys.exit(1)

print(f"\nTeams with Week {prev_week} data: {len(prev_week_data)}")

# Get each team's luck from previous week
team_luck_map = {}
for _, row in prev_week_data.iterrows():
    team_luck_map[row['team']] = {
        'luck': row['luck'],
        'exp_diff': row['expected_diff'],
        'actual_diff': row['actual_diff'],
    }

lucky_teams = [t for t, d in team_luck_map.items() if d['luck'] >= LUCK_THRESHOLD]
unlucky_teams = [t for t, d in team_luck_map.items() if d['luck'] <= -LUCK_THRESHOLD]

print(f"\nLuck summary:")
print(f"  +{LUCK_THRESHOLD} or more (lucky): {len(lucky_teams)} teams")
if lucky_teams:
    for t in sorted(lucky_teams):
        print(f"    {t}: {team_luck_map[t]['luck']:+.1f}")

print(f"  -{LUCK_THRESHOLD} or less (unlucky): {len(unlucky_teams)} teams")
if unlucky_teams:
    for t in sorted(unlucky_teams):
        print(f"    {t}: {team_luck_map[t]['luck']:+.1f}")

# =============================================================================
# STEP 2: Load betting lines for target week
# =============================================================================
print("\n" + "=" * 100)
if args.current_week:
    print(f"STEP 2: Using current week lines from Step 0")
    print("=" * 100)
    df_lines = df_current_week_lines.copy()
else:
    print(f"STEP 2: Loading betting lines for Week {target_week}")
    print("=" * 100)
    
    # Load all historical betting lines
    lines_dir = Path("/Users/thomasmyles/dev/betting/data/01_input/the-odds-api/nfl/game_lines/historical")
    csv_files = sorted(glob.glob(str(lines_dir / "nfl_game_lines_*.csv")))
    
    # Add London games if exists
    london_file = lines_dir / "2025_game_lines_london.csv"
    if london_file.exists():
        csv_files.append(str(london_file))
    
    if not csv_files:
        print(f"\n❌ ERROR: No betting lines found in {lines_dir}")
        sys.exit(1)
    
    # Load and combine all lines
    dfs = []
    for csv_file in csv_files:
        df_temp = pd.read_csv(csv_file)
        dfs.append(df_temp)
    
    df_lines = pd.concat(dfs, ignore_index=True)
    df_lines['game_time'] = pd.to_datetime(df_lines['game_time'])

# Add team abbreviations
df_lines = add_team_abbr_columns(df_lines)

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

if args.current_week:
    # For current week mode, use all games we found
    week_games = df_consensus.copy()
    
    # Determine target week from the games (assume all in same week)
    # Try to get week from results file if it exists for backtesting
    results_path = Path("/Users/thomasmyles/dev/betting/data/03_intermediate/nfl_games_with_spreads_and_results.csv")
    if results_path.exists():
        df_results = pd.read_csv(results_path)
        week_games = week_games.merge(
            df_results[['game_id', 'week']],
            on='game_id',
            how='left'
        )
        if week_games['week'].notna().any():
            target_week = int(week_games['week'].mode().iloc[0])
            print(f"\nDetermined week: {target_week}")
        else:
            # Estimate week based on date
            target_week = prev_week + 1
            print(f"\nEstimated week: {target_week} (based on most recent luck data)")
    else:
        # Estimate week
        target_week = prev_week + 1
        print(f"\nEstimated week: {target_week} (based on most recent luck data)")
    
    print(f"Found {len(week_games)} upcoming games")
    
else:
    # For historical mode, filter by week number
    # Get week numbers from results file
    results_path = Path("/Users/thomasmyles/dev/betting/data/03_intermediate/nfl_games_with_spreads_and_results.csv")
    
    if not results_path.exists():
        print(f"\n❌ ERROR: Results file not found at {results_path}")
        print("\n   Run this first:")
        print("   python backtesting/20251126_nfl_spread_covering_vs_score_differential.py")
        sys.exit(1)
    
    df_results = pd.read_csv(results_path)
    
    # Merge to get week numbers
    df_consensus = df_consensus.merge(
        df_results[['game_id', 'week']],
        on='game_id',
        how='left'
    )
    
    # Filter to target week
    week_games = df_consensus[df_consensus['week'] == target_week].copy()
    
    if len(week_games) == 0:
        print(f"\n❌ No games found for Week {target_week}")
        print(f"   Available weeks: {sorted(df_consensus['week'].dropna().unique())}")
        sys.exit(1)
    
    print(f"\nFound {len(week_games)} games in Week {target_week}")

# =============================================================================
# STEP 3: Create output with 1 row per team
# =============================================================================
print("\n" + "=" * 100)
print("STEP 3: Building team-level output with strategy flags")
print("=" * 100)

teams_output = []

for _, game in week_games.iterrows():
    # Convert game time to ET for display
    from zoneinfo import ZoneInfo
    game_time_utc = pd.to_datetime(game['game_time'])
    if game_time_utc.tz is None:
        game_time_utc = game_time_utc.tz_localize('UTC')
    game_time_et = game_time_utc.astimezone(ZoneInfo('America/New_York'))
    
    # Process away team
    away_team = game['away_abbr']
    away_spread = game['consensus_spread']
    away_spread_abs = abs(away_spread)
    away_is_fav = away_spread < 0
    
    # Get spread category
    if away_spread_abs <= 3:
        away_spread_cat = '≤3'
    elif away_spread_abs <= 7:
        away_spread_cat = '3.5-7'
    else:
        away_spread_cat = '7.5+'
    
    # Get luck from previous week
    away_luck_data = team_luck_map.get(away_team, {})
    away_luck = away_luck_data.get('luck', 0)
    
    # Apply strategies
    away_primary = (
        away_luck <= -LUCK_THRESHOLD and 
        away_is_fav and 
        away_spread_abs <= PRIMARY_MAX_SPREAD
    )
    
    away_secondary = (
        away_luck >= LUCK_THRESHOLD and 
        not away_is_fav and 
        away_spread_abs >= SECONDARY_MIN_SPREAD
    )
    
    teams_output.append({
        'team': away_team,
        'opponent': game['home_abbr'],
        'location': 'away',
        'game_id': game['game_id'],
        'game_time': game['game_time'],
        'game_time_et': game_time_et.strftime('%Y-%m-%d %H:%M ET'),
        'last_week': prev_week,
        'last_week_luck': away_luck,
        'last_week_exp_diff': away_luck_data.get('exp_diff', np.nan),
        'last_week_actual_diff': away_luck_data.get('actual_diff', np.nan),
        'this_week': target_week,
        'this_week_spread': away_spread,
        'spread_abs': away_spread_abs,
        'spread_category': away_spread_cat,
        'is_favorite': away_is_fav,
        'num_books': game['num_books'],
        'strat_primary_back_unlucky_fav_small_spread': away_primary,
        'strat_secondary_back_lucky_dog_large_spread': away_secondary,
    })
    
    # Process home team
    home_team = game['home_abbr']
    home_spread = -away_spread
    home_spread_abs = abs(home_spread)
    home_is_fav = home_spread < 0
    
    # Get spread category
    if home_spread_abs <= 3:
        home_spread_cat = '≤3'
    elif home_spread_abs <= 7:
        home_spread_cat = '3.5-7'
    else:
        home_spread_cat = '7.5+'
    
    # Get luck from previous week
    home_luck_data = team_luck_map.get(home_team, {})
    home_luck = home_luck_data.get('luck', 0)
    
    # Apply strategies
    home_primary = (
        home_luck <= -LUCK_THRESHOLD and 
        home_is_fav and 
        home_spread_abs <= PRIMARY_MAX_SPREAD
    )
    
    home_secondary = (
        home_luck >= LUCK_THRESHOLD and 
        not home_is_fav and 
        home_spread_abs >= SECONDARY_MIN_SPREAD
    )
    
    teams_output.append({
        'team': home_team,
        'opponent': away_team,
        'location': 'home',
        'game_id': game['game_id'],
        'game_time': game['game_time'],
        'game_time_et': game_time_et.strftime('%Y-%m-%d %H:%M ET'),
        'last_week': prev_week,
        'last_week_luck': home_luck,
        'last_week_exp_diff': home_luck_data.get('exp_diff', np.nan),
        'last_week_actual_diff': home_luck_data.get('actual_diff', np.nan),
        'this_week': target_week,
        'this_week_spread': home_spread,
        'spread_abs': home_spread_abs,
        'spread_category': home_spread_cat,
        'is_favorite': home_is_fav,
        'num_books': game['num_books'],
        'strat_primary_back_unlucky_fav_small_spread': home_primary,
        'strat_secondary_back_lucky_dog_large_spread': home_secondary,
    })

df_output = pd.DataFrame(teams_output)

print(f"\nCreated output with {len(df_output)} rows ({len(df_output)//2} games × 2 teams)")

# =============================================================================
# STEP 4: Verbose mode - explain reasoning for each team
# =============================================================================
if args.verbose_mode:
    print("\n" + "=" * 100)
    print(f"VERBOSE MODE: Detailed analysis for each team")
    print("=" * 100)
    
    for _, team in df_output.iterrows():
        print(f"\n{'='*100}")
        print(f"Team: {team['team']} ({team['location']}) vs {team['opponent']}")
        print(f"Game Time: {team['game_time']}")
        print(f"{'='*100}")
        
        print(f"\n📊 Last Week (Week {team['last_week']}) Performance:")
        print(f"   Luck: {team['last_week_luck']:+.1f}")
        print(f"   Expected Diff: {team['last_week_exp_diff']:+.1f}")
        print(f"   Actual Diff: {team['last_week_actual_diff']:+.1f}")
        
        # Categorize luck
        if team['last_week_luck'] >= LUCK_THRESHOLD:
            luck_status = f"🍀 LUCKY (+{LUCK_THRESHOLD} or more)"
        elif team['last_week_luck'] <= -LUCK_THRESHOLD:
            luck_status = f"💔 UNLUCKY (-{LUCK_THRESHOLD} or less)"
        else:
            luck_status = f"😐 NEUTRAL (between -{LUCK_THRESHOLD} and +{LUCK_THRESHOLD})"
        print(f"   Status: {luck_status}")
        
        print(f"\n📈 This Week (Week {team['this_week']}) Betting Line:")
        print(f"   Spread: {team['this_week_spread']:+.1f}")
        print(f"   Role: {'⭐ FAVORITE' if team['is_favorite'] else '🐶 UNDERDOG'}")
        print(f"   Spread Category: {team['spread_category']}")
        
        print(f"\n🎯 Strategy Evaluation:")
        
        # Primary strategy check
        print(f"\n   PRIMARY (Back Unlucky Fav ≤7):")
        if team['last_week_luck'] <= -LUCK_THRESHOLD:
            print(f"      ✓ Unlucky last week ({team['last_week_luck']:+.1f})")
        else:
            print(f"      ✗ Not unlucky enough ({team['last_week_luck']:+.1f}, need ≤-{LUCK_THRESHOLD})")
        
        if team['is_favorite']:
            print(f"      ✓ Is a favorite ({team['this_week_spread']:+.1f})")
        else:
            print(f"      ✗ Is an underdog ({team['this_week_spread']:+.1f})")
        
        if team['spread_abs'] <= PRIMARY_MAX_SPREAD:
            print(f"      ✓ Spread is ≤7 ({team['spread_abs']:.1f})")
        else:
            print(f"      ✗ Spread too large ({team['spread_abs']:.1f}, need ≤{PRIMARY_MAX_SPREAD})")
        
        if team['strat_primary_back_unlucky_fav_small_spread']:
            print(f"      ✅ PRIMARY MATCH - Expected: 67-73% ATS, +27-39% ROI")
        else:
            print(f"      ❌ Does not qualify")
        
        # Secondary strategy check
        print(f"\n   SECONDARY (Back Lucky Dog ≥7):")
        if team['last_week_luck'] >= LUCK_THRESHOLD:
            print(f"      ✓ Lucky last week ({team['last_week_luck']:+.1f})")
        else:
            print(f"      ✗ Not lucky enough ({team['last_week_luck']:+.1f}, need ≥+{LUCK_THRESHOLD})")
        
        if not team['is_favorite']:
            print(f"      ✓ Is an underdog ({team['this_week_spread']:+.1f})")
        else:
            print(f"      ✗ Is a favorite ({team['this_week_spread']:+.1f})")
        
        if team['spread_abs'] >= SECONDARY_MIN_SPREAD:
            print(f"      ✓ Spread is ≥7 ({team['spread_abs']:.1f})")
        else:
            print(f"      ✗ Spread too small ({team['spread_abs']:.1f}, need ≥{SECONDARY_MIN_SPREAD})")
        
        if team['strat_secondary_back_lucky_dog_large_spread']:
            print(f"      ✅ SECONDARY MATCH - Expected: 71% ATS, +36% ROI")
        else:
            print(f"      ❌ Does not qualify")
        
        # Final verdict
        if team['strat_primary_back_unlucky_fav_small_spread'] or team['strat_secondary_back_lucky_dog_large_spread']:
            print(f"\n💰 BETTING RECOMMENDATION: BET {team['team']} {team['this_week_spread']:+.1f}")
        else:
            print(f"\n🚫 NO BET")

# =============================================================================
# STEP 5: Display summary and save results
# =============================================================================
print("\n" + "=" * 100)
print(f"BETTING OPPORTUNITIES FOR WEEK {target_week}")
print("=" * 100)

# Filter to only plays
df_plays = df_output[(df_output['strat_primary_back_unlucky_fav_small_spread']) | (df_output['strat_secondary_back_lucky_dog_large_spread'])].copy()

if len(df_plays) == 0:
    print("\n❌ No strategy matches found for this week")
    print(f"\nBreakdown of {len(df_output)} teams checked:")
    print(f"  Lucky (+{LUCK_THRESHOLD}): {(df_output['last_week_luck'] >= LUCK_THRESHOLD).sum()}")
    print(f"  Unlucky (-{LUCK_THRESHOLD}): {(df_output['last_week_luck'] <= -LUCK_THRESHOLD).sum()}")
    print(f"  Favorites: {df_output['is_favorite'].sum()}")
    print(f"  Underdogs: {(~df_output['is_favorite']).sum()}")
else:
    primary_plays = df_plays[df_plays['strat_primary_back_unlucky_fav_small_spread']]
    secondary_plays = df_plays[df_plays['strat_secondary_back_lucky_dog_large_spread']]
    
    print(f"\n✅ Found {len(df_plays)} plays ({len(primary_plays)} primary, {len(secondary_plays)} secondary)\n")
    
    if len(primary_plays) > 0:
        print("🔥 PRIMARY STRATEGY: Unlucky Favorites (spread ≤7)")
        print("   Expected: 67-73% ATS, +27-39% ROI")
        print("-" * 100)
        
        for _, play in primary_plays.iterrows():
            vs_at = '@' if play['location'] == 'away' else 'vs'
            print(f"\n  ✅ BET: {play['team']} {play['this_week_spread']:+.1f}")
            print(f"     Game: {play['team']} {vs_at} {play['opponent']}")
            print(f"     Last week (W{play['last_week']}) luck: {play['last_week_luck']:+.1f}")
            print(f"     This week spread: {play['this_week_spread']:+.1f} ({play['spread_category']} favorite)")
    
    if len(secondary_plays) > 0:
        print(f"\n\n💎 SECONDARY STRATEGY: Lucky Big Underdogs (spread ≥7)")
        print("   Expected: 71% ATS, +36% ROI")
        print("-" * 100)
        
        for _, play in secondary_plays.iterrows():
            vs_at = '@' if play['location'] == 'away' else 'vs'
            print(f"\n  ✅ BET: {play['team']} {play['this_week_spread']:+.1f}")
            print(f"     Game: {play['team']} {vs_at} {play['opponent']}")
            print(f"     Last week (W{play['last_week']}) luck: {play['last_week_luck']:+.1f}")
            print(f"     This week spread: {play['this_week_spread']:+.1f} ({play['spread_category']} underdog)")

# Save outputs
from datetime import datetime
output_dir = Path("/Users/thomasmyles/dev/betting/data/04_output/nfl/todays_plays")
output_dir.mkdir(parents=True, exist_ok=True)

# Create timestamp for filename
timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')

# Always save all teams - main output
all_teams_path = output_dir / f"nfl_luck_regression_all_teams_week_{target_week}_{timestamp}.csv"
df_output.to_csv(all_teams_path, index=False)

print(f"\n{'=' * 100}")
print(f"✅ Saved all {len(df_output)} teams to: {all_teams_path}")

# Also save filtered plays for convenience
if len(df_plays) > 0:
    plays_path = output_dir / f"nfl_luck_regression_plays_week_{target_week}_{timestamp}.csv"
    df_plays.to_csv(plays_path, index=False)
    print(f"💾 Also saved {len(df_plays)} plays (filtered) to: {plays_path}")

print("=" * 100)

# Summary stats
total_teams = len(df_output)
primary_count = df_output['strat_primary_back_unlucky_fav_small_spread'].sum()
secondary_count = df_output['strat_secondary_back_lucky_dog_large_spread'].sum()

print(f"\n📊 Summary:")
print(f"   Total teams analyzed: {total_teams}")
print(f"   Primary strategy matches: {primary_count}")
print(f"   Secondary strategy matches: {secondary_count}")
print(f"   Total betting opportunities: {primary_count + secondary_count}")

print("\n✅ COMPLETE")
print("=" * 100)

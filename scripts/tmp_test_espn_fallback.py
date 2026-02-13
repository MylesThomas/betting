"""
Temporary test script to compare NBA API vs ESPN API data.
Goal: Make ESPN API return exact same columns as NBA API.

Test with:
    python scripts/tmp_test_espn_fallback.py --date 2026-02-12
"""

import sys
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import urllib3
import ssl

# ============================================================================
# SSL FIX FOR MACOS (same as fetch_nba_player_props.py)
# ============================================================================
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests with timeout
original_request = requests.Session.request
def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    kwargs.setdefault('timeout', 30)
    return original_request(self, *args, **kwargs)
requests.Session.request = patched_request

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Test imports
try:
    from nba_api.stats.endpoints import playergamelogs
    print("✅ nba_api imported")
except ImportError as e:
    print(f"❌ nba_api import failed: {e}")
    sys.exit(1)


def fetch_nba_api_data(date_str, season='2025-26'):
    """Fetch from NBA API (current method)"""
    print(f"\n{'='*80}")
    print(f"FETCHING FROM NBA API")
    print(f"{'='*80}")
    
    try:
        game_logs = playergamelogs.PlayerGameLogs(
            season_nullable=season,
            season_type_nullable='Regular Season',
            date_from_nullable=date_str,
            date_to_nullable=date_str
        )
        
        df = game_logs.get_data_frames()[0]
        
        if df.empty:
            print("❌ No games found")
            return pd.DataFrame()
        
        print(f"✅ Found {len(df)} player performances")
        print(f"\nColumns ({len(df.columns)}):")
        for col in df.columns:
            print(f"  - {col}")
        
        print(f"\nSample data (first player):")
        print(df.head(1).T)
        
        return df
        
    except Exception as e:
        print(f"❌ NBA API failed: {e}")
        return pd.DataFrame()


def fetch_espn_api_data(date_str):
    """Fetch from ESPN API (new method)"""
    print(f"\n{'='*80}")
    print(f"FETCHING FROM ESPN API")
    print(f"{'='*80}")
    
    try:
        # Convert date format: 2026-02-12 → 20260212
        espn_date = date_str.replace('-', '')
        
        # Step 1: Get scoreboard for date
        scoreboard_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={espn_date}"
        print(f"Fetching scoreboard: {scoreboard_url}")
        
        response = requests.get(scoreboard_url, timeout=10)
        response.raise_for_status()
        scoreboard = response.json()
        
        events = scoreboard.get('events', [])
        print(f"✅ Found {len(events)} games")
        
        if not events:
            print("❌ No games found")
            return pd.DataFrame()
        
        # Step 2: Fetch box score for each game
        all_players = []
        
        for i, event in enumerate(events, 1):
            game_id = event['id']
            competition = event['competitions'][0]
            
            # Get team info
            away_team = competition['competitors'][1]  # Away is index 1
            home_team = competition['competitors'][0]  # Home is index 0
            
            away_abbr = away_team['team']['abbreviation']
            home_abbr = home_team['team']['abbreviation']
            
            print(f"\nGame {i}/{len(events)}: {away_abbr} @ {home_abbr}")
            
            # Fetch detailed box score
            box_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
            print(f"  Fetching box score: {box_url}")
            
            box_response = requests.get(box_url, timeout=10)
            box_response.raise_for_status()
            box_data = box_response.json()
            
            # Parse players
            players = parse_espn_box_score(box_data, date_str, away_abbr, home_abbr)
            print(f"  ✅ Parsed {len(players)} players")
            all_players.extend(players)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_players)
        
        print(f"\n✅ Total players parsed: {len(df)}")
        print(f"\nColumns ({len(df.columns)}):")
        for col in df.columns:
            print(f"  - {col}")
        
        print(f"\nSample data (first player):")
        print(df.head(1).T)
        
        return df
        
    except Exception as e:
        print(f"❌ ESPN API failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def parse_espn_box_score(box_data, date_str, away_abbr, home_abbr):
    """Parse ESPN box score into NBA API format"""
    players = []
    
    boxscore = box_data.get('boxscore', {})
    if not boxscore:
        return players
    
    teams_data = boxscore.get('players', [])
    
    for team_data in teams_data:
        team_info = team_data.get('team', {})
        team_abbr = team_info.get('abbreviation', '')
        team_name = team_info.get('displayName', '')
        
        # Determine if home or away
        is_home = (team_abbr == home_abbr)
        matchup = f"{team_abbr} vs. {away_abbr}" if is_home else f"{team_abbr} @ {home_abbr}"
        
        # Get statistics section
        statistics = team_data.get('statistics', [])
        if not statistics:
            continue
        
        # Usually statistics[0] has the player stats
        stat_section = statistics[0]
        athletes = stat_section.get('athletes', [])
        
        for athlete_data in athletes:
            athlete = athlete_data.get('athlete', {})
            stats = athlete_data.get('stats', [])
            
            if not stats:
                continue
            
            # ESPN stats order (confirmed from API):
            # [0]=MIN, [1]=PTS, [2]=FG, [3]=3PT, [4]=FT, [5]=REB, [6]=AST, 
            # [7]=TO, [8]=STL, [9]=BLK, [10]=OREB, [11]=DREB, [12]=PF, [13]=+/-
            
            player_dict = {
                # IDs and names
                'PLAYER_ID': athlete.get('id', ''),
                'PLAYER_NAME': athlete.get('displayName', ''),
                'TEAM_ID': team_info.get('id', ''),
                'TEAM_NAME': team_name,
                'TEAM_ABBREVIATION': team_abbr,
                
                # Game info
                'GAME_ID': box_data.get('header', {}).get('id', ''),
                'GAME_DATE': date_str,
                'MATCHUP': matchup,
                
                # Stats (using correct order)
                'MIN': parse_minutes(stats[0]) if len(stats) > 0 else 0,
                'PTS': safe_int(stats[1]) if len(stats) > 1 else 0,
                'REB': safe_int(stats[5]) if len(stats) > 5 else 0,
                'AST': safe_int(stats[6]) if len(stats) > 6 else 0,
                'TOV': safe_int(stats[7]) if len(stats) > 7 else 0,
                'STL': safe_int(stats[8]) if len(stats) > 8 else 0,
                'BLK': safe_int(stats[9]) if len(stats) > 9 else 0,
                'OREB': safe_int(stats[10]) if len(stats) > 10 else 0,
                'DREB': safe_int(stats[11]) if len(stats) > 11 else 0,
                'PF': safe_int(stats[12]) if len(stats) > 12 else 0,
                'PLUS_MINUS': safe_int(stats[13]) if len(stats) > 13 else 0,
                
                # Field goals (index 2: "7-12" format)
                'FGM': parse_made(stats[2]) if len(stats) > 2 else 0,
                'FGA': parse_attempts(stats[2]) if len(stats) > 2 else 0,
                'FG_PCT': calculate_pct(stats[2]) if len(stats) > 2 else 0.0,
                
                # 3-pointers (index 3: "2-5" format)
                'FG3M': parse_made(stats[3]) if len(stats) > 3 else 0,
                'FG3A': parse_attempts(stats[3]) if len(stats) > 3 else 0,
                'FG3_PCT': calculate_pct(stats[3]) if len(stats) > 3 else 0.0,
                
                # Free throws (index 4: "4-4" format)
                'FTM': parse_made(stats[4]) if len(stats) > 4 else 0,
                'FTA': parse_attempts(stats[4]) if len(stats) > 4 else 0,
                'FT_PCT': calculate_pct(stats[4]) if len(stats) > 4 else 0.0,
                
                # Win/Loss (will need to determine from game result)
                'WL': 'TBD',  # Will populate this after parsing both teams
            }
            
            players.append(player_dict)
    
    return players


def parse_made(fg_string):
    """Parse '7-12' to get made (7)"""
    if not fg_string or '-' not in str(fg_string):
        return 0
    return int(str(fg_string).split('-')[0])


def parse_attempts(fg_string):
    """Parse '7-12' to get attempts (12)"""
    if not fg_string or '-' not in str(fg_string):
        return 0
    return int(str(fg_string).split('-')[1])


def calculate_pct(fg_string):
    """Parse '7-12' to calculate percentage (0.583)"""
    made = parse_made(fg_string)
    attempts = parse_attempts(fg_string)
    if attempts == 0:
        return 0.0
    return round(made / attempts, 3)


def parse_minutes(min_string):
    """Parse '35:24' to get total minutes (35)"""
    if not min_string or ':' not in str(min_string):
        return 0
    return int(str(min_string).split(':')[0])


def safe_int(value):
    """Safely parse int, handling empty/None/string values"""
    if not value or value == '':
        return 0
    try:
        # If it contains a dash, take the first number (made, not attempted)
        if '-' in str(value):
            return parse_made(value)
        return int(value)
    except (ValueError, TypeError):
        return 0


def compare_dataframes(nba_df, espn_df):
    """Compare the two DataFrames"""
    print(f"\n{'='*80}")
    print(f"COMPARISON")
    print(f"{'='*80}")
    
    print(f"\nRow counts:")
    print(f"  NBA API:  {len(nba_df)} players")
    print(f"  ESPN API: {len(espn_df)} players")
    
    if nba_df.empty or espn_df.empty:
        print("\n❌ One or both DataFrames are empty - cannot compare")
        return
    
    print(f"\nColumn comparison:")
    nba_cols = set(nba_df.columns)
    espn_cols = set(espn_df.columns)
    
    print(f"  NBA API columns: {len(nba_cols)}")
    print(f"  ESPN API columns: {len(espn_cols)}")
    
    missing_in_espn = nba_cols - espn_cols
    extra_in_espn = espn_cols - nba_cols
    
    if missing_in_espn:
        print(f"\n⚠️  Missing in ESPN (need to add):")
        for col in sorted(missing_in_espn):
            print(f"    - {col}")
    
    if extra_in_espn:
        print(f"\n✅ Extra in ESPN (can drop):")
        for col in sorted(extra_in_espn):
            print(f"    - {col}")
    
    common_cols = nba_cols & espn_cols
    print(f"\n✅ Common columns: {len(common_cols)}")
    
    # Compare data types for common columns
    print(f"\nData type comparison (common columns):")
    for col in sorted(common_cols):
        nba_type = nba_df[col].dtype
        espn_type = espn_df[col].dtype
        match = "✅" if nba_type == espn_type else "⚠️"
        print(f"  {match} {col}: NBA={nba_type}, ESPN={espn_type}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default='2026-02-12', help='Date to test (YYYY-MM-DD)')
    parser.add_argument('--season', default='2025-26', help='NBA season')
    args = parser.parse_args()
    
    print(f"Testing date: {args.date}")
    print(f"Season: {args.season}")
    
    # Fetch from both sources
    nba_df = fetch_nba_api_data(args.date, args.season)
    espn_df = fetch_espn_api_data(args.date)
    
    # Compare
    compare_dataframes(nba_df, espn_df)
    
    # Save to CSV for manual inspection
    output_dir = Path.home() / 'Downloads' / 'tmp' / 'testing_nba_api_fallback'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not nba_df.empty:
        nba_path = output_dir / 'nba_api_sample.csv'
        nba_df.to_csv(nba_path, index=False)
        print(f"\n✅ NBA API data saved to {nba_path}")
    
    if not espn_df.empty:
        espn_path = output_dir / 'espn_api_sample.csv'
        espn_df.to_csv(espn_path, index=False)
        print(f"✅ ESPN API data saved to {espn_path}")


if __name__ == '__main__':
    main()

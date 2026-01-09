"""
Find tonight's NBA player prop plays using the 3D Role-Spread-Scorer Points Model.

Strategy Overview:
This script implements the 3D Role-Spread-Scorer Points Model, which identifies betting
edges by categorizing players based on THREE factors:
    1. Player Role: Scoring tier based on their points prop line
    2. Game Spread: Team's competitive situation (favorite/underdog/pick'em)
    3. Scorer Type: Rim Attacker (≥40% points within 6ft) vs Perimeter (<40%)

The model uses historical data to find combinations where over/under hit rates
significantly deviate from baseline, creating positive expected ROI opportunities.

Context:
Takes tonight's games, bins each player by their points line tier, team spread, AND
scorer type, then matches them against pre-defined 3D strategies with proven positive ROI.

Usage:
    # Simple: Auto-loads strategies from S3 for current season
    python scripts/find_role_spread_scorer_points_model_plays.py
    
    # Specify season
    python scripts/find_role_spread_scorer_points_model_plays.py --season 2025-26
    
    # Specific date
    python scripts/find_role_spread_scorer_points_model_plays.py --date 2026-01-06
    
    # Adjust ROI threshold
    python scripts/find_role_spread_scorer_points_model_plays.py --min-roi 7.0
    
    # Show all players (not just plays)
    python scripts/find_role_spread_scorer_points_model_plays.py --show-all
    
    # Specify rim scorer threshold (must match data generation)
    python scripts/find_role_spread_scorer_points_model_plays.py --rim-scorer-pct 40
"""

import sys
from pathlib import Path

# Add project root to path
import os
import json
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

import pandas as pd
import argparse
from datetime import datetime, date

# Import player name normalization and team mapping
sys.path.insert(0, str(root / 'src'))
from player_name_utils import normalize_player_name, get_name_mappings
from team_utils import TEAM_NAME_TO_ABBR, load_player_team_cache


# =============================================================================
# STRATEGIES (from analyze_player_props_matrix.py analysis)
# =============================================================================

# These strategies are based on historical analysis with 50+ game sample sizes
# Format: (line_tier, spread_bin, bet_side, edge, roi, games)

STRATEGIES_DETAILED = {
    # =========================================================================
    # OVER STRATEGIES (Top 10 from detailed granularity analysis)
    # =========================================================================
    'high_star_small_dog_over': {
        'line_tier': '25-30 (High Star)',
        'spread_bin': '2-6 Dog',
        'bet_side': 'OVER',
        'edge': 7.3,
        'roi': 7.0,
        'games': 107,
        'hit_rate': 56.1,
    },
    'star_pickem_over': {
        'line_tier': '20-25 (Star)',
        'spread_bin': 'Pick\'em (-2 to +2)',
        'bet_side': 'OVER',
        'edge': 7.1,
        'roi': 6.7,
        'games': 93,
        'hit_rate': 55.9,
    },
    'role_player_big_dog_over': {
        'line_tier': '10-15 (Role Player)',
        'spread_bin': '10-15 Dog',
        'bet_side': 'OVER',
        'edge': 6.7,
        'roi': 5.9,
        'games': 173,
        'hit_rate': 55.5,
    },
    'star_big_fav_over': {
        'line_tier': '20-25 (Star)',
        'spread_bin': '10-15 Fav',
        'bet_side': 'OVER',
        'edge': 6.1,
        'roi': 4.8,
        'games': 52,
        'hit_rate': 54.9,
    },
    'high_star_big_dog_over': {
        'line_tier': '25-30 (High Star)',
        'spread_bin': '10-15 Dog',
        'bet_side': 'OVER',
        'edge': 5.9,
        'roi': 4.4,
        'games': 64,
        'hit_rate': 54.7,
    },
    'star_medium_fav_over': {
        'line_tier': '20-25 (Star)',
        'spread_bin': '6-10 Fav',
        'bet_side': 'OVER',
        'edge': 5.8,
        'roi': 4.1,
        'games': 121,
        'hit_rate': 54.5,
    },
    'bench_small_dog_over': {
        'line_tier': '5-10 (Bench)',
        'spread_bin': '2-6 Dog',
        'bet_side': 'OVER',
        'edge': 4.7,
        'roi': 2.1,
        'games': 379,
        'hit_rate': 53.5,
    },
    'star_medium_dog_over': {
        'line_tier': '20-25 (Star)',
        'spread_bin': '6-10 Dog',
        'bet_side': 'OVER',
        'edge': 4.6,
        'roi': 1.8,
        'games': 120,
        'hit_rate': 53.3,
    },
    'bench_medium_fav_over': {
        'line_tier': '5-10 (Bench)',
        'spread_bin': '6-10 Fav',
        'bet_side': 'OVER',
        'edge': 4.3,
        'roi': 1.3,
        'games': 328,
        'hit_rate': 53.0,
    },
    'bench_huge_dog_over': {
        'line_tier': '5-10 (Bench)',
        'spread_bin': '15+ Dog',
        'bet_side': 'OVER',
        'edge': 4.2,
        'roi': 1.1,
        'games': 51,
        'hit_rate': 52.9,
    },
    
    # =========================================================================
    # UNDER STRATEGIES (Top 10 from detailed granularity analysis)
    # =========================================================================
    'bench_pickem_under': {
        'line_tier': '5-10 (Bench)',
        'spread_bin': 'Pick\'em (-2 to +2)',
        'bet_side': 'UNDER',
        'edge': 7.3,
        'roi': 11.8,
        'games': 210,
        'hit_rate': 58.6,
    },
    'star_small_fav_under': {
        'line_tier': '20-25 (Star)',
        'spread_bin': '2-6 Fav',
        'bet_side': 'UNDER',
        'edge': 5.4,
        'roi': 8.2,
        'games': 150,
        'hit_rate': 56.7,
    },
    'bench_huge_fav_under': {
        'line_tier': '5-10 (Bench)',
        'spread_bin': '15+ Fav',
        'bet_side': 'UNDER',
        'edge': 4.8,
        'roi': 6.9,
        'games': 50,
        'hit_rate': 56.0,
    },
    'star_small_dog_under': {
        'line_tier': '20-25 (Star)',
        'spread_bin': '2-6 Dog',
        'bet_side': 'UNDER',
        'edge': 4.5,
        'roi': 6.3,
        'games': 149,
        'hit_rate': 55.7,
    },
    'bench_big_fav_under': {
        'line_tier': '5-10 (Bench)',
        'spread_bin': '10-15 Fav',
        'bet_side': 'UNDER',
        'edge': 4.2,
        'roi': 5.8,
        'games': 202,
        'hit_rate': 55.4,
    },
    'high_role_pickem_under': {
        'line_tier': '15-20 (High Role)',
        'spread_bin': 'Pick\'em (-2 to +2)',
        'bet_side': 'UNDER',
        'edge': 4.0,
        'roi': 5.4,
        'games': 154,
        'hit_rate': 55.2,
    },
    'high_role_big_dog_under': {
        'line_tier': '15-20 (High Role)',
        'spread_bin': '10-15 Dog',
        'bet_side': 'UNDER',
        'edge': 3.4,
        'roi': 4.3,
        'games': 108,
        'hit_rate': 54.6,
    },
    'bench_big_dog_under': {
        'line_tier': '5-10 (Bench)',
        'spread_bin': '10-15 Dog',
        'bet_side': 'UNDER',
        'edge': 2.8,
        'roi': 3.2,
        'games': 164,
        'hit_rate': 54.0,
    },
    'high_role_small_dog_under': {
        'line_tier': '15-20 (High Role)',
        'spread_bin': '2-6 Dog',
        'bet_side': 'UNDER',
        'edge': 2.6,
        'roi': 2.7,
        'games': 264,
        'hit_rate': 53.8,
    },
    'bench_small_fav_under': {
        'line_tier': '5-10 (Bench)',
        'spread_bin': '2-6 Fav',
        'bet_side': 'UNDER',
        'edge': 2.4,
        'roi': 2.4,
        'games': 402,
        'hit_rate': 53.6,
    },
}

STRATEGIES_STANDARD = {
    # TODO: Add standard strategies when we run standard granularity analysis
    # For now, just use detailed as default
}


# =============================================================================
# BINNING FUNCTIONS (must match analyze_player_props_matrix.py)
# =============================================================================

def bin_points_line(line, granularity='detailed'):
    """Bin player points line into tiers"""
    if pd.isna(line):
        return 'Unknown'
    
    if granularity == 'standard':
        if line < 10:
            return '<10 (Bench)'
        elif line < 20:
            return '10-20 (Role)'
        elif line < 30:
            return '20-30 (Star)'
        else:
            return '30+ (Superstar)'
    else:  # detailed
        if line < 5:
            return '<5 (Deep Bench)'
        elif line < 10:
            return '5-10 (Bench)'
        elif line < 15:
            return '10-15 (Role Player)'
        elif line < 20:
            return '15-20 (High Role)'
        elif line < 25:
            return '20-25 (Star)'
        elif line < 30:
            return '25-30 (High Star)'
        else:
            return '30+ (Superstar)'


def bin_team_spread(spread, granularity='detailed'):
    """
    Bin team spread into categories
    
    Args:
        spread: Team spread (positive = underdog, negative = favorite)
    """
    if pd.isna(spread):
        return 'Unknown'
    
    if granularity == 'standard':
        if spread < -5:
            return 'Favorite'
        elif spread <= 5:
            return 'Pick\'em'
        else:
            return 'Underdog'
    else:  # detailed
        if spread < -15:
            return '15+ Fav'
        elif spread < -10:
            return '10-15 Fav'
        elif spread < -6:
            return '6-10 Fav'
        elif spread < -2:
            return '2-6 Fav'
        elif spread <= 2:
            return 'Pick\'em (-2 to +2)'
        elif spread <= 6:
            return '2-6 Dog'
        elif spread <= 10:
            return '6-10 Dog'
        elif spread <= 15:
            return '10-15 Dog'
        else:
            return '15+ Dog'


# =============================================================================
# DATA LOADING
# =============================================================================

def load_strategies(json_file, min_roi=5.0):
    """
    Load strategies from JSON file (REQUIRED)
    
    Args:
        json_file: Path to strategies JSON file (REQUIRED)
        min_roi: Minimum ROI threshold to filter strategies
    
    Returns:
        Dict of strategies {strategy_key: strategy_data}
    """
    if not json_file:
        raise ValueError(
            "❌ --strategies-json or --season is REQUIRED\n"
            "   Generate strategies with: python analysis/analyze_points_props_role_spread_model.py --season 2025-26"
        )
    
    # Load from JSON file or S3
    print(f"📊 Loading strategies from {json_file}...")
    try:
        import json
        
        # Check if S3 URI
        if str(json_file).startswith('s3://'):
            # Parse S3 URI
            s3_uri = str(json_file)
            parts = s3_uri.replace('s3://', '').split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ''
            
            # Load from S3
            import boto3
            from io import BytesIO
            s3 = boto3.client('s3')
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj['Body'].read().decode('utf-8'))
        else:
            # Load from local file
            with open(json_file, 'r') as f:
                data = json.load(f)
        
        all_strategies = data['strategies']
        print(f"   ✅ Loaded {len(all_strategies)} total strategies")
        print(f"   📅 Generated: {data.get('generated_at', 'Unknown')}")
        print(f"   📈 Data through: {data.get('data_through', 'Unknown')}")
        
        # Filter by min_roi
        filtered_strategies = [s for s in all_strategies if s['roi'] >= min_roi]
        print(f"   🔍 Filtered to {len(filtered_strategies)} strategies with ROI >= {min_roi}%")
        
        if len(filtered_strategies) == 0:
            print(f"   ⚠️  WARNING: No strategies meet ROI threshold of {min_roi}%")
        
        # Convert to dict format {strategy_key: strategy_data}
        strategies = {}
        for i, strat in enumerate(filtered_strategies):
            key = f"strat_{i}"
            strategies[key] = strat
        
        return strategies
        
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ Strategies file not found: {json_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ Invalid JSON in strategies file: {e}")
    except Exception as e:
        raise Exception(f"❌ Error loading strategies: {e}")


def load_player_scorer_types(season='2025-26', rim_scorer_pct=40):
    """
    Load player scorer type classifications from S3 merged data
    
    Args:
        season: NBA season
        rim_scorer_pct: Threshold used for rim scorer classification
    
    Returns:
        dict: {player_name: scorer_type}
    """
    print(f"\n📊 Loading player scorer type data...")
    
    # Load from S3 merged data
    s3_key = f"data/03_intermediate/player_props_with_actuals_{season}_rim{rim_scorer_pct}.csv"
    s3_uri = f"s3://nba-betting-mt/{s3_key}"
    
    try:
        import boto3
        from io import BytesIO
        
        s3 = boto3.client('s3')
        bucket = 'nba-betting-mt'
        obj = s3.get_object(Bucket=bucket, Key=s3_key)
        
        # Read CSV
        df = pd.read_csv(BytesIO(obj['Body'].read()))
        
        # Get unique player-scorer_type mappings
        if 'scorer_type' not in df.columns:
            print(f"   ⚠️  WARNING: scorer_type column not found in data")
            return {}
        
        # Create mapping (take most recent scorer_type for each player)
        scorer_map = df[['PLAYER_NAME', 'scorer_type']].dropna().drop_duplicates('PLAYER_NAME').set_index('PLAYER_NAME')['scorer_type'].to_dict()
        
        # Count by type
        rim_count = sum(1 for v in scorer_map.values() if 'Rim' in str(v))
        perim_count = sum(1 for v in scorer_map.values() if 'Perimeter' in str(v))
        
        print(f"   ✅ Loaded scorer types for {len(scorer_map)} players")
        print(f"      Rim Attackers (≥{rim_scorer_pct}%): {rim_count}")
        print(f"      Perimeter (<{rim_scorer_pct}%): {perim_count}")
        
        return scorer_map
        
    except Exception as e:
        print(f"   ⚠️  Error loading scorer type data: {e}")
        print(f"   Continuing without scorer type matching...")
        return {}


def load_tonights_games(target_date=None, use_s3=False):
    """
    Load tonight's games with player props and team spreads
    
    Uses The Odds API to fetch:
    1. Today's NBA games
    2. Player points props for each game  
    3. Game spreads for each team
    
    Also loads player-team mapping from historical game data
    
    Returns:
        DataFrame with columns: PLAYER_NAME, points_line, team_abbr, team_spread, opponent
    """
    import requests
    import ssl
    import urllib3
    from datetime import datetime
    from zoneinfo import ZoneInfo
    
    # Disable SSL warnings (common with macOS)
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Load API key
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv('ODDS_API_KEY')
    
    if not api_key or api_key == 'your_api_key_here':
        print("❌ No valid API key found!")
        print("Get your API key at: https://the-odds-api.com/")
        print("Add it to .env file as: ODDS_API_KEY=your_key")
        # print("\n⚠️  Using mock data for now...\n")
        
        # Return mock data
        return -99
    
    # Load player-team mapping from cache
    print(f"📋 Loading player-team mapping from {'S3' if use_s3 else 'local cache'}...")
    try:
        if use_s3:
            # Load from S3
            import boto3
            from io import BytesIO
            s3 = boto3.client('s3')
            bucket = 'nba-betting-mt'
            key = 'data/02_cache/player_team_cache.csv'
            
            obj = s3.get_object(Bucket=bucket, Key=key)
            df_cache = pd.read_csv(BytesIO(obj['Body'].read()))
            
            player_team_map = dict(zip(df_cache['player_normalized'], df_cache['team']))
            cache_timestamp = df_cache['timestamp'].iloc[0] if len(df_cache) > 0 else None
            
            print(f"   ✅ Loaded {len(player_team_map):,} player-team mappings from S3")
            if cache_timestamp:
                print(f"   📅 Cache timestamp: {cache_timestamp}\n")
        else:
            # Load from local using utility function
            cache_data = load_player_team_cache()
            player_team_map = cache_data['mapping']
            cache_timestamp = cache_data['timestamp']
            
            if player_team_map:
                print(f"   ✅ Loaded {len(player_team_map):,} player-team mappings from local cache")
                if cache_timestamp:
                    print(f"   📅 Cache timestamp: {cache_timestamp}\n")
                else:
                    print()
            else:
                print(f"   ⚠️  No player-team mapping available in cache")
                print(f"   Will skip players without known teams\n")
    except Exception as e:
        print(f"   ⚠️  Error loading player-team mapping: {e}")
        print(f"   Will skip players without known teams\n")
        player_team_map = {}
    
    # API Configuration
    API_BASE_URL = 'https://api.the-odds-api.com/v4'
    SPORT = 'basketball_nba'
    TIMEZONE = 'America/New_York'
    
    print(f"📊 Fetching today's NBA games and player props...")
    print(f"📅 {datetime.now(ZoneInfo(TIMEZONE)).strftime('%Y-%m-%d %H:%M:%S ET')}\n")
    
    try:
        # Step 1: Get today's NBA events
        url = f"{API_BASE_URL}/sports/{SPORT}/events"
        params = {'apiKey': api_key}
        response = requests.get(url, params=params, verify=False)
        response.raise_for_status()
        
        events = response.json()
        
        # Filter for today's games
        tz = ZoneInfo(TIMEZONE)
        now = datetime.now(tz)
        today = now.date()
        
        todays_events = []
        for event in events:
            event_time_utc = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
            event_time_local = event_time_utc.astimezone(tz)
            
            if event_time_local.date() == today:
                todays_events.append(event)
        
        if not todays_events:
            print("❌ No NBA games found for today")
            print("⚠️  Using mock data...\n")
            return pd.DataFrame({
                'PLAYER_NAME': ['Jalen Brunson', 'Karl-Anthony Towns'],
                'points_line': [28.5, 23.5],
                'team_abbr': ['NYK', 'NYK'],
                'team_spread': [3.0, 3.0],
                'opponent': ['BOS', 'BOS'],
            })
        
        print(f"✅ Found {len(todays_events)} games today\n")
        
        # Step 2: Fetch player props AND spreads for each game
        all_player_data = []
        unmapped_players = []  # Track players without team mapping
        game_info = []  # Store game metadata (tip time, teams, spreads)
        
        for i, event in enumerate(todays_events, 1):
            event_id = event['id']
            away_team = event['away_team']
            home_team = event['home_team']
            event_time_utc = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
            event_time_local = event_time_utc.astimezone(tz)
            
            print(f"📥 Game {i}/{len(todays_events)}: {away_team} @ {home_team}")
            
            # Get player points props
            url = f"{API_BASE_URL}/sports/{SPORT}/events/{event_id}/odds"
            params = {
                'apiKey': api_key,
                'regions': 'us',
                'markets': 'player_points,spreads',  # Get both player props and spreads
                'oddsFormat': 'american',
            }
            
            try:
                response = requests.get(url, params=params, verify=False)
                response.raise_for_status()
                odds_data = response.json()
                
                # Save raw API response to S3 for debugging/review
                try:
                    import boto3
                    from io import StringIO
                    
                    s3 = boto3.client('s3')
                    bucket = 'the-odds-api-mt'
                    
                    # Create timestamp in ET for filename
                    et_tz = ZoneInfo('America/New_York')
                    timestamp = datetime.now(et_tz).strftime('%Y%m%d_%H%M%S')
                    
                    # Save to live_game_odds/ with timestamp and game info
                    game_slug = f"{away_team.replace(' ', '_')}_at_{home_team.replace(' ', '_')}"
                    key = f"nba/live_game_odds/{timestamp}_{game_slug}.json"
                    
                    s3.put_object(
                        Bucket=bucket,
                        Key=key,
                        Body=json.dumps(odds_data, indent=2),
                        ContentType='application/json'
                    )
                    print(f"   💾 Saved raw odds to s3://{bucket}/{key}")
                except Exception as e:
                    print(f"   ⚠️  Failed to save raw odds to S3: {e}")
                
                # Extract spreads first (to map teams to spreads)
                team_spreads = {}
                for bookmaker in odds_data.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'spreads':
                            for outcome in market.get('outcomes', []):
                                team_name = outcome['name']
                                spread = outcome.get('point', 0)
                                # Use first spread found for each team
                                if team_name not in team_spreads:
                                    team_spreads[team_name] = spread
                
                # Store game info for later use
                away_abbr_temp = TEAM_NAME_TO_ABBR.get(away_team, away_team)
                home_abbr_temp = TEAM_NAME_TO_ABBR.get(home_team, home_team)
                game_info.append({
                    'away_team': away_abbr_temp,
                    'home_team': home_abbr_temp,
                    'away_spread': team_spreads.get(away_team, 0),
                    'home_spread': team_spreads.get(home_team, 0),
                    'game_time': event_time_local,
                })
                
                # Extract player props
                for bookmaker in odds_data.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'player_points':
                            # Group by (player, line) to deduplicate
                            player_lines_seen = set()
                            
                            for outcome in market.get('outcomes', []):
                                player = outcome.get('description', 'Unknown')
                                line = outcome.get('point')
                                
                                key = (player, line)
                                if key in player_lines_seen:
                                    continue
                                player_lines_seen.add(key)
                                
                                # Normalize player name to match cache
                                player_normalized = normalize_player_name(player)
                                
                                # Apply name mappings (Odds API → NBA API)
                                name_mappings = get_name_mappings()
                                player_normalized = name_mappings.get(player_normalized, player_normalized)
                                
                                # Determine player's team using cache mapping
                                player_team_abbr = player_team_map.get(player_normalized)
                                
                                # Convert API's full team names to abbreviations for comparison
                                away_abbr = TEAM_NAME_TO_ABBR.get(away_team, away_team)
                                home_abbr = TEAM_NAME_TO_ABBR.get(home_team, home_team)
                                
                                if player_team_abbr and player_team_abbr in [away_abbr, home_abbr]:
                                    # We know the team from cache
                                    opponent_abbr = home_abbr if player_team_abbr == away_abbr else away_abbr
                                    
                                    # Get spread for player's team (API returns spreads with full names)
                                    player_team_full = away_team if player_team_abbr == away_abbr else home_team
                                    spread = team_spreads.get(player_team_full, 0)
                                    
                                    # Get opponent full name
                                    opponent_full = home_team if player_team_abbr == away_abbr else away_team
                                    
                                    # Get odds (price) - typically in American format (e.g., -110, +120)
                                    odds = outcome.get('price', -110)  # Default to -110 if missing
                                    bet_side = outcome.get('name', 'Unknown')  # 'Over' or 'Under'
                                    
                                    all_player_data.append({
                                        'PLAYER_NAME': player,
                                        'points_line': line,
                                        'team_abbr': player_team_abbr,
                                        'team_spread': spread,
                                        'opponent': opponent_abbr,
                                        'game_time': event_time_local,
                                        'bookmaker': bookmaker['title'],  # Track bookmaker
                                        'odds': odds,  # Store odds for detailed display
                                        'bet_side': bet_side,  # 'Over' or 'Under'
                                    })
                                else:
                                    # Track unmapped player (cache might be outdated or player recently traded)
                                    unmapped_players.append({
                                        'player': player,
                                        'normalized': player_normalized,
                                        'game': f"{away_team} @ {home_team}",
                                        'in_cache': player_normalized in player_team_map,
                                        'cached_team': player_team_map.get(player_normalized, 'N/A'),
                                    })
                
                print(f"   ✅ Found props and spreads")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        if not all_player_data:
            print("\n❌ No player props found with team mapping")
            print("⚠️  Using mock data...\n")
            return pd.DataFrame({
                'PLAYER_NAME': ['Jalen Brunson', 'Karl-Anthony Towns'],
                'points_line': [28.5, 23.5],
                'team_abbr': ['NYK', 'NYK'],
                'team_spread': [3.0, 3.0],
                'opponent': ['BOS', 'BOS'],
            })
        
        df = pd.DataFrame(all_player_data)
        
        # Remove duplicates (same player might appear in multiple bookmaker markets)
        # Include bet_side since we now track Over/Under separately
        df = df.drop_duplicates(subset=['PLAYER_NAME', 'points_line', 'team_abbr', 'bet_side'])
        
        total_props_before_consensus = len(df)
        total_players_before_consensus = df['PLAYER_NAME'].nunique()
        print(f"   Raw props before consensus: {total_props_before_consensus} props for {total_players_before_consensus} players")
        
        # Get consensus line for each player (median line)
        # This removes the issue of multiple lines per player (21.5, 22.5, 23.5, etc.)
        df_consensus = df.groupby('PLAYER_NAME', as_index=False).agg({
            'points_line': 'median',  # Use median line as consensus
            'team_abbr': 'first',
            'team_spread': 'first',
            'opponent': 'first',
            'game_time': 'first',
        })
        
        # For each consensus line, find which bookmakers offer that exact line
        # This helps users know where to actually place the bet
        def get_bookmakers_for_consensus(player_name, consensus_line):
            """
            Find all bookmakers offering lines within ±0.5 of consensus line.
            This handles cases where median falls between bookmaker lines (e.g., median=12.5 but books offer 12.0 and 13.0)
            Returns: Comma-separated list of bookmaker names (for backward compatibility)
            """
            player_rows = df[df['PLAYER_NAME'] == player_name]
            # Accept lines within ±0.5 of consensus (e.g., if consensus=12.5, accept 12.0, 12.5, 13.0)
            matching_rows = player_rows[abs(player_rows['points_line'] - consensus_line) <= 0.5]
            books = matching_rows['bookmaker'].unique().tolist()
            return ', '.join(sorted(books)) if books else ''
        
        def get_bookmaker_details_for_side(player_name, consensus_line, side):
            """
            Get detailed bookmaker info (name, line, odds) for ONE side only.
            
            Args:
                player_name: Player name
                consensus_line: Median line
                side: 'Over' or 'Under'
            
            Returns: JSON string with list of {bookmaker, line, odds} dicts
            """
            import json
            player_rows = df[df['PLAYER_NAME'] == player_name]
            # Filter by side FIRST, then check line proximity
            side_rows = player_rows[player_rows['bet_side'].str.upper() == side.upper()]
            matching_rows = side_rows[abs(side_rows['points_line'] - consensus_line) <= 0.5]
            
            # Build list of {bookmaker, line, odds}
            details = []
            for _, row in matching_rows.iterrows():
                details.append({
                    'bookmaker': row['bookmaker'],
                    'line': row['points_line'],
                    'odds': row['odds']
                })
            
            # Sort by bookmaker name
            details = sorted(details, key=lambda x: x['bookmaker'])
            
            return json.dumps(details) if details else '[]'
        
        df_consensus['bookmakers'] = df_consensus.apply(
            lambda row: get_bookmakers_for_consensus(row['PLAYER_NAME'], row['points_line']),
            axis=1
        )
        
        # Create separate columns for Over and Under odds
        df_consensus['bookmaker_details_over'] = df_consensus.apply(
            lambda row: get_bookmaker_details_for_side(row['PLAYER_NAME'], row['points_line'], 'Over'),
            axis=1
        )
        
        df_consensus['bookmaker_details_under'] = df_consensus.apply(
            lambda row: get_bookmaker_details_for_side(row['PLAYER_NAME'], row['points_line'], 'Under'),
            axis=1
        )
        
        # Count bookmakers
        df_consensus['num_bookmakers'] = df_consensus['bookmakers'].apply(
            lambda x: len([b for b in x.split(', ') if b])
        )
        
        # Validate: every consensus line must have at least one bookmaker
        assert (df_consensus['bookmakers'] != '').all(), "Empty bookmakers found - consensus lines must come from bookmakers"
        assert (df_consensus['num_bookmakers'] > 0).all(), "No bookmakers found for consensus lines"
        
        players_mapped = df_consensus['PLAYER_NAME'].nunique()
        total_props = len(df_consensus)
        total_unmapped = len(set(u['normalized'] for u in unmapped_players))
        
        print(f"\n✅ Loaded {total_props} player props from {len(todays_events)} games (consensus lines)")
        print(f"   Successfully mapped: {players_mapped} players")
        if total_unmapped > 0:
            print(f"   Unmapped (skipped): {total_unmapped} players")
        
        # Log unmapped players if any
        if unmapped_players:
            # Deduplicate unmapped players
            unmapped_df = pd.DataFrame(unmapped_players)
            unmapped_df = unmapped_df.drop_duplicates(subset=['normalized'])
            
            print(f"\n⚠️  {len(unmapped_df)} players without team mapping:")
            for _, row in unmapped_df.iterrows():
                if row['in_cache']:
                    # Player is in cache but team doesn't match this game (likely traded/wrong mapping)
                    print(f"   - {row['player']} (cached as {row['cached_team']}, not in {row['game']})")
                else:
                    # Player not in cache at all
                    print(f"   - {row['player']} (not in cache, game: {row['game']})")
        
        print()
        
        return df_consensus, game_info
        
    except Exception as e:
        print(f"\n❌ Error loading games: {e}")
        print("⚠️  Using mock data...\n")
        import traceback
        traceback.print_exc()
        return pd.DataFrame({
            'PLAYER_NAME': ['Jalen Brunson', 'Karl-Anthony Towns'],
            'points_line': [28.5, 23.5],
            'team_abbr': ['NYK', 'NYK'],
            'team_spread': [3.0, 3.0],
            'opponent': ['BOS', 'BOS'],
        })


# =============================================================================
# PLAY FINDING
# =============================================================================

def find_plays(df_games, strategies, scorer_map=None, granularity='detailed'):
    """
    Find betting plays by matching games to strategies (3D: includes scorer_type)
    
    Args:
        df_games: DataFrame with today's player props
        strategies: Dict of strategies to match against
        scorer_map: Dict of {player_name: scorer_type} (optional for 3D matching)
        granularity: 'standard' or 'detailed'
    
    Returns:
        DataFrame with plays and reasoning
    """
    if not strategies:
        print(f"No strategies provided")
        return pd.DataFrame()
    
    # Bin each player/team
    df_games['line_tier'] = df_games['points_line'].apply(lambda x: bin_points_line(x, granularity))
    df_games['spread_bin'] = df_games['team_spread'].apply(lambda x: bin_team_spread(x, granularity))
    
    # Add scorer_type to df_games if scorer_map is provided
    if scorer_map:
        df_games['scorer_type'] = df_games['PLAYER_NAME'].map(scorer_map)
    
    plays = []
    
    for idx, row in df_games.iterrows():
        player = row['PLAYER_NAME']
        line = row['points_line']
        line_tier = row['line_tier']
        spread_bin = row['spread_bin']
        team = row['team_abbr']
        opp = row['opponent']
        spread = row['team_spread']
        player_scorer_type = row.get('scorer_type', None) if scorer_map else None
        game_time = row['game_time']
        bookmakers = row['bookmakers']
        num_bookmakers = row['num_bookmakers']
        bookmaker_details_over = row['bookmaker_details_over']
        bookmaker_details_under = row['bookmaker_details_under']
        
        # Check if this combination matches any strategy
        for strat_name, strat in strategies.items():
            # Match on line_tier, spread_bin, and scorer_type (if provided)
            line_match = strat['line_tier'] == line_tier
            spread_match = strat['spread_bin'] == spread_bin
            
            # For 3D strategies, also match on scorer_type
            scorer_match = True
            if 'scorer_type' in strat:
                if player_scorer_type:
                    scorer_match = strat['scorer_type'] == player_scorer_type
                else:
                    # Skip strategy if it requires scorer_type but player doesn't have it
                    continue
            
            if line_match and spread_match and scorer_match:
                # Generate strategy display name from bins + scorer_type + bet side
                if 'scorer_type' in strat:
                    strategy_display_name = f"{strat['line_tier']} + {strat['spread_bin']} + {strat['scorer_type']} {strat['bet_side']}"
                else:
                    strategy_display_name = f"{strat['line_tier']} + {strat['spread_bin']} {strat['bet_side']}"
                
                # Select correct bookmaker_details based on bet side
                if strat['bet_side'] == 'OVER':
                    bookmaker_details = bookmaker_details_over
                else:
                    bookmaker_details = bookmaker_details_under
                
                play_data = {
                    'player': player,
                    'line': line,
                    'bet_side': strat['bet_side'],
                    'team': team,
                    'opponent': opp,
                    'spread': spread,
                    'line_tier': line_tier,
                    'spread_bin': spread_bin,
                    'strategy_key': strat_name,
                    'strategy_name': strategy_display_name,
                    'strategy_roi': strat['roi'],
                    'strategy_edge': strat['edge'],
                    'strategy_hit_rate': strat['hit_rate'],
                    'strategy_games': strat['games'],
                    'reason': f"{strat['bet_side']} - {line_tier} in {spread_bin} games ({strat['edge']:+.1f}% edge, {strat['roi']:+.1f}% ROI, {strat['games']} games)",
                    'game_time': game_time,
                    'bookmakers': bookmakers,
                    'num_bookmakers': num_bookmakers,
                    'bookmaker_details': bookmaker_details,
                }
                
                # Add scorer_type if present
                if 'scorer_type' in strat:
                    play_data['scorer_type'] = strat['scorer_type']
                    play_data['reason'] = f"{strat['bet_side']} - {line_tier} + {strat['scorer_type']} in {spread_bin} games ({strat['edge']:+.1f}% edge, {strat['roi']:+.1f}% ROI, {strat['games']} games)"
                
                plays.append(play_data)
    
    return pd.DataFrame(plays)


# =============================================================================
# OUTPUT FORMATTING & SAVING
# =============================================================================

def save_plays_to_s3(df_plays, target_date, season='2025-26'):
    """
    Save plays to S3 as CSV for tracking performance (3D: includes scorer_type)
    
    Args:
        df_plays: DataFrame with plays
        target_date: Date string (YYYY-MM-DD)
        season: NBA season
    """
    if df_plays.empty:
        print(f"\n💾 No plays to save for {target_date}")
        return
    
    # Prepare CSV columns (use actual column names from find_plays)
    base_columns = [
        'player', 'team', 'opponent', 'bet_side', 'line', 'spread',
        'line_tier', 'spread_bin'
    ]
    
    # Add scorer_type if present
    if 'scorer_type' in df_plays.columns:
        base_columns.append('scorer_type')
    
    base_columns.extend([
        'strategy_name', 'strategy_roi', 'strategy_edge', 
        'strategy_hit_rate', 'strategy_games'
    ])
    
    # Add game_time, bookmakers, bookmaker_details (always present from API fetch)
    base_columns.extend(['game_time', 'bookmakers', 'num_bookmakers', 'bookmaker_details'])
    
    csv_data = df_plays[base_columns].copy()
    
    # Rename columns for clarity in saved CSV
    csv_data = csv_data.rename(columns={
        'strategy_roi': 'expected_roi',
        'strategy_edge': 'edge_vs_baseline',
        'strategy_hit_rate': 'hit_rate',
        'strategy_games': 'games_in_sample'
    })
    
    # Calculate edge vs breakeven (52.38% for -110 odds)
    csv_data['edge_vs_breakeven'] = csv_data['hit_rate'] - 52.38
    
    # Add metadata
    csv_data.insert(0, 'date', target_date)
    csv_data.insert(1, 'season', season)
    
    # S3 path - save to 3d/ subfolder
    bucket = 'nba-betting-mt'
    key = f'data/04_output/plays/role_spread_points_model/3d/{target_date}.csv'
    
    # Backup filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_key = f'data/04_output/plays/role_spread_points_model/3d/{target_date}_backup_{timestamp}.csv'
    
    try:
        import boto3
        from io import StringIO
        
        s3 = boto3.client('s3')
        
        # Check if file already exists, if so create backup
        try:
            s3.head_object(Bucket=bucket, Key=key)
            # File exists, copy it to backup
            s3.copy_object(
                Bucket=bucket,
                CopySource={'Bucket': bucket, 'Key': key},
                Key=backup_key
            )
            print(f"\n💾 Created backup: s3://{bucket}/{backup_key}")
        except s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                # File doesn't exist, no backup needed
                pass
            else:
                raise
        
        # Save new file
        csv_buffer = StringIO()
        csv_data.to_csv(csv_buffer, index=False)
        
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        
        print(f"💾 Saved {len(csv_data)} plays to S3: s3://{bucket}/{key}")
        
    except Exception as e:
        print(f"\n⚠️  Failed to save plays to S3: {e}")


def print_plays(df_plays, all_games_info):
    """Pretty print plays grouped by game, showing ALL games"""
    
    print(f"\n{'='*80}")
    print(f"🎯 TONIGHT'S PLAYS ({date.today()})")
    print(f"{'='*80}\n")
    
    # Sort games by tip time
    all_games_info = sorted(all_games_info, key=lambda g: g['game_time'])
    
    if len(df_plays) == 0:
        print("❌ No plays found matching strategies\n")
        # Still show all games even if no plays
        for game_num, game in enumerate(all_games_info, 1):
            tip_time = game['game_time'].strftime('%I:%M %p ET')
            print(f"{'='*80}")
            print(f"🏀 GAME {game_num}/{len(all_games_info)}: {game['away_team']} @ {game['home_team']}")
            print(f"⏰ Tip: {tip_time}")
            print(f"{'='*80}\n")
            print(f"📍 {game['away_team']}: No plays")
            print(f"🏠 {game['home_team']}: No plays\n")
        return
    
    # Add game_teams column for grouping
    def get_game_teams(row):
        teams = sorted([row['team'], row['opponent']])
        return tuple(teams)
    
    df_plays['game_teams'] = df_plays.apply(get_game_teams, axis=1)
    
    # Create a map of plays by game
    plays_by_game = {}
    for game_teams, group in df_plays.groupby('game_teams'):
        plays_by_game[game_teams] = group
    
    # Show all games (even ones without plays), sorted by tip time
    for game_num, game in enumerate(all_games_info, 1):
        away_team = game['away_team']
        home_team = game['home_team']
        tip_time = game['game_time'].strftime('%I:%M %p ET')
        game_teams = tuple(sorted([away_team, home_team]))
        
        print(f"{'='*80}")
        print(f"🏀 GAME {game_num}/{len(all_games_info)}: {away_team} @ {home_team}")
        print(f"⏰ Tip: {tip_time}")
        print(f"{'='*80}\n")
        
        # Check if this game has any plays
        if game_teams not in plays_by_game:
            print(f"📍 {away_team}: No plays")
            print(f"🏠 {home_team}: No plays\n")
            continue
        
        game_plays = plays_by_game[game_teams]
        
        # Group plays by team
        away_plays = game_plays[game_plays['team'] == away_team]
        home_plays = game_plays[game_plays['team'] == home_team]
        
        # Show away team plays
        if len(away_plays) > 0:
            print(f"📍 {away_team} plays ({len(away_plays)}):")
            print(f"{'─'*80}")
            
            # Sort by ROI descending within team
            away_plays = away_plays.sort_values('strategy_roi', ascending=False)
            
            for idx, play in away_plays.iterrows():
                breakeven_rate = 52.38
                edge_vs_breakeven = play['strategy_hit_rate'] - breakeven_rate
                
                print(f"🔥 {play['bet_side']}: {play['player']} {play['line']} pts")
                print(f"   Strategy: {play['strategy_name']}")
                print(f"   ├─ Historical: {play['strategy_hit_rate']:.1f}% hit rate ({play['strategy_games']} games)")
                print(f"   ├─ Edge vs Baseline: {play['strategy_edge']:+.1f}% | Edge vs Breakeven: {edge_vs_breakeven:+.1f}%")
                print(f"   └─ Expected ROI: {play['strategy_roi']:+.1f}%")
                print()
        else:
            print(f"📍 {away_team}: No plays\n")
        
        # Show home team plays
        if len(home_plays) > 0:
            print(f"🏠 {home_team} plays ({len(home_plays)}):")
            print(f"{'─'*80}")
            
            # Sort by ROI descending within team
            home_plays = home_plays.sort_values('strategy_roi', ascending=False)
            
            for idx, play in home_plays.iterrows():
                breakeven_rate = 52.38
                edge_vs_breakeven = play['strategy_hit_rate'] - breakeven_rate
                
                print(f"🔥 {play['bet_side']}: {play['player']} {play['line']} pts")
                print(f"   Strategy: {play['strategy_name']}")
                print(f"   ├─ Historical: {play['strategy_hit_rate']:.1f}% hit rate ({play['strategy_games']} games)")
                print(f"   ├─ Edge vs Baseline: {play['strategy_edge']:+.1f}% | Edge vs Breakeven: {edge_vs_breakeven:+.1f}%")
                print(f"   └─ Expected ROI: {play['strategy_roi']:+.1f}%")
                print()
        else:
            print(f"🏠 {home_team}: No plays\n")
        
        print()
    
    print(f"{'='*80}")
    print(f"Total plays: {len(df_plays)}")
    print(f"Avg ROI: {df_plays['strategy_roi'].mean():.1f}%")
    print(f"{'='*80}\n")


def print_no_plays_reasoning(df_games, strategies, granularity='detailed'):
    """Show why we don't have plays for certain games"""
    
    df_games['line_tier'] = df_games['points_line'].apply(lambda x: bin_points_line(x, granularity))
    df_games['spread_bin'] = df_games['team_spread'].apply(lambda x: bin_team_spread(x, granularity))
    
    print(f"\n{'='*80}")
    print("🔍 ALL PLAYERS & STRATEGY MATCHES")
    print(f"{'='*80}\n")
    
    for idx, row in df_games.iterrows():
        player = row['PLAYER_NAME']
        line = row['points_line']
        line_tier = row['line_tier']
        spread_bin = row['spread_bin']
        
        # Check for matches
        matches = []
        for strat_name, strat in strategies.items():
            if strat['line_tier'] == line_tier and strat['spread_bin'] == spread_bin:
                matches.append(strat)
        
        if matches:
            print(f"✅ {player} {line} pts")
            print(f"   Category: {line_tier} + {spread_bin}")
            for strat in matches:
                print(f"   → {strat['bet_side']}: {strat['edge']:+.1f}% edge, {strat['roi']:+.1f}% ROI")
        else:
            print(f"❌ {player} {line} pts")
            print(f"   Category: {line_tier} + {spread_bin}")
            print(f"   → No strategy for this combination")
        print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Find tonight\'s NBA player prop plays (3D: with scorer type)')
    parser.add_argument('--season', type=str, default='2025-26',
                       help='NBA season (e.g., 2025-26). Auto-generates S3 path for strategies.')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--granularity', choices=['standard', 'detailed'], default='detailed',
                       help='Binning granularity (detailed = 7x9 grid, standard = 4x6 grid)')
    parser.add_argument('--strategies-json', type=str, default=None,
                       help='Path to strategies JSON (local or S3 URI). If not provided, loads from S3 using --season.')
    parser.add_argument('--min-roi', type=float, default=5.0,
                       help='Minimum ROI threshold (default: 5.0%%)')
    parser.add_argument('--rim-scorer-pct', type=int, default=40,
                       help='Rim scorer threshold percentage (default: 40). Must match data generation.')
    parser.add_argument('--s3', action='store_true',
                       help='Load player-team cache from S3')
    parser.add_argument('--save-s3', action='store_true', default=True,
                       help='Save plays to S3 for tracking (default: True)')
    parser.add_argument('--show-all', action='store_true',
                       help='Show reasoning for all players, not just plays')
    
    args = parser.parse_args()
    
    # Auto-generate S3 path if strategies-json not provided
    if not args.strategies_json:
        args.strategies_json = f's3://nba-betting-mt/data/03_intermediate/points_by_role_gamespread_6feet_strategies_{args.season}_rim{args.rim_scorer_pct}.json'
        print(f"💡 Using 3D strategies from: {args.strategies_json}")
    
    # Load strategies
    strategies = load_strategies(args.strategies_json, args.min_roi)
    
    # Load player scorer types for 3D matching
    scorer_map = load_player_scorer_types(args.season, args.rim_scorer_pct)
    
    # Load data
    print(f"\n📊 Loading games for {args.date or 'today'}...")
    result = load_tonights_games(args.date, use_s3=args.s3)
    
    # Unpack result
    if isinstance(result, tuple):
        df_games, all_games_info = result
    else:
        # Fallback for mock data (doesn't return game info)
        df_games = result
        all_games_info = []
    
    print(f"   Found {len(df_games)} players with props\n")
    
    # Find plays using loaded strategies (with scorer_type matching)
    df_plays = find_plays(df_games, strategies, scorer_map=scorer_map, granularity=args.granularity)
    
    # Note: ROI filtering already happened in load_strategies()
    # Show filter info if strategies were filtered
    if args.strategies_json:
        print(f"🔍 Using strategies with ROI >= {args.min_roi}%\n")
    
    # Output
    if args.show_all:
        print_no_plays_reasoning(df_games, strategies, args.granularity)
    
    print_plays(df_plays, all_games_info)
    
    # Save plays to S3 for tracking
    if args.save_s3:
        target_date = args.date if args.date else date.today().strftime('%Y-%m-%d')
        save_plays_to_s3(df_plays, target_date, args.season)


if __name__ == '__main__':
    main()


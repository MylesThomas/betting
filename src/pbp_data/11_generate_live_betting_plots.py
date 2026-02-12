"""
Live Betting Plot Generator

Purpose:
Generate real-time Monte Carlo probability plots for live NBA games.
Separate from signal detection (script 10) for performance - plots are slower.

Process:
1. Fetch live games from ESPN API
2. Get active players in each game
3. Load player profiles and pregame lines
4. Fetch play-by-play data from ESPN
5. Build probability curve (run MC at multiple time points)
6. Generate plot with current game state marker
7. Save to local directory AND upload to S3

Usage:
    # Run once (generate plots for all live games)
    python src/pbp_data/11_generate_live_betting_plots.py
    
    # Run continuously with auto-refresh
    python src/pbp_data/11_generate_live_betting_plots.py --loop --interval 180
    
    # Test mode with specific players
    python src/pbp_data/11_generate_live_betting_plots.py --test-with-fake-data

Output:
    - Local: ~/Downloads/tmp/live_betting_plots/{YYYYMMDD}-{player_name}-{game_id}.png
    - S3: s3://nba-betting-mt/data/04_output/live_player_odds/plots/{YYYYMMDD}-{player_name}-{game_id}.png
"""

import sys
import requests
import boto3
import pandas as pd
import numpy as np
import json
import os
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load .env file from project root
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pbp_data.monte_carlo_utils import (
    load_player_profile,
    monte_carlo_simulate_bet,
    find_vegas_adjustment,
    get_consensus_prop_line,
    create_ggplot,
    get_data_paths
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# S3 configuration
S3_BUCKET = "nba-betting-mt"
S3_PLOT_PREFIX = "data/04_output/live_player_odds/plots"

# Local plot directory
LOCAL_PLOT_DIR = Path.home() / "Downloads" / "tmp" / "live_betting_plots"
LOCAL_PLOT_DIR.mkdir(exist_ok=True, parents=True)

# ESPN API
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

# Plotting parameters
N_SIMULATIONS = 2000  # Faster for plotting (still accurate)
MC_SAMPLE_INTERVAL = 2  # Sample MC every 2 minutes for probability curve
MAX_PLAYERS_PER_GAME = 10  # Top scorers only

# Initialize boto3
s3_client = boto3.client('s3')

# Cache for player profiles and vegas adjustments (reuse across iterations)
_PLAYER_PROFILE_CACHE = {}
_VEGAS_ADJUSTMENT_CACHE = {}


# =============================================================================
# TEST MODE - FAKE DATA
# =============================================================================

def generate_fake_live_games() -> List[Dict]:
    """Generate fake live games for testing."""
    return [
        {
            'game_id': '401810642',
            'away_team': 'Memphis Grizzlies',
            'home_team': 'Denver Nuggets',
            'away_score': 44,
            'home_score': 53,
            'quarter': 2,
            'clock': '3:28',
            'game_date': '2026-02-11'
        }
    ]


def generate_fake_active_players(game_id: str) -> List[Dict]:
    """Generate fake active players for testing."""
    return [
        {
            'player_name': 'Nikola Jokic',
            'player_id': '3112335',
            'team': 'Denver Nuggets',
            'current_points': 9.0,
            'minutes_played': 12.5
        }
    ]


# =============================================================================
# STEP 1: FETCH LIVE GAMES
# =============================================================================

def fetch_live_games(test_mode: bool = False) -> List[Dict]:
    """
    Fetch currently live NBA games from ESPN API.
    
    Args:
        test_mode: If True, return fake data for testing
    
    Returns:
        List of game dictionaries with game_id, teams, score, clock, etc.
    """
    if test_mode:
        return generate_fake_live_games()
    
    try:
        response = requests.get(ESPN_SCOREBOARD_URL, timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()
        
        live_games = []
        
        for event in data.get('events', []):
            status = event['status']['type']['name']
            
            # Only include in-progress games
            if status == 'STATUS_IN_PROGRESS':
                competition = event['competitions'][0]
                
                game_info = {
                    'game_id': event['id'],
                    'away_team': competition['competitors'][1]['team']['displayName'],
                    'home_team': competition['competitors'][0]['team']['displayName'],
                    'away_score': int(competition['competitors'][1]['score']),
                    'home_score': int(competition['competitors'][0]['score']),
                    'quarter': event['status']['period'],
                    'clock': event['status']['displayClock'],
                    'game_date': event['date'][:10],  # YYYY-MM-DD
                }
                
                live_games.append(game_info)
        
        return live_games
    
    except Exception as e:
        print(f"❌ Error fetching live games: {e}")
        return []


# =============================================================================
# STEP 2: GET ACTIVE PLAYERS
# =============================================================================

def get_active_players(game_id: str, test_mode: bool = False) -> List[Dict]:
    """
    Get active players from a live game's boxscore.
    
    Args:
        game_id: ESPN game ID
        test_mode: If True, return fake data
    
    Returns:
        List of player dictionaries with name, points, minutes, etc.
    """
    if test_mode:
        return generate_fake_active_players(game_id)
    
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()
        
        players = []
        
        # Parse boxscore for both teams
        boxscore = data.get('boxscore', {})
        teams = boxscore.get('teams', [])
        
        for team in teams:
            team_name = team['team']['displayName']
            statistics = team.get('statistics', [])
            
            for stat_group in statistics:
                for athlete in stat_group.get('athletes', []):
                    stats = athlete.get('stats', [])
                    
                    points = 0
                    minutes = 0
                    
                    for stat_val in stats:
                        if 'PTS' in str(stat_val):
                            try:
                                points = float(stat_val) if stat_val != '--' else 0
                            except:
                                points = 0
                        if 'MIN' in str(stat_val):
                            try:
                                if ':' in str(stat_val):
                                    mins, secs = str(stat_val).split(':')
                                    minutes = float(mins) + float(secs) / 60
                                else:
                                    minutes = float(stat_val) if stat_val != '--' else 0
                            except:
                                minutes = 0
                    
                    if minutes > 0:
                        player_info = {
                            'player_name': athlete['athlete']['displayName'],
                            'player_id': athlete['athlete'].get('id'),
                            'team': team_name,
                            'current_points': points,
                            'minutes_played': minutes,
                        }
                        players.append(player_info)
        
        players.sort(key=lambda x: x['current_points'], reverse=True)
        return players[:MAX_PLAYERS_PER_GAME]
    
    except Exception as e:
        print(f"⚠️  Error getting active players for game {game_id}: {e}")
        return []


# =============================================================================
# STEP 3: FETCH PLAY-BY-PLAY DATA
# =============================================================================

def fetch_live_play_by_play(game_id: str) -> Optional[Dict]:
    """
    Fetch live play-by-play data from ESPN API.
    
    Args:
        game_id: ESPN game ID
    
    Returns:
        Dictionary with plays, boxscore, header, etc. or None if failed
    """
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()
        
        return data
    
    except Exception as e:
        print(f"   ⚠️  Error fetching play-by-play: {e}")
        return None


# =============================================================================
# STEP 4: BUILD PROBABILITY CURVE
# =============================================================================

def calculate_game_minute(quarter: int, clock: str) -> float:
    """
    Calculate current game minute from quarter and clock.
    
    Args:
        quarter: Quarter number (1-4)
        clock: Clock string (e.g., "3:28")
    
    Returns:
        Game minute (0-48)
    """
    try:
        if ':' in clock:
            mins, secs = map(int, clock.split(':'))
            time_remaining = mins + secs / 60.0
        else:
            time_remaining = float(clock) / 60.0
    except:
        time_remaining = 0
    
    quarter_start = (quarter - 1) * 12
    game_minute = quarter_start + (12 - time_remaining)
    
    return game_minute


def build_probability_curve(
    player_name: str,
    player_profile: Dict,
    pregame_line: float,
    vegas_adjustment: float,
    pbp_data: Dict,
    current_game_minute: float,
    n_sims: int = N_SIMULATIONS
) -> pd.DataFrame:
    """
    Build probability curve by running MC at multiple time points.
    
    Args:
        player_name: Player name
        player_profile: Player's historical stats
        pregame_line: Pregame prop line
        vegas_adjustment: PPM multiplier
        pbp_data: Play-by-play data from ESPN
        current_game_minute: Current game minute
        n_sims: Number of simulations per point
    
    Returns:
        DataFrame with game_minute, cumulative_points, prob_over
    """
    # Parse plays to track player scoring
    plays = pbp_data.get('plays', [])
    
    play_data = []
    cumulative_points = 0
    
    for play in plays:
        quarter = play.get('period', {}).get('number', 1)
        
        # Calculate game minute
        clock_display = play.get('clock', {}).get('displayValue', '12:00')
        try:
            if isinstance(clock_display, (int, float)):
                time_left_in_quarter = float(clock_display) / 60.0
            elif ':' in str(clock_display):
                mins, secs = map(int, str(clock_display).split(':'))
                time_left_in_quarter = mins + secs / 60.0
            else:
                time_left_in_quarter = float(clock_display) / 60.0
            
            quarter_start = (quarter - 1) * 12
            game_minute = quarter_start + (12 - time_left_in_quarter)
        except:
            game_minute = (quarter - 1) * 12
        
        description = play.get('text', '')
        
        # Check if player scored
        points_this_play = 0
        if player_name in description:
            if 'makes' in description.lower() or 'free throw' in description.lower():
                if '3-pt' in description.lower() or 'three point' in description.lower():
                    points_this_play = 3
                elif '2-pt' in description.lower() or 'two point' in description.lower():
                    points_this_play = 2
                elif 'free throw' in description.lower() and 'makes' in description.lower():
                    points_this_play = 1
        
        cumulative_points += points_this_play
        
        play_data.append({
            'quarter': quarter,
            'game_minute': game_minute,
            'cumulative_points': cumulative_points,
        })
    
    # Convert to DataFrame and sort
    df_plays = pd.DataFrame(play_data)
    df_plays = df_plays.sort_values('game_minute').reset_index(drop=True)
    
    # Keep only up to current minute
    df_plays = df_plays[df_plays['game_minute'] <= current_game_minute].copy()
    
    # Sample points for MC: every N minutes + current minute
    sample_minutes = list(range(0, int(current_game_minute) + 1, MC_SAMPLE_INTERVAL))
    sample_minutes.append(current_game_minute)
    sample_minutes = sorted(set(sample_minutes))
    
    # Run MC at each sample point
    prob_data = []
    for minute in sample_minutes:
        # Find player's points at this minute
        plays_up_to = df_plays[df_plays['game_minute'] <= minute]
        if len(plays_up_to) > 0:
            points_at_minute = plays_up_to.iloc[-1]['cumulative_points']
        else:
            points_at_minute = 0
        
        # Run MC simulation
        prob = monte_carlo_simulate_bet(
            player_profile=player_profile,
            current_minute=minute,
            current_points=points_at_minute,
            prop_line=pregame_line,
            n_simulations=n_sims,
            vegas_adjustment=vegas_adjustment,
            score_differential=None,
            debug=False
        )
        
        prob_data.append({
            'game_minute': minute,
            'cumulative_points': points_at_minute,
            'prob_over': prob
        })
    
    df_probs = pd.DataFrame(prob_data)
    
    # Merge with full play data
    df_plot = df_plays.merge(df_probs, on='game_minute', how='left', suffixes=('', '_mc'))
    df_plot['prob_over'] = df_plot['prob_over'].ffill().bfill()
    df_plot['cumulative_points'] = df_plot['cumulative_points'].ffill()
    
    return df_plot


# =============================================================================
# STEP 5: GENERATE AND SAVE PLOT
# =============================================================================

def generate_and_save_plot(
    player: Dict,
    game: Dict,
    df_plot: pd.DataFrame,
    pregame_line: float,
    player_profile: Dict,
    current_game_minute: float
) -> Optional[str]:
    """
    Generate plot and save to both local and S3.
    
    Args:
        player: Player info
        game: Game info
        df_plot: DataFrame with probability curve
        pregame_line: Pregame prop line
        player_profile: Player's historical stats
        current_game_minute: Current game minute
    
    Returns:
        Local plot path if successful, None otherwise
    """
    player_name = player['player_name']
    player_id = player['player_id']
    game_id = game['game_id']
    game_date = game['game_date']
    away_team = game['away_team']
    home_team = game['home_team']
    current_points = player['current_points']
    
    # Generate plot filename
    game_date_str = game_date.replace('-', '')  # YYYYMMDD
    player_name_clean = player_name.replace(" ", "_")
    plot_filename = f"{game_date_str}-{player_name_clean}-{game_id}.png"
    
    local_plot_path = LOCAL_PLOT_DIR / plot_filename
    s3_key = f"{S3_PLOT_PREFIX}/{plot_filename}"
    
    # Generate plot using existing create_ggplot function
    try:
        plot_path = create_ggplot(
            df=df_plot,
            prop_line=pregame_line,
            player_name=player_name,
            player_id=player_id,
            game_id=game_id,
            game_date=game_date,
            away_team=away_team,
            home_team=home_team,
            final_points=current_points,
            result="IN PROGRESS",
            plot_dir=LOCAL_PLOT_DIR,
            bet_placement_minute=None,  # No bet placement marker
            current_game_minute=current_game_minute  # Mark current state
        )
        
        if not plot_path:
            return None
        
        # Upload to S3
        try:
            with open(plot_path, 'rb') as f:
                s3_client.put_object(
                    Bucket=S3_BUCKET,
                    Key=s3_key,
                    Body=f.read(),
                    ContentType='image/png'
                )
            print(f"      ✅ Uploaded to S3: s3://{S3_BUCKET}/{s3_key}")
        except Exception as e:
            print(f"      ⚠️  Failed to upload to S3: {e}")
        
        return str(plot_path)
    
    except Exception as e:
        print(f"   ❌ Error generating plot: {e}")
        return None


# =============================================================================
# STEP 6: ANALYZE AND PLOT PLAYER
# =============================================================================

def analyze_and_plot_player(
    player: Dict,
    game: Dict,
    n_sims: int = N_SIMULATIONS,
    test_mode: bool = False
) -> Optional[str]:
    """
    Analyze player and generate plot.
    
    Args:
        player: Player info
        game: Game info
        n_sims: Number of MC simulations
        test_mode: If True, use fake pregame lines
    
    Returns:
        Plot path if successful, None otherwise
    """
    player_name = player['player_name']
    current_points = player['current_points']
    game_id = game['game_id']
    
    try:
        # Step 1: Load player profile (with caching)
        if player_name in _PLAYER_PROFILE_CACHE:
            player_profile = _PLAYER_PROFILE_CACHE[player_name]
        else:
            player_profile = load_player_profile(player_name)
            _PLAYER_PROFILE_CACHE[player_name] = player_profile
        
        # Step 2: Get pregame line
        if test_mode:
            fake_pregame_lines = {
                'Nikola Jokic': 26.5,
                'Jamal Murray': 22.5,
                'LeBron James': 25.5
            }
            pregame_line = fake_pregame_lines.get(player_name)
            if pregame_line:
                print(f"      📊 Pregame line (TEST): {pregame_line}")
        else:
            pregame_line = get_consensus_prop_line(
                player_name,
                game['game_date'],
                market="player_points"
            )
            if pregame_line:
                print(f"      📊 Pregame line: {pregame_line}")
        
        if not pregame_line:
            print(f"      ⚠️  No pregame line found")
            return None
        
        # Step 3: Get vegas adjustment (with caching)
        cache_key = f"{player_name}_{pregame_line}"
        if cache_key in _VEGAS_ADJUSTMENT_CACHE:
            vegas_adjustment = _VEGAS_ADJUSTMENT_CACHE[cache_key]
        else:
            print(f"      🎲 Calculating Vegas adjustment...")
            vegas_adjustment = find_vegas_adjustment(
                player_profile,
                pregame_line,
                n_simulations=5000
            )
            _VEGAS_ADJUSTMENT_CACHE[cache_key] = vegas_adjustment
            print(f"         Vegas adjustment: {vegas_adjustment:.4f}")
        
        # Step 4: Calculate current game minute
        current_game_minute = calculate_game_minute(game['quarter'], game['clock'])
        print(f"      ⏱️  Game minute: {current_game_minute:.1f}")
        
        # Step 5: Fetch play-by-play data
        print(f"      📥 Fetching play-by-play data...")
        pbp_data = fetch_live_play_by_play(game_id)
        
        if not pbp_data:
            print(f"      ⚠️  No play-by-play data available")
            return None
        
        # Step 6: Build probability curve
        print(f"      📈 Building probability curve...")
        df_plot = build_probability_curve(
            player_name=player_name,
            player_profile=player_profile,
            pregame_line=pregame_line,
            vegas_adjustment=vegas_adjustment,
            pbp_data=pbp_data,
            current_game_minute=current_game_minute,
            n_sims=n_sims
        )
        
        print(f"      📊 Generated {len(df_plot)} data points")
        
        # Step 7: Generate and save plot
        print(f"      🎨 Generating plot...")
        plot_path = generate_and_save_plot(
            player=player,
            game=game,
            df_plot=df_plot,
            pregame_line=pregame_line,
            player_profile=player_profile,
            current_game_minute=current_game_minute
        )
        
        if plot_path:
            print(f"      ✅ Local plot: {plot_path}")
            return plot_path
        else:
            return None
    
    except Exception as e:
        print(f"   ❌ Error analyzing {player_name}: {e}")
        return None


# =============================================================================
# STEP 7: CLEANUP OLD PLOTS
# =============================================================================

def cleanup_completed_games(live_game_ids: List[str]):
    """
    Delete plots for games that are no longer live.
    
    Args:
        live_game_ids: List of currently live game IDs
    """
    try:
        # Get all plots in local directory
        local_plots = list(LOCAL_PLOT_DIR.glob("*.png"))
        
        deleted_count = 0
        for plot_path in local_plots:
            # Extract game_id from filename: {YYYYMMDD}-{player_name}-{game_id}.png
            parts = plot_path.stem.split('-')
            if len(parts) >= 3:
                plot_game_id = parts[-1]
                
                if plot_game_id not in live_game_ids:
                    plot_path.unlink()
                    deleted_count += 1
        
        if deleted_count > 0:
            print(f"   🧹 Cleaned up {deleted_count} plot(s) from completed games")
    
    except Exception as e:
        print(f"   ⚠️  Error during cleanup: {e}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution loop."""
    
    parser = argparse.ArgumentParser(description="Live betting plot generator")
    parser.add_argument("--n-sims", type=int, default=2000, help="Number of MC simulations (default 2000)")
    parser.add_argument("--test-with-fake-data", action="store_true", help="Run in test mode with fake data")
    parser.add_argument("--loop", action="store_true", help="Run continuously (refresh plots)")
    parser.add_argument("--interval", type=int, default=180, help="Update interval in seconds (default 180)")
    args = parser.parse_args()
    
    test_mode = args.test_with_fake_data
    n_sims = args.n_sims
    
    print("="*80)
    print("LIVE BETTING PLOT GENERATOR")
    print("="*80)
    print()
    print("📋 Process Overview:")
    print("   1. Fetch live games from ESPN")
    print("   2. Get active players in each game")
    print("   3. Load player profiles and pregame lines")
    print("   4. Fetch play-by-play data")
    print("   5. Build probability curve (run MC at multiple time points)")
    print("   6. Generate plot with current game state marker")
    print("   7. Save to local directory and upload to S3")
    print("   8. Cleanup plots for completed games")
    print()
    print(f"⚙️  Configuration:")
    print(f"   - Mode: {'TEST (Fake Data)' if test_mode else 'LIVE (Real Data)'}")
    print(f"   - MC Simulations: {n_sims:,} per time point")
    print(f"   - Sample Interval: Every {MC_SAMPLE_INTERVAL} minutes")
    print(f"   - Max Players Per Game: {MAX_PLAYERS_PER_GAME}")
    print(f"   - Local plots: {LOCAL_PLOT_DIR}")
    print(f"   - S3 bucket: s3://{S3_BUCKET}/{S3_PLOT_PREFIX}/")
    if args.loop:
        print(f"   - Loop mode: Update every {args.interval} seconds")
    print()
    
    iteration = 0
    
    while True:
        iteration += 1
        
        if args.loop:
            print("="*80)
            print(f"ITERATION #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            print()
        
        # =====================================================================
        # STEP 1: FETCH LIVE GAMES
        # =====================================================================
        print("="*80)
        print(f"STEP 1: Fetching live games {'(TEST MODE)' if test_mode else 'from ESPN'}...")
        print("="*80)
        
        live_games = fetch_live_games(test_mode=test_mode)
        
        if not live_games:
            print("❌ No live games found")
            if not args.loop:
                return
            else:
                print(f"   ⏳ Waiting {args.interval} seconds before next check...")
                time.sleep(args.interval)
                continue
        
        print(f"✅ Found {len(live_games)} live game(s)")
        for game in live_games:
            print(f"   🏀 {game['away_team']} ({game['away_score']}) @ {game['home_team']} ({game['home_score']})")
            print(f"      Q{game['quarter']} - {game['clock']}")
        print()
        
        live_game_ids = [g['game_id'] for g in live_games]
        
        # Track generated plots
        generated_plots = []
        
        # =====================================================================
        # STEP 2-7: PROCESS EACH GAME
        # =====================================================================
        for game in live_games:
            game_id = game['game_id']
            
            print("="*80)
            print(f"STEP 2: Getting active players for {game['away_team']} @ {game['home_team']}...")
            print("="*80)
            
            players = get_active_players(game_id, test_mode=test_mode)
            
            if not players:
                print("⚠️  No active players found")
                continue
            
            print(f"✅ Found {len(players)} active player(s)")
            for p in players:
                print(f"   - {p['player_name']} ({p['team']}): {p['current_points']} pts, {p['minutes_played']:.1f} min")
            print()
            
            print("="*80)
            print(f"STEP 3-7: Generating plots for each player...")
            print("="*80)
            
            for player in players:
                print(f"   🔄 Processing {player['player_name']}...")
                
                plot_path = analyze_and_plot_player(
                    player=player,
                    game=game,
                    n_sims=n_sims,
                    test_mode=test_mode
                )
                
                if plot_path:
                    generated_plots.append(plot_path)
                    print(f"      ✅ Plot generated successfully")
                else:
                    print(f"      ⚠️  Plot generation failed")
            
            print()
        
        # =====================================================================
        # STEP 8: CLEANUP
        # =====================================================================
        print("="*80)
        print(f"STEP 8: Cleaning up plots for completed games...")
        print("="*80)
        
        cleanup_completed_games(live_game_ids)
        print()
        
        # =====================================================================
        # SUMMARY
        # =====================================================================
        print("="*80)
        print("✅ PLOT GENERATION COMPLETE")
        print("="*80)
        print()
        print(f"📊 Generated {len(generated_plots)} plot(s)")
        
        if generated_plots:
            print()
            print("📂 Local plots:")
            for plot in generated_plots:
                print(f"   - {plot}")
            print()
            print("💡 To open all plots:")
            print(f'   open {" ".join([f\'"{p}\' for p in generated_plots])}')
        
        print()
        
        # Exit or continue loop
        if not args.loop:
            break
        else:
            print(f"⏳ Waiting {args.interval} seconds before next update...")
            print()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()

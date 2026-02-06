"""
Step 3: Process cached JSON data into Parquet files.

Reads from ~/Downloads/tmp/player_points_monte_carlo/pbp_data/*.json
Outputs to data/*.parquet

No API calls - processes cached data only.

Usage:
    python src/pbp_data/03_process_data.py [--verbose]
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

from .config import PBP_DATA_DIR, OUTPUT_DIR


def parse_clock_to_seconds(clock_str, quarter):
    """
    Convert game clock to total seconds elapsed from start of game.
    
    Args:
        clock_str: Time string like "10:23" or "5.2"
        quarter: Quarter number (1-4 for regulation, 5+ for OT)
    
    Returns:
        Total seconds elapsed from game start
    """
    if not clock_str or ':' not in str(clock_str):
        try:
            remaining = float(clock_str)
        except:
            return 0
    else:
        parts = str(clock_str).split(':')
        mins = int(parts[0])
        secs = float(parts[1])
        remaining = mins * 60 + secs
    
    # Quarter lengths
    if quarter <= 4:
        quarter_len = 720  # 12 minutes in seconds
        elapsed_this_quarter = quarter_len - remaining
        previous_quarters = (quarter - 1) * quarter_len
    else:
        quarter_len = 300  # 5 minutes OT in seconds
        elapsed_this_quarter = quarter_len - remaining
        previous_quarters = 4 * 720 + (quarter - 5) * 300
    
    return previous_quarters + elapsed_this_quarter


def get_starting_lineups(game_data):
    """
    Extract starting 5 players for each team.
    
    Returns:
        dict: {team_id: [player_id1, player_id2, player_id3, player_id4, player_id5]}
    """
    boxscore = game_data.get('boxscore', {})
    teams = boxscore.get('players', [])
    
    starting_lineups = {}
    
    for team_data in teams:
        team_id = str(team_data.get('team', {}).get('id'))
        stats = team_data.get('statistics', [])
        
        if not stats:
            continue
        
        starters = []
        for player in stats[0].get('athletes', []):
            if player.get('starter', False):
                player_id = str(player.get('athlete', {}).get('id'))
                starters.append(player_id)
        
        if len(starters) == 5:
            starting_lineups[team_id] = starters
    
    return starting_lineups


def get_player_names(game_data):
    """Get player_id -> player_name mapping from boxscore."""
    boxscore = game_data.get('boxscore', {})
    teams = boxscore.get('players', [])
    
    names = {}
    
    for team_data in teams:
        stats = team_data.get('statistics', [])
        if not stats:
            continue
        
        for player in stats[0].get('athletes', []):
            player_id = str(player.get('athlete', {}).get('id'))
            player_name = player.get('athlete', {}).get('displayName')
            names[player_id] = player_name
    
    return names


def parse_game_to_minute_by_minute(game_data, game_id, game_date):
    """
    Parse a single game's play-by-play into minute-by-minute data.
    
    Tracks both playing time (actual seconds on court) and cumulative points.
    Uses brute force approach: maintains 5-player roster per team, adds elapsed
    time to all 10 players on court between plays.
    
    Returns:
        DataFrame with columns: game_id, game_date, player_id, player_name,
                               minute, playing_seconds, cumulative_points
    """
    plays = game_data.get('plays', [])
    
    if not plays:
        return pd.DataFrame()
    
    # Get starting lineups (5 players per team)
    team_rosters = get_starting_lineups(game_data)
    player_names = get_player_names(game_data)
    
    if len(team_rosters) != 2:
        # Can't track without proper rosters
        return pd.DataFrame()
    
    # Initialize tracking
    playing_seconds = {}  # player_id -> total seconds played
    cumulative_points = {}  # player_id -> total points scored
    
    # Initialize all starting players
    for team_id, players in team_rosters.items():
        for player_id in players:
            playing_seconds[player_id] = 0
            cumulative_points[player_id] = 0
    
    # Store snapshots at each game clock minute
    minute_data = []
    
    # Track game state
    last_game_seconds = 0
    last_minute = -1
    
    # Process plays in chronological order (already sorted)
    for play in plays:
        quarter = play.get('period', {}).get('number', 1)
        clock = play.get('clock', {}).get('displayValue', '0:00')
        game_seconds = parse_clock_to_seconds(clock, quarter)
        
        # Calculate time elapsed since last play
        time_delta = game_seconds - last_game_seconds
        
        # Add time to all 10 players currently on court (5 per team)
        if 0 < time_delta < 120:  # Sanity check: 0-120 seconds
            for team_id, roster in team_rosters.items():
                for player_id in roster:
                    playing_seconds[player_id] += time_delta
        
        # Handle scoring
        if play.get('scoringPlay', False):
            participants = play.get('participants', [])
            if participants:
                scorer_id = str(participants[0].get('athlete', {}).get('id'))
                points = play.get('scoreValue', 0)
                
                if scorer_id not in cumulative_points:
                    cumulative_points[scorer_id] = 0
                    playing_seconds[scorer_id] = 0
                
                cumulative_points[scorer_id] += points
        
        # Handle substitution - swap player in 5-man roster
        play_type = play.get('type', {}).get('text', '')
        if 'substitution' in play_type.lower():
            team_id = str(play.get('team', {}).get('id'))
            participants = play.get('participants', [])
            
            if len(participants) >= 2 and team_id in team_rosters:
                player_in_id = str(participants[0].get('athlete', {}).get('id'))
                player_out_id = str(participants[1].get('athlete', {}).get('id'))
                
                # Remove player_out, add player_in
                if player_out_id in team_rosters[team_id]:
                    team_rosters[team_id].remove(player_out_id)
                
                if player_in_id not in team_rosters[team_id]:
                    team_rosters[team_id].append(player_in_id)
                
                # Initialize new player if needed
                if player_in_id not in playing_seconds:
                    playing_seconds[player_in_id] = 0
                    cumulative_points[player_in_id] = 0
        
        # Take snapshot at each new minute
        current_minute = int(game_seconds // 60)
        if current_minute != last_minute:
            # Record for all players who have played
            for player_id in playing_seconds.keys():
                minute_data.append({
                    'game_id': game_id,
                    'game_date': game_date,
                    'player_id': player_id,
                    'player_name': player_names.get(player_id, 'Unknown'),
                    'minute': current_minute,
                    'playing_seconds': playing_seconds[player_id],
                    'cumulative_points': cumulative_points[player_id],
                })
            
            last_minute = current_minute
        
        last_game_seconds = game_seconds
    
    return pd.DataFrame(minute_data)


def track_minutes_from_substitutions(game_data, game_id, game_date):
    """
    Track minutes played per quarter using substitution data.
    
    Returns:
        DataFrame with columns: game_id, game_date, player_id, player_name, quarter, minutes_played
    """
    plays = game_data.get('plays', [])
    boxscore = game_data.get('boxscore', {})
    
    # Get starters from boxscore
    starters = set()
    player_map = {}
    
    players_data = boxscore.get('players', [])
    for team_data in players_data:
        stats = team_data.get('statistics', [])
        if stats:
            athletes = stats[0].get('athletes', [])
            for player in athletes:
                athlete = player.get('athlete', {})
                player_id = str(athlete.get('id'))
                player_name = athlete.get('displayName', '')
                is_starter = player.get('starter', False)
                
                player_map[player_id] = player_name
                
                if is_starter:
                    starters.add(player_id)
    
    # Track time on court for each player by quarter
    # Format: {player_id: {quarter: [(time_in, time_out), ...]}}
    player_court_time = defaultdict(lambda: defaultdict(list))
    
    # Track who's on court
    on_court = set(starters)  # Start with starters
    
    # Track last time for each quarter
    last_time_by_quarter = {}
    
    # Process substitutions by quarter
    for quarter in [1, 2, 3, 4, 5]:
        quarter_plays = [p for p in plays if p.get('period', {}).get('number') == quarter]
        
        if not quarter_plays:
            continue
        
        # At start of quarter (except Q1), check for subs
        if quarter > 1:
            # Look for 12:00 substitutions
            start_subs = [p for p in quarter_plays if p.get('clock', {}).get('displayValue') == '12:00' 
                         and 'substitution' in p.get('type', {}).get('text', '').lower()]
            
            for sub in start_subs:
                # Parse "X enters the game for Y"
                text = sub.get('text', '')
                if ' enters the game for ' in text:
                    parts = text.split(' enters the game for ')
                    player_in = parts[0].strip()
                    player_out = parts[1].strip()
                    
                    # Find player IDs (match by name)
                    player_in_id = None
                    player_out_id = None
                    
                    for pid, pname in player_map.items():
                        if pname == player_in:
                            player_in_id = pid
                        if pname == player_out:
                            player_out_id = pid
                    
                    if player_out_id and player_out_id in on_court:
                        on_court.remove(player_out_id)
                    if player_in_id:
                        on_court.add(player_in_id)
        
        # Players on court at start of quarter play the full quarter (simplified)
        # In reality, track every sub, but for V1 this is good enough
        for player_id in on_court:
            player_court_time[player_id][quarter].append((0, 12))  # Full quarter
    
    # Convert to DataFrame
    minutes_data = []
    
    for player_id, quarters in player_court_time.items():
        for quarter, time_segments in quarters.items():
            # Sum time segments
            total_minutes = sum(end - start for start, end in time_segments)
            
            minutes_data.append({
                'game_id': game_id,
                'game_date': game_date,
                'player_id': player_id,
                'player_name': player_map.get(player_id, ''),
                'quarter': quarter,
                'minutes_played': total_minutes,
            })
    
    return pd.DataFrame(minutes_data)


def process_all_games(verbose=False):
    """
    Process all cached JSON files into DataFrames.
    
    Returns:
        Tuple of (minute_by_minute_df, quarter_stats_df, validation_df)
    """
    json_files = sorted(PBP_DATA_DIR.glob('*.json'))
    
    if verbose:
        print(f"📊 Processing {len(json_files)} cached games")
        print()
    
    all_minute_data = []
    all_quarter_data = []
    all_validation_data = []
    failed_files = []
    
    for i, json_file in enumerate(json_files):
        if verbose and (i+1) % 50 == 0:
            print(f"  Processed {i+1}/{len(json_files)} games...")
        
        # Parse filename to get game_id and date
        filename = json_file.stem  # e.g., "20260204_401810584"
        parts = filename.split('_')
        date_str = parts[0]
        game_id = parts[1]
        game_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        # Load JSON with error handling
        try:
            with open(json_file, 'r') as f:
                game_data = json.load(f)
        except json.JSONDecodeError as e:
            failed_files.append((json_file.name, str(e)))
            if verbose:
                print(f"  ❌ Skipping {json_file.name}: JSON decode error")
            continue
        except Exception as e:
            failed_files.append((json_file.name, str(e)))
            if verbose:
                print(f"  ❌ Skipping {json_file.name}: {e}")
            continue
        
        # Parse minute-by-minute
        minute_df = parse_game_to_minute_by_minute(game_data, game_id, game_date)
        if len(minute_df) > 0:
            all_minute_data.append(minute_df)
        
        # Track minutes (simplified for now - will enhance later)
        # For V1, we'll skip minutes tracking and just validate against boxscore totals
    
    # Report failed files
    if failed_files and verbose:
        print()
        print(f"⚠️  Failed to process {len(failed_files)} files:")
        for filename, error in failed_files[:10]:  # Show first 10
            print(f"     {filename}")
        if len(failed_files) > 10:
            print(f"     ... and {len(failed_files) - 10} more")
    
    # Combine all games
    if all_minute_data:
        minute_by_minute_df = pd.concat(all_minute_data, ignore_index=True)
    else:
        minute_by_minute_df = pd.DataFrame()
    
    return minute_by_minute_df


def main():
    parser = argparse.ArgumentParser(description='Process NBA play-by-play data')
    parser.add_argument('--verbose', action='store_true', help='Print progress')
    args = parser.parse_args()
    
    if args.verbose:
        print(f"🏀 Processing play-by-play data")
        print()
    
    # Process all games
    minute_by_minute_df = process_all_games(verbose=args.verbose)
    
    if args.verbose:
        print()
        print(f"✅ Processed {len(minute_by_minute_df):,} minute-by-minute rows")
        print(f"   {minute_by_minute_df['game_id'].nunique()} games")
        print(f"   {minute_by_minute_df['player_id'].nunique()} players")
    
    # Save to Parquet
    output_file = OUTPUT_DIR / 'minute_by_minute.parquet'
    minute_by_minute_df.to_parquet(output_file, index=False, engine='pyarrow', compression='snappy')
    
    if args.verbose:
        print()
        print(f"💾 Saved to: {output_file}")
        print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

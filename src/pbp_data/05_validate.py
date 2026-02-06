"""
Step 5: Validate parsed data against official box scores.

Reads from:
- ~/Downloads/tmp/player_points_monte_carlo/pbp_data/*.json (box scores)
- data/player_profiles.parquet (our calculated totals)

Outputs to:
- data/metadata/validation_results.parquet

Validates:
- Total points per game match box score (±1 point tolerance)
- Minutes played per game match box score (±1 minute tolerance)

Usage:
    python src/pbp_data/05_validate.py [--verbose]
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

from .config import PBP_DATA_DIR, OUTPUT_DIR


def extract_boxscore_totals(game_data, game_id):
    """
    Extract official box score totals from ESPN JSON.
    
    Returns:
        List of dicts with: game_id, player_id, player_name, official_points, official_minutes
    """
    boxscore = game_data.get('boxscore', {})
    players_data = boxscore.get('players', [])
    
    results = []
    
    for team_data in players_data:
        stats = team_data.get('statistics', [])
        if not stats:
            continue
        
        # Get stat column mappings
        stat_labels = stats[0].get('labels', [])
        
        # Find indices for MIN and PTS
        try:
            min_idx = stat_labels.index('MIN')
            pts_idx = stat_labels.index('PTS')
        except ValueError:
            # If labels not found, skip this team
            continue
        
        athletes = stats[0].get('athletes', [])
        
        for player in athletes:
            athlete = player.get('athlete', {})
            player_id = str(athlete.get('id'))
            player_name = athlete.get('displayName', '')
            
            # Get stats array
            player_stats = player.get('stats', [])
            
            if len(player_stats) > max(min_idx, pts_idx):
                minutes_str = player_stats[min_idx]
                points_str = player_stats[pts_idx]
                
                # Parse minutes (format: "35" or "35:23")
                try:
                    if ':' in str(minutes_str):
                        parts = minutes_str.split(':')
                        official_minutes = int(parts[0]) + int(parts[1]) / 60.0
                    else:
                        official_minutes = float(minutes_str)
                except:
                    official_minutes = 0.0
                
                # Parse points
                try:
                    official_points = int(points_str)
                except:
                    official_points = 0
                
                results.append({
                    'game_id': game_id,
                    'player_id': player_id,
                    'player_name': player_name,
                    'official_points': official_points,
                    'official_minutes': official_minutes,
                })
    
    return results


def validate_all_games(verbose=False):
    """
    Compare our calculated totals with official box scores.
    
    Returns:
        DataFrame with validation results
    """
    json_files = sorted(PBP_DATA_DIR.glob('*.json'))
    
    if verbose:
        print(f"📊 Validating {len(json_files)} games")
        print()
    
    all_official = []
    
    # Extract official box scores
    for i, json_file in enumerate(json_files):
        if verbose and (i+1) % 100 == 0:
            print(f"  Extracted {i+1}/{len(json_files)} box scores...")
        
        # Parse filename (format: {date}_{game_id}.json)
        filename = json_file.stem
        parts = filename.split('_')
        date_str = parts[0]
        game_id = parts[1]
        
        # Load JSON
        with open(json_file, 'r') as f:
            game_data = json.load(f)
        
        # Extract box score
        official = extract_boxscore_totals(game_data, game_id)
        all_official.extend(official)
    
    official_df = pd.DataFrame(all_official)
    
    if verbose:
        print()
        print(f"✅ Extracted {len(official_df)} player-game box scores")
        print()
    
    # Load our calculated data
    minute_file = OUTPUT_DIR / 'minute_by_minute.parquet'
    minute_df = pd.read_parquet(minute_file)
    
    # Calculate our totals (convert playing_seconds to minutes)
    our_totals = minute_df.groupby(['game_id', 'player_id']).agg({
        'cumulative_points': 'max',
        'playing_seconds': 'max',
        'player_name': 'first',
    }).reset_index()
    
    # Convert seconds to minutes
    our_totals['calculated_minutes'] = our_totals['playing_seconds'] / 60.0
    
    our_totals = our_totals[['game_id', 'player_id', 'cumulative_points', 'calculated_minutes', 'player_name']]
    our_totals.columns = ['game_id', 'player_id', 'calculated_points', 'calculated_minutes', 'player_name']
    
    # Merge
    comparison = official_df.merge(
        our_totals,
        on=['game_id', 'player_id'],
        how='outer',
        suffixes=('_official', '_calculated')
    )
    
    # Fill NaNs (players in one dataset but not the other)
    comparison['official_points'] = comparison['official_points'].fillna(0)
    comparison['calculated_points'] = comparison['calculated_points'].fillna(0)
    comparison['official_minutes'] = comparison['official_minutes'].fillna(0)
    comparison['calculated_minutes'] = comparison['calculated_minutes'].fillna(0)
    
    # Calculate differences
    comparison['points_diff'] = comparison['calculated_points'] - comparison['official_points']
    comparison['minutes_diff'] = comparison['calculated_minutes'] - comparison['official_minutes']
    
    # Validation flags
    comparison['points_valid'] = comparison['points_diff'].abs() <= 1
    comparison['minutes_valid'] = comparison['minutes_diff'].abs() <= 1
    comparison['fully_valid'] = comparison['points_valid'] & comparison['minutes_valid']
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description='Validate parsed data')
    parser.add_argument('--verbose', action='store_true', help='Print progress')
    args = parser.parse_args()
    
    if args.verbose:
        print(f"🏀 Validating parsed data")
        print()
    
    # Validate
    validation_df = validate_all_games(verbose=args.verbose)
    
    # Print summary
    total = len(validation_df)
    points_valid = validation_df['points_valid'].sum()
    minutes_valid = validation_df['minutes_valid'].sum()
    fully_valid = validation_df['fully_valid'].sum()
    
    if args.verbose:
        print(f"📊 Validation Results:")
        print(f"   Total player-games: {total:,}")
        print(f"   Points valid (±1): {points_valid:,} ({100*points_valid/total:.1f}%)")
        print(f"   Minutes valid (±1): {minutes_valid:,} ({100*minutes_valid/total:.1f}%)")
        print(f"   Fully valid: {fully_valid:,} ({100*fully_valid/total:.1f}%)")
        print()
        
        # Show problematic cases
        invalid = validation_df[~validation_df['fully_valid']]
        if len(invalid) > 0:
            print(f"⚠️  {len(invalid)} invalid cases found:")
            print()
            
            # Points issues
            points_issues = invalid[~invalid['points_valid']].nlargest(5, 'points_diff', keep='all')
            if len(points_issues) > 0:
                print(f"   Top points discrepancies:")
                for _, row in points_issues.head().iterrows():
                    print(f"      {row['player_name_official']}: {row['calculated_points']:.0f} calc vs {row['official_points']:.0f} official (diff: {row['points_diff']:+.0f})")
                print()
            
            # Minutes issues
            minutes_issues = invalid[~invalid['minutes_valid']].nlargest(5, 'minutes_diff', keep='all')
            if len(minutes_issues) > 0:
                print(f"   Top minutes discrepancies:")
                for _, row in minutes_issues.head().iterrows():
                    print(f"      {row['player_name_official']}: {row['calculated_minutes']:.1f} calc vs {row['official_minutes']:.1f} official (diff: {row['minutes_diff']:+.1f})")
    
    # Save to Parquet
    metadata_dir = OUTPUT_DIR / 'metadata'
    metadata_dir.mkdir(exist_ok=True)
    
    output_file = metadata_dir / 'validation_results.parquet'
    validation_df.to_parquet(output_file, index=False, engine='pyarrow', compression='snappy')
    
    if args.verbose:
        print()
        print(f"💾 Saved to: {output_file}")
        print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

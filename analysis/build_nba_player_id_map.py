"""
Build comprehensive NBA player ID mapping from NBA API game logs.

This script scans NBA API game logs across multiple seasons to create a
complete player name -> player ID mapping for use in visualizations.

Usage:
    python3 analysis/build_nba_player_id_map.py --seasons 2023-24 2024-25 2025-26
"""

import boto3
import pandas as pd
from io import StringIO
import json
import sys
from pathlib import Path

def normalize_player_name(name):
    """Normalize player name for consistent matching."""
    # Remove periods, convert to title case, handle common variations
    name = name.strip().replace('.', '').replace("'", "").replace('-', ' ')
    # Handle common patterns
    name = name.title()
    # Specific mappings for known variations
    name_map = {
        'Cj Mccollum': 'CJ McCollum',
        'Rj Barrett': 'RJ Barrett',
        'Og Anunoby': 'OG Anunoby',
        'Pj Washington': 'PJ Washington',
        'Tj Mcconnell': 'TJ McConnell',
    }
    return name_map.get(name, name)

def build_player_id_map(seasons):
    """Build player ID map from NBA API game logs."""
    s3 = boto3.client('s3')
    player_map = {}
    
    for season in seasons:
        print(f"📅 Processing season {season}...")
        
        # List all game logs for the season
        response = s3.list_objects_v2(
            Bucket='nba-api-mt',
            Prefix=f'player_game_logs/{season}/'
        )
        
        if 'Contents' not in response:
            print(f"   ⚠️  No game logs found for {season}")
            continue
        
        file_count = 0
        for obj in response['Contents']:
            key = obj['Key']
            if not key.endswith('.csv'):
                continue
            
            # Read CSV
            response_obj = s3.get_object(Bucket='nba-api-mt', Key=key)
            df = pd.read_csv(StringIO(response_obj['Body'].read().decode('utf-8')))
            
            # Add to map
            for _, row in df.iterrows():
                player_name = normalize_player_name(row['PLAYER_NAME'])
                player_id = str(row['PLAYER_ID'])
                
                # Store both original and normalized names
                player_map[player_name] = player_id
                player_map[row['PLAYER_NAME']] = player_id
            
            file_count += 1
        
        print(f"   ✅ Processed {file_count} files, {len(player_map)} unique players so far")
    
    return player_map

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Build NBA player ID map')
    parser.add_argument('--seasons', nargs='+', default=['2023-24', '2024-25', '2025-26'],
                        help='NBA seasons to scan (e.g., 2023-24 2024-25)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("🏀 BUILDING NBA PLAYER ID MAP")
    print("=" * 80)
    print(f"Seasons: {', '.join(args.seasons)}\n")
    
    player_map = build_player_id_map(args.seasons)
    
    # Save to JSON
    output_path = Path.home() / 'Downloads' / 'tmp' / 'nba_player_id_map.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(player_map, f, indent=2)
    
    print(f"\n✅ Saved {len(player_map)} player mappings to: {output_path}")
    
    # Show some examples
    print("\n📋 Sample mappings:")
    for i, (name, player_id) in enumerate(list(player_map.items())[:10]):
        print(f"   {name}: {player_id}")

if __name__ == '__main__':
    main()

"""
Quick debug script to find players without scorer_type classification.

Context: Investigating why bench_pickem_under (2D) has more plays than the sum 
of bench_pickem_rim_under + bench_pickem_perimeter_under (3D). Found that players
without scorer_type classification only appear in 2D strategies.

Usage:
    cd betting-repo
    python tmp/debug_unclassified_players.py
    python tmp/debug_unclassified_players.py --check-players-without-match

Created: 2026-01-18 5:30pm CT
"""

import pandas as pd
import boto3
from io import BytesIO
import argparse
import sys
from pathlib import Path

# Add src to path for imports
root = Path(__file__).parent.parent
sys.path.insert(0, str(root / 'src'))
from player_name_utils import normalize_player_name

# =============================================================================
# CONFIG
# =============================================================================

SEASON = '2025-26'
RIM_SCORER_PCT = 40
DATE = '2026-01-18'

# =============================================================================
# LOAD DATA
# =============================================================================

def load_player_scorer_types(season=SEASON, rim_scorer_pct=RIM_SCORER_PCT):
    """Load player scorer type classifications from S3"""
    print(f"\n📊 Loading player scorer type data...")
    
    s3_key = f"data/03_intermediate/player_props_with_actuals_{season}_rim{rim_scorer_pct}.csv"
    
    s3 = boto3.client('s3')
    bucket = 'nba-betting-mt'
    obj = s3.get_object(Bucket=bucket, Key=s3_key)
    
    df = pd.read_csv(BytesIO(obj['Body'].read()))
    
    if 'scorer_type' not in df.columns:
        print(f"   ⚠️  WARNING: scorer_type column not found in data")
        return {}, df
    
    # Normalize player names before creating mapping
    df['PLAYER_NAME_NORMALIZED'] = df['PLAYER_NAME'].apply(normalize_player_name)
    
    # Create mapping (take most recent scorer_type for each player)
    scorer_map = df[['PLAYER_NAME_NORMALIZED', 'scorer_type']].dropna().drop_duplicates('PLAYER_NAME_NORMALIZED').set_index('PLAYER_NAME_NORMALIZED')['scorer_type'].to_dict()
    
    rim_count = sum(1 for v in scorer_map.values() if 'Rim' in str(v))
    perim_count = sum(1 for v in scorer_map.values() if 'Perimeter' in str(v))
    
    print(f"   ✅ Loaded scorer types for {len(scorer_map)} players")
    print(f"      Rim Attackers (≥{rim_scorer_pct}%): {rim_count}")
    print(f"      Perimeter (<{rim_scorer_pct}%): {perim_count}")
    
    return scorer_map, df


def load_todays_props(date=DATE):
    """Load today's player props from S3"""
    print(f"\n📊 Loading today's player props ({date})...")
    
    # Try 2D file first (has all players)
    s3_key = f"data/04_output/plays/role_spread_points_model/2d/{date}.csv"
    
    s3 = boto3.client('s3')
    bucket = 'nba-betting-mt'
    
    try:
        obj = s3.get_object(Bucket=bucket, Key=s3_key)
        df = pd.read_csv(BytesIO(obj['Body'].read()))
        print(f"   ✅ Loaded {len(df)} player props")
        return df
    except Exception as e:
        print(f"   ❌ Error loading props: {e}")
        return pd.DataFrame()


# =============================================================================
# MAIN
# =============================================================================

def check_why_player_not_in_data(player_name, df_full):
    """
    Check why a player doesn't have scorer_type in the dataset.
    
    Possible reasons:
    1. Player not in dataset at all (rookie, recently traded, etc.)
    2. Player in dataset but no shot chart data
    3. Player in dataset but not enough games to calculate rim %
    """
    print(f"\n{'='*80}")
    print(f"🔍 WHY DOESN'T {player_name} HAVE SCORER_TYPE?")
    print(f"{'='*80}\n")
    
    # Check if player exists in full dataset
    player_data = df_full[df_full['PLAYER_NAME'] == player_name]
    
    if len(player_data) == 0:
        print(f"❌ {player_name} NOT FOUND in dataset at all")
        print(f"   Possible reasons:")
        print(f"   - Rookie (not in training data)")
        print(f"   - Recently traded/signed")
        print(f"   - Name mismatch (check spelling/normalization)")
        print(f"   - Has not played enough games this season")
        return
    
    print(f"✅ {player_name} IS in dataset ({len(player_data)} rows)")
    
    # Check if scorer_type column exists and what the values are
    if 'scorer_type' in player_data.columns:
        scorer_types = player_data['scorer_type'].unique()
        print(f"\n📊 scorer_type values for this player:")
        print(f"   {scorer_types}")
        
        if pd.isna(scorer_types).all():
            print(f"\n❌ All scorer_type values are NULL/NaN")
            print(f"   Possible reasons:")
            
            # Check if shot chart data exists
            if 'rim_attempt_pct' in player_data.columns:
                rim_pcts = player_data['rim_attempt_pct'].dropna()
                if len(rim_pcts) == 0:
                    print(f"   - No rim_attempt_pct data (missing shot chart data)")
                else:
                    print(f"   - Has rim_attempt_pct data: {rim_pcts.describe()}")
                    print(f"   - But scorer_type still NULL (data pipeline issue?)")
            else:
                print(f"   - rim_attempt_pct column doesn't exist")
            
            # Check game counts
            if 'GAME_DATE' in player_data.columns:
                n_games = player_data['GAME_DATE'].nunique()
                print(f"   - Games in dataset: {n_games}")
                if n_games < 5:
                    print(f"   - ⚠️ Very few games ({n_games}) - may not have enough shot data")
    else:
        print(f"\n❌ scorer_type column doesn't exist in dataset")
    
    # Show sample rows
    print(f"\n📋 Sample rows for {player_name}:")
    print("─" * 80)
    cols_to_show = ['GAME_DATE', 'PLAYER_NAME', 'PTS']
    if 'rim_attempt_pct' in player_data.columns:
        cols_to_show.append('rim_attempt_pct')
    if 'scorer_type' in player_data.columns:
        cols_to_show.append('scorer_type')
    
    print(player_data[cols_to_show].head(10).to_string())
    print()


def main():
    parser = argparse.ArgumentParser(description='Debug players without scorer_type classification')
    parser.add_argument('--check-players-without-match', action='store_true',
                       help='Investigate WHY unclassified players lack scorer_type')
    args = parser.parse_args()
    
    print("="*80)
    print("🔍 DEBUG: Players WITHOUT scorer_type classification")
    print("="*80)
    
    # Load scorer type mapping and full dataset
    scorer_map, df_full = load_player_scorer_types()
    
    # Load today's props
    df_props = load_todays_props()
    
    if df_props.empty:
        print("\n❌ No props data found")
        return
    
    # Get unique players from today
    players_today = df_props['player'].unique()
    print(f"\n📊 Today's unique players: {len(players_today)}")
    
    # Normalize player names for matching
    players_today_normalized = [normalize_player_name(p) for p in players_today]
    
    # Find unclassified players
    unclassified = [p for i, p in enumerate(players_today) if players_today_normalized[i] not in scorer_map]
    
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"{'='*80}\n")
    
    if unclassified:
        print(f"❌ Found {len(unclassified)} players WITHOUT scorer_type:\n")
        
        # Get details for each unclassified player
        for player in unclassified:
            player_rows = df_props[df_props['player'] == player]
            
            # Get representative row
            row = player_rows.iloc[0]
            
            print(f"  {player:30s} | Line: {row['line']:4.1f} ({row['line_tier']:20s}) | Spread: {row['spread']:+5.1f} ({row['spread_bin']:20s})")
            
            # Show strategies they match
            strategies = player_rows['strategy_name'].unique()
            for strat in strategies:
                print(f"      → Strategy: {strat}")
            print()
        
        # Show distribution by line_tier + spread_bin
        print("\nDistribution by bucket:")
        print("─" * 80)
        
        unclassified_rows = df_props[df_props['player'].isin(unclassified)]
        distribution = unclassified_rows.groupby(['line_tier', 'spread_bin']).agg(
            players=('player', 'nunique'),
            plays=('player', 'count')
        ).reset_index().sort_values('players', ascending=False)
        
        for _, row in distribution.iterrows():
            print(f"  {row['line_tier']:20s} + {row['spread_bin']:20s}: {row['players']:2d} players, {row['plays']:2d} plays")
        
        # If --check-players-without-match, investigate each one
        if args.check_players_without_match:
            for player in unclassified:
                check_why_player_not_in_data(player, df_full)
        
    else:
        print("✅ All players have scorer_type classification")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()


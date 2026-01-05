"""
Analyze Close-Range Shooting (0-6 feet) for All Players

Quick analysis script to demonstrate what you can do with the shot distance data.
This reads the shot chart CSVs and analyzes close-range efficiency.
"""

import pandas as pd
import os
from glob import glob

# Path to shot charts
SHOT_CHARTS_DIR = "/Users/thomasmyles/dev/betting/data/01_input/nba_api/shot_charts/2024_25"

def analyze_player_close_range(csv_path, max_distance=6):
    """
    Analyze close-range shooting for a single player
    
    Args:
        csv_path: Path to player's shot chart CSV
        max_distance: Maximum distance in feet (default 6)
    
    Returns:
        Dict with analysis
    """
    df = pd.read_csv(csv_path)
    
    if df.empty:
        return None
    
    player_name = df['PLAYER_NAME'].iloc[0]
    
    # Filter to close-range shots
    close_shots = df[df['SHOT_DISTANCE'] <= max_distance]
    
    if close_shots.empty:
        return None
    
    # Calculate stats
    total_shots = len(df)
    close_attempts = len(close_shots)
    close_makes = close_shots['SHOT_MADE_FLAG'].sum()
    close_fg_pct = (close_makes / close_attempts * 100) if close_attempts > 0 else 0
    close_pct_of_total = (close_attempts / total_shots * 100) if total_shots > 0 else 0
    
    # Get most common shot types at close range
    top_shot_types = close_shots['ACTION_TYPE'].value_counts().head(3).to_dict()
    
    return {
        'player': player_name,
        'total_shots': total_shots,
        'close_attempts': close_attempts,
        'close_pct_of_total': round(close_pct_of_total, 1),
        'close_fg_pct': round(close_fg_pct, 1),
        'close_makes': int(close_makes),
        'top_shot_types': list(top_shot_types.keys())
    }


def analyze_all_players(max_distance=6):
    """
    Analyze close-range shooting for all players with shot data
    """
    print("="*80)
    print(f"ANALYZING CLOSE-RANGE SHOOTING (0-{max_distance} feet)")
    print("="*80)
    
    # Get all shot chart files
    shot_files = glob(os.path.join(SHOT_CHARTS_DIR, "*.csv"))
    
    if not shot_files:
        print(f"\n❌ No shot chart files found in {SHOT_CHARTS_DIR}")
        print("\nRun this first:")
        print("  python scripts/test_fetch_shot_charts.py")
        print("  OR")
        print("  python scripts/fetch_all_nba_shot_charts.py --auto")
        return None
    
    print(f"\n📂 Found {len(shot_files)} player shot charts")
    print(f"📍 Analyzing shots within {max_distance} feet of the basket...\n")
    
    results = []
    
    for csv_path in shot_files:
        analysis = analyze_player_close_range(csv_path, max_distance)
        if analysis:
            results.append(analysis)
    
    if not results:
        print("❌ No data to analyze")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by close-range efficiency
    df = df.sort_values('close_fg_pct', ascending=False)
    
    print("="*80)
    print(f"RESULTS: {len(df)} Players")
    print("="*80)
    
    print("\n🎯 BEST CLOSE-RANGE FINISHERS (sorted by FG%):")
    print(df[['player', 'close_attempts', 'close_fg_pct', 'close_pct_of_total']].to_string(index=False))
    
    print("\n\n📊 LEAGUE STATS:")
    print(f"   Average close-range FG%: {df['close_fg_pct'].mean():.1f}%")
    print(f"   Average close-range attempts: {df['close_attempts'].mean():.1f} per player")
    print(f"   Highest close-range FG%: {df['close_fg_pct'].max():.1f}%")
    print(f"   Lowest close-range FG%: {df['close_fg_pct'].min():.1f}%")
    
    # Show top 3 shot types for best finisher
    best_finisher = df.iloc[0]
    print(f"\n🏀 {best_finisher['player']} - Most Common Shots at Rim:")
    for i, shot_type in enumerate(best_finisher['top_shot_types'], 1):
        print(f"   {i}. {shot_type}")
    
    return df


def compare_distance_ranges():
    """
    Compare different distance ranges (0-3 feet vs 3-6 feet vs 6-10 feet)
    """
    print("\n" + "="*80)
    print("COMPARING DIFFERENT SHOT DISTANCE RANGES")
    print("="*80)
    
    shot_files = glob(os.path.join(SHOT_CHARTS_DIR, "*.csv"))
    
    results = []
    
    for csv_path in shot_files:
        df = pd.read_csv(csv_path)
        
        if df.empty:
            continue
        
        player_name = df['PLAYER_NAME'].iloc[0]
        
        # 0-3 feet (at the rim)
        rim_shots = df[df['SHOT_DISTANCE'] <= 3]
        rim_fg_pct = (rim_shots['SHOT_MADE_FLAG'].sum() / len(rim_shots) * 100) if not rim_shots.empty else 0
        
        # 3-6 feet (short range)
        short_shots = df[(df['SHOT_DISTANCE'] > 3) & (df['SHOT_DISTANCE'] <= 6)]
        short_fg_pct = (short_shots['SHOT_MADE_FLAG'].sum() / len(short_shots) * 100) if not short_shots.empty else 0
        
        # 6-10 feet (floater range)
        floater_shots = df[(df['SHOT_DISTANCE'] > 6) & (df['SHOT_DISTANCE'] <= 10)]
        floater_fg_pct = (floater_shots['SHOT_MADE_FLAG'].sum() / len(floater_shots) * 100) if not floater_shots.empty else 0
        
        results.append({
            'player': player_name,
            'rim_attempts': len(rim_shots),
            'rim_fg_pct': round(rim_fg_pct, 1),
            'short_attempts': len(short_shots),
            'short_fg_pct': round(short_fg_pct, 1),
            'floater_attempts': len(floater_shots),
            'floater_fg_pct': round(floater_fg_pct, 1)
        })
    
    df = pd.DataFrame(results)
    
    # Filter to players with meaningful sample sizes
    df_filtered = df[(df['rim_attempts'] >= 20) & (df['short_attempts'] >= 10) & (df['floater_attempts'] >= 10)]
    
    if df_filtered.empty:
        print("\n⚠️  Not enough data for distance comparison")
        return None
    
    print(f"\n📊 Distance Range Comparison ({len(df_filtered)} players with 20+ rim attempts):")
    print("\nColumns: Rim (0-3ft) | Short (3-6ft) | Floater (6-10ft)")
    print(df_filtered.sort_values('rim_fg_pct', ascending=False).to_string(index=False))
    
    return df


if __name__ == "__main__":
    # Analyze close-range shooting (0-6 feet)
    close_range_df = analyze_all_players(max_distance=6)
    
    # Compare different distance ranges
    if close_range_df is not None and len(close_range_df) > 0:
        distance_comparison = compare_distance_ranges()
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    
    print("\n💡 Next Steps:")
    print("  1. Fetch all ~500 players: python scripts/fetch_all_nba_shot_charts.py --auto")
    print("  2. Read any player's shots: pd.read_csv('path/to/player.csv')")
    print("  3. Filter by distance: df[df['SHOT_DISTANCE'] <= 6]")
    print("  4. Filter by shot type: df[df['ACTION_TYPE'] == 'Driving Layup Shot']")
    print("  5. Filter by opponent: df[df['VTM'] == 'GSW'] or df[df['HTM'] == 'GSW']")


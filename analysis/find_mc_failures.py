"""
Find games where Monte Carlo gives 0% probability but player keeps scoring.

This helps identify systematic model failures.

Usage:
    python analysis/find_mc_failures.py --threshold 0.05
"""

import duckdb
import pandas as pd
from pathlib import Path

# Find all MC result CSVs
PLOTS_DIR = Path.home() / 'dev' / 'betting' / 'src' / 'pbp_data' / 'tmp' / 'plots'

def find_zero_prob_failures():
    """Find games where MC goes to 0% but player keeps scoring after."""
    
    # Combine all CSVs
    csv_pattern = str(PLOTS_DIR / 'monte_carlo_pbp_*.csv')
    
    con = duckdb.connect()
    
    # Find problematic patterns
    query = f"""
    WITH game_data AS (
        SELECT 
            *,
            LAG(cumulative_points, 10) OVER (PARTITION BY game_id ORDER BY game_minute) as points_10_plays_ago,
            LAG(prob_over, 10) OVER (PARTITION BY game_id ORDER BY game_minute) as prob_10_plays_ago
        FROM read_csv_auto('{csv_pattern}', union_by_name=true)
    )
    SELECT 
        game_id,
        game_date,
        player_name,
        game_minute,
        cumulative_points,
        points_10_plays_ago,
        prob_over,
        prob_10_plays_ago,
        cumulative_points - points_10_plays_ago as points_gained,
        prop_line
    FROM game_data
    WHERE prob_over <= 0.05  -- MC says 5% or less
      AND cumulative_points > points_10_plays_ago  -- But player scored
      AND game_minute >= 24  -- Only check 2nd half
      AND points_10_plays_ago IS NOT NULL
    ORDER BY game_id, game_minute
    """
    
    failures = con.execute(query).df()
    con.close()
    
    return failures


def summarize_failures(failures):
    """Summarize the failure cases."""
    
    if len(failures) == 0:
        print("✅ No failures found!")
        return
    
    print(f"\n⚠️  Found {len(failures)} instances of MC at ~0% while player scoring\n")
    
    # Group by game
    games = failures.groupby(['game_id', 'player_name']).agg({
        'game_minute': ['min', 'max'],
        'points_gained': 'sum',
        'prob_over': 'mean'
    }).reset_index()
    
    games.columns = ['game_id', 'player', 'first_minute', 'last_minute', 'points_gained', 'avg_prob']
    
    print("Games with systematic MC failures:")
    print("="*80)
    for _, row in games.iterrows():
        print(f"\n{row['player']} - Game {row['game_id']}")
        print(f"  Minutes {row['first_minute']:.0f}-{row['last_minute']:.0f}: MC avg {row['avg_prob']*100:.1f}%")
        print(f"  But gained {row['points_gained']:.0f} points during this period")
    
    print("\n" + "="*80)
    print(f"\nTotal problematic games: {len(games)}")
    
    return games


def main():
    print("🔍 Searching for Monte Carlo failures...")
    print("   (MC at 0% but player keeps scoring)\n")
    
    failures = find_zero_prob_failures()
    games = summarize_failures(failures)
    
    if len(failures) > 0:
        # Save to CSV
        output_file = PLOTS_DIR.parent / 'mc_failures.csv'
        failures.to_csv(output_file, index=False)
        print(f"\n💾 Detailed failures saved to: {output_file}")


if __name__ == '__main__':
    main()

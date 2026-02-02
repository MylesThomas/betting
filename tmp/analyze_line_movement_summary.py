"""
Single SQL Query Line Movement Summary

Returns 1 row per game with movement deltas and summary stats.

Author: Thomas Myles
Date: 2026-02-01
"""

import duckdb
import pandas as pd
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

COMBINED_ODDS_PATH = Path.home() / 'Downloads' / 'tmp' / 'combined_odds.parquet'
COMBINED_ESPN_PATH = Path.home() / 'Downloads' / 'tmp' / 'combined_espn.parquet'

# Filter config - change this to filter by game status
GAME_STATUS_FILTER = 'in'  # 'in' = live games only, 'pre' = pre-game, 'post' = finished, None = all games


# =============================================================================
# MAIN QUERY
# =============================================================================

def get_line_movement_summary(status_filter=GAME_STATUS_FILTER):
    """
    Get line movement summary with 1 row per game.
    
    Args:
        status_filter: Game status to filter ('in', 'pre', 'post', or None for all)
        
    Returns:
        DataFrame with movement summary
    """
    con = duckdb.connect()
    
    # Build WHERE clause for status filter
    status_where = f"AND e_latest.game_status = '{status_filter}'" if status_filter else ""
    
    query = f"""
    WITH odds_summary AS (
        SELECT
            away_team,
            home_team,
            MIN(fetched_at) as first_snapshot_time,
            MAX(fetched_at) as last_snapshot_time,
            COUNT(DISTINCT fetched_at) as num_snapshots,
            COUNT(DISTINCT bookmaker) as num_bookmakers,
            COUNT(*) as total_records
        FROM '{COMBINED_ODDS_PATH}'
        GROUP BY away_team, home_team
    ),
    
    first_odds AS (
        SELECT
            o.away_team,
            o.home_team,
            MEDIAN(o.away_spread) as first_away_spread,
            MEDIAN(o.home_spread) as first_home_spread,
            MEDIAN(o.away_ml) as first_away_ml,
            MEDIAN(o.home_ml) as first_home_ml
        FROM '{COMBINED_ODDS_PATH}' o
        INNER JOIN odds_summary s
            ON o.away_team = s.away_team
            AND o.home_team = s.home_team
            AND o.fetched_at = s.first_snapshot_time
        GROUP BY o.away_team, o.home_team
    ),
    
    last_odds AS (
        SELECT
            o.away_team,
            o.home_team,
            MEDIAN(o.away_spread) as last_away_spread,
            MEDIAN(o.home_spread) as last_home_spread,
            MEDIAN(o.away_ml) as last_away_ml,
            MEDIAN(o.home_ml) as last_home_ml
        FROM '{COMBINED_ODDS_PATH}' o
        INNER JOIN odds_summary s
            ON o.away_team = s.away_team
            AND o.home_team = s.home_team
            AND o.fetched_at = s.last_snapshot_time
        GROUP BY o.away_team, o.home_team
    ),
    
    latest_espn AS (
        SELECT
            away_team_espn,
            home_team_espn,
            away_score,
            home_score,
            game_status,
            game_status_description,
            period,
            display_clock,
            time_remaining_minutes
        FROM '{COMBINED_ESPN_PATH}'
        WHERE collection_timestamp = (SELECT MAX(collection_timestamp) FROM '{COMBINED_ESPN_PATH}')
    )
    
    SELECT
        s.away_team,
        s.home_team,
        
        -- Current game state
        e_latest.game_status,
        e_latest.away_score,
        e_latest.home_score,
        e_latest.period,
        e_latest.display_clock,
        
        -- First snapshot odds
        f.first_away_spread,
        f.first_home_spread,
        f.first_away_ml,
        f.first_home_ml,
        
        -- Last snapshot odds
        l.last_away_spread,
        l.last_home_spread,
        l.last_away_ml,
        l.last_home_ml,
        
        -- Movement deltas
        (l.last_away_spread - f.first_away_spread) as away_spread_movement,
        (l.last_home_spread - f.first_home_spread) as home_spread_movement,
        (l.last_away_ml - f.first_away_ml) as away_ml_movement,
        (l.last_home_ml - f.first_home_ml) as home_ml_movement,
        
        -- Summary stats
        s.num_snapshots,
        s.num_bookmakers,
        s.total_records,
        s.first_snapshot_time,
        s.last_snapshot_time,
        
        -- Time tracking
        EXTRACT(EPOCH FROM (s.last_snapshot_time - s.first_snapshot_time)) / 60 as minutes_tracked
        
    FROM odds_summary s
    INNER JOIN first_odds f
        ON s.away_team = f.away_team
        AND s.home_team = f.home_team
    INNER JOIN last_odds l
        ON s.away_team = l.away_team
        AND s.home_team = l.home_team
    LEFT JOIN latest_espn e_latest
        ON s.away_team = e_latest.away_team_espn
        AND s.home_team = e_latest.home_team_espn
    WHERE 1=1
        {status_where}
    ORDER BY 
        CASE WHEN e_latest.game_status = 'in' THEN 0 ELSE 1 END,
        ABS(l.last_away_spread - f.first_away_spread) DESC
    """
    
    df = con.execute(query).df()
    con.close()
    
    return df


# =============================================================================
# DISPLAY
# =============================================================================

def display_summary(df):
    """Pretty print the summary dataframe."""
    if len(df) == 0:
        print("❌ No games found")
        return
    
    print("\n" + "="*120)
    print(f"📊 LINE MOVEMENT SUMMARY ({len(df)} games)")
    print("="*120)
    
    for idx, row in df.iterrows():
        matchup = f"{row['away_team']} @ {row['home_team']}"
        
        # Game state
        if pd.notna(row['game_status']):
            status = row['game_status']
            if status == 'in' and pd.notna(row['away_score']):
                score_str = f"[{int(row['away_score'])}-{int(row['home_score'])}]"
            else:
                score_str = ""
        else:
            status = "N/A"
            score_str = ""
        
        print(f"\n🏀 {matchup:<55} {status:<6} {score_str}")
        print("-" * 120)
        
        # First snapshot
        first_spread = f"{row['first_away_spread']:+.1f}/{row['first_home_spread']:+.1f}" if pd.notna(row['first_away_spread']) else "N/A"
        first_ml = f"{int(row['first_away_ml']):+d}/{int(row['first_home_ml']):+d}" if pd.notna(row['first_away_ml']) else "N/A"
        
        # Last snapshot
        last_spread = f"{row['last_away_spread']:+.1f}/{row['last_home_spread']:+.1f}" if pd.notna(row['last_away_spread']) else "N/A"
        last_ml = f"{int(row['last_away_ml']):+d}/{int(row['last_home_ml']):+d}" if pd.notna(row['last_away_ml']) else "N/A"
        
        # Movement
        spread_move = f"{row['away_spread_movement']:+.1f}" if pd.notna(row['away_spread_movement']) else "N/A"
        ml_move = f"{int(row['away_ml_movement']):+d}" if pd.notna(row['away_ml_movement']) else "N/A"
        
        print(f"  First:     Spread: {first_spread:<15} ML: {first_ml:<20}")
        print(f"  Last:      Spread: {last_spread:<15} ML: {last_ml:<20}")
        print(f"  Movement:  Spread: {spread_move:<15} ML: {ml_move:<20}")
        print(f"  Tracked:   {row['num_snapshots']} snapshots over {row['minutes_tracked']:.1f} minutes ({row['num_bookmakers']} bookmakers, {row['total_records']} records)")
    
    print("\n" + "="*120 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the line movement summary query."""
    print("\n" + "="*120)
    print("📈 LIVE ODDS LINE MOVEMENT SUMMARY")
    print("="*120)
    
    # Check if combined files exist
    if not COMBINED_ODDS_PATH.exists() or not COMBINED_ESPN_PATH.exists():
        print("\n❌ Combined parquet files not found!")
        print(f"   Expected: {COMBINED_ODDS_PATH}")
        print(f"   Expected: {COMBINED_ESPN_PATH}")
        print("\n   Run this first: python tmp/read_all_live_odds_parquet.py\n")
        return
    
    print(f"\n✅ Found combined files:")
    print(f"   Odds: {COMBINED_ODDS_PATH}")
    print(f"   ESPN: {COMBINED_ESPN_PATH}")
    
    filter_msg = f"'{GAME_STATUS_FILTER}' games only" if GAME_STATUS_FILTER else "all games"
    print(f"   Filter: {filter_msg}")
    
    # Get summary
    df = get_line_movement_summary(GAME_STATUS_FILTER)
    
    # Display
    display_summary(df)
    
    # Return dataframe for interactive use
    print("💡 Tip: DataFrame is available as 'df' variable for further analysis\n")
    return df


if __name__ == '__main__':
    df = main()

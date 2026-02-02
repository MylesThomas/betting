"""
Analyze Live Odds Line Movement Over Time

Uses DuckDB to join odds + ESPN data and show how lines moved
from first snapshot to most recent for live games.

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


# =============================================================================
# FUNCTIONS
# =============================================================================

def get_available_games():
    """Get list of games in the dataset."""
    con = duckdb.connect()
    
    query = f"""
        SELECT DISTINCT
            away_team,
            home_team,
            MIN(fetched_at) as first_snapshot,
            MAX(fetched_at) as last_snapshot,
            COUNT(DISTINCT fetched_at) as num_snapshots
        FROM '{COMBINED_ODDS_PATH}'
        GROUP BY away_team, home_team
        ORDER BY first_snapshot
    """
    
    games = con.execute(query).df()
    con.close()
    
    return games


def analyze_game_line_movement(away_team: str, home_team: str):
    """
    Show line movement for a specific game from first to last snapshot.
    
    Args:
        away_team: Away team name
        home_team: Home team name
    """
    con = duckdb.connect()
    
    print(f"\n{'='*80}")
    print(f"📊 LINE MOVEMENT: {away_team} @ {home_team}")
    print(f"{'='*80}\n")
    
    # Get line movement with scores (joined with ESPN)
    query = f"""
        SELECT 
            o.fetched_at as collection_timestamp,
            o.bookmaker,
            o.away_spread,
            o.home_spread,
            o.away_ml,
            o.home_ml,
            e.away_score,
            e.home_score,
            e.game_status,
            e.period,
            e.time_remaining_minutes
        FROM '{COMBINED_ODDS_PATH}' o
        LEFT JOIN '{COMBINED_ESPN_PATH}' e
            ON o.fetched_at = e.collection_timestamp
            AND o.away_team = e.away_team_espn
            AND o.home_team = e.home_team_espn
        WHERE o.away_team = '{away_team}'
          AND o.home_team = '{home_team}'
        ORDER BY o.fetched_at, o.bookmaker
    """
    
    df = con.execute(query).df()
    con.close()
    
    if len(df) == 0:
        print("❌ No data found for this game")
        return
    
    print(f"📈 Found {len(df):,} records across {df['collection_timestamp'].nunique()} snapshots\n")
    
    # Group by timestamp to show consensus at each snapshot
    print(f"{'Time':<20} {'Status':<8} {'Score':<12} {'Spread':<15} {'ML':<20}")
    print("-" * 80)
    
    for timestamp in df['collection_timestamp'].unique():
        snapshot = df[df['collection_timestamp'] == timestamp]
        
        # Get timestamp in readable format
        time_str = pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        # Get game state (same for all bookmakers at this timestamp)
        game_status = snapshot['game_status'].iloc[0] if pd.notna(snapshot['game_status'].iloc[0]) else 'N/A'
        away_score = snapshot['away_score'].iloc[0]
        home_score = snapshot['home_score'].iloc[0]
        
        score_str = f"{int(away_score)}-{int(home_score)}" if pd.notna(away_score) and pd.notna(home_score) else 'N/A'
        
        # Calculate median odds across bookmakers
        median_away_spread = snapshot['away_spread'].median()
        median_home_spread = snapshot['home_spread'].median()
        median_away_ml = snapshot['away_ml'].median()
        median_home_ml = snapshot['home_ml'].median()
        
        spread_str = f"{median_away_spread:+.1f}/{median_home_spread:+.1f}" if pd.notna(median_away_spread) else "N/A"
        
        ml_str = f"{int(median_away_ml):+d}/{int(median_home_ml):+d}" if pd.notna(median_away_ml) and pd.notna(median_home_ml) else "N/A"
        
        print(f"{time_str:<20} {game_status:<8} {score_str:<12} {spread_str:<15} {ml_str:<20}")
    
    # Summary stats
    print("\n" + "="*80)
    print("📊 MOVEMENT SUMMARY")
    print("="*80)
    
    first_snapshot = df[df['collection_timestamp'] == df['collection_timestamp'].min()]
    last_snapshot = df[df['collection_timestamp'] == df['collection_timestamp'].max()]
    
    first_spread = first_snapshot['away_spread'].median()
    last_spread = last_snapshot['away_spread'].median()
    spread_movement = last_spread - first_spread if pd.notna(first_spread) and pd.notna(last_spread) else None
    
    first_ml = first_snapshot['away_ml'].median()
    last_ml = last_snapshot['away_ml'].median()
    ml_movement = last_ml - first_ml if pd.notna(first_ml) and pd.notna(last_ml) else None
    
    print(f"Away Team Spread: {first_spread:+.1f} → {last_spread:+.1f} (moved {spread_movement:+.1f})" if spread_movement is not None else "Spread data incomplete")
    print(f"Away Team ML: {int(first_ml):+d} → {int(last_ml):+d} (moved {int(ml_movement):+d})" if ml_movement is not None else "ML data incomplete")
    
    print()


def show_all_games_latest_snapshot():
    """Show consensus odds for all games at the most recent snapshot."""
    con = duckdb.connect()
    
    print(f"\n{'='*80}")
    print(f"📸 LATEST SNAPSHOT - ALL GAMES")
    print(f"{'='*80}\n")
    
    # Get latest timestamp
    latest_time = con.execute(f"SELECT MAX(fetched_at) FROM '{COMBINED_ODDS_PATH}'").fetchone()[0]
    print(f"Latest snapshot: {latest_time}\n")
    
    # Get consensus for all games at latest timestamp
    query = f"""
        SELECT 
            o.away_team,
            o.home_team,
            MEDIAN(o.away_spread) as away_spread,
            MEDIAN(o.home_spread) as home_spread,
            MEDIAN(o.away_ml) as away_ml,
            MEDIAN(o.home_ml) as home_ml,
            e.away_score,
            e.home_score,
            e.game_status
        FROM '{COMBINED_ODDS_PATH}' o
        LEFT JOIN '{COMBINED_ESPN_PATH}' e
            ON o.fetched_at = e.collection_timestamp
            AND o.away_team = e.away_team_espn
            AND o.home_team = e.home_team_espn
        WHERE o.fetched_at = '{latest_time}'
        GROUP BY o.away_team, o.home_team, e.away_score, e.home_score, e.game_status
        ORDER BY e.game_status DESC, o.away_team
    """
    
    games = con.execute(query).df()
    con.close()
    
    if len(games) == 0:
        print("❌ No games found")
        return
    
    # Separate live vs upcoming
    live_games = games[games['game_status'] == 'in']
    upcoming_games = games[games['game_status'] != 'in']
    
    if len(live_games) > 0:
        print(f"🔥 LIVE GAMES ({len(live_games)}):\n")
        for _, game in live_games.iterrows():
            matchup = f"{game['away_team']} @ {game['home_team']}"
            score = f"[{int(game['away_score'])}-{int(game['home_score'])}]" if pd.notna(game['away_score']) else ""
            spread = f"{game['away_spread']:+.1f}/{game['home_spread']:+.1f}" if pd.notna(game['away_spread']) else "N/A"
            ml = f"{int(game['away_ml']):+d}/{int(game['home_ml']):+d}" if pd.notna(game['away_ml']) else "N/A"
            print(f"  {matchup:<50} {score:<10} Spread: {spread:<12} ML: {ml}")
    
    if len(upcoming_games) > 0:
        print(f"\n📊 UPCOMING/FINISHED GAMES ({len(upcoming_games)}):\n")
        for _, game in upcoming_games.iterrows():
            matchup = f"{game['away_team']} @ {game['home_team']}"
            spread = f"{game['away_spread']:+.1f}/{game['home_spread']:+.1f}" if pd.notna(game['away_spread']) else "N/A"
            ml = f"{int(game['away_ml']):+d}/{int(game['home_ml']):+d}" if pd.notna(game['away_ml']) else "N/A"
            status = game['game_status'] if pd.notna(game['game_status']) else 'N/A'
            print(f"  {matchup:<50} [{status:<4}] Spread: {spread:<12} ML: {ml}")
    
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Analyze live odds line movement."""
    print("\n" + "="*80)
    print("📈 LIVE ODDS LINE MOVEMENT ANALYZER")
    print("="*80)
    
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
    
    # Show latest snapshot for all games
    show_all_games_latest_snapshot()
    
    # Get available games
    print("\n" + "="*80)
    print("📋 AVAILABLE GAMES")
    print("="*80)
    
    games = get_available_games()
    
    if len(games) == 0:
        print("\n❌ No games found")
        return
    
    print(f"\nFound {len(games)} games:\n")
    for i, game in games.iterrows():
        print(f"  [{i+1}] {game['away_team']} @ {game['home_team']}")
        print(f"      Snapshots: {game['num_snapshots']} | First: {game['first_snapshot']} | Last: {game['last_snapshot']}")
    
    # Analyze each game's line movement
    print("\n" + "="*80)
    print("📈 DETAILED LINE MOVEMENT (All Games)")
    print("="*80)
    
    for _, game in games.iterrows():
        analyze_game_line_movement(game['away_team'], game['home_team'])
    
    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

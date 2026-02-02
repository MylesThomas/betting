"""
Analyze collected odds data to understand consensus line calculations.

PURPOSE:
Show SQL patterns for computing consensus lines from live odds data.

CONTEXT:
After collecting live odds with multiple bookmakers per game, we need to:
1. Define "consensus line" - e.g., median spread across all books
2. Track how consensus shifts over time
3. Identify which books are ahead/behind consensus

Expected schema from collect_live_game_data.py:
- collection_timestamp: When we queried (ISO timestamp)
- game_id: Unique game identifier
- away_team / home_team: Team names
- bookmaker: Book name (e.g., 'fanduel', 'draftkings')
- away_spread / home_spread: Spread values
- away_spread_price / home_spread_price: Prices (e.g., -110)
- away_ml / home_ml: Moneyline odds

USAGE:
    python tmp/analyze_consensus_lines.py
    python tmp/analyze_consensus_lines.py --matchup "Milwaukee Bucks @ Boston Celtics"
"""

import duckdb
import argparse
from pathlib import Path

# Point to collected data
DATA_DIR = Path.home() / 'Downloads' / 'tmp'
ODDS_FILE = DATA_DIR / 'odds_api_20260201.parquet'

def show_schema():
    """Show the actual schema of collected data."""
    print("=" * 80)
    print("DATA SCHEMA")
    print("=" * 80)
    
    if not ODDS_FILE.exists():
        print(f"❌ No data file found: {ODDS_FILE}")
        print(f"Run: python tmp/collect_live_game_data.py --interval 30")
        return False
    
    con = duckdb.connect(':memory:')
    
    query = f"""
    DESCRIBE SELECT * FROM '{ODDS_FILE}' LIMIT 1;
    """
    
    result = con.execute(query).fetchdf()
    print(result.to_string(index=False))
    
    # Show row count
    count_query = f"SELECT COUNT(*) as total_rows FROM '{ODDS_FILE}'"
    count = con.execute(count_query).fetchone()[0]
    print(f"\nTotal rows: {count}")
    print()
    
    return True


def example_consensus_spread_query(matchup_filter=None):
    """
    SQL Example 1: Consensus spread + ML at a given collection_timestamp.
    
    Strategy: 
    - Use MEDIAN across all bookmakers for a game
    - Only include books updated in last 30s (fresh odds)
    - Get min/max ranges to see market width
    
    Args:
        matchup_filter: Optional string like "Milwaukee Bucks @ Boston Celtics"
    """
    print("=" * 80)
    print("QUERY 1: CONSENSUS SPREAD + ML (MEDIAN ACROSS FRESH BOOKS)")
    if matchup_filter:
        print(f"Filtered to: {matchup_filter}")
    print("=" * 80)
    
    con = duckdb.connect(':memory:')
    
    # Build WHERE clause for matchup filter
    matchup_where = ""
    if matchup_filter:
        matchup_where = f"AND (away_team || ' @ ' || home_team) = '{matchup_filter}'"
    
    sql = f"""
    -- Get consensus spread + ML for each game at each collection_timestamp
    -- Only use bookmakers that updated in last 30 seconds
    WITH fresh_odds AS (
        SELECT 
            *,
            -- Calculate seconds since bookmaker last updated
            EXTRACT(EPOCH FROM (
                collection_timestamp::TIMESTAMP - bookmaker_last_update::TIMESTAMP
            )) AS seconds_since_update
        FROM '{ODDS_FILE}'
        WHERE 1=1
            {matchup_where}
    ),
    consensus AS (
        SELECT 
            collection_timestamp,
            game_id,
            away_team,
            home_team,
            
            -- SPREAD CONSENSUS
            MEDIAN(away_spread) AS consensus_away_spread,
            MEDIAN(home_spread) AS consensus_home_spread,
            MIN(away_spread) AS min_away_spread,
            MAX(away_spread) AS max_away_spread,
            
            -- MONEYLINE CONSENSUS  
            MEDIAN(away_ml) AS consensus_away_ml,
            MEDIAN(home_ml) AS consensus_home_ml,
            MIN(away_ml) AS min_away_ml,
            MAX(away_ml) AS max_away_ml,
            
            -- Metadata
            COUNT(DISTINCT bookmaker) AS num_fresh_books,
            MIN(seconds_since_update) AS newest_update_seconds,
            MAX(seconds_since_update) AS oldest_update_seconds
            
        FROM fresh_odds
        WHERE away_spread IS NOT NULL
            AND seconds_since_update <= 30  -- Only fresh odds (< 30s old)
        GROUP BY collection_timestamp, game_id, away_team, home_team
    )
    SELECT 
        STRFTIME(TIMEZONE('America/New_York', collection_timestamp::TIMESTAMPTZ), '%I:%M:%S %p') AS time_et,
        away_team || ' @ ' || home_team AS matchup,
        
        -- Spread market (both sides)
        consensus_away_spread,
        consensus_home_spread,
        min_away_spread || ' to ' || max_away_spread AS spread_range,
        
        -- ML market (both sides)
        consensus_away_ml,
        consensus_home_ml,
        min_away_ml || ' to ' || max_away_ml AS ml_range,
        
        num_fresh_books,
        ROUND(newest_update_seconds, 1) AS newest_update_seconds,
        ROUND(oldest_update_seconds, 1) AS oldest_update_seconds
        
    FROM consensus
    ORDER BY collection_timestamp DESC
    LIMIT 10;
    """
    
    try:
        result = con.execute(sql).fetchdf()
        print(result.to_string(index=False))
        print(f"\n✅ Returned {len(result)} rows\n")
    except Exception as e:
        print(f"❌ Query failed: {e}\n")


def example_consensus_with_books_query(matchup_filter=None):
    """
    SQL Example 2: Show all books vs consensus at a specific time.
    
    Strategy: 
    - Join individual books back to consensus to see deviations
    - Only compare fresh books (updated in last 30s)
    - Show deviations for both spread and ML
    
    Args:
        matchup_filter: Optional string like "Milwaukee Bucks @ Boston Celtics"
    """
    print("=" * 80)
    print("QUERY 2: INDIVIDUAL BOOKS VS CONSENSUS (SPREAD + ML)")
    if matchup_filter:
        print(f"Filtered to: {matchup_filter}")
    print("=" * 80)
    
    con = duckdb.connect(':memory:')
    
    # Build WHERE clause for matchup filter
    matchup_where = ""
    if matchup_filter:
        matchup_where = f"AND (away_team || ' @ ' || home_team) = '{matchup_filter}'"
    
    sql = f"""
    -- Show how each book compares to consensus
    WITH fresh_odds AS (
        SELECT 
            *,
            EXTRACT(EPOCH FROM (
                collection_timestamp::TIMESTAMP - bookmaker_last_update::TIMESTAMP
            )) AS seconds_since_update
        FROM '{ODDS_FILE}'
        WHERE EXTRACT(EPOCH FROM (
            collection_timestamp::TIMESTAMP - bookmaker_last_update::TIMESTAMP
        )) <= 30  -- Only fresh books
            {matchup_where}
    ),
    consensus AS (
        SELECT 
            collection_timestamp,
            game_id,
            MEDIAN(away_spread) AS consensus_away_spread,
            MEDIAN(home_spread) AS consensus_home_spread,
            MEDIAN(away_ml) AS consensus_away_ml,
            MEDIAN(home_ml) AS consensus_home_ml
        FROM fresh_odds
        WHERE away_spread IS NOT NULL
        GROUP BY collection_timestamp, game_id
    )
    SELECT 
        STRFTIME(TIMEZONE('America/New_York', o.collection_timestamp::TIMESTAMPTZ), '%I:%M:%S %p') AS time_et,
        o.away_team || ' @ ' || o.home_team AS matchup,
        o.bookmaker,
        
        -- Spread market (both sides)
        o.away_spread AS book_away_spread,
        o.home_spread AS book_home_spread,
        c.consensus_away_spread,
        c.consensus_home_spread,
        ROUND(o.away_spread - c.consensus_away_spread, 1) AS away_spread_vs_consensus,
        
        -- ML market (both sides)
        o.away_ml AS book_away_ml,
        o.home_ml AS book_home_ml,
        c.consensus_away_ml,
        c.consensus_home_ml,
        o.away_ml - c.consensus_away_ml AS away_ml_vs_consensus,
        
        ROUND(o.seconds_since_update, 1) AS seconds_old
        
    FROM fresh_odds o
    JOIN consensus c 
        ON o.collection_timestamp = c.collection_timestamp 
        AND o.game_id = c.game_id
    WHERE o.away_spread IS NOT NULL
    ORDER BY o.collection_timestamp DESC, ABS(o.away_spread - c.consensus_away_spread) DESC
    LIMIT 20;
    """
    
    try:
        result = con.execute(sql).fetchdf()
        print(result.to_string(index=False))
        print(f"\n✅ Returned {len(result)} rows\n")
    except Exception as e:
        print(f"❌ Query failed: {e}\n")


def example_line_movement_query(matchup_filter=None):
    """
    SQL Example 3: Track consensus line movement over time (spread + ML).
    
    Strategy: 
    - Compare consensus at different collection_timestamps for same game
    - Only use fresh books (< 30s old) for each consensus calculation
    - Track both spread and ML movement
    
    Args:
        matchup_filter: Optional string like "Milwaukee Bucks @ Boston Celtics"
    """
    print("=" * 80)
    print("QUERY 3: CONSENSUS LINE MOVEMENT OVER TIME (SPREAD + ML)")
    if matchup_filter:
        print(f"Filtered to: {matchup_filter}")
    print("=" * 80)
    
    con = duckdb.connect(':memory:')
    
    # Build WHERE clause for matchup filter
    matchup_where = ""
    if matchup_filter:
        matchup_where = f"AND (away_team || ' @ ' || home_team) = '{matchup_filter}'"
    
    sql = f"""
    -- Track how consensus spread + ML changes during the game
    WITH fresh_odds AS (
        SELECT 
            *,
            EXTRACT(EPOCH FROM (
                collection_timestamp::TIMESTAMP - bookmaker_last_update::TIMESTAMP
            )) AS seconds_since_update
        FROM '{ODDS_FILE}'
        WHERE EXTRACT(EPOCH FROM (
            collection_timestamp::TIMESTAMP - bookmaker_last_update::TIMESTAMP
        )) <= 30
            {matchup_where}
    ),
            consensus AS (
        SELECT 
            collection_timestamp,
            game_id,
            away_team,
            home_team,
            MEDIAN(away_spread) AS consensus_away_spread,
            MEDIAN(home_spread) AS consensus_home_spread,
            MEDIAN(away_ml) AS consensus_away_ml,
            MEDIAN(home_ml) AS consensus_home_ml,
            COUNT(DISTINCT bookmaker) AS num_fresh_books
        FROM fresh_odds
        WHERE away_spread IS NOT NULL
        GROUP BY collection_timestamp, game_id, away_team, home_team
    ),
    with_lag AS (
        SELECT 
            *,
            LAG(consensus_away_spread) OVER (
                PARTITION BY game_id 
                ORDER BY collection_timestamp
            ) AS prev_spread,
            LAG(consensus_away_ml) OVER (
                PARTITION BY game_id 
                ORDER BY collection_timestamp
            ) AS prev_ml,
            LAG(collection_timestamp) OVER (
                PARTITION BY game_id 
                ORDER BY collection_timestamp
            ) AS prev_collection_timestamp
        FROM consensus
    )
    SELECT 
        STRFTIME(TIMEZONE('America/New_York', collection_timestamp::TIMESTAMPTZ), '%I:%M:%S %p') AS time_et,
        away_team || ' @ ' || home_team AS matchup,
        
        -- Spread movement (both sides)
        consensus_away_spread AS current_away_spread,
        consensus_home_spread AS current_home_spread,
        prev_spread AS prev_away_spread,
        ROUND(consensus_away_spread - prev_spread, 1) AS spread_movement,
        
        -- ML movement (both sides)
        consensus_away_ml AS current_away_ml,
        consensus_home_ml AS current_home_ml,
        prev_ml AS prev_away_ml,
        consensus_away_ml - prev_ml AS ml_movement,
        
        -- Time between collections
        ROUND(EXTRACT(EPOCH FROM (collection_timestamp::TIMESTAMP - prev_collection_timestamp::TIMESTAMP)), 0) AS seconds_elapsed,
        
        num_fresh_books
        
    FROM with_lag
    WHERE prev_spread IS NOT NULL
    ORDER BY collection_timestamp DESC, ABS(consensus_away_spread - prev_spread) DESC
    LIMIT 20;
    """
    
    try:
        result = con.execute(sql).fetchdf()
        print(result.to_string(index=False))
        print(f"\n✅ Returned {len(result)} rows\n")
    except Exception as e:
        print(f"❌ Query failed: {e}\n")


def example_best_price_query(matchup_filter=None):
    """
    SQL Example 4: Find best price for a given spread.
    
    Strategy: For a target spread (e.g., -5.5), find which book offers best price.
    
    Args:
        matchup_filter: Optional string like "Milwaukee Bucks @ Boston Celtics"
    """
    print("=" * 80)
    print("QUERY 4: BEST PRICE FOR TARGET SPREAD")
    if matchup_filter:
        print(f"Filtered to: {matchup_filter}")
    print("=" * 80)
    
    con = duckdb.connect(':memory:')
    
    # Build WHERE clause for matchup filter
    matchup_where = ""
    if matchup_filter:
        matchup_where = f"AND (away_team || ' @ ' || home_team) = '{matchup_filter}'"
    
    sql = f"""
    -- Find best price for spread closest to consensus
    WITH fresh_odds AS (
        SELECT 
            *,
            EXTRACT(EPOCH FROM (
                collection_timestamp::TIMESTAMP - bookmaker_last_update::TIMESTAMP
            )) AS seconds_since_update
        FROM '{ODDS_FILE}'
        WHERE EXTRACT(EPOCH FROM (
            collection_timestamp::TIMESTAMP - bookmaker_last_update::TIMESTAMP
        )) <= 30
            {matchup_where}
    ),
    consensus AS (
        SELECT 
            collection_timestamp,
            game_id,
            MEDIAN(away_spread) AS consensus_away_spread
        FROM fresh_odds
        WHERE away_spread IS NOT NULL
        GROUP BY collection_timestamp, game_id
    ),
    ranked AS (
        SELECT 
            STRFTIME(TIMEZONE('America/New_York', o.collection_timestamp::TIMESTAMPTZ), '%I:%M:%S %p') AS time_et,
            o.away_team || ' @ ' || o.home_team AS matchup,
            c.consensus_away_spread,
            o.bookmaker,
            o.away_spread,
            o.away_spread_price,
            o.collection_timestamp,
            
            -- Rank books by price (higher is better for bettor)
            RANK() OVER (
                PARTITION BY o.collection_timestamp, o.game_id 
                ORDER BY o.away_spread_price DESC
            ) AS price_rank
            
        FROM fresh_odds o
        JOIN consensus c 
            ON o.collection_timestamp = c.collection_timestamp 
            AND o.game_id = c.game_id
        WHERE o.away_spread IS NOT NULL
            AND ABS(o.away_spread - c.consensus_away_spread) < 0.5  -- Within 0.5 of consensus
    )
    SELECT 
        time_et,
        matchup,
        consensus_away_spread,
        bookmaker,
        away_spread,
        away_spread_price,
        price_rank
    FROM ranked
    WHERE price_rank <= 3  -- Top 3 prices only
    ORDER BY collection_timestamp DESC, price_rank
    LIMIT 20;
    """
    
    try:
        result = con.execute(sql).fetchdf()
        print(result.to_string(index=False))
        print(f"\n✅ Returned {len(result)} rows\n")
    except Exception as e:
        print(f"❌ Query failed: {e}\n")


def main():
    """Run all queries on actual data."""
    parser = argparse.ArgumentParser(
        description='Analyze consensus lines from collected odds data'
    )
    parser.add_argument(
        '--matchup',
        type=str,
        help='Filter to specific matchup (e.g., "Milwaukee Bucks @ Boston Celtics")'
    )
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("CONSENSUS LINE CALCULATION - ACTUAL QUERIES")
    if args.matchup:
        print(f"FILTERED TO: {args.matchup}")
    print("=" * 80 + "\n")
    
    # Check if data exists
    has_data = show_schema()
    
    if not has_data:
        return
    
    # Run all queries with optional matchup filter
    example_consensus_spread_query(args.matchup)
    example_consensus_with_books_query(args.matchup)
    example_line_movement_query(args.matchup)
    example_best_price_query(args.matchup)
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key patterns for consensus lines:

1. CONSENSUS = MEDIAN(spread/ML) across FRESH books only
   - Fresh = updated in last 30 seconds (use bookmaker_last_update)
   - More robust than AVG (not skewed by outliers)
   - Stale odds can mislead - always filter by recency!

2. MIN/MAX RANGES show market width
   - Tight range = books agree (sharp line)
   - Wide range = uncertainty or slow books
   - Track ranges for both spread AND moneyline

3. TRACK DEVIATIONS = book_value - consensus_value
   - Find books ahead/behind market (both spread + ML)
   - Identify sharp books (move first) vs slow books
   - Only compare books with fresh updates (< 30s)

4. LINE MOVEMENT = current_consensus - previous_consensus
   - Track movement for BOTH spread and ML
   - Detect when market is shifting
   - Join with ESPN scores to see why (e.g., injury, momentum)

5. BEST PRICE = find highest price at consensus spread
   - E.g., everyone at -5.5, but DraftKings at -105 vs FanDuel at -110
   - Value for bettors = better odds on same outcome
   - Only search among fresh books

CRITICAL: Always filter by recency (bookmaker_last_update)
- Stale odds (> 30s old) can skew consensus
- During live games, books update at different speeds
- Sharp books move first, slow books lag behind

Next steps:
- Run collector during games to build real dataset
- Test these queries on actual data
- Tune recency threshold (30s vs 60s?)
- Consider weighting by book reputation (sharp vs recreational)
""")


if __name__ == '__main__':
    main()

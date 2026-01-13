"""
Load and Join NFL QB Playoff Game Logs with Rushing Props

Context:
--------
Combines two data sources:
1. QB playoff game logs (2001-2024): ESPN box scores with actual rushing yards
2. Historical rushing props (2023-26): The Odds API betting lines

Goal:
-----
Join these datasets to analyze:
- How QBs perform vs their rushing props in playoff games
- First playoff game rushing props specifically
- Bookmaker line accuracy

Data Sources:
-------------
1. s3://nfl-betting-mt/data/01_input/espn_web/playoffs/qb/gamelogs/
   - 584 playoff games, 109 QBs, seasons 2001-2024
   - Columns: athlete, season, date, opponent, attempts, passing_yards, rushing_yds, etc.

2. s3://the-odds-api-mt/nfl/historical_player_props/{season}/player_rush_yds/
   - Betting lines from 2023-24, 2024-25, 2025-26 playoff seasons
   - Columns: player_name, commence_time, bookmaker, line, over_price, under_price, etc.

Join Strategy:
--------------
Match on:
- Player name (fuzzy matching needed: "Patrick Mahomes" vs "P. Mahomes")
- Date (standardize both to YYYY-MM-DD)
- Season (ensure props match game season)

Usage:
------
cd betting
python analysis/load_qb_playoff_rushing_data.py

Returns:
--------
- df_qb: All QB playoff game logs
- df_props: All rushing props
- df_joined: Merged dataset (QB games with props)

Created: 2026-01-12
Author: Thomas Myles
"""

import pandas as pd
import boto3
from io import StringIO
import os
from datetime import datetime
import re

# =============================================================================
# CONFIGURATION
# =============================================================================

# S3 buckets
NFL_BETTING_BUCKET = 'nfl-betting-mt'
ODDS_API_BUCKET = 'the-odds-api-mt'

# S3 paths
QB_GAMELOGS_PREFIX = 'data/01_input/espn_web/playoffs/qb/gamelogs'
PROPS_PREFIX = 'nfl/historical_player_props'

# Seasons with playoff props data
PROP_SEASONS = ['2023-24', '2024-25', '2025-26']

s3_client = boto3.client('s3')

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def list_s3_files(bucket, prefix):
    """List all files in S3 bucket with prefix"""
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    files = []
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.csv'):
                    files.append(obj['Key'])
    
    return files

def load_csv_from_s3(bucket, key):
    """Load CSV file from S3 into DataFrame"""
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

def normalize_name(name):
    """
    Normalize player name for matching.
    
    Examples:
        "Patrick Mahomes II" -> "patrick mahomes"
        "P. Mahomes" -> "p mahomes"
        "Jalen Hurts" -> "jalen hurts"
    """
    name = str(name).lower().strip()
    # Remove suffixes (Jr, Sr, II, III, IV)
    name = re.sub(r'\s+(jr\.?|sr\.?|ii|iii|iv)$', '', name, flags=re.IGNORECASE)
    # Remove periods and extra spaces
    name = re.sub(r'\.', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name

def parse_espn_date(date_str, season):
    """
    Parse ESPN date format to YYYY-MM-DD.
    
    ESPN format: "Sat 1/11" or "Sun 2/9"
    Need to infer year from season (playoffs span two calendar years)
    
    Args:
        date_str: Like "Sat 1/11" or "Sun 2/9"
        season: Like 2024 (means 2024-25 season)
    
    Returns:
        str: YYYY-MM-DD format
    """
    # Extract month and day
    match = re.search(r'(\d{1,2})/(\d{1,2})', date_str)
    if not match:
        return None
    
    month, day = int(match.group(1)), int(match.group(2))
    
    # Playoffs are Jan/Feb of NEXT calendar year
    # Example: 2024 season = playoffs in Jan/Feb 2025
    year = season + 1
    
    # Handle edge case: if month is 12, it's pre-playoffs (shouldn't happen)
    if month == 12:
        year = season
    
    return f"{year}-{month:02d}-{day:02d}"

def parse_odds_api_date(date_str):
    """
    Parse Odds API datetime to YYYY-MM-DD.
    
    Odds API format: "2025-01-11T18:30:00Z" or similar ISO format
    
    Returns:
        str: YYYY-MM-DD format
    """
    try:
        dt = pd.to_datetime(date_str)
        return dt.strftime('%Y-%m-%d')
    except:
        return None

# =============================================================================
# LOAD QB GAME LOGS
# =============================================================================

def load_all_qb_gamelogs():
    """Load master QB playoff game log file"""
    print("Loading QB playoff game logs from S3...")
    
    # List files in QB gamelogs directory
    files = list_s3_files(NFL_BETTING_BUCKET, QB_GAMELOGS_PREFIX)
    
    # Find master file (most recent)
    master_files = [f for f in files if 'nfl_all_qb_playoff_gamelogs' in f]
    
    if not master_files:
        raise ValueError("No master QB gamelog file found!")
    
    # Find the file matching our prop seasons (2023-2025)
    # Prefer recent narrow range over full history, as we only have props for 2023-26
    def parse_file_range(filename):
        """Extract start_year, end_year, date from filename"""
        # Example: nfl_all_qb_playoff_gamelogs_2023_2025_20260112.csv
        import re
        match = re.search(r'(\d{4})_(\d{4})_(\d{8})', filename)
        if match:
            start_year = int(match.group(1))
            end_year = int(match.group(2))
            date = int(match.group(3))
            
            # Prioritize files that cover our prop seasons (2023-2025)
            covers_prop_seasons = (start_year <= 2023 and end_year >= 2025)
            
            # Sort by: covers prop seasons (True first), then most recent date
            return (covers_prop_seasons, date)
        return (False, 0)
    
    # Sort to get file that covers prop seasons with most recent date
    master_file = sorted(master_files, key=parse_file_range, reverse=True)[0]
    
    print(f"  Loading: {master_file}")
    df = load_csv_from_s3(NFL_BETTING_BUCKET, master_file)
    
    # Clean and standardize
    df['athlete_normalized'] = df['athlete'].apply(normalize_name)
    df['game_date'] = df.apply(lambda row: parse_espn_date(row['date'], row['season']), axis=1)
    
    # Add season_str for matching with props (e.g., 2024 -> "2024-25")
    df['season_str'] = df['season'].apply(lambda x: f"{x}-{str(x+1)[-2:]}")
    
    # Add flag for FIRST playoff game for each QB
    # Sort by athlete and date, then mark first game
    df = df.sort_values(['athlete', 'game_date']).reset_index(drop=True)
    df['is_first_playoff_game'] = ~df.duplicated(subset=['athlete'], keep='first')
    
    # Count and show first playoff games
    first_games_count = df['is_first_playoff_game'].sum()
    
    print(f"  ✅ Loaded {len(df)} playoff games for {df['athlete'].nunique()} QBs")
    print(f"     {first_games_count} first playoff games identified")
    
    return df

# =============================================================================
# LOAD RUSHING PROPS
# =============================================================================

def load_rushing_props_for_season(season):
    """Load all rushing props for a specific season"""
    print(f"\n  {season}:")
    
    season_prefix = f"{PROPS_PREFIX}/{season}/player_rush_yds"
    files = list_s3_files(ODDS_API_BUCKET, season_prefix)
    
    if not files:
        print(f"    ⚠️  No files found")
        return pd.DataFrame()
    
    print(f"    Found {len(files)} files")
    
    # Load all CSV files for this season
    dfs = []
    for file_key in files:
        df = load_csv_from_s3(ODDS_API_BUCKET, file_key)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"    ✅ Loaded {len(combined)} prop records")
    
    return combined

def load_all_rushing_props():
    """Load rushing props for all available seasons"""
    print("\nLoading historical rushing props from S3...")
    
    all_props = []
    
    for season in PROP_SEASONS:
        df = load_rushing_props_for_season(season)
        if len(df) > 0:
            df['season_str'] = season
            all_props.append(df)
    
    if not all_props:
        print("\n  ⚠️  No props data found")
        return pd.DataFrame()
    
    combined = pd.concat(all_props, ignore_index=True)
    
    # Clean and standardize
    # Props data has player name in 'player' column
    if 'player' not in combined.columns:
        raise ValueError(f"Cannot find 'player' column in props data. Columns: {combined.columns.tolist()}")
    
    combined['player_normalized'] = combined['player'].apply(normalize_name)
    
    # Parse game_time to game_date
    if 'game_time' in combined.columns:
        combined['game_date'] = combined['game_time'].apply(parse_odds_api_date)
    else:
        raise ValueError("Cannot find game_time column in props data")
    
    print(f"\n  ✅ Total: {len(combined)} prop records across {len(all_props)} seasons")
    return combined

# =============================================================================
# JOIN DATASETS
# =============================================================================

def join_qb_games_with_props(df_qb, df_props):
    """
    Join QB game logs with rushing props.
    
    Join Strategy (by player + game):
    ---------------------------------
    For each QB's playoff game, find the betting prop for THAT specific game:
    
    1. Player match: "Patrick Mahomes" (QB data) = "Patrick Mahomes" (prop data)
    2. Game match: "2025-01-18" (game date) = "2025-01-18" (prop commence date)
    3. Season match: "2024-25" (ensures correct season)
    
    Example:
        QB Game: Mahomes, 2025-01-18, rushed for 45 yards
        Prop: Mahomes, 2025-01-18, line was 25.5 yards
        Result: Joined row shows actual (45) vs line (25.5)
    
    Returns:
        DataFrame: Each row = one QB's game with actual rushing yards + prop line
    """
    print("\n" + "="*80)
    print("JOINING QB GAMES WITH PROPS (BY PLAYER + GAME)")
    print("="*80)
    
    if len(df_props) == 0:
        print("⚠️  No props data to join")
        return pd.DataFrame()
    
    # Show BEFORE filtering: all QB data
    print(f"\nBEFORE FILTERING (All QB data):")
    print(f"  Total QB playoff games: {len(df_qb)}")
    print(f"  QBs: {df_qb['athlete'].nunique()}")
    print(f"  Seasons: {df_qb['season'].min()}-{df_qb['season'].max()}")
    
    # Filter QB data to only prop seasons (2023-24 and later)
    df_qb_prop_seasons = df_qb[df_qb['season_str'].isin(PROP_SEASONS)].copy()
    
    print(f"\nAFTER FILTERING (Only prop seasons: {', '.join(PROP_SEASONS)}):")
    print(f"  QB games in prop seasons: {len(df_qb_prop_seasons)}")
    print(f"  QBs in prop seasons: {df_qb_prop_seasons['athlete'].nunique()}")
    print(f"  Unique game dates: {df_qb_prop_seasons['game_date'].nunique()}")
    
    # Inner join: only keep QB games that have matching props
    # Join keys ensure we match EACH PLAYER'S SPECIFIC GAME
    df_joined = df_qb_prop_seasons.merge(
        df_props,
        left_on=['athlete_normalized', 'game_date', 'season_str'],
        right_on=['player_normalized', 'game_date', 'season_str'],
        how='inner',
        suffixes=('_actual', '_prop')  # Distinguish overlapping columns
    )
    
    print(f"\nJOIN RESULTS:")
    print(f"  ✅ Matched: {len(df_joined)} QB game + prop combinations")
    print(f"     Unique QBs with props: {df_joined['athlete'].nunique()}")
    print(f"     Unique games with props: {df_joined['game_date'].nunique()}")
    
    # Calculate join rate
    join_rate_games = (len(df_joined) / len(df_qb_prop_seasons) * 100) if len(df_qb_prop_seasons) > 0 else 0
    join_rate_qbs = (df_joined['athlete'].nunique() / df_qb_prop_seasons['athlete'].nunique() * 100) if df_qb_prop_seasons['athlete'].nunique() > 0 else 0
    
    print(f"\n  📊 JOIN RATE:")
    print(f"     Games: {join_rate_games:.1f}% ({len(df_joined)}/{len(df_qb_prop_seasons)})")
    print(f"     QBs: {join_rate_qbs:.1f}% ({df_joined['athlete'].nunique()}/{df_qb_prop_seasons['athlete'].nunique()})")
    
    # Sample matched games
    if len(df_joined) > 0:
        print(f"\n  Sample Matched Games (multiple bookmakers per game):")
        for _, row in df_joined[['athlete', 'game_date', 'opponent']].drop_duplicates().head(5).iterrows():
            print(f"    - {row['athlete']} on {row['game_date']} vs {row['opponent']}")
    
    return df_joined

def get_consensus_lines(df_joined):
    """
    Aggregate multiple bookmaker lines to get consensus (median) for each player/game.
    
    Strategy:
    ---------
    For each unique player + game combination:
    1. We have 5-10 different bookmaker lines (DraftKings, FanDuel, etc.)
    2. Take MEDIAN of prop_line across all bookmakers
    3. Result: ONE row per player per game with consensus line
    
    Example:
        Before:
            Mahomes, 2025-01-18, DraftKings, line=25.5
            Mahomes, 2025-01-18, FanDuel, line=26.5
            Mahomes, 2025-01-18, BetMGM, line=25.0
        
        After:
            Mahomes, 2025-01-18, consensus_line=25.5 (median)
    
    Returns:
        DataFrame: One row per player per game with consensus line
    """
    print("\n" + "="*80)
    print("CALCULATING CONSENSUS LINES (MEDIAN BY PLAYER/GAME)")
    print("="*80)
    
    if len(df_joined) == 0:
        print("⚠️  No joined data to aggregate")
        return pd.DataFrame()
    
    print(f"\nBefore aggregation: {len(df_joined)} rows (multiple bookmakers)")
    
    # Determine column names after merge (they might have _actual or _prop suffixes)
    season_col = 'season_actual' if 'season_actual' in df_joined.columns else 'season'
    
    # Group by player + game and aggregate
    groupby_cols = ['athlete', 'athlete_id', 'game_date', season_col, 'season_str', 'opponent']
    
    # Add playoff_round if it exists
    if 'playoff_round' in df_joined.columns:
        groupby_cols.append('playoff_round')
    
    # Build aggregation dict
    agg_dict = {
        # Actual game stats (same across all bookmakers, just take first)
        'rushing_yds': 'first',
        'rushing_car': 'first',
        'rushing_avg': 'first',
        'passing_yards': 'first',
        'passing_tds': 'first',
        'attempts': 'first',
        'result': 'first',
        
        # First playoff game flag
        'is_first_playoff_game': 'first',
        
        # Consensus prop line (MEDIAN across bookmakers)
        'prop_line': 'median',
        
        # Count how many bookmakers
        'bookmaker': 'count',
    }
    
    consensus = df_joined.groupby(
        groupby_cols,
        dropna=False
    ).agg(agg_dict).reset_index()
    
    # Rename season column back to 'season' if it was suffixed
    if season_col == 'season_actual':
        consensus = consensus.rename(columns={'season_actual': 'season'})
    
    # Rename for clarity
    consensus = consensus.rename(columns={
        'prop_line': 'consensus_line',
        'bookmaker': 'num_bookmakers'
    })
    
    # Calculate difference: actual vs consensus
    consensus['diff_vs_consensus'] = consensus['rushing_yds'] - consensus['consensus_line']
    consensus['beat_line'] = consensus['diff_vs_consensus'] > 0
    
    print(f"After aggregation: {len(consensus)} rows (one per player/game)")
    print(f"\nConsensus Stats:")
    print(f"  Unique QBs: {consensus['athlete'].nunique()}")
    print(f"  Unique games: {consensus['game_date'].nunique()}")
    print(f"  Avg bookmakers per game: {consensus['num_bookmakers'].mean():.1f}")
    
    # Show sample
    print(f"\nSample Consensus Lines:")
    display_cols = ['athlete', 'game_date', 'is_first_playoff_game', 'rushing_yds', 'consensus_line', 'diff_vs_consensus', 'num_bookmakers']
    print(consensus[display_cols].head(10).to_string(index=False))
    
    # Show first playoff games specifically
    first_games = consensus[consensus['is_first_playoff_game'] == True].sort_values('game_date')
    print(f"\n🏈 FIRST PLAYOFF GAMES:")
    print(f"   {len(first_games)} QBs in their first playoff game")
    if len(first_games) > 0:
        print(f"\n   All First Playoff Games:")
        print(f"     {'HIT':<4} {'QB':<20} {'Date':<12} {'Opponent':<8} {'Result':<8} | {'Rush Yds':>8} vs {'Line':>5} | {'Diff':>6} | {'Pass Yds':<8} {'Pass TD':<7}")
        print(f"     {'-'*4} {'-'*20} {'-'*12} {'-'*8} {'-'*8}   {'-'*8}    {'-'*5}   {'-'*6}   {'-'*8} {'-'*7}")
        for _, row in first_games.iterrows():
            result_emoji = "✅" if row['beat_line'] else "❌"
            diff = row['diff_vs_consensus']
            result_str = row.get('result', 'N/A')
            opponent = row.get('opponent', 'N/A')
            pass_yds = row.get('passing_yards', 0)
            pass_tds = row.get('passing_tds', 0)
            
            print(f"     {result_emoji:<4} {row['athlete']:20s} {row['game_date']:<12} {opponent:<8} {result_str:<8} | {row['rushing_yds']:>8.0f} vs {row['consensus_line']:>5.1f} | {diff:>+6.1f} | {pass_yds:>8.0f} {pass_tds:>7.0f}")
    
    # === GROUP BY FIRST PLAYOFF GAME: OVER/UNDER RATES ===
    print(f"\n" + "="*80)
    print("OVER/UNDER RATES BY FIRST PLAYOFF GAME STATUS")
    print("="*80)
    
    groupby_first_game = consensus.groupby('is_first_playoff_game').agg({
        'beat_line': ['count', 'sum', 'mean'],
        'diff_vs_consensus': 'mean'
    }).round(3)
    
    # Flatten column names
    groupby_first_game.columns = ['total_games', 'beat_line_count', 'beat_line_rate', 'avg_diff']
    groupby_first_game = groupby_first_game.reset_index()
    
    # Add formatted columns for display
    groupby_first_game['beat_line_pct'] = (groupby_first_game['beat_line_rate'] * 100).round(1)
    groupby_first_game['record'] = groupby_first_game.apply(
        lambda x: f"{int(x['beat_line_count'])}/{int(x['total_games'])}", axis=1
    )
    
    # Display results
    for _, row in groupby_first_game.iterrows():
        first_game_status = "FIRST PLAYOFF GAME" if row['is_first_playoff_game'] else "VETERAN (not first)"
        emoji = "🆕" if row['is_first_playoff_game'] else "🏆"
        
        print(f"\n{emoji} {first_game_status}:")
        print(f"   Total Games: {int(row['total_games'])}")
        print(f"   Beat Line: {row['record']} ({row['beat_line_pct']:.1f}%)")
        print(f"   Avg Difference: {row['avg_diff']:+.1f} yards vs consensus")
    
    # Calculate the edge
    first_rate = groupby_first_game[groupby_first_game['is_first_playoff_game'] == True]['beat_line_pct'].values
    veteran_rate = groupby_first_game[groupby_first_game['is_first_playoff_game'] == False]['beat_line_pct'].values
    
    if len(first_rate) > 0 and len(veteran_rate) > 0:
        edge = first_rate[0] - veteran_rate[0]
        print(f"\n{'🔥' if edge > 0 else '❄️'} EDGE: First playoff game QBs hit at {edge:+.1f}% higher rate")
        
        if edge > 5:
            print(f"   💡 INSIGHT: First playoff game QBs significantly OUTPERFORM their lines!")
        elif edge < -5:
            print(f"   ⚠️  INSIGHT: First playoff game QBs significantly UNDERPERFORM their lines!")
        else:
            print(f"   ℹ️  INSIGHT: No significant edge detected (within 5%)")
    
    # === DETAILED BREAKDOWN: ALL QB/GAME COMBINATIONS ===
    print(f"\n" + "="*80)
    print("DETAILED BREAKDOWN: ALL QB/GAME COMBINATIONS")
    print("="*80)
    
    # First playoff games
    veteran_games = consensus[consensus['is_first_playoff_game'] == False].sort_values('game_date')
    
    print(f"\n🆕 FIRST PLAYOFF GAMES ({len(first_games)} games):")
    print(f"{'HIT':<4} {'QB':<20} {'Date':<12} {'Opponent':<8} {'Result':<8} | {'Rush Yds':>8} vs {'Line':>5} | {'Diff':>6}")
    print(f"{'-'*4} {'-'*20} {'-'*12} {'-'*8} {'-'*8}   {'-'*8}    {'-'*5}   {'-'*6}")
    for _, row in first_games.iterrows():
        result_emoji = "✅" if row['beat_line'] else "❌"
        diff = row['diff_vs_consensus']
        result_str = row.get('result', 'N/A')
        opponent = row.get('opponent', 'N/A')
        print(f"{result_emoji:<4} {row['athlete']:20s} {row['game_date']:<12} {opponent:<8} {result_str:<8} | {row['rushing_yds']:>8.0f} vs {row['consensus_line']:>5.1f} | {diff:>+6.1f}")
    
    print(f"\n🏆 VETERAN QB GAMES ({len(veteran_games)} games):")
    print(f"{'HIT':<4} {'QB':<20} {'Date':<12} {'Opponent':<8} {'Result':<8} | {'Rush Yds':>8} vs {'Line':>5} | {'Diff':>6}")
    print(f"{'-'*4} {'-'*20} {'-'*12} {'-'*8} {'-'*8}   {'-'*8}    {'-'*5}   {'-'*6}")
    for _, row in veteran_games.iterrows():
        result_emoji = "✅" if row['beat_line'] else "❌"
        diff = row['diff_vs_consensus']
        result_str = row.get('result', 'N/A')
        opponent = row.get('opponent', 'N/A')
        print(f"{result_emoji:<4} {row['athlete']:20s} {row['game_date']:<12} {opponent:<8} {result_str:<8} | {row['rushing_yds']:>8.0f} vs {row['consensus_line']:>5.1f} | {diff:>+6.1f}")
    
    return consensus

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function: load and join all data"""
    print("="*80)
    print("NFL QB PLAYOFF RUSHING ANALYSIS")
    print("="*80)
    
    # Step 1: Load QB game logs
    df_qb = load_all_qb_gamelogs()
    
    # Step 2: Load rushing props
    df_props = load_all_rushing_props()
    
    # Step 3: Join datasets (multiple bookmakers per game)
    df_joined = join_qb_games_with_props(df_qb, df_props)
    
    # Step 4: Get consensus lines (one row per player/game)
    df_consensus = get_consensus_lines(df_joined)
    
    # Step 5: Summary
    print(f"\n{'='*80}")
    print("FINAL DATA SUMMARY")
    print(f"{'='*80}")
    print(f"QB game logs: {len(df_qb)} games (all history)")
    print(f"Rushing props: {len(df_props)} prop lines (all bookmakers)")
    print(f"Joined (raw): {len(df_joined)} rows (multiple bookmakers per game)")
    print(f"Consensus: {len(df_consensus)} rows (one per player/game)")
    
    if len(df_consensus) > 0:
        print(f"\nConsensus Performance:")
        beat_pct = (df_consensus['beat_line'].sum() / len(df_consensus) * 100)
        print(f"  QBs that beat consensus: {df_consensus['beat_line'].sum()}/{len(df_consensus)} ({beat_pct:.1f}%)")
        print(f"  Avg difference: {df_consensus['diff_vs_consensus'].mean():.1f} yards")
    
    return df_qb, df_props, df_joined, df_consensus

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    df_qb, df_props, df_joined, df_consensus = main()


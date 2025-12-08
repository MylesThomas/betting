#!/usr/bin/env python3
"""
NBA Props vs Actuals Analysis

Reads historical props data, matches with actual game results,
calculates consensus lines and errors per market.

Goal for --high-level mode:
1. Find overall trends in the markets 

Goal for --liquidity mode:
1. find markets where the books are not as confident/good at pricing
2. find where they are weak
3. bet on those focused parts of the markets

Usage:
    python nba_props_vs_actuals.py --mode high-level    # Market-level summary
    python nba_props_vs_actuals.py --mode player        # (coming soon) Player-level analysis
    python nba_props_vs_actuals.py --mode line-value    # (coming soon) Line value analysis
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import glob
import sys

# Add src to path for utils imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from odds_utils import odds_to_implied_probability, calculate_vig, calculate_vig_attribution
from player_name_utils import normalize_player_name

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
PROPS_DIR = PROJECT_ROOT / 'data' / '01_input' / 'the-odds-api' / 'nba' / 'historical_props' / '2025-26'
GAME_LOGS_FILE = PROJECT_ROOT / 'data' / '01_input' / 'nba_api' / 'season_game_logs' / 'combined_2025_26.csv'
INTERMEDIATE_DIR = PROJECT_ROOT / 'data' / '03_intermediate' / 'nba_props_analysis'
OUTPUT_DIR = None

# Minimum minutes played to include in analysis
# Filters out DNPs, garbage time players, and early injuries
MIN_MINUTES_PLAYED = 10

# =============================================================================
# CATEGORICAL METRICS CONFIG
# =============================================================================

# Category 1: Liquidity (number of books offering the prop)
LIQUIDITY_THIN_MAX = 2      # 1-2 books = "thin"
LIQUIDITY_MED_MAX = 4       # 3-4 books = "medium"
# 5+ books = "liquid"

# Category 2: Vig Structure (over_vig vs under_vig)
VIG_ASYMMETRY_THRESHOLD = 0.01  # 1% difference to be considered asymmetric
# symmetric: |over_vig - under_vig| < threshold
# over_heavy: over_vig > under_vig + threshold
# under_heavy: under_vig > over_vig + threshold

# Category 3: Line Quartile (within-market percentile)
# Q1 = 0-25%, Q2 = 25-50%, Q3 = 50-75%, Q4 = 75-100%
# Computed dynamically per market

# Category 4: Consensus Line (binary)
# consensus = line with lowest total_vig for this game/player/market
# alt = all other lines

# Category 5: Market Structure (replaces simple liquidity)
# Based on n_books at THIS line + line dispersion + odds dispersion
MARKET_STRUCTURE_ODDS_DISP_THRESHOLD = 0.03  # 3% implied range = "odds_disagree"
MARKET_STRUCTURE_LINE_SPREAD_THRESHOLD = 3   # >3 points = "line_disagree"

# =============================================================================
# COMBO ANALYSIS CONFIG (for granular edge discovery)
# =============================================================================

# Minimum sample size for combo analysis
COMBO_MIN_N = 15

# Category columns available for combo analysis
CATEGORY_COLUMNS = {
    'market': 'market',
    'structure': 'cat_market_structure',
    'vig': 'cat_vig_structure', 
    'position': 'cat_line_position',
    'quartile': 'cat_line_quartile',
    'liquidity': 'cat_liquidity',
}

# Combos to analyze (list of tuples of category names)
# Comment out combos that don't show edge (but keep for documentation)
ANALYSIS_COMBOS = [
    # === 1-WAY (baseline) ===
    ('structure',),
    ('vig',),
    
    # === 2-WAY COMBOS ===
    ('structure', 'vig'),           # GOOD: odds_disagree+under_heavy shows edge
    ('vig', 'position'),            # GOOD: only_line+skewed_vig shows edge
    ('market', 'structure'),        # CHECK: per-market structure patterns
    # ('market', 'vig'),            # WEAK: no clear pattern
    # ('structure', 'position'),    # WEAK: position alone not predictive
    
    # === 3-WAY COMBOS ===
    ('market', 'structure', 'vig'),     # GOOD: granular market patterns
    ('structure', 'vig', 'position'),   # GOOD: only_line+vig combos
    # ('market', 'vig', 'position'),    # TOO SPARSE
    
    # === 4-WAY COMBOS (most granular) ===
    ('market', 'structure', 'vig', 'position'),  # DETAILED: small N but high edge
    # ('market', 'structure', 'vig', 'quartile'), # TOO SPARSE
]

# How many top segments to show per combo level
COMBO_TOP_N = 15

# =============================================================================

# Market to actual stat column mapping
MARKET_TO_STAT = {
    'player_points': 'pts',
    'player_rebounds': 'reb',
    'player_assists': 'ast',
    'player_threes': 'threes_made',
    'player_blocks': 'blk',
    'player_steals': 'stl',
    'player_points_rebounds_assists': 'pts_reb_ast',  # computed
}

# Binary markets (no line, just yes/no)
BINARY_MARKETS = ['player_double_double', 'player_triple_double']

# Available analysis modes
AVAILABLE_MODES = ['high-level', 'player', 'line-value', 'liquidity', 'game-player-market', 'game-player-market-line']


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Note: odds_to_implied_probability, calculate_vig imported from src/odds_utils.py
# normalize_player_name imported from src/player_name_utils.py

def american_odds_to_implied_prob(odds: float) -> float:
    """Wrapper for odds_to_implied_probability that handles NaN/zero."""
    if pd.isna(odds) or odds == 0:
        return np.nan
    return odds_to_implied_probability(odds)


def normalize_name_for_matching(name: str) -> str:
    """
    Normalize player name for matching (lowercase).
    Uses the imported normalize_player_name but converts to lowercase.
    """
    if pd.isna(name):
        return ''
    normalized = normalize_player_name(name)
    return normalized.lower() if normalized else ''


def calculate_double_double(row: pd.Series) -> int:
    """Check if player got a double-double (2+ categories with 10+)."""
    categories = [row['pts'], row['reb'], row['ast'], row['stl'], row['blk']]
    doubles = sum(1 for c in categories if c >= 10)
    return 1 if doubles >= 2 else 0


def calculate_triple_double(row: pd.Series) -> int:
    """Check if player got a triple-double (3+ categories with 10+)."""
    categories = [row['pts'], row['reb'], row['ast'], row['stl'], row['blk']]
    doubles = sum(1 for c in categories if c >= 10)
    return 1 if doubles >= 3 else 0


# =============================================================================
# CATEGORICAL METRICS FUNCTIONS
# =============================================================================

def calc_category_liquidity(n_books: int) -> str:
    """
    Category 1: Liquidity based on number of books.
    
    Returns: 'thin' (1-2), 'medium' (3-4), or 'liquid' (5+)
    """
    if pd.isna(n_books):
        return 'unknown'
    if n_books <= LIQUIDITY_THIN_MAX:
        return f'thin (1-{LIQUIDITY_THIN_MAX} books)'
    elif n_books <= LIQUIDITY_MED_MAX:
        return f'medium (3-{LIQUIDITY_MED_MAX} books)'
    else:
        return f'liquid (5+ books)'


def calc_category_vig_structure(over_vig: float, under_vig: float) -> str:
    """
    Category 2: Vig structure based on asymmetry.
    
    Returns: 'symmetric', 'over_heavy', or 'under_heavy'
    """
    if pd.isna(over_vig) or pd.isna(under_vig):
        return 'unknown'
    
    diff = over_vig - under_vig
    
    if abs(diff) < VIG_ASYMMETRY_THRESHOLD:
        return f'symmetric (<= {VIG_ASYMMETRY_THRESHOLD*100}% difference)'
    elif diff > 0:
        return f'over_heavy (> {VIG_ASYMMETRY_THRESHOLD*100}% difference)'  # over bettors paying more vig
    else:
        return f'under_heavy (< {VIG_ASYMMETRY_THRESHOLD*100}% difference)'  # under bettors paying more vig


def calc_category_line_quartile(line: float, market: str, market_percentiles: dict) -> str:
    """
    Category 3: Line quartile within the market.
    
    Args:
        line: The line value
        market: The market name
        market_percentiles: Dict with {market: {'q25': x, 'q50': y, 'q75': z}}
    
    Returns: 'Q1' (0-25%), 'Q2' (25-50%), 'Q3' (50-75%), 'Q4' (75-100%)
    """
    if pd.isna(line) or market not in market_percentiles:
        return 'unknown'
    
    pcts = market_percentiles[market]
    
    if line <= pcts['q25']:
        return 'Q1 (0-25%)'  # Bottom 25%
    elif line <= pcts['q50']:
        return 'Q2 (25-50%)'  # 25-50%
    elif line <= pcts['q75']:
        return 'Q3 (50-75%)'  # 50-75%
    else:
        return 'Q4 (75-100%)'  # Top 25%


def calc_market_percentiles(df: pd.DataFrame, line_col: str = 'line', market_col: str = 'market') -> dict:
    """
    Calculate line percentiles for each market.
    
    Returns: {market: {'q25': x, 'q50': y, 'q75': z}}
    """
    percentiles = {}
    
    for market in df[market_col].unique():
        market_lines = df[df[market_col] == market][line_col].dropna()
        if len(market_lines) > 0:
            percentiles[market] = {
                'q25': market_lines.quantile(0.25),
                'q50': market_lines.quantile(0.50),
                'q75': market_lines.quantile(0.75),
            }
    
    return percentiles


def calc_is_consensus_line(df: pd.DataFrame) -> pd.Series:
    """
    Category 4: Binary flag for consensus line.
    
    For each game/player/market, the consensus line is the one with the lowest total_vig.
    All other lines are "alt" lines.
    
    Args:
        df: DataFrame with columns ['date', 'player', 'market', 'total_vig']
    
    Returns: Series of 1 (consensus) or 0 (alt)
    """
    # Find min vig per game/player/market
    min_vig = df.groupby(['date', 'player', 'market'])['total_vig'].transform('min')
    
    # Mark as consensus if this line has the min vig
    # Handle ties by marking all tied lines as consensus
    is_consensus = (df['total_vig'] == min_vig).astype(int)
    
    return is_consensus


def calc_category_market_structure(n_books: int, prop_line_spread: float, line_implied_range: float) -> str:
    """
    Category 5: Market structure based on shoppability and dispersion.
    
    This replaces simple liquidity with a more nuanced view:
    - captive: Only 1 book offers this line (can't shop)
    - odds_disagree: Multiple books, high odds dispersion at same line (>3%)
    - line_disagree: Multiple books, wide line spread (>3 points)
    - consensus: Multiple books, tight line spread (<=1)
    - moderate: Everything else
    
    Args:
        n_books: Number of books offering THIS line
        prop_line_spread: Max line - min line across all books for this prop
        line_implied_range: Range of implied probs at THIS line
    
    Returns: 'captive', 'odds_disagree', 'line_disagree', 'consensus', or 'moderate'
    """
    if pd.isna(n_books) or n_books == 1:
        return 'captive'
    
    # Check odds dispersion first (takes priority)
    if pd.notna(line_implied_range) and line_implied_range > MARKET_STRUCTURE_ODDS_DISP_THRESHOLD:
        return 'odds_disagree'
    
    # Then check line spread
    if pd.notna(prop_line_spread):
        if prop_line_spread > MARKET_STRUCTURE_LINE_SPREAD_THRESHOLD:
            return 'line_disagree'
        elif prop_line_spread <= 1:
            return 'consensus'
    
    return 'moderate'


def add_categorical_columns(df: pd.DataFrame, is_aggregated: bool = False) -> pd.DataFrame:
    """
    Add all categorical columns to a DataFrame.
    
    Args:
        df: DataFrame with the required columns
        is_aggregated: If True, use 'n_books' column; if False, calculate from individual rows
    
    Returns: DataFrame with added category columns
    """
    df = df.copy()
    
    # Category 1: Liquidity
    if 'n_books' in df.columns:
        df['cat_liquidity'] = df['n_books'].apply(calc_category_liquidity)
    elif 'n' in df.columns:
        df['cat_liquidity'] = df['n'].apply(calc_category_liquidity)
    
    # Category 2: Vig Structure
    if 'over_vig' in df.columns and 'under_vig' in df.columns:
        df['cat_vig_structure'] = df.apply(
            lambda row: calc_category_vig_structure(row['over_vig'], row['under_vig']),
            axis=1
        )
    elif 'avg_over_vig' in df.columns and 'avg_under_vig' in df.columns:
        df['cat_vig_structure'] = df.apply(
            lambda row: calc_category_vig_structure(row['avg_over_vig'], row['avg_under_vig']),
            axis=1
        )
    
    # Category 3: Line Quartile
    line_col = 'line' if 'line' in df.columns else 'avg_line'
    if line_col in df.columns and 'market' in df.columns:
        market_pcts = calc_market_percentiles(df, line_col, 'market')
        df['cat_line_quartile'] = df.apply(
            lambda row: calc_category_line_quartile(row[line_col], row['market'], market_pcts),
            axis=1
        )
    
    # Category 4: Consensus Line (only for non-aggregated data)
    if not is_aggregated and all(col in df.columns for col in ['date', 'player', 'market', 'total_vig']):
        df['is_consensus_line'] = calc_is_consensus_line(df)
    
    # Category 5: Market Structure (for game-player-market-line mode)
    if all(col in df.columns for col in ['n_books', 'prop_line_spread', 'line_implied_range']):
        df['cat_market_structure'] = df.apply(
            lambda row: calc_category_market_structure(
                row['n_books'], 
                row['prop_line_spread'], 
                row['line_implied_range']
            ),
            axis=1
        )
    
    return df


# =============================================================================
# DATA LOADING (shared across all modes)
# =============================================================================

def load_all_props() -> pd.DataFrame:
    """Load all props CSV files into a single DataFrame."""
    props_files = glob.glob(str(PROPS_DIR / 'props_*.csv'))
    
    if not props_files:
        raise FileNotFoundError(f"No props files found in {PROPS_DIR}")
    
    print(f"📂 Loading {len(props_files)} props files...")
    
    dfs = []
    for f in sorted(props_files):
        try:
            df = pd.read_csv(f)
            # Extract date from filename
            filename = Path(f).stem
            date_str = filename.split('_')[1]  # props_2025-10-22_all_markets.csv -> 2025-10-22
            df['prop_date'] = date_str
            dfs.append(df)
        except Exception as e:
            print(f"  ⚠️ Error loading {f}: {e}")
    
    props_df = pd.concat(dfs, ignore_index=True)
    print(f"  ✅ Loaded {len(props_df):,} total prop records")
    
    return props_df


def load_game_logs() -> pd.DataFrame:
    """Load game logs, filter by minutes, and add computed columns."""
    print(f"📂 Loading game logs from {GAME_LOGS_FILE}...")
    
    df = pd.read_csv(GAME_LOGS_FILE)
    initial_count = len(df)
    
    # Filter by minimum minutes played BEFORE any other processing
    df = df[df['minutes'] >= MIN_MINUTES_PLAYED].copy()
    filtered_count = initial_count - len(df)
    
    print(f"  📊 Raw records: {initial_count:,}")
    print(f"  🔽 Filtered out {filtered_count:,} records with < {MIN_MINUTES_PLAYED} min played")
    print(f"  ✅ Keeping {len(df):,} records with >= {MIN_MINUTES_PLAYED} min played")
    
    # Add computed columns
    df['pts_reb_ast'] = df['pts'] + df['reb'] + df['ast']
    df['double_double'] = df.apply(calculate_double_double, axis=1)
    df['triple_double'] = df.apply(calculate_triple_double, axis=1)
    
    # Normalize player name for matching
    df['player_normalized'] = df['player'].apply(normalize_name_for_matching)
    
    # Convert date to string format matching props
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    print(f"  📊 Double-doubles: {df['double_double'].sum():,}")
    print(f"  📊 Triple-doubles: {df['triple_double'].sum():,}")
    
    return df


def calculate_consensus_lines(props_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate consensus line for each player/game/market combination.
    Consensus = average line across all bookmakers.
    """
    print("\n📊 Calculating consensus lines...")
    
    # Filter out rows with no line (binary markets)
    numeric_props = props_df[props_df['line'].notna()].copy()
    
    # Group by player, date, market and calculate consensus
    consensus = numeric_props.groupby(['player', 'prop_date', 'market']).agg({
        'line': ['mean', 'std', 'count'],
        'over_odds': 'mean',
        'under_odds': 'mean',
        'game': 'first',
    }).reset_index()
    
    # Flatten column names
    consensus.columns = ['player', 'prop_date', 'market', 
                         'consensus_line', 'line_std', 'num_books',
                         'avg_over_odds', 'avg_under_odds', 'game']
    
    # Calculate implied probabilities from odds
    consensus['implied_over_prob'] = consensus['avg_over_odds'].apply(american_odds_to_implied_prob)
    consensus['implied_under_prob'] = consensus['avg_under_odds'].apply(american_odds_to_implied_prob)
    
    # Add normalized player name
    consensus['player_normalized'] = consensus['player'].apply(normalize_name_for_matching)
    
    print(f"  ✅ Created {len(consensus):,} consensus lines")
    print(f"  📊 Markets: {consensus['market'].nunique()}")
    print(f"  📊 Unique players: {consensus['player'].nunique()}")
    
    return consensus


def match_props_to_actuals(consensus_df: pd.DataFrame, game_logs_df: pd.DataFrame) -> pd.DataFrame:
    """Match props with actual game results."""
    print("\n🔗 Matching props to actuals...")
    
    matched_rows = []
    unmatched_count = 0
    
    for _, prop in consensus_df.iterrows():
        # Find matching game log
        player_norm = prop['player_normalized']
        date = prop['prop_date']
        market = prop['market']
        
        # Match on normalized name and date
        match = game_logs_df[
            (game_logs_df['player_normalized'] == player_norm) & 
            (game_logs_df['date'] == date)
        ]
        
        if len(match) == 0:
            unmatched_count += 1
            continue
        
        game_log = match.iloc[0]
        
        # Get actual stat value based on market
        if market in MARKET_TO_STAT:
            stat_col = MARKET_TO_STAT[market]
            actual = game_log[stat_col]
        elif market == 'player_double_double':
            actual = game_log['double_double']
        elif market == 'player_triple_double':
            actual = game_log['triple_double']
        else:
            continue
        
        # Calculate error metrics
        # Error from BOOK's perspective: negative = line too low, positive = line too high
        consensus_line = prop['consensus_line']
        
        if pd.notna(consensus_line) and consensus_line != 0:
            error = consensus_line - actual  # Book's error: line - actual
            abs_error = abs(error)
            error_pct = error / consensus_line  # Relative to line size
            abs_error_pct = abs(error_pct)
            hit_over = 1 if actual > consensus_line else 0
            hit_under = 1 if actual < consensus_line else 0
            push = 1 if actual == consensus_line else 0
        else:
            # Binary market (double/triple double) or zero line
            error = None
            abs_error = None
            error_pct = None
            abs_error_pct = None
            hit_over = actual  # 1 if they got it, 0 if not
            hit_under = 1 - actual
            push = 0
        
        matched_rows.append({
            'player': prop['player'],
            'player_normalized': player_norm,
            'date': date,
            'game': prop['game'],
            'market': market,
            'consensus_line': consensus_line,
            'line_std': prop['line_std'],
            'num_books': prop['num_books'],
            'avg_over_odds': prop['avg_over_odds'],
            'avg_under_odds': prop['avg_under_odds'],
            'implied_over_prob': prop['implied_over_prob'],
            'implied_under_prob': prop['implied_under_prob'],
            'actual': actual,
            'error': error,
            'abs_error': abs_error,
            'error_pct': error_pct,
            'abs_error_pct': abs_error_pct,
            'hit_over': hit_over,
            'hit_under': hit_under,
            'push': push,
            # Additional context from game log
            'minutes': game_log['minutes'],
            'result': game_log['result'],
            'opponent': game_log['opponent'],
            'home_away': game_log['home_away'],
        })
    
    matched_df = pd.DataFrame(matched_rows)
    
    match_rate = len(matched_df) / len(consensus_df) * 100 if len(consensus_df) > 0 else 0
    print(f"  ✅ Matched {len(matched_df):,} / {len(consensus_df):,} props ({match_rate:.1f}%)")
    print(f"  ⚠️ Unmatched: {unmatched_count:,}")
    
    return matched_df


def load_consensus_lines_data() -> pd.DataFrame:
    """
    Load and process all data. Returns matched DataFrame.
    This is the shared data pipeline for all analysis modes.
    """
    print("=" * 80)
    print("🏀 NBA PROPS VS ACTUALS - DATA LOADING")
    print("=" * 80)
    
    # Load raw data
    props_df = load_all_props()
    game_logs_df = load_game_logs()
    
    # Calculate consensus lines
    consensus_df = calculate_consensus_lines(props_df)
    
    # Match props to actuals
    matched_df = match_props_to_actuals(consensus_df, game_logs_df)
    
    if len(matched_df) == 0:
        raise ValueError("No matches found! Check player name matching.")
    
    return matched_df


# =============================================================================
# MODE: HIGH-LEVEL (Market-level summary)
# =============================================================================

def calculate_market_summary(matched_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics per market."""
    print("\n📈 Calculating market summaries...")
    
    summaries = []
    
    for market in matched_df['market'].unique():
        market_data = matched_df[matched_df['market'] == market]
        
        is_binary = market in BINARY_MARKETS
        
        summary = {
            'market': market,
            'n_props': len(market_data),
            'n_unique_players': market_data['player'].nunique(),
        }
        
        if not is_binary:
            # Numeric market stats
            hit_over_rate = market_data['hit_over'].mean()
            hit_under_rate = market_data['hit_under'].mean()
            avg_implied_over = market_data['implied_over_prob'].mean()
            avg_implied_under = market_data['implied_under_prob'].mean()
            
            summary.update({
                'avg_line': market_data['consensus_line'].mean(),
                'avg_actual': market_data['actual'].mean(),
                'avg_error': market_data['error'].mean(),
                'std_error': market_data['error'].std(),
                'avg_abs_error': market_data['abs_error'].mean(),
                'median_error': market_data['error'].median(),
                'hit_over_rate': hit_over_rate,
                'hit_under_rate': hit_under_rate,
                'push_rate': market_data['push'].mean(),
                'avg_implied_over': avg_implied_over,
                'avg_implied_under': avg_implied_under,
                'edge_over': hit_over_rate - avg_implied_over,
                'edge_under': hit_under_rate - avg_implied_under,
            })
        else:
            # Binary market stats
            summary.update({
                'avg_line': None,
                'avg_actual': market_data['actual'].mean(),  # This is hit rate
                'avg_error': None,
                'std_error': None,
                'avg_abs_error': None,
                'median_error': None,
                'hit_over_rate': market_data['hit_over'].mean(),  # = hit rate for binary
                'hit_under_rate': market_data['hit_under'].mean(),
                'push_rate': None,
                'avg_implied_over': None,
                'avg_implied_under': None,
                'edge_over': None,
                'edge_under': None,
            })
        
        summaries.append(summary)
    
    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values('n_props', ascending=False)
    
    return summary_df


def print_high_level_summary(summary_df: pd.DataFrame):
    """Print formatted high-level summary."""
    print("\n" + "=" * 130)
    print("📊 HIGH-LEVEL MARKET SUMMARY")
    print("=" * 130)
    
    # Numeric markets
    numeric_markets = summary_df[summary_df['avg_error'].notna()]
    
    print("\n📈 NUMERIC MARKETS (Over/Under Lines):")
    print("-" * 130)
    print(f"{'Market':<24} {'N':>6} {'Pred':>7} {'Actual':>7} {'Error':>7} {'AbsErr':>7} {'Over%':>7} {'ImpOvr':>7} {'EdgeOvr':>8} {'Under%':>7} {'ImpUnd':>7} {'EdgeUnd':>8}")
    print("-" * 130)
    
    for _, row in numeric_markets.iterrows():
        market_short = row['market'].replace('player_', '')
        print(f"{market_short:<24} {row['n_props']:>6,} {row['avg_line']:>7.2f} {row['avg_actual']:>7.2f} {row['avg_error']:>+7.2f} {row['avg_abs_error']:>7.2f} {row['hit_over_rate']:>6.1%} {row['avg_implied_over']:>6.1%} {row['edge_over']:>+7.1%} {row['hit_under_rate']:>6.1%} {row['avg_implied_under']:>6.1%} {row['edge_under']:>+7.1%}")
    
    print("-" * 130)
    
    # Binary markets
    binary_markets = summary_df[summary_df['avg_error'].isna()]
    
    if len(binary_markets) > 0:
        print("\n🎯 BINARY MARKETS (Yes/No - Hit Rate):")
        print("-" * 100)
        print(f"{'Market':<32} {'N':>6} {'Hit Rate':>10}")
        print("-" * 100)
        
        for _, row in binary_markets.iterrows():
            market_short = row['market'].replace('player_', '')
            print(f"{market_short:<32} {row['n_props']:>6,} {row['hit_over_rate']:>9.1%}")
    
    print("=" * 130)


def run_high_level_analysis(matched_df: pd.DataFrame):
    """Run high-level market summary analysis."""
    print("\n" + "=" * 80)
    print("🎯 MODE: HIGH-LEVEL ANALYSIS")
    print("=" * 80)
    
    # Calculate market summary
    summary_df = calculate_market_summary(matched_df)
    
    # Print summary
    print_high_level_summary(summary_df)
    
    # Save outputs
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    
    matched_file = INTERMEDIATE_DIR / 'props_vs_actuals_matched.csv'
    summary_file = INTERMEDIATE_DIR / 'market_summary.csv'
    
    matched_df.to_csv(matched_file, index=False)
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\n💾 Saved:")
    print(f"   - Full matched data: {matched_file}")
    print(f"   - Market summary: {summary_file}")
    
    return summary_df


# =============================================================================
# MODE: PLAYER (Player-level analysis) - Coming soon
# =============================================================================

def run_player_analysis(matched_df: pd.DataFrame):
    """Run player-level analysis."""
    print("\n" + "=" * 80)
    print("🎯 MODE: PLAYER ANALYSIS")
    print("=" * 80)
    print("\n⚠️ Player-level analysis not yet implemented.")
    print("   This will show per-player edge, consistency, etc.")
    # TODO: Implement player-level analysis


# =============================================================================
# MODE: LINE-VALUE (Line value analysis) - Coming soon
# =============================================================================

def run_line_value_analysis(matched_df: pd.DataFrame):
    """Run line value analysis."""
    print("\n" + "=" * 80)
    print("🎯 MODE: LINE-VALUE ANALYSIS")
    print("=" * 80)
    print("\n⚠️ Line-value analysis not yet implemented.")
    print("   This will show edge by line range (e.g., points < 10 vs > 20).")
    # TODO: Implement line-value analysis


# =============================================================================
# MODE: LIQUIDITY (Per-prop liquidity metrics)
# =============================================================================

def calculate_liquidity_metrics(props_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate liquidity metrics at the player/game/market level.
    
    Metrics:
    - num_books: number of bookmakers offering this prop
    - num_distinct_lines: number of unique line values
    - line_min, line_max, line_mean, line_median, line_std, line_spread
    - implied_over_prob_*: min/max/mean/median/std of implied over probabilities
    """
    print("\n📊 Calculating per-prop liquidity metrics...")
    
    # Filter to numeric markets only
    numeric_props = props_df[props_df['line'].notna()].copy()
    
    # Calculate implied probabilities for each row
    numeric_props['implied_over_prob'] = numeric_props['over_odds'].apply(american_odds_to_implied_prob)
    numeric_props['implied_under_prob'] = numeric_props['under_odds'].apply(american_odds_to_implied_prob)
    
    # Group by player/date/market and calculate metrics
    liquidity = numeric_props.groupby(['player', 'prop_date', 'market']).agg({
        # Line metrics
        'line': ['count', 'nunique', 'min', 'max', 'mean', 'median', 'std'],
        # Implied probability metrics (over)
        'implied_over_prob': ['min', 'max', 'mean', 'median', 'std'],
        # Implied probability metrics (under)
        'implied_under_prob': ['min', 'max', 'mean', 'median', 'std'],
        # Raw odds metrics
        'over_odds': ['min', 'max', 'mean', 'std'],
        'under_odds': ['min', 'max', 'mean', 'std'],
        # Bookmaker count
        'bookmaker': 'nunique',
        # Game info
        'game': 'first',
    }).reset_index()
    
    # Flatten column names
    liquidity.columns = [
        'player', 'prop_date', 'market',
        'num_lines', 'num_distinct_lines', 'line_min', 'line_max', 'line_mean', 'line_median', 'line_std',
        'imp_over_min', 'imp_over_max', 'imp_over_mean', 'imp_over_median', 'imp_over_std',
        'imp_under_min', 'imp_under_max', 'imp_under_mean', 'imp_under_median', 'imp_under_std',
        'over_odds_min', 'over_odds_max', 'over_odds_mean', 'over_odds_std',
        'under_odds_min', 'under_odds_max', 'under_odds_mean', 'under_odds_std',
        'num_books', 'game',
    ]
    
    # Add derived metrics
    liquidity['line_spread'] = liquidity['line_max'] - liquidity['line_min']
    liquidity['line_std'] = liquidity['line_std'].fillna(0)
    liquidity['imp_over_std'] = liquidity['imp_over_std'].fillna(0)
    liquidity['imp_under_std'] = liquidity['imp_under_std'].fillna(0)
    liquidity['imp_over_spread'] = liquidity['imp_over_max'] - liquidity['imp_over_min']
    
    # Add normalized player name for matching
    liquidity['player_normalized'] = liquidity['player'].apply(normalize_name_for_matching)
    
    print(f"  ✅ Calculated liquidity for {len(liquidity):,} props")
    
    return liquidity


def join_liquidity_with_actuals(liquidity_df: pd.DataFrame, game_logs_df: pd.DataFrame) -> pd.DataFrame:
    """Join liquidity metrics with actual game outcomes."""
    print("\n🔗 Joining liquidity metrics with actuals...")
    
    matched_rows = []
    unmatched_count = 0
    
    for _, prop in liquidity_df.iterrows():
        player_norm = prop['player_normalized']
        date = prop['prop_date']
        market = prop['market']
        
        # Match on normalized name and date
        match = game_logs_df[
            (game_logs_df['player_normalized'] == player_norm) & 
            (game_logs_df['date'] == date)
        ]
        
        if len(match) == 0:
            unmatched_count += 1
            continue
        
        game_log = match.iloc[0]
        
        # Get actual stat value
        if market in MARKET_TO_STAT:
            stat_col = MARKET_TO_STAT[market]
            actual = game_log[stat_col]
        else:
            continue
        
        # Create row with all liquidity metrics + actual
        # Error from BOOK's perspective: negative = line too low, positive = line too high
        row = prop.to_dict()
        row['actual'] = actual
        line_mean = prop['line_mean']
        row['error'] = line_mean - actual  # Book's error
        row['abs_error'] = abs(row['error'])
        row['error_pct'] = row['error'] / line_mean if line_mean != 0 else None
        row['abs_error_pct'] = abs(row['error_pct']) if row['error_pct'] is not None else None
        row['hit_over'] = 1 if actual > line_mean else 0
        row['hit_under'] = 1 if actual < line_mean else 0
        row['minutes'] = game_log['minutes']
        
        matched_rows.append(row)
    
    matched_df = pd.DataFrame(matched_rows)
    
    match_rate = len(matched_df) / len(liquidity_df) * 100 if len(liquidity_df) > 0 else 0
    print(f"  ✅ Matched {len(matched_df):,} / {len(liquidity_df):,} props ({match_rate:.1f}%)")
    
    return matched_df


def calculate_outcome_distribution_by_market(matched_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate outcome distribution statistics by market."""
    from scipy import stats as scipy_stats
    
    print("\n📈 Calculating outcome distributions by market...")
    
    summaries = []
    
    for market in matched_df['market'].unique():
        mkt_data = matched_df[matched_df['market'] == market]
        actuals = mkt_data['actual'].dropna()
        
        if len(actuals) < 10:
            continue
        
        summary = {
            'market': market.replace('player_', ''),
            'n_props': len(mkt_data),
            # Outcome distribution
            'actual_min': actuals.min(),
            'actual_max': actuals.max(),
            'actual_mean': actuals.mean(),
            'actual_median': actuals.median(),
            'actual_std': actuals.std(),
            'actual_skew': scipy_stats.skew(actuals),
            'actual_kurtosis': scipy_stats.kurtosis(actuals),
            # Liquidity metrics (averages)
            'avg_num_books': mkt_data['num_books'].mean(),
            'avg_num_distinct_lines': mkt_data['num_distinct_lines'].mean(),
            'avg_line_spread': mkt_data['line_spread'].mean(),
            'avg_imp_over_spread': mkt_data['imp_over_spread'].mean(),
            # Error metrics
            'avg_error': mkt_data['error'].mean(),
            'avg_abs_error': mkt_data['abs_error'].mean(),
            'hit_over_rate': mkt_data['hit_over'].mean(),
        }
        summaries.append(summary)
    
    return pd.DataFrame(summaries).sort_values('n_props', ascending=False)


def print_liquidity_summary(market_summary: pd.DataFrame, prop_liquidity: pd.DataFrame):
    """Print formatted liquidity summary."""
    print("\n" + "=" * 140)
    print("📊 LIQUIDITY ANALYSIS - MARKET SUMMARY")
    print("=" * 140)
    
    # Market-level summary
    print("\n📈 MARKET LIQUIDITY & OUTCOME DISTRIBUTIONS:")
    print("-" * 140)
    print(f"{'Market':<24} {'N':>6} {'AvgBooks':>9} {'DistLines':>10} {'LineSprd':>9} {'ImpSprd':>8} │ {'ActMean':>8} {'ActStd':>8} {'Skew':>7} {'Kurt':>7} │ {'Error':>7} {'Over%':>7}")
    print("-" * 140)
    
    for _, row in market_summary.iterrows():
        print(f"{row['market']:<24} {row['n_props']:>6,} {row['avg_num_books']:>9.1f} {row['avg_num_distinct_lines']:>10.2f} {row['avg_line_spread']:>9.2f} {row['avg_imp_over_spread']:>7.1%} │ {row['actual_mean']:>8.2f} {row['actual_std']:>8.2f} {row['actual_skew']:>7.2f} {row['actual_kurtosis']:>7.2f} │ {row['avg_error']:>+7.2f} {row['hit_over_rate']:>6.1%}")
    
    print("-" * 140)
    
    # Distribution of liquidity tiers
    print("\n📊 LIQUIDITY TIER DISTRIBUTION (by num_books):")
    print("-" * 80)
    
    for market in prop_liquidity['market'].unique():
        mkt_data = prop_liquidity[prop_liquidity['market'] == market]
        total = len(mkt_data)
        
        tier_1_2 = (mkt_data['num_books'] <= 2).sum() / total * 100
        tier_3_4 = ((mkt_data['num_books'] >= 3) & (mkt_data['num_books'] <= 4)).sum() / total * 100
        tier_5_plus = (mkt_data['num_books'] >= 5).sum() / total * 100
        
        mkt_short = market.replace('player_', '')
        print(f"{mkt_short:<28} 1-2 books: {tier_1_2:>5.1f}%  |  3-4 books: {tier_3_4:>5.1f}%  |  5+ books: {tier_5_plus:>5.1f}%")
    
    print("=" * 140)


def calculate_error_by_book(props_df: pd.DataFrame, game_logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate error for each bookmaker's lines individually.
    This shows which books are sharpest vs softest.
    """
    print("\n📊 Calculating error by bookmaker...")
    
    # Filter to numeric markets only
    numeric_props = props_df[props_df['line'].notna()].copy()
    
    # Add normalized player name
    numeric_props['player_normalized'] = numeric_props['player'].apply(normalize_name_for_matching)
    
    # Prepare game logs for joining
    game_logs_df = game_logs_df.copy()
    
    matched_rows = []
    
    for _, prop in numeric_props.iterrows():
        player_norm = prop['player_normalized']
        date = prop['prop_date']
        market = prop['market']
        
        # Match on normalized name and date
        match = game_logs_df[
            (game_logs_df['player_normalized'] == player_norm) & 
            (game_logs_df['date'] == date)
        ]
        
        if len(match) == 0:
            continue
        
        game_log = match.iloc[0]
        
        # Get actual stat value
        if market in MARKET_TO_STAT:
            stat_col = MARKET_TO_STAT[market]
            actual = game_log[stat_col]
        else:
            continue
        
        # Error from BOOK's perspective: negative = line too low, positive = line too high
        line = prop['line']
        error = line - actual
        error_pct = error / line if line != 0 else None
        
        matched_rows.append({
            'player': prop['player'],
            'date': date,
            'market': market,
            'bookmaker': prop['bookmaker'],
            'line': line,
            'actual': actual,
            'error': error,
            'abs_error': abs(error),
            'error_pct': error_pct,
            'abs_error_pct': abs(error_pct) if error_pct is not None else None,
            'hit_over': 1 if actual > line else 0,
            'hit_under': 1 if actual < prop['line'] else 0,
        })
    
    book_df = pd.DataFrame(matched_rows)
    print(f"  ✅ Matched {len(book_df):,} individual book lines")
    
    return book_df


def calculate_book_summary(book_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate error metrics by bookmaker."""
    
    book_summary = book_df.groupby('bookmaker').agg({
        'error': ['mean', 'std'],
        'abs_error': 'mean',
        'hit_over': 'mean',
        'line': 'count',
    }).reset_index()
    
    book_summary.columns = ['bookmaker', 'avg_error', 'std_error', 'avg_abs_error', 'hit_over_rate', 'n_lines']
    book_summary = book_summary.sort_values('avg_abs_error')
    
    return book_summary


def calculate_book_by_market_summary(book_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate error metrics by bookmaker AND market."""
    
    summary = book_df.groupby(['bookmaker', 'market']).agg({
        'error': 'mean',
        'abs_error': 'mean',
        'hit_over': 'mean',
        'line': 'count',
    }).reset_index()
    
    summary.columns = ['bookmaker', 'market', 'avg_error', 'avg_abs_error', 'hit_over_rate', 'n_lines']
    
    return summary


def print_book_summary(book_summary: pd.DataFrame, book_by_market: pd.DataFrame):
    """Print bookmaker accuracy summary."""
    print("\n" + "=" * 100)
    print("📚 BOOKMAKER ACCURACY ANALYSIS")
    print("=" * 100)
    
    print("\n📈 OVERALL ACCURACY BY BOOK (sorted by abs error - lower = sharper):")
    print("-" * 100)
    print(f"{'Bookmaker':<20} {'N Lines':>10} {'AvgError':>10} {'AbsError':>10} {'StdError':>10} {'Over%':>8}")
    print("-" * 100)
    
    for _, row in book_summary.iterrows():
        print(f"{row['bookmaker']:<20} {row['n_lines']:>10,} {row['avg_error']:>+10.2f} {row['avg_abs_error']:>10.2f} {row['std_error']:>10.2f} {row['hit_over_rate']:>7.1%}")
    
    print("-" * 100)
    
    # Show best/worst book by market
    print("\n📊 SHARPEST BOOK BY MARKET (lowest abs error):")
    print("-" * 80)
    
    for market in book_by_market['market'].unique():
        mkt_data = book_by_market[book_by_market['market'] == market]
        # Only consider books with at least 50 lines in this market
        mkt_data = mkt_data[mkt_data['n_lines'] >= 50]
        if len(mkt_data) == 0:
            continue
        
        best = mkt_data.loc[mkt_data['avg_abs_error'].idxmin()]
        worst = mkt_data.loc[mkt_data['avg_abs_error'].idxmax()]
        
        mkt_short = market.replace('player_', '')
        print(f"{mkt_short:<24} Best: {best['bookmaker']:<12} (AbsErr: {best['avg_abs_error']:.2f})  |  Worst: {worst['bookmaker']:<12} (AbsErr: {worst['avg_abs_error']:.2f})")
    
    print("=" * 100)


def join_raw_props_with_actuals(props_df: pd.DataFrame, game_logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join INDIVIDUAL book lines (not aggregated) with actual game results.
    Each row = one bookmaker's line for one player/game/market.
    """
    print("\n🔗 Joining individual book lines with actuals (NO aggregation)...")
    
    # Filter to numeric markets only
    numeric_props = props_df[props_df['line'].notna()].copy()
    
    # Add normalized player name
    numeric_props['player_normalized'] = numeric_props['player'].apply(normalize_name_for_matching)
    
    # Calculate implied probability for each line
    numeric_props['implied_over_prob'] = numeric_props['over_odds'].apply(american_odds_to_implied_prob)
    numeric_props['implied_under_prob'] = numeric_props['under_odds'].apply(american_odds_to_implied_prob)
    
    matched_rows = []
    
    for _, prop in numeric_props.iterrows():
        player_norm = prop['player_normalized']
        date = prop['prop_date']
        market = prop['market']
        
        # Match on normalized name and date
        match = game_logs_df[
            (game_logs_df['player_normalized'] == player_norm) & 
            (game_logs_df['date'] == date)
        ]
        
        if len(match) == 0:
            continue
        
        game_log = match.iloc[0]
        
        # Get actual stat value
        if market in MARKET_TO_STAT:
            stat_col = MARKET_TO_STAT[market]
            actual = game_log[stat_col]
        else:
            continue
        
        # Error from BOOK's perspective: negative = line too low, positive = line too high
        line = prop['line']
        error = line - actual
        error_pct = error / line if line != 0 else None
        
        matched_rows.append({
            'player': prop['player'],
            'player_normalized': player_norm,
            'date': date,
            'game': prop['game'],
            'matchup': game_log['matchup'],  # e.g., "OKC @ GSW"
            'opponent': game_log['opponent'],
            'home_away': game_log['home_away'],
            'market': market,
            'bookmaker': prop['bookmaker'],
            'line': line,
            'over_odds': prop['over_odds'],
            'under_odds': prop['under_odds'],
            'implied_over_prob': prop['implied_over_prob'],
            'implied_under_prob': prop['implied_under_prob'],
            'actual': actual,
            'error': error,
            'abs_error': abs(error),
            'error_pct': error_pct,
            'abs_error_pct': abs(error_pct) if error_pct is not None else None,
            'hit_over': 1 if actual > line else 0,
            'hit_under': 1 if actual < line else 0,
            'push': 1 if actual == line else 0,
            'minutes': game_log['minutes'],
        })
    
    matched_df = pd.DataFrame(matched_rows)
    
    total_props = len(numeric_props)
    match_rate = len(matched_df) / total_props * 100 if total_props > 0 else 0
    print(f"  ✅ Matched {len(matched_df):,} / {total_props:,} individual book lines ({match_rate:.1f}%)")
    
    return matched_df


def calculate_liquidity_per_prop(matched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add liquidity metrics to each row by calculating stats across books for same player/date/market.
    This adds context about how liquid each prop is without losing the individual book rows.
    """
    print("\n📊 Adding liquidity context to each row...")
    
    # Calculate liquidity stats per player/date/market
    liquidity_stats = matched_df.groupby(['player', 'date', 'market']).agg({
        'bookmaker': 'nunique',
        'line': ['nunique', 'min', 'max', 'std'],
        'implied_over_prob': ['min', 'max', 'std'],
    }).reset_index()
    
    liquidity_stats.columns = ['player', 'date', 'market', 
                               'num_books', 'num_distinct_lines', 
                               'line_min', 'line_max', 'line_std',
                               'imp_over_min', 'imp_over_max', 'imp_over_std']
    
    liquidity_stats['line_spread'] = liquidity_stats['line_max'] - liquidity_stats['line_min']
    liquidity_stats['imp_over_spread'] = liquidity_stats['imp_over_max'] - liquidity_stats['imp_over_min']
    liquidity_stats['line_std'] = liquidity_stats['line_std'].fillna(0)
    
    # Merge back to individual rows
    matched_df = matched_df.merge(
        liquidity_stats[['player', 'date', 'market', 'num_books', 'num_distinct_lines', 
                         'line_spread', 'line_std', 'imp_over_spread']],
        on=['player', 'date', 'market'],
        how='left'
    )
    
    print(f"  ✅ Added liquidity context to {len(matched_df):,} rows")
    
    return matched_df


def calculate_market_summary_from_raw(matched_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate market summary from individual book lines."""
    from scipy import stats as scipy_stats
    
    summaries = []
    
    for market in matched_df['market'].unique():
        mkt_data = matched_df[matched_df['market'] == market]
        actuals = mkt_data['actual'].dropna()
        
        if len(actuals) < 10:
            continue
        
        summary = {
            'market': market.replace('player_', ''),
            'n_book_lines': len(mkt_data),
            'n_unique_props': mkt_data.groupby(['player', 'date']).ngroups,
            # Liquidity
            'avg_num_books': mkt_data.groupby(['player', 'date'])['bookmaker'].nunique().mean(),
            'avg_line_spread': mkt_data.groupby(['player', 'date'])['line_spread'].first().mean(),
            # Outcome distribution
            'actual_mean': actuals.mean(),
            'actual_std': actuals.std(),
            'actual_skew': scipy_stats.skew(actuals),
            'actual_kurtosis': scipy_stats.kurtosis(actuals),
            # Error (per book line)
            'avg_error': mkt_data['error'].mean(),
            'avg_abs_error': mkt_data['abs_error'].mean(),
            'hit_over_rate': mkt_data['hit_over'].mean(),
        }
        summaries.append(summary)
    
    return pd.DataFrame(summaries).sort_values('n_book_lines', ascending=False)


def run_liquidity_analysis(props_df: pd.DataFrame, game_logs_df: pd.DataFrame):
    """Run liquidity analysis mode - NO aggregation, keeps individual book lines."""
    print("\n" + "=" * 80)
    print("🎯 MODE: LIQUIDITY ANALYSIS (Individual Book Lines)")
    print("=" * 80)
    
    # Join individual book lines with actuals (NO aggregation)
    matched_df = join_raw_props_with_actuals(props_df, game_logs_df)
    
    if len(matched_df) == 0:
        print("❌ No matches found!")
        return
    
    # Add liquidity context to each row
    matched_df = calculate_liquidity_per_prop(matched_df)
    
    # Calculate summaries
    market_summary = calculate_market_summary_from_raw(matched_df)
    book_summary = calculate_book_summary(matched_df)
    book_by_market = calculate_book_by_market_summary(matched_df)
    
    # Print summaries
    print("\n" + "=" * 130)
    print("📊 MARKET SUMMARY (from individual book lines)")
    print("=" * 130)
    print(f"\n{'Market':<24} {'BookLines':>10} {'UniqProps':>10} {'AvgBooks':>9} {'LineSprd':>9} │ {'ActMean':>8} {'ActStd':>8} │ {'Error':>7} {'AbsErr':>8} {'Over%':>7}")
    print("-" * 130)
    
    for _, row in market_summary.iterrows():
        print(f"{row['market']:<24} {row['n_book_lines']:>10,} {row['n_unique_props']:>10,} {row['avg_num_books']:>9.1f} {row['avg_line_spread']:>9.2f} │ {row['actual_mean']:>8.2f} {row['actual_std']:>8.2f} │ {row['avg_error']:>+7.2f} {row['avg_abs_error']:>8.2f} {row['hit_over_rate']:>6.1%}")
    
    print("-" * 130)
    
    # Print book summary
    print_book_summary(book_summary, book_by_market)
    
    # Save outputs
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    
    raw_file = INTERMEDIATE_DIR / 'liquidity_individual_book_lines.csv'
    market_file = INTERMEDIATE_DIR / 'liquidity_market_summary.csv'
    book_summary_file = INTERMEDIATE_DIR / 'book_accuracy_summary.csv'
    
    matched_df.to_csv(raw_file, index=False)
    market_summary.to_csv(market_file, index=False)
    book_summary.to_csv(book_summary_file, index=False)
    
    print(f"\n💾 Saved:")
    print(f"   - Individual book lines: {raw_file} ({len(matched_df):,} rows)")
    print(f"   - Market summary: {market_file}")
    print(f"   - Book accuracy: {book_summary_file}")
    
    return matched_df, market_summary, book_summary


# =============================================================================
# MODE: GAME-PLAYER-MARKET (Group by game + player + market, aggregate across books)
# =============================================================================

def run_game_player_market_analysis(props_df: pd.DataFrame, game_logs_df: pd.DataFrame):
    """
    Analyze at game + player + market level with list columns.
    
    Each row = one game + one player + one market (e.g., 2025-11-20 + LeBron + points)
    Aggregates across bookmakers for the same game.
    
    Includes: lst_lines, lst_bookmakers, lst_vig, and 5 vig attribution metrics.
    """
    print("\n" + "=" * 80)
    print("🎯 MODE: GAME-PLAYER-MARKET ANALYSIS")
    print("=" * 80)
    
    # First get individual book lines matched with actuals
    matched_df = join_raw_props_with_actuals(props_df, game_logs_df)
    
    if len(matched_df) == 0:
        print("❌ No matches found!")
        return
    
    # Add vig attribution columns (5 metrics per line)
    print("\n📊 Calculating vig attribution metrics...")
    vig_attrs = matched_df.apply(
        lambda row: calculate_vig_attribution(row['implied_over_prob'], row['implied_under_prob']),
        axis=1
    )
    matched_df['total_vig'] = vig_attrs.apply(lambda x: x['total_vig'])
    matched_df['over_vig'] = vig_attrs.apply(lambda x: x['over_vig'])
    matched_df['under_vig'] = vig_attrs.apply(lambda x: x['under_vig'])
    matched_df['fair_over'] = vig_attrs.apply(lambda x: x['fair_over'])
    matched_df['fair_under'] = vig_attrs.apply(lambda x: x['fair_under'])
    
    print("\n📊 Aggregating to game + player + market level...")
    
    # Group by date + player + market (each game is separate)
    game_player_market = matched_df.groupby(['date', 'player', 'market']).agg({
        # Scalar columns (will be NA in aggregated view - from first row for reference)
        'game': 'first',
        'actual': 'first',  # Same for all rows in group
        'minutes': 'first',
        
        # Count columns
        'bookmaker': ['nunique', 'count'],  # num unique books, total lines
        
        # List columns - bookmakers
        'bookmaker': lambda x: list(x),
        
        # List columns - lines
        'line': lambda x: list(x),
        
        # List columns - odds
        'over_odds': lambda x: list(x),
        'under_odds': lambda x: list(x),
        
        # List columns - implied probs
        'implied_over_prob': lambda x: list(x),
        'implied_under_prob': lambda x: list(x),
        
        # List columns - errors
        'error': lambda x: list(x),
        'error_pct': lambda x: list(x),
        'hit_over': lambda x: list(x),
        
        # List columns - vig attribution (5 metrics)
        'total_vig': lambda x: list(x),
        'over_vig': lambda x: list(x),
        'under_vig': lambda x: list(x),
        'fair_over': lambda x: list(x),
        'fair_under': lambda x: list(x),
    }).reset_index()
    
    # Flatten column names
    game_player_market.columns = ['date', 'player', 'market', 'game', 'actual', 'minutes',
                                   'lst_bookmakers', 'lst_lines', 'lst_over_odds', 'lst_under_odds',
                                   'lst_implied_over', 'lst_implied_under',
                                   'lst_errors', 'lst_error_pcts', 'lst_hit_over',
                                   'lst_total_vig', 'lst_over_vig', 'lst_under_vig',
                                   'lst_fair_over', 'lst_fair_under']
    
    # Add distinct lines list
    game_player_market['lst_distinct_lines'] = game_player_market['lst_lines'].apply(lambda x: sorted(set(x)))
    
    # Add aggregate stats
    game_player_market['n'] = game_player_market['lst_lines'].apply(len)
    game_player_market['n_books'] = game_player_market['lst_bookmakers'].apply(lambda x: len(set(x)))
    game_player_market['n_distinct_lines'] = game_player_market['lst_distinct_lines'].apply(len)
    
    # Line stats
    game_player_market['avg_line'] = game_player_market['lst_lines'].apply(np.mean)
    game_player_market['std_line'] = game_player_market['lst_lines'].apply(lambda x: np.std(x) if len(x) > 1 else 0)
    game_player_market['min_line'] = game_player_market['lst_lines'].apply(np.min)
    game_player_market['max_line'] = game_player_market['lst_lines'].apply(np.max)
    game_player_market['line_spread'] = game_player_market['max_line'] - game_player_market['min_line']
    
    # Error stats
    game_player_market['avg_error'] = game_player_market['lst_errors'].apply(np.mean)
    game_player_market['std_error'] = game_player_market['lst_errors'].apply(lambda x: np.std(x) if len(x) > 1 else 0)
    game_player_market['avg_error_pct'] = game_player_market['lst_error_pcts'].apply(lambda x: np.nanmean(x))
    game_player_market['hit_over_rate'] = game_player_market['lst_hit_over'].apply(np.mean)
    
    # Vig attribution stats (averages across books)
    game_player_market['avg_total_vig'] = game_player_market['lst_total_vig'].apply(lambda x: np.nanmean(x))
    game_player_market['avg_over_vig'] = game_player_market['lst_over_vig'].apply(lambda x: np.nanmean(x))
    game_player_market['avg_under_vig'] = game_player_market['lst_under_vig'].apply(lambda x: np.nanmean(x))
    game_player_market['avg_fair_over'] = game_player_market['lst_fair_over'].apply(lambda x: np.nanmean(x))
    game_player_market['avg_fair_under'] = game_player_market['lst_fair_under'].apply(lambda x: np.nanmean(x))
    
    # Scalar columns set to NA (since aggregated)
    game_player_market['line'] = np.nan
    game_player_market['bookmaker'] = np.nan
    game_player_market['over_odds'] = np.nan
    game_player_market['under_odds'] = np.nan
    game_player_market['implied_over_prob'] = np.nan
    game_player_market['implied_under_prob'] = np.nan
    game_player_market['error'] = np.nan
    game_player_market['error_pct'] = np.nan
    game_player_market['abs_error'] = np.nan
    game_player_market['abs_error_pct'] = np.nan
    game_player_market['total_vig'] = np.nan
    game_player_market['over_vig'] = np.nan
    game_player_market['under_vig'] = np.nan
    game_player_market['fair_over'] = np.nan
    game_player_market['fair_under'] = np.nan
    
    # Add categorical columns (using aggregated vig values)
    print("\n📊 Adding categorical columns...")
    game_player_market = add_categorical_columns(game_player_market, is_aggregated=True)
    # Note: is_consensus_line is NA for aggregated data (would need individual lines)
    game_player_market['is_consensus_line'] = np.nan
    
    print(f"  ✅ Created {len(game_player_market):,} game-player-market combinations")
    
    # Print summary
    print("\n" + "=" * 140)
    print("📊 GAME-PLAYER-MARKET SUMMARY (sample of 20 rows)")
    print("=" * 140)
    print(f"{'Date':<12} {'Player':<20} {'Market':<16} {'N':>4} {'AvgLine':>8} {'Actual':>7} {'Error':>7} {'Err%':>7} {'TotVig':>7} {'OvrVig':>7} {'UndVig':>7}")
    print("-" * 140)
    
    sample = game_player_market.sample(min(20, len(game_player_market)))
    for _, row in sample.iterrows():
        mkt = row['market'].replace('player_', '')[:14]
        player = row['player'][:18]
        err_pct = row['avg_error_pct'] if pd.notna(row['avg_error_pct']) else 0
        print(f"{row['date']:<12} {player:<20} {mkt:<16} {row['n']:>4} {row['avg_line']:>8.1f} {row['actual']:>7.1f} {row['avg_error']:>+7.2f} {err_pct:>+6.0%} {row['avg_total_vig']:>6.1%} {row['avg_over_vig']:>6.1%} {row['avg_under_vig']:>6.1%}")
    
    print("-" * 140)
    print("=" * 140)
    
    # Save
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    output_file = INTERMEDIATE_DIR / 'game_player_market_analysis.csv'
    
    # Convert lists to strings for CSV
    csv_df = game_player_market.copy()
    for col in csv_df.columns:
        if csv_df[col].dtype == 'object' and len(csv_df) > 0:
            first_val = csv_df[col].iloc[0] if len(csv_df) > 0 else None
            if isinstance(first_val, list):
                csv_df[col] = csv_df[col].apply(str)
    
    csv_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved: {output_file} ({len(game_player_market):,} rows)")
    
    return game_player_market


# =============================================================================
# MODE: GAME-PLAYER-MARKET-LINE (Individual book lines, n=1 per row)
# =============================================================================

def run_game_player_market_line_analysis(props_df: pd.DataFrame, game_logs_df: pd.DataFrame):
    """
    Analyze at game + player + market + LINE VALUE level.
    
    Each row = one game + one player + one market + one unique line value
    Aggregates across bookmakers that offer the same line value.
    
    Example: If DraftKings and FanDuel both offer 25.5, they're aggregated into one row.
             If Bovada offers 26.5, that's a separate row.
    """
    print("\n" + "=" * 80)
    print("🎯 MODE: GAME-PLAYER-MARKET-LINE ANALYSIS (group by line value)")
    print("=" * 80)
    
    # Get individual book lines matched with actuals
    matched_df = join_raw_props_with_actuals(props_df, game_logs_df)
    
    if len(matched_df) == 0:
        print("❌ No matches found!")
        return
    
    # Add vig attribution columns (5 metrics per line)
    print("\n📊 Calculating vig attribution metrics...")
    vig_attrs = matched_df.apply(
        lambda row: calculate_vig_attribution(row['implied_over_prob'], row['implied_under_prob']),
        axis=1
    )
    matched_df['total_vig'] = vig_attrs.apply(lambda x: x['total_vig'])
    matched_df['over_vig'] = vig_attrs.apply(lambda x: x['over_vig'])
    matched_df['under_vig'] = vig_attrs.apply(lambda x: x['under_vig'])
    matched_df['fair_over'] = vig_attrs.apply(lambda x: x['fair_over'])
    matched_df['fair_under'] = vig_attrs.apply(lambda x: x['fair_under'])
    
    # First, calculate prop-level stats (for context: how many distinct lines for this prop?)
    prop_stats = matched_df.groupby(['date', 'player', 'market']).agg(
        prop_n_books=('bookmaker', 'nunique'),
        prop_n_distinct_lines=('line', 'nunique'),
        prop_min_line=('line', 'min'),
        prop_max_line=('line', 'max'),
        prop_line_std=('line', 'std'),  # Standard deviation of lines across books
        prop_line_mean=('line', 'mean'),  # Mean line
        prop_min_vig=('total_vig', 'min'),  # For consensus line detection
        # Odds dispersion at prop level (across ALL lines/books)
        prop_over_odds_min=('over_odds', 'min'),
        prop_over_odds_max=('over_odds', 'max'),
        prop_implied_over_min=('implied_over_prob', 'min'),
        prop_implied_over_max=('implied_over_prob', 'max'),
    ).reset_index()
    prop_stats['prop_line_spread'] = prop_stats['prop_max_line'] - prop_stats['prop_min_line']
    prop_stats['prop_over_odds_spread'] = prop_stats['prop_over_odds_max'] - prop_stats['prop_over_odds_min']
    prop_stats['prop_implied_range'] = prop_stats['prop_implied_over_max'] - prop_stats['prop_implied_over_min']
    
    # Group by game + player + market + LINE VALUE
    print("\n📊 Grouping by game + player + market + line value...")
    game_player_market_line = matched_df.groupby(['date', 'player', 'market', 'line']).agg(
        # How many books offer THIS exact line value
        n=('bookmaker', 'count'),
        n_books=('bookmaker', 'nunique'),
        
        # Game/team info (same for all books on this prop)
        game=('game', 'first'),
        matchup=('matchup', 'first'),
        opponent=('opponent', 'first'),
        home_away=('home_away', 'first'),
        
        # Actual value (same for all books on this prop)
        actual=('actual', 'first'),
        hit_over=('hit_over', 'first'),  # Same for all since same line value
        
        # List of bookmakers offering this line
        lst_bookmakers=('bookmaker', lambda x: list(x)),
        
        # Odds lists (from books offering this exact line)
        lst_over_odds=('over_odds', lambda x: list(x)),
        lst_under_odds=('under_odds', lambda x: list(x)),
        lst_implied_over=('implied_over_prob', lambda x: list(x)),
        lst_implied_under=('implied_under_prob', lambda x: list(x)),
        
        # Vig lists
        lst_total_vig=('total_vig', lambda x: list(x)),
        lst_over_vig=('over_vig', lambda x: list(x)),
        lst_under_vig=('under_vig', lambda x: list(x)),
        lst_fair_over=('fair_over', lambda x: list(x)),
        lst_fair_under=('fair_under', lambda x: list(x)),
        
        # Error lists (same actual but same line, so errors should be same)
        lst_errors=('error', lambda x: list(x)),
        lst_error_pcts=('error_pct', lambda x: list(x)),
        
        # Sanity check columns (should be uniform within each row)
        lst_lines=('line', lambda x: list(x)),  # All same value
        lst_hit_over=('hit_over', lambda x: list(x)),  # All 1s or all 0s
        lst_actuals=('actual', lambda x: list(x)),  # All same value
    ).reset_index()
    
    # Merge prop-level stats
    game_player_market_line = game_player_market_line.merge(
        prop_stats, on=['date', 'player', 'market'], how='left'
    )
    
    # Calculate aggregate stats
    game_player_market_line['avg_line'] = game_player_market_line['line']  # It's the line value itself
    game_player_market_line['std_line'] = 0  # No variation within same line value
    game_player_market_line['n_distinct_lines'] = game_player_market_line['prop_n_distinct_lines']
    
    # Note: prop_min_line, prop_max_line, prop_line_spread are already from parent level (game-player-market)
    # Add boolean flags for easy filtering
    game_player_market_line['is_min_line'] = (game_player_market_line['line'] == game_player_market_line['prop_min_line']).astype(int)
    game_player_market_line['is_max_line'] = (game_player_market_line['line'] == game_player_market_line['prop_max_line']).astype(int)
    
    # Add categorical: line position (only_line, min, max, middle)
    def calc_line_position(row):
        if row['prop_line_spread'] == 0:
            return 'only_line'
        elif row['is_min_line'] == 1:
            return 'min'
        elif row['is_max_line'] == 1:
            return 'max'
        else:
            return 'middle'
    
    game_player_market_line['cat_line_position'] = game_player_market_line.apply(calc_line_position, axis=1)
    
    # Error stats (error is same for all books with same line value)
    game_player_market_line['error'] = game_player_market_line['line'] - game_player_market_line['actual']
    game_player_market_line['abs_error'] = abs(game_player_market_line['error'])
    game_player_market_line['error_pct'] = game_player_market_line['error'] / game_player_market_line['line']
    game_player_market_line['abs_error_pct'] = abs(game_player_market_line['error_pct'])
    game_player_market_line['avg_error'] = game_player_market_line['error']
    game_player_market_line['avg_error_pct'] = game_player_market_line['error_pct']
    game_player_market_line['std_error'] = 0
    game_player_market_line['hit_over_rate'] = game_player_market_line['hit_over'].astype(float)
    
    # Vig stats (average across books offering this line)
    game_player_market_line['avg_total_vig'] = game_player_market_line['lst_total_vig'].apply(lambda x: np.nanmean(x))
    game_player_market_line['avg_over_vig'] = game_player_market_line['lst_over_vig'].apply(lambda x: np.nanmean(x))
    game_player_market_line['avg_under_vig'] = game_player_market_line['lst_under_vig'].apply(lambda x: np.nanmean(x))
    game_player_market_line['avg_fair_over'] = game_player_market_line['lst_fair_over'].apply(lambda x: np.nanmean(x))
    game_player_market_line['avg_fair_under'] = game_player_market_line['lst_fair_under'].apply(lambda x: np.nanmean(x))
    
    # Scalar vig columns (avg across books for this line)
    game_player_market_line['total_vig'] = game_player_market_line['avg_total_vig']
    game_player_market_line['over_vig'] = game_player_market_line['avg_over_vig']
    game_player_market_line['under_vig'] = game_player_market_line['avg_under_vig']
    game_player_market_line['fair_over'] = game_player_market_line['avg_fair_over']
    game_player_market_line['fair_under'] = game_player_market_line['avg_fair_under']
    
    # Implied probability averages
    game_player_market_line['implied_over_prob'] = game_player_market_line['lst_implied_over'].apply(lambda x: np.nanmean(x))
    game_player_market_line['implied_under_prob'] = game_player_market_line['lst_implied_under'].apply(lambda x: np.nanmean(x))
    
    # ==========================================================================
    # ODDS DISPERSION METRICS (at line level - same line across different books)
    # ==========================================================================
    print("\n📊 Calculating odds dispersion metrics...")
    
    # Odds spread at this line (max - min odds across books offering this exact line)
    game_player_market_line['line_over_odds_min'] = game_player_market_line['lst_over_odds'].apply(lambda x: np.nanmin(x) if len(x) > 0 else np.nan)
    game_player_market_line['line_over_odds_max'] = game_player_market_line['lst_over_odds'].apply(lambda x: np.nanmax(x) if len(x) > 0 else np.nan)
    game_player_market_line['line_under_odds_min'] = game_player_market_line['lst_under_odds'].apply(lambda x: np.nanmin(x) if len(x) > 0 else np.nan)
    game_player_market_line['line_under_odds_max'] = game_player_market_line['lst_under_odds'].apply(lambda x: np.nanmax(x) if len(x) > 0 else np.nan)
    
    # Odds spread (difference between best and worst odds at this line)
    game_player_market_line['line_over_odds_spread'] = game_player_market_line['line_over_odds_max'] - game_player_market_line['line_over_odds_min']
    game_player_market_line['line_under_odds_spread'] = game_player_market_line['line_under_odds_max'] - game_player_market_line['line_under_odds_min']
    
    # Implied probability range at this line
    game_player_market_line['line_implied_over_min'] = game_player_market_line['lst_implied_over'].apply(lambda x: np.nanmin(x) if len(x) > 0 else np.nan)
    game_player_market_line['line_implied_over_max'] = game_player_market_line['lst_implied_over'].apply(lambda x: np.nanmax(x) if len(x) > 0 else np.nan)
    game_player_market_line['line_implied_range'] = game_player_market_line['line_implied_over_max'] - game_player_market_line['line_implied_over_min']
    
    # Vig range at this line
    game_player_market_line['line_vig_min'] = game_player_market_line['lst_total_vig'].apply(lambda x: np.nanmin(x) if len(x) > 0 else np.nan)
    game_player_market_line['line_vig_max'] = game_player_market_line['lst_total_vig'].apply(lambda x: np.nanmax(x) if len(x) > 0 else np.nan)
    game_player_market_line['line_vig_range'] = game_player_market_line['line_vig_max'] - game_player_market_line['line_vig_min']
    
    # NA columns for compatibility
    game_player_market_line['bookmaker'] = np.nan  # Multiple bookmakers aggregated
    game_player_market_line['over_odds'] = np.nan
    game_player_market_line['under_odds'] = np.nan
    
    # Sanity check: lst_distinct_lines should be length 1 (only one unique line per row)
    game_player_market_line['lst_distinct_lines'] = game_player_market_line['lst_lines'].apply(lambda x: list(set(x)))
    
    # Is this line the consensus (lowest avg vig for this prop)?
    game_player_market_line['is_consensus_line'] = (
        (game_player_market_line['total_vig'] == 
         game_player_market_line.groupby(['date', 'player', 'market'])['total_vig'].transform('min'))
    ).astype(int)
    
    # Add categorical columns
    print("\n📊 Adding categorical columns...")
    game_player_market_line = add_categorical_columns(game_player_market_line, is_aggregated=False)
    
    print(f"\n  ✅ {len(game_player_market_line):,} game-player-market-line rows")
    
    # Print summary
    print("\n" + "=" * 170)
    print("📊 GAME-PLAYER-MARKET-LINE SUMMARY (sample of 20 rows)")
    print("=" * 170)
    print(f"{'Date':<12} {'Player':<18} {'Market':<14} {'Line':>6} {'#Books':>6} {'Actual':>7} {'Error':>7} {'Err%':>6} {'TotVig':>7} {'OvrVig':>7} {'UndVig':>7} {'FairOvr':>8} {'Hit':>4} {'Liq':>8} {'Cons':>5}")
    print("-" * 170)
    
    sample = game_player_market_line.sample(min(20, len(game_player_market_line)))
    for _, row in sample.iterrows():
        mkt = row['market'].replace('player_', '')[:12]
        player = row['player'][:16]
        hit = "✓" if row['hit_over'] else "✗"
        err_pct = row['error_pct'] if pd.notna(row['error_pct']) else 0
        fair_over = row['fair_over'] if pd.notna(row['fair_over']) else 0
        total_vig = row['total_vig'] if pd.notna(row['total_vig']) else 0
        over_vig = row['over_vig'] if pd.notna(row['over_vig']) else 0
        under_vig = row['under_vig'] if pd.notna(row['under_vig']) else 0
        liq = row.get('cat_liquidity', 'unk')[:6]
        cons = "✓" if row.get('is_consensus_line', 0) == 1 else ""
        print(f"{row['date']:<12} {player:<18} {mkt:<14} {row['line']:>6.1f} {row['n_books']:>6} {row['actual']:>7.1f} {row['error']:>+7.2f} {err_pct:>+5.0%} {total_vig:>6.1%} {over_vig:>6.1%} {under_vig:>6.1%} {fair_over:>7.1%} {hit:>4} {liq:>8} {cons:>5}")
    
    print("-" * 170)
    
    # Categorical summary
    print("\n📊 CATEGORICAL SUMMARY:")
    print(f"   Liquidity: {game_player_market_line['cat_liquidity'].value_counts().to_dict()}")
    print(f"   Vig Structure: {game_player_market_line['cat_vig_structure'].value_counts().to_dict()}")
    print(f"   Line Quartile: {game_player_market_line['cat_line_quartile'].value_counts().to_dict()}")
    print(f"   Line Position: {game_player_market_line['cat_line_position'].value_counts().to_dict()}")
    print(f"   Consensus Lines: {game_player_market_line['is_consensus_line'].sum():,} / {len(game_player_market_line):,} ({game_player_market_line['is_consensus_line'].mean()*100:.1f}%)")
    
    # Vig stats
    print("\n📊 VIG ATTRIBUTION STATS (across all line values):")
    print(f"   Total Vig - Mean: {game_player_market_line['total_vig'].mean():.2%}  Std: {game_player_market_line['total_vig'].std():.2%}")
    print(f"   Over Vig  - Mean: {game_player_market_line['over_vig'].mean():.2%}  Std: {game_player_market_line['over_vig'].std():.2%}")
    print(f"   Under Vig - Mean: {game_player_market_line['under_vig'].mean():.2%}  Std: {game_player_market_line['under_vig'].std():.2%}")
    print(f"   Fair Over - Mean: {game_player_market_line['fair_over'].mean():.1%}  Std: {game_player_market_line['fair_over'].std():.1%}")
    print(f"   Fair Under- Mean: {game_player_market_line['fair_under'].mean():.1%}  Std: {game_player_market_line['fair_under'].std():.1%}")
    
    print("=" * 170)
    
    # Save
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    output_file = INTERMEDIATE_DIR / 'game_player_market_line_analysis.csv'
    game_player_market_line.to_csv(output_file, index=False)
    print(f"\n💾 Saved: {output_file} ({len(game_player_market_line):,} rows)")
    
    return game_player_market_line


# =============================================================================
# VERBOSE EDGE ANALYSIS
# =============================================================================

def print_verbose_edge_analysis(df: pd.DataFrame):
    """
    Print detailed edge analysis tables for each categorical metric.
    Called when --verbose flag is used.
    """
    
    # Define categorical columns and their display names
    # Get unique values dynamically from the data
    CATEGORIES = [
        ('cat_liquidity', 'LIQUIDITY', sorted(df['cat_liquidity'].dropna().unique())),
        ('cat_vig_structure', 'VIG STRUCTURE', sorted(df['cat_vig_structure'].dropna().unique())),
        ('cat_line_quartile', 'LINE QUARTILE', sorted(df['cat_line_quartile'].dropna().unique())),
        ('cat_line_position', 'LINE POSITION', ['only_line', 'min', 'middle', 'max']),
        ('is_consensus_line', 'CONSENSUS LINE', [1, 0]),
    ]
    
    # Add market structure if available (game-player-market-line mode)
    if 'cat_market_structure' in df.columns:
        CATEGORIES.insert(0, ('cat_market_structure', 'MARKET STRUCTURE', 
            ['captive', 'consensus', 'moderate', 'line_disagree', 'odds_disagree']))
    
    # Display names for consensus line (1=consensus, 0=alt)
    CONSENSUS_DISPLAY = {1: 'consensus', 0: 'alt'}
    
    markets = sorted(df['market'].unique())
    
    # ==========================================================================
    # PART 1: Overall Edge by Category
    # ==========================================================================
    print("\n" + "=" * 120)
    print("📊 VERBOSE: OVERALL EDGE BY CATEGORY")
    print("=" * 120)
    
    for cat_col, cat_name, cat_values in CATEGORIES:
        print(f"\n{'─' * 120}")
        print(f"📈 {cat_name}")
        print(f"{'─' * 140}")
        print(f"{'Category':<16} {'N':>10} {'Ovr%':>7} {'Und%':>7} {'ImpOvr':>8} {'ImpUnd':>8} {'OvrEdge':>9} {'UndEdge':>9} {'BestSide':>10} {'BestEdge':>9} {'AvgVig':>7}")
        print(f"{'─' * 140}")
        
        results = []
        for cat in cat_values:
            subset = df[df[cat_col] == cat]
            if len(subset) < 10:
                continue
            
            n = len(subset)
            over_rate = subset['hit_over'].mean()
            under_rate = 1 - over_rate
            implied_over = subset['implied_over_prob'].mean()
            implied_under = subset['implied_under_prob'].mean()
            
            # Edge for BOTH sides
            over_edge = over_rate - implied_over
            under_edge = under_rate - implied_under
            
            # Which side is better?
            if over_edge > under_edge:
                best_side = 'OVER'
                best_edge = over_edge
            else:
                best_side = 'UNDER'
                best_edge = under_edge
            
            avg_vig = subset['total_vig'].mean()
            
            results.append((cat, n, over_rate, under_rate, implied_over, implied_under,
                          over_edge, under_edge, best_side, best_edge, avg_vig))
        
        # Sort by best_edge descending
        results.sort(key=lambda x: x[9], reverse=True)
        
        for cat, n, over_rate, under_rate, imp_over, imp_under, ovr_edge, und_edge, best_side, best_edge, avg_vig in results:
            # Convert 1/0 to consensus/alt for display
            if cat_col == 'is_consensus_line':
                cat_str = CONSENSUS_DISPLAY.get(cat, str(cat))[:14]
            else:
                cat_str = str(cat)[:14]
            
            # Color indicators
            ovr_profit = '✅' if ovr_edge > 0 else ''
            und_profit = '✅' if und_edge > 0 else ''
            
            print(f"{cat_str:<16} {n:>10,} {over_rate:>6.1%} {under_rate:>6.1%} {imp_over:>7.1%} {imp_under:>7.1%} {ovr_edge*100:>+7.1f}%{ovr_profit} {und_edge*100:>+7.1f}%{und_profit} {best_side:>10} {best_edge*100:>+8.1f}% {avg_vig:>6.1%}")
    
    # ==========================================================================
    # PART 2: Edge by Category × Market
    # ==========================================================================
    print("\n\n" + "=" * 140)
    print("📊 VERBOSE: EDGE BY CATEGORY × MARKET")
    print("=" * 140)
    
    for cat_col, cat_name, cat_values in CATEGORIES:
        print(f"\n{'━' * 160}")
        print(f"📈 {cat_name} × MARKET")
        print(f"{'━' * 160}")
        print(f"{'Market':<24} {'Category':<14} {'N':>7} {'Ovr%':>6} {'Und%':>6} {'OvrEdge':>8} {'UndEdge':>8} {'Best':>6} {'BestEdge':>9} {'AvgLine':>8} {'AvgVig':>6}")
        print(f"{'─' * 160}")
        
        results = []
        for market in markets:
            for cat in cat_values:
                subset = df[(df['market'] == market) & (df[cat_col] == cat)]
                if len(subset) < 30:  # Min sample size
                    continue
                
                n = len(subset)
                over_rate = subset['hit_over'].mean()
                under_rate = 1 - over_rate
                implied_over = subset['implied_over_prob'].mean()
                implied_under = subset['implied_under_prob'].mean()
                
                # Edge for BOTH sides
                over_edge = over_rate - implied_over
                under_edge = under_rate - implied_under
                
                # Which side is better?
                if over_edge > under_edge:
                    best_side = 'OVER'
                    best_edge = over_edge
                else:
                    best_side = 'UNDER'
                    best_edge = under_edge
                
                avg_line = subset['line'].mean()
                avg_vig = subset['total_vig'].mean()
                
                # Convert 1/0 to consensus/alt for display
                if cat_col == 'is_consensus_line':
                    cat_display = CONSENSUS_DISPLAY.get(cat, str(cat))
                else:
                    cat_display = str(cat)
                
                results.append((market.replace('player_', ''), cat_display, n, over_rate, under_rate,
                              over_edge, under_edge, best_side, best_edge, avg_line, avg_vig))
        
        # Sort by best_edge descending
        results.sort(key=lambda x: x[8], reverse=True)
        
        for mkt, cat, n, over_rate, under_rate, ovr_edge, und_edge, best_side, best_edge, avg_line, avg_vig in results:
            ovr_flag = '✅' if ovr_edge > 0 else ''
            und_flag = '✅' if und_edge > 0 else ''
            print(f"{mkt:<24} {cat:<14} {n:>7,} {over_rate:>5.1%} {under_rate:>5.1%} {ovr_edge*100:>+6.1f}%{ovr_flag} {und_edge*100:>+6.1f}%{und_flag} {best_side:>6} {best_edge*100:>+8.1f}% {avg_line:>8.1f} {avg_vig:>5.1%}")
    
    # ==========================================================================
    # PART 3: Cross-Category Analysis (2-way combinations)
    # ==========================================================================
    print("\n\n" + "=" * 160)
    print("📊 VERBOSE: CROSS-CATEGORY COMBINATIONS (Liquidity × Vig Structure)")
    print("=" * 160)
    print(f"{'Liquidity':<20} {'VigStructure':<20} {'N':>8} {'Ovr%':>6} {'Und%':>6} {'OvrEdge':>9} {'UndEdge':>9} {'Best':>6} {'BestEdge':>9} {'AvgVig':>7}")
    print(f"{'─' * 160}")
    
    liq_values = df['cat_liquidity'].dropna().unique()
    vig_values = df['cat_vig_structure'].dropna().unique()
    
    results = []
    for liq in liq_values:
        for vig in vig_values:
            subset = df[(df['cat_liquidity'] == liq) & (df['cat_vig_structure'] == vig)]
            if len(subset) < 50:
                continue
            
            n = len(subset)
            over_rate = subset['hit_over'].mean()
            under_rate = 1 - over_rate
            implied_over = subset['implied_over_prob'].mean()
            implied_under = subset['implied_under_prob'].mean()
            
            over_edge = over_rate - implied_over
            under_edge = under_rate - implied_under
            
            if over_edge > under_edge:
                best_side = 'OVER'
                best_edge = over_edge
            else:
                best_side = 'UNDER'
                best_edge = under_edge
            
            avg_vig = subset['total_vig'].mean()
            
            results.append((str(liq)[:18], str(vig)[:18], n, over_rate, under_rate, 
                          over_edge, under_edge, best_side, best_edge, avg_vig))
    
    results.sort(key=lambda x: x[8], reverse=True)
    for liq, vig, n, over_rate, under_rate, ovr_edge, und_edge, best_side, best_edge, avg_vig in results:
        ovr_flag = '✅' if ovr_edge > 0 else ''
        und_flag = '✅' if und_edge > 0 else ''
        print(f"{liq:<20} {vig:<20} {n:>8,} {over_rate:>5.1%} {under_rate:>5.1%} {ovr_edge*100:>+7.1f}%{ovr_flag} {und_edge*100:>+7.1f}%{und_flag} {best_side:>6} {best_edge*100:>+8.1f}% {avg_vig:>6.1%}")
    
    # ==========================================================================
    # PART 4: Best Opportunities Summary
    # ==========================================================================
    print("\n\n" + "=" * 160)
    print("🎯 VERBOSE: TOP 20 EXPLOITABLE SEGMENTS (Market × Category - BOTH Over & Under)")
    print("=" * 160)
    print(f"{'Rank':<5} {'Side':<6} {'Market':<22} {'Category':<16} {'Type':<14} {'N':>7} {'HitRate':>8} {'Implied':>8} {'Edge':>8} {'AvgVig':>7}")
    print(f"{'─' * 160}")
    
    all_opportunities = []
    
    for cat_col, cat_name, cat_values in CATEGORIES:
        for market in markets:
            for cat in cat_values:
                subset = df[(df['market'] == market) & (df[cat_col] == cat)]
                if len(subset) < 50:
                    continue
                
                n = len(subset)
                over_rate = subset['hit_over'].mean()
                under_rate = 1 - over_rate
                implied_over = subset['implied_over_prob'].mean()
                implied_under = subset['implied_under_prob'].mean()
                
                over_edge = over_rate - implied_over
                under_edge = under_rate - implied_under
                avg_vig = subset['total_vig'].mean()
                
                # Convert 1/0 to consensus/alt for display
                if cat_col == 'is_consensus_line':
                    cat_display = CONSENSUS_DISPLAY.get(cat, str(cat))
                else:
                    cat_display = str(cat)
                
                # Add OVER opportunity if profitable
                if over_edge > 0:
                    all_opportunities.append((
                        'OVER',
                        market.replace('player_', ''),
                        cat_display,
                        cat_name,
                        n,
                        over_rate,
                        implied_over,
                        over_edge,
                        avg_vig
                    ))
                
                # Add UNDER opportunity if profitable
                if under_edge > 0:
                    all_opportunities.append((
                        'UNDER',
                        market.replace('player_', ''),
                        cat_display,
                        cat_name,
                        n,
                        under_rate,
                        implied_under,
                        under_edge,
                        avg_vig
                    ))
    
    # Sort by edge and take top 20
    all_opportunities.sort(key=lambda x: x[7], reverse=True)
    
    for i, (side, mkt, cat, cat_type, n, hit_rate, implied, edge, avg_vig) in enumerate(all_opportunities[:20], 1):
        print(f"{i:<5} {side:<6} {mkt:<22} {cat:<16} {cat_type:<14} {n:>7,} {hit_rate:>7.1%} {implied:>7.1%} {edge*100:>+7.1f}% {avg_vig:>6.1%}")
    
    # ==========================================================================
    # PART 5: Bookmaker Analysis (if lst_bookmakers available)
    # ==========================================================================
    if 'lst_bookmakers' in df.columns and df['lst_bookmakers'].notna().any():
        print("\n\n" + "=" * 120)
        print("📊 VERBOSE: BOOKMAKER FREQUENCY IN PROFITABLE SEGMENTS")
        print("=" * 120)
        
        # Filter to profitable thin liquidity
        profitable_thin = df[(df['cat_liquidity'] == 'thin') & 
                            (df['hit_over'] == True)]
        
        if len(profitable_thin) > 0:
            # Explode bookmaker lists and count
            from collections import Counter
            all_books = []
            for books in profitable_thin['lst_bookmakers'].dropna():
                if isinstance(books, list):
                    all_books.extend(books)
                elif isinstance(books, str):
                    # Handle string representation of list
                    try:
                        import ast
                        books_list = ast.literal_eval(books)
                        all_books.extend(books_list)
                    except:
                        pass
            
            if all_books:
                book_counts = Counter(all_books)
                print(f"\nBookmakers in THIN liquidity winning OVERS:")
                print(f"{'Bookmaker':<30} {'Count':>10} {'%':>8}")
                print(f"{'─' * 50}")
                total = sum(book_counts.values())
                for book, count in book_counts.most_common(15):
                    print(f"{book:<30} {count:>10,} {count/total:>7.1%}")
    
    print("\n" + "=" * 120)
    print("✅ VERBOSE ANALYSIS COMPLETE")
    print("=" * 120)
    
    return all_opportunities


def print_combo_analysis(df: pd.DataFrame):
    """
    Analyze edge for category combinations defined in ANALYSIS_COMBOS.
    This allows systematic exploration of granular segments.
    
    Combos that don't show edge can be commented out in the config
    to document "paths not to go down."
    """
    import numpy as np
    
    print("\n" + "=" * 120)
    print("📊 COMBO ANALYSIS: MULTI-DIMENSIONAL EDGE DISCOVERY")
    print("=" * 120)
    print(f"   Min sample size: {COMBO_MIN_N}")
    print(f"   Top segments per combo: {COMBO_TOP_N}")
    
    # Helper to parse list columns
    def parse_list(x):
        if isinstance(x, (list, np.ndarray)):
            return list(x)
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return []
        try: 
            return eval(str(x).replace('nan', 'None'))
        except: 
            return []
    
    def safe_min(lst):
        if not lst:
            return np.nan
        vals = [v for v in lst if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return np.min(vals) if vals else np.nan
    
    # Pre-compute best implied if not already present
    if 'best_over_implied' not in df.columns and 'lst_implied_over' in df.columns:
        df = df.copy()
        df['best_over_implied'] = df['lst_implied_over'].apply(lambda x: safe_min(parse_list(x)))
        df['best_under_implied'] = df['lst_implied_under'].apply(lambda x: safe_min(parse_list(x)))
    
    df['hit_under'] = 1 - df['hit_over']
    
    # Simplify vig for display
    if 'cat_vig_structure' in df.columns:
        df['vig_short'] = df['cat_vig_structure'].apply(lambda x: str(x).split()[0] if pd.notna(x) else 'unk')
    
    all_combo_results = []
    
    for combo in ANALYSIS_COMBOS:
        print(f"\n{'─' * 120}")
        combo_name = ' × '.join(combo)
        print(f"📈 COMBO: {combo_name}")
        print(f"{'─' * 120}")
        
        # Map category names to columns
        combo_cols = []
        valid_combo = True
        for cat in combo:
            col = CATEGORY_COLUMNS.get(cat)
            if col is None:
                print(f"   ⚠️ Unknown category: {cat}")
                valid_combo = False
                break
            # Use simplified vig if available
            if cat == 'vig' and 'vig_short' in df.columns:
                col = 'vig_short'
            if col not in df.columns:
                print(f"   ⚠️ Column not in data: {col}")
                valid_combo = False
                break
            combo_cols.append((cat, col))
        
        if not valid_combo:
            continue
        
        # Get all unique combinations
        results = []
        
        # Build segment filter dynamically
        from itertools import product
        
        # Get unique values for each dimension
        dim_values = []
        for cat_name, col in combo_cols:
            if cat_name == 'market':
                vals = [v.replace('player_', '') for v in df[col].dropna().unique()]
            else:
                vals = [str(v).split()[0] if pd.notna(v) else 'unk' for v in df[col].dropna().unique()]
            dim_values.append(list(set(vals)))
        
        # Iterate through all combinations
        for val_combo in product(*dim_values):
            # Build filter
            mask = pd.Series([True] * len(df))
            segment_parts = []
            
            for (cat_name, col), val in zip(combo_cols, val_combo):
                if cat_name == 'market':
                    mask = mask & (df[col].str.replace('player_', '') == val)
                else:
                    mask = mask & (df[col].astype(str).str.split().str[0] == val)
                segment_parts.append(val)
            
            subset = df[mask]
            if len(subset) < COMBO_MIN_N:
                continue
            
            # Calculate edges
            hit_o = subset['hit_over'].mean()
            hit_u = subset['hit_under'].mean()
            
            # Use best implied if available, otherwise avg
            if 'best_over_implied' in subset.columns:
                impl_o = subset['best_over_implied'].mean()
                impl_u = subset['best_under_implied'].mean()
            else:
                impl_o = subset['implied_over_prob'].mean()
                impl_u = subset['implied_under_prob'].mean()
            
            edge_o = hit_o - impl_o
            edge_u = hit_u - impl_u
            best_edge = max(edge_o, edge_u)
            best_side = 'OVER' if edge_o > edge_u else 'UNDER'
            
            results.append({
                'segment': '+'.join(segment_parts),
                'n': len(subset),
                'over_edge': edge_o,
                'under_edge': edge_u,
                'best_side': best_side,
                'best_edge': best_edge,
                'combo': combo_name
            })
        
        if not results:
            print("   No segments with sufficient sample size")
            continue
        
        # Sort by best edge and show top N
        results_df = pd.DataFrame(results).sort_values('best_edge', ascending=False)
        
        # Count positive edge segments
        positive = (results_df['best_edge'] > 0).sum()
        total = len(results_df)
        print(f"   Segments with positive edge: {positive} / {total} ({positive/total*100:.0f}%)")
        print()
        
        print(f"   {'Segment':<50} {'N':>5} {'Side':>6} {'Edge':>7}")
        print(f"   {'─' * 70}")
        
        for _, r in results_df.head(COMBO_TOP_N).iterrows():
            flag = '✓✓✓' if r['best_edge'] > 0.10 else ('✓✓' if r['best_edge'] > 0.05 else ('✓' if r['best_edge'] > 0 else ''))
            print(f"   {r['segment']:<50} {r['n']:>5} {r['best_side']:>6} {r['best_edge']:>+6.1%} {flag}")
        
        all_combo_results.extend(results)
    
    # Summary: Best segments across ALL combos
    if all_combo_results:
        print(f"\n{'=' * 120}")
        print("🏆 TOP SEGMENTS ACROSS ALL COMBOS")
        print(f"{'=' * 120}")
        
        all_df = pd.DataFrame(all_combo_results).sort_values('best_edge', ascending=False)
        
        print(f"\n{'Segment':<55} {'Combo':<30} {'N':>5} {'Side':>6} {'Edge':>7}")
        print(f"{'─' * 110}")
        
        for _, r in all_df.head(20).iterrows():
            flag = '✓✓✓' if r['best_edge'] > 0.10 else ('✓✓' if r['best_edge'] > 0.05 else ('✓' if r['best_edge'] > 0 else ''))
            print(f"{r['segment']:<55} {r['combo']:<30} {r['n']:>5} {r['best_side']:>6} {r['best_edge']:>+6.1%} {flag}")
    
    print(f"\n{'=' * 120}")
    print("✅ COMBO ANALYSIS COMPLETE")
    print(f"{'=' * 120}")


def extract_top_strategies_data(df: pd.DataFrame, all_opportunities: list, top_n: int = 5) -> pd.DataFrame:
    """
    Extract rows from the dataframe that match the top N strategies.
    
    Args:
        df: The game-player-market-line dataframe
        all_opportunities: List of tuples from print_verbose_edge_analysis
        top_n: Number of top strategies to extract (default 5)
    
    Returns:
        DataFrame with rows matching top N strategies, plus strategy metadata columns
    """
    print("\n" + "=" * 120)
    print(f"📊 EXTRACTING TOP {top_n} STRATEGIES DATA")
    print("=" * 120)
    
    # Map category names back to column names
    CAT_NAME_TO_COL = {
        'MARKET STRUCTURE': 'cat_market_structure',
        'LIQUIDITY': 'cat_liquidity',
        'VIG STRUCTURE': 'cat_vig_structure',
        'LINE QUARTILE': 'cat_line_quartile',
        'LINE POSITION': 'cat_line_position',
        'CONSENSUS LINE': 'is_consensus_line',
    }
    
    # Map consensus display names back to values
    CONSENSUS_DISPLAY_REVERSE = {'consensus': 1, 'alt': 0}
    
    # Sort by edge and take top N
    sorted_opps = sorted(all_opportunities, key=lambda x: x[7], reverse=True)[:top_n]
    
    print(f"\n🎯 Top {top_n} strategies selected:")
    print(f"{'Rank':<5} {'Side':<6} {'Market':<24} {'Category':<24} {'Type':<16} {'N':>7} {'Edge':>8} {'HitRate':>9} {'Implied':>9} {'AvgVig':>8}")
    print("-" * 130)
    
    for i, (side, mkt, cat, cat_type, n, hit_rate, implied, edge, avg_vig) in enumerate(sorted_opps, 1):
        print(f"{i:<5} {side:<6} {mkt:<24} {cat:<24} {cat_type:<16} {n:>7,} {edge*100:>+7.1f}% {hit_rate*100:>8.1f}% {implied*100:>8.1f}% {avg_vig*100:>7.2f}%")
    
    # Build combined filter for all top strategies
    all_strategy_dfs = []
    
    for strategy_rank, (side, mkt, cat, cat_type, n, hit_rate, implied, edge, avg_vig) in enumerate(sorted_opps, 1):
        # Get the column name for this category type
        cat_col = CAT_NAME_TO_COL.get(cat_type)
        if not cat_col:
            print(f"  ⚠️ Unknown category type: {cat_type}")
            continue
        
        # Add player_ prefix back to market
        full_market = f"player_{mkt}"
        
        # Convert category value for consensus line
        if cat_type == 'CONSENSUS LINE':
            cat_value = CONSENSUS_DISPLAY_REVERSE.get(cat, cat)
        else:
            cat_value = cat
        
        # Build filter
        market_mask = df['market'] == full_market
        cat_mask = df[cat_col] == cat_value
        
        # Side filter - OVER means we want hit_over==1 outcomes (for strategy validation)
        # But we want to include ALL rows that match the strategy criteria, not just winners
        # The side tells us which direction to bet
        
        subset = df[market_mask & cat_mask].copy()
        
        if len(subset) == 0:
            print(f"  ⚠️ No rows found for strategy {strategy_rank}: {side} {mkt} | {cat}")
            continue
        
        # Add strategy metadata columns
        subset['strategy_rank'] = strategy_rank
        subset['strategy_side'] = side
        subset['strategy_market'] = mkt
        subset['strategy_category'] = cat
        subset['strategy_category_type'] = cat_type
        subset['strategy_edge'] = edge
        subset['strategy_implied'] = implied
        subset['strategy_hit_rate'] = hit_rate
        
        # Calculate if this specific bet would have won
        if side == 'OVER':
            subset['strategy_bet_won'] = subset['hit_over'].astype(int)
        else:
            subset['strategy_bet_won'] = (1 - subset['hit_over']).astype(int)
        
        all_strategy_dfs.append(subset)
        
        print(f"\n  {'='*100}")
        print(f"  ✅ STRATEGY {strategy_rank}: {side} {mkt.upper()} | {cat} ({cat_type})")
        print(f"  {'='*100}")
        print(f"     📋 Filter: market='{full_market}' AND {cat_col}='{cat_value}'")
        print(f"     📋 Bet Side: {side}")
        print(f"     📋 Total Rows: {len(subset):,}")
        
        # Win/Loss breakdown
        wins = subset['strategy_bet_won'].sum()
        losses = len(subset) - wins
        win_rate = subset['strategy_bet_won'].mean()
        print(f"\n     📊 Performance:")
        print(f"        Win Rate: {win_rate:.1%} ({wins} W / {losses} L)")
        print(f"        Expected Edge: {edge*100:+.2f}%")
        print(f"        Implied Prob: {implied*100:.1f}%")
        print(f"        Avg Vig: {avg_vig*100:.2f}%")
        
        # Date range
        date_min = subset['date'].min()
        date_max = subset['date'].max()
        unique_dates = subset['date'].nunique()
        print(f"\n     📅 Date Range:")
        print(f"        From: {date_min} to {date_max}")
        print(f"        Unique Dates: {unique_dates}")
        
        # Player breakdown
        unique_players = subset['player'].nunique()
        top_players = subset.groupby('player').size().sort_values(ascending=False).head(5)
        print(f"\n     👤 Players ({unique_players} unique):")
        for player, count in top_players.items():
            player_wins = subset[subset['player'] == player]['strategy_bet_won'].sum()
            player_wr = player_wins / count if count > 0 else 0
            print(f"        {player}: {count} bets ({player_wins}W/{count-player_wins}L, {player_wr:.0%})")
        if unique_players > 5:
            print(f"        ... and {unique_players - 5} more players")
        
        # Line statistics
        print(f"\n     📏 Line Stats:")
        print(f"        Mean Line: {subset['line'].mean():.2f}")
        print(f"        Median Line: {subset['line'].median():.2f}")
        print(f"        Min/Max: {subset['line'].min():.1f} / {subset['line'].max():.1f}")
        print(f"        Std Dev: {subset['line'].std():.2f}")
        
        # Actual outcome stats
        print(f"\n     📈 Actual Outcome Stats:")
        print(f"        Mean Actual: {subset['actual'].mean():.2f}")
        print(f"        Median Actual: {subset['actual'].median():.2f}")
        print(f"        Actual vs Line Diff (mean): {(subset['actual'] - subset['line']).mean():+.2f}")
        
        # Books breakdown
        if 'n_books' in subset.columns:
            print(f"\n     📚 Bookmaker Coverage:")
            print(f"        Avg Books: {subset['n_books'].mean():.1f}")
            print(f"        Min/Max Books: {subset['n_books'].min()} / {subset['n_books'].max()}")
    
    if not all_strategy_dfs:
        print("  ❌ No data found for any strategies!")
        return pd.DataFrame()
    
    # Combine all strategies
    combined_df = pd.concat(all_strategy_dfs, ignore_index=True)
    
    # Sort by strategy rank, then date
    combined_df = combined_df.sort_values(['strategy_rank', 'date', 'player', 'market'])
    
    # Verbose summary
    print("\n" + "=" * 120)
    print("📊 COMBINED STRATEGIES SUMMARY")
    print("=" * 120)
    
    print(f"\n📈 Overview:")
    print(f"   Total rows: {len(combined_df):,}")
    print(f"   Unique games (dates): {combined_df['date'].nunique()}")
    print(f"   Unique players: {combined_df['player'].nunique()}")
    print(f"   Markets: {combined_df['market'].unique().tolist()}")
    print(f"   Date Range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    
    print("\n📊 Breakdown by Strategy:")
    print(f"{'Rank':<5} {'Side':<6} {'Market':<16} {'Category':<20} {'CatType':<16} {'Rows':>6} {'W':>5} {'L':>5} {'WinRate':>8} {'Edge':>8} {'Implied':>9}")
    print("-" * 120)
    
    for rank in sorted(combined_df['strategy_rank'].unique()):
        strat_data = combined_df[combined_df['strategy_rank'] == rank]
        row = strat_data.iloc[0]
        win_rate = strat_data['strategy_bet_won'].mean()
        wins = strat_data['strategy_bet_won'].sum()
        losses = len(strat_data) - wins
        print(f"{rank:<5} {row['strategy_side']:<6} {row['strategy_market']:<16} {row['strategy_category']:<20} {row['strategy_category_type']:<16} {len(strat_data):>6,} {wins:>5} {losses:>5} {win_rate:>7.1%} {row['strategy_edge']*100:>+7.1f}% {row['strategy_implied']*100:>8.1f}%")
    
    print("\n📊 Overall Performance:")
    overall_win_rate = combined_df['strategy_bet_won'].mean()
    total_wins = combined_df['strategy_bet_won'].sum()
    total_losses = len(combined_df) - total_wins
    print(f"   Combined Win Rate: {overall_win_rate:.1%} ({total_wins} W / {total_losses} L)")
    print(f"   Average Edge: {combined_df['strategy_edge'].mean()*100:+.1f}%")
    print(f"   Weighted Edge (by row count): {(combined_df['strategy_edge'] * combined_df['strategy_rank'].map(lambda x: len(combined_df[combined_df['strategy_rank'] == x]))).sum() / len(combined_df) * 100:+.1f}%")
    
    # Cross-strategy player analysis
    print("\n👤 Top Players Across All Strategies:")
    player_stats = combined_df.groupby('player').agg({
        'strategy_bet_won': ['sum', 'count', 'mean'],
        'strategy_rank': lambda x: x.unique().tolist()
    }).reset_index()
    player_stats.columns = ['player', 'wins', 'total', 'win_rate', 'strategies']
    player_stats = player_stats.sort_values('total', ascending=False).head(10)
    
    print(f"   {'Player':<25} {'Bets':>6} {'W':>5} {'L':>5} {'WinRate':>8} {'Strategies'}")
    print(f"   {'-'*80}")
    for _, p in player_stats.iterrows():
        losses = p['total'] - p['wins']
        strat_list = ','.join(map(str, sorted(p['strategies'])))
        print(f"   {p['player']:<25} {p['total']:>6} {p['wins']:>5} {losses:>5} {p['win_rate']:>7.1%}  #{strat_list}")
    
    # Column order for output
    strategy_cols = [
        'strategy_rank', 'strategy_side', 'strategy_market', 'strategy_category', 
        'strategy_category_type', 'strategy_edge', 'strategy_implied', 'strategy_hit_rate',
        'strategy_bet_won'
    ]
    
    core_cols = [
        'date', 'player', 'game', 'matchup', 'opponent', 'home_away',
        'market', 'line', 'actual', 'hit_over',
        'n_books', 'n_distinct_lines', 
        'prop_min_line', 'prop_max_line', 'prop_line_spread', 'is_min_line', 'is_max_line',
        'total_vig', 'over_vig', 'under_vig', 'fair_over', 'fair_under',
        'implied_over_prob', 'implied_under_prob',
        'error', 'error_pct', 'abs_error',
    ]
    
    categorical_cols = [
        'cat_liquidity', 'cat_vig_structure', 'cat_line_quartile', 'cat_line_position', 'is_consensus_line'
    ]
    
    list_cols = [col for col in combined_df.columns if col.startswith('lst_')]
    
    # Reorder columns
    ordered_cols = strategy_cols + core_cols + categorical_cols + list_cols
    # Add any remaining columns not yet included
    remaining_cols = [c for c in combined_df.columns if c not in ordered_cols]
    final_col_order = ordered_cols + remaining_cols
    # Filter to columns that actually exist
    final_col_order = [c for c in final_col_order if c in combined_df.columns]
    
    combined_df = combined_df[final_col_order]
    
    # Save to intermediate
    output_file = INTERMEDIATE_DIR / 'top_strategies_game_player_market_line.csv'
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 120)
    print(f"💾 SAVED: {output_file}")
    print(f"   Rows: {len(combined_df):,}")
    print(f"   Columns: {len(combined_df.columns)}")
    print("=" * 120)
    
    # Verbose column info
    print("\n📋 Output Columns:")
    print("   Strategy metadata: " + ", ".join(strategy_cols))
    print("   Core data: " + ", ".join([c for c in core_cols if c in combined_df.columns]))
    print("   Categories: " + ", ".join([c for c in categorical_cols if c in combined_df.columns]))
    if list_cols:
        print("   List columns: " + ", ".join(list_cols[:5]) + ("..." if len(list_cols) > 5 else ""))
    
    return combined_df


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='NBA Props vs Actuals Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nba_props_vs_actuals.py --mode high-level              # Market-level summary (consensus)
  python nba_props_vs_actuals.py --mode liquidity               # Per-prop liquidity + book accuracy
  python nba_props_vs_actuals.py --mode game-player-market      # Game + player + market (aggregate across books)
  python nba_props_vs_actuals.py --mode game-player-market-line # Game + player + market + line value (aggregate books with same line)
  python nba_props_vs_actuals.py --mode game-player-market-line --verbose  # With detailed edge tables
        """
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=AVAILABLE_MODES,
        default='high-level',
        help=f'Analysis mode. Options: {", ".join(AVAILABLE_MODES)}. Default: high-level'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed edge analysis tables for each categorical metric'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    result_df = None
    
    # Modes that need raw props data (not consensus)
    RAW_PROPS_MODES = ['liquidity', 'game-player-market', 'game-player-market-line']
    
    if args.mode in RAW_PROPS_MODES:
        print("=" * 80)
        print(f"🏀 NBA PROPS VS ACTUALS - {args.mode.upper()} ANALYSIS")
        if args.verbose:
            print("📊 VERBOSE MODE ENABLED - Will print detailed edge tables")
        print("=" * 80)
        
        # Load raw data
        props_df = load_all_props()
        game_logs_df = load_game_logs()
        
        if args.mode == 'liquidity':
            result_df = run_liquidity_analysis(props_df, game_logs_df)
        elif args.mode == 'game-player-market':
            result_df = run_game_player_market_analysis(props_df, game_logs_df)
        elif args.mode == 'game-player-market-line':
            result_df = run_game_player_market_line_analysis(props_df, game_logs_df)
        
        # Print verbose edge analysis if flag is set
        if args.verbose and result_df is not None:
            all_opportunities = print_verbose_edge_analysis(result_df)
            
            # Run combo analysis (multi-dimensional edge discovery)
            print_combo_analysis(result_df)
            
            # Extract top N strategies data to CSV
            if all_opportunities:
                top_n = 5  # Take top 5 strategies
                strategies_df = extract_top_strategies_data(result_df, all_opportunities, top_n=top_n)
    else:
        # Load consensus-matched data for other modes
        matched_df = load_consensus_lines_data()
        
        if args.mode == 'high-level':
            run_high_level_analysis(matched_df)
        elif args.mode == 'player':
            run_player_analysis(matched_df)
        elif args.mode == 'line-value':
            run_line_value_analysis(matched_df)
    
    print(f"\n✅ Analysis complete!")
    
    return result_df


if __name__ == '__main__':
    matched_df = main()

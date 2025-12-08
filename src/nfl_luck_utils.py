"""
NFL Luck Analysis Utilities

Simple utilities for NFL luck analysis (overperformance/underperformance).
"""

import pandas as pd
from pathlib import Path
import glob
from typing import Optional, Dict, List

from nfl_team_utils import add_team_abbr_columns, normalize_unexpected_points_abbr
from config import DATA_ROOT, NFL_2025_BYE_WEEKS

# =============================================================================
# PATHS
# =============================================================================

NFL_LINES_DIR = DATA_ROOT / "01_input/the-odds-api/nfl/game_lines/historical"
NFL_LINES_UPCOMING_DIR = DATA_ROOT / "01_input/the-odds-api/nfl/game_lines/upcoming"
UNEXPECTED_POINTS_DIR = DATA_ROOT / "01_input/unexpected_points"
INTERMEDIATE_DIR = DATA_ROOT / "03_intermediate"
OUTPUT_DIR = DATA_ROOT / "04_output/nfl"


# =============================================================================
# CATEGORY CONSTANTS
# =============================================================================

LUCK_CATEGORIES: List[str] = ['Lucky', 'Neutral', 'Unlucky']
SPREAD_CATEGORIES: List[str] = ['0-3', '3.5-7', '7.5+']


# =============================================================================
# SIMPLE FUNCTIONS
# =============================================================================

def get_nfl_week(game_date: pd.Timestamp, season: int = 2025) -> int:
    """Calculate NFL week from game date. Week 1 of 2025 starts Sept 4."""
    week1_start = pd.Timestamp('2025-09-04', tz='America/New_York')
    
    if game_date.tz is None:
        game_date = game_date.tz_localize('UTC')
    
    game_date_et = game_date.tz_convert('America/New_York')
    days_since = (game_date_et - week1_start).days
    
    return max(1, (days_since // 7) + 1)


def categorize_luck(luck_value: float, threshold: float = 7.0) -> str:
    """Categorize luck: 'Lucky' if >= threshold, 'Unlucky' if <= -threshold, else 'Neutral'."""
    if luck_value >= threshold:
        return 'Lucky'
    elif luck_value <= -threshold:
        return 'Unlucky'
    return 'Neutral'


def categorize_spread(spread: float) -> str:
    """Categorize spread: '0-3', '3.5-7', or '7.5+'."""
    abs_spread = abs(spread)
    if abs_spread <= 3:
        return '0-3'
    elif abs_spread <= 7:
        return '3.5-7'
    return '7.5+'


def calculate_roi(win_pct: float, odds: int = -110) -> float:
    """Calculate ROI % for a given win percentage at American odds."""
    if odds < 0:
        decimal_odds = 1 + (100 / abs(odds))
    else:
        decimal_odds = 1 + (odds / 100)
    return (win_pct * decimal_odds - 1) * 100


# =============================================================================
# DATA LOADING
# =============================================================================

def load_nfl_betting_lines(include_upcoming: bool = False) -> pd.DataFrame:
    """Load all NFL betting lines from historical dir."""
    csv_files = sorted(glob.glob(str(NFL_LINES_DIR / "nfl_game_lines_*.csv")))
    
    # Add London games
    london_file = NFL_LINES_DIR / "2025_game_lines_london.csv"
    if london_file.exists():
        csv_files.append(str(london_file))
    
    if include_upcoming and NFL_LINES_UPCOMING_DIR.exists():
        csv_files.extend(glob.glob(str(NFL_LINES_UPCOMING_DIR / "nfl_game_lines_*.csv")))
    
    dfs = [pd.read_csv(f) for f in csv_files if Path(f).exists()]
    df = pd.concat(dfs, ignore_index=True)
    
    df['game_time'] = pd.to_datetime(df['game_time'])
    if df['game_time'].dt.tz is None:
        df['game_time'] = df['game_time'].dt.tz_localize('UTC')
    
    # Filter to 2025 season
    season_start = pd.Timestamp('2025-09-01', tz='UTC')
    return df[df['game_time'] >= season_start].copy()


def calculate_consensus_lines(df_lines: pd.DataFrame) -> pd.DataFrame:
    """Calculate median spread for each game."""
    df_lines = add_team_abbr_columns(df_lines)
    
    results = []
    for game_id, g in df_lines.groupby('game_id'):
        spreads = g['away_spread'].dropna()
        if len(spreads) == 0:
            continue
        
        results.append({
            'game_id': game_id,
            'game_time': g['game_time'].iloc[0],
            'away_team': g['away_team'].iloc[0],
            'home_team': g['home_team'].iloc[0],
            'away_abbr': g['away_abbr'].iloc[0],
            'home_abbr': g['home_abbr'].iloc[0],
            'consensus_spread': spreads.median(),
            'num_books': len(spreads),
        })
    
    return pd.DataFrame(results)


def load_unexpected_points_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load Unexpected Points data with luck calculated.
    
    Luck Components:
    - offensive_luck: your_score - your_adj_score (you scored more/less than expected)
    - defensive_luck: opp_adj_score - opp_score (opponent scored less/more than expected)
    - luck (total): offensive_luck + defensive_luck (full game luck)
    
    Note: Total luck is zero-sum per game (one team's luck = -1 × opponent's luck)
    """
    if file_path is None:
        xlsx_files = sorted(UNEXPECTED_POINTS_DIR.glob("Unexpected Points*.xlsx"))
        if not xlsx_files:
            raise FileNotFoundError(f"No Unexpected Points files in {UNEXPECTED_POINTS_DIR}")
        file_path = xlsx_files[-1]
    
    df = pd.read_excel(file_path, sheet_name="2025 Adjusted Scores")
    df['team_canonical'] = df['team'].apply(normalize_unexpected_points_abbr)
    
    # Calculate offensive luck (old method - kept for reference)
    df['offensive_luck'] = df['score'] - df['adj_score']
    
    # Calculate defensive luck by looking up opponent's data in same game
    df['defensive_luck'] = 0.0
    
    for game_id in df['game_id'].unique():
        game_mask = df['game_id'] == game_id
        game_rows = df[game_mask]
        
        if len(game_rows) != 2:
            continue
        
        # Get both teams' data
        team1_idx = game_rows.index[0]
        team2_idx = game_rows.index[1]
        
        team1_score = df.loc[team1_idx, 'score']
        team1_adj = df.loc[team1_idx, 'adj_score']
        team2_score = df.loc[team2_idx, 'score']
        team2_adj = df.loc[team2_idx, 'adj_score']
        
        # Defensive luck = opponent's adj_score - opponent's actual score
        # (if opponent scored less than expected, you got lucky defensively)
        df.loc[team1_idx, 'defensive_luck'] = team2_adj - team2_score
        df.loc[team2_idx, 'defensive_luck'] = team1_adj - team1_score
    
    # Total luck = offensive + defensive
    df['luck'] = df['offensive_luck'] + df['defensive_luck']
    
    return df


def build_prior_luck_lookup(df_up: pd.DataFrame) -> Dict:
    """
    Build lookup for prior week luck. Returns dict with:
    - 'by_team_week': {(team, week): {'luck': total, 'offensive_luck': off, 'defensive_luck': def}}
    - 'weeks_played': {team: [week1, week2, ...]}
    """
    by_team_week = {}
    weeks_played = {}
    
    for _, row in df_up.iterrows():
        team = row['team_canonical']
        week = row['week']
        
        by_team_week[(team, week)] = {
            'luck': row['luck'],
            'offensive_luck': row.get('offensive_luck', row['luck']),  # fallback for old data
            'defensive_luck': row.get('defensive_luck', 0.0),
        }
        
        if team not in weeks_played:
            weeks_played[team] = []
        weeks_played[team].append(week)
    
    for team in weeks_played:
        weeks_played[team] = sorted(weeks_played[team])
    
    return {'by_team_week': by_team_week, 'weeks_played': weeks_played}


def get_prior_week_luck(lookup: Dict, team: str, current_week: int) -> Optional[float]:
    """Get total luck from team's last played game before current_week (handles byes)."""
    luck_data = get_prior_week_luck_detailed(lookup, team, current_week)
    if luck_data is None:
        return None
    return luck_data['luck']


def get_prior_week_luck_detailed(lookup: Dict, team: str, current_week: int) -> Optional[Dict]:
    """
    Get detailed luck breakdown from team's last played game before current_week.
    
    Returns dict with:
    - 'luck': total luck (offensive + defensive)
    - 'offensive_luck': how much team over/under performed scoring
    - 'defensive_luck': how much opponent over/under performed scoring
    
    Returns None if no prior game found.
    """
    weeks_played = lookup['weeks_played']
    by_team_week = lookup['by_team_week']
    
    if team not in weeks_played:
        return None
    
    prior_weeks = [w for w in weeks_played[team] if w < current_week]
    if not prior_weeks:
        return None
    
    last_played = max(prior_weeks)
    return by_team_week.get((team, last_played))


def get_luck_matchup_ats_results(df: pd.DataFrame, luck_cat_a: str, luck_cat_b: str) -> tuple:
    """
    Get ATS (against the spread) results for luck category matchups.
    
    Example: get_luck_matchup_ats_results(df, 'Lucky', 'Unlucky') returns how often
    the Lucky team covered vs how often the Unlucky team covered.
    
    Args:
        df: DataFrame with 'away_luck_cat', 'home_luck_cat', 'away_covered' columns
        luck_cat_a: First luck category ('Lucky', 'Neutral', or 'Unlucky')
        luck_cat_b: Second luck category ('Lucky', 'Neutral', or 'Unlucky')
    
    Returns:
        Tuple of (luck_cat_a_covers, luck_cat_b_covers, total_games)
    """
    subset = df[
        ((df['away_luck_cat'] == luck_cat_a) & (df['home_luck_cat'] == luck_cat_b)) |
        ((df['away_luck_cat'] == luck_cat_b) & (df['home_luck_cat'] == luck_cat_a))
    ]
    
    if len(subset) == 0:
        return 0, 0, 0
    
    luck_cat_a_covers = 0
    luck_cat_b_covers = 0
    
    for _, game in subset.iterrows():
        if game['away_luck_cat'] == luck_cat_a:
            if game['away_covered']:
                luck_cat_a_covers += 1
            else:
                luck_cat_b_covers += 1
        else:
            if game['away_covered']:
                luck_cat_b_covers += 1
            else:
                luck_cat_a_covers += 1
    
    return luck_cat_a_covers, luck_cat_b_covers, len(subset)

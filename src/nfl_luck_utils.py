"""
NFL Luck Analysis Utilities

Simple utilities for NFL luck analysis (overperformance/underperformance).
"""

import pandas as pd
from pathlib import Path
import glob
from typing import Optional, Dict, List

from nfl_team_utils import add_team_abbr_columns, normalize_unexpected_points_abbr
from config import DATA_ROOT, NFL_2020_BYE_WEEKS, NFL_2021_BYE_WEEKS, NFL_2022_BYE_WEEKS, NFL_2023_BYE_WEEKS, NFL_2024_BYE_WEEKS, NFL_2025_BYE_WEEKS

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

def get_bye_weeks(season: int = 2025) -> Dict[str, int]:
    """Get bye weeks dictionary for specified season."""
    if season == 2020:
        return NFL_2020_BYE_WEEKS
    elif season == 2021:
        return NFL_2021_BYE_WEEKS
    elif season == 2022:
        return NFL_2022_BYE_WEEKS
    elif season == 2023:
        return NFL_2023_BYE_WEEKS
    elif season == 2024:
        return NFL_2024_BYE_WEEKS
    elif season == 2025:
        return NFL_2025_BYE_WEEKS
    else:
        raise ValueError(f"Bye weeks not configured for season {season}")


def get_nfl_week(game_date: pd.Timestamp, season: int = 2025) -> int:
    """
    Calculate NFL week from game date.
    
    Season start dates (Week 1 Thursday):
    - 2020: Sept 10 (HOU @ KC)
    - 2021: Sept 9 (DAL @ TB)
    - 2022: Sept 8 (LAR vs BUF)
    - 2023: Sept 7 (DET @ KC)
    - 2024: Sept 5 (BAL @ KC)
    - 2025: Sept 4
    """
    season_starts = {
        2020: pd.Timestamp('2020-09-10', tz='America/New_York'),
        2021: pd.Timestamp('2021-09-09', tz='America/New_York'),
        2022: pd.Timestamp('2022-09-08', tz='America/New_York'),
        2023: pd.Timestamp('2023-09-07', tz='America/New_York'),
        2024: pd.Timestamp('2024-09-05', tz='America/New_York'),
        2025: pd.Timestamp('2025-09-04', tz='America/New_York'),
    }
    
    if season not in season_starts:
        raise ValueError(f"Season {season} not supported. Use 2020-2025.")
    
    week1_start = season_starts[season]
    
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

def load_nfl_betting_lines(include_upcoming: bool = False, season: int = 2025) -> pd.DataFrame:
    """
    Load all NFL betting lines from historical dir.
    
    Args:
        include_upcoming: Include upcoming games dir. Default: False.
        season: Season to filter (2020-2025). Default: 2025.
    """
    csv_files = sorted(glob.glob(str(NFL_LINES_DIR / "nfl_game_lines_*.csv")))
    
    # Add season-specific London games
    london_file = NFL_LINES_DIR / f"{season}_game_lines_london.csv"
    if london_file.exists():
        csv_files.append(str(london_file))
    
    if include_upcoming and NFL_LINES_UPCOMING_DIR.exists():
        csv_files.extend(glob.glob(str(NFL_LINES_UPCOMING_DIR / "nfl_game_lines_*.csv")))
    
    dfs = [pd.read_csv(f) for f in csv_files if Path(f).exists()]
    df = pd.concat(dfs, ignore_index=True)
    
    df['game_time'] = pd.to_datetime(df['game_time'])
    if df['game_time'].dt.tz is None:
        df['game_time'] = df['game_time'].dt.tz_localize('UTC')
    
    # Filter to specified season
    if season == 2020:
        season_start = pd.Timestamp('2020-09-01', tz='UTC')
        season_end = pd.Timestamp('2021-02-28', tz='UTC')
        return df[(df['game_time'] >= season_start) & (df['game_time'] <= season_end)].copy()
    elif season == 2021:
        season_start = pd.Timestamp('2021-09-01', tz='UTC')
        season_end = pd.Timestamp('2022-02-28', tz='UTC')
        return df[(df['game_time'] >= season_start) & (df['game_time'] <= season_end)].copy()
    elif season == 2022:
        season_start = pd.Timestamp('2022-09-01', tz='UTC')
        season_end = pd.Timestamp('2023-02-28', tz='UTC')
        return df[(df['game_time'] >= season_start) & (df['game_time'] <= season_end)].copy()
    elif season == 2023:
        season_start = pd.Timestamp('2023-09-01', tz='UTC')
        season_end = pd.Timestamp('2024-02-28', tz='UTC')
        return df[(df['game_time'] >= season_start) & (df['game_time'] <= season_end)].copy()
    elif season == 2024:
        season_start = pd.Timestamp('2024-09-01', tz='UTC')
        season_end = pd.Timestamp('2025-02-28', tz='UTC')
        return df[(df['game_time'] >= season_start) & (df['game_time'] <= season_end)].copy()
    else:  # 2025
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


def load_unexpected_points_data(file_path: Optional[Path] = None, season: int = 2025) -> pd.DataFrame:
    """
    Load Unexpected Points data with luck calculated.
    
    Args:
        file_path: Path to Excel file. If None, uses most recent file.
        season: Season to load (2020-2025). Default: 2025.
    
    Sheet naming convention:
    - 2020, 2021: "Adjusted Scores 2012-2021" (combined sheet)
    - 2022, 2023: "Adjusted Scores 2022" or "Adjusted Scores 2023"
    - 2024, 2025+: "{year} Adjusted Scores" (e.g., "2024 Adjusted Scores")
    - Before 2012: Not supported (raises ValueError)
    
    Luck Components:
    - offensive_luck: your_score - your_adj_score (you scored more/less than expected)
    - defensive_luck: opp_adj_score - opp_score (opponent scored less/more than expected)
    - luck (total): offensive_luck + defensive_luck (full game luck)
    
    Note: Total luck is zero-sum per game (one team's luck = -1 × opponent's luck)
    """
    if file_path is None:
        xlsx_files = list(UNEXPECTED_POINTS_DIR.glob("Unexpected Points*.xlsx"))
        if not xlsx_files:
            raise FileNotFoundError(f"No Unexpected Points files in {UNEXPECTED_POINTS_DIR}")
        # Sort by modification time (most recent first) to get the latest file
        xlsx_files = sorted(xlsx_files, key=lambda f: f.stat().st_mtime, reverse=True)
        file_path = xlsx_files[0]
        print(f"Using latest Unexpected Points file: {file_path}")
    
    # Determine sheet name based on season
    if season < 2012:
        raise ValueError(f"Season {season} not supported. Unexpected Points data only available from 2012 onwards.")
    elif 2012 <= season <= 2021:
        sheet_name = "Adjusted Scores 2012-2021"
    elif season in [2022, 2023]:
        sheet_name = f"Adjusted Scores {season}"
    else:  # 2024, 2025, future seasons
        sheet_name = f"{season} Adjusted Scores"
    
    print(f"Loading sheet: '{sheet_name}'")
    df = pd.read_excel(file_path, sheet_name=sheet_name)
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
    Build lookup for prior week luck with full game context. Returns dict with:
    - 'by_team_week': {(team, week): {luck details + game context}}
    - 'weeks_played': {team: [week1, week2, ...]}
    
    Each entry includes:
    - luck, offensive_luck, defensive_luck
    - week, opponent, score, opp_score, adj_score, opp_adj_score, won
    """
    by_team_week = {}
    weeks_played = {}
    
    # First pass: collect all games by game_id to find opponents
    games_by_id = {}
    for _, row in df_up.iterrows():
        game_id = row['game_id']
        if game_id not in games_by_id:
            games_by_id[game_id] = []
        games_by_id[game_id].append(row)
    
    # Second pass: build lookup with opponent info
    for _, row in df_up.iterrows():
        team = row['team_canonical']
        week = row['week']
        game_id = row['game_id']
        score = row['score']
        adj_score = row['adj_score']
        
        # Find opponent in same game
        game_rows = games_by_id[game_id]
        opp_row = None
        for gr in game_rows:
            if gr['team_canonical'] != team:
                opp_row = gr
                break
        
        opponent = opp_row['team_canonical'] if opp_row is not None else 'UNK'
        opp_score = opp_row['score'] if opp_row is not None else 0
        opp_adj_score = opp_row['adj_score'] if opp_row is not None else 0
        
        by_team_week[(team, week)] = {
            'luck': row['luck'],
            'offensive_luck': row.get('offensive_luck', row['luck']),
            'defensive_luck': row.get('defensive_luck', 0.0),
            'week': week,
            'opponent': opponent,
            'score': score,
            'opp_score': opp_score,
            'adj_score': adj_score,
            'opp_adj_score': opp_adj_score,
            'won': score > opp_score,
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
    - 'week': the week number of the prior game
    - 'opponent': opponent team abbreviation
    - 'score': team's actual score
    - 'opp_score': opponent's actual score
    - 'adj_score': team's expected score
    - 'opp_adj_score': opponent's expected score
    - 'won': True if team won
    
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
        df: DataFrame with 'away_prior_luck_cat', 'home_prior_luck_cat', 'away_covered' columns
        luck_cat_a: First luck category ('Lucky', 'Neutral', or 'Unlucky')
        luck_cat_b: Second luck category ('Lucky', 'Neutral', or 'Unlucky')
    
    Returns:
        Tuple of (luck_cat_a_covers, luck_cat_b_covers, total_games)
    """
    subset = df[
        ((df['away_prior_luck_cat'] == luck_cat_a) & (df['home_prior_luck_cat'] == luck_cat_b)) |
        ((df['away_prior_luck_cat'] == luck_cat_b) & (df['home_prior_luck_cat'] == luck_cat_a))
    ]
    
    if len(subset) == 0:
        return 0, 0, 0
    
    luck_cat_a_covers = 0
    luck_cat_b_covers = 0
    
    for _, game in subset.iterrows():
        if game['away_prior_luck_cat'] == luck_cat_a:
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

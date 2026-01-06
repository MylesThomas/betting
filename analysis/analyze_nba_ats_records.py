"""
NBA Against The Spread (ATS) Analysis for 2025-26 Season

Analyzes which teams are performing well ATS, especially focusing on teams 
that might have losing records but cover spreads frequently. These teams 
represent good betting value.

Key metrics:
- ATS record (wins-losses-pushes)
- ATS win percentage
- Average game margin (how much they win/lose by)
- Average spread margin (how much they beat/miss spread by)
- Home vs Away ATS splits
- As favorite vs underdog ATS performance

Usage:
    # Run analysis only (console output + CSV)
    python analysis/analyze_nba_ats_records.py --season 2025
    
    # Run analysis + create R visualization
    python analysis/analyze_nba_ats_records.py --season 2025 --plot

Output:
    - CSV: data/04_output/nba_ats_analysis/nba_ats_rankings_YYYY_MM_YYYYMMDD.csv
    - CSV: data/04_output/nba_ats_analysis/nba_ats_game_results_YYYY_MM_YYYYMMDD.csv
    - PNG: content/viz/nba/nba_ats_rankings_YYYY_MM.png (if --plot flag used)

Data Sources:
    - Betting lines: The Odds API (DraftKings) via S3 bucket 'the-odds-api-mt'
    - Game results: nba_api via data/01_input/nba_api/historical/

Key Insights from Recent Analysis (2025-26 through Jan 4):
    - Best ATS: Suns 23-11 (67.6%), 76ers 20-13 (60.6%), Nuggets/Celtics 20-14 (58.8%)
    - Worst ATS: Cavaliers 12-24 (33.3%), Kings 12-23 (34.3%), Clippers 14-20 (41.2%)
    - Suns are elite when favored (13-2, 86.7%) - best favorite cover rate
    - Cavaliers: Good team (20-16 W-L) but WORST ATS - massively overvalued by market
    - Bulls/Blazers crush as dogs but terrible as favorites - market inefficiency
    - Pelicans: Bad team (8-28) but 21-15 ATS (58.3%) - market overcorrected

Author: Thomas Myles
Date: 2026-01-04
"""

import pandas as pd
import numpy as np
import boto3
from io import BytesIO
from datetime import datetime
from zoneinfo import ZoneInfo
import sys
import os
import argparse
from pathlib import Path

# Add src to path for config
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from config_loader import get_config

CONFIG = get_config()

# Constants
RATE_LIMIT_DELAY = 0.6  # NBA API rate limit
S3_BUCKET = 'the-odds-api-mt'

# Team name mapping (Odds API → nba_api)
TEAM_NAME_MAP = {
    'Los Angeles Clippers': 'LA Clippers',
    # Add more if needed
}


def get_season_string(season_year):
    """
    Convert season year to string format
    
    Args:
        season_year: Year season starts (e.g., 2025 for 2025-26)
    
    Returns:
        String like "2025-26"
    """
    return f"{season_year}-{str(season_year + 1)[-2:]}"


def normalize_team_name(team_name, reverse=False):
    """
    Normalize team names between Odds API and nba_api
    
    Args:
        team_name: Team name to normalize
        reverse: If True, map nba_api → Odds API instead
    
    Returns:
        Normalized team name
    """
    if reverse:
        # Reverse mapping (nba_api → Odds API)
        reverse_map = {v: k for k, v in TEAM_NAME_MAP.items()}
        return reverse_map.get(team_name, team_name)
    else:
        # Forward mapping (Odds API → nba_api)
        return TEAM_NAME_MAP.get(team_name, team_name)


def load_all_game_lines(season_year):
    """
    Load all NBA game lines for specified season from S3
    
    Args:
        season_year: Year season starts (e.g., 2025 for 2025-26)
    
    Returns:
        DataFrame with all game lines
    """
    season_str = get_season_string(season_year)
    s3_prefix = f'nba/historical_game_lines/{season_str}/'
    
    s3 = boto3.client('s3')
    
    # List all CSV files for the season
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_prefix)
    
    all_dfs = []
    
    for obj in response.get('Contents', []):
        key = obj['Key']
        
        # Skip non-CSV files and failures file
        if not key.endswith('.csv') or 'failed' in key.lower():
            continue
        
        # Read CSV from S3
        try:
            response = s3.get_object(Bucket=S3_BUCKET, Key=key)
            df = pd.read_csv(BytesIO(response['Body'].read()))
            all_dfs.append(df)
        except Exception as e:
            print(f"⚠️  Error reading {key}: {e}")
    
    if not all_dfs:
        raise ValueError("No game lines found in S3")
    
    # Combine all dataframes
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Convert timestamps
    df['game_time'] = pd.to_datetime(df['game_time'])
    df['odds_pull_time'] = pd.to_datetime(df['odds_pull_time'])
    
    return df


def get_nba_scores(season_year):
    """
    Load NBA game scores from nba-api data
    
    Args:
        season_year: Year season starts (e.g., 2025 for 2025-26)
    
    Returns:
        DataFrame with game results (team, opponent, points, opp_points, date)
    """
    season_str = get_season_string(season_year).replace('-', '_')
    scores_path = Path(__file__).parent.parent / 'data' / '01_input' / 'nba_api' / 'historical' / f'nba_games_{season_str}.csv'
    
    if not scores_path.exists():
        raise ValueError(f"NBA scores file not found: {scores_path}\n" 
                        f"Run: python scripts/fetch_nba_team_game_results.py --season {season_year}")
    
    # Load data
    scores = pd.read_csv(scores_path)
    
    # Convert game date
    scores['GAME_DATE'] = pd.to_datetime(scores['GAME_DATE'])
    
    # Keep only what we need
    scores = scores[['TEAM_NAME', 'GAME_DATE', 'MATCHUP', 'WL', 'PTS']].copy()
    scores.columns = ['team', 'game_date', 'matchup', 'win_loss', 'points']
    
    return scores


def calculate_ats_records(lines_df, scores_df):
    """
    Calculate ATS records for each team
    
    Args:
        lines_df: DataFrame with betting lines
        scores_df: DataFrame with game results
    
    Returns:
        DataFrame with team ATS records
    """
    # Filter to spread market only
    spreads = lines_df[lines_df['market'] == 'spread'].copy()
    
    # Use a single bookmaker (consensus) - let's use DraftKings
    spreads = spreads[spreads['bookmaker_key'] == 'draftkings']
    
    # If DraftKings not available, use first available bookmaker
    if spreads.empty:
        bookmaker = lines_df[lines_df['market'] == 'spread']['bookmaker_key'].iloc[0]
        spreads = lines_df[(lines_df['market'] == 'spread') & 
                          (lines_df['bookmaker_key'] == bookmaker)].copy()
        print(f"ℹ️  Using {bookmaker} lines (DraftKings not available)")
    
    # Get unique games
    games = spreads.groupby('game_id').first().reset_index()
    
    # Convert game times to ET timezone for proper date matching
    et_tz = ZoneInfo('America/New_York')
    
    results = []
    
    for _, game in games.iterrows():
        # Convert game time to ET to get the correct US date
        game_time_utc = pd.to_datetime(game['game_time'])
        if game_time_utc.tzinfo is None:
            game_time_utc = game_time_utc.tz_localize('UTC')
        game_time_et = game_time_utc.tz_convert(et_tz)
        game_date = game_time_et.date()  # Use ET date for matching
        
        away_team = normalize_team_name(game['away_team'])  # Normalize
        home_team = normalize_team_name(game['home_team'])  # Normalize
        away_spread = game['away_line']
        home_spread = game['home_line']
        
        # Find game result in scores
        away_score_row = scores_df[
            (scores_df['team'] == away_team) & 
            (scores_df['game_date'].dt.date == game_date)
        ]
        
        home_score_row = scores_df[
            (scores_df['team'] == home_team) & 
            (scores_df['game_date'].dt.date == game_date)
        ]
        
        if away_score_row.empty or home_score_row.empty:
            # Game hasn't been played yet or scores not available
            continue
        
        away_points = away_score_row.iloc[0]['points']
        home_points = home_score_row.iloc[0]['points']
        
        # Calculate actual margin (away perspective)
        actual_margin = away_points - home_points
        
        # Away team ATS result
        # The spread is the adjustment: +5 means add 5 to their score
        # So: adjusted_margin = actual_margin + spread
        away_ats_margin = actual_margin + away_spread  # FIXED: was "actual_margin - away_spread"
        if away_ats_margin > 0:
            away_ats_result = 'W'
        elif away_ats_margin < 0:
            away_ats_result = 'L'
        else:
            away_ats_result = 'P'  # Push
        
        # Home team ATS result
        home_ats_margin = -actual_margin + home_spread  # FIXED: was "-actual_margin - home_spread"
        if home_ats_margin > 0:
            home_ats_result = 'W'
        elif home_ats_margin < 0:
            home_ats_result = 'L'
        else:
            home_ats_result = 'P'  # Push
        
        # Determine if team was favorite or underdog
        away_fav = 'Favorite' if away_spread < 0 else 'Underdog' if away_spread > 0 else 'Pick'
        home_fav = 'Favorite' if home_spread < 0 else 'Underdog' if home_spread > 0 else 'Pick'
        
        # Away team record
        results.append({
            'team': away_team,
            'game_date': game_date,
            'opponent': home_team,
            'location': 'Away',
            'spread': away_spread,
            'actual_margin': actual_margin,
            'ats_margin': away_ats_margin,
            'ats_result': away_ats_result,
            'fav_dog': away_fav,
            'team_points': away_points,
            'opp_points': home_points
        })
        
        # Home team record
        results.append({
            'team': home_team,
            'game_date': game_date,
            'opponent': away_team,
            'location': 'Home',
            'spread': home_spread,
            'actual_margin': -actual_margin,
            'ats_margin': home_ats_margin,
            'ats_result': home_ats_result,
            'fav_dog': home_fav,
            'team_points': home_points,
            'opp_points': away_points
        })
    
    return pd.DataFrame(results)


def summarize_ats_records(ats_df):
    """
    Summarize ATS records by team
    
    Args:
        ats_df: DataFrame with game-level ATS results
    
    Returns:
        DataFrame with team ATS summaries, sorted by ATS win %
    """
    summaries = []
    
    for team in sorted(ats_df['team'].unique()):
        team_games = ats_df[ats_df['team'] == team]
        
        # Overall ATS record
        wins = len(team_games[team_games['ats_result'] == 'W'])
        losses = len(team_games[team_games['ats_result'] == 'L'])
        pushes = len(team_games[team_games['ats_result'] == 'P'])
        total = wins + losses  # Exclude pushes from win %
        
        win_pct = wins / total if total > 0 else 0
        
        # Average ATS margin
        avg_ats_margin = team_games['ats_margin'].mean()
        
        # Home/Away splits
        home_games = team_games[team_games['location'] == 'Home']
        away_games = team_games[team_games['location'] == 'Away']
        
        home_wins = len(home_games[home_games['ats_result'] == 'W'])
        home_total = len(home_games[home_games['ats_result'] != 'P'])
        home_pct = home_wins / home_total if home_total > 0 else 0
        
        away_wins = len(away_games[away_games['ats_result'] == 'W'])
        away_total = len(away_games[away_games['ats_result'] != 'P'])
        away_pct = away_wins / away_total if away_total > 0 else 0
        
        # Favorite/Underdog splits
        fav_games = team_games[team_games['fav_dog'] == 'Favorite']
        dog_games = team_games[team_games['fav_dog'] == 'Underdog']
        
        fav_wins = len(fav_games[fav_games['ats_result'] == 'W'])
        fav_losses = len(fav_games[fav_games['ats_result'] == 'L'])
        fav_pushes = len(fav_games[fav_games['ats_result'] == 'P'])
        fav_total = fav_wins + fav_losses  # Exclude pushes from win %
        fav_pct = fav_wins / fav_total if fav_total > 0 else 0
        
        dog_wins = len(dog_games[dog_games['ats_result'] == 'W'])
        dog_losses = len(dog_games[dog_games['ats_result'] == 'L'])
        dog_pushes = len(dog_games[dog_games['ats_result'] == 'P'])
        dog_total = dog_wins + dog_losses  # Exclude pushes from win %
        dog_pct = dog_wins / dog_total if dog_total > 0 else 0
        
        # Average ATS margin as favorite/underdog (AFTER fav_games/dog_games are defined)
        avg_ats_margin_fav = fav_games['ats_margin'].mean() if len(fav_games) > 0 else 0
        avg_ats_margin_dog = dog_games['ats_margin'].mean() if len(dog_games) > 0 else 0
        
        # Average game margin (actual margin of victory/defeat)
        avg_game_margin = team_games['actual_margin'].mean()
        avg_game_margin_fav = fav_games['actual_margin'].mean() if len(fav_games) > 0 else 0
        avg_game_margin_dog = dog_games['actual_margin'].mean() if len(dog_games) > 0 else 0
        
        # Average betting line (spread)
        avg_betting_line = team_games['spread'].mean()
        avg_betting_line_fav = fav_games['spread'].mean() if len(fav_games) > 0 else 0
        avg_betting_line_dog = dog_games['spread'].mean() if len(dog_games) > 0 else 0
        
        # Actual win/loss record
        actual_wins = len(team_games[team_games['team_points'] > team_games['opp_points']])
        actual_losses = len(team_games[team_games['team_points'] < team_games['opp_points']])
        actual_win_pct = actual_wins / (actual_wins + actual_losses) if (actual_wins + actual_losses) > 0 else 0
        
        summaries.append({
            'team': team,
            'games': len(team_games),
            'ats_record': f"{wins}-{losses}-{pushes}",
            'ats_win_pct': win_pct,
            'avg_game_margin': avg_game_margin,
            'avg_game_margin_fav': avg_game_margin_fav,
            'avg_game_margin_dog': avg_game_margin_dog,
            'avg_ats_margin': avg_ats_margin,
            'avg_ats_margin_fav': avg_ats_margin_fav,
            'avg_ats_margin_dog': avg_ats_margin_dog,
            'avg_betting_line': avg_betting_line,
            'avg_betting_line_fav': avg_betting_line_fav,
            'avg_betting_line_dog': avg_betting_line_dog,
            'actual_record': f"{actual_wins}-{actual_losses}",
            'actual_win_pct': actual_win_pct,
            'home_ats_pct': home_pct,
            'away_ats_pct': away_pct,
            'as_fav_ats_pct': fav_pct,
            'as_dog_ats_pct': dog_pct,
            'as_fav_games': fav_total,
            'as_dog_games': dog_total,
            'as_fav_wins': fav_wins,
            'as_fav_losses': fav_losses,
            'as_fav_pushes': fav_pushes,
            'as_dog_wins': dog_wins,
            'as_dog_losses': dog_losses,
            'as_dog_pushes': dog_pushes
        })
    
    summary_df = pd.DataFrame(summaries)
    
    # Sort by ATS win percentage
    summary_df = summary_df.sort_values('ats_win_pct', ascending=False)
    
    return summary_df


def print_ats_rankings(summary_df, season_year):
    """
    Print formatted ATS rankings
    
    Args:
        summary_df: DataFrame with team ATS summaries
        season_year: Year season starts
    """
    season_str = get_season_string(season_year)
    
    print(f"\n{'='*180}")
    print(f"🏀 NBA {season_str} SEASON - AGAINST THE SPREAD (ATS) RANKINGS")
    print(f"{'='*180}")
    print(f"As of: {datetime.now().strftime('%Y-%m-%d %I:%M %p ET')}")
    print(f"{'='*180}\n")
    
    # Main table
    print(f"{'Rank':<6} {'Team':<25} {'W-L':<10} {'Win%':<7} {'ATS':<12} {'ATS%':<7} "
          f"{'Fav':<10} {'Fav%':<7} {'Dog':<10} {'Dog%':<7} "
          f"{'Avg Margin':<12} {'Fav Margin':<12} {'Dog Margin':<12} "
          f"{'Avg Line':<10} {'Fav Line':<10} {'Dog Line':<10}")
    print(f"{'-'*180}")
    
    for idx, row in summary_df.iterrows():
        rank = summary_df.index.get_loc(idx) + 1
        
        # Format W-L record and percentage
        wl_record = row['actual_record']
        wl_pct = f"{row['actual_win_pct']:.1%}"
        
        # Format overall ATS record and percentage
        ats_record = row['ats_record']
        ats_pct = f"{row['ats_win_pct']:.1%}"
        
        # Format ATS as favorite
        fav_wins = int(row['as_fav_ats_pct'] * row['as_fav_games'])
        fav_losses = row['as_fav_games'] - fav_wins
        ats_fav = f"{fav_wins}-{fav_losses}" if row['as_fav_games'] > 0 else "N/A"
        fav_pct = f"{row['as_fav_ats_pct']:.1%}" if row['as_fav_games'] > 0 else "N/A"
        
        # Format ATS as underdog
        dog_wins = int(row['as_dog_ats_pct'] * row['as_dog_games'])
        dog_losses = row['as_dog_games'] - dog_wins
        ats_dog = f"{dog_wins}-{dog_losses}" if row['as_dog_games'] > 0 else "N/A"
        dog_pct = f"{row['as_dog_ats_pct']:.1%}" if row['as_dog_games'] > 0 else "N/A"
        
        # Format margins and lines
        avg_margin = f"{row['avg_ats_margin']:+.1f}"
        fav_margin = f"{row['avg_ats_margin_fav']:+.1f}" if row['as_fav_games'] > 0 else "N/A"
        dog_margin = f"{row['avg_ats_margin_dog']:+.1f}" if row['as_dog_games'] > 0 else "N/A"
        
        avg_line = f"{row['avg_betting_line']:+.1f}"
        fav_line = f"{row['avg_betting_line_fav']:+.1f}" if row['as_fav_games'] > 0 else "N/A"
        dog_line = f"{row['avg_betting_line_dog']:+.1f}" if row['as_dog_games'] > 0 else "N/A"
        
        print(f"{rank:<6} {row['team']:<25} {wl_record:<10} {wl_pct:<7} {ats_record:<12} {ats_pct:<7} "
              f"{ats_fav:<10} {fav_pct:<7} {ats_dog:<10} {dog_pct:<7} "
              f"{avg_margin:<12} {fav_margin:<12} {dog_margin:<12} "
              f"{avg_line:<10} {fav_line:<10} {dog_line:<10}")
    
    print(f"\n{'='*180}")
    print(f"💎 BEST BETTING VALUE (Good ATS despite losing record)")
    print(f"{'='*180}")
    
    # Teams with losing record but good ATS
    value_teams = summary_df[
        (summary_df['actual_win_pct'] < 0.500) &  # Losing record
        (summary_df['ats_win_pct'] > 0.500) &     # Winning ATS
        (summary_df['games'] >= 10)                # Minimum sample size
    ].head(5)
    
    if not value_teams.empty:
        for idx, row in value_teams.iterrows():
            rank = summary_df.index.get_loc(idx) + 1
            print(f"\n#{rank} {row['team']}")
            print(f"   Record: {row['actual_record']} ({row['actual_win_pct']:.1%}) | ATS: {row['ats_record']} ({row['ats_win_pct']:.1%})")
            print(f"   → Market undervalues by {(row['ats_win_pct'] - row['actual_win_pct']) * 100:.1f} percentage points")
    else:
        print("\nNo teams found with losing record but winning ATS record")



def save_results(summary_df, ats_df, season_year):
    """
    Save ATS analysis results to CSV
    
    Args:
        summary_df: Team-level ATS summaries
        ats_df: Game-level ATS results
        season_year: Year season starts
    """
    season_str = get_season_string(season_year).replace('-', '_')
    output_dir = Path(__file__).parent.parent / 'data' / '04_output' / 'nba_ats_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    date_str = datetime.now().strftime('%Y%m%d')
    
    # Save team summaries
    summary_path = output_dir / f'nba_ats_rankings_{season_str}_{date_str}.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n💾 Saved team ATS summaries: {summary_path}")
    
    # Save game-level results
    games_path = output_dir / f'nba_ats_game_results_{season_str}_{date_str}.csv'
    ats_df.to_csv(games_path, index=False)
    print(f"💾 Saved game-level ATS results: {games_path}")



def get_team_logos():
    """Get NBA team logos from ESPN"""
    logo_map = {
        'Atlanta Hawks': 'https://a.espncdn.com/i/teamlogos/nba/500/atl.png',
        'Boston Celtics': 'https://a.espncdn.com/i/teamlogos/nba/500/bos.png',
        'Brooklyn Nets': 'https://a.espncdn.com/i/teamlogos/nba/500/bkn.png',
        'Charlotte Hornets': 'https://a.espncdn.com/i/teamlogos/nba/500/cha.png',
        'Chicago Bulls': 'https://a.espncdn.com/i/teamlogos/nba/500/chi.png',
        'Cleveland Cavaliers': 'https://a.espncdn.com/i/teamlogos/nba/500/cle.png',
        'Dallas Mavericks': 'https://a.espncdn.com/i/teamlogos/nba/500/dal.png',
        'Denver Nuggets': 'https://a.espncdn.com/i/teamlogos/nba/500/den.png',
        'Detroit Pistons': 'https://a.espncdn.com/i/teamlogos/nba/500/det.png',
        'Golden State Warriors': 'https://a.espncdn.com/i/teamlogos/nba/500/gs.png',
        'Houston Rockets': 'https://a.espncdn.com/i/teamlogos/nba/500/hou.png',
        'Indiana Pacers': 'https://a.espncdn.com/i/teamlogos/nba/500/ind.png',
        'LA Clippers': 'https://a.espncdn.com/i/teamlogos/nba/500/lac.png',
        'Los Angeles Clippers': 'https://a.espncdn.com/i/teamlogos/nba/500/lac.png',
        'Los Angeles Lakers': 'https://a.espncdn.com/i/teamlogos/nba/500/lal.png',
        'Memphis Grizzlies': 'https://a.espncdn.com/i/teamlogos/nba/500/mem.png',
        'Miami Heat': 'https://a.espncdn.com/i/teamlogos/nba/500/mia.png',
        'Milwaukee Bucks': 'https://a.espncdn.com/i/teamlogos/nba/500/mil.png',
        'Minnesota Timberwolves': 'https://a.espncdn.com/i/teamlogos/nba/500/min.png',
        'New Orleans Pelicans': 'https://a.espncdn.com/i/teamlogos/nba/500/no.png',
        'New York Knicks': 'https://a.espncdn.com/i/teamlogos/nba/500/ny.png',
        'Oklahoma City Thunder': 'https://a.espncdn.com/i/teamlogos/nba/500/okc.png',
        'Orlando Magic': 'https://a.espncdn.com/i/teamlogos/nba/500/orl.png',
        'Philadelphia 76ers': 'https://a.espncdn.com/i/teamlogos/nba/500/phi.png',
        'Phoenix Suns': 'https://a.espncdn.com/i/teamlogos/nba/500/phx.png',
        'Portland Trail Blazers': 'https://a.espncdn.com/i/teamlogos/nba/500/por.png',
        'Sacramento Kings': 'https://a.espncdn.com/i/teamlogos/nba/500/sac.png',
        'San Antonio Spurs': 'https://a.espncdn.com/i/teamlogos/nba/500/sa.png',
        'Toronto Raptors': 'https://a.espncdn.com/i/teamlogos/nba/500/tor.png',
        'Utah Jazz': 'https://a.espncdn.com/i/teamlogos/nba/500/utah.png',
        'Washington Wizards': 'https://a.espncdn.com/i/teamlogos/nba/500/wsh.png',
    }
    return logo_map


def create_ats_visualization(summary_df, season_year):
    """
    Create publication-quality ATS table using R's gt package
    
    Args:
        summary_df: DataFrame with team ATS summaries
        season_year: Year season starts
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        import subprocess
        import platform
        
        print("   ✅ rpy2 loaded successfully\n")
        
    except ImportError as e:
        print(f"❌ Error: rpy2 not installed")
        print(f"   Install: pip install rpy2")
        print(f"   Also ensure R is installed: brew install r")
        return
    
    season_str = get_season_string(season_year)
    logo_map = get_team_logos()
    
    # Prepare display dataframe
    df_viz = summary_df.copy()
    df_viz['logo_url'] = df_viz['team'].map(logo_map)
    
    # Assign ranks with tie handling (teams with same ATS% get same rank)
    df_viz['rank'] = df_viz['ats_win_pct'].rank(method='min', ascending=False).astype(int)
    
    # Calculate numeric values for gradients (percentages as 0-100)
    df_viz['win_pct_num'] = (df_viz['actual_win_pct'] * 100).round(1)
    df_viz['ats_pct_num'] = (df_viz['ats_win_pct'] * 100).round(1)
    df_viz['fav_pct_num'] = df_viz.apply(
        lambda row: round(row['as_fav_ats_pct'] * 100, 1) if row['as_fav_games'] > 0 else None,
        axis=1
    )
    df_viz['dog_pct_num'] = df_viz.apply(
        lambda row: round(row['as_dog_ats_pct'] * 100, 1) if row['as_dog_games'] > 0 else None,
        axis=1
    )
    
    # Keep numeric values for game margins and ATS margins (already in correct range)
    df_viz['avg_game_margin_num'] = df_viz['avg_game_margin'].round(1)
    df_viz['fav_game_margin_num'] = df_viz.apply(
        lambda row: round(row['avg_game_margin_fav'], 1) if row['as_fav_games'] > 0 else None,
        axis=1
    )
    df_viz['dog_game_margin_num'] = df_viz.apply(
        lambda row: round(row['avg_game_margin_dog'], 1) if row['as_dog_games'] > 0 else None,
        axis=1
    )
    
    df_viz['avg_ats_margin_num'] = df_viz['avg_ats_margin'].round(1)
    df_viz['fav_ats_margin_num'] = df_viz.apply(
        lambda row: round(row['avg_ats_margin_fav'], 1) if row['as_fav_games'] > 0 else None,
        axis=1
    )
    df_viz['dog_ats_margin_num'] = df_viz.apply(
        lambda row: round(row['avg_ats_margin_dog'], 1) if row['as_dog_games'] > 0 else None,
        axis=1
    )
    df_viz['avg_line_num'] = df_viz['avg_betting_line'].round(1)
    df_viz['fav_line_num'] = df_viz.apply(
        lambda row: round(row['avg_betting_line_fav'], 1) if row['as_fav_games'] > 0 else None,
        axis=1
    )
    df_viz['dog_line_num'] = df_viz.apply(
        lambda row: round(row['avg_betting_line_dog'], 1) if row['as_dog_games'] > 0 else None,
        axis=1
    )
    
    # Format Fav/Dog records (W-L-P format like main ATS)
    df_viz['fav_record'] = df_viz.apply(
        lambda row: f"{row['as_fav_wins']}-{row['as_fav_losses']}-{row['as_fav_pushes']}" if row['as_fav_games'] > 0 else 'N/A',
        axis=1
    )
    df_viz['dog_record'] = df_viz.apply(
        lambda row: f"{row['as_dog_wins']}-{row['as_dog_losses']}-{row['as_dog_pushes']}" if row['as_dog_games'] > 0 else 'N/A',
        axis=1
    )
    
    # Select columns for visualization (using numeric columns for gradient coloring)
    # Order: Rank, Team, W-L, Win%, then 3 sections (Avg/Fav/Dog) each with ATS record, ATS%, Line, Game Margin, ATS Margin
    table_df = df_viz[[
        'rank', 'team', 'logo_url', 'actual_record', 'win_pct_num',
        'ats_record', 'ats_pct_num', 'avg_line_num', 'avg_game_margin_num', 'avg_ats_margin_num',
        'fav_record', 'fav_pct_num', 'fav_line_num', 'fav_game_margin_num', 'fav_ats_margin_num',
        'dog_record', 'dog_pct_num', 'dog_line_num', 'dog_game_margin_num', 'dog_ats_margin_num'
    ]].copy()
    
    # Rename for display
    table_df.columns = [
        'Rank', 'Team', 'logo_url', 'W-L', 'Win%',
        'ATS', 'ATS%', 'Avg Spread', 'Avg Game Margin', 'Avg Spread Margin',
        'Fav ATS', 'Fav ATS%', 'Avg Fav Spread', 'Avg Fav Game Margin', 'Avg Fav Spread Margin',
        'Dog ATS', 'Dog ATS%', 'Avg Dog Spread', 'Avg Dog Game Margin', 'Avg Dog Spread Margin'
    ]
    
    # Convert to R
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(table_df)
    
    ro.globalenv['ats_data'] = r_df
    
    # Output path
    output_dir = Path(__file__).parent.parent / 'content/viz/nba'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'nba_ats_rankings_{season_str.replace("-", "_")}.png'
    
    print(f"   💾 Output: {output_path.name}\n")
    
    # R code for visualization
    r_code = f'''
    .libPaths(c("~/R/library", .libPaths()))
    
    library(gt)
    library(gtExtras)
    library(dplyr)
    
    table <- ats_data %>%
      gt() %>%
      
      # Add team logos
      gt_img_rows(columns = logo_url, height = 20) %>%
      
      # Format percentage columns (add % symbol)
      fmt_number(
        columns = c(`Win%`, `ATS%`, `Fav ATS%`, `Dog ATS%`),
        decimals = 1,
        pattern = "{{x}}%"
      ) %>%
      
      # Format margin columns (add +/- and 1 decimal)
      fmt_number(
        columns = c(`Avg Game Margin`, `Avg Fav Game Margin`, `Avg Dog Game Margin`, 
                    `Avg Spread Margin`, `Avg Fav Spread Margin`, `Avg Dog Spread Margin`),
        decimals = 1,
        force_sign = TRUE
      ) %>%
      
      # Format line columns (add +/- and 1 decimal)
      fmt_number(
        columns = c(`Avg Spread`, `Avg Fav Spread`, `Avg Dog Spread`),
        decimals = 1,
        force_sign = TRUE
      ) %>%
      
      # Title
      tab_header(
        title = md("**NBA {season_str} Against The Spread (ATS) Rankings**"),
        subtitle = md("**Game Margin** = avg margin of victory/defeat | **Spread Margin** = avg pts beat/miss spread by")
      ) %>%
      
      # Column alignment
      cols_align(align = "center", columns = everything()) %>%
      cols_align(align = "left", columns = c(Team)) %>%
      
      # Hide logo column header
      cols_label(logo_url = "") %>%
      
      # Color gradients for percentage columns (0% red -> 100% green)
      data_color(
        columns = `Win%`,
        method = "numeric",
        palette = c("#d62728", "#ff9999", "#ffffff", "#90EE90", "#00b300"),
        domain = c(0, 100),
        na_color = "#e8e8e8"
      ) %>%
      
      data_color(
        columns = `ATS%`,
        method = "numeric",
        palette = c("#d62728", "#ff9999", "#ffffff", "#90EE90", "#00b300"),
        domain = c(0, 100),
        na_color = "#e8e8e8"
      ) %>%
      
      data_color(
        columns = `Fav ATS%`,
        method = "numeric",
        palette = c("#d62728", "#ff9999", "#ffffff", "#90EE90", "#00b300"),
        domain = c(0, 100),
        na_color = "#e8e8e8"
      ) %>%
      
      data_color(
        columns = `Dog ATS%`,
        method = "numeric",
        palette = c("#d62728", "#ff9999", "#ffffff", "#90EE90", "#00b300"),
        domain = c(0, 100),
        na_color = "#e8e8e8"
      ) %>%
      
      # Color gradients for game margin columns (-15 red -> +15 green)
      data_color(
        columns = `Avg Game Margin`,
        method = "numeric",
        palette = c("#d62728", "#ff9999", "#ffffff", "#90EE90", "#00b300"),
        domain = c(-15, 15),
        na_color = "#e8e8e8"
      ) %>%
      
      data_color(
        columns = `Avg Fav Game Margin`,
        method = "numeric",
        palette = c("#d62728", "#ff9999", "#ffffff", "#90EE90", "#00b300"),
        domain = c(-15, 15),
        na_color = "#e8e8e8"
      ) %>%
      
      data_color(
        columns = `Avg Dog Game Margin`,
        method = "numeric",
        palette = c("#d62728", "#ff9999", "#ffffff", "#90EE90", "#00b300"),
        domain = c(-15, 15),
        na_color = "#e8e8e8"
      ) %>%
      
      # Color gradients for ATS margin columns (-15 red -> +15 green)
      data_color(
        columns = `Avg Spread Margin`,
        method = "numeric",
        palette = c("#d62728", "#ff9999", "#ffffff", "#90EE90", "#00b300"),
        domain = c(-15, 15),
        na_color = "#e8e8e8"
      ) %>%
      
      data_color(
        columns = `Avg Fav Spread Margin`,
        method = "numeric",
        palette = c("#d62728", "#ff9999", "#ffffff", "#90EE90", "#00b300"),
        domain = c(-15, 15),
        na_color = "#e8e8e8"
      ) %>%
      
      data_color(
        columns = `Avg Dog Spread Margin`,
        method = "numeric",
        palette = c("#d62728", "#ff9999", "#ffffff", "#90EE90", "#00b300"),
        domain = c(-15, 15),
        na_color = "#e8e8e8"
      ) %>%
      
      # NO gradient for line columns - they're descriptive, not evaluative
      # (Being a -12 favorite doesn't mean "bad", it just means strong team)
      
      # Column widths (make narrower to fit all columns)
      cols_width(
        Rank ~ px(50),
        Team ~ px(140),
        logo_url ~ px(45),
        `W-L` ~ px(65),
        `Win%` ~ px(55),
        ATS ~ px(75),
        `ATS%` ~ px(55),
        `Avg Spread` ~ px(70),
        `Avg Game Margin` ~ px(85),
        `Avg Spread Margin` ~ px(90),
        `Fav ATS` ~ px(65),
        `Fav ATS%` ~ px(55),
        `Avg Fav Spread` ~ px(75),
        `Avg Fav Game Margin` ~ px(90),
        `Avg Fav Spread Margin` ~ px(95),
        `Dog ATS` ~ px(65),
        `Dog ATS%` ~ px(55),
        `Avg Dog Spread` ~ px(75),
        `Avg Dog Game Margin` ~ px(90),
        `Avg Dog Spread Margin` ~ px(95)
      ) %>%
      
      # Add vertical dividers between sections
      # After Win% (separate actual record from avg betting stats)
      tab_style(
        style = cell_borders(
          sides = "right",
          color = "#2c3e50",
          weight = px(2)
        ),
        locations = list(
          cells_body(columns = `Win%`),
          cells_column_labels(columns = `Win%`)
        )
      ) %>%
      
      # After Avg Spread Margin (separate avg section from fav section)
      tab_style(
        style = cell_borders(
          sides = "right",
          color = "#2c3e50",
          weight = px(2)
        ),
        locations = list(
          cells_body(columns = `Avg Spread Margin`),
          cells_column_labels(columns = `Avg Spread Margin`)
        )
      ) %>%
      
      # After Avg Fav Spread Margin (separate fav section from dog section)  
      tab_style(
        style = cell_borders(
          sides = "right",
          color = "#2c3e50",
          weight = px(2)
        ),
        locations = list(
          cells_body(columns = `Avg Fav Spread Margin`),
          cells_column_labels(columns = `Avg Fav Spread Margin`)
        )
      ) %>%
      
      # Style headers
      tab_style(
        style = list(
          cell_text(weight = "bold", size = px(11)),
          cell_fill(color = "#e8e8e8")
        ),
        locations = cells_column_labels(everything())
      ) %>%
      
      # Bold team names and rank
      tab_style(
        style = cell_text(weight = "600", size = px(11)),
        locations = cells_body(columns = c(Rank, Team))
      ) %>%
      
      # Smaller body text to fit all columns
      tab_style(
        style = cell_text(size = px(10)),
        locations = cells_body(columns = everything())
      ) %>%
      
      # Zebra striping
      opt_row_striping(row_striping = TRUE) %>%
      
      # Table options
      tab_options(
        table.font.names = "Arial",
        table.font.size = px(10),
        heading.title.font.size = px(22),
        heading.subtitle.font.size = px(13),
        heading.padding = px(4),
        column_labels.padding = px(2),
        data_row.padding = px(1),
        table.border.bottom.width = px(2),
        table.border.bottom.color = "#2c3e50",
        column_labels.border.bottom.width = px(2),
        column_labels.border.bottom.color = "#2c3e50",
        table.background.color = "#f8f9fa",
        row.striping.background_color = "#f0f0f0"
      ) %>%
      
      # Footer
      tab_source_note(
        source_note = md("**Data:** DraftKings (The Odds API) & nba_api | {datetime.now().strftime('%B %d, %Y')}")
      )
    
    # Save (wider to fit all columns including game margin)
    gtsave(table, "{str(output_path)}", vwidth = 2400, vheight = 2500)
    '''
    
    print("   🔧 Executing R code...\n")
    
    try:
        ro.r(r_code)
        print(f"   ✅ Visualization saved!\n")
        print(f"   🖼️  {output_path}\n")
        
        # Auto-open
        try:
            if platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', str(output_path)])
                print("   📂 Opening PNG...\n")
        except:
            pass
            
    except Exception as e:
        print(f"❌ Error creating visualization:")
        print(f"   {e}")
        print("\n💡 Install R packages:")
        print("   R -e 'install.packages(c(\"gt\", \"gtExtras\", \"dplyr\", \"webshot2\"))'")


def main():
    """Main analysis pipeline"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Analyze NBA team performance against the spread (ATS)'
    )
    parser.add_argument(
        '--season',
        type=int,
        default=2025,
        help='Season start year (e.g., 2025 for 2025-26 season). Default: 2025'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Create R visualization (gt table) of ATS rankings'
    )
    
    args = parser.parse_args()
    season_year = args.season
    season_str = get_season_string(season_year)
    
    print(f"\n{'='*100}")
    print(f"🏀 LOADING NBA {season_str} BETTING LINES AND RESULTS")
    print(f"{'='*100}")
    
    # Load betting lines from S3
    print("\n📥 Loading betting lines from S3...")
    lines_df = load_all_game_lines(season_year)
    print(f"✅ Loaded {len(lines_df):,} betting lines from {lines_df['game_id'].nunique()} games")
    
    # Load game results
    print("\n📥 Loading game results from nba-api...")
    scores_df = get_nba_scores(season_year)
    print(f"✅ Loaded results for {len(scores_df):,} team-games")
    
    # Calculate ATS records
    print("\n📊 Calculating ATS records...")
    ats_df = calculate_ats_records(lines_df, scores_df)
    print(f"✅ Calculated ATS results for {len(ats_df):,} team-games")
    
    # Summarize by team
    print("\n📈 Summarizing by team...")
    summary_df = summarize_ats_records(ats_df)
    
    # Print rankings
    print_ats_rankings(summary_df, season_year)
    
    # Save results
    save_results(summary_df, ats_df, season_year)
    
    # Create visualization if requested
    if args.plot:
        print(f"\n{'='*100}")
        print(f"🎨 CREATING R VISUALIZATION")
        print(f"{'='*100}\n")
        create_ats_visualization(summary_df, season_year)
    
    print(f"\n{'='*100}")
    print(f"✅ ANALYSIS COMPLETE")
    print(f"{'='*100}\n")



if __name__ == '__main__':
    main()


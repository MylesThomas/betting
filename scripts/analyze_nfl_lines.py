"""
Analyze ALL NFL betting lines data from historical directory

Loads all CSV files and analyzes the complete dataset

Usage:
    python analyze_nfl_lines.py              # Standard analysis
    python analyze_nfl_lines.py --debug      # Include Browns debugging section
"""

import pandas as pd
from pathlib import Path
import glob
import argparse

# 2025 NFL Bye Weeks Schedule
BYE_WEEKS = {
    5: ['Atlanta Falcons', 'Chicago Bears', 'Green Bay Packers', 'Pittsburgh Steelers'],
    6: ['Houston Texans', 'Minnesota Vikings'],
    7: ['Baltimore Ravens', 'Buffalo Bills'],
    8: ['Arizona Cardinals', 'Detroit Lions', 'Jacksonville Jaguars', 'Las Vegas Raiders', 
        'Los Angeles Rams', 'Seattle Seahawks'],
    9: ['Cleveland Browns', 'New York Jets', 'Philadelphia Eagles', 'Tampa Bay Buccaneers'],
    10: ['Cincinnati Bengals', 'Dallas Cowboys', 'Kansas City Chiefs', 'Tennessee Titans'],
    11: ['Indianapolis Colts', 'New Orleans Saints'],
    12: ['Denver Broncos', 'Los Angeles Chargers', 'Miami Dolphins', 'Washington Commanders'],
    13: [],
    14: ['Carolina Panthers', 'New England Patriots', 'New York Giants', 'San Francisco 49ers']
}

# Create team -> bye week mapping
TEAM_BYE_WEEK = {}
for week, teams in BYE_WEEKS.items():
    for team in teams:
        TEAM_BYE_WEEK[team] = week

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Analyze NFL betting lines')
parser.add_argument('--debug', action='store_true', 
                   help='Show debugging sections (Browns games, specific dates)')
args = parser.parse_args()

# Directory with all historical data
data_dir = 'data/01_input/the-odds-api/nfl/game_lines/historical'

print("="*80)
print("NFL BETTING LINES ANALYSIS - FULL SEASON")
print("="*80)

# Find all CSV files (daily files + London games)
csv_files = glob.glob(f"{data_dir}/nfl_game_lines_*.csv")
london_file = f"{data_dir}/2025_game_lines_london.csv"

# Add London games file if it exists
if Path(london_file).exists():
    csv_files.append(london_file)

csv_files = sorted(csv_files)

print(f"\nFound {len(csv_files)} CSV files")
if csv_files:
    print(f"First file: {Path(csv_files[0]).name}")
    print(f"Last file: {Path(csv_files[-1]).name}")
    if Path(london_file).exists():
        print(f"Includes: 2025_game_lines_london.csv (London games)")

# Load all files
dfs = []
for csv_file in csv_files:
    df_temp = pd.read_csv(csv_file)
    dfs.append(df_temp)
    
df_all = pd.concat(dfs, ignore_index=True)

print(f"\nTotal rows loaded: {len(df_all):,}")

# Convert game_time to datetime and then convert to ET (America/New_York)
# Note: datetime is already timezone-aware from CSV
df_all['game_time'] = pd.to_datetime(df_all['game_time']).dt.tz_convert('America/New_York')

# Filter to 2025 season only (after Sept 1, 2025)
# Make timezone-aware in ET to match game_time
print(f"Rows before filtering to 2025 season (>= 2025-09-01): {len(df_all):,}")
season_start = pd.to_datetime('2025-09-01').tz_localize('America/New_York')
df_all = df_all[df_all['game_time'] >= season_start].copy()


print(f"Rows after filtering to 2025 season (>= 2025-09-01): {len(df_all):,}")

# Get date range
date_range_start = df_all['game_time'].min().date()
date_range_end = df_all['game_time'].max().date()

print(f"Date range: {date_range_start} to {date_range_end}")

# Basic stats
print(f"\n{'='*80}")
print("BASIC STATS")
print(f"{'='*80}")
print(f"Unique games: {df_all['game_id'].nunique()}")
print(f"Unique bookmakers: {df_all['bookmaker'].nunique()}")
print(f"Markets: {df_all['market'].unique().tolist()}")

# Bookmaker list
print(f"\nBookmakers:")
for book in sorted(df_all['bookmaker'].unique()):
    count = len(df_all[df_all['bookmaker'] == book])
    print(f"  • {book:<20s} ({count:,} lines)")

# Games per date
print(f"\n{'='*80}")
print("GAMES PER DATE")
print(f"{'='*80}")

df_all['game_date'] = df_all['game_time'].dt.date
games_per_date = df_all.groupby('game_date')['game_id'].nunique().sort_index()

print(f"\n{'Date':<15s} {'Games':<10s}")
print("-" * 25)
for date, num_games in games_per_date.items():
    day_name = pd.to_datetime(date).strftime('%A')[:3]
    print(f"{str(date):<15s} {num_games:<4d} ({day_name})")

# Games per team
print(f"\n{'='*80}")
print("GAMES PER TEAM")
print(f"{'='*80}\n")

# Count away and home games
away_games = df_all.groupby('away_team')['game_id'].nunique()
home_games = df_all.groupby('home_team')['game_id'].nunique()

# Combine and sum
all_teams = pd.concat([away_games, home_games]).groupby(level=0).sum().sort_values(ascending=False)

print(f"{'Team':<30s} {'Games':<10s}")
print("-" * 40)
for team, games in all_teams.items():
    print(f"{team:<30s} {games:<10d}")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Total teams: {len(all_teams)}")
print(f"Total game dates: {len(games_per_date)}")
print(f"Total unique games: {df_all['game_id'].nunique()}")
print(f"Min games per team: {all_teams.min()}")
print(f"Max games per team: {all_teams.max()}")
print(f"Average games per team: {all_teams.mean():.1f}")

# Coverage check
expected_games_per_team = len(games_per_date) // 2  # Rough estimate
coverage_pct = (all_teams.min() / expected_games_per_team * 100) if expected_games_per_team > 0 else 0

print(f"\nSeason coverage:")
print(f"  Dates with games: {len(games_per_date)}")
print(f"  Weeks covered: ~{len(games_per_date) / 3:.1f} weeks")
print(f"  Min games/team: {all_teams.min()} ({coverage_pct:.0f}% of expected)")

# Data quality checks
print(f"\n{'='*80}")
print("DATA QUALITY")
print(f"{'='*80}")

# Check for missing spreads
missing_spreads = df_all['away_spread'].isna().sum()
print(f"Missing away_spread: {missing_spreads} ({missing_spreads/len(df_all)*100:.1f}%)")

# Check for missing odds
missing_odds = df_all['away_odds'].isna().sum()
print(f"Missing away_odds: {missing_odds} ({missing_odds/len(df_all)*100:.1f}%)")

# Lines per game
lines_per_game = df_all.groupby('game_id').size()
print(f"\nLines per game:")
print(f"  Mean: {lines_per_game.mean():.1f}")
print(f"  Min: {lines_per_game.min()}")
print(f"  Max: {lines_per_game.max()}")

print(f"\n{'='*80}")
print("BYE WEEK VALIDATION")
print(f"{'='*80}")

# Determine current week based on latest game date (Nov 24 = Week 12)
CURRENT_WEEK = 12

print(f"Data through Week {CURRENT_WEEK} (Nov 24, 2025)")
print(f"\nExpected games by team:")
print(f"  • Bye weeks 5-{CURRENT_WEEK}: 11 games (12 weeks - 1 bye)")
print(f"  • Bye weeks {CURRENT_WEEK+1}+: 12 games (no bye yet)\n")

# Check each team
mismatches = []
for team, games in all_teams.items():
    bye_week = TEAM_BYE_WEEK.get(team, None)
    
    if bye_week is None:
        expected = 12  # No bye week listed
        status = "⚠️  No bye week in schedule"
    elif bye_week <= CURRENT_WEEK:
        expected = 11  # Had bye already
        status = f"Bye Week {bye_week}"
    else:
        expected = 12  # Bye week in future
        status = f"Bye Week {bye_week} (future)"
    
    actual = int(games)
    
    if actual != expected:
        mismatches.append({
            'team': team,
            'expected': expected,
            'actual': actual,
            'bye_week': bye_week,
            'status': status
        })

if mismatches:
    print(f"❌ MISMATCHES FOUND: {len(mismatches)}")
    print(f"\n{'Team':<35s} {'Expected':<10s} {'Actual':<10s} {'Status':<20s}")
    print("-" * 75)
    for m in mismatches:
        print(f"{m['team']:<35s} {m['expected']:<10d} {m['actual']:<10d} {m['status']:<20s}")
else:
    teams_11_games = sorted([t for t, bw in TEAM_BYE_WEEK.items() if bw <= CURRENT_WEEK])
    teams_12_games = sorted([t for t, bw in TEAM_BYE_WEEK.items() if bw > CURRENT_WEEK])
    
    print(f"✅ ALL TEAMS HAVE CORRECT NUMBER OF GAMES")
    print(f"   • {len(teams_11_games)} teams with 11 games (bye completed)")
    print(f"   • {len(teams_12_games)} teams with 12 games (bye upcoming)")
    
    if args.debug:
        print(f"\n  Teams with 11 games (bye weeks 5-12):")
        for team in teams_11_games:
            bye_week = TEAM_BYE_WEEK[team]
            print(f"    • {team:<35s} (Bye Week {bye_week})")
        
        print(f"\n  Teams with 12 games (bye week 13+):")
        for team in teams_12_games:
            bye_week = TEAM_BYE_WEEK[team]
            print(f"    • {team:<35s} (Bye Week {bye_week})")

print(f"\n{'='*80}")
print("✅ ANALYSIS COMPLETE")
print(f"{'='*80}")

# DEBUG SECTION (only shown with --debug flag)
if args.debug:
    print(f"\n{'='*80}")
    print("DEBUG: CLEVELAND BROWNS GAMES")
    print(f"{'='*80}\n")

    browns_games = df_all[(df_all['away_team'] == 'Cleveland Browns') | (df_all['home_team'] == 'Cleveland Browns')].copy()
    browns_games_unique = browns_games.drop_duplicates(subset=['game_id'])[['game_time', 'away_team', 'home_team']].sort_values('game_time')

    print(f"{'Date':<20s} {'Away Team':<30s} {'Home Team':<30s}")
    print("-" * 80)
    for _, row in browns_games_unique.iterrows():
        game_date = row['game_time'].strftime('%Y-%m-%d (%a)')
        print(f"{game_date:<20s} {row['away_team']:<30s} {row['home_team']:<30s}")

    # Check Oct 5 and Nov 2 games
    print(f"\n{'='*80}")
    print("DEBUG: GAMES ON OCT 5, 2025")
    print(f"{'='*80}\n")

    oct5_games = df_all[df_all['game_time'].dt.date == pd.to_datetime('2025-10-05').date()].drop_duplicates(subset=['game_id'])[['game_time', 'away_team', 'home_team']].sort_values('game_time')
    for _, row in oct5_games.iterrows():
        print(f"{row['away_team']:<30s} @ {row['home_team']:<30s}")

    print(f"\n{'='*80}")
    print("DEBUG: GAMES ON NOV 2, 2025")
    print(f"{'='*80}\n")

    nov2_games = df_all[df_all['game_time'].dt.date == pd.to_datetime('2025-11-02').date()].drop_duplicates(subset=['game_id'])[['game_time', 'away_team', 'home_team']].sort_values('game_time')
    for _, row in nov2_games.iterrows():
        print(f"{row['away_team']:<30s} @ {row['home_team']:<30s}")

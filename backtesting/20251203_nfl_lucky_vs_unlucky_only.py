"""
NFL Model: Only Lucky vs Unlucky Matchups

Test if regression effect shows up specifically when:
- One team was Lucky last week
- Other team was Unlucky last week
- Bet the Unlucky team

This matches the find_nfl_luck_regression_plays_both_teams.py strategy exactly.

Usage:
    python backtesting/20251203_nfl_lucky_vs_unlucky_only.py
    python backtesting/20251203_nfl_lucky_vs_unlucky_only.py --threshold 5
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
from datetime import datetime

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from config import NFL_LUCK_THRESHOLD_DEFAULT

parser = argparse.ArgumentParser()
parser.add_argument('--threshold', type=float, default=NFL_LUCK_THRESHOLD_DEFAULT,
                    help=f'Luck threshold (default: {NFL_LUCK_THRESHOLD_DEFAULT})')
args = parser.parse_args()

threshold = args.threshold

print("=" * 120)
print("NFL: LUCKY vs UNLUCKY MATCHUPS ONLY")
print("=" * 120)
print("")
print(f"Strategy: When LUCKY team plays UNLUCKY team → Bet UNLUCKY team")
print(f"Threshold: ±{threshold}")
print("")


def categorize_luck(luck_value, threshold):
    """Categorize luck into Lucky/Neutral/Unlucky."""
    if luck_value is None or pd.isna(luck_value):
        return 'Unknown'
    if luck_value >= threshold:
        return 'Lucky'
    elif luck_value <= -threshold:
        return 'Unlucky'
    else:
        return 'Neutral'


# Load data
intermediate_dir = PROJECT_ROOT / 'data' / '03_intermediate'
data_path = intermediate_dir / "nfl_games_with_spreads_and_results.csv"

df_games = pd.read_csv(data_path)
df_games = df_games.sort_values(['week', 'game_time']).reset_index(drop=True)

print(f"Loaded {len(df_games)} games")

# First, calculate prior week luck for each team in each game
# We need to join: for each game, get each team's luck from PRIOR week

team_game_rows = []
for idx, game in df_games.iterrows():
    team_game_rows.append({
        'game_id': game['game_id'], 'week': game['week'],
        'team': game['away_abbr'], 'is_home': False,
        'spread': game['consensus_spread'], 'actual_margin': game['actual_margin'],
        'covered': game['away_covered'],
        'team_adj_score': game['away_adj_score'], 'opp_adj_score': game['home_adj_score'],
    })
    team_game_rows.append({
        'game_id': game['game_id'], 'week': game['week'],
        'team': game['home_abbr'], 'is_home': True,
        'spread': -game['consensus_spread'], 'actual_margin': -game['actual_margin'],
        'covered': game['home_covered'],
        'team_adj_score': game['home_adj_score'], 'opp_adj_score': game['away_adj_score'],
    })

df_team_games = pd.DataFrame(team_game_rows)

# For each team-game, calculate their luck in that game
df_team_games['adj_margin'] = df_team_games['team_adj_score'] - df_team_games['opp_adj_score']
df_team_games['luck'] = df_team_games['actual_margin'] - df_team_games['adj_margin']

# Create a lookup: for each team and week, what was their luck?
luck_lookup = {}
for _, row in df_team_games.iterrows():
    key = (row['team'], row['week'])
    luck_lookup[key] = row['luck']

# Now for each game, get PRIOR week luck for both teams
games_with_prior_luck = []

for _, game in df_games.iterrows():
    week = game['week']
    away = game['away_abbr']
    home = game['home_abbr']
    
    # Get prior week luck (skip week 1)
    if week == 1:
        continue
    
    # Find prior week for each team (handle bye weeks)
    away_prior_luck = None
    home_prior_luck = None
    
    # Look back from current week to find most recent game
    for prior_week in range(week - 1, 0, -1):
        if away_prior_luck is None and (away, prior_week) in luck_lookup:
            away_prior_luck = luck_lookup[(away, prior_week)]
        if home_prior_luck is None and (home, prior_week) in luck_lookup:
            home_prior_luck = luck_lookup[(home, prior_week)]
        if away_prior_luck is not None and home_prior_luck is not None:
            break
    
    if away_prior_luck is None or home_prior_luck is None:
        continue  # Skip if missing prior data
    
    away_luck_cat = categorize_luck(away_prior_luck, threshold)
    home_luck_cat = categorize_luck(home_prior_luck, threshold)
    
    games_with_prior_luck.append({
        'game_id': game['game_id'],
        'week': week,
        'away': away,
        'home': home,
        'spread': game['consensus_spread'],  # Away team spread
        'actual_margin': game['actual_margin'],  # Away - Home
        'away_covered': game['away_covered'],
        'home_covered': game['home_covered'],
        'away_prior_luck': away_prior_luck,
        'home_prior_luck': home_prior_luck,
        'away_luck_cat': away_luck_cat,
        'home_luck_cat': home_luck_cat,
    })

df_analysis = pd.DataFrame(games_with_prior_luck)

# Filter to Lucky vs Unlucky matchups only
df_lu = df_analysis[
    ((df_analysis['away_luck_cat'] == 'Lucky') & (df_analysis['home_luck_cat'] == 'Unlucky')) |
    ((df_analysis['away_luck_cat'] == 'Unlucky') & (df_analysis['home_luck_cat'] == 'Lucky'))
].copy()

print(f"\nTotal games with prior luck data: {len(df_analysis)}")
print(f"Lucky vs Unlucky matchups: {len(df_lu)}")
print("")

if len(df_lu) == 0:
    print("No Lucky vs Unlucky matchups found!")
    sys.exit(0)

# Strategy: Bet the UNLUCKY team
# Determine which team is unlucky and if they covered
results = []

for _, row in df_lu.iterrows():
    if row['away_luck_cat'] == 'Unlucky':
        # Bet away team
        bet_team = row['away']
        bet_spread = row['spread']
        bet_covered = row['away_covered']
        opp_team = row['home']
        opp_luck = row['home_prior_luck']
        bet_luck = row['away_prior_luck']
    else:
        # Bet home team
        bet_team = row['home']
        bet_spread = -row['spread']
        bet_covered = row['home_covered']
        opp_team = row['away']
        opp_luck = row['away_prior_luck']
        bet_luck = row['home_prior_luck']
    
    results.append({
        'week': row['week'],
        'bet_team': bet_team,
        'bet_spread': bet_spread,
        'opp_team': opp_team,
        'bet_luck': bet_luck,  # Unlucky (negative)
        'opp_luck': opp_luck,  # Lucky (positive)
        'covered': bet_covered,
    })

df_results = pd.DataFrame(results)

# Walk-forward results by week
print("=" * 120)
print("RESULTS BY WEEK: Betting UNLUCKY team in Lucky vs Unlucky matchups")
print("=" * 120)
print("")

total_bets = 0
total_wins = 0

for week in sorted(df_results['week'].unique()):
    week_bets = df_results[df_results['week'] == week]
    wins = week_bets['covered'].sum()
    bets = len(week_bets)
    total_bets += bets
    total_wins += wins
    
    win_pct = wins / bets * 100 if bets > 0 else 0
    result_icon = "✅" if wins > bets / 2 else ("⚠️" if wins == bets / 2 else "❌")
    
    print(f"Week {week:>2d}: {wins}/{bets} ({win_pct:>5.1f}%) {result_icon}")
    
    for _, bet in week_bets.iterrows():
        icon = "✅" if bet['covered'] else "❌"
        print(f"         {icon} {bet['bet_team']} {bet['bet_spread']:+.1f} vs {bet['opp_team']} (unlucky: {bet['bet_luck']:+.1f}, lucky: {bet['opp_luck']:+.1f})")

print("")
print("=" * 120)
print("OVERALL SUMMARY")
print("=" * 120)
print("")
print(f"Total Bets: {total_bets}")
print(f"Wins: {total_wins}")
print(f"Losses: {total_bets - total_wins}")
print(f"Win Rate: {total_wins / total_bets * 100:.1f}%")

roi = ((total_wins / total_bets * 1.909) - 1) * 100
print(f"ROI (at -110): {roi:+.1f}%")
print("")

# Break-even is 52.4% at -110 odds
break_even = 52.4
if total_wins / total_bets * 100 > break_even:
    print(f"✅ PROFITABLE! Above {break_even}% break-even")
else:
    print(f"❌ Not profitable. Below {break_even}% break-even")

print("")
print("=" * 120)
print("✅ COMPLETE")
print("=" * 120)




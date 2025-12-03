"""
NFL Spread Covering Analysis: BOTH Teams' Luck (Two-Dimensional)

Instead of just looking at one team's luck, this script analyzes the matchup
between both teams' prior week luck to predict spread covering.

Key Question:
When a "lucky" team (overperformed last week) plays an "unlucky" team 
(underperformed last week), who covers the spread?

Luck = score - adj_score (from Unexpected Points data)
- Positive luck = overperformed (scored more than expected)
- Negative luck = underperformed (scored less than expected)

Luck Categories (based on --threshold):
- Lucky: luck >= +threshold
- Neutral: -threshold < luck < +threshold
- Unlucky: luck <= -threshold

Threshold Guidelines:
- --threshold 3: More games qualify, larger sample, weaker signal
- --threshold 5: Balanced (recommended starting point)
- --threshold 7: ~1 TD of variance, stronger signal, smaller sample

Arguments:
- --threshold N: Set luck cutoff (default: 3, from config.py)
- --group-by-spread: Break down by spread size (0-3, 3.5-7, 7.5+)
- --include-fav-dog: Split by unlucky team's role (favorite vs underdog)
- --show-games: Display individual Lucky vs Unlucky games
- --team ABBR: Show one team's game-by-game path through the season
- --data-quality-check: Verify all teams have expected games based on bye weeks
- --debug: Show detailed debugging info

Usage:
    # Basic analysis
    python backtesting/20251202_nfl_both_teams_luck_analysis.py --threshold 5
    python backtesting/20251202_nfl_both_teams_luck_analysis.py --threshold 3
    
    # Group by spread size
    python backtesting/20251202_nfl_both_teams_luck_analysis.py --threshold 5 --group-by-spread
    
    # Full breakdown: spread + unlucky team's role (6 analyses)
    python backtesting/20251202_nfl_both_teams_luck_analysis.py --threshold 5 --group-by-spread --include-fav-dog
    
    # Show individual games
    python backtesting/20251202_nfl_both_teams_luck_analysis.py --threshold 3 --show-games
    
    # Show one team's path through the season
    python backtesting/20251202_nfl_both_teams_luck_analysis.py --threshold 5 --team GB
    python backtesting/20251202_nfl_both_teams_luck_analysis.py --threshold 3 --team DET
    
    # Verify data quality
    python backtesting/20251202_nfl_both_teams_luck_analysis.py --data-quality-check
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from nfl_luck_utils import (
    categorize_luck,
    categorize_spread,
    get_nfl_week,
    load_nfl_betting_lines,
    calculate_consensus_lines,
    load_unexpected_points_data,
    build_prior_luck_lookup,
    get_prior_week_luck,
    get_luck_matchup_ats_results,
    calculate_roi,
    LUCK_CATEGORIES,
    SPREAD_CATEGORIES,
    NFL_2025_BYE_WEEKS,
)
from nfl_team_utils import normalize_unexpected_points_abbr
from config import NFL_LUCK_THRESHOLD_DEFAULT

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Analyze NFL spread covering with both teams luck')
parser.add_argument('--debug', action='store_true', 
                   help='Show detailed debugging info')
parser.add_argument('--threshold', type=float, default=NFL_LUCK_THRESHOLD_DEFAULT,
                   help=f'Luck threshold for categorization (default: {NFL_LUCK_THRESHOLD_DEFAULT})')
parser.add_argument('--team', type=str, default=None,
                   help='Show one team\'s path through the season (e.g., --team GB)')
parser.add_argument('--show-games', action='store_true',
                   help='Show individual games for each matchup type')
parser.add_argument('--group-by-spread', action='store_true',
                   help='Break down results by spread ranges (≤3, 3-7, >7)')
parser.add_argument('--include-fav-dog', action='store_true',
                   help='Further break down by favorite/underdog (use with --group-by-spread for 6 tables)')
parser.add_argument('--data-quality-check', action='store_true',
                   help='Verify each team has expected games based on bye weeks')
args = parser.parse_args()

threshold = args.threshold

print("=" * 100)
print("NFL SPREAD COVERING: BOTH TEAMS' LUCK ANALYSIS")
print("=" * 100)
print(f"\nLuck threshold: ±{threshold}")
print(f"  Lucky: luck >= +{threshold}")
print(f"  Neutral: -{threshold} < luck < +{threshold}")
print(f"  Unlucky: luck <= -{threshold}")

# =============================================================================
# STEP 1: Load all NFL betting lines
# =============================================================================
print("\n" + "=" * 100)
print("STEP 1: Loading NFL betting lines")
print("=" * 100)

df_lines = load_nfl_betting_lines()
print(f"Total betting lines loaded: {len(df_lines):,}")

# =============================================================================
# STEP 2: Calculate consensus best line for each game
# =============================================================================
print("\n" + "=" * 100)
print("STEP 2: Calculating consensus lines")
print("=" * 100)

df_consensus = calculate_consensus_lines(df_lines)
print(f"Consensus lines calculated: {len(df_consensus)} games")

# =============================================================================
# STEP 3: Load Unexpected Points data (game results)
# =============================================================================
print("\n" + "=" * 100)
print("STEP 3: Loading Unexpected Points data")
print("=" * 100)

# Use the latest file
up_path = Path("/Users/thomasmyles/dev/betting/data/01_input/unexpected_points/Unexpected Points Subscriber Data (tuesday before week 14).xlsx")
df_up = load_unexpected_points_data(file_path=up_path)

print(f"Unexpected Points data loaded: {len(df_up)} rows")
print(f"Weeks: {df_up['week'].min()} to {df_up['week'].max()}")

# =============================================================================
# STEP 4: Create team-week lookup for prior week luck
# =============================================================================
print("\n" + "=" * 100)
print("STEP 4: Building prior-week luck lookup (handles bye weeks)")
print("=" * 100)

# Build prior luck lookup
luck_lookup = build_prior_luck_lookup(df_up)

print(f"Teams tracked: {len(luck_lookup['weeks_played'])}")

# =============================================================================
# STEP 5: Join betting lines with game results and prior week luck
# =============================================================================
print("\n" + "=" * 100)
print("STEP 5: Joining data and computing prior week luck for both teams")
print("=" * 100)

# Restructure UP data by game
up_games = []
for game_id, game_group in df_up.groupby('game_id'):
    if len(game_group) != 2:
        continue
    
    team1 = game_group.iloc[0]
    team2 = game_group.iloc[1]
    
    up_games.append({
        'game_id_up': game_id,
        'season': team1['season'],
        'week': team1['week'],
        'team1_abbr': team1['team_canonical'],
        'team1_score': team1['score'],
        'team1_adj_score': team1['adj_score'],
        'team1_luck': team1['luck'],
        'team2_abbr': team2['team_canonical'],
        'team2_score': team2['score'],
        'team2_adj_score': team2['adj_score'],
        'team2_luck': team2['luck'],
    })

df_up_games = pd.DataFrame(up_games)
print(f"Restructured UP data: {len(df_up_games)} games")

# Match betting lines with UP games
matched_games = []

for idx, bet_game in df_consensus.iterrows():
    away = bet_game['away_abbr']
    home = bet_game['home_abbr']
    game_time = bet_game['game_time']
    
    # Estimate the NFL week from game date
    estimated_week = get_nfl_week(game_time)
    
    # Find matching game in UP data (by teams)
    match = df_up_games[
        ((df_up_games['team1_abbr'] == away) & (df_up_games['team2_abbr'] == home)) |
        ((df_up_games['team1_abbr'] == home) & (df_up_games['team2_abbr'] == away))
    ]
    
    if len(match) == 0:
        continue
    
    if len(match) > 1:
        # Multiple matches (same teams play twice) - use week to disambiguate
        # Find the match closest to the estimated week
        match = match.copy()
        match['week_diff'] = abs(match['week'] - estimated_week)
        match = match.sort_values('week_diff').iloc[:1]
    
    up_game = match.iloc[0]
    week = up_game['week']
    
    # Determine which team is away/home in UP data
    if up_game['team1_abbr'] == away:
        away_score = up_game['team1_score']
        away_luck = up_game['team1_luck']
        home_score = up_game['team2_score']
        home_luck = up_game['team2_luck']
    else:
        away_score = up_game['team2_score']
        away_luck = up_game['team2_luck']
        home_score = up_game['team1_score']
        home_luck = up_game['team1_luck']
    
    # Get PRIOR WEEK luck for both teams (handles bye weeks - looks at last played game)
    away_prior_luck = get_prior_week_luck(luck_lookup, away, week)
    home_prior_luck = get_prior_week_luck(luck_lookup, home, week)
    
    # Skip week 1 games (no prior week) or games where we don't have prior data
    if away_prior_luck is None or home_prior_luck is None:
        continue
    
    # Calculate actual game outcome vs spread
    actual_margin = away_score - home_score  # From away perspective
    away_covered = (actual_margin + bet_game['consensus_spread']) > 0
    
    # Categorize prior week luck for both teams
    away_luck_cat = categorize_luck(away_prior_luck, threshold)
    home_luck_cat = categorize_luck(home_prior_luck, threshold)
    
    # Categorize spread size
    abs_spread = abs(bet_game['consensus_spread'])
    spread_cat = categorize_spread(bet_game['consensus_spread'])
    
    # Determine favorite/underdog from away perspective
    # Negative spread = away is favorite, positive = away is underdog
    away_is_favorite = bet_game['consensus_spread'] < 0
    
    matched_games.append({
        'game_id': bet_game['game_id'],
        'game_time': bet_game['game_time'],
        'week': week,
        'away_abbr': away,
        'home_abbr': home,
        'consensus_spread': bet_game['consensus_spread'],
        'abs_spread': abs_spread,
        'spread_cat': spread_cat,
        'away_is_favorite': away_is_favorite,
        'home_is_favorite': not away_is_favorite,
        'away_score': away_score,
        'home_score': home_score,
        'actual_margin': actual_margin,
        'away_covered': away_covered,
        'home_covered': not away_covered,
        # This game's luck
        'away_luck': away_luck,
        'home_luck': home_luck,
        # PRIOR week's luck (the key feature!)
        'away_prior_luck': away_prior_luck,
        'home_prior_luck': home_prior_luck,
        'away_luck_cat': away_luck_cat,
        'home_luck_cat': home_luck_cat,
        # Matchup type (away's luck vs home's luck)
        'matchup_type': f"{away_luck_cat} vs {home_luck_cat}",
    })

df_matched = pd.DataFrame(matched_games)

print(f"\n✅ Successfully matched {len(df_matched)} games with prior week luck!")
print(f"   (Week 1 games excluded - no prior week data)")

# Show sample
print("\nSample matched games with prior week luck:")
print(df_matched[[
    'week', 'away_abbr', 'home_abbr', 
    'away_prior_luck', 'home_prior_luck',
    'away_luck_cat', 'home_luck_cat',
    'away_covered'
]].head(10).to_string())

# =============================================================================
# STEP 6: Helper function to print matchup analysis
# =============================================================================

def print_matchup_results(df, cat1, cat2, label):
    """Print results for a matchup type using get_luck_matchup_ats_results from nfl_luck_utils."""
    c1_covers, c2_covers, total = get_luck_matchup_ats_results(df, cat1, cat2)
    
    if total == 0:
        print(f"\n{label}: No games")
        return
    
    c1_pct = c1_covers / total * 100
    c2_pct = c2_covers / total * 100
    c1_roi = calculate_roi(c1_pct / 100)
    c2_roi = calculate_roi(c2_pct / 100)
    
    c1_edge = "💰" if c1_pct > 52.5 else ("💸" if c1_pct < 47.5 else "⚖️")
    c2_edge = "💰" if c2_pct > 52.5 else ("💸" if c2_pct < 47.5 else "⚖️")
    
    print(f"\n{label} ({total} games)")
    print(f"   {cat1} team covers: {c1_covers}-{total-c1_covers} ({c1_pct:.1f}%) | ROI: {c1_roi:+.1f}% {c1_edge}")
    print(f"   {cat2} team covers: {c2_covers}-{total-c2_covers} ({c2_pct:.1f}%) | ROI: {c2_roi:+.1f}% {c2_edge}")

# =============================================================================
# STEP 7: Main Matchup Analysis (No Home/Away - Just Luck Categories)
# =============================================================================
# Store for later use (needed by multiple sections)
lucky_vs_unlucky = df_matched[
    ((df_matched['away_luck_cat'] == 'Lucky') & (df_matched['home_luck_cat'] == 'Unlucky')) |
    ((df_matched['away_luck_cat'] == 'Unlucky') & (df_matched['home_luck_cat'] == 'Lucky'))
]

# Skip full analysis if only running data quality check
if not args.data_quality_check:
    print("\n" + "=" * 100)
    print("STEP 7: MATCHUP ANALYSIS BY LUCK CATEGORY")
    print("=" * 100)
    
    print("\n🔥 KEY MATCHUP: LUCKY vs UNLUCKY")
    print("-" * 60)
    print_matchup_results(df_matched, 'Lucky', 'Unlucky', "Lucky Team vs Unlucky Team")
    
    print("\n\n🎯 OTHER MATCHUP TYPES:")
    print("-" * 60)
    print_matchup_results(df_matched, 'Lucky', 'Neutral', "Lucky Team vs Neutral Team")
    print_matchup_results(df_matched, 'Unlucky', 'Neutral', "Unlucky Team vs Neutral Team")
    
    # Same-category matchups (just show sample size, no "who covers" since same category)
    print("\n\n📊 SAME-CATEGORY MATCHUPS (Sample Sizes):")
    print("-" * 60)
    
    lucky_lucky = df_matched[
        (df_matched['away_luck_cat'] == 'Lucky') & (df_matched['home_luck_cat'] == 'Lucky')
    ]
    unlucky_unlucky = df_matched[
        (df_matched['away_luck_cat'] == 'Unlucky') & (df_matched['home_luck_cat'] == 'Unlucky')
    ]
    neutral_neutral = df_matched[
        (df_matched['away_luck_cat'] == 'Neutral') & (df_matched['home_luck_cat'] == 'Neutral')
    ]
    
    print(f"   Lucky vs Lucky: {len(lucky_lucky)} games")
    print(f"   Unlucky vs Unlucky: {len(unlucky_unlucky)} games")
    print(f"   Neutral vs Neutral: {len(neutral_neutral)} games (baseline)")

# =============================================================================
# STEP 9: Grouped analysis by spread size and/or favorite/underdog
# =============================================================================
if (args.group_by_spread or args.include_fav_dog) and not args.data_quality_check:
    print("\n" + "=" * 100)
    print("GROUPED ANALYSIS: SPREAD SIZE" + (" + UNLUCKY TEAM'S ROLE" if args.include_fav_dog else ""))
    print("=" * 100)
    
    spread_cats = SPREAD_CATEGORIES
    order = LUCK_CATEGORIES
    
    def analyze_subset(df_subset, label):
        """Analyze a subset of games - focus on Lucky vs Unlucky results."""
        if len(df_subset) == 0:
            print(f"\n{label}: No games")
            return
        
        print(f"\n{'='*80}")
        print(f"{label} (n={len(df_subset)} total games)")
        print(f"{'='*80}")
        
        # Lucky vs Unlucky in this subset
        lu_c1, lu_c2, lu_total = get_luck_matchup_ats_results(df_subset, 'Lucky', 'Unlucky')
        ln_c1, ln_c2, ln_total = get_luck_matchup_ats_results(df_subset, 'Lucky', 'Neutral')
        un_c1, un_c2, un_total = get_luck_matchup_ats_results(df_subset, 'Unlucky', 'Neutral')
        
        if lu_total > 0:
            unlucky_pct = lu_c2 / lu_total * 100
            unlucky_roi = ((unlucky_pct / 100 * 1.909) - 1) * 100
            edge = "💰" if unlucky_pct > 52.5 else ("💸" if unlucky_pct < 47.5 else "⚖️")
            print(f"\n🎯 Lucky vs Unlucky: {lu_total} games")
            print(f"   Unlucky covers: {lu_c2}-{lu_c1} ({unlucky_pct:.1f}%) | ROI: {unlucky_roi:+.1f}% {edge}")
        else:
            print(f"\n🎯 Lucky vs Unlucky: No games")
        
        if ln_total > 0:
            lucky_pct = ln_c1 / ln_total * 100
            print(f"\n   Lucky vs Neutral: {ln_total} games | Lucky covers: {ln_c1}-{ln_c2} ({lucky_pct:.1f}%)")
        
        if un_total > 0:
            unlucky_pct = un_c1 / un_total * 100
            print(f"   Unlucky vs Neutral: {un_total} games | Unlucky covers: {un_c1}-{un_c2} ({unlucky_pct:.1f}%)")
    
    def get_lucky_vs_unlucky_with_roles(df_subset):
        """
        Extract Lucky vs Unlucky games and determine the unlucky team's role (fav/dog).
        Returns list of dicts with game info and unlucky team's role.
        """
        lu_games = []
        
        lu_subset = df_subset[
            ((df_subset['away_luck_cat'] == 'Lucky') & (df_subset['home_luck_cat'] == 'Unlucky')) |
            ((df_subset['away_luck_cat'] == 'Unlucky') & (df_subset['home_luck_cat'] == 'Lucky'))
        ]
        
        for _, game in lu_subset.iterrows():
            if game['away_luck_cat'] == 'Unlucky':
                # Away team is unlucky
                unlucky_is_favorite = game['away_is_favorite']
                unlucky_covered = game['away_covered']
            else:
                # Home team is unlucky
                unlucky_is_favorite = game['home_is_favorite']
                unlucky_covered = game['home_covered']
            
            lu_games.append({
                'game': game,
                'unlucky_is_favorite': unlucky_is_favorite,
                'unlucky_covered': unlucky_covered,
                'spread_cat': game['spread_cat'],
            })
        
        return lu_games
    
    # Get all Lucky vs Unlucky games with role info
    all_lu_games = get_lucky_vs_unlucky_with_roles(df_matched)
    
    # Loop through spread categories
    for spread_cat in spread_cats:
        df_spread = df_matched[df_matched['spread_cat'] == spread_cat]
        
        if not args.include_fav_dog:
            # Just spread grouping - show full 3x3 matrix
            analyze_subset(df_spread, f"SPREAD {spread_cat}")
        else:
            # Spread + Fav/Dog grouping based on UNLUCKY team's role
            # Filter Lucky vs Unlucky games by spread and unlucky team's role
            lu_in_spread = [g for g in all_lu_games if g['spread_cat'] == spread_cat]
            
            for role, role_label in [(True, 'UNLUCKY IS FAVORITE'), (False, 'UNLUCKY IS UNDERDOG')]:
                lu_filtered = [g for g in lu_in_spread if g['unlucky_is_favorite'] == role]
                
                print(f"\n{'='*80}")
                print(f"SPREAD {spread_cat} | {role_label} (n={len(lu_filtered)} Lucky vs Unlucky games)")
                print(f"{'='*80}")
                
                if len(lu_filtered) > 0:
                    unlucky_covers = sum(1 for g in lu_filtered if g['unlucky_covered'])
                    total = len(lu_filtered)
                    unlucky_pct = unlucky_covers / total * 100
                    lucky_pct = 100 - unlucky_pct
                    unlucky_roi = ((unlucky_pct / 100 * 1.909) - 1) * 100
                    
                    print(f"\n🎯 Unlucky team covers: {unlucky_covers}-{total-unlucky_covers} ({unlucky_pct:.1f}%) | ROI: {unlucky_roi:+.1f}%")
                    print(f"   Lucky team covers: {total-unlucky_covers}-{unlucky_covers} ({lucky_pct:.1f}%)")
                    
                    if unlucky_pct > 52.5:
                        print(f"   ✅ EDGE: Bet unlucky team")
                    elif lucky_pct > 52.5:
                        print(f"   ⚡ EDGE: Bet lucky team")
                    else:
                        print(f"   ⚖️  No clear edge")
                else:
                    print(f"\n   No Lucky vs Unlucky games in this category")
    
    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY: UNLUCKY TEAM ATS BY SPREAD + ROLE")
    print("=" * 100)
    
    print(f"\n{'Group':<45s} {'Sample':<10s} {'Unlucky ATS':<15s} {'ROI':<10s}")
    print("-" * 85)
    
    for spread_cat in spread_cats:
        lu_in_spread = [g for g in all_lu_games if g['spread_cat'] == spread_cat]
        
        if not args.include_fav_dog:
            # Just by spread
            if len(lu_in_spread) > 0:
                unlucky_covers = sum(1 for g in lu_in_spread if g['unlucky_covered'])
                total = len(lu_in_spread)
                unlucky_pct = unlucky_covers / total * 100
                unlucky_roi = ((unlucky_pct / 100 * 1.909) - 1) * 100
                record = f"{unlucky_covers}-{total-unlucky_covers} ({unlucky_pct:.0f}%)"
                roi_str = f"{unlucky_roi:+.1f}%"
                edge = "💰" if unlucky_pct > 52.5 else ("💸" if unlucky_pct < 47.5 else "⚖️")
                print(f"Spread {spread_cat:<40s} {total:<10d} {record:<15s} {roi_str:<10s} {edge}")
            else:
                print(f"Spread {spread_cat:<40s} {'0':<10s} {'-':<15s} {'-':<10s}")
        else:
            # By spread + unlucky team's role
            for role, role_label in [(True, 'Unlucky is Fav'), (False, 'Unlucky is Dog')]:
                lu_filtered = [g for g in lu_in_spread if g['unlucky_is_favorite'] == role]
                label = f"Spread {spread_cat} | {role_label}"
                
                if len(lu_filtered) > 0:
                    unlucky_covers = sum(1 for g in lu_filtered if g['unlucky_covered'])
                    total = len(lu_filtered)
                    unlucky_pct = unlucky_covers / total * 100
                    unlucky_roi = ((unlucky_pct / 100 * 1.909) - 1) * 100
                    record = f"{unlucky_covers}-{total-unlucky_covers} ({unlucky_pct:.0f}%)"
                    roi_str = f"{unlucky_roi:+.1f}%"
                    edge = "💰" if unlucky_pct > 52.5 else ("💸" if unlucky_pct < 47.5 else "⚖️")
                    print(f"{label:<45s} {total:<10d} {record:<15s} {roi_str:<10s} {edge}")
                else:
                    print(f"{label:<45s} {'0':<10s} {'-':<15s} {'-':<10s}")

# =============================================================================
# STEP 10: Show individual games if requested
# =============================================================================
if args.show_games and not args.data_quality_check:
    print("\n" + "=" * 100)
    print("INDIVIDUAL GAMES: LUCKY vs UNLUCKY MATCHUPS")
    print("=" * 100)
    
    for _, game in lucky_vs_unlucky.sort_values('week').iterrows():
        away = game['away_abbr']
        home = game['home_abbr']
        week = int(game['week'])
        
        if game['away_luck_cat'] == 'Lucky':
            lucky_team = away
            unlucky_team = home
            lucky_loc = 'Away'
            lucky_prior = game['away_prior_luck']
            unlucky_prior = game['home_prior_luck']
            lucky_covered = game['away_covered']
        else:
            lucky_team = home
            unlucky_team = away
            lucky_loc = 'Home'
            lucky_prior = game['home_prior_luck']
            unlucky_prior = game['away_prior_luck']
            lucky_covered = game['home_covered']
        
        result_str = "Lucky ✓" if lucky_covered else "Unlucky ✓"
        spread_str = f"{game['consensus_spread']:+.1f}"
        score_str = f"{int(game['away_score'])}-{int(game['home_score'])}"
        
        print(f"Wk{week:2d} | {away:>4s} @ {home:<4s} | Spread: {spread_str:>6s} | Score: {score_str:>7s} | Lucky={lucky_team} ({lucky_loc}, +{lucky_prior:.1f}) vs Unlucky={unlucky_team} ({unlucky_prior:+.1f}) → {result_str}")

# =============================================================================
# STEP 11: Team Path Through Season
# =============================================================================
if args.team and not args.data_quality_check:
    team = args.team.upper()
    print("\n" + "=" * 100)
    print(f"TEAM PATH: {team} - Game-by-Game Season Analysis")
    print("=" * 100)
    
    # Get ALL games for this team from UP data (including week 1)
    team_up_data = df_up[df_up['team_canonical'] == team].sort_values('week')
    
    if len(team_up_data) == 0:
        print(f"\n⚠️  No games found for team: {team}")
        print(f"   Available teams: {sorted(df_up['team_canonical'].unique())}")
    else:
        bye_week = NFL_2025_BYE_WEEKS.get(team, None)
        print(f"\nShowing {len(team_up_data)} games for {team} (bye week {bye_week})")
        print(f"Threshold: ±{threshold} for Lucky/Unlucky classification")
        print()
        
        # Header
        print(f"{'Wk':<4s} {'Opp':<5s} {'Score':<10s} {'ATS':<5s} {'Spread':<8s} "
              f"{'Team Luck':<12s} {'Opp Luck':<12s} {'Matchup Type':<25s} {'Result':<15s}")
        print("-" * 110)
        
        # Track stats
        wins = 0
        losses = 0
        ats_wins = 0
        ats_losses = 0
        lu_matchups = 0
        lu_correct = 0
        
        for _, team_row in team_up_data.iterrows():
            week = int(team_row['week'])
            team_score = int(team_row['score'])
            game_id = team_row['game_id']
            
            # Find opponent from the same game
            opp_row = df_up[(df_up['game_id'] == game_id) & (df_up['team_canonical'] != team)]
            if len(opp_row) == 0:
                continue
            opp_row = opp_row.iloc[0]
            opp = opp_row['team_canonical']
            opp_score = int(opp_row['score'])
            
            # Get prior luck for both teams
            team_prior_luck = get_prior_week_luck(luck_lookup, team, week)
            opp_prior_luck = get_prior_week_luck(luck_lookup, opp, week)
            
            # Try to get spread from consensus lines
            spread = None
            spread_str = "-"
            team_covered = None
            ats_str = "-"
            
            # Find matching consensus line
            consensus_match = df_consensus[
                ((df_consensus['away_abbr'] == team) & (df_consensus['home_abbr'] == opp)) |
                ((df_consensus['away_abbr'] == opp) & (df_consensus['home_abbr'] == team))
            ]
            
            if len(consensus_match) > 0:
                # Use week estimation to pick correct game if teams play twice
                if len(consensus_match) > 1:
                    consensus_match = consensus_match.copy()
                    consensus_match['week_diff'] = abs(consensus_match['game_time'].apply(get_nfl_week) - week)
                    consensus_match = consensus_match.sort_values('week_diff').iloc[:1]
                
                bet_game = consensus_match.iloc[0]
                
                # Determine spread from team's perspective
                if bet_game['away_abbr'] == team:
                    spread = bet_game['consensus_spread']
                else:
                    spread = -bet_game['consensus_spread']
                
                spread_str = f"{spread:+.1f}"
                
                # Calculate ATS
                actual_margin = team_score - opp_score
                team_covered = (actual_margin + spread) > 0 if spread is not None else None
                ats_str = "✓" if team_covered else "✗"
            
            # Win/Loss
            won = team_score > opp_score
            tie = team_score == opp_score
            if won:
                wins += 1
            elif not tie:
                losses += 1
            
            if team_covered is not None:
                if team_covered:
                    ats_wins += 1
                else:
                    ats_losses += 1
            
            # Determine luck categories
            if team_prior_luck is None or opp_prior_luck is None:
                team_luck_cat = "N/A"
                opp_luck_cat = "N/A"
                matchup = "Week 1 (no prior)" if week == 1 else "Missing data"
                team_luck_str = "-"
                opp_luck_str = "-"
            else:
                def categorize_luck(luck_value, thresh):
                    if luck_value >= thresh:
                        return 'Lucky'
                    elif luck_value <= -thresh:
                        return 'Unlucky'
                    else:
                        return 'Neutral'
                
                team_luck_cat = categorize_luck(team_prior_luck, threshold)
                opp_luck_cat = categorize_luck(opp_prior_luck, threshold)
                matchup = f"{team_luck_cat} vs {opp_luck_cat}"
                team_luck_str = f"{team_prior_luck:+.1f} ({team_luck_cat[0]})"
                opp_luck_str = f"{opp_prior_luck:+.1f} ({opp_luck_cat[0]})"
            
            # Check if this is a Lucky vs Unlucky matchup
            is_lu_matchup = (team_luck_cat == 'Lucky' and opp_luck_cat == 'Unlucky') or \
                           (team_luck_cat == 'Unlucky' and opp_luck_cat == 'Lucky')
            
            result_str = ""
            if is_lu_matchup and team_covered is not None:
                lu_matchups += 1
                if team_luck_cat == 'Unlucky':
                    expected_cover = team_covered
                else:
                    expected_cover = not team_covered
                
                if expected_cover:
                    lu_correct += 1
                    result_str = "✅ Unlucky covered"
                else:
                    result_str = "❌ Lucky covered"
            
            # Highlight Lucky vs Unlucky matchups
            if is_lu_matchup:
                matchup = f"⭐ {matchup}"
            
            # Format score
            score_str = f"{team_score}-{opp_score}"
            if won:
                score_str += " W"
            elif tie:
                score_str += " T"
            else:
                score_str += " L"
            
            print(f"{week:<4d} {opp:<5s} {score_str:<10s} {ats_str:<5s} {spread_str:<8s} "
                  f"{team_luck_str:<12s} {opp_luck_str:<12s} {matchup:<25s} {result_str:<15s}")
        
        # Summary
        print("-" * 110)
        total_games = wins + losses
        print(f"\n📊 {team} Season Summary:")
        print(f"   Record: {wins}-{losses}" + (f" ({wins/total_games*100:.1f}%)" if total_games > 0 else ""))
        if ats_wins + ats_losses > 0:
            print(f"   ATS: {ats_wins}-{ats_losses} ({ats_wins/(ats_wins+ats_losses)*100:.1f}%)")
        
        if lu_matchups > 0:
            print(f"\n🎯 Lucky vs Unlucky Matchups: {lu_matchups}")
            print(f"   Unlucky team covered: {lu_correct}/{lu_matchups} ({lu_correct/lu_matchups*100:.1f}%)")
        else:
            print(f"\n🎯 Lucky vs Unlucky Matchups: 0")

# =============================================================================
# STEP 12: Data Quality Check
# =============================================================================
if args.data_quality_check:
    print("\n" + "=" * 100)
    print("DATA QUALITY CHECK")
    print("=" * 100)
    
    # -------------------------------------------------------------------------
    # Part 1: Join Coverage between Betting Lines and Unexpected Points
    # -------------------------------------------------------------------------
    print("\n📊 JOIN COVERAGE: Betting Lines ↔ Unexpected Points")
    print("-" * 70)
    
    # Count unique games in each source
    up_game_count = len(df_up_games)
    betting_game_count = len(df_consensus)
    matched_game_count = len(df_matched)
    
    # Games in UP but not matched to betting lines
    up_game_keys = set()
    for _, g in df_up_games.iterrows():
        up_game_keys.add((g['team1_abbr'], g['team2_abbr'], g['week']))
        up_game_keys.add((g['team2_abbr'], g['team1_abbr'], g['week']))
    
    matched_game_keys = set()
    for _, g in df_matched.iterrows():
        matched_game_keys.add((g['away_abbr'], g['home_abbr'], g['week']))
        matched_game_keys.add((g['home_abbr'], g['away_abbr'], g['week']))
    
    # Find unmatched UP games (excluding week 1 which has no prior luck)
    unmatched_up_games = []
    for _, g in df_up_games.iterrows():
        if g['week'] == 1:
            continue  # Week 1 expected to be unmatched (no prior luck)
        key1 = (g['team1_abbr'], g['team2_abbr'], g['week'])
        key2 = (g['team2_abbr'], g['team1_abbr'], g['week'])
        if key1 not in matched_game_keys and key2 not in matched_game_keys:
            unmatched_up_games.append(g)
    
    # Calculate coverage
    # UP games minus week 1 games = games that SHOULD be matched
    week1_games = len(df_up_games[df_up_games['week'] == 1])
    up_matchable = up_game_count - week1_games
    
    if up_matchable > 0:
        join_coverage = matched_game_count / up_matchable * 100
    else:
        join_coverage = 0
    
    print(f"\n   Unexpected Points games: {up_game_count}")
    print(f"   Betting lines games: {betting_game_count}")
    print(f"   Week 1 games (no prior luck): {week1_games}")
    print(f"   Matchable UP games (week 2+): {up_matchable}")
    print(f"   Successfully matched: {matched_game_count}")
    print(f"   Join coverage: {join_coverage:.1f}%")
    
    if join_coverage >= 100:
        print(f"\n   ✅ 100% join coverage!")
    elif join_coverage >= 95:
        print(f"\n   ⚠️  {100-join_coverage:.1f}% of games not matched")
    else:
        print(f"\n   ❌ Only {join_coverage:.1f}% coverage - investigate missing games")
    
    if unmatched_up_games:
        print(f"\n   Unmatched games ({len(unmatched_up_games)}):")
        for g in unmatched_up_games[:10]:  # Show first 10
            print(f"      Week {g['week']}: {g['team1_abbr']} vs {g['team2_abbr']}")
        if len(unmatched_up_games) > 10:
            print(f"      ... and {len(unmatched_up_games) - 10} more")
    
    # -------------------------------------------------------------------------
    # Part 2: Games per Team
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("📊 GAMES PER TEAM")
    print("-" * 70)
    
    # Get current max week in UP data
    max_week = df_up['week'].max()
    print(f"\nData through Week {max_week}")
    print(f"Bye weeks factored in for each team")
    
    # For each team, calculate expected games and actual matched games
    all_teams = sorted(set(df_matched['away_abbr'].unique()) | set(df_matched['home_abbr'].unique()))
    
    print(f"\n{'Team':<6s} {'Bye':<5s} {'Expected':<10s} {'In UP':<8s} {'Matched':<10s} {'Missing':<10s} {'Status':<8s}")
    print("-" * 70)
    
    issues = []
    
    for team in all_teams:
        bye_week = NFL_2025_BYE_WEEKS.get(team, None)
        
        # Expected games through max_week (minus bye if it's passed)
        expected_games = max_week
        if bye_week and bye_week <= max_week:
            expected_games -= 1
        
        # Games in UP data
        up_games_count = len(df_up[df_up['team_canonical'] == team])
        
        # Games in matched data (with prior luck available)
        team_matched = df_matched[
            (df_matched['away_abbr'] == team) | (df_matched['home_abbr'] == team)
        ]
        matched_count = len(team_matched)
        
        # Expected matched = expected games - 1 (week 1 has no prior)
        expected_matched = expected_games - 1
        
        # Check for issues
        missing = expected_matched - matched_count
        
        if up_games_count < expected_games:
            status = "⚠️ UP"
            issues.append(f"{team}: Missing {expected_games - up_games_count} game(s) in Unexpected Points data")
        elif matched_count < expected_matched:
            status = "⚠️ Match"
            issues.append(f"{team}: {missing} game(s) not matched (likely opponent bye week)")
        else:
            status = "✓"
        
        bye_str = str(bye_week) if bye_week else "-"
        print(f"{team:<6s} {bye_str:<5s} {expected_games:<10d} {up_games_count:<8d} {matched_count:<10d} {missing:<10d} {status:<8s}")
    
    if issues:
        print(f"\n⚠️  Issues Found ({len(issues)}):")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print(f"\n✓ All teams have expected game counts!")
    
    print(f"\nNote: 'Missing' games are usually due to opponent bye weeks (no prior luck for opponent)")

# =============================================================================
# STEP 13: Save results
# =============================================================================
print("\n" + "=" * 100)
print("Saving results")
print("=" * 100)

intermediate_dir = Path("/Users/thomasmyles/dev/betting/data/03_intermediate")
intermediate_dir.mkdir(parents=True, exist_ok=True)

output_path = intermediate_dir / f"nfl_both_teams_luck_analysis_threshold_{int(threshold)}.csv"
df_matched.to_csv(output_path, index=False)

print(f"✅ Saved to: {output_path}")
print(f"   Rows: {len(df_matched)}")

# =============================================================================
# SUMMARY
# =============================================================================
if not args.data_quality_check:
    print("\n" + "=" * 100)
    print("SUMMARY & BETTING IMPLICATIONS")
    print("=" * 100)
    
    print(f"""
Analysis complete using ±{threshold} luck threshold.

Key Finding:
  When a LUCKY team (overperformed by +{threshold}+ last week) plays an 
  UNLUCKY team (underperformed by -{threshold}+ last week):
""")
    
    # Recalculate aggregate stats (may have been overwritten by grouped analysis)
    if len(lucky_vs_unlucky) > 0:
        summary_lucky_covers = 0
        summary_unlucky_covers = 0
        
        for _, game in lucky_vs_unlucky.iterrows():
            if game['away_luck_cat'] == 'Lucky':
                if game['away_covered']:
                    summary_lucky_covers += 1
                else:
                    summary_unlucky_covers += 1
            else:
                if game['away_covered']:
                    summary_unlucky_covers += 1
                else:
                    summary_lucky_covers += 1
        
        summary_total = len(lucky_vs_unlucky)
        summary_lucky_pct = summary_lucky_covers / summary_total * 100
        summary_unlucky_pct = summary_unlucky_covers / summary_total * 100
        
        print(f"  • Sample size: {summary_total} games")
        print(f"  • Lucky team covers: {summary_lucky_pct:.1f}%")
        print(f"  • Unlucky team covers: {summary_unlucky_pct:.1f}%")
        
        if summary_unlucky_pct > 55:
            print(f"\n  🎯 ACTIONABLE: Strong edge betting the UNLUCKY team!")
            print(f"     Regression to mean appears significant.")
        elif summary_unlucky_pct > 52.5:
            print(f"\n  📈 MODERATE: Slight edge betting the UNLUCKY team.")
        elif summary_lucky_pct > 55:
            print(f"\n  ⚡ CONTRARIAN: Lucky teams continue to dominate!")
            print(f"     Momentum > regression in this sample.")
        else:
            print(f"\n  ⚖️  NEUTRAL: No significant edge either way.")
    else:
        print(f"  No Lucky vs Unlucky matchups found.")
    
    print(f"""
Next Steps:
  1. Run with different thresholds: --threshold 5, --threshold 10
  2. View individual games: --show-games
  3. Cross-reference with spread size to refine further
""")
    
    print("=" * 100)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 100)


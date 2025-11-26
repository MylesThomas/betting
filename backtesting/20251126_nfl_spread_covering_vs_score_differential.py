"""
NFL Spread Covering Analysis: Score Differential Impact

Analyzes whether a team's score vs. adj_score differential in week N
predicts their ability to cover the spread in week N+1.

Methodology:
1. Load all NFL game lines (spreads) from historical data
2. Calculate consensus best line for each game
3. Join with Unexpected Points data (game results + adj_scores)
4. Track week-to-week: Does overperformance in week N lead to failing to cover in week N+1?

Key Questions:
- Do teams that overperformed (score >> adj_score) fail to cover next week?
- Do teams that underperformed (score << adj_score) beat the spread next week?
- Is there a regression-to-mean effect in spread covering?

Usage:
    python backtesting/20251126_nfl_spread_covering_vs_score_differential.py              # Standard run
    python backtesting/20251126_nfl_spread_covering_vs_score_differential.py --debug      # Show unmatched games
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import sys
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from nfl_team_utils import add_team_abbr_columns, normalize_unexpected_points_abbr

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Analyze NFL spread covering vs score differential')
parser.add_argument('--debug', action='store_true', 
                   help='Show detailed debugging info including unmatched games')
parser.add_argument('--observe-trends', action='store_true',
                   help='Observe sequential trends for each team')
parser.add_argument('--team', type=str, default='all',
                   help='Team abbreviation to analyze (e.g., "GB") or "all" for all teams')
parser.add_argument('--threshold', type=float, default=7.0,
                   help='Luck threshold for extreme performance tracking (default: 7.0)')
parser.add_argument('--group-by-spread', action='store_true',
                   help='Break down results by spread ranges (0-3, 3-7, 7+)')
parser.add_argument('--include-fav-dog', action='store_true',
                   help='Further break down by favorite/underdog (requires --group-by-spread)')
args = parser.parse_args()

print("=" * 100)
print("NFL SPREAD COVERING vs SCORE DIFFERENTIAL ANALYSIS")
print("=" * 100)

# =============================================================================
# STEP 1: Load all NFL betting lines
# =============================================================================
print("\n" + "=" * 100)
print("STEP 1: Loading NFL betting lines")
print("=" * 100)

lines_dir = Path("/Users/thomasmyles/dev/betting/data/01_input/the-odds-api/nfl/game_lines/historical")
csv_files = sorted(glob.glob(str(lines_dir / "nfl_game_lines_*.csv")))

# Add London games file if exists
london_file = lines_dir / "2025_game_lines_london.csv"
if london_file.exists():
    csv_files.append(str(london_file))

print(f"Found {len(csv_files)} CSV files")

# Load all betting lines
dfs = []
for csv_file in csv_files:
    df_temp = pd.read_csv(csv_file)
    dfs.append(df_temp)

df_lines = pd.concat(dfs, ignore_index=True)

print(f"Total betting lines loaded: {len(df_lines):,}")
print(f"Unique games: {df_lines['game_id'].nunique()}")
print(f"Unique bookmakers: {df_lines['bookmaker'].nunique()}")

# Convert game_time to datetime
df_lines['game_time'] = pd.to_datetime(df_lines['game_time'])

# Filter to 2025 season only (Sept 1, 2025+)
season_start = pd.to_datetime('2025-09-01').tz_localize('America/New_York')
df_lines = df_lines[df_lines['game_time'] >= season_start].copy()

print(f"Lines in 2025 season: {len(df_lines):,}")

# Add team abbreviations for joining
df_lines = add_team_abbr_columns(df_lines)

# =============================================================================
# STEP 2: Calculate consensus best line for each game
# =============================================================================
print("\n" + "=" * 100)
print("STEP 2: Calculating consensus best lines")
print("=" * 100)

# For each game, get the consensus (median/mode) spread and average odds
# We'll take AWAY perspective (away_spread)

consensus_lines = []

for game_id, game_group in df_lines.groupby('game_id'):
    # Get unique game info
    away_team = game_group['away_team'].iloc[0]
    home_team = game_group['home_team'].iloc[0]
    away_abbr = game_group['away_abbr'].iloc[0]
    home_abbr = game_group['home_abbr'].iloc[0]
    game_time = game_group['game_time'].iloc[0]
    
    # Spread lines (away perspective)
    spreads = game_group['away_spread'].dropna()
    
    if len(spreads) == 0:
        continue
    
    # Consensus spread: median (most common middle value)
    consensus_spread = spreads.median()
    
    # Also track the range (best and worst lines available)
    best_spread_away = spreads.max()  # Highest spread = best for away bettors
    worst_spread_away = spreads.min()  # Lowest spread = worst for away bettors
    
    # Home perspective is opposite (negative of away spread)
    best_spread_home = -worst_spread_away  # Best home line = negative of worst away
    worst_spread_home = -best_spread_away  # Worst home line = negative of best away
    
    # Number of books offering this game
    num_books = len(spreads)
    
    # Standard deviation of spreads (line shopping value)
    spread_std = spreads.std() if len(spreads) > 1 else 0
    
    consensus_lines.append({
        'game_id': game_id,
        'game_time': game_time,
        'away_team': away_team,
        'home_team': home_team,
        'away_abbr': away_abbr,
        'home_abbr': home_abbr,
        'consensus_spread': consensus_spread,  # Away perspective
        'best_spread_away': best_spread_away,
        'worst_spread_away': worst_spread_away,
        'best_spread_home': best_spread_home,
        'worst_spread_home': worst_spread_home,
        'spread_std': spread_std,
        'num_books': num_books,
    })

df_consensus = pd.DataFrame(consensus_lines)

print(f"\nConsensus lines calculated: {len(df_consensus)} games")
print(f"Date range: {df_consensus['game_time'].min().date()} to {df_consensus['game_time'].max().date()}")
print(f"\nSample consensus lines:")
print(df_consensus[['game_time', 'away_abbr', 'home_abbr', 'consensus_spread', 'num_books']].head(10))

print(f"\nLine shopping value (sample with best/worst):")
sample_with_range = df_consensus[df_consensus['spread_std'] > 0].head(5)
for _, row in sample_with_range.iterrows():
    print(f"  {row['away_abbr']} @ {row['home_abbr']}: Away {row['worst_spread_away']:+.1f} to {row['best_spread_away']:+.1f} | Home {row['worst_spread_home']:+.1f} to {row['best_spread_home']:+.1f} (σ={row['spread_std']:.2f})")

# =============================================================================
# STEP 3: Load Unexpected Points data (game results)
# =============================================================================
print("\n" + "=" * 100)
print("STEP 3: Loading Unexpected Points data")
print("=" * 100)

up_path = Path("/Users/thomasmyles/dev/betting/data/01_input/unexpected_points/Unexpected Points Subscriber Data.xlsx")
df_up = pd.read_excel(up_path, sheet_name="2025 Adjusted Scores")

print(f"Unexpected Points data loaded: {len(df_up)} rows")
print(f"Teams: {df_up['team'].nunique()}")
print(f"Weeks: {df_up['week'].min()} to {df_up['week'].max()}")
print(f"Columns: {df_up.columns.tolist()}")

# Normalize Unexpected Points team abbreviations to canonical format
# (e.g., "LA" -> "LAR" for Rams)
print(f"\nNormalizing team abbreviations...")
df_up['team_canonical'] = df_up['team'].apply(normalize_unexpected_points_abbr)

# Check if any mappings were applied
normalized_count = (df_up['team'] != df_up['team_canonical']).sum()
if normalized_count > 0:
    print(f"  Normalized {normalized_count} rows (data source quirks)")
    # Show what was normalized
    normalized_teams = df_up[df_up['team'] != df_up['team_canonical']][['team', 'team_canonical']].drop_duplicates()
    for _, row in normalized_teams.iterrows():
        print(f"    {row['team']} -> {row['team_canonical']}")

# Calculate score differential
df_up['score_diff'] = df_up['score'] - df_up['adj_score']

# =============================================================================
# STEP 4: Join betting lines with game results
# =============================================================================
print("\n" + "=" * 100)
print("STEP 4: Joining betting lines with game results")
print("=" * 100)

# For each consensus line game, we need to find the corresponding game in UP data
# Strategy: Match on game_id from both datasets

# First, let's see what game_ids look like in both datasets
print("\nSample game_ids from consensus lines:")
print(df_consensus['game_id'].head(3).tolist())

print("\nSample game_ids from Unexpected Points:")
print(df_up['game_id'].head(3).tolist())

# Add game_time column to UP data for matching
# We need to match games by teams and approximate date

# Create a joining key: for each betting line game, create week + teams
df_consensus['game_date'] = df_consensus['game_time'].dt.date
df_consensus['game_key'] = (
    df_consensus['away_abbr'].astype(str) + '_' + 
    df_consensus['home_abbr'].astype(str) + '_' +
    df_consensus['game_date'].astype(str)
)

# For UP data, create matching keys (one per team)
df_up['away_home'] = 'away'  # Placeholder, need to determine

# Actually, UP data has both teams per game, so let's restructure it
# Group by game_id to get both teams

print("\nRestructuring Unexpected Points data by game...")

up_games = []
for game_id, game_group in df_up.groupby('game_id'):
    if len(game_group) != 2:
        continue
    
    team1 = game_group.iloc[0]
    team2 = game_group.iloc[1]
    
    # Determine away/home (we'll need to match with betting lines)
    # For now, just structure it
    
    game_info = {
        'game_id_up': game_id,
        'season': team1['season'],
        'week': team1['week'],
        'team1_abbr': team1['team_canonical'],  # Use canonical abbreviation
        'team1_score': team1['score'],
        'team1_adj_score': team1['adj_score'],
        'team1_score_diff': team1['score_diff'],
        'team2_abbr': team2['team_canonical'],  # Use canonical abbreviation
        'team2_score': team2['score'],
        'team2_adj_score': team2['adj_score'],
        'team2_score_diff': team2['score_diff'],
    }
    
    up_games.append(game_info)

df_up_games = pd.DataFrame(up_games)

print(f"Restructured UP data: {len(df_up_games)} games")
print("\nSample UP games:")
print(df_up_games[['week', 'team1_abbr', 'team2_abbr', 'team1_score', 'team2_score']].head(10))

# Now match betting lines with UP games by teams
print("\nMatching games...")

# For each consensus line, find matching UP game
matched_games = []
unmatched_games = []

for idx, bet_game in df_consensus.iterrows():
    away = bet_game['away_abbr']
    home = bet_game['home_abbr']
    
    # Find matching game in UP data (either order of teams)
    match = df_up_games[
        ((df_up_games['team1_abbr'] == away) & (df_up_games['team2_abbr'] == home)) |
        ((df_up_games['team1_abbr'] == home) & (df_up_games['team2_abbr'] == away))
    ]
    
    if len(match) == 0:
        unmatched_games.append({
            'game_id': bet_game['game_id'],
            'game_time': bet_game['game_time'],
            'game_date': bet_game['game_date'],
            'away_team': bet_game['away_team'],
            'away_abbr': away,
            'home_team': bet_game['home_team'],
            'home_abbr': home,
            'consensus_spread': bet_game['consensus_spread'],
            'num_books': bet_game['num_books'],
        })
        continue
    
    if len(match) > 1:
        # Multiple matches - shouldn't happen, take first
        if args.debug:
            print(f"⚠️  Multiple matches for {away} @ {home}")
        match = match.iloc[:1]
    
    up_game = match.iloc[0]
    
    # Determine which team is away/home in UP data
    if up_game['team1_abbr'] == away:
        away_score = up_game['team1_score']
        away_adj_score = up_game['team1_adj_score']
        away_score_diff = up_game['team1_score_diff']
        home_score = up_game['team2_score']
        home_adj_score = up_game['team2_adj_score']
        home_score_diff = up_game['team2_score_diff']
    else:
        away_score = up_game['team2_score']
        away_adj_score = up_game['team2_adj_score']
        away_score_diff = up_game['team2_score_diff']
        home_score = up_game['team1_score']
        home_adj_score = up_game['team1_adj_score']
        home_score_diff = up_game['team1_score_diff']
    
    # Calculate actual game outcome vs spread
    actual_margin = away_score - home_score  # From away perspective
    spread_result = actual_margin - bet_game['consensus_spread']
    
    # Did away team cover?
    # If spread is +7.5 (away is underdog), they cover if they lose by 7 or less
    # actual_margin + spread > 0 means cover
    # Example: spread = +7.5, lose by 3 (margin = -3): -3 + 7.5 = 4.5 > 0 ✓ cover
    # Example: spread = -7.5 (favorite), win by 10 (margin = +10): 10 - 7.5 = 2.5 > 0 ✓ cover
    away_covered = (actual_margin + bet_game['consensus_spread']) > 0
    
    # Alternative: spread_result > 0 means they beat the spread
    # If spread was +7.5 and they lost by 3, margin = -3
    # spread_result = -3 - 7.5 = -10.5 < 0... that's wrong
    
    # Correct calculation:
    # Away team covers if: actual_margin > -consensus_spread
    # Or: actual_margin + consensus_spread > 0
    # This handles both positive and negative spreads correctly
    
    matched_games.append({
        # Game identification
        'game_id': bet_game['game_id'],
        'game_time': bet_game['game_time'],
        'week': up_game['week'],
        'season': up_game['season'],
        
        # Teams
        'away_team': bet_game['away_team'],
        'away_abbr': away,
        'home_team': bet_game['home_team'],
        'home_abbr': home,
        
        # Betting lines
        'consensus_spread': bet_game['consensus_spread'],
        'best_spread_away': bet_game['best_spread_away'],
        'spread_std': bet_game['spread_std'],
        'num_books': bet_game['num_books'],
        
        # Actual results
        'away_score': away_score,
        'home_score': home_score,
        'actual_margin': actual_margin,
        
        # Adjusted scores (expected)
        'away_adj_score': away_adj_score,
        'home_adj_score': home_adj_score,
        
        # Score differentials (luck/overperformance)
        'away_score_diff': away_score_diff,
        'home_score_diff': home_score_diff,
        
        # Spread covering
        'away_covered': away_covered,
        'home_covered': not away_covered,
        'cover_margin': actual_margin + bet_game['consensus_spread'],  # How much they beat/missed spread by
    })

df_matched = pd.DataFrame(matched_games)

print(f"\n✅ Successfully matched {len(df_matched)} games!")
print(f"Match rate: {len(df_matched) / len(df_consensus) * 100:.1f}%")

# Show sample
print("\nSample matched games:")
print(df_matched[[
    'week', 'away_abbr', 'home_abbr', 
    'consensus_spread', 'actual_margin', 
    'away_covered', 'away_score_diff', 'home_score_diff'
]].head(15))

# =============================================================================
# STEP 5: Save joined data
# =============================================================================
print("\n" + "=" * 100)
print("STEP 5: Saving joined data")
print("=" * 100)

intermediate_dir = Path("/Users/thomasmyles/dev/betting/data/03_intermediate")
intermediate_dir.mkdir(parents=True, exist_ok=True)

output_path = intermediate_dir / "nfl_games_with_spreads_and_results.csv"
df_matched.to_csv(output_path, index=False)

print(f"✅ Saved to: {output_path}")
print(f"   Rows: {len(df_matched)}")
print(f"   Columns: {len(df_matched.columns)}")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
print("\n" + "=" * 100)
print("SUMMARY STATISTICS")
print("=" * 100)

print(f"\n📊 Coverage by week:")
weeks = df_matched.groupby('week').size().sort_index()
for week, count in weeks.items():
    print(f"  Week {int(week):2d}: {count:2d} games")

print(f"\n📊 Overall spread covering:")
away_covers = df_matched['away_covered'].sum()
away_games = len(df_matched)
print(f"  Away teams covered: {away_covers}/{away_games} ({away_covers/away_games*100:.1f}%)")
print(f"  Home teams covered: {away_games - away_covers}/{away_games} ({(away_games - away_covers)/away_games*100:.1f}%)")

print(f"\n📊 Score differential distribution:")
print(f"  Away teams:")
print(f"    Mean score_diff: {df_matched['away_score_diff'].mean():+.2f}")
print(f"    Median score_diff: {df_matched['away_score_diff'].median():+.2f}")
print(f"  Home teams:")
print(f"    Mean score_diff: {df_matched['home_score_diff'].mean():+.2f}")
print(f"    Median score_diff: {df_matched['home_score_diff'].median():+.2f}")

# =============================================================================
# DEBUG: Show unmatched games
# =============================================================================
if args.debug and len(unmatched_games) > 0:
    print("\n" + "=" * 100)
    print("DEBUG: UNMATCHED GAMES")
    print("=" * 100)
    print(f"\nTotal unmatched: {len(unmatched_games)} games")
    print(f"These betting lines have no corresponding game in Unexpected Points data:\n")
    
    df_unmatched = pd.DataFrame(unmatched_games)
    df_unmatched = df_unmatched.sort_values('game_time')
    
    print(f"{'Date':<12s} {'Away':<25s} {'Home':<25s} {'Spread':<8s} {'Books':<6s}")
    print("-" * 80)
    for _, game in df_unmatched.iterrows():
        date_str = game['game_date'].strftime('%Y-%m-%d')
        away_str = f"{game['away_abbr']} ({game['away_team']})"[:24]
        home_str = f"{game['home_abbr']} ({game['home_team']})"[:24]
        spread_str = f"{game['consensus_spread']:+.1f}"
        books_str = str(game['num_books'])
        
        print(f"{date_str:<12s} {away_str:<25s} {home_str:<25s} {spread_str:<8s} {books_str:<6s}")
    
    print("\nPossible reasons for unmatched games:")
    print("  • Game postponed/rescheduled")
    print("  • London/international games with date mismatch")
    print("  • Unexpected Points data not yet updated for these weeks")
    print("  • Team name abbreviation mismatch")

print("\n" + "=" * 100)
print("✅ DATA PREPARATION COMPLETE")
print("=" * 100)

if not args.observe_trends:
    print("\nNext steps:")
    print("1. Analyze if score_diff in week N predicts covering in week N+1")
    print("2. Track team-by-team: overperformance → fail to cover next week?")
    print("3. Build predictive model based on adj_score vs actual score")
    print("\nRun methodology analysis to dig deeper!")
    print("\n💡 Try: python3 backtesting/20251126_nfl_spread_covering_vs_score_differential.py --observe-trends --team GB")
    if not args.debug and len(unmatched_games) > 0:
        print(f"💡 Tip: Run with --debug flag to see {len(unmatched_games)} unmatched games")

# =============================================================================
# OBSERVE TRENDS: Sequential game-by-game analysis
# =============================================================================
if args.observe_trends:
    print("\n" + "=" * 100)
    print("OBSERVE TRENDS: Sequential Game-by-Game Analysis")
    print("=" * 100)
    
    # Filter teams if specified
    if args.team.upper() != 'ALL':
        teams_to_analyze = [args.team.upper()]
        print(f"\nAnalyzing team: {args.team.upper()}")
    else:
        teams_to_analyze = sorted(df_matched['away_abbr'].unique())
        print(f"\nAnalyzing all {len(teams_to_analyze)} teams")
    
    threshold = args.threshold
    
    print(f"\nTracking metrics (threshold: ±{threshold}):")
    print("  • ExpDiff: Expected score differential (from adj_score)")
    print("  • ActDiff: Actual score differential (from actual score)")
    print("  • Luck: Over/underperformance (ActDiff - ExpDiff)")
    print("  • SeasonΣ: Season running total of luck")
    print("  • W-L: Current win/loss record")
    print("  • ATS: Current against the spread record")
    print(f"  • Post+{threshold}: Shows 'Y' if this game is after a +{threshold} luck game")
    print(f"  • Post-{threshold}: Shows 'Y' if this game is after a -{threshold} luck game")
    
    # Track aggregate stats across all teams
    total_after_plus_wins = 0
    total_after_plus_losses = 0
    total_after_plus_ats_wins = 0
    total_after_plus_ats_losses = 0
    total_after_plus_spreads = []  # Track spreads for ROI calculation
    
    total_after_minus_wins = 0
    total_after_minus_losses = 0
    total_after_minus_ats_wins = 0
    total_after_minus_ats_losses = 0
    total_after_minus_spreads = []  # Track spreads for ROI calculation
    
    # Collect all game-by-game data for CSV export
    all_game_data = []
    
    for team in teams_to_analyze:
        print("\n" + "=" * 100)
        print(f"TEAM: {team}")
        print("=" * 100)
        
        # Get all games for this team (home or away)
        team_games = df_matched[
            (df_matched['away_abbr'] == team) | (df_matched['home_abbr'] == team)
        ].sort_values('week').copy()
        
        if len(team_games) == 0:
            print(f"  No games found for {team}")
            continue
        
        # Initialize tracking variables
        season_score_diff_total = 0
        wins = 0
        losses = 0
        ats_wins = 0
        ats_losses = 0
        
        # Track performance after extreme games
        after_plus7_wins = 0
        after_plus7_losses = 0
        after_plus7_ats_wins = 0
        after_plus7_ats_losses = 0
        
        after_minus7_wins = 0
        after_minus7_losses = 0
        after_minus7_ats_wins = 0
        after_minus7_ats_losses = 0
        
        # Previous game variables
        prev_expected_diff = None
        prev_actual_diff = None
        prev_score_diff = None
        prev_was_plus7 = False
        prev_was_minus7 = False
        
        print(f"\n{'Week':<6s} {'Opp':<6s} {'Result':<10s} {'ATS':<8s} {'ExpDiff':<10s} {'ActDiff':<10s} {'Luck':<8s} {'SeasonΣ':<10s} {'W-L':<8s} {'ATS':<8s} {'Post+{:<4s}'.format(str(int(threshold)))} {'Post-{:<4s}'.format(str(int(threshold)))}")
        print("-" * 110)
        
        for idx, game in team_games.iterrows():
            week = int(game['week'])
            
            # Determine if team was away or home
            is_away = game['away_abbr'] == team
            
            if is_away:
                opp = game['home_abbr']
                team_score = game['away_score']
                opp_score = game['home_score']
                team_adj_score = game['away_adj_score']
                opp_adj_score = game['home_adj_score']
                team_score_diff = game['away_score_diff']
                covered = game['away_covered']
            else:
                opp = game['away_abbr']
                team_score = game['home_score']
                opp_score = game['away_score']
                team_adj_score = game['home_adj_score']
                opp_adj_score = game['away_adj_score']
                team_score_diff = game['home_score_diff']
                covered = game['home_covered']
            
            # Calculate expected and actual score differentials
            expected_diff = team_adj_score - opp_adj_score  # What was expected
            actual_diff = team_score - opp_score  # What actually happened
            luck = actual_diff - expected_diff  # How much they over/underperformed
            
            # Game result
            won = team_score > opp_score
            
            # Update W/L records
            if won:
                wins += 1
            else:
                losses += 1
            
            if covered:
                ats_wins += 1
            else:
                ats_losses += 1
            
            # Check if this game is after extreme performance
            is_post_plus = prev_was_plus7
            is_post_minus = prev_was_minus7
            
            # Get the spread for this game (from team's perspective)
            if is_away:
                team_spread = game['consensus_spread']  # Away spread
            else:
                team_spread = -game['consensus_spread']  # Home spread (flip sign)
            
            # Track post extreme performance
            if is_post_plus:
                if won:
                    after_plus7_wins += 1
                else:
                    after_plus7_losses += 1
                
                if covered:
                    after_plus7_ats_wins += 1
                else:
                    after_plus7_ats_losses += 1
                
                # Track spread for aggregate stats
                total_after_plus_spreads.append(team_spread)
            
            if is_post_minus:
                if won:
                    after_minus7_wins += 1
                else:
                    after_minus7_losses += 1
                
                if covered:
                    after_minus7_ats_wins += 1
                else:
                    after_minus7_ats_losses += 1
                
                # Track spread for aggregate stats
                total_after_minus_spreads.append(team_spread)
            
            # Update season total
            season_score_diff_total += team_score_diff
            
            # Format output
            result_str = f"{team_score}-{opp_score}" + (" W" if won else " L")
            ats_str = "✓" if covered else "✗"
            exp_diff_str = f"{expected_diff:+.1f}"
            act_diff_str = f"{actual_diff:+.1f}"
            luck_str = f"{luck:+.1f}"
            season_sum_str = f"{season_score_diff_total:+.1f}"
            record_str = f"{wins}-{losses}"
            ats_record_str = f"{ats_wins}-{ats_losses}"
            
            # Post +/-threshold indicators
            post_plus_indicator = "Y" if is_post_plus else "-"
            post_minus_indicator = "Y" if is_post_minus else "-"
            
            print(f"{week:<6d} {opp:<6s} {result_str:<10s} {ats_str:<8s} {exp_diff_str:<10s} {act_diff_str:<10s} {luck_str:<8s} {season_sum_str:<10s} {record_str:<8s} {ats_record_str:<8s} {post_plus_indicator:<8s} {post_minus_indicator:<8s}")
            
            # Categorize spread
            abs_spread = abs(team_spread)
            if abs_spread <= 3:
                spread_category = '0-3 (close)'
            elif abs_spread <= 7:
                spread_category = '3-7 (moderate)'
            else:
                spread_category = '7+ (large)'
            
            # Determine if favorite or underdog
            is_favorite = team_spread < 0
            fav_dog = 'Favorite' if is_favorite else 'Underdog'
            
            # Combined category
            spread_fav_category = f"{spread_category} {fav_dog}"
            
            # Collect data for CSV export
            all_game_data.append({
                'team': team,
                'week': week,
                'opponent': opp,
                'is_away': is_away,
                'team_score': team_score,
                'opp_score': opp_score,
                'won': won,
                'covered': covered,
                'spread': team_spread,
                'spread_abs': abs_spread,
                'spread_category': spread_category,
                'is_favorite': is_favorite,
                'fav_dog': fav_dog,
                'spread_fav_category': spread_fav_category,
                'expected_diff': expected_diff,
                'actual_diff': actual_diff,
                'luck': luck,
                'season_luck_sum': season_score_diff_total,
                'season_wins': wins,
                'season_losses': losses,
                'season_ats_wins': ats_wins,
                'season_ats_losses': ats_losses,
                'is_post_plus_threshold': is_post_plus,
                'is_post_minus_threshold': is_post_minus,
                'after_plus_wins': after_plus7_wins,
                'after_plus_losses': after_plus7_losses,
                'after_plus_ats_wins': after_plus7_ats_wins,
                'after_plus_ats_losses': after_plus7_ats_losses,
                'after_minus_wins': after_minus7_wins,
                'after_minus_losses': after_minus7_losses,
                'after_minus_ats_wins': after_minus7_ats_wins,
                'after_minus_ats_losses': after_minus7_ats_losses,
                'game_id': game['game_id'],
                'game_time': game['game_time'],
            })
            
            # Update previous game tracking
            prev_expected_diff = expected_diff
            prev_actual_diff = actual_diff
            prev_score_diff = team_score_diff
            prev_was_plus7 = luck >= threshold
            prev_was_minus7 = luck <= -threshold
        
        # Add to aggregate totals
        total_after_plus_wins += after_plus7_wins
        total_after_plus_losses += after_plus7_losses
        total_after_plus_ats_wins += after_plus7_ats_wins
        total_after_plus_ats_losses += after_plus7_ats_losses
        
        total_after_minus_wins += after_minus7_wins
        total_after_minus_losses += after_minus7_losses
        total_after_minus_ats_wins += after_minus7_ats_wins
        total_after_minus_ats_losses += after_minus7_ats_losses
        
        # Summary
        print("\n" + "-" * 100)
        print("SUMMARY:")
        print(f"  Season Record: {wins}-{losses} ({wins/(wins+losses)*100:.1f}% win rate)")
        print(f"  ATS Record: {ats_wins}-{ats_losses} ({ats_wins/(ats_wins+ats_losses)*100:.1f}% cover rate)")
        print(f"  Season Score Diff Total: {season_score_diff_total:+.1f}")
        print(f"  Avg Score Diff per Game: {season_score_diff_total/len(team_games):+.2f}")
        
        if after_plus7_wins + after_plus7_losses > 0:
            print(f"\n  After +{threshold} Overperformance Games:")
            print(f"    Record: {after_plus7_wins}-{after_plus7_losses} ({after_plus7_wins/(after_plus7_wins+after_plus7_losses)*100:.1f}%)")
            print(f"    ATS: {after_plus7_ats_wins}-{after_plus7_ats_losses} ({after_plus7_ats_wins/(after_plus7_ats_wins+after_plus7_ats_losses)*100:.1f}%)")
        else:
            print(f"\n  After +{threshold} Overperformance Games: No games (0 +{threshold} overperformances)")
        
        if after_minus7_wins + after_minus7_losses > 0:
            print(f"\n  After -{threshold} Underperformance Games:")
            print(f"    Record: {after_minus7_wins}-{after_minus7_losses} ({after_minus7_wins/(after_minus7_wins+after_minus7_losses)*100:.1f}%)")
            print(f"    ATS: {after_minus7_ats_wins}-{after_minus7_ats_losses} ({after_minus7_ats_wins/(after_minus7_ats_wins+after_minus7_ats_losses)*100:.1f}%)")
        else:
            print(f"\n  After -{threshold} Underperformance Games: No games (0 -{threshold} underperformances)")
    
    # Save detailed tracking data to CSV
    if len(all_game_data) > 0:
        df_tracking = pd.DataFrame(all_game_data)
        
        intermediate_dir = Path("/Users/thomasmyles/dev/betting/data/03_intermediate")
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = f"nfl_game_by_game_tracking_threshold_{int(threshold)}.csv"
        output_path = intermediate_dir / output_filename
        
        df_tracking.to_csv(output_path, index=False)
        
        print(f"\n💾 Saved detailed tracking data: {output_path}")
        print(f"   Rows: {len(df_tracking)}")
        print(f"   Columns: {len(df_tracking.columns)}")
        print(f"   Teams: {df_tracking['team'].nunique()}")
    
    # Print aggregate summary if analyzing all teams
    if args.team.upper() == 'ALL':
        print("\n" + "=" * 100)
        print("LEAGUE-WIDE AGGREGATE STATISTICS")
        print("=" * 100)
        
        total_plus_games = total_after_plus_wins + total_after_plus_losses
        total_minus_games = total_after_minus_wins + total_after_minus_losses
        
        print(f"\nThreshold: ±{threshold} luck points")
        
        if args.group_by_spread:
            print("Breakdown by spread category\n")
        else:
            print("\n")
        
        if total_plus_games > 0:
            plus_win_rate = total_after_plus_wins / total_plus_games * 100
            plus_ats_rate = total_after_plus_ats_wins / total_plus_games * 100
            avg_plus_spread = np.mean(total_after_plus_spreads)
            
            print(f"📈 AFTER +{threshold} OVERPERFORMANCE (Lucky) Games:")
            print(f"   Sample size: {total_plus_games} games")
            print(f"   Win/Loss: {total_after_plus_wins}-{total_after_plus_losses} ({plus_win_rate:.1f}%)")
            print(f"   ATS: {total_after_plus_ats_wins}-{total_after_plus_ats_losses} ({plus_ats_rate:.1f}%)")
            print(f"   Avg Spread: {avg_plus_spread:+.2f} ({'favorites' if avg_plus_spread < 0 else 'underdogs'})")
            
            # ROI calculation (assuming -110 odds)
            # Break-even at 52.38%, so anything above/below shows edge
            plus_roi = ((plus_ats_rate / 100 * 1.909) - 1) * 100  # 1.909 = 210/110 payout
            print(f"   Expected ROI: {plus_roi:+.1f}% (at -110 odds)")
            
            # Interpretation
            if plus_win_rate > 55:
                print(f"   🔥 Signal: MOMENTUM (teams riding the wave)")
            elif plus_win_rate < 45:
                print(f"   📉 Signal: REVERSAL (regression to mean)")
            else:
                print(f"   ➡️  Signal: NEUTRAL (no clear trend)")
            
            if plus_ats_rate > 52.5:
                print(f"   💰 Betting: BACK them (positive ATS edge)")
            elif plus_ats_rate < 47.5:
                print(f"   💸 Betting: FADE them (negative ATS edge)")
            else:
                print(f"   ⚖️  Betting: No edge")
        else:
            print(f"📈 AFTER +{threshold} OVERPERFORMANCE: No data")
        
        print()
        
        if total_minus_games > 0:
            minus_win_rate = total_after_minus_wins / total_minus_games * 100
            minus_ats_rate = total_after_minus_ats_wins / total_minus_games * 100
            avg_minus_spread = np.mean(total_after_minus_spreads)
            
            print(f"📉 AFTER -{threshold} UNDERPERFORMANCE (Unlucky) Games:")
            print(f"   Sample size: {total_minus_games} games")
            print(f"   Win/Loss: {total_after_minus_wins}-{total_after_minus_losses} ({minus_win_rate:.1f}%)")
            print(f"   ATS: {total_after_minus_ats_wins}-{total_after_minus_ats_losses} ({minus_ats_rate:.1f}%)")
            print(f"   Avg Spread: {avg_minus_spread:+.2f} ({'favorites' if avg_minus_spread < 0 else 'underdogs'})")
            
            # ROI calculation (assuming -110 odds)
            minus_roi = ((minus_ats_rate / 100 * 1.909) - 1) * 100  # 1.909 = 210/110 payout
            print(f"   Expected ROI: {minus_roi:+.1f}% (at -110 odds)")
            
            # Interpretation
            if minus_win_rate > 55:
                print(f"   📈 Signal: BOUNCE BACK (unlucky teams revert up)")
            elif minus_win_rate < 45:
                print(f"   📉 Signal: CONTINUED STRUGGLE (negative momentum)")
            else:
                print(f"   ➡️  Signal: NEUTRAL (no clear trend)")
            
            if minus_ats_rate > 52.5:
                print(f"   💰 Betting: BACK them (positive ATS edge)")
            elif minus_ats_rate < 47.5:
                print(f"   💸 Betting: FADE them (negative ATS edge)")
            else:
                print(f"   ⚖️  Betting: No edge")
        else:
            print(f"📉 AFTER -{threshold} UNDERPERFORMANCE: No data")
        
        print("\n" + "-" * 100)
        print("INTERPRETATION GUIDE:")
        print("  • Win rate ~50% = neutral (no momentum or reversal)")
        print("  • Win rate >55% = momentum/continuation signal")
        print("  • Win rate <45% = reversal/regression signal")
        print("  • ATS >52.5% = profitable betting angle")
        print("  • ATS <47.5% = fade opportunity")
        
        # Spread category breakdown if requested
        if args.group_by_spread and len(all_game_data) > 0:
            print("\n" + "=" * 100)
            print("BREAKDOWN BY SPREAD CATEGORY")
            print("=" * 100)
            
            df_all = pd.DataFrame(all_game_data)
            
            # After +threshold breakdown
            df_plus = df_all[df_all['is_post_plus_threshold'] == True]
            if len(df_plus) > 0:
                if args.include_fav_dog:
                    print(f"\n📈 AFTER +{threshold} OVERPERFORMANCE BY SPREAD + FAV/DOG:")
                    print(f"{'Spread + Role':<30s} {'Sample':<10s} {'ATS Record':<15s} {'ATS%':<10s} {'ROI':<10s} {'Avg Spread':<12s}")
                    print("-" * 90)
                    
                    # Show by spread range, then split by fav/dog
                    for spread_range in ['0-3 (close)', '3-7 (moderate)', '7+ (large)']:
                        for fav_dog in ['Favorite', 'Underdog']:
                            category = f"{spread_range} {fav_dog}"
                            df_cat = df_plus[df_plus['spread_fav_category'] == category]
                            
                            if len(df_cat) > 0:
                                ats_wins = df_cat['covered'].sum()
                                ats_total = len(df_cat)
                                ats_pct = ats_wins / ats_total * 100
                                roi = ((ats_pct / 100 * 1.909) - 1) * 100
                                avg_spread = df_cat['spread'].mean()
                                record_str = f"{ats_wins}-{ats_total-ats_wins}"
                                
                                print(f"{category:<30s} {ats_total:<10d} {record_str:<15s} {ats_pct:<10.1f} {roi:+<10.1f} {avg_spread:+<12.2f}")
                            else:
                                print(f"{category:<30s} {'0':<10s} {'-':<15s} {'-':<10s} {'-':<10s} {'-':<12s}")
                else:
                    print(f"\n📈 AFTER +{threshold} OVERPERFORMANCE BY SPREAD:")
                    print(f"{'Spread Range':<20s} {'Sample':<10s} {'ATS Record':<15s} {'ATS%':<10s} {'ROI':<10s} {'Avg Spread':<12s}")
                    print("-" * 80)
                    
                    for category in ['0-3 (close)', '3-7 (moderate)', '7+ (large)']:
                        df_cat = df_plus[df_plus['spread_category'] == category]
                        if len(df_cat) > 0:
                            ats_wins = df_cat['covered'].sum()
                            ats_total = len(df_cat)
                            ats_pct = ats_wins / ats_total * 100
                            roi = ((ats_pct / 100 * 1.909) - 1) * 100
                            avg_spread = df_cat['spread'].mean()
                            record_str = f"{ats_wins}-{ats_total-ats_wins}"
                            
                            print(f"{category:<20s} {ats_total:<10d} {record_str:<15s} {ats_pct:<10.1f} {roi:+<10.1f} {avg_spread:+<12.2f}")
                        else:
                            print(f"{category:<20s} {'0':<10s} {'-':<15s} {'-':<10s} {'-':<10s} {'-':<12s}")
            
            # After -threshold breakdown
            df_minus = df_all[df_all['is_post_minus_threshold'] == True]
            if len(df_minus) > 0:
                if args.include_fav_dog:
                    print(f"\n📉 AFTER -{threshold} UNDERPERFORMANCE BY SPREAD + FAV/DOG:")
                    print(f"{'Spread + Role':<30s} {'Sample':<10s} {'ATS Record':<15s} {'ATS%':<10s} {'ROI':<10s} {'Avg Spread':<12s}")
                    print("-" * 90)
                    
                    # Show by spread range, then split by fav/dog
                    for spread_range in ['0-3 (close)', '3-7 (moderate)', '7+ (large)']:
                        for fav_dog in ['Favorite', 'Underdog']:
                            category = f"{spread_range} {fav_dog}"
                            df_cat = df_minus[df_minus['spread_fav_category'] == category]
                            
                            if len(df_cat) > 0:
                                ats_wins = df_cat['covered'].sum()
                                ats_total = len(df_cat)
                                ats_pct = ats_wins / ats_total * 100
                                roi = ((ats_pct / 100 * 1.909) - 1) * 100
                                avg_spread = df_cat['spread'].mean()
                                record_str = f"{ats_wins}-{ats_total-ats_wins}"
                                
                                print(f"{category:<30s} {ats_total:<10d} {record_str:<15s} {ats_pct:<10.1f} {roi:+<10.1f} {avg_spread:+<12.2f}")
                            else:
                                print(f"{category:<30s} {'0':<10s} {'-':<15s} {'-':<10s} {'-':<10s} {'-':<12s}")
                else:
                    print(f"\n📉 AFTER -{threshold} UNDERPERFORMANCE BY SPREAD:")
                    print(f"{'Spread Range':<20s} {'Sample':<10s} {'ATS Record':<15s} {'ATS%':<10s} {'ROI':<10s} {'Avg Spread':<12s}")
                    print("-" * 80)
                    
                    for category in ['0-3 (close)', '3-7 (moderate)', '7+ (large)']:
                        df_cat = df_minus[df_minus['spread_category'] == category]
                        if len(df_cat) > 0:
                            ats_wins = df_cat['covered'].sum()
                            ats_total = len(df_cat)
                            ats_pct = ats_wins / ats_total * 100
                            roi = ((ats_pct / 100 * 1.909) - 1) * 100
                            avg_spread = df_cat['spread'].mean()
                            record_str = f"{ats_wins}-{ats_total-ats_wins}"
                            
                            print(f"{category:<20s} {ats_total:<10d} {record_str:<15s} {ats_pct:<10.1f} {roi:+<10.1f} {avg_spread:+<12.2f}")
                        else:
                            print(f"{category:<20s} {'0':<10s} {'-':<15s} {'-':<10s} {'-':<10s} {'-':<12s}")
            
            print("\n" + "-" * 100)
            if args.include_fav_dog:
                print("HYPOTHESIS TEST:")
                print("  ✓ Unlucky favorites vs unlucky underdogs - which bounces back more?")
                print("  ✓ Lucky favorites vs lucky underdogs - true dominance or just variance?")
                print("  ✓ Key insight: Separate 'bad team on a bad day' from 'good team got unlucky'")
            else:
                print("HYPOTHESIS TEST:")
                print("  ✓ If close games (0-3) have best ROI → Only bet when near pick'em")
                print("  ✓ If large spreads (7+) have negative ROI → Avoid huge underdogs")
                print("  ✓ Compare to validate: unlucky teams bounce back ONLY when not massive dogs")
                print(f"\n💡 Add --include-fav-dog to split by favorite/underdog for deeper insights")
    
    print("\n" + "=" * 100)
    print("✅ TREND OBSERVATION COMPLETE")
    print("=" * 100)


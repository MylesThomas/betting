"""
Build NCAAB Team Name Mapping (ESPN → Odds API) - V2

Returns single dataframe with:
- team_name_espn: ESPN team name
- team_name_odds_api: Matched Odds API name (null if needs manual review)
- match_type: 'exact', 'fuzzy', or 'no_match'
- potential_matches_odds_api: Fuzzy matches with confidence scores

Usage in notebook:
```python
from tmp.build_ncaab_team_name_mapping_v2 import get_team_mapping_dfs

df = get_team_mapping_dfs(season='2024-25', use_cache=True)

# Filter to teams needing manual review
needs_review = df[df['team_name_odds_api'].isna()]
display(needs_review)

# For each row, decide if fuzzy match is correct or find the right match
# Then update team_name_odds_api column
```

CLI Usage:
```bash
# Step 1: Clear old cache and rebuild
rm -rf ~/Downloads/tmp/ncaab_cache/
python tmp/build_ncaab_team_name_mapping_v2.py --season 2024-25

# Step 2: Analyze the cache (shows team stats, games played, wins, etc.)
python tmp/build_ncaab_team_name_mapping_v2.py --season 2024-25 --analyze-cache

# Step 3: Use cache for fast iteration
python tmp/build_ncaab_team_name_mapping_v2.py --season 2024-25 --use-cache
```

Author: Thomas Myles  
Date: 2026-01-15
"""

import sys
import pandas as pd
from pathlib import Path

# Setup paths
def find_project_root():
    """Find project root by looking for .gitignore file."""
    current = Path.cwd()
    while current != current.parent:
        if (current / '.gitignore').exists():
            return current
        current = current.parent
    return Path.cwd()

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'tmp'))

from join_ncaab_outcomes_and_lines import (
    load_game_outcomes, 
    load_game_lines, 
    SEASON_DATES,
    get_cache_path
)
from difflib import SequenceMatcher


def calculate_similarity(s1, s2):
    """Calculate similarity ratio between two strings (0-100)."""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio() * 100


def find_fuzzy_matches(espn_team, odds_teams, threshold=60, top_n=3):
    """Find top N fuzzy matches for an ESPN team from odds teams."""
    matches = []
    for odds_team in odds_teams:
        similarity = calculate_similarity(espn_team, odds_team)
        if similarity >= threshold:
            matches.append((odds_team, similarity))
    
    # Sort by similarity (highest first) and return top N
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:top_n]


def analyze_cache(season='2024-25'):
    """
    Analyze cached outcomes data and show team statistics.
    
    Args:
        season: Season string (e.g., '2024-25')
    
    Returns:
        None (prints analysis)
    """
    if season not in SEASON_DATES:
        raise ValueError(f"Unknown season: {season}. Available: {list(SEASON_DATES.keys())}")
    
    start_date, end_date = SEASON_DATES[season]
    
    # Check if cache exists
    cache_path_outcomes = get_cache_path('outcomes', start_date, end_date)
    cache_path_lines = get_cache_path('lines', start_date, end_date)
    
    if not cache_path_outcomes.exists():
        print(f"❌ Cache not found for {season}")
        print(f"   Expected: {cache_path_outcomes}")
        print(f"\n💡 Run without --analyze-cache first to build cache:")
        print(f"   python tmp/build_ncaab_team_name_mapping_v2.py --season {season}")
        return
    
    print(f"{'='*80}")
    print(f"CACHE ANALYSIS: {season}")
    print(f"{'='*80}")
    print(f"Cache location: {cache_path_outcomes.parent}")
    
    # Load cached data
    df_outcomes = pd.read_parquet(cache_path_outcomes)
    
    unique_teams = pd.concat([df_outcomes['HOME_TEAM'], df_outcomes['AWAY_TEAM']]).nunique()
    
    print(f"\n📊 Outcomes Cache:")
    print(f"   Total games: {len(df_outcomes):,}")
    print(f"   Date range: {df_outcomes['GAME_DATE'].min()} to {df_outcomes['GAME_DATE'].max()}")
    print(f"   Unique teams: {unique_teams}")
    print(f"   Avg games per team: {len(df_outcomes) * 2 / unique_teams:.1f}")
    
    # Team stats
    print(f"\n🏀 Team Statistics:")
    
    # Count games per team (both home and away)
    home_games = df_outcomes.groupby('HOME_TEAM').size()
    away_games = df_outcomes.groupby('AWAY_TEAM').size()
    total_games = (home_games.add(away_games, fill_value=0)).sort_values(ascending=False)
    
    print(f"\n   Top 10 Teams by Games Played:")
    for i, (team, games) in enumerate(total_games.head(10).items(), 1):
        print(f"      {i:2}. {team:<35} {int(games):3} games")
    
    # Win statistics (if available)
    if 'HOME_WL' in df_outcomes.columns and 'AWAY_WL' in df_outcomes.columns:
        # Home wins
        home_wins = df_outcomes[df_outcomes['HOME_WL'] == 'W'].groupby('HOME_TEAM').size()
        # Away wins
        away_wins = df_outcomes[df_outcomes['AWAY_WL'] == 'W'].groupby('AWAY_TEAM').size()
        # Total wins
        total_wins = (home_wins.add(away_wins, fill_value=0)).sort_values(ascending=False)
        
        # Calculate win percentage
        win_pct = (total_wins / total_games * 100).sort_values(ascending=False)
        
        print(f"\n   Top 10 Teams by Wins:")
        for i, (team, wins) in enumerate(total_wins.head(10).items(), 1):
            games = int(total_games[team])
            pct = win_pct[team]
            print(f"      {i:2}. {team:<35} {int(wins):3} wins / {games:3} games ({pct:.1f}%)")
        
        # Teams with most games and high win rate (min 20 games)
        qualified = total_games[total_games >= 20]
        qualified_win_pct = win_pct[qualified.index].sort_values(ascending=False)
        
        print(f"\n   Top 10 Teams by Win % (min 20 games):")
        for i, (team, pct) in enumerate(qualified_win_pct.head(10).items(), 1):
            games = int(total_games[team])
            wins = int(total_wins[team])
            print(f"      {i:2}. {team:<35} {pct:5.1f}% ({wins:2}-{games-wins:2})")
    
    # Check lines cache if exists
    if cache_path_lines.exists():
        df_lines = pd.read_parquet(cache_path_lines)
        
        # Get distinct teams from lines
        lines_teams = pd.concat([df_lines['home_team'], df_lines['away_team']]).nunique()
        
        print(f"\n📊 Lines Cache:")
        print(f"   Total game lines: {len(df_lines):,}")
        print(f"   Date range: {df_lines['date'].min()} to {df_lines['date'].max()}")
        print(f"   Unique teams in lines: {lines_teams}")
        print(f"   Avg games per team: {len(df_lines) * 2 / lines_teams:.1f}")
        print(f"   Coverage: {len(df_lines) / len(df_outcomes) * 100:.1f}% of games have lines")
    
    print(f"\n{'='*80}")
    print("✅ Cache analysis complete!")
    print(f"{'='*80}")


def get_d1_team_matching_df(season='2024-25', use_cache=True, min_games=5):
    """
    Get dataframe of D1 team matching status with W/L and ATS stats.
    
    Returns DataFrame with columns:
    - team_name_espn: ESPN team name (original)
    - team_name_odds_api: Odds API team name (or None if not matched)
    - matched: True if matched, False otherwise
    - espn_games: Total games played
    - espn_wins: Total wins
    - espn_losses: Total losses
    - espn_win_pct: Win percentage
    - ats_games: Games with ATS results
    - ats_wins: ATS wins
    - ats_losses: ATS losses
    - ats_pushes: ATS pushes
    - ats_win_pct: ATS win percentage
    
    Args:
        season: Season string (e.g., '2024-25')
        use_cache: Whether to use cached data
        min_games: Minimum games for a team to be considered D1
    
    Returns:
        DataFrame with D1 teams, matching status, and stats
    
    Usage:
        df = get_d1_team_matching_df(season='2024-25', use_cache=True)
        
        # Filter to unmatched teams
        unmatched = df[~df['matched']]
        
        # Sort by win percentage
        df.sort_values('espn_win_pct', ascending=False)
    """
    from join_ncaab_outcomes_and_lines import normalize_team_name
    from config_loader import get_config
    
    CONFIG = get_config()
    non_d1_teams = set(CONFIG.get('ncaab_non_d1_teams', []))
    
    if season not in SEASON_DATES:
        raise ValueError(f"Unknown season: {season}. Available: {list(SEASON_DATES.keys())}")
    
    start_date, end_date = SEASON_DATES[season]
    
    print(f"Loading data for {season}...")
    
    # Load data
    outcomes_df = load_game_outcomes(start_date, end_date, use_cache=use_cache)
    lines_df = load_game_lines(start_date, end_date, use_cache=use_cache)
    
    # Filter to D1 teams only (≥min_games AND not in exclusion list)
    home_counts = outcomes_df.groupby('HOME_TEAM').size()
    away_counts = outcomes_df.groupby('AWAY_TEAM').size()
    total_counts = home_counts.add(away_counts, fill_value=0)
    
    # Apply filters: min_games AND not in non-D1 list
    d1_teams = set(
        team for team in total_counts[total_counts >= min_games].index
        if team not in non_d1_teams
    )
    
    print(f"Excluded non-D1 teams: {len(non_d1_teams)}")
    
    # Get unique teams from odds API
    odds_teams = {}  # normalized -> original name mapping
    for team in pd.concat([lines_df['home_team'], lines_df['away_team']]).unique():
        normalized = normalize_team_name(team)
        odds_teams[normalized] = team
    
    print(f"ESPN D1 teams (≥{min_games} games): {len(d1_teams)}")
    print(f"Odds API teams: {len(odds_teams)}")
    
    # Calculate ESPN W/L stats
    espn_stats = {}
    for team in d1_teams:
        # Home games
        home_games = outcomes_df[outcomes_df['HOME_TEAM'] == team]
        home_wins = (home_games['HOME_WL'] == 'W').sum()
        home_losses = (home_games['HOME_WL'] == 'L').sum()
        
        # Away games
        away_games = outcomes_df[outcomes_df['AWAY_TEAM'] == team]
        away_wins = (away_games['AWAY_WL'] == 'W').sum()
        away_losses = (away_games['AWAY_WL'] == 'L').sum()
        
        total_wins = home_wins + away_wins
        total_losses = home_losses + away_losses
        total_games = total_wins + total_losses
        win_pct = total_wins / total_games if total_games > 0 else 0
        
        espn_stats[team] = {
            'games': total_games,
            'wins': total_wins,
            'losses': total_losses,
            'win_pct': win_pct
        }
    
    # Calculate ATS stats by joining outcomes and lines
    print("Calculating ATS stats...")
    
    # Normalize team names for joining
    outcomes_df = outcomes_df.copy()
    outcomes_df['home_norm'] = outcomes_df['HOME_TEAM'].apply(normalize_team_name)
    outcomes_df['away_norm'] = outcomes_df['AWAY_TEAM'].apply(normalize_team_name)
    
    lines_df = lines_df.copy()
    lines_df['home_norm'] = lines_df['home_team'].apply(normalize_team_name)
    lines_df['away_norm'] = lines_df['away_team'].apply(normalize_team_name)
    
    # Join outcomes with lines
    joined = outcomes_df.merge(
        lines_df,
        left_on=['GAME_DATE', 'home_norm', 'away_norm'],
        right_on=['date', 'home_norm', 'away_norm'],
        how='inner',
        suffixes=('', '_line')
    )
    
    # Calculate ATS results
    # Home team perspective: (home_score - away_score) + spread > 0 means cover
    # Away team perspective: (away_score - home_score) - spread > 0 means cover
    joined['home_ats_margin'] = (joined['HOME_SCORE'] - joined['AWAY_SCORE']) + joined['consensus_spread']
    joined['away_ats_margin'] = (joined['AWAY_SCORE'] - joined['HOME_SCORE']) - joined['consensus_spread']
    
    # Classify ATS results (use small epsilon for pushes)
    epsilon = 0.01
    joined['home_ats_result'] = joined['home_ats_margin'].apply(
        lambda x: 'W' if x > epsilon else ('L' if x < -epsilon else 'P')
    )
    joined['away_ats_result'] = joined['away_ats_margin'].apply(
        lambda x: 'W' if x > epsilon else ('L' if x < -epsilon else 'P')
    )
    
    # Calculate ATS stats per team (by normalized name)
    ats_stats = {}
    
    # Process home games
    home_ats = joined.groupby('home_norm')['home_ats_result'].value_counts().unstack(fill_value=0)
    for norm_team in home_ats.index:
        ats_stats[norm_team] = {
            'ats_wins': home_ats.loc[norm_team].get('W', 0),
            'ats_losses': home_ats.loc[norm_team].get('L', 0),
            'ats_pushes': home_ats.loc[norm_team].get('P', 0)
        }
    
    # Add away games
    away_ats = joined.groupby('away_norm')['away_ats_result'].value_counts().unstack(fill_value=0)
    for norm_team in away_ats.index:
        if norm_team not in ats_stats:
            ats_stats[norm_team] = {'ats_wins': 0, 'ats_losses': 0, 'ats_pushes': 0}
        ats_stats[norm_team]['ats_wins'] += away_ats.loc[norm_team].get('W', 0)
        ats_stats[norm_team]['ats_losses'] += away_ats.loc[norm_team].get('L', 0)
        ats_stats[norm_team]['ats_pushes'] += away_ats.loc[norm_team].get('P', 0)
    
    # Calculate totals and win percentage
    for norm_team in ats_stats:
        wins = ats_stats[norm_team]['ats_wins']
        losses = ats_stats[norm_team]['ats_losses']
        pushes = ats_stats[norm_team]['ats_pushes']
        total = wins + losses + pushes
        
        ats_stats[norm_team]['ats_games'] = total
        ats_stats[norm_team]['ats_win_pct'] = wins / (wins + losses) if (wins + losses) > 0 else 0.0
    
    # Build matching dataframe
    results = []
    
    for espn_team in sorted(d1_teams):
        # Normalize ESPN team name
        espn_normalized = normalize_team_name(espn_team)
        
        # Check if normalized match exists in odds API
        if espn_normalized in odds_teams:
            matched_team = odds_teams[espn_normalized]
            is_matched = True
            ats_data = ats_stats.get(espn_normalized, {})
        else:
            matched_team = None
            is_matched = False
            ats_data = {}
        
        # Get ESPN stats
        stats = espn_stats.get(espn_team, {})
        
        results.append({
            'team_name_espn': espn_team,
            'team_name_odds_api': matched_team,
            'matched': is_matched,
            'espn_games': stats.get('games', 0),
            'espn_wins': stats.get('wins', 0),
            'espn_losses': stats.get('losses', 0),
            'espn_win_pct': stats.get('win_pct', 0.0),
            'ats_games': ats_data.get('ats_games', 0),
            'ats_wins': ats_data.get('ats_wins', 0),
            'ats_losses': ats_data.get('ats_losses', 0),
            'ats_pushes': ats_data.get('ats_pushes', 0),
            'ats_win_pct': ats_data.get('ats_win_pct', 0.0)
        })
    
    df = pd.DataFrame(results)
    
    # Summary
    matched_count = df['matched'].sum()
    unmatched_count = len(df) - matched_count
    
    print(f"\nResults:")
    print(f"   Total D1 teams: {len(df)}")
    print(f"   Matched: {matched_count} ({matched_count/len(df)*100:.1f}%)")
    print(f"   Unmatched: {unmatched_count} ({unmatched_count/len(df)*100:.1f}%)")
    
    return df


def get_team_mapping_dfs(season='2024-25', use_cache=True, fuzzy_threshold=60):
    """
    Get team mapping dataframe with fuzzy matches for manual review.
    
    Args:
        season: Season string (e.g., '2024-25')
        use_cache: Whether to use cached data
        fuzzy_threshold: Minimum similarity score (0-100) for fuzzy matches
    
    Returns:
        df: DataFrame with columns:
            - team_name_espn: ESPN team name
            - team_name_odds_api: Matched Odds API name (null if no exact match)
            - potential_matches_odds_api: List of fuzzy matches with scores
    """
    
    if season not in SEASON_DATES:
        raise ValueError(f"Unknown season: {season}. Available: {list(SEASON_DATES.keys())}")
    
    start_date, end_date = SEASON_DATES[season]
    
    print(f"Loading data for {season} ({start_date} to {end_date})...")
    
    # Load data
    outcomes_df = load_game_outcomes(start_date, end_date, use_cache=use_cache)
    lines_df = load_game_lines(start_date, end_date, use_cache=use_cache)
    
    print(f"\n📊 Data loaded:")
    print(f"   Outcomes: {len(outcomes_df)} games")
    print(f"   Lines: {len(lines_df)} games")
    print(f"   💡 Lines > Outcomes because lines include FUTURE games not yet played")
    
    # Get unique teams from each source
    espn_home = outcomes_df[['GAME_DATE', 'HOME_TEAM']].copy()
    espn_home.columns = ['date', 'team']
    espn_away = outcomes_df[['GAME_DATE', 'AWAY_TEAM']].copy()
    espn_away.columns = ['date', 'team']
    espn_all = pd.concat([espn_home, espn_away], ignore_index=True)
    
    odds_home = lines_df[['date', 'home_team']].copy()
    odds_home.columns = ['date', 'team']
    odds_away = lines_df[['date', 'away_team']].copy()
    odds_away.columns = ['date', 'team']
    odds_all = pd.concat([odds_home, odds_away], ignore_index=True)
    
    # Get unique team lists
    espn_teams = sorted(espn_all['team'].unique())
    odds_teams = sorted(odds_all['team'].unique())
    
    print(f"\n📋 Unique teams:")
    print(f"   ESPN: {len(espn_teams)}")
    print(f"   Odds API: {len(odds_teams)}")
    
    # Join on date + team name to find exact matches
    joined = espn_all.merge(
        odds_all,
        on=['date', 'team'],
        how='left'
    )
    
    # Get teams that have at least one exact match
    matched_teams = set(joined[joined['team'].notna()]['team'].unique())
    
    # Build result dataframe
    results = []
    
    print(f"\n🔍 Finding fuzzy matches (threshold={fuzzy_threshold}%)...")
    
    for espn_team in espn_teams:
        if espn_team in matched_teams:
            # Exact match found
            results.append({
                'team_name_espn': espn_team,
                'team_name_odds_api': espn_team,  # Same name
                'match_type': 'exact',
                'potential_matches_odds_api': None
            })
        else:
            # No exact match - find fuzzy matches
            fuzzy = find_fuzzy_matches(espn_team, odds_teams, threshold=fuzzy_threshold)
            
            if fuzzy:
                match_str = '; '.join([f"{name} ({score:.1f}%)" for name, score in fuzzy])
            else:
                match_str = None
            
            results.append({
                'team_name_espn': espn_team,
                'team_name_odds_api': None,
                'match_type': 'fuzzy' if fuzzy else 'no_match',
                'potential_matches_odds_api': match_str
            })
    
    df = pd.DataFrame(results)
    
    # Summary stats
    exact_matches = (df['match_type'] == 'exact').sum()
    fuzzy_available = (df['match_type'] == 'fuzzy').sum()
    no_matches = (df['match_type'] == 'no_match').sum()
    
    print(f"\n✅ Results:")
    print(f"   Exact matches: {exact_matches}")
    print(f"   Need manual review (have fuzzy matches): {fuzzy_available}")
    print(f"   No matches found: {no_matches}")
    print(f"\n💡 Review rows where team_name_odds_api is null to complete the mapping")
    
    return df


# CLI execution
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=str, default='2024-25')
    parser.add_argument('--use-cache', action='store_true')
    parser.add_argument('--analyze-cache', action='store_true',
                       help='Analyze cached data and show team statistics')
    args = parser.parse_args()
    
    # If analyze-cache flag, run analysis and exit
    if args.analyze_cache:
        analyze_cache(season=args.season)
    else:
        df = get_team_mapping_dfs(
            season=args.season,
            use_cache=args.use_cache,
            fuzzy_threshold=60
        )
        
        print(f"\n{'='*80}")
        print("TEAM MAPPING RESULTS")
        print(f"{'='*80}")
        
        # Show teams needing manual review
        needs_review = df[df['team_name_odds_api'].isna()].copy()
        
        if len(needs_review) > 0:
            print(f"\n⚠️  {len(needs_review)} teams need manual review:\n")
            print(needs_review.to_string(index=False, max_colwidth=80))
        else:
            print("\n✅ All teams matched!")
        
        # Show sample of exact matches
        exact = df[df['match_type'] == 'exact'].head(10)
        print(f"\n✅ Sample of exact matches:")
        print(exact[['team_name_espn', 'team_name_odds_api']].to_string(index=False))


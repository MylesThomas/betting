"""
Analyze vig in NFL Super Bowl futures markets.

Purpose:
- Read most recent NFL championship futures
- Calculate implied probabilities from odds
- Calculate total vig across all teams
- Remove vig to get true implied probabilities
- Show how much sportsbooks take on futures

Context:
Futures markets typically have much higher vig than game lines.
Game lines: ~4-5% vig
Futures: Often 20-40%+ vig

Usage:
    cd /Users/thomasmyles/dev/betting
    python3 analysis/analyze_nfl_championship_futures_vig.py
"""

import pandas as pd
import os
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from odds_utils import odds_to_implied_probability, probability_to_american_odds


def get_all_nfl_teams_with_records():
    """
    Get all 32 NFL teams with their current W-L records from Unexpected Points data.
    
    Reads the most recent Unexpected Points Excel file and calculates W-L records
    from the 2025 Adjusted Scores sheet.
    """
    # Team abbreviation to full name mapping
    team_name_map = {
        'ARI': 'Arizona Cardinals',
        'ATL': 'Atlanta Falcons',
        'BAL': 'Baltimore Ravens',
        'BUF': 'Buffalo Bills',
        'CAR': 'Carolina Panthers',
        'CHI': 'Chicago Bears',
        'CIN': 'Cincinnati Bengals',
        'CLE': 'Cleveland Browns',
        'DAL': 'Dallas Cowboys',
        'DEN': 'Denver Broncos',
        'DET': 'Detroit Lions',
        'GB': 'Green Bay Packers',
        'HOU': 'Houston Texans',
        'IND': 'Indianapolis Colts',
        'JAX': 'Jacksonville Jaguars',
        'KC': 'Kansas City Chiefs',
        'LA': 'Los Angeles Rams',
        'LAC': 'Los Angeles Chargers',
        'LV': 'Las Vegas Raiders',
        'MIA': 'Miami Dolphins',
        'MIN': 'Minnesota Vikings',
        'NE': 'New England Patriots',
        'NO': 'New Orleans Saints',
        'NYG': 'New York Giants',
        'NYJ': 'New York Jets',
        'PHI': 'Philadelphia Eagles',
        'PIT': 'Pittsburgh Steelers',
        'SEA': 'Seattle Seahawks',
        'SF': 'San Francisco 49ers',
        'TB': 'Tampa Bay Buccaneers',
        'TEN': 'Tennessee Titans',
        'WAS': 'Washington Commanders',
    }
    
    # Find most recent unexpected points file (by modification time)
    up_dir = repo_root / 'data/01_input/unexpected_points'
    up_files = list(up_dir.glob('Unexpected Points Subscriber Data*.xlsx'))
    
    if not up_files:
        raise FileNotFoundError(f"No Unexpected Points files found in {up_dir}")
    
    # Sort by modification time (most recent first)
    up_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    most_recent = up_files[0]
    print(f"📊 Reading team records from: {most_recent.name}")
    
    # Read 2025 Adjusted Scores
    df = pd.read_excel(most_recent, sheet_name='2025 Adjusted Scores')
    
    # Build game results
    games = []
    for game_id in df['game_id'].unique():
        game_df = df[df['game_id'] == game_id]
        if len(game_df) == 2:
            team1, team2 = game_df.iloc[0], game_df.iloc[1]
            games.append({
                'team1': team1['team'],
                'score1': team1['score'],
                'team2': team2['team'],
                'score2': team2['score']
            })
    
    # Calculate W-L-T records
    records = {abbr: {'wins': 0, 'losses': 0, 'ties': 0} for abbr in team_name_map.keys()}
    
    for game in games:
        team1, team2 = game['team1'], game['team2']
        score1, score2 = game['score1'], game['score2']
        
        if score1 > score2:
            records[team1]['wins'] += 1
            records[team2]['losses'] += 1
        elif score2 > score1:
            records[team2]['wins'] += 1
            records[team1]['losses'] += 1
        else:  # Tie game
            records[team1]['ties'] += 1
            records[team2]['ties'] += 1
    
    # Convert to DataFrame with full team names
    records_list = []
    for abbr, rec in records.items():
        full_name = team_name_map[abbr]
        # Format: W-L-T, but omit -T if no ties
        if rec['ties'] > 0:
            record_str = f"{rec['wins']}-{rec['losses']}-{rec['ties']}"
        else:
            record_str = f"{rec['wins']}-{rec['losses']}"
        records_list.append({'team': full_name, 'record': record_str})
    
    return pd.DataFrame(records_list)


def get_most_recent_futures_file():
    """Get the most recent NFL Super Bowl futures CSV file"""
    futures_dir = repo_root / 'data/01_input/the-odds-api/nfl/futures'
    
    # Get all CSV files
    csv_files = sorted(futures_dir.glob('nfl_super_bowl_futures_*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No futures files found in {futures_dir}")
    
    # Return most recent (last in sorted list)
    most_recent = csv_files[-1]
    print(f"📁 Reading: {most_recent.name}\n")
    return most_recent


def calculate_vig_by_bookmaker(df):
    """Calculate vig for each bookmaker"""
    results = []
    
    for bookmaker in df['bookmaker'].unique():
        bm_df = df[df['bookmaker'] == bookmaker].copy()
        
        # Calculate implied probability for each team
        bm_df['implied_prob'] = bm_df['odds'].apply(odds_to_implied_probability)
        
        # Sum all implied probabilities (should be > 1.0 due to vig)
        total_implied = bm_df['implied_prob'].sum()
        
        # Vig is the excess over 100%
        vig = total_implied - 1.0
        vig_pct = vig * 100
        
        results.append({
            'bookmaker': bookmaker,
            'num_teams': len(bm_df),
            'total_implied_prob': total_implied,
            'vig': vig,
            'vig_pct': vig_pct
        })
    
    return pd.DataFrame(results).sort_values('vig_pct', ascending=False)


def calculate_fair_probabilities(df, bookmaker):
    """Calculate fair (vig-free) probabilities for a bookmaker"""
    bm_df = df[df['bookmaker'] == bookmaker].copy()
    
    # Calculate implied probabilities
    bm_df['implied_prob'] = bm_df['odds'].apply(odds_to_implied_probability)
    
    # Sum of all implied probabilities
    total_implied = bm_df['implied_prob'].sum()
    
    # Fair probability = normalize to sum to 1.0
    bm_df['fair_prob'] = bm_df['implied_prob'] / total_implied
    
    # Vig on each team (how much extra probability bookmaker added)
    bm_df['vig_amount'] = bm_df['implied_prob'] - bm_df['fair_prob']
    
    return bm_df[['team', 'odds', 'implied_prob', 'fair_prob', 'vig_amount']].sort_values('fair_prob', ascending=False)


def main():
    """Analyze NFL championship futures vig"""
    
    print("="*80)
    print("NFL SUPER BOWL FUTURES VIG ANALYSIS")
    print("="*80 + "\n")
    
    # Read most recent futures file
    futures_file = get_most_recent_futures_file()
    df = pd.read_csv(futures_file)
    
    print(f"📊 Total odds entries: {len(df)}")
    print(f"🏈 Unique teams: {df['team'].nunique()}")
    print(f"📚 Bookmakers: {df['bookmaker'].nunique()}")
    print(f"   {', '.join(df['bookmaker'].unique())}\n")
    
    # Calculate vig by bookmaker
    print("="*80)
    print("VIG BY BOOKMAKER")
    print("="*80 + "\n")
    
    vig_df = calculate_vig_by_bookmaker(df)
    
    print(f"{'Bookmaker':<20} {'Teams':<8} {'Total Implied':<15} {'Vig %':<10}")
    print("-"*80)
    for _, row in vig_df.iterrows():
        print(f"{row['bookmaker']:<20} {row['num_teams']:<8} "
              f"{row['total_implied_prob']:>6.4f} ({row['total_implied_prob']*100:>6.2f}%)  "
              f"{row['vig_pct']:>6.2f}%")
    
    avg_vig = vig_df['vig_pct'].mean()
    print("-"*80)
    print(f"{'AVERAGE VIG':<20} {'':<8} {'':<15} {avg_vig:>6.2f}%\n")
    
    # Show fair probabilities for bookmaker with lowest vig
    best_bookmaker = vig_df.iloc[-1]['bookmaker']  # Last row = lowest vig
    
    print("="*80)
    print(f"FAIR PROBABILITIES (VIG REMOVED) - {best_bookmaker.upper()}")
    print("="*80 + "\n")
    
    fair_df = calculate_fair_probabilities(df, best_bookmaker)
    
    print(f"{'Rank':<6} {'Team':<35} {'Odds':<10} {'Implied %':<12} {'Fair %':<12}")
    print("-"*80)
    for i, row in fair_df.head(15).iterrows():
        odds_str = f"{row['odds']:+.0f}" if row['odds'] > 0 else f"{row['odds']:.0f}"
        print(f"{fair_df.index.get_loc(i)+1:<6} {row['team']:<35} {odds_str:<10} "
              f"{row['implied_prob']*100:>6.2f}%      {row['fair_prob']*100:>6.2f}%")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print(f"\n1. Average vig across all bookmakers: {avg_vig:.2f}%")
    print(f"2. Bookmaker with best (lowest) vig: {best_bookmaker} ({vig_df.iloc[-1]['vig_pct']:.2f}%)")
    print(f"3. Bookmaker with worst (highest) vig: {vig_df.iloc[0]['bookmaker']} ({vig_df.iloc[0]['vig_pct']:.2f}%)")
    print(f"\n💡 For comparison:")
    print(f"   - Game lines (spread/total): ~4-5% vig")
    print(f"   - Futures markets: {avg_vig:.1f}% vig")
    print(f"   - Futures markets have {avg_vig/4.5:.1f}x more vig than game lines!")
    
    # Show what this means for a bettor
    print(f"\n📉 What this means:")
    print(f"   If all teams had equal odds, sportsbooks would win {avg_vig:.1f}% of all bets")
    print(f"   To break even, bettors need to be {avg_vig:.1f}% better than 'fair' odds")
    
    # Calculate average odds across all bookmakers for each team
    print("\n" + "="*80)
    print("AVERAGE ODDS ACROSS ALL BOOKMAKERS")
    print("="*80 + "\n")
    
    # Calculate implied probability for each entry
    df['implied_prob'] = df['odds'].apply(odds_to_implied_probability)
    
    # For each team, find which book has the best odds (lowest implied prob)
    best_books = []
    for team in df['team'].unique():
        team_df = df[df['team'] == team]
        best_idx = team_df['implied_prob'].idxmin()
        best_book = df.loc[best_idx, 'bookmaker']
        best_odds = df.loc[best_idx, 'odds']
        best_books.append({
            'team': team,
            'best_book': best_book,
            'best_odds': best_odds
        })
    
    best_books_df = pd.DataFrame(best_books)
    
    # Group by team and calculate averages, min, max
    team_avg = df.groupby('team').agg({
        'implied_prob': ['mean', 'min', 'max'],
        'odds': 'count'
    })
    
    # Flatten column names
    team_avg.columns = ['implied_prob_avg', 'implied_prob_min', 'implied_prob_max', 'num_books']
    team_avg = team_avg.reset_index()
    
    # Merge best book info
    team_avg = team_avg.merge(best_books_df, on='team')
    
    # Calculate fair probability (remove vig)
    # Total implied prob across all teams (average across books)
    total_implied_avg = team_avg['implied_prob_avg'].sum()
    
    # Fair probability = normalize to sum to 1.0
    team_avg['fair_prob'] = team_avg['implied_prob_avg'] / total_implied_avg
    
    # Convert fair probability to American odds
    team_avg['fair_odds'] = team_avg['fair_prob'].apply(
        lambda p: probability_to_american_odds(p * 100)
    )
    
    # Calculate line shopping opportunity (spread between min and max)
    team_avg['shopping_spread_pct'] = (team_avg['implied_prob_max'] - team_avg['implied_prob_min']) * 100
    
    # Get all 32 NFL teams with W/L records
    all_teams = get_all_nfl_teams_with_records()
    
    # Merge with odds data (outer join to include teams without odds)
    team_avg = all_teams.merge(team_avg, on='team', how='left')
    
    # For teams without odds, fill with placeholder values
    team_avg['best_book'] = team_avg['best_book'].fillna('-')
    team_avg['best_odds'] = team_avg['best_odds'].fillna(0)  # Will display as '-'
    team_avg['implied_prob_avg'] = team_avg['implied_prob_avg'].fillna(0.0)
    team_avg['implied_prob_min'] = team_avg['implied_prob_min'].fillna(0.0)
    team_avg['implied_prob_max'] = team_avg['implied_prob_max'].fillna(0.0)
    team_avg['fair_prob'] = team_avg['fair_prob'].fillna(0.0001)  # < 0.1%
    team_avg['fair_odds'] = team_avg['fair_odds'].fillna(100000)  # +100000
    team_avg['num_books'] = team_avg['num_books'].fillna(0).astype(int)
    team_avg['shopping_spread_pct'] = team_avg['shopping_spread_pct'].fillna(0.0)
    
    # Calculate W/L percentage for sorting
    team_avg['wins'] = team_avg['record'].str.split('-').str[0].astype(int)
    team_avg['losses'] = team_avg['record'].str.split('-').str[1].astype(int)
    team_avg['win_pct'] = team_avg['wins'] / (team_avg['wins'] + team_avg['losses'])
    
    # Sort by: teams with odds first (by fair_prob desc), then teams without odds (by W/L desc)
    team_avg['has_odds'] = team_avg['num_books'] > 0
    team_avg = team_avg.sort_values(['has_odds', 'fair_prob', 'win_pct'], ascending=[False, False, False])
    
    print(f"{'Rank':<6} {'Team':<25} {'W-L':<8} {'Best Book':<12} {'Best Odds':<12} {'Implied %':<12} {'Fair Odds':<12} {'Fair %':<10}")
    print("-"*120)
    for i, (idx, row) in enumerate(team_avg.iterrows(), 1):
        # Handle teams without odds
        if row['num_books'] == 0:
            best_odds_str = '-'
            fair_odds_str = '+100000'
            implied_str = '0.00%'
            fair_str = '<0.1%'
        else:
            best_odds_str = f"{row['best_odds']:+.0f}" if row['best_odds'] > 0 else f"{row['best_odds']:.0f}"
            fair_odds_str = f"{row['fair_odds']:+.0f}" if row['fair_odds'] > 0 else f"{row['fair_odds']:.0f}"
            implied_str = f"{row['implied_prob_avg']*100:>5.2f}%"
            fair_str = f"{row['fair_prob']*100:>5.2f}%"
        
        print(f"{i:<6} {row['team']:<25} "
              f"{row['record']:<8} "
              f"{row['best_book']:<12} "
              f"{best_odds_str:<12} "
              f"{implied_str:<12} "
              f"{fair_odds_str:<12} "
              f"{fair_str:<10}")
    
    print("-"*110)
    print(f"{'TOTAL':<6} {'':<25} {'':<12} {'':<12} {team_avg['implied_prob_avg'].sum()*100:>5.2f}%      {'':<12} 100.00%")
    
    # Show biggest line shopping opportunities
    print("\n" + "="*80)
    print("BIGGEST LINE SHOPPING OPPORTUNITIES")
    print("="*80 + "\n")
    
    top_shopping = team_avg.nlargest(5, 'shopping_spread_pct')[['team', 'best_book', 'best_odds', 'shopping_spread_pct', 'implied_prob_min', 'implied_prob_max']]
    
    print(f"{'Rank':<6} {'Team':<30} {'Best Book':<12} {'Best Odds':<12} {'Spread':<10}")
    print("-"*80)
    for i, (idx, row) in enumerate(top_shopping.iterrows(), 1):
        best_odds_str = f"{row['best_odds']:+.0f}" if row['best_odds'] > 0 else f"{row['best_odds']:.0f}"
        print(f"{i:<6} {row['team']:<30} "
              f"{row['best_book']:<12} "
              f"{best_odds_str:<12} "
              f"{row['shopping_spread_pct']:.2f}%")
    
    print("-"*80)
    print(f"\n💡 Shopping tip: The 'Spread' shows how much difference there is between")
    print(f"   the best and worst odds. Bigger spread = more important to shop around!")
    
    # Save to CSV
    output_file = repo_root / 'data/04_output/nfl/nfl_championship_fair_odds.csv'
    team_avg.to_csv(output_file, index=False)
    print(f"\n💾 Saved team averages to: {output_file}")
    
    return team_avg


if __name__ == "__main__":
    team_avg_df = main()


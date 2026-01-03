"""
Analyze vig in NBA Championship futures markets.

Context:
Created to extend the NFL futures workflow to NBA. Thomas wants to post weekly
for both NFL and NBA championship futures analysis.

Purpose:
- Read most recent NBA championship futures
- Calculate implied probabilities from odds
- Calculate total vig across all teams
- Remove vig to get true implied probabilities
- Show how much sportsbooks take on futures
- Compare with team records from ESPN API

Futures markets typically have much higher vig than game lines.
Game lines: ~4-5% vig
Futures: Often 20-40%+ vig

Usage:
    cd /Users/thomasmyles/dev/betting
    python3 analysis/analyze_nba_championship_futures_vig.py
"""

import pandas as pd
import os
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from odds_utils import odds_to_implied_probability, probability_to_american_odds


def get_most_recent_futures_file():
    """Get the most recent NBA Championship futures CSV file"""
    futures_dir = repo_root / 'data/01_input/the-odds-api/nba/futures'
    
    # Get all CSV files
    csv_files = sorted(futures_dir.glob('nba_championship_futures_*.csv'))
    
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
    """Analyze NBA championship futures vig"""
    
    print("="*80)
    print("NBA CHAMPIONSHIP FUTURES VIG ANALYSIS")
    print("="*80 + "\n")
    
    # Read most recent futures file
    futures_file = get_most_recent_futures_file()
    df = pd.read_csv(futures_file)
    
    print(f"📊 Total odds entries: {len(df)}")
    print(f"🏀 Unique teams: {df['team'].nunique()}")
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
        'odds': 'count',
        'record': 'first'  # Get record from ESPN API (all rows for a team should have same record)
    })
    
    # Flatten column names
    team_avg.columns = ['implied_prob_avg', 'implied_prob_min', 'implied_prob_max', 'num_books', 'record']
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
    
    # Calculate W/L percentage for sorting (if record exists)
    team_avg['has_record'] = team_avg['record'].notna() & (team_avg['record'] != '')
    team_avg['wins'] = 0
    team_avg['losses'] = 0
    team_avg['win_pct'] = 0.0
    
    for idx, row in team_avg.iterrows():
        if row['has_record']:
            parts = row['record'].split('-')
            if len(parts) >= 2:
                team_avg.at[idx, 'wins'] = int(parts[0])
                team_avg.at[idx, 'losses'] = int(parts[1])
                total_games = int(parts[0]) + int(parts[1])
                if total_games > 0:
                    team_avg.at[idx, 'win_pct'] = int(parts[0]) / total_games
    
    # Sort by fair_prob desc (best championship odds first)
    team_avg = team_avg.sort_values('fair_prob', ascending=False)
    
    print(f"{'Rank':<6} {'Team':<30} {'W-L':<8} {'Best Book':<12} {'Best Odds':<12} {'Implied %':<12} {'Fair Odds':<12} {'Fair %':<10}")
    print("-"*120)
    for i, (idx, row) in enumerate(team_avg.iterrows(), 1):
        record_str = row['record'] if row['has_record'] else '-'
        best_odds_str = f"{row['best_odds']:+.0f}" if row['best_odds'] > 0 else f"{row['best_odds']:.0f}"
        fair_odds_str = f"{row['fair_odds']:+.0f}" if row['fair_odds'] > 0 else f"{row['fair_odds']:.0f}"
        implied_str = f"{row['implied_prob_avg']*100:>5.2f}%"
        fair_str = f"{row['fair_prob']*100:>5.2f}%"
        
        print(f"{i:<6} {row['team']:<30} "
              f"{record_str:<8} "
              f"{row['best_book']:<12} "
              f"{best_odds_str:<12} "
              f"{implied_str:<12} "
              f"{fair_odds_str:<12} "
              f"{fair_str:<10}")
    
    print("-"*120)
    print(f"{'TOTAL':<6} {'':<30} {'':<8} {'':<12} {'':<12} {team_avg['implied_prob_avg'].sum()*100:>5.2f}%      {'':<12} 100.00%")
    
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
    os.makedirs(repo_root / 'data/04_output/nba', exist_ok=True)
    output_file = repo_root / 'data/04_output/nba/nba_championship_fair_odds.csv'
    team_avg.to_csv(output_file, index=False)
    print(f"\n💾 Saved team averages to: {output_file}")
    
    # Also save metadata with average vig
    metadata_file = repo_root / 'data/04_output/nba/nba_championship_metadata.csv'
    metadata_df = pd.DataFrame([{
        'timestamp': pd.Timestamp.now(),
        'avg_vig_pct': avg_vig,
        'num_teams': len(df['team'].unique()),
        'num_bookmakers': len(df['bookmaker'].unique())
    }])
    metadata_df.to_csv(metadata_file, index=False)
    print(f"💾 Saved metadata to: {metadata_file}")
    
    return team_avg


if __name__ == "__main__":
    team_avg_df = main()


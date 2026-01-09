"""
Analyze NBA MVP Odds - Calculate Fair Odds and Vig

Context:
Similar to championship futures analysis, but for MVP award.
Takes FanDuel MVP odds (hardcoded) and calculates fair odds by removing vig.

Usage:
    python3 analysis/analyze_nba_mvp_vig.py

Input:
    data/01_input/fanduel/nba/mvp/nba_mvp_odds_YYYYMMDD_HHMMSS.csv (latest)

Output:
    data/04_output/nba/mvp/nba_mvp_fair_odds_YYYYMMDD_HHMMSS.csv

Output columns:
    - player: Player name
    - fanduel_odds: Raw FanDuel odds
    - fanduel_implied_prob: Implied probability from FanDuel
    - fair_prob: True probability after removing market vig
    - fair_odds: What odds would be with zero vig
    - vig_pct: Bookmaker edge on this specific player
    - fetch_date: When odds were pulled from FanDuel
"""

import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime
import glob

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from odds_utils import odds_to_implied_probability, implied_probability_to_odds


def load_latest_mvp_odds():
    """Load the most recent MVP odds CSV"""
    input_dir = repo_root / 'data/01_input/fanduel/nba/mvp'
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Get all MVP odds files
    csv_files = list(input_dir.glob('nba_mvp_odds_*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No MVP odds files found in {input_dir}")
    
    # Get most recent file
    latest_file = max(csv_files, key=os.path.getmtime)
    
    print(f"📂 Loading: {latest_file.name}")
    
    df = pd.read_csv(latest_file)
    return df


def calculate_fair_odds(df):
    """
    Calculate fair odds by removing market vig (proportional method).
    
    Same method used for championship futures analysis.
    
    Args:
        df: DataFrame with columns [player, odds, implied_prob]
    
    Returns:
        DataFrame with added columns [fair_prob, fair_odds, vig_pct]
    """
    # Calculate total market probability (includes vig)
    total_market_prob = df['implied_prob'].sum()
    
    # Fair probability = remove vig proportionally
    # fair_prob = implied_prob / total_market_prob
    df['fair_prob'] = df['implied_prob'] / total_market_prob
    
    # Convert fair probability back to American odds
    df['fair_odds'] = df['fair_prob'].apply(implied_probability_to_odds)
    
    # Calculate vig % for each player (absolute difference in probability points)
    # This matches the calculation in championship futures analysis
    # vig_pct = (implied_prob - fair_prob) * 100
    df['vig_pct'] = (df['implied_prob'] - df['fair_prob']) * 100
    
    return df


def main():
    """Main analysis function"""
    print("="*80)
    print("NBA MVP VIG ANALYSIS")
    print("="*80)
    
    # Load latest MVP odds
    print("\n1️⃣ Loading MVP odds from FanDuel...")
    df = load_latest_mvp_odds()
    
    print(f"   ✅ Loaded {len(df)} players")
    
    # Calculate fair odds
    print("\n2️⃣ Calculating fair odds (removing vig)...")
    df = calculate_fair_odds(df)
    
    # Calculate market vig
    total_market_prob = df['implied_prob'].sum()
    market_vig_pct = (total_market_prob - 1.0) * 100
    average_vig_pct = df['vig_pct'].mean()
    
    print(f"   ✅ Market Vig: {market_vig_pct:.1f}%")
    print(f"   ✅ Average Vig per Player: {average_vig_pct:.1f}%")
    
    # Show results
    print("\n" + "="*80)
    print("FAIR ODDS vs FANDUEL ODDS")
    print("="*80)
    
    print(f"\n{'Player':<30}{'FanDuel':<12}{'Fair Odds':<12}{'Vig %':<10}")
    print("-" * 64)
    
    for row in df.head(10).itertuples():
        fd_odds_str = f"{int(row.odds):+d}"
        fair_odds_str = f"{int(row.fair_odds):+d}"
        vig_str = f"{row.vig_pct:+.1f}%"
        
        print(f"{row.player:<30}{fd_odds_str:<12}{fair_odds_str:<12}{vig_str:<10}")
    
    # Save to CSV
    print("\n3️⃣ Saving results...")
    
    # Rename columns for clarity
    df_output = df.rename(columns={
        'odds': 'fanduel_odds',
        'implied_prob': 'fanduel_implied_prob'
    })
    
    # Reorder columns
    df_output = df_output[[
        'player',
        'fanduel_odds',
        'fanduel_implied_prob',
        'fair_prob',
        'fair_odds',
        'vig_pct',
        'fetch_date'
    ]]
    
    # Save
    output_dir = repo_root / 'data/04_output/nba/mvp'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'nba_mvp_fair_odds_{timestamp}.csv'
    
    df_output.to_csv(output_file, index=False)
    
    print(f"   ✅ Saved to: {output_file}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nMarket Vig: {market_vig_pct:.1f}%")
    print(f"Players: {len(df)}")
    print(f"Favorite: {df.iloc[0]['player']} ({df.iloc[0]['odds']:+d})")
    print(f"Longshot: {df.iloc[-1]['player']} ({df.iloc[-1]['odds']:+d})")
    
    print("\nKey Insights:")
    print(f"- FanDuel charges {market_vig_pct:.1f}% vig on MVP market")
    print(f"- Average vig per player: {average_vig_pct:.1f}%")
    
    # Find best and worst vig
    best_vig_player = df.loc[df['vig_pct'].idxmin()]
    worst_vig_player = df.loc[df['vig_pct'].idxmax()]
    
    print(f"- Lowest vig: {best_vig_player['player']} ({best_vig_player['vig_pct']:+.1f}%)")
    print(f"- Highest vig: {worst_vig_player['player']} ({worst_vig_player['vig_pct']:+.1f}%)")
    
    print("\n" + "="*80)
    print("NEXT STEP")
    print("="*80)
    print("\nGenerate visualization:")
    print("   python3 analysis/viz_nba_mvp_gt.py")


if __name__ == "__main__":
    main()


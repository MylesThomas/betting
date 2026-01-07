"""
NBA MVP Odds from FanDuel (Hardcoded)

Context:
Thomas wants to track NBA MVP odds similar to championship futures workflow.
The Odds API doesn't support MVP futures, so we hardcode FanDuel odds manually.

Usage:
    python3 scripts/fetch_nba_mvp_odds_fanduel.py

Output:
    data/01_input/fanduel/nba/mvp/nba_mvp_odds_YYYYMMDD_HHMMSS.csv

CSV columns:
    - bookmaker: Always 'fanduel'
    - player: Player name
    - odds: American odds (e.g., -450, +750)
    - implied_prob: Implied probability from odds
    - fetch_date: Date these odds were manually entered

How to update:
1. Go to FanDuel → NBA → Awards → MVP
2. Copy odds into CURRENT_MVP_ODDS dict below
3. Update FETCH_DATE to today's date
4. Run this script
"""

import pandas as pd
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for odds_utils
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from odds_utils import odds_to_implied_probability


# =============================================================================
# HARDCODED MVP ODDS (UPDATE MANUALLY FROM FANDUEL)
# =============================================================================

# Date these odds were fetched from FanDuel (update when you update odds)
FETCH_DATE = "2025-01-07"  # YYYY-MM-DD

# Current MVP odds from FanDuel
# Format: {'Player Name': american_odds}
CURRENT_MVP_ODDS = {
    'Shai Gilgeous-Alexander': -450,
    'Luka Doncic': +750,
    'Cade Cunningham': +2000,
    'Jaylen Brown': +4000,
    'Jalen Brunson': +5000,
    'Anthony Edwards': +10000,
    'Tyrese Maxey': +10000,
    'Donovan Mitchell': +20000,
    'Kawhi Leonard': +100000,
    'Stephen Curry': +100000,
    'Alperen Sengun': +100000,
    'Kevin Durant': +100000,
}

# Season end date from FanDuel
SEASON_END_DATE = "Apr 14, 6:00pm CT"


# =============================================================================
# FUNCTIONS
# =============================================================================

def create_mvp_dataframe(odds_dict, fetch_date):
    """
    Create DataFrame from hardcoded MVP odds.
    
    Args:
        odds_dict: Dict of player -> odds
        fetch_date: Date string (YYYY-MM-DD) when odds were fetched
    
    Returns:
        DataFrame with columns: bookmaker, player, odds, implied_prob, fetch_date
    """
    rows = []
    
    for player, odds in odds_dict.items():
        implied_prob = odds_to_implied_probability(odds)
        
        rows.append({
            'bookmaker': 'fanduel',
            'player': player,
            'odds': odds,
            'implied_prob': implied_prob,
            'fetch_date': fetch_date
        })
    
    df = pd.DataFrame(rows)
    
    # Sort by implied probability descending (highest prob = favorite)
    df = df.sort_values('implied_prob', ascending=False).reset_index(drop=True)
    
    return df


def save_mvp_odds(df, timestamp=None):
    """
    Save MVP odds to CSV with timestamp.
    
    Args:
        df: DataFrame with MVP odds
        timestamp: Optional timestamp string (YYYYMMDD_HHMMSS)
    
    Returns:
        Path to saved file
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create output directory
    output_dir = repo_root / 'data/01_input/fanduel/nba/mvp'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    output_file = output_dir / f'nba_mvp_odds_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    
    return output_file


def main():
    """Main function"""
    print("="*80)
    print("NBA MVP ODDS FROM FANDUEL (HARDCODED)")
    print("="*80)
    
    print(f"\n📅 Fetch Date: {FETCH_DATE}")
    print(f"📊 Season Ends: {SEASON_END_DATE}")
    print(f"🏀 Total Players: {len(CURRENT_MVP_ODDS)}")
    
    # Create DataFrame
    df = create_mvp_dataframe(CURRENT_MVP_ODDS, FETCH_DATE)
    
    # Calculate total market probability (vig)
    total_prob = df['implied_prob'].sum()
    vig_pct = (total_prob - 1.0) * 100
    
    print(f"\n📈 Market Vig: {vig_pct:.1f}%")
    print(f"   (Total implied probability: {total_prob:.4f})")
    
    # Display top candidates
    print("\n" + "="*80)
    print("TOP MVP CANDIDATES")
    print("="*80)
    
    print(f"\n{'Rank':<6}{'Player':<30}{'Odds':<10}{'Implied Prob':<15}")
    print("-" * 61)
    
    for i, row in enumerate(df.head(10).itertuples(), 1):
        odds_str = f"{row.odds:+d}"  # Format as +750 or -450
        prob_str = f"{row.implied_prob*100:.1f}%"
        print(f"{i:<6}{row.player:<30}{odds_str:<10}{prob_str:<15}")
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = save_mvp_odds(df, timestamp)
    
    print("\n" + "="*80)
    print("✅ SAVED")
    print("="*80)
    print(f"\n💾 Output: {output_file}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Run analysis to calculate fair odds and vig:")
    print("   python3 analysis/analyze_nba_mvp_vig.py")
    print("\n2. Generate visualization:")
    print("   python3 analysis/viz_nba_mvp_gt.py")
    print("\n3. To update odds:")
    print("   - Edit CURRENT_MVP_ODDS dict in this file")
    print("   - Update FETCH_DATE to today")
    print("   - Re-run this script")


if __name__ == "__main__":
    main()


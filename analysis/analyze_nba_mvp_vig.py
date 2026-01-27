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
from s3_utils import get_latest_file_from_s3, read_df_from_s3, upload_df_to_s3


def load_latest_mvp_odds():
    """Load the most recent MVP odds CSV from S3"""
    bucket = 'nba-betting-mt'
    prefix = 'data/01_input/fanduel/nba/mvp/'
    
    # Get most recent file
    latest_key = get_latest_file_from_s3(bucket, prefix)
    
    if not latest_key:
        raise FileNotFoundError(f"No MVP odds files found in s3://{bucket}/{prefix}")
    
    filename = latest_key.split('/')[-1]
    print(f"📂 Loading from S3: {filename}")
    
    df = read_df_from_s3(bucket, latest_key)
    return df


def calculate_fair_odds(df):
    """
    Calculate fair odds by removing market vig (proportional method).
    
    Same method used for championship futures analysis.
    Only calculates for players currently on the board (odds not None).
    
    Args:
        df: DataFrame with columns [player, odds, implied_prob]
    
    Returns:
        DataFrame with added columns [fair_prob, fair_odds, vig_pct]
    """
    # Only calculate for players currently on the board
    on_board_mask = df['odds'].notna()
    
    # Calculate total market probability (includes vig) - only for players on board
    total_market_prob = df.loc[on_board_mask, 'implied_prob'].sum()
    
    # Fair probability = remove vig proportionally (only for players on board)
    df.loc[on_board_mask, 'fair_prob'] = df.loc[on_board_mask, 'implied_prob'] / total_market_prob
    
    # Convert fair probability back to American odds (only for players on board)
    df.loc[on_board_mask, 'fair_odds'] = df.loc[on_board_mask, 'fair_prob'].apply(implied_probability_to_odds)
    
    # Calculate vig % for each player (only for players on board)
    df.loc[on_board_mask, 'vig_pct'] = (df.loc[on_board_mask, 'implied_prob'] - df.loc[on_board_mask, 'fair_prob']) * 100
    
    # For players not on board, set fair_prob, fair_odds, vig_pct to None
    df.loc[~on_board_mask, 'fair_prob'] = None
    df.loc[~on_board_mask, 'fair_odds'] = None
    df.loc[~on_board_mask, 'vig_pct'] = None
    
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
    
    # Calculate market vig (only for players on board)
    on_board_mask = df['odds'].notna()
    total_market_prob = df.loc[on_board_mask, 'implied_prob'].sum()
    market_vig_pct = (total_market_prob - 1.0) * 100
    average_vig_pct = df.loc[on_board_mask, 'vig_pct'].mean()
    
    print(f"   ✅ Market Vig: {market_vig_pct:.1f}%")
    print(f"   ✅ Average Vig per Player: {average_vig_pct:.1f}%")
    
    # Show results
    print("\n" + "="*80)
    print("FAIR ODDS vs FANDUEL ODDS")
    print("="*80)
    
    print(f"\n{'Player':<30}{'FanDuel':<12}{'Fair Odds':<12}{'Vig %':<10}")
    print("-" * 64)
    
    for row in df.head(20).itertuples():  # Show more rows to include removed players
        fd_odds_str = f"{int(row.odds):+d}" if pd.notna(row.odds) else "-"
        fair_odds_str = f"{int(row.fair_odds):+d}" if pd.notna(row.fair_odds) else "-"
        vig_str = f"{row.vig_pct:+.1f}%" if pd.notna(row.vig_pct) else "-"
        
        print(f"{row.player:<30}{fd_odds_str:<12}{fair_odds_str:<12}{vig_str:<10}")
    
    # Save to CSV
    print("\n3️⃣ Saving results...")
    
    # Rename columns for clarity
    df_output = df.rename(columns={
        'odds': 'fanduel_odds',
        'implied_prob': 'fanduel_implied_prob'
    })
    
    # Reorder columns (include season_start columns if they exist)
    output_columns = [
        'player',
        'fanduel_odds',
        'fanduel_implied_prob',
        'fair_prob',
        'fair_odds',
        'vig_pct',
        'fetch_date'
    ]
    
    # Add season start columns if they exist in the dataframe
    if 'season_start_odds' in df_output.columns:
        output_columns.insert(2, 'season_start_odds')
    if 'season_start_date' in df_output.columns:
        output_columns.insert(3, 'season_start_date')
    if 'last_week_odds' in df_output.columns:
        output_columns.insert(4, 'last_week_odds')
    if 'last_week_date' in df_output.columns:
        output_columns.insert(5, 'last_week_date')
    
    df_output = df_output[output_columns]
    
    # Save to S3
    bucket = 'nba-betting-mt'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    s3_key = f'data/04_output/nba/mvp/nba_mvp_fair_odds_{timestamp}.csv'
    
    s3_uri = upload_df_to_s3(df_output, bucket, s3_key)
    
    print(f"   ✅ Saved to S3: {s3_uri}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    on_board_df = df[df['odds'].notna()]
    removed_df = df[df['odds'].isna()]
    
    print(f"\nMarket Vig: {market_vig_pct:.1f}%")
    print(f"Players on Board: {len(on_board_df)}")
    print(f"Players Removed: {len(removed_df)}")
    print(f"Total Tracked: {len(df)}")
    
    if len(on_board_df) > 0:
        print(f"Favorite: {on_board_df.iloc[0]['player']} ({int(on_board_df.iloc[0]['odds']):+d})")
        print(f"Longshot: {on_board_df.iloc[-1]['player']} ({int(on_board_df.iloc[-1]['odds']):+d})")
    
    print("\nKey Insights:")
    print(f"- FanDuel charges {market_vig_pct:.1f}% vig on MVP market")
    print(f"- Average vig per player: {average_vig_pct:.1f}%")
    
    # Find best and worst vig (only among players on board)
    if len(on_board_df) > 0:
        best_vig_player = on_board_df.loc[on_board_df['vig_pct'].idxmin()]
        worst_vig_player = on_board_df.loc[on_board_df['vig_pct'].idxmax()]
        
        print(f"- Lowest vig: {best_vig_player['player']} ({best_vig_player['vig_pct']:+.1f}%)")
        print(f"- Highest vig: {worst_vig_player['player']} ({worst_vig_player['vig_pct']:+.1f}%)")
    
    if len(removed_df) > 0:
        print(f"\n- Removed from board since season start:")
        for player in removed_df['player'].head(5):
            print(f"    • {player}")
    
    print("\n" + "="*80)
    print("NEXT STEP")
    print("="*80)
    print("\nGenerate visualization:")
    print("   python3 analysis/viz_nba_mvp_gt.py")


if __name__ == "__main__":
    main()


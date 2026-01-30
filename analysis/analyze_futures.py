"""
Generic Championship Futures Analysis

Analyzes championship futures for NFL, NBA, NCAAF, or NCAAB.

Usage:
    python3 analysis/analyze_futures.py --sport nfl
    python3 analysis/analyze_futures.py --sport nba
    python3 analysis/analyze_futures.py --sport ncaaf
    python3 analysis/analyze_futures.py --sport ncaab

Outputs:
    - data/04_output/{sport}/{sport}_championship_fair_odds.csv
    - data/04_output/{sport}/{sport}_championship_metadata.csv
"""

import sys
import argparse
import pandas as pd
import yaml
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from futures_analysis import (
    get_most_recent_futures_file,
    calculate_vig_by_bookmaker,
    calculate_fair_probabilities,
    calculate_team_averages,
    save_analysis_outputs
)
from odds_utils import odds_to_implied_probability
import numpy as np


def load_configs():
    """Load futures config."""
    futures_config_path = repo_root / 'config' / 'futures_config.yaml'
    
    with open(futures_config_path) as f:
        futures_config = yaml.safe_load(f)
    
    return futures_config


def main():
    parser = argparse.ArgumentParser(
        description='Analyze championship futures for NFL, NBA, NCAAF, or NCAAB'
    )
    parser.add_argument(
        '--sport',
        type=str,
        required=True,
        choices=['nfl', 'nba', 'ncaaf', 'ncaab'],
        help='Sport to analyze (nfl, nba, ncaaf, ncaab)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=99999,
        help='Limit analysis to top N teams by fair probability (default: all teams)'
    )
    args = parser.parse_args()
    
    sport = args.sport.lower()
    
    # Load configs
    futures_config = load_configs()
    sport_config = futures_config['sports'][sport]
    
    # Print header
    emoji = sport_config['emoji']
    display_name = sport_config['display_name']
    print("=" * 80)
    print(f"{emoji} {display_name.upper()} CHAMPIONSHIP FUTURES VIG ANALYSIS")
    print("=" * 80 + "\n")
    
    # Get most recent futures file
    input_dir = repo_root / sport_config['input_dir']
    file_prefix = sport_config['file_prefix']
    s3_bucket_base = futures_config.get('s3_bucket_base')
    
    try:
        futures_file = get_most_recent_futures_file(input_dir, file_prefix, s3_bucket=s3_bucket_base)
        print(f"📁 Reading: {futures_file.name}\n")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"\n💡 Tip: Run fetch script first:")
        print(f"   python3 scripts/fetch_championship_futures.py")
        sys.exit(1)
    
    # Read CSV
    df = pd.read_csv(futures_file)
    
    print(f"📊 Total odds entries: {len(df)}")
    print(f"{emoji} Unique teams: {df['team'].nunique()}")
    print(f"📚 Bookmakers: {df['bookmaker'].nunique()}")
    print(f"   {', '.join(df['bookmaker'].unique())}\n")
    
    # Calculate vig by bookmaker
    print("=" * 80)
    print("VIG BY BOOKMAKER")
    print("=" * 80 + "\n")
    
    vig_df = calculate_vig_by_bookmaker(df)
    
    print(f"{'Bookmaker':<20} {'Teams':<8} {'Total Implied':<15} {'Vig %':<10}")
    print("-" * 80)
    for _, row in vig_df.iterrows():
        print(f"{row['bookmaker']:<20} {row['num_teams']:<8} "
              f"{row['total_implied_prob']:>6.4f} ({row['total_implied_prob']*100:>6.2f}%)  "
              f"{row['vig_pct']:>6.2f}%")
    
    avg_vig = vig_df['vig_pct'].mean()
    print("-" * 80)
    print(f"{'AVERAGE VIG':<20} {'':<8} {'':<15} {avg_vig:>6.2f}%\n")
    
    # Show fair probabilities for best bookmaker
    best_bookmaker = vig_df.iloc[-1]['bookmaker']
    
    print("=" * 80)
    print(f"FAIR PROBABILITIES (VIG REMOVED) - {best_bookmaker.upper()}")
    print("=" * 80 + "\n")
    
    fair_df = calculate_fair_probabilities(df, best_bookmaker)
    
    print(f"{'Rank':<6} {'Team':<40} {'Odds':<10} {'Implied %':<12} {'Fair %':<12}")
    print("-" * 80)
    for i, (idx, row) in enumerate(fair_df.head(20).iterrows(), 1):
        odds_str = f"{row['odds']:+.0f}" if row['odds'] > 0 else f"{row['odds']:.0f}"
        print(f"{i:<6} {row['team']:<40} {odds_str:<10} "
              f"{row['implied_prob']*100:>6.2f}%      {row['fair_prob']*100:>6.2f}%")
    
    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print(f"\n1. Average vig across all bookmakers: {avg_vig:.2f}%")
    print(f"2. Bookmaker with best (lowest) vig: {best_bookmaker} ({vig_df.iloc[-1]['vig_pct']:.2f}%)")
    print(f"3. Bookmaker with worst (highest) vig: {vig_df.iloc[0]['bookmaker']} ({vig_df.iloc[0]['vig_pct']:.2f}%)")
    print(f"\n💡 For comparison:")
    print(f"   - Game lines (spread/total): ~4-5% vig")
    print(f"   - Futures markets: {avg_vig:.1f}% vig")
    print(f"   - Futures markets have {avg_vig/4.5:.1f}x more vig than game lines!")
    print(f"\n📉 What this means:")
    print(f"   If all teams had equal odds, sportsbooks would win {avg_vig:.1f}% of all bets")
    print(f"   To break even, bettors need to be {avg_vig:.1f}% better than 'fair' odds")
    
    # Calculate team averages
    print("\n" + "=" * 80)
    print("AVERAGE ODDS ACROSS ALL BOOKMAKERS")
    print("=" * 80 + "\n")
    
    # Calculate implied probability for each entry
    df['implied_prob'] = df['odds'].apply(odds_to_implied_probability)
    
    team_avg = calculate_team_averages(df)
    
    # Add historical odds if configured
    print("\n" + "=" * 80)
    print("HISTORICAL ODDS (PRESEASON & LAST WEEK)")
    print("=" * 80 + "\n")
    
    historical_config = sport_config.get('historical_odds')
    if historical_config:
        # Preseason odds
        preseason_config = historical_config.get('preseason', {})
        if preseason_config and preseason_config.get('odds'):
            preseason_date = preseason_config.get('date', 'Unknown')
            preseason_label = preseason_config.get('label', 'Preseason')
            preseason_odds_dict = preseason_config['odds']
            
            # Map preseason odds to teams
            team_avg['season_start_odds'] = team_avg['team'].map(preseason_odds_dict)
            team_avg['season_start_date'] = preseason_date
            team_avg['season_start_label'] = preseason_label
            
            matched_count = team_avg['season_start_odds'].notna().sum()
            print(f"   📅 {preseason_label} ({preseason_date}): {matched_count}/{len(team_avg)} teams matched")
        else:
            print("   ⚠️  No preseason odds configured")
            team_avg['season_start_odds'] = np.nan
            team_avg['season_start_date'] = None
            team_avg['season_start_label'] = None
        
        # Last week odds
        last_week_config = historical_config.get('last_week', {})
        if last_week_config and last_week_config.get('odds'):
            last_week_date = last_week_config.get('date', 'Unknown')
            last_week_label = last_week_config.get('label', 'Last Week')
            last_week_odds_dict = last_week_config['odds']
            
            # Map last week odds to teams
            team_avg['last_week_odds'] = team_avg['team'].map(last_week_odds_dict)
            team_avg['last_week_date'] = last_week_date
            team_avg['last_week_label'] = last_week_label
            
            matched_count = team_avg['last_week_odds'].notna().sum()
            print(f"   📅 {last_week_label} ({last_week_date}): {matched_count}/{len(team_avg)} teams matched")
        else:
            print("   ⚠️  No last week odds configured")
            team_avg['last_week_odds'] = np.nan
            team_avg['last_week_date'] = None
            team_avg['last_week_label'] = None
        
        print()
    else:
        print("   ℹ️  No historical odds configured for this sport\n")
        team_avg['season_start_odds'] = np.nan
        team_avg['season_start_date'] = None
        team_avg['season_start_label'] = None
        team_avg['last_week_odds'] = np.nan
        team_avg['last_week_date'] = None
        team_avg['last_week_label'] = None
    
    # Apply top-n filter if specified
    total_teams = len(team_avg)
    if args.top_n < total_teams:
        print(f"\n⚠️  Filtering to top {args.top_n} teams by fair probability")
        team_avg = team_avg.head(args.top_n)
        print(f"   Showing {len(team_avg)} of {total_teams} teams\n")
    else:
        print()
    
    print(f"{'Rank':<6} {'Team':<40} {'Record':<10} {'Best Book':<12} {'Best Odds':<12} "
          f"{'Implied %':<12} {'Fair Odds':<12} {'Fair %':<10}")
    print("-" * 130)
    
    for i, (idx, row) in enumerate(team_avg.iterrows(), 1):
        best_odds_str = f"{row['best_odds']:+.0f}" if row['best_odds'] > 0 else f"{row['best_odds']:.0f}"
        fair_odds_str = f"{row['fair_odds']:+.0f}" if row['fair_odds'] > 0 else f"{row['fair_odds']:.0f}"
        implied_str = f"{row['implied_prob_avg']*100:>5.2f}%"
        fair_str = f"{row['fair_prob']*100:>5.2f}%"
        
        print(f"{i:<6} {row['team']:<40} "
              f"{row['record']:<10} "
              f"{row['best_book']:<12} "
              f"{best_odds_str:<12} "
              f"{implied_str:<12} "
              f"{fair_odds_str:<12} "
              f"{fair_str:<10}")
    
    print("-" * 130)
    print(f"{'TOTAL':<6} {'':<40} {'':<10} {'':<12} {'':<12} "
          f"{team_avg['implied_prob_avg'].sum()*100:>5.2f}%      {'':<12} 100.00%")
    
    # Line shopping opportunities
    print("\n" + "=" * 80)
    print("BIGGEST LINE SHOPPING OPPORTUNITIES")
    print("=" * 80 + "\n")
    
    top_shopping = team_avg.nlargest(5, 'shopping_spread_pct')
    
    print(f"{'Rank':<6} {'Team':<40} {'Best Book':<12} {'Best Odds':<12} {'Spread':<10}")
    print("-" * 90)
    
    for i, (idx, row) in enumerate(top_shopping.iterrows(), 1):
        best_odds_str = f"{row['best_odds']:+.0f}" if row['best_odds'] > 0 else f"{row['best_odds']:.0f}"
        print(f"{i:<6} {row['team']:<40} {row['best_book']:<12} "
              f"{best_odds_str:<12} {row['shopping_spread_pct']:>5.2f}%")
    
    print("-" * 90)
    print("\n💡 Shopping tip: The 'Spread' shows how much difference there is between")
    print("   the best and worst odds. Bigger spread = more important to shop around!")
    
    # Save outputs
    output_dir = repo_root / sport_config['output_dir']
    output_prefix = sport_config['output_prefix']
    
    # Get save settings from config
    save_locally = futures_config.get('save_locally', False)
    s3_bucket = sport_config.get('s3_output_bucket')
    s3_path = sport_config.get('s3_analysis_path')
    
    team_avg_file, metadata_file = save_analysis_outputs(
        team_avg, vig_df, output_dir, output_prefix,
        save_locally=save_locally,
        s3_bucket=s3_bucket,
        s3_path=s3_path
    )
    
    if save_locally:
        print(f"\n💾 Saved team averages to: {team_avg_file}")
        print(f"💾 Saved metadata to: {metadata_file}")


if __name__ == '__main__':
    main()

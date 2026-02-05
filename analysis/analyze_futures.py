"""
Championship Futures Analysis - S3-Driven

Reads all data from S3 CSV files (no config updates needed).

Usage:
    python3 analysis/analyze_futures.py \
        --sport nba \
        --top-n 20 \
        --preseason s3://the-odds-api-mt/nba/futures/preseason_2024-25.csv \
        --last-week s3://the-odds-api-mt/nba/futures/2026-01-27/nba_championship_futures_20260127_175837.csv \
        --this-week s3://the-odds-api-mt/nba/futures/2026-02-04/nba_championship_futures_20260204_163703.csv

Output:
    - data/04_output/{sport}/{sport}_championship_fair_odds.csv
    - Uploads to S3
"""

import sys
import argparse
import pandas as pd
import yaml
import boto3
from io import StringIO
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from odds_utils import odds_to_implied_probability, probability_to_american_odds


def load_config():
    """Load futures config."""
    config_path = repo_root / 'config' / 'futures_config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_s3_path(s3_path):
    """Parse S3 path into bucket and key."""
    if not s3_path.startswith('s3://'):
        raise ValueError(f"Invalid S3 path: {s3_path}")
    
    path = s3_path[5:]  # Remove 's3://'
    bucket, key = path.split('/', 1)
    return bucket, key


def load_csv_from_s3(s3_path):
    """Load CSV from S3 path."""
    bucket, key = parse_s3_path(s3_path)
    
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket=bucket, Key=key)
    csv_string = response['Body'].read().decode('utf-8')
    
    return pd.read_csv(StringIO(csv_string))


def get_median_odds_per_team(df):
    """
    Get median odds per team (most representative of market).
    
    Returns dict: {team: median_odds}
    """
    # Group by team and get median odds
    median_odds = df.groupby('team')['odds'].median().to_dict()
    return median_odds


def build_analysis_dataframe(preseason_df, last_week_df, current_df):
    """
    Build analysis dataframe from three S3 CSVs.
    
    Args:
        preseason_df: DataFrame from preseason S3 CSV
        last_week_df: DataFrame from last week's S3 CSV
        current_df: DataFrame from current week's S3 CSV
    
    Returns:
        DataFrame with all calculated fields
    """
    # Get median odds per team for each timepoint
    preseason_median = get_median_odds_per_team(preseason_df)
    last_week_median = get_median_odds_per_team(last_week_df)
    current_median = get_median_odds_per_team(current_df)
    
    # Get current records per team (take first record for each team since it's the same across bookmakers)
    # Fill NaN with "NA" for teams not in ESPN API data
    current_records = current_df.groupby('team')['record'].first().fillna("NA").to_dict()
    
    # Get all unique teams
    all_teams = set()
    all_teams.update(preseason_median.keys())
    all_teams.update(last_week_median.keys())
    all_teams.update(current_median.keys())
    
    # Build rows
    rows = []
    for team in all_teams:
        # Preseason
        preseason = preseason_median.get(team)
        preseason_implied = odds_to_implied_probability(preseason) if preseason else None
        
        # Last week
        last_week = last_week_median.get(team)
        last_week_implied = odds_to_implied_probability(last_week) if last_week else None
        
        # Current (median odds)
        current = current_median.get(team)
        current_implied = odds_to_implied_probability(current) if current else None
        
        # Current record
        record = current_records.get(team, '-')
        
        rows.append({
            'team': team,
            'record': record,
            'preseason_odds': preseason,
            'preseason_implied_prob': preseason_implied,
            'last_week_odds': last_week,
            'last_week_implied_prob': last_week_implied,
            'current_odds': current,
            'current_implied_prob': current_implied,
        })
    
    df = pd.DataFrame(rows)
    
    # Calculate fair probabilities (normalize current to sum to 100%)
    total_implied = df['current_implied_prob'].sum()
    df['fair_prob'] = df['current_implied_prob'] / total_implied
    df['fair_odds'] = df['fair_prob'].apply(
        lambda p: probability_to_american_odds(p * 100) if pd.notna(p) else None
    )
    
    # Calculate vig (difference between implied and fair)
    df['vig_pct'] = (df['current_implied_prob'] - df['fair_prob']) * 100
    
    # Calculate differences (in percentage points)
    # Treat missing historical data as 0% implied probability (not on the board = 0%)
    df['diff_preseason'] = (
        df['current_implied_prob'] - df['preseason_implied_prob'].fillna(0)
    ) * 100
    df['diff_last_week'] = (
        df['current_implied_prob'] - df['last_week_implied_prob'].fillna(0)
    ) * 100
    
    # Sort by fair probability (best teams first)
    df = df.sort_values('fair_prob', ascending=False, na_position='last').reset_index(drop=True)
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Analyze futures from S3 CSVs')
    parser.add_argument('--sport', type=str, required=True, choices=['nfl', 'nba', 'ncaaf', 'ncaab'])
    parser.add_argument('--top-n', type=int, default=99999, help='Limit to top N teams')
    parser.add_argument('--preseason', type=str, required=True, help='S3 path to preseason CSV')
    parser.add_argument('--last-week', type=str, required=True, help='S3 path to last week CSV')
    parser.add_argument('--this-week', type=str, required=True, help='S3 path to current week CSV')
    args = parser.parse_args()
    
    # Load config (for display settings only)
    config = load_config()
    sport_config = config['sports'][args.sport]
    
    # Print header
    emoji = sport_config['emoji']
    display_name = sport_config['display_name']
    print("=" * 80)
    print(f"{emoji} {display_name.upper()} CHAMPIONSHIP FUTURES ANALYSIS (S3-DRIVEN)")
    print("=" * 80 + "\n")
    
    # Load CSV files from S3
    print(f"📥 Loading data from S3...")
    print(f"   Preseason: {args.preseason}")
    preseason_df = load_csv_from_s3(args.preseason)
    
    print(f"   Last Week: {args.last_week}")
    last_week_df = load_csv_from_s3(args.last_week)
    
    print(f"   This Week: {args.this_week}")
    current_df = load_csv_from_s3(args.this_week)
    print(f"   ✅ Loaded\n")
    
    # Build analysis dataframe
    df = build_analysis_dataframe(preseason_df, last_week_df, current_df)
    
    # Apply top-n filter
    total_teams = len(df)
    if args.top_n < total_teams:
        df = df.head(args.top_n)
        print(f"📊 Showing top {args.top_n} of {total_teams} teams\n")
    
    # Print summary
    print("Data Sources:")
    print(f"  Preseason: {args.preseason}")
    print(f"  Last Week: {args.last_week}")
    print(f"  This Week: {args.this_week}")
    print(f"\nTeams analyzed: {len(df)}")
    print(f"Average vig: {df['vig_pct'].mean():.1f}%\n")
    
    # Print table
    print("=" * 140)
    print(f"{'Rank':<5} {'Team':<35} {'Current':<10} {'Implied%':<10} {'Fair%':<10} {'Vig%':<8} {'Δ Pre':<10} {'Δ LW':<10}")
    print("=" * 140)
    
    for i, row in df.iterrows():
        current_str = f"{int(row['current_odds']):+d}" if pd.notna(row['current_odds']) else "-"
        implied_str = f"{row['current_implied_prob']*100:.1f}%" if pd.notna(row['current_implied_prob']) else "-"
        fair_str = f"{row['fair_prob']*100:.1f}%" if pd.notna(row['fair_prob']) else "-"
        vig_str = f"{row['vig_pct']:.1f}%" if pd.notna(row['vig_pct']) else "-"
        diff_pre_str = f"{row['diff_preseason']:+.1f}pp" if pd.notna(row['diff_preseason']) else "-"
        diff_lw_str = f"{row['diff_last_week']:+.1f}pp" if pd.notna(row['diff_last_week']) else "-"
        
        print(f"{i+1:<5} {row['team']:<35} {current_str:<10} {implied_str:<10} {fair_str:<10} {vig_str:<8} {diff_pre_str:<10} {diff_lw_str:<10}")
    
    print("=" * 140)
    
    # Save output
    output_dir = repo_root / sport_config['output_dir']
    output_prefix = sport_config['output_prefix']
    output_file = output_dir / f'{output_prefix}_fair_odds.csv'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved to: {output_file}")
    
    # Upload to S3 if configured
    s3_bucket = sport_config.get('s3_output_bucket')
    s3_path = sport_config.get('s3_analysis_path')
    
    if s3_bucket and s3_path:
        try:
            s3_client = boto3.client('s3')
            
            s3_key = f"{s3_path}/{output_prefix}_fair_odds.csv"
            csv_data = df.to_csv(index=False)
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=s3_key,
                Body=csv_data,
                ContentType='text/csv'
            )
            print(f"☁️  Uploaded to s3://{s3_bucket}/{s3_key}")
        except Exception as e:
            print(f"⚠️  S3 upload failed: {e}")


if __name__ == '__main__':
    main()

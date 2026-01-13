"""
Analyze NFL QB rushing performance in first playoff game vs. subsequent games.

Hypothesis: Do QBs rush more in their first playoff game?

Usage:
    python3 analysis/analyze_qb_first_playoff_rush.py
    
Or drag into notebook:
    from analysis.analyze_qb_first_playoff_rush import load_and_analyze
    df = load_and_analyze()
"""

import pandas as pd
import boto3
from io import StringIO

S3_BUCKET = 'nfl-betting-mt'
S3_PREFIX = 'data/01_input/espn_web/playoffs/qb/gamelogs'


def load_all_qb_files():
    """Load all QB playoff files from S3 and mark first playoff game."""
    s3 = boto3.client('s3')
    
    # List all QB files
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
    files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.csv') and 'all_qb' not in obj['Key']]
    
    dfs = []
    for file_key in files:
        # Read CSV from S3
        obj = s3.get_object(Bucket=S3_BUCKET, Key=file_key)
        df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        
        # Sort by season and date
        df = df.sort_values(['season', 'date'])
        
        # Mark first playoff game
        df['first_playoff_game_binary'] = False
        df.iloc[0, df.columns.get_loc('first_playoff_game_binary')] = True
        
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


def analyze_first_game_rushing(df):
    """Compare rushing yards: first playoff game vs. all others."""
    
    # Convert to numeric
    df['rushing_yds'] = pd.to_numeric(df['rushing_yds'], errors='coerce')
    
    # Split into first vs. other games
    first_games = df[df['first_playoff_game_binary'] == True]
    other_games = df[df['first_playoff_game_binary'] == False]
    
    print("\n" + "="*80)
    print("QB RUSHING: FIRST PLAYOFF GAME vs. OTHER GAMES")
    print("="*80)
    
    print(f"\nFirst Playoff Games: {len(first_games)} games")
    print(f"  Mean rushing yards: {first_games['rushing_yds'].mean():.1f}")
    print(f"  Median rushing yards: {first_games['rushing_yds'].median():.1f}")
    print(f"  Std dev: {first_games['rushing_yds'].std():.1f}")
    
    print(f"\nOther Playoff Games: {len(other_games)} games")
    print(f"  Mean rushing yards: {other_games['rushing_yds'].mean():.1f}")
    print(f"  Median rushing yards: {other_games['rushing_yds'].median():.1f}")
    print(f"  Std dev: {other_games['rushing_yds'].std():.1f}")
    
    diff = first_games['rushing_yds'].mean() - other_games['rushing_yds'].mean()
    print(f"\nDifference: {diff:+.1f} yards (first game vs. others)")
    
    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(first_games['rushing_yds'].dropna(), 
                                       other_games['rushing_yds'].dropna())
    print(f"T-test p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("✅ Statistically significant difference!")
    else:
        print("❌ No significant difference")
    
    print("="*80)
    
    return df


def load_and_analyze():
    """Main function: load data and run analysis."""
    print("Loading QB playoff data from S3...")
    df = load_all_qb_files()
    print(f"✅ Loaded {len(df)} total playoff games")
    
    df = analyze_first_game_rushing(df)
    
    return df


if __name__ == '__main__':
    df = load_and_analyze()
    
    # Save for further analysis
    df.to_csv('data/04_output/qb_playoff_rush_analysis.csv', index=False)
    print(f"\n💾 Saved to: data/04_output/qb_playoff_rush_analysis.csv")


"""
Quick Analysis of Historical Player Props Data

Reads all player_threes CSV files and provides analysis
"""

import pandas as pd
from pathlib import Path
import glob

def load_all_props():
    """Load all player_threes CSV files into one DataFrame"""
    
    # Find all CSV files
    props_dir = Path(__file__).parent.parent / 'data' / '01_input' / 'the-odds-api' / 'nba' / 'historical_props'
    csv_files = glob.glob(str(props_dir / 'props_*_player_threes.csv'))
    
    if not csv_files:
        print("No CSV files found in the-odds-api/nba/historical_props/")
        return None
    
    print(f"Found {len(csv_files)} CSV files")
    print(f"Loading data...")
    
    # Load all files
    dfs = []
    for csv_file in sorted(csv_files):
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    # Combine into one DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)
    
    print(f"✅ Loaded {len(combined_df):,} total prop rows from {len(csv_files)} dates\n")
    
    return combined_df


def analyze_props(df):
    """Run quick analysis on the props data"""
    
    print("="*70)
    print("PLAYER PROPS ANALYSIS")
    print("="*70)
    
    # Basic stats
    print("\n📊 BASIC STATS")
    print(f"Total prop entries: {len(df):,}")
    print(f"Unique players: {df['player'].nunique()}")
    print(f"Unique games: {df['game'].nunique()}")
    print(f"Unique dates: {df['game_time'].apply(lambda x: x[:10]).nunique()}")
    print(f"Bookmakers: {', '.join(df['bookmaker'].unique())}")
    
    # Extract date from game_time
    df['date'] = pd.to_datetime(df['game_time']).dt.date
    
    # Props per date
    print("\n📅 PROPS PER DATE")
    props_per_date = df.groupby('date').size()
    print(f"Average props per date: {props_per_date.mean():.0f}")
    print(f"Min props in a date: {props_per_date.min()}")
    print(f"Max props in a date: {props_per_date.max()}")
    
    # Games per date
    print("\n🏀 GAMES PER DATE")
    games_per_date = df.groupby('date')['game'].nunique()
    print(f"Average games per date: {games_per_date.mean():.1f}")
    print(f"Min games in a date: {games_per_date.min()}")
    print(f"Max games in a date: {games_per_date.max()}")
    
    # Top players by appearances
    print("\n⭐ TOP 10 PLAYERS BY APPEARANCES")
    player_counts = df['player'].value_counts().head(10)
    for i, (player, count) in enumerate(player_counts.items(), 1):
        print(f"{i:2}. {player:30} - {count:4} props")
    
    # Line distribution
    print("\n📈 LINE DISTRIBUTION (3-point props)")
    line_dist = df['line'].value_counts().sort_index()
    for line, count in line_dist.items():
        pct = (count / len(df)) * 100
        print(f"  {line} threes: {count:5} props ({pct:5.1f}%)")
    
    # Bookmaker coverage
    print("\n🎰 BOOKMAKER COVERAGE")
    bookmaker_counts = df['bookmaker'].value_counts()
    for bookmaker, count in bookmaker_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {bookmaker:20} - {count:6} props ({pct:5.1f}%)")
    
    # Sample data
    print("\n📋 SAMPLE DATA (first 5 props)")
    print(df[['player', 'game', 'line', 'bookmaker', 'over_odds', 'under_odds']].head())
    
    # Data quality checks
    print("\n🔍 DATA QUALITY")
    missing_over = df['over_odds'].isna().sum()
    missing_under = df['under_odds'].isna().sum()
    missing_line = df['line'].isna().sum()
    
    print(f"Missing over_odds: {missing_over:,} ({missing_over/len(df)*100:.2f}%)")
    print(f"Missing under_odds: {missing_under:,} ({missing_under/len(df)*100:.2f}%)")
    print(f"Missing line: {missing_line:,} ({missing_line/len(df)*100:.2f}%)")
    
    # Props with both over and under
    both_odds = df[df['over_odds'].notna() & df['under_odds'].notna()]
    print(f"Props with both over/under: {len(both_odds):,} ({len(both_odds)/len(df)*100:.1f}%)")
    
    print("\n" + "="*70)
    
    return df


def main():
    """Main analysis function"""
    
    # Load data
    df = load_all_props()
    
    if df is None:
        return
    
    # Analyze
    df = analyze_props(df)
    
    # Save combined file for further analysis
    output_file = Path(__file__).parent.parent / 'data' / '01_input' / 'the-odds-api' / 'nba' / 'historical_props' / 'combined_props_player_threes.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Saved combined data to: {output_file}")


if __name__ == "__main__":
    main()


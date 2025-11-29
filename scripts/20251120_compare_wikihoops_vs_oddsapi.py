"""
Script to compare Wikihoops game counts with The-Odds-API calendar data.
Only shows mismatches that need manual verification.

Author: Script generated on 2024-11-20
"""

import pandas as pd
from pathlib import Path
import sys

def load_wikihoops_data(wikihoops_file):
    """Load the most recent Wikihoops scraped data."""
    if not Path(wikihoops_file).exists():
        print(f"Error: Wikihoops file not found: {wikihoops_file}")
        sys.exit(1)
    
    df = pd.read_csv(wikihoops_file)
    df['date'] = pd.to_datetime(df['date'])
    return df

def load_oddsapi_data(oddsapi_file):
    """Load The-Odds-API calendar data."""
    if not Path(oddsapi_file).exists():
        print(f"Error: Odds API file not found: {oddsapi_file}")
        sys.exit(1)
    
    df = pd.read_csv(oddsapi_file)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def compare_sources(wikihoops_df, oddsapi_df):
    """
    Compare game counts between Wikihoops and The-Odds-API.
    Returns DataFrame of mismatches.
    """
    # Merge on date
    merged = wikihoops_df.merge(
        oddsapi_df[['Date', 'Num_Games']], 
        left_on='date', 
        right_on='Date', 
        how='outer',
        suffixes=('_wiki', '_api')
    )
    
    # Use the date from either source
    merged['date'] = merged['date'].fillna(merged['Date'])
    merged = merged.drop('Date', axis=1)
    
    # Fill NaN values for comparison
    merged['num_games_wiki'] = merged['num_games'].fillna(-1)
    merged['num_games_api'] = merged['Num_Games'].fillna(-1)
    
    # Find mismatches
    mismatches = merged[merged['num_games_wiki'] != merged['num_games_api']].copy()
    
    # Clean up the display
    mismatches['date'] = mismatches['date'].dt.strftime('%Y-%m-%d')
    mismatches['num_games_wiki'] = mismatches['num_games_wiki'].replace(-1, 'Missing')
    mismatches['num_games_api'] = mismatches['num_games_api'].replace(-1, 'Missing')
    
    # Add a status column
    mismatches['issue_type'] = 'Count Mismatch'
    mismatches.loc[mismatches['num_games_wiki'] == 'Missing', 'issue_type'] = 'Missing from Wikihoops'
    mismatches.loc[mismatches['num_games_api'] == 'Missing', 'issue_type'] = 'Missing from Odds API'
    mismatches.loc[mismatches['status'] == 'error', 'issue_type'] = 'Wikihoops Scrape Error'
    
    return mismatches[['date', 'num_games_wiki', 'num_games_api', 'status', 'issue_type']].sort_values('date')

def main():
    """Main function to compare and display results."""
    
    # Set file paths
    project_root = Path(__file__).parent.parent
    
    # Look for the most recent Wikihoops file
    data_dir = project_root / 'data'
    wikihoops_files = sorted(data_dir.glob('wikihoops_games_count_*.csv'))
    
    if not wikihoops_files:
        print("Error: No Wikihoops data files found in data/ directory")
        print("Please run the scraper first: python scripts/20251120_scrape_wikihoops_games_count.py")
        sys.exit(1)
    
    wikihoops_file = wikihoops_files[-1]  # Most recent
    oddsapi_file = project_root / 'api_setup' / 'nba_calendar' / 'daily_summary_2024_25.csv'
    
    print("="*80)
    print("COMPARING WIKIHOOPS vs THE-ODDS-API GAME COUNTS")
    print("="*80)
    print(f"\nWikihoops Data: {wikihoops_file.name}")
    print(f"Odds API Data:  {oddsapi_file.name}")
    print()
    
    # Load data
    wikihoops_df = load_wikihoops_data(wikihoops_file)
    oddsapi_df = load_oddsapi_data(oddsapi_file)
    
    print(f"Wikihoops dates: {len(wikihoops_df)}")
    print(f"Odds API dates:  {len(oddsapi_df)}")
    
    # Compare
    mismatches = compare_sources(wikihoops_df, oddsapi_df)
    
    if len(mismatches) == 0:
        print("\n" + "="*80)
        print("✓ ALL GAME COUNTS MATCH! No manual verification needed.")
        print("="*80)
    else:
        print(f"\n{'='*80}")
        print(f"⚠ FOUND {len(mismatches)} MISMATCHES REQUIRING MANUAL VERIFICATION")
        print(f"{'='*80}\n")
        
        # Display mismatches in a nice format
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        print(mismatches.to_string(index=False))
        
        # Save to file
        output_file = data_dir / 'wikihoops_oddsapi_mismatches.csv'
        mismatches.to_csv(output_file, index=False)
        print(f"\n{'='*80}")
        print(f"Mismatches saved to: {output_file}")
        print(f"{'='*80}")
        
        # Summary by issue type
        print("\n" + "="*80)
        print("SUMMARY BY ISSUE TYPE:")
        print("="*80)
        summary = mismatches['issue_type'].value_counts()
        for issue_type, count in summary.items():
            print(f"  {issue_type}: {count}")

if __name__ == "__main__":
    main()




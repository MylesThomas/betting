"""
Script to scrape the number of NBA games per day from Wikihoops.com
Covers dates from 2024-01-01 to today.

Author: Script generated on 2024-11-20
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import pandas as pd
import re
from pathlib import Path
import urllib3

# Suppress SSL warnings when verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def scrape_games_count(date_str):
    """
    Scrape the number of games for a specific date from Wikihoops.
    
    Args:
        date_str: Date in format YYYY-MM-DD
        
    Returns:
        Tuple of (date_str, num_games, status)
    """
    url = f"https://wikihoops.com/games/{date_str}/?ref=gameday-page"
    
    try:
        # Add headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        # Disable SSL verification to avoid certificate issues on macOS
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for the text pattern "X GAMES OF HOOPS ON [DAY]"
        # This appears in an h3 or heading element
        text = soup.get_text()
        
        # Pattern to match "9 GAMES OF HOOPS ON MONDAY"
        pattern = r'(\d+)\s+GAMES?\s+OF\s+HOOPS\s+ON\s+\w+'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            num_games = int(match.group(1))
            return (date_str, num_games, 'success')
        else:
            # Check if it's a date with no games
            if 'no games' in text.lower() or 'no nba' in text.lower():
                return (date_str, 0, 'no_games')
            else:
                return (date_str, None, 'not_found')
                
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {date_str}: {e}")
        return (date_str, None, 'error')
    except Exception as e:
        print(f"Error parsing {date_str}: {e}")
        return (date_str, None, 'parse_error')

def generate_date_range(start_date, end_date):
    """Generate list of dates between start_date and end_date."""
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    return dates

def main():
    """Main function to scrape all dates and save results."""
    
    # Set date range
    start_date = datetime(2024, 1, 1)
    end_date = datetime.now()
    
    print(f"Scraping Wikihoops game counts from {start_date.date()} to {end_date.date()}")
    print(f"Total days to scrape: {(end_date - start_date).days + 1}")
    
    # Generate all dates
    dates = generate_date_range(start_date, end_date)
    
    # Store results
    results = []
    
    # Scrape each date
    for i, date_str in enumerate(dates, 1):
        print(f"[{i}/{len(dates)}] Scraping {date_str}...", end=' ')
        
        date_str_result, num_games, status = scrape_games_count(date_str)
        results.append({
            'date': date_str_result,
            'num_games': num_games,
            'status': status
        })
        
        print(f"Games: {num_games if num_games is not None else 'N/A'} ({status})")
        
        # Be polite - add delay between requests
        time.sleep(1)
        
        # Periodic save every 50 requests
        if i % 50 == 0:
            df = pd.DataFrame(results)
            output_path = Path(__file__).parent.parent / 'data' / 'wikihoops_games_count_temp.csv'
            output_path.parent.mkdir(exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"  → Saved temporary progress to {output_path}")
    
    # Save final results
    df = pd.DataFrame(results)
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(exist_ok=True)
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f'wikihoops_games_count_{timestamp}.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Scraping complete!")
    print(f"Total dates scraped: {len(results)}")
    print(f"Successful: {len(df[df['status'] == 'success'])}")
    print(f"No games: {len(df[df['status'] == 'no_games'])}")
    print(f"Errors: {len(df[df['status'].isin(['error', 'parse_error', 'not_found'])])}")
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}")
    
    # Print summary statistics
    if not df[df['num_games'].notna()].empty:
        print(f"\nGame Count Statistics:")
        print(f"  Total games scraped: {df['num_games'].sum()}")
        print(f"  Average games per day: {df['num_games'].mean():.2f}")
        print(f"  Max games in a day: {df['num_games'].max()}")
        print(f"  Days with games: {len(df[df['num_games'] > 0])}")

if __name__ == "__main__":
    main()


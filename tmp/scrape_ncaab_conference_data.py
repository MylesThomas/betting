"""
Scrape NCAA D1 basketball conference data from Wikipedia.

This script fetches the table from Wikipedia listing all D1 men's basketball programs
with their conferences, then saves to CSV and creates a conference mapping.

Source: https://en.wikipedia.org/wiki/List_of_NCAA_Division_I_men%27s_basketball_programs
"""

import pandas as pd
import requests
from io import StringIO
import ssl
import urllib.request
import urllib3

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# URL to Wikipedia page
URL = "https://en.wikipedia.org/wiki/List_of_NCAA_Division_I_men%27s_basketball_programs"

def scrape_conference_table():
    """
    Scrape the D1 basketball programs table from Wikipedia.
    
    Returns:
        pd.DataFrame with columns: School, Nickname, Conference
    """
    print(f"📥 Fetching data from Wikipedia...")
    print(f"   URL: {URL}\n")
    
    # Fetch HTML content with requests (handles SSL better)
    # Add user-agent to avoid Wikipedia blocking
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(URL, headers=headers, timeout=30, verify=False)
    response.raise_for_status()
    
    # Use pandas read_html to parse all tables from the HTML content
    tables = pd.read_html(StringIO(response.text))
    
    print(f"✅ Found {len(tables)} tables on the page\n")
    
    # The main table should be the largest one
    # It has columns: School, Nickname, Home arena, Conference, Tournament appearances, etc.
    main_table = tables[0]
    
    print(f"📊 Main table shape: {main_table.shape}")
    print(f"   Columns: {list(main_table.columns)}\n")
    
    # Extract relevant columns
    df = main_table[['School', 'Nickname', 'Conference']].copy()
    
    # Clean up the data
    print("🧹 Cleaning data...")
    
    # Remove any rows with NaN values
    df = df.dropna()
    
    # Strip whitespace
    df['School'] = df['School'].str.strip()
    df['Nickname'] = df['Nickname'].str.strip()
    df['Conference'] = df['Conference'].str.strip()
    
    # Create team name in format used by ESPN (School Nickname without "University of" etc)
    df['team_name_espn'] = df.apply(create_espn_team_name, axis=1)
    
    print(f"   ✅ {len(df)} teams loaded\n")
    
    return df


def create_espn_team_name(row):
    """
    Create team name in ESPN format from School and Nickname.
    
    ESPN typically uses: [Short School Name] [Nickname]
    Examples:
    - "Duke Blue Devils" (not "Duke University Blue Devils")
    - "North Carolina Tar Heels" (not "University of North Carolina Tar Heels")
    - "UConn Huskies" (not "University of Connecticut Huskies")
    """
    school = row['School']
    nickname = row['Nickname']
    
    # Remove common prefixes/suffixes
    school = school.replace('University of ', '')
    school = school.replace('University', '').strip()
    school = school.replace('College of ', '')
    school = school.replace('State University', 'State').strip()
    
    # Handle specific cases
    if 'UMass' in school:
        school = school  # Keep as is
    elif school.startswith('University at '):
        school = school.replace('University at ', '')
    
    # Clean up extra spaces
    school = ' '.join(school.split())
    
    # Combine
    team_name = f"{school} {nickname}".strip()
    
    return team_name


def save_to_csv(df, filepath='data/ncaab_conferences.csv'):
    """Save dataframe to CSV."""
    df.to_csv(filepath, index=False)
    print(f"💾 Saved to: {filepath}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}\n")


def create_conference_mapping(df):
    """
    Create a dictionary mapping team names to conferences.
    
    Returns:
        dict: {team_name: conference}
    """
    mapping = dict(zip(df['team_name_espn'], df['Conference']))
    return mapping


def print_conference_stats(df):
    """Print statistics about conferences."""
    print("📈 Conference Statistics:")
    print("=" * 80)
    
    conference_counts = df['Conference'].value_counts()
    
    for conf, count in conference_counts.items():
        print(f"   {conf:<35} {count:>3} teams")
    
    print("=" * 80)
    print(f"   TOTAL: {len(df)} teams across {len(conference_counts)} conferences\n")


def main():
    """Main execution."""
    print("=" * 80)
    print("NCAA D1 BASKETBALL CONFERENCE SCRAPER")
    print("=" * 80)
    print()
    
    # Scrape the data
    df = scrape_conference_table()
    
    # Print statistics
    print_conference_stats(df)
    
    # Save to CSV
    save_to_csv(df, 'tmp/ncaab_conferences.csv')
    
    # Show sample
    print("📋 Sample of data:")
    print(df[['team_name_espn', 'Conference']].head(20).to_string(index=False))
    print(f"   ... ({len(df) - 20} more teams)\n")
    
    # Create mapping
    conf_map = create_conference_mapping(df)
    
    print(f"✅ Conference mapping created with {len(conf_map)} teams")
    print(f"   Example: {list(conf_map.items())[0]}\n")
    
    print("=" * 80)
    print("✅ DONE!")
    print("=" * 80)
    
    return df, conf_map


if __name__ == "__main__":
    df, conf_map = main()


"""
Build NCAAB Team Name Mapping (Odds API → ESPN)

Purpose:
Extract unique team names from historical data (both Odds API and ESPN),
then create a mapping dictionary where Odds API names are keys and ESPN names are values.

Context:
- The Odds API uses abbreviations: "Boston Univ.", "Coppin St", "Miss Valley St"
- ESPN uses full names: "Boston University", "Coppin State", "Mississippi Valley State"
- ESPN is the source of truth
- Mapping needed for lambda_function_track_live_odds.py to match games to scores

Data Sources:
1. Odds API data: data/01_input/the-odds-api/ncaab/ (historical game lines)
2. ESPN data: Need to scrape or use existing game results data

Created: 2026-02-16
"""

import os
import sys
from pathlib import Path
import pandas as pd
from collections import defaultdict

# Find project root
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

DATA_ROOT = project_root / 'data'


def extract_odds_api_teams():
    """Extract all unique team names from Odds API data."""
    print("="*80)
    print("EXTRACTING ODDS API TEAM NAMES")
    print("="*80)
    
    teams = set()
    
    # Check multiple potential data sources
    sources = [
        DATA_ROOT / '01_input/the-odds-api/ncaab/futures',
        DATA_ROOT / '01_input/the-odds-api/ncaab/game_lines',
        DATA_ROOT / '04_output/ncaab',
    ]
    
    files_found = 0
    for source_dir in sources:
        if not source_dir.exists():
            print(f"⚠️  Directory not found: {source_dir}")
            continue
        
        # Process CSV files
        csv_files = list(source_dir.glob('*.csv'))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Look for team name columns
                team_cols = [col for col in df.columns if 'team' in col.lower()]
                
                for col in team_cols:
                    # Convert to string and filter out non-string values
                    team_values = df[col].dropna().astype(str).unique()
                    teams.update(team_values)
                
                files_found += 1
                if files_found <= 3:  # Show first few files
                    print(f"✅ Processed: {csv_file.name} ({len(team_cols)} team columns)")
            except Exception as e:
                print(f"⚠️  Error reading {csv_file.name}: {e}")
        
        # Process Parquet files
        parquet_files = list(source_dir.glob('*.parquet'))
        for pq_file in parquet_files:
            try:
                df = pd.read_parquet(pq_file)
                
                # Look for team name columns
                team_cols = [col for col in df.columns if 'team' in col.lower()]
                
                for col in team_cols:
                    # Convert to string and filter out non-string values
                    team_values = df[col].dropna().astype(str).unique()
                    teams.update(team_values)
                
                files_found += 1
                if files_found <= 3:
                    print(f"✅ Processed: {pq_file.name} ({len(team_cols)} team columns)")
            except Exception as e:
                print(f"⚠️  Error reading {pq_file.name}: {e}")
    
    if files_found > 3:
        print(f"... and {files_found - 3} more files")
    
    print(f"\n✅ Found {len(teams)} unique teams from Odds API\n")
    return sorted(teams)


def extract_espn_teams():
    """Extract all unique team names from ESPN data."""
    print("="*80)
    print("EXTRACTING ESPN TEAM NAMES")
    print("="*80)
    
    teams = set()
    
    # Check for ESPN game results data
    sources = [
        DATA_ROOT / '01_input/espn',
        DATA_ROOT / '02_cache/espn',
        DATA_ROOT / '01_input/game_results',
    ]
    
    files_found = 0
    for source_dir in sources:
        if not source_dir.exists():
            print(f"⚠️  Directory not found: {source_dir}")
            continue
        
        # Look for NCAAB files
        all_files = list(source_dir.rglob('*.csv')) + list(source_dir.rglob('*.parquet'))
        ncaab_files = [f for f in all_files if 'ncaab' in str(f).lower() or 'college' in str(f).lower()]
        
        for data_file in ncaab_files:
            try:
                if data_file.suffix == '.csv':
                    df = pd.read_csv(data_file)
                else:
                    df = pd.read_parquet(data_file)
                
                # Look for team name columns
                team_cols = [col for col in df.columns if 'team' in col.lower()]
                
                for col in team_cols:
                    # Convert to string and filter out non-string values
                    team_values = df[col].dropna().astype(str).unique()
                    teams.update(team_values)
                
                files_found += 1
                if files_found <= 3:
                    print(f"✅ Processed: {data_file.name} ({len(team_cols)} team columns)")
            except Exception as e:
                print(f"⚠️  Error reading {data_file.name}: {e}")
    
    if files_found == 0:
        print("⚠️  No ESPN data found - will need to fetch from API")
    else:
        if files_found > 3:
            print(f"... and {files_found - 3} more files")
        print(f"\n✅ Found {len(teams)} unique teams from ESPN\n")
    
    return sorted(teams)


def build_mapping(odds_teams, espn_teams):
    """
    Build mapping dictionary: Odds API team name → ESPN team name.
    
    Strategy:
    1. Exact matches (most teams)
    2. Fuzzy matches (expand abbreviations)
    3. Manual review list (couldn't auto-match)
    """
    print("="*80)
    print("BUILDING MAPPING")
    print("="*80)
    
    mapping = {}
    exact_matches = 0
    fuzzy_matches = 0
    unmatched = []
    
    # Create normalized lookup for ESPN teams
    espn_lookup = {team.lower(): team for team in espn_teams}
    
    for odds_team in odds_teams:
        # Try exact match first
        if odds_team in espn_teams:
            mapping[odds_team] = odds_team
            exact_matches += 1
            continue
        
        # Try case-insensitive match
        if odds_team.lower() in espn_lookup:
            mapping[odds_team] = espn_lookup[odds_team.lower()]
            exact_matches += 1
            continue
        
        # Try fuzzy matching (expand abbreviations)
        normalized = odds_team
        
        # "St " → "State "
        if " St " in normalized:
            normalized = normalized.replace(" St ", " State ")
        
        # "Univ." → "University"
        if "Univ." in normalized:
            normalized = normalized.replace("Univ.", "University")
        
        # "Miss " → "Mississippi " (at start)
        if normalized.startswith("Miss "):
            normalized = normalized.replace("Miss ", "Mississippi ", 1)
        
        # Check if normalized version matches
        if normalized in espn_teams:
            mapping[odds_team] = normalized
            fuzzy_matches += 1
            print(f"🔄 Fuzzy match: '{odds_team}' → '{normalized}'")
            continue
        
        if normalized.lower() in espn_lookup:
            mapping[odds_team] = espn_lookup[normalized.lower()]
            fuzzy_matches += 1
            print(f"🔄 Fuzzy match: '{odds_team}' → '{espn_lookup[normalized.lower()]}'")
            continue
        
        # Couldn't match - add to review list
        unmatched.append(odds_team)
    
    print(f"\n✅ Exact matches: {exact_matches}")
    print(f"🔄 Fuzzy matches: {fuzzy_matches}")
    print(f"⚠️  Unmatched: {len(unmatched)}")
    
    if unmatched:
        print(f"\n🔍 Teams needing manual review:")
        for team in unmatched[:20]:
            print(f"   '{team}'")
        if len(unmatched) > 20:
            print(f"   ... and {len(unmatched) - 20} more")
    
    return mapping, unmatched


def generate_mapping_code(mapping):
    """Generate Python dictionary code for the mapping."""
    print("\n" + "="*80)
    print("GENERATED MAPPING CODE")
    print("="*80)
    print("\nODDS_API_TO_ESPN_NCAAB = {")
    for odds_name, espn_name in sorted(mapping.items()):
        if odds_name != espn_name:  # Only include non-identical mappings
            print(f'    "{odds_name}": "{espn_name}",')
    print("}")
    print()


def main():
    print("\n" + "="*80)
    print("NCAAB TEAM NAME MAPPING BUILDER")
    print("="*80)
    print("\nThis script builds a mapping dictionary:")
    print("  Odds API team name → ESPN team name")
    print("\nESPN is the source of truth (for game scores/status)")
    print("Odds API names need to be normalized to match ESPN\n")
    
    # Extract team names from both sources
    odds_teams = extract_odds_api_teams()
    espn_teams = extract_espn_teams()
    
    if not odds_teams:
        print("❌ No Odds API teams found. Check data directory structure.")
        return
    
    if not espn_teams:
        print("⚠️  No ESPN teams found in local data.")
        print("   Options:")
        print("   1. Fetch from ESPN API (will gather teams from live scoreboards)")
        print("   2. Use existing game results data")
        print("   3. Continue with Odds API only (will generate fuzzy rules)")
        print()
        
        response = input("Fetch from ESPN API now? (y/n): ").lower()
        if response == 'y':
            print("\n🔄 Fetching NCAAB teams from ESPN API...")
            import requests
            
            url = 'http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard'
            response = requests.get(url, timeout=10)
            data = response.json()
            
            events = data.get('events', [])
            for event in events:
                competition = event['competitions'][0]
                for competitor in competition['competitors']:
                    espn_teams.append(competitor['team']['displayName'])
            
            espn_teams = sorted(set(espn_teams))
            print(f"✅ Fetched {len(espn_teams)} teams from ESPN (today's games only)\n")
    
    # Build mapping
    mapping, unmatched = build_mapping(odds_teams, espn_teams)
    
    # Generate code
    generate_mapping_code(mapping)
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Odds API teams: {len(odds_teams)}")
    print(f"ESPN teams: {len(espn_teams)}")
    print(f"Mapped: {len(mapping)}")
    print(f"Unmatched: {len(unmatched)}")
    print(f"Coverage: {len(mapping)/len(odds_teams)*100:.1f}%")
    print()
    
    if unmatched:
        print("⚠️  Action required:")
        print("   1. Review unmatched teams above")
        print("   2. Add manual mappings to ODDS_API_TO_ESPN_NCAAB dict")
        print("   3. Re-run this script to verify 100% coverage")
    else:
        print("✅ All teams mapped successfully!")
    print()


if __name__ == '__main__':
    main()

"""
Create conference mapping for NCAAB teams.

This script:
1. Loads our existing ESPN team names from the D1 team data
2. Loads Wikipedia conference data  
3. Matches them using fuzzy matching
4. Creates a final conference mapping dictionary

Usage:
    python tmp/create_conference_mapping.py --season 2024-25
"""

import pandas as pd
import sys
from pathlib import Path
from difflib import SequenceMatcher

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'tmp'))
sys.path.append(str(project_root / 'src'))

# Import the function to get D1 teams
from build_ncaab_team_name_mapping_v2 import get_d1_team_matching_df
from ncaab_conference_data import NCAAB_CONFERENCE_MAPPING_2025_26 as MANUAL_CONFERENCE_MAPPING


def normalize_for_matching(name):
    """
    Normalize team name for fuzzy matching.
    
    Remove common words that cause issues:
    - "University", "State", "of", "the"
    - Keep mascots and key identifiers
    """
    name = name.lower()
    
    # Remove common university terms
    for term in ['university', 'college', 'institute', 'of', 'the', 'at', 'state']:
        name = name.replace(term, '')
    
    # Clean up spaces
    name = ' '.join(name.split())
    
    return name


def fuzzy_match_score(str1, str2):
    """Calculate fuzzy match score between two strings."""
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def match_teams_to_conferences(espn_teams, wikipedia_df):
    """
    Match ESPN team names to Wikipedia conference data.
    
    Args:
        espn_teams: List of ESPN team names
        wikipedia_df: DataFrame with columns [School, Nickname, Conference, team_name_espn]
    
    Returns:
        dict: {espn_team_name: conference}
    """
    print("🔗 Matching ESPN teams to Wikipedia conferences...\n")
    
    mapping = {}
    unmatched_espn = []
    
    for espn_team in espn_teams:
        # Try manual mapping first
        if espn_team in MANUAL_CONFERENCE_MAPPING:
            conference = MANUAL_CONFERENCE_MAPPING[espn_team]
            mapping[espn_team] = conference
            continue
        
        # Try exact match (case insensitive)
        wiki_exact = wikipedia_df[
            wikipedia_df['team_name_espn'].str.lower() == espn_team.lower()
        ]
        
        if len(wiki_exact) > 0:
            conference = wiki_exact.iloc[0]['Conference']
            mapping[espn_team] = conference
            continue
        
        # Try fuzzy matching on normalized names
        espn_normalized = normalize_for_matching(espn_team)
        
        best_match = None
        best_score = 0
        
        for _, row in wikipedia_df.iterrows():
            wiki_team = row['team_name_espn']
            wiki_normalized = normalize_for_matching(wiki_team)
            
            score = fuzzy_match_score(espn_normalized, wiki_normalized)
            
            if score > best_score:
                best_score = score
                best_match = row
        
        # If we have a good match (>0.75), use it
        if best_score > 0.75:
            conference = best_match['Conference']
            mapping[espn_team] = conference
            print(f"   ✅ Matched: {espn_team} → {best_match['team_name_espn']} ({conference}) [score: {best_score:.2f}]")
        else:
            unmatched_espn.append(espn_team)
            print(f"   ❌ No match: {espn_team} (best: {best_match['team_name_espn']}, score: {best_score:.2f})")
    
    print(f"\n📊 Matching Results:")
    print(f"   Matched:   {len(mapping)} teams")
    print(f"   Unmatched: {len(unmatched_espn)} teams\n")
    
    if unmatched_espn:
        print(f"❌ Unmatched ESPN teams:")
        for team in unmatched_espn[:20]:  # Show first 20
            print(f"   - {team}")
        if len(unmatched_espn) > 20:
            print(f"   ... and {len(unmatched_espn) - 20} more")
        print()
    
    return mapping, unmatched_espn


def clean_conference_name(conference):
    """
    Clean conference name by removing footnote markers.
    
    Wikipedia has footnotes like "Big Ten[i]" which should just be "Big Ten"
    """
    # Remove anything in brackets
    import re
    conference = re.sub(r'\[.*?\]', '', conference)
    return conference.strip()


def main(season='2024-25'):
    """Main execution."""
    print("=" * 80)
    print("NCAAB CONFERENCE MAPPING CREATOR")
    print("=" * 80)
    print()
    
    # Load D1 teams from our data
    print(f"📥 Loading D1 teams for {season} season...")
    d1_df = get_d1_team_matching_df(season=season, use_cache=True)
    espn_teams = d1_df['team_name_espn'].tolist()
    print(f"   ✅ Loaded {len(espn_teams)} ESPN team names\n")
    
    # Load Wikipedia conference data
    print("📥 Loading Wikipedia conference data...")
    wiki_df = pd.read_csv('tmp/ncaab_conferences.csv')
    print(f"   ✅ Loaded {len(wiki_df)} Wikipedia teams\n")
    
    # Clean conference names
    wiki_df['Conference'] = wiki_df['Conference'].apply(clean_conference_name)
    
    # Match teams
    conf_mapping, unmatched = match_teams_to_conferences(espn_teams, wiki_df)
    
    # Save mapping to CSV
    output_df = pd.DataFrame([
        {'team_name_espn': team, 'conference': conf}
        for team, conf in sorted(conf_mapping.items())
    ])
    
    output_path = 'tmp/ncaab_conference_mapping.csv'
    output_df.to_csv(output_path, index=False)
    print(f"💾 Saved conference mapping to: {output_path}")
    print(f"   {len(output_df)} teams with conferences\n")
    
    # Print conference statistics
    print("📈 Conference Distribution:")
    print("=" * 80)
    conf_counts = output_df['conference'].value_counts()
    for conf, count in conf_counts.head(20).items():
        print(f"   {conf:<35} {count:>3} teams")
    print("=" * 80)
    print()
    
    # Show sample
    print("📋 Sample of mapping:")
    print(output_df.head(20).to_string(index=False))
    print(f"   ... ({len(output_df) - 20} more teams)\n")
    
    print("=" * 80)
    print("✅ DONE!")
    print("=" * 80)
    
    return output_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create NCAAB conference mapping')
    parser.add_argument('--season', default='2024-25', help='Season (e.g., 2024-25)')
    
    args = parser.parse_args()
    
    df = main(season=args.season)


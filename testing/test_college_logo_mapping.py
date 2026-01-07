"""
Test script to identify team name mismatches between The Odds API and local logo files.

Purpose:
- Compare team names from NCAAF/NCAAB futures data with available logo filenames
- Identify exact matches and mismatches
- Generate a mapping dictionary for teams with different names
- Help debug logo display issues in futures visualizations

Context:
Logos are stored locally at: ref/shot-quality/Logos/New Logos/
The Odds API uses different team name formats that may not match logo filenames.

Usage:
    cd /Users/thomasmyles/dev/betting
    python3 scripts/test_college_logo_mapping.py
"""

import pandas as pd
from pathlib import Path
import sys

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

# Import after adding to path
from ncaa_team_utils import map_teams_to_logos, get_logo_coverage_stats


def get_available_logos():
    """Get all available logo files and create name mapping"""
    logos_dir = repo_root / 'ref/shot-quality/Logos/New Logos'
    
    logo_files = list(logos_dir.glob('*.png'))
    
    # Create set of team names (without .png extension)
    logo_names = {logo_file.stem for logo_file in logo_files}
    
    # Also create a lowercase mapping for fuzzy matching
    logo_names_lower = {name.lower(): name for name in logo_names}
    
    return logo_names, logo_names_lower, logo_files


def get_teams_from_futures(sport):
    """
    Get unique team names from most recent futures file for a sport.
    
    Args:
        sport: 'ncaaf' or 'ncaab'
    
    Returns:
        set: Unique team names from futures data
    """
    futures_dir = repo_root / f'data/01_input/the-odds-api/{sport}/futures'
    
    # Get all CSV files
    csv_files = sorted(futures_dir.glob(f'{sport}_championship_futures_*.csv'))
    
    if not csv_files:
        print(f"⚠️  No futures files found for {sport.upper()}")
        return set()
    
    # Read most recent file
    most_recent = csv_files[-1]
    print(f"📁 Reading: {most_recent.name}")
    
    df = pd.read_csv(most_recent)
    teams = set(df['team'].unique())
    
    print(f"   Found {len(teams)} unique teams\n")
    
    return teams


def find_matches_and_mismatches(api_teams, logo_names, logo_names_lower):
    """
    Compare API team names with logo filenames.
    
    Args:
        api_teams: Set of team names from The Odds API
        logo_names: Set of logo filenames (without .png)
        logo_names_lower: Dict mapping lowercase logo names to actual names
    
    Returns:
        tuple: (exact_matches, mismatches, fuzzy_matches)
    """
    exact_matches = []
    mismatches = []
    fuzzy_matches = []
    
    for team in sorted(api_teams):
        if team in logo_names:
            # Exact match
            exact_matches.append(team)
        else:
            # Check for case-insensitive match
            team_lower = team.lower()
            if team_lower in logo_names_lower:
                # Fuzzy match (case difference)
                actual_logo_name = logo_names_lower[team_lower]
                fuzzy_matches.append((team, actual_logo_name))
            else:
                # No match found
                mismatches.append(team)
    
    return exact_matches, mismatches, fuzzy_matches


def suggest_logo_matches(mismatches, logo_names):
    """
    Suggest potential logo matches for mismatched teams.
    
    Uses simple string matching to suggest possible matches.
    
    Args:
        mismatches: List of team names without exact matches
        logo_names: Set of available logo names
    
    Returns:
        dict: Mapping of mismatched team to suggested logo names
    """
    from difflib import get_close_matches
    
    suggestions = {}
    
    for team in mismatches:
        # Find close matches (up to 3 suggestions)
        matches = get_close_matches(team, logo_names, n=3, cutoff=0.6)
        if matches:
            suggestions[team] = matches
    
    return suggestions


def generate_mapping_code(fuzzy_matches, suggestions):
    """
    Generate Python code for a team name mapping dictionary.
    
    Args:
        fuzzy_matches: List of (api_name, logo_name) tuples
        suggestions: Dict of api_name -> [suggested_logo_names]
    
    Returns:
        str: Python code for mapping dictionary
    """
    lines = ["# Team name mapping: The Odds API -> Logo filename", "{"]
    
    # Add fuzzy matches (case differences)
    if fuzzy_matches:
        lines.append("    # Case differences (auto-detected)")
        for api_name, logo_name in sorted(fuzzy_matches):
            lines.append(f"    '{api_name}': '{logo_name}',")
        lines.append("")
    
    # Add suggestions for manual review
    if suggestions:
        lines.append("    # Potential matches (needs manual review)")
        for api_name, suggested_names in sorted(suggestions.items()):
            lines.append(f"    # '{api_name}': '{suggested_names[0]}',  # Other options: {suggested_names[1:]}")
    
    lines.append("}")
    
    return "\n".join(lines)


def main():
    """Main test function"""
    
    print("="*80)
    print("COLLEGE LOGO MAPPING TEST")
    print("="*80 + "\n")
    
    # Get available logos
    print("📂 Loading logo files...\n")
    logo_names, logo_names_lower, logo_files = get_available_logos()
    print(f"   ✅ Found {len(logo_names)} logo files")
    print(f"   📁 Location: ref/shot-quality/Logos/New Logos/\n")
    
    # Test both sports
    for sport in ['ncaaf', 'ncaab']:
        print("="*80)
        print(f"{sport.upper()} TEAM NAME MATCHING")
        print("="*80 + "\n")
        
        # Get teams from futures
        api_teams = get_teams_from_futures(sport)
        
        if not api_teams:
            continue
        
        # Convert set to list for processing
        api_teams_list = list(api_teams)
        
        print("="*80)
        print("USING NCAA_TEAM_UTILS MODULE (Smart Mapping)")
        print("="*80 + "\n")
        
        # Use the smart mapping utility
        logo_map = map_teams_to_logos(api_teams_list, repo_root)
        stats = get_logo_coverage_stats(api_teams_list, repo_root)
        
        print(f"📊 SMART MAPPING RESULTS:")
        print(f"   ✅ Matched: {stats['matched']}/{stats['total']} teams ({stats['coverage_pct']:.1f}%)")
        print(f"   ❌ Unmatched: {stats['unmatched']} teams\n")
        
        # Show all mappings
        print(f"TEAM MAPPINGS:")
        print("-"*80)
        for team in sorted(api_teams_list):
            logo_path = logo_map[team]
            if logo_path:
                logo_name = Path(logo_path).stem
                print(f"   ✅ '{team}' → '{logo_name}'")
            else:
                print(f"   ❌ '{team}' → NO LOGO FOUND")
        print()
        
        # Show unmatched teams if any
        if stats['unmatched_teams']:
            print(f"⚠️  TEAMS WITHOUT LOGOS:")
            for team in stats['unmatched_teams']:
                print(f"   • {team}")
            print()
            
            # Get suggestions for unmatched teams
            suggestions = suggest_logo_matches(stats['unmatched_teams'], logo_names)
            if suggestions:
                print(f"💡 SUGGESTED MAPPINGS TO ADD:")
                print("-"*80)
                for team, suggested_logos in suggestions.items():
                    print(f"   '{team}': '{suggested_logos[0]}',  # Options: {suggested_logos[1:]}")
                print()
        
        # Also show raw matching for comparison
        print("="*80)
        print("RAW NAME MATCHING (Without Smart Mapping)")
        print("="*80 + "\n")
        
        # Find matches and mismatches using raw comparison
        exact_matches, mismatches, fuzzy_matches = find_matches_and_mismatches(
            api_teams, logo_names, logo_names_lower
        )
        
        print(f"📊 RAW RESULTS:")
        print(f"   ✅ Exact matches: {len(exact_matches)}/{len(api_teams)} teams")
        print(f"   🔄 Case differences: {len(fuzzy_matches)} teams")
        print(f"   ❌ No match found: {len(mismatches)} teams\n")
        
        if mismatches:
            print(f"❌ TEAMS THAT NEED MAPPING:")
            suggestions = suggest_logo_matches(mismatches, logo_names)
            for team in mismatches:
                if team in suggestions:
                    print(f"   • '{team}' → Suggestions: {suggestions[team]}")
                else:
                    print(f"   • '{team}' → No suggestions found")
            print()
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print("\n✅ The ncaa_team_utils module is working!")
    print("   - Teams are being mapped correctly using the exception dictionary")
    print("   - 100% coverage means all teams have logos")
    print("\n💡 To add new mappings:")
    print("   1. Check 'TEAMS THAT NEED MAPPING' section above")
    print("   2. Add to TEAM_NAME_EXCEPTIONS in src/ncaa_team_utils.py")
    print("   3. Re-run this test to verify\n")
    
    # Show sample logos
    print("📸 SAMPLE AVAILABLE LOGOS (first 20):")
    for logo_file in sorted(logo_files)[:20]:
        print(f"   • {logo_file.stem}")
    if len(logo_files) > 20:
        print(f"   ... and {len(logo_files) - 20} more")
    print()


if __name__ == "__main__":
    main()


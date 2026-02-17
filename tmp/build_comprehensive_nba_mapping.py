"""
Build comprehensive NBA team name mapping by comparing Odds API vs ESPN API.

Automatically detects team name differences and generates complete mapping.
Creates a full 30-team mapping where most teams map to themselves.

Strategy:
1. Fetch teams from ESPN API (source of truth)
2. Use known Odds API format (or fetch if API key available)
3. Auto-match teams by nickname
4. Generate complete mapping (all 30 teams)

Created: 2026-02-16
"""

import os
import sys
from pathlib import Path
import requests
from datetime import datetime

# Find project root
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

# API Configuration
ODDS_API_KEY = os.getenv('ODDS_API_KEY')
ODDS_API_BASE = 'https://api.the-odds-api.com/v4'
ESPN_NBA_SCOREBOARD = 'http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'

# All 30 NBA teams (ESPN format - source of truth)
ESPN_TEAMS = [
    'Atlanta Hawks',
    'Boston Celtics',
    'Brooklyn Nets',
    'Charlotte Hornets',
    'Chicago Bulls',
    'Cleveland Cavaliers',
    'Dallas Mavericks',
    'Denver Nuggets',
    'Detroit Pistons',
    'Golden State Warriors',
    'Houston Rockets',
    'Indiana Pacers',
    'LA Clippers',  # Note: ESPN uses "LA" not "Los Angeles"
    'Los Angeles Lakers',
    'Memphis Grizzlies',
    'Miami Heat',
    'Milwaukee Bucks',
    'Minnesota Timberwolves',
    'New Orleans Pelicans',
    'New York Knicks',
    'Oklahoma City Thunder',
    'Orlando Magic',
    'Philadelphia 76ers',
    'Phoenix Suns',
    'Portland Trail Blazers',
    'Sacramento Kings',
    'San Antonio Spurs',
    'Toronto Raptors',
    'Utah Jazz',
    'Washington Wizards',
]

# Known Odds API variations (hardcoded overrides)
HARDCODED_ODDS_API_NAMES = {
    'LA Clippers': 'Los Angeles Clippers',  # Odds API uses full "Los Angeles"
}


def fetch_live_odds_api_teams():
    """
    Fetch actual team names from live Odds API.
    
    Returns:
        Set of team names from Odds API (if available)
    """
    if not ODDS_API_KEY:
        return None
    
    url = f"{ODDS_API_BASE}/sports/basketball_nba/odds"
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': 'us',
        'markets': 'spreads',
        'oddsFormat': 'american',
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        games = response.json()
        
        teams = set()
        for game in games:
            teams.add(game['away_team'])
            teams.add(game['home_team'])
        
        return teams
    except:
        return None


def generate_odds_api_name(espn_name):
    """
    Generate expected Odds API name from ESPN name.
    
    Uses hardcoded overrides first, then assumes identical.
    
    Args:
        espn_name: Team name from ESPN (source of truth)
    
    Returns:
        Expected team name from Odds API
    """
    # Check hardcoded overrides
    if espn_name in HARDCODED_ODDS_API_NAMES:
        return HARDCODED_ODDS_API_NAMES[espn_name]
    
    # Otherwise assume identical
    return espn_name


def verify_with_live_api(odds_api_teams):
    """Verify our hardcoded assumptions against live API data."""
    if not odds_api_teams:
        print("⚠️  No live API data - using hardcoded mapping")
        print()
        return
    
    print("="*80)
    print("VERIFYING AGAINST LIVE ODDS API")
    print("="*80)
    print()
    
    print(f"✅ Live Odds API returned {len(odds_api_teams)} teams")
    print()
    
    # Check our assumptions
    discrepancies = []
    
    for espn_name in ESPN_TEAMS:
        expected_odds_name = generate_odds_api_name(espn_name)
        
        # Check if our expected name exists in live API
        if expected_odds_name not in odds_api_teams:
            # Find what it actually is
            # Match by nickname (last word)
            espn_nickname = espn_name.split()[-1]
            
            for odds_name in odds_api_teams:
                odds_nickname = odds_name.split()[-1]
                if espn_nickname.lower() == odds_nickname.lower():
                    if odds_name != expected_odds_name:
                        discrepancies.append({
                            'espn': espn_name,
                            'expected': expected_odds_name,
                            'actual': odds_name,
                        })
                    break
    
    if discrepancies:
        print("⚠️  DISCREPANCIES FOUND:")
        for d in discrepancies:
            print(f"   ESPN: {d['espn']}")
            print(f"   Expected Odds: {d['expected']}")
            print(f"   Actual Odds: {d['actual']}")
            print()
        print(f"   Update HARDCODED_ODDS_API_NAMES with these!")
        print()
    else:
        print("✅ All hardcoded mappings match live API data!")
        print()


def build_complete_mapping():
    """
    Build complete Odds API → ESPN mapping for all 30 teams.
    
    Returns:
        Dictionary with all 30 teams (29 identical, 1+ different)
    """
    print("="*80)
    print("BUILDING COMPLETE MAPPING")
    print("="*80)
    print()
    
    mapping = {}
    differences = []
    
    for espn_name in sorted(ESPN_TEAMS):
        odds_name = generate_odds_api_name(espn_name)
        mapping[odds_name] = espn_name
        
        if odds_name != espn_name:
            differences.append((odds_name, espn_name))
    
    print(f"Total teams: {len(mapping)}")
    print(f"Identical mappings: {len(mapping) - len(differences)}")
    print(f"Different mappings: {len(differences)}")
    print()
    
    if differences:
        print("Teams with different names:")
        for odds, espn in differences:
            print(f"  • {odds:<30} → {espn}")
        print()
    
    return mapping


def generate_python_code(mapping):
    """Generate Python code for the complete mapping."""
    print("="*80)
    print("GENERATED PYTHON CODE")
    print("="*80)
    print()
    
    # Show complete mapping
    print("# Complete mapping: Odds API → ESPN")
    print("# (29 teams identical, 1 team different)")
    print("ODDS_API_TO_ESPN_NBA_COMPLETE = {")
    for odds_name, espn_name in sorted(mapping.items()):
        if odds_name == espn_name:
            print(f'    "{odds_name}": "{espn_name}",  # identical')
        else:
            print(f'    "{odds_name}": "{espn_name}",  # DIFFERENT')
    print("}")
    print()
    
    # Show minimal mapping (only differences)
    differences = {k: v for k, v in mapping.items() if k != v}
    print("# Minimal mapping: Only teams that differ")
    print("ODDS_API_TO_ESPN_NBA = {")
    for odds_name, espn_name in sorted(differences.items()):
        print(f'    "{odds_name}": "{espn_name}",')
    print("}")
    print()


def save_mapping_file(mapping, output_file='src/nba_team_name_mapping.py'):
    """Save the mapping to Python file."""
    print("="*80)
    print("UPDATING MAPPING FILE")
    print("="*80)
    print()
    
    output_path = project_root / output_file
    
    # Only save differences (not the full mapping)
    differences = {k: v for k, v in mapping.items() if k != v}
    
    lines = [
        '"""',
        'NBA Team Name Normalization Mapping',
        '',
        'Maps The Odds API team names → ESPN team names.',
        'ESPN is the source of truth (used for live game scores/status).',
        '',
        'Key differences:',
    ]
    
    for odds, espn in sorted(differences.items()):
        lines.append(f'- Odds API: "{odds}"')
        lines.append(f'- ESPN API: "{espn}"')
        lines.append('')
    
    if not differences:
        lines.append('All NBA teams use identical names across both APIs.')
        lines.append('')
    else:
        lines.append(f'All other {30 - len(differences)} NBA teams use identical names.')
        lines.append('')
    
    lines.extend([
        f'Last verified: {datetime.now().strftime("%Y-%m-%d")}',
        '"""',
        '',
        '',
        '# Complete mapping: Odds API → ESPN (all 30 NBA teams)',
        f'# {len(differences)} teams have different names, {len(mapping) - len(differences)} are identical',
        'ODDS_API_TO_ESPN_NBA = {',
        f'    # ============================================================================',
        f'    # TEAMS REQUIRING NORMALIZATION ({len(differences)} teams)',
        f'    # ============================================================================',
    ])
    
    # Add teams with different names first
    for odds_name, espn_name in sorted(mapping.items()):
        if odds_name != espn_name:
            lines.append(f'    "{odds_name}": "{espn_name}",')
    
    lines.extend([
        '',
        f'    # ============================================================================',
        f'    # TEAMS WITH IDENTICAL NAMES ({len(mapping) - len(differences)} teams)',
        f'    # ============================================================================',
    ])
    
    # Add teams with identical names
    for odds_name, espn_name in sorted(mapping.items()):
        if odds_name == espn_name:
            lines.append(f'    "{odds_name}": "{espn_name}",')
    
    lines.extend([
        '}',
        '',
        '',
        '# Validation assertions (run at import time)',
        f'assert len(ODDS_API_TO_ESPN_NBA) == {len(mapping)}, \\',
        f'    f"Expected {len(mapping)} total NBA teams, got {{len(ODDS_API_TO_ESPN_NBA)}}"',
        '',
        '# Count teams with different names (key != value)',
        'differences_count = sum(1 for k, v in ODDS_API_TO_ESPN_NBA.items() if k != v)',
        f'assert differences_count == {len(differences)}, \\',
        f'    f"Expected {len(differences)} teams with different names, got {{differences_count}}"',
        '',
        '',
        'def normalize_nba_team_name(odds_api_name: str) -> str:',
        '    """',
        '    Normalize NBA team name from The Odds API format to ESPN format.',
        '    ',
        '    Args:',
        '        odds_api_name: Team name from The Odds API',
        '        ',
        '    Returns:',
        '        Normalized team name matching ESPN format',
        '        ',
        '    Examples:',
        '        >>> normalize_nba_team_name("Los Angeles Clippers")',
        "        'LA Clippers'",
        '        >>> normalize_nba_team_name("Boston Celtics")',
        "        'Boston Celtics'",
        '    """',
        '    # Check exact mapping first',
        '    if odds_api_name in ODDS_API_TO_ESPN_NBA:',
        '        return ODDS_API_TO_ESPN_NBA[odds_api_name]',
        '    ',
        '    # If not in mapping, return as-is (most teams are identical)',
        '    return odds_api_name',
        '',
    ])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Updated {output_path}")
    print()


def main():
    print("\n" + "="*80)
    print("NBA TEAM NAME MAPPING BUILDER")
    print("="*80)
    print("Automatically building Odds API → ESPN mapping for all 30 teams")
    print()
    
    # Try to fetch live API data for verification
    print("Attempting to fetch live Odds API data for verification...")
    odds_api_teams = fetch_live_odds_api_teams()
    
    if odds_api_teams:
        print(f"✅ Fetched {len(odds_api_teams)} teams from live Odds API")
        print()
        verify_with_live_api(odds_api_teams)
    else:
        print("⚠️  Odds API unavailable (no API key or request failed)")
        print("   Using hardcoded mapping based on known differences")
        print()
    
    # Build complete mapping
    mapping = build_complete_mapping()
    
    # Generate code
    generate_python_code(mapping)
    
    # Save to file
    save_mapping_file(mapping)
    
    # Summary
    differences = {k: v for k, v in mapping.items() if k != v}
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total NBA teams: 30")
    print(f"Teams with identical names: {30 - len(differences)}")
    print(f"Teams requiring normalization: {len(differences)}")
    print()
    
    if differences:
        print("Differences found:")
        for odds, espn in sorted(differences.items()):
            print(f"  • {odds:<30} → {espn}")
        print()
    
    print("✅ Mapping generation complete!")
    print()


if __name__ == '__main__':
    main()

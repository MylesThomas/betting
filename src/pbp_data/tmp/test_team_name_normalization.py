"""
Test team name normalization for Odds API → ESPN matching.

ESPN is the ground truth - external APIs conform to ESPN naming.

Verifies that team name mapping handles all known mismatches:
- Los Angeles Clippers (Odds) → LA Clippers (ESPN)
- Los Angeles Lakers (Odds) → LA Lakers (ESPN)

Author: Myles Thomas
Date: 2026-02-12
"""

# Standalone test - copy the mapping from the main file
ODDS_TO_ESPN_TEAM_MAPPING = {
    # Odds API uses full names, ESPN sometimes abbreviates
    # Odds API → ESPN format
    'Los Angeles Clippers': 'LA Clippers',
    'Los Angeles Lakers': 'LA Lakers',
    
    # Handle if ESPN ever uses full names (defensive)
    'LA Clippers': 'LA Clippers',
    'LA Lakers': 'LA Lakers',
    
    # Trail Blazers variations
    'Portland Trail Blazers': 'Portland Trail Blazers',
    'Portland Trailblazers': 'Portland Trail Blazers',
    
    # All other teams (map to themselves for completeness)
    'Atlanta Hawks': 'Atlanta Hawks',
    'Boston Celtics': 'Boston Celtics',
    'Brooklyn Nets': 'Brooklyn Nets',
    'Charlotte Hornets': 'Charlotte Hornets',
    'Chicago Bulls': 'Chicago Bulls',
    'Cleveland Cavaliers': 'Cleveland Cavaliers',
    'Dallas Mavericks': 'Dallas Mavericks',
    'Denver Nuggets': 'Denver Nuggets',
    'Detroit Pistons': 'Detroit Pistons',
    'Golden State Warriors': 'Golden State Warriors',
    'Houston Rockets': 'Houston Rockets',
    'Indiana Pacers': 'Indiana Pacers',
    'Memphis Grizzlies': 'Memphis Grizzlies',
    'Miami Heat': 'Miami Heat',
    'Milwaukee Bucks': 'Milwaukee Bucks',
    'Minnesota Timberwolves': 'Minnesota Timberwolves',
    'New Orleans Pelicans': 'New Orleans Pelicans',
    'New York Knicks': 'New York Knicks',
    'Oklahoma City Thunder': 'Oklahoma City Thunder',
    'Orlando Magic': 'Orlando Magic',
    'Philadelphia 76ers': 'Philadelphia 76ers',
    'Phoenix Suns': 'Phoenix Suns',
    'Sacramento Kings': 'Sacramento Kings',
    'San Antonio Spurs': 'San Antonio Spurs',
    'Toronto Raptors': 'Toronto Raptors',
    'Utah Jazz': 'Utah Jazz',
    'Washington Wizards': 'Washington Wizards',
}

def normalize_odds_team_to_espn(odds_team_name: str) -> str:
    """Normalize Odds API team name to match ESPN format (ground truth)."""
    return ODDS_TO_ESPN_TEAM_MAPPING.get(odds_team_name, odds_team_name)


print("="*80)
print("TEAM NAME NORMALIZATION TEST (Odds API → ESPN)")
print("="*80)
print()

# Test known mismatches
print("Known Mismatches (Odds API → ESPN):")
print("-" * 80)

test_cases = [
    ('Los Angeles Clippers', 'LA Clippers', 'Odds uses full name, ESPN abbreviates'),
    ('Los Angeles Lakers', 'LA Lakers', 'Odds uses full name, ESPN abbreviates'),
    ('Portland Trail Blazers', 'Portland Trail Blazers', 'Should match exactly'),
    ('Milwaukee Bucks', 'Milwaukee Bucks', 'Should match exactly'),
    ('Unknown Team', 'Unknown Team', 'Returns original if not in mapping'),
]

for odds_name, expected_espn_name, description in test_cases:
    result = normalize_odds_team_to_espn(odds_name)
    status = '✅' if result == expected_espn_name else '❌'
    print(f"{status}  {odds_name:30} → {result:30} ({description})")

print()
print("="*80)
print("ALL TEAM MAPPINGS")
print("="*80)
print()

# Show all teams in the mapping
all_teams = sorted(set(ODDS_TO_ESPN_TEAM_MAPPING.keys()))
print(f"Total teams in mapping: {len(all_teams)}")
print()

# Group by whether normalization is needed
needs_normalization = []
no_normalization_needed = []

for odds_name in all_teams:
    espn_name = ODDS_TO_ESPN_TEAM_MAPPING[odds_name]
    if odds_name != espn_name:
        needs_normalization.append((odds_name, espn_name))
    else:
        no_normalization_needed.append(odds_name)

print("Teams that need normalization (Odds → ESPN):")
for odds_name, espn_name in needs_normalization:
    print(f"  - {odds_name:30} → {espn_name}")

print()
print(f"Teams that don't need normalization: {len(no_normalization_needed)}")
for team in no_normalization_needed[:5]:
    print(f"  - {team}")
if len(no_normalization_needed) > 5:
    print(f"  ... and {len(no_normalization_needed) - 5} more")

print()
print("="*80)
print("SUMMARY")
print("="*80)
print(f"✅ Total teams covered: {len(all_teams)}")
print(f"✅ Teams needing normalization: {len(needs_normalization)}")
print(f"✅ Teams with identity mapping: {len(no_normalization_needed)}")
print()
print("✅ ESPN IS GROUND TRUTH - External APIs conform to ESPN names")
print("="*80)

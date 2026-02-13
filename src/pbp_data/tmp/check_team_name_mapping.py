"""
Check for team name mismatches between ESPN and The Odds API.

Common issues:
- LA Clippers vs Los Angeles Clippers
- LA Lakers vs Los Angeles Lakers
- Different abbreviations

This script will:
1. Fetch all teams from both APIs
2. Compare team names
3. Identify mismatches
4. Generate a normalization mapping
"""
import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).resolve()
while not (project_root / '.gitignore').exists():
    project_root = project_root.parent
    if project_root == project_root.parent:
        raise FileNotFoundError("Could not find project root (no .gitignore)")

env_path = project_root / '.env'
load_dotenv(env_path)

ODDS_API_KEY = os.environ.get('ODDS_API_KEY', '')

if not ODDS_API_KEY:
    print("❌ ODDS_API_KEY not found in .env")
    sys.exit(1)

print(f"✅ API Key loaded: {ODDS_API_KEY[:8]}...")
print()

# ============================================================================
# STEP 1: Get ESPN teams
# ============================================================================

print("="*80)
print("STEP 1: Fetch ESPN scoreboard")
print("="*80)

espn_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
try:
    response = requests.get(espn_url, timeout=10, verify=False)
    response.raise_for_status()
    
    data = response.json()
    espn_games = data.get('events', [])
    
    espn_teams = set()
    for game in espn_games:
        competitors = game.get('competitions', [{}])[0].get('competitors', [])
        for comp in competitors:
            team_name = comp.get('team', {}).get('displayName', '')
            if team_name:
                espn_teams.add(team_name)
    
    print(f"✅ Found {len(espn_teams)} unique teams from ESPN")
    print("\nESPN Team Names:")
    for team in sorted(espn_teams):
        print(f"  - {team}")
    print()
    
except Exception as e:
    print(f"❌ Exception: {e}")
    sys.exit(1)

# ============================================================================
# STEP 2: Get Odds API teams
# ============================================================================

print("="*80)
print("STEP 2: Fetch The Odds API events")
print("="*80)

url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
params = {'apiKey': ODDS_API_KEY}

try:
    response = requests.get(url, params=params, timeout=10, verify=False)
    response.raise_for_status()
    
    events = response.json()
    
    odds_teams = set()
    for event in events:
        home_team = event.get('home_team')
        away_team = event.get('away_team')
        if home_team:
            odds_teams.add(home_team)
        if away_team:
            odds_teams.add(away_team)
    
    print(f"✅ Found {len(odds_teams)} unique teams from Odds API")
    print("\nOdds API Team Names:")
    for team in sorted(odds_teams):
        print(f"  - {team}")
    print()
    
except Exception as e:
    print(f"❌ Exception: {e}")
    sys.exit(1)

# ============================================================================
# STEP 3: Compare and find mismatches
# ============================================================================

print("="*80)
print("STEP 3: Compare team names")
print("="*80)

# Find exact matches
exact_matches = espn_teams & odds_teams
print(f"✅ Exact matches: {len(exact_matches)}")
for team in sorted(exact_matches):
    print(f"  ✓ {team}")
print()

# Find ESPN teams not in Odds API
espn_only = espn_teams - odds_teams
if espn_only:
    print(f"⚠️  ESPN teams not in Odds API: {len(espn_only)}")
    for team in sorted(espn_only):
        print(f"  • {team}")
    print()

# Find Odds API teams not in ESPN
odds_only = odds_teams - espn_teams
if odds_only:
    print(f"⚠️  Odds API teams not in ESPN: {len(odds_only)}")
    for team in sorted(odds_only):
        print(f"  • {team}")
    print()

# ============================================================================
# STEP 4: Attempt fuzzy matching for mismatches
# ============================================================================

print("="*80)
print("STEP 4: Fuzzy matching for mismatches")
print("="*80)

def normalize_team_name(name: str) -> str:
    """Normalize team name for comparison"""
    # Remove common prefixes/variations
    name = name.lower().strip()
    name = name.replace('la ', 'los angeles ')
    name = name.replace('ny ', 'new york ')
    return name

# Build normalization mapping
team_mapping = {}

for espn_team in espn_only:
    best_match = None
    best_score = 0
    
    espn_norm = normalize_team_name(espn_team)
    
    for odds_team in odds_only:
        odds_norm = normalize_team_name(odds_team)
        
        # Simple substring matching
        if espn_norm in odds_norm or odds_norm in espn_norm:
            # Count matching words
            espn_words = set(espn_norm.split())
            odds_words = set(odds_norm.split())
            common_words = espn_words & odds_words
            score = len(common_words)
            
            if score > best_score:
                best_score = score
                best_match = odds_team
    
    if best_match:
        team_mapping[espn_team] = best_match
        print(f"✅ MATCH: '{espn_team}' (ESPN) → '{best_match}' (Odds API)")
    else:
        print(f"❌ NO MATCH: '{espn_team}' (ESPN)")

print()

# ============================================================================
# STEP 5: Generate normalization code
# ============================================================================

print("="*80)
print("STEP 5: Normalization mapping for code")
print("="*80)

print("\nAdd this to 10_live_betting_signal_generator.py:\n")

code = '''
# Team name normalization (ESPN → Odds API format)
TEAM_NAME_MAPPING = {
'''

for espn_name, odds_name in sorted(team_mapping.items()):
    code += f"    '{espn_name}': '{odds_name}',\n"

code += '''}

def normalize_team_for_odds_api(espn_team_name: str) -> str:
    """
    Normalize ESPN team name to match Odds API format.
    
    Args:
        espn_team_name: Team name from ESPN API
    
    Returns:
        Normalized team name for Odds API matching
    """
    return TEAM_NAME_MAPPING.get(espn_team_name, espn_team_name)


# Update match_espn_to_odds_event to use normalization:
def match_espn_to_odds_event(game: Dict, odds_lookup: Dict[Tuple[str, str], str]) -> Optional[str]:
    """
    Match ESPN game to Odds API event by team names.
    
    Args:
        game: ESPN game dict with 'home_team' and 'away_team'
        odds_lookup: Dict mapping (away_team, home_team) -> odds_event_id
    
    Returns:
        Odds API event ID or None if no match
    """
    away = game.get('away_team')
    home = game.get('home_team')
    
    if not away or not home:
        return None
    
    # Normalize team names for Odds API
    away_normalized = normalize_team_for_odds_api(away)
    home_normalized = normalize_team_for_odds_api(home)
    
    # Try exact match with normalized names
    if (away_normalized, home_normalized) in odds_lookup:
        return odds_lookup[(away_normalized, home_normalized)]
    
    # Try original names (fallback)
    if (away, home) in odds_lookup:
        return odds_lookup[(away, home)]
    
    # Try fuzzy match (in case team names differ slightly)
    for (odds_away, odds_home), event_id in odds_lookup.items():
        if (away_normalized.lower() in odds_away.lower() or odds_away.lower() in away_normalized.lower()) and \\
           (home_normalized.lower() in odds_home.lower() or odds_home.lower() in home_normalized.lower()):
            return event_id
    
    return None
'''

print(code)
print()

# ============================================================================
# STEP 6: Test the mapping
# ============================================================================

print("="*80)
print("STEP 6: Test mapping with current games")
print("="*80)

# Fetch current ESPN games again
try:
    response = requests.get(espn_url, timeout=10, verify=False)
    data = response.json()
    espn_games = data.get('events', [])
    
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
    params = {'apiKey': ODDS_API_KEY}
    response = requests.get(url, params=params, timeout=10, verify=False)
    events = response.json()
    
    # Build odds lookup with normalized names
    odds_lookup = {}
    for event in events:
        away = event.get('away_team')
        home = event.get('home_team')
        event_id = event.get('id')
        if away and home and event_id:
            odds_lookup[(away, home)] = event_id
    
    print(f"\nTesting {len(espn_games)} ESPN games:\n")
    
    for game in espn_games:
        espn_id = game.get('id')
        competitors = game.get('competitions', [{}])[0].get('competitors', [])
        
        home_team = None
        away_team = None
        for comp in competitors:
            team_name = comp.get('team', {}).get('displayName', '')
            if comp.get('homeAway') == 'home':
                home_team = team_name
            else:
                away_team = team_name
        
        if not home_team or not away_team:
            continue
        
        # Apply normalization
        away_norm = team_mapping.get(away_team, away_team)
        home_norm = team_mapping.get(home_team, home_team)
        
        # Try to match
        if (away_norm, home_norm) in odds_lookup:
            odds_id = odds_lookup[(away_norm, home_norm)]
            print(f"✅ MATCH: {away_team} @ {home_team}")
            print(f"   ESPN ID: {espn_id}")
            print(f"   Odds ID: {odds_id}")
            if away_norm != away_team or home_norm != home_team:
                print(f"   Normalized: {away_norm} @ {home_norm}")
        else:
            print(f"❌ NO MATCH: {away_team} @ {home_team}")
        print()
    
except Exception as e:
    print(f"❌ Error: {e}")

print("="*80)
print("SUMMARY")
print("="*80)
print(f"Total mismatches found: {len(team_mapping)}")
print(f"Normalization mapping generated above ⬆️")
print("="*80)

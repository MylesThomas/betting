"""
NCAA Team Utilities

Purpose:
- Map NCAA team names from The Odds API to local logo filenames
- Handle edge cases where API names differ from logo filenames
- Support both NCAAF and NCAAB team name normalization

Context:
Logos are stored at: ref/shot-quality/Logos/New Logos/
Logo files use standardized names (e.g., "Alabama.png", "Duke.png")
The Odds API often includes mascots (e.g., "Alabama Crimson Tide", "Duke Blue Devils")

Usage:
    from ncaa_team_utils import map_team_name_to_logo
    
    # Get logo filename for a team
    logo_name = map_team_name_to_logo("Indiana Hoosiers")
    # Returns: "Indiana"
    
    # Get full logo path
    logo_path = get_team_logo_path("Oregon Ducks")
    # Returns: Path to Oregon.png
"""

from pathlib import Path


# =============================================================================
# TEAM NAME MAPPINGS (The Odds API → Logo filename)
# =============================================================================

# Edge cases where The Odds API name differs from logo filename
# Format: "API Team Name": "Logo Filename (without .png)"
TEAM_NAME_EXCEPTIONS = {
    # NCAAF - Football teams with mascots
    "Indiana Hoosiers": "Indiana",
    "Miami Hurricanes": "Miami FL",
    "Ole Miss Rebels": "Mississippi",
    "Oregon Ducks": "Oregon",
    
    # NCAAB - Basketball teams with mascots (comprehensive list)
    "Akron Zips": "Akron",
    "Alabama Crimson Tide": "Alabama",
    "Arizona St Sun Devils": "Arizona St.",
    "Arizona Wildcats": "Arizona",
    "Arkansas Razorbacks": "Arkansas",
    "Auburn Tigers": "Auburn",
    "BYU Cougars": "BYU",
    "Baylor Bears": "Baylor",
    "Boise State Broncos": "Boise St.",
    "Boston College Eagles": "Boston College",
    "Butler Bulldogs": "Butler",
    "California Golden Bears": "California",
    "Cincinnati Bearcats": "Cincinnati",
    "Clemson Tigers": "Clemson",
    "Colorado Buffaloes": "Colorado",
    "Colorado St Rams": "Colorado St.",
    "Creighton Bluejays": "Creighton",
    "Dayton Flyers": "Dayton",
    "DePaul Blue Demons": "DePaul",
    "Drake Bulldogs": "Drake",
    "Duke Blue Devils": "Duke",
    "Florida Gators": "Florida",
    "Florida St Seminoles": "Florida St.",
    "George Mason Patriots": "George Mason",
    "Georgetown Hoyas": "Georgetown",
    "Georgia Bulldogs": "Georgia",
    "Georgia Tech Yellow Jackets": "Georgia Tech",
    "Gonzaga Bulldogs": "Gonzaga",
    "Grand Canyon Antelopes": "Grand Canyon",
    "High Point Panthers": "High Point",
    "Houston Cougars": "Houston",
    "Illinois Fighting Illini": "Illinois",
    "Iowa Hawkeyes": "Iowa",
    "Iowa State Cyclones": "Iowa St.",
    "Kansas Jayhawks": "Kansas",
    "Kansas St Wildcats": "Kansas St.",
    "Kentucky Wildcats": "Kentucky",
    "LSU Tigers": "LSU",
    "Liberty Flames": "Liberty",
    "Louisville Cardinals": "Louisville",
    "Marquette Golden Eagles": "Marquette",
    "Maryland Terrapins": "Maryland",
    "McNeese Cowboys": "McNeese St.",
    "Memphis Tigers": "Memphis",
    "Michigan St Spartans": "Michigan St.",
    "Michigan Wolverines": "Michigan",
    "Minnesota Golden Gophers": "Minnesota",
    "Mississippi St Bulldogs": "Mississippi St.",
    "Missouri Tigers": "Missouri",
    "NC State Wolfpack": "N.C. State",
    "Nebraska Cornhuskers": "Nebraska",
    "Nevada Wolf Pack": "Nevada",
    "New Mexico Lobos": "New Mexico",
    "North Carolina Tar Heels": "North Carolina",
    "North Texas Mean Green": "North Texas",
    "Northwestern Wildcats": "Northwestern",
    "Notre Dame Fighting Irish": "Notre Dame",
    "Ohio State Buckeyes": "Ohio St.",
    "Oklahoma Sooners": "Oklahoma",
    "Oklahoma St Cowboys": "Oklahoma St.",
    "Penn State Nittany Lions": "Penn St.",
    "Pittsburgh Panthers": "Pittsburgh",
    "Providence Friars": "Providence",
    "Purdue Boilermakers": "Purdue",
    "Rutgers Scarlet Knights": "Rutgers",
    "SMU Mustangs": "SMU",
    "Saint Louis Billikens": "Saint Louis",
    "Saint Mary's Gaels": "Saint Mary's",
    "San Diego St Aztecs": "San Diego St.",
    "San Francisco Dons": "San Francisco",
    "Seton Hall Pirates": "Seton Hall",
    "South Carolina Gamecocks": "South Carolina",
    "St. John's Red Storm": "St. John's",
    "Stanford Cardinal": "Stanford",
    "Syracuse Orange": "Syracuse",
    "TCU Horned Frogs": "TCU",
    "Tennessee Volunteers": "Tennessee",
    "Texas A&M Aggies": "Texas A&M",
    "Texas Longhorns": "Texas",
    "Texas Tech Red Raiders": "Texas Tech",
    "Troy Trojans": "Troy",
    "UAB Blazers": "UAB",
    "UC Irvine Anteaters": "UC Irvine",
    "UC San Diego Tritons": "UC San Diego",
    "UCF Knights": "UCF",
    "UCLA Bruins": "UCLA",
    "UConn Huskies": "Connecticut",
    "UNLV Rebels": "UNLV",
    "USC Trojans": "USC",
    "Utah State Aggies": "Utah St.",
    "Utah Utes": "Utah",
    "VCU Rams": "VCU",
    "Vanderbilt Commodores": "Vanderbilt",
    "Villanova Wildcats": "Villanova",
    "Virginia Cavaliers": "Virginia",
    "Virginia Tech Hokies": "Virginia Tech",
    "Wake Forest Demon Deacons": "Wake Forest",
    "Washington Huskies": "Washington",
    "West Virginia Mountaineers": "West Virginia",
    "Wisconsin Badgers": "Wisconsin",
    "Xavier Musketeers": "Xavier",
    "Yale Bulldogs": "Yale",
    
    # Additional state abbreviations that might differ
    "Miami (FL)": "Miami FL",
    "Miami (OH)": "Miami OH",
    "UNC": "North Carolina",
}


# =============================================================================
# ESPN TEAM IDs (for fetching records via ESPN API)
# =============================================================================

"""
ESPN NCAAB Team IDs

ESPN's API provides team records for college basketball. To fetch records, you need the ESPN team ID.

HOW TO FIND ESPN TEAM IDs:
1. Method 1 - Team Page URL:
   - Go to team's ESPN page (e.g., espn.com/mens-college-basketball/team/_/id/150/duke-blue-devils)
   - The team ID is in the URL: /id/150/ means team ID is 150

2. Method 2 - API Testing:
   curl "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/2026/types/2/teams/150/record"

HOW TO ADD MORE TEAMS:
1. Find the ESPN team ID using methods above
2. Add to ESPN_NCAAB_TEAM_IDS dictionary below
3. Use The Odds API team name format (with mascot) as the key
   Example: 'Duke Blue Devils': 150

IMPORTANT NOTES:
- Team IDs are specific to college basketball (football uses different IDs)
- ESPN uses season 2026 for 2025-26 academic year
- Records include both conference + non-conference games
- Some mid-major teams may not have IDs or complete data

TESTING A TEAM ID:
    import requests
    team_id = 150  # Duke
    season = 2026
    url = f'https://sports.core.api.espn.com/.../teams/{team_id}/record'
    response = requests.get(url, timeout=5, verify=False)
    data = response.json()
    for item in data.get('items', []):
        if item.get('type') == 'total' and item.get('name') == 'overall':
            print(f"Record: {item.get('summary')}")
"""

# ESPN team IDs for major college basketball programs
# Format: The Odds API team name (with mascot) -> ESPN team ID
ESPN_NCAAB_TEAM_IDS = {
    # Power 5 + Top Programs
    'Duke Blue Devils': 150,
    'North Carolina Tar Heels': 153,
    'Kansas Jayhawks': 2305,
    'Kentucky Wildcats': 96,
    'Gonzaga Bulldogs': 2250,
    'Villanova Wildcats': 222,
    'Michigan St Spartans': 127,
    'UCLA Bruins': 26,
    'Arizona Wildcats': 12,
    'Purdue Boilermakers': 2509,
    'Houston Cougars': 248,
    'Connecticut Huskies': 41,
    'Baylor Bears': 239,
    'Illinois Fighting Illini': 356,
    'Wisconsin Badgers': 275,
    'Texas Longhorns': 251,
    'Tennessee Volunteers': 2633,
    'Auburn Tigers': 2,
    'Alabama Crimson Tide': 333,
    'Arkansas Razorbacks': 8,
    'Iowa State Cyclones': 66,
    'Florida Gators': 57,
    'Michigan Wolverines': 130,
    'Ohio State Buckeyes': 194,
    'Indiana Hoosiers': 84,
    'Maryland Terrapins': 120,
    'Virginia Cavaliers': 258,
    'Xavier Musketeers': 2752,
    'Creighton Bluejays': 156,
    'Marquette Golden Eagles': 269,
    
    # Additional ACC
    'Virginia Tech Hokies': 259,
    'Clemson Tigers': 228,
    'Florida St Seminoles': 52,
    'Syracuse Orange': 183,
    'Pittsburgh Panthers': 221,
    'NC State Wolfpack': 152,
    'Boston College Eagles': 103,
    'Wake Forest Demon Deacons': 154,
    'Georgia Tech Yellow Jackets': 59,
    'Miami Hurricanes': 2390,
    
    # Additional Big Ten
    'Penn State Nittany Lions': 213,
    'Northwestern Wildcats': 77,
    'Rutgers Scarlet Knights': 164,
    'Minnesota Golden Gophers': 135,
    'Iowa Hawkeyes': 2294,
    'Nebraska Cornhuskers': 158,
    
    # Additional Big 12
    'Texas Tech Red Raiders': 2641,
    'Oklahoma Sooners': 201,
    'Oklahoma St Cowboys': 197,
    'TCU Horned Frogs': 2628,
    'Kansas St Wildcats': 2306,
    'West Virginia Mountaineers': 277,
    
    # Additional SEC
    'Mississippi St Bulldogs': 344,
    'Ole Miss Rebels': 145,
    'LSU Tigers': 99,
    'Missouri Tigers': 142,
    'South Carolina Gamecocks': 2579,
    'Vanderbilt Commodores': 238,
    'Georgia Bulldogs': 61,
    
    # Additional Big East
    'Providence Friars': 2507,
    'Seton Hall Pirates': 2550,
    "St. John's Red Storm": 2599,
    'Butler Bulldogs': 2086,
    'DePaul Blue Demons': 305,
    
    # WCC & Mountain West
    "Saint Mary's Gaels": 2608,
    'San Francisco Dons': 2599,
    'San Diego St Aztecs': 21,
    'Boise State Broncos': 68,
    'Nevada Wolf Pack': 2440,
    'Colorado St Rams': 36,
    'UNLV Rebels': 2439,
    'New Mexico Lobos': 167,
    
    # Other Major Programs
    'Memphis Tigers': 235,
    'SMU Mustangs': 2567,
    'Dayton Flyers': 2168,
    'VCU Rams': 2670,
    'Colorado Buffaloes': 38,
    'California Golden Bears': 25,
    'Stanford Cardinal': 24,
    'USC Trojans': 30,
    'UCF Knights': 2116,
    'Utah Utes': 254,
    'Arizona St Sun Devils': 9,
    'Louisville Cardinals': 97,
    'Grand Canyon Antelopes': 2253,
    'UAB Blazers': 5,
    'Akron Zips': 2006,
    'Drake Bulldogs': 2181,
    'Troy Trojans': 2653,
    'Yale Bulldogs': 43,
    'Utah State Aggies': 328,
    'BYU Cougars': 252,
    'Notre Dame Fighting Irish': 87,
    
    # UConn aliases
    'UConn Huskies': 41,
}


# ESPN team IDs for college football programs
# Format: The Odds API team name (with mascot) -> ESPN team ID
# NOTE: Starting with 2025-26 CFP playoff teams only. Expand as needed.
# TODO 2027: Update for next season's playoff teams
ESPN_NCAAF_TEAM_IDS = {
    # 2025-26 College Football Playoff Teams (Current top 4 from The Odds API)
    'Indiana Hoosiers': 84,
    'Miami Hurricanes': 2390,
    'Oregon Ducks': 2483,
    'Ole Miss Rebels': 145,
}


# =============================================================================
# TEAM NAME MAPPING FUNCTIONS
# =============================================================================

def map_team_name_to_logo(team_name: str) -> str:
    """
    Map a team name from The Odds API to the corresponding logo filename.
    
    Uses exception mapping for known edge cases, otherwise returns the team name as-is
    (assumes 1-to-1 mapping).
    
    Args:
        team_name: Team name from The Odds API (e.g., "Indiana Hoosiers")
        
    Returns:
        str: Logo filename without .png extension (e.g., "Indiana")
        
    Examples:
        >>> map_team_name_to_logo("Indiana Hoosiers")
        'Indiana'
        >>> map_team_name_to_logo("Alabama")
        'Alabama'
        >>> map_team_name_to_logo("Miami Hurricanes")
        'Miami FL'
    """
    # Check if team is in exceptions dictionary
    if team_name in TEAM_NAME_EXCEPTIONS:
        return TEAM_NAME_EXCEPTIONS[team_name]
    
    # Default: assume 1-to-1 mapping
    return team_name


def get_team_logo_path(team_name: str, repo_root: Path = None) -> Path:
    """
    Get the full path to a team's logo file.
    
    Args:
        team_name: Team name from The Odds API
        repo_root: Path to repository root (defaults to auto-detection)
        
    Returns:
        Path: Full path to logo PNG file, or None if not found
        
    Examples:
        >>> path = get_team_logo_path("Indiana Hoosiers")
        >>> print(path)
        /Users/.../betting/ref/shot-quality/Logos/New Logos/Indiana.png
    """
    # Auto-detect repo root if not provided
    if repo_root is None:
        repo_root = Path(__file__).parent.parent
    
    # Map team name to logo filename
    logo_name = map_team_name_to_logo(team_name)
    
    # Build path to logo file
    logo_path = repo_root / 'ref/shot-quality/Logos/New Logos' / f'{logo_name}.png'
    
    # Return path (will return even if file doesn't exist - caller can check)
    return logo_path


def get_all_available_logos(repo_root: Path = None) -> dict:
    """
    Get all available team logos as a mapping dictionary.
    
    Scans the logos directory and creates a mapping of logo names to file paths.
    
    Args:
        repo_root: Path to repository root (defaults to auto-detection)
        
    Returns:
        dict: Mapping of logo names (without .png) to absolute file paths
        
    Examples:
        >>> logos = get_all_available_logos()
        >>> print(logos['Alabama'])
        /Users/.../betting/ref/shot-quality/Logos/New Logos/Alabama.png
    """
    # Auto-detect repo root if not provided
    if repo_root is None:
        repo_root = Path(__file__).parent.parent
    
    logos_dir = repo_root / 'ref/shot-quality/Logos/New Logos'
    
    # Get all PNG files
    logo_files = list(logos_dir.glob('*.png'))
    
    # Create mapping: logo name (without .png) → absolute path
    logo_map = {}
    for logo_file in logo_files:
        logo_name = logo_file.stem  # Filename without extension
        logo_map[logo_name] = str(logo_file.absolute())
    
    return logo_map


def map_teams_to_logos(team_names: list, repo_root: Path = None) -> dict:
    """
    Map a list of team names to their logo file paths.
    
    This is the main function to use in visualization scripts.
    
    Args:
        team_names: List of team names from The Odds API
        repo_root: Path to repository root (defaults to auto-detection)
        
    Returns:
        dict: Mapping of original team names to logo file paths (or None if not found)
        
    Examples:
        >>> teams = ["Indiana Hoosiers", "Alabama", "Miami Hurricanes"]
        >>> logo_map = map_teams_to_logos(teams)
        >>> print(logo_map["Indiana Hoosiers"])
        /Users/.../betting/ref/shot-quality/Logos/New Logos/Indiana.png
    """
    # Auto-detect repo root if not provided
    if repo_root is None:
        repo_root = Path(__file__).parent.parent
    
    # Get all available logos
    available_logos = get_all_available_logos(repo_root)
    
    # Map each team name
    team_logo_map = {}
    for team_name in team_names:
        # Map API name to logo name
        logo_name = map_team_name_to_logo(team_name)
        
        # Check if logo exists
        if logo_name in available_logos:
            team_logo_map[team_name] = available_logos[logo_name]
        else:
            # Logo not found - set to None
            team_logo_map[team_name] = None
    
    return team_logo_map


def get_logo_coverage_stats(team_names: list, repo_root: Path = None) -> dict:
    """
    Get statistics on logo coverage for a list of teams.
    
    Useful for debugging and reporting how many teams have logos.
    
    Args:
        team_names: List of team names from The Odds API
        repo_root: Path to repository root (defaults to auto-detection)
        
    Returns:
        dict: Statistics including total, matched, unmatched teams
        
    Examples:
        >>> teams = ["Indiana Hoosiers", "Alabama", "Unknown Team"]
        >>> stats = get_logo_coverage_stats(teams)
        >>> print(f"Matched: {stats['matched']}/{stats['total']}")
        Matched: 2/3
    """
    # Map teams to logos
    team_logo_map = map_teams_to_logos(team_names, repo_root)
    
    # Calculate stats
    total = len(team_names)
    matched = sum(1 for path in team_logo_map.values() if path is not None)
    unmatched = total - matched
    
    # Get list of unmatched teams
    unmatched_teams = [team for team, path in team_logo_map.items() if path is None]
    
    return {
        'total': total,
        'matched': matched,
        'unmatched': unmatched,
        'coverage_pct': (matched / total * 100) if total > 0 else 0.0,
        'unmatched_teams': unmatched_teams
    }


# =============================================================================
# TESTING / DEBUG FUNCTIONS
# =============================================================================

def test_team_mapping():
    """Test the team name mapping with known examples"""
    test_cases = [
        ("Indiana Hoosiers", "Indiana"),
        ("Alabama", "Alabama"),
        ("Miami Hurricanes", "Miami FL"),
        ("Ole Miss Rebels", "Mississippi"),
        ("Oregon Ducks", "Oregon"),
    ]
    
    print("Testing team name mapping:")
    print("-" * 60)
    
    for api_name, expected_logo in test_cases:
        result = map_team_name_to_logo(api_name)
        status = "✅" if result == expected_logo else "❌"
        print(f"{status} '{api_name}' → '{result}' (expected: '{expected_logo}')")
    
    print("-" * 60)


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_team_mapping()
    
    print("\nGetting all available logos...")
    logos = get_all_available_logos()
    print(f"Found {len(logos)} logo files")
    
    print("\nSample logos:")
    for i, (name, path) in enumerate(list(logos.items())[:10]):
        print(f"  • {name}")
    print(f"  ... and {len(logos) - 10} more")


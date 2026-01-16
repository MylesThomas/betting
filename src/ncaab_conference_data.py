"""
NCAAB Conference Mappings

Conference affiliations for NCAA Division I Men's Basketball teams.

IMPORTANT: These mappings are specific to the 2025-26 season.
           Conferences change due to realignment - update annually!

Last Updated: January 2026
Source: https://en.wikipedia.org/wiki/List_of_NCAA_Division_I_men%27s_basketball_programs

Major Changes for 2025-26:
- UCLA, USC join Big Ten
- Stanford, California join ACC  
- SMU joins ACC
- BYU, Colorado, Arizona, Arizona State join Big 12
- Texas, Oklahoma join SEC
- Oregon, Washington join Big Ten

TODO 2027: Update conference mappings for 2026-27 season realignment.
"""

# Conference mappings: ESPN team name → Conference
# 
# These mappings are for teams where ESPN uses different naming conventions
# than the official university names (abbreviations, nicknames, etc.)
NCAAB_CONFERENCE_MAPPING_2025_26 = {
    # Big Ten (18 teams) - Added UCLA, USC, Washington in 2024
    'Wisconsin Badgers': 'Big Ten',
    'Illinois Fighting Illini': 'Big Ten',
    'Indiana Hoosiers': 'Big Ten',
    'Maryland Terrapins': 'Big Ten',
    'Penn State Nittany Lions': 'Big Ten',
    'Rutgers Scarlet Knights': 'Big Ten',
    'Nebraska Cornhuskers': 'Big Ten',
    'Washington Huskies': 'Big Ten',
    'UCLA Bruins': 'Big Ten',
    'USC Trojans': 'Big Ten',
    
    # ACC (18 teams) - Added Stanford, California, SMU in 2024
    'California Golden Bears': 'ACC',
    'Georgia Tech Yellow Jackets': 'ACC',
    'Miami Hurricanes': 'ACC',
    'Virginia Tech Hokies': 'ACC',
    'SMU Mustangs': 'ACC',
    'North Carolina Tar Heels': 'ACC',
    'NC State Wolfpack': 'ACC',
    
    # Big 12 (16 teams) - Added Colorado, BYU, Cincinnati, UCF, Houston in recent years
    'BYU Cougars': 'Big 12',
    'Colorado Buffaloes': 'Big 12',
    'Texas Longhorns': 'Big 12',
    'TCU Horned Frogs': 'Big 12',
    'Oklahoma State Cowboys': 'Big 12',
    'UCF Knights': 'Big 12',
    
    # SEC (16 teams) - Added Texas, Oklahoma in 2024
    'LSU Tigers': 'SEC',
    'Ole Miss Rebels': 'SEC',
    
    # Big East (11 teams)
    'UConn Huskies': 'Big East',
    
    # Mountain West (12 teams)
    'Fresno State Bulldogs': 'Mountain West',
    'San Diego State Aztecs': 'Mountain West',
    'Nevada Wolf Pack': 'Mountain West',
    'New Mexico Lobos': 'Mountain West',
    'Air Force Falcons': 'Mountain West',
    'UNLV Rebels': 'Mountain West',
    'UTEP Miners': 'Mountain West',
    
    # West Coast Conference (11 teams)
    'Saint Mary\'s Gaels': 'West Coast',
    'UC San Diego Tritons': 'West Coast',
    
    # UC System (Big West)
    'UC Davis Aggies': 'Big West',
    'UC Irvine Anteaters': 'Big West',
    'UC Riverside Highlanders': 'Big West',
    'UC Santa Barbara Gauchos': 'Big West',
    
    # Cal State System (Big West)
    'Cal Poly Mustangs': 'Big West',
    'Cal State Bakersfield Roadrunners': 'Big West',
    'Cal State Fullerton Titans': 'Big West',
    'Cal State Northridge Matadors': 'Big West',
    'Long Beach State Beach': 'Big West',
    'Sacramento State Hornets': 'Big West',
    'Hawai\'i Rainbow Warriors': 'Big West',
    
    # UNC System
    'UNC Asheville Bulldogs': 'Big South',
    'UNC Greensboro Spartans': 'Southern',
    'UNC Wilmington Seahawks': 'CAA',
    'Charlotte 49ers': 'American',
    
    # UMass System
    'Massachusetts Minutemen': 'A-10',
    'UMass Lowell River Hawks': 'America East',
    
    # UT System
    'UTSA Roadrunners': 'American',
    'UT Arlington Mavericks': 'WAC',
    'UT Martin Skyhawks': 'Ohio Valley',
    
    # UL System
    'Louisiana Ragin\' Cajuns': 'Sun Belt',
    'UL Monroe Warhawks': 'Sun Belt',
    'Louisiana Tech Bulldogs': 'Conference USA',
    
    # Military Academies
    'Army Black Knights': 'Patriot League',
    'Navy Midshipmen': 'Patriot League',
    
    # Other Abbreviations/Short Names
    'VCU Rams': 'A-10',
    'UAB Blazers': 'American',
    'UIC Flames': 'Missouri Valley',
    'UMBC Retrievers': 'America East',
    'UAlbany Great Danes': 'America East',
    'VMI Keydets': 'Southern',
    
    # Name Variations
    'Saint Joseph\'s Hawks': 'A-10',
    'Saint Francis Red Flash': 'NEC',
    'Loyola Chicago Ramblers': 'A-10',
    'Loyola Maryland Greyhounds': 'Patriot League',
    'Chattanooga Mocs': 'Southern',
    'Green Bay Phoenix': 'Horizon League',
    'Milwaukee Panthers': 'Horizon League',
    'Omaha Mavericks': 'The Summit',
    'Kansas City Roos': 'The Summit',
    'IU Indianapolis Jaguars': 'Horizon League',
    
    # HBCUs
    'Alabama A&M Bulldogs': 'SWAC',
    'Florida A&M Rattlers': 'MEAC',
    'Arkansas-Pine Bluff Golden Lions': 'SWAC',
    'North Carolina A&T Aggies': 'CAA',
    
    # Other State Schools
    'Middle Tennessee Blue Raiders': 'Conference USA',
    'Austin Peay Governors': 'ASUN',
    'Southeast Missouri State Redhawks': 'Ohio Valley',
    'Southern Illinois Salukis': 'Missouri Valley',
    'SIU Edwardsville Cougars': 'Ohio Valley',
    'Bowling Green Falcons': 'MAC',
    'Central Connecticut Blue Devils': 'NEC',
    'McNeese Cowboys': 'Southland',
    'Nicholls Colonels': 'Southland',
    'Stephen F. Austin Lumberjacks': 'WAC',
    'Sam Houston Bearkats': 'Conference USA',
    'Tarleton State Texans': 'WAC',
    'Southern Miss Golden Eagles': 'Sun Belt',
    'Little Rock Trojans': 'Ohio Valley',
    'NJIT Highlanders': 'America East',
    'Purdue Fort Wayne Mastodons': 'Horizon League',
    'Queens University Royals': 'ASUN',
}


def get_team_conference(team_name: str, season: str = '2025-26') -> str:
    """
    Get the conference for a given team.
    
    Args:
        team_name: Team name in ESPN format (e.g., "Wisconsin Badgers")
        season: Season string (e.g., "2025-26"). Currently only 2025-26 is supported.
    
    Returns:
        Conference name (e.g., "Big Ten")
        
    Raises:
        ValueError: If team is not found in mapping
        NotImplementedError: If season is not 2025-26
    """
    if season != '2025-26':
        raise NotImplementedError(
            f"Conference mappings are only available for 2025-26 season. "
            f"Requested season: {season}. Please update NCAAB_CONFERENCE_MAPPING_{season.replace('-', '_')}."
        )
    
    if team_name not in NCAAB_CONFERENCE_MAPPING_2025_26:
        raise ValueError(
            f"Team '{team_name}' not found in conference mapping. "
            f"This may be due to an unmatched team name or missing manual mapping."
        )
    
    return NCAAB_CONFERENCE_MAPPING_2025_26[team_name]


def is_conference_game(team1: str, team2: str, season: str = '2025-26') -> bool:
    """
    Determine if a game is a conference game (both teams in same conference).
    
    Args:
        team1: First team name in ESPN format
        team2: Second team name in ESPN format
        season: Season string (e.g., "2025-26")
    
    Returns:
        True if both teams are in the same conference, False otherwise
    """
    try:
        conf1 = get_team_conference(team1, season)
        conf2 = get_team_conference(team2, season)
        return conf1 == conf2
    except (ValueError, NotImplementedError):
        # If either team is not found or season not supported, return False
        return False


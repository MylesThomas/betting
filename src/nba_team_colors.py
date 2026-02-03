"""
NBA team colors for visualization.

Primary and secondary colors for each NBA team, following official team branding.
Colors are in hex format for use with matplotlib.

Usage:
    from src.nba_team_colors import get_team_color
    
    color = get_team_color('Los Angeles Lakers', 'primary')
    # Returns: '#552583' (Lakers purple)

Author: Myles Thomas
Date: 2026-02-02
"""

# NBA Team Colors (Primary, Secondary)
# Source: Official team branding
NBA_TEAM_COLORS = {
    'Atlanta Hawks': {
        'primary': '#E03A3E',
        'secondary': '#C1D32F'
    },
    'Boston Celtics': {
        'primary': '#007A33',
        'secondary': '#BA9653'
    },
    'Brooklyn Nets': {
        'primary': '#000000',
        'secondary': '#FFFFFF'
    },
    'Charlotte Hornets': {
        'primary': '#1D1160',
        'secondary': '#00788C'
    },
    'Chicago Bulls': {
        'primary': '#CE1141',
        'secondary': '#000000'
    },
    'Cleveland Cavaliers': {
        'primary': '#860038',
        'secondary': '#FDBB30'
    },
    'Dallas Mavericks': {
        'primary': '#00538C',
        'secondary': '#002B5E'
    },
    'Denver Nuggets': {
        'primary': '#0E2240',
        'secondary': '#FEC524'
    },
    'Detroit Pistons': {
        'primary': '#C8102E',
        'secondary': '#1D42BA'
    },
    'Golden State Warriors': {
        'primary': '#1D428A',
        'secondary': '#FFC72C'
    },
    'Houston Rockets': {
        'primary': '#CE1141',
        'secondary': '#000000'
    },
    'Indiana Pacers': {
        'primary': '#002D62',
        'secondary': '#FDBB30'
    },
    'Los Angeles Clippers': {
        'primary': '#C8102E',
        'secondary': '#1D428A'
    },
    'Los Angeles Lakers': {
        'primary': '#552583',
        'secondary': '#FDB927'
    },
    'Memphis Grizzlies': {
        'primary': '#5D76A9',
        'secondary': '#12173F'
    },
    'Miami Heat': {
        'primary': '#98002E',
        'secondary': '#F9A01B'
    },
    'Milwaukee Bucks': {
        'primary': '#00471B',
        'secondary': '#EEE1C6'
    },
    'Minnesota Timberwolves': {
        'primary': '#0C2340',
        'secondary': '#236192'
    },
    'New Orleans Pelicans': {
        'primary': '#0C2340',
        'secondary': '#C8102E'
    },
    'New York Knicks': {
        'primary': '#006BB6',
        'secondary': '#F58426'
    },
    'Oklahoma City Thunder': {
        'primary': '#007AC1',
        'secondary': '#EF3B24'
    },
    'Orlando Magic': {
        'primary': '#0077C0',
        'secondary': '#C4CED4'
    },
    'Philadelphia 76ers': {
        'primary': '#006BB6',
        'secondary': '#ED174C'
    },
    'Phoenix Suns': {
        'primary': '#1D1160',
        'secondary': '#E56020'
    },
    'Portland Trail Blazers': {
        'primary': '#E03A3E',
        'secondary': '#000000'
    },
    'Sacramento Kings': {
        'primary': '#5A2D81',
        'secondary': '#63727A'
    },
    'San Antonio Spurs': {
        'primary': '#C4CED4',
        'secondary': '#000000'
    },
    'Toronto Raptors': {
        'primary': '#CE1141',
        'secondary': '#000000'
    },
    'Utah Jazz': {
        'primary': '#002B5C',
        'secondary': '#00471B'
    },
    'Washington Wizards': {
        'primary': '#002B5C',
        'secondary': '#E31837'
    }
}


def get_team_color(team_name: str, color_type: str = 'primary') -> str:
    """
    Get the color for an NBA team.
    
    Args:
        team_name: Full team name (e.g., 'Los Angeles Lakers')
        color_type: 'primary' or 'secondary'
        
    Returns:
        Hex color code
        
    Examples:
        >>> get_team_color('Los Angeles Lakers', 'primary')
        '#552583'
        >>> get_team_color('Boston Celtics', 'secondary')
        '#BA9653'
    """
    # Default colors if team not found
    default_colors = {'primary': '#1f77b4', 'secondary': '#ff7f0e'}
    
    if team_name in NBA_TEAM_COLORS:
        return NBA_TEAM_COLORS[team_name][color_type]
    
    return default_colors[color_type]


def get_team_colors_dict(team_name: str) -> dict:
    """
    Get both primary and secondary colors for a team.
    
    Args:
        team_name: Full team name
        
    Returns:
        Dictionary with 'primary' and 'secondary' keys
    """
    default_colors = {'primary': '#1f77b4', 'secondary': '#ff7f0e'}
    return NBA_TEAM_COLORS.get(team_name, default_colors)

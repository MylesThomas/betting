"""
Utility functions for working with American odds and spread bets.

This module provides functions for:
- Converting odds to implied probabilities
- Calculating bet amounts and profits
- Working with American odds format
- Spread cover: single source of truth for "did this side cover?" (see did_cover_spread)
"""


def did_cover_spread(home_score: float, away_score: float, home_spread: float, bet_home: bool):
    """
    Whether the bet side covered the spread. SINGLE SOURCE OF TRUTH for spread-cover logic.

    Convention: home_spread is the HOME team's spread (negative when home is favored).
    E.g. home_spread = -10.5 means home is favored by 10.5; home covers only when
    (home_score - away_score) > 10.5.

    Formula:
      - Bet home: cover when (home_score - away_score) + home_spread > 0  => margin > -home_spread
      - Bet away: cover when (away_score - home_score) - home_spread > 0  => away_margin > home_spread
      (away's line is -home_spread; away covers when away_margin > home_spread, e.g. -7 > -10.5)

    Args:
        home_score: Home team final score.
        away_score: Away team final score.
        home_spread: Home team's spread (e.g. -10.5 = home favored by 10.5). Use None/NaN if no line.
        bet_home: True if evaluating whether home covered, False for away.

    Returns:
        True if that side covered, False if they did not, None if home_spread is None or NaN.

    Examples:
        >>> did_cover_spread(75, 68, -10.5, True)   # Arizona 75-68, bet home -10.5 → margin 7, need >10.5
        False
        >>> did_cover_spread(76, 68, -10.5, True)   # margin 8, still no cover
        False
        >>> did_cover_spread(79, 68, -10.5, True)   # margin 11, cover
        True
        >>> did_cover_spread(68, 75, -10.5, False)  # bet away +10.5, away lost by 7 → away_margin -7, -7+10.5=3.5>0
        True
    """
    import math
    if home_spread is None or (isinstance(home_spread, float) and math.isnan(home_spread)):
        return None
    home_margin = home_score - away_score
    if bet_home:
        diff = home_margin + home_spread
    else:
        diff = -home_margin - home_spread
    if diff == 0:
        return None  # push
    return diff > 0


def calculate_bet_amount(odds, target_win=100):
    """
    Calculate bet amount to win $100 (or return $100 for + odds).
    
    Args:
        odds: American odds (e.g., -110, +130)
        target_win: Target profit amount (default $100)
    
    Returns:
        Bet amount needed
    
    Examples:
        >>> calculate_bet_amount(-110, 100)  # Bet $110 to win $100
        110.0
        >>> calculate_bet_amount(+150, 100)  # Bet $66.67 to win $100
        66.67
    """
    if odds < 0:
        # Negative odds: bet more to win target_win
        # Formula: bet_amount = (abs(odds) / 100) * target_win
        return abs(odds) / 100 * target_win
    else:
        # Positive odds: bet less to win target_win  
        # Formula: bet_amount = (100 / odds) * target_win
        return 100 / odds * target_win


def calculate_profit(odds, bet_amount):
    """
    Calculate profit from a winning bet.
    
    Args:
        odds: American odds
        bet_amount: Amount wagered
    
    Returns:
        Profit (not including original stake)
    
    Examples:
        >>> calculate_profit(-110, 110)  # Win $100 on a $110 bet at -110
        100.0
        >>> calculate_profit(+150, 100)  # Win $150 on a $100 bet at +150
        150.0
    """
    if odds < 0:
        return bet_amount * (100 / abs(odds))
    else:
        return bet_amount * (odds / 100)


def odds_to_implied_probability(odds):
    """
    Convert American odds to implied probability.
    
    Args:
        odds: American odds (e.g., -110, +130)
    
    Returns:
        Implied probability as a decimal (e.g., 0.5455 for 54.55%)
    
    Examples:
        >>> odds_to_implied_probability(-110)  # 52.38% implied probability
        0.5238095238095238
        >>> odds_to_implied_probability(+150)  # 40% implied probability
        0.4
    """
    if odds < 0:
        # Negative odds: |odds| / (|odds| + 100)
        return abs(odds) / (abs(odds) + 100)
    else:
        # Positive odds: 100 / (odds + 100)
        return 100 / (odds + 100)


def american_odds_to_percentage_probability(odds):
    """
    Convert American odds to implied probability percentage.
    
    Args:
        odds: American odds (e.g., -110, +130)
    
    Returns:
        Implied probability as percentage (e.g., 54.55 for 54.55%)
    
    Examples:
        >>> american_odds_to_probability(-110)  # 52.38%
        52.38095238095238
        >>> american_odds_to_probability(+150)  # 40%
        40.0
    """
    if odds < 0:
        return abs(odds) / (abs(odds) + 100) * 100
    else:
        return 100 / (odds + 100) * 100


def probability_to_american_odds(prob_pct):
    """
    Convert implied probability percentage to American odds.
    
    Args:
        prob_pct: Probability as percentage (e.g., 54.55 for 54.55%)
    
    Returns:
        American odds (negative for favorites, positive for underdogs)
    
    Examples:
        >>> probability_to_american_odds(52.38)  # ~-110 odds
        -110.0
        >>> probability_to_american_odds(40.0)  # +150 odds
        150.0
        >>> probability_to_american_odds(50.0)  # Even odds
        -100.0
    """
    if prob_pct >= 50:
        # Negative odds (favorite)
        return -prob_pct / (100 - prob_pct) * 100
    else:
        # Positive odds (underdog)
        return (100 - prob_pct) / prob_pct * 100


def implied_probability_to_odds(prob):
    """
    Convert implied probability (as decimal) to American odds.
    
    Args:
        prob: Probability as decimal (e.g., 0.5455 for 54.55%)
    
    Returns:
        American odds (negative for favorites, positive for underdogs)
    
    Examples:
        >>> implied_probability_to_odds(0.5238)  # ~-110 odds
        -110.0
        >>> implied_probability_to_odds(0.40)  # +150 odds
        150.0
        >>> implied_probability_to_odds(0.50)  # Even odds
        -100.0
    """
    # Convert decimal to percentage and use existing function
    prob_pct = prob * 100
    return probability_to_american_odds(prob_pct)


def calculate_vig(implied_over, implied_under):
    """
    Calculate vig (overround/juice) from implied probabilities.
    
    Vig = implied_over + implied_under - 1
    
    Args:
        implied_over: Implied probability for over (as decimal, e.g., 0.5238)
        implied_under: Implied probability for under (as decimal, e.g., 0.5238)
    
    Returns:
        Vig as decimal (e.g., 0.0476 for 4.76% vig)
    
    Examples:
        >>> calculate_vig(0.5238, 0.5238)  # -110/-110 → 4.76% vig
        0.0476
        >>> calculate_vig(0.5, 0.5)  # Fair odds → 0% vig
        0.0
    """
    import numpy as np
    import pandas as pd
    
    if pd.isna(implied_over) or pd.isna(implied_under):
        return np.nan
    return implied_over + implied_under - 1


def calculate_vig_attribution(implied_over, implied_under):
    """
    Calculate total vig and attribute it to each side (over/under).
    
    Uses proportional attribution: each side's vig is the difference between
    its implied probability and its "fair" probability (normalized to sum to 1).
    
    Fair probabilities:
        fair_over = implied_over / (implied_over + implied_under)
        fair_under = implied_under / (implied_over + implied_under)
    
    Vig attribution:
        over_vig = implied_over - fair_over
        under_vig = implied_under - fair_under
    
    Args:
        implied_over: Implied probability for over (as decimal, e.g., 0.5238)
        implied_under: Implied probability for under (as decimal, e.g., 0.5238)
    
    Returns:
        dict with keys:
            - 'total_vig': Total vig (over_vig + under_vig)
            - 'over_vig': Vig attributed to the over side
            - 'under_vig': Vig attributed to the under side
            - 'fair_over': Fair probability for over (no vig)
            - 'fair_under': Fair probability for under (no vig)
    
    Examples:
        >>> calculate_vig_attribution(0.5238, 0.5238)  # -110/-110 (symmetric)
        {'total_vig': 0.0476, 'over_vig': 0.0238, 'under_vig': 0.0238, 
         'fair_over': 0.5, 'fair_under': 0.5}
        
        >>> calculate_vig_attribution(0.60, 0.4348)  # -150/+130 (asymmetric)
        {'total_vig': 0.0348, 'over_vig': 0.0201, 'under_vig': 0.0147,
         'fair_over': 0.5799, 'fair_under': 0.4201}
    """
    import numpy as np
    import pandas as pd
    
    if pd.isna(implied_over) or pd.isna(implied_under):
        return {
            'total_vig': np.nan,
            'over_vig': np.nan,
            'under_vig': np.nan,
            'fair_over': np.nan,
            'fair_under': np.nan
        }
    
    # Sum of implied probabilities (will be > 1 due to vig)
    implied_sum = implied_over + implied_under
    
    # Fair probabilities: normalize to sum to 1
    fair_over = implied_over / implied_sum
    fair_under = implied_under / implied_sum
    
    # Sanity check: fair probabilities must sum to 1
    assert abs((fair_over + fair_under) - 1) < 0.001, \
        f"Fair probabilities should sum to 1, got {fair_over + fair_under}"
    
    # Vig on each side: how much "extra" probability is baked in
    over_vig = implied_over - fair_over
    under_vig = implied_under - fair_under
    
    # Total vig (should equal implied_sum - 1)
    total_vig = over_vig + under_vig
    
    return {
        'total_vig': total_vig,
        'over_vig': over_vig,
        'under_vig': under_vig,
        'fair_over': fair_over,
        'fair_under': fair_under
    }

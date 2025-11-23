"""
Utility functions for working with American odds.

This module provides functions for:
- Converting odds to implied probabilities
- Calculating bet amounts and profits
- Working with American odds format
"""


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


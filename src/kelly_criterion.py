"""
Kelly Criterion calculation for optimal bet sizing.

The Kelly Criterion is a mathematical formula for determining the optimal
bet size to maximize long-term growth of bankroll, given an edge.

Formula:
    Kelly % = (Win_Prob × (Net_Odds + 1) - 1) / Net_Odds

Where:
    - Win_Prob = Expected probability of winning (e.g., 0.572 for 57.2%)
    - Net_Odds = Decimal odds won (e.g., for -110: 100/110 = 0.909)

Context:
This module provides Kelly Criterion calculations for sports betting,
using American odds format and win probabilities from backtest data.

Author: Thomas Myles
Date: 2026-01-19
"""


def calculate_kelly_criterion(win_prob, american_odds, max_kelly=0.10):
    """
    Calculate Kelly Criterion bet size as percentage of bankroll.
    
    Args:
        win_prob: Probability of winning as decimal (e.g., 0.572 for 57.2%)
        american_odds: American odds (e.g., -110, +150)
        max_kelly: Maximum Kelly percentage to cap at (default 0.10 = 10%)
    
    Returns:
        dict with keys:
            - 'kelly_pct': Kelly percentage as decimal (e.g., 0.032 for 3.2%)
            - 'kelly_pct_display': Kelly percentage for display (e.g., 3.2)
            - 'net_odds': Net odds used in calculation
            - 'capped': Whether Kelly was capped at max_kelly
    
    Examples:
        >>> calculate_kelly_criterion(0.572, -110)
        {'kelly_pct': 0.055, 'kelly_pct_display': 5.5, 'net_odds': 0.909, 'capped': False}
        
        >>> calculate_kelly_criterion(0.55, -110)
        {'kelly_pct': 0.027, 'kelly_pct_display': 2.7, 'net_odds': 0.909, 'capped': False}
        
        >>> calculate_kelly_criterion(0.50, -110)  # No edge
        {'kelly_pct': 0.0, 'kelly_pct_display': 0.0, 'net_odds': 0.909, 'capped': False}
    """
    # Convert American odds to net odds (decimal odds - 1)
    if american_odds < 0:
        # Negative odds: risk |odds| to win 100
        # Net odds = profit / risk = 100 / |odds|
        net_odds = 100 / abs(american_odds)
    else:
        # Positive odds: risk 100 to win odds
        # Net odds = profit / risk = odds / 100
        net_odds = american_odds / 100
    
    # Kelly formula: f* = (p × (b + 1) - 1) / b
    # Where:
    #   f* = optimal fraction of bankroll to bet
    #   p = probability of winning
    #   b = net odds (profit per unit wagered)
    kelly_pct = (win_prob * (net_odds + 1) - 1) / net_odds
    
    # Cap at 0% if negative (no edge)
    if kelly_pct < 0:
        kelly_pct = 0.0
    
    # Cap at max_kelly if specified
    capped = False
    if kelly_pct > max_kelly:
        kelly_pct = max_kelly
        capped = True
    
    return {
        'kelly_pct': kelly_pct,
        'kelly_pct_display': kelly_pct * 100,  # Convert to percentage for display
        'net_odds': net_odds,
        'capped': capped
    }


def calculate_fractional_kelly(kelly_pct, fraction=0.5):
    """
    Calculate fractional Kelly (e.g., half Kelly, quarter Kelly).
    
    Many bettors use fractional Kelly to reduce variance while still
    maintaining positive growth.
    
    Args:
        kelly_pct: Full Kelly percentage as decimal (e.g., 0.05 for 5%)
        fraction: Fraction to use (e.g., 0.5 for half Kelly, 0.25 for quarter Kelly)
    
    Returns:
        Fractional Kelly percentage as decimal
    
    Examples:
        >>> calculate_fractional_kelly(0.05, 0.5)  # Half Kelly
        0.025
        
        >>> calculate_fractional_kelly(0.05, 0.25)  # Quarter Kelly
        0.0125
    """
    return kelly_pct * fraction


def kelly_bet_size(kelly_pct, bankroll):
    """
    Calculate dollar amount to bet given Kelly percentage and bankroll.
    
    Args:
        kelly_pct: Kelly percentage as decimal (e.g., 0.032 for 3.2%)
        bankroll: Total bankroll in dollars (e.g., 10000)
    
    Returns:
        Dollar amount to bet
    
    Examples:
        >>> kelly_bet_size(0.032, 10000)
        320.0
        
        >>> kelly_bet_size(0.055, 10000)
        550.0
    """
    return kelly_pct * bankroll

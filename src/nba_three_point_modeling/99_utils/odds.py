"""Odds math helpers used across modeling, pricing, and backtesting."""

from __future__ import annotations

from typing import Tuple


def american_to_implied_prob(american_odds: float) -> float:
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    return (-american_odds) / ((-american_odds) + 100.0)


def implied_prob_to_american(probability: float) -> float:
    """Convert implied probability (0,1) to American odds."""
    if probability <= 0.0 or probability >= 1.0:
        raise ValueError(f"Probability must be between 0 and 1, got {probability}")
    if probability >= 0.5:
        return -100.0 * probability / (1.0 - probability)
    return 100.0 * (1.0 - probability) / probability


def remove_vig_two_way(prob_over: float, prob_under: float) -> Tuple[float, float]:
    """Normalize two-way implied probabilities to no-vig probabilities."""
    total = prob_over + prob_under
    return prob_over / total, prob_under / total


def target_profit_stake(american_odds: float, target_profit: float = 100.0) -> float:
    """
    Stake sizing rule: amount required to win target_profit.

    Examples:
    -110 -> 110 stake for 100 win
    +200 -> 50 stake for 100 win
    """
    if american_odds < 0:
        return target_profit * ((-american_odds) / 100.0)
    return target_profit * (100.0 / american_odds)


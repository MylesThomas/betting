"""
Formatting utilities for Streamlit dashboard.

Shared formatters used across NBA and NFL arb dashboards for consistent display.
"""


def format_large_number(num):
    """
    Format large numbers with K/M/B suffixes for clean display.
    
    Examples:
        1234 -> "1.2K"
        12345 -> "12.3K"
        1234567 -> "1.2M"
        1234567890 -> "1.2B"
    
    Args:
        num: Number to format (int or float)
    
    Returns:
        Formatted string with appropriate suffix
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return f"{num:.0f}"


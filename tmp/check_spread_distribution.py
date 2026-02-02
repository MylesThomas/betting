"""
Check if whole-point spreads (2.0, 3.0, 4.0) are from consensus averaging
or if bookmakers actually post them.
"""

import pandas as pd
from pathlib import Path

# Load cached 2025-26 data (before merge)
cache_dir = Path.home() / 'Downloads' / 'tmp'
lines_df = pd.read_parquet(cache_dir / 'nba_2025-26_lines.parquet')

# Filter to spreads only
spreads = lines_df[lines_df['market'] == 'spreads'].copy()

print("=" * 60)
print("SPREAD DISTRIBUTION - RAW DATA (2025-26)")
print("=" * 60)

# Check away spreads
away_spreads = spreads['away_spread'].dropna()
home_spreads = spreads['home_spread'].dropna()

print(f"\nTotal spread lines: {len(spreads)}")
print(f"Unique bookmakers: {spreads['bookmaker'].nunique()}")
print(f"Bookmakers: {sorted(spreads['bookmaker'].unique())}")

print("\n" + "=" * 60)
print("AWAY SPREADS - checking for whole numbers")
print("=" * 60)

# Count whole vs half-point spreads
away_whole = (away_spreads % 1 == 0).sum()
away_half = (away_spreads % 1 != 0).sum()

print(f"Whole-point spreads (2.0, 3.0, etc.): {away_whole}")
print(f"Half-point spreads (2.5, 3.5, etc.): {away_half}")
print(f"Percentage whole-point: {away_whole / len(away_spreads) * 100:.1f}%")

# Show examples of whole-point spreads
whole_spreads = spreads[spreads['away_spread'] % 1 == 0][['game_date', 'away_team', 'home_team', 'away_spread', 'bookmaker']].head(10)
print("\nSample of whole-point spreads from raw data:")
print(whole_spreads.to_string(index=False))

print("\n" + "=" * 60)
print("MOST COMMON SPREAD VALUES (raw from bookmakers)")
print("=" * 60)
spread_counts = away_spreads.abs().value_counts().head(20)
print(spread_counts)

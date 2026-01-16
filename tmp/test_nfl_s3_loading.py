"""
Quick test to verify NFL lines load from S3 correctly.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from nfl_luck_utils import load_nfl_betting_lines

print("=" * 80)
print("TESTING NFL LINES S3 LOADING")
print("=" * 80)

# Test loading 2025 season from S3
print("\n📈 Loading 2025 season from S3...")
df = load_nfl_betting_lines(season=2025)

if df.empty:
    print("❌ ERROR: No data loaded!")
else:
    print(f"\n✅ SUCCESS!")
    print(f"   Total lines: {len(df):,}")
    print(f"   Unique games: {df['game_id'].nunique()}")
    print(f"   Unique bookmakers: {df['bookmaker'].nunique()}")
    print(f"   Date range: {df['game_time'].min()} to {df['game_time'].max()}")
    
    # Show sample
    print(f"\n📋 Sample data:")
    print(df.head(3))
    
    print(f"\n✅ S3 loading working correctly!")


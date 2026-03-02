"""
Temporary script to inspect nba_prop_strategies.parquet: columns, sample rows (LeBron, null GAME_ID).
Uses market_median_value_{market} and actual_{stat}. Column in parquet is GAME_ID (uppercase).
"""
import pandas as pd
from pathlib import Path

PARQUET = Path.home() / "Downloads" / "tmp" / "nba_prop_strategies.parquet"
if not PARQUET.exists():
    print(f"Missing {PARQUET}")
    exit(1)

df = pd.read_parquet(PARQUET)
print("Columns:", list(df.columns))
print("Shape:", df.shape)
median_cols = sorted(c for c in df.columns if c.startswith("market_median_value_"))
# actual_points, actual_rebounds, etc. (not actual_pts/actual_reb from logs)
actual_cols = [c for c in df.columns if c in ("actual_points", "actual_rebounds", "actual_assists", "actual_steals", "actual_blocks", "actual_threes", "actual_points_rebounds_assists", "actual_double_double", "actual_triple_double")]
print("\n--- LeBron James, GAME_ID null (first 2 rows) ---")
lb = df[(df["player"] == "LeBron James") & (df["GAME_ID"].isna())].head(2)
for idx, row in lb.iterrows():
    print(f"\nRow: game_date={row['game_date']}, season={row['season']}")
    for c in median_cols[:5]:
        print(f"  {c}={row[c]}")
    for c in actual_cols[:5]:
        print(f"  {c}={row[c]}")

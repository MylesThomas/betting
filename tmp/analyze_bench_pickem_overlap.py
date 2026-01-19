"""
Analyze the overlap between bench_pickem_under (2D) and its 3D components.

Context: Investigating why bench_pickem_perimeter_under + bench_pickem_rim_under 
does not equal bench_pickem_under. Hypothesis: 2D includes players without 
scorer_type classification, while 3D only includes players with defined scorer_type.

Created: 2026-01-18 5:20pm CT
"""

import pandas as pd

# Load the filtered plays
df_2d = pd.read_csv('tmp/2026-01-18_2d_top3.csv')
df_3d = pd.read_csv('tmp/2026-01-18_3d_top3.csv')

print("="*80)
print("ANALYZING BENCH PICKEM UNDER OVERLAP")
print("="*80)

# Filter for bench_pickem_under strategy from 2D
bench_pickem_2d = df_2d[df_2d['strategy_name'] == 'bench_pickem_under'].copy()
print(f"\n2D bench_pickem_under: {len(bench_pickem_2d)} plays")

# Filter for bench_pickem rim/perimeter from 3D
bench_pickem_rim_3d = df_3d[df_3d['strategy_name'] == 'bench_pickem_rim_under'].copy()
bench_pickem_perimeter_3d = df_3d[df_3d['strategy_name'] == 'bench_pickem_perimeter_under'].copy()

print(f"3D bench_pickem_rim_under: {len(bench_pickem_rim_3d)} plays")
print(f"3D bench_pickem_perimeter_under: {len(bench_pickem_perimeter_3d)} plays")
print(f"3D total (rim + perimeter): {len(bench_pickem_rim_3d) + len(bench_pickem_perimeter_3d)} plays")

# Combine 3D plays
bench_pickem_3d_combined = pd.concat([bench_pickem_rim_3d, bench_pickem_perimeter_3d])

print(f"\n❓ Expected: 2D count = 3D rim + 3D perimeter")
print(f"   Actual: {len(bench_pickem_2d)} = {len(bench_pickem_rim_3d)} + {len(bench_pickem_perimeter_3d)}")
print(f"   Match: {len(bench_pickem_2d) == len(bench_pickem_3d_combined)}")

# Find players in 2D but not in 3D
players_2d = set(bench_pickem_2d['player'].unique())
players_3d = set(bench_pickem_3d_combined['player'].unique())

missing_in_3d = players_2d - players_3d
extra_in_3d = players_3d - players_2d

print(f"\n📊 PLAYER OVERLAP:")
print(f"   Players in 2D only: {len(missing_in_3d)}")
print(f"   Players in 3D only: {len(extra_in_3d)}")
print(f"   Players in both: {len(players_2d & players_3d)}")

# Show players missing in 3D (likely no scorer_type)
if missing_in_3d:
    print(f"\n🔍 PLAYERS IN 2D BUT NOT IN 3D (likely no scorer_type):")
    print("─" * 80)
    missing_df = bench_pickem_2d[bench_pickem_2d['player'].isin(missing_in_3d)]
    
    # Check if scorer_type column exists
    if 'scorer_type' in missing_df.columns:
        print(f"\nScorer type distribution for missing players:")
        print(missing_df['scorer_type'].value_counts(dropna=False))
        print()
        for _, row in missing_df.iterrows():
            scorer = row.get('scorer_type', 'N/A')
            print(f"   {row['player']:30s} | scorer_type: {scorer}")
    else:
        print("\nNo scorer_type column in 2D data")
        for _, row in missing_df.iterrows():
            print(f"   {row['player']:30s}")

# Show players in 3D but not in 2D (shouldn't happen)
if extra_in_3d:
    print(f"\n⚠️ PLAYERS IN 3D BUT NOT IN 2D (unexpected!):")
    print("─" * 80)
    extra_df = bench_pickem_3d_combined[bench_pickem_3d_combined['player'].isin(extra_in_3d)]
    for _, row in extra_df.iterrows():
        scorer = row.get('scorer_type', 'N/A')
        print(f"   {row['player']:30s} | scorer_type: {scorer}")

# Check column overlap
print(f"\n📋 COLUMN COMPARISON:")
print(f"   2D columns: {list(bench_pickem_2d.columns)}")
print(f"   3D columns: {list(bench_pickem_3d_combined.columns)}")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
if len(missing_in_3d) > 0:
    print(f"✅ Hypothesis CONFIRMED: 2D includes {len(missing_in_3d)} player(s) without scorer_type")
    print(f"   These players are in bench_pickem_under but NOT in rim/perimeter variants")
else:
    print(f"❌ Hypothesis REJECTED: All 2D players have scorer_type classification")
    print(f"   Need to investigate other reasons for the discrepancy")


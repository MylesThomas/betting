"""
Check if bench_pickem_perimeter_under has any YTD plays in 2025-26

Looking at actual tracked plays from this season to see if:
1. The strategy has generated any plays
2. If so, what the performance is
"""

import pandas as pd
import boto3
from io import StringIO

s3_client = boto3.client('s3')
bucket = 'nba-betting-mt'

# Try to load YTD tracking data for 2025-26
print("="*80)
print("CHECKING 2025-26 YTD TRACKED PLAYS")
print("="*80)

# Try both 2D and 3D tracking files
for strategy_type in ['2d', '3d']:
    print(f"\n{'='*80}")
    print(f"Checking {strategy_type.upper()} tracking data...")
    print("="*80)
    
    try:
        # Try to list all tracking files for this season
        prefix = f'data/04_output/plays/role_spread_points_model/{strategy_type}/tracking/'
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        if 'Contents' not in response:
            print(f"  ⚠️  No tracking files found")
            continue
        
        # Get the most recent tracking file
        tracking_files = [obj['Key'] for obj in response['Contents'] if 'tracking_summary_2025-26' in obj['Key']]
        
        if not tracking_files:
            print(f"  ⚠️  No 2025-26 tracking summary found")
            continue
        
        # Use the most recent one
        tracking_file = sorted(tracking_files)[-1]
        print(f"  ✅ Found: {tracking_file}")
        
        # Load it
        response = s3_client.get_object(Bucket=bucket, Key=tracking_file)
        df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
        
        print(f"  📊 Loaded {len(df)} tracked plays")
        
        # Filter to bench pick'em strategies
        bench_pickem = df[
            (df['line_tier'] == '5-10 (Bench)') &
            (df['spread_bin'] == 'Pick\'em (-2 to +2)')
        ]
        
        if len(bench_pickem) > 0:
            print(f"\n  📊 Bench Pick'em plays: {len(bench_pickem)}")
            
            if strategy_type == '3d' and 'scorer_type' in df.columns:
                scorer_counts = bench_pickem['scorer_type'].value_counts()
                print(f"\n  By scorer_type:")
                for scorer, count in scorer_counts.items():
                    print(f"    {scorer}: {count} plays")
                
                # Check for perimeter specifically
                perimeter = bench_pickem[bench_pickem['scorer_type'] == 'Perimeter (<40.0%)']
                if len(perimeter) > 0:
                    print(f"\n  🎯 FOUND PERIMETER PLAYS!")
                    print(f"    Total: {len(perimeter)}")
                    wins = (perimeter['result'] == 'WIN').sum()
                    losses = (perimeter['result'] == 'LOSS').sum()
                    profit = perimeter['profit'].sum()
                    print(f"    W-L: {wins}-{losses}")
                    print(f"    Profit: ${profit:,.2f}")
        else:
            print(f"  ⚠️  No bench pick'em plays found")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

print(f"\n{'='*80}")
print("CONCLUSION")
print("="*80)
print("""
If no perimeter plays were found:
  - The strategy may be too rare to generate plays frequently
  - Market makers might not offer lines on bench perimeter players in pick'em games
  - You might need to wait longer for this strategy to generate plays

This is why you're seeing all 0's in the v2 config!
""")


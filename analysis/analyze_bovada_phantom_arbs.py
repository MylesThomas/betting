"""
Analyze Bovada involvement in arbitrage opportunities from existing data.
No API calls needed - uses recent arb files from data/04_output/nba/arbs/.

PROBLEM STATEMENT:
    82% of arbs involve Bovada, and Bovada arbs are 3.2x more profitable than others.
    This suggests Bovada's lines in the-odds-api are STALE - they don't update as 
    frequently as other bookmakers. When you check the actual Bovada website, these 
    lines have already moved, creating "phantom arbs" that don't actually exist.

WHAT THIS SCRIPT DOES:
    - Loads recent arb data files (last 100 files with actual opportunities)
    - Calculates how often each bookmaker appears in arbitrage opportunities
    - Compares Bovada's involvement rate and average profit vs other bookmakers
    - Identifies if Bovada arbs are disproportionately profitable (smoking gun for stale data)

EXPECTED RESULTS:
    Before Staleness Filtering:
        - 82% of arbs involve Bovada
        - Bovada arbs average 8.65% profit (3.2x higher than others)
        - Most are phantom arbs (don't exist on actual website)
    
    After Staleness Filtering (with is_stale column):
        - ~40-50% of arbs involve Bovada (only fresh lines)
        - Bovada arbs average ~3-4% profit (closer to market)
        - Phantom arbs filtered out automatically via is_stale flag

SOLUTION:
    See docs/BOVADA_STALENESS_IMPLEMENTATION.md for the complete implementation
    that adds staleness filtering to find_nba_arb_opportunities.py. The fix tracks
    'last_update' timestamps from the API and flags/filters stale lines before they
    appear in email alerts.
"""

import pandas as pd
from pathlib import Path
from collections import Counter

def main():
    print("="*80)
    print("🔍 BOVADA PHANTOM ARB ANALYSIS")
    print("="*80)
    
    # Load recent arb files with actual data
    data_dir = Path('/Users/thomasmyles/dev/betting/data/04_output/nba/arbs')
    
    # Get files from last 7 days that have data (>1KB)
    all_files = list(data_dir.glob('arb_output_202512*.csv'))
    arb_files = [f for f in all_files if f.stat().st_size > 1000]  # Only files with data
    arb_files = sorted(arb_files)[-100:]  # Last 100 files with data
    
    print(f"\n📂 Loading {len(arb_files)} recent files with actual arbs...")
    
    all_arbs = []
    for f in arb_files:
        try:
            df = pd.read_csv(f)
            if len(df) > 0 and 'is_arb' in df.columns:
                arbs = df[df['is_arb'] == True].copy()
                arbs['source_file'] = f.name
                all_arbs.append(arbs)
        except:
            pass
    
    if not all_arbs:
        print("❌ No arb data found")
        return
    
    df = pd.concat(all_arbs, ignore_index=True)
    print(f"✅ Loaded {len(df)} total arb opportunities\n")
    
    # Analyze bookmaker involvement
    print("="*80)
    print("📊 BOOKMAKER INVOLVEMENT IN ARBS")
    print("="*80)
    
    if 'best_over_book' not in df.columns or 'best_under_book' not in df.columns:
        print("❌ Missing bookmaker columns")
        return
    
    # Count how often each bookmaker appears
    over_books = Counter(df['best_over_book'].dropna())
    under_books = Counter(df['best_under_book'].dropna())
    
    # Combine counts
    all_books = set(over_books.keys()) | set(under_books.keys())
    book_stats = []
    
    for book in all_books:
        over_count = over_books.get(book, 0)
        under_count = under_books.get(book, 0)
        total = over_count + under_count
        involved = len(df[(df['best_over_book'] == book) | (df['best_under_book'] == book)])
        
        book_stats.append({
            'bookmaker': book,
            'over_side': over_count,
            'under_side': under_count,
            'total_appearances': total,
            'arbs_involved': involved,
            'pct_of_all_arbs': (involved / len(df)) * 100
        })
    
    stats_df = pd.DataFrame(book_stats).sort_values('arbs_involved', ascending=False)
    
    print(f"\n{stats_df.to_string(index=False)}\n")
    
    # Bovada-specific analysis
    print("="*80)
    print("🎯 BOVADA SPECIFIC ANALYSIS")
    print("="*80)
    
    bovada_arbs = df[(df['best_over_book'] == 'bovada') | (df['best_under_book'] == 'bovada')]
    bovada_count = len(bovada_arbs)
    bovada_pct = (bovada_count / len(df)) * 100
    
    print(f"\nBovada involved in: {bovada_count} / {len(df)} arbs ({bovada_pct:.1f}%)")
    
    if bovada_count > 0:
        print(f"\n🎰 Sample high-profit Bovada arbs:")
        sample_cols = ['player', 'market', 'line', 'best_over_book', 'best_over_odds', 
                      'best_under_book', 'best_under_odds', 'expected_profit_pct']
        sample_cols = [c for c in sample_cols if c in bovada_arbs.columns]
        
        sample = bovada_arbs.nlargest(10, 'expected_profit_pct')[sample_cols]
        print(sample.to_string(index=False))
        
        # Which side is Bovada usually on?
        bovada_over_only = len(df[(df['best_over_book'] == 'bovada') & (df['best_under_book'] != 'bovada')])
        bovada_under_only = len(df[(df['best_under_book'] == 'bovada') & (df['best_over_book'] != 'bovada')])
        
        print(f"\n📊 Bovada positioning:")
        print(f"  On OVER side only: {bovada_over_only}")
        print(f"  On UNDER side only: {bovada_under_only}")
        
        # Average profit for Bovada arbs vs others
        bovada_avg_profit = bovada_arbs['expected_profit_pct'].mean()
        non_bovada_arbs = df[~((df['best_over_book'] == 'bovada') | (df['best_under_book'] == 'bovada'))]
        non_bovada_avg_profit = non_bovada_arbs['expected_profit_pct'].mean() if len(non_bovada_arbs) > 0 else 0
        
        print(f"\n💰 Profit comparison:")
        print(f"  Bovada arbs avg profit: {bovada_avg_profit:.2f}%")
        print(f"  Non-Bovada arbs avg profit: {non_bovada_avg_profit:.2f}%")
        
        if bovada_avg_profit > non_bovada_avg_profit * 1.5:
            print(f"\n🚨 SMOKING GUN!")
            print(f"  Bovada arbs are {bovada_avg_profit/non_bovada_avg_profit:.1f}x more profitable")
            print(f"  This suggests Bovada's lines are STALE")
            print(f"  → Other books move their lines")
            print(f"  → Bovada's API doesn't update")
            print(f"  → Phantom arb appears with huge edge")
            print(f"  → But Bovada's website already adjusted!")
    
    print("\n" + "="*80)
    print("💡 CONCLUSION")
    print("="*80)
    
    print(f"\nFrom your email: 11 out of 13 arbs (84.6%) involved Bovada")
    print(f"From recent data: {bovada_pct:.1f}% of arbs involve Bovada")
    
    if bovada_pct > 50:
        print(f"\n🎯 VERDICT: Bovada is over-represented in arbitrage opportunities.")
        print(f"\n📋 RECOMMENDED ACTIONS:")
        print(f"  1. ❌ Exclude Bovada from automated alerts")
        print(f"  2. ⚠️  If keeping Bovada, always verify manually on their website")
        print(f"  3. 📊 Add 'VERIFY_BOVADA' flag to high-profit Bovada arbs")
        print(f"  4. 🔍 Track phantom arb rate (arbs that don't exist when checked)")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()


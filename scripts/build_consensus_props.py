"""
Build Consensus Props Dataset

Takes raw historical props and creates clean consensus dataset:
- One row per player per game
- Consensus line using penalty-based methodology
- Full price tracking (best/avg/worst for over and under)
- Bookmaker tracking (who offers best prices)
- Alternative lines available
- Arbitrage detection (negative vig opportunities)

TIMESTAMP COLUMNS (if available in raw data):
- bookmaker_last_update: When bookmaker updated odds
- market_last_update: When player_threes market specifically updated
Use these to filter stale lines and verify arbitrage windows

NOTE: Data fetched BEFORE 2025-03-31 does NOT have timestamps.
Data fetched on/after 2025-03-31 has bookmaker_last_update and market_last_update columns.

IMPORTANT: From our analysis, bookmakers update odds within a couple minutes of each other.
This means arbitrage opportunities identified in the data should still exist at the time of fetch.
If arbs don't actually exist when you try to bet them, it's likely due to the historical data
being fetched after the arbitrage window closed (not due to stale timestamps in the data).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob


def american_odds_to_probability(odds):
    """Convert American odds to implied probability percentage"""
    if odds == 0:
        raise ValueError("Odds cannot be 0")
    
    if odds < 0:
        probability = abs(odds) / (abs(odds) + 100)
    else:
        probability = 100 / (odds + 100)
    
    return probability * 100


def calculate_vig(over_odds, under_odds):
    """
    Calculate vig (distance from 100% total probability)
    
    Returns SIGNED value:
    - Positive vig = bookmaker has edge (total > 100%)
    - Negative vig = ARBITRAGE OPPORTUNITY (total < 100%)
    - Zero vig = fair odds (total = 100%)
    """
    over_prob = american_odds_to_probability(over_odds)
    under_prob = american_odds_to_probability(under_odds)
    total_prob = over_prob + under_prob
    return total_prob - 100.0  # Keep the sign!


def find_consensus_line_for_player(player_df):
    """
    Find consensus line using penalty-based approach.
    
    Key Points:
    1. Pick consensus line (penalty-based: most books unless other line is WAY better)
    2. At consensus line, aggregate prices across ALL bookmakers:
       - Best odds (for betting)
       - Average odds (for market view)
       - Worst odds (to avoid)
    3. Track which bookmaker offers best price on each side
    
    Returns full data structure with pricing and alternatives.
    """
    PENALTY_PER_BOOK = 1.5  # % penalty per missing bookmaker
    
    # Analyze each line
    line_analysis = []
    
    for line in sorted(player_df['line'].unique()):
        line_df = player_df[player_df['line'] == line].copy()
        
        # Calculate vig for each bookmaker at this line
        vigs = []
        for _, row in line_df.iterrows():
            vig = calculate_vig(row['over_odds'], row['under_odds'])
            vigs.append(vig)
        
        avg_vig = np.mean(vigs)
        num_bookmakers = len(line_df)
        
        line_analysis.append({
            'line': line,
            'num_bookmakers': num_bookmakers,
            'avg_vig': avg_vig,
            'df': line_df
        })
    
    # Sort by number of bookmakers (descending)
    line_analysis.sort(key=lambda x: x['num_bookmakers'], reverse=True)
    
    # Start with line that has most bookmakers
    consensus = line_analysis[0]
    
    # Check if any challenger can beat consensus
    for challenger in line_analysis[1:]:
        book_diff = consensus['num_bookmakers'] - challenger['num_bookmakers']
        required_advantage = PENALTY_PER_BOOK * book_diff
        actual_advantage = consensus['avg_vig'] - challenger['avg_vig']
        
        if actual_advantage > required_advantage:
            consensus = challenger
            break
    
    # Extract full pricing data for consensus line
    # NOTE: Consensus line may have 1 to N bookmakers offering it
    # We aggregate across ALL bookmakers at this line
    consensus_df = consensus['df']
    num_books = len(consensus_df)
    
    # BEST PRICES (for value betting - shop across all bookmakers)
    over_best_odds = int(consensus_df['over_odds'].max())
    over_best_book = consensus_df.loc[consensus_df['over_odds'].idxmax(), 'bookmaker']
    
    under_best_odds = int(consensus_df['under_odds'].max())
    under_best_book = consensus_df.loc[consensus_df['under_odds'].idxmax(), 'bookmaker']
    
    # AVERAGE PRICES (market consensus view - average across all N bookmakers)
    # Handle edge cases where odds might be 0 or NaN
    over_avg_odds = consensus_df['over_odds'].replace(0, np.nan).mean()
    under_avg_odds = consensus_df['under_odds'].replace(0, np.nan).mean()
    
    # If all were 0/NaN, fall back to best odds
    if pd.isna(over_avg_odds) or over_avg_odds == 0:
        over_avg_odds = float(over_best_odds)
    if pd.isna(under_avg_odds) or under_avg_odds == 0:
        under_avg_odds = float(under_best_odds)
    
    # WORST PRICES (to avoid - know the bad end of market)
    over_worst_odds = int(consensus_df['over_odds'].min())
    over_worst_book = consensus_df.loc[consensus_df['over_odds'].idxmin(), 'bookmaker']
    
    under_worst_odds = int(consensus_df['under_odds'].min())
    under_worst_book = consensus_df.loc[consensus_df['under_odds'].idxmin(), 'bookmaker']
    
    # MARKET EFFICIENCY
    # Best possible vig if you shop for best odds on both sides
    best_market_vig = calculate_vig(over_best_odds, under_best_odds)
    # Average vig across all bookmakers (already calculated above)
    avg_vig_from_avg_odds = calculate_vig(over_avg_odds, under_avg_odds)
    
    # ARBITRAGE DETECTION
    is_arbitrage = best_market_vig < 0  # Negative vig = arb opportunity
    arb_profit_pct = abs(best_market_vig) if is_arbitrage else 0  # Guaranteed profit %
    
    # Get alternative lines (non-consensus lines available)
    alternative_lines = [la['line'] for la in line_analysis if la['line'] != consensus['line']]
    
    return {
        'consensus_line': consensus['line'],
        'num_bookmakers': num_books,  # How many books offer consensus line
        'avg_vig': round(consensus['avg_vig'], 2),
        
        # OVER side: prices aggregated across all N bookmakers at consensus line
        'over_best_odds': over_best_odds,        # Best for betting
        'over_best_book': over_best_book,        # Who offers best
        'over_avg_odds': round(over_avg_odds, 1),  # Average across N books
        'over_worst_odds': over_worst_odds,      # Worst (avoid)
        'over_worst_book': over_worst_book,      # Who offers worst
        
        # UNDER side: prices aggregated across all N bookmakers at consensus line
        'under_best_odds': under_best_odds,      # Best for betting
        'under_best_book': under_best_book,      # Who offers best
        'under_avg_odds': round(under_avg_odds, 1),  # Average across N books
        'under_worst_odds': under_worst_odds,    # Worst (avoid)
        'under_worst_book': under_worst_book,    # Who offers worst
        
        # Market efficiency metrics
        'best_market_vig': round(best_market_vig, 2),  # SIGNED: negative = arbitrage!
        'avg_market_vig': round(avg_vig_from_avg_odds, 2),  # Using average odds
        
        # Arbitrage detection
        'is_arbitrage': is_arbitrage,  # True if negative vig
        'arb_opportunity': 1 if is_arbitrage else 0,  # Binary 1/0 flag
        'arb_profit_pct': round(arb_profit_pct, 2),  # Guaranteed profit %
        
        # Alternative lines (non-consensus lines available)
        'alternative_lines': alternative_lines if alternative_lines else None,
        'num_alternative_lines': len(alternative_lines)
    }


def process_all_props():
    """
    Process all historical props CSVs and build consensus dataset.
    
    For each player/game:
    1. Find consensus line (penalty-based)
    2. Aggregate prices across ALL bookmakers offering that line:
       - Best odds (for value)
       - Average odds (market view)  
       - Worst odds (to avoid)
    3. Track alternative lines available
    """
    
    # Find all props files
    props_dir = Path(__file__).parent.parent / 'historical_props'
    csv_files = sorted(glob.glob(str(props_dir / 'props_*_player_threes.csv')))
    
    if not csv_files:
        print("❌ No props CSV files found!")
        return None
    
    print(f"Found {len(csv_files)} CSV files to process")
    print(f"Building consensus props dataset...")
    print(f"  - Picks consensus line per player/game (penalty-based)")
    print(f"  - Aggregates prices across N bookmakers at consensus line")
    print(f"  - Tracks best/avg/worst odds on each side\n")
    
    all_consensus_rows = []
    
    for i, csv_file in enumerate(csv_files, 1):
        filename = Path(csv_file).name
        date_str = filename.split('_')[1]  # Extract date from filename
        
        print(f"[{i}/{len(csv_files)}] Processing {date_str}...", end=' ')
        
        df = pd.read_csv(csv_file)
        
        # Group by player and game
        grouped = df.groupby(['player', 'game'])
        
        date_consensus = []
        for (player, game), player_df in grouped:
            consensus_data = find_consensus_line_for_player(player_df)
            
            # Build consensus row
            row = {
                'date': date_str,
                'player': player,
                'game': game,
                'game_time': player_df['game_time'].iloc[0],
                **consensus_data
            }
            
            date_consensus.append(row)
        
        all_consensus_rows.extend(date_consensus)
        print(f"✅ {len(date_consensus)} players")
    
    # Create DataFrame
    consensus_df = pd.DataFrame(all_consensus_rows)
    
    print(f"\n{'='*70}")
    print(f"CONSENSUS DATASET BUILT")
    print(f"{'='*70}")
    print(f"Total rows: {len(consensus_df):,}")
    print(f"Unique players: {consensus_df['player'].nunique()}")
    print(f"Unique games: {consensus_df['game'].nunique()}")
    print(f"Date range: {consensus_df['date'].min()} to {consensus_df['date'].max()}")
    
    # Show distribution of bookmaker counts at consensus line
    print(f"\n📊 Bookmakers at Consensus Line (how many books we're aggregating):")
    book_dist = consensus_df['num_bookmakers'].value_counts().sort_index()
    for num_books, count in book_dist.items():
        pct = (count / len(consensus_df)) * 100
        avg_note = f" (averaging across {num_books} books)" if num_books > 1 else " (single book)"
        print(f"   {num_books} bookmakers: {count:5} props ({pct:5.1f}%){avg_note}")
    
    return consensus_df


def analyze_bookmaker_performance(consensus_df):
    """Analyze which bookmakers consistently offer best prices"""
    
    print(f"\n{'='*70}")
    print("BOOKMAKER PERFORMANCE ANALYSIS")
    print(f"{'='*70}")
    
    # Count best over prices
    print("\n📊 Best OVER Prices (most often):")
    over_best_counts = consensus_df['over_best_book'].value_counts().head(10)
    for book, count in over_best_counts.items():
        pct = (count / len(consensus_df)) * 100
        print(f"   {book:20} - {count:5} times ({pct:5.1f}%)")
    
    # Count best under prices
    print("\n📊 Best UNDER Prices (most often):")
    under_best_counts = consensus_df['under_best_book'].value_counts().head(10)
    for book, count in under_best_counts.items():
        pct = (count / len(consensus_df)) * 100
        print(f"   {book:20} - {count:5} times ({pct:5.1f}%)")
    
    # Count worst over prices (to avoid)
    print("\n📊 Worst OVER Prices (most often):")
    over_worst_counts = consensus_df['over_worst_book'].value_counts().head(10)
    for book, count in over_worst_counts.items():
        pct = (count / len(consensus_df)) * 100
        print(f"   {book:20} - {count:5} times ({pct:5.1f}%)")
    
    # Count worst under prices (to avoid)
    print("\n📊 Worst UNDER Prices (most often):")
    under_worst_counts = consensus_df['under_worst_book'].value_counts().head(10)
    for book, count in under_worst_counts.items():
        pct = (count / len(consensus_df)) * 100
        print(f"   {book:20} - {count:5} times ({pct:5.1f}%)")


def analyze_alternative_lines(consensus_df):
    """Analyze availability of alternative lines"""
    
    print(f"\n{'='*70}")
    print("ALTERNATIVE LINES ANALYSIS")
    print(f"{'='*70}")
    
    props_with_alts = consensus_df[consensus_df['num_alternative_lines'] > 0]
    
    print(f"\nProps with alternative lines: {len(props_with_alts):,} ({len(props_with_alts)/len(consensus_df)*100:.1f}%)")
    print(f"Props with single line only: {len(consensus_df) - len(props_with_alts):,}")
    
    print(f"\nDistribution of alternative lines:")
    alt_dist = consensus_df['num_alternative_lines'].value_counts().sort_index()
    for num_alts, count in alt_dist.items():
        pct = (count / len(consensus_df)) * 100
        print(f"   {num_alts} alternatives: {count:5} props ({pct:5.1f}%)")


def analyze_arbitrage_opportunities(consensus_df):
    """Analyze arbitrage opportunities in the dataset"""
    
    print(f"\n{'='*70}")
    print("🚨 ARBITRAGE OPPORTUNITIES 🚨")
    print(f"{'='*70}")
    
    arb_props = consensus_df[consensus_df['is_arbitrage'] == True]
    
    print(f"\nTotal props: {len(consensus_df):,}")
    print(f"Arbitrage opportunities: {len(arb_props):,} ({len(arb_props)/len(consensus_df)*100:.1f}%)")
    
    if len(arb_props) > 0:
        print(f"\nProfit distribution:")
        print(f"   Average profit: {arb_props['arb_profit_pct'].mean():.2f}%")
        print(f"   Median profit: {arb_props['arb_profit_pct'].median():.2f}%")
        print(f"   Max profit: {arb_props['arb_profit_pct'].max():.2f}%")
        print(f"   Min profit: {arb_props['arb_profit_pct'].min():.2f}%")
        
        print(f"\n📊 Top 10 Arbitrage Opportunities:")
        top_arbs = arb_props.nlargest(10, 'arb_profit_pct')
        for i, row in enumerate(top_arbs.iterrows(), 1):
            idx, data = row
            print(f"\n   {i}. {data['player']} - {data['game']}")
            print(f"      Date: {data['date']} | Line: {data['consensus_line']} threes")
            print(f"      Over: {data['over_best_odds']:>4} @ {data['over_best_book']}")
            print(f"      Under: {data['under_best_odds']:>4} @ {data['under_best_book']}")
            print(f"      💰 Guaranteed Profit: {data['arb_profit_pct']:.2f}%")
    else:
        print("\n✅ No arbitrage opportunities found (all bookmakers have positive vig)")


def build_combined_props():
    """
    Build combined props dataset by appending all individual CSV files.
    Just adds a 'date' column and concatenates everything.
    """
    props_dir = Path(__file__).parent.parent / 'historical_props'
    csv_files = sorted(glob.glob(str(props_dir / 'props_*_player_threes.csv')))
    
    if not csv_files:
        print("❌ No props CSV files found!")
        return None
    
    print(f"\n{'='*70}")
    print("BUILDING COMBINED PROPS DATASET")
    print(f"{'='*70}")
    print(f"Appending {len(csv_files)} CSV files...")
    
    all_dfs = []
    
    for csv_file in csv_files:
        filename = Path(csv_file).name
        date_str = filename.split('_')[1]  # Extract date from filename
        
        df = pd.read_csv(csv_file)
        df['date'] = date_str
        all_dfs.append(df)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\n✅ Combined dataset built:")
    print(f"   Total rows: {len(combined_df):,}")
    print(f"   Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    
    return combined_df


def main():
    """Build consensus props dataset and analyze"""
    
    print("="*70)
    print("CONSENSUS PROPS BUILDER")
    print("="*70)
    print()
    
    # Build combined dataset (all raw props appended)
    combined_df = build_combined_props()
    
    if combined_df is not None:
        combined_output = Path(__file__).parent.parent / 'historical_props' / 'combined_props_player_threes.csv'
        combined_df.to_csv(combined_output, index=False)
        print(f"   💾 Saved: {combined_output}")
    
    # Build consensus dataset
    consensus_df = process_all_props()
    
    if consensus_df is None:
        return
    
    # Save to CSV
    output_file = Path(__file__).parent.parent / 'historical_props' / 'consensus_props_player_threes.csv'
    consensus_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved consensus dataset: {output_file}")
    
    # Analysis sections
    analyze_bookmaker_performance(consensus_df)
    analyze_alternative_lines(consensus_df)
    analyze_arbitrage_opportunities(consensus_df)
    
    print(f"\n{'='*70}")
    print("SAMPLE DATA")
    print(f"{'='*70}")
    print(consensus_df.head(10).to_string())
    
    print(f"\n{'='*70}")
    print("✅ Consensus dataset ready for trend analysis!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()


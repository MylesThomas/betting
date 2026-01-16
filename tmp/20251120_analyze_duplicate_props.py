"""
Analyze players with multiple prop lines in a single game

Focus on opening day: Celtics vs Knicks
"""

import pandas as pd
from pathlib import Path
import numpy as np

def american_odds_to_probability(odds):
    """
    Convert American odds to implied probability percentage.
    
    Args:
        odds: American odds (e.g., -141, +106, -100, +100)
    
    Returns:
        Probability as a percentage (0-100)
    
    Examples:
        -110 → 52.38% (favorite)
        +110 → 47.62% (underdog)
        -100 → 50.00% (even)
        +100 → 50.00% (even)
    """
    if odds == 0:
        raise ValueError("Odds cannot be 0")
    
    if odds < 0:
        # Negative odds (favorite)
        # Formula: |odds| / (|odds| + 100)
        probability = abs(odds) / (abs(odds) + 100)
    else:
        # Positive odds (underdog)
        # Formula: 100 / (odds + 100)
        probability = 100 / (odds + 100)
    
    return probability * 100  # Convert to percentage


def calculate_balance_score(over_odds, under_odds):
    """
    Calculate how close a line is to fair (100% total probability).
    
    Converts odds to probabilities and finds distance from 100% total.
    Bookmakers add "juice" (vig), so total probability is usually > 100%.
    
    Lower score = more balanced = closer to fair odds
    
    Returns: distance from 100% total probability
    
    Examples:
        -110/-110 → 52.38% + 52.38% = 104.76% → score = 4.76
        +105/-110 → 48.78% + 52.38% = 101.16% → score = 1.16 (better!)
        -100/+100 → 50.00% + 50.00% = 100.00% → score = 0.00 (perfect!)
    """
    over_prob = american_odds_to_probability(over_odds)
    under_prob = american_odds_to_probability(under_odds)
    
    total_prob = over_prob + under_prob
    
    # Distance from fair 100%
    return abs(total_prob - 100.0)


def find_consensus_line(player_df):
    """
    Find the consensus line using penalty-based approach.
    
    Algorithm:
    1. Start with line that has most bookmakers (consensus)
    2. Check if any other line is good enough to overcome penalty
    3. Penalty = 1.5% per bookmaker difference
    4. Track full pricing data for consensus line
    
    Returns: (consensus_data, all_lines_analysis)
    """
    PENALTY_PER_BOOK = 1.5  # % penalty per missing bookmaker
    
    # Analyze each line
    line_analysis = []
    
    for line in sorted(player_df['line'].unique()):
        line_df = player_df[player_df['line'] == line].copy()
        
        # Calculate vig for each bookmaker at this line
        vigs = []
        for _, row in line_df.iterrows():
            vig = calculate_balance_score(row['over_odds'], row['under_odds'])
            vigs.append(vig)
        
        avg_vig = np.mean(vigs)
        num_bookmakers = len(line_df)
        
        line_analysis.append({
            'line': line,
            'num_bookmakers': num_bookmakers,
            'avg_vig': avg_vig,
            'min_vig': min(vigs),
            'max_vig': max(vigs),
            'bookmakers': line_df['bookmaker'].tolist(),
            'df': line_df
        })
    
    # Sort by number of bookmakers (descending) - most books = initial consensus
    line_analysis.sort(key=lambda x: x['num_bookmakers'], reverse=True)
    
    # Start with line that has most bookmakers
    consensus = line_analysis[0]
    consensus_reason = f"Most bookmakers ({consensus['num_bookmakers']})"
    
    # Check if any challenger can beat consensus
    for i, challenger in enumerate(line_analysis[1:], 1):
        book_diff = consensus['num_bookmakers'] - challenger['num_bookmakers']
        required_advantage = PENALTY_PER_BOOK * book_diff
        actual_advantage = consensus['avg_vig'] - challenger['avg_vig']
        
        if actual_advantage > required_advantage:
            # Challenger overcomes penalty!
            old_consensus = consensus
            consensus = challenger
            consensus_reason = (f"Overcame penalty: {actual_advantage:.2f}% advantage > "
                              f"{required_advantage:.2f}% required (beats {old_consensus['num_bookmakers']} books)")
            break
    
    # Extract full pricing data for consensus line
    consensus_df = consensus['df']
    
    # Calculate best market vig (using best odds on each side)
    best_over = consensus_df['over_odds'].max()
    best_under = consensus_df['under_odds'].max()  # Max under (closest to 0) is best
    best_market_vig = calculate_balance_score(best_over, best_under)
    
    consensus_data = {
        'line': consensus['line'],
        'num_bookmakers': consensus['num_bookmakers'],
        'avg_vig': consensus['avg_vig'],
        'reason': consensus_reason,
        
        # Over side pricing
        'over_best_odds': int(consensus_df['over_odds'].max()),
        'over_avg_odds': consensus_df['over_odds'].mean(),
        'over_worst_odds': int(consensus_df['over_odds'].min()),
        'over_best_book': consensus_df.loc[consensus_df['over_odds'].idxmax(), 'bookmaker'],
        
        # Under side pricing (max is best since closer to 0)
        'under_best_odds': int(consensus_df['under_odds'].max()),
        'under_avg_odds': consensus_df['under_odds'].mean(),
        'under_worst_odds': int(consensus_df['under_odds'].min()),
        'under_best_book': consensus_df.loc[consensus_df['under_odds'].idxmax(), 'bookmaker'],
        
        # Market efficiency
        'best_market_vig': best_market_vig,
        'bookmakers': consensus['bookmakers']
    }
    
    return consensus_data, line_analysis


def analyze_opening_day():
    """Analyze opening day Celtics vs Knicks game for duplicate player props"""
    
    # Load opening day data
    props_file = Path(__file__).parent.parent / 'data' / '01_input' / 'the-odds-api' / 'nba' / 'historical_props' / 'props_2024-10-22_player_threes.csv'
    
    if not props_file.exists():
        print("Opening day file not found!")
        return
    
    df = pd.read_csv(props_file)
    
    print("="*80)
    print("OPENING DAY ANALYSIS: Multiple Props Per Player")
    print("="*80)
    print(f"\nTotal props: {len(df)}")
    print(f"Unique players: {df['player'].nunique()}")
    print(f"Games: {df['game'].unique()}")
    
    # Analyze each game
    for game in df['game'].unique():
        print(f"\n{'='*80}")
        print(f"GAME: {game}")
        print(f"{'='*80}")
        
        game_df = df[df['game'] == game].copy()
        
        print(f"\nTotal props in this game: {len(game_df)}")
        print(f"Unique players: {game_df['player'].nunique()}")
        
        # Count props per player
        player_counts = game_df['player'].value_counts()
        
        print(f"\n📊 Props per player distribution:")
        print(f"   Players with 1 prop:  {(player_counts == 1).sum()}")
        print(f"   Players with 2+ props: {(player_counts > 1).sum()}")
        
        # Find players with multiple props
        players_with_multiple = player_counts[player_counts > 1].index
        
        if len(players_with_multiple) == 0:
            print("\n✅ No players with multiple props in this game")
            continue
        
        print(f"\n{'='*80}")
        print(f"PLAYERS WITH MULTIPLE PROPS: {len(players_with_multiple)} players")
        print(f"{'='*80}")
        
        for player in sorted(players_with_multiple):
            player_df = game_df[game_df['player'] == player].sort_values('bookmaker')
            
            print(f"\n{'─'*80}")
            print(f"🏀 {player}")
            print(f"{'─'*80}")
            
            # Check if different lines or just different bookmakers
            unique_lines = player_df['line'].nunique()
            unique_bookmakers = player_df['bookmaker'].nunique()
            
            print(f"Total entries: {len(player_df)} | Unique lines: {unique_lines} | Bookmakers: {unique_bookmakers}")
            
            if unique_lines == 1:
                # Single line - show full pricing details
                line = player_df['line'].iloc[0]
                print(f"\n✅ SINGLE LINE: {line} threes")
                print(f"   {unique_bookmakers} bookmakers agree")
                
                # Calculate vigs and pricing
                vigs = []
                for _, row in player_df.iterrows():
                    vig = calculate_balance_score(row['over_odds'], row['under_odds'])
                    vigs.append(vig)
                
                avg_vig = np.mean(vigs)
                
                # Get best prices
                best_over = int(player_df['over_odds'].max())
                best_under = int(player_df['under_odds'].max())
                over_best_book = player_df.loc[player_df['over_odds'].idxmax(), 'bookmaker']
                under_best_book = player_df.loc[player_df['under_odds'].idxmax(), 'bookmaker']
                
                best_market_vig = calculate_balance_score(best_over, best_under)
                
                over_best_p = american_odds_to_probability(best_over)
                under_best_p = american_odds_to_probability(best_under)
                
                print(f"\n   📊 PRICING DETAILS:")
                print(f"   Over:")
                print(f"      Best:  {best_over:>4} ({over_best_p:.2f}%) @ {over_best_book}")
                print(f"      Avg:   {player_df['over_odds'].mean():>6.1f}")
                print(f"      Worst: {int(player_df['over_odds'].min()):>4}")
                print(f"   Under:")
                print(f"      Best:  {best_under:>4} ({under_best_p:.2f}%) @ {under_best_book}")
                print(f"      Avg:   {player_df['under_odds'].mean():>6.1f}")
                print(f"      Worst: {int(player_df['under_odds'].min()):>4}")
                print(f"")
                print(f"   Market efficiency:")
                print(f"      Best market vig: {best_market_vig:.2f}%")
                print(f"      Avg market vig:  {avg_vig:.2f}%")
                
            else:
                # Multiple lines - find consensus using penalty-based approach
                print(f"\n⚠️  MULTIPLE LINES - Analyzing with penalty-based consensus...")
                
                consensus_data, all_lines_analysis = find_consensus_line(player_df)
                
                print(f"\n   📋 All Lines with Details:")
                for i, line_info in enumerate(all_lines_analysis, 1):
                    is_consensus = (line_info['line'] == consensus_data['line'])
                    
                    print(f"\n   Option {i}: Line {line_info['line']} threes")
                    print(f"      Bookmakers ({line_info['num_bookmakers']}): {', '.join(line_info['bookmakers'])}")
                    
                    # Show odds for each bookmaker at this line
                    for _, row in line_info['df'].iterrows():
                        over_p = american_odds_to_probability(row['over_odds'])
                        under_p = american_odds_to_probability(row['under_odds'])
                        total_p = over_p + under_p
                        vig = total_p - 100
                        
                        print(f"         {row['bookmaker']:20} | {row['over_odds']:>4}/{row['under_odds']:>4} "
                              f"→ {over_p:.2f}%/{under_p:.2f}% = {total_p:.2f}% (vig: {vig:.2f}%)")
                    
                    print(f"      Average vig: {line_info['avg_vig']:.2f}%")
                    
                    if is_consensus:
                        print(f"      🏆 WINNER!")
                
                # Show consensus with full pricing details
                over_best_p = american_odds_to_probability(consensus_data['over_best_odds'])
                under_best_p = american_odds_to_probability(consensus_data['under_best_odds'])
                
                print(f"\n   ✅ CONSENSUS PICK:")
                print(f"      Line: {consensus_data['line']} threes")
                print(f"      Reason: {consensus_data['reason']}")
                print(f"      Bookmakers: {consensus_data['num_bookmakers']}")
                print(f"")
                print(f"      📊 PRICING DETAILS:")
                print(f"      Over:")
                print(f"         Best:  {consensus_data['over_best_odds']:>4} ({over_best_p:.2f}%) @ {consensus_data['over_best_book']}")
                print(f"         Avg:   {consensus_data['over_avg_odds']:>6.1f}")
                print(f"         Worst: {consensus_data['over_worst_odds']:>4}")
                print(f"      Under:")
                print(f"         Best:  {consensus_data['under_best_odds']:>4} ({under_best_p:.2f}%) @ {consensus_data['under_best_book']}")
                print(f"         Avg:   {consensus_data['under_avg_odds']:>6.1f}")
                print(f"         Worst: {consensus_data['under_worst_odds']:>4}")
                print(f"")
                print(f"      Market efficiency:")
                print(f"         Best market vig: {consensus_data['best_market_vig']:.2f}%")
                print(f"         Avg market vig:  {consensus_data['avg_vig']:.2f}%")


def main():
    analyze_opening_day()
    
    print("\n" + "="*80)
    print("CONSENSUS LINE METHODOLOGY SUMMARY")
    print("="*80)
    print("""
✅ IMPLEMENTED APPROACH: "Penalty-Based Consensus with Full Price Tracking"

STEP 1: Calculate Vig for Each Line
   - Convert odds to implied probabilities
   - Vig = (over% + under%) - 100%
   - Lower vig = more efficient market

STEP 2: Apply Penalty-Based Consensus
   - Start with line that has MOST bookmakers (initial consensus)
   - For each challenger line to beat consensus:
     Required advantage = 1.5% × (consensus_books - challenger_books)
   - If challenger's vig advantage > required, challenger wins
   
   Example: 1 book vs 5 books
   - Required advantage = 1.5% × 4 = 6.0%
   - 1-book line must be 6% more efficient to win
   
STEP 3: Track Full Pricing for Consensus Line
   - Best over odds (highest positive/closest to 0 negative)
   - Average over odds
   - Best under odds (closest to 0)
   - Average under odds
   - Which bookmaker offers best price on each side

WHY THIS WORKS:
✅ Respects market consensus (multiple books agreeing)
✅ Allows single-book lines to win if SIGNIFICANTLY better
✅ Tracks both average prices (market view) and best prices (value)
✅ Penalty scales with bookmaker difference

EXAMPLES:
- 1 book at 6.3% vig vs 5 books at 6.4% vig
  → Need 6% advantage, only have 0.1% → 5 books win ✅
  
- 1 book at 2.0% vig vs 5 books at 8.5% vig
  → Need 6% advantage, have 6.5% → 1 book wins ✅
    """)


if __name__ == "__main__":
    main()


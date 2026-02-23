"""
Find betting opportunities in tonight's NBA games based on underdog-unders strategy.

STRATEGY: Underdog-Unders (Fade Hot Streaks)
    Bet UNDER with POSITIVE ODDS after hot streaks:
    1. Player is on a hot streak (hit their 3pt prop 2-10 consecutive games)
    2. Market is still pricing the UNDER as an underdog (positive odds like +110, +135)
    3. We bet UNDER expecting regression to the mean
    
    Based on backtest: python backtesting/20251121_nba_3pt_prop_miss_streaks.py --underdog-unders
    
    Backtest Performance:
    - 3-game hot streaks: 241 bets, 46.5% win rate, +5.82% ROI ✅
    - 5-game hot streaks: 64 bets, 50.0% win rate, +1.75% ROI ✅

QUICK START:
    # Step 1: Fetch data first (see fetch_live_data.py)
    python implementation/fetch_live_data.py --test
    
    # Step 2: Find opportunities
    python implementation/find_todays_plays.py

USAGE:
    # Find opportunities using latest data
    python implementation/find_todays_plays.py
    
    # Custom data directory
    python implementation/find_todays_plays.py --data-dir data/custom/
    
    # Custom output directory
    python implementation/find_todays_plays.py --output-dir data/my_results/
    
    # Adjust streak length filters
    python implementation/find_todays_plays.py --min-streak 3 --max-streak 8

OUTPUT:
    Console: Detailed betting opportunities with player stats and trends
    CSV: data/results/recommended_bets_YYYYMMDD.csv

EXAMPLE OUTPUT:
    ====================================================================
    OPPORTUNITY #1
    ====================================================================
    
    🏀 Player:     Stephen Curry
    🎯 Game:       Golden State Warriors @ Los Angeles Lakers
    
    📊 Bet:        UNDER 4.5 threes
    💰 Odds:       +125
    📈 Implied:    44.4%
    💵 Bet Amount: $80.00 (to win $100)
    
    🔥 Hot Streak: 5 consecutive games hitting OVER
       Recent performance: [5, 6, 5, 7, 6]
       Recent attempts:    [10, 12, 11, 13, 12]
       Avg 3PA in streak:  11.6 ↑
    
    📝 Strategy:   Fade the hot streak - expecting regression to mean
    ⚠️  Confidence: Medium

TROUBLESHOOTING:
    No opportunities found?
    - This is normal - the strategy has specific criteria
    - Try on nights with more games (10+ games)
    - The test data may not have qualifying scenarios
    
    Want more strategies?
    - Check backtesting/20251121_nba_3pt_prop_miss_streaks.py for other approaches
    - Run --streak flag for cold-streak overs
    - Run --blind-under for blind under betting analysis

IMPORTANT NOTES:
    ⚠️  Risk Disclaimer:
    - This is based on historical backtesting
    - Past performance does not guarantee future results
    - Always bet responsibly within your limits
    - Verify all data before placing bets
    - Track your results to validate the strategy

Author: Myles Thomas
Date: 2025-11-21
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import utility functions
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from player_name_utils import normalize_player_name


def load_latest_data(data_dir='data/live'):
    """
    Load the most recent props and game results data.
    
    Args:
        data_dir: Directory containing the data files
        
    Returns:
        Tuple of (props_df, game_results_df, metadata)
    """
    data_path = Path(data_dir)
    
    # Find most recent files
    props_files = sorted(data_path.glob('props_today_*.csv'))
    results_files = sorted(data_path.glob('game_results_season_*.csv'))
    metadata_files = sorted(data_path.glob('metadata_*.json'))
    
    if not props_files:
        raise FileNotFoundError(f"No props files found in {data_dir}")
    if not results_files:
        raise FileNotFoundError(f"No game results files found in {data_dir}")
    
    props_file = props_files[-1]
    results_file = results_files[-1]
    
    print("Loading data...")
    print(f"  Props: {props_file}")
    print(f"  Game results: {results_file}")
    
    props_df = pd.read_csv(props_file)
    game_results_df = pd.read_csv(results_file)
    
    # Load metadata if available
    metadata = None
    if metadata_files:
        import json
        metadata_file = metadata_files[-1]
        print(f"  Metadata: {metadata_file}")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    print()
    return props_df, game_results_df, metadata


def calculate_consensus_props(props_df):
    """
    Calculate consensus lines and best odds from multiple bookmakers.
    
    Args:
        props_df: Raw props DataFrame with multiple bookmakers
        
    Returns:
        DataFrame with consensus lines and best odds per player per game
    """
    # Check if this is already processed data (from arbs/) or raw API data
    if 'over_odds' in props_df.columns and 'under_odds' in props_df.columns:
        # Already processed format from arbs/
        # Just need to normalize player names and aggregate by bookmaker
        consensus = []
        
        for (event_id, player), group in props_df.groupby(['event_id', 'player']):
            # Player names are already in the right format
            player_normalized = normalize_player_name(player)
            
            consensus.append({
                'game_id': event_id,
                'player': player_normalized,
                'player_name': player,
                'home_team': group['game'].iloc[0].split(' @ ')[1] if ' @ ' in group['game'].iloc[0] else '',
                'away_team': group['game'].iloc[0].split(' @ ')[0] if ' @ ' in group['game'].iloc[0] else '',
                'commence_time': group['game_time'].iloc[0],
                'consensus_line': np.median(group['line'].values),
                'over_best_odds': int(max(group['over_odds'].values)),
                'over_avg_odds': int(np.mean(group['over_odds'].values)),
                'under_best_odds': int(max(group['under_odds'].values)),
                'under_avg_odds': int(np.mean(group['under_odds'].values)),
                'num_bookmakers': len(group['bookmaker'].unique()),
            })
        
        return pd.DataFrame(consensus)
    
    else:
        # Raw API format - need to pivot and process
        # Normalize player names
        props_df['player'] = props_df['player_name'].apply(normalize_player_name)
        
        # Pivot to get over/under odds
        over_df = props_df[props_df['over_under'] == 'Over'].copy()
        under_df = props_df[props_df['over_under'] == 'Under'].copy()
        
        # Group by player and game
        consensus = []
        
        for (game_id, player), group in props_df.groupby(['game_id', 'player']):
            over_odds = over_df[(over_df['game_id'] == game_id) & (over_df['player'] == player)]['odds'].values
            under_odds = under_df[(under_df['game_id'] == game_id) & (under_df['player'] == player)]['odds'].values
            lines = group['line'].values
            
            if len(over_odds) == 0 or len(under_odds) == 0 or len(lines) == 0:
                continue
            
            consensus.append({
                'game_id': game_id,
                'player': player,
                'player_name': group['player_name'].iloc[0],
                'home_team': group['home_team'].iloc[0],
                'away_team': group['away_team'].iloc[0],
                'commence_time': group['commence_time'].iloc[0],
                'consensus_line': np.median(lines),
                'over_best_odds': int(max(over_odds)),
                'over_avg_odds': int(np.mean(over_odds)),
                'under_best_odds': int(max(under_odds)),
                'under_avg_odds': int(np.mean(under_odds)),
                'num_bookmakers': len(group['bookmaker_key'].unique()),
            })
        
        return pd.DataFrame(consensus)


def detect_hot_streaks(player, game_results_df, min_streak=2, max_streak=10):
    """
    Detect if a player is on a hot streak (hitting their 3pt prop consecutively).
    
    Args:
        player: Normalized player name
        game_results_df: Historical game results
        min_streak: Minimum consecutive hits to qualify
        max_streak: Maximum streak length to check
        
    Returns:
        Dict with streak info or None if no streak
    """
    # Normalize player names in game results
    game_results_df = game_results_df.copy()
    
    # Check if player column exists, if not create it from player_name
    if 'player' not in game_results_df.columns:
        if 'player_name' in game_results_df.columns:
            game_results_df['player'] = game_results_df['player_name'].apply(normalize_player_name)
        else:
            # Already has player column, just normalize it
            game_results_df['player'] = game_results_df['player'].apply(normalize_player_name)
    else:
        game_results_df['player'] = game_results_df['player'].apply(normalize_player_name)
    
    # Get player's recent games (sorted by date, most recent last)
    player_games = game_results_df[game_results_df['player'] == player].copy()
    
    if len(player_games) == 0:
        return None
    
    # Use 'date' column instead of 'game_date'
    date_col = 'date' if 'date' in player_games.columns else 'game_date'
    player_games = player_games.sort_values(date_col)
    
    # We need historical props to know what the line was
    # For now, use a simple heuristic: estimate line from recent performance
    # In production, you'd want historical props data
    
    # Estimate typical line from season average
    season_avg = player_games['threes_made'].mean()
    typical_lines = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    estimated_line = min(typical_lines, key=lambda x: abs(x - season_avg))
    
    # Check for consecutive hits (made >= estimated line)
    player_games['hit'] = player_games['threes_made'] >= estimated_line
    
    # Count consecutive hits from the end (most recent games)
    current_streak = 0
    streak_games = []
    
    for idx in reversed(player_games.index):
        game = player_games.loc[idx]
        if game['hit']:
            current_streak += 1
            streak_games.insert(0, game)
            
            if current_streak >= max_streak:
                break
        else:
            break
    
    # Return streak info if it qualifies
    if current_streak >= min_streak:
        return {
            'streak_length': current_streak,
            'estimated_line': estimated_line,
            'recent_makes': [g['threes_made'] for g in streak_games],
            'recent_attempts': [g['threes_attempted'] for g in streak_games],
            'recent_dates': [g[date_col] for g in streak_games],
            'games': streak_games,
        }
    
    return None


def find_underdog_under_opportunities(consensus_props, game_results_df):
    """
    Find underdog-under betting opportunities in tonight's games.
    
    Criteria:
    1. Player is on a hot streak (2-10 consecutive hits)
    2. UNDER odds are positive (underdog, e.g., +110, +135)
    3. Line is consistent with recent props
    
    Args:
        consensus_props: Tonight's consensus props
        game_results_df: Season-to-date game results
        
    Returns:
        DataFrame with recommended bets
    """
    opportunities = []
    
    print("="*100)
    print("SCANNING FOR UNDERDOG-UNDER OPPORTUNITIES")
    print("="*100)
    print()
    print("Strategy: Bet UNDER with positive odds after hot streaks (2-10 consecutive hits)")
    print()
    
    players_checked = 0
    hot_streaks_found = 0
    
    for idx, prop in consensus_props.iterrows():
        player = prop['player']
        line = prop['consensus_line']
        under_odds = prop['under_best_odds']
        
        players_checked += 1
        
        # Check if under is an underdog (positive odds)
        if under_odds <= 0:
            continue
        
        # Detect hot streak
        streak_info = detect_hot_streaks(player, game_results_df, min_streak=2, max_streak=10)
        
        if streak_info is None:
            continue
        
        hot_streaks_found += 1
        
        # Calculate implied probability
        implied_prob = 100 / (under_odds + 100) * 100  # Convert to percentage
        
        # Calculate bet amount (to win $100)
        bet_amount = 100 / under_odds * 100
        
        # Get trend info
        streak_3pa = streak_info['recent_attempts']
        trend_3pa = "↑" if len(streak_3pa) >= 2 and streak_3pa[-1] > streak_3pa[0] else "↓" if len(streak_3pa) >= 2 else "→"
        avg_3pa = np.mean(streak_3pa) if streak_3pa else 0
        
        opportunities.append({
            'player': player,
            'player_name': prop['player_name'],
            'game': f"{prop['away_team']} @ {prop['home_team']}",
            'commence_time': prop['commence_time'],
            'line': line,
            'bet_direction': 'UNDER',
            'odds': under_odds,
            'implied_probability': implied_prob,
            'bet_amount': bet_amount,
            'streak_length': streak_info['streak_length'],
            'estimated_line': streak_info['estimated_line'],
            'recent_makes': streak_info['recent_makes'],
            'recent_attempts': streak_info['recent_attempts'],
            'avg_3pa_in_streak': avg_3pa,
            'trend_3pa': trend_3pa,
            'confidence': 'Medium' if streak_info['streak_length'] >= 3 else 'Low',
        })
    
    print(f"Players checked: {players_checked}")
    print(f"Hot streaks found: {hot_streaks_found}")
    print(f"Underdog-under opportunities: {len(opportunities)}")
    print()
    
    return pd.DataFrame(opportunities)


def display_opportunities(opportunities_df):
    """
    Display betting opportunities in a nice format.
    
    Args:
        opportunities_df: DataFrame with recommended bets
    """
    if len(opportunities_df) == 0:
        print("="*100)
        print("NO BETTING OPPORTUNITIES FOUND")
        print("="*100)
        print()
        print("No players met the criteria:")
        print("  ❌ On a hot streak (2-10 consecutive hits)")
        print("  ❌ UNDER odds are positive (underdog)")
        print()
        return
    
    print("="*100)
    print(f"FOUND {len(opportunities_df)} BETTING OPPORTUNITY(IES)")
    print("="*100)
    print()
    
    for idx, opp in opportunities_df.iterrows():
        print(f"{'='*100}")
        print(f"OPPORTUNITY #{idx + 1}")
        print(f"{'='*100}")
        print()
        print(f"🏀 Player:     {opp['player_name']}")
        print(f"🎯 Game:       {opp['game']}")
        print(f"🕐 Time:       {opp['commence_time']}")
        print()
        print(f"📊 Bet:        UNDER {opp['line']} threes")
        print(f"💰 Odds:       {opp['odds']:+d}")
        print(f"📈 Implied:    {opp['implied_probability']:.1f}%")
        print(f"💵 Bet Amount: ${opp['bet_amount']:.2f} (to win $100)")
        print()
        print(f"🔥 Hot Streak: {opp['streak_length']} consecutive games hitting OVER")
        print(f"   Recent performance: {opp['recent_makes']}")
        print(f"   Recent attempts:    {opp['recent_attempts']}")
        print(f"   Avg 3PA in streak:  {opp['avg_3pa_in_streak']:.1f} {opp['trend_3pa']}")
        print()
        print(f"📝 Strategy:   Fade the hot streak - expecting regression to mean")
        print(f"⚠️  Confidence: {opp['confidence']}")
        print()
    
    print("="*100)


def save_recommendations(opportunities_df, output_dir='data/results'):
    """
    Save recommended bets to CSV file.
    
    Args:
        opportunities_df: DataFrame with recommended bets
        output_dir: Directory to save results
    """
    if len(opportunities_df) == 0:
        print("No bets to save.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    today_str = datetime.now().strftime('%Y%m%d')
    output_path = f'{output_dir}/recommended_bets_{today_str}.csv'
    
    opportunities_df.to_csv(output_path, index=False)
    
    print(f"💾 Saved recommendations to: {output_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Find betting opportunities in tonight\'s NBA games',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find opportunities using latest data
  python implementation/find_todays_plays.py
  
  # Specify custom data directory
  python implementation/find_todays_plays.py --data-dir data/live
  
  # Save results to custom location
  python implementation/find_todays_plays.py --output-dir data/custom/
        """
    )
    
    parser.add_argument('--data-dir', type=str, default='data/live',
                       help='Directory containing props and game results data')
    parser.add_argument('--output-dir', type=str, default='data/results',
                       help='Directory to save recommended bets')
    parser.add_argument('--min-streak', type=int, default=2,
                       help='Minimum streak length to qualify (default: 2)')
    parser.add_argument('--max-streak', type=int, default=10,
                       help='Maximum streak length to check (default: 10)')
    
    args = parser.parse_args()
    
    print()
    print("🎲 NBA 3PT PROP BETTING OPPORTUNITY FINDER")
    print("="*100)
    print()
    
    try:
        # Load data
        props_df, game_results_df, metadata = load_latest_data(args.data_dir)
        
        # Calculate consensus props
        print("Calculating consensus props...")
        consensus_props = calculate_consensus_props(props_df)
        print(f"  {len(consensus_props)} props for tonight's games")
        print()
        
        # Find opportunities
        opportunities = find_underdog_under_opportunities(consensus_props, game_results_df)
        
        # Display opportunities
        display_opportunities(opportunities)
        
        # Save recommendations
        save_recommendations(opportunities, args.output_dir)
        
        print()
        print("="*100)
        print("✅ ANALYSIS COMPLETE")
        print("="*100)
        print()
        
        if len(opportunities) > 0:
            print("⚠️  IMPORTANT REMINDERS:")
            print("   - This is based on historical backtesting - past performance doesn't guarantee future results")
            print("   - Always verify the line is correct before placing bets")
            print("   - Bet responsibly and within your limits")
            print("   - Track your results to validate the strategy")
        print()
        
    except Exception as e:
        print()
        print("="*100)
        print(f"❌ ERROR: {e}")
        print("="*100)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


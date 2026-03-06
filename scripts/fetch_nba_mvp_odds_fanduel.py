"""
NBA MVP Odds from FanDuel (Hardcoded)

Context:
Thomas wants to track NBA MVP odds similar to championship futures workflow.
The Odds API doesn't support MVP futures, so we hardcode FanDuel odds manually.

Usage:
    python3 scripts/fetch_nba_mvp_odds_fanduel.py

Output:
    data/01_input/fanduel/nba/mvp/nba_mvp_odds_YYYYMMDD_HHMMSS.csv

CSV columns:
    - bookmaker: Always 'fanduel'
    - player: Player name
    - odds: American odds (e.g., -450, +750)
    - implied_prob: Implied probability from odds
    - fetch_date: Date these odds were manually entered

How to update:
1. Go to FanDuel → NBA → Awards → MVP
2. Copy odds into CURRENT_MVP_ODDS dict below
3. Update FETCH_DATE to today's date
4. Run this script
"""

import pandas as pd
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for odds_utils
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from odds_utils import odds_to_implied_probability
from s3_utils import upload_df_to_s3


# =============================================================================
# MVP ODDS HISTORY (UPDATE MANUALLY FROM FANDUEL)
# =============================================================================
# Timestamped snapshots for tracking odds movement and comparing to previous weeks
# 
# How to update:
# 1. Go to https://sportsbook.fanduel.com/navigation/nba?tab=awards
# 2. Copy today's odds
# 3. Add a new dict to MVP_ODDS_HISTORY list below with today's date
# 4. Run this script to see comparison vs. season start and previous week

MVP_ODDS_HISTORY = [
    {
        'date': '20251021',
        'fetch_date': '2025-10-21',
        'label': 'Preseason',
        'odds': {
            'Shai Gilgeous-Alexander': +250,
            'Nikola Jokic': +220,
            'Luka Doncic': +380,
            'Giannis Antetokounmpo': +900,
            'Victor Wembanyama': +1500,
            'Anthony Edwards': +2000,
            'Cade Cunningham': +5000,
            'Jaylen Brown': +10000,  # Fixed typo: was "Jalen"
            'Jalen Brunson': +5000,
            'Tyrese Maxey': +8000,
            'Joel Embiid': +7000,
            'Kevin Durant': +8000,
            'Donovan Mitchell': +50000,
            'Stephen Curry': +50000,
            'Anthony Davis': +10000,
            'Ja Morant': +12000,
            'LeBron James': +15000,
            'Paolo Banchero': +10000,
            'Evan Mobley': +25000,
            'Pascal Siakam': +25000,
        }
    },
    {
        'date': '20260109',
        'fetch_date': '2026-01-09',
        'label': 'Week 14',
        'odds': {
            'Shai Gilgeous-Alexander': -450,
            'Luka Doncic': +750,
            'Cade Cunningham': +1800,
            'Jaylen Brown': +4000,
            'Jalen Brunson': +7500,
            'Anthony Edwards': +10000,
            'Tyrese Maxey': +10000,
            'Donovan Mitchell': +20000,
            'Kawhi Leonard': +100000,
            'Stephen Curry': +100000,
            'Alperen Sengun': +100000,
            'Kevin Durant': +100000,
        }
    },
    {
        'date': '20260121',
        'fetch_date': '2026-01-21',
        'label': 'Week 16',
        'odds': {
            'Shai Gilgeous-Alexander': -350,
            'Luka Doncic': +800,
            'Cade Cunningham': +1200,
            'Jaylen Brown': +2000,
            'Anthony Edwards': +5000,
            'Tyrese Maxey': +7500,
            'Jalen Brunson': +10000,
            'Donovan Mitchell': +20000,
            'Kawhi Leonard': +100000,
            'Stephen Curry': +100000,
            'Kevin Durant': +100000,
        }
    },
    {
        'date': '20260127',
        'fetch_date': '2026-01-27',
        'label': 'Week 17',
        'odds': {
            'Shai Gilgeous-Alexander': -400,
            'Luka Doncic': +1000,
            'Cade Cunningham': +1200,
            'Jaylen Brown': +2500,
            'Anthony Edwards': +5000,
            'Tyrese Maxey': +7500,
            'Jalen Brunson': +25000,
            'Donovan Mitchell': +40000,
            'Kawhi Leonard': +100000,
            'Stephen Curry': +100000,
            'Kevin Durant': +100000,
        }
    },
    {
        'date': '20260202',
        'fetch_date': '2026-02-02',
        'label': 'Week 18',
        'odds': {
            'Shai Gilgeous-Alexander': -320,
            'Nikola Jokic': +500,
            'Luka Doncic': +1400,
            'Cade Cunningham': +2700,
            'Jaylen Brown': +6000,
            'Anthony Edwards': +25000,
            'Tyrese Maxey': +25000,
            'Jalen Brunson': +30000,
            'Donovan Mitchell': +40000,
            'Kawhi Leonard': +100000,
            'Kevin Durant': +100000,
        }
    },
    {
        'date': '20260210',
        'fetch_date': '2026-02-10',
        'label': 'Week 19',
        'odds': {
            'Shai Gilgeous-Alexander': -220,
            'Nikola Jokic': +300,
            'Cade Cunningham': +2000,
            'Luka Doncic': +2000,
            'Jaylen Brown': +4000,
            'Anthony Edwards': +15000,
            'Jalen Brunson': +25000,
            'Tyrese Maxey': +50000,
            'Donovan Mitchell': +50000,
            'Kawhi Leonard': +100000,
            'Kevin Durant': +100000,
        }
    },
    {
        'date': '20260223',
        'fetch_date': '2026-02-23',
        'label': 'Week 20',
        'odds': {
            'Shai Gilgeous-Alexander': -145,
            'Nikola Jokic': +270,
            'Cade Cunningham': +650,
            'Victor Wembanyama': +2500,
            'Luka Doncic': +4000,
            'Jaylen Brown': +5000,
            'Donovan Mitchell': +10000,
            'Anthony Edwards': +25000,
            'Jalen Brunson': +50000,
            'Kawhi Leonard': +100000,
            'Tyrese Maxey': +100000,
            'Kevin Durant': +100000,
        }
    },
    {
        'date': '20260306',
        'fetch_date': '2026-03-06',
        'label': 'Week 21',
        'odds': {
            'Shai Gilgeous-Alexander': -320,
            'Nikola Jokic': +600,
            'Cade Cunningham': +1500,
            'Victor Wembanyama': +2000,
            'Jaylen Brown': +6000,
            'Luka Doncic': +10000,
            'Donovan Mitchell': +25000,
            'Jalen Brunson': +50000,
            'Anthony Edwards': +50000,
            'Kawhi Leonard': +100000,
            'Tyrese Maxey': +100000,
            'Kevin Durant': +100000,
        }
    }
]

# Helper functions to access odds history
def get_latest_odds():
    """Get the most recent MVP odds snapshot."""
    return MVP_ODDS_HISTORY[-1]

def get_first_odds():
    """Get the first MVP odds snapshot (season start)."""
    return MVP_ODDS_HISTORY[0]

def get_previous_odds():
    """Get the second-to-last MVP odds snapshot."""
    if len(MVP_ODDS_HISTORY) >= 2:
        return MVP_ODDS_HISTORY[-2]
    return None

# Current odds (always points to latest in history)
CURRENT_MVP_ODDS = get_latest_odds()['odds']
FETCH_DATE = get_latest_odds()['fetch_date']

# Season end date from FanDuel
SEASON_END_DATE = "Apr 14, 7:00pm ET"


# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================

def compare_odds(current_snapshot, previous_snapshot):
    """
    Compare two odds snapshots and return insights on movement.
    
    Args:
        current_snapshot: Dict with 'date', 'fetch_date', 'label', 'odds'
        previous_snapshot: Dict with 'date', 'fetch_date', 'label', 'odds'
    
    Returns:
        Dict with:
            - movers_up: List of (player, old_odds, new_odds)
            - movers_down: List of (player, old_odds, new_odds)
            - new_players: List of player names
            - removed_players: List of player names
    """
    current_odds = current_snapshot['odds']
    previous_odds = previous_snapshot['odds']
    
    movers_up = []      # Odds got better (favorites strengthened, longshots shortened)
    movers_down = []    # Odds got worse (favorites weakened, longshots lengthened)
    new_players = []
    removed_players = []
    
    # Find new and removed players
    current_players = set(current_odds.keys())
    previous_players = set(previous_odds.keys())
    
    new_players = sorted(list(current_players - previous_players))
    removed_players = sorted(list(previous_players - current_players))
    
    # Compare odds for players in both snapshots
    for player in current_players & previous_players:
        old_odds = previous_odds[player]
        new_odds = current_odds[player]
        
        if old_odds != new_odds:
            # Convert to implied probability to determine movement direction
            old_prob = odds_to_implied_probability(old_odds)
            new_prob = odds_to_implied_probability(new_odds)
            
            if new_prob > old_prob:
                # Player's chances improved
                movers_up.append((player, old_odds, new_odds))
            else:
                # Player's chances worsened
                movers_down.append((player, old_odds, new_odds))
    
    # Sort by magnitude of change (largest probability change first)
    movers_up.sort(key=lambda x: abs(odds_to_implied_probability(x[2]) - odds_to_implied_probability(x[1])), reverse=True)
    movers_down.sort(key=lambda x: abs(odds_to_implied_probability(x[2]) - odds_to_implied_probability(x[1])), reverse=True)
    
    return {
        'movers_up': movers_up,
        'movers_down': movers_down,
        'new_players': new_players,
        'removed_players': removed_players
    }


def display_odds_comparison(current_snapshot, previous_snapshot, comparison_label):
    """
    Display a formatted comparison between two odds snapshots.
    
    Args:
        current_snapshot: Current odds dict
        previous_snapshot: Previous odds dict
        comparison_label: String like "vs. Season Start" or "vs. Last Week"
    """
    comparison = compare_odds(current_snapshot, previous_snapshot)
    
    print("\n" + "="*80)
    print(f"📊 ODDS MOVEMENTS: {comparison_label}")
    print(f"   {previous_snapshot['label']} ({previous_snapshot['fetch_date']}) → {current_snapshot['label']} ({current_snapshot['fetch_date']})")
    print("="*80)
    
    # Movers up (improved chances)
    if comparison['movers_up']:
        print(f"\n📈 IMPROVED ODDS ({len(comparison['movers_up'])} players):")
        for player, old_odds, new_odds in comparison['movers_up'][:10]:  # Top 10
            old_prob = odds_to_implied_probability(old_odds) * 100
            new_prob = odds_to_implied_probability(new_odds) * 100
            change = new_prob - old_prob
            print(f"   ✅ {player:<30} {old_odds:>+7d} → {new_odds:>+7d}  ({old_prob:.1f}% → {new_prob:.1f}%, +{change:.1f}pp)")
    
    # Movers down (worsened chances)
    if comparison['movers_down']:
        print(f"\n📉 WORSENED ODDS ({len(comparison['movers_down'])} players):")
        for player, old_odds, new_odds in comparison['movers_down'][:10]:  # Top 10
            old_prob = odds_to_implied_probability(old_odds) * 100
            new_prob = odds_to_implied_probability(new_odds) * 100
            change = new_prob - old_prob
            print(f"   ❌ {player:<30} {old_odds:>+7d} → {new_odds:>+7d}  ({old_prob:.1f}% → {new_prob:.1f}%, {change:.1f}pp)")
    
    # New players
    if comparison['new_players']:
        print(f"\n🆕 NEW TO BOARD ({len(comparison['new_players'])} players):")
        for player in comparison['new_players']:
            odds = current_snapshot['odds'][player]
            prob = odds_to_implied_probability(odds) * 100
            print(f"   • {player:<30} {odds:>+7d}  ({prob:.1f}%)")
    
    # Removed players
    if comparison['removed_players']:
        print(f"\n🚫 REMOVED FROM BOARD ({len(comparison['removed_players'])} players):")
        for player in comparison['removed_players']:
            old_odds = previous_snapshot['odds'][player]
            print(f"   • {player:<30} was {old_odds:>+7d}")
    
    # If no changes
    if not any([comparison['movers_up'], comparison['movers_down'], 
                comparison['new_players'], comparison['removed_players']]):
        print("\n   ℹ️  No changes in odds")


# =============================================================================
# FUNCTIONS
# =============================================================================

def create_mvp_dataframe(odds_dict, fetch_date):
    """
    Create DataFrame from hardcoded MVP odds with full comparison to season start and last week.
    
    Uses a full outer join approach:
    - Players on current board but not at season start: season_start_odds = None (will show "NEW")
    - Players at season start but not on current board: odds = None (will show "-")
    
    Args:
        odds_dict: Dict of player -> odds (current)
        fetch_date: Date string (YYYY-MM-DD) when odds were fetched
    
    Returns:
        DataFrame with columns: bookmaker, player, odds, implied_prob, fetch_date, 
                                season_start_odds, last_week_odds
    """
    rows = []
    
    # Get season start odds for comparison
    season_start_snapshot = get_first_odds()
    season_start_odds_dict = season_start_snapshot['odds']
    
    # Get last week odds for comparison
    previous_snapshot = get_previous_odds()
    previous_odds_dict = previous_snapshot['odds'] if previous_snapshot else {}
    
    # Get all unique players from current, season start, and last week (full outer join)
    all_players = set(odds_dict.keys()) | set(season_start_odds_dict.keys()) | set(previous_odds_dict.keys())
    
    for player in all_players:
        current_odds = odds_dict.get(player, None)
        season_start_odds = season_start_odds_dict.get(player, None)
        last_week_odds = previous_odds_dict.get(player, None)
        
        # Calculate implied prob (only if player is on current board)
        if current_odds is not None:
            implied_prob = odds_to_implied_probability(current_odds)
        else:
            # Player was removed from board - set implied_prob to 0 so they sort to bottom
            implied_prob = 0.0
        
        rows.append({
            'bookmaker': 'fanduel',
            'player': player,
            'odds': current_odds,
            'implied_prob': implied_prob,
            'fetch_date': fetch_date,
            'season_start_odds': season_start_odds,
            'season_start_date': season_start_snapshot['fetch_date'],
            'last_week_odds': last_week_odds,
            'last_week_date': previous_snapshot['fetch_date'] if previous_snapshot else None
        })
    
    df = pd.DataFrame(rows)
    
    # Sort by implied probability descending, then by difference (for removed players)
    # This puts active players first (by current odds), then removed players by their drop magnitude
    df = df.sort_values(['implied_prob', 'season_start_odds'], ascending=[False, True]).reset_index(drop=True)
    
    return df


def save_mvp_odds(df, timestamp=None):
    """
    Save MVP odds to S3 with timestamp.
    
    Args:
        df: DataFrame with MVP odds
        timestamp: Optional timestamp string (YYYYMMDD_HHMMSS)
    
    Returns:
        S3 URI of saved file
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # S3 path
    bucket = 'nba-betting-mt'
    s3_key = f'data/01_input/fanduel/nba/mvp/nba_mvp_odds_{timestamp}.csv'
    
    # Upload to S3
    s3_uri = upload_df_to_s3(df, bucket, s3_key)
    
    return s3_uri


def main():
    """Main function"""
    print("="*80)
    print("NBA MVP ODDS FROM FANDUEL")
    print("="*80)
    
    current_snapshot = get_latest_odds()
    print(f"\n📅 Using Latest Snapshot: {current_snapshot['label']} ({current_snapshot['fetch_date']})")
    print(f"📊 Total Snapshots in History: {len(MVP_ODDS_HISTORY)}")
    
    print("\n" + "="*80)
    
    print(f"\n📅 Fetch Date: {FETCH_DATE}")
    print(f"📊 Season Ends: {SEASON_END_DATE}")
    print(f"🏀 Total Players: {len(CURRENT_MVP_ODDS)}")
    
    # Create DataFrame
    df = create_mvp_dataframe(CURRENT_MVP_ODDS, FETCH_DATE)
    
    # Calculate total market probability (vig)
    total_prob = df['implied_prob'].sum()
    vig_pct = (total_prob - 1.0) * 100
    
    print(f"\n📈 Market Vig: {vig_pct:.1f}%")
    print(f"   (Total implied probability: {total_prob:.4f})")
    
    # Display top candidates
    print("\n" + "="*80)
    print("TOP MVP CANDIDATES")
    print("="*80)
    
    print(f"\n{'Rank':<6}{'Player':<30}{'Odds':<10}{'Implied Prob':<15}")
    print("-" * 61)
    
    for i, row in enumerate(df.head(20).itertuples(), 1):  # Show more to include removed players
        # Handle None/NaN values for removed players
        if pd.notna(row.odds):
            odds_str = f"{int(row.odds):+d}"
            prob_str = f"{row.implied_prob*100:.1f}%"
        else:
            odds_str = "-"
            prob_str = "(removed)"
        print(f"{i:<6}{row.player:<30}{odds_str:<10}{prob_str:<15}")
    
    # Show odds comparisons
    current_snapshot = get_latest_odds()
    first_snapshot = get_first_odds()
    previous_snapshot = get_previous_odds()
    
    # Compare to season start (first observation)
    if first_snapshot and current_snapshot['date'] != first_snapshot['date']:
        display_odds_comparison(current_snapshot, first_snapshot, "vs. SEASON START")
    
    # Compare to previous week (n-1 observation)
    if previous_snapshot and current_snapshot['date'] != previous_snapshot['date']:
        display_odds_comparison(current_snapshot, previous_snapshot, "vs. PREVIOUS UPDATE")
    
    # Save to S3
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    s3_uri = save_mvp_odds(df, timestamp)
    
    print("\n" + "="*80)
    print("✅ SAVED TO S3")
    print("="*80)
    print(f"\n💾 Output: {s3_uri}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Run analysis to calculate fair odds and vig:")
    print("   python3 analysis/analyze_nba_mvp_vig.py")
    print("\n2. Generate visualization:")
    print("   python3 analysis/viz_nba_mvp_gt.py")
    print("\n3. To add new odds next time:")
    print("   - Go to: https://sportsbook.fanduel.com/navigation/nba?tab=awards")
    print("   - Add a new dict to MVP_ODDS_HISTORY in this file")
    print("   - Run the full workflow again")
    print("\n💡 TIP: Use the odds movements above for your tweet insights!")


if __name__ == "__main__":
    main()


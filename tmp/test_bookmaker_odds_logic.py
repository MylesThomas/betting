"""
Test Bookmaker Odds Logic

Investigates suspicious bookmaker_details where higher UNDER lines show better (+) odds.

Example from email:
    Books (2): BetRivers (21.5 @ -114), BetRivers (22.5 @ +110)
    
Expected logic for UNDER bets:
    - Higher line (22.5) = easier to hit = worse odds (more negative)
    - Lower line (21.5) = harder to hit = better odds (less negative/positive)
    
But we're seeing the opposite! Need to check:
    1. Are we pulling the correct side's odds? (under vs over)
    2. Is the API returning incorrect data?
    3. Is our parsing logic wrong?

Usage:
    cd betting
    python tmp/test_bookmaker_odds_logic.py

Author: Thomas Myles
Date: 2026-01-09
"""

import pandas as pd
import boto3
from io import StringIO
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def load_todays_plays_from_s3(date_str='2026-01-09', strategy='2d'):
    """Load today's plays from S3"""
    print(f"📥 Loading {strategy.upper()} plays for {date_str}...")
    
    s3 = boto3.client('s3')
    bucket = 'nba-betting-mt'
    key = f'data/04_output/plays/role_spread_points_model/{strategy}/{date_str}.csv'
    
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        print(f"   ✅ Loaded {len(df)} plays\n")
        return df
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        return None


def load_raw_props_from_s3(date_str='2026-01-09', season='2025-26'):
    """Load raw player props from The Odds API (before consensus/median)"""
    print(f"📥 Loading raw props for {date_str}...")
    
    s3 = boto3.client('s3')
    bucket = 'the-odds-api-mt'
    key = f'nba/historical_player_props/{season}/{date_str}.csv'
    
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        print(f"   ✅ Loaded {len(df)} raw prop records\n")
        return df
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        return None


def parse_bookmaker_details(bookmaker_details_json):
    """Parse bookmaker_details JSON string"""
    try:
        return json.loads(bookmaker_details_json)
    except:
        return []


def analyze_suspicious_play(df_plays, player_name):
    """Analyze a specific player's bookmaker details"""
    
    play = df_plays[df_plays['player'] == player_name]
    
    if play.empty:
        print(f"❌ Player '{player_name}' not found in plays")
        return
    
    play = play.iloc[0]
    
    print("="*80)
    print(f"🔍 ANALYZING: {play['player']}")
    print("="*80)
    print(f"Bet Side: {play['bet_side']}")
    print(f"Median Line: {play['line']}")
    print(f"Strategy: {play['strategy_name']}")
    print()
    
    # Parse bookmaker details
    details = parse_bookmaker_details(play['bookmaker_details'])
    
    if not details:
        print("❌ No bookmaker details found")
        return
    
    print(f"📊 BOOKMAKER DETAILS ({len(details)} entries):")
    print("-"*80)
    
    # Group by side
    over_details = [d for d in details if d.get('side', '').upper() == 'OVER']
    under_details = [d for d in details if d.get('side', '').upper() == 'UNDER']
    
    print(f"\n📈 OVER Odds ({len(over_details)} entries):")
    for i, book_data in enumerate(over_details, 1):
        bookmaker = book_data.get('bookmaker', 'Unknown')
        line = book_data.get('line', 'N/A')
        odds = book_data.get('odds', 'N/A')
        side = book_data.get('side', 'N/A')
        
        # Calculate distance from median
        if isinstance(line, (int, float)) and isinstance(play['line'], (int, float)):
            distance = line - play['line']
            distance_str = f"({distance:+.1f} from median)"
        else:
            distance_str = ""
        
        # Check logic for OVER
        if isinstance(odds, (int, float)):
            odds_str = f"{odds:+d}" if odds > 0 else f"{odds:d}"
        else:
            odds_str = str(odds)
        
        print(f"{i}. {bookmaker}: {line} @ {odds_str} {distance_str}")
    
    print(f"\n📉 UNDER Odds ({len(under_details)} entries):")
    for i, book_data in enumerate(under_details, 1):
        bookmaker = book_data.get('bookmaker', 'Unknown')
        line = book_data.get('line', 'N/A')
        odds = book_data.get('odds', 'N/A')
        side = book_data.get('side', 'N/A')
        
        # Calculate distance from median
        if isinstance(line, (int, float)) and isinstance(play['line'], (int, float)):
            distance = line - play['line']
            distance_str = f"({distance:+.1f} from median)"
        else:
            distance_str = ""
        
        # Check logic for UNDER: Higher line = easier = should have worse odds
        if isinstance(line, (int, float)) and line > play['line']:
            expected = "should be MORE negative (e.g., -120, -130)"
        elif isinstance(line, (int, float)) and line < play['line']:
            expected = "should be LESS negative or positive (e.g., -105, +100)"
        else:
            expected = "should be around -110"
        
        if isinstance(odds, (int, float)):
            odds_str = f"{odds:+d}" if odds > 0 else f"{odds:d}"
            # Flag suspicious odds
            if line > play['line'] and odds > 0:
                suspicious = "🚨 SUSPICIOUS - Higher line with plus odds!"
            else:
                suspicious = "✅"
        else:
            odds_str = str(odds)
            suspicious = ""
        
        print(f"{i}. {bookmaker}: {line} @ {odds_str} {distance_str} {suspicious}")
        print(f"   Expected: {expected}")
        print()
    
    # Show which side we're actually betting
    print("-"*80)
    print(f"🎯 OUR BET: {play['bet_side']} {play['line']}")
    print(f"   We should only see {play['bet_side']} odds in the email!")
    print()
    
    return play, details


def check_raw_props_for_player(df_raw, player_name):
    """Check raw props data from The Odds API for this player"""
    
    if df_raw is None:
        print("⚠️  No raw props data available")
        return
    
    player_props = df_raw[df_raw['PLAYER_NAME'] == player_name]
    
    if player_props.empty:
        print(f"❌ '{player_name}' not found in raw props")
        return
    
    print("="*80)
    print(f"📋 RAW ODDS API DATA for {player_name}")
    print("="*80)
    print(f"Total records: {len(player_props)}")
    print()
    
    # Group by bookmaker
    for bookmaker in player_props['bookmaker'].unique():
        book_props = player_props[player_props['bookmaker'] == bookmaker]
        print(f"📚 {bookmaker} ({len(book_props)} records):")
        
        for _, row in book_props.iterrows():
            line = row.get('points_line', 'N/A')
            odds = row.get('odds', 'N/A')
            
            if isinstance(odds, (int, float)):
                odds_str = f"{odds:+.0f}" if odds > 0 else f"{odds:.0f}"
            else:
                odds_str = str(odds)
            
            print(f"   Line {line} @ {odds_str}")
        print()


def main():
    """Main test function"""
    
    print("\n" + "="*80)
    print("🧪 BOOKMAKER ODDS LOGIC TEST")
    print("="*80)
    print("Investigating: Why do higher UNDER lines show plus odds?")
    print()
    
    # Configuration
    date_str = '2026-01-09'
    season = '2025-26'
    test_player = 'Trey Murphy III'  # From the example
    strategy = '2d'
    
    # Load today's plays
    df_plays = load_todays_plays_from_s3(date_str, strategy)
    
    if df_plays is None or df_plays.empty:
        print("❌ No plays data - cannot proceed")
        return
    
    # Show all players in plays
    print("👥 PLAYERS IN TODAY'S PLAYS:")
    print("-"*80)
    for i, player in enumerate(df_plays['player'].unique(), 1):
        bet_side = df_plays[df_plays['player'] == player]['bet_side'].iloc[0]
        line = df_plays[df_plays['player'] == player]['line'].iloc[0]
        print(f"{i}. {player} - {bet_side} {line}")
    print()
    
    # Analyze the suspicious play
    if test_player in df_plays['player'].values:
        play, details = analyze_suspicious_play(df_plays, test_player)
    else:
        print(f"⚠️  '{test_player}' not in today's plays, analyzing first player instead...")
        test_player = df_plays['player'].iloc[0]
        play, details = analyze_suspicious_play(df_plays, test_player)
    
    # Load and check raw props
    df_raw = load_raw_props_from_s3(date_str, season)
    check_raw_props_for_player(df_raw, test_player)
    
    # Summary
    print("="*80)
    print("💡 DIAGNOSIS")
    print("="*80)
    print()
    print("Possible Issues:")
    print("1. ❓ Are we pulling OVER odds instead of UNDER odds?")
    print("2. ❓ Is The Odds API returning the wrong side's odds?")
    print("3. ❓ Is our median calculation causing the issue?")
    print("4. ❓ Are both sides (over/under) being mixed in bookmaker_details?")
    print()
    print("Next Steps:")
    print("- Check the raw CSV to see which 'outcome' field each row has")
    print("- Verify we're filtering for the correct outcome (Over vs Under)")
    print("- Check if BetRivers actually offers both lines or if it's a duplicate")
    print()


if __name__ == '__main__':
    main()


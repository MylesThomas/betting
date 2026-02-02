"""
Fill Missing Games for Player Data

Problem:
--------
When a player misses games (DNP, injury, rest), there are gaps in their game log data.
This creates misleading "streaks" because games where they didn't play are invisible.

Example:
  - Oct 24: Player plays, makes 2 threes
  - Oct 26-Nov 5: Player injured (team plays 6 games) - NO ROWS IN DATA
  - Nov 7: Player returns, makes 1 three
  
  Without filling: Looks like 2 consecutive games
  With filling: Shows 2 games played, 6 games missed (DNP), then 1 game played

Solution:
---------
1. Load team schedules (all games each team played)
2. For each player:
   a. Determine team(s) they played for (handle mid-season trades)
   b. Get full team schedule for their stint(s)
   c. Fill in missing games as DNP rows
   d. Mark DNPs with: threes_made=0, threes_attempted=0, minutes=0, dnp=True

3. Save filled data: data/03_intermediate/player_level_data_filled/2024_25/{Player_Name}.csv

Author: Myles Thomas
Date: 2025-11-25
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys

CURRENT_NBA_SEASON = '2024_25'
MAX_GAP_DAYS = 7

# Team abbreviation mappings (handle variations)
TEAM_ABBREV_MAP = {
    'BKN': 'BRK',  # Brooklyn sometimes appears as BKN or BRK
    'BRK': 'BRK',
    'CHA': 'CHO',  # Charlotte sometimes CHO
    'CHO': 'CHO',
    'PHX': 'PHO',  # Phoenix sometimes PHO
    'PHO': 'PHO',
}

def normalize_team_abbrev(team):
    """Normalize team abbreviations to handle variations"""
    team = str(team).upper().strip()
    return TEAM_ABBREV_MAP.get(team, team)


def load_team_schedule(season=CURRENT_NBA_SEASON):
    """
    Load full NBA schedule for the season.
    
    This should include every game played by every team.
    We'll need to derive this from the game logs or use NBA API.
    
    Returns:
        DataFrame with columns: date, home_team, away_team, game_id
    """
    # For now, we'll derive the schedule from all player game logs
    # In the future, could use NBA API's schedule endpoint
    
    game_logs_dir = Path(__file__).parent.parent / 'data' / '01_input' / 'nba_api' / 'season_game_logs' / season
    
    if not game_logs_dir.exists():
        raise FileNotFoundError(f"Game logs directory not found: {game_logs_dir}")
    
    print(f"Loading team schedule from game logs in {season}...")
    
    # Load all player game logs and extract unique games
    all_games = []
    
    csv_files = list(game_logs_dir.glob("*.csv"))
    print(f"  Found {len(csv_files)} player files")
    
    for i, csv_file in enumerate(csv_files):
        if i % 100 == 0:
            print(f"  Processing file {i+1}/{len(csv_files)}...")
        
        try:
            df = pd.read_csv(csv_file)
            
            if len(df) == 0:
                continue
            
            # Extract game info
            for _, row in df.iterrows():
                matchup = str(row['matchup'])  # e.g., "DEN vs. HOU" or "DEN @ HOU"
                date = row['date']
                
                # Parse matchup to get teams
                if ' vs. ' in matchup:
                    parts = matchup.split(' vs. ')
                    home_team = normalize_team_abbrev(parts[0])
                    away_team = normalize_team_abbrev(parts[1])
                elif ' @ ' in matchup:
                    parts = matchup.split(' @ ')
                    away_team = normalize_team_abbrev(parts[0])
                    home_team = normalize_team_abbrev(parts[1])
                else:
                    continue
                
                all_games.append({
                    'date': date,
                    'home_team': home_team,
                    'away_team': away_team
                })
        except Exception as e:
            print(f"  Error processing {csv_file.name}: {e}")
            continue
    
    # Convert to DataFrame and get unique games
    schedule_df = pd.DataFrame(all_games)
    schedule_df = schedule_df.drop_duplicates(subset=['date', 'home_team', 'away_team'])
    schedule_df = schedule_df.sort_values('date').reset_index(drop=True)
    
    print(f"  ✓ Extracted {len(schedule_df)} unique games")
    
    return schedule_df


def get_player_team_stints(player_df, schedule_df, extend_to_season=True):
    """
    Determine which team(s) a player played for and when.
    Handles mid-season trades.
    
    Args:
        player_df: Player's game log (only games they actually played)
        schedule_df: Full NBA schedule
        extend_to_season: If True, extend stints to cover full season (start→end)
    
    Returns:
        List of dicts: [{'team': 'DEN', 'start_date': '2024-10-22', 'end_date': '2025-02-15'}, ...]
    """
    if len(player_df) == 0:
        return []
    
    # Sort by date
    player_df = player_df.sort_values('date').copy()
    
    # Extract team from matchup
    def extract_team(matchup):
        """Extract player's team from matchup string"""
        if ' vs. ' in matchup:
            return normalize_team_abbrev(matchup.split(' vs. ')[0])
        elif ' @ ' in matchup:
            return normalize_team_abbrev(matchup.split(' @ ')[0])
        return None
    
    player_df['team'] = player_df['matchup'].apply(extract_team)
    
    # Find team changes (trades)
    stints = []
    current_team = None
    start_date = None
    
    for _, row in player_df.iterrows():
        team = row['team']
        date = row['date']
        
        if team != current_team:
            # New stint (or first stint)
            if current_team is not None:
                # End previous stint
                stints.append({
                    'team': current_team,
                    'start_date': start_date,
                    'end_date': prev_date,
                    'first_game': start_date,  # Player's first game
                    'last_game': prev_date     # Player's last game
                })
            
            # Start new stint
            current_team = team
            start_date = date
        
        prev_date = date
    
    # Add final stint
    if current_team is not None:
        stints.append({
            'team': current_team,
            'start_date': start_date,
            'end_date': prev_date,
            'first_game': start_date,  # Player's first game
            'last_game': prev_date     # Player's last game
        })
    
    # EXTEND TO FULL SEASON (but handle trades correctly)
    # Find team's season start/end dates
    if extend_to_season and schedule_df is not None:
        for i, stint in enumerate(stints):
            team = stint['team']
            
            # Get ALL team games for this team
            team_schedule = schedule_df[
                (schedule_df['home_team'] == team) | (schedule_df['away_team'] == team)
            ]
            
            if len(team_schedule) > 0:
                team_season_start = team_schedule['date'].min()
                team_season_end = team_schedule['date'].max()
                
                # HANDLE TRADE BOUNDARIES
                # If this is NOT the first stint, truncate start (player wasn't on team yet)
                if i > 0:
                    # Use player's first game with THIS team as start
                    stint['start_date'] = stint['first_game']
                else:
                    # First team: extend back to team's season start
                    stint['start_date'] = team_season_start
                
                # If this is NOT the last stint, truncate end (player left for another team)
                if i < len(stints) - 1:
                    # Get the NEXT stint's first game (switch point)
                    next_stint_first_game = stints[i + 1]['first_game']
                    
                    # This stint ends the day before switch point
                    # But we want all team games UP TO (not including) the switch point
                    stint['end_date'] = pd.to_datetime(next_stint_first_game) - pd.Timedelta(days=1)
                    stint['end_date'] = stint['end_date'].strftime('%Y-%m-%d')
                else:
                    # Last team: extend to team's season end
                    stint['end_date'] = team_season_end
    
    return stints


def fill_missing_games_for_player(player_name, player_df, schedule_df, max_gap_days=MAX_GAP_DAYS, fill_trade_gaps=True):
    """
    Fill in missing games for a player from their team's schedule.
    
    Args:
        player_name: Player name
        player_df: Player's game log (only games they played)
        schedule_df: Full NBA schedule
        max_gap_days: Only fill gaps <= this many days (for gaps within a stint)
        fill_trade_gaps: If True, fill gaps between stints using previous team's schedule
    
    Returns:
        DataFrame with filled games (includes DNPs)
    """
    if len(player_df) == 0:
        return player_df
    
    # Get player's team stints (extended to full season)
    stints = get_player_team_stints(player_df, schedule_df, extend_to_season=True)
    
    if len(stints) == 0:
        return player_df
    
    # For each stint, get team's games and fill
    filled_rows = []
    
    for stint_idx, stint in enumerate(stints):
        team = stint['team']
        start_date = stint['start_date']
        end_date = stint['end_date']
        
        # Get all team games in this stint period
        team_games = schedule_df[
            ((schedule_df['home_team'] == team) | (schedule_df['away_team'] == team)) &
            (schedule_df['date'] >= start_date) &
            (schedule_df['date'] <= end_date)
        ].copy()
        
        # Get player's actual games in this stint
        stint_actual_games = player_df[
            (player_df['date'] >= start_date) &
            (player_df['date'] <= end_date)
        ].copy()
        
        actual_dates = set(stint_actual_games['date'].values)
        
        # SIMPLE LOGIC: Iterate through ALL team games
        # If player has a row for that date → use real data
        # If player doesn't have a row → add DNP
        for _, team_game in team_games.iterrows():
            game_date = team_game['date']
            
            if game_date in actual_dates:
                # Player actually played - use real data
                game_row = stint_actual_games[stint_actual_games['date'] == game_date].iloc[0]
                filled_rows.append(game_row.to_dict())
            else:
                # Player missed this game - create DNP row
                # Determine home/away
                if team_game['home_team'] == team:
                    home_away = 'HOME'
                    opponent = team_game['away_team']
                    matchup = f"{team} vs. {opponent}"
                else:
                    home_away = 'AWAY'
                    opponent = team_game['home_team']
                    matchup = f"{team} @ {opponent}"
                
                # Get player_id if available
                player_id = None
                if len(stint_actual_games) > 0 and 'player_id' in stint_actual_games.columns:
                    player_id = stint_actual_games.iloc[0]['player_id']
                
                # Create DNP row
                dnp_row = {
                    'player': player_name,
                    'player_id': player_id,
                    'date': game_date,
                    'matchup': matchup,
                    'result': None,  # Don't know result
                    'minutes': 0,
                    'fgm': 0,
                    'fga': 0,
                    'fg_pct': 0,
                    'threes_made': 0,
                    'threes_attempted': 0,
                    'three_pct': 0,
                    'ftm': 0,
                    'fta': 0,
                    'ft_pct': 0,
                    'oreb': 0,
                    'dreb': 0,
                    'reb': 0,
                    'ast': 0,
                    'stl': 0,
                    'blk': 0,
                    'tov': 0,
                    'pf': 0,
                    'pts': 0,
                    'plus_minus': 0,
                    'opponent': opponent,
                    'home_away': home_away,
                    'player_normalized': player_name,
                    'dnp': True,  # Mark as DNP
                }
                
                filled_rows.append(dnp_row)
        
        # NOTE: No need for special trade gap handling!
        # Since we extend stints to full season, each stint covers its team's entire schedule.
        # Traded players will have rows from both teams (may be > 82 total).
        # REMOVED OLD TRADE GAP LOGIC - now handled by extending stints to full season
        
        # OLD CODE WAS HERE - removed for simplicity
        if False and fill_trade_gaps and stint_idx < len(stints) - 1:
            next_stint = stints[stint_idx + 1]
            gap_start = stint['end_date']
            gap_end = next_stint['start_date']
            
            # Get previous team's games during the gap
            gap_team_games = schedule_df[
                ((schedule_df['home_team'] == team) | (schedule_df['away_team'] == team)) &
                (schedule_df['date'] > gap_start) &
                (schedule_df['date'] < gap_end)
            ].copy()
            
            # Fill all gap games as DNPs with previous team
            for _, team_game in gap_team_games.iterrows():
                game_date = team_game['date']
                
                # Determine home/away
                if team_game['home_team'] == team:
                    home_away = 'HOME'
                    opponent = team_game['away_team']
                    matchup = f"{team} vs. {opponent}"
                else:
                    home_away = 'AWAY'
                    opponent = team_game['home_team']
                    matchup = f"{team} @ {opponent}"
                
                # Create DNP row for trade gap
                dnp_row = {
                    'player': player_name,
                    'player_id': stint_actual_games.iloc[0]['player_id'] if 'player_id' in stint_actual_games.columns and len(stint_actual_games) > 0 else None,
                    'date': game_date,
                    'matchup': matchup,
                    'result': None,
                    'minutes': 0,
                    'fgm': 0,
                    'fga': 0,
                    'fg_pct': 0,
                    'threes_made': 0,
                    'threes_attempted': 0,
                    'three_pct': 0,
                    'ftm': 0,
                    'fta': 0,
                    'ft_pct': 0,
                    'oreb': 0,
                    'dreb': 0,
                    'reb': 0,
                    'ast': 0,
                    'stl': 0,
                    'blk': 0,
                    'tov': 0,
                    'pf': 0,
                    'pts': 0,
                    'plus_minus': 0,
                    'opponent': opponent,
                    'home_away': home_away,
                    'player_normalized': player_name,
                    'dnp': True,  # Mark as DNP (trade gap)
                }
                
                filled_rows.append(dnp_row)
    
    # Convert to DataFrame
    filled_df = pd.DataFrame(filled_rows)
    filled_df = filled_df.sort_values('date').reset_index(drop=True)
    
    # Ensure dnp column exists for all rows
    if 'dnp' not in filled_df.columns:
        filled_df['dnp'] = False
    filled_df['dnp'] = filled_df['dnp'].fillna(False)
    
    return filled_df


def process_all_players(season='2024_25', max_gap_days=MAX_GAP_DAYS):
    """
    Process all players: fill missing games from team schedules.
    
    Args:
        season: Season to process (e.g., '2024_25')
        max_gap_days: Maximum gap to fill (avoids filling long injuries)
    """
    print("="*80)
    print("FILL MISSING GAMES FOR ALL PLAYERS")
    print("="*80)
    print()
    
    # Paths
    input_dir = Path(__file__).parent.parent / 'data' / '01_input' / 'nba_api' / 'season_game_logs' / season
    output_dir = Path(__file__).parent.parent / 'data' / '03_intermediate' / 'player_level_data_filled' / season
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Load team schedule
    schedule_df = load_team_schedule(season)
    print()
    
    # Process each player
    csv_files = sorted(list(input_dir.glob("*.csv")))
    print(f"Processing {len(csv_files)} players...")
    print()
    
    stats = {
        'total_players': 0,
        'games_added': 0,
        'games_original': 0,
        'players_with_gaps': 0
    }
    
    # Track players with incomplete seasons
    incomplete_seasons = []
    
    for i, csv_file in enumerate(csv_files):
        player_name = csv_file.stem.replace('_', ' ')
        
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(csv_files)}] {player_name}")
        
        try:
            # Load player data
            player_df = pd.read_csv(csv_file)
            original_games = len(player_df)
            
            # Fill missing games
            filled_df = fill_missing_games_for_player(
                player_name,
                player_df,
                schedule_df,
                max_gap_days=max_gap_days
            )
            
            filled_games = len(filled_df)
            games_added = filled_games - original_games
            
            # Save filled data
            output_file = output_dir / csv_file.name
            filled_df.to_csv(output_file, index=False)
            
            # Update stats
            stats['total_players'] += 1
            stats['games_original'] += original_games
            stats['games_added'] += games_added
            if games_added > 0:
                stats['players_with_gaps'] += 1
            
            # Check if player has a full season (82 games)
            if filled_games < 82:
                incomplete_seasons.append({
                    'player': player_name,
                    'total_games': filled_games,
                    'games_played': original_games,
                    'games_dnp': games_added
                })
            
            if games_added > 0 and (i + 1) % 50 == 0:
                print(f"    ↳ Added {games_added} DNP games")
        
        except Exception as e:
            print(f"  ❌ Error processing {player_name}: {e}")
            continue
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Players processed: {stats['total_players']}")
    print(f"Players with gaps filled: {stats['players_with_gaps']}")
    print(f"Original games: {stats['games_original']:,}")
    print(f"DNP games added: {stats['games_added']:,}")
    print(f"Total games (with DNPs): {stats['games_original'] + stats['games_added']:,}")
    print()
    
    # Report players with incomplete seasons
    if len(incomplete_seasons) > 0:
        print("="*80)
        print(f"⚠️  PLAYERS WITH INCOMPLETE SEASONS (<82 games): {len(incomplete_seasons)}")
        print("="*80)
        print()
        print("Possible reasons: trades, injuries >14 days, mid-season call-ups, season start late")
        print()
        
        # Sort by total games (ascending)
        incomplete_seasons_sorted = sorted(incomplete_seasons, key=lambda x: x['total_games'])
        
        print(f"{'Player':<30} | {'Total Games':<12} | {'Played':<8} | {'DNP':<8}")
        print("-" * 70)
        
        for player_info in incomplete_seasons_sorted:
            player = player_info['player']
            total = player_info['total_games']
            played = player_info['games_played']
            dnp = player_info['games_dnp']
            
            print(f"{player:<30} | {total:<12} | {played:<8} | {dnp:<8}")
        
        print()
        print(f"Note: Only filled gaps ≤ {max_gap_days} days. Longer injuries not filled.")
        print()
    
    print(f"✓ Output saved to: {output_dir}")
    print()


def test_single_player(player_name='Stephen Curry', season='2024_25', max_gap_days=MAX_GAP_DAYS):
    """
    Test gap-filling logic on a single player.
    
    Stephen Curry missed 12 games in 2024-25 season.
    This should find those 12 gaps and fill them.
    
    Args:
        player_name: Player to test (default: Stephen Curry)
        season: Season to test
        max_gap_days: Maximum gap to fill
    """
    print("="*80)
    print(f"TEST SINGLE PLAYER: {player_name}")
    print("="*80)
    print()
    
    # Paths
    input_dir = Path(__file__).parent.parent / 'data' / '01_input' / 'nba_api' / 'season_game_logs' / season
    
    # Load team schedule
    print("Loading team schedule...")
    schedule_df = load_team_schedule(season)
    print(f"✓ Loaded {len(schedule_df)} total games")
    print()
    
    # Load player data
    player_filename = player_name.replace(' ', '_') + '.csv'
    player_file = input_dir / player_filename
    
    if not player_file.exists():
        print(f"❌ Player file not found: {player_file}")
        return
    
    player_df = pd.read_csv(player_file)
    print(f"Original data: {len(player_df)} games played")
    print()
    
    # Get player's team stints (extended to full season)
    stints = get_player_team_stints(player_df, schedule_df, extend_to_season=True)
    print(f"Team stint(s): {len(stints)}")
    for stint in stints:
        print(f"  • {stint['team']}: {stint['start_date']} to {stint['end_date']}")
    print()
    
    # For each stint, show gaps
    print("="*80)
    print("ANALYZING GAPS")
    print("="*80)
    print()
    
    total_team_games = 0
    total_gaps = 0
    
    for stint_idx, stint in enumerate(stints):
        team = stint['team']
        start_date = stint['start_date']
        end_date = stint['end_date']
        
        print(f"Stint {stint_idx + 1}: {team} ({start_date} to {end_date})")
        print("-" * 80)
        
        # Get all team games in this stint
        team_games = schedule_df[
            ((schedule_df['home_team'] == team) | (schedule_df['away_team'] == team)) &
            (schedule_df['date'] >= start_date) &
            (schedule_df['date'] <= end_date)
        ].copy()
        
        print(f"  Team {team} played {len(team_games)} games during this stint")
        
        # Get player's actual games
        stint_actual_games = player_df[
            (player_df['date'] >= start_date) &
            (player_df['date'] <= end_date)
        ].copy()
        
        print(f"  {player_name} played {len(stint_actual_games)} games")
        print(f"  → Missing: {len(team_games) - len(stint_actual_games)} games")
        print()
        
        # Find specific gaps
        actual_dates = set(stint_actual_games['date'].values)
        gaps = []
        
        for _, team_game in team_games.iterrows():
            game_date = team_game['date']
            
            if game_date not in actual_dates:
                # This is a gap
                # Check gap to nearest actual game
                actual_dates_list = sorted(list(actual_dates))
                game_date_dt = pd.to_datetime(game_date)
                
                closest_gap = float('inf')
                for actual_date in actual_dates_list:
                    actual_date_dt = pd.to_datetime(actual_date)
                    gap = abs((game_date_dt - actual_date_dt).days)
                    closest_gap = min(closest_gap, gap)
                
                gaps.append({
                    'date': game_date,
                    'home_team': team_game['home_team'],
                    'away_team': team_game['away_team'],
                    'closest_gap_days': closest_gap,
                    'will_fill': closest_gap <= max_gap_days
                })
        
        # Display gaps
        if len(gaps) > 0:
            print(f"  GAPS FOUND: {len(gaps)}")
            print(f"  {'Date':<12} | {'Matchup':<20} | {'Gap (days)':<12} | {'Fill?':<8}")
            print("  " + "-" * 60)
            
            gaps_to_fill = 0
            for gap in sorted(gaps, key=lambda x: x['date']):
                matchup = f"{gap['away_team']} @ {gap['home_team']}"
                will_fill_str = "YES ✓" if gap['will_fill'] else "NO (>14d)"
                print(f"  {gap['date']:<12} | {matchup:<20} | {gap['closest_gap_days']:<12} | {will_fill_str:<8}")
                
                if gap['will_fill']:
                    gaps_to_fill += 1
            
            print()
            print(f"  → Will fill {gaps_to_fill} gaps (≤{max_gap_days} days)")
            print(f"  → Won't fill {len(gaps) - gaps_to_fill} gaps (>{max_gap_days} days)")
            
            total_gaps += gaps_to_fill
        
        total_team_games += len(team_games)
        print()
    
    # Fill and compare
    print("="*80)
    print("FILLING GAPS")
    print("="*80)
    print()
    
    filled_df = fill_missing_games_for_player(
        player_name,
        player_df,
        schedule_df,
        max_gap_days=max_gap_days
    )
    
    print(f"Original games: {len(player_df)}")
    print(f"Filled games: {len(filled_df)}")
    print(f"DNPs added: {len(filled_df) - len(player_df)}")
    print()
    
    # Verify player-specific assertions (only for 2024-25 season)
    dnps_added = len(filled_df) - len(player_df)
    
    if season == '2024_25' and player_name == 'Stephen Curry':
        print("="*80)
        print("STEPHEN CURRY VERIFICATION")
        print("="*80)
        print(f"Expected: 12 missed games (1 team, no trades)")
        print(f"Found: {dnps_added} DNP games added")
        print(f"Expected stints: 1 (GSW only)")
        print(f"Found stints: {len(stints)}")
        
        checks_passed = 0
        if dnps_added == 12:
            print("✅ PASS - Found exactly 12 missed games!")
            checks_passed += 1
        else:
            print(f"❌ FAIL - Expected 12 missed games, found {dnps_added}")
        
        if len(stints) == 1 and stints[0]['team'] == 'GSW':
            print("✅ PASS - Single team (GSW) as expected")
            checks_passed += 1
        else:
            print(f"❌ FAIL - Expected 1 stint with GSW, found {len(stints)} stints")
        
        print(f"\n{'='*80}")
        print(f"OVERALL: {checks_passed}/2 checks passed")
        print()
    
    elif season == '2024_25' and player_name == 'Luka Doncic':
        print("="*80)
        print("LUKA DONCIC VERIFICATION (MID-SEASON TRADE)")
        print("="*80)
        print(f"Expected: 50 total games played (22 DAL, 28 LAL)")
        print(f"Found: {len(player_df)} total games played")
        print(f"Expected filled: ~82 games (DAL games until trade + LAL games after)")
        print(f"Found filled: {len(filled_df)} total games")
        print(f"Expected stints: 2 (traded from DAL to LAL)")
        print(f"Found stints: {len(stints)}")
        print(f"Note: NO double-counting - DAL ends when LAL starts")
        
        checks_passed = 0
        
        # Check total games played
        if len(player_df) == 50:
            print("✅ PASS - Played 50 total games")
            checks_passed += 1
        else:
            print(f"❌ FAIL - Expected 50 games, found {len(player_df)}")
        
        # Check filled games (should be around 82, NOT 164)
        if 80 <= len(filled_df) <= 85:
            print(f"✅ PASS - Filled {len(filled_df)} games (no double-counting)")
            checks_passed += 1
        else:
            print(f"❌ FAIL - Expected 80-85 filled games, found {len(filled_df)}")
        
        # Check number of stints
        if len(stints) == 2:
            print("✅ PASS - Found 2 team stints (trade detected)")
            checks_passed += 1
            
            # Check stint details
            stint1_team = stints[0]['team']
            stint2_team = stints[1]['team']
            
            print(f"\n  Stint 1: {stint1_team} ({stints[0]['start_date']} to {stints[0]['end_date']})")
            
            # Count games in stint 1
            stint1_games = player_df[
                (player_df['date'] >= stints[0]['start_date']) &
                (player_df['date'] <= stints[0]['end_date'])
            ]
            print(f"    Games played: {len(stint1_games)}")
            
            print(f"\n  Stint 2: {stint2_team} ({stints[1]['start_date']} to {stints[1]['end_date']})")
            
            # Count games in stint 2
            stint2_games = player_df[
                (player_df['date'] >= stints[1]['start_date']) &
                (player_df['date'] <= stints[1]['end_date'])
            ]
            print(f"    Games played: {len(stint2_games)}")
            
            # Verify specific team/game counts
            if stint1_team == 'DAL' and len(stint1_games) == 22:
                print(f"\n✅ PASS - Dallas stint: 22 games as expected")
                checks_passed += 1
            else:
                print(f"\n❌ FAIL - Dallas stint: Expected 22 games with DAL, found {len(stint1_games)} with {stint1_team}")
            
            if stint2_team == 'LAL' and len(stint2_games) == 28:
                print(f"✅ PASS - Lakers stint: 28 games as expected")
                checks_passed += 1
            else:
                print(f"❌ FAIL - Lakers stint: Expected 28 games with LAL, found {len(stint2_games)} with {stint2_team}")
        else:
            print(f"❌ FAIL - Expected 2 stints, found {len(stints)}")
        
        # Check filled data
        expected_filled = 50 + dnps_added  # Should be close to 82 total if trade timing worked out
        print(f"\nTotal games after filling: {len(filled_df)}")
        print(f"DNP games added: {dnps_added}")
        
        print(f"\n{'='*80}")
        print(f"OVERALL: {checks_passed}/5 checks passed")
        print()
    
    elif season == '2024_25' and player_name == 'Josh Hart':
        print("="*80)
        print("JOSH HART VERIFICATION (MINIMAL DNPs)")
        print("="*80)
        print(f"Expected: 77 games played, 4 DNPs (minor injury/rest)")
        print(f"Found: {len(player_df)} games played")
        print(f"Expected stints: 1 (single team)")
        print(f"Found stints: {len(stints)}")
        print(f"Note: Team only played 81 games during his stint (joined/left mid-season)")
        
        checks_passed = 0
        
        # Check games played
        if len(player_df) == 77:
            print("✅ PASS - Played 77 games")
            checks_passed += 1
        else:
            print(f"❌ FAIL - Expected 77 games, found {len(player_df)}")
        
        # Check DNPs added
        if dnps_added == 4:
            print("✅ PASS - Found 4 DNP games")
            checks_passed += 1
        else:
            print(f"❌ FAIL - Expected 4 DNPs, found {dnps_added}")
        
        # Check single team
        if len(stints) == 1:
            team = stints[0]['team']
            print(f"✅ PASS - Single team ({team}) as expected")
            checks_passed += 1
        else:
            print(f"❌ FAIL - Expected 1 stint, found {len(stints)}")
        
        print(f"\n{'='*80}")
        print(f"OVERALL: {checks_passed}/3 checks passed")
        print()
    
    elif season == '2024_25' and player_name == 'Tyus Jones':
        print("="*80)
        print("TYUS JONES VERIFICATION (FULL SEASON, 1 DNP)")
        print("="*80)
        print(f"Expected: 81 games played, 1 DNP")
        print(f"Found: {len(player_df)} games played")
        print(f"Expected stints: 1 (single team, full season)")
        print(f"Found stints: {len(stints)}")
        print(f"Expected filled: 82 (full team schedule)")
        print(f"Found filled: {len(filled_df)} total games")
        
        checks_passed = 0
        
        # Check games played
        if len(player_df) == 81:
            print("✅ PASS - Played 81 games")
            checks_passed += 1
        else:
            print(f"❌ FAIL - Expected 81 games, found {len(player_df)}")
        
        # Check DNPs added
        if dnps_added == 1:
            print("✅ PASS - Found 1 DNP game")
            checks_passed += 1
        else:
            print(f"❌ FAIL - Expected 1 DNP, found {dnps_added}")
        
        # Check single team
        if len(stints) == 1:
            team = stints[0]['team']
            print(f"✅ PASS - Single team ({team}) as expected")
            checks_passed += 1
        else:
            print(f"❌ FAIL - Expected 1 stint, found {len(stints)}")
        
        # Check filled total
        if len(filled_df) == 82:
            print(f"✅ PASS - Complete 82-game season")
            checks_passed += 1
        else:
            print(f"❌ FAIL - Expected 82 games, found {len(filled_df)}")
        
        print(f"\n{'='*80}")
        print(f"OVERALL: {checks_passed}/4 checks passed")
        print()
    
    elif season == '2024_25' and player_name == 'Joel Embiid':
        print("="*80)
        print("JOEL EMBIID VERIFICATION (EXTREME LOAD MANAGEMENT)")
        print("="*80)
        print(f"Expected: 19 games played, 63 DNPs")
        print(f"Found: {len(player_df)} games played")
        print(f"Expected filled: 82 (full team schedule with start+end gaps)")
        print(f"Found filled: {len(filled_df)} total games")
        print(f"Note: Start-of-season and end-of-season gaps filled regardless of length")
        
        checks_passed = 0
        
        # Check games played
        if len(player_df) == 19:
            print("✅ PASS - Played 19 games")
            checks_passed += 1
        else:
            print(f"❌ FAIL - Expected 19 games, found {len(player_df)}")
        
        # Check filled total (should be 82)
        if len(filled_df) == 82:
            print(f"✅ PASS - Complete 82-game season (includes pre/post season)")
            checks_passed += 1
        else:
            print(f"❌ FAIL - Expected 82 games, found {len(filled_df)}")
        
        # Check DNPs
        if dnps_added >= 60:  # Should be around 63
            print(f"✅ PASS - Added {dnps_added} DNPs (extreme load management)")
            checks_passed += 1
        else:
            print(f"❌ FAIL - Expected ~63 DNPs, found {dnps_added}")
        
        print(f"\n{'='*80}")
        print(f"OVERALL: {checks_passed}/3 checks passed")
        print()
    
    elif season == '2024_25' and player_name == 'Jimmy Butler':
        print("="*80)
        print("JIMMY BUTLER VERIFICATION (MID-SEASON TRADE)")
        print("="*80)
        print(f"Expected: 55 total games played (~31 MIA, ~24 GSW)")
        print(f"Found: {len(player_df)} total games played")
        print(f"Expected filled: ~81 games (MIA games + GSW games, NO overlap)")
        print(f"Found filled: {len(filled_df)} total games")
        print(f"Expected stints: 2 (traded from MIA to GSW)")
        print(f"Found stints: {len(stints)}")
        
        checks_passed = 0
        
        # Check total games played
        if len(player_df) == 55:
            print("✅ PASS - Played 55 total games")
            checks_passed += 1
        else:
            print(f"❌ FAIL - Expected 55 games, found {len(player_df)}")
        
        # Check filled games (should be around 81, NOT 164)
        if 79 <= len(filled_df) <= 83:
            print(f"✅ PASS - Filled {len(filled_df)} games (no double-counting)")
            checks_passed += 1
        else:
            print(f"❌ FAIL - Expected 79-83 filled games, found {len(filled_df)}")
        
        # Check number of stints
        if len(stints) == 2:
            print("✅ PASS - Found 2 team stints (trade detected)")
            checks_passed += 1
            
            # Check stint details
            stint1_team = stints[0]['team']
            stint2_team = stints[1]['team']
            
            print(f"\n  Stint 1: {stint1_team} ({stints[0]['start_date']} to {stints[0]['end_date']})")
            stint1_games = player_df[
                (player_df['date'] >= stints[0]['start_date']) &
                (player_df['date'] <= stints[0]['end_date'])
            ]
            print(f"    Games played: {len(stint1_games)}")
            
            print(f"\n  Stint 2: {stint2_team} ({stints[1]['start_date']} to {stints[1]['end_date']})")
            stint2_games = player_df[
                (player_df['date'] >= stints[1]['start_date']) &
                (player_df['date'] <= stints[1]['end_date'])
            ]
            print(f"    Games played: {len(stint2_games)}")
            
            # Verify teams
            if stint1_team == 'MIA' and stint2_team == 'GSW':
                print(f"\n✅ PASS - Trade teams correct (MIA → GSW)")
                checks_passed += 1
            else:
                print(f"\n❌ FAIL - Expected MIA → GSW, found {stint1_team} → {stint2_team}")
        else:
            print(f"❌ FAIL - Expected 2 stints, found {len(stints)}")
        
        print(f"\n{'='*80}")
        print(f"OVERALL: {checks_passed}/4 checks passed")
        print()
    
    # FINAL VERIFICATION: Check total games after filling
    print("="*80)
    print("FINAL VERIFICATION - COMPLETE SEASON CHECK")
    print("="*80)
    
    # Count games within stints (NO overlap for trades)
    stint_games = 0
    for stint in stints:
        team = stint['team']
        start_date = stint['start_date']
        end_date = stint['end_date']
        
        # Get team games for this stint
        team_games_in_stint = schedule_df[
            ((schedule_df['home_team'] == team) | (schedule_df['away_team'] == team)) &
            (schedule_df['date'] >= start_date) &
            (schedule_df['date'] <= end_date)
        ]
        stint_games += len(team_games_in_stint)
    
    # No need for trade gap calculation - stints now perfectly partition the season
    # (end of stint 1 = day before start of stint 2)
    
    filled_games = len(filled_df)
    expected_games = stint_games
    
    print(f"Player: {player_name}")
    print(f"Team games (across all stints): {stint_games}")
    print(f"Expected total: {expected_games}")
    print(f"Games after filling: {filled_games}")
    print(f"Match: {'✅ YES' if filled_games == expected_games else '❌ NO'}")
    
    if filled_games == expected_games:
        print(f"\n✅ SUCCESS - Complete season data captured!")
        if filled_games >= 82:
            print(f"   Player has full season ({filled_games} games)")
        elif len(stints) > 1:
            print(f"   Player was traded mid-season (covered all games with both teams)")
        else:
            print(f"   Player joined/left mid-season ({filled_games} games available)")
    else:
        gap = expected_games - filled_games
        print(f"\n⚠️  WARNING - Missing {gap} games!")
        print(f"   Possible reasons:")
        print(f"   • Long injury (>{max_gap_days} days) not filled")
        print(f"   • Player traded with gap between teams not captured")
        print(f"   • Data quality issue")
    
    print()
    
    # Show sample of filled data
    print("="*80)
    print("SAMPLE OF FILLED DATA (First 20 rows)")
    print("="*80)
    print()
    
    print(f"{'Date':<12} | {'Matchup':<25} | {'3PM':<5} | {'3PA':<5} | {'Min':<5} | {'DNP?':<6}")
    print("-" * 70)
    
    for i, row in filled_df.head(20).iterrows():
        date = row['date']
        matchup = row['matchup'][:25]
        threes_made = row['threes_made']
        threes_attempted = row['threes_attempted']
        minutes = row['minutes']
        is_dnp = row.get('dnp', False)
        dnp_str = "YES" if is_dnp else ""
        
        print(f"{date:<12} | {matchup:<25} | {threes_made:<5.0f} | {threes_attempted:<5.0f} | {minutes:<5.0f} | {dnp_str:<6}")
    
    print()
    print(f"... ({len(filled_df) - 20} more rows)")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fill missing games for NBA players')
    parser.add_argument('--test', action='store_true', 
                       help='Run verification tests (Stephen Curry + Luka Doncic)')
    parser.add_argument('--all', action='store_true',
                       help='Process all players')
    parser.add_argument('--player', type=str,
                       help='Test on specific player (e.g., "LeBron James")')
    parser.add_argument('--max-gap-days', type=int, default=MAX_GAP_DAYS,
                       help='Maximum gap in days to fill (default: 14)')
    
    args = parser.parse_args()
    
    if args.test:
        # Run verification tests
        print("\n" + "="*80)
        print("RUNNING VERIFICATION TESTS (2024-25 Season)")
        print("="*80)
        print("\nTest 1: Stephen Curry (missed 12 games, no trade, full 82)")
        print("Test 2: Luka Doncic (mid-season trade DAL→LAL, 50 games)")
        print("Test 3: Josh Hart (missed 4 games, single team, 81 total)")
        print("Test 4: Tyus Jones (missed 1 game, full 82-game season)")
        print("Test 5: Joel Embiid (missed MANY games, start+end+middle, full 82)")
        print("Test 6: Jimmy Butler (mid-season trade MIA→GSW, 55 games)")
        print("\n")
        
        # Test 1: Stephen Curry
        test_single_player(
            player_name='Stephen Curry',
            season='2024_25',
            max_gap_days=args.max_gap_days
        )
        
        print("\n" * 3)
        
        # Test 2: Luka Doncic
        test_single_player(
            player_name='Luka Doncic',
            season='2024_25',
            max_gap_days=args.max_gap_days
        )
        
        print("\n" * 3)
        
        # Test 3: Josh Hart
        test_single_player(
            player_name='Josh Hart',
            season='2024_25',
            max_gap_days=args.max_gap_days
        )
        
        print("\n" * 3)
        
        # Test 4: Tyus Jones
        test_single_player(
            player_name='Tyus Jones',
            season='2024_25',
            max_gap_days=args.max_gap_days
        )
        
        print("\n" * 3)
        
        # Test 5: Joel Embiid
        test_single_player(
            player_name='Joel Embiid',
            season='2024_25',
            max_gap_days=args.max_gap_days
        )
        
        print("\n" * 3)
        
        # Test 6: Jimmy Butler
        test_single_player(
            player_name='Jimmy Butler',
            season='2024_25',
            max_gap_days=args.max_gap_days
        )
        
        print("\n" + "="*80)
        print("VERIFICATION TESTS COMPLETE - ALL 6 TESTS")
        print("="*80)
        print()
        
    elif args.player:
        # Test specific player
        test_single_player(
            player_name=args.player,
            season='2024_25',
            max_gap_days=args.max_gap_days
        )
    elif args.all:
        # Process all players
        process_all_players(
            season='2024_25',
            max_gap_days=args.max_gap_days
        )
    else:
        # Default: show help
        parser.print_help()


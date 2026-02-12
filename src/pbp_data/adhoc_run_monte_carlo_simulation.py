"""
Ad-hoc Monte Carlo Simulation for Single Player/Moment

Purpose:
Takes a player and specific game context (quarter, time, current points) and runs
Monte Carlo simulation to estimate probability of hitting a prop line.

Example from image:
- Player: Nikola Jokic
- Prop: Under 25.5 points
- Context: Q2 with 10:49 remaining, current points scored
- Juice Reel model valued bet at -$19.56 (18% probability)
- Our model: ???

Usage:
    # Basic example (Jokic in Q2 with 10:49 left, 0 points, 25.5 line)
    python src/pbp_data/adhoc_run_monte_carlo_simulation.py \
        --player-name "Nikola Jokic" \
        --quarter 2 \
        --time-remaining "10:49" \
        --current-points 0 \
        --prop-line 25.5 \
        --n-sims 10000
    
    # With vegas adjustment (calibrate to 50% at game start)
    python src/pbp_data/adhoc_run_monte_carlo_simulation.py \
        --player-name "Nikola Jokic" \
        --quarter 2 \
        --time-remaining "10:49" \
        --current-points 0 \
        --prop-line 25.5 \
        --n-sims 10000 \
        --vegas-adjust

Output:
    - Probability of going OVER the prop line
    - Game minute calculation
    - Vegas adjustment factor (if enabled)
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pbp_data.monte_carlo_utils import (
    load_player_profile,
    monte_carlo_simulate_bet,
    find_vegas_adjustment,
    get_data_paths,
    create_ggplot
)
import pandas as pd
import json


def parse_time_remaining(time_str):
    """
    Parse time remaining string (MM:SS) to minutes as float.
    
    Args:
        time_str: Time string like "10:49" or "5:30"
    
    Returns:
        float: Minutes remaining (e.g., 10.817 for 10:49)
    """
    if ':' in time_str:
        mins, secs = map(int, time_str.split(':'))
        return mins + secs / 60.0
    else:
        # Assume it's just minutes
        return float(time_str)


def calculate_game_minute(quarter, time_remaining_in_quarter):
    """
    Calculate current game minute from quarter and time remaining.
    
    NBA quarters are 12 minutes each:
    - Q1: minutes 0-12
    - Q2: minutes 12-24
    - Q3: minutes 24-36
    - Q4: minutes 36-48
    
    Args:
        quarter: Quarter number (1-4)
        time_remaining_in_quarter: Minutes remaining in quarter
    
    Returns:
        float: Current game minute (0-48)
    """
    quarter_start = (quarter - 1) * 12
    quarter_length = 12.0
    time_elapsed_in_quarter = quarter_length - time_remaining_in_quarter
    game_minute = quarter_start + time_elapsed_in_quarter
    
    return game_minute


def main():
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo simulation for single player at specific moment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Nikola Jokic in Q2 with 10:49 left, 9 points scored, U26.5 at -110
  python src/pbp_data/adhoc_run_monte_carlo_simulation.py \\
      --player-name "Nikola Jokic" \\
      --quarter 2 \\
      --time-remaining "10:49" \\
      --current-points 9 \\
      --prop-line 26.5 \\
      --n-sims 10000 \\
      --vegas-adjust

  # Luka Doncic at halftime (Q2 ended), 15 points scored, U34.5 line
  python src/pbp_data/adhoc_run_monte_carlo_simulation.py \\
      --player-name "Luka Doncic" \\
      --quarter 2 \\
      --time-remaining "0:00" \\
      --current-points 15 \\
      --prop-line 34.5 \\
      --n-sims 10000 \\
      --vegas-adjust
        """
    )
    
    parser.add_argument("--player-name", required=True, help="Player name (e.g., 'Nikola Jokic')")
    parser.add_argument("--quarter", type=int, required=True, help="Current quarter (1-4)")
    parser.add_argument("--time-remaining", required=True, help="Time remaining in quarter (MM:SS format, e.g., '10:49')")
    parser.add_argument("--current-points", type=float, required=True, help="Points scored so far")
    parser.add_argument("--prop-line", type=float, required=True, help="Prop line (e.g., 26.5)")
    parser.add_argument("--n-sims", type=int, default=10000, help="Number of Monte Carlo simulations (default: 10000)")
    parser.add_argument("--vegas-adjust", action="store_true", help="Apply Vegas adjustment (calibrate to 50% at game start)")
    parser.add_argument("--plot", action="store_true", help="Generate visualization plot (requires --game-id)")
    parser.add_argument("--game-id", help="ESPN game ID for fetching play-by-play data (required if --plot is used)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.plot and not args.game_id:
        parser.error("--game-id is required when --plot is used")
    
    print("="*80)
    print("AD-HOC MONTE CARLO SIMULATION")
    print("="*80)
    print()
    
    # Parse inputs
    player_name = args.player_name
    quarter = args.quarter
    time_remaining_str = args.time_remaining
    current_points = args.current_points
    prop_line = args.prop_line
    n_sims = args.n_sims
    vegas_adjust = args.vegas_adjust
    
    print(f"📊 Inputs:")
    print(f"   Player: {player_name}")
    print(f"   Quarter: Q{quarter}")
    print(f"   Time Remaining: {time_remaining_str}")
    print(f"   Current Points: {current_points}")
    print(f"   Prop Line: {prop_line}")
    print(f"   Simulations: {n_sims:,}")
    print(f"   Vegas Adjustment: {'Yes' if vegas_adjust else 'No'}")
    print()
    
    # Parse time remaining
    time_remaining = parse_time_remaining(time_remaining_str)
    print(f"⏰ Parsed time remaining: {time_remaining:.3f} minutes")
    
    # Calculate game minute
    game_minute = calculate_game_minute(quarter, time_remaining)
    print(f"⏱️  Current game minute: {game_minute:.3f}")
    print()
    
    # Load player profile
    print(f"📥 Loading player profile for {player_name}...")
    try:
        player_profile = load_player_profile(player_name)
        print(f"   ✅ Loaded profile:")
        print(f"      - Games: {player_profile['num_games']}")
        print(f"      - Avg PPG: {player_profile['avg_points_per_game']:.1f}")
        print(f"      - Avg MPG: {player_profile['avg_minutes_per_game']:.1f}")
        print()
    except Exception as e:
        print(f"   ❌ Failed to load player profile: {e}")
        return
    
    # Determine Vegas adjustment
    vegas_adjustment = 1.0
    if vegas_adjust:
        print(f"🎲 Calculating Vegas adjustment (calibrating to prop line {prop_line})...")
        print(f"   Using binary search to find PPM multiplier that gives 50% prob at game start")
        vegas_adjustment = find_vegas_adjustment(player_profile, prop_line, n_simulations=n_sims)
        print(f"   ✅ Vegas adjustment factor: {vegas_adjustment:.4f}")
        print(f"      (Assumes pregame line {prop_line} was efficient at 50%)")
        print()
    
    # Run Monte Carlo simulation
    print(f"🎲 Running Monte Carlo simulation ({n_sims:,} iterations)...")
    prob_over = monte_carlo_simulate_bet(
        player_profile=player_profile,
        current_minute=game_minute,
        current_points=current_points,
        prop_line=prop_line,
        n_simulations=n_sims,
        vegas_adjustment=vegas_adjustment,
        score_differential=None,  # Unknown for ad-hoc query
        debug=False
    )
    
    print(f"   ✅ Simulation complete")
    print()
    
    # Output results
    print("="*80)
    print("RESULTS")
    print("="*80)
    print()
    print(f"🎯 Probability of OVER {prop_line}: {prob_over:.1%}")
    print(f"🎯 Probability of UNDER {prop_line}: {(1 - prob_over):.1%}")
    print()
    
    # Comparison to image example (if this is Jokic)
    if "jokic" in player_name.lower() and abs(prop_line - 26.5) < 0.1:
        print("📸 Comparison to Juice Reel example (from image):")
        print(f"   - Juice Reel: Valued bet at -$19.56 (implied ~18% chance of OVER)")
        print(f"   - Our model: {prob_over:.1%} probability of OVER")
        print()
        
        # Calculate implied value of UNDER 26.5 bet at -110
        # -110 means: risk $11 to win $10 (risk $22 to win $20, etc.)
        # If we bet $20 at -110: we risk $20 to win $18.18
        # EV = P(win) * $18.18 - P(lose) * $20
        prob_under = 1 - prob_over
        win_amount = 20 * (100 / 110)  # Win amount for $20 bet at -110
        ev = prob_under * win_amount - prob_over * 20
        
        print(f"💰 Expected Value of UNDER 26.5 bet at -110 ($20 risked):")
        print(f"   - Our model EV: ${ev:.2f}")
        print(f"   - Juice Reel live value: -$19.56")
        print(f"   - Difference: ${ev - (-19.56):.2f}")
        print()
    
    print("="*80)
    print("✅ SIMULATION COMPLETE")
    print("="*80)
    
    # Generate plot if requested
    if args.plot:
        print()
        print("="*80)
        print("GENERATING PLOT")
        print("="*80)
        print()
        
        game_id = args.game_id
        
        # Load play-by-play data
        pbp_file = Path(__file__).parent / "tmp" / f"live_game_{game_id}.json"
        
        if not pbp_file.exists():
            print(f"❌ Play-by-play data not found: {pbp_file}")
            print(f"   Please fetch the game data first using the live game tracker")
            return
        
        print(f"📥 Loading play-by-play data from: {pbp_file.name}")
        
        with open(pbp_file, 'r') as f:
            pbp_data = json.load(f)
        
        # Parse game metadata
        boxscore = pbp_data['boxscore']
        teams = boxscore['teams']
        away_team = teams[0]['team']['displayName']
        home_team = teams[1]['team']['displayName']
        
        # Get game date from header
        header = pbp_data.get('header', {})
        game_date_str = header.get('competitions', [{}])[0].get('date', '')
        if game_date_str:
            game_dt_utc = pd.to_datetime(game_date_str)
            game_dt_et = game_dt_utc.tz_convert("America/New_York")
            game_date = game_dt_et.strftime("%Y-%m-%d")
        else:
            game_date = "unknown"
        
        print(f"   🏀 {away_team} @ {home_team}")
        print(f"   📅 {game_date}")
        print()
        
        # Parse plays and track player points
        print(f"📊 Parsing plays to track {player_name}'s scoring...")
        plays = pbp_data['plays']
        
        play_data = []
        cumulative_points = 0
        
        for play in plays:
            quarter = play.get('period', {}).get('number', 1)
            
            # Calculate game minute
            clock_display = play.get('clock', {}).get('displayValue', '12:00')
            try:
                if isinstance(clock_display, (int, float)):
                    time_left_in_quarter = float(clock_display) / 60.0
                elif ':' in str(clock_display):
                    mins, secs = map(int, str(clock_display).split(':'))
                    time_left_in_quarter = mins + secs / 60.0
                else:
                    time_left_in_quarter = float(clock_display) / 60.0
                
                quarter_start = (quarter - 1) * 12
                game_minute = quarter_start + (12 - time_left_in_quarter)
            except Exception:
                game_minute = (quarter - 1) * 12
            
            description = play.get('text', '')
            
            # Check if player scored
            points_this_play = 0
            if player_name in description:
                if 'makes' in description.lower() or 'free throw' in description.lower():
                    if '3-pt' in description.lower() or 'three point' in description.lower():
                        points_this_play = 3
                    elif '2-pt' in description.lower() or 'two point' in description.lower():
                        points_this_play = 2
                    elif 'free throw' in description.lower() and 'makes' in description.lower():
                        points_this_play = 1
            
            cumulative_points += points_this_play
            
            play_data.append({
                'quarter': quarter,
                'game_minute': game_minute,
                'cumulative_points': cumulative_points,
            })
        
        # Convert to DataFrame and sort by game_minute
        df_plays = pd.DataFrame(play_data)
        df_plays = df_plays.sort_values('game_minute').reset_index(drop=True)
        
        # Keep only up to current minute (user's specified time, not live game time)
        df_plays = df_plays[df_plays['game_minute'] <= game_minute].copy()
        
        # Override final cumulative points with user's input (more accurate than PBP parsing)
        if len(df_plays) > 0:
            df_plays.loc[df_plays.index[-1], 'cumulative_points'] = current_points
        
        print(f"   ✅ Tracked {len(df_plays)} plays up to minute {game_minute:.2f}")
        print(f"   📈 Using provided points: {current_points} (PBP parsed: {cumulative_points})")
        print()
        
        # Build probability curve by running MC at sample points
        print(f"📈 Building probability curve (running MC at multiple time points)...")
        
        # Sample points: every 2 minutes + current minute
        sample_minutes = list(range(0, int(game_minute) + 1, 2)) + [game_minute]
        sample_minutes = sorted(set(sample_minutes))
        
        prob_data = []
        for minute in sample_minutes:
            # Find player's points at this minute
            plays_up_to = df_plays[df_plays['game_minute'] <= minute]
            if len(plays_up_to) > 0:
                points_at_minute = plays_up_to.iloc[-1]['cumulative_points']
            else:
                points_at_minute = 0
            
            # Run MC simulation
            prob = monte_carlo_simulate_bet(
                player_profile=player_profile,
                current_minute=minute,
                current_points=points_at_minute,
                prop_line=prop_line,
                n_simulations=n_sims,
                vegas_adjustment=vegas_adjustment,
                score_differential=None,
                debug=False
            )
            
            prob_data.append({
                'game_minute': minute,
                'cumulative_points': points_at_minute,
                'prob_over': prob
            })
            
            print(f"   ⏱️  Minute {minute:.1f}: {points_at_minute} pts, {prob:.1%} prob over")
        
        df_probs = pd.DataFrame(prob_data)
        
        # Merge with full play data
        df_plot = df_plays.merge(df_probs, on='game_minute', how='left', suffixes=('', '_mc'))
        df_plot['prob_over'] = df_plot['prob_over'].ffill().bfill()
        df_plot['cumulative_points'] = df_plot['cumulative_points'].ffill()
        
        print()
        print(f"📊 Generating plot...")
        
        # Get player_id (try to extract from boxscore or use 0 as fallback)
        player_id = player_profile.get('player_id', 0)
        
        # Determine final result (game not complete, so mark as "IN PROGRESS")
        final_points = current_points
        result = "IN PROGRESS"
        
        # Use the ORIGINAL user-specified game_minute for the marker (not the end of data)
        observation_minute = calculate_game_minute(args.quarter, parse_time_remaining(args.time_remaining))
        
        # Get the actual current game minute (end of data)
        current_game_state_minute = df_plays['game_minute'].max()
        
        # Generate plot
        plot_dir = Path(__file__).parent / "tmp" / "plots"
        plot_dir.mkdir(exist_ok=True, parents=True)
        
        plot_path = create_ggplot(
            df=df_plot,
            prop_line=prop_line,
            player_name=player_name,
            player_id=player_id,
            game_id=game_id,
            game_date=game_date,
            away_team=away_team,
            home_team=home_team,
            final_points=final_points,
            result=result,
            plot_dir=plot_dir,
            bet_placement_minute=observation_minute,  # Model comparison point
            current_game_minute=current_game_state_minute  # Current game state
        )
        
        if plot_path:
            print(f"   ✅ Plot saved: {plot_path}")
            print()
            print(f"📂 To open the plot:")
            print(f"   open \"{plot_path}\"")
        else:
            print(f"   ❌ Plot generation failed")
        
        print()


if __name__ == "__main__":
    main()

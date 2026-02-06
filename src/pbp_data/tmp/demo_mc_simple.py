"""
Simple Monte Carlo demo for Luka Doncic - one game, first play debug.

Usage:
    cd /Users/thomasmyles/dev/betting
    python src/pbp_data/tmp/demo_mc_simple.py
"""

import duckdb
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PLAYER_PROFILES = DATA_DIR / "player_profiles.parquet"
MINUTE_BY_MINUTE = DATA_DIR / "minute_by_minute.parquet"


def load_player_profile(player_name):
    """Load player's quarterly distributions from parquet."""
    con = duckdb.connect()
    result = con.execute(f"""
        SELECT 
            q1_minutes_history,
            q2_minutes_history,
            q3_minutes_history,
            q4_minutes_history,
            points_per_minute_history,
            avg_points_per_game, 
            num_games
        FROM '{PLAYER_PROFILES}'
        WHERE player_name = '{player_name}'
    """).fetchone()
    con.close()
    
    return {
        'q1_minutes_history': result[0],
        'q2_minutes_history': result[1],
        'q3_minutes_history': result[2],
        'q4_minutes_history': result[3],
        'game_ppm_history': result[4],  # NEW: Full-game PPM
        'avg_ppg': result[5],
        'num_games': result[6]
    }


def get_quarter_info(current_minute):
    """Return (current_quarter, time_left_in_quarter)."""
    if current_minute < 12:
        return 1, 12.0 - current_minute
    elif current_minute < 24:
        return 2, 24.0 - current_minute
    elif current_minute < 36:
        return 3, 36.0 - current_minute
    else:
        return 4, 48.0 - current_minute


def find_ppm_adjustment_for_50_percent(prop_line, player_profile, n_sims=5000, tolerance=0.001):
    """
    Binary search to find exact PPM adjustment that gives 50% probability at game start.
    """
    print(f"   🔍 Finding PPM adjustment to start at 50%...")
    
    low, high = -0.5, 0.5  # Search range for adjustment
    best_adjustment = 0
    
    for iteration in range(20):  # Max 20 iterations
        mid = (low + high) / 2
        
        # Test this adjustment
        hits = 0
        for _ in range(n_sims):
            game_ppm = random.choice(player_profile['game_ppm_history']) + mid
            projected_points = 0
            
            # Full game projection
            for q in range(1, 5):
                minutes = random.choice(player_profile[f'q{q}_minutes_history'])
                projected_points += minutes * game_ppm
            
            if projected_points > prop_line:
                hits += 1
        
        prob = hits / n_sims
        
        if abs(prob - 0.50) < tolerance:
            best_adjustment = mid
            print(f"      ✅ Found adjustment: {best_adjustment:+.4f} (gives {prob:.1%})")
            break
        elif prob < 0.50:
            low = mid  # Need higher adjustment
        else:
            high = mid  # Need lower adjustment
        
        best_adjustment = mid
    
    return best_adjustment


def monte_carlo(current_minute, current_points, prop_line, player_profile, ppm_adjustment=0, 
                n_sims=10000, debug=False):
    """
    Run Monte Carlo simulation from current game state.
    
    Args:
        ppm_adjustment: Fixed PPM adjustment (calculated once at game start to give 50%)
    
    Returns: probability of going OVER prop_line
    """
    hits = 0
    debug_sims = []
    
    current_quarter, time_left_in_quarter = get_quarter_info(current_minute)
    
    for sim_idx in range(n_sims):
        projected_points = current_points
        sim_details = {'sim': sim_idx + 1, 'start': current_points, 'quarters': []}
        
        # Sample base PPM and apply FIXED adjustment
        base_ppm = random.choice(player_profile['game_ppm_history'])
        game_ppm = base_ppm + ppm_adjustment
        
        # PROJECT CURRENT QUARTER (partial)
        if time_left_in_quarter > 0:
            typical_q_minutes = random.choice(player_profile[f'q{current_quarter}_minutes_history'])
            proportion_left = time_left_in_quarter / 12.0
            projected_minutes = typical_q_minutes * proportion_left
            q_points = projected_minutes * game_ppm
            projected_points += q_points
            
            if debug and sim_idx < 5:
                sim_details['quarters'].append({
                    'q': current_quarter,
                    'type': 'partial',
                    'time_left': time_left_in_quarter,
                    'typical_min': typical_q_minutes,
                    'proj_min': projected_minutes,
                    'ppm': game_ppm,
                    'points': q_points
                })
        
        # PROJECT FUTURE QUARTERS
        for future_q in range(current_quarter + 1, 5):
            minutes = random.choice(player_profile[f'q{future_q}_minutes_history'])
            q_points = minutes * game_ppm
            projected_points += q_points
            
            if debug and sim_idx < 5:
                sim_details['quarters'].append({
                    'q': future_q,
                    'type': 'full',
                    'minutes': minutes,
                    'ppm': game_ppm,
                    'points': q_points
                })
        
        if projected_points > prop_line:
            hits += 1
        
        if debug and sim_idx < 5:
            sim_details['final'] = projected_points
            sim_details['hit'] = projected_points > prop_line
            sim_details['game_ppm'] = game_ppm
            debug_sims.append(sim_details)
    
    # Print debug
    if debug:
        print(f"\n🔍 MONTE CARLO DEBUG")
        if ppm_adjustment != 0:
            print(f"   🎰 PPM Adjustment: {ppm_adjustment:+.4f} (applied to all simulations)")
        print(f"\n   Current state: Minute {current_minute:.1f}, {current_points} pts")
        print(f"   Prop line: Over {prop_line}")
        print(f"   Player avg: {player_profile['avg_ppg']:.1f} ppg")
        print(f"\n   First 5 simulations:")
        
        for sim in debug_sims:
            base_ppm = sim['game_ppm'] - ppm_adjustment
            if ppm_adjustment != 0:
                ppm_display = f"{base_ppm:.3f} + {ppm_adjustment:.3f} = {sim['game_ppm']:.3f}"
            else:
                ppm_display = f"{sim['game_ppm']:.3f}"
            
            print(f"\n   Sim {sim['sim']}: Start={sim['start']:.1f}pts, Game PPM={ppm_display}")
            for q in sim['quarters']:
                if q['type'] == 'partial':
                    print(f"      Q{q['q']} (partial, {q['time_left']:.1f}min left): "
                          f"{q['proj_min']:.1f}min × {q['ppm']:.3f}ppm = {q['points']:.1f}pts")
                else:
                    print(f"      Q{q['q']} (full): {q['minutes']:.1f}min × {q['ppm']:.3f}ppm = {q['points']:.1f}pts")
            print(f"      → Final: {sim['final']:.1f}pts ({'✅ OVER' if sim['hit'] else '❌ UNDER'})")
        
        prob = hits / n_sims
        print(f"\n   📊 RESULT: {hits}/{n_sims} hit OVER")
        print(f"      → Probability: {prob:.1%}")
    
    return hits / n_sims


def main():
    print("="*80)
    print("SIMPLE MONTE CARLO - FIRST PLAY DEBUG")
    print("="*80)
    
    player_name = "Luka Doncic"
    prop_line = 33.5
    n_sims = 50000
    
    # Load profile
    print(f"\n📊 Loading {player_name}'s profile...")
    profile = load_player_profile(player_name)
    print(f"   ✅ {profile['num_games']} games, {profile['avg_ppg']:.1f} ppg avg")
    
    # Get one game
    con = duckdb.connect()
    game = con.execute(f"""
        SELECT game_id, game_date
        FROM '{MINUTE_BY_MINUTE}'
        WHERE player_name = '{player_name}'
        GROUP BY game_id, game_date
        ORDER BY game_date DESC
        LIMIT 1
    """).fetchone()
    con.close()
    
    game_id, game_date = game[0], game[1]
    print(f"\n🏀 Game: {game_id} on {game_date}")
    print(f"   Prop line: Over {prop_line}")
    
    # Run Monte Carlo for first play (minute 0, 0 points)
    print(f"\n🎲 Running simulations...")
    
    # Find adjustment for 50% at game start
    print(f"\n{'='*80}")
    print(f"🎰 FINDING VEGAS ADJUSTMENT (to start at 50%)")
    print(f"{'='*80}")
    ppm_adjustment = find_ppm_adjustment_for_50_percent(prop_line, profile, n_sims=5000)
    
    # Run WITHOUT adjustment
    print(f"\n{'='*80}")
    print(f"1️⃣  BASELINE (No Adjustment)")
    print(f"{'='*80}")
    prob_baseline = monte_carlo(
        current_minute=0.0,
        current_points=0,
        prop_line=prop_line,
        player_profile=profile,
        ppm_adjustment=0,
        n_sims=n_sims,
        debug=True
    )
    
    # Run WITH adjustment
    print(f"\n{'='*80}")
    print(f"2️⃣  VEGAS-ADJUSTED (Should start at ~50%)")
    print(f"{'='*80}")
    prob_vegas = monte_carlo(
        current_minute=0.0,
        current_points=0,
        prop_line=prop_line,
        player_profile=profile,
        ppm_adjustment=ppm_adjustment,
        n_sims=n_sims,
        debug=True
    )
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE")
    print(f"{'='*80}")
    print(f"\n📊 COMPARISON:")
    print(f"   Baseline:       {prob_baseline:.1%}")
    print(f"   Vegas-adjusted: {prob_vegas:.1%} (target: 50%)")
    print(f"   PPM adjustment: {ppm_adjustment:+.4f}")
    print()


if __name__ == "__main__":
    main()

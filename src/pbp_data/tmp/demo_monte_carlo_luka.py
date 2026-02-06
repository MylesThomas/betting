"""
Demo: Monte Carlo simulation for Luka Doncic game.

Goal:
1. Load Luka's historical data (quarterly distributions)
2. Pick one of his games
3. For each minute of the game, run 10,000 Monte Carlo simulations
4. Plot:
   - Top: Probability of covering Over 30.5 over time
   - Bottom: Actual points scored over time

Usage:
    cd /Users/thomasmyles/dev/betting
    python src/pbp_data/tmp/demo_monte_carlo_luka.py
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MINUTE_BY_MINUTE = DATA_DIR / "minute_by_minute.parquet"


# =============================================================================
# LOAD PLAYER PROFILE
# =============================================================================

def load_player_profile(player_name):
    """
    Load player's historical data (quarterly distributions) using DuckDB.
    
    Returns:
        dict with quarterly distributions (lists)
    """
    con = duckdb.connect()
    
    # Run SQL to build player profiles
    con.execute(f"""
        -- Step 1: Game-level stats
        CREATE OR REPLACE TEMP TABLE game_level_stats AS
        SELECT 
            game_id,
            game_date,
            player_id,
            player_name,
            MAX(playing_seconds) / 60.0 AS total_minutes,
            MAX(cumulative_points) AS total_points
        FROM '{MINUTE_BY_MINUTE}'
        GROUP BY game_id, game_date, player_id, player_name;
        
        CREATE OR REPLACE TEMP TABLE game_stats_with_ppm AS
        SELECT 
            *,
            CASE 
                WHEN total_minutes > 0 THEN total_points / total_minutes 
                ELSE 0 
            END AS points_per_minute
        FROM game_level_stats;
        
        -- Step 2: Quarterly splits
        CREATE OR REPLACE TEMP TABLE quarter_splits AS
        SELECT 
            game_id,
            game_date,
            player_id,
            player_name,
            
            -- Q1 (minutes 0-11)
            MAX(CASE WHEN minute <= 11 THEN playing_seconds ELSE 0 END) / 60.0 AS q1_minutes,
            MAX(CASE WHEN minute <= 11 THEN cumulative_points ELSE 0 END) AS q1_points,
            
            -- Q2 (minutes 12-23)
            (MAX(CASE WHEN minute <= 23 THEN playing_seconds ELSE 0 END) - 
             MAX(CASE WHEN minute <= 11 THEN playing_seconds ELSE 0 END)) / 60.0 AS q2_minutes,
            (MAX(CASE WHEN minute <= 23 THEN cumulative_points ELSE 0 END) - 
             MAX(CASE WHEN minute <= 11 THEN cumulative_points ELSE 0 END)) AS q2_points,
            
            -- Q3 (minutes 24-35)
            (MAX(CASE WHEN minute <= 35 THEN playing_seconds ELSE 0 END) - 
             MAX(CASE WHEN minute <= 23 THEN playing_seconds ELSE 0 END)) / 60.0 AS q3_minutes,
            (MAX(CASE WHEN minute <= 35 THEN cumulative_points ELSE 0 END) - 
             MAX(CASE WHEN minute <= 23 THEN cumulative_points ELSE 0 END)) AS q3_points,
            
            -- Q4 (minutes 36-47)
            (MAX(CASE WHEN minute <= 47 THEN playing_seconds ELSE 0 END) - 
             MAX(CASE WHEN minute <= 35 THEN playing_seconds ELSE 0 END)) / 60.0 AS q4_minutes,
            (MAX(CASE WHEN minute <= 47 THEN cumulative_points ELSE 0 END) - 
             MAX(CASE WHEN minute <= 35 THEN cumulative_points ELSE 0 END)) AS q4_points
            
        FROM '{MINUTE_BY_MINUTE}'
        GROUP BY game_id, game_date, player_id, player_name;
        
        CREATE OR REPLACE TEMP TABLE quarter_splits_with_ppm AS
        SELECT 
            *,
            CASE WHEN q1_minutes > 0 THEN q1_points / q1_minutes ELSE 0 END AS q1_ppm,
            CASE WHEN q2_minutes > 0 THEN q2_points / q2_minutes ELSE 0 END AS q2_ppm,
            CASE WHEN q3_minutes > 0 THEN q3_points / q3_minutes ELSE 0 END AS q3_ppm,
            CASE WHEN q4_minutes > 0 THEN q4_points / q4_minutes ELSE 0 END AS q4_ppm
        FROM quarter_splits;
    """)
    
    # Get player profile
    query = f"""
    SELECT 
        g.player_id,
        g.player_name,
        COUNT(*) AS num_games,
        
        -- Summary stats
        AVG(g.total_points) AS avg_points_per_game,
        AVG(g.total_minutes) AS avg_minutes_per_game,
        
        -- Q1 distributions
        LIST(q.q1_minutes ORDER BY q.game_date DESC) AS q1_minutes_history,
        LIST(q.q1_ppm ORDER BY q.game_date DESC) AS q1_ppm_history,
        
        -- Q2 distributions
        LIST(q.q2_minutes ORDER BY q.game_date DESC) AS q2_minutes_history,
        LIST(q.q2_ppm ORDER BY q.game_date DESC) AS q2_ppm_history,
        
        -- Q3 distributions
        LIST(q.q3_minutes ORDER BY q.game_date DESC) AS q3_minutes_history,
        LIST(q.q3_ppm ORDER BY q.game_date DESC) AS q3_ppm_history,
        
        -- Q4 distributions
        LIST(q.q4_minutes ORDER BY q.game_date DESC) AS q4_minutes_history,
        LIST(q.q4_ppm ORDER BY q.game_date DESC) AS q4_ppm_history
        
    FROM game_stats_with_ppm g
    LEFT JOIN quarter_splits_with_ppm q 
        ON g.game_id = q.game_id 
        AND g.player_id = q.player_id
    WHERE g.player_name = '{player_name}'
    GROUP BY g.player_id, g.player_name
    """
    
    result = con.execute(query).fetchone()
    
    if not result:
        raise ValueError(f"Player {player_name} not found")
    
    profile = {
        'player_id': result[0],
        'player_name': result[1],
        'num_games': result[2],
        'avg_points_per_game': result[3],
        'avg_minutes_per_game': result[4],
        'q1_minutes_history': result[5],
        'q1_ppm_history': result[6],
        'q2_minutes_history': result[7],
        'q2_ppm_history': result[8],
        'q3_minutes_history': result[9],
        'q3_ppm_history': result[10],
        'q4_minutes_history': result[11],
        'q4_ppm_history': result[12],
    }
    
    con.close()
    
    return profile


# =============================================================================
# LOAD GAME DATA
# =============================================================================

def get_player_games(player_name):
    """Get list of games for a player."""
    con = duckdb.connect()
    
    query = f"""
    SELECT DISTINCT 
        game_id,
        game_date,
        MAX(cumulative_points) AS final_points
    FROM '{MINUTE_BY_MINUTE}'
    WHERE player_name = '{player_name}'
    GROUP BY game_id, game_date
    ORDER BY game_date DESC
    """
    
    df = con.execute(query).df()
    con.close()
    
    return df


def load_game_minute_by_minute(player_name, game_id):
    """Load minute-by-minute data for a specific game."""
    con = duckdb.connect()
    
    query = f"""
    SELECT 
        minute,
        cumulative_points,
        playing_seconds / 60.0 AS minutes_played
    FROM '{MINUTE_BY_MINUTE}'
    WHERE player_name = '{player_name}'
    AND game_id = '{game_id}'
    ORDER BY minute
    """
    
    df = con.execute(query).df()
    con.close()
    
    return df


# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

def monte_carlo_simulate_bet(
    player_profile,
    current_minute,
    current_points,
    prop_line,
    n_simulations=10000,
    debug=False
):
    """
    Run Monte Carlo simulation for remaining game.
    
    Args:
        player_profile: dict with quarterly distributions
        current_minute: Current game minute (0-47)
        current_points: Points scored so far
        prop_line: Target line (e.g., 30.5)
        n_simulations: Number of simulations
        debug: If True, print first 5 simulations
    
    Returns:
        prob_over: Probability of hitting over
    """
    # Determine current quarter and time remaining
    if current_minute < 12:
        current_quarter = 1
        time_remaining_in_quarter = 12 - current_minute
    elif current_minute < 24:
        current_quarter = 2
        time_remaining_in_quarter = 24 - current_minute
    elif current_minute < 36:
        current_quarter = 3
        time_remaining_in_quarter = 36 - current_minute
    else:
        current_quarter = 4
        time_remaining_in_quarter = 48 - current_minute
    
    # If game is over, return deterministic result
    if time_remaining_in_quarter <= 0 and current_quarter >= 4:
        return 1.0 if current_points > prop_line else 0.0
    
    hits = 0
    debug_sims = []
    
    for sim_num in range(n_simulations):
        projected_final_points = current_points
        sim_details = {
            'sim': sim_num + 1,
            'start': current_points,
            'quarters': []
        }
        
        # Current quarter (partial) - project remaining time
        if time_remaining_in_quarter > 0:
            minutes_key = f'q{current_quarter}_minutes_history'
            ppm_key = f'q{current_quarter}_ppm_history'
            
            minutes_history = player_profile[minutes_key]
            ppm_history = player_profile[ppm_key]
            
            # Filter out zeros
            minutes_history = [m for m in minutes_history if m > 0]
            ppm_history = [p for p in ppm_history if p > 0]
            
            if minutes_history and ppm_history:
                # Sample typical minutes played in this quarter
                typical_minutes_this_quarter = random.choice(minutes_history)
                
                # Scale by proportion of quarter remaining
                # If 12 min left in a 12 min quarter → play full typical minutes
                # If 6 min left in a 12 min quarter → play half typical minutes
                quarter_length = 12.0  # NBA quarters are 12 minutes
                proportion_remaining = time_remaining_in_quarter / quarter_length
                projected_minutes_remaining = typical_minutes_this_quarter * proportion_remaining
                
                # Sample PPM
                current_q_ppm = random.choice(ppm_history)
                
                # Calculate points
                remaining_quarter_points = current_q_ppm * projected_minutes_remaining
                projected_final_points += remaining_quarter_points
                
                if debug and sim_num < 5:
                    sim_details['quarters'].append({
                        'q': current_quarter,
                        'type': 'partial',
                        'game_time_left': time_remaining_in_quarter,
                        'typical_min': typical_minutes_this_quarter,
                        'proportion': proportion_remaining,
                        'proj_min': projected_minutes_remaining,
                        'ppm': current_q_ppm,
                        'points': remaining_quarter_points
                    })
        
        # Future quarters
        for future_quarter in range(current_quarter + 1, 5):
            minutes_key = f'q{future_quarter}_minutes_history'
            ppm_key = f'q{future_quarter}_ppm_history'
            
            minutes_history = player_profile[minutes_key]
            ppm_history = player_profile[ppm_key]
            
            # Filter out zeros
            minutes_history = [m for m in minutes_history if m > 0]
            ppm_history = [p for p in ppm_history if p > 0]
            
            if minutes_history and ppm_history:
                future_q_minutes = random.choice(minutes_history)
                future_q_ppm = random.choice(ppm_history)
                future_quarter_points = future_q_ppm * future_q_minutes
                projected_final_points += future_quarter_points
                
                if debug and sim_num < 5:
                    sim_details['quarters'].append({
                        'q': future_quarter,
                        'type': 'full',
                        'minutes': future_q_minutes,
                        'ppm': future_q_ppm,
                        'points': future_quarter_points
                    })
        
        # Check if bet hits
        if projected_final_points > prop_line:
            hits += 1
        
        if debug and sim_num < 5:
            sim_details['final'] = projected_final_points
            sim_details['hit'] = projected_final_points > prop_line
            debug_sims.append(sim_details)
    
    # Print debug info
    if debug:
        print(f"\n🔍 DEBUG: Monte Carlo Simulation Breakdown")
        print(f"   Starting state: Minute {current_minute}, {current_points} pts scored")
        print(f"   Target: Over {prop_line} pts")
        print(f"   Running {n_simulations:,} simulations...")
        print()
        print("   First 5 simulations:")
        
        for sim in debug_sims:
            print(f"\n  Sim {sim['sim']}: Start={sim['start']:.1f}")
            for q in sim['quarters']:
                if q['type'] == 'partial':
                    print(f"    Q{q['q']} (partial, {q['game_time_left']:.1f}min game clock left):")
                    print(f"      Typical Q{q['q']} mins: {q['typical_min']:.1f} → Projected: {q['proj_min']:.1f} (×{q['proportion']:.2f})")
                    print(f"      {q['ppm']:.3f} PPM × {q['proj_min']:.1f}min = {q['points']:.1f} pts")
                else:
                    print(f"    Q{q['q']} (full): {q['ppm']:.3f} PPM × {q['minutes']:.1f}min = {q['points']:.1f} pts")
            print(f"    → Final: {sim['final']:.1f} pts ({'✅ HIT OVER' if sim['hit'] else '❌ STAY UNDER'})")
        
        print()
        print(f"   📊 RESULTS AFTER {n_simulations:,} SIMULATIONS:")
        print(f"      Simulations OVER {prop_line}:  {hits:,} / {n_simulations:,}")
        print(f"      Simulations UNDER {prop_line}: {n_simulations - hits:,} / {n_simulations:,}")
        print(f"      → Probability OVER:  {hits / n_simulations:.1%}")
        print(f"      → Probability UNDER: {(n_simulations - hits) / n_simulations:.1%}")
    
    prob_over = hits / n_simulations
    return prob_over


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_monte_carlo_results(minute_data, prob_data, prop_line, player_name, game_id, game_date):
    """
    Plot probability over time and actual points over time.
    
    Args:
        minute_data: DataFrame with columns [minute, cumulative_points]
        prob_data: DataFrame with columns [minute, prob_over]
        prop_line: Target line
        player_name: Player name
        game_id: Game ID
        game_date: Game date
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Top plot: Probability over time
    ax1.plot(prob_data['minute'], prob_data['prob_over'] * 100, 
             linewidth=2, color='#1f77b4', label=f'Prob Over {prop_line}')
    ax1.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50% (coin flip)')
    ax1.fill_between(prob_data['minute'], 0, prob_data['prob_over'] * 100, 
                      alpha=0.3, color='#1f77b4')
    ax1.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{player_name} - Monte Carlo Simulation (Over {prop_line} pts)\n' + 
                  f'Game: {game_id} on {game_date}', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 100)
    
    # Add quarter markers
    for q_start in [0, 12, 24, 36]:
        ax1.axvline(q_start, color='red', linestyle='--', alpha=0.3, linewidth=1)
    
    # Bottom plot: Actual points over time
    ax2.plot(minute_data['minute'], minute_data['cumulative_points'], 
             linewidth=2, color='#2ca02c', label='Actual Points')
    ax2.axhline(prop_line, color='red', linestyle='--', alpha=0.7, 
                linewidth=2, label=f'Target: {prop_line} pts')
    ax2.fill_between(minute_data['minute'], 0, minute_data['cumulative_points'], 
                      alpha=0.3, color='#2ca02c')
    ax2.set_xlabel('Game Time (minutes)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Points Scored', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Add quarter markers
    for q_start in [0, 12, 24, 36]:
        ax2.axvline(q_start, color='red', linestyle='--', alpha=0.3, linewidth=1)
    
    # Add quarter labels
    quarter_labels = ['Q1', 'Q2', 'Q3', 'Q4']
    quarter_positions = [6, 18, 30, 42]
    for label, pos in zip(quarter_labels, quarter_positions):
        ax2.text(pos, ax2.get_ylim()[1] * 0.95, label, 
                ha='center', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save figure
    output_file = PROJECT_ROOT / "src" / "pbp_data" / "tmp" / f"monte_carlo_{player_name.replace(' ', '_')}_{game_id}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved plot: {output_file}")
    
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("MONTE CARLO SIMULATION DEMO - LUKA DONCIC")
    print("=" * 80)
    print()
    
    # Config
    player_name = "Luka Doncic"
    prop_line = 30.5
    n_simulations = 1000  # Using 1000 for faster demo
    
    # Load player profile
    print(f"📊 Loading {player_name}'s historical data...")
    profile = load_player_profile(player_name)
    
    print(f"   ✅ Loaded {profile['num_games']} games")
    print(f"   📈 Average: {profile['avg_points_per_game']:.1f} PPG, {profile['avg_minutes_per_game']:.1f} MPG")
    
    # Debug: Check historical performance vs prop line
    con = duckdb.connect()
    actual_hit_rate = con.execute(f"""
        SELECT 
            COUNT(*) FILTER (WHERE total_points > {prop_line}) AS games_over,
            COUNT(*) AS total_games,
            COUNT(*) FILTER (WHERE total_points > {prop_line}) * 100.0 / COUNT(*) AS pct_over
        FROM (
            SELECT 
                game_id,
                MAX(cumulative_points) AS total_points
            FROM '{MINUTE_BY_MINUTE}'
            WHERE player_name = '{player_name}'
            GROUP BY game_id
        )
    """).fetchone()
    con.close()
    
    print(f"   🎯 Historical: {actual_hit_rate[0]}/{actual_hit_rate[1]} games Over {prop_line} ({actual_hit_rate[2]:.1f}%)")
    print()
    
    # Debug: Show quarterly distributions
    print("📊 Quarterly Distributions:")
    for q in [1, 2, 3, 4]:
        minutes = [m for m in profile[f'q{q}_minutes_history'] if m > 0]
        ppm = [p for p in profile[f'q{q}_ppm_history'] if p > 0]
        if minutes and ppm:
            avg_min = sum(minutes) / len(minutes)
            avg_ppm = sum(ppm) / len(ppm)
            avg_pts = avg_min * avg_ppm
            print(f"   Q{q}: {avg_min:.1f} min × {avg_ppm:.3f} PPM = {avg_pts:.1f} pts (avg)")
    print()
    
    # Get games
    print(f"🏀 Finding {player_name}'s games...")
    games = get_player_games(player_name)
    print(f"   ✅ Found {len(games)} games")
    print()
    
    # Pick a game (use most recent game with 25+ points for interesting example)
    interesting_games = games[games['final_points'] >= 25]
    if len(interesting_games) == 0:
        interesting_games = games
    
    game = interesting_games.iloc[0]
    game_id = game['game_id']
    game_date = game['game_date']
    final_points = game['final_points']
    
    print(f"🎯 Selected game: {game_id} on {game_date}")
    print(f"   Final score: {final_points} points")
    print(f"   Prop line: Over/Under {prop_line}")
    print()
    
    # Load minute-by-minute data
    print(f"📈 Loading minute-by-minute data...")
    game_data = load_game_minute_by_minute(player_name, game_id)
    print(f"   ✅ Loaded {len(game_data)} minutes")
    print()
    
    # Run Monte Carlo for each minute
    print(f"🎲 Running Monte Carlo simulation ({n_simulations:,} iterations per minute)...")
    print()
    
    results = []
    
    for idx, row in game_data.iterrows():
        minute = row['minute']
        current_points = row['cumulative_points']
        
        # Run simulation (debug first minute only)
        prob_over = monte_carlo_simulate_bet(
            profile,
            minute,
            current_points,
            prop_line,
            n_simulations,
            debug=(minute == 0)
        )
        
        results.append({
            'minute': minute,
            'prob_over': prob_over,
        })
        
        # Print progress every 5 minutes
        if int(minute) % 5 == 0:
            print(f"   Minute {int(minute):2d}: {current_points:2.0f} pts → {prob_over:.1%} chance of Over {prop_line}")
    
    prob_df = pd.DataFrame(results)
    
    print()
    print(f"✅ Simulation complete!")
    print()
    
    # Summary stats
    final_prob = prob_df.iloc[-1]['prob_over']
    hit_over = final_points > prop_line
    
    print(f"📊 Summary:")
    print(f"   Final points: {final_points}")
    print(f"   Prop line: Over {prop_line}")
    print(f"   Result: {'✅ HIT' if hit_over else '❌ MISS'}")
    print(f"   Final probability: {final_prob:.1%}")
    print()
    
    # Save results to CSV
    print("💾 Saving results to CSV...")
    results_df = game_data.copy()
    results_df['prob_over'] = prob_df['prob_over']
    results_df['prob_under'] = 1 - prob_df['prob_over']
    results_df['points_needed'] = prop_line - results_df['cumulative_points']
    
    # Add metadata columns
    results_df.insert(0, 'player_name', player_name)
    results_df.insert(1, 'game_id', game_id)
    results_df.insert(2, 'game_date', game_date)
    results_df.insert(3, 'prop_line', prop_line)
    
    # Reorder columns for readability
    results_df = results_df[[
        'player_name', 'game_id', 'game_date', 'prop_line',
        'minute', 'cumulative_points', 'minutes_played', 'points_needed',
        'prob_over', 'prob_under'
    ]]
    
    # Save CSV
    output_csv = PROJECT_ROOT / "src" / "pbp_data" / "tmp" / f"monte_carlo_{player_name.replace(' ', '_')}_{game_id}.csv"
    results_df.to_csv(output_csv, index=False, float_format='%.4f')
    print(f"   ✅ Saved: {output_csv}")
    print(f"   📊 {len(results_df)} rows (1 per minute)")
    print()
    
    # Show sample of results
    print("📋 Sample results (first 10 minutes):")
    print()
    print(results_df.head(10).to_string(index=False))
    print()
    print("📋 Sample results (last 10 minutes):")
    print()
    print(results_df.tail(10).to_string(index=False))
    print()
    
    # Plot
    print("📊 Generating plot...")
    plot_monte_carlo_results(game_data, prob_df, prop_line, player_name, game_id, game_date)
    
    print()
    print("=" * 80)
    print("✅ COMPLETE")
    print("=" * 80)
    print()
    print("Output files:")
    print(f"  - CSV:  {output_csv}")
    print(f"  - Plot: {output_csv.with_suffix('.png')}")


if __name__ == "__main__":
    main()

"""
Demo: Monte Carlo simulation for NBA player props with play-by-play updates.

Goal:
1. Load player's historical data (quarterly distributions)
2. Load a game's play-by-play data
3. For each play in the game, run Monte Carlo simulations
4. Plot:
   - Top: Probability of covering Over prop line over time (with smoothing)
   - Bottom: Actual points scored over time (colored by pace line)

Features:
- Uses real consensus prop lines from S3 data
- Vegas adjustment to start at 50% probability
- R/ggplot2 for publication-quality plots
- Team logos and player headshots
- Pace line visualization (green when ahead, red when behind)

Usage:
    # Single game
    python src/pbp_data/tmp/demo_monte_carlo_pbp.py --player-name "Luka Doncic" --game-id 401809820 --n-sims 1000
    
    # All games
    python src/pbp_data/tmp/demo_monte_carlo_pbp.py --player-name "LeBron James" --game-id all --n-sims 1000
    
    # With consensus prop lines
    python src/pbp_data/tmp/demo_monte_carlo_pbp.py --player-name "Luka Doncic" --game-id all --n-sims 1000 --use-consensus
"""

import duckdb
import pandas as pd
import argparse
import sys
from pathlib import Path

# Import all functions from monte_carlo_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pbp_data.monte_carlo_utils import (
    get_project_root,
    get_data_paths,
    load_player_profile,
    get_consensus_prop_line,
    load_play_by_play,
    monte_carlo_simulate_bet,
    find_vegas_adjustment,
    create_ggplot,
)


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = get_project_root()
PATHS = get_data_paths()
MINUTE_BY_MINUTE = PATHS['minute_by_minute']
PLOT_DIR = PROJECT_ROOT / "src" / "pbp_data" / "tmp" / "plots"
PLOT_DIR.mkdir(exist_ok=True, parents=True)


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_game(player_name, game_id, n_sims, use_consensus):
    """Process a single game."""
    print(f"\n{'='*80}")
    print(f"Game: {game_id}")
    print(f"{'='*80}")
    
    # Load player profile
    player_profile = load_player_profile(player_name)
    
    # Load play-by-play data
    print(f"📥 Loading play-by-play data...")
    pbp_df, metadata = load_play_by_play(game_id, player_name)
    
    away_team = metadata['away_team']
    home_team = metadata['home_team']
    game_date = metadata['game_date']
    commence_time_et = metadata['commence_time_et']
    
    print(f"   ✅ {away_team} @ {home_team}")
    if commence_time_et:
        print(f"   🕐 Tipoff: {commence_time_et.strftime('%Y-%m-%d %I:%M %p %Z')}")
    
    # Get prop line
    if use_consensus:
        prop_line = get_consensus_prop_line(player_name, game_date)
        if prop_line:
            print(f"   📊 Consensus prop line: {prop_line}")
        else:
            # Round average to nearest 0.5 (like sportsbooks do)
            avg = player_profile['avg_points_per_game']
            prop_line = round(avg * 2) / 2
            print(f"   ⚠️  No consensus prop line found, using rounded average: {prop_line}")
    else:
        # Round average to nearest 0.5 (like sportsbooks do)
        avg = player_profile['avg_points_per_game']
        prop_line = round(avg * 2) / 2
        print(f"   📊 Using player average as prop line: {prop_line}")
    
    # Find Vegas adjustment (one-time, at game start)
    print(f"   🎲 Calibrating Vegas adjustment...")
    vegas_adjustment = find_vegas_adjustment(player_profile, prop_line, n_simulations=n_sims)
    print(f"   ✅ Vegas adjustment: {vegas_adjustment:.4f}")
    
    # Run Monte Carlo for each play
    print(f"   🎲 Running Monte Carlo simulation ({n_sims:,} iterations per play)...")
    
    results = []
    for idx, row in pbp_df.iterrows():
        game_minute = row['game_minute']
        current_points = row['cumulative_points']
        
        prob_over = monte_carlo_simulate_bet(
            player_profile,
            game_minute,
            current_points,
            prop_line,
            n_simulations=n_sims,
            vegas_adjustment=vegas_adjustment,
            debug=False
        )
        
        results.append({
            'game_minute': game_minute,
            'quarter': row['quarter'],
            'cumulative_points': current_points,
            'description': row['description'],
            'prob_over': prob_over,
        })
    
    results_df = pd.DataFrame(results)
    
    # Determine result
    final_points = int(pbp_df.iloc[-1]['cumulative_points'])
    result = "HIT" if final_points > prop_line else "MISS"
    
    # Save CSV
    player_name_clean = player_name.replace(" ", "_")
    csv_file = PLOT_DIR / f"monte_carlo_pbp_{player_name_clean}_{game_id}_{game_date}.csv"
    
    save_df = results_df.copy()
    save_df.insert(0, 'player_name', player_name)
    save_df.insert(1, 'game_id', game_id)
    save_df.insert(2, 'game_date', game_date)
    save_df.insert(3, 'prop_line', prop_line)
    save_df['final_points'] = final_points
    save_df['result'] = result
    
    save_df.to_csv(csv_file, index=False, float_format='%.4f')
    
    # Create plot
    print(f"   📊 Generating plot...")
    plot_file = create_ggplot(
        results_df,
        prop_line,
        player_name,
        player_profile['player_id'],
        game_id,
        game_date,
        away_team,
        home_team,
        final_points,
        result,
        plot_dir=PLOT_DIR
    )
    
    if plot_file:
        print(f"   💾 Plot saved: {plot_file}")
    
    print(f"   💾 CSV saved: {csv_file}")
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETE")
    print(f"{'='*80}")
    print(f"\n📊 Result: {final_points} pts ({result})")
    print(f"   Starting prob: {results_df.iloc[0]['prob_over']:.1%}")
    print(f"   Final prob: {results_df.iloc[-1]['prob_over']:.1%}")
    
    return {
        'game_id': game_id,
        'game_date': game_date,
        'final_points': final_points,
        'prop_line': prop_line,
        'result': result,
        'starting_prob': results_df.iloc[0]['prob_over'],
        'final_prob': results_df.iloc[-1]['prob_over'],
        'num_plays': len(results_df),
    }


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo simulation for NBA player props")
    parser.add_argument("--player-name", type=str, required=True, help="Player name (e.g., 'Luka Doncic')")
    parser.add_argument("--game-id", type=str, required=True, help="Game ID or 'all'")
    parser.add_argument("--n-sims", type=int, default=1000, help="Number of Monte Carlo simulations per play")
    parser.add_argument("--use-consensus", action="store_true", help="Use consensus prop lines from S3 data")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"MONTE CARLO SIMULATION - {args.player_name}")
    print("=" * 80)
    print()
    
    # Load player profile to get games
    player_profile = load_player_profile(args.player_name)
    
    # Get games
    if args.game_id == "all":
        con = duckdb.connect()
        games_df = con.execute(f"""
            SELECT DISTINCT game_id, game_date
            FROM '{MINUTE_BY_MINUTE}'
            WHERE player_name = '{args.player_name}'
            ORDER BY game_date ASC
        """).df()
        con.close()
        
        game_ids = games_df['game_id'].tolist()
        print(f"📊 Found {len(game_ids)} games for {args.player_name}")
    else:
        game_ids = [args.game_id]
    
    # Process each game
    summaries = []
    for i, game_id in enumerate(game_ids, 1):
        print(f"\n[{i}/{len(game_ids)}]")
        
        try:
            summary = process_game(args.player_name, game_id, args.n_sims, args.use_consensus)
            summaries.append(summary)
        except Exception as e:
            print(f"   ❌ Error processing game {game_id}: {e}")
            continue
    
    # Save summary
    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_file = PLOT_DIR / "monte_carlo_summary.csv"
        summary_df.to_csv(summary_file, index=False, float_format='%.3f')
        print(f"\n💾 Summary saved: {summary_file}")
    
    print(f"\n{'='*80}")
    print("✅ ALL GAMES COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

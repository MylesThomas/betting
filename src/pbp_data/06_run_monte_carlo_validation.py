"""
Script 06: Run Monte Carlo Validation for All Player-Games

Purpose:
- Run Monte Carlo simulations for top N players (by PPG)
- Save ALL play-by-play predictions for Brier score analysis
- Generate visualization plots for each game
- Cache everything to ~/Downloads/tmp/monte_carlo_validation/

Usage:
    # Top 50 players
    python src/pbp_data/06_run_monte_carlo_validation.py --top-n 50 --n-sims 1000
    
    # All players
    python src/pbp_data/06_run_monte_carlo_validation.py --top-n 0 --n-sims 1000
    
    # Resume from specific player (skip already completed)
    python src/pbp_data/06_run_monte_carlo_validation.py --top-n 50 --start-from "Luka Doncic"

Output:
    ~/Downloads/tmp/monte_carlo_validation/predictions.parquet
    ~/Downloads/tmp/monte_carlo_validation/plots/monte_carlo_pbp_*.png
"""

import duckdb
import pandas as pd
import argparse
import sys
from pathlib import Path

# Import functions from monte_carlo_utils
sys.path.insert(0, str(Path(__file__).parent.parent))
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

# Validation cache directory
VALIDATION_DIR = Path.home() / "Downloads" / "tmp" / "monte_carlo_validation"
VALIDATION_DIR.mkdir(exist_ok=True, parents=True)
PREDICTIONS_DIR = VALIDATION_DIR / "predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR = VALIDATION_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True, parents=True)


# =============================================================================
# MAIN VALIDATION LOGIC
# =============================================================================

def get_top_players_by_ppg(top_n=50):
    """Get top N players by PPG."""
    con = duckdb.connect()
    
    query = f"""
    WITH game_stats AS (
        SELECT 
            player_name,
            player_id,
            game_id,
            MAX(cumulative_points) AS game_points
        FROM '{MINUTE_BY_MINUTE}'
        GROUP BY player_name, player_id, game_id
    )
    SELECT 
        player_name,
        player_id,
        COUNT(DISTINCT game_id) AS num_games,
        AVG(game_points) AS avg_ppg
    FROM game_stats
    GROUP BY player_name, player_id
    HAVING num_games >= 5
    ORDER BY avg_ppg DESC
    """
    
    if top_n > 0:
        query += f" LIMIT {top_n}"
    
    df = con.execute(query).df()
    con.close()
    
    return df


def get_player_games(player_name):
    """Get all games for a player, sorted chronologically."""
    con = duckdb.connect()
    
    query = f"""
    SELECT DISTINCT game_id, game_date
    FROM '{MINUTE_BY_MINUTE}'
    WHERE player_name = ?
    ORDER BY game_date ASC
    """
    
    df = con.execute(query, [player_name]).df()
    con.close()
    
    return df


def get_prediction_filename(player_name, game_id):
    """Get filename for a player-game prediction file."""
    player_name_clean = player_name.replace(" ", "_").replace("'", "")
    return PREDICTIONS_DIR / f"{player_name_clean}_{game_id}.parquet"


def save_predictions(predictions_df, player_name, game_id):
    """Save predictions to individual parquet file per player-game."""
    prediction_file = get_prediction_filename(player_name, game_id)
    predictions_df.to_parquet(prediction_file, index=False)


def check_game_already_processed(player_name, game_id):
    """Check if a player-game combination has already been processed."""
    prediction_file = get_prediction_filename(player_name, game_id)
    return prediction_file.exists()


def process_player_game(player_name, game_id, player_profile, n_sims):
    """Process a single player-game and return predictions DataFrame."""
    try:
        # Load play-by-play data
        pbp_df, metadata = load_play_by_play(game_id, player_name)
        
        away_team = metadata['away_team']
        home_team = metadata['home_team']
        game_date = metadata['game_date']
        
        # Get prop line (use consensus if available, otherwise use average)
        prop_line = get_consensus_prop_line(player_name, game_date)
        if prop_line is None:
            avg = player_profile['avg_points_per_game']
            prop_line = round(avg * 2) / 2
        
        # Find Vegas adjustment
        vegas_adjustment = find_vegas_adjustment(player_profile, prop_line, n_simulations=n_sims)
        
        # Run Monte Carlo for each play
        predictions = []
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
            
            predictions.append({
                'player_name': player_name,
                'player_id': player_profile['player_id'],
                'game_id': game_id,
                'game_date': game_date,
                'away_team': away_team,
                'home_team': home_team,
                'prop_line': prop_line,
                'vegas_adjustment': vegas_adjustment,
                'play_id': row['play_id'],
                'quarter': row['quarter'],
                'game_minute': game_minute,
                'cumulative_points': current_points,
                'prob_over': prob_over,
            })
        
        # Add final result to all rows
        final_points = int(pbp_df.iloc[-1]['cumulative_points'])
        result = "HIT" if final_points > prop_line else "MISS"
        
        predictions_df = pd.DataFrame(predictions)
        predictions_df['final_points'] = final_points
        predictions_df['result'] = result
        
        # Generate plot
        plot_file = create_ggplot(
            predictions_df[['game_minute', 'cumulative_points', 'prob_over', 'quarter']],
            prop_line,
            player_name,
            player_profile['player_id'],
            game_id,
            game_date,
            away_team,
            home_team,
            final_points,
            result,
            plot_dir=PLOTS_DIR
        )
        
        return predictions_df
        
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run Monte Carlo validation")
    parser.add_argument("--top-n", type=int, default=50, help="Top N players by PPG (0 = all)")
    parser.add_argument("--n-sims", type=int, default=1000, help="Number of simulations per play")
    parser.add_argument("--start-from", type=str, default=None, help="Start from specific player")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MONTE CARLO VALIDATION - BATCH PROCESSING")
    print("=" * 80)
    print(f"\nMode: {'All players' if args.top_n == 0 else f'Top {args.top_n} players by PPG'}")
    print(f"Simulations per play: {args.n_sims:,}")
    print(f"Output: {PREDICTIONS_DIR}")
    print(f"Plots: {PLOTS_DIR}")
    print()
    
    # Get top players
    players_df = get_top_players_by_ppg(top_n=args.top_n)
    print(f"📊 Processing {len(players_df)} players")
    print(f"\nTop 10:")
    print(players_df.head(10)[['player_name', 'avg_ppg', 'num_games']].to_string(index=False))
    print()
    
    # Filter players if starting from a specific one
    if args.start_from:
        start_idx = players_df[players_df['player_name'] == args.start_from].index
        if len(start_idx) > 0:
            players_df = players_df.iloc[start_idx[0]:].reset_index(drop=True)
            print(f"▶️  Resuming from: {args.start_from}")
            print(f"   Remaining players: {len(players_df)}")
        else:
            print(f"⚠️  Player '{args.start_from}' not found in top {args.top_n}, processing all")
    
    # Main processing loop
    total_games_processed = 0
    total_games_skipped = 0
    
    for player_idx, player_row in players_df.iterrows():
        player_name = player_row['player_name']
        
        print(f"\n{'='*80}")
        print(f"[{player_idx + 1}/{len(players_df)}] {player_name}")
        print(f"{'='*80}")
        print(f"   PPG: {player_row['avg_ppg']:.1f} | Games: {player_row['num_games']}")
        
        try:
            # Load player profile
            player_profile = load_player_profile(player_name)
            
            # Get all games for this player
            games_df = get_player_games(player_name)
            print(f"   📅 Found {len(games_df)} games")
            
            # Process each game
            for game_idx, game_row in games_df.iterrows():
                game_id = game_row['game_id']
                game_date = game_row['game_date']
                
                # Check if already processed
                if check_game_already_processed(player_name, game_id):
                    print(f"      ⏭️  Game {game_id} ({game_date}) - already processed")
                    total_games_skipped += 1
                    continue
                
                print(f"      🎲 Processing game {game_id} ({game_date})...")
                
                # Process this game
                predictions_df = process_player_game(
                    player_name, 
                    game_id, 
                    player_profile, 
                    args.n_sims
                )
                
                if predictions_df is not None:
                    # Save predictions
                    save_predictions(predictions_df, player_name, game_id)
                    total_games_processed += 1
                    print(f"      ✅ Saved {len(predictions_df)} predictions")
                
        except Exception as e:
            print(f"   ❌ Error processing player {player_name}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("✅ VALIDATION BATCH COMPLETE")
    print(f"{'='*80}")
    print(f"   Games processed: {total_games_processed}")
    print(f"   Games skipped (already done): {total_games_skipped}")
    print(f"   Predictions directory: {PREDICTIONS_DIR}")
    print(f"   Total prediction files: {len(list(PREDICTIONS_DIR.glob('*.parquet')))}")
    print(f"   Plots directory: {PLOTS_DIR}")


if __name__ == "__main__":
    main()

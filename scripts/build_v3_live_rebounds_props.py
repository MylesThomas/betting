"""
Build v3-shaped parquet from live player_rebounds props for prod scoring.

Reads live CSV (from fetch_nba_player_rebounds_live.py), resolves game_id via NBA schedule,
and runs the same canonical line logic as v2_build_rebounds_universe.

Usage:
    python scripts/build_v3_live_rebounds_props.py \
        --live-csv data/live_props/2026-03-24/20260324T230914Z.csv \
        --output data/live_props/2026-03-24/v3_live.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from nba_schedule_utils import resolve_game_id, get_schedule_for_date
from player_team_history.name_normalization import normalize_from_odds_api

# Import v2 logic
sys.path.insert(0, str(project_root))
from src.nba_rebounds_modeling.00_research.scripts.v2_build_rebounds_universe import (
    build_market_panel,
    build_v3_props_raw,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_args():
    parser = argparse.ArgumentParser(description="Build v3 live rebounds props")
    parser.add_argument('--live-csv', type=str, required=True, help="Path or s3:// URI to live props CSV")
    parser.add_argument('--output', type=str, required=True, help="Output parquet path")
    parser.add_argument('--date', type=str, help="Target date YYYY-MM-DD (inferred from CSV if not provided)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    logging.info(f"Reading live props from {args.live_csv}")
    if args.live_csv.startswith("s3://"):
        import boto3
        from io import BytesIO
        bucket, key = args.live_csv.replace("s3://", "").split("/", 1)
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(BytesIO(obj['Body'].read()))
    else:
        df = pd.read_csv(args.live_csv)
        
    if df.empty:
        logging.warning("Live CSV is empty. Exiting.")
        sys.exit(0)
        
    # Infer date if not provided
    if args.date:
        target_date = args.date
    else:
        # Parse game_time to ET date
        df['game_time_dt'] = pd.to_datetime(df['game_time'])
        target_date = df['game_time_dt'].dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d').mode()[0]
        
    logging.info(f"Target date: {target_date}")
    
    # 1. Prepare props dataframe for v2 functions
    props = df.copy()
    props['player_normalized'] = props['player'].apply(normalize_from_odds_api)
    props['date'] = target_date
    props['line'] = pd.to_numeric(props['prop_line'], errors='coerce')
    props['odds_over'] = pd.to_numeric(props['over_odds'], errors='coerce')
    props['odds_under'] = pd.to_numeric(props['under_odds'], errors='coerce')
    
    # 2. Resolve game_id for each row
    logging.info("Resolving game_ids...")
    schedule_df = get_schedule_for_date(target_date)
    
    # Cache resolutions to avoid redundant lookups
    game_id_cache = {}
    
    def get_game_id(row):
        matchup_key = f"{row['home_team']}_{row['away_team']}"
        if matchup_key not in game_id_cache:
            gid = resolve_game_id(row['home_team'], row['away_team'], target_date, schedule_df)
            game_id_cache[matchup_key] = gid
        return game_id_cache[matchup_key]
        
    props['game_id'] = props.apply(get_game_id, axis=1)
    
    missing_gids = props[props['game_id'].isna()]
    if not missing_gids.empty:
        logging.warning(f"Could not resolve game_id for {len(missing_gids)} rows (e.g. {missing_gids.iloc[0]['home_team']} vs {missing_gids.iloc[0]['away_team']})")
        props = props.dropna(subset=['game_id'])
        
    if props.empty:
        logging.error("No rows left after dropping missing game_ids.")
        sys.exit(1)
        
    # 3. Build logs stub
    # v2 functions need: season, date, player_normalized, game_id, REB
    logs_stub = props[['season', 'date', 'player_normalized', 'game_id']].drop_duplicates().copy()
    logs_stub['REB'] = np.nan  # Pre-game, outcome is unknown
    
    # 4. Run v2 logic
    logging.info("Building market panel...")
    panel, book_line = build_market_panel(props, logs_stub)
    
    logging.info("Building v3 props raw...")
    v3_raw = build_v3_props_raw(book_line, logs_stub, panel)
    
    # 5. Save output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    v3_raw.to_parquet(out_path, index=False)
    
    logging.info(f"Wrote v3 live parquet to {out_path} ({len(v3_raw)} rows)")


if __name__ == "__main__":
    main()

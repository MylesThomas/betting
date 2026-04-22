"""
Fetch Live NBA Player Rebounds Props

Fetches player_rebounds market for the current (or upcoming) NBA slate from The Odds API.
Saves raw JSONL and normalized CSV to S3.

Usage:
    python scripts/fetch_nba_player_rebounds_live.py --s3
    python scripts/fetch_nba_player_rebounds_live.py --date 2026-03-17 --s3
    python scripts/fetch_nba_player_rebounds_live.py --dry-run
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

import ssl
import urllib3

# ============================================================================
# SSL FIX FOR MACOS
# ============================================================================
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests with timeout
original_request = requests.Session.request
def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    kwargs.setdefault('timeout', 10)
    return original_request(self, *args, **kwargs)
requests.Session.request = patched_request

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from odds_api_parser import median_home_away_spreads_from_event, parse_player_props
from season_utils import get_current_nba_season

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
API_KEY = os.getenv('ODDS_API_KEY') or os.getenv('THE_ODDS_API_KEY')
BASE_URL = 'https://api.the-odds-api.com/v4'
SPORT_KEY = 'basketball_nba'
MARKET = 'player_rebounds'
ODDS_FORMAT = 'american'
REGION = 'us'

S3_BUCKET = 'the-odds-api-mt'
S3_PREFIX_RAW = 'nba/live_player_props/player_rebounds/raw'
S3_PREFIX_CSV = 'nba/live_player_props/player_rebounds/csv'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_args():
    parser = argparse.ArgumentParser(description="Fetch live NBA rebounds props")
    parser.add_argument('--date', type=str, help="Target date YYYY-MM-DD (defaults to today ET)")
    parser.add_argument('--s3', action='store_true', help="Upload to S3")
    parser.add_argument('--dry-run', action='store_true', help="Print stats, do not upload")
    parser.add_argument('--season', type=str, help="Override season (e.g., 2025-26)")
    parser.add_argument('--output-csv', type=str, default="", help="Optional explicit local CSV output path")
    return parser.parse_args()


def get_upcoming_events() -> list:
    """Fetch all upcoming events."""
    endpoint = f"sports/{SPORT_KEY}/events"
    params = {
        'apiKey': API_KEY,
    }
    
    logging.info(f"Fetching upcoming events from {endpoint}...")
    response = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
    if response.status_code != 200:
        logging.error(f"API Error: {response.text}")
    response.raise_for_status()
    
    return response.json()


def get_event_odds(event_id: str) -> dict:
    """Fetch live odds for a specific event."""
    endpoint = f"sports/{SPORT_KEY}/events/{event_id}/odds"
    params = {
        'apiKey': API_KEY,
        'regions': REGION,
        # spreads: same quota as extra market; needed before historical_game_lines CSV exists for slate date
        'markets': f'{MARKET},spreads',
        'oddsFormat': ODDS_FORMAT
    }
    
    import time
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
            if response.status_code == 422:
                # Some games might not have player props posted yet
                logging.warning(f"Event {event_id} has no odds for {MARKET} yet.")
                return None
            if response.status_code != 200:
                logging.error(f"API Error for event {event_id}: {response.text}")
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            if attempt < max_retries - 1:
                logging.warning(f"Connection/Timeout error for event {event_id}, retrying... ({e})")
                time.sleep(2)
            else:
                raise


def filter_events_by_date(events: list, target_date: str) -> list:
    """Filter events to only those commencing on the target date (ET)."""
    filtered = []
    for event in events:
        commence_time = event.get('commence_time')
        if not commence_time:
            continue
            
        # Parse UTC time and convert to ET to match date
        dt_utc = pd.to_datetime(commence_time)
        dt_et = dt_utc.tz_convert('America/New_York')
        event_date = dt_et.strftime('%Y-%m-%d')
        
        if event_date == target_date:
            filtered.append(event)
            
    return filtered


def write_to_s3(bucket: str, key: str, body: str):
    s3 = boto3.client('s3')
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    logging.info(f"Uploaded to s3://{bucket}/{key}")


def main():
    args = parse_args()
    
    if not API_KEY:
        logging.error("API_KEY not found in environment")
        sys.exit(1)
        
    # Determine target date
    if args.date:
        target_date = args.date
    else:
        # Default to today in ET
        target_date = pd.Timestamp.now(tz='America/New_York').strftime('%Y-%m-%d')
        
    season = args.season or get_current_nba_season()
    fetch_ts_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    
    logging.info(f"Target Date: {target_date} | Season: {season} | Fetch TS: {fetch_ts_utc}")
    
    # 1. Fetch all upcoming events
    try:
        all_events = get_upcoming_events()
    except Exception as e:
        logging.error(f"Failed to fetch events: {e}")
        sys.exit(1)
        
    # 2. Filter to target date
    target_events = filter_events_by_date(all_events, target_date)
    logging.info(f"Found {len(target_events)} events for {target_date} out of {len(all_events)} total upcoming")
    
    if not target_events:
        logging.warning("No events found for target date. Exiting.")
        sys.exit(0)
        
    # 3. Fetch odds for each target event
    odds_responses = []
    for event in target_events:
        event_id = event['id']
        logging.info(f"Fetching odds for event {event_id} ({event['home_team']} vs {event['away_team']})")
        odds_data = get_event_odds(event_id)
        if odds_data:
            odds_responses.append(odds_data)
        time.sleep(0.6)  # Rate limit
        
    if not odds_responses:
        logging.warning("No odds data returned for any events. Exiting.")
        sys.exit(0)
        
    # 4. Parse props
    props_list = parse_player_props(odds_responses, target_market=MARKET)
    logging.info(f"Parsed {len(props_list)} prop rows")
    
    if not props_list:
        logging.warning("No props found in the events. Exiting.")
        sys.exit(0)
        
    # 4. Convert to DataFrame and format like historical CSV
    df = pd.DataFrame(props_list)
    df['fetch_date'] = fetch_ts_utc
    df['season'] = season

    df["home_spread_line"] = np.nan
    df["away_spread_line"] = np.nan
    for ev in odds_responses:
        eid = str(ev.get("id") or "")
        if not eid:
            continue
        hs, aws = median_home_away_spreads_from_event(ev)
        mask = df["odds_api_event_id"].astype(str) == eid
        if hs is not None:
            df.loc[mask, "home_spread_line"] = hs
        if aws is not None:
            df.loc[mask, "away_spread_line"] = aws
    
    # Reorder columns to match historical if possible
    col_order = [
        'player', 'away_team', 'home_team', 'game_time', 'market', 
        'prop_line', 'over_odds', 'under_odds', 'bookmaker', 
        'bookmaker_last_update', 'market_last_update', 'fetch_date', 'season',
        'home_spread_line', 'away_spread_line',
        'odds_api_event_id'
    ]
    # Ensure all columns exist
    for col in col_order:
        if col not in df.columns:
            df[col] = None
            
    df = df[col_order]
    
    if args.dry_run:
        logging.info("DRY RUN: Sample of parsed props:")
        print(df.head().to_string())
        return
        
    if args.s3:
        # A. Write raw JSONL
        raw_key = f"{S3_PREFIX_RAW}/{season}/{target_date}/{fetch_ts_utc}.jsonl"
        raw_body = "\n".join(json.dumps(e) for e in target_events)
        write_to_s3(S3_BUCKET, raw_key, raw_body)
        
        # B. Write CSV (versioned)
        csv_key = f"{S3_PREFIX_CSV}/{season}/{target_date}/runs/{fetch_ts_utc}.csv"
        csv_body = df.to_csv(index=False)
        write_to_s3(S3_BUCKET, csv_key, csv_body)
        
        # C. Write CSV (latest)
        latest_key = f"{S3_PREFIX_CSV}/{season}/{target_date}/latest.csv"
        write_to_s3(S3_BUCKET, latest_key, csv_body)
        
        logging.info("S3 upload complete.")
    if args.output_csv:
        out_path = Path(args.output_csv).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logging.info(f"Saved locally to {out_path}")
    elif not args.s3:
        # Local save for testing
        out_dir = project_root / 'live_props_tmp' / target_date
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{fetch_ts_utc}.csv"
        df.to_csv(out_path, index=False)
        logging.info(f"Saved locally to {out_path}")


if __name__ == "__main__":
    main()

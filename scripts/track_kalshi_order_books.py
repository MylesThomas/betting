"""
Track Kalshi Prediction Market Order Books Hourly

[... keeping full docstring from before ...]
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
import os
import requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import ssl
import urllib3
from typing import Optional, Tuple, Dict, List
import math
import boto3
from io import StringIO
import json
import statistics

# Load environment variables
load_dotenv()

# Fix SSL certificate issues (for API calls)
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Add src to path by finding project root (look for .gitignore)
def find_project_root() -> Path:
    """Find project root by looking for .gitignore file."""
    # In Lambda, we don't need project root - everything is in /var/task
    if 'AWS_LAMBDA_FUNCTION_NAME' in os.environ:
        return Path('/var/task')
    
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / '.gitignore').exists():
            return parent
    
    # Fallback to current directory if .gitignore not found
    return current

PROJECT_ROOT = find_project_root()
# Only add src to path if it exists (won't exist in Lambda)
src_path = PROJECT_ROOT / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================

# Display timezone
DISPLAY_TIMEZONE = 'America/New_York'  # Eastern Time for logging

# Kalshi API Configuration
KALSHI_API_BASE = 'https://api.elections.kalshi.com/trade-api/v2'

# Time windows for comparison
LOOKBACK_WINDOW_1H = timedelta(hours=1)
LOOKBACK_WINDOW_24H = timedelta(hours=24)
LOOKBACK_WINDOW_168H = timedelta(hours=168)  # 1 week
SNAPSHOT_TIME_TOLERANCE = timedelta(minutes=5)  # Allow 5min variance when finding snapshots

# Market selection criteria
MIN_VOLUME = 100000  # Minimum 100K contracts traded
EXCLUDED_CATEGORIES = ['sports', 'nba', 'nfl', 'mlb', 'nhl']  # Avoid sports markets

# AWS Configuration
S3_BUCKET = os.getenv('S3_BUCKET', 'kalshi-order-book-snapshots')
SNS_TOPIC_ARN = os.getenv('SNS_TOPIC_ARN', '')
IS_LAMBDA = 'AWS_LAMBDA_FUNCTION_NAME' in os.environ

# Initialize boto3 clients
s3_client = boto3.client('s3')
sns_client = boto3.client('sns') if IS_LAMBDA else None

# Signal thresholds (Phase 3 - market-relative percentiles)
BASELINE_HOURS_REQUIRED = 48  # Need 48h of data before generating signals
IMBALANCE_PERCENTILE_HIGH = 85  # Top 15% = extreme
IMBALANCE_PERCENTILE_LOW = 15   # Bottom 15% = extreme
DEPTH_RATIO_PERCENTILE_HIGH = 85
DEPTH_RATIO_PERCENTILE_LOW = 15

# Timestamp format for filenames
TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'

# S3 Paths
MARKETS_CONFIG_KEY = 'config/tracked_markets.json'

# =============================================================================
# S3 HELPER FUNCTIONS
# =============================================================================

def get_s3_snapshot_key(market_ticker: str, timestamp: datetime) -> str:
    """Generate S3 key for order book snapshot."""
    timestamp_str = timestamp.strftime(TIMESTAMP_FORMAT)
    return f"data/01_input/kalshi/order_books/{market_ticker}_{timestamp_str}.json"


def get_s3_baseline_key(market_ticker: str) -> str:
    """Generate S3 key for market baseline."""
    return f"data/04_output/kalshi/market_baselines/{market_ticker}_baseline.json"


def list_s3_snapshots(market_ticker: str) -> List[str]:
    """
    List all snapshot files in S3 for a given market.
    
    Returns:
        List of S3 keys sorted by timestamp
    """
    prefix = f"data/01_input/kalshi/order_books/{market_ticker}_"
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=prefix
        )
        
        if 'Contents' not in response:
            return []
        
        return sorted([obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.json')])
    except Exception as e:
        print(f"Warning: Failed to list S3 snapshots for {market_ticker}: {e}")
        return []


def save_json_to_s3(data: dict, s3_key: str) -> bool:
    """Save JSON data to S3."""
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=json.dumps(data, indent=2)
        )
        return True
    except Exception as e:
        print(f"Error: Failed to save to S3: {s3_key}")
        print(f"   {e}")
        return False


def load_json_from_s3(s3_key: str) -> Optional[dict]:
    """Load JSON data from S3."""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception as e:
        print(f"Warning: Failed to load from S3: {s3_key}")
        print(f"   {e}")
        return None


def find_snapshot_near_time_s3(market_ticker: str, target_time: datetime) -> Optional[str]:
    """
    Find snapshot S3 key closest to target_time within SNAPSHOT_TIME_TOLERANCE.
    
    Returns:
        S3 key (path) or None if not found
    """
    all_snapshots = list_s3_snapshots(market_ticker)
    
    if not all_snapshots:
        return None
    
    best_key = None
    best_diff = None
    
    for s3_key in all_snapshots:
        # Extract timestamp from key: data/.../TICKER_20241224_120000.json
        filename = s3_key.split('/')[-1]
        parts = filename.replace('.json', '').split('_')
        if len(parts) < 3:
            continue
            
        timestamp_str = f"{parts[-2]}_{parts[-1]}"
        
        try:
            file_time = datetime.strptime(timestamp_str, TIMESTAMP_FORMAT)
            # Make timezone-aware
            if target_time.tzinfo:
                file_time = file_time.replace(tzinfo=timezone.utc)
            
            time_diff = abs(file_time - target_time)
            
            if time_diff <= SNAPSHOT_TIME_TOLERANCE:
                if best_diff is None or time_diff < best_diff:
                    best_diff = time_diff
                    best_key = s3_key
        except ValueError:
            continue
    
    return best_key


# =============================================================================
# MARKET CONFIG MANAGEMENT
# =============================================================================

def load_markets_config() -> dict:
    """
    Load tracked markets config from S3.
    
    Returns:
        Dict with 'markets' list and metadata, or empty structure if not found
    """
    config = load_json_from_s3(MARKETS_CONFIG_KEY)
    
    if config is None:
        # Initialize empty config
        config = {
            'markets': [],
            'last_updated': datetime.now(timezone.utc).astimezone(ZoneInfo(DISPLAY_TIMEZONE)).strftime('%Y-%m-%d %H:%M:%S ET')
        }
        save_json_to_s3(config, MARKETS_CONFIG_KEY)
    
    return config


def save_markets_config(config: dict):
    """Save markets config to S3."""
    config['last_updated'] = datetime.now(timezone.utc).astimezone(ZoneInfo(DISPLAY_TIMEZONE)).strftime('%Y-%m-%d %H:%M:%S ET')
    save_json_to_s3(config, MARKETS_CONFIG_KEY)


def add_market_to_config(market_ticker: str, category: str, initial_volume: int, initial_price: float):
    """
    Add a new market to tracked config (if not already present).
    
    Returns:
        True if added, False if already exists
    """
    config = load_markets_config()
    
    # Check if already tracked
    existing_tickers = {m['ticker'] for m in config['markets']}
    if market_ticker in existing_tickers:
        return False
    
    # Add new market
    now_et = datetime.now(timezone.utc).astimezone(ZoneInfo(DISPLAY_TIMEZONE))
    config['markets'].append({
        'ticker': market_ticker,
        'date_added': now_et.strftime('%Y-%m-%d %H:%M:%S ET'),
        'category': category,
        'initial_volume': initial_volume,
        'initial_price': initial_price,
        'active': True
    })
    
    save_markets_config(config)
    return True


# =============================================================================
# KALSHI API FUNCTIONS
# =============================================================================

def fetch_kalshi_markets(limit: int = 100) -> Optional[List[dict]]:
    """
    Fetch available markets from Kalshi API.
    
    Returns:
        List of market dicts or None on error
    """
    url = f"{KALSHI_API_BASE}/markets"
    params = {
        'limit': limit,
        'status': 'open'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('markets', [])
    except Exception as e:
        print(f"Error fetching Kalshi markets: {e}")
        return None


def fetch_order_book(market_ticker: str) -> Optional[dict]:
    """
    Fetch full order book for a specific market.
    
    Returns:
        Order book data or None on error
    """
    url = f"{KALSHI_API_BASE}/markets/{market_ticker}/orderbook"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching order book for {market_ticker}: {e}")
        return None


# =============================================================================
# ORDER BOOK ANALYSIS
# =============================================================================

def analyze_order_book(order_book_data: dict) -> Optional[dict]:
    """
    Calculate metrics from full order book distribution.
    
    Returns:
        Dict with imbalance, depth ratio, and other metrics, or None if invalid data
    """
    if not order_book_data or 'orderbook' not in order_book_data:
        return None
    
    book = order_book_data['orderbook']
    yes_orders = book.get('yes', [])  # Format: [[price_cents, size], ...]
    no_orders = book.get('no', [])
    
    if not yes_orders or not no_orders:
        return None
    
    # Best bid/ask (YES side = buying event, NO side = selling event)
    best_yes_price = yes_orders[-1][0] / 100  # Highest YES price
    best_yes_size = yes_orders[-1][1]
    best_no_price = no_orders[-1][0] / 100   # Highest NO price  
    best_no_size = no_orders[-1][1]
    
    # Mid price (YES + NO should ≈ 1.0)
    mid_price = best_yes_price
    spread = abs(1.0 - best_yes_price - best_no_price)
    
    # Top-of-book imbalance (Stoikov's key signal)
    total_top = best_yes_size + best_no_size
    bid_imbalance = best_yes_size / total_top if total_top > 0 else 0.5
    
    # Total depth across all levels
    total_yes_depth = sum(size for _, size in yes_orders)
    total_no_depth = sum(size for _, size in no_orders)
    total_depth = total_yes_depth + total_no_depth
    
    # Depth ratio (for detecting hidden support/resistance)
    depth_ratio = total_yes_depth / total_no_depth if total_no_depth > 0 else 1.0
    
    # Weighted average prices
    weighted_yes_price = sum(p/100 * s for p, s in yes_orders) / total_yes_depth if total_yes_depth > 0 else mid_price
    weighted_no_price = sum(p/100 * s for p, s in no_orders) / total_no_depth if total_no_depth > 0 else (1.0 - mid_price)
    
    return {
        'mid_price': round(mid_price, 4),
        'spread': round(spread, 4),
        'best_yes_price': round(best_yes_price, 4),
        'best_yes_size': best_yes_size,
        'best_no_price': round(best_no_price, 4),
        'best_no_size': best_no_size,
        'bid_imbalance': round(bid_imbalance, 4),
        'total_yes_depth': total_yes_depth,
        'total_no_depth': total_no_depth,
        'total_depth': total_depth,
        'depth_ratio': round(depth_ratio, 4),
        'weighted_yes_price': round(weighted_yes_price, 4),
        'weighted_no_price': round(weighted_no_price, 4),
        'num_yes_levels': len(yes_orders),
        'num_no_levels': len(no_orders),
    }


def save_order_book_snapshot(market_ticker: str, order_book_data: dict, metrics: dict, timestamp: datetime) -> bool:
    """
    Save order book snapshot to S3.
    
    Stores full order book + calculated metrics for later analysis.
    """
    s3_key = get_s3_snapshot_key(market_ticker, timestamp)
    
    snapshot = {
        'timestamp': timestamp.isoformat(),
        'market_ticker': market_ticker,
        'order_book': order_book_data,
        'metrics': metrics
    }
    
    return save_json_to_s3(snapshot, s3_key)


# =============================================================================
# BASELINE MANAGEMENT (Phase 3)
# =============================================================================

def load_baseline(market_ticker: str) -> Optional[dict]:
    """Load market-specific baseline if it exists."""
    s3_key = get_s3_baseline_key(market_ticker)
    return load_json_from_s3(s3_key)


def update_baseline(market_ticker: str, current_metrics: dict):
    """
    Update rolling 48h baseline for a market.
    
    Tracks percentiles of key metrics to enable market-relative signal detection.
    """
    s3_key = get_s3_baseline_key(market_ticker)
    baseline = load_json_from_s3(s3_key)
    
    if baseline is None:
        # Initialize new baseline
        baseline = {
            'market_ticker': market_ticker,
            'first_seen': datetime.now(timezone.utc).isoformat(),
            'samples': []
        }
    
    # Add current sample
    baseline['samples'].append({
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'metrics': {
            'bid_imbalance': current_metrics.get('bid_imbalance', 0.5),
            'depth_ratio': current_metrics.get('depth_ratio', 1.0),
            'spread': current_metrics.get('spread', 0.02),
            'total_depth': current_metrics.get('total_depth', 0),
        }
    })
    
    # Keep only last 48h
    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
    baseline['samples'] = [
        s for s in baseline['samples']
        if datetime.fromisoformat(s['timestamp']) > cutoff
    ]
    
    # Calculate statistics
    if baseline['samples']:
        hours_of_data = len(baseline['samples']) / 12  # Assuming hourly snapshots
        baseline['hours_of_data'] = hours_of_data
        baseline['last_updated'] = datetime.now(timezone.utc).isoformat()
        baseline['ready_for_alerts'] = hours_of_data >= BASELINE_HOURS_REQUIRED
        
        # Calculate percentiles for each metric
        def calc_percentiles(values):
            if not values:
                return {}
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            return {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std': statistics.stdev(values) if n > 1 else 0,
                'p15': sorted_vals[max(0, int(n * 0.15))] if n > 0 else 0,
                'p85': sorted_vals[min(n-1, int(n * 0.85))] if n > 0 else 0,
                'p10': sorted_vals[max(0, int(n * 0.10))] if n > 0 else 0,
                'p90': sorted_vals[min(n-1, int(n * 0.90))] if n > 0 else 0,
            }
        
        # Extract metric arrays
        imbalances = [s['metrics']['bid_imbalance'] for s in baseline['samples']]
        depth_ratios = [s['metrics']['depth_ratio'] for s in baseline['samples']]
        spreads = [s['metrics']['spread'] for s in baseline['samples']]
        depths = [s['metrics']['total_depth'] for s in baseline['samples']]
        
        baseline['metrics_stats'] = {
            'bid_imbalance': calc_percentiles(imbalances),
            'depth_ratio': calc_percentiles(depth_ratios),
            'spread': calc_percentiles(spreads),
            'total_depth': calc_percentiles(depths),
        }
    
    # Save updated baseline
    save_json_to_s3(baseline, s3_key)
    return baseline


# =============================================================================
# SIGNAL DETECTION
# =============================================================================

def check_signals(market_ticker: str, current_metrics: dict, baseline: dict) -> Optional[List[dict]]:
    """
    Check all signals for a market (Phase 3 only - requires baseline).
    
    Returns:
        List of detected signals, or None if insufficient baseline data
    """
    if not baseline or not baseline.get('ready_for_alerts', False):
        return None
    
    stats = baseline['metrics_stats']
    signals = []
    
    # 1. Order Book Imbalance Signal
    current_imbalance = current_metrics['bid_imbalance']
    imbalance_stats = stats['bid_imbalance']
    
    if current_imbalance >= imbalance_stats['p85']:
        signals.append({
            'type': 'SMART_MONEY_BID',
            'signal': 'FOLLOW_UP',
            'value': current_imbalance,
            'percentile': 85,
            'reason': f"Imbalance {current_imbalance:.2f} (top 15% for this market) - Smart money buying"
        })
    elif current_imbalance <= imbalance_stats['p15']:
        signals.append({
            'type': 'SMART_MONEY_ASK',
            'signal': 'FOLLOW_DOWN',
            'value': current_imbalance,
            'percentile': 15,
            'reason': f"Imbalance {current_imbalance:.2f} (bottom 15% for this market) - Smart money selling"
        })
    
    # 2. Depth Ratio Signal
    current_depth_ratio = current_metrics['depth_ratio']
    depth_stats = stats['depth_ratio']
    
    if current_depth_ratio >= depth_stats['p85']:
        signals.append({
            'type': 'DEEP_BID_SUPPORT',
            'signal': 'FOLLOW_UP',
            'value': current_depth_ratio,
            'percentile': 85,
            'reason': f"Depth ratio {current_depth_ratio:.2f}x (top 15%) - Strong bid support"
        })
    elif current_depth_ratio <= depth_stats['p15']:
        signals.append({
            'type': 'DEEP_ASK_RESISTANCE',
            'signal': 'FOLLOW_DOWN',
            'value': current_depth_ratio,
            'percentile': 15,
            'reason': f"Depth ratio {current_depth_ratio:.2f}x (bottom 15%) - Strong ask resistance"
        })
    
    return signals if signals else []


def analyze_signal_alignment(signals: List[dict]) -> dict:
    """
    Analyze how signals align (all bullish, all bearish, or conflicting).
    
    Returns:
        Dict with alignment type and conviction level
    """
    if not signals:
        return {'alignment': 'NONE', 'conviction': 'NONE'}
    
    # Count directional signals
    bullish = sum(1 for s in signals if 'UP' in s['signal'])
    bearish = sum(1 for s in signals if 'DOWN' in s['signal'])
    
    if bullish > 0 and bearish == 0:
        conviction = 'HIGH' if len(signals) >= 2 else 'MODERATE'
        return {'alignment': 'BULLISH', 'conviction': conviction}
    elif bearish > 0 and bullish == 0:
        conviction = 'HIGH' if len(signals) >= 2 else 'MODERATE'
        return {'alignment': 'BEARISH', 'conviction': conviction}
    elif bullish > 0 and bearish > 0:
        return {'alignment': 'CONFLICTING', 'conviction': 'LOW'}
    else:
        return {'alignment': 'NEUTRAL', 'conviction': 'NONE'}


# =============================================================================
# EMAIL FORMATTING
# =============================================================================

def format_market_signals_text(market_ticker: str, current_metrics: dict, signals: List[dict], alignment: dict) -> str:
    """Format a single market's signals as plain text for email."""
    lines = []
    
    # Market header
    lines.append(f"\n{market_ticker}")
    lines.append(f"  Price: {current_metrics['mid_price']:.2f}")
    lines.append(f"  Spread: {current_metrics['spread']:.4f}")
    lines.append("")
    
    # Show each signal
    for signal in signals:
        emoji = "🔺" if 'UP' in signal['signal'] else "🔻"
        lines.append(f"  {emoji} {signal['type']} (p{signal['percentile']})")
        lines.append(f"     {signal['reason']}")
    
    # Alignment analysis
    lines.append("")
    if alignment['alignment'] == 'BULLISH':
        lines.append(f"  💡 {alignment['conviction']} CONVICTION → FOLLOW UP")
        lines.append(f"     {len(signals)} signal(s) agree on upward direction")
    elif alignment['alignment'] == 'BEARISH':
        lines.append(f"  💡 {alignment['conviction']} CONVICTION → FOLLOW DOWN")
        lines.append(f"     {len(signals)} signal(s) agree on downward direction")
    elif alignment['alignment'] == 'CONFLICTING':
        lines.append(f"  ⚠️  CONFLICTING SIGNALS → WAIT FOR CLARITY")
        lines.append(f"     Signals disagree on direction")
    
    return "\n".join(lines)


def format_signals_email(actionable_markets: List[dict], neutral_markets: List[str], timestamp: datetime) -> str:
    """
    Format all signals into plain text email.
    
    Args:
        actionable_markets: List of dicts with market_ticker, signals, metrics, alignment
        neutral_markets: List of market tickers with no signals
        timestamp: Current run time
    """
    time_et = timestamp.astimezone(ZoneInfo(DISPLAY_TIMEZONE))
    time_str = time_et.strftime('%b %d, %Y %I:%M %p ET')
    
    lines = []
    lines.append("=" * 80)
    lines.append("🚨 KALSHI TRADING SIGNALS")
    lines.append("=" * 80)
    lines.append(f"Time: {time_str}")
    lines.append("")
    
    # Summary
    lines.append(f"Markets tracked: {len(actionable_markets) + len(neutral_markets)}")
    lines.append(f"Actionable signals: {len(actionable_markets)}")
    lines.append(f"Neutral markets: {len(neutral_markets)}")
    lines.append("")
    
    # Actionable markets
    if actionable_markets:
        lines.append("=" * 80)
        lines.append("📊 ACTIONABLE MARKETS")
        lines.append("=" * 80)
        
        # Sort by conviction (HIGH first)
        actionable_markets.sort(key=lambda x: (
            0 if x['alignment']['conviction'] == 'HIGH' else 
            1 if x['alignment']['conviction'] == 'MODERATE' else 2
        ))
        
        for market in actionable_markets:
            lines.append(format_market_signals_text(
                market['market_ticker'],
                market['metrics'],
                market['signals'],
                market['alignment']
            ))
            lines.append("-" * 80)
    else:
        lines.append("=" * 80)
        lines.append("✅ NO ACTIONABLE SIGNALS DETECTED")
        lines.append("=" * 80)
    
    # Neutral markets (collapsed)
    if neutral_markets:
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"📋 NEUTRAL MARKETS ({len(neutral_markets)})")
        lines.append("=" * 80)
        lines.append("")
        lines.append("Markets with no signals (all metrics within normal ranges):")
        for ticker in sorted(neutral_markets):
            lines.append(f"  • {ticker}")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def send_email_via_sns(subject: str, body: str):
    """Send plain text email via AWS SNS."""
    if not SNS_TOPIC_ARN:
        print("Warning: SNS_TOPIC_ARN not set, skipping email")
        return
    
    if not IS_LAMBDA:
        print("Note: Running locally, email only sent in Lambda")
        return
    
    try:
        sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject,
            Message=body
        )
        print(f"✅ Email sent via SNS: {subject}")
    except Exception as e:
        print(f"Error: Failed to send email via SNS: {e}")


# =============================================================================
# MAIN LOGIC
# =============================================================================

def discover_and_add_new_markets():
    """
    Discover high-volume Kalshi markets and add to tracking config.
    
    Returns:
        Number of new markets added
    """
    print("\n📡 Discovering new Kalshi markets...")
    
    markets = fetch_kalshi_markets(limit=200)
    if not markets:
        print("   ❌ Failed to fetch markets from Kalshi API")
        return 0
    
    print(f"   Fetched {len(markets)} markets from API")
    
    # Debug: Show first few markets
    print(f"   Analyzing markets (min volume: {MIN_VOLUME:,})...")
    
    added_count = 0
    candidates = []
    
    for market in markets:
        ticker = market.get('ticker')
        volume = market.get('volume', 0)
        category = market.get('category', 'unknown').lower()
        yes_bid = market.get('yes_bid')
        
        # Collect candidates for debugging
        if volume >= MIN_VOLUME / 10:  # Show markets with at least 10K volume for debugging
            candidates.append({
                'ticker': ticker,
                'volume': volume,
                'category': category,
                'yes_bid': yes_bid
            })
        
        # Skip if below volume threshold
        if volume < MIN_VOLUME:
            continue
        
        # Skip sports markets
        if any(exc in category for exc in EXCLUDED_CATEGORIES):
            print(f"   ⏭️  Skipped {ticker} (sports category: {category})")
            continue
        
        # Skip if no liquidity
        if yes_bid is None:
            print(f"   ⏭️  Skipped {ticker} (no yes_bid)")
            continue
        
        # Try to add (will skip if already tracked)
        initial_price = yes_bid / 100
        if add_market_to_config(ticker, category, volume, initial_price):
            print(f"   ✅ Added {ticker} (volume: {volume:,}, price: {initial_price:.2f}, category: {category})")
            added_count += 1
    
    # Debug output
    if added_count == 0:
        print(f"\n   ℹ️  No markets met criteria (volume >= {MIN_VOLUME:,})")
        if candidates:
            print(f"   📊 Top markets by volume (showing {min(5, len(candidates))}):")
            candidates.sort(key=lambda x: x['volume'], reverse=True)
            for i, m in enumerate(candidates[:5], 1):
                print(f"      {i}. {m['ticker']}: {m['volume']:,} volume, category: {m['category']}")
        else:
            print("   ⚠️  No markets found with volume >= 10K")
    else:
        print(f"   ✅ Added {added_count} new market(s) to tracking")
    
    return added_count


def process_market(market_ticker: str, timestamp: datetime) -> Optional[dict]:
    """
    Process a single market: fetch order book, calculate signals.
    
    Returns:
        Dict with market data and signals, or None if skipped
    """
    # Fetch current order book
    order_book_data = fetch_order_book(market_ticker)
    if not order_book_data:
        return None
    
    # Analyze order book
    metrics = analyze_order_book(order_book_data)
    if not metrics:
        return None
    
    # Save snapshot
    save_order_book_snapshot(market_ticker, order_book_data, metrics, timestamp)
    
    # Update baseline
    baseline = update_baseline(market_ticker, metrics)
    
    # Check if ready for signals
    if not baseline.get('ready_for_alerts', False):
        hours = baseline.get('hours_of_data', 0)
        print(f"   ⏳ {market_ticker}: Calibrating baseline ({hours:.0f}h / {BASELINE_HOURS_REQUIRED}h)")
        return None
    
    # Detect signals
    signals = check_signals(market_ticker, metrics, baseline)
    
    if not signals:
        return {'market_ticker': market_ticker, 'signals': [], 'metrics': metrics}
    
    # Analyze signal alignment
    alignment = analyze_signal_alignment(signals)
    
    return {
        'market_ticker': market_ticker,
        'signals': signals,
        'metrics': metrics,
        'alignment': alignment
    }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Track Kalshi order books and generate trading signals'
    )
    parser.add_argument('--prod-run', action='store_true',
                       help='Production mode (no prompts)')
    parser.add_argument('--check-api', action='store_true',
                       help='Check API connection and discover markets only')
    parser.add_argument('--skip-discovery', action='store_true',
                       help='Skip market discovery, only process tracked markets')
    
    args = parser.parse_args()
    
    timestamp = datetime.now(timezone.utc)
    time_et = timestamp.astimezone(ZoneInfo(DISPLAY_TIMEZONE))
    
    print("=" * 80)
    print("KALSHI ORDER BOOK TRACKER")
    print("=" * 80)
    print(f"Time: {time_et.strftime('%Y-%m-%d %H:%M:%S ET')}")
    print("")
    
    # Step 1: Load existing markets config
    print("📋 Loading tracked markets...")
    config = load_markets_config()
    active_markets = [m for m in config['markets'] if m.get('active', True)]
    print(f"   Currently tracking: {len(active_markets)} markets")
    
    # Step 2: Discover new markets (unless skipped or check-api only)
    if not args.skip_discovery:
        new_count = discover_and_add_new_markets()
        if new_count > 0:
            # Reload config
            config = load_markets_config()
            active_markets = [m for m in config['markets'] if m.get('active', True)]
    
    if args.check_api:
        print("\n✅ API check complete")
        return
    
    # Step 3: Process each market
    print(f"\n📊 Processing {len(active_markets)} markets...")
    
    actionable_markets = []
    neutral_markets = []
    
    for market_config in active_markets:
        ticker = market_config['ticker']
        print(f"\n   Processing {ticker}...")
        
        result = process_market(ticker, timestamp)
        
        if result is None:
            continue  # Skipped (no data or calibrating)
        
        if result['signals']:
            actionable_markets.append(result)
            print(f"   ✅ {ticker}: {len(result['signals'])} signal(s) detected")
        else:
            neutral_markets.append(ticker)
            print(f"   ↔️  {ticker}: No signals (neutral)")
    
    # Step 4: Generate and send alerts
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Markets processed: {len(active_markets)}")
    print(f"Actionable signals: {len(actionable_markets)}")
    print(f"Neutral markets: {len(neutral_markets)}")
    
    if actionable_markets or IS_LAMBDA:
        # Generate email
        email_body = format_signals_email(actionable_markets, neutral_markets, timestamp)
        
        if actionable_markets:
            subject = f"🚨 Kalshi Trading Signals - {time_et.strftime('%b %d, %Y %I:%M %p ET')}"
        else:
            subject = f"✅ Kalshi Check - No Signals - {time_et.strftime('%b %d, %Y %I:%M %p ET')}"
        
        # Send email (only in Lambda)
        send_email_via_sns(subject, email_body)
        
        # Print to console
        if not IS_LAMBDA:
            print("\n" + email_body)
    
    print("\n✅ Complete")


# =============================================================================
# AWS LAMBDA HANDLER
# =============================================================================

def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    Entry point when running in Lambda.
    """
    try:
        print("Lambda function started")
        print(f"Event: {json.dumps(event)}")
        
        # Run main logic (will auto-detect Lambda environment)
        main()
        
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Kalshi order book tracking complete'})
        }
    except Exception as e:
        print(f"Error in Lambda handler: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


if __name__ == '__main__':
    main()



if __name__ == '__main__':
    print("Kalshi order book tracker - implementation in progress...")

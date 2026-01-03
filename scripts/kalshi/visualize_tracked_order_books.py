"""
Visualize Kalshi Order Book Data from S3

Query and visualize order book snapshots collected by kalshi_order_book_tracker.py.
Generates two key charts for each market:
1. Price change over time (mid-price, bid, ask)
2. Order book distribution (before/after comparison)

Usage:
    # View all snapshots for a market
    python scripts/kalshi/visualize_tracked_order_books.py KXGREENLAND-29
    
    # Compare two specific time periods
    python scripts/kalshi/visualize_tracked_order_books.py KXGREENLAND-29 --compare 24h
    
    # Export charts for email
    python scripts/kalshi/visualize_tracked_order_books.py KXGREENLAND-29 --export
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import boto3
import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Setup
plt.style.use('seaborn-v0_8-whitegrid')
DISPLAY_TIMEZONE = 'America/New_York'

# AWS S3
S3_BUCKET = os.getenv('S3_BUCKET_KALSHI', 'kalshi-order-book-snapshots')
s3_client = boto3.client('s3')

# Add project root to path
current = Path(__file__).resolve().parent
project_root = current.parent.parent
import sys
sys.path.insert(0, str(project_root))


# =============================================================================
# S3 DATA LOADING
# =============================================================================

def list_market_snapshots(ticker: str) -> List[Dict]:
    """
    List all order book snapshots for a market from S3.
    
    Returns list of dicts with 'key', 'timestamp', 'size' for each snapshot.
    """
    prefix = f"data/01_input/kalshi/order_books/"
    
    snapshots = []
    paginator = s3_client.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            if not key.endswith('.json'):
                continue
            
            # Extract filename: TICKER_YYYYMMDD_HHMMSS.json
            filename = key.split('/')[-1]
            
            # Check if this file belongs to our ticker
            if not filename.startswith(ticker):
                continue
            
            # Extract timestamp from filename
            if '_' not in filename:
                continue
            
            parts = filename.replace('.json', '').split('_')
            if len(parts) < 3:
                continue
            
            try:
                date_str = parts[-2]  # YYYYMMDD
                time_str = parts[-1]  # HHMMSS
                
                timestamp = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                timestamp = timestamp.replace(tzinfo=timezone.utc)
                
                snapshots.append({
                    'key': key,
                    'timestamp': timestamp,
                    'size': obj['Size'],
                    'filename': filename
                })
            except (ValueError, IndexError):
                continue
    
    # Sort by timestamp
    snapshots.sort(key=lambda x: x['timestamp'])
    
    return snapshots


def load_snapshot(s3_key: str) -> Optional[Dict]:
    """Load a single snapshot from S3."""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        data = json.loads(response['Body'].read().decode('utf-8'))
        return data
    except Exception as e:
        print(f"Error loading {s3_key}: {e}")
        return None


def load_all_snapshots(ticker: str) -> pd.DataFrame:
    """
    Load all snapshots for a market and return as DataFrame.
    
    Returns DataFrame with columns:
    - timestamp: datetime
    - yes_bid, yes_ask, no_bid, no_ask: prices
    - yes_bid_size, yes_ask_size, no_bid_size, no_ask_size: sizes
    - mid_price: (yes_bid + yes_ask) / 2
    - spread: yes_ask - yes_bid
    - bid_imbalance: yes_bid_size / (yes_bid_size + yes_ask_size)
    - depth_ratio: (yes_bid_size + no_bid_size) / (yes_ask_size + no_ask_size)
    """
    print(f"📊 Loading snapshots for {ticker}...")
    
    snapshot_list = list_market_snapshots(ticker)
    
    if not snapshot_list:
        print(f"   No snapshots found for {ticker}")
        return pd.DataFrame()
    
    print(f"   Found {len(snapshot_list)} snapshots")
    
    rows = []
    for snap_info in snapshot_list:
        data = load_snapshot(snap_info['key'])
        if not data:
            continue
        
        # Parse order book structure
        ob = data.get('order_book', {}).get('orderbook', {})
        yes_orders = ob.get('yes', [])
        no_orders = ob.get('no', [])
        
        # Orders are [price, size] pairs
        # For yes side: lowest ask price is the best offer, highest bid is best demand
        # For no side: similar logic
        
        # Find best bid/ask for yes (convert cents to decimal)
        yes_bid = max([order[0] / 100 for order in yes_orders if order[0] <= 50], default=None)
        yes_bid_size = sum([order[1] for order in yes_orders if order[0] / 100 == yes_bid]) if yes_bid else 0
        yes_ask = min([order[0] / 100 for order in yes_orders if order[0] > 50], default=None)
        yes_ask_size = sum([order[1] for order in yes_orders if order[0] / 100 == yes_ask]) if yes_ask else 0
        
        # For no side
        no_bid = max([order[0] / 100 for order in no_orders if order[0] <= 50], default=None)
        no_bid_size = sum([order[1] for order in no_orders if order[0] / 100 == no_bid]) if no_bid else 0
        no_ask = min([order[0] / 100 for order in no_orders if order[0] > 50], default=None)
        no_ask_size = sum([order[1] for order in no_orders if order[0] / 100 == no_ask]) if no_ask else 0
        
        # Calculate metrics
        mid_price = (yes_bid + yes_ask) / 2 if (yes_bid and yes_ask) else None
        spread = (yes_ask - yes_bid) if (yes_bid and yes_ask) else None
        
        bid_imbalance = None
        if yes_bid_size + yes_ask_size > 0:
            bid_imbalance = yes_bid_size / (yes_bid_size + yes_ask_size)
        
        total_bid_depth = yes_bid_size + no_bid_size
        total_ask_depth = yes_ask_size + no_ask_size
        depth_ratio = None
        if total_ask_depth > 0:
            depth_ratio = total_bid_depth / total_ask_depth
        
        rows.append({
            'timestamp': snap_info['timestamp'],
            'yes_bid': yes_bid,
            'yes_ask': yes_ask,
            'no_bid': no_bid,
            'no_ask': no_ask,
            'yes_bid_size': yes_bid_size,
            'yes_ask_size': yes_ask_size,
            'no_bid_size': no_bid_size,
            'no_ask_size': no_ask_size,
            'mid_price': mid_price,
            'spread': spread,
            'bid_imbalance': bid_imbalance,
            'depth_ratio': depth_ratio,
        })
    
    df = pd.DataFrame(rows)
    
    if not df.empty:
        # Convert to ET for display
        df['timestamp_et'] = df['timestamp'].dt.tz_convert(DISPLAY_TIMEZONE)
    
    print(f"   ✅ Loaded {len(df)} snapshots")
    print(f"   Time range: {df['timestamp_et'].min()} to {df['timestamp_et'].max()}")
    
    return df


# =============================================================================
# VISUALIZATION: PRICE OVER TIME
# =============================================================================

def plot_price_over_time(df: pd.DataFrame, ticker: str, save_path: Optional[str] = None):
    """
    Plot price movement over time.
    
    Shows:
    - Mid price (main line)
    - Bid/ask band (shaded area)
    - Spread (secondary axis)
    """
    if df.empty:
        print("No data to plot")
        return
    
    # Filter out rows with no price data
    df_valid = df[df['mid_price'].notna()].copy()
    
    if df_valid.empty:
        print("No valid price data to plot (all markets may have no bids/asks)")
        return
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Plot mid price
    ax1.plot(df_valid['timestamp_et'], df_valid['mid_price'], 
             label='Mid Price', color='blue', linewidth=2, marker='o', markersize=4)
    
    # Plot bid/ask band
    ax1.fill_between(df_valid['timestamp_et'], df_valid['yes_bid'], df_valid['yes_ask'], 
                      alpha=0.2, color='blue', label='Bid-Ask Spread')
    
    ax1.set_xlabel('Time (ET)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price (cents)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{ticker} - Price Movement Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Secondary axis for spread
    ax2 = ax1.twinx()
    spread_valid = df_valid[df_valid['spread'].notna()]
    if not spread_valid.empty:
        ax2.plot(spread_valid['timestamp_et'], spread_valid['spread'], 
                 label='Spread', color='red', linewidth=1, alpha=0.7, linestyle='--')
        ax2.set_ylabel('Spread (cents)', fontsize=12, fontweight='bold', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   💾 Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# VISUALIZATION: ORDER BOOK COMPARISON
# =============================================================================

def plot_order_book_comparison(df: pd.DataFrame, ticker: str, 
                               compare_hours: int = 24, save_path: Optional[str] = None):
    """
    Compare order book at two time points (now vs X hours ago).
    
    Shows side-by-side order book distributions with deltas.
    """
    if df.empty or len(df) < 2:
        print("Not enough data for comparison")
        return
    
    # Get most recent snapshot
    current = df.iloc[-1]
    
    # Get snapshot from X hours ago (or earliest if not enough data)
    target_time = current['timestamp'] - timedelta(hours=compare_hours)
    past = df.iloc[(df['timestamp'] - target_time).abs().argsort()[0]]
    
    time_diff_hours = (current['timestamp'] - past['timestamp']).total_seconds() / 3600
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- PAST ORDER BOOK ---
    ax1.barh(['Yes Bid', 'Yes Ask', 'No Bid', 'No Ask'],
             [past['yes_bid_size'], past['yes_ask_size'], 
              past['no_bid_size'], past['no_ask_size']],
             color=['green', 'red', 'lightgreen', 'lightcoral'])
    
    ax1.set_xlabel('Size (contracts)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{past["timestamp_et"].strftime("%b %d, %I:%M %p ET")}\n' + 
                  f'Mid: {past["mid_price"]:.1f}¢  Spread: {past["spread"]:.1f}¢',
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # --- CURRENT ORDER BOOK ---
    ax2.barh(['Yes Bid', 'Yes Ask', 'No Bid', 'No Ask'],
             [current['yes_bid_size'], current['yes_ask_size'], 
              current['no_bid_size'], current['no_ask_size']],
             color=['green', 'red', 'lightgreen', 'lightcoral'])
    
    ax2.set_xlabel('Size (contracts)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{current["timestamp_et"].strftime("%b %d, %I:%M %p ET")}\n' + 
                  f'Mid: {current["mid_price"]:.1f}¢  Spread: {current["spread"]:.1f}¢',
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Overall title
    fig.suptitle(f'{ticker} - Order Book Comparison ({time_diff_hours:.1f}h apart)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   💾 Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# VISUALIZATION: SIGNAL METRICS
# =============================================================================

def plot_signal_metrics(df: pd.DataFrame, ticker: str, save_path: Optional[str] = None):
    """
    Plot signal detection metrics over time.
    
    Shows:
    - Bid imbalance (top)
    - Depth ratio (bottom)
    
    With thresholds marked (p15, p85 from baseline if available).
    """
    if df.empty:
        print("No data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # --- BID IMBALANCE ---
    ax1.plot(df['timestamp_et'], df['bid_imbalance'], 
             label='Bid Imbalance', color='purple', linewidth=2, marker='o', markersize=4)
    ax1.axhline(0.85, color='red', linestyle='--', alpha=0.5, label='High threshold (0.85)')
    ax1.axhline(0.15, color='red', linestyle='--', alpha=0.5, label='Low threshold (0.15)')
    ax1.axhline(0.50, color='gray', linestyle='-', alpha=0.3, label='Neutral (0.50)')
    ax1.set_ylabel('Bid Imbalance', fontsize=12, fontweight='bold')
    ax1.set_title(f'{ticker} - Signal Metrics Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # --- DEPTH RATIO ---
    ax2.plot(df['timestamp_et'], df['depth_ratio'], 
             label='Depth Ratio', color='orange', linewidth=2, marker='o', markersize=4)
    ax2.axhline(3.0, color='red', linestyle='--', alpha=0.5, label='High threshold (3.0x)')
    ax2.axhline(0.33, color='red', linestyle='--', alpha=0.5, label='Low threshold (0.33x)')
    ax2.axhline(1.0, color='gray', linestyle='-', alpha=0.3, label='Neutral (1.0x)')
    ax2.set_xlabel('Time (ET)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Depth Ratio (Bid/Ask)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   💾 Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize Kalshi order book data from S3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('ticker', type=str, help='Market ticker (e.g., KXGREENLAND-29)')
    parser.add_argument('--compare', type=int, default=24, 
                       help='Hours to compare for order book (default: 24)')
    parser.add_argument('--export', action='store_true',
                       help='Export charts to PNG files (for email)')
    parser.add_argument('--output-dir', type=str, default='output/kalshi_charts',
                       help='Directory to save charts (default: output/kalshi_charts)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"VISUALIZING ORDER BOOKS: {args.ticker}")
    print("=" * 80)
    
    # Load data
    df = load_all_snapshots(args.ticker)
    
    if df.empty:
        print("\n❌ No data found - market may not be tracked yet")
        return
    
    # Print summary stats
    print("\n📈 Summary Statistics:")
    print(f"   Snapshots: {len(df)}")
    print(f"   Time span: {(df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600:.1f} hours")
    
    if pd.notna(df['mid_price'].iloc[-1]):
        print(f"   Current mid price: {df['mid_price'].iloc[-1]:.2f}¢")
        print(f"   Price range: {df['mid_price'].min():.2f}¢ - {df['mid_price'].max():.2f}¢")
    else:
        print(f"   Current mid price: N/A (no bid/ask)")
    
    if pd.notna(df['spread'].mean()):
        print(f"   Avg spread: {df['spread'].mean():.2f}¢")
    
    if not df['bid_imbalance'].isna().all() and pd.notna(df['bid_imbalance'].iloc[-1]):
        print(f"   Current bid imbalance: {df['bid_imbalance'].iloc[-1]:.2f}")
    if not df['depth_ratio'].isna().all() and pd.notna(df['depth_ratio'].iloc[-1]):
        print(f"   Current depth ratio: {df['depth_ratio'].iloc[-1]:.2f}x")
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    
    if args.export:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export all charts
        print("\n1️⃣ Price over time...")
        plot_price_over_time(df, args.ticker, 
                           save_path=output_dir / f"{args.ticker}_price_over_time.png")
        
        print("\n2️⃣ Order book comparison...")
        plot_order_book_comparison(df, args.ticker, compare_hours=args.compare,
                                  save_path=output_dir / f"{args.ticker}_order_book_comparison.png")
        
        print("\n3️⃣ Signal metrics...")
        plot_signal_metrics(df, args.ticker,
                          save_path=output_dir / f"{args.ticker}_signal_metrics.png")
        
        print(f"\n✅ Charts saved to: {output_dir}")
    else:
        # Show interactively
        print("\n1️⃣ Price over time...")
        plot_price_over_time(df, args.ticker)
        
        print("\n2️⃣ Order book comparison...")
        plot_order_book_comparison(df, args.ticker, compare_hours=args.compare)
        
        print("\n3️⃣ Signal metrics...")
        plot_signal_metrics(df, args.ticker)


if __name__ == '__main__':
    main()


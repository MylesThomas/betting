"""
Visualize Kalshi Order Book Snapshots from S3

This script queries order book snapshots saved by kalshi_order_book_tracker.py
and generates visualizations to understand market movements and order book dynamics.

PURPOSE
-------
After kalshi_order_book_tracker.py runs hourly and saves snapshots to S3, this script
helps analyze the data by creating two key visualizations for any tracked market:

1. **Price Movement Over Time**: Line chart showing mid-price evolution
2. **Order Book Before/After**: Side-by-side comparison of order book distribution

WORKFLOW
--------
1. Query S3 for all snapshots of a given market ticker
2. Load snapshot JSON files and extract price/order book data
3. Generate time series data (timestamp, mid_price, spread, volume)
4. Create visualizations:
   - Chart 1: Price over time with spread bands
   - Chart 2: Order book comparison at two different timestamps
5. Save charts locally or display interactively

DATA SOURCE
-----------
S3 Bucket: kalshi-order-book-snapshots
Path: data/01_input/kalshi/order_books/{TICKER}/snapshot_{timestamp}.json

Each snapshot contains:
- timestamp: ISO format datetime
- market_ticker: Kalshi ticker (e.g., KXGREENLAND-29)
- yes_bid: Best yes price (cents)
- yes_ask: Best yes ask price (cents)
- no_bid: Best no price (cents)
- no_ask: Best no ask price (cents)
- orderbook: Full order book with yes_orders and no_orders arrays

USAGE
-----
**Basic Usage - View recent price movements:**
```bash
python scripts/kalshi/visualize_kalshi_snapshots.py KXGREENLAND-29
```

**Show order book comparison (last 24h vs now):**
```bash
python scripts/kalshi/visualize_kalshi_snapshots.py KXGREENLAND-29 --compare 24h
```

**Custom time range:**
```bash
python scripts/kalshi/visualize_kalshi_snapshots.py KXGREENLAND-29 --start "2025-12-27" --end "2025-12-28"
```

**Save charts to file:**
```bash
python scripts/kalshi/visualize_kalshi_snapshots.py KXGREENLAND-29 --save charts/
```

**Interactive mode (opens in browser):**
```bash
python scripts/kalshi/visualize_kalshi_snapshots.py KXGREENLAND-29 --interactive
```

**List available markets in S3:**
```bash
python scripts/kalshi/visualize_kalshi_snapshots.py --list
```

**Show statistics only (no charts):**
```bash
python scripts/kalshi/visualize_kalshi_snapshots.py KXGREENLAND-29 --stats-only
```

CHART 1: PRICE MOVEMENT OVER TIME
----------------------------------
X-axis: Time (hourly snapshots)
Y-axis: Price (cents, 0-100 range)

Lines plotted:
- Yes Bid (green line): Best bid price over time
- Yes Ask (red line): Best ask price over time
- The gap between lines inherently shows the spread
- Mid price (optional, dashed): (yes_bid + yes_ask) / 2

Annotations:
- Fill events (detected volume changes)
- Baseline calibration markers (first 48h)
- Signal triggers (if baseline complete)

CHART 2: ORDER BOOK BEFORE/AFTER
---------------------------------
Side-by-side bar charts showing full order book distribution:

Left panel: "Before" snapshot (e.g., 24h ago or user-specified)
- X-axis: Price levels (1-99 cents)
- Y-axis: Volume at each price level
- Color: Green for bids, Red for asks

Right panel: "After" snapshot (most recent or user-specified)
- Same axes as left panel
- Delta indicators: Show volume changes at each level

Metrics shown:
- Best bid/ask before → after (with delta)
- Total volume before → after (with % change)
- Spread before → after (tightening/widening)
- Imbalance before → after (bid_size / total_size)

EXAMPLES
--------
1. **Check if data is being collected:**
   python scripts/kalshi/visualize_kalshi_snapshots.py KXGREENLAND-29

2. **Analyze yesterday's price action:**
   python scripts/kalshi/visualize_kalshi_snapshots.py KXGREENLAND-29 --start "2025-12-27"

3. **Compare order book before/after news event:**
   python scripts/kalshi/visualize_kalshi_snapshots.py KXGREENLAND-29 \
       --compare-times "2025-12-28T10:00" "2025-12-28T14:00"

4. **Generate charts for email alerts (future integration):**
   python scripts/kalshi/visualize_kalshi_snapshots.py KXGREENLAND-29 \
       --save /tmp/ --format png --no-display

DEPENDENCIES
------------
- boto3 (S3 access)
- pandas (data manipulation)
- matplotlib (chart generation)
- seaborn (styling, optional)

ENVIRONMENT VARIABLES
---------------------
- S3_BUCKET_KALSHI: S3 bucket name (default: kalshi-order-book-snapshots)
- AWS_REGION_NAME: AWS region (default: us-east-2)

RELATED FILES
-------------
- scripts/kalshi_order_book_tracker.py (data collection)
- scripts/kalshi/view_kalshi_order_book.py (local order book viewer)

TODO / FUTURE ENHANCEMENTS
---------------------------
- [ ] Add plotly for interactive charts
- [ ] Generate charts directly in Lambda for SES inline images
- [ ] Volume profile heatmap (price level × time)
- [ ] Correlation analysis between multiple markets
- [ ] Anomaly detection overlays (sudden spread widening, etc.)
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys
import argparse
import os
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import boto3
import json
from typing import List, Dict, Optional, Tuple

# Optional: seaborn for prettier charts
try:
    import seaborn as sns
    sns.set_palette("husl")
    plt.style.use('seaborn-v0_8-darkgrid')
except ImportError:
    plt.style.use('ggplot')  # Fallback style

# Load environment variables
load_dotenv()

# Add src to path by finding project root
def find_project_root() -> Path:
    """Find project root by looking for .gitignore file."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / '.gitignore').exists():
            return parent
    return current

PROJECT_ROOT = find_project_root()
src_path = PROJECT_ROOT / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================

DISPLAY_TIMEZONE = 'America/New_York'  # Eastern Time for display
S3_BUCKET = os.getenv('S3_BUCKET_KALSHI', 'kalshi-order-book-snapshots')

# Initialize boto3 client
s3_client = boto3.client('s3')

# =============================================================================
# S3 QUERY FUNCTIONS
# =============================================================================

def list_all_markets() -> List[str]:
    """
    List all market tickers that have snapshots in S3.
    
    Returns:
        List of market tickers
    """
    prefix = 'data/01_input/kalshi/order_books/'
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix)
        
        tickers = set()
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Extract ticker from filename like: KXGREENLAND-29_20251228_052308.json
                    filename = obj['Key'].split('/')[-1]
                    if filename.endswith('.json'):
                        # Ticker is everything before the first underscore
                        ticker = filename.split('_')[0]
                        tickers.add(ticker)
        
        return sorted(list(tickers))
    except Exception as e:
        print(f"Error listing markets from S3: {e}")
        return []


def get_snapshot_keys(market_ticker: str) -> List[str]:
    """
    Get all S3 keys for snapshots of a given market.
    
    Args:
        market_ticker: Kalshi market ticker (e.g., KXGREENLAND-29)
    
    Returns:
        List of S3 keys sorted by timestamp
    """
    prefix = f'data/01_input/kalshi/order_books/{market_ticker}_'
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix)
        
        keys = []
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith('.json'):
                        keys.append(obj['Key'])
        
        return sorted(keys)
    except Exception as e:
        print(f"Error fetching snapshot keys: {e}")
        return []


def load_snapshot_from_s3(s3_key: str) -> Optional[dict]:
    """
    Load a single snapshot JSON from S3.
    
    Args:
        s3_key: S3 key for the snapshot file
    
    Returns:
        Snapshot dict or None if error
    """
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        print(f"Error loading {s3_key}: {e}")
        return None


def load_all_snapshots(market_ticker: str) -> pd.DataFrame:
    """
    Load all snapshots for a market into a DataFrame.
    
    Args:
        market_ticker: Kalshi market ticker
    
    Returns:
        DataFrame with columns: timestamp, yes_bid, yes_ask, no_bid, no_ask, 
                                mid_price, spread, orderbook, metrics
    """
    keys = get_snapshot_keys(market_ticker)
    
    if not keys:
        print(f"No snapshots found for {market_ticker}")
        return pd.DataFrame()
    
    print(f"Loading {len(keys)} snapshots from S3...")
    
    data = []
    for key in keys:
        snapshot = load_snapshot_from_s3(key)
        if snapshot:
            # Parse timestamp
            ts = datetime.fromisoformat(snapshot['timestamp'].replace('Z', '+00:00'))
            
            # Extract metrics if available
            metrics = snapshot.get('metrics', {})
            
            data.append({
                'timestamp': ts,
                'mid_price': metrics.get('mid_price'),
                'spread': metrics.get('spread'),
                'best_yes_price': metrics.get('best_yes_price'),
                'best_no_price': metrics.get('best_no_price'),
                'bid_imbalance': metrics.get('bid_imbalance'),
                'depth_ratio': metrics.get('depth_ratio'),
                'total_depth': metrics.get('total_depth'),
                'orderbook': snapshot.get('order_book')
            })
    
    df = pd.DataFrame(data)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✅ Loaded {len(df)} snapshots")
    return df


# =============================================================================
# CHART GENERATION FUNCTIONS
# =============================================================================

def save_chart_to_s3(fig, market_ticker: str, chart_type: str) -> str:
    """
    Save matplotlib figure to S3 for use in SES emails.
    
    Args:
        fig: Matplotlib figure object
        market_ticker: Market ticker
        chart_type: 'price' or 'orderbook'
    
    Returns:
        S3 public URL for the chart
    """
    from io import BytesIO
    
    # Save figure to bytes
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    
    # Generate S3 key (in email-charts/ for public access)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    s3_key = f"email-charts/{market_ticker}_{chart_type}_{timestamp}.png"
    
    # Upload to S3
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=buf.getvalue(),
            ContentType='image/png'
        )
        
        # Generate public URL
        public_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"
        print(f"📊 Saved {chart_type} chart to S3: {s3_key}")
        
        return public_url
    except Exception as e:
        print(f"❌ Error saving chart to S3: {e}")
        return ""


def create_price_chart(df: pd.DataFrame, market_ticker: str, save_to_s3: bool = False, 
                      save_path: Optional[str] = None) -> str:
    """
    Create Chart 1: Price movement over time showing bid/ask spread.
    
    Args:
        df: DataFrame with price data
        market_ticker: Market ticker for title
        save_to_s3: If True, save to S3 instead of local file
        save_path: Optional local path to save chart (ignored if save_to_s3=True)
    
    Returns:
        S3 URL if saved to S3, local path if saved locally, or empty string
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Convert to ET timezone for x-axis
    df['timestamp_et'] = df['timestamp'].dt.tz_convert(ZoneInfo(DISPLAY_TIMEZONE))
    
    # Calculate Yes Ask from Yes Bid + Spread
    df['yes_ask'] = df['best_yes_price'] + df['spread']
    
    # Plot Yes Bid and Yes Ask lines
    ax.plot(df['timestamp_et'], df['best_yes_price'], 
            color='#2ecc71', linewidth=2.5, label='Yes Bid (sell price)', marker='o', markersize=4)
    ax.plot(df['timestamp_et'], df['yes_ask'], 
            color='#e74c3c', linewidth=2.5, label='Yes Ask (buy price)', marker='s', markersize=4)
    
    # Fill between bid and ask to show actual spread
    ax.fill_between(df['timestamp_et'], 
                     df['best_yes_price'], 
                     df['yes_ask'],
                     alpha=0.2, color='gray', label='Spread (Ask - Bid)')
    
    # Add mid-price as dashed line
    ax.plot(df['timestamp_et'], df['mid_price'],
            color='black', linewidth=1.5, linestyle='--', label='Mid Price', alpha=0.7)
    
    # Add vertical time markers (24h, 48h, 1w, 2w, 3w, 1mo, 2mo, ..., 1y, 2y)
    latest_time = df['timestamp_et'].iloc[-1]
    earliest_time = df['timestamp_et'].iloc[0]
    
    # Define time intervals
    time_markers = [
        ('24h', timedelta(hours=24)),
        ('48h', timedelta(hours=48)),
        ('1w', timedelta(weeks=1)),
        ('2w', timedelta(weeks=2)),
        # ('3w', timedelta(weeks=3)),
        ('1mo', timedelta(days=30)),
        ('2mo', timedelta(days=60)),
        ('3mo', timedelta(days=90)),
        ('4mo', timedelta(days=120)),
        ('5mo', timedelta(days=150)),
        ('6mo', timedelta(days=180)),
        ('7mo', timedelta(days=210)),
        ('8mo', timedelta(days=240)),
        ('9mo', timedelta(days=270)),
        ('10mo', timedelta(days=300)),
        ('11mo', timedelta(days=330)),
        ('1y', timedelta(days=365)),
        ('2y', timedelta(days=730)),
    ]
    
    # Draw vertical lines for each interval that exists in the data
    for label, delta in time_markers:
        marker_time = latest_time - delta
        
        # Only draw if this time is within our data range
        if marker_time >= earliest_time:
            ax.axvline(x=marker_time, color='black', linewidth=1, linestyle=':', alpha=0.4, zorder=1)
            
            # Add label at the top of the chart
            ax.text(marker_time, ax.get_ylim()[1], f' {label}',
                   rotation=90, verticalalignment='top', horizontalalignment='right',
                   fontsize=8, alpha=0.6, color='black')
    
    # Formatting
    ax.set_xlabel('Time (ET)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price (cents)', fontsize=12, fontweight='bold')
    ax.set_title(f'{market_ticker} - Price Movement Over Time\n(Gap = Spread)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Y-axis: 0-100 cents (full probability range)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}¢'))
    
    # X-axis: format timestamps nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %I:%M %p'))
    plt.xticks(rotation=45, ha='right')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Add data collection info
    hours = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 3600
    info_text = f'Snapshots: {len(df)} | Timespan: {hours:.1f}h'
    if hours >= 48:
        info_text += ' | ✅ Baseline Ready'
    else:
        info_text += f' | ⏳ Calibrating ({hours:.0f}/48h)'
    
    ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save to S3 or local file
    if save_to_s3:
        url = save_chart_to_s3(plt.gcf(), market_ticker, 'price')
        plt.close()
        return url
    elif save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved price chart to: {save_path}")
        plt.close()
        return str(save_path)
    else:
        plt.show()
        return ""


def create_orderbook_comparison(df: pd.DataFrame, market_ticker: str, 
                                before_idx: int = 0, after_idx: int = -1,
                                save_to_s3: bool = False,
                                save_path: Optional[str] = None) -> str:
    """
    Create Chart 2: Order book comparison (before vs after).
    Histogram showing volume at each price level (1-99 cents).
    
    Args:
        df: DataFrame with order book data
        market_ticker: Market ticker for title
        before_idx: Index for "before" snapshot (default: first)
        after_idx: Index for "after" snapshot (default: last)
        save_to_s3: If True, save to S3 instead of local file
        save_path: Optional local path (ignored if save_to_s3=True)
    
    Returns:
        S3 URL if saved to S3, local path if saved locally, or empty string
    """
    # Get before/after snapshots
    before_snap = df.iloc[before_idx]
    after_snap = df.iloc[after_idx]
    
    before_orderbook = before_snap['orderbook']
    after_orderbook = after_snap['orderbook']
    
    if not before_orderbook or not after_orderbook:
        print("⚠️  No order book data available")
        return ""
    
    # Extract yes/no orders
    before_yes = before_orderbook.get('orderbook', {}).get('yes', [])
    before_no = before_orderbook.get('orderbook', {}).get('no', [])
    after_yes = after_orderbook.get('orderbook', {}).get('yes', [])
    after_no = after_orderbook.get('orderbook', {}).get('no', [])
    
    # Build volume by price level
    def build_volume_dict(yes_orders, no_orders):
        volume = {}
        
        # Yes orders (bids for "yes" outcome)
        for order in yes_orders:
            price_cents = order[0]  # price in cents
            size = order[1]
            volume[price_cents] = volume.get(price_cents, 0) + size
        
        # No orders (bids for "no" outcome - complement side)
        for order in no_orders:
            price_cents = order[0]
            size = order[1]
            # No orders sit on opposite side (100 - price)
            complement_price = 100 - price_cents
            volume[complement_price] = volume.get(complement_price, 0) + size
        
        return volume
    
    before_volume = build_volume_dict(before_yes, before_no)
    after_volume = build_volume_dict(after_yes, after_no)
    
    # Calculate delta (what changed)
    all_prices = sorted(set(list(before_volume.keys()) + list(after_volume.keys())))
    delta_volume = {}
    for price in all_prices:
        before_vol = before_volume.get(price, 0)
        after_vol = after_volume.get(price, 0)
        delta = after_vol - before_vol
        if delta != 0:  # Only include prices with changes
            delta_volume[price] = delta
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7), sharey=True)
    
    # Convert to ET for display
    before_time = before_snap['timestamp'].astimezone(ZoneInfo(DISPLAY_TIMEZONE))
    after_time = after_snap['timestamp'].astimezone(ZoneInfo(DISPLAY_TIMEZONE))
    
    # Plot BEFORE
    if before_volume:
        prices = sorted(before_volume.keys())
        volumes = [before_volume[p] for p in prices]
        
        ax1.bar(prices, volumes, color='#3498db', alpha=0.7, edgecolor='black', width=1.0)
        ax1.set_xlabel('Price Level (cents)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Volume (contracts)', fontsize=11, fontweight='bold')
        ax1.set_title(f'BEFORE\n{before_time.strftime("%m/%d %I:%M %p ET")}',
                      fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_xlim(0, 100)
        
        # Add vertical line at current best bid price (market price)
        best_bid = before_snap['best_yes_price'] * 100 if before_snap['best_yes_price'] else 0
        if best_bid > 0:
            ax1.axvline(x=best_bid, color='black', linewidth=2.5, linestyle='--', 
                       label=f'Market Price: {best_bid:.0f}¢', zorder=10)
            # Shade left side (bids) and right side (asks)
            ax1.axvspan(0, best_bid, alpha=0.05, color='green', label='Bids (buy YES / sell NO)')
            ax1.axvspan(best_bid, 100, alpha=0.05, color='red', label='Asks (sell YES / buy NO)')
        
        # Add metrics
        total_vol = sum(volumes)
        ax1.text(0.02, 0.98, 
                 f'Total Volume: {total_vol:,}\nBest Bid: {best_bid:.0f}¢',
                 transform=ax1.transAxes,
                 fontsize=9,
                 verticalalignment='top',
                 horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Add legend
        ax1.legend(loc='upper right', fontsize=8)
    
    # Plot AFTER
    if after_volume:
        prices = sorted(after_volume.keys())
        volumes = [after_volume[p] for p in prices]
        
        ax2.bar(prices, volumes, color='#e74c3c', alpha=0.7, edgecolor='black', width=1.0)
        ax2.set_xlabel('Price Level (cents)', fontsize=11, fontweight='bold')
        ax2.set_title(f'AFTER\n{after_time.strftime("%m/%d %I:%M %p ET")}',
                      fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xlim(0, 100)
        
        # Add vertical line at current best bid price
        best_bid = after_snap['best_yes_price'] * 100 if after_snap['best_yes_price'] else 0
        if best_bid > 0:
            ax2.axvline(x=best_bid, color='black', linewidth=2.5, linestyle='--', 
                       label=f'Market Price: {best_bid:.0f}¢', zorder=10)
            # Shade left side (bids) and right side (asks)
            ax2.axvspan(0, best_bid, alpha=0.05, color='green', label='Bids (buy YES / sell NO)')
            ax2.axvspan(best_bid, 100, alpha=0.05, color='red', label='Asks (sell YES / buy NO)')
        
        # Calculate delta from before
        total_vol = sum(volumes)
        before_total = sum(before_volume.values()) if before_volume else 0
        before_best = before_snap['best_yes_price'] * 100 if before_snap['best_yes_price'] else 0
        vol_delta = total_vol - before_total
        price_delta = best_bid - before_best
        
        ax2.text(0.02, 0.98,
                 f'Total Volume: {total_vol:,} ({vol_delta:+,})\nBest Bid: {best_bid:.0f}¢ ({price_delta:+.1f}¢)',
                 transform=ax2.transAxes,
                 fontsize=9,
                 verticalalignment='top',
                 horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # Add legend
        ax2.legend(loc='upper right', fontsize=8)
    
    # Plot DELTA (what changed)
    if delta_volume:
        prices = sorted(delta_volume.keys())
        deltas = [delta_volume[p] for p in prices]
        colors = ['#27ae60' if d > 0 else '#e74c3c' for d in deltas]
        
        ax3.bar(prices, deltas, color=colors, alpha=0.7, edgecolor='black', width=1.0)
        ax3.set_xlabel('Price Level (cents)', fontsize=11, fontweight='bold')
        ax3.set_title(f'DELTA (Changes)\n{before_time.strftime("%m/%d %I:%M %p")} → {after_time.strftime("%I:%M %p ET")}',
                      fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_xlim(0, 100)
        ax3.axhline(y=0, color='black', linewidth=1.5, linestyle='-', alpha=0.5)
        
        # Add vertical line at AFTER market price
        before_best_bid = before_snap['best_yes_price'] * 100 if before_snap['best_yes_price'] else 0
        after_best_bid = after_snap['best_yes_price'] * 100 if after_snap['best_yes_price'] else 0
        
        if after_best_bid > 0:
            ax3.axvline(x=after_best_bid, color='#2ecc71', linewidth=2.5, linestyle='--', 
                       label=f'Current Price: {after_best_bid:.0f}¢', zorder=10, alpha=0.8)
        
        # If price moved, show the old price and highlight the move
        price_moved = False
        if before_best_bid > 0 and after_best_bid > 0 and abs(before_best_bid - after_best_bid) >= 1:
            price_moved = True
            ax3.axvline(x=before_best_bid, color='#e67e22', linewidth=2.5, linestyle=':', 
                       label=f'Previous Price: {before_best_bid:.0f}¢', zorder=10, alpha=0.8)
            
            # Add arrow showing price movement
            arrow_y = ax3.get_ylim()[1] * 0.85  # Position at 85% of chart height
            ax3.annotate('', xy=(after_best_bid, arrow_y), xytext=(before_best_bid, arrow_y),
                        arrowprops=dict(arrowstyle='->', color='#e67e22', lw=3, alpha=0.9))
            
            # Add price change label
            price_change = after_best_bid - before_best_bid
            mid_point = (before_best_bid + after_best_bid) / 2
            ax3.text(mid_point, arrow_y * 1.05, f'{price_change:+.0f}¢',
                    fontsize=10, fontweight='bold', color='#e67e22',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#e67e22', alpha=0.9))
        
        # Calculate metrics
        volume_added = sum(d for d in deltas if d > 0)
        volume_removed = abs(sum(d for d in deltas if d < 0))
        net_change = sum(deltas)
        
        # Build metrics text
        metrics_text = f'Volume Added: {volume_added:,}\nVolume Removed: {volume_removed:,}\nNet Change: {net_change:+,}'
        
        # If price moved significantly, flag it and identify where the fill likely happened
        if price_moved:
            price_change = after_best_bid - before_best_bid
            # Check for large volume removed near the old price
            volume_at_old_price = abs(delta_volume.get(int(before_best_bid), 0))
            if volume_at_old_price > 1000:  # Significant fill
                metrics_text += f'\n\n🚨 PRICE MOVED {price_change:+.0f}¢\n'
                metrics_text += f'Large fill at {before_best_bid:.0f}¢\n'
                metrics_text += f'({volume_at_old_price:,} contracts)'
            else:
                metrics_text += f'\n\n📊 Price moved {price_change:+.0f}¢'
        
        ax3.text(0.02, 0.98,
                 metrics_text,
                 transform=ax3.transAxes,
                 fontsize=9,
                 verticalalignment='top',
                 horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#27ae60', alpha=0.7, label='New volume (added)'),
            Patch(facecolor='#e74c3c', alpha=0.7, label='Filled/cancelled (removed)')
        ]
        if after_best_bid > 0:
            legend_elements.insert(0, plt.Line2D([0], [0], color='#2ecc71', linewidth=2.5, 
                                                  linestyle='--', label=f'Current Price: {after_best_bid:.0f}¢'))
        if price_moved and before_best_bid > 0:
            legend_elements.insert(1, plt.Line2D([0], [0], color='#e67e22', linewidth=2.5, 
                                                  linestyle=':', label=f'Previous Price: {before_best_bid:.0f}¢'))
        ax3.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Overall title
    fig.suptitle(f'{market_ticker} - Order Book Comparison', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save to S3 or local file
    if save_to_s3:
        url = save_chart_to_s3(plt.gcf(), market_ticker, 'orderbook')
        plt.close()
        return url
    elif save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved order book chart to: {save_path}")
        plt.close()
        return str(save_path)
    else:
        plt.show()
        return ""


# =============================================================================
# STATISTICS FUNCTIONS
# =============================================================================

def print_market_stats(market_ticker: str):
    """
    Print statistics about snapshots for a market.
    
    Args:
        market_ticker: Kalshi market ticker
    """
    print("=" * 80)
    print(f"SNAPSHOT STATISTICS: {market_ticker}")
    print("=" * 80)
    
    df = load_all_snapshots(market_ticker)
    
    if df.empty:
        print("❌ No snapshots found")
        return
    
    # Convert timestamps to ET for display
    df['timestamp_et'] = df['timestamp'].dt.tz_convert(ZoneInfo(DISPLAY_TIMEZONE))
    
    # Basic stats
    print(f"\n📊 Data Collection:")
    print(f"   Total snapshots: {len(df)}")
    print(f"   First snapshot: {df['timestamp_et'].iloc[0].strftime('%Y-%m-%d %H:%M:%S ET')}")
    print(f"   Last snapshot:  {df['timestamp_et'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S ET')}")
    
    # Time span
    time_span = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
    hours = time_span.total_seconds() / 3600
    print(f"   Time span: {hours:.1f} hours ({hours/24:.1f} days)")
    print(f"   Baseline status: {'✅ Ready (48h+)' if hours >= 48 else f'⏳ Calibrating ({hours:.0f}h / 48h)'}")
    
    # Price stats
    print(f"\n💰 Price Statistics:")
    if df['mid_price'].iloc[-1] is not None:
        print(f"   Current mid-price: {df['mid_price'].iloc[-1]:.2f}¢")
        print(f"   Current spread: {df['spread'].iloc[-1]:.4f}¢")
        print(f"   Price range: {df['mid_price'].min():.2f}¢ - {df['mid_price'].max():.2f}¢")
        print(f"   Avg spread: {df['spread'].mean():.4f}¢")
        print(f"   Volatility (std): {df['mid_price'].std():.4f}¢")
    else:
        print(f"   ⚠️  No price data available (market may not have active orders)")
        return
    
    # Recent movement
    if len(df) > 1:
        recent_change = df['mid_price'].iloc[-1] - df['mid_price'].iloc[0]
        recent_pct = (recent_change / df['mid_price'].iloc[0]) * 100 if df['mid_price'].iloc[0] != 0 else 0
        print(f"\n📈 Movement Since Start:")
        print(f"   Change: {recent_change:+.2f}¢ ({recent_pct:+.2f}%)")
        
        if len(df) >= 24:
            change_24h = df['mid_price'].iloc[-1] - df['mid_price'].iloc[-24]
            pct_24h = (change_24h / df['mid_price'].iloc[-24]) * 100 if df['mid_price'].iloc[-24] != 0 else 0
            print(f"   24h change: {change_24h:+.2f}¢ ({pct_24h:+.2f}%)")
    
    print("\n" + "=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize Kalshi order book snapshots from S3',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('ticker', nargs='?', help='Market ticker (e.g., KXGREENLAND-29)')
    parser.add_argument('--list', action='store_true', help='List all markets with snapshots')
    parser.add_argument('--stats-only', action='store_true', help='Show statistics only (no charts)')
    parser.add_argument('--compare', help='Compare order book over time (e.g., 24h)')
    parser.add_argument('--save', help='Save charts to local directory (deprecated - use --s3)')
    parser.add_argument('--s3', action='store_true', help='Save charts to S3 (for email use)')
    parser.add_argument('--open', action='store_true', help='Open S3 chart URLs in browser after saving')
    parser.add_argument('--interactive', action='store_true', help='Open interactive chart in browser')
    
    args = parser.parse_args()
    
    # List markets mode
    if args.list:
        print("📋 Markets with snapshots in S3:")
        print("=" * 80)
        markets = list_all_markets()
        if markets:
            for i, ticker in enumerate(markets, 1):
                print(f"{i:3d}. {ticker}")
            print(f"\nTotal: {len(markets)} markets")
        else:
            print("No markets found")
        return
    
    # Require ticker for other modes
    if not args.ticker:
        parser.error("ticker is required (unless using --list)")
    
    ticker = args.ticker.upper()
    
    # Stats only mode
    if args.stats_only:
        print_market_stats(ticker)
        return
    
    # Chart generation mode (default)
    print(f"📊 Generating charts for {ticker}...")
    df = load_all_snapshots(ticker)
    
    if df.empty:
        print("❌ No data found for this market")
        return
    
    if df['mid_price'].iloc[-1] is None:
        print("⚠️  No price data available (market may not have active orders)")
        print("Run with --stats-only to see what data exists")
        return
    
    # Determine save mode (S3 vs local)
    save_to_s3 = args.s3
    save_dir = None
    chart_urls = []  # Track URLs for --open flag
    
    if args.save and not save_to_s3:
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Saving charts locally to: {save_dir}")
    elif save_to_s3:
        print(f"☁️  Saving charts to S3 bucket: {S3_BUCKET}")
    
    # Generate Chart 1: Price movement
    print("\n📈 Chart 1: Price Movement Over Time")
    
    if save_to_s3:
        chart1_url = create_price_chart(df, ticker, save_to_s3=True)
        print(f"   Public URL: {chart1_url}")
        chart_urls.append(chart1_url)
    elif save_dir:
        chart1_path = save_dir / f"{ticker}_price_movement.png"
        create_price_chart(df, ticker, save_path=str(chart1_path))
    else:
        create_price_chart(df, ticker)
    
    # Generate Chart 2: Order book comparison
    if len(df) >= 2:
        print("\n📊 Chart 2: Order Book Comparison")
        
        # Determine before/after indices
        before_idx = 0
        after_idx = -1
        
        if args.compare:
            # Parse compare argument (e.g., "24h")
            if args.compare.endswith('h'):
                try:
                    hours = int(args.compare[:-1])
                    target_time = df['timestamp'].iloc[-1] - timedelta(hours=hours)
                    # Find closest snapshot
                    before_idx = (df['timestamp'] - target_time).abs().idxmin()
                except:
                    print(f"⚠️  Invalid compare format: {args.compare}, using first snapshot")
        
        if save_to_s3:
            chart2_url = create_orderbook_comparison(df, ticker, before_idx, after_idx, save_to_s3=True)
            print(f"   Public URL: {chart2_url}")
            chart_urls.append(chart2_url)
        elif save_dir:
            chart2_path = save_dir / f"{ticker}_orderbook_comparison.png"
            create_orderbook_comparison(df, ticker, before_idx, after_idx, save_path=str(chart2_path))
        else:
            create_orderbook_comparison(df, ticker, before_idx, after_idx)
    else:
        print("\n⚠️  Need at least 2 snapshots for order book comparison")
    
    # Open URLs in browser if requested
    if args.open and chart_urls:
        import webbrowser
        print("\n🌐 Opening charts in browser...")
        for url in chart_urls:
            webbrowser.open(url)
        print(f"   Opened {len(chart_urls)} chart(s) in your default browser")
    
    if save_to_s3:
        print("\n💡 Charts saved to S3 with public URLs for use in SES emails")
        if not args.open:
            print("   Use --open flag to automatically open charts in browser")
    elif not save_dir:
        print("\n💡 Tip: Use --s3 to save images to S3 for email use")
    
    print("\n✅ Complete")


if __name__ == '__main__':
    main()



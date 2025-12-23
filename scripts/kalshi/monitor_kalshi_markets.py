"""
Monitor Kalshi prediction markets for price and order book changes.

Tracks full order book distribution to detect:
- Price movements
- Liquidity changes
- Order book imbalances
- Large orders appearing/disappearing

Every run:
1. Fetch current order book for tracked markets
2. Save full order book as JSON
3. Calculate summary metrics and save to CSV
4. Compare to previous snapshot
5. Alert if significant changes detected

Usage:
    python scripts/monitor_kalshi_markets.py

Setup as cron job for regular monitoring:
    */10 * * * * cd /Users/thomasmyles/dev/betting && python scripts/monitor_kalshi_markets.py
"""

import requests
import json
import csv
import os
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Kalshi API
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Markets to monitor (start with 2)
MARKETS_TO_MONITOR = [
    "KXELONMARS-99",          # Elon Mars - highest volume
    "KXPERSONPRESFUENTES-45"  # President - political, active
]

# Paths
BASE_DIR = Path(__file__).parent.parent
ORDER_BOOKS_DIR = BASE_DIR / "data" / "04_output" / "prediction_markets" / "order_books"
SUMMARY_CSV = BASE_DIR / "data" / "04_output" / "prediction_markets" / "snapshots_summary.csv"

# Create directories if needed
ORDER_BOOKS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)

# Alert thresholds (we'll refine these as we see data)
ALERT_PRICE_CHANGE_PCT = 3.0      # Alert if price moves >3%
ALERT_DEPTH_CHANGE_PCT = 50.0     # Alert if liquidity drops >50%
ALERT_IMBALANCE_SHIFT = 0.15      # Alert if imbalance shifts >15%


# =============================================================================
# API FUNCTIONS
# =============================================================================

def get_order_book(market_ticker):
    """Fetch full order book for a market."""
    url = f"{KALSHI_API_BASE}/markets/{market_ticker}/orderbook"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error fetching {market_ticker}: {e}")
        return None


# =============================================================================
# ORDER BOOK ANALYSIS
# =============================================================================

def calculate_health_score(yes_orders, no_orders, mid_price, total_depth):
    """
    Calculate composite health score (0-100) for order book quality.
    
    Combines: weighted depth, spread, balance, concentration, execution quality.
    Higher score = better tradeable market.
    """
    if not yes_orders or not no_orders or total_depth == 0:
        return 0
    
    scores = {}
    
    # 1. Weighted depth (liquidity near price matters more)
    def calc_weighted_depth(orders, ref_price, decay=0.2):
        return sum(size * (1.0 / (1.0 + decay * abs(p/100 - ref_price) * 100))
                   for p, size in orders)
    
    yes_weighted = calc_weighted_depth(yes_orders, mid_price)
    no_weighted = calc_weighted_depth(no_orders, 1.0 - mid_price)
    weighted_total = yes_weighted + no_weighted
    
    scores['weighted_depth'] = min(100, (weighted_total / 10000) * 100)
    
    # 2. Spread
    best_yes = yes_orders[-1][0] / 100
    best_no = no_orders[-1][0] / 100
    spread = abs(1.0 - best_yes - best_no)
    scores['spread'] = max(0, 100 - spread * 1000)
    
    # 3. Balance
    yes_total = sum(s for _, s in yes_orders)
    imbalance = yes_total / total_depth
    if 0.3 <= imbalance <= 0.7:
        scores['balance'] = 100
    else:
        scores['balance'] = max(0, 100 - abs(imbalance - 0.5) * 200)
    
    # 4. Concentration
    largest = max(max(s for _, s in yes_orders), max(s for _, s in no_orders))
    concentration_ratio = largest / total_depth
    if concentration_ratio < 0.2:
        scores['concentration'] = 100
    elif concentration_ratio > 0.5:
        scores['concentration'] = 0
    else:
        scores['concentration'] = 100 - ((concentration_ratio - 0.2) / 0.3) * 100
    
    # Weighted composite (simplified - no execution for speed)
    weights = {
        'weighted_depth': 0.35,
        'spread': 0.30,
        'balance': 0.20,
        'concentration': 0.15
    }
    
    return sum(scores[k] * weights[k] for k in scores)


def analyze_order_book(order_book_data):
    """
    Calculate metrics from full order book distribution.
    
    Returns dict with:
    - mid_price
    - best_bid, best_ask, spread
    - bid/ask depth at various levels
    - total_depth
    - imbalance_ratio
    - largest_bid, largest_ask (detect walls)
    - weighted_avg_bid_price, weighted_avg_ask_price
    - health_score (0-100, composite quality metric)
    """
    if not order_book_data or 'orderbook' not in order_book_data:
        return None
    
    book = order_book_data['orderbook']
    yes_orders = book.get('yes', [])  # Format: [[price_cents, size], ...]
    no_orders = book.get('no', [])
    
    if not yes_orders or not no_orders:
        return None
    
    # Best bid/ask (YES side is buying the event, NO side is selling)
    best_yes_bid = yes_orders[-1][0] / 100  # Highest YES price (buying event)
    best_no_bid = no_orders[-1][0] / 100    # Highest NO price (selling event)
    
    # Mid price (since YES + NO should ~= 1.0)
    mid_price = best_yes_bid
    spread = abs(1.0 - best_yes_bid - best_no_bid)
    
    # Calculate depth at different price levels from mid
    def calc_depth_within(orders, mid, distance_pct):
        """Sum size of orders within distance_pct of mid price."""
        lower = mid * (1 - distance_pct)
        upper = mid * (1 + distance_pct)
        return sum(size for price_cents, size in orders 
                   if lower <= price_cents/100 <= upper)
    
    # YES side depth (bids)
    bid_depth_1pct = calc_depth_within(yes_orders, mid_price, 0.01)
    bid_depth_2pct = calc_depth_within(yes_orders, mid_price, 0.02)
    bid_depth_5pct = calc_depth_within(yes_orders, mid_price, 0.05)
    bid_depth_total = sum(size for _, size in yes_orders)
    
    # NO side depth (asks)
    ask_depth_1pct = calc_depth_within(no_orders, 1.0 - mid_price, 0.01)
    ask_depth_2pct = calc_depth_within(no_orders, 1.0 - mid_price, 0.02)
    ask_depth_5pct = calc_depth_within(no_orders, 1.0 - mid_price, 0.05)
    ask_depth_total = sum(size for _, size in no_orders)
    
    # Total depth
    total_depth_5pct = bid_depth_5pct + ask_depth_5pct
    total_depth_all = bid_depth_total + ask_depth_total
    
    # Imbalance ratio (0.5 = balanced, >0.5 = more bids, <0.5 = more asks)
    imbalance_ratio = bid_depth_5pct / (total_depth_5pct + 1)  # +1 to avoid div by zero
    
    # Find largest single orders (walls)
    largest_bid = max(yes_orders, key=lambda x: x[1])
    largest_ask = max(no_orders, key=lambda x: x[1])
    
    # Weighted average prices
    weighted_bid_price = sum(p/100 * s for p, s in yes_orders) / (bid_depth_total + 1)
    weighted_ask_price = sum(p/100 * s for p, s in no_orders) / (ask_depth_total + 1)
    
    # Calculate health score
    health_score = calculate_health_score(yes_orders, no_orders, mid_price, total_depth_all)
    
    return {
        'mid_price': round(mid_price, 4),
        'best_bid': round(best_yes_bid, 4),
        'best_ask': round(1.0 - best_yes_bid, 4),
        'spread': round(spread, 4),
        'bid_depth_1pct': bid_depth_1pct,
        'bid_depth_2pct': bid_depth_2pct,
        'bid_depth_5pct': bid_depth_5pct,
        'bid_depth_total': bid_depth_total,
        'ask_depth_1pct': ask_depth_1pct,
        'ask_depth_2pct': ask_depth_2pct,
        'ask_depth_5pct': ask_depth_5pct,
        'ask_depth_total': ask_depth_total,
        'total_depth_5pct': total_depth_5pct,
        'total_depth_all': total_depth_all,
        'imbalance_ratio': round(imbalance_ratio, 3),
        'largest_bid_price': largest_bid[0] / 100,
        'largest_bid_size': largest_bid[1],
        'largest_ask_price': largest_ask[0] / 100,
        'largest_ask_size': largest_ask[1],
        'weighted_bid_price': round(weighted_bid_price, 4),
        'weighted_ask_price': round(weighted_ask_price, 4),
        'num_bid_levels': len(yes_orders),
        'num_ask_levels': len(no_orders),
        'health_score': round(health_score, 1),
    }


# =============================================================================
# DATA STORAGE
# =============================================================================

def save_order_book_json(market_ticker, order_book_data, timestamp):
    """Save full order book as JSON for detailed analysis."""
    filename = f"{market_ticker}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    filepath = ORDER_BOOKS_DIR / filename
    
    with open(filepath, 'w') as f:
        json.dump({
            'timestamp': timestamp.isoformat(),
            'market_ticker': market_ticker,
            'data': order_book_data
        }, f, indent=2)
    
    return filepath


def save_summary_csv(market_ticker, metrics, timestamp):
    """Append summary metrics to CSV."""
    file_exists = SUMMARY_CSV.exists()
    
    row = {
        'timestamp': timestamp.isoformat(),
        'market_ticker': market_ticker,
        **metrics
    }
    
    fieldnames = list(row.keys())
    
    with open(SUMMARY_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_previous_snapshot(market_ticker):
    """Load most recent snapshot for comparison."""
    if not SUMMARY_CSV.exists():
        return None
    
    # Read last row for this market
    previous = None
    with open(SUMMARY_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['market_ticker'] == market_ticker:
                previous = row
    
    return previous


# =============================================================================
# ALERTING
# =============================================================================

def check_alerts(market_ticker, current_metrics, previous_metrics):
    """Compare current vs previous and return alerts."""
    alerts = []
    
    if not previous_metrics:
        return alerts
    
    # Price change
    prev_price = float(previous_metrics['mid_price'])
    curr_price = current_metrics['mid_price']
    price_change_pct = ((curr_price - prev_price) / prev_price) * 100
    
    if abs(price_change_pct) >= ALERT_PRICE_CHANGE_PCT:
        alerts.append({
            'type': 'PRICE_MOVE',
            'message': f"Price moved {price_change_pct:+.2f}% ({prev_price:.3f} → {curr_price:.3f})",
            'severity': 'HIGH'
        })
    
    # Liquidity change
    prev_depth = float(previous_metrics['total_depth_5pct'])
    curr_depth = current_metrics['total_depth_5pct']
    depth_change_pct = ((curr_depth - prev_depth) / prev_depth) * 100 if prev_depth > 0 else 0
    
    if depth_change_pct < -ALERT_DEPTH_CHANGE_PCT:
        alerts.append({
            'type': 'LIQUIDITY_DROP',
            'message': f"Liquidity dropped {depth_change_pct:.1f}% ({prev_depth:,.0f} → {curr_depth:,.0f})",
            'severity': 'HIGH'
        })
    
    # Health score degradation
    if 'health_score' in previous_metrics and 'health_score' in current_metrics:
        prev_health = float(previous_metrics['health_score'])
        curr_health = current_metrics['health_score']
        health_change = curr_health - prev_health
        
        if curr_health < 40:
            alerts.append({
                'type': 'POOR_HEALTH',
                'message': f"Market health poor: {curr_health:.1f}/100 (was {prev_health:.1f})",
                'severity': 'HIGH'
            })
        elif health_change < -20:
            alerts.append({
                'type': 'HEALTH_DROP',
                'message': f"Market health dropped {health_change:.1f} points ({prev_health:.1f} → {curr_health:.1f})",
                'severity': 'HIGH'
            })
    
    # Imbalance shift
    prev_imbalance = float(previous_metrics['imbalance_ratio'])
    curr_imbalance = current_metrics['imbalance_ratio']
    imbalance_shift = abs(curr_imbalance - prev_imbalance)
    
    if imbalance_shift >= ALERT_IMBALANCE_SHIFT:
        alerts.append({
            'type': 'IMBALANCE_SHIFT',
            'message': f"Order book imbalance shifted {imbalance_shift:.3f} ({prev_imbalance:.3f} → {curr_imbalance:.3f})",
            'severity': 'MEDIUM'
        })
    
    # Large wall detection
    if current_metrics['largest_bid_size'] > 5000:
        alerts.append({
            'type': 'BID_WALL',
            'message': f"Large bid wall: {current_metrics['largest_bid_size']:,} at {current_metrics['largest_bid_price']:.3f}",
            'severity': 'INFO'
        })
    
    if current_metrics['largest_ask_size'] > 5000:
        alerts.append({
            'type': 'ASK_WALL',
            'message': f"Large ask wall: {current_metrics['largest_ask_size']:,} at {current_metrics['largest_ask_price']:.3f}",
            'severity': 'INFO'
        })
    
    return alerts


def print_alerts(market_ticker, alerts):
    """Print alerts to console (later: send email)."""
    if not alerts:
        return
    
    print(f"\n{'='*80}")
    print(f"🚨 ALERTS: {market_ticker}")
    print(f"{'='*80}")
    
    for alert in alerts:
        emoji = "🔴" if alert['severity'] == 'HIGH' else "🟡"
        print(f"{emoji} [{alert['type']}] {alert['message']}")
    
    print(f"{'='*80}\n")


# =============================================================================
# MAIN MONITORING LOOP
# =============================================================================

def monitor_market(market_ticker, timestamp):
    """Monitor a single market."""
    print(f"\n📊 Monitoring {market_ticker}...")
    
    # Fetch order book
    order_book_data = get_order_book(market_ticker)
    if not order_book_data:
        print(f"   ⚠️  Could not fetch order book")
        return
    
    # Analyze order book
    metrics = analyze_order_book(order_book_data)
    if not metrics:
        print(f"   ⚠️  Could not analyze order book")
        return
    
    # Save data
    json_path = save_order_book_json(market_ticker, order_book_data, timestamp)
    save_summary_csv(market_ticker, metrics, timestamp)
    
    # Load previous for comparison
    previous = load_previous_snapshot(market_ticker)
    
    # Check for alerts
    alerts = check_alerts(market_ticker, metrics, previous)
    
    # Print summary
    print(f"   ✅ Price: {metrics['mid_price']:.3f}")
    print(f"   ✅ Depth (5%): {metrics['total_depth_5pct']:,}")
    print(f"   ✅ Imbalance: {metrics['imbalance_ratio']:.3f}")
    print(f"   ✅ Saved: {json_path.name}")
    
    if alerts:
        print_alerts(market_ticker, alerts)
    
    return metrics, alerts


def main():
    """Main monitoring function."""
    print("="*80)
    print("KALSHI MARKET MONITOR")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Markets: {', '.join(MARKETS_TO_MONITOR)}")
    
    timestamp = datetime.now()
    all_alerts = []
    
    for market_ticker in MARKETS_TO_MONITOR:
        try:
            metrics, alerts = monitor_market(market_ticker, timestamp)
            if alerts:
                all_alerts.extend([(market_ticker, a) for a in alerts])
        except Exception as e:
            print(f"❌ Error monitoring {market_ticker}: {e}")
    
    # Summary
    print("\n" + "="*80)
    if all_alerts:
        print(f"✅ Monitoring complete. {len(all_alerts)} alerts triggered.")
    else:
        print(f"✅ Monitoring complete. No alerts.")
    print("="*80)


if __name__ == "__main__":
    main()


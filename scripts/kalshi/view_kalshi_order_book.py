"""
View and analyze Kalshi order book distribution.

Helps visualize the full order book to understand:
- Where liquidity sits
- How spread out vs concentrated it is
- Presence of walls or gaps
- Shape of distribution

Usage:
    python scripts/view_kalshi_order_book.py MARKET_TICKER

Example:
    python scripts/view_kalshi_order_book.py KXELONMARS-99
"""

import sys
import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
ORDER_BOOKS_DIR = BASE_DIR / "data" / "04_output" / "prediction_markets" / "order_books"

# =============================================================================
# FILL DETECTION CONFIG
# =============================================================================

# Legitimate liquidity is defined as orders within reasonable distance of best bid
# Spread = price ± min(MAX_SPREAD_CENTS, SPREAD_PCT * price)
MAX_SPREAD_CENTS = 0.05  # Maximum 5 cents from best bid
SPREAD_PCT = 0.20  # OR 20% of price (whichever is smaller)

# Fill significance thresholds
FILL_THRESHOLD_PCT = 0.03  # 3% of legitimate volume
MIN_FILL_SIZE = 250  # Absolute minimum contracts to flag as significant

# Display options
ALERT_ON_SIGNIFICANT = True  # Show 🔥 section for significant fills
LOG_ALL_FILLS = True  # Show 📊 section for complete fill record

# =============================================================================
# OVERREACTION SCORE CONFIG (Domer's fade vs follow framework)
# =============================================================================

# Fill velocity thresholds (contracts per minute)
FILL_VELOCITY_HIGH = 2000  # >2000/min = panic (overreaction)
FILL_VELOCITY_MODERATE = 1000  # 1000-2000/min = active
FILL_VELOCITY_LOW = 300  # <300/min = orderly (potential underreaction)

# Aggression ratio thresholds (aggressive fills / total volume)
AGGRESSION_PANIC = 0.75  # >75% aggressive = retail stampede (overreaction)
AGGRESSION_ORDERLY = 0.40  # <40% aggressive = patient capital (potential underreaction)

# Spread widening thresholds
SPREAD_CHAOS = 3.0  # 3x+ widening = confusion (overreaction)
SPREAD_MODERATE = 1.5  # 1.5-3x widening = active

# Liquidity drain thresholds
DEPTH_DRAIN = 0.6  # <60% remaining = exhaustion (overreaction)
DEPTH_MODERATE = 0.8  # 60-80% remaining = active
DEPTH_GROWTH = 1.2  # >120% depth = accumulation (potential underreaction)


def get_spread_threshold(price):
    """
    Calculate legitimate spread threshold based on price.
    
    Returns: min(MAX_SPREAD_CENTS, SPREAD_PCT * price)
    
    Examples:
        price=0.10 → min(0.05, 0.02) = 0.02 → range [0.08-0.12]
        price=0.50 → min(0.05, 0.10) = 0.05 → range [0.45-0.55]
        price=0.90 → min(0.05, 0.18) = 0.05 → range [0.85-0.95]
    """
    return min(MAX_SPREAD_CENTS, SPREAD_PCT * price)


def get_legitimate_orders(orders, best_price):
    """
    Filter orders that are within legitimate trading range of best price.
    
    Args:
        orders: List of (price_cents, size) tuples
        best_price: Best bid price in decimal (e.g., 0.07)
    
    Returns:
        List of (price_cents, size) for orders within threshold
    """
    if not orders or best_price == 0:
        return []
    
    threshold = get_spread_threshold(best_price)
    legit_orders = []
    
    for price_cents, size in orders:
        price_dec = price_cents / 100
        if abs(price_dec - best_price) <= threshold:
            legit_orders.append((price_cents, size))
    
    return legit_orders


def load_latest_order_book(market_ticker):
    """Load most recent order book JSON for a market."""
    files = sorted(ORDER_BOOKS_DIR.glob(f"{market_ticker}_*.json"))
    if not files:
        print(f"No order book data found for {market_ticker}")
        return None
    
    latest = files[-1]
    with open(latest, 'r') as f:
        return json.load(f)


def load_previous_order_book(market_ticker):
    """Load second-most recent order book JSON for comparison."""
    files = sorted(ORDER_BOOKS_DIR.glob(f"{market_ticker}_*.json"))
    if len(files) < 2:
        return None
    
    previous = files[-2]
    with open(previous, 'r') as f:
        return json.load(f)


def visualize_order_book(order_book_data):
    """Create ASCII visualization of order book distribution."""
    if not order_book_data:
        return
    
    book = order_book_data['data']['orderbook']
    yes_orders = book.get('yes', [])
    no_orders = book.get('no', [])
    
    print("\n" + "="*80)
    print("ORDER BOOK DISTRIBUTION")
    print("="*80)
    print(f"Timestamp: {order_book_data['timestamp']}")
    print(f"Market: {order_book_data['market_ticker']}")
    print()
    print("ℹ️  These are UNFILLED LIMIT ORDERS waiting for counterparties.")
    print("   If not filled by expiration, orders are cancelled (no loss/gain).")
    
    # Fill in missing price levels with 0s
    yes_filled = fill_missing_prices(yes_orders)
    no_filled = fill_missing_prices(no_orders)
    
    # Calculate best prices for spread display
    best_yes_bid = yes_orders[-1][0] / 100 if yes_orders else 0
    best_no_bid = no_orders[-1][0] / 100 if no_orders else 0
    best_no_ask_as_yes = 1.0 - best_no_bid  # Convert NO price to YES equivalent
    
    # YES side (bids - buying the event)
    print("\n--- YES SIDE (Bids - Buying Event) ---")
    print(f"{'Price':>8} {'Size':>10} {'Visual'}")
    print("-" * 50)
    
    max_size = max(max((s for _, s in yes_filled), default=0),
                   max((s for _, s in no_filled), default=0))
    
    display_orders_compressed(list(reversed(yes_filled)), max_size)
    
    # Market spread separator
    print("\n" + "━" * 80)
    spread_width = abs(best_no_ask_as_yes - best_yes_bid)
    print(f"        MARKET SPREAD: {best_yes_bid:.2f} (YES bid) ←→ {best_no_ask_as_yes:.2f} (NO ask as YES) | Width: {spread_width:.2f}")
    print("━" * 80)
    
    # NO side (asks - selling the event)
    print("\n--- NO SIDE (Asks - Selling Event) ---")
    print(f"{'Price':>8} {'Size':>10} {'Visual'}")
    print("-" * 50)
    
    display_orders_compressed(no_filled, max_size)
    
    # Summary stats
    print("\n" + "="*80)
    print("DISTRIBUTION STATS - Unfilled Limit Orders")
    print("="*80)
    
    yes_total = sum(s for _, s in yes_orders)
    no_total = sum(s for _, s in no_orders)
    total = yes_total + no_total
    
    yes_weighted_price = sum(p/100 * s for p, s in yes_orders) / yes_total if yes_total > 0 else 0
    no_weighted_price = sum(p/100 * s for p, s in no_orders) / no_total if no_total > 0 else 0
    
    print(f"YES side: {len(yes_orders)} levels, {yes_total:,} total size (unfilled bids)")
    print(f"NO side:  {len(no_orders)} levels, {no_total:,} total size (unfilled asks)")
    print(f"Total liquidity: {total:,} contracts waiting in book")
    print(f"")
    print(f"Weighted avg YES price: {yes_weighted_price:.2f}")
    print(f"Weighted avg NO price:  {no_weighted_price:.2f}")
    print(f"")
    print(f"Best YES bid: {yes_orders[-1][0]/100:.2f}")
    print(f"Best NO bid:  {no_orders[-1][0]/100:.2f}")
    print(f"Implied spread: {abs(1.0 - yes_orders[-1][0]/100 - no_orders[-1][0]/100):.2f}")
    
    # Concentration analysis
    yes_prices = [p/100 for p, _ in yes_orders]
    no_prices = [p/100 for p, _ in no_orders]
    
    print(f"")
    print(f"YES price range: {min(yes_prices):.2f} to {max(yes_prices):.2f}")
    print(f"NO price range:  {min(no_prices):.2f} to {max(no_prices):.2f}")
    
    # Find largest orders (walls)
    largest_yes = max(yes_orders, key=lambda x: x[1])
    largest_no = max(no_orders, key=lambda x: x[1])
    
    print(f"")
    print(f"Largest YES order: {largest_yes[1]:,} at {largest_yes[0]/100:.2f} (unfilled limit buy)")
    print(f"Largest NO order:  {largest_no[1]:,} at {largest_no[0]/100:.2f} (unfilled limit buy)")
    print(f"")
    print(f"💡 NOTE: To detect FILLS, compare snapshots over time.")
    print(f"   Shrinking/disappearing orders = someone took the other side.")
    
    # Check if orders are concentrated or spread out
    yes_std = calculate_weighted_std([p/100 for p, _ in yes_orders],
                                     [s for _, s in yes_orders])
    no_std = calculate_weighted_std([p/100 for p, _ in no_orders],
                                    [s for _, s in no_orders])
    
    print(f"")
    print(f"YES price concentration (std): {yes_std:.2f} {'(tight)' if yes_std < 0.02 else '(spread out)'}")
    print(f"NO price concentration (std):  {no_std:.2f} {'(tight)' if no_std < 0.02 else '(spread out)'}")


def fill_missing_prices(orders):
    """Fill in missing price levels with 0 size for complete distribution.
    
    Always fills from 1 cent to 99 cents for complete market view.
    """
    # Convert to dict for easy lookup
    price_dict = {price: size for price, size in orders}
    
    # Fill in all prices from 1 to 99 cents (full market range)
    filled_orders = []
    for price_cents in range(1, 100):
        size = price_dict.get(price_cents, 0)
        filled_orders.append((price_cents, size))
    
    return filled_orders


def display_orders_compressed(orders, max_size):
    """Display orders, compressing consecutive zeros into summary lines."""
    if not orders:
        return
    
    i = 0
    while i < len(orders):
        price_cents, size = orders[i]
        
        if size == 0:
            # Find consecutive zeros
            start_price = price_cents
            zero_count = 0
            while i < len(orders) and orders[i][1] == 0:
                zero_count += 1
                i += 1
            end_price = orders[i-1][0]
            
            # Show compressed summary
            if zero_count > 1:
                print(f"{start_price/100:>8.2f}-{end_price/100:.2f}  [gap: {zero_count} levels]")
            else:
                price = start_price / 100
                bar_length = int((0 / max_size) * 40) if max_size > 0 else 0
                bar = "█" * bar_length
                print(f"{price:>8.2f} {0:>10,} {bar}")
        else:
            # Show individual non-zero entry
            price = price_cents / 100
            bar_length = int((size / max_size) * 40) if max_size > 0 else 0
            bar = "█" * bar_length
            print(f"{price:>8.2f} {size:>10,} {bar}")
            i += 1


def calculate_weighted_std(prices, weights):
    """Calculate weighted standard deviation."""
    if not prices or sum(weights) == 0:
        return 0
    
    weighted_mean = sum(p * w for p, w in zip(prices, weights)) / sum(weights)
    variance = sum(w * (p - weighted_mean)**2 for p, w in zip(prices, weights)) / sum(weights)
    return variance ** 0.5


def detect_fills(prev_book, curr_book):
    """
    Compare two order book snapshots and detect fill activity.
    
    Returns dict with:
        - yes_fills: List of {price, delta, pct_of_volume} for YES side
        - no_fills: List of {price, delta, pct_of_volume} for NO side
        - yes_legit_range: (min, max) for legitimate YES prices
        - no_legit_range: (min, max) for legitimate NO prices
        - time_delta: Seconds between snapshots
        - momentum: Overall market direction
    """
    if not prev_book or not curr_book:
        return None
    
    prev_yes = prev_book['data']['orderbook'].get('yes', [])
    prev_no = prev_book['data']['orderbook'].get('no', [])
    curr_yes = curr_book['data']['orderbook'].get('yes', [])
    curr_no = curr_book['data']['orderbook'].get('no', [])
    
    # Get best prices
    prev_best_yes = prev_yes[-1][0] / 100 if prev_yes else 0
    curr_best_yes = curr_yes[-1][0] / 100 if curr_yes else 0
    prev_best_no = prev_no[-1][0] / 100 if prev_no else 0
    curr_best_no = curr_no[-1][0] / 100 if curr_no else 0
    
    # Get legitimate orders only
    prev_yes_legit = get_legitimate_orders(prev_yes, prev_best_yes)
    curr_yes_legit = get_legitimate_orders(curr_yes, curr_best_yes)
    prev_no_legit = get_legitimate_orders(prev_no, prev_best_no)
    curr_no_legit = get_legitimate_orders(curr_no, curr_best_no)
    
    # Calculate legitimate volume
    prev_yes_vol = sum(s for _, s in prev_yes_legit)
    prev_no_vol = sum(s for _, s in prev_no_legit)
    
    # Set thresholds
    yes_threshold = max(MIN_FILL_SIZE, prev_yes_vol * FILL_THRESHOLD_PCT) if prev_yes_vol > 0 else MIN_FILL_SIZE
    no_threshold = max(MIN_FILL_SIZE, prev_no_vol * FILL_THRESHOLD_PCT) if prev_no_vol > 0 else MIN_FILL_SIZE
    
    # Convert to dicts for comparison
    prev_yes_dict = {p: s for p, s in prev_yes_legit}
    curr_yes_dict = {p: s for p, s in curr_yes_legit}
    prev_no_dict = {p: s for p, s in prev_no_legit}
    curr_no_dict = {p: s for p, s in curr_no_legit}
    
    # Detect fills on YES side
    yes_fills = []
    all_yes_prices = set(prev_yes_dict.keys()) | set(curr_yes_dict.keys())
    for price in sorted(all_yes_prices):
        prev_size = prev_yes_dict.get(price, 0)
        curr_size = curr_yes_dict.get(price, 0)
        delta = curr_size - prev_size
        
        if delta != 0:
            yes_fills.append({
                'price': price / 100,
                'delta': delta,
                'prev_size': prev_size,
                'curr_size': curr_size,
                'pct_of_volume': abs(delta) / prev_yes_vol * 100 if prev_yes_vol > 0 else 0,
                'significant': abs(delta) >= yes_threshold
            })
    
    # Detect fills on NO side
    no_fills = []
    all_no_prices = set(prev_no_dict.keys()) | set(curr_no_dict.keys())
    for price in sorted(all_no_prices):
        prev_size = prev_no_dict.get(price, 0)
        curr_size = curr_no_dict.get(price, 0)
        delta = curr_size - prev_size
        
        if delta != 0:
            no_fills.append({
                'price': price / 100,
                'delta': delta,
                'prev_size': prev_size,
                'curr_size': curr_size,
                'pct_of_volume': abs(delta) / prev_no_vol * 100 if prev_no_vol > 0 else 0,
                'significant': abs(delta) >= no_threshold
            })
    
    # Calculate time delta
    prev_time = datetime.fromisoformat(prev_book['timestamp'].replace('Z', '+00:00'))
    curr_time = datetime.fromisoformat(curr_book['timestamp'].replace('Z', '+00:00'))
    time_delta = (curr_time - prev_time).total_seconds()
    
    # Calculate momentum
    yes_price_change = curr_best_yes - prev_best_yes
    no_price_change = curr_best_no - prev_best_no
    
    yes_filled_total = sum(abs(f['delta']) for f in yes_fills if f['delta'] < 0)
    yes_added_total = sum(f['delta'] for f in yes_fills if f['delta'] > 0)
    no_filled_total = sum(abs(f['delta']) for f in no_fills if f['delta'] < 0)
    no_added_total = sum(f['delta'] for f in no_fills if f['delta'] > 0)
    
    # Determine momentum
    momentum = "NEUTRAL"
    if yes_price_change > 0 or (yes_filled_total > no_filled_total and yes_added_total > 0):
        momentum = "BULLISH"
    elif yes_price_change < 0 or (no_filled_total > yes_filled_total and no_added_total > 0):
        momentum = "BEARISH"
    
    # Get legitimate ranges
    yes_threshold_val = get_spread_threshold(curr_best_yes)
    no_threshold_val = get_spread_threshold(curr_best_no)
    
    return {
        'yes_fills': yes_fills,
        'no_fills': no_fills,
        'yes_legit_range': (curr_best_yes - yes_threshold_val, curr_best_yes + yes_threshold_val),
        'no_legit_range': (curr_best_no - no_threshold_val, curr_best_no + no_threshold_val),
        'time_delta': time_delta,
        'momentum': momentum,
        'yes_price_change': yes_price_change,
        'no_price_change': no_price_change,
        'yes_filled_total': yes_filled_total,
        'yes_added_total': yes_added_total,
        'no_filled_total': no_filled_total,
        'no_added_total': no_added_total,
        'prev_yes_vol': prev_yes_vol,
        'prev_no_vol': prev_no_vol,
        'curr_yes_vol': sum(s for _, s in curr_yes_legit),
        'curr_no_vol': sum(s for _, s in curr_no_legit),
    }


def display_fill_detection(fill_data):
    """Display fill detection results in a clear format."""
    if not fill_data:
        return
    
    print("\n" + "="*80)
    print(f"FILL DETECTION (Since Last Snapshot: {fill_data['time_delta']:.0f} seconds ago)")
    print("="*80)
    
    # Show legitimate ranges
    yes_min, yes_max = fill_data['yes_legit_range']
    no_min, no_max = fill_data['no_legit_range']
    print(f"\nLegitimate Liquidity Defined As:")
    print(f"  YES: [{yes_min:.2f}-{yes_max:.2f}] (±{get_spread_threshold((yes_min+yes_max)/2):.2f} from best)")
    print(f"  NO:  [{no_min:.2f}-{no_max:.2f}] (±{get_spread_threshold((no_min+no_max)/2):.2f} from best)")
    
    # Significant fills section
    if ALERT_ON_SIGNIFICANT:
        yes_sig = [f for f in fill_data['yes_fills'] if f['significant']]
        no_sig = [f for f in fill_data['no_fills'] if f['significant']]
        
        print(f"\n🔥 SIGNIFICANT FILLS (>{FILL_THRESHOLD_PCT*100:.0f}% of legit volume OR >{MIN_FILL_SIZE} contracts):")
        if yes_sig or no_sig:
            for f in yes_sig:
                action = "filled" if f['delta'] < 0 else "added"
                print(f"  ✅ YES {f['price']:.2f}: {abs(f['delta']):,} contracts {action} ({f['pct_of_volume']:.1f}% of volume)")
            for f in no_sig:
                action = "filled" if f['delta'] < 0 else "added"
                print(f"  ✅ NO {f['price']:.2f}: {abs(f['delta']):,} contracts {action} ({f['pct_of_volume']:.1f}% of volume)")
        else:
            print(f"  ❌ No significant fills detected")
    
    # All fills section
    if LOG_ALL_FILLS and (fill_data['yes_fills'] or fill_data['no_fills']):
        print(f"\n📊 ALL FILLS IN LEGIT RANGE:")
        
        if fill_data['yes_fills']:
            print(f"   YES side:")
            for f in sorted(fill_data['yes_fills'], key=lambda x: x['price'], reverse=True):
                if f['delta'] < 0:
                    print(f"     {f['price']:.2f}: {abs(f['delta']):,} contracts filled")
                else:
                    print(f"     {f['price']:.2f}: +{f['delta']:,} contracts added (new liquidity)")
        
        if fill_data['no_fills']:
            print(f"   NO side:")
            for f in sorted(fill_data['no_fills'], key=lambda x: x['price']):
                if f['delta'] < 0:
                    print(f"     {f['price']:.2f}: {abs(f['delta']):,} contracts filled")
                else:
                    print(f"     {f['price']:.2f}: +{f['delta']:,} contracts added (new liquidity)")
    
    # Summary
    print(f"\n📈 SUMMARY:")
    
    # Volume changes
    yes_vol_change = fill_data['curr_yes_vol'] - fill_data['prev_yes_vol']
    no_vol_change = fill_data['curr_no_vol'] - fill_data['prev_no_vol']
    yes_vol_pct = yes_vol_change / fill_data['prev_yes_vol'] * 100 if fill_data['prev_yes_vol'] > 0 else 0
    no_vol_pct = no_vol_change / fill_data['prev_no_vol'] * 100 if fill_data['prev_no_vol'] > 0 else 0
    
    print(f"  • YES legit volume: {fill_data['prev_yes_vol']:,} → {fill_data['curr_yes_vol']:,} ({yes_vol_change:+,}, {yes_vol_pct:+.1f}%)")
    print(f"  • NO legit volume:  {fill_data['prev_no_vol']:,} → {fill_data['curr_no_vol']:,} ({no_vol_change:+,}, {no_vol_pct:+.1f}%)")
    
    # Activity
    if fill_data['yes_filled_total'] > 0:
        print(f"  • Total YES filled: {fill_data['yes_filled_total']:,} contracts ({fill_data['yes_filled_total']/fill_data['prev_yes_vol']*100:.1f}% of prev volume)")
    if fill_data['yes_added_total'] > 0:
        print(f"  • Total YES added: {fill_data['yes_added_total']:,} contracts (new bids)")
    if fill_data['no_filled_total'] > 0:
        print(f"  • Total NO filled: {fill_data['no_filled_total']:,} contracts ({fill_data['no_filled_total']/fill_data['prev_no_vol']*100:.1f}% of prev volume)")
    if fill_data['no_added_total'] > 0:
        print(f"  • Total NO added: {fill_data['no_added_total']:,} contracts (new bids)")
    
    # Price movement
    if fill_data['yes_price_change'] != 0:
        direction = "↗️" if fill_data['yes_price_change'] > 0 else "↘️"
        print(f"  {direction} Best YES bid moved: {fill_data['yes_price_change']:+.2f} cents")
    
    # Momentum
    momentum_emoji = {"BULLISH": "↗️", "BEARISH": "↘️", "NEUTRAL": "↔️"}
    print(f"\n  Market Direction: {momentum_emoji[fill_data['momentum']]}  {fill_data['momentum']}")


def compare_snapshots(market_ticker, num_snapshots=3):
    """Show evolution of order book over last N snapshots."""
    files = sorted(ORDER_BOOKS_DIR.glob(f"{market_ticker}_*.json"))[-num_snapshots:]
    
    if len(files) < 2:
        print(f"\nNeed at least 2 snapshots to compare. Run monitor a few times first.")
        return
    
    print("\n" + "="*80)
    print("ORDER BOOK EVOLUTION")
    print("="*80)
    
    snapshots = []
    for f in files:
        with open(f, 'r') as file:
            data = json.load(file)
            book = data['data']['orderbook']
            yes_total = sum(s for _, s in book.get('yes', []))
            no_total = sum(s for _, s in book.get('no', []))
            best_yes = book['yes'][-1][0]/100 if book.get('yes') else 0
            
            snapshots.append({
                'time': data['timestamp'],
                'yes_total': yes_total,
                'no_total': no_total,
                'total': yes_total + no_total,
                'price': best_yes,
                'imbalance': yes_total / (yes_total + no_total) if (yes_total + no_total) > 0 else 0
            })
    
    print(f"{'Timestamp':25} {'Price':>8} {'Total Depth':>12} {'Imbalance':>10} {'Change'}")
    print("-" * 80)
    
    for i, snap in enumerate(snapshots):
        time_str = snap['time'][:19]
        price_change = ""
        depth_change = ""
        
        if i > 0:
            price_diff = ((snap['price'] - snapshots[i-1]['price']) / snapshots[i-1]['price']) * 100
            depth_diff = ((snap['total'] - snapshots[i-1]['total']) / snapshots[i-1]['total']) * 100
            price_change = f"Price: {price_diff:+.1f}%"
            depth_change = f"Depth: {depth_diff:+.1f}%"
        
        print(f"{time_str:25} {snap['price']:>8.2f} {snap['total']:>12,} {snap['imbalance']:>10.2f} {price_change} {depth_change}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/view_kalshi_order_book.py MARKET_TICKER")
        print("\nAvailable markets:")
        if ORDER_BOOKS_DIR.exists():
            markets = set([f.name.split('_')[0] for f in ORDER_BOOKS_DIR.glob("*.json")])
            for market in sorted(markets):
                print(f"  - {market}")
        else:
            print("  No data yet. Run monitor_kalshi_markets.py first.")
        return
    
    market_ticker = sys.argv[1]
    
    # Load and visualize latest
    order_book = load_latest_order_book(market_ticker)
    if order_book:
        visualize_order_book(order_book)
    
    # Detect fills by comparing with previous snapshot
    prev_book = load_previous_order_book(market_ticker)
    if prev_book and order_book:
        fill_data = detect_fills(prev_book, order_book)
        if fill_data:
            display_fill_detection(fill_data)
    else:
        print("\n" + "="*80)
        print("FILL DETECTION")
        print("="*80)
        print("\n⚠️  Need at least 2 snapshots to detect fills.")
        print("   Run monitor_kalshi_markets.py a few more times to build history.")
    
    # Show evolution
    compare_snapshots(market_ticker)
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()


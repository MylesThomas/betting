"""
Analyze and refine order book quantification methodology.

This script explores different metrics to quantify order book quality:
1. Weighted depth (liquidity near price matters more)
2. Liquidity health score
3. Execution quality (slippage for different order sizes)
4. Market impact (how much would price move if you bought X)

Goal: Find metrics that best capture "tradeable" vs "thin/distorted" markets

Usage:
    python scripts/analyze_order_book_metrics.py MARKET_TICKER
"""

import sys
import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
ORDER_BOOKS_DIR = BASE_DIR / "data" / "04_output" / "prediction_markets" / "order_books"
SUMMARY_CSV = BASE_DIR / "data" / "04_output" / "prediction_markets" / "snapshots_summary.csv"


def load_latest_order_book(market_ticker):
    """Load most recent order book JSON."""
    files = sorted(ORDER_BOOKS_DIR.glob(f"{market_ticker}_*.json"))
    if not files:
        return None
    with open(files[-1], 'r') as f:
        return json.load(f)


def calculate_weighted_depth(orders, mid_price, decay_rate=0.1):
    """
    Calculate depth with exponential decay based on distance from mid.
    
    Liquidity far from mid matters less. Decay rate controls how fast.
    Higher decay_rate = only care about nearby liquidity
    
    Args:
        orders: [(price_cents, size), ...]
        mid_price: Current mid price (0-1)
        decay_rate: How quickly to discount far liquidity
    
    Returns:
        Weighted depth score
    """
    weighted_total = 0
    
    for price_cents, size in orders:
        price = price_cents / 100
        distance = abs(price - mid_price)
        weight = 1.0 / (1.0 + decay_rate * distance * 100)  # *100 to scale distance
        weighted_total += size * weight
    
    return weighted_total


def calculate_slippage(orders, order_size, side='buy'):
    """
    Calculate average price you'd get for order_size.
    
    This simulates "walking the book" - as you buy/sell, you consume
    liquidity at each level, getting progressively worse prices.
    
    Args:
        orders: [(price_cents, size), ...]
        order_size: How many shares you want to buy/sell
        side: 'buy' or 'sell'
    
    Returns:
        (avg_price, slippage_pct, levels_consumed)
    """
    if not orders:
        return None, None, 0
    
    # For buying YES, we consume from highest to lowest price
    # For selling, we'd do the opposite (but simplified here)
    sorted_orders = sorted(orders, reverse=(side=='buy'), key=lambda x: x[0])
    
    remaining = order_size
    total_cost = 0
    levels_consumed = 0
    
    for price_cents, available_size in sorted_orders:
        price = price_cents / 100
        
        if remaining <= 0:
            break
        
        filled = min(remaining, available_size)
        total_cost += filled * price
        remaining -= filled
        levels_consumed += 1
    
    if remaining > 0:
        # Not enough liquidity
        return None, None, levels_consumed
    
    avg_price = total_cost / order_size
    entry_price = sorted_orders[0][0] / 100
    slippage_pct = ((avg_price - entry_price) / entry_price) * 100
    
    return avg_price, slippage_pct, levels_consumed


def calculate_market_impact(orders, order_size):
    """
    Estimate how much price would move if you placed order_size.
    
    Different from slippage - this is the *permanent* price change
    you'd cause by absorbing liquidity.
    
    Returns:
        Price impact as % of current best price
    """
    if not orders:
        return None
    
    sorted_orders = sorted(orders, reverse=True, key=lambda x: x[0])
    
    # Find what price level you'd end up at
    remaining = order_size
    final_price = sorted_orders[0][0] / 100
    
    for price_cents, size in sorted_orders:
        if remaining <= 0:
            break
        final_price = price_cents / 100
        remaining -= size
    
    initial_price = sorted_orders[0][0] / 100
    impact_pct = ((final_price - initial_price) / initial_price) * 100
    
    return impact_pct


def calculate_liquidity_health_score(yes_orders, no_orders, mid_price):
    """
    Composite score for overall book health (0-100).
    
    Factors:
    1. Total depth (more is better)
    2. Weighted depth near mid (nearby liquidity matters most)
    3. Balance (not too one-sided)
    4. Spread (tighter is better)
    5. Concentration (not dominated by one huge order)
    6. Execution quality (low slippage for typical order sizes)
    
    Returns:
        Score 0-100, higher is healthier
    """
    if not yes_orders or not no_orders:
        return 0
    
    scores = {}
    
    # 1. Total depth score (normalize to typical market size)
    yes_total = sum(s for _, s in yes_orders)
    no_total = sum(s for _, s in no_orders)
    total_depth = yes_total + no_total
    
    # Score: 0 if <1000, 100 if >100k, linear between
    depth_score = min(100, (total_depth / 100000) * 100)
    scores['depth'] = depth_score
    
    # 2. Weighted depth score (liquidity near mid matters more)
    yes_weighted = calculate_weighted_depth(yes_orders, mid_price, decay_rate=0.2)
    no_weighted = calculate_weighted_depth(no_orders, 1.0 - mid_price, decay_rate=0.2)
    weighted_total = yes_weighted + no_weighted
    
    weighted_score = min(100, (weighted_total / 10000) * 100)
    scores['weighted_depth'] = weighted_score
    
    # 3. Balance score (imbalance between 0.3-0.7 is healthy)
    imbalance = yes_total / (total_depth + 1)
    if 0.3 <= imbalance <= 0.7:
        balance_score = 100
    else:
        # Penalize as we get more extreme
        balance_score = max(0, 100 - abs(imbalance - 0.5) * 200)
    scores['balance'] = balance_score
    
    # 4. Spread score (tighter is better)
    best_yes = yes_orders[-1][0] / 100
    best_no = no_orders[-1][0] / 100
    spread = abs(1.0 - best_yes - best_no)
    
    # 0% spread = 100, 10% spread = 0
    spread_score = max(0, 100 - spread * 1000)
    scores['spread'] = spread_score
    
    # 5. Concentration score (not dominated by walls)
    largest_yes = max(s for _, s in yes_orders)
    largest_no = max(s for _, s in no_orders)
    largest = max(largest_yes, largest_no)
    
    concentration_ratio = largest / total_depth
    # If largest order is <20% of total: good (100)
    # If >50% of total: bad (0)
    if concentration_ratio < 0.2:
        concentration_score = 100
    elif concentration_ratio > 0.5:
        concentration_score = 0
    else:
        concentration_score = 100 - ((concentration_ratio - 0.2) / 0.3) * 100
    scores['concentration'] = concentration_score
    
    # 6. Execution quality (slippage for typical trade sizes)
    # Test with 1000 share order
    avg_price, slippage, _ = calculate_slippage(yes_orders, 1000, 'buy')
    if slippage is not None:
        # <1% slippage = 100, >5% = 0
        # Use abs() since slippage can be negative (price improvement)
        execution_score = max(0, min(100, 100 - abs(slippage) * 20))
    else:
        execution_score = 0  # Can't even fill 1000 shares
    scores['execution'] = execution_score
    
    # Weighted average (you can tune these weights)
    weights = {
        'depth': 0.15,
        'weighted_depth': 0.25,  # Most important - liquidity near price
        'balance': 0.15,
        'spread': 0.20,  # Very important for trading
        'concentration': 0.10,
        'execution': 0.15
    }
    
    final_score = sum(scores[k] * weights[k] for k in scores)
    
    return final_score, scores


def analyze_market(market_ticker):
    """Run full analysis on a market."""
    print("="*80)
    print(f"ORDER BOOK QUANTIFICATION ANALYSIS: {market_ticker}")
    print("="*80)
    
    # Load data
    order_book = load_latest_order_book(market_ticker)
    if not order_book:
        print(f"No data for {market_ticker}")
        return
    
    book = order_book['data']['orderbook']
    yes_orders = book.get('yes', [])
    no_orders = book.get('no', [])
    
    if not yes_orders or not no_orders:
        print("Incomplete order book")
        return
    
    mid_price = yes_orders[-1][0] / 100
    
    print(f"\nMarket: {market_ticker}")
    print(f"Mid Price: {mid_price:.3f}")
    print(f"Timestamp: {order_book['timestamp']}")
    
    # Basic stats
    yes_total = sum(s for _, s in yes_orders)
    no_total = sum(s for _, s in no_orders)
    
    print(f"\n--- BASIC METRICS ---")
    print(f"YES side: {len(yes_orders)} levels, {yes_total:,} size")
    print(f"NO side:  {len(no_orders)} levels, {no_total:,} size")
    print(f"Total depth: {yes_total + no_total:,}")
    print(f"Imbalance: {yes_total/(yes_total+no_total):.3f}")
    
    # Weighted depth with different decay rates
    print(f"\n--- WEIGHTED DEPTH (liquidity near price matters more) ---")
    for decay in [0.05, 0.1, 0.2, 0.5]:
        yes_weighted = calculate_weighted_depth(yes_orders, mid_price, decay)
        no_weighted = calculate_weighted_depth(no_orders, 1.0 - mid_price, decay)
        total_weighted = yes_weighted + no_weighted
        print(f"Decay rate {decay:.2f}: {total_weighted:>10,.0f} (YES: {yes_weighted:,.0f}, NO: {no_weighted:,.0f})")
    
    print("\n💡 Higher decay = only count nearby liquidity")
    print("💡 Lower decay = count liquidity across entire book")
    
    # Execution quality for different sizes
    print(f"\n--- EXECUTION QUALITY (slippage for different order sizes) ---")
    print(f"{'Order Size':>12} {'Avg Price':>12} {'Slippage':>12} {'Levels Used'}")
    print("-" * 60)
    
    for size in [100, 500, 1000, 2000, 5000]:
        avg_price, slippage, levels = calculate_slippage(yes_orders, size, 'buy')
        if avg_price:
            print(f"{size:>12,} {avg_price:>12.4f} {slippage:>11.2f}% {levels:>12}")
        else:
            print(f"{size:>12,} {'NOT ENOUGH LIQUIDITY':>40}")
    
    print("\n💡 Low slippage = can trade without moving price much")
    print("💡 Many levels = price gets worse as you buy more")
    
    # Market impact
    print(f"\n--- MARKET IMPACT (how much would price move?) ---")
    print(f"{'If you bought':>15} {'Price would move'}")
    print("-" * 40)
    
    for size in [1000, 5000, 10000, 20000]:
        impact = calculate_market_impact(yes_orders, size)
        if impact:
            print(f"{size:>15,} {impact:>14.2f}%")
        else:
            print(f"{size:>15,} {'Not enough liquidity':>20}")
    
    print("\n💡 High impact = thin book, you're a whale")
    print("💡 Low impact = deep book, you can trade freely")
    
    # Health score
    print(f"\n--- LIQUIDITY HEALTH SCORE ---")
    health_score, component_scores = calculate_liquidity_health_score(yes_orders, no_orders, mid_price)
    
    print(f"\nOverall Score: {health_score:.1f} / 100")
    print(f"\nComponent Breakdown:")
    for component, score in component_scores.items():
        bar_length = int(score / 2)
        bar = "█" * bar_length
        print(f"  {component:20s} {score:>5.1f} {bar}")
    
    print(f"\n💡 Health score combines all factors into one number")
    print(f"💡 >80 = excellent, tradeable with size")
    print(f"💡 60-80 = good, normal trading")
    print(f"💡 40-60 = okay, be careful with size")
    print(f"💡 <40 = poor, thin/distorted market")
    
    # Summary
    print(f"\n{'='*80}")
    print("TRADING IMPLICATIONS")
    print(f"{'='*80}")
    
    if health_score >= 80:
        print("✅ EXCELLENT market quality - safe to trade with size")
    elif health_score >= 60:
        print("✅ GOOD market quality - normal trading conditions")
    elif health_score >= 40:
        print("⚠️  FAIR market quality - reduce position sizes")
    else:
        print("❌ POOR market quality - very thin, watch for manipulation")
    
    # Specific warnings
    if component_scores['concentration'] < 50:
        print("⚠️  WARNING: Order book dominated by large walls")
    
    if component_scores['balance'] < 50:
        print("⚠️  WARNING: Very one-sided order book")
    
    if component_scores['execution'] < 50:
        print("⚠️  WARNING: High slippage even for small orders")
    
    print()


def compare_markets():
    """Compare health scores across all monitored markets."""
    if not SUMMARY_CSV.exists():
        print("No summary data yet")
        return
    
    df = pd.read_csv(SUMMARY_CSV)
    
    # Get latest snapshot per market
    latest = df.sort_values('timestamp').groupby('market_ticker').tail(1)
    
    print("\n" + "="*80)
    print("MARKET COMPARISON")
    print("="*80)
    
    for _, row in latest.iterrows():
        ticker = row['market_ticker']
        
        # Load full order book to calculate health
        order_book = load_latest_order_book(ticker)
        if not order_book:
            continue
        
        book = order_book['data']['orderbook']
        yes_orders = book.get('yes', [])
        no_orders = book.get('no', [])
        
        if yes_orders and no_orders:
            mid_price = yes_orders[-1][0] / 100
            health, _ = calculate_liquidity_health_score(yes_orders, no_orders, mid_price)
            
            print(f"\n{ticker}")
            print(f"  Price: {mid_price:.3f}")
            print(f"  Total Depth: {row['total_depth_all']:,.0f}")
            print(f"  Health Score: {health:.1f} / 100")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_order_book_metrics.py MARKET_TICKER")
        print("\nAvailable markets:")
        if ORDER_BOOKS_DIR.exists():
            markets = set([f.name.split('_')[0] for f in ORDER_BOOKS_DIR.glob("*.json")])
            for market in sorted(markets):
                print(f"  - {market}")
        else:
            print("  No data yet. Run monitor_kalshi_markets.py first.")
        return
    
    market_ticker = sys.argv[1]
    analyze_market(market_ticker)
    
    # Also show comparison
    compare_markets()


if __name__ == "__main__":
    main()


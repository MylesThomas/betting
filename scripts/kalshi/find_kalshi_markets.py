"""
Test script to explore Kalshi API and find markets to monitor.

This script will:
1. Connect to Kalshi API
2. List available markets
3. Show order book structure for a sample market
4. Help us pick 2 markets to monitor

Usage:
    python scripts/test_kalshi_markets.py
"""

import requests
import json
from datetime import datetime
import os

# Kalshi API endpoints
KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_API_DEMO = "https://demo-api.kalshi.co/trade-api/v2"

# Use demo for testing (no auth needed), switch to live later
API_BASE = KALSHI_API_DEMO


def get_markets(limit=20, status="open"):
    """
    Fetch available markets from Kalshi.
    
    Args:
        limit: Number of markets to return
        status: Market status (open, closed, settled)
    """
    url = f"{API_BASE}/markets"
    params = {
        "limit": limit,
        "status": status
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("markets", [])
    except Exception as e:
        print(f"Error fetching markets: {e}")
        return []


def get_order_book(market_ticker):
    """
    Fetch full order book for a specific market.
    
    Args:
        market_ticker: The ticker symbol for the market
    """
    url = f"{API_BASE}/markets/{market_ticker}/orderbook"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        print(f"Error fetching order book for {market_ticker}: {e}")
        return None


def display_markets(markets):
    """Pretty print available markets."""
    print("\n" + "="*80)
    print("AVAILABLE KALSHI MARKETS")
    print("="*80)
    
    for i, market in enumerate(markets, 1):
        ticker = market.get("ticker", "N/A")
        title = market.get("title", "N/A")
        volume = market.get("volume", 0)
        yes_price = market.get("yes_bid", 0)
        
        print(f"\n{i}. {ticker}")
        print(f"   Title: {title}")
        print(f"   Volume: ${volume:,}")
        print(f"   Current Price: {yes_price}¢")


def display_order_book(order_book, market_ticker):
    """Pretty print order book structure."""
    print("\n" + "="*80)
    print(f"ORDER BOOK: {market_ticker}")
    print("="*80)
    
    if not order_book:
        print("No order book data available")
        return
    
    # Display structure
    print("\nOrder Book Keys:", list(order_book.keys()))
    print("\nFull JSON:")
    print(json.dumps(order_book, indent=2))
    
    # Try to parse bids/asks if available
    yes_bids = order_book.get("yes", [])
    no_bids = order_book.get("no", [])
    
    if yes_bids:
        print("\n--- YES SIDE (Bids) ---")
        for level in yes_bids[:10]:  # Top 10 levels
            price = level.get("price", 0)
            size = level.get("size", 0)
            print(f"  Price: {price}¢  |  Size: {size}")
    
    if no_bids:
        print("\n--- NO SIDE (Asks) ---")
        for level in no_bids[:10]:  # Top 10 levels
            price = level.get("price", 0)
            size = level.get("size", 0)
            print(f"  Price: {price}¢  |  Size: {size}")


def main():
    """Main exploration function."""
    global API_BASE
    
    print("="*80)
    print("KALSHI API EXPLORER")
    print("="*80)
    print(f"Using API: {API_BASE}")
    
    # Step 1: Get available markets
    print("\n[1/2] Fetching available markets...")
    markets = get_markets(limit=20)
    
    if not markets:
        print("\n⚠️  No markets found. Trying live API...")
        API_BASE = KALSHI_API_BASE
        markets = get_markets(limit=20)
    
    if not markets:
        print("\n❌ Could not fetch markets. Check API connection.")
        return
    
    display_markets(markets)
    
    # Step 2: Get order book for first market as example
    if markets:
        sample_ticker = markets[0].get("ticker")
        print(f"\n[2/2] Fetching order book for sample market: {sample_ticker}")
        order_book = get_order_book(sample_ticker)
        display_order_book(order_book, sample_ticker)
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR MONITORING")
    print("="*80)
    
    # Sort by volume to find most active
    sorted_markets = sorted(markets, key=lambda x: x.get("volume", 0), reverse=True)
    
    print("\nTop 5 by Volume (most active/liquid):")
    for i, market in enumerate(sorted_markets[:5], 1):
        ticker = market.get("ticker")
        title = market.get("title", "N/A")
        volume = market.get("volume", 0)
        print(f"{i}. {ticker} - ${volume:,}")
        print(f"   {title}")
    
    print("\n✅ Next step: Pick 2 markets from above and we'll start monitoring!")


if __name__ == "__main__":
    main()


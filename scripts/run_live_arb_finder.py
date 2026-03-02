"""
Live local arb finder: runs every N seconds during configured "awake" hours (ET).

Self-contained: all NBA arb logic lives in this file. No imports from other project modules.
Flow: load today's NBA events → for each game fetch odds → parse props → find arbs → print (no files saved).
Then sleep INTERVAL seconds and repeat. NFL still runs find_nfl_arb_opportunities.py as subprocess.

Usage:
    ODDS_API_KEY in .env. From repo root:
    python scripts/run_live_arb_finder.py --sport nba --interval 60 --profit-threshold 5.0
"""

import argparse
import os
import subprocess
import sys
import time
import ssl
import urllib3
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# Disable SSL warnings for API (match finder behavior)
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ET = ZoneInfo("America/New_York")
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

# --- The Odds API ---
API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"
REGIONS = "us"
ODDS_FORMAT = "american"
DATE_FORMAT = "iso"

# --- NBA defaults (no double-double / triple-double) ---
NBA_DEFAULT_MARKETS = (
    "player_points,player_rebounds,player_assists,player_threes,"
    "player_blocks,player_steals,player_points_rebounds_assists"
)
EXCLUDED_BOOKMAKERS = {"bovada", "williamhill_us"}
BASE_WAGER = 100

MARKET_DISPLAY = {
    "player_threes": "Threes",
    "player_points": "Points",
    "player_rebounds": "Rebounds",
    "player_assists": "Assists",
    "player_blocks": "Blocks",
    "player_steals": "Steals",
    "player_points_rebounds_assists": "Pts+Reb+Ast",
}


def now_et():
    return datetime.now(ET)


def inside_window(now, start_hour: int, end_hour: int) -> bool:
    if start_hour <= end_hour:
        return start_hour <= now.hour <= end_hour
    return now.hour >= start_hour or now.hour <= end_hour


def american_to_probability(odds):
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def calculate_arb_profit(over_odds, under_odds):
    over_prob = american_to_probability(over_odds)
    under_prob = american_to_probability(under_odds)
    total_prob = over_prob + under_prob
    is_arb = total_prob < 1.0
    expected_profit_pct = ((1 / total_prob) - 1) * 100
    return {
        "is_arb": is_arb,
        "expected_profit_pct": expected_profit_pct,
        "over_prob": over_prob,
        "under_prob": under_prob,
        "total_prob": total_prob,
    }


def calculate_bet_amounts(over_odds, under_odds, total_stake=100):
    over_prob = american_to_probability(over_odds)
    under_prob = american_to_probability(under_odds)
    over_stake = (over_prob / (over_prob + under_prob)) * total_stake
    under_stake = (under_prob / (over_prob + under_prob)) * total_stake
    over_return = over_stake * (1 + over_odds / 100) if over_odds > 0 else over_stake * (1 + 100 / abs(over_odds))
    under_return = under_stake * (1 + under_odds / 100) if under_odds > 0 else under_stake * (1 + 100 / abs(under_odds))
    profit = min(over_return, under_return) - total_stake
    return {
        "over_stake": round(over_stake, 2),
        "under_stake": round(under_stake, 2),
        "guaranteed_profit": round(profit, 2),
    }


def get_todays_nba_events(api_key):
    url = f"{API_BASE}/sports/{SPORT}/events"
    r = requests.get(url, params={"apiKey": api_key}, verify=False)
    r.raise_for_status()
    events = r.json()
    today = now_et().date()
    out = []
    for ev in events:
        t = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")).astimezone(ET)
        if t.date() == today:
            out.append(ev)
    return out, {"remaining": r.headers.get("x-requests-remaining", "unknown")}


def get_event_odds(api_key, event_id, markets):
    url = f"{API_BASE}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": REGIONS,
        "markets": markets,
        "oddsFormat": ODDS_FORMAT,
        "dateFormat": DATE_FORMAT,
    }
    r = requests.get(url, params=params, verify=False)
    r.raise_for_status()
    return r.json(), {"remaining": r.headers.get("x-requests-remaining", "unknown")}


def parse_event_props_to_df(event_data, api_fetch_time):
    if "data" in event_data:
        event_data = event_data["data"]
    game_info = f"{event_data['away_team']} @ {event_data['home_team']}"
    game_time = event_data.get("commence_time")
    event_id = event_data.get("id")
    fetch_et = api_fetch_time.astimezone(ET).strftime("%Y-%m-%d %H:%M:%S ET")
    rows = []
    for book in event_data.get("bookmakers", []):
        bkey = book["key"]
        if bkey in EXCLUDED_BOOKMAKERS:
            continue
        for market in book.get("markets", []):
            mkey = market["key"]
            mlast = market.get("last_update")
            minutes_stale = None
            if mlast:
                try:
                    last_dt = datetime.fromisoformat(mlast.replace("Z", "+00:00"))
                    if last_dt.tzinfo is None:
                        last_dt = last_dt.replace(tzinfo=timezone.utc)
                    minutes_stale = round((api_fetch_time - last_dt).total_seconds() / 60, 2)
                except Exception:
                    pass
            by_key = {}
            for outcome in market.get("outcomes", []):
                player = outcome.get("description", "Unknown")
                line = outcome.get("point")
                odds = outcome.get("price")
                name = outcome.get("name")
                key = (player, line)
                if key not in by_key:
                    by_key[key] = {
                        "event_id": event_id,
                        "player": player,
                        "market": mkey,
                        "line": line,
                        "bookmaker": bkey,
                        "game": game_info,
                        "game_time": game_time,
                        "market_last_update": mlast,
                        "minutes_stale": minutes_stale,
                        "fetch_time_et": fetch_et,
                    }
                if name == "Over":
                    by_key[key]["over_odds"] = odds
                elif name == "Under":
                    by_key[key]["under_odds"] = odds
            rows.extend(by_key.values())
    return pd.DataFrame(rows)


def find_best_odds_per_player(props_df):
    if props_df.empty:
        return pd.DataFrame()
    best = []
    for (market, player, line), group in props_df.groupby(["market", "player", "line"]):
        over_rows = group[group["over_odds"].notna()]
        under_rows = group[group["under_odds"].notna()]
        if over_rows.empty or under_rows.empty:
            continue
        best_over = over_rows.loc[over_rows["over_odds"].idxmax()]
        best_under = under_rows.loc[under_rows["under_odds"].idxmax()]
        arb = calculate_arb_profit(best_over["over_odds"], best_under["under_odds"])
        bet = calculate_bet_amounts(best_over["over_odds"], best_under["under_odds"], BASE_WAGER) if arb["is_arb"] else None
        best.append({
            "player": player,
            "market": market,
            "line": line,
            "best_over_odds": best_over["over_odds"],
            "best_over_book": best_over["bookmaker"],
            "best_over_implied": arb["over_prob"],
            "best_under_odds": best_under["under_odds"],
            "best_under_book": best_under["bookmaker"],
            "best_under_implied": arb["under_prob"],
            "total_prob": arb["total_prob"],
            "expected_profit_pct": arb["expected_profit_pct"],
            "is_arb": arb["is_arb"],
            "over_stake": bet["over_stake"] if bet else None,
            "under_stake": bet["under_stake"] if bet else None,
            "guaranteed_profit": bet["guaranteed_profit"] if bet else None,
            "game": group["game"].iloc[0],
            "game_time": group["game_time"].iloc[0],
            "over_minutes_stale": best_over.get("minutes_stale"),
            "under_minutes_stale": best_under.get("minutes_stale"),
            "over_last_update": best_over.get("market_last_update"),
            "under_last_update": best_under.get("market_last_update"),
        })
    return pd.DataFrame(best)


def add_staleness_flags(df, max_staleness_minutes):
    df = df.copy()
    df["max_staleness"] = df[["over_minutes_stale", "under_minutes_stale"]].fillna(0).max(axis=1)
    df["is_stale"] = df["max_staleness"] > max_staleness_minutes
    return df


def filter_fresh(df):
    return df[df["is_stale"] == False].copy()


def _seconds_ago(iso_utc_str):
    if pd.isna(iso_utc_str) or not iso_utc_str:
        return "—"
    try:
        dt = datetime.fromisoformat(str(iso_utc_str).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        secs = int((datetime.now(timezone.utc) - dt).total_seconds())
        return f"{secs} sec ago"
    except Exception:
        return "—"


def _tip_off_et(game_time):
    if pd.isna(game_time) or not game_time:
        return "—"
    try:
        dt = datetime.fromisoformat(str(game_time).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        et = dt.astimezone(ET)
        h = et.hour % 12 or 12
        suf = "am" if et.hour < 12 else "pm"
        return f"{h}{suf} ET"
    except Exception:
        return str(game_time)


def _short_game(s):
    if pd.isna(s) or not s:
        return "—"
    parts = str(s).split(" @ ")
    if len(parts) != 2:
        return s
    away = parts[0].strip().split()[-1]
    home = parts[1].strip().split()[-1]
    return f"{away} @ {home}"


def _markets_list(markets_str):
    """Return list of (key, display_name) in order for the given markets string."""
    keys = [m.strip() for m in markets_str.split(",") if m.strip()]
    return [(k, MARKET_DISPLAY.get(k, k)) for k in keys]


def _log_verbose_player_lines(best_df, mdisplay):
    """Log each (player, line) with combined implied %% and arb vs book-edge."""
    for _, row in best_df.iterrows():
        total_pct = row["total_prob"] * 100
        if row["total_prob"] <= 1.0:
            label = "<= 100% = ARB"
        else:
            label = "> 100% = book edge"
        print(f"      [verbose] {row['player']} {row['line']} ({mdisplay}): combined implied {total_pct:.2f}% ({label})")


def display_arbs(df, min_profit_pct=0.0, inline=False):
    """Print arb rows. If inline=True, no big header (for per-market output)."""
    if df.empty:
        return 0
    arbs = df[(df["expected_profit_pct"] > min_profit_pct) & (df["is_arb"])].copy()
    arbs = arbs.sort_values("expected_profit_pct", ascending=False)
    if arbs.empty:
        return 0
    if not inline:
        print("\n" + "=" * 80)
        print(f"ARBITRAGE OPPORTUNITIES: {len(arbs)}")
        print("=" * 80 + "\n")
    for _, row in arbs.iterrows():
        mkt = MARKET_DISPLAY.get(row["market"], row["market"])
        print(f"🏀 {row['player']} → {row['line']} {mkt}")
        print(f"{_short_game(row['game'])}, tip off at {_tip_off_et(row['game_time'])}")
        bet = calculate_bet_amounts(row["best_over_odds"], row["best_under_odds"], BASE_WAGER)
        for side, line_val, odds, book, implied, last_up, stake in [
            ("Over", row["line"], row["best_over_odds"], row["best_over_book"], row["best_over_implied"], row.get("over_last_update"), bet["over_stake"]),
            ("Under", row["line"], row["best_under_odds"], row["best_under_book"], row["best_under_implied"], row.get("under_last_update"), bet["under_stake"]),
        ]:
            print(f"- {side} {line_val}: {odds:+} ({book}) → {implied:.2%} | {_seconds_ago(last_up)} | Bet ${stake:.2f}")
        print("-" * 80)
    return len(arbs)


def run_nba_live(api_key: str, markets, profit_threshold: float, max_staleness_minutes: float = 1.0, verbose: bool = False):
    markets_str = markets or NBA_DEFAULT_MARKETS
    markets_order = _markets_list(markets_str)
    events, _ = get_todays_nba_events(api_key)
    if not events:
        print("No NBA games today.")
        return
    fetch_time = datetime.now(timezone.utc)
    n = len(events)
    for i, event in enumerate(events, 1):
        label = f"{event['away_team']} @ {event['home_team']}"
        print(f"--- Game {i}/{n}: {label} ---")
        try:
            odds, _ = get_event_odds(api_key, event["id"], markets_str)
            props = parse_event_props_to_df(odds, fetch_time)
        except Exception as e:
            print(f"   Fetch error: {e}\n")
            continue
        if props.empty:
            print("   No props\n")
            continue
        game_arb_count = 0
        for mkey, mdisplay in markets_order:
            props_m = props[props["market"] == mkey]
            if props_m.empty:
                print(f"   No arbs for {mdisplay}")
                continue
            best = find_best_odds_per_player(props_m)
            if best.empty:
                print(f"   No arbs for {mdisplay}")
                continue
            if verbose:
                print(f"   [{mdisplay}] player-line combos (combined implied):")
                _log_verbose_player_lines(best, mdisplay)
            best = add_staleness_flags(best, max_staleness_minutes)
            fresh = filter_fresh(best)
            n_arbs = display_arbs(fresh, min_profit_pct=profit_threshold, inline=True)
            if n_arbs == 0:
                print(f"   No arbs for {mdisplay}")
            else:
                game_arb_count += n_arbs
        if game_arb_count == 0:
            print("   No arbs this cycle.")
        print()


def run_nfl_finder(extra_args: list):
    cmd = [sys.executable, str(SCRIPT_DIR / "find_nfl_arb_opportunities.py")] + extra_args
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def main():
    parser = argparse.ArgumentParser(description="Live arb finder: NBA (in-process) and/or NFL (subprocess).")
    parser.add_argument("--interval", type=int, default=60, metavar="SEC", help="Seconds between runs (default 60)")
    parser.add_argument("--start-hour", type=int, default=8, metavar="H", help="Window start ET 0-23 (default 8)")
    parser.add_argument("--end-hour", type=int, default=23, metavar="H", help="Window end ET 0-23 (default 23)")
    parser.add_argument("--sport", choices=("nba", "nfl", "both"), default="both", help="Which sport(s)")
    parser.add_argument("--nba-markets", type=str, default=None, help="NBA markets (default: points, rebounds, assists, threes, blocks, steals, pts+reb+ast)")
    parser.add_argument("--profit-threshold", type=float, default=0.0, metavar="PCT", help="Min profit %% to show arbs (default 0)")
    parser.add_argument("--verbose", action="store_true", help="Log each player-line combo with combined implied %% (<= 100%% = arb, > 100%% = book edge)")
    parser.add_argument("--nfl-all-markets", action="store_true", help="NFL: all markets")
    args = parser.parse_args()
    os.chdir(REPO_ROOT)

    if not (0 <= args.start_hour <= 23 and 0 <= args.end_hour <= 23):
        print("--start-hour and --end-hour must be 0-23", file=sys.stderr)
        sys.exit(1)
    if args.interval < 1:
        print("--interval must be >= 1", file=sys.stderr)
        sys.exit(1)

    load_dotenv()
    api_key = (os.getenv("ODDS_API_KEY") or "").strip()
    if not api_key or api_key == "your_api_key_here":
        api_key = None

    nfl_extra = ["--week"]
    if args.nfl_all_markets:
        nfl_extra.append("--all-markets")

    nba_markets_str = args.nba_markets or NBA_DEFAULT_MARKETS
    nba_markets_display = [MARKET_DISPLAY.get(m.strip(), m.strip()) for m in nba_markets_str.split(",") if m.strip()]

    print("=" * 60)
    print("LIVE ARB FINDER (terminal only)")
    print("=" * 60)
    print("Args:")
    print(f"   --interval {args.interval}  --start-hour {args.start_hour}  --end-hour {args.end_hour}")
    print(f"   --sport {args.sport}  --profit-threshold {args.profit_threshold}  --verbose {args.verbose}")
    if args.nba_markets:
        print(f"   --nba-markets {args.nba_markets}")
    else:
        print(f"   --nba-markets (default)")
    print("Excluded books:", ", ".join(sorted(EXCLUDED_BOOKMAKERS)))
    print("Markets participating:", ", ".join(nba_markets_display))
    print("Stop with Ctrl+C.")
    print("=" * 60)

    run_count = 0
    while True:
        now = now_et()
        if not inside_window(now, args.start_hour, args.end_hour):
            print(f"[{now.strftime('%Y-%m-%d %H:%M ET')}] Outside window — sleeping {args.interval}s")
            time.sleep(args.interval)
            continue
        run_count += 1
        print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S ET')}] Run #{run_count}")
        print("-" * 60)
        if args.sport in ("nba", "both"):
            if api_key:
                run_nba_live(api_key, args.nba_markets, args.profit_threshold, max_staleness_minutes=1.0, verbose=args.verbose)
            else:
                print("No ODDS_API_KEY. Add to .env.")
        if args.sport in ("nfl", "both"):
            run_nfl_finder(nfl_extra)
        print("-" * 60)
        print(f"Next run in {args.interval}s...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

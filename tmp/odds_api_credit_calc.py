"""
Odds API credit calculator — reads config directly from run_nfl_historical_backfill.py.

Observed rates (2025-06-19 live run):
  Global snapshot endpoint (event_id map): 10 credits × num_regions / call
  Per-event endpoint (market fetch):       4.2 credits × num_regions / market / call
"""

import re
from pathlib import Path

BACKFILL_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_nfl_historical_backfill.py"

# ── Parse backfill script ──────────────────────────────────────────────────────
src = BACKFILL_SCRIPT.read_text()

# Active markets (uncommented lines inside ALL_MARKETS block)
start  = src.index("ALL_MARKETS = [")
end    = src.index("\n]", start)
block  = src[start:end]
markets = []
for line in block.splitlines():
    stripped = line.strip()
    if stripped.startswith("#") or not stripped:
        continue
    for m in re.findall(r'"([\w_]+)"', stripped):
        if m not in ("CUT", "ALL_MARKETS"):
            markets.append(m)

# Regions (count comma-separated values in REGIONS = "...")
regions_str = re.search(r'REGIONS\s*=\s*"([^"]+)"', src).group(1)
num_regions = len(regions_str.split(","))

# Bookmakers
bookmakers_str = re.search(r'BOOKMAKERS\s*=\s*\[([^\]]+)\]', src).group(1)
bookmakers = [b.strip().strip('"') for b in bookmakers_str.split(",")]

# Seasons list
seasons_str = re.search(r'SEASONS\s*=\s*\[([^\]]+)\]', src).group(1)
seasons = [int(s.strip()) for s in seasons_str.split(",")]

# Credit stop threshold
threshold = int(re.search(r'CREDIT_STOP_THRESHOLD\s*=\s*(\d+)', src).group(1))

# ── Constants ──────────────────────────────────────────────────────────────────
CREDITS_PER_REGION_SNAPSHOT   = 10.0  # event_id map: global snapshot endpoint
CREDITS_PER_REGION_PER_MARKET = 4.2   # market fetch: per-event endpoint
GAMES_PER_SEASON    = 285
GAMEDAYS_PER_SEASON = 64
PLAN_CREDITS        = 100_000
CREDITS_REMAINING   = 86_060  # update manually after each run

# ── Calculations ───────────────────────────────────────────────────────────────
n_markets   = len(markets)
n_seasons   = len(seasons)

event_map_per_season   = GAMEDAYS_PER_SEASON * CREDITS_PER_REGION_SNAPSHOT * num_regions
market_fetch_per_game  = n_markets * CREDITS_PER_REGION_PER_MARKET * num_regions
market_fetch_per_season = market_fetch_per_game * GAMES_PER_SEASON
total_per_season       = event_map_per_season + market_fetch_per_season
total_all              = total_per_season * n_seasons

seasons_from_remaining = (CREDITS_REMAINING - threshold) / total_per_season
seasons_from_reset     = (PLAN_CREDITS - threshold) / total_per_season

# ── Output ─────────────────────────────────────────────────────────────────────
W = 52
print(f"{'─'*W}")
print(f"  SOURCE: {BACKFILL_SCRIPT.name}")
print(f"{'─'*W}")
print(f"  Markets        : {n_markets}  ({', '.join(markets[:3])}{'...' if n_markets > 3 else ''})")
print(f"  Regions        : {num_regions}  ({regions_str})")
print(f"  Bookmakers     : {bookmakers}")
print(f"  Seasons        : {n_seasons}  {seasons}")
print(f"  Stop threshold : {threshold:,}")
print()
print(f"{'─'*W}")
print(f"  CREDITS")
print(f"{'─'*W}")
print(f"  Event ID map   : {event_map_per_season:>10,.0f}  /season")
print(f"  Market fetch   : {market_fetch_per_season:>10,.0f}  /season  ({n_markets} mkts × {GAMES_PER_SEASON} games × {CREDITS_PER_REGION_PER_MARKET*num_regions:.1f} cr)")
print(f"  Total/season   : {total_per_season:>10,.0f}")
print(f"  Total all ({n_seasons} seasons): {total_all:>8,.0f}")
print()
print(f"{'─'*W}")
print(f"  BUDGET")
print(f"{'─'*W}")
print(f"  Plan/month     : {PLAN_CREDITS:>10,}")
print(f"  Remaining now  : {CREDITS_REMAINING:>10,}  (update CREDITS_REMAINING if stale)")
print(f"  Seasons now    : {seasons_from_remaining:>10.1f}  (before reset)")
print(f"  Seasons/reset  : {seasons_from_reset:>10.1f}  (from fresh 100k)")
fits = total_all <= CREDITS_REMAINING
print(f"  All {n_seasons} seasons fit? : {'YES ✅' if fits else f'NO ❌  ({total_all - CREDITS_REMAINING:,.0f} short — wait for reset or reduce seasons)'}")
print(f"{'─'*W}")

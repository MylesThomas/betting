"""
Quick sanity check — reads every strategies/*.yaml and prints each strategy key.
Usage: python config/strategies/read_strategies.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config_loader import load_strategies

all_strats = load_strategies()

for sport, strategies in sorted(all_strats.items()):
    print(f"\n[{sport.upper()}]")
    for key, cfg in strategies.items():
        status    = cfg.get("status", "unknown")
        direction = cfg.get("direction", "?")
        oos       = cfg.get("oos", {})
        roi       = oos.get("roi")
        units     = oos.get("units")
        n_bets    = oos.get("n_bets")
        roi_str   = f"{roi*100:.1f}%" if roi is not None else "n/a"
        units_str = str(units) if units is not None else "n/a"
        n_str     = str(n_bets) if n_bets is not None else "n/a"
        print(f"  {key:<20} dir={direction:<6} status={status:<20} oos_roi={roi_str:<8} units={units_str:<8} n={n_str}")

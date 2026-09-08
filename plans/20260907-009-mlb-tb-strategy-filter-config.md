# MLB TB Strategy Filter Config

**Date:** 2026-09-07

## What changed

Added explicit strategy filter params to the snapshot analysis pipeline so CLV and tightening computations are scoped to exactly the rows the model bets on (`batter_total_bases`, line=1.5, non-null under odds).

## Files changed

| File | Change |
|------|--------|
| `config.yaml` | Added `markets: [batter_total_bases]` to `strategy` block |
| `compute_clv.py` | Added `strategy_only=True, markets=None, lines=None` params; filters df before CLV computation |
| `compute_tightening.py` | Same params; filters first-seen rows before per-game aggregation |
| `settle_total_bases.py` | Loads config once at import (`_STRATEGY_MARKETS`, `_STRATEGY_LINES`); passes to both call sites |

## Design decisions

- **Snapshot fetch unchanged** — `snapshot_props.py` still fetches both `batter_total_bases` and `batter_total_bases_alternate`; filtering happens downstream
- **`strategy_only=True` default** — both functions are strategy-scoped by default; pass `strategy_only=False` for ad-hoc analysis across all lines/markets
- **Caller owns config load** — `settle_total_bases.py` loads config and passes explicit lists; `compute_clv` and `compute_tightening` don't read config.yaml themselves

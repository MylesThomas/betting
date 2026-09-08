# MLB Total Bases — Dedup & Book-Count Analysis

**Summary**: Deduping to best-book per (player, game, line) nearly doubles OOS ROI; the 2–4 books bucket is a money loser in IS.

**Last updated**: 2026-07-24

**Source**: `knowledge-base/raw/20260724-mlb-total-bases-dedup-analysis.html`

---

## Context

The total bases spine is at book-grain — one row per (player, game, bookmaker, line). With ~7.8 books per slot on average, the raw backtest counts 7–8 correlated bets for every qualifying opportunity. This analysis answers:
1. What happens to performance if we dedupe to one bet per (player, game, line)?
2. Does the number of books offering a line predict edge quality?

## Setup

- IS: 2024–2025 | OOS: 2026
- Strategy tested: UNDER 1.5, dogs (+odds), edge ≥ 5pp (canonical live strategy)
- Dedupe method: keep row with highest `under_price` (best available book)

## Key findings

### 1. Deduping improves ROI at every edge threshold

| | OOS raw | OOS deduped |
|---|---|---|
| n bets (edge≥5, dogs, 1.5) | 443 | 120 |
| Units | +33.2 | +17.8 |
| ROI | +7.5% | **+14.8%** |

Deduped ROI is consistently higher across all thresholds in both IS and OOS. The raw strategy is not finding more independent edges — it's re-betting the same outcome through multiple books.

### 2. The 2–4 books bucket is a red flag (IS)

At canonical params (edge≥5pp, dogs, 1.5), split by number of books offering the line:

| Bucket | IS n | IS win% | IS ROI | OOS n | OOS win% | OOS ROI |
|---|---|---|---|---|---|---|
| 1 book | 13 | 61.5% | +39.5% | 10 | 80.0% | +92.8% |
| 2–4 books | 354 | **37.3%** | **−17.9%** | 54 | 48.1% | +7.9% |
| 5+ books | 1,861 | 51.4% | +11.6% | 379 | 48.0% | +5.2% |

- **5+ books** drives all IS returns and holds in OOS
- **2–4 books** is a money loser in IS (too small to confirm in OOS at n=54)
- 1 book samples are too tiny to trust (n=13 / n=10)

OOS doesn't confirm the 2–4 bucket is bad (+7.9%), but IS signal warrants monitoring. Don't filter yet — wait for another half season of OOS data.

## Live Lambda implication

The Lambda currently places **one bet per qualifying (player, game, line, book)**. On a typical 8-book day, one bad player outcome hits 8×. True position per unique opportunity is ~8× what the deduped backtest implies. Consider switching to best-book-only to align live sizing with the deduped results.

## Rules

1. When reporting ROI for total bases, note whether it's raw (per-book) or deduped (per-opportunity) — they differ by ~2×
2. Don't filter the 2–4 books bucket yet (OOS n too small); revisit after 2026 season ends
3. Deduped best-book is the more honest representation of edge per opportunity

## Related

- [[data-quirks]]
- [[edge-calibration]]
- [[roi-and-pnl]]

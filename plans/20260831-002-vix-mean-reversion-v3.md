# VIX Mean Reversion — v3 Plan
Date: 2026-08-31
Session log: `knowledge-base/raw/20260830-vix-mean-reversion.html`
Scripts: `src/qr/vix_mean_reversion/`

---

## What v2 established (locked)

| param | value |
|---|---|
| fair_value_method | 126d SMA |
| entry_threshold | 10% below SMA |
| exit_threshold | 2% below SMA (early exit) |
| direction | long only |
| result | 229 trades, 96.5% win rate, +35.06u, CAGR 139.7%, calmar 99.14, avg hold 27.3d |

v2 is a strong signal. The problem: **VIX is not investable.** Everything in v3 is about finding an investable vehicle that captures the edge.

---

## What P1a showed (VIX calls — OptionsDX 2010–2023)

| hold bucket | n | win rate | avg P&L | expired worthless |
|---|---|---|---|---|
| 0–14d | 41 | **87.8%** | **+15.0%** | 0% |
| 15–30d | 20 | 45.0% | +8.0% | 0% |
| 31–60d | 14 | 0% | −53.5% | 29% |
| 60d+ | 15 | 0% | −100% | 100% |

**Core problem:** ATM VIX calls at signal entries cost a median **24.2% of VIX level** as premium. Median actual VIX move was +14.5%. Only 19% of entries see a VIX move large enough to cover the premium. Options pricing erases the edge unless VIX spikes fast.

**Exception:** The 0–14d bucket (fast spikes) has a strong edge on calls. If we can target those entries only, options become viable.

---

## v3 Track Results (all complete)

| track | vehicle | n | win rate | total u | verdict |
|---|---|---|---|---|---|
| A — SPY | SPY long | 208 | 36.5% | +0.68u | ❌ Wrong direction — VIX UP = SPY DOWN |
| B — VIX calls DTE≥90 ATM | VIX ATM calls | 72 | 45.8% | −7.28u | ❌ 24% premium hurdle vs 14.5% median VIX move |
| C — Filtered F1∧F2 | VIX calls + filter | 9 | 77.8% | +0.77u | ⚠️ Edge real but n=9 (0.6/yr) — not implementable |

**Core problem:** Every investable instrument has a structural cost that erases the edge on most trades. The 0–14d fast-spike bucket is profitable for options (87.8% win rate) but fast spikes cannot be predicted from entry-day features.

---

## v3 Tracks (archived)

---

### Track A — SPY as vehicle (highest priority)

**Hypothesis:** When VIX is ≥10% below its 126d SMA, the market is calm and SPY tends to continue drifting up until VIX reverts. Go long SPY on entry signal, exit on exit signal.

**Why SPY first:**
- Free data (yfinance), full history back to 1993
- No expiry, no premium, no roll costs
- Directly tradeable via any broker
- Most comparable to the VIX close baseline

**What changes:** P&L is now SPY % return during the VIX-signal hold period, not VIX % return.

**Key questions:**
- How correlated is SPY return with VIX return during our hold windows?
- Does the edge survive on SPY (which is a different series than VIX)?
- How do 2008, 2020, and 2022 look — VIX spikes coincide with SPY crashes, which is the opposite direction. Need to verify the logic: low VIX → long SPY means we're long during calm markets, which should be fine. But if VIX mean-reverts by spiking (fear event), SPY drops — that's a loss.

**Script:** `20260831_spy_backtest.py`
**Data:** yfinance `SPY`
**HTML section:** `TRACK A — SPY Vehicle Backtest`

**Passing bar:**
- n_trades ≥ 50 (within SPY history window)
- win_rate ≥ 0.50
- total_units > 0 (net profitable)
- calmar > 1

---

### Track B — VIX calls, DTE ≥ 90, strike sweep

**Hypothesis:** The main problem with P1a was expiry wipeout on long-hold trades (60d+ expired worthless). Buying 90 DTE calls at entry gives enough runway to hold through the full trade without expiry. Separately, OTM calls (delta ~0.25–0.30) have lower premium hurdles than ATM.

**What changes from P1a:**
- Min DTE at entry: 90 (was 40)
- Strike sweep: ATM (delta ~0.50) vs. slightly OTM (delta ~0.30)
- Same entry/exit signals as v2

**Key question:** Does removing the expiry wipeout (15 full-loss trades in P1a) turn the overall P&L positive? The 24.2% premium hurdle on ATM doesn't change — but losses are partial rather than -100%.

**Script:** `20260831_p1a_v2_dte90.py`
**Data:** OptionsDX 2010–2023 (already extracted)
**HTML section:** `TRACK B — VIX Calls DTE 90 + Strike Sweep`

**Output table:** one row per (DTE_min, strike_type) combo — at minimum: (40, ATM), (90, ATM), (90, OTM delta~0.30)

---

### Track C — Filtered entries (momentum signal) + VIX calls

**Hypothesis:** The 0–14d bucket on calls is the only profitable regime (87.8% win rate, +15% avg). The P1b-ii short-hold feature analysis found that VIX rising in the 5 days before entry correlates with shorter holds (21d avg vs 28d avg). If we restrict options entries to "VIX rising at entry" moments, we get more fast-spike trades and avoid the slow grind entries that expire worthless.

**Filter definition:**
- Enter only if VIX is higher today than 5 days ago (mom_5d > 0)
- Use ATM calls, DTE ≥ 40 (sufficient for 21d avg hold)
- Same v2 exit signal

**Key question:** Does the momentum filter improve options win rate enough to make the strategy profitable overall? What fraction of v2 trades survive the filter (need ≥ 30 for statistical meaningfulness).

**From P1b-ii:** 28 of 229 v2 trades had VIX rising at entry — small sample. May need to loosen filter (mom_5d vs mom_3d) or use a different feature.

**Script:** `20260831_p1a_v3_filtered.py`
**Data:** OptionsDX 2010–2023
**HTML section:** `TRACK C — Filtered Entries Options Backtest`

---

## Remaining paths (v4)

| path | why | data cost | status |
|---|---|---|---|
| **SPX puts when VIX low** | Buy SPX puts when VIX ≤ SMA×0.90; profit when SPX drops (= VIX spikes). Same signal, mirror instrument. Low IV at entry = cheap puts. | OptionsDX SPX data — same free download | **Ready to test** |
| VIX futures (paid) | Direct expression, no premium hurdle, no contango for long | CBOE DataShop ~$50–200 one-time | Deferred — data purchase decision needed |
| UVIX (2× long VIX futures ETF) | Retail-tradeable, no options complexity | yfinance — free but only 2022–present | Too short for backtest; track live |

**Recommended next:** SPX puts test using OptionsDX SPX data (free, same download site as VIX).

---

## Decision criteria for v3 config (archived)

After running all three tracks:

| check | requirement |
|---|---|
| n_trades in-sample | ≥ 30 for options tracks; ≥ 50 for SPY |
| win_rate | ≥ 0.55 |
| total_units | > 0 |
| calmar | > 1 |
| profitable years | ≥ 2/3 of years in sample |

If multiple tracks pass: prefer the one with the best calmar. SPY is preferred over options if metrics are close — simpler execution, no expiry management.

If no track passes: the strategy does not have a viable investable vehicle at this stage. Document and park until VIX futures data is available (P2 path — would require CBOE historical futures data, not free).

---

## Open questions for v3

1. **SPY directionality:** Our signal fires when VIX is LOW — we go long SPY during calm markets. VIX mean-reverts by spiking (fear), which would hurt SPY. Need to verify: are the SPY returns during our hold windows driven by continued calm, or do we also capture pre-spike calm? Check 2008 and 2020 specifically.

2. **DTE selection for Track B:** At low VIX, 90 DTE calls are cheaper in absolute terms but still expensive as % of VIX. Check if DTE 60 is a better balance (lower premium vs. 90, less expiry risk vs. 40).

3. **Strike selection:** For VIX calls, OTM delta-0.25 calls have a lower dollar premium but need a bigger absolute VIX move. At VIX=15, a 20-strike call (33% OTM) might cost $1.50 vs $3.50 ATM — but needs VIX to spike past 21.50 to break even. Need the EDA to show actual strike/premium table at entry-level VIX to decide.

4. **Track C filter size:** If the momentum filter leaves only 28 trades (2010–2023), that's too thin. Consider widening to: VIX rising over last 3d, OR VIX below SMA for ≤ 7 days (freshness). Check trade count before running full backtest.

---

## HTML approach — all additive

All new sections append to `20260830-vix-mean-reversion.html`.

| section tag | content |
|---|---|
| `TRACK A` | SPY backtest: equity curve, year-by-year, summary vs v2 VIX close |
| `TRACK B` | VIX calls DTE 90 sweep table, equity curves by config |
| `TRACK C` | Filtered entries: trade count, win rate, comparison to unfiltered P1a |
| `V3 DECISION` | Final verdict table, chosen config, production path |

---

## Production path (once v3 config locked)

Same as v2 production path but updated for the chosen vehicle.

For SPY:
- Daily: fetch VIX close + SPY close
- Signal: if VIX ≤ 126d_SMA × 0.90 → buy SPY at next open
- Exit: if VIX ≥ 126d_SMA × 0.98 → sell SPY at next open
- Slippage: T+1 open execution (not close) — measure signal-day close vs T+1 open gap

For VIX calls:
- Same signal logic
- On entry: buy ATM or OTM call with DTE ≥ chosen min
- On exit: sell call at mid
- Need broker that supports VIX options (IBKR, TastyTrade)
- Extra param: what if no liquid contract available on signal day?

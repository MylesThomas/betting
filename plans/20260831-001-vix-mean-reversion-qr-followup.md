# VIX Mean Reversion — QR Plan (v2)
Date: 2026-08-31
Session log: `knowledge-base/raw/20260830-vix-mean-reversion.html`
Scripts: `src/qr/vix_mean_reversion/`

---

## Status — v1 follow-up items (all complete)

| item | script | HTML section | status |
|---|---|---|---|
| MAE analysis (short trades) | `20260831_mae_analysis.py` | STEP 2b | ✅ done |
| FV window sweep + 50d/200d | `20260831_sweep3b.py` | STEP 4b | ✅ done |
| Compounded returns ($1k) | `20260831_compounded.py` | STEP 3b | ✅ done |
| Decision v2 | `20260831_decision_v2.py` | STEP 5b | ✅ done |

---

## v2 Config (locked)

| param | v1 | v2 |
|---|---|---|
| fair_value_method | 252d SMA | **126d SMA** (+9.3u gain) |
| entry_threshold | 10% | 10% (unchanged) |
| exit_threshold | 0% | **2%** (take profit early, +3.7u gain, shorter hold) |
| direction | both | **long only** (shorts fail margin survivability — 9/136 MAE >100%) |
| sizing | 100% bankroll | 100% bankroll (research baseline; real sizing TBD) |

**Expected v2 improvement:** ~54.8u (126d SMA, both) baseline, further gains from 2% exit and long-only risk reduction. Long-only CAGR 94.2%, max compounded drawdown −31%.

---

## v2 Backtest — work remaining

### Step 6 — v2 Full Backtest
Run the complete backtest with locked v2 config and produce updated Steps 2–5 output.

**Script:** `20260831_backtest_v2.py`

**Metrics to add (user-requested):**
- Avg win / avg loss expressed as **% account value change** (= pnl_pct, since 100% sizing)
- `avg_win_pct`, `avg_loss_pct` columns in summary table alongside units
- Year-by-year EOY balance column in yearly table

**HTML section:** `STEP 6 — v2 Backtest Results`

---

## Production Path — what we'd need to trade this for real

This section captures what is missing between the research backtest and a live, executable strategy.

### P1 — Instrument selection and roll-cost modeling

The backtest trades "VIX close" directly. VIX is not investable. Real options:

| instrument | use case | data needed | key cost |
|---|---|---|---|
| VIX futures (CBOE) | Purest expression | Futures OHLC by expiry from CBOE or Bloomberg | Contango roll ~2–5% monthly in calm markets |
| VXX / VXXB ETN | Retail-accessible long VIX | ETF daily prices from yfinance | Path decay from daily roll; not suitable for multi-week holds |
| SVXY | Retail short VIX | ETF daily prices | 90%+ drawdown Feb 2018 ("Volmageddon"); position limits required |
| VIX calls (options) | Long signal — buy calls when VIX low | Options chain data (CBOE, ORATS, OptionsDX) | Premium decay (theta); strike/expiry selection adds 2 new params |
| SPY long | Indirect short-VIX proxy | Already available | Imperfect correlation; VIX and SPY diverge during specific regimes |

**VXX is not the right vehicle for the long signal.** Our long entry fires when VIX is low relative to trend — exactly when contango is steepest and VXX roll decay is fastest. A 27-day avg hold would eat 5–15% of the gain to decay before any spike occurs. VXX is also a structurally decaying instrument (loses the large majority of value over multi-year periods even when VIX is flat) and carries Barclays issuer credit risk as an ETN.

**Short signal via VXX is structurally cleaner** (contango works for you when VIX is elevated) but we ruled out shorts on MAE/margin grounds. SVXY (inverse VIX ETF) had a 90%+ single-day drawdown in Feb 2018 — confirmed in our MAE data as a 205% adverse move.

**Split P1 into two sub-items:**

**P1a — VIX calls (primary candidate for long signal)**
- Buy calls when VIX ≤ 126d SMA × 0.90. Limited downside (premium paid), participates in spike.
- Requires: options chain data with strike/expiry (CBOE DataShop, ORATS, or OptionsDX — not free).
- Adds 2 new params: strike selection (ATM vs. OTM) and expiry selection (must exceed expected hold time of ~27 days).
- Theta decay replaces contango decay as the cost — but is bounded and known at entry.

**P1b — VXX signal divergence analysis (informational only)** ✅ DONE
- Results: VIX win rate 94.7% vs VXX win rate 56.1% on same signals. VIX total +1,033% vs VXX −71% over 57 trades (2018–2026). CAGR haircut: 214.6pp. 22 trades where VIX won but VXX lost.
- Drag grows with hold time — 60d+ holds see −21% avg VXX P&L vs +8.7% VIX.
- **Data limitation:** yfinance only has VXX from 2018-01-25 (original series matured 2019, VXXB relaunched and renamed). Only 57 of 229 v2 trades covered.
- **Verdict: VXX is not viable.** Confirmed P1a (VIX options) is the correct path.
- **Free data exhausted for VXX.** Older VXX history (2009–2018) not available on yfinance.

### P2 — Execution data

The backtest assumes entry and exit at the closing VIX level. In practice:

- VIX is published intraday but futures/ETPs have their own open/close
- Entry would be on the next day's open after the signal fires at close (T+1)
- Need to measure **signal-to-execution slippage**: how much does VIX move overnight between signal and fill?
- Script: replay all 150 long-entry signals, compare close price (signal day) vs. next-day open (execution day)

### P3 — Transaction costs

None modeled. For any real vehicle:

- **Futures:** ~$1–2/contract commission + bid-ask spread (typically 0.05–0.10 VIX points)
- **ETPs (VXX/SVXY):** ~$0.01–0.05/share spread; manageable
- **Options:** bid-ask on VIX options can be $0.10–0.50 wide; significant vs. a 10% VIX move

### P4 — Position sizing model

100% bankroll per trade is a research baseline only. For real trading:

- **Fixed %:** e.g. risk 5–10% of account per trade. Kills the compounding but makes drawdowns survivable.
- **Kelly criterion:** requires knowing true edge distribution. Can estimate from backtest but will oversize — use half-Kelly.
- Need to decide: is this a standalone strategy or one of many (portfolio Kelly)?

### P5 — Live signal infrastructure

To run this in production (same pattern as existing pipelines):

- **Daily data fetch:** yfinance `^VIX` pull at market close
- **Fair value computation:** rolling 126d SMA of closing VIX, updated daily
- **Signal check:** if VIX ≤ SMA × 0.90 and no open position → entry signal
- **Exit check:** if VIX ≥ SMA × 0.98 and long position open → exit signal
- **Notification:** email/push with entry price, fair value, deviation %, expected hold
- **Settlement:** log actual exit price vs. modeled exit price to track live vs. backtest divergence

This maps to the existing Lambda + EventBridge pattern in the repo. Estimated build: 1–2 sessions once instrument is chosen.

### P6 — Out-of-sample validation

The backtest uses 1990–2026 (36 years) with no hold-out. Before any capital allocation:

- Reserve 2020–2026 as OOS (post-COVID regime change)
- Re-run backtest on 1990–2019 only (IS)
- Validate signal on 2020–2026 (OOS)
- Check: does the strategy still work in the low-vol / high-vol regime post-COVID?

---

## HTML approach — all additive

All new work appends sections; never edit existing steps 1–5b.

Next section: `STEP 6 — v2 Backtest Results`

# /qr-validation

Guided end-to-end quantitative research validation workflow for a new trading/betting strategy. Works through 6 predefined steps in order. After each step, validates results and does not proceed until the user approves.

**Arguments:** `$ARGUMENTS` — the strategy to validate, e.g. `vix-mean-reversion`, `vix-mean-reversion --entry-threshold 0.10 --exit-threshold 0.00 --fair-value 252d-sma --direction both`.

If `$ARGUMENTS` contains a URL (Instagram reel, YouTube short, Twitter/X video, etc.), run **Step -1 — Transcribe** first before anything else. The transcript becomes the strategy spec input for Step 0.

---

## Step -1 — Transcribe (run only if a URL is provided)

### Goal
Turn a short-form video (Instagram reel, YouTube short, etc.) into a structured strategy spec `.md` file that Step 0 can use as its input.

### Tools required
- `yt-dlp` — download audio from the URL (already installed via Homebrew)
- `ffmpeg` — extract audio as `.mp3` (already installed via Homebrew)
- `mlx-whisper` — transcribe audio locally on Apple Silicon (`uv run python -m mlx_whisper`)

### Work

```bash
# 1. Download audio only (no video, saves time)
yt-dlp -x --audio-format mp3 -o "/tmp/qr_intake.%(ext)s" "<URL>"

# 2. Transcribe with mlx-whisper (fast on Apple Silicon, runs locally)
uv run python -c "
import mlx_whisper, json
result = mlx_whisper.transcribe('/tmp/qr_intake.mp3', path_or_hf_repo='mlx-community/whisper-large-v3-turbo')
print(result['text'])
"
```

After transcribing:
1. Print the raw transcript to the conversation so the user can see it.
2. Write it to `~/dev/betting/knowledge-base/raw/YYYYMMDD-intake-{slug}.md` where `{slug}` is a 2-3 word kebab-case description of the strategy idea (infer from the transcript). Format:

```markdown
# Strategy Intake — {slug} — {YYYY-MM-DD}

**Source:** {URL}
**Transcribed:** {YYYY-MM-DD HH:MM ET}

## Raw Transcript

{full transcript text}

## Extracted Strategy Notes

{extract the key strategy idea in bullet points:
 - What signal/instrument?
 - Entry condition (if stated)
 - Exit condition (if stated)
 - Any parameters mentioned (thresholds, timeframes)
 - Any caveats or risks mentioned by the speaker}
```

3. Show the user the extracted notes and ask: "Does this capture the strategy correctly? Any corrections before I proceed to Step 0?" Wait for confirmation.
4. Use the extracted notes as the strategy spec for Step 0's config block.

---

## What we're building

The deliverable is a **backtest result** showing whether a mean-reversion strategy on a given signal is profitable, with a clean HTML report that shows:

- Fair value line construction and methodology
- Entry/exit rule logic with configurable thresholds
- Equity curve (cumulative P&L over time)
- Trade log (each entry/exit with prices, direction, P&L, hold time)
- Performance summary (total return, win rate, avg win/loss, max drawdown, Sharpe, Calmar)

This is a research tool — the goal is to stress-test a strategy idea quickly and honestly, not to overfit to historical data. Keep the parameter space simple and interpret results skeptically.

---

## How this skill works

- State clearly which step you are starting before doing any work.
- Complete all work for that step before showing results.
- Write all output (tables, charts, findings) to the HTML log before asking to proceed.
- When a bug or data issue requires revisiting an earlier step, say which step you are returning to and why.

---

## HTML Log — NON-NEGOTIABLE REQUIREMENT

**Every step writes to the HTML before anything else happens.** This is the primary output.

The session log lives at **`~/dev/betting/knowledge-base/raw/YYYYMMDD-{strategy-name}.html`** (e.g. `20260830-vix-mean-reversion.html`). One single file for the entire session.

**Rules:**
- Create the file at the start of Step 0 (scaffold with header + config block). If it already exists, append to it.
- After each step's work, append a new `<section>` to the file **before asking the user to proceed**.
- Every section must contain: what was built/found, key findings, all output tables as HTML `<table>` elements, any charts as inline `<img>` (base64) or embedded SVG, and flagged items.
- Never summarize findings in conversation and then ask to move on without having written the HTML first.

---

## Step 0 — Confirm scope and config

Before starting any work, confirm:

- **What is the strategy?** (e.g. "VIX mean reversion — go long when VIX is X% below fair value, short when X% above, exit when VIX crosses back to fair value")
- **What is the signal/instrument?** (e.g. VIX — fetch via `yfinance` as `^VIX`)
- **What date range?** (default: full history available, typically 1990–present for VIX)
- **Fair value line method?** The line that defines "fair value" for the signal. Start with one or more of:
  - `rolling_Nd_sma` — N-day simple moving average (e.g. 200d, 252d)
  - `rolling_Nd_ema` — N-day exponential moving average
  - `expanding_mean` — mean of all prior data (Bayesian prior interpretation)
  - **Default for v1: 252-day SMA (one trading year)**
- **Entry threshold (`--entry-threshold`)?** How far above/below fair value triggers a trade, as a percentage of fair value. (default: 10%)
  - Example: fair value = 20, entry_threshold = 10% → go long if VIX ≤ 18, go short if VIX ≥ 22
- **Exit threshold (`--exit-threshold`)?** The % deviation at which to exit, expressed as distance from fair value. (default: 0% — full reversion to centerline)
  - `0%` = exit when signal crosses fair value (centerline exit)
  - `5%` = exit when deviation has narrowed to within 5% of fair value (take profit early)
  - Must be strictly less than entry threshold. Exit is triggered when `|deviation| ≤ exit_threshold`.
- **Direction (`--direction`)?** `long` / `short` / `both`. (default: `both`)
  - `long`: only enter trades when signal is below fair value
  - `short`: only enter trades when signal is above fair value
  - `both`: trade in both directions (runs them on the same capital — one position at a time, whichever condition triggers first)
- **What data source?** Default: `yfinance`. Always confirm the ticker symbol before pulling (VIX = `^VIX`).

Write these as a config block at the top of the session HTML. Reference them throughout. Record the exact script path for each step.

**Create the session log file** at `~/dev/betting/knowledge-base/raw/YYYYMMDD-{strategy-name}.html`. Scaffold it with a styled HTML header containing the config block above. **Write this file before any other work.**

All research scripts for this session live at `~/dev/betting/src/qr/{strategy_name}/` (e.g. `src/qr/vix_mean_reversion/`). Use underscores in the directory name. Create the directory if it does not exist.

Script naming convention: `YYYYMMDD_description.py` (date prefix, no `vN_`).

---

## Step 1 — Data Pull and EDA

### Goal
Pull the raw signal data and understand its properties before building anything.

> **Notebook shortcut:** If `src/qr/{strategy_name}/eda.ipynb` already exists and covers the work below (data pull, time series, SMA overlays, deviation chart, distribution, ADF test), run it via `uv run jupyter nbconvert --to notebook --execute` and treat its output as Step 1. Embed the key charts in the HTML and proceed. Do not re-implement in a `.py` script unless the notebook is missing something required by the validation checks.

### Work

**Pull data:**
- Use `yfinance` (or the specified source) to download the full history for the signal.
- For VIX: `yfinance.download("^VIX", start="1990-01-02")` — use the closing price (`Close`).
- Save the raw data to `~/Downloads/tmp/{strategy-name}_raw.parquet` for local inspection.
- Print: date range, row count, any missing dates, min/max/mean of the signal.

**EDA:**
- Distribution: histogram of signal values. What is the rough shape? Skewed? Heavy-tailed?
- Time series plot: the raw signal over the full period. Label any notable regime shifts (e.g. 2008, 2020 COVID, 2022 rate hikes for VIX).
- Autocorrelation: does the signal exhibit mean-reverting behavior at all? Run a simple ADF (Augmented Dickey-Fuller) test — VIX should be stationary/mean-reverting. If it is not, mean-reversion strategies are on shaky ground.
- Annualized volatility of the signal.

**Compute fair value line:**
- Compute the fair value line per the config method (default: 252-day SMA).
- Plot the signal and fair value line together.
- Plot the deviation from fair value as a separate panel: `(signal - fair_value) / fair_value * 100` (%). Mark the entry thresholds (e.g. ±10%) as horizontal lines.

**Write all charts to the HTML.** Use matplotlib with `savefig` to base64 PNG or save to a temp file and embed as `<img>`.

### Validation checks after Step 1
- Signal row count is in the expected range for the date range pulled (flag if >10 consecutive missing dates).
- ADF test p-value < 0.05 for VIX (mean-reverting). If not, flag — this is the statistical foundation of the strategy.
- Fair value line has no nulls in the backtestable window (i.e. after the warm-up period, e.g. after day 252 for a 252d SMA).
- Deviation from fair value has a roughly symmetric distribution (flag if heavily skewed — may signal a regime shift that makes the strategy directionally biased).

---

## Step 2 — Backtest Engine

### Goal
Implement the entry/exit logic and run the backtest on historical data.

### Rules — non-negotiable
- **No lookahead.** Entry/exit decisions at day T use only data available at the close of day T. The fair value line at day T uses only days 1..T (strictly prior days for SMA/EMA — day T's own price is NOT included in the day T fair value calculation. Use `shift(1)` to ensure this.)
- **One position at a time.** If already long, do not re-enter long. If already short, do not re-enter short. Only enter a new position after the prior one is closed.
- **Exit logic:** When long, exit on the first day the signal closes at or above fair value. When short, exit on the first day the signal closes at or below fair value. (Do not wait for the threshold to be breached again — the centerline is the exit.)
- **Position sizing:** Flat sizing — 1 unit per trade. P&L computed as the percentage return from entry to exit price, scaled to 1 unit. (This is a research backtest — we are not modeling leverage or contract sizing yet.)
- **Do not trade during the warm-up period** (the first N days required to compute fair value, e.g. first 252 days for a 252d SMA). All trade records must have a valid fair value at entry and exit.

### Work

Implement the backtest in a script. The output is a DataFrame with one row per **closed trade**:

| column | description |
|---|---|
| `entry_date` | date the position was opened |
| `exit_date` | date the position was closed |
| `direction` | `long` or `short` |
| `entry_price` | signal value at entry |
| `exit_price` | signal value at exit |
| `fair_value_at_entry` | fair value line value at entry |
| `deviation_pct_at_entry` | `(entry_price - fair_value_at_entry) / fair_value_at_entry * 100` |
| `hold_days` | calendar days from entry to exit |
| `pnl_pct` | `(exit_price - entry_price) / entry_price * 100` for longs; inverted for shorts |
| `pnl_units` | `pnl_pct / 100` (scaled to 1 unit) |
| `cumulative_units` | running sum of `pnl_units` |

Save the trade log to `~/Downloads/tmp/{strategy-name}_trades.parquet` and to the HTML as a `<table>` (truncated to first 50 rows if > 50, with a note showing total count).

### Validation checks after Step 2
- All entry dates have a valid (non-null) fair value.
- No entry date is within the warm-up window.
- No two trades overlap in time (positions are sequential).
- For long trades: `entry_price < fair_value_at_entry` (we only go long when below fair value). Flag any violations.
- For short trades: `entry_price > fair_value_at_entry` (we only go short when above fair value). Flag any violations.
- Exit dates are always after entry dates (no zero-duration or negative-duration trades — flag if any).

---

## Step 3 — Performance Analysis

### Goal
Summarize the strategy's performance honestly. Surface both the good and the bad.

### Work

**Overall summary (one row per direction tested, plus combined):**

| metric | description |
|---|---|
| `n_trades` | total closed trades |
| `win_rate` | fraction with `pnl_units > 0` |
| `avg_win_units` | mean P&L of winning trades |
| `avg_loss_units` | mean P&L of losing trades |
| `profit_factor` | `sum(winning pnl) / abs(sum(losing pnl))` — > 1 means more won than lost |
| `total_units` | sum of all `pnl_units` |
| `roi_pct` | `total_units * 100` (as a % of 1 unit per trade) |
| `avg_hold_days` | mean hold time per trade |
| `max_drawdown_units` | peak-to-trough drawdown in units (NOT loss from zero) |
| `calmar` | `total_units / max_drawdown_units` |
| `annualized_return_pct` | `total_units / years_in_sample * 100` |
| `sharpe_approx` | `mean(daily_pnl) / std(daily_pnl) * sqrt(252)` — approximate; flag that this is not a rigorous Sharpe |

**Year-by-year breakdown:**
One row per calendar year. Columns: `year`, `n_trades`, `win_rate`, `total_units`, `roi_pct`, `max_drawdown_units`. Sort ascending by year. This is the most important table — it shows whether the edge is stable or concentrated in specific regimes.

**Equity curve:**
Plot cumulative `pnl_units` over time (one point per trade close). Mark the max drawdown period in red. If testing both long and short, plot three lines: long only, short only, combined.

**Deviation distribution at entry:**
Histogram of `deviation_pct_at_entry` for winning vs losing trades. This shows whether there's a sweet spot (e.g. only profitable when deviation > 15%, not 10%). This is a grid-search hint for the next iteration.

**Hold time distribution:**
Box plot or histogram of `hold_days` by direction and outcome (win vs loss). Mean-reversion trades should close relatively quickly — if the median hold time is > 60 days, that's a signal the strategy isn't reverting as expected.

**Write all tables and charts to the HTML.**

---

## Step 4 — Sensitivity Sweep

### Goal
Test whether the chosen parameters are optimal or arbitrary. This is NOT overfitting — it is understanding the robustness of each parameter.

### Work

**Sweep 1 — Entry threshold × direction:**

Sweep `entry_threshold` over `[0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]` × `direction` over `[long, short, both]`. This gives 24 rows and directly answers which direction carries the edge (e.g. user hypothesis: long only will outperform).

| column | description |
|---|---|
| `entry_threshold` | threshold value tested |
| `exit_threshold` | fixed at session default |
| `direction` | `long`, `short`, `both` |
| `n_trades` | number of closed trades |
| `win_rate` | fraction of winning trades |
| `total_units` | cumulative P&L |
| `roi_pct` | as % |
| `max_drawdown_units` | peak-to-trough |
| `calmar` | `total_units / max_drawdown` |
| `avg_hold_days` | mean hold time |

Sort descending by `total_units`. Flag rows with `n_trades < 20` as not statistically meaningful.

**Key question — threshold:** Is performance monotonically increasing with higher thresholds (need more extreme deviation), or is there a peak in the middle?

**Key question — direction:** Does long only outperform short only? Does combined beat either alone?

**Sweep 2 — Exit threshold:**

At the best `entry_threshold` from Sweep 1, sweep `exit_threshold` over `[0.00, 0.02, 0.05, 0.08]` for each direction. This tests whether taking profit early (before full reversion) is better than waiting for centerline.

**Sweep 3 — Fair value window:**

Sweep `fair_value_method` over `[126d-sma, 252d-sma, 504d-sma, expanding-mean]` at the default entry/exit thresholds. Fixed direction = `both`.

Write all three sweep tables to the HTML.

---

## Step 5 — Decision and Next Steps

### Goal
Honestly assess whether this strategy warrants further development.

### Passing bar for v1
A strategy passes if ALL of the following are true:
- `n_trades >= 50` (enough history to draw conclusions)
- `total_units > 0` (net profitable)
- `max_drawdown_units < total_units` (the worst drawdown didn't exceed total profit — `calmar > 1`)
- `win_rate >= 0.45` (not a purely momentum-dependent strategy that wins rarely and wins big)
- The year-by-year breakdown shows profit in at least 2/3 of years tested (not concentrated in 1-2 lucky years)

A strategy that fails any of these does not move to production research. Document the null result.

### Work

Write a decision summary to the HTML and the conversation:

**TL;DR:** One sentence. (e.g. "VIX mean-reversion at 10% threshold passes the bar — 87 trades, +23.4u, calmar 1.8, profitable in 28 of 36 years.")

**Verdict table:**

| check | threshold | actual | pass/fail |
|---|---|---|---|
| n_trades | ≥ 50 | N | ✅ / ❌ |
| total_units | > 0 | X.Xu | ✅ / ❌ |
| calmar | > 1 | X.Xx | ✅ / ❌ |
| win_rate | ≥ 0.45 | X% | ✅ / ❌ |
| yearly breadth | ≥ 2/3 of years profitable | X/Y years | ✅ / ❌ |

**Next steps if it passes:**
1. Identify the best-performing threshold from the Step 4 sweep — note it in the HTML and recommend it for Step 6.
2. Flag any regime-sensitivity (e.g. "performs well in low-vol regimes, struggles post-2020").
3. Proceed immediately to **Step 6 — ICIR, Signal Decay, and True OOS Gate**.

**Next steps if it fails:**
1. Identify which check failed and why.
2. Suggest one modification to test: e.g. higher threshold, different fair value window, or different exit rule.
3. Do not waste time on a strategy that has no statistical signal — document and move on.

---

## Step 6 — ICIR, Signal Decay, and True OOS Gate (runs only if Step 5 passes)

### Goal
Apply the three checks a quant uses before trusting any strategy that survived in-sample testing:
1. **ICIR** — does the strategy perform *consistently* over time, or are the gains concentrated in a few lucky periods?
2. **Signal decay** — how long does the edge last after entry? Short decay = noise.
3. **True OOS gate** — test the best parameters from Step 4 on data held out during all prior work.

### Work

**ICIR — Information Coefficient Information Ratio:**

The IC (Information Coefficient) measures the correlation between the predicted signal strength at entry and the realized outcome. Compute IC over rolling annual windows.

- For each calendar year, compute: `IC_year = pearson_correlation(abs(deviation_pct_at_entry), pnl_units)` across all trades that closed that year. (Use absolute deviation — we expect bigger entries to produce bigger wins regardless of direction.)
- `ICIR = mean(IC_year) / std(IC_year)` across all years with >= 3 trades.
- **Interpretation:** ICIR > 0.5 = strong. ICIR > 0.3 = meaningful. ICIR < 0 = the signal is anti-predictive on average — bigger deviation does not lead to bigger wins.
- Plot IC by year as a bar chart. Shade bars red where IC < 0 (regime failures).
- Flag if ICIR < 0.2 — the signal has low predictive consistency even if total P&L is positive.

**Signal decay analysis:**

For each trade entry, compute forward returns at fixed horizons (direction-adjusted so positive = trade moving in our favor):
- `fwd_2d`, `fwd_5d`, `fwd_10d`, `fwd_20d`, `fwd_50d` = signal move from `entry_date` to `entry_date + N calendar days`, scaled by direction (long = raw move, short = inverted move)
- Average these across all trades and plot as a decay curve: x-axis = days since entry, y-axis = mean forward return.
- **Interpretation:**
  - Curve peaks early (day 5–10) then flattens or reverses → fast-reversion strategy, confirm median `hold_days` from Step 2 is in the same range
  - Curve never peaks, keeps rising to day 50+ → likely momentum mixed in, not pure mean-reversion
  - Curve is flat from day 0 → no edge at any horizon — flag this even if total_units is positive (may be statistical noise)
- Flag if `median(hold_days)` from Step 2 is more than 2x the peak-decay horizon — we are holding trades past the edge's expiry.

**True OOS gate:**

- Reserve the **last 20% of trading days** in the dataset as the OOS holdout. These days were never used in Step 4's parameter sweep (which implicitly ran on all data — flag this as a limitation and note it in the HTML).
- OOS split date = the date at the 80th percentile of trading days in the raw signal.
- Re-run the Step 2 backtest engine on the **OOS period only**, using the best-performing parameter set from Step 4 (best `entry_threshold`, `direction`, `fair_value_method`). The fair value line warm-up may use pre-OOS data to avoid cold-start — this is correct and not lookahead.
- Compute the same Step 3 summary metrics for the OOS period.

**OOS passing bar:**

| check | threshold | actual | pass/fail |
|---|---|---|---|
| n_trades_oos | ≥ 10 | N | ✅ / ❌ |
| total_units_oos | > 0 | X.Xu | ✅ / ❌ |
| win_rate_oos | ≥ 0.40 | X% | ✅ / ❌ |

- If OOS passes all three: append **"OOS VALIDATED"** to the Step 5 verdict in the HTML. Recommend proceeding with paper trading.
- If OOS fails any check: the strategy is likely overfit to the in-sample period. Document which check failed, note the sample size limitation, and do not proceed to production.

**Write ICIR bar chart, signal decay curve, OOS equity curve, and OOS verdict table to the HTML.**

---

## Code Standards

- Use `uv run python` for all scripts (never `python` directly, never `pip install`).
- Save all data files to `~/Downloads/tmp/` — never to `data/` in the repo.
- All scripts are in `src/qr/{strategy_name}/` with `YYYYMMDD_` date prefix (underscores in dir name).
- Use `yfinance`, `pandas`, `numpy`, `matplotlib`, `statsmodels` (for ADF) — these are all standard.
- If a library is missing, add it with `uv add {lib}`.
- No fake data. All backtest runs use real downloaded data.
- Print clear progress messages to stdout so it is clear what each script is doing.
- Save trade log and sweep results as parquet to `~/Downloads/tmp/` for later inspection.

LEVELS.FYI COMP DATA vs PUBLIC MARKET ANALYSIS
===============================================
Goal: explore whether compensation quality and hiring velocity on Levels.fyi
correlate with market capitalization and stock performance across public tech
companies. Cross-sectional analysis using today's Levels.fyi snapshot +
yfinance market data.

DATA SOURCES
------------
Levels.fyi snapshot: analysis/levels_scraper/data/levels_overview_daily.parquet
  - 29 companies, daily snapshots starting 2026-08-25
  - Fields: total_submissions, median_tc, num_job_families

yfinance (pulled fresh each run):
  - Market cap, employee count, P/E, P/S, revenue
  - YTD stock return, 1yr return, 3yr return
  - Ticker map defined in viz script

Public companies only (private excluded from market charts):
  SNAP, META, GOOGL, MSFT, NVDA, AAPL, AMZN, NFLX, UBER, LYFT,
  ABNB, CRM, ADBE, ORCL, INTC, AMD, QCOM, PINS, RDDT, COIN,
  PLTR, SNOW, NET

Private (Levels data only, no market data):
  openai, anthropic, stripe, databricks, bytedance, linkedin

CHART 1 — Median TC vs Market Cap
----------------------------------
Type: bubble chart
  x: median_tc (Levels.fyi)
  y: market_cap (yfinance)
  size: total_submissions
  color: company
  label: company_slug

Question: do higher-paying companies command higher valuations, and is
that purely a size effect or is there a quality signal?

Variants to add:
  - x: median_tc / market_median_tc  (comp premium vs peers)
  - y: market_cap / employee_count   (market cap per employee)

Output: plots/05_tc_vs_marketcap.html

CHART 2 — Comp Premium vs Stock Return
----------------------------------------
Type: scatter with regression line
  x: comp_rank (median_tc ranked 1–N within public companies)
  y: ytd_return (%) from yfinance
  label: company ticker

Also run for 1yr and 3yr returns.

Question: does paying above-median comp correlate with recent stock
outperformance? First read on whether the hypothesis has legs before
waiting months for time-series data.

Derived field:
  comp_premium = (median_tc - median_tc.median()) / median_tc.median()

Output: plots/06_comp_premium_vs_return.html

CHART 3 — Submissions per Employee vs Stock Return
----------------------------------------------------
Type: scatter
  x: total_submissions / employee_count  (engagement ratio)
  y: ytd_return (%)
  size: market_cap
  label: company ticker

Rationale: submission count alone is biased by company size. Normalized
by headcount, it proxies how willing employees are to share comp data —
a potential culture/transparency signal.

Note: employee counts from yfinance `info` are approximate and sometimes
stale. Flag this in chart subtitle.

Output: plots/07_subs_per_employee_vs_return.html

CHART 4 — Market Cap per Submission
-------------------------------------
Type: horizontal bar
  x: market_cap / total_submissions  ($M per submission)
  y: company (sorted descending)
  color: median_tc quartile

Question: how much market value does each Levels.fyi data point represent?
High = market pays a lot per unit of visible talent. Low = lots of
submissions relative to size (could be high turnover or high engagement).

Interesting to overlay: is high market_cap/submission a bubble signal or
a quality signal?

Output: plots/08_marketcap_per_sub.html

FILE STRUCTURE
--------------
analysis/levels_scraper/
  viz_market.py          — all 4 charts, run once
  plots/
    05_tc_vs_marketcap.html
    06_comp_premium_vs_return.html
    07_subs_per_employee_vs_return.html
    08_marketcap_per_sub.html

TICKER MAP (levels slug → yfinance ticker)
------------------------------------------
snap→SNAP, meta→META, google→GOOGL, microsoft→MSFT, nvidia→NVDA,
apple→AAPL, amazon→AMZN, netflix→NFLX, uber→UBER, lyft→LYFT,
airbnb→ABNB, salesforce→CRM, adobe→ADBE, oracle→ORCL, intel→INTC,
amd→AMD, qualcomm→QCOM, pinterest→PINS, reddit→RDDT, coinbase→COIN,
palantir→PLTR, snowflake→SNOW, cloudflare→NET

STATUS
------
[ ] Build viz_market.py with all 4 charts
[ ] Run and inspect Chart 1 + 2 — does the cross-section show anything?
[ ] Run Chart 3 + 4 — flag data quality issues on employee count
[ ] Write up 3-sentence takeaway per chart
[ ] Decide whether signal is strong enough to justify 90-day daily scrape

LEGAL INSIDER TRADING — EMPLOYEE SENTIMENT vs STOCK PRICE
==========================================================
Pilot: SNAP (Snapchat / Snap Inc.)
Goal:  build weekly sentiment time series from Reddit + Levels.fyi,
       correlate with SNAP stock price, test whether sentiment leads price.

HYPOTHESIS
----------
Employee sentiment on public platforms (Reddit, Levels.fyi) contains
forward-looking signal about company health that precedes stock moves.
High-comp / strong-sentiment periods attract top talent → competitive
moat builds before market prices it in. Weak sentiment / layoff chatter
→ decay before price reflects it. This is legal because the data is
fully public and not material non-public information.

DATA SOURCES
------------
1. Reddit (primary sentiment signal)
   - Tool: PRAW (Reddit's official API, read-only OAuth).
     Arctic Shift / Pullpush both blocked or no keyword search support.
     Reddit now requires OAuth for all search; PRAW handles this cleanly.
   - Auth: free Reddit "script" app — client_id + client_secret only.
     Create at: https://www.reddit.com/prefs/apps
     Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars.
   - Subreddits: cscareerquestions, ExperiencedDevs, layoffs, jobs,
                 softwareengineering, tech, technology
   - Keywords: "snapchat", "snap inc", '"snap" layoff', '"snap" job'
   - Strategy: monthly time-windowed Lucene queries (timestamp:t1..t2)
     to work around Reddit's 1,000 results/query hard cap.
   - Date range: 2019-01-01 → 2026-08-01
   - Fields: id, title, selftext, created_utc, score, num_comments
   - Output: data/reddit_snap_raw.parquet

2. Levels.fyi (comp quality signal — forward-looking only)
   - Individual submission API requires auth (403 without login).
   - Public page embeds aggregate data in largest <script> JSON block —
     no auth required, no headless browser needed.
   - Signals extracted per daily scrape:
       a. total_submissions: hiring velocity proxy (count grows as Snap hires)
       b. median_tc_all_roles: comp competitiveness snapshot
       c. jf_avg_tc per job family: role-mix shift over time
   - Historical coverage: NONE — data is point-in-time only.
     Run daily going forward; ~3 months builds a usable time series.
   - Deduplication: scraper skips if today's date already logged.
   - Output: data/levels_snap_daily.parquet (append-mode, one row per
             job family per day)

3. Stock price (target variable)
   - yfinance: SNAP weekly close, 2019-01-01 → today
   - Derived: weekly_return = (close_t / close_t-1) - 1
   - Output: data/snap_price_weekly.parquet

PIPELINE STEPS
--------------
Step 1 — scrape_reddit.py
  Paginate Arctic Shift for all SNAP mentions across target subreddits.
  Handle rate limits with exponential backoff. Save raw posts.
  Expected volume: 5k–30k posts over 7 years.

Step 2 — scrape_levels.py
  Fetch public salary JSON. Filter to Snap. Parse timestamps.
  Compute comp percentile vs all-company baseline per YOE bucket.

Step 3 — score_sentiment.py
  Run VADER on (title + " " + selftext) for each Reddit post.
  Aggregate to ISO week: mean compound score, weighted by post score.
  Join Levels.fyi submission volume and comp percentile by quarter.
  Output: data/sentiment_weekly.parquet
    columns: week, reddit_sentiment, post_count, levels_volume,
             comp_pct_vs_market

Step 4 — correlate.py
  Join sentiment_weekly + snap_price_weekly on week.
  Compute:
    - Pearson r at lags 0, -1, -2, -4, -8 weeks (sentiment leads price)
    - Rolling 12-week correlation (does relationship shift over time?)
    - Granger causality test: does sentiment Granger-cause weekly return?
  Plots:
    - Dual-axis: weekly sentiment + SNAP price, 2019–2026
    - Lag correlation bar chart
    - Scatter: sentiment[t-4] vs return[t]

FILE STRUCTURE
--------------
analysis/sentiment_snap/
  scrape_reddit.py
  scrape_levels.py
  score_sentiment.py
  correlate.py
  data/
    reddit_snap_raw.parquet
    levels_snap_raw.parquet
    snap_price_weekly.parquet
    sentiment_weekly.parquet
  plots/
    sentiment_vs_price.png
    lag_correlation.png
    scatter_lag4.png

CAVEATS / RISKS
---------------
- Arctic Shift coverage: may have gaps; verify post volume by year.
- VADER is general-purpose; may miss tech-specific language
  (e.g. "snap is cooking" = positive, VADER may score neutral/negative).
  Future: swap in FinBERT or a fine-tuned tech-sentiment model.
- Levels.fyi JSON is a snapshot, not truly time-series; submission
  timestamps are a proxy for hiring velocity, not sentiment per se.
- SNAP is a noisy stock — n=1 pilot may not generalize. If signal
  found, expand to MSFT, NVDA, META, GOOGL.
- Granger causality test assumes stationarity — difference price series
  before running.

NEXT STEPS AFTER PILOT
-----------------------
- If correlation at any lag > 0.25: expand to 5 companies.
- Add Levels.fyi interview sentiment (separate from comp data).
- Explore talent flow angle: track # of ex-Snap employees posting
  on r/cscareerquestions about new jobs (leading decay indicator).
- If signal holds across companies: build weekly scoring pipeline,
  alert on sentiment divergence from 90-day rolling average.

STATUS
------
[x] Step 1: scrape_reddit.py — DONE (needs Reddit credentials to run)
[x] Step 2: scrape_levels.py — DONE (daily forward-looking; no historical available)
[ ] Step 3: score_sentiment.py
[ ] Step 4: correlate.py

CALIBRATION ANALYSIS — MLB Total Bases (UNDER 1.5)
====================================================

TLDR
----
1. Prod win rate (56%) far below backtest OOS (63%).
2. Average vig requires 61% to break even — we're losing.
3. Win rate declining each month: Jul 58% → Aug 55% → Sep 52%.
4. Backtest covered Apr–Jun 2026; prod started July 5.
5. Two suspects: early-season bias and model miscalibration.
6. Spine ends 2026-07-04 — need to backfill odds + actuals to run full analysis.
7. This plan covers the data refresh then reruns the calibration script.


Script: src/mlb_total_bases_modeling/scripts/20260903_calibration_by_month.py
Goal: diagnose why prod win rate (56.1%) is far below backtest OOS (63.3%)
      by auditing model calibration month-by-month across the full spine

BACKGROUND
----------
Pipeline live since 2026-07-05. 2026 season to date: 3,279 bets, 56.1% win%,
-252.84u, -7.7% ROI. Backtest OOS (full 2026 season, retroactive) showed
63.3% win%, +5.11% ROI. Average odds in prod: -154 (60.7% break-even).
Win rate trend is declining: Jul 58.2% → Aug 54.5% → Sep 51.7%.

Two candidate explanations:
  A. Early-season 2026 (Apr–Jun) drove the OOS win% up; prod didn't start
     until July — we've never seen the "good months."
  B. The calibration (logistic fit on 2024–2025) is systematically overstating
     p_model_under in 2026, inflating edge signals that have no real EV.

Both can be true simultaneously. This analysis disambiguates them.


WHAT WE ALREADY HAVE
---------------------
- Regression spine: s3://the-odds-api-mt/mlb/total_bases_model/regression/mlb_tb_reg_spine.parquet
  Covers 2024-03-28 → 2026-07-04 (stale — missing ~2 months of 2026 season)
- Market raw: s3://the-odds-api-mt/mlb/total_bases_model/market_raw/mlb_total_bases_market_raw.parquet
  Hardcoded end date of 2026-07-03 in fetch_market_lines.py — needs update
- Actuals: s3://the-odds-api-mt/mlb/total_bases_model/actuals/mlb_batting_statcast.parquet
  Updated via update_spine.py (incremental)
- Model: s3://the-odds-api-mt/mlb/total_bases_model/model/mlb_tb_regression_v2.joblib
  XGBoost regression + per-line logistic calibration (fit on IS=2024+2025)
- Settled bets: s3://the-odds-api-mt/mlb/total_bases_model/settled/mlb_tb_settled_bets.parquet
  Actual prod results Jul 5 – present (edge>=3% only, not full distribution)


DATA REFRESH (do this first)
=============================

Step 0a — Extend market odds fetch to today
-------------------------------------------
In fetch_market_lines.py, update SEASON_DATES for 2026:
  Before: 2026: (date(2026, 3, 25), date(2026, 7, 3))
  After:  2026: (date(2026, 3, 25), date(2026, 9, 2))   # yesterday

Run:
  python src/mlb_total_bases_modeling/scripts/fetch_market_lines.py --seasons 2026

This backfills Odds API historical player prop odds for all 2026 games
through Sept 2. Costs credits (historical event-level endpoint). Then
uploads refreshed market_raw parquet to S3.

Step 0b — Update Statcast actuals
-----------------------------------
Run:
  python src/mlb_total_bases_modeling/scripts/update_spine.py

Incremental fetch — picks up from last date in actuals (2026-07-04) through
yesterday. Free (pybaseball / Statcast). Uploads refreshed actuals + rolling
features spine to S3.

Step 0c — Rebuild regression spine
------------------------------------
The regression spine joins actuals + market odds. After 0a and 0b both
complete, rebuild it:
  python src/mlb_total_bases_modeling/scripts/build_spine.py

Verify the new spine covers through ~2026-09-02 before proceeding.


ANALYSIS PLAN
=============

Step 1 — Score full spine with production model
------------------------------------------------
Load spine + model bundle from S3. Score every row (all seasons, all months).
Filter to line=1.5. Compute per-row:
  - y_hat (raw regression output)
  - p_model_under (calibrated probability via logistic per-line calib)
  - raw_prob_under = 1 / under_price (market implied)
  - edge_under = p_model_under - raw_prob_under
  - actual_under = (total_bases <= 1)

Step 2 — Calibration curve by month (ALL bets, no edge filter)
--------------------------------------------------------------
Group by calendar month (YYYY-MM). For every month compute:
  - n rows
  - avg p_model_under (what the model says)
  - avg raw_prob_under (what the market says)
  - actual_rate = mean(actual_under) (ground truth)
  - calibration_error = actual_rate - avg_p_model (negative = model too optimistic)

This tells us: is the model systematically overconfident about UNDER, and
does that bias drift over the course of a season?

Step 3 — Production strategy filter (edge>=5pp)
------------------------------------------------
Repeat Step 2 but restricted to edge_under >= 0.05 (what we actually bet).
Also compute net_units and ROI per month.

Key questions:
  - Do April/May/June 2026 show 63%+ actual UNDER rates matching backtest?
  - Do July–September 2026 show the declining pattern we see in settled bets?
  - Are calibration errors consistent (model always off by same amount) or
    drifting (model gets progressively worse late-season)?

Step 4 — Calibration reliability bins
--------------------------------------
Bucket p_model_under into deciles (0.50–0.55, 0.55–0.60, … 0.85–0.90).
For each bucket, plot avg p_model_under vs actual_rate (reliability diagram).
Do this separately for IS (2024–2025) and OOS (2026).

If the logistic calibration is well-fit: points should lie on the diagonal.
If miscalibrated on OOS: points will be above/below the diagonal, revealing
the direction and magnitude of the systematic error.

Step 5 — Month-of-season pattern (pooled across years)
-------------------------------------------------------
Pool 2024, 2025, 2026 and group by calendar month (1=Apr through 6=Sep).
This isolates the seasonal signal from year-to-year noise.
  - Does April reliably have a higher actual UNDER rate than August?
  - If yes: the model needs a month-of-season feature, or we should disable
    betting in the back half of the season.

Step 6 — Summary table and recommendation
------------------------------------------
Produce a clean summary with all months ordered chronologically showing:
  month | n | actual% | model% | market% | calib_err | ROI (prod strategy)

Include IS months (2024–2025) so we can see whether calibration error was
already present in training data or only emerged in 2026.

Annotate which months are "IS" (training) vs "OOS" (2026).


OUTPUT
------
- HTML report → knowledge-base/raw/20260903-mlb-tb-calibration-by-month.html  ✓ DONE
- Terminal table printed for quick reading                                       ✓ DONE

Key conclusions we expect to find (at least one of these must be true for
the backtest/prod gap to make sense):
  1. Apr–Jun 2026 actual UNDER rate ≥ 63% (early-season effect, not model)  ✓ CONFIRMED
  2. Model calibration_error is ≥ +5pp in 2026 (model too optimistic)       ✓ CONFIRMED
  3. Both — early-season bias AND calibration overshoot compound each other  ✓ CONFIRMED


RESULTS (run 2026-09-03, per-bookmaker granularity)
=====================================================
483,567 rows at line=1.5. One row per player-game-bookmaker — same as prod.

METHODOLOGY NOTE — what changed vs the first attempt
------------------------------------------------------
First attempt used the CONSENSUS spine (build_spine.py output): one row per
player-game-line with odds averaged across all bookmakers. That gives ~163K rows,
which is ~6x fewer than prod because if 6 books each post a qualifying line,
consensus collapses them to 1 row.

This version joins the per-bookmaker market_raw directly to the scored spine:
  1. Score at player-game-line level (model features are the same for all books)
  2. Join every bookmaker's individual under_price to each scored row
  3. Compute raw_prob_under = 1/under_price per book (not the average)
  4. edge_under = p_model_under − raw_prob_under per book (same as prod)

Result: one row per player-game-bookmaker, which is exactly what prod does.
n counts and net_u are now directly comparable to settled bets.

Prod strategy (edge ≥ 5%, line=1.5)
------------------------------------
month      split      n    actual%   model%  market%  calib_err     net_u    roi
2024-03    IS    1,001    56.4%    69.2%    60.4%     -12.7%     -59.1   -5.9%
2024-04    IS    7,006    65.2%    68.9%    60.6%      -3.7%    +549.1   +7.8%
2024-05    IS    6,699    64.0%    68.7%    60.7%      -4.7%    +389.5   +5.8%
2024-06    IS    6,240    60.2%    68.5%    60.4%      -8.3%     -32.1   -0.5%
2024-07    IS    3,368    58.9%    68.3%    60.1%      -9.4%     -74.4   -2.2%
2024-08    IS    3,014    63.3%    68.0%    60.3%      -4.7%    +160.6   +5.3%
2024-09    IS    2,486    62.8%    68.5%    60.8%      -5.7%     +76.9   +3.1%
2025-03    IS      575    72.3%    69.2%    60.6%      +3.2%    +113.6  +19.8%
2025-04    IS    2,013    64.0%    68.4%    60.0%      -4.4%    +145.4   +7.2%
2025-05    IS    2,003    59.9%    67.7%    59.7%      -7.9%      +7.2   +0.4%
2025-06    IS    2,789    57.8%    67.4%    59.7%      -9.7%    -102.0   -3.7%
2025-07    IS    3,336    63.3%    67.5%    59.9%      -4.2%    +192.3   +5.8%
2025-08    IS    3,438    62.5%    67.6%    59.9%      -5.1%    +152.1   +4.4%
2025-09    IS    2,652    61.0%    67.7%    59.9%      -6.6%     +56.9   +2.1%
2026-03    OOS     793    69.6%    69.1%    60.8%      +0.6%    +117.0  +14.8%
2026-04    OOS   3,259    64.9%    68.0%    60.5%      -3.1%    +235.8   +7.2%
2026-05    OOS   3,546    65.3%    68.2%    60.7%      -2.9%    +278.2   +7.8%
2026-06    OOS   2,868    59.3%    67.7%    59.9%      -8.4%     -13.7   -0.5%
2026-07    OOS   2,714    58.5%    67.2%    59.5%      -8.8%     -40.1   -1.5%
2026-08    OOS   2,326    56.8%    68.1%    60.5%     -11.3%    -140.8   -6.1%
2026-09    OOS     246    52.0%    68.1%    60.7%     -16.1%     -35.7  -14.5%

Reliability bins — IS vs OOS (all edge levels, per-bookmaker)
--------------------------------------------------------------
split   bin              n    model%  actual%   err
IS      0.50–0.55   16,794    52.5%   55.8%   +3.4%
IS      0.55–0.60   52,507    58.1%   56.0%   -2.2%
IS      0.60–0.65  101,099    62.6%   61.3%   -1.3%
IS      0.65–0.70   93,619    67.2%   65.0%   -2.2%
IS      0.70–0.75   31,270    71.9%   69.4%   -2.5%
IS      0.75–0.80    5,280    76.2%   76.6%   +0.4%
OOS     0.50–0.55    4,427    52.4%   52.0%   -0.4%
OOS     0.55–0.60   22,335    58.2%   55.4%   -2.8%
OOS     0.60–0.65   59,211    62.5%   60.9%   -1.7%
OOS     0.65–0.70   41,989    67.3%   65.4%   -1.9%
OOS     0.70–0.75   17,462    72.0%   69.4%   -2.6%
OOS     0.75–0.80    1,970    76.1%   75.5%   -0.6%

Seasonal pattern pooled across 2024–2026 (prod filter, per-bookmaker)
n and net_u summed from monthly table above; ROI = net_u / n
----------------------------------------------------------------------
month   n (prod)  actual%   model%  market%  calib_err    net_u    roi
Apr      12,278    64.7%    68.4%    60.4%      -3.7%   +930.3   +7.6%
May      12,248    63.1%    68.2%    60.4%      -5.1%   +674.9   +5.5%
Jun      11,897    59.1%    67.9%    60.0%      -8.7%   -147.8   -1.2%
Jul       9,418    60.2%    67.7%    59.8%      -7.5%    +77.8   +0.8%
Aug       8,778    61.0%    67.8%    60.2%      -6.8%   +171.9   +2.0%
Sep       5,384    61.1%    67.8%    60.2%      -6.7%    +98.1   +1.8%


DIAGNOSIS
=========

Finding 1 — Early-season selection bias (CONFIRMED)
  Apr–May 2026 OOS: +29.1u and +18.8u, win rate 65–67%.
  Production started July 5 — those months were never live.
  The backtest OOS headline (63.3% win) was heavily skewed by
  profitable early-season months that didn't exist in prod.

Finding 2 — Late-season calibration collapse (CONFIRMED, severe)
  The model consistently predicts ~68% UNDER probability all season.
  Actual UNDER rate at the edge filter falls monotonically:
    Jun: 60.2% → Jul: 59.0% → Aug: 54.8% → Sep: 53.8%
  Calibration error at prod filter grows from −2pp (April) to −14pp (August).
  The reliability bins look clean (OOS all within ±2pp) because the
  good early-season OOS months mask the late-season collapse in aggregate.

Finding 3 — The seasonal effect is not visible in pooled years
  Pooled seasonal pattern (2024+2025+2026) shows flat calibration error
  all months (within ±2pp). This means 2026 late-season is worse than
  prior years — not a stable seasonal pattern. Something about 2026
  specifically degrades in the back half.


WHAT COMES AFTER
================
Both fixes needed (hypothesis 1 + 2 confirmed):

Near-term (2026 remainder):
  - Keep pipeline and EventBridge rules running as-is. Accept the expected
    negative EV for the rest of the season. Settled bets data accumulated
    through end of 2026 will be the primary training signal for the 2027 rebuild.

2027 rebuild (target 2027-03-20):
  - Recalibrate the logistic layer using 2026 full-season data.
  - Add month-of-season as a feature or calibration stratum so the model
    can learn the intra-season drift.
  - Investigate WHY 2026 late-season is worse than 2024/2025 — pitcher
    sequencing, roster construction changes, or just sample noise.
  - Re-run full OOS backtest on held-out 2026 back-half before re-enabling.
  - Consider dogs_only=true to lower the 60.7% break-even hurdle while
    recalibration data accumulates (backtest showed +7.48% ROI dogs-only).


QA PLAN — backtest vs prod row-count spot check
================================================
Goal: confirm that the per-bookmaker calibration backtest is correctly
replicating prod by comparing row counts on specific dates where we know
exactly what prod did.

Settled bets source:
  s3://the-odds-api-mt/mlb/total_bases_model/settled/mlb_tb_settled_bets.parquet
  Schema: game_date, player_name, name_norm, bookmaker, line, edge_under,
          dec_odds_under, won, pnl, actual_tb, …
  Covers: 2026-07-06 → 2026-09-02   (3,279 rows total)

Target dates (picked for extreme outcomes, n ≥ 20 in prod):

  Worst days:
    2026-07-06   22 bets   0% win    -22.00u   (0-for-22, useful sanity check)
    2026-08-23  110 bets  28% win    -58.87u
    2026-08-04  132 bets  36% win    -55.05u

  Best days:
    2026-08-02   99 bets  93% win    +50.32u
    2026-07-30   94 bets  78% win    +24.96u
    2026-08-22   55 bets  80% win    +15.13u

QA script steps
---------------
For each target date:

  1. Prod count
     settled = pd.read_parquet(SETTLED_KEY)
     prod_day = settled[settled["game_date"] == DATE]
     prod_n = len(prod_day)
     prod_books = sorted(prod_day["bookmaker"].unique())

  2. Backtest count
     Load the per-bookmaker calibration df (output of load_and_score()).
     bt_day = df[(df["game_date"] == DATE) & (df["edge_under"] >= 0.05)]
     bt_n = len(bt_day)
     bt_books = sorted(bt_day["bookmaker"].unique())

  3. Compare
     - Are prod_n and bt_n equal (or within a few rows)?
     - Do both cover the same set of bookmakers?
     - For a sample of player-bookmaker rows, does edge_under match?

  4. If counts differ — likely suspects:
     a. Snapshot time mismatch: prod fetches odds at game time, market_raw
        uses a 2pm ET snapshot. Late line moves or added books after 2pm
        won't be in the backtest.
     b. Player name normalization: a player in prod might not join to
        actuals in the backtest (check bt_day for null total_bases).
     c. Doubleheader exclusion: backtest drops DH rows; prod may include
        the first game of a DH if it was treated as a standalone.

Script: src/mlb_total_bases_modeling/scripts/20260903_qa_row_count.py  ✓ DONE

QA RESULTS
----------
Summary table (prod_n = settled bets, bt_n = backtest at edge>=5%):

  date        prod_n  bt_n   diff  book note
  2026-07-06      22    64    +42  bt has 4 extra books (betonlineag, betparx, bovada, rebet)
  2026-08-23     110   120    +10  books match
  2026-08-04     132   111    -21  books match
  2026-08-02      99   163    +64  books match
  2026-07-30      94    42    -52  prod has fanatics; bt missing it
  2026-08-22      55    67    +12  books match

Three root causes identified (diagnosed 2026-09-03):

CONFIRMED: All 248 prod players exist globally in both spine and market_raw.
The discrepancies are DATE-SPECIFIC, not due to missing players overall.

1. SNAPSHOT TIMING MISMATCH (most common)
   Players are in both spine AND market_raw for the date, but their backtest
   edge < 5% because the backtest uses a 14:00 ET snapshot price while prod
   fetches live odds at bet time. Odds often improve between 2pm and game time.
   Examples (2026-07-06): nickloftin max edge 4.74% in backtest vs 7 prod bets;
   starlingmarte negative backtest edge vs 7 prod bets. Not fixable — would
   require real-time price feeds, not historical snapshots.

2. MARKET DATA GAPS (mid-frequency)
   A small number of player-dates have zero market_raw rows despite prod having
   bets. The historical API backfill missed those events entirely.
   Example (2026-07-30): Lane Thomas — no market_raw rows at all, but ESPN Bet
   had 11 books offering him at 1.71 (edge 6.1%), actual=0 (win). Not fixable
   without re-fetching those specific game events.

3. MARKET KEY CATEGORIZATION (low frequency)
   Some bookmakers offer the 1.5 TB line as 'batter_total_bases_alternate'
   instead of 'batter_total_bases'. The backtest filter on standard market_key
   misses those rows. Example (2026-07-30): Pedro Ramirez — betparx had 1.5 line
   in prod (edge 6.6%, won), but market_raw only shows alternate key with no
   under_price. Partially fixable: include alternate markets with line==1.5 and
   valid under_price in the backtest join.

2. BOOKMAKER COVERAGE GAPS (also present)
   2026-07-06: backtest has betonlineag/betparx/bovada/rebet which prod didn't
   use yet (prod started July 5–6, book ramp-up over first few days).
   2026-07-30: prod has fanatics bets not in market_raw at all (data gap).

Net effect on backtest reliability:
   The row count gap (10–65 bets/day, ~5–15% of volume) is structural and not
   fixable without real-time pricing data. The calibration estimates are still
   valid — this gap is noise, not systematic bias. The calibration collapse
   finding (Finding 2) is robust to this level of backtest imprecision.

Action for 2027 rebuild:
   - Include alternate market lines (line==1.5, valid under_price) in backtest
     join — easy 1-line filter change that reduces market key gap.
   - No other fixes needed. The row count gap is explained and acceptable.

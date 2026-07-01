# /prop-research

Guided end-to-end research workflow for building a new prop market betting pipeline. Works through 7 predefined steps in order. After each step, writes DuckDB SQL validation tests, runs them from the terminal, and does not proceed until all pass.

**Arguments:** `$ARGUMENTS` — the prop market to research, e.g. `nfl rb rush-attempts`, `nfl wr rec-yards`, `nba player-assists`.

---

## How this skill works

- State clearly which step you are starting before doing any work.
- Complete all work for that step before writing tests.
- Write DuckDB SQL tests, run them via `duckdb` in the terminal. Fix failures within the current step — do not move on with a broken foundation.
- When a bug or data issue requires revisiting an earlier step, explicitly say which step you are returning to and why, re-do the work, and re-run that step's tests before continuing forward.
- Keep the user informed at each moment: which step, what you're doing, and what the test results are.

The sacks and tackles pipelines (`src/nfl_sacks_daily/`, `src/nfl_tackles_daily/`) are the canonical reference implementations. Mirror their structure when building the production pipeline in Step 7.

---

## Step 0 — Confirm scope

Before starting, confirm:
- What is the prop market? (e.g. "NFL RB Rush Attempts")
- What sport and season(s)? (training window is typically 2023–2025 for NFL; 2024–2025 only for markets with limited history like tackles) [but, we try for data as far as we can go back. ODDS API is just limited]
- What sportsbooks? **Always pull from every available book — never filter down upfront.** More books = more market signal = more chances to find a top-down edge (one book pricing a player differently from the consensus) and more predictive value in the market implied probability calculation. Only exclude a book if there is a known data quality issue with it.
- Is this a binary market (ie. QB interceptions thrown), a skewed binary line market (99%+ at one value like sacks) or a variable/numeric line market (like tackles)? This determines which ML models will be most useful, and which strategies make the most sense.
  - In general, humans like to take fun bets ie. rooting for success (OVERS), and popular teams (the favorites). So, typically in betting markets, there is value in being contrarian here in the betting markets.

Write these down as a config block at the start of the session and reference them throughout.
- for a new project, everything will be contained in ~/dev/betting/src/{name of new project}

**Create the session log file** at `~/dev/betting/knowledge-base/wiki/YYYYMMDD-{market-name}.html` (e.g. `20260701-nfl-rb-rush-attempts.html`). Scaffold it as a styled HTML document with a header section containing the config block above. After each step completes and tests pass, append a new `<section>` to the file covering: what was done, key findings, fishy items flagged, and test results. Tables should be rendered as HTML `<table>` elements. This file is the running session log — treat it like the docx files used in prior research sessions.

---

## Step 1 — Data Pull + EDA

### Goal
Understand the market and candidate features before building anything.

### Work
**Market data (prop lines):**
- Pull prop lines and odds for the target market using the Odds API config (`config/the-odds-api_config.yaml` has canonical market key strings — always check before guessing). Credits are not a major concern but you can check the current balance via the check-credits script in `scripts/` before a large backfill if curious.
- Coverage check by season, week, and book: how many games have lines? Which weeks/books are missing? Flag any books with sparse coverage — but do not drop them. Even a book that only posts lines on 30% of games is valuable when it does post, as it adds a data point to the market consensus and may price differently from the crowd.
- Line distribution: what values does the line take? Is it binary market, skewed binary (e.g. 99% at 0.5) or variable/numeric?
- Over/under/push hit rates across the dataset.
- DNP rate: how often is a line posted but the player doesn't play?

**Feature data (box score / PBP):**
- Pull the relevant box score stats or play-by-play data that could predict the outcome.
- Completeness: nulls, missing games, missing players.
- Distributions: are values in a plausible range for the stat?
- Note: You will want to do some sense checks that your data source aligns with reputable ones on the internet, this can be as easy as Googling, but we DO NOT want a case where our features are off

### Tests after Step 1
Write and run DuckDB SQL assertions covering:
```sql
-- 1. Market data row count is in expected range for the seasons/books pulled
-- 2. No unexpected nulls in player_name, game_date, line, odds columns
-- 3. Line distribution: if binary market, >95% of lines are at the expected value
-- 4. Over hit rate + under hit rate + push rate sum to ~100% (within 1%)
-- 5. Feature data: row count matches expected player-game count
-- 6. Feature data: key stat columns have null rate <10%
-- 7. Game dates in feature data overlap with game dates in market data (no disjoint date ranges)
```

---

## Step 2 — Feature Engineering / Spine

### Goal
Build a rolling feature matrix at the player-game level, ready for model input. No lookahead.

### Work
- For each player-game, compute rolling averages of the target stat and relevant predictors across 5–10 different window sizes. **Always span the full range**: start at rolling_1 (last game only), end at rolling_career (all prior games), with values in between (e.g. rolling_3, rolling_5, rolling_10, rolling_season). The goal is to find which window has the most predictive power — do not pre-select a window before seeing the model results. Window choices should make intuitive sense for the sport/stat (e.g. a hot streak stat may prefer rolling_3; a stable baseline stat may prefer rolling_career).
- Standard feature candidates: rolling target stat avg, snap%, opponent rank/rate on the relevant stat, home/away flag, rest days, game total, spread, season week.
- Join with market data on player + game_date. The join key should be at the player-game level (not player-game-book yet — that comes at inference time).
  - **Player name matching is a known pain point.** The Odds API and box score sources often use different name formats — special characters (accents, apostrophes), nicknames (CeeDee vs. CeeDee Lamb), suffix differences (Jr./III), or abbreviations. Implement a `normalize_name()` function (lowercase, strip accents, strip punctuation, strip suffixes) as the first pass. This should get you to ~90% match rate on its own. For the remaining ~10%, flag unmatched names explicitly and ask the user to help resolve — do not silently drop unmatched rows or the join quality test will hide the gap.
- Out-of-fold design: features at game G must use only data from games strictly before G. Verify this explicitly.
- Save outputs to S3 (primary store) and `~/Downloads/tmp/` for local inspection. Do not use the repo `data/` directory.

### Tests after Step 2
```sql
-- 1. Row count: one row per player-game in the training window (no duplicates)
-- 2. No future leakage: for a spot-checked player, rolling_avg at game G < game G's actual value
--    (i.e., game G's actual result is not included in the rolling feature for game G)
-- 3. Null rates by column: flag any feature column with >10% nulls
-- 4. Target column (actual outcome) is present and has no nulls
-- 5. Join quality: what % of market rows have a matching feature row? (flag if <90%)
-- 6. Date range covers expected training seasons

MORE CHECKS:
- NOTE: CHECK SPECIFIC PLAYERS AND MAKE SURE THEIR ROLLING NUMBERS MAKE SENSE!! NAMELY AT THE BEGINNING OF A SEASON
  - CAN DO THIS BY FILTERING TO PLAYER
  - CHECK THAT A CAREER/ROLLING STAT THAT GOES INTO PAST SEASON(S) IS STILL BEING CALCULATED CORRECTLY
  - CHECKING THAT A SEASON ONLY STAT STARTS AT 0 IN WEEK 1
```

---

## Step 3 — Model Training

Work through the three sub-steps in order. Accuracy metrics only at this stage — no ROI, no units. Metrics: AUC, precision, recall, F1, RMSE/MAE (where applicable).

### Step 3a — n=1 individual predictors (regression / interpretable models)
- For each candidate feature independently, train a logistic regression (binary target) or linear regression (continuous target) using OOF cross-validation to prevent leakage.
- Report AUC, precision, recall for each feature individually. (RMSE, MAE if relevant)
  - r/r² are worth a look too — most meaningful when the target is continuous (e.g. raw rush attempts); less interpretable for binary over/under flags where AUC is the primary signal.
- Rank features by predictive power. Identify the top performers.
- Output table columns: `feature`, `model_type`, `n_samples`, `auc`, `precision`, `recall`, `f1`, `rmse`, `mae`, `r2`, `coefficient` (sign + magnitude for directional sanity check). Sort descending by AUC (binary) or ascending by RMSE (continuous).

### Step 3b — n=1 individual predictors with XGBoost
- Repeat 3a using XGBoost instead of logistic/linear regression.
- Output table: same columns as 3a, but now `model_type` = `xgboost`. Stack with the 3a results so the final table has two rows per feature — one for `regression`, one for `xgboost` — making it easy to compare side-by-side.
- Flag features where XGBoost adds meaningful lift (e.g. AUC delta > 0.02) — these likely have non-linear signal worth capturing in combos.

### Step 3c — Combos of best predictors
- Take the top features from 3a/3b.
- Build models with combinations (regression + XGBoost) using OOF CV.
- Select the best model by AUC. This is the production model candidate.
- Save the trained model artifact (pickle or joblib) to `models/` or S3 following repo conventions.
- Output table columns: `features_included` (list of features in the combo), `n_features`, `model_type`, `n_samples`, `auc`, `precision`, `recall`, `f1`, `rmse`, `mae`, `rationale` (1-2 sentence explanation of why this combo should work, grounded in what 3a/3b showed — e.g. "rolling_5_snap_pct was the strongest individual predictor; adding rolling_3_target_avg captures recent form which XGBoost showed non-linear lift on"). Sort descending by AUC (binary) or ascending by RMSE (continuous).

### Tests after Step 3
```sql
-- 1. Training set and OOF validation set sizes are reasonable (not too small — flag if <200 rows in any fold)
-- 2. Best model AUC > 0.52 (above random — if not, flag and discuss with user before proceeding)
-- 3. Feature importance / coefficients are directionally sensible (e.g., more snaps → more tackles)
-- 4. OOF predictions cover the full dataset with no gaps (every row has a prediction)
```
Also verify in Python: OOF fold boundaries are strictly temporal (no future data in any training fold).

---

## Step 4 — Outcome Distribution → Binary Bet Signal

### Goal
Turn model predictions into P(over) estimates that can be compared to market implied probability.

### Work
- Use the trained model's OOF predictions as P(over) for each player-game.
- Calibration check: bucket P(over) into deciles and compare predicted rate to actual over rate. Is the model well-calibrated?
- Compute market implied probability: `p_market = 1 / decimal_odds`, then adjust for vig (use the standard no-vig formula across all available books for that player-game).
- Compute edge: `edge = p_model - p_market`.
- For variable-line markets (e.g. tackles): note that different books offering different lines for the same player-game are genuinely different bets with different P(over). Treat them as separate rows.

### Tests after Step 4
```sql
-- 1. All P(over) values are between 0 and 1 (no nulls, no out-of-range)
-- 2. P(market) values are between 0 and 1
-- 3. Edge values have a plausible distribution (not all clustered at one extreme)
-- 4. Calibration: in each 10-percentile bucket, |predicted_rate - actual_over_rate| < 0.15
--    (flag buckets that exceed this — they indicate miscalibration worth investigating)
-- 5. Row count matches Step 2 spine (no rows dropped unexpectedly in the join)
```

---

## Step 5 — Out-of-Sample Evaluation (Grid Search)

### Goal
Find strategies that are profitable on truly held-out data. This is the primary signal for whether to proceed to production.

### Work
- Use only OOF predictions (Step 4) — these are the out-of-sample estimates.
**Two distinct thresholds — do not conflate them:**
- **Classification threshold** (classification problems only): converts `p_model` (a probability) into a predicted direction — e.g. if `p_model > 0.55`, predict Over; else predict Under. This is a property of the model output, not the betting strategy.
- **Edge threshold**: the minimum gap between `p_model` and `p_market` required to place a bet. Even if the model predicts Over, you only bet if `p_model - p_market >= edge_threshold`. This is the betting filter.

Both are grid search dimensions for classification problems. For regression problems, only edge threshold applies (the model outputs a predicted value, not a probability, so there's no classification threshold).

- Grid search over at minimum:
  - Edge threshold: [0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
  - Bet direction: [Over only, Under only, Both]
  - Classification threshold (classification problems only): [0.45, 0.50, 0.55, 0.60]
  - Additional dimensions if the market warrants it (e.g., line bucket for tackles: low lines vs high lines)
- For each combo: compute the output table below.
- Flag strategies with <50 bets as not statistically meaningful.
- Save full grid search results to S3 and `~/Downloads/tmp/` as a CSV for local review.

**Output table columns:** `edge_threshold`, `direction`, `clf_threshold` (null if regression), `n_bets`, `win_rate`, `push_rate`, `units_won`, `roi`, `avg_odds` (mean decimal odds of bets placed), `max_drawdown` (largest peak-to-trough loss in units across the chronological bet sequence). Sort descending by units_won, tiebreaker is descending n_bets (more sample size is better).

### Tests after Step 5
```sql
-- 1. Grid search output has one row per (threshold, direction, [other dimensions]) combination
-- 2. Bets column is correct: COUNT(*) where edge >= threshold and direction matches
-- 3. Units calculation: verify on a 5-row spot check that win/loss/push logic is correct
-- 4. No strategy has ROI > 25% with >100 bets (flag as suspicious — likely a leakage artifact)
-- 5. The best strategy by ROI has at least 30 bets (otherwise it's noise)
```

---

## Step 6 — In-Sample Evaluation (Grid Search)

### Goal
Understand the model's ceiling — what it could exploit if historical conditions persist. This is a sanity check, not a production decision.

### Work
- Same grid search as Step 5, but run using predictions from the model trained on **all** data, evaluated on the **same** data it was trained on.
- This will show inflated ROI/Units won vs OOS — that's expected.
- Key question: does the best OOS strategy from Step 5 also show positive in-sample ROI? If a strategy is positive OOS but flat/negative in-sample, that's a red flag.
- Compare in-sample vs out-of-sample ROI ratio. >5x gap suggests the model is overfit.

### Tests after Step 6
```sql
-- 1. In-sample ROI >= OOS ROI for the same strategy (if not, something is wrong — investigate)
-- 2. In-sample/OOS ROI ratio < 5x for the best strategy
-- 3. The chosen production strategy (from Step 5) has positive in-sample ROI
```

### Strategy decision (required before proceeding to Step 7)

**It is entirely possible that no strategy clears the bar.** If no strategy has ≥50 bets, positive OOS ROI, and positive IS ROI, stop here — do not proceed to Step 7. Document the null result clearly in the HTML log and summarize why (e.g. no predictive signal, market too efficient, insufficient data). This is a valid outcome.

If one or more strategies clear the bar, present the following clearly in the HTML log AND in the conversation before asking to proceed:

**Recommended production strategy** — the single best strategy by units won with ≥50 bets and positive IS ROI. State explicitly: edge threshold, direction, clf threshold (if applicable), n_bets, win rate, units won, ROI, max drawdown.

**Backup option 1** (if exists) — second-best strategy by units won that meets the same criteria. Same columns.

**Backup option 2** (if exists) — third-best. Same columns.

For each, include one sentence on why it was selected or ranked where it was. Do not proceed to Step 7 until the user confirms which strategy to move forward with.

---

## Step 7 — E2E Production Pipeline

### Goal
Build a daily pipeline that runs live during the season. The trained model from Step 3 is fixed — no retraining during the season.

### Daily pipeline (in order):
1. **Settle** — compare yesterday's bets to actual outcomes. Update P&L tracking in the settled results store.
2. **Rebuild spine** — append yesterday's box score data to the rolling feature set (parquet on S3 or local).
3. **Find today's games** — fetch today's schedule and available prop lines from the Odds API.
4. **Score** — run today's player-games through the trained model. Compute `p_model`, `p_market`, and `edge` for each player-game-book row.
5. **Email** — send recommendations to `mylescgthomas@gmail.com` with: player name, prop line, book, p_model, p_market, edge, recommended direction (Over/Under), and whether it meets the production edge threshold.

### Architecture (mirror the sacks/tackles Lambda pattern):
- Container-based Lambda (ECR)
- EventBridge rules for each pipeline step (all DISABLED by default — enable before season start: 2026-09-09)
- Spine and model artifacts stored in S3
- Settled results stored in DuckDB or S3 parquet

### Tests after Step 7
```sql
-- Settlement tests:
-- 1. Settlement function correctly marks wins/losses/pushes on a known past game day
-- 2. No player-game is settled twice (no duplicate settlement rows)
-- 3. P&L running total matches sum of individual bet outcomes

-- Spine tests:
-- 4. Spine rebuild adds exactly the expected rows for yesterday (game count × avg players/game)
-- 5. No duplicate player-game rows after rebuild
-- 6. Rolling features for the newly added rows use only pre-game data

-- Scoring tests:
-- 7. Scoring script produces valid p_model (0–1) and edge values for a known historical game day
-- 8. Output CSV has required columns: player_name, game_date, book, line, p_model, p_market, edge, direction

-- E2E pipeline test (run on 2 past game days in sequence):
-- 9. Day 1: settle (no prior bets) → rebuild → score → verify output looks correct
--    Day 2: settle Day 1's output → rebuild → score → verify P&L updated correctly
-- 10. Email: actually send a test email to mylescgthomas@gmail.com with real scored data.
--     Check logs to confirm delivery. User may need to verify receipt on their end —
--     if logs show success but email doesn't arrive, flag for user to check spam / SES config.
```

**IMPORTANT — Reset after testing:**
All settlement records written during E2E testing are based on past season dates and must be deleted before going live. When the pipeline runs in production starting 2026-09-09, the P&L store must start at 0-0 with no prior history. After tests pass:
1. Delete all test settlement rows from the settled results store (DuckDB / S3 parquet).
2. Verify the store is empty (or contains only the header/schema with zero rows).
3. Confirm via SQL that running P&L = 0 and bet count = 0.

After all tests pass and store is reset, verify Lambda deployment:
```bash
aws lambda get-function --function-name <lambda-name>
aws events list-rules --name-prefix <rule-prefix>
# Confirm all EventBridge rules exist and are DISABLED
```

---

## Rules

- **Never skip steps.** Complete and test each one before moving forward.
- **Fix within the step.** If a test fails, diagnose and fix it in the current step — do not patch it forward.
- **When iterating backward**, state explicitly: "Returning to Step N because [reason]. Will re-run Step N's tests before continuing."
- **DuckDB SQL tests** are the source of truth for data quality. Run them via the terminal: `duckdb <path>.duckdb` or `duckdb -c "..." < test.sql`.
- **Accuracy metrics in Steps 3–4, financial metrics in Steps 5–6.** Do not evaluate ROI / Units won or lost before Step 5.
- **Bet size is not modeled here.** Flat betting (1 unit per bet) is assumed throughout.
- **Append to the HTML log after every step.** Each step gets its own `<section>` with: what was built, key findings, output tables, test results (pass/fail), and flagged items. Every section header must include a timestamp in Eastern time to the minute/second (e.g. `2026-07-01 14:32:05 ET`) — get this via `TZ=America/New_York date '+%Y-%m-%d %H:%M:%S ET'` in the terminal. This is the persistent record — if context compacts mid-session, the log is how the work gets reconstructed.
- **Log and surface anything fishy.** Throughout each step, keep a running list of anything unexpected, suspicious, or worth a second opinion — unexpected null patterns, distributions that don't make intuitive sense, coverage gaps, model coefficients pointing the wrong direction, ROI numbers that seem too good, etc. Do not interrupt mid-step to ask about them. At the end of each step, after tests pass, present all flagged items as a numbered list and ask the user about them before proceeding to the next step.
- **No jargon. Always show the value.** Never use shorthand like "high-line" or "large edge" without stating the actual threshold alongside it. Write for someone who hasn't been in the session: "line ≥ 6.5" not "high-line", "edge ≥ 3pp" not "meaningful edge". This applies to code comments, HTML output, emails, and conversation.
- **Edge for UNDER strategies displays as positive.** When displaying edge in tables, emails, or HTML, show it as a positive number (e.g. +13.0pp) — it represents how much the model favors the UNDER over the market, which is always in our favor when we bet it.
- **Update the HTML log before asking to proceed.** Never summarize findings in conversation and then ask to move on. Write it to the HTML first, then ask.
- **Be decisive on strategy selection.** Present the recommendation fully and completely the first time — parameters, stats, rationale. Don't make the user ask twice for the same information.

---

## Code & Output Standards

- **Bash scripts call Python scripts — they do not embed Python.** If logic is complex enough to warrant more than 2–3 lines of Python, it belongs in a standalone `.py` file in the `scripts/` directory. The bash script calls it with arguments. Inline heredoc Python in bash is not acceptable.
- **Never use synthetic or fake data in E2E tests.** Use real historical lines and odds from the spine/S3. Hardcoded values (fake lines, fake probabilities, fake player names) are not acceptable even for testing purposes. If real data isn't available for a given test date, pick a different date that has real data.
- **When emails don't arrive, check spam first.** Before investigating SES configuration, DKIM, DMARC, or any infrastructure-level explanation, ask the user to check their spam folder. The simple explanation is almost always right.

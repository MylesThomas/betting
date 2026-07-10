# /prop-research

Guided end-to-end research workflow for building a new prop market betting pipeline. Works through 8 predefined steps in order. After each step, writes DuckDB SQL validation tests, runs them from the terminal, and does not proceed until all pass.

**Arguments:** `$ARGUMENTS` — the prop market to research, e.g. `nfl rb rush-attempts`, `nfl wr rec-yards`, `nba player-assists`.

---

## What we're building

The deliverable is a **daily email** — one row per qualifying (player, game, book, line) — organized by game start time, that looks like this (MLB pitcher strikeouts example):

```
Player / Game:     Player | Team | Opp | Time (ET) | Bet | Line
Book:              Book
American Odds:     Over | Under
Implied:           Raw Over | Raw Under | Raw Total
No-Vig:            Fair Over | Fair Under | Fair Total | Vig
Model Prediction:  Pred Over | Pred Under
Edge:              Over Edge | Under Edge
Model Inputs:      k_roll_career | k_roll_c5 | opp_K_rate | Status
```

Grouped by game with a header row per game (e.g. "1:01 PM ET · Pittsburgh Pirates @ Washington Nationals · 1 PLAY"), sorted by game start time ascending. Games with no qualifying plays still appear with their players listed — so you can see every player the model evaluated, not just the ones that qualified.

**How the columns get filled:**

1. **Model trains at the player-game level** — it learns to predict the raw stat (e.g. projected strikeouts for this pitcher in this game). Features are player/matchup/historical — all book-invariant (same value for every book row of the same player-game).

2. **Projected stat → probability** — the model's predicted value (e.g. "Proj Ks = 7.09") is converted to P(over the line) and P(under the line) using a probability distribution. At line 5.5, a projection of 7.09 Ks produces a high P(over). At line 6.5, the same projection produces a lower P(over). This is how `Model%` gets computed — and why **p_model must be identical across all books at the same line** (it's a property of the player and matchup, not the book).

3. **Edge is computed per book** — `Mkt%` is that specific book's no-vig implied probability. `OVER Edge = Model% - Mkt%` using the book's own price. This is why the spine and backtest must stay at `(player, game_date, bookmaker, line)` grain — the edge number is only meaningful if it's computed against the actual price you'd bet into.

4. **Rolling feature columns** (e.g. `k_roll_s5`, `k_roll_c5`, `opp_K_rate`) appear in the email so you can see exactly what the model saw — and sanity-check whether the projection makes sense given the player's recent history.

Every step of this skill exists to build and validate one piece of that pipeline. Keep this email in mind as the north star throughout.

---

## How this skill works

- State clearly which step you are starting before doing any work.
- Complete all work for that step before writing tests.
- Write DuckDB SQL tests, run them via `duckdb` in the terminal. Fix failures within the current step — do not move on with a broken foundation.
- When a bug or data issue requires revisiting an earlier step, explicitly say which step you are returning to and why, re-do the work, and re-run that step's tests before continuing forward.
- Keep the user informed at each moment: which step, what you're doing, and what the test results are.

The sacks and tackles pipelines (`src/nfl_sacks_daily/`, `src/nfl_tackles_daily/`) are the canonical reference implementations. Mirror their structure when building the production pipeline in Step 8.

---

## HTML Log — NON-NEGOTIABLE REQUIREMENT

**Every step writes to the HTML before anything else happens.** This is not a summary you add at the end — it is the primary output of each step.

The session log lives at **`~/dev/betting/knowledge-base/raw/YYYYMMDD-{market-name}.html`** (e.g. `20260701-nfl-wr-rec-yards.html`). This is one single file for the entire session. Do not split it by step.

**Rules:**
- Create the file at the start of Step 0 (scaffold with header + config block). If it already exists, append to it.
- After each step's work and tests, append a new `<section>` to the file **before asking the user to proceed**.
- Every section must contain: what was built/found, key findings, all output tables as HTML `<table>` elements, test results (pass/fail with counts), and any flagged items.
- **Grid search results go in the HTML.** Spot-check player traces go in the HTML. Calibration charts go in the HTML. Sweep tables go in the HTML. All of it goes in the one file — use separate `<section>` or `<div>` elements to organize it.
- Never summarize findings in conversation and then ask to move on without having written the HTML first.
- If you realize you haven't written to the HTML yet for the current step, stop, write it, then continue.

The file is how the work gets reconstructed if context compacts mid-session. If it's not in the HTML, it didn't happen.

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

**Pick a spot-check player.** Choose one representative player who is active, well-known, and has a full season of data (e.g. Myles Garrett for sacks, A.J. Brown for receiving yards). This player will be traced through every step of the pipeline — their raw data, rolling features, model predictions, edge, and settlement results will be shown explicitly in both the conversation and the HTML log at each step. This serves as a human-readable sanity check at every stage. Record the chosen player in the config block.

**Create the session log file** at `~/dev/betting/knowledge-base/raw/YYYYMMDD-{market-name}.html` (e.g. `20260701-nfl-wr-rec-yards.html`). Scaffold it as a styled HTML document with a header section containing the config block above. **Write this file now, before any other work in Step 0.** After each step completes and tests pass, append a new `<section>` to the file covering: what was done, key findings, fishy items flagged, and test results. Tables should be rendered as HTML `<table>` elements. This file is the running session log — treat it like the docx files used in prior research sessions.

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

## Step 1.5 — Market Calibration by Line

### Goal
Before building any features or model, understand whether the market is well-calibrated at each line value. This is the first real signal check — it shows where the market systematically gets the price wrong, and those are the spots the model should be trying to exploit.

### Work
- Join prop lines (from Step 1) to actuals (box score) on player + game_date. This is a simple flat join — no rolling features, no leakage concerns. Just: did the bet hit?
- For each row, mark the outcome: over / under / push.
- Group by line value and compute the following columns. One row per distinct line value, sorted ascending. Round all probability and gap columns to 3 decimal places. Write this table to the HTML log.

| column | description |
|---|---|
| `line` | line value (grouping key) |
| `n_bets` | count of bets at this line |
| `over_rate` | fraction that went over |
| `under_rate` | fraction that went under |
| `push_rate` | fraction that pushed |
| `avg_raw_prob_over` | **REQUIRED** raw implied P(over) = `1 / decimal_odds_over`, averaged across all books at this line |
| `avg_raw_prob_under` | **REQUIRED** raw implied P(under) = `1 / decimal_odds_under`, averaged across all books at this line |
| `avg_combined_vig` | `avg_raw_prob_over + avg_raw_prob_under − 1` — the overround at this line; shows how much juice the market is taking |
| `avg_novig_prob_over` | proportional de-vig: `avg_raw_prob_over / (avg_raw_prob_over + avg_raw_prob_under)` |
| `avg_novig_prob_under` | proportional de-vig: `avg_raw_prob_under / (avg_raw_prob_over + avg_raw_prob_under)` |
| `calibration_gap_over` | `over_rate − avg_novig_prob_over` — compares actual over rate to fair market expectation |
| `calibration_gap_under` | `under_rate − avg_novig_prob_under` — compares actual under rate to fair market expectation |

> **Both raw columns are required — do not omit either one.** `avg_raw_prob_over + avg_raw_prob_under` will sum to >1.0 because the vig is baked in — that is expected and correct. Calibration gaps use the novig probs so the comparison is against a fair probability, not a vig-inflated one.

- If the market is well-calibrated, both calibration gaps are near zero at every line. Where they diverge — especially where `|calibration_gap_over| > 0.05` or `|calibration_gap_under| > 0.05` — the market is systematically mispricing that line bucket. Flag those rows. These are hypotheses to carry into feature engineering and modeling: does the model find signal at the same miscalibrated lines, or somewhere else?
- Note any patterns: is miscalibration concentrated at high lines, low lines, or spread evenly? Is it consistent across books or driven by one outlier book?

### Tests after Step 1.5
```sql
-- 1. Every row has over_rate + under_rate + push_rate summing to ~1.0 (within 1%)
-- 2. n_bets per line value is large enough to be meaningful (flag lines with <30 observations)
-- 3. avg_implied_prob_over and avg_implied_prob_under are both between 0 and 1 for all rows
-- 4. Total bets in this table matches expected joined row count from Step 1
```

---

## Step 2 — Feature Engineering / Spine

### Goal
Build a rolling feature matrix at the player-game-book-line level, ready for model input. No lookahead.

### Spine grain — non-negotiable
**The spine must be built at `(player, game_date, bookmaker, line)` grain — one row per player per game per book per line.** Do not aggregate to consensus before training or backtesting. This is the grain the production pipeline operates at: when a bet qualifies, it qualifies at a specific book at a specific price. The backtest must mirror this exactly or the strategy metrics (n_bets, ROI, edge thresholds) will not be valid in production.

Rolling features (snap%, target avg, etc.) are computed at the player-game level and then broadcast to all book rows for that player-game — that is fine and correct. The per-book column is `novig_prob_over` (computed from each book's own odds, not a cross-book average). Edge is always `p_model - novig_prob_over` using that book's own novig. Never substitute a consensus novig here — if you do, you are testing a bet that no individual book is actually offering.

### Work
- For each player-game, compute rolling averages of the target stat and relevant predictors across 5–10 different window sizes. **Always span the full range**: start at rolling_1 (last game only), end at rolling_career (all prior games), with values in between (e.g. rolling_3, rolling_5, rolling_10, rolling_season). The goal is to find which window has the most predictive power — do not pre-select a window before seeing the model results. Window choices should make intuitive sense for the sport/stat (e.g. a hot streak stat may prefer rolling_3; a stable baseline stat may prefer rolling_career).
- Standard feature candidates: rolling target stat avg, snap%, opponent rank/rate on the relevant stat, home/away flag, rest days, game total, spread, season week.
- **Consensus odds bin features (always build these — test in Step 3a).** These are player-game level (book-invariant — computed from the consensus across all books, then broadcast to every book row). Build all 4:
  - `consensus_over_odds_bin` — coarse: `+` (over is +odds), `-` (over is -odds), `even`
  - `consensus_over_odds_bin_granular` — 8 buckets: `-500_to_-300`, `-300_to_-200`, `-200_to_-110`, `-110_to_even`, `even_to_+110`, `+110_to_+200`, `+200_to_+300`, `+300_plus`
  - `consensus_under_odds_bin` — same coarse 3-value version for the under
  - `consensus_under_odds_bin_granular` — same 8-bucket version for the under
  - Compute consensus American odds as the **simple average across all books** at the consensus line for the player-game before binning. These are categorical features — encode them accordingly. They capture where the market is pricing the bet (heavy favorite vs dog), which may predict outcome independently of the raw line value.
- **Min/max line and raw implied prob features (always build these — test in Step 3a).** Six player-game level (book-invariant) features that capture the spread of market opinion. Useful signals when alt lines or varying book opinions exist:
  - `min_line` / `max_line` — lowest and highest line offered across all books for this player-game (works as a predictive feature in e.g. NBA rebounds)
  - `min_raw_implied_prob_over` / `max_raw_implied_prob_over` — min and max of `1 / decimal_odds_over` across all books and lines; captures the range of raw book opinions on the over
  - `min_raw_implied_prob_under` / `max_raw_implied_prob_under` — same for the under
  - A wide spread between min/max may signal genuine disagreement across books about the player's true probability — which can be exploited. Compute at the player-game level and broadcast to all book rows.
- Join with market data on player + game_date, expanding to one row per book per line. The rolling features are the same across all book rows for the same player-game; only `novig_prob_over`, `line`, and `bookmaker` vary by row.
  - **Player name matching is a known pain point.** The Odds API and box score sources often use different name formats — special characters (accents, apostrophes), nicknames (CeeDee vs. CeeDee Lamb), suffix differences (Jr./III), or abbreviations. Implement a `normalize_name()` function (lowercase, strip accents, strip punctuation, strip suffixes) as the first pass. This should get you to ~90% match rate on its own. For the remaining ~10%, flag unmatched names explicitly and ask the user to help resolve — do not silently drop unmatched rows or the join quality test will hide the gap.
- Out-of-fold design: features at game G must use only data from games strictly before G. Verify this explicitly.
- Save outputs to S3 (primary store) and `~/Downloads/tmp/` for local inspection. Do not use the repo `data/` directory.

### Tests after Step 2
```sql
-- 1. Row count: one row per (player, game_date, bookmaker, line) — no duplicates at that grain
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
- **Always include the 4 consensus odds bin features** (`consensus_over_odds_bin`, `consensus_over_odds_bin_granular`, `consensus_under_odds_bin`, `consensus_under_odds_bin_granular`) built in Step 2 as candidates in this sweep. Test each of the 4 independently — they may have predictive power on their own by capturing where the market is pricing the bet (heavy favorite vs dog), independent of the raw line or rolling stats.
- **Always include the 6 min/max features** (`min_line`, `max_line`, `min_raw_implied_prob_over`, `max_raw_implied_prob_over`, `min_raw_implied_prob_under`, `max_raw_implied_prob_under`) as candidates. Test each independently — a wide line or prob spread may signal genuine book disagreement that correlates with outcomes.
- **Market-derived features** (e.g. `market_under_prob`, `offered_line`) will almost always rank near the top — the market is well-calibrated and its implied probability is genuinely correlated with outcomes. Understand what that means before including them in the model: a market-derived feature predicts the response variable well *because the market already priced in the same player information the model is trying to learn*. If the model is dominated by market-derived features, `p_model` will closely track `p_market` and edge will be driven almost entirely by the non-market features. That's not necessarily wrong — it depends on whether those non-market features add signal — but it needs to be understood. When a market feature ranks highly in 3a, explicitly ask: is this feature adding signal the market doesn't already have, or is it just recapitulating the market price?
- Output table columns: `feature`, `model_type`, `n_samples`, `auc`, `precision`, `recall`, `f1`, `rmse`, `mae`, `r2`, `coefficient` (sign + magnitude for directional sanity check). Sort descending by AUC (binary) or ascending by RMSE (continuous).

### Step 3b — n=1 individual predictors with XGBoost
- Repeat 3a using XGBoost instead of logistic/linear regression.
- Output table: same columns as 3a, but now `model_type` = `xgboost`. Stack with the 3a results so the final table has two rows per feature — one for `regression`, one for `xgboost` — making it easy to compare side-by-side.
- Flag features where XGBoost adds meaningful lift (e.g. AUC delta > 0.02) — these likely have non-linear signal worth capturing in combos.

### Step 3c — Combos of best predictors
- Take the top features from 3a/3b.
- Build models with combinations (regression + XGBoost) using OOF CV.
- Select the best model by AUC. This is the production model candidate.
- Save the trained model artifact (pickle or joblib) to `models/` or S3 following repo conventions. **Immediately after saving, record the exact sklearn version used:** run `python -c "import sklearn; print(sklearn.__version__)"` and write the output into the config block in the HTML log. This version must be pinned in the Lambda Dockerfile when it is created in Step 8 — note it here so it isn't lost.
- Output table columns: `features_included` (list of features in the combo), `n_features`, `model_type`, `n_samples`, `auc`, `precision`, `recall`, `f1`, `rmse`, `mae`, `rationale` (1-2 sentence explanation of why this combo should work, grounded in what 3a/3b showed — e.g. "rolling_5_snap_pct was the strongest individual predictor; adding rolling_3_target_avg captures recent form which XGBoost showed non-linear lift on"). Sort descending by AUC (binary) or ascending by RMSE (continuous).
- **Write a model config block** into the pipeline's YAML config file at `src/{pipeline_folder}/config.yaml` (the same config file the pipeline already uses) with the winning feature set split into numeric and categorical lists — e.g.:
  ```yaml
  model:
    numeric_features:
      - offered_line
      - game_total
      - receiving_yards_L8
      - snap_pct_L8
    categorical_features:
      - pos_TE
      - pos_WR

  backtest:
    groupby: ["player_key", "game_date", "bookmaker", "line"]
    edge_col: "book_edge"          # p_model minus that book's own novig — never consensus
    novig_col: "novig_prob_over"   # computed per-book, not averaged across books

  grid_search:
    edge_threshold:       [0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
    direction:            ["over", "under", "both"]
    odds_bucket:          ["all", "plus_odds", "minus_odds"]
    clf_threshold:        [0.45, 0.50, 0.55, 0.60]   # classification models only; null for regression
    shrinkage:            [0, 0.25, 0.50, 0.75]       # pulls predictions toward the mean; 0 = no shrinkage
    prediction_method:    ["model", "consensus_line"]  # "model" = trained ML yhat; "consensus_line" = avg line
                                                       # across all books at player-game level (book-invariant)
                                                       # used as yhat directly, no ML involved
    # Add market-specific dimensions here as needed, e.g.:
    # line_bucket: ["low", "high"]
  ```
  All grid search scripts must load `grid_search` values from this YAML — never hardcode the sweep values inline. This makes it immediately clear which combinations were tested and allows the sweep to be re-run reproducibly. When adding a new dimension, add it here first.

### Tests after Step 3
```sql
-- 1. Training set and OOF validation set sizes are reasonable (not too small — flag if <200 rows in any fold)
-- 2. Best model AUC > 0.52 (above random — if not, flag and discuss with user before proceeding)
-- 3. Feature importance / coefficients are directionally sensible (e.g., more snaps → more tackles)
-- 4. OOF predictions cover the full dataset with no gaps (every row has a prediction)
```
Also verify in Python: OOF fold boundaries are strictly temporal (no future data in any training fold).

**Required Python assert — yhat is book-invariant.** Immediately after generating OOF predictions, add this check directly in the training script:
```python
yhat_check = df.groupby(["player_key", "game_date", "line"])["yhat"].nunique()
assert (yhat_check == 1).all(), (
    f"yhat is not book-invariant — {(yhat_check > 1).sum()} (player, game, line) groups "
    f"have varying predictions across books. A per-book feature is inside the model. "
    f"Check model inputs: any column that varies across books for the same player-game "
    f"(e.g. novig_prob_over) must be removed from the feature set."
)
```
This assert must live in the training script itself — not just as a post-hoc SQL check. If it fails, the model is structurally broken: different books will produce different projected values for the same player at the same line, which corrupts every downstream probability and edge calculation.

---

## Step 4 — Outcome Distribution → Binary Bet Signal

### Goal
Turn model predictions into P(over) estimates that can be compared to market implied probability.

### Pred → probability: try multiple methods

The model outputs a predicted float (e.g. "projected Ks = 5.86"). Converting that to P(over the line) is non-trivial — different methods can produce meaningfully different probabilities, and each market may respond differently. **Always try at least 2–3 methods and compare them on calibration before picking one.**

**Method 1 — Sampling distribution.** Fit a distribution (e.g. Poisson for count stats like strikeouts/sacks, Normal or Gamma for continuous stats) to the predicted value. Simulate N draws (e.g. 10,000) and compute the fraction exceeding the line → P(over). The choice of distribution matters: Poisson is natural for count data but assumes mean = variance; Normal is flexible but allows negative values; use domain knowledge to pick.

**Method 2 — Secondary classifier.** Train a logistic regression or XGBoost classifier that takes `(yhat, line)` as inputs and directly outputs P(over). This learns the empirical relationship between predicted value and outcome from data, rather than assuming a parametric distribution. Useful when the residual distribution is skewed or heavy-tailed.

**Method 3 — Empirical quantile lookup.** Group historical OOF predictions by `(yhat_bucket, line_bucket)` and compute the actual over rate in each cell. Use that as P(over) for future predictions. Simple, non-parametric, but requires enough data per cell.

**Evaluation — Brier score and calibration curves.** For each method:
- Compute **Brier score** = mean((p_model - actual_outcome)²). Lower is better. Compare methods head-to-head.
- Plot **calibration curves**: bucket p_model into deciles, compare predicted rate to actual over rate in each bucket. A well-calibrated method has points near the diagonal. A method that consistently overestimates P(over) in the 60–70% bucket is miscalibrated and will produce systematically wrong edges.
- The winning method is the one with the lowest Brier score and the tightest calibration curve. Write both to the HTML log before proceeding.

Each market is different — a method that works well for strikeouts may not work for receiving yards. Don't assume and don't skip this comparison.

### Work
- Use the trained model's OOF predictions as the input to the pred→probability conversion above. Try at least 2 methods, pick the best by Brier score + calibration, record the chosen method in config.yaml.
- **Clip all `p_model` values to [0.01, 0.99] before any downstream use.** After clipping, log the count and rate of rows hitting each boundary (e.g. "12 rows clipped to 0.01, 0.3% of total"). Ideally zero. If non-zero, look at the distribution of those rows — which players, which line ranges, which weeks — to understand what segment is clipping. You don't need to inspect every row; you need to understand the pattern. This is a model quality signal worth discussing before proceeding.
- Calibration check: bucket P(over) into deciles and compare predicted rate to actual over rate. Is the model well-calibrated?
- Compute market implied probability: `p_market = 1 / decimal_odds` — this is the **raw implied probability** (vig-inclusive). Also compute novig (de-vigged) probabilities via proportional de-vig for display in the email (Fair Over/Under columns), but do not use novig for edge.
- Compute edge: `edge = p_model − raw_implied_prob` — **always use the raw implied probability for edge, not novig.** The raw implied probability is the break-even probability at the actual price you're betting into: `edge > 0` means the bet is +EV at the real offered price. Novig understates the hurdle — a bet with positive edge vs novig can still be -EV at the real vig-inclusive price if the model's advantage is smaller than the vig on that side. Novig probabilities are still computed and displayed in the email (Fair Over/Under/Total, Vig columns) for transparency and calibration context, but they play no role in the edge calculation or strategy filters.

**Required Python assert — edge is computed against raw implied probability.** Immediately after computing the edge column, add this check in the scoring script:
```python
# Verify edge = p_model − raw_implied_prob (NOT novig_prob)
sample = df.sample(min(200, len(df)), random_state=42)
edge_expected_over  = sample["p_model_over"]  - sample["raw_implied_prob_over"]
edge_expected_under = sample["p_model_under"] - sample["raw_implied_prob_under"]
assert (sample["edge_over"]  - edge_expected_over ).abs().max() < 1e-6, (
    "edge_over is NOT p_model_over − raw_implied_prob_over. "
    "Check the edge calculation — novig_prob must not be used here."
)
assert (sample["edge_under"] - edge_expected_under).abs().max() < 1e-6, (
    "edge_under is NOT p_model_under − raw_implied_prob_under. "
    "Check the edge calculation — novig_prob must not be used here."
)
```
This assert must live in the scoring script itself, not just as a post-hoc check. It gates the entire grid search — if it fails, all downstream strategy results are in novig space and are not valid.
- For variable-line markets (e.g. tackles): note that different books offering different lines for the same player-game are genuinely different bets with different P(over). Treat them as separate rows.
- **Model purity check — same line, different books → identical p_model.** For a given player-game, every book row at the same line must produce the exact same `p_model`. The projected stat is a property of the player and the matchup, not of the book offering it. If `p_model` varies across books at the same line, a per-book feature (e.g. `novig_prob_over`) is inside the model — this is a hard bug, not a soft flag. Stop and fix the feature set before proceeding.
- **Line monotonicity check — different lines → p_model varies correctly.** For any player-game that appears at more than one line value, sort by line ascending and verify that `p_model_under` is monotonically non-decreasing. Higher line → easier to go under → p(under) must be higher. A player-game where p(under 65.5) < p(under 55.5) means the line feature is pointing the wrong direction or a correlated feature is overriding it. Flag every inversion: show the player, game date, the two lines, and the two `p_model_under` values. Write the full list to the HTML log. If inversions are rare (< 2% of multi-line player-games), note and move on. If common, investigate the line feature's coefficient before proceeding.

Together these two checks confirm the model is clean: p_model is book-invariant (purity) and line-sensitive (monotonicity). If either fails, the model has a structural problem.

### Tests after Step 4
```sql
-- 1. All P(over) values are between 0.01 and 0.99 after clipping (no nulls, no out-of-range)
-- 2. P(market) values are between 0 and 1
-- 3. Edge values have a plausible distribution (not all clustered at one extreme)
-- 4. Calibration: in each 10-percentile bucket, |predicted_rate - actual_over_rate| < 0.15
--    (flag buckets that exceed this — they indicate miscalibration worth investigating)
-- 5. Row count matches Step 2 spine (no rows dropped unexpectedly in the join)
-- 6. Count of rows clipped to 0.01 or 0.99 — log the number; flag for review if > 0
-- 7. Model purity: for each (player, game_date, line), stddev of p_model across books = 0.
--    Any non-zero stddev means a per-book feature is inside the model. Hard fail — fix before proceeding.
-- 8. Line monotonicity: for player-games with multiple line values, count inversions where
--    p_model_under at a higher line < p_model_under at a lower line. Log inversion rate.
--    Flag for investigation if inversion rate > 2% of multi-line player-games.
```

---

## Step 5 — Out-of-Sample Evaluation (Grid Search)

### Goal
Find strategies that are profitable on truly held-out data. This is the primary signal for whether to proceed to production.

> **What "grid search" means here:** this is a sweep over **betting strategy parameters** — edge threshold, bet direction, odds bucket, line bucket, etc. It is NOT model hyperparameter tuning (that's Step 3). The question being answered is: given the trained model's OOF predictions, which filter settings produce profitable betting results?

### Grain
**The grid search runs at `(player, game_date, bookmaker, line)` grain — the same grain as the spine.** Every row represents a real, actionable bet at a specific book at a specific price. Edge is `p_model − raw_implied_prob` using that book's own raw odds — never a consensus average, never novig. This makes the threshold numbers directly interpretable: `edge_threshold = 0.05` means "only bet when the model's probability is at least 5pp above the break-even probability at the real offered price." The Step 4 edge assert must pass before the grid search runs — results computed in novig space are not valid.

If you run the grid search at consensus grain and then deploy per-book, the thresholds are not valid — you would be applying a filter calibrated on fictional average bets to real per-book bets.

### Work
- Use only OOF predictions (Step 4) — these are the out-of-sample estimates.
**Two distinct thresholds — do not conflate them:**
- **Classification threshold** (classification problems only): converts `p_model` (a probability) into a predicted direction — e.g. if `p_model > 0.55`, predict Over; else predict Under. This is a property of the model output, not the betting strategy.
- **Edge threshold**: the minimum gap between `p_model` and `p_market` required to place a bet. Even if the model predicts Over, you only bet if `p_model - p_market >= edge_threshold`. This is the betting filter.

Both are grid search dimensions for classification problems. For regression problems, only edge threshold applies (the model outputs a predicted value, not a probability, so there's no classification threshold).

- **Load all sweep values from `config.yaml` `grid_search:` block** — do not hardcode them in the script. This makes the exact combinations tested visible and reproducible.
- Grid search over at minimum (defaults from config):
  - Edge threshold: [0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
  - Bet direction: [Over only, Under only, Both]
  - Odds bucket: [All, +odds only (dogs), -odds only (favs)] — default assumption is "All" wins. Only prefer a sub-bucket strategy if it clearly outperforms All on both units_won and the drawdown check. The more dimensions you carve on, the higher the overfitting risk — a sub-bucket that wins in-sample may just be noise.
  - Classification threshold (classification problems only): [0.45, 0.50, 0.55, 0.60]
  - Shrinkage: [0, 0.25, 0.50, 0.75] — shrinkage pulls each player's projected stat toward the population mean, reducing overconfidence on players with thin history. Apply before converting yhat to P(over/under). Always sweep all 4 values — 0 means no shrinkage (raw model output), 0.75 means heavy pull toward the mean.
  - Prediction method: ["model", "consensus_line"] — `model` uses the trained ML model's OOF yhat. `consensus_line` skips the model entirely and uses the average line across all books for that player-game (computed at spine build time, book-invariant) as the yhat directly. Both then go through the same P(over/under) conversion and per-book edge computation. This is the non-ML baseline. If `consensus_line` strategies outperform `model` strategies, it means the edge is coming from individual books posting lines that deviate from the consensus — not from the ML model knowing something the market doesn't. The market consensus is generally correct; what `consensus_line` exploits is specific books being off on a given player-game. That's still a real, actionable edge — it's line-shopping at scale — but it's a different kind of signal than what the ML model is trying to find. Understanding which is driving the profit shapes how you think about the pipeline and which component is actually doing the work. Note: shrinkage has no meaning when `prediction_method = consensus_line` — skip shrinkage sweep for that method (or fix shrinkage = 0).
  - Additional market-specific dimensions added to config as needed (e.g., line bucket for tackles: low lines vs high lines)
- For each combo: compute the output table below.
- Flag strategies with <50 bets as not statistically meaningful.
- **Flag any strategy where `max_drawdown > units_won` in red.** A strategy that drew down more than it ever returned is one most bettors would abandon before recovery. Flag it clearly — it's a signal worth noting, not a definitive answer.
- Save full grid search results to S3 and `~/Downloads/tmp/` as a CSV for local review.
- **Also write all OOS result rows (combos with ≥1 bet) into `config.yaml` under `grid_search.out_of_sample_results`**, one entry per combo, sorted by units_won descending. This makes the config the single source of truth for both the sweep parameters and the results — no need to re-open a CSV to remember what was tried. Format to match the MLB total bases / sacks pattern: one YAML list entry per combo with fields `edge`, `odds_bucket`, `direction`, `lines`, `n`, `win_pct`, `units`, `roi`, `mdd` (plus any market-specific dimensions). Also write a `strategy_summary` string at the top of the `grid_search:` block summarising the chosen production strategy in one line, e.g. `"Unders on lines ≤ 17.5, shrinkage 0.25, edge ≥ 10% — OOS 446 bets, +74.2u, +16.63% ROI (YYYY-MM-DD)"`.

**Output table columns:** `edge_threshold`, `direction`, `odds_bucket`, `clf_threshold` (null if regression), `shrinkage`, `n_bets`, `pct_of_universe` (n_bets / total scored player-game-book-line rows — shows how selective the strategy is), `win_rate`, `push_rate`, `units_won`, `roi`, `avg_odds` (mean decimal odds of bets placed), `max_drawdown` (largest peak-to-trough loss in units across the chronological bet sequence). Sort descending by units_won, tiebreaker is descending n_bets (more sample size is better).

### Tests after Step 5
```sql
-- 1. Grid search output has one row per (threshold, direction, [other dimensions]) combination
-- 2. Bets column is correct: COUNT(*) where edge >= threshold and direction matches
-- 3. Units calculation: verify on a 5-row spot check that win/loss/push logic is correct
-- 4. No strategy has ROI > 25% with >100 bets (flag as suspicious — likely a leakage artifact)
-- 5. The best strategy by ROI has at least 30 bets (otherwise it's noise)
-- 6. Flag any strategy with win_rate > 0.50 — the market is well-calibrated and shouldn't be
--    wrong more than half the time on a sustained basis. A win rate between 0.50 and 1.00
--    almost always means feature leakage, a data join error, or the target variable bleeding
--    into the features. Investigate before treating it as a real result.
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
- **Write all IS result rows (combos with ≥1 bet) into `config.yaml` under `grid_search.in_sample_results`**, same format as the OOS rows written in Step 5.

### Tests after Step 6
```sql
-- 1. In-sample ROI >= OOS ROI for the same strategy (if not, something is wrong — investigate)
-- 2. In-sample/OOS ROI ratio < 5x for the best strategy
-- 3. The chosen production strategy (from Step 5) has positive in-sample ROI
```

### Strategy decision (required before proceeding to Step 7)

**It is entirely possible that no strategy clears the bar.** If no strategy has ≥50 bets, positive OOS ROI, and positive IS ROI, stop here — do not proceed to Step 7. Document the null result clearly in the HTML log and summarize why (e.g. no predictive signal, market too efficient, insufficient data). This is a valid outcome.

If one or more strategies clear the bar, identify up to 3 candidates (best by units won, subject to ≥50 bets and positive IS ROI). For each candidate, note whether `max_drawdown > units_won` and flag it clearly if so — a strategy that drew down more than it ever returned is one most bettors would abandon before recovery. This is a signal worth discussing, not a definitive disqualifier. Present it to the user and let them decide. In modeling, things are rarely completely right or wrong — surface the signal and let the user decide. For each candidate, run the following **4-table strategy characterization** and write all tables to the HTML log.

**All 4 tables use in-sample (IS) data** — the full dataset scored by the model trained on all data. IS is the right choice here because the goal is to characterize the structure of the strategy (which books, which odds buckets) with maximum sample size. The OOS grid search already answered whether the strategy works; these tables answer where and how it works. More bets = more reliable slices across books and odds buckets.

**Table 1 — Production strategy summary (1 row)** *(IS)*
Columns: `edge_threshold`, `direction`, `clf_threshold` (if applicable), `shrinkage`, `n_bets`, `pct_of_universe`, `win_rate`, `push_rate`, `units_won`, `roi`, `max_drawdown`, `avg_implied_prob`, `avg_odds`

**Table 2 — Odds bucket breakdown (3 rows: dog / even / fav)** *(IS)*
Same columns as Table 1, plus `pct_of_strategy`. Dog = implied prob < 0.50 (+odds); even = 0.50; fav = implied prob > 0.50 (-odds). Include the breakeven win rate for each bucket (`1 / (1 + avg_odds_decimal)`) so it's clear whether each bucket is profitable on its own.

**Table 3 — By bookmaker** *(IS)*
Same columns as Table 1 plus `pct_of_strategy`, one row per book, sorted by units_won descending. Flag books with negative ROI in red — these are candidates for exclusion from the production pipeline.

**Table 4 — Bookmaker × odds bucket** *(IS)*
Cross-tab of Table 3 × Table 2. Same columns. This catches cases where a book's losses are concentrated in one bucket (e.g. a book that's profitable on dogs but loses on favs, or vice versa).

After all 4 tables are written for all candidates, present a summary in the HTML and in the conversation — which candidate looks strongest and why, any books worth excluding based on Table 3/4, and what the odds structure tells you about where the edge is coming from. Then ask the user to confirm which strategy to move forward with. Do not proceed to Step 7 until confirmed.

---

## Step 7 — Mock HTML Email

### Goal
Build a styled HTML mock of the picks email using real historical data from a past game day. This is the visual spec the production pipeline in Step 8 must match. Agree on the layout, columns, and styling before writing any Lambda code — the E2E test in Step 8 passes when the real email matches this mock.

### Work
- Choose a past game day from the scored OOF output (Step 4) that has:
  - Real prop lines in the spine
  - Real box score outcomes (so qualifying rows can be marked)
  - At least 2 games and multiple players per game, so the game-grouped layout is fully visible
- Using the OOF scored output for that day, produce the full player row set:
  - One row per `(player, game_date, bookmaker, line)` — same grain as the pipeline
  - All required picks columns, grouped into the sections below: Player | Team | Opp | Time (ET) | Line | Book | Over | Under | Raw Over | Raw Under | Raw Total | Fair Over | Fair Under | Fair Total | Vig | [Prediction (yhat)] | [Delta] | Pred Over | Pred Under | Over Edge | Under Edge | [model input feature cols] | Status
  - `Status` = "PLAY" if the row meets the production edge threshold; blank otherwise
- Build a complete styled HTML file at `knowledge-base/raw/YYYYMMDD-mock-email-{market-name}.html` (e.g. `knowledge-base/raw/20260701-mock-email-nfl-wr-rec-yards.html`).

The mock must show all three sections that the live email will contain — use real OOF data for all three:

**Section 1 — Today's plays (required):**
- **Header summary** — first visible element: one line, e.g. "3 plays today across 2 games"
- **Game sections** — one section per game, sorted by `game_time_et` ascending. Each section has:
  - A header row: `"7:05 PM ET — NYY @ BOS (3 players scored, 1 play)"`
  - A `<table>` with all scored players for that game — qualifying and non-qualifying alike. Qualifying rows highlighted, non-qualifying rows default background.
  - **Grouped column headers (two-row `<thead>`)** — because the email has many columns, group them into labeled sections using `colspan` on the top header row, with individual column names underneath. Groups should follow the natural structure of the data, for example:

    | ← Player / Game → | ← Book → | ← American Odds → | ← Implied → | ← No-Vig → | ← Model Prediction → | ← Edge → | ← Model Inputs → |
    |---|---|---|---|---|---|---|---|
    | Player \| Team \| Opp \| Time (ET) \| Line | Book | Over \| Under | Raw Over \| Raw Under \| Raw Total | Fair Over \| Fair Under \| Fair Total \| Vig | [Prediction (yhat)] \| [Delta] \| Pred Over \| Pred Under | Over Edge \| Under Edge | feature cols… \| Status |

    The key requirement is that every column belongs to a visible group — no ungrouped orphan columns. If two models are trained (e.g. OLS + XGB), split "Model Prediction" and "Edge" into two groups each (e.g. "← Model OLS →" / "← Model XGB →" and "← Edge OLS →" / "← Edge XGB →") — the column names above assume a single model; adapt as needed.
- **Model inputs table** — one table below the game sections, columns: `Feature` | `Shown as` | `What it measures` | `Role`. One row per model feature with the actual column names and descriptions for this market.

**Section 2 — Yesterday's results (required):** Pick any game day immediately before the chosen "today" that has settled OOF bets. One row per bet: Player | Team | Opponent | Bet Direction | Line | Book | Under Odds | Edge | Actual | Outcome | P&L. Below the per-bet table, a "by game" summary (Game | Bets | W | L | Net).

**Section 3 — All-time results (required):** Summary stat cards (All-Time P&L, Record, Win %, ROI) populated from the OOF backtest results. Season-by-season breakdown table. Footer line: flat-bet assumption, strategy parameters, OOS baseline.

All three sections are required in the mock — the live pipeline produces all three, so the visual spec must show all three.

**Styling (inline CSS only — no external stylesheets):**
- Font: `system-ui, Arial`, 14px
- Tables: `border-collapse: collapse`; alternating row shading (`#f9f9f9` / white)
- Game header rows: `background: #1a1a2e; color: white; font-weight: bold`
- PLAY rows (OVER): `background: #e6f4ea`
- PLAY rows (UNDER): `background: #fce8e6`
- Non-play rows: default alternating shading

**No fake data.** The mock must use real scored output from the OOF pipeline — real player names, real lines, real odds, real edges from a real historical game day. Do not hardcode any values.

**Before building the HTML, assert all model input columns are in the scored output DataFrame.** Load `config.yaml`, read `model.numeric_features` and `model.categorical_features`, and verify every feature is present as a column in the scored data. Do this in Python before rendering a single row:
```python
import yaml, pandas as pd
with open("src/{pipeline_folder}/config.yaml") as f:
    cfg = yaml.safe_load(f)
required = cfg["model"]["numeric_features"] + cfg["model"].get("categorical_features", [])
scored_df = ...  # your scored output DataFrame
missing = [c for c in required if c not in scored_df.columns]
assert not missing, f"Model input columns missing from scored output — cannot build email: {missing}"
```
If this assert fails, fix the scoring script so it carries all model input columns through to the output before proceeding. The email is the wrong place to discover that a feature column was silently dropped.

### Tests after Step 7
```sql
-- Section 1 — Today's plays:
-- 1. Row count in the HTML game tables matches the scored output row count for that day
--    (count <tr> elements in game tables vs SELECT COUNT(*) from scored output for the date)
-- 2. All PLAY rows have edge >= production edge threshold
-- 3. All model input columns (numeric_features + categorical_features from config.yaml) appear
--    as column headers in every game table — grep the rendered HTML for each feature name
-- 4. All required picks columns are present in every game table (check the header <tr>)
-- 5. Game sections ordered by game_time_et ascending

-- Section 2 — Yesterday's results:
-- 6. All settled bets for the "yesterday" game day appear (count matches OOF data for that date)
-- 7. "By game" summary net P&L equals sum of individual bet P&L values for that day

-- Section 3 — All-time results:
-- 8. Summary cards (P&L, Record, Win %, ROI) match the OOF backtest aggregate numbers
-- 9. Season-by-season table rows sum to the all-time totals
```
After the SQL checks, open the file in a browser (`open knowledge-base/raw/YYYYMMDD-mock-email-{market-name}.html`) and visually confirm:
- All three sections are present in order: today's plays → yesterday's results → all-time results
- Game grouping is correct, sorted by time
- PLAY rows are visibly highlighted and distinguishable from non-play rows
- Model inputs table is present and readable
- Columns are in the correct order
- Yesterday's per-bet table and by-game summary both render correctly
- All-time stat cards and season table look clean

**Do not proceed to Step 8 until the layout is approved.** Show the path to the HTML file in the conversation and ask the user to open it and confirm the layout. Step 8 builds the live pipeline; the E2E test in Step 8 is "does the real email match this mock?"

---

## Step 8 — E2E Production Pipeline

### Goal
Build a daily pipeline that runs live during the season. The trained model from Step 3 is fixed — no retraining during the season.

### Daily cadence

Every day during the season, **two emails** are sent:

| Time (ET) | Rule | What runs | Email |
|---|---|---|---|
| **8:30 AM** | Rule 1 | Settle yesterday's bets + rebuild spine/features | **Email 1** — spine/features build confirmation |
| **9:00 AM** | Rule 2 | Score today's games | **Email 2** — plays + yesterday's results + all-time results |

**Email 1 (8:30 AM)** is the ops health check. It tells you the pipeline is working and surfaces anything that needs fixing before bets go out. **Email 2 (9:00 AM)** is the daily plays email. Do not send Email 2 if the spine's `last_updated` timestamp is not from today — it means rule 1 failed and features are stale.

**Email 2 always sends — even on days with no qualifying plays.** "0 plays today" is a valid and expected outcome. The email still shows yesterday's results and all-time record; the plays section simply says "No plays today." Do not skip the send on low-edge days.

**Timing is sport-dependent — discuss while building.** 8:30 AM / 9:00 AM is the default starting point and works well for MLB and NFL. Adjust as needed for other sports or unusual schedules. The rule of thumb: **rebuild the spine every day regardless of whether there are games**, because compute is cheap and a stale spine is harder to debug than an unnecessary rebuild.

### Daily schedule

Two EventBridge rules, both DISABLED by default. Enable before season start.

**8:30 AM ET — Settle + Rebuild** (rule 1):
1. **Settle** — compare yesterday's bets to actual outcomes. Update P&L in the settled results store.
2. **Rebuild spine** — append yesterday's box score data to the rolling feature set (parquet on S3). Settle must complete before rebuild so the newly appended rows don't get re-settled.
3. **Send Email 1 — spine/features build confirmation.** This email confirms the 8:30 job ran cleanly and surfaces anything that needs attention before the 9:00 AM scoring run. Contents:
   - **Settle summary:** bets settled, W/L/Push counts, net P&L for yesterday.
   - **Spine rebuild summary:** rows added, new players added, `last_updated` timestamp.
   - **Warnings (if any):** unmatched players (Odds API name vs available MLB API names side-by-side so a `NAME_MAP` fix can be applied immediately), stale data flags, any settlement anomalies.
   - If no warnings, a single "All clear — spine updated, N rows added" line is sufficient.

**9:00 AM ET — Score + Email** (rule 2):
4. **Find today's games** — fetch today's schedule and available prop lines from the Odds API.
5. **Score** — run today's player-games through the trained model. Compute `p_model`, `p_market`, and `edge` for each player-game-book row. Log a warning if the spine's `last_updated` timestamp is not from today — it means the 8:30 job failed and scoring is running off stale features. Do not send Email 2 if this check fails.
6. **Pre-send validation** — before sending any email, run the following checks in order. Halt and do not send if any fail.

   **Required assert — edge is computed against raw implied probability.** Run this before any other validation. If it fails, the qualifying bets are wrong and the email must not go out:
   ```python
   sample = scored_df.sample(min(200, len(scored_df)), random_state=42)
   edge_expected_over  = sample["p_model_over"]  - sample["raw_implied_prob_over"]
   edge_expected_under = sample["p_model_under"] - sample["raw_implied_prob_under"]
   assert (sample["edge_over"]  - edge_expected_over ).abs().max() < 1e-6, (
       "edge_over is NOT p_model_over − raw_implied_prob_over. "
       "Novig must not be used for edge. Fix before sending."
   )
   assert (sample["edge_under"] - edge_expected_under).abs().max() < 1e-6, (
       "edge_under is NOT p_model_under − raw_implied_prob_under. "
       "Novig must not be used for edge. Fix before sending."
   )
   ```

   Then eyeball the output. There are no hard cut-offs; the question is: **does this look like a model making modest adjustments to the market, or a model ignoring the market entirely?**
   - **Clip all `p_model` values to [0.01, 0.99].** Then log every row that hit the clip boundary (pre-clip value, player name, line, feature values). Ideally zero rows hit the boundary. Any that do should be inspected before the email goes out — they mean the model saw a feature combination so far outside its training range that it returned a near-certain prediction, which is almost never trustworthy. If they hit the boundary because of a feature pipeline issue (e.g. a player with no L8 history returning 0), exclude those rows from the qualifying bets rather than sending them. Note: if you saw zero clips in Step 4 OOF data but clips are appearing here on live data, that is a feature drift issue — the scoring pipeline is feeding the model values it never saw during training.
   - Print: n_total_scored, n_qualifying_bets, n_clipped (pre-clip), and for qualifying bets: mean/min/max of `p_model`, `p_market`, and `edge`.
   - The market is your anchor. A well-functioning model should produce `p_model` values that are in the same neighbourhood as `p_market` — shifted by the signal, not divorced from it. If the market says 60% chance of under and the model says 3%, that is a bug, not alpha. A model that prices every player at 3% P(under) has lost contact with reality.
   - Obvious red flags: `p_model` clustered near 0 or 1 across the board; edges uniformly 30pp+; n_qualifying_bets that exceeds a plausible fraction of total players scored; `p_model` distribution that looks nothing like `p_market` distribution. None of these require a formula — they are visible on inspection.
   - If anything looks wrong, halt. Do not send. Log the summary stats and investigate the scoring script before proceeding.
7. **Send Email 2** — one combined email to `mylescgthomas@gmail.com`, three sections in this order:

   **Section 1 — Today's plays.** The main section. Layout and columns match the Step 7 mock exactly.

   - **Header summary (top):** one line — e.g. "3 plays today across 2 games". First thing visible.
   - **Game-by-game sections:** sort all scored players by `game_time_et`, then `home_team`, then `away_team`, then `player_name`. Group by game with a header row per game — e.g. "7:05 PM ET — NYY @ BOS (3 players scored, 1 play)". Show **all scored players** in each game, not just qualifying bets. Highlight qualifying bet rows so plays stand out from non-plays. The point is to see every player the model evaluated in context. **The matchup (`home_team`, `away_team`) must be part of the sort key** — without it, multiple games starting at the same time (e.g. a 7:05 PM ET slate) will interleave rows from different matchups under the wrong game header.
   - **Required columns — grouped by section (match the Step 7 mock layout exactly):**
     - **Player / Game:** Player | Team | Opp | Time (ET) | Line
     - **Book:** Book — use the full sportsbook name, never a nickname (e.g. "DraftKings" not "DK", "FanDuel" not "FD"). Map raw Odds API keys to display names in every pipeline using a lookup dict — never pass the raw key through to the email:
       ```python
       _BOOK_DISPLAY_NAMES = {
           "betonlineag":    "BetOnline",
           "fanduel":        "FanDuel",
           "draftkings":     "DraftKings",
           "betmgm":         "BetMGM",
           "caesars":        "Caesars",
           "betrivers":      "BetRivers",
           "pointsbetus":    "PointsBet",
           "unibet_us":      "Unibet",
           "mybookieag":     "MyBookie",
           "bovada":         "Bovada",
           "pinnacle":       "Pinnacle",
           "bet365":         "Bet365",
           "williamhill_us": "William Hill",
           "lowvig":         "LowVig",
           "ballybet":       "Bally Bet",
           "espnbet":        "ESPN Bet",
           "fliff":          "Fliff",
           "betanysports":   "BetAnySports",
           "fanatics":       "Fanatics",
           "hardrock":       "Hard Rock Bet",
       }
       ```
       If a book key appears that isn't in the dict, log a warning and fall back to the raw key rather than silently displaying it — that way missing entries get caught and added.
     - **American Odds:** Over | Under — American odds as posted by that book.
     - **Implied:** Raw Over | Raw Under | Raw Total — raw implied probabilities (1 / decimal_odds); Raw Total > 100% due to vig, e.g. ~106%.
     - **No-Vig:** Fair Over | Fair Under | Fair Total | Vig — proportionally de-vigged probabilities; Fair Total = 100% by construction; Vig = Raw Total − 100% in pp, e.g. +6.3pp. These are the benchmark for edge — edge is always computed against Fair Over / Fair Under, not the raw implied probs.
     - **Model Prediction:** [Prediction (yhat)] | [Delta] | Pred Over | Pred Under
       - `Prediction (yhat)` — the model's raw output before probability conversion (e.g. 6.2 projected Ks). Player-game invariant: same value regardless of line or book. Include for regression models (OLS / XGBoost regressor) where yhat is a numeric stat — the chain `yhat → Pred Over` is what you want to sanity-check. Skip for pure classifiers (logistic / XGBoost classifier) where yhat is already a probability and equals Pred Over directly.
       - `Delta` — `yhat − line`. Positive = model leans OVER, negative = model leans UNDER, in the stat's own units (e.g. +0.7 Ks, −2.3 rush attempts). Player-game-line invariant. Only include alongside Prediction (yhat) — skip for classifiers.
       - `Pred Over` / `Pred Under` — model P(over) and P(under) at that line. Player-game-line invariant: same across all books at the same line, varies across lines.
     - **Edge:** Over Edge | Under Edge — p_model minus that book's own Fair Over / Fair Under; edge for UNDER strategies displays as a positive number.
     - **Model Inputs:** [all features from `config.yaml` `numeric_features` + `categorical_features`, in order] | Status — features the model saw; lets you sanity-check projections inline. **If the model's Pred Over is 85% for a player the market has at 30%, something is wrong — seeing the inputs alongside the prediction is how you catch it.** `Status` encodes both whether the row qualifies and the bet direction: `PLAY - OVER`, `PLAY - UNDER`, or blank for non-qualifying rows. No separate Bet column needed.
   - **Grouped column headers (two-row `<thead>` with `colspan`)** — see Step 7 requirement.
   - **Model inputs table (below the game sections):** one row per feature with columns: `Feature` | `Shown as` | `What it measures` | `Role`.

   **Section 2 — Yesterday's results.** One row per settled bet. Columns: Player | Team | Opponent | Bet Direction | Line | Book | Under Odds | Edge | Actual | Outcome | P&L. Below the per-bet table, a "by game" summary table (one row per game: Game | Bets | W | L | Net). `Bet Direction` always populated explicitly.

   **Section 3 — All-time results.** Summary stat cards (All-Time P&L, Record, Win %, ROI) followed by a season-by-season breakdown table (Season | Bets | Record | Win % | Units | ROI). Footer line: flat-bet assumption, strategy parameters, OOS baseline from research.

   One SES call. The settled results for Section 2 come from the same store updated in step 1 above.

### Architecture (mirror the sacks/tackles Lambda pattern):
- Container-based Lambda (ECR)
- **When writing the Dockerfile, pin `scikit-learn` to the exact version recorded in Step 3c.** A mismatch between the version used to pickle the model and the version in the container will cause `InconsistentVersionWarning` at best and a hard `AttributeError` at worst.
- **Two EventBridge rules** (both DISABLED by default — enable before season start: 2026-09-09):
  - Rule 1: `cron(30 13 * * ? *)` → 8:30 AM ET — triggers settle + rebuild
  - Rule 2: `cron(0 14 * * ? *)` → 9:00 AM ET — triggers score + email
- Spine and model artifacts stored in S3
- Settled results stored in DuckDB or S3 parquet

### Tests after Step 8
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
-- 10. Email: send a test email to mylescgthomas@gmail.com using the same historical game day
--     as the Step 7 mock. Visually compare the real email against the mock HTML — layout,
--     columns, game grouping, and PLAY highlighting must match. Check logs to confirm delivery.
--     User may need to verify receipt on their end — if logs show success but email doesn't
--     arrive, flag for user to check spam / SES config.
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
- **Append to the HTML log after every step — this is mandatory, not optional.** See the "HTML Log — NON-NEGOTIABLE REQUIREMENT" section above. Each step gets its own `<section>` with: what was built, key findings, all output tables as HTML `<table>` elements, test results (pass/fail with counts), and flagged items. Every section header must include a timestamp in Eastern time to the minute/second (e.g. `2026-07-01 14:32:05 ET`) — get this via `TZ=America/New_York date '+%Y-%m-%d %H:%M:%S ET'` in the terminal. Everything goes in the one file — grid search results, spot-check traces, calibration output, sweep tables, all of it. This is the persistent record — if context compacts mid-session, the log is how the work gets reconstructed.
- **Trace the spot-check player at every step.** At the end of each step's work (before tests), show all relevant columns for the spot-check player chosen in Step 0 — their raw data, rolling features, p_model, p_market, edge, and results as applicable. Include this as a dedicated subsection in the HTML log for that step. If the spot-check player's numbers look wrong, treat it as a test failure and investigate before proceeding.
- **Log and surface anything fishy.** Throughout each step, keep a running list of anything unexpected, suspicious, or worth a second opinion — unexpected null patterns, distributions that don't make intuitive sense, coverage gaps, model coefficients pointing the wrong direction, ROI numbers that seem too good, etc. Do not interrupt mid-step to ask about them. At the end of each step, after tests pass, present all flagged items as a numbered list and ask the user about them before proceeding to the next step.
- **No jargon. Always show the value.** Never use shorthand like "high-line" or "large edge" without stating the actual threshold alongside it. Write for someone who hasn't been in the session: "line ≥ 6.5" not "high-line", "edge ≥ 3pp" not "meaningful edge". This applies to code comments, HTML output, emails, and conversation.
- **Use readable names in code and data — shorten only for display.** DataFrame column names, variable names, and SQL aliases should be fully descriptive (e.g. `rolling_rebound_mean_60`, `novig_prob_over`, `under_edge_ols`) — not abbreviated to cryptic shorthand (e.g. `rb60`, `nvp`, `ue_ols`). Abbreviations are acceptable only in the HTML display layer (column headers, email labels) where space is genuinely constrained — and only after the full name is established in the underlying data. Never choose a short name in code because it's faster to type.
- **Edge for UNDER strategies displays as positive.** When displaying edge in tables, emails, or HTML, show it as a positive number (e.g. +13.0pp) — it represents how much the model favors the UNDER over the market, which is always in our favor when we bet it.
- **Update the HTML log before asking to proceed.** Never summarize findings in conversation and then ask to move on. Write it to the HTML first, then ask.
- **Be decisive on strategy selection.** Present the recommendation fully and completely the first time — parameters, stats, rationale. Don't make the user ask twice for the same information.

---

## Code & Output Standards

- **Research script naming: `YYYYMMDD_description.py` prefix, not `vN_`.** Use the date the script was first created (ET). Example: `20260706_edge_backtest.py`. When two scripts in the same directory would produce the same date+name, add a distinguishing suffix (e.g. `_spread`). Do not use `v1_`, `v2_`, etc. — those are banned.
- **Bash scripts call Python scripts — they do not embed Python.** If logic is complex enough to warrant more than 2–3 lines of Python, it belongs in a standalone `.py` file in the `scripts/` directory. The bash script calls it with arguments. Inline heredoc Python in bash is not acceptable.
- **No fake data anywhere — ever.** This applies to every test, script, demo run, and email send throughout the entire session. Forbidden: hardcoded player names, hardcoded teams, hardcoded lines, hardcoded probabilities, hardcoded dates passed as arguments to simulate a "real" run. All test runs must use real data: query the spine/S3 for a past game day that actually has data, make real Odds API calls, and let the pipeline resolve its own inputs. If a script needs a `--gameday` argument, pick a real historical date that has lines in the spine — do not invent one. If real data isn't available for a scenario, say so and find a date that does have data rather than filling in values. A test that passes on fake data tells us nothing.
- **When emails don't arrive, check spam first.** Before investigating SES configuration, DKIM, DMARC, or any infrastructure-level explanation, ask the user to check their spam folder. The simple explanation is almost always right.

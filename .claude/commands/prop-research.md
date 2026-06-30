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
- What sportsbooks? (default: all available)
- Is this a binary market (ie. QB interceptions thrown), a skewed binary line market (99%+ at one value like sacks) or a variable/numeric line market (like tackles)? This determines which ML models will be most useful, and which strategies make the most sense.
  - In general, humans like to take fun bets ie. rooting for success (OVERS), and popular teams (the favorites). So, typically in betting markets, there is value in being contrarian here in the betting markets.

Write these down as a config block at the start of the session and reference them throughout.
- for a new project, everything will be contained in ~/dev/betting/src/{name of new project}

---

## Step 1 — Data Pull + EDA

### Goal
Understand the market and candidate features before building anything.

### Work
**Market data (prop lines):**
- Pull prop lines and odds for the target market using the Odds API config (`config/the-odds-api_config.yaml` has canonical market key strings — always check before guessing).
- Coverage check by season, week, and book: how many games have lines? Which weeks/books are missing?
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
- For each player-game, compute rolling averages of the target stat and relevant predictors: last N games (try N=3, 5, 10), season-to-date, opponent-adjusted where feasible.
- Standard feature candidates: rolling target stat avg, snap%, opponent rank/rate on the relevant stat, home/away flag, rest days, game total, spread, season week.
- Join with market data on player + game_date. The join key should be at the player-game level (not player-game-book yet — that comes at inference time).
- Out-of-fold design: features at game G must use only data from games strictly before G. Verify this explicitly.
- Save to parquet (follow the `data/` directory conventions already in the repo).

### Tests after Step 2
```sql
-- 1. Row count: one row per player-game in the training window (no duplicates)
-- 2. No future leakage: for a spot-checked player, rolling_avg at game G < game G's actual value
--    (i.e., game G's actual result is not included in the rolling feature for game G)
-- 3. Null rates by column: flag any feature column with >10% nulls
-- 4. Target column (actual outcome) is present and has no nulls
-- 5. Join quality: what % of market rows have a matching feature row? (flag if <90%)
-- 6. Date range covers expected training seasons
```

---

## Step 3 — Model Training

Work through the three sub-steps in order. Accuracy metrics only at this stage — no ROI, no units. Metrics: AUC, precision, recall, F1, RMSE/MAE (where applicable).

### Step 3a — n=1 individual predictors (regression / interpretable models)
- For each candidate feature independently, train a logistic regression (binary target) or linear regression (continuous target) using OOF cross-validation to prevent leakage.
- Report AUC, precision, recall for each feature individually.
- Rank features by predictive power. Identify the top performers.

### Step 3b — n=1 individual predictors with XGBoost
- Repeat 3a using XGBoost instead of logistic/linear regression.
- Compare vs regression for each feature. Flag where XGBoost adds meaningful lift.

### Step 3c — Combos of best predictors
- Take the top features from 3a/3b.
- Build models with combinations (regression + XGBoost) using OOF CV.
- Select the best model by AUC. This is the production model candidate.
- Save the trained model artifact (pickle or joblib) to `models/` or S3 following repo conventions.

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
- Grid search over at minimum:
  - Edge threshold: [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
  - Bet direction: [Over only, Under only, Both]
  - Additional dimensions if the market warrants it (e.g., line bucket for tackles: low lines vs high lines)
- For each combo: number of bets, win rate, units won/lost (at -110 equiv), ROI.
- Identify the best-performing strategies. Note that strategies with <50 bets are not statistically meaningful — flag these.
- Save the full grid search results to `data/` as a CSV.

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
- This will show inflated ROI vs OOS — that's expected.
- Key question: does the best OOS strategy from Step 5 also show positive in-sample ROI? If a strategy is positive OOS but flat/negative in-sample, that's a red flag.
- Compare in-sample vs out-of-sample ROI ratio. >5x gap suggests the model is overfit.

### Tests after Step 6
```sql
-- 1. In-sample ROI >= OOS ROI for the same strategy (if not, something is wrong — investigate)
-- 2. In-sample/OOS ROI ratio < 5x for the best strategy
-- 3. The chosen production strategy (from Step 5) has positive in-sample ROI
```

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

-- E2E pipeline test:
-- 9. Run the full pipeline on 2 historical game days in sequence:
--    Day 1: settle (no prior bets) → rebuild → score → verify output
--    Day 2: settle Day 1's output → rebuild → score → verify P&L updated correctly
-- 10. Email template renders with real data (send a test email to mylescgthomas@gmail.com)
```

After all tests pass, verify Lambda deployment:
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
- **Accuracy metrics in Steps 3–4, financial metrics in Steps 5–6.** Do not evaluate ROI before Step 5.
- **Bet size is not modeled here.** Flat betting (1 unit per bet) is assumed throughout.

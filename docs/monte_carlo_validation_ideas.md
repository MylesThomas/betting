# Monte Carlo Validation Ideas

## Context
We need ways to validate that our Monte Carlo simulation is producing accurate probability predictions for player prop bets.

## Key Challenge
**We can't just compare to the pace line** - that assumes linear scoring, which isn't realistic. Players have:
- Quarter-specific scoring patterns
- Variable minutes (blowouts, rest)
- Hot/cold streaks
- Different roles in different games

## Validation Approaches

### 1. Calibration Analysis (Brier Score)
**What it measures:** Are our probabilities well-calibrated?

**Method:**
- Bucket all predictions by probability (0-10%, 10-20%, ..., 90-100%)
- For each bucket, calculate the actual hit rate
- Compare predicted probability vs actual hit rate
- **Good model:** 50% predictions should hit ~50% of the time

**Implementation:**
```python
# For all games, group by probability buckets
for prob_bucket in range(0, 100, 10):
    games_in_bucket = df[(df['prob_over'] >= prob_bucket) & (df['prob_over'] < prob_bucket + 10)]
    actual_hit_rate = games_in_bucket['result'].mean()
    expected_prob = prob_bucket + 5  # midpoint
    
    print(f"{prob_bucket}-{prob_bucket+10}%: Predicted {expected_prob}%, Actual {actual_hit_rate*100:.1f}%")
```

**Metric:** Brier Score = mean((predicted_prob - actual_outcome)^2)
- Lower is better
- 0 = perfect, 0.25 = random guessing


### 2. Time-Series Coherence
**What it measures:** Do probabilities change sensibly as the game progresses?

**Checks:**
- ✅ Probability should start near 50% (Vegas adjustment)
- ✅ Probability should increase when player scores
- ✅ Probability should decrease when time passes with no points
- ✅ Probability should be ~100% when they go over the line
- ✅ Probability should be ~0% when mathematically impossible

**Red flags:**
- Probability jumps erratically
- Probability decreases when player scores
- Probability stays flat despite game progression


### 3. Cross-Validation by Quarter
**What it measures:** Does the model work equally well in all quarters?

**Method:**
- Split validation by quarter (Q1, Q2, Q3, Q4)
- Calculate calibration separately for each quarter
- Check if model is biased in certain quarters

**Example:**
```sql
-- Check hit rates by quarter when model says 50-60%
SELECT 
    quarter,
    AVG(CASE WHEN actual_points > prop_line THEN 1 ELSE 0 END) as actual_hit_rate,
    COUNT(*) as n_predictions
FROM monte_carlo_results
WHERE prob_over BETWEEN 0.5 AND 0.6
GROUP BY quarter
```


### 4. Profit Simulation (Kelly Criterion)
**What it measures:** If we bet using the model, would we make money?

**Method:**
- For each game, compare our MC probability to Vegas line
- If MC says 60% but line implies 50%, we have +EV
- Simulate betting strategy:
  - Only bet when edge > 5%
  - Size bets using Kelly Criterion
  - Track ROI over all games

**Red flags:**
- Negative ROI (model is worse than market)
- ROI only from a few games (overfitting)


### 5. Comparison to Naive Baselines
**What it measures:** Is our complex model actually better than simple heuristics?

**Baselines to beat:**
- **Baseline 1:** Always predict player's season average
- **Baseline 2:** Linear pace (points / minutes * 48)
- **Baseline 3:** Current PPM extrapolated (current_points / minutes_elapsed * 48)

**Method:**
```python
# At each minute, compare MC prob vs naive baselines
mc_prob = monte_carlo_probability
pace_prob = 1 if (current_points / minutes_elapsed * 48) > prop_line else 0
avg_prob = 1 if player_season_avg > prop_line else 0

# Track Brier score for each
mc_brier = mean((mc_prob - actual)^2)
pace_brier = mean((pace_prob - actual)^2)
avg_brier = mean((avg_prob - actual)^2)
```


### 6. Variance Check (MC Stability)
**What it measures:** Are our simulations stable with enough iterations?

**Method:**
- Run MC with different random seeds
- Compare probability distributions
- Should be stable with n_sims >= 1000

**Test:**
```python
probs = []
for seed in range(10):
    random.seed(seed)
    prob = monte_carlo(...)
    probs.append(prob)

std_dev = np.std(probs)
print(f"Std dev across seeds: {std_dev:.3f}")
# Should be < 0.01 for stable predictions
```


### 7. Edge Cases Validation
**What it measures:** Does model handle extreme scenarios correctly?

**Test cases:**
- Player at 0 points with 1 minute left, need 30 points → prob should be ~0%
- Player at 29 points with 1 minute left, need 30 points → prob should be moderate (depends on PPM)
- Player at 35 points with 10 minutes left, need 30 points → prob should be ~100%
- Player has 0 points at halftime → prob should drop but not to 0% (comeback possible)


### 8. Historical Backtest
**What it measures:** How would the model have performed on past games?

**Method:**
- Run MC on all historical games
- For each game, save probabilities at key moments:
  - Start of game
  - End of Q1, Q2, Q3
  - Final 5 minutes
- Calculate calibration and Brier score
- Compare to Vegas closing line accuracy


## Recommended Priority

1. **Start with:** Calibration Analysis (Brier Score) - easiest and most important
2. **Then:** Time-Series Coherence - visual sanity check
3. **Then:** Edge Cases - catch bugs
4. **Finally:** Profit Simulation - ultimate test

## Implementation Plan

```bash
# 1. Combine all play-by-play CSVs into one file
duckdb -c "
    COPY (
        SELECT * FROM read_csv_auto('src/pbp_data/tmp/plots/monte_carlo_pbp_*.csv')
    ) TO 'analysis/monte_carlo_validation_data.parquet' (FORMAT PARQUET)
"

# 2. Create validation script
python analysis/validate_monte_carlo.py \\
    --data analysis/monte_carlo_validation_data.parquet \\
    --output analysis/monte_carlo_validation_report.html
```

## Key Metrics to Track

| Metric | Good | Acceptable | Bad |
|--------|------|------------|-----|
| Brier Score | < 0.15 | 0.15-0.20 | > 0.20 |
| Calibration Error | < 5% | 5-10% | > 10% |
| ROI (if betting) | > 5% | 0-5% | < 0% |
| Prob Stability (std) | < 0.01 | 0.01-0.02 | > 0.02 |


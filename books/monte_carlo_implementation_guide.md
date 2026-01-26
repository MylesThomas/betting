# Monte Carlo Prediction Models - Implementation Guide

**Source:** Personal notes from "Monte Carlo or Bust: Simple Simulations for Aspiring Sports Bettors"

## Overview

This guide extracts the key methodologies from the book for implementing prediction models for sports betting.

## 1. Dixon-Coles Model (Soccer/Football)

### Core Concept
Model the number of goals each team will score using attack/defense strengths vs league averages, then use Poisson distribution for probabilities.

### Formulas

**Attack Strength:**
```
Team Attack Strength = Team Mean xG / League Mean xG
(calculated separately for home and away)
```

**Defense Strength:**
```
Team Defense Strength = Team Mean xG Conceded / League Mean xG Conceded  
(calculated separately for home and away)
```

**Expected Goals:**
```
Home Team xG = Home Attack × Away Defense × League Home Mean xG
Away Team xG = Away Attack × Home Defense × League Away Mean xG
```

### Poisson Distribution
- Use `POISSON(x, μ, FALSE)` to get probability of exactly x goals
- Where μ = expected goals from Dixon-Coles
- Calculate probabilities for 0-6 goals per team
- Use multiplication rule for scoreline probabilities
- Use addition rule for win/draw/loss probabilities

### Match Outcome Probabilities
```
Home Win = Sum of all scorelines where home > away
Draw = Sum of (0-0, 1-1, 2-2, 3-3, 4-4, 5-5, 6-6)
Away Win = Sum of all scorelines where away > home
```

**Convert to odds:**
```
Fair Odds = 1 / Probability
```

## 2. NBA Model (Basketball)

### Key Differences from Soccer
- Points are **normally distributed** (not Poisson)
- Much higher scoring
- No draws
- Home court advantage exists but is weaker than soccer

### Formulas

**Expected Points:**
```
Team Expected Points = (Team Points Per Game / League PPG) × Opponent Defensive Rating × League PPG
(similar structure to Dixon-Coles but for points)
```

**Normal Distribution:**
- Use `NORM.DIST(x, μ, σ, TRUE)` for probabilities
- μ = expected points
- σ = standard deviation of team's scoring

**Win Probability (shortcut method):**
```
Points Difference ~ Normal(μ_diff, σ_diff)
μ_diff = Team A Expected - Team B Expected
σ_diff = √(σ_A² + σ_B²)

P(Team A wins) = 1 - NORM.DIST(0, μ_diff, σ_diff, TRUE)
```

## 3. NFL Model (Football)

### Approach
Similar to soccer - points are closer to Poisson distribution than normal.

### Key Metrics
- EPA (Expected Points Added) as equivalent to xG
- Offensive/Defensive efficiency
- Home field advantage (~2.5 points)

### Model Structure
Same as Dixon-Coles but with:
- Points instead of goals
- EPA instead of xG
- Poisson distribution for point totals (0-60+)

## 4. Monte Carlo Simulation

### Purpose
When calculating exact probabilities is too complex (e.g., season simulations with 380+ games).

### Process
1. **Use random number generator** - `RAND()` in Excel
2. **Invert distribution function** - `POISSON_INV(RAND(), μ)` or `NORM.INV(RAND(), μ, σ)`
3. **Simulate outcomes** - Run 10,000+ iterations
4. **Calculate probabilities** - Count outcomes / total iterations

### Example Use Cases
- Season-long league table predictions
- Playoff qualification chances
- Championship probabilities
- Series outcomes (best-of-7)

## 5. Key Concepts

### Expected Value (EV)
```
Expected Value = (Bookmaker Odds / Model Fair Odds) - 1
Positive EV = value bet
```

### Removing Bookmaker Margin
Book discusses logarithmic function method:
```
Fair probability = (Unfair probability)^exponent
Where exponent solves: sum of fair probabilities = 1.0
```

### Favorite-Longshot Bias
- Bookmakers apply larger margins to longshots
- Favorites often have better value than odds suggest
- Important when comparing model to bookmaker odds

## 6. Data Requirements

### Minimum Data Needed
- Historical game results (scores, home/away)
- Team performance metrics (points scored/allowed)
- League averages for the relevant time period

### Enhanced Data
- Advanced metrics (xG, EPA, efficiency ratings)
- Recent form weighting
- Injury reports
- Rest days/travel

## 7. Model Validation

### Critical Points
- **One result proves nothing** - need 100s or 1000s of bets
- Compare model outputs to closing odds (most efficient)
- Track actual results vs expected over large sample
- Be alert to systematic errors in assumptions

### Statistical Testing
- Use Monte Carlo to generate distribution of expected outcomes
- Compare actual outcome to expected distribution
- Statistical significance if actual falls in <5% tail of distribution

## 8. Python Implementation Notes

### Libraries Needed
```python
import numpy as np
from scipy.stats import poisson, norm
import pandas as pd
```

### Key Functions
- `poisson.pmf(k, mu)` - Poisson probability mass function
- `norm.pdf(x, loc, scale)` - Normal probability density
- `norm.cdf(x, loc, scale)` - Normal cumulative distribution
- `np.random.poisson(lam)` - Random Poisson sample
- `np.random.normal(loc, scale)` - Random normal sample

## Next Steps for Implementation

1. Choose sport (NBA, NFL, or both)
2. Gather historical data
3. Calculate team strength metrics
4. Implement Dixon-Coles or Normal distribution model
5. Build Monte Carlo simulator
6. Compare outputs to bookmaker odds
7. Backtest on historical data
8. Identify +EV opportunities

---

**Full book chapter reference:** `books/monte_carlo_or_bust.md`

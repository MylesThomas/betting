# Monte Carlo Prediction Models - Reference Notes

**Personal notes from book chapters for implementation purposes**

## Dixon-Coles Model (Soccer → Adaptable to Other Sports)

### Attack/Defense Strength Calculation
- Attack Strength = Team Mean Metric / League Mean Metric
- Defense Strength = Team Mean Conceded / League Mean Conceded  
- Calculate separately for home/away

### Expected Goals/Points
- Home Expected = Home Attack × Away Defense × League Home Mean
- Away Expected = Away Attack × Home Defense × League Away Mean

### Probability Distributions
- **Soccer/NFL**: Poisson distribution for scoring
- **NBA**: Normal distribution for points

### Monte Carlo Simulation Process
1. Calculate expected outcome (goals/points)
2. Use random number + inverse distribution function
3. Simulate 10,000+ iterations
4. Count outcomes to determine probabilities

### Key Implementation Functions (Python)
```python
from scipy.stats import poisson, norm
import numpy as np

# Poisson (for soccer/NFL)
poisson.pmf(k, mu)  # probability of exactly k goals
np.random.poisson(mu, size)  # random samples

# Normal (for NBA)  
norm.cdf(x, loc, scale)  # cumulative probability
np.random.normal(loc, scale, size)  # random samples
```

### Finding Value
```python
expected_value = (bookmaker_odds / model_fair_odds) - 1
# Bet if expected_value > 0
```

---

**Full raw transcription available at:** `books/monte_carlo_or_bust.md` (for detailed reference)

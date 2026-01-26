# Monte Carlo Prediction Models - Implementation Strategy

**Source:** Personal notes from Monte Carlo or Bust by Joseph Buchdahl  
**Purpose:** Implementation guide for building prediction models for NBA/NFL betting

---

## Executive Summary

This guide extracts the core methodologies from the book for building statistical prediction models that can identify +EV betting opportunities. The approach uses attack/defense strength metrics, probability distributions, and Monte Carlo simulations to estimate "true" odds and compare them against bookmaker lines.

**Key Insight:** If we can build a model that accurately predicts game outcomes, we can identify when bookmakers have mispriced markets and bet with positive expected value.

---

## Core Methodology: The Dixon-Coles Model

### Overview
The Dixon-Coles model (1997, Journal of Applied Statistics) calculates the probability of match outcomes by:
1. Measuring team attack/defense strength relative to league averages
2. Calculating expected scoring (goals/points) for each team
3. Using probability distributions to model all possible outcomes
4. Comparing model odds to bookmaker odds to find value

### Key Principle
**Team strength is relative, not absolute**. A team's performance is measured against the league average, accounting for home/away splits.

---

## Step 1: Calculate Team Strength Metrics

### Attack Strength Formula
```
Attack Strength = (Team Mean Scoring) / (League Mean Scoring)
```

**Calculate separately for home and away:**
- Home Attack Strength = Team Home Scoring / League Home Avg
- Away Attack Strength = Team Away Scoring / League Away Avg

**Example (from book - Premier League soccer):**
- League home mean xG: 455.94 / 288 games = 1.583 xG
- Aston Villa home xG: 17.96 / 13 games = 1.382 xG
- Aston Villa home attack strength: 1.382 / 1.583 = 0.873

**Higher number = stronger attack**

### Defense Strength Formula
```
Defense Strength = (Team Mean Conceded) / (League Mean Conceded)
```

**Key insight:** League mean conceded is the inverse of league mean scored:
- League home mean conceded = League away mean scored
- League away mean conceded = League home mean scored

**Example:**
- League home mean conceded: 1.337 xG (= away team scoring avg)
- Aston Villa home conceded: 27.04 / 13 games = 2.08 xG
- Aston Villa home defense strength: 2.08 / 1.337 = 1.556

**Lower number = stronger defense**

---

## Step 2: Calculate Expected Scoring

### Formula
```
Home Team Expected = Home Attack × Away Defense × League Home Mean
Away Team Expected = Away Attack × Home Defense × League Away Mean
```

**Example (Aston Villa vs Sheffield United):**
```
Aston Villa xG = 0.873 × 0.915 × 1.583 = 1.264 xG
Sheffield United xG = 0.767 × 1.556 × 1.337 = 1.596 xG
```

This tells us: if this game could be played infinite times, Villa would average 1.264 goals, Sheffield United 1.596 goals.

---

## Step 3: Model Outcome Probabilities

### For Soccer/NFL: Poisson Distribution

**Why Poisson?** Goals/points scored in lower-scoring sports follow a Poisson distribution where:
- Events occur independently
- Events occur at a constant average rate
- Mean = Variance

**Python Implementation:**
```python
from scipy.stats import poisson

def calculate_poisson_probabilities(expected_goals, max_goals=6):
    """
    Calculate probability of scoring 0-max_goals
    
    Args:
        expected_goals: Expected goals from Dixon-Coles (μ)
        max_goals: Maximum goals to calculate (default 6)
    
    Returns:
        Dictionary of {goals: probability}
    """
    probabilities = {}
    for k in range(max_goals + 1):
        probabilities[k] = poisson.pmf(k, expected_goals)
    return probabilities

# Example
home_probs = calculate_poisson_probabilities(1.264)  # Aston Villa
away_probs = calculate_poisson_probabilities(1.596)  # Sheffield United

# home_probs = {0: 0.2825, 1: 0.3571, 2: 0.2257, 3: 0.0951, ...}
```

### Calculate Match Outcome Probabilities

**Use multiplication rule for specific scorelines:**
```python
def calculate_match_probabilities(home_probs, away_probs):
    """
    Calculate probabilities for all scorelines and match outcomes
    
    Returns:
        dict with 'home_win', 'draw', 'away_win' probabilities
    """
    home_win = 0
    draw = 0
    away_win = 0
    
    for home_goals, home_prob in home_probs.items():
        for away_goals, away_prob in away_probs.items():
            scoreline_prob = home_prob * away_prob
            
            if home_goals > away_goals:
                home_win += scoreline_prob
            elif home_goals == away_goals:
                draw += scoreline_prob
            else:
                away_win += scoreline_prob
    
    return {
        'home_win': home_win,
        'draw': draw,
        'away_win': away_win
    }
```

**Book Example Results:**
- Home Win (Aston Villa): 30.24%
- Draw: 24.59%
- Away Win (Sheffield United): 45.00%

### Convert to Fair Odds

```python
def probability_to_odds(probability):
    """Convert probability to decimal odds"""
    return 1 / probability

# Example
fair_odds = {
    'home': probability_to_odds(0.3024),  # 3.31
    'draw': probability_to_odds(0.2459),  # 4.07
    'away': probability_to_odds(0.4500),  # 2.22
}
```

---

## Step 4: For NBA - Normal Distribution Approach

### Why Normal for NBA?
Basketball has much higher scoring, and points follow a **normal distribution** rather than Poisson.

**Key differences:**
- Need mean (μ) AND standard deviation (σ)
- Use `scipy.stats.norm` instead of `poisson`
- No draws - only need win probability

### NBA Expected Points Formula
Same structure as Dixon-Coles but with points instead of goals:

```python
def calculate_nba_expected_points(team_stats, opponent_stats, league_stats, is_home):
    """
    Calculate expected points for NBA game
    
    Args:
        team_stats: dict with 'off_rating', 'def_rating'
        opponent_stats: dict with 'off_rating', 'def_rating'
        league_stats: dict with 'home_avg_points', 'away_avg_points'
        is_home: bool
    
    Returns:
        Expected points for the team
    """
    if is_home:
        # Home offense vs opponent away defense
        attack_strength = team_stats['off_rating'] / league_stats['league_off_rating']
        defense_opponent = opponent_stats['def_rating_away'] / league_stats['league_def_rating_away']
        expected = attack_strength * defense_opponent * league_stats['home_avg_points']
    else:
        # Away offense vs opponent home defense
        attack_strength = team_stats['off_rating'] / league_stats['league_off_rating']
        defense_opponent = opponent_stats['def_rating_home'] / league_stats['league_def_rating_home']
        expected = attack_strength * defense_opponent * league_stats['away_avg_points']
    
    return expected
```

### Calculate Win Probability (Normal Distribution)

**Method 1: Monte Carlo Simulation**
```python
import numpy as np
from scipy.stats import norm

def simulate_nba_game_mc(home_exp, home_std, away_exp, away_std, n_sims=10000):
    """
    Simulate NBA game using Monte Carlo
    
    Args:
        home_exp: Home team expected points (μ)
        home_std: Home team standard deviation (σ)
        away_exp: Away team expected points
        away_std: Away team standard deviation
        n_sims: Number of simulations
    
    Returns:
        Probability of home win
    """
    home_scores = np.random.normal(home_exp, home_std, n_sims)
    away_scores = np.random.normal(away_exp, away_std, n_sims)
    
    home_wins = np.sum(home_scores > away_scores)
    return home_wins / n_sims
```

**Method 2: Analytical (Shortcut)**

The book provides a mathematical shortcut for normal distributions:

```python
def calculate_nba_win_prob_analytical(home_exp, home_std, away_exp, away_std):
    """
    Calculate win probability using normal distribution mathematics
    
    Points difference is also normally distributed with:
    - Mean = μ_home - μ_away
    - Std Dev = √(σ_home² + σ_away²)
    
    P(Home wins) = P(Difference > 0)
    """
    diff_mean = home_exp - away_exp
    diff_std = np.sqrt(home_std**2 + away_std**2)
    
    # P(X > 0) = 1 - P(X <= 0) = 1 - CDF(0)
    home_win_prob = 1 - norm.cdf(0, diff_mean, diff_std)
    
    return home_win_prob
```

**Book Example (NBA Finals - Toronto vs Golden State):**
- Toronto home: 111.34 expected, std dev 11.15
- Golden State away: 111.61 expected, std dev 12.31
- Difference: -0.27 mean, 16.67 std dev
- Toronto win prob: 49.44% (essentially a coin flip)

---

## Step 5: Monte Carlo Simulation for Complex Scenarios

### When to Use Monte Carlo

Use Monte Carlo when:
1. **Too many combinations** to calculate exactly (e.g., full season with 380+ games)
2. **Series outcomes** (best-of-7 playoffs)
3. **Tournament simulations**
4. **Complex probability trees**

### Monte Carlo Process

```python
def monte_carlo_season_simulation(team_strengths, schedule, n_iterations=10000):
    """
    Simulate full season using Monte Carlo
    
    Args:
        team_strengths: Dict of team metrics
        schedule: List of (home_team, away_team) tuples
        n_iterations: Number of season simulations
    
    Returns:
        Dictionary with finishing probabilities
    """
    results = {team: [] for team in team_strengths}
    
    for iteration in range(n_iterations):
        # Initialize points for this iteration
        points = {team: 0 for team in team_strengths}
        
        # Simulate each game
        for home_team, away_team in schedule:
            # Calculate expected scoring
            home_exp = calculate_expected_score(home_team, away_team, is_home=True)
            away_exp = calculate_expected_score(away_team, home_team, is_home=False)
            
            # Use inverse Poisson/Normal to get random outcome
            # For Poisson (soccer/NFL):
            home_score = poisson.rvs(home_exp)
            away_score = poisson.rvs(away_exp)
            
            # For Normal (NBA):
            # home_score = norm.rvs(home_exp, home_std)
            # away_score = norm.rvs(away_exp, away_std)
            
            # Award points
            if home_score > away_score:
                points[home_team] += 3  # Win
            elif home_score == away_score:
                points[home_team] += 1  # Draw
                points[away_team] += 1
            else:
                points[away_team] += 3  # Away win
        
        # Store results
        for team, team_points in points.items():
            results[team].append(team_points)
    
    # Calculate probabilities
    probabilities = {}
    for team, point_totals in results.items():
        probabilities[team] = {
            'mean_points': np.mean(point_totals),
            'median_points': np.median(point_totals),
            'std_dev': np.std(point_totals)
        }
    
    return probabilities
```

**Book Example:** 
- Premier League 2019/20: 92 remaining games = 3^92 possible outcomes (too many to calculate)
- Used 10,000 Monte Carlo iterations instead
- Predicted finishing points within ±3 points for 12 of 20 teams

---

## Step 6: Finding Value (Expected Value)

### The Key Formula

```python
def calculate_expected_value(bookmaker_odds, model_fair_odds):
    """
    Calculate expected value of a bet
    
    Args:
        bookmaker_odds: Odds offered by bookmaker (decimal)
        model_fair_odds: Fair odds from your model (decimal)
    
    Returns:
        Expected value as percentage
    
    Positive EV = value bet (place bet)
    Negative EV = bad bet (avoid)
    """
    expected_value = (bookmaker_odds / model_fair_odds) - 1
    return expected_value

# Example from book
# Aston Villa fair odds: 3.31 (from model)
# Bookmaker odds: 3.50
ev = calculate_expected_value(3.50, 3.31)  # 0.0574 or 5.74% EV

# Sheffield United fair odds: 2.22
# Bookmaker odds: 2.43
ev = calculate_expected_value(2.43, 2.22)  # 0.0946 or 9.46% EV
```

**Critical Insight from Book:**
> "We aren't interested in finding winners—although it goes without saying that it's nice to win—we are interested in finding value."

One result tells you nothing. You need 100s or 1000s of bets to validate a model.

---

## Step 7: Removing Bookmaker Margin

### Why This Matters

Bookmaker odds include a profit margin (vig/overround). Before comparing to your model, you must remove this margin to get "fair" probabilities.

**Problem:** Bookmaker probabilities sum to > 100%

**Example:**
- Home: 2.00 (50%)
- Draw: 3.00 (33.3%)
- Away: 4.00 (25%)
- Total: 108.3% (8.3% margin)

### Favorite-Longshot Bias

**Key Discovery:** Bookmakers don't apply margin evenly. They apply MORE margin to longshots than favorites.

**Why?**
1. Bettors overestimate likelihood of rare events
2. Greater uncertainty in pricing longshots
3. Higher potential liability for bookmakers

### Logarithmic Margin Removal (Book's Preferred Method)

```python
def remove_margin_logarithmic(odds_dict):
    """
    Remove bookmaker margin using logarithmic method
    
    This accounts for favorite-longshot bias by applying
    margin proportional to the odds.
    
    Args:
        odds_dict: Dict of {outcome: decimal_odds}
    
    Returns:
        Dict of {outcome: fair_odds}
    """
    # Convert odds to implied probabilities
    implied_probs = {k: 1/v for k, v in odds_dict.items()}
    total_prob = sum(implied_probs.values())
    
    if total_prob <= 1.0:
        return odds_dict  # Already fair
    
    # Find exponent that makes probabilities sum to 1
    # Using iterative approach
    from scipy.optimize import fsolve
    
    def equation(exponent):
        fair_probs = {k: p**exponent for k, p in implied_probs.items()}
        return sum(fair_probs.values()) - 1.0
    
    exponent = fsolve(equation, 0.95)[0]
    
    # Apply exponent to get fair probabilities
    fair_probs = {k: p**exponent for k, p in implied_probs.items()}
    
    # Convert back to odds
    fair_odds = {k: 1/p for k, p in fair_probs.items()}
    
    return fair_odds
```

---

## Implementation Roadmap for NBA/NFL

### Phase 1: Data Collection (Week 1-2)

**What you need:**

1. **Historical game results:**
   - Date, Home Team, Away Team, Final Score
   - Minimum: Current season + 1-2 previous seasons

2. **Team statistics:**
   - **NBA:** Offensive Rating, Defensive Rating, Pace
   - **NFL:** Points Per Game, Points Allowed, EPA (if available)
   
3. **Bookmaker odds (critical):**
   - Opening odds and closing odds
   - Moneyline, spread, totals
   - Multiple bookmakers if possible (for CLV analysis)

**Data sources:**
- the-odds-api (you're already using)
- NBA: basketball-reference.com, nba.com/stats
- NFL: pro-football-reference.com

### Phase 2: Build Basic Model (Week 3-4)

**Step-by-step:**

1. **Calculate league averages:**
```python
def calculate_league_averages(games_df):
    """Calculate home/away league averages"""
    home_avg = games_df['home_score'].mean()
    away_avg = games_df['away_score'].mean()
    return {'home': home_avg, 'away': away_avg}
```

2. **Calculate team strengths:**
```python
def calculate_team_strengths(games_df, team, league_avg):
    """Calculate attack/defense strength for a team"""
    # Filter home games
    home_games = games_df[games_df['home_team'] == team]
    home_scored = home_games['home_score'].mean()
    home_allowed = home_games['away_score'].mean()
    
    # Filter away games
    away_games = games_df[games_df['away_team'] == team]
    away_scored = away_games['away_score'].mean()
    away_allowed = away_games['home_score'].mean()
    
    return {
        'home_attack': home_scored / league_avg['home'],
        'home_defense': home_allowed / league_avg['away'],
        'away_attack': away_scored / league_avg['away'],
        'away_defense': away_allowed / league_avg['home']
    }
```

3. **Predict single game:**
```python
def predict_game(home_team, away_team, team_strengths, league_avg):
    """Predict game outcome"""
    # Calculate expected points
    home_exp = (
        team_strengths[home_team]['home_attack'] * 
        team_strengths[away_team]['away_defense'] * 
        league_avg['home']
    )
    away_exp = (
        team_strengths[away_team]['away_attack'] * 
        team_strengths[home_team]['home_defense'] * 
        league_avg['away']
    )
    
    # For NBA: Use normal distribution
    home_std = 12.0  # Estimate from historical data
    away_std = 12.0
    
    win_prob = calculate_nba_win_prob_analytical(
        home_exp, home_std, away_exp, away_std
    )
    
    fair_odds = {
        'home': 1 / win_prob,
        'away': 1 / (1 - win_prob)
    }
    
    return fair_odds
```

4. **Compare to bookmaker:**
```python
def find_value_bets(fair_odds, bookmaker_odds, threshold=0.05):
    """Find bets with positive EV above threshold"""
    value_bets = []
    
    for outcome in ['home', 'away']:
        ev = calculate_expected_value(
            bookmaker_odds[outcome],
            fair_odds[outcome]
        )
        
        if ev > threshold:
            value_bets.append({
                'outcome': outcome,
                'ev': ev,
                'fair_odds': fair_odds[outcome],
                'bookmaker_odds': bookmaker_odds[outcome]
            })
    
    return value_bets
```

### Phase 3: Validation (Week 5-6)

**Backtest on historical data:**

```python
def backtest_model(games_df, team_strengths_func, league_avg):
    """
    Test model on historical games
    
    Returns accuracy metrics and Brier score
    """
    predictions = []
    actuals = []
    
    for idx, game in games_df.iterrows():
        # Calculate expected outcome
        fair_odds = predict_game(
            game['home_team'], 
            game['away_team'],
            team_strengths_func(games_df[:idx]),  # Only use past data
            league_avg
        )
        
        home_win_prob = 1 / fair_odds['home']
        predictions.append(home_win_prob)
        actuals.append(1 if game['home_score'] > game['away_score'] else 0)
    
    # Calculate Brier score (lower is better)
    brier_score = np.mean((np.array(predictions) - np.array(actuals))**2)
    
    return brier_score
```

**Key Validation Metrics:**
1. **Brier Score:** Measures calibration of probabilities (should be < 0.20)
2. **ROI:** Return on investment if betting all +EV opportunities
3. **CLV:** Closing Line Value (are your fair odds better than opening odds?)

### Phase 4: Advanced Enhancements

**Improvements to consider:**

1. **Time-weighted strength:**
   - Recent games weighted more heavily
   - Dixon-Coles used exponential decay

2. **Rest days:**
   - Back-to-back games in NBA
   - Short rest in NFL

3. **Injuries:**
   - Adjust team strength for missing players

4. **Travel:**
   - Cross-country games in NBA

5. **Pace adjustment (NBA):**
   - Faster teams increase variance
   - Adjust expected totals by pace factor

6. **Strength of schedule:**
   - Teams may be under/overrated based on opponents faced

---

## Critical Lessons from the Book

### 1. Single Results Mean Nothing

> "To know whether this model was reliable and accurate, we would need to use it over hundreds, indeed probably thousands of matches, yes, really thousands."

**Don't judge your model by:**
- One winning bet
- One losing bet
- Even 10 or 20 bets

**Do judge your model by:**
- 500+ bet sample
- Consistent positive EV
- Brier score on thousands of predictions

### 2. Model Assumptions Matter

**Potential flaws to watch for:**
- **Independence assumption:** Are outcomes truly independent? (Players have emotions/memory)
- **Distribution fit:** Does Poisson/Normal actually fit your sport?
- **Sample size:** Too few games = noise; too many games = stale data
- **Home advantage:** Has it changed? (COVID example from book)

### 3. You Need a Better Model Than Bookmakers

> "If the model you are using is already reflected in the odds, because other people are using it too, it won't be good enough to help you overcome the bookmakers' margins."

**Dixon-Coles is well-known.** Bookmakers use it. You need to add something they don't have:
- Better data (advanced metrics)
- Faster updates (real-time injury news)
- Better understanding of your teams
- Unique insights (player matchups, etc.)

### 4. Closing Odds Are Most Efficient

The book emphasizes: **Closing odds contain the most information**

Why?
- Maximum money has been bet
- Maximum information has been incorporated
- Sharpest bettors have moved the line

**Implication:** If your model beats closing odds consistently, you're onto something valuable.

---

## Quick Start Checklist

- [ ] Collect 2-3 seasons of game results
- [ ] Calculate league home/away averages
- [ ] Build team strength calculator (attack/defense)
- [ ] Implement expected scoring formula
- [ ] Choose distribution (Poisson for NFL, Normal for NBA)
- [ ] Build probability calculator
- [ ] Get current bookmaker odds
- [ ] Remove bookmaker margin
- [ ] Calculate expected value
- [ ] Backtest on historical data (500+ games)
- [ ] Track bets and validate model performance
- [ ] Iterate and improve

---

## File Structure for Implementation

```
/analysis/
├── monte_carlo_models/
│   ├── __init__.py
│   ├── dixon_coles.py          # Core model
│   ├── distributions.py        # Poisson/Normal helpers
│   ├── margin_removal.py       # Remove bookmaker vig
│   ├── monte_carlo.py          # Season simulations
│   └── value_finder.py         # EV calculator
│
├── data/
│   ├── nba_games.csv           # Historical games
│   ├── nfl_games.csv
│   └── bookmaker_odds.csv      # Odds data
│
└── scripts/
    ├── calculate_team_strengths.py
    ├── predict_today_games.py
    ├── backtest_model.py
    └── find_value_bets.py
```

---

## Next Steps

1. **Start simple:** Basic Dixon-Coles on NBA/NFL moneylines
2. **Validate thoroughly:** Backtest on 500+ games before betting real money
3. **Track everything:** Log all predictions and outcomes
4. **Iterate:** Add enhancements one at a time, test each addition
5. **Be patient:** Model validation takes time (months, not days)

**Remember:** The goal isn't to predict winners. The goal is to find mispriced markets where our model disagrees with the bookmaker, and we can demonstrate we're right over a large sample.

---

**Full chapter reference:** `books/monte_carlo_or_bust.md` (raw OCR transcription)

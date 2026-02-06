# Monte Carlo Simulation Variables for Bet Prediction

## Problem Statement
Predict the probability of a bet hitting (e.g., "Over 30.5 points") given:
- Player's historical performance
- Current game state (score so far, time remaining)
- Game context (opponent, home/away, etc.)

## Core Philosophy
Rather than assuming normal distribution, use the **actual empirical distribution** of player performance and simulate remaining game scenarios.

---

## Variable Categories

### 1. Historical Performance Data (Distribution Inputs)

#### Player Statistics History
- **`points_per_game_history`** (list): All games this season/last N games
  - Better than just mean/std: captures actual distribution shape
  - Can be skewed (e.g., star players occasionally explode for 50+)
  - Handles multi-modal distributions (e.g., player has "good games" and "bad games")

- **`minutes_per_game_history`** (list): Actual minutes played each game
  - Critical for estimating remaining playing time
  - Accounts for blowouts, foul trouble, DNP-Rest, etc.

- **`points_per_minute_history`** (list): Efficiency metric
  - `points / minutes` for each game
  - More stable for projecting with irregular minutes
  - Useful for calculating: `pred_remaining_points = pred_remaining_minutes * sample(points_per_minute_history)`

#### Quarter/Half Splits
- **`first_half_points_history`** (list): Points scored in 1H of each game
- **`second_half_points_history`** (list): Points scored in 2H of each game
- **`first_half_minutes_history`** (list): Minutes played in 1H
- **`second_half_minutes_history`** (list): Minutes played in 2H

**Why this matters:** 
- Starters typically play more consistent 1H minutes
- 2H is more variable (blowouts, foul trouble, rest)
- Luka example: if he has 0 at halftime, his 2H distribution is more relevant than full-game distribution

#### Quarter-by-Quarter (Optional, more granular)
- **`q1_points_history`**, **`q2_points_history`**, **`q3_points_history`**, **`q4_points_history`**
- Allows modeling specific quarter patterns
- Some players are "4th quarter players"

---

### 2. Current Game State

#### Points Accumulated
- **`current_points`** (int): Points scored so far in THIS game
- **`points_needed`** (float): Target line - current_points
  - Example: Player has 12 pts, line is 30.5, needs 18.5 more

#### Time Information
- **`time_remaining_in_game`** (float): Minutes left in game (48 min for NBA)
- **`current_quarter`** (int): 1, 2, 3, 4, or OT
- **`time_remaining_in_quarter`** (float): Minutes left in current quarter
- **`minutes_played_so_far`** (float): How much they've played already

#### Predicted Remaining Playing Time
- **`pred_remaining_minutes_to_play`** (float): Estimated minutes they'll play rest of game
  
**Calculation approach:**
```python
# Method 1: Historical average approach
avg_total_minutes = mean(minutes_per_game_history)
pred_remaining_minutes = max(0, avg_total_minutes - minutes_played_so_far)

# Method 2: Conditional on game situation (better)
if game_is_close:
    # Use games where they played heavy minutes
    pred_remaining_minutes = sample from high-minute games
elif team_is_winning_big:
    # Use blowout games
    pred_remaining_minutes = sample from blowout games (less minutes)
```

---

### 3. Game Context Variables

#### Opponent Quality
- **`opponent_defensive_rating`** (float): How good is opponent's defense
- **`opponent_pace`** (float): How fast does opponent play (possessions per game)
- **`historical_performance_vs_opponent`** (list): Player's past games vs this team
  - More relevant than overall average if you have enough data

#### Home/Away
- **`is_home_game`** (bool): Home vs away split matters for many players
- **`home_points_history`** (list): Performance at home
- **`away_points_history`** (list): Performance on road

#### Game Script / Situation
- **`current_score_differential`** (int): Is team winning/losing and by how much
- **`game_is_close`** (bool): Within 10 points
- **`team_is_winning_big`** (bool): Up by 20+ (affects playing time)
- **`team_is_losing_big`** (bool): Down by 20+ (garbage time, or benched)

**Why this matters:**
- If team up 30, starters sit → 0 additional points
- If team down 25, stars might play heavy minutes to try comeback
- If close game, stars play normal/heavy minutes

#### Usage / Role Context
- **`usage_rate`** (float): What % of team's possessions does player use when on court
- **`is_starter`** (bool): Starter vs bench affects minutes distribution
- **`team_possessions_remaining`** (estimated): How many possessions left in game
  - Depends on pace and time remaining

---

### 4. Situational Modifiers

#### Injury / Rest
- **`is_playing_back_to_back`** (bool): Players often less efficient
- **`days_rest`** (int): 0, 1, 2+ days
- **`is_probable_injury`** (bool): Playing through minor injury

#### Recent Form
- **`last_5_games_points`** (list): Hot/cold streaks matter
- **`last_5_games_minutes`** (list): Recent playing time trends
- Can be more predictive than season-long averages

#### Foul Situation
- **`current_fouls`** (int): If player has 4-5 fouls, might sit more
- **`foul_trouble_likely`** (bool): Some players foul more than others

---

## Monte Carlo Simulation Approach

### Basic Algorithm

```python
def simulate_bet_outcome(
    current_points,
    target_line,
    pred_remaining_minutes,
    points_per_minute_history,
    n_simulations=10000
):
    """
    Simulate whether player will hit the over on their points line.
    
    Returns:
        prob_over: Probability of hitting over (0 to 1)
    """
    hits = 0
    
    for _ in range(n_simulations):
        # Sample from historical points-per-minute distribution
        ppm = random.choice(points_per_minute_history)
        
        # Predict additional points
        additional_points = ppm * pred_remaining_minutes
        
        # Calculate final points
        final_points = current_points + additional_points
        
        # Check if over
        if final_points > target_line:
            hits += 1
    
    prob_over = hits / n_simulations
    return prob_over
```

### Advanced Algorithm (Context-Aware)

```python
def simulate_bet_outcome_advanced(
    current_points,
    target_line,
    minutes_played_so_far,
    time_remaining_in_game,
    current_quarter,
    score_differential,
    # Historical data
    minutes_per_game_history,
    points_per_minute_by_half,  # Dict: {'1H': [...], '2H': [...]}
    points_per_minute_by_situation,  # Dict: {'close': [...], 'blowout': [...]}
    n_simulations=10000
):
    """
    Context-aware simulation considering game situation.
    """
    hits = 0
    
    for _ in range(n_simulations):
        # 1. Estimate remaining minutes (sample from similar games)
        similar_games = filter_games_by_situation(
            minutes_per_game_history, 
            current_situation=score_differential
        )
        typical_total_minutes = random.choice(similar_games)
        pred_remaining_minutes = max(0, typical_total_minutes - minutes_played_so_far)
        
        # 2. Sample points-per-minute based on context
        if current_quarter <= 2:
            # Use first-half efficiency
            ppm = random.choice(points_per_minute_by_half['1H'])
        else:
            # Use second-half efficiency
            ppm = random.choice(points_per_minute_by_half['2H'])
        
        # 3. Adjust for game situation
        if abs(score_differential) > 20:
            # Blowout: use lower efficiency samples
            ppm = random.choice(points_per_minute_by_situation['blowout'])
        
        # 4. Calculate final points
        additional_points = ppm * pred_remaining_minutes
        final_points = current_points + additional_points
        
        # 5. Check if over
        if final_points > target_line:
            hits += 1
    
    prob_over = hits / n_simulations
    return prob_over
```

---

## Key Insights

### 1. Don't Assume Normal Distribution
- Player performance is often **right-skewed** (occasionally huge games)
- Using actual historical distribution captures this
- Sampling from empirical data > fitting parametric distribution

### 2. Time Remaining ≠ Minutes Remaining to Play
- 10 minutes left in game ≠ player will play 10 minutes
- Need to model playing time separately
- Consider: blowouts, foul trouble, coach decisions

### 3. Context Matters Enormously
- Luka with 0 at halftime in a **close game** → likely plays heavy 2H minutes
- Luke Kornet with 0 at halftime in a **blowout** → probably done for the night
- Must condition on game situation

### 4. Half Splits > Full Game Distribution
- If player has 0 at halftime, their 2H distribution is more relevant
- Don't sample from full-game distribution (includes the 0 they already have)

### 5. Starter vs Bench Matters
- Starters: more predictable minutes (30-36 mpg)
- Bench: highly variable minutes (0-25 mpg depending on game flow)
- Need different models for different player types

---

## Practical Example: Luka at Halftime

**Scenario:** 
- Luka Doncic, Over 30.5 points
- Currently has 0 points at halftime
- Close game (within 5 points)
- Played 18 minutes in 1H

**Variables:**
```python
current_points = 0
target_line = 30.5
points_needed = 30.5
minutes_played_so_far = 18
time_remaining_in_game = 24  # 2H in NBA

# Historical data (example)
minutes_per_game_history = [34, 36, 35, 33, 37, ...]  # Usually 33-37 mpg
second_half_minutes_history = [16, 18, 17, 16, 19, ...]
second_half_points_history = [15, 22, 18, 20, 16, ...]
points_per_minute_2H = [0.94, 1.22, 1.06, 1.25, 0.84, ...]

# Predicted remaining minutes
typical_total_minutes = 35  # Sample from close games
pred_remaining_minutes = 35 - 18 = 17 minutes

# Monte Carlo
for sim in range(10000):
    ppm_2H = random.choice(points_per_minute_2H)  # e.g., 1.15
    remaining_pts = ppm_2H * 17  # e.g., 19.5
    final_pts = 0 + 19.5 = 19.5
    # Check if > 30.5 → No in this sim
    
# Result: Maybe 20-30% chance of hitting over
# (Not 0%, but definitely not 50% anymore)
```

---

## Next Steps

1. **Data Collection:** Pull player game logs with quarter/half splits
2. **Feature Engineering:** Calculate points-per-minute, minutes by situation
3. **Build Simulation Engine:** Implement Monte Carlo with context awareness
4. **Validation:** Backtest on historical in-game scenarios
5. **Live Integration:** Connect to live game data (APIs like ESPN, NBA.com)
6. **Calibration:** Ensure simulated probabilities match real-world outcomes

---

## Additional Metrics to Consider (Phase 2)

- **Shot attempts per game** (volume indicator)
- **True shooting percentage** (efficiency)
- **Assist rate** (for assist props)
- **Rebound rate** (for rebound props)
- **Opponent-specific matchup data** (does opponent have good defender for this player)
- **Referee crew** (some refs call more fouls → affects playing time)
- **Travel schedule** (West Coast → East Coast back-to-back)
- **Playoff implications** (teams rest stars when game doesn't matter)

These can be added once core model is working.

---

## Quick V1 Solution

A minimal viable product to get started with real-time probability predictions.

### Inputs (Required)

**Player Historical Data (Pre-computed):**
```python
{
    "player_id": "luka_doncic",
    "player_name": "Luka Doncic",
    "season": "2025-26",
    "num_games": 46,
    
    # Full game distributions (list of values, one per game)
    "total_points_history": [33, 28, 35, 30, 27, ...],  # Total points each game
    "total_minutes_history": [34.2, 36.1, 35.8, 33.5, ...],  # Total minutes each game
    "points_per_minute_history": [0.95, 1.15, 0.88, 0.89, ...],  # PPM each game
    
    # Quarter-by-quarter distributions (list of values, one per game)
    # For each quarter: sample MINUTES and PPM separately
    "q1_points_history": [8, 6, 10, 7, 5, ...],  # Q1 points each game
    "q1_minutes_history": [8.5, 9.0, 8.2, 7.8, ...],  # Q1 minutes each game
    "q1_ppm_history": [0.94, 0.67, 1.22, 0.90, ...],  # Q1 PPM each game
    
    "q2_points_history": [9, 7, 8, 10, 6, ...],
    "q2_minutes_history": [8.8, 9.2, 9.5, 8.5, ...],
    "q2_ppm_history": [1.02, 0.76, 0.84, 1.18, ...],
    
    "q3_points_history": [8, 9, 10, 7, 8, ...],
    "q3_minutes_history": [8.5, 9.1, 9.0, 8.8, ...],
    "q3_ppm_history": [0.94, 0.99, 1.11, 0.80, ...],
    
    "q4_points_history": [8, 6, 7, 6, 8, ...],
    "q4_minutes_history": [8.4, 8.8, 9.1, 8.4, ...],
    "q4_ppm_history": [0.95, 0.68, 0.77, 0.71, ...],
    
    # Summary stats (for quick reference)
    "avg_points_per_game": 32.8,
    "avg_minutes_per_game": 35.2,
    "avg_ppm": 0.93,
    "std_points": 4.2,
    "p25_points": 29.0,
    "p50_points": 33.0,
    "p75_points": 36.0,
}
```

**DuckDB Queries to Build Player Inputs:**

```sql
-- Step 1: Get game-level stats per player (total points, total minutes)
CREATE OR REPLACE TEMP TABLE game_level_stats AS
SELECT 
    game_id,
    game_date,
    player_id,
    player_name,
    MAX(playing_seconds) / 60.0 AS total_minutes,
    MAX(cumulative_points) AS total_points
FROM '~/dev/betting/data/minute_by_minute.parquet'
GROUP BY game_id, game_date, player_id, player_name;

-- Add PPM
CREATE OR REPLACE TEMP TABLE game_stats_with_ppm AS
SELECT 
    *,
    CASE 
        WHEN total_minutes > 0 THEN total_points / total_minutes 
        ELSE 0 
    END AS points_per_minute
FROM game_level_stats;

-- Step 2: Get quarterly splits (Q1, Q2, Q3, Q4)
-- Q1 = minutes 0-11, Q2 = 12-23, Q3 = 24-35, Q4 = 36-47, OT = 48+
CREATE OR REPLACE TEMP TABLE quarter_splits AS
SELECT 
    game_id,
    game_date,
    player_id,
    player_name,
    
    -- Q1 (minutes 0-11)
    MAX(CASE WHEN minute <= 11 THEN playing_seconds ELSE 0 END) / 60.0 AS q1_minutes,
    MAX(CASE WHEN minute <= 11 THEN cumulative_points ELSE 0 END) AS q1_points,
    
    -- Q2 (minutes 12-23)
    (MAX(CASE WHEN minute <= 23 THEN playing_seconds ELSE 0 END) - 
     MAX(CASE WHEN minute <= 11 THEN playing_seconds ELSE 0 END)) / 60.0 AS q2_minutes,
    (MAX(CASE WHEN minute <= 23 THEN cumulative_points ELSE 0 END) - 
     MAX(CASE WHEN minute <= 11 THEN cumulative_points ELSE 0 END)) AS q2_points,
    
    -- Q3 (minutes 24-35)
    (MAX(CASE WHEN minute <= 35 THEN playing_seconds ELSE 0 END) - 
     MAX(CASE WHEN minute <= 23 THEN playing_seconds ELSE 0 END)) / 60.0 AS q3_minutes,
    (MAX(CASE WHEN minute <= 35 THEN cumulative_points ELSE 0 END) - 
     MAX(CASE WHEN minute <= 23 THEN cumulative_points ELSE 0 END)) AS q3_points,
    
    -- Q4 (minutes 36-47)
    (MAX(CASE WHEN minute <= 47 THEN playing_seconds ELSE 0 END) - 
     MAX(CASE WHEN minute <= 35 THEN playing_seconds ELSE 0 END)) / 60.0 AS q4_minutes,
    (MAX(CASE WHEN minute <= 47 THEN cumulative_points ELSE 0 END) - 
     MAX(CASE WHEN minute <= 35 THEN cumulative_points ELSE 0 END)) AS q4_points
    
FROM '~/dev/betting/data/minute_by_minute.parquet'
GROUP BY game_id, game_date, player_id, player_name;

-- Add PPM for each quarter
CREATE OR REPLACE TEMP TABLE quarter_splits_with_ppm AS
SELECT 
    *,
    CASE WHEN q1_minutes > 0 THEN q1_points / q1_minutes ELSE 0 END AS q1_ppm,
    CASE WHEN q2_minutes > 0 THEN q2_points / q2_minutes ELSE 0 END AS q2_ppm,
    CASE WHEN q3_minutes > 0 THEN q3_points / q3_minutes ELSE 0 END AS q3_ppm,
    CASE WHEN q4_minutes > 0 THEN q4_points / q4_minutes ELSE 0 END AS q4_ppm
FROM quarter_splits;

-- Step 3: Build player profile (1 row per player with all distributions)
SELECT 
    g.player_id,
    g.player_name,
    COUNT(*) AS num_games,
    
    -- Summary stats
    AVG(g.total_points) AS avg_points_per_game,
    AVG(g.total_minutes) AS avg_minutes_per_game,
    AVG(g.points_per_minute) AS avg_ppm,
    STDDEV(g.total_points) AS std_points,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY g.total_points) AS p25_points,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY g.total_points) AS p50_points,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY g.total_points) AS p75_points,
    
    -- Full game distributions (ordered by most recent first)
    LIST(g.total_points ORDER BY g.game_date DESC) AS total_points_history,
    LIST(g.total_minutes ORDER BY g.game_date DESC) AS total_minutes_history,
    LIST(g.points_per_minute ORDER BY g.game_date DESC) AS points_per_minute_history,
    
    -- Q1 distributions
    LIST(q.q1_points ORDER BY q.game_date DESC) AS q1_points_history,
    LIST(q.q1_minutes ORDER BY q.game_date DESC) AS q1_minutes_history,
    LIST(q.q1_ppm ORDER BY q.game_date DESC) AS q1_ppm_history,
    
    -- Q2 distributions
    LIST(q.q2_points ORDER BY q.game_date DESC) AS q2_points_history,
    LIST(q.q2_minutes ORDER BY q.game_date DESC) AS q2_minutes_history,
    LIST(q.q2_ppm ORDER BY q.game_date DESC) AS q2_ppm_history,
    
    -- Q3 distributions
    LIST(q.q3_points ORDER BY q.game_date DESC) AS q3_points_history,
    LIST(q.q3_minutes ORDER BY q.game_date DESC) AS q3_minutes_history,
    LIST(q.q3_ppm ORDER BY q.game_date DESC) AS q3_ppm_history,
    
    -- Q4 distributions
    LIST(q.q4_points ORDER BY q.game_date DESC) AS q4_points_history,
    LIST(q.q4_minutes ORDER BY q.game_date DESC) AS q4_minutes_history,
    LIST(q.q4_ppm ORDER BY q.game_date DESC) AS q4_ppm_history
    
FROM game_stats_with_ppm g
LEFT JOIN quarter_splits_with_ppm q 
    ON g.game_id = q.game_id 
    AND g.player_id = q.player_id
GROUP BY g.player_id, g.player_name
HAVING COUNT(*) >= 10  -- Only players with 10+ games
ORDER BY avg_points_per_game DESC;
```

**Example: Get Kevin Durant's data**
```sql
-- Quick query for one player
SELECT 
    player_name,
    num_games,
    ROUND(avg_points_per_game, 1) AS avg_pts,
    ROUND(avg_minutes_per_game, 1) AS avg_mins,
    ROUND(avg_ppm, 3) AS avg_ppm,
    ROUND(std_points, 1) AS std_pts,
    p25_points,
    p50_points,
    p75_points,
    -- Show last 5 games (most recent)
    total_points_history[1:5] AS last_5_games_points,
    total_minutes_history[1:5] AS last_5_games_minutes,
    -- Show Q1 scoring in last 5 games
    q1_points_history[1:5] AS last_5_q1_points,
    q2_points_history[1:5] AS last_5_q2_points,
    q3_points_history[1:5] AS last_5_q3_points,
    q4_points_history[1:5] AS last_5_q4_points
FROM player_profiles
WHERE player_name = 'Kevin Durant';
```

**Example: Verify quarterly data sums to total**
```sql
-- Sanity check: Q1+Q2+Q3+Q4 should equal total points
SELECT 
    player_name,
    game_date,
    total_points,
    q1_points + q2_points + q3_points + q4_points AS quarterly_sum,
    total_points - (q1_points + q2_points + q3_points + q4_points) AS difference
FROM player_profiles
WHERE player_name = 'Kevin Durant'
AND ABS(total_points - (q1_points + q2_points + q3_points + q4_points)) > 1
LIMIT 10;
```

**Optional: Extract game situation context (close vs blowout)**
```sql
-- For V2: Add game situation context to predict remaining minutes
-- This requires score data (not in minute_by_minute.parquet yet)
-- But here's the structure for when you add it:

CREATE OR REPLACE TEMP TABLE game_situations AS
SELECT 
    game_id,
    game_date,
    player_id,
    player_name,
    total_minutes,
    
    -- Define game situation (would need score data)
    CASE 
        WHEN final_margin <= 10 THEN 'close'
        WHEN final_margin > 20 THEN 'blowout'
        ELSE 'competitive'
    END AS game_situation
    
FROM game_level_stats;

-- Then aggregate minutes by situation
SELECT 
    player_id,
    player_name,
    LIST(total_minutes ORDER BY game_date DESC) FILTER (WHERE game_situation = 'close') AS close_game_minutes,
    LIST(total_minutes ORDER BY game_date DESC) FILTER (WHERE game_situation = 'blowout') AS blowout_minutes,
    AVG(total_minutes) FILTER (WHERE game_situation = 'close') AS avg_minutes_close_game,
    AVG(total_minutes) FILTER (WHERE game_situation = 'blowout') AS avg_minutes_blowout
FROM game_situations
GROUP BY player_id, player_name;
```

**Live Game State (Updated Every Minute):**
```python
{
    "game_id": "2026020501_DAL_BOS",
    "timestamp": "2026-02-05 20:35:00",
    
    # Player state
    "player_id": "luka_doncic",
    "current_points": 8,
    "minutes_played_so_far": 18.5,
    "current_fouls": 2,
    "is_on_court": True,  # Currently playing
    
    # Quarter breakdown of points/minutes so far
    "q1_points": 3,
    "q1_minutes": 8.2,
    "q2_points": 5,
    "q2_minutes": 10.3,
    "q3_points_so_far": 0,  # Currently in Q3
    "q3_minutes_so_far": 3.5,  # Currently in Q3
    
    # Game state
    "current_quarter": 3,
    "time_remaining_in_quarter": 8.5,  # minutes left in Q3
    "time_remaining_in_game": 20.5,  # total minutes left (Q3 + Q4)
    
    # Score context (affects playing time prediction)
    "team_score": 62,
    "opponent_score": 58,
    "score_differential": 4,  # positive = team winning
    "game_is_close": True,  # Within 10 points
    
    # Prop bet
    "prop_type": "points",
    "prop_line": 30.5,
    "bet_side": "over",  # or "under"
}
```

**Additional Context for V1 (Optional but Helpful):**
```python
{
    # Starter vs bench (affects minutes distribution)
    "is_starter": True,  # Starters play more predictable minutes
    
    # Recent trends (V1 can skip, but useful)
    "last_3_games_avg_points": 31.2,
    "last_3_games_avg_minutes": 34.5,
    
    # Injury/rest status (V1 skips, but important for V2)
    "is_probable_injury": False,
    "days_rest": 1,  # 0 = back-to-back, 1 = normal rest
}
```

---

### Monte Carlo Simulation Logic (Quarter-by-Quarter)

**Key Principle:** 
- **Past quarters**: Use actual points scored (already in `current_points`)
- **Current quarter (partial)**: Sample PPM, multiply by remaining time in quarter
- **Future quarters**: Sample BOTH minutes and PPM, multiply together

**Pseudocode:**

```python
def monte_carlo_simulate_bet(
    player_profile,      # Historical data (quarterly distributions)
    game_state,          # Current game state
    prop_line,           # Target line (e.g., 30.5 points)
    n_simulations=10000
):
    """
    Simulate remaining game to predict probability of hitting over.
    
    Samples quarter-by-quarter:
    - Current quarter (partial): sample PPM, multiply by time remaining
    - Future quarters: sample minutes AND PPM, multiply together
    """
    
    hits = 0
    
    for sim in range(n_simulations):
        # Start with points already scored
        projected_final_points = game_state['current_points']
        
        # =================================================================
        # CURRENT QUARTER (PARTIAL) - only project remaining time
        # =================================================================
        current_quarter = game_state['current_quarter']  # e.g., 3
        time_remaining_in_quarter = game_state['time_remaining_in_quarter']  # e.g., 8.5 min
        
        if time_remaining_in_quarter > 0:
            # Sample PPM for current quarter
            current_q_ppm = random.choice(player_profile[f'q{current_quarter}_ppm_history'])
            
            # Project points for remaining time in current quarter
            remaining_quarter_points = current_q_ppm * time_remaining_in_quarter
            projected_final_points += remaining_quarter_points
        
        # =================================================================
        # FUTURE QUARTERS - sample both minutes AND PPM
        # =================================================================
        for future_quarter in range(current_quarter + 1, 5):  # e.g., Q4 if currently in Q3
            # Sample minutes for this future quarter
            future_q_minutes = random.choice(player_profile[f'q{future_quarter}_minutes_history'])
            
            # Sample PPM for this future quarter
            future_q_ppm = random.choice(player_profile[f'q{future_quarter}_ppm_history'])
            
            # Project points for this future quarter
            future_quarter_points = future_q_ppm * future_q_minutes
            projected_final_points += future_quarter_points
        
        # =================================================================
        # OVERTIME (if needed)
        # =================================================================
        # For V1: Skip OT prediction (rare)
        # For V2: If time_remaining suggests OT possible, sample OT distributions
        
        # =================================================================
        # CHECK IF BET HITS
        # =================================================================
        if projected_final_points > prop_line:
            hits += 1
    
    # Calculate probability
    prob_over = hits / n_simulations
    prob_under = 1 - prob_over
    
    return {
        'prob_over': prob_over,
        'prob_under': prob_under,
        'n_simulations': n_simulations
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

# Player: Luka Doncic (averages 33 PPG)
player_profile = {
    'player_name': 'Luka Doncic',
    # Q1 history
    'q1_minutes_history': [8.5, 9.0, 8.2, 7.8, 9.1, ...],
    'q1_ppm_history': [0.94, 0.67, 1.22, 0.90, 1.05, ...],
    # Q2 history
    'q2_minutes_history': [8.8, 9.2, 9.5, 8.5, 9.0, ...],
    'q2_ppm_history': [1.02, 0.76, 0.84, 1.18, 0.95, ...],
    # Q3 history
    'q3_minutes_history': [8.5, 9.1, 9.0, 8.8, 9.2, ...],
    'q3_ppm_history': [0.94, 0.99, 1.11, 0.80, 1.05, ...],
    # Q4 history
    'q4_minutes_history': [8.4, 8.8, 9.1, 8.4, 8.9, ...],
    'q4_ppm_history': [0.95, 0.68, 0.77, 0.71, 1.10, ...],
}

# Current game state: Q3 with 8.5 min remaining, has 8 points
game_state = {
    'current_points': 8,           # Points so far (Q1+Q2+partial Q3)
    'current_quarter': 3,          # Currently in Q3
    'time_remaining_in_quarter': 8.5,  # 8.5 min left in Q3
    
    # Breakdown (for debugging/logging)
    'q1_points': 3,    # Actual Q1 points
    'q2_points': 5,    # Actual Q2 points
    'q3_points_so_far': 0,  # Q3 partial (3.5 min played, no points yet)
}

prop_line = 30.5  # Over/Under 30.5 points

# Run simulation
result = monte_carlo_simulate_bet(player_profile, game_state, prop_line, n_simulations=10000)

print(f"Prob Over {prop_line}: {result['prob_over']:.1%}")
print(f"Prob Under {prop_line}: {result['prob_under']:.1%}")

# Example output:
# Prob Over 30.5: 34.7%
# Prob Under 30.5: 65.3%
```

**Why This Works:**

1. **No double-counting**: Past points are in `current_points`, we only project forward
2. **Partial quarter handling**: Only sample PPM for current quarter (minutes already being played)
3. **Future quarter flexibility**: Sample both minutes AND PPM (captures variance in both)
4. **Quarter-specific patterns**: Uses Q3 distribution for Q3, Q4 for Q4 (some players are 4th quarter players)

**Edge Cases:**

- **Game over** (Q4 time remaining = 0): `projected_final_points = current_points` → deterministic 0% or 100%
- **Start of quarter** (just started Q3): `time_remaining_in_quarter = 12.0` → use full Q3 PPM
- **End of quarter** (Q3 about to end): `time_remaining_in_quarter = 0.1` → minimal Q3 contribution

### Outputs

**Probability Prediction:**
```python
{
    "timestamp": "2026-02-05 20:35:00",
    "game_clock": "Q3 8:30",
    
    # Main output
    "prob_over": 0.653,  # 65.3% chance of hitting over
    "prob_under": 0.347,  # 34.7% chance of staying under
    
    # Current state
    "current_points": 8,
    "target_line": 30.5,
    "points_needed": 22.5,
    
    # Model internals (for debugging/analysis)
    "pred_remaining_minutes": 15.2,  # Expected minutes rest of game
    "pred_additional_points_mean": 17.5,  # Expected points to be added
    "pred_additional_points_p10": 8.2,  # 10th percentile
    "pred_additional_points_p50": 16.8,  # Median
    "pred_additional_points_p90": 28.3,  # 90th percentile
    "pred_final_points_mean": 25.5,  # 8 + 17.5
    "pred_final_points_p10": 16.2,
    "pred_final_points_p50": 24.8,
    "pred_final_points_p90": 36.3,
    
    # Simulation metadata
    "n_simulations": 10000,
    "model_version": "v1_simple",
}
```

**Time Series Output (Store Every Minute):**
```python
[
    {
        "game_clock": "Q1 12:00",
        "minutes_elapsed": 0,
        "current_points": 0,
        "prob_over": 0.524,  # Slightly better than opening line
    },
    {
        "game_clock": "Q1 8:23",
        "minutes_elapsed": 3.6,
        "current_points": 5,
        "prob_over": 0.612,  # Hot start
    },
    {
        "game_clock": "Q2 0:00",
        "minutes_elapsed": 24,
        "current_points": 8,
        "prob_over": 0.428,  # Cold half, probability dropped
    },
    {
        "game_clock": "Q3 8:30",
        "minutes_elapsed": 27.5,
        "current_points": 8,
        "prob_over": 0.347,  # Still cold, probability dropping
    },
    # ... continue every minute
    {
        "game_clock": "Q4 0:00",
        "minutes_elapsed": 48,
        "current_points": 28,
        "prob_over": 0.0,  # Game over, didn't hit
    }
]
```

### Success Metrics (Backtesting)

#### 0. Unit Tests with Hard-Coded Examples (Sanity Checks)

**Goal:** Model behaves sensibly on obvious scenarios before running full backtest.

**Test Cases:**

```python
def test_obvious_scenarios():
    """Test model on edge cases and obvious scenarios."""
    
    # Setup: Player averages 30 PPG, historical data
    player_data = get_luka_historical_data()
    prop_line = 30.5
    
    # TEST 1: Start of game - should be ~50% (player averages 30)
    game_state = {"current_points": 0, "minutes_played": 0, "quarter": 1, "time_remaining": 48}
    prob = simulate(player_data, game_state, prop_line)
    assert 0.45 <= prob <= 0.55, f"Start of game should be ~50%, got {prob}"
    
    # TEST 2: On pace at halftime - should be ~50%
    game_state = {"current_points": 15, "minutes_played": 18, "quarter": 2, "time_remaining": 24}
    prob = simulate(player_data, game_state, prop_line)
    assert 0.45 <= prob <= 0.55, f"On pace at half should be ~50%, got {prob}"
    
    # TEST 3: Almost there at halftime - should be very high
    game_state = {"current_points": 29, "minutes_played": 18, "quarter": 2, "time_remaining": 24}
    prob = simulate(player_data, game_state, prop_line)
    assert prob >= 0.90, f"29 pts at half should be >90%, got {prob}"
    
    # TEST 4: Already over - should be 100%
    game_state = {"current_points": 32, "minutes_played": 30, "quarter": 3, "time_remaining": 12}
    prob = simulate(player_data, game_state, prop_line)
    assert prob == 1.0, f"Already over should be 100%, got {prob}"
    
    # TEST 5: Needs unrealistic amount - should be very low
    game_state = {"current_points": 5, "minutes_played": 36, "quarter": 4, "time_remaining": 3}
    prob = simulate(player_data, game_state, prop_line)
    assert prob <= 0.05, f"Need 26 pts in 3 min should be <5%, got {prob}"
    
    # TEST 6: Game over, didn't hit - should be 0%
    game_state = {"current_points": 28, "minutes_played": 35, "quarter": 4, "time_remaining": 0}
    prob = simulate(player_data, game_state, prop_line)
    assert prob == 0.0, f"Game over under line should be 0%, got {prob}"
    
    # TEST 7: Cold first half but plenty of time - should be reasonable
    game_state = {"current_points": 8, "minutes_played": 18, "quarter": 2, "time_remaining": 24}
    prob = simulate(player_data, game_state, prop_line)
    assert 0.25 <= prob <= 0.45, f"Cold half should be 25-45%, got {prob}"
    
    # TEST 8: Hot start - should be elevated
    game_state = {"current_points": 12, "minutes_played": 6, "quarter": 1, "time_remaining": 42}
    prob = simulate(player_data, game_state, prop_line)
    assert 0.60 <= prob <= 0.80, f"Hot start should be 60-80%, got {prob}"

def test_bench_player_scenarios():
    """Test model on bench player with less predictable minutes."""
    
    # Setup: Backup big, averages 7 PPG, 15 MPG
    player_data = get_bench_player_data()
    prop_line = 10.5
    
    # TEST: In garbage time with 8 pts - depends on if they stay in
    game_state = {"current_points": 8, "minutes_played": 12, "quarter": 4, "time_remaining": 5}
    prob = simulate(player_data, game_state, prop_line)
    # Should be low because bench players often pulled in garbage time
    assert 0.05 <= prob <= 0.30, f"Bench player garbage time should be low, got {prob}"

def test_probability_monotonicity():
    """Test that probabilities change sensibly over time."""
    
    player_data = get_luka_historical_data()
    prop_line = 30.5
    
    # Scenario: Player scores 5 points, time passes
    state_t1 = {"current_points": 10, "minutes_played": 12, "quarter": 2, "time_remaining": 30}
    state_t2 = {"current_points": 15, "minutes_played": 15, "quarter": 2, "time_remaining": 27}
    
    prob_t1 = simulate(player_data, state_t1, prop_line)
    prob_t2 = simulate(player_data, state_t2, prop_line)
    
    # Should increase when points scored
    assert prob_t2 > prob_t1, "Probability should increase when points scored"
    
    # Scenario: No points scored, time passes
    state_t3 = {"current_points": 15, "minutes_played": 18, "quarter": 2, "time_remaining": 24}
    prob_t3 = simulate(player_data, state_t3, prop_line)
    
    # Should decrease when time passes with no points
    assert prob_t3 < prob_t2, "Probability should decrease when time passes with no points"
```

**Why this matters:**
- Catches major bugs before running expensive backtests
- Validates model logic on extreme cases (0%, 100% scenarios)
- Tests less obvious scenarios (cold start, hot start) against intuition
- Ensures monotonicity properties hold

**Run these tests:**
- Before every backtest
- After every model change
- As part of CI/CD pipeline

---

#### 1. Calibration (Most Important)

#### 0. Unit Tests with Hard-Coded Examples (Sanity Checks)

**Goal:** Model behaves sensibly on obvious scenarios before running full backtest.

**Test Cases:**

```python
def test_obvious_scenarios():
    """Test model on edge cases and obvious scenarios."""
    
    # Setup: Player averages 30 PPG, historical data
    player_data = get_luka_historical_data()
    prop_line = 30.5
    
    # TEST 1: Start of game - should be ~50% (player averages 30)
    game_state = {"current_points": 0, "minutes_played": 0, "quarter": 1, "time_remaining": 48}
    prob = simulate(player_data, game_state, prop_line)
    assert 0.45 <= prob <= 0.55, f"Start of game should be ~50%, got {prob}"
    
    # TEST 2: On pace at halftime - should be ~50%
    game_state = {"current_points": 15, "minutes_played": 18, "quarter": 2, "time_remaining": 24}
    prob = simulate(player_data, game_state, prop_line)
    assert 0.45 <= prob <= 0.55, f"On pace at half should be ~50%, got {prob}"
    
    # TEST 3: Almost there at halftime - should be very high
    game_state = {"current_points": 29, "minutes_played": 18, "quarter": 2, "time_remaining": 24}
    prob = simulate(player_data, game_state, prop_line)
    assert prob >= 0.90, f"29 pts at half should be >90%, got {prob}"
    
    # TEST 4: Already over - should be 100%
    game_state = {"current_points": 32, "minutes_played": 30, "quarter": 3, "time_remaining": 12}
    prob = simulate(player_data, game_state, prop_line)
    assert prob == 1.0, f"Already over should be 100%, got {prob}"
    
    # TEST 5: Needs unrealistic amount - should be very low
    game_state = {"current_points": 5, "minutes_played": 36, "quarter": 4, "time_remaining": 3}
    prob = simulate(player_data, game_state, prop_line)
    assert prob <= 0.05, f"Need 26 pts in 3 min should be <5%, got {prob}"
    
    # TEST 6: Game over, didn't hit - should be 0%
    game_state = {"current_points": 28, "minutes_played": 35, "quarter": 4, "time_remaining": 0}
    prob = simulate(player_data, game_state, prop_line)
    assert prob == 0.0, f"Game over under line should be 0%, got {prob}"
    
    # TEST 7: Cold first half but plenty of time - should be reasonable
    game_state = {"current_points": 8, "minutes_played": 18, "quarter": 2, "time_remaining": 24}
    prob = simulate(player_data, game_state, prop_line)
    assert 0.25 <= prob <= 0.45, f"Cold half should be 25-45%, got {prob}"
    
    # TEST 8: Hot start - should be elevated
    game_state = {"current_points": 12, "minutes_played": 6, "quarter": 1, "time_remaining": 42}
    prob = simulate(player_data, game_state, prop_line)
    assert 0.60 <= prob <= 0.80, f"Hot start should be 60-80%, got {prob}"

def test_bench_player_scenarios():
    """Test model on bench player with less predictable minutes."""
    
    # Setup: Backup big, averages 7 PPG, 15 MPG
    player_data = get_bench_player_data()
    prop_line = 10.5
    
    # TEST: In garbage time with 8 pts - depends on if they stay in
    game_state = {"current_points": 8, "minutes_played": 12, "quarter": 4, "time_remaining": 5}
    prob = simulate(player_data, game_state, prop_line)
    # Should be low because bench players often pulled in garbage time
    assert 0.05 <= prob <= 0.30, f"Bench player garbage time should be low, got {prob}"

def test_probability_monotonicity():
    """Test that probabilities change sensibly over time."""
    
    player_data = get_luka_historical_data()
    prop_line = 30.5
    
    # Scenario: Player scores 5 points, time passes
    state_t1 = {"current_points": 10, "minutes_played": 12, "quarter": 2, "time_remaining": 30}
    state_t2 = {"current_points": 15, "minutes_played": 15, "quarter": 2, "time_remaining": 27}
    
    prob_t1 = simulate(player_data, state_t1, prop_line)
    prob_t2 = simulate(player_data, state_t2, prop_line)
    
    # Should increase when points scored
    assert prob_t2 > prob_t1, "Probability should increase when points scored"
    
    # Scenario: No points scored, time passes
    state_t3 = {"current_points": 15, "minutes_played": 18, "quarter": 2, "time_remaining": 24}
    prob_t3 = simulate(player_data, state_t3, prop_line)
    
    # Should decrease when time passes with no points
    assert prob_t3 < prob_t2, "Probability should decrease when time passes with no points"
```

**Why this matters:**
- Catches major bugs before running expensive backtests
- Validates model logic on extreme cases (0%, 100% scenarios)
- Tests less obvious scenarios (cold start, hot start) against intuition
- Ensures monotonicity properties hold

**Run these tests:**
- Before every backtest
- After every model change
- As part of CI/CD pipeline

---

#### 1. Calibration (Most Important)

**Goal:** When model says 70% chance of over, it should hit ~70% of the time.

**Test:**
```python
# For all predictions made during season
predictions_at_70pct = filter(predictions, lambda p: 0.65 <= p.prob_over <= 0.75)

actual_hit_rate = sum(p.actual_outcome == 'over' for p in predictions_at_70pct) / len(predictions_at_70pct)

# Should be close to 0.70
assert 0.65 <= actual_hit_rate <= 0.75  # Within 5% tolerance
```

**Calibration Plot:**
- Bucket predictions into bins (0-10%, 10-20%, ..., 90-100%)
- For each bin, calculate actual hit rate
- Plot predicted vs actual
- Perfect calibration = diagonal line

**Metrics:**
- **Brier Score:** `mean((predicted_prob - actual_outcome)^2)` - Lower is better
  - Perfect = 0.0, Random = 0.25
  - Good model < 0.20
- **Log Loss:** `mean(-log(predicted_prob_of_actual_outcome))` - Lower is better
  - Penalizes confident wrong predictions heavily

#### 2. Discrimination

**Goal:** Model should assign higher probabilities to bets that actually hit.

**Metric:**
- **ROC AUC:** Area under receiver operating characteristic curve
  - 0.5 = random, 1.0 = perfect
  - Good model > 0.65
- **Mean Probability Difference:**
  - `mean(prob_over | actual=over) - mean(prob_over | actual=under)`
  - Should be significantly positive (e.g., > 0.15)

#### 3. Profitability (Ultimate Test)

**Goal:** Can you make money betting based on model predictions?

**Simulation:**
```python
# Kelly Criterion or Fixed Stake betting
for prediction in all_predictions:
    if prediction.prob_over > 0.55:  # Edge threshold
        # Bet on over
        if prediction.actual_outcome == 'over':
            profit += stake
        else:
            profit -= stake

roi = profit / total_risked
```

**Metrics:**
- **ROI:** Return on investment (should be > 0%)
- **Win Rate:** % of bets that won
- **Sharpe Ratio:** Risk-adjusted return
- **Max Drawdown:** Largest losing streak

#### 4. Time Series Coherence

**Goal:** Probabilities should change smoothly and logically over time.

**Tests:**

a) **Monotonicity (mostly):**
```python
# If player scores points, prob_over should increase (usually)
if current_points_t2 > current_points_t1:
    assert prob_over_t2 >= prob_over_t1  # (with rare exceptions)

# If time passes with no points, prob_over should decrease
if current_points_t2 == current_points_t1 and time_passed > 0:
    assert prob_over_t2 <= prob_over_t1
```

b) **Convergence at game end:**
```python
# At end of game, probability should be 0 or 1
if game_over:
    if final_points > line:
        assert prob_over == 1.0
    else:
        assert prob_over == 0.0
```

c) **Reasonable trajectories:**
- Plot probability over time for sample games
- Should look smooth, not erratic jumps
- Should respond appropriately to scoring bursts and droughts

#### 5. Benchmarks

Compare model to simple baselines:

**Baseline 1: Static Probability (Opening Line)**
- Just use 50% (or implied odds from opening line) entire game
- Model should beat this easily

**Baseline 2: Linear Projection (Deterministic)**
```python
# Simple projection: current pace * remaining time
points_per_minute_so_far = current_points / minutes_played if minutes_played > 0 else 0
projected_total = current_points + (points_per_minute_so_far * pred_remaining_minutes)
prob_over = 1.0 if projected_total > line else 0.0
```
- Model should beat this significantly (doesn't capture variance)
- Always returns 0% or 100%, not calibrated

**Baseline 3: Bet Cover Probability (Simple Monte Carlo)**
```python
def bet_cover_baseline(current_points, points_needed, pred_remaining_minutes, 
                       season_avg_ppm, season_std_ppm, n_sims=10000):
    """
    Simple baseline: start at 50%, adjust based on points per minute.
    
    Uses normal distribution for PPM (crude but simple proxy).
    """
    hits = 0
    for _ in range(n_sims):
        # Sample PPM from normal distribution (using season stats)
        ppm_sample = np.random.normal(season_avg_ppm, season_std_ppm)
        ppm_sample = max(0, ppm_sample)  # Can't be negative
        
        # Project additional points
        additional_points = ppm_sample * pred_remaining_minutes
        final_points = current_points + additional_points
        
        if final_points > line:
            hits += 1
    
    return hits / n_sims

# Example usage:
# Luka averages 0.95 PPM with std of 0.25
prob = bet_cover_baseline(
    current_points=8, 
    points_needed=22.5,
    pred_remaining_minutes=17,
    season_avg_ppm=0.95,
    season_std_ppm=0.25
)
```

**Why this baseline is useful:**
- Starts at ~50% at game start (if player averages the line)
- Fluctuates based on current pace and time remaining
- Captures variance (unlike Baseline 2)
- Simple enough to implement quickly
- **Good sanity check:** Full model should beat this
  - Full model uses actual distribution (not normal)
  - Full model uses half splits
  - Full model uses game context

**Expected performance:**
- This baseline should be decent but not great
- Brier Score: ~0.20-0.22
- ROC AUC: ~0.58-0.62
- Full model should beat by 10-15%

**Baseline 4: Historical Average**
```python
# Just use season average
prob_over = fraction of games where player scored > line
```
- Model should beat this (doesn't use live data)

---

### How Baselines Relate to Each Other

**Sophistication ranking:**
1. **Baseline 1 (Static)** - Worst, ignores everything
2. **Baseline 4 (Historical Avg)** - Ignores live game state
3. **Baseline 2 (Linear)** - Uses live state but no variance
4. **Baseline 3 (Bet Cover)** - Uses live state + variance, but crude distribution
5. **Full Monte Carlo Model** - Best, uses empirical distributions + context

**Your model should beat all of these progressively:**
- If you can't beat Baseline 1 → Model is broken
- If you can't beat Baseline 4 → Model isn't using live data properly
- If you can't beat Baseline 2 → Model's variance estimates are wrong
- If you can't beat Baseline 3 → Model's improvements (empirical dist, context) aren't adding value

### Backtesting Implementation

```python
# Pseudocode for backtesting pipeline

# 1. Get historical games
historical_games = load_games(season="2024-25", player="luka_doncic")

# 2. For each game, simulate minute-by-minute predictions
all_predictions = []

for game in historical_games:
    # Get player's historical data UP TO this game (no lookahead bias)
    hist_data = get_player_history(player, before_date=game.date, last_n_games=20)
    
    # Simulate the game minute-by-minute
    for minute in range(0, 48):
        game_state = game.get_state_at_minute(minute)
        
        # Run Monte Carlo simulation
        prediction = monte_carlo_simulate(
            player_history=hist_data,
            game_state=game_state,
            prop_line=30.5,
            n_simulations=10000
        )
        
        # Store prediction
        prediction['actual_outcome'] = 'over' if game.final_points > 30.5 else 'under'
        all_predictions.append(prediction)

# 3. Calculate metrics
calibration_plot(all_predictions)
brier_score = calculate_brier_score(all_predictions)
roc_auc = calculate_roc_auc(all_predictions)
roi = calculate_betting_roi(all_predictions, strategy='kelly')

# 4. Generate report
print(f"Brier Score: {brier_score:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Betting ROI: {roi:.2%}")
```

### Success Criteria for V1

**Minimum Bar (Model is working):**
- ✅ Brier Score < 0.22
- ✅ ROC AUC > 0.60
- ✅ Calibration error < 0.10 (predicted vs actual within 10%)
- ✅ Beats all 3 baselines

**Good Model (Ready for real money):**
- ✅ Brier Score < 0.18
- ✅ ROC AUC > 0.68
- ✅ Calibration error < 0.05
- ✅ Positive ROI (>3%) on backtest with Kelly betting

**Great Model (Edge over market):**
- ✅ Brier Score < 0.15
- ✅ ROC AUC > 0.75
- ✅ Calibration error < 0.03
- ✅ ROI > 8% on backtest
- ✅ Profitable across multiple seasons, multiple players

### Data Requirements for Backtesting

**Minimum:**
- Game logs with timestamps (minute-by-minute scoring)
- Player box scores (points, minutes played by half/quarter)
- ~100 games per player (1-2 seasons)

**Ideal:**
- Play-by-play data with exact timestamps
- Second-by-second scoring updates
- Multiple seasons (3+) for robustness
- Multiple players (test generalization)

**Sources:**
- NBA API (stats.nba.com)
- Basketball Reference (play-by-play tables)
- ESPN API
- Sports betting odds archives (for comparing to market)

### V1 Simplifications (To Add Later)

**What we're skipping for V1:**
- ❌ Opponent-specific adjustments (use season averages)
- ❌ Home/away splits (combine all games)
- ❌ Recent form weighting (equal weight to all games in window)
- ❌ Foul trouble adjustments (assume typical minutes)
- ❌ Lineup-specific data (who else is on court)
- ❌ Real-time injury updates
- ❌ Advanced stats (usage rate, true shooting %)

**Why skip these:**
- Keep model simple to start
- Easier to debug when things go wrong
- Faster to implement and test
- Can add incrementally and measure improvement

**Add them only if:**
- V1 works and is calibrated
- Clear hypothesis for how it improves model
- Can A/B test improvement on held-out data

---

## Live Bet Tracking

### Example: February 6, 2026

================================================================================
📋 PLAYS SUMMARY
================================================================================

Detroit Pistons vs New York Knicks @ 7:40pm et
- New York Knicks -1.5 (STEAM FAVORITE)

Milwaukee Bucks vs Indiana Pacers @ 8:10pm et
- Indiana Pacers -1.5 (STEAM FAVORITE)

Minnesota Timberwolves vs New Orleans Pelicans @ 8:10pm et
- New Orleans Pelicans +9.5 (STEAM UNDERDOG) (CURRENT)

================================================================================

STEAM_DETECTED: YES

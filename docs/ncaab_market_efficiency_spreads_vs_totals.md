# NCAAB Market Efficiency: Spreads vs Totals

**Date:** 2026-01-18  
**Analysis:** Comparing market efficiency for game spreads vs game totals

---

## 🎯 Question

Are NCAAB totals (over/under) markets less efficient than spread markets?

---

## 📊 Results Summary

### Spreads (from conference analysis)
- **Best Conference (ACC):** Model MAE advantage of **+0.04 points**
- **Win Rate:** 48.7% (below 50% breakeven)
- **Verdict:** Market is too efficient, no exploitable edge

### Totals (this analysis)
- **Overall:** Model MAE **+0.05 points worse** than market
- **Win Rate:** 49.8% (below 50% breakeven)
- **Verdict:** Market is equally efficient, no exploitable edge

---

## 📈 Walk-Forward Validation: Totals

**Test Setup:**
- Season: 2024-25
- Total games: 5,467
- Test dates: 148
- Method: Train on all history before each date, test on that date

**Model Features:**
- `x1_home`: Home team's avg last 10 implied scores
- `x1_away`: Away team's avg last 10 implied scores  
- `x2_home`: Home team's current implied score
- `x2_away`: Away team's current implied score
- `x3`: Binary conference game indicator

**Results:**
```
Model MAE:  12.70 points
Market MAE: 12.65 points
Difference: +0.05 points (Market wins)

Model beats market: 2,722/5,467 games (49.8%)
```

---

## 🔍 Model Snapshot Comparison: Totals

**Test Setup:**
- Train model every 10 days
- Test each model on ALL future games
- Shows how model evolves over season

**Results (15 snapshots):**
```
Avg Model MAE:  12.57 points
Avg Market MAE: 12.53 points
Avg Difference: +0.05 points (Market wins)

Model wins: 8/15 snapshots (53.3%)
```

**Best Snapshot:**
- Date: 2025-03-28 (late season)
- Model advantage: -0.13 points
- Still not exploitable (very small edge, likely noise)

**Worst Snapshot:**
- Date: 2024-11-04 (early season)
- Model disadvantage: +0.78 points
- Market sharper than model early in season

---

## 💡 Key Insights

### 1. Totals Are Not Easier Than Spreads
Both markets are highly efficient:
- Spreads: Best conference edge of +0.04 points
- Totals: Overall edge of -0.05 points (market wins)

### 2. Both Below 50% Win Rate
- Spreads (ACC): 48.7% win rate
- Totals (Overall): 49.8% win rate
- **Need >52-55% to beat vig and be profitable**

### 3. Markets Learn Quickly
Early season (Nov 4): Model loses by 0.78 points  
Mid season (Dec-Jan): Model ~equal to market  
Late season (Mar 28): Model wins by 0.13 points (but sample size = 22 games)

### 4. The Model Converges to Market
As more data accumulates, model predictions converge to market lines.
This suggests market is already pricing in all our features efficiently.

---

## 🚫 Why Can't We Beat the Market?

### Market Already Knows:
1. **Team Strength** (x1: rolling avg implied scores)
   - Market sets lines based on same historical performance
2. **Current Matchup** (x2: current implied scores)
   - Market adjusts for specific opponent matchups  
3. **Conference Games** (x3: conference indicator)
   - Market knows conference effects

### We Need:
- Information market doesn't have
- Faster updates than market
- Better situational modeling
- Lower-liquidity markets

---

## 🎯 Next Steps (Recommendations)

### Option 1: Opening vs Closing Lines ⭐⭐⭐
- Test if opening lines are less efficient
- Market may not be instant-efficient
- Potential timing edge

### Option 2: Small Conference Deep Dive ⭐⭐
- Focus on low-volume conferences
- MEAC, SWAC, Summit League
- Less sharp money = potential edge

### Option 3: Game Situations ⭐⭐
- Back-to-backs (rest disadvantage)
- Long travel games
- Rivalry games
- Tournament vs regular season

### Option 4: First Half Lines ⭐
- Lower liquidity = potential edge
- Different dynamics than full game

---

## 📝 Conclusion

**NCAAB totals markets are just as efficient as spreads markets.**

Both show:
- Near-zero MAE advantage over market
- Win rates below 50% breakeven
- Model convergence to market prices

**We cannot beat the market on game-level totals or spreads** with publicly available data and linear regression on team strength features.

Must pivot to:
1. **Timing strategies** (opening vs closing)
2. **Lower-liquidity markets** (small conferences, first half)
3. **Situational factors** (rest, travel, matchups)
4. **Player props** (less efficient than game lines)

---

## 📁 Files

**Analysis Scripts:**
- `analysis/ml_pricing_ncaab_games_v2.py` - Spreads model
- `analysis/ml_pricing_ncaab_totals.py` - Totals model (this analysis)
- `analysis/conference_inefficiency_hunt.py` - Conference-specific spreads

**Data:**
- Uses S3-cached game outcomes and betting lines
- Season: 2024-25 (Nov 4, 2024 - Apr 7, 2025)
- 5,589 games with totals lines
- 373 D1 teams

---

**Bottom Line:** Markets are efficient. Linear models on public data can't beat them. Need different approach.


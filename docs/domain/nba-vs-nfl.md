# NBA vs NFL: Sport-Specific Betting Patterns

**Status:** 🚧 DRAFT - Needs your review  
**Last updated:** 2026-02-13

**⚠️ REVIEW INSTRUCTIONS:**
- ➕ Add YOUR specific observations
- 📝 Correct general knowledge that doesn't match reality
- ✅ Mark what's accurate
- ❌ Flag what's wrong

---

## Market Volume & Liquidity

### NBA

**Game frequency:**
- Games almost **every day** during season (Oct-Apr)
- 82 games per team over ~6 months
- 15-30 games per night (league-wide)
- 1,230 regular season games total

**Betting volume:**
- **High frequency, moderate $ per game**
- Average NBA bet: ~$51
- More games = betting spread across more events
- Player props widely available
- Sharp bettors very active

**Data availability:**
- Tons of historical data per season
- Large sample sizes for analysis
- Easier to model (more games = more data)

### NFL

**Game frequency:**
- Games once per week (mostly Sundays)
- 17 games per team over ~18 weeks
- 13-16 games per week (league-wide)
- 272 regular season games total (league-wide)

**Betting volume:**
- **HIGHER $ per game than NBA** (more liquid per game)
- Average NFL bet: ~$83 vs NBA: ~$51
- NFL teams worth more ($7.1B vs $5.4B average)
- Total NFL season handle: ~$30B+ (concentrated over 272 games)
- Public very active (more popular sport in US)

**Data availability:**
- Smaller sample sizes (17 games vs 82)
- Each game matters more (higher variance)
- Harder to model (less data per season)

**⚠️ YOUR INPUT:**
- Does high NBA volume make it easier or harder to find edges?
- Do you focus on NBA because of daily opportunities?

---

## Market Efficiency

### NBA Efficiency

**Game lines (spreads/totals):**
- **Highly efficient** (very hard to beat)
- Sharp bettors extremely active
- Edges are small (<2%)
- Need sophisticated models to compete

**Player props (main markets):**
- **Moderately efficient**
- Points, rebounds, assists are well-priced
- Still beatable with good data/models
- Books adjust quickly to sharp action

**Why it's efficient:**
- High volume attracts sharp bettors
- Lots of data available publicly
- Books invest in good models
- Quick price corrections
- **BUT:** More opportunity than NFL (more games = more edges to find)
- Props markets still beatable with good data/models

### NFL Efficiency

**Game lines:**
- **EXTREMELY efficient** (<1% of people can beat it long-term)
- Sharp bettors dominate
- Opening lines very sharp
- Closing lines nearly impossible to beat consistently

**Player props:**
- **Less developed** than NBA
- Fewer markets (mainly QB/RB/WR stats)
- Less historical data to model
- Lower limits (books protecting themselves)

**Why it's SO efficient:**
- Massive public interest = massive sharp interest
- Weekly games = more time for price discovery
- Concentrated betting (only 272 games) = sharps focus here
- Books have had decades to perfect NFL models
- Very hard to find edge

**⚠️ YOUR INPUT:**
- Is this why you DON'T focus much on NFL?
- Too efficient to beat consistently?

**⚠️ YOUR INPUT:**
- Have you bet NFL? How does efficiency compare to NBA in your experience?
- Which sport do you think has better opportunities?

---

## Types of Bets Available

### NBA

**Game-level bets:**
- Moneyline (winner)
- Spread (margin of victory)
- Totals (combined score)
- Quarter/Half lines
- Team totals

**Player props (extensive):**
- Points, Rebounds, Assists (main)
- Threes, Steals, Blocks
- Combo props (PRA, PA, PR, RA)
- Double-doubles, Triple-doubles
- First basket scorer
- Alternate lines for all of the above

**Liquidity:**
- Main player props: **Liquid**
- Secondary props: Moderate
- Typical limits: $500-$2,000 on most main props
- **Lower $ per game than NFL** (spread across 1,230 games)

### NFL

**Game-level bets:**
- Moneyline, Spread, Totals
- Quarter/Half lines
- Team totals
- Extremely high limits ($10k+)

**Player props (less extensive):**
- QB: Passing yards, TDs, completions, interceptions
- RB: Rushing yards, TDs, receptions
- WR/TE: Receiving yards, TDs, receptions
- Defense: Sacks, tackles (limited)
- Anytime TD scorer (popular)

**Liquidity:**
- **Game lines: Extremely liquid** ($10k-$50k+ limits)
- **Higher $ per game than NBA** (~$83 avg bet vs $51)
- Player props: Lower limits ($100-$500 typical)
- Concentrated betting (only 272 games vs NBA's 1,230)

**⚠️ YOUR INPUT:**
- Is this why you focus on NBA? Better prop markets?

---

## Data Sources & Quality

### NBA Data Sources

**Used in this codebase:**
- **NBA API** (stats.nba.com) - Game logs, player stats, team stats
- **DraftKings API** - Player props, odds
- **The Odds API** - Multi-book odds aggregation
- **ESPN API** - Scores, schedules (fallback)

**Data quality:**
- **Excellent** for NBA
- Rich play-by-play data available
- Shot charts, tracking data (SportRadar)
- Real-time box scores
- Injury reports (mostly accurate)

**Data challenges:**
- Player name normalization (Jr., III, etc.)
- Team abbreviations vary by source
- Historical data cleaning needed

### NFL Data Sources

**Used in this codebase:**
- **The Odds API** - Spreads, totals, odds
- **Unexpected Points** (Google Sheet) - EPA/CPOE data
- **ESPN** - Schedules, scores

**Data quality:**
- **Fragmented** compared to NBA
- No single authoritative API
- Play-by-play data exists but harder to access
- EPA/CPOE from third parties (Next Gen Stats, PFF)

**Data challenges:**
- Injury reports less reliable (Wednes day/Friday updates)
- Weather data integration needed
- Limited historical play-by-play

**⚠️ YOUR INPUT:**
- Are there other NFL data sources you'd want to use?
- Is data fragmentation why you focus on NBA?

---

## Betting Strategy Differences

### NBA Strategies (Used in This Codebase)

**Player usage-based models:**
- Usage rate correlates with props
- Matchup matters (pace, defense)
- Back-to-backs affect performance
- Garbage time affects stat accumulation

**Monte Carlo simulations:**
- Simulate game outcomes
- Project player stats probabilistically
- Account for variance

**Key factors:**
- Pace (possessions per game)
- Defensive rating of opponent
- Recent performance trends
- Rest days (back-to-back games) - **but priced in**
- Home vs away

**Important:** Most of these factors are already **priced into the lines**. Simply betting based on "high pace game" or "back-to-back" is not profitable without finding where the market is **mispriced**.

### NFL Strategies (Used in This Codebase)

**Luck regression:**
- Teams with extreme turnover differential regress
- Point differential > wins → undervalued
- Unlucky favorites identified via Expected Points Added

**Opponent-specific:**
- Pass defense vs rush defense
- Weather impact on passing/totals
- Home field advantage

**Key factors:**
- Turnovers (high variance, regress to mean)
- EPA vs actual points (luck indicator)
- Strength of schedule
- Rest (TNF, MNF, short weeks)

**⚠️ YOUR INPUT:**
- What NBA strategies have worked best for you?
- Which factors matter most in your models?
- Have you tried NFL strategies beyond what's documented?

---

## Variance & Sample Size

### NBA

**Lower variance:**
- 82 games per season (larger sample)
- More possessions per game (~100)
- Stats more stable (law of large numbers)

**Implication for betting:**
- Easier to identify true talent vs luck
- Models converge faster
- But... smaller edges (market knows this too)

**Sample size needs:**
- 20-30 games of data is meaningful
- Full season (82 games) very robust
- Multi-season data even better

### NFL

**Higher variance:**
- Only 17 games per season
- Fewer possessions per game
- Single events matter more (one turnover swings game)

**Implication for betting:**
- Harder to separate skill from luck
- Models need more seasons of data
- Larger edges possible (market less certain)
- But also more risk

**Sample size needs:**
- Single season is noisy
- 3+ seasons needed for robust analysis
- Even then, roster turnover complicates

**⚠️ YOUR INPUT:**
- Is NBA's lower variance an advantage for modeling?
- Does NFL's higher variance make it harder or create more opportunity?

---

## Timing & Schedule Patterns

### NBA

**Season structure:**
- October - April (regular season)
- April - June (playoffs)
- Games every night (not always same teams)

**Back-to-backs:**
- Common (teams play consecutive nights)
- Affects performance, props
- **Important caveat:** Rest factors are largely **priced into lines**
- Books adjust for back-to-backs (not automatic value)
- Like game context - relationship exists but already incorporated

**Rest patterns:**
- 0 days rest (back-to-back)
- 1 day rest (normal)
- 2+ days rest (well-rested)
- Load management (stars sit strategically)

**YOUR ACTUAL APPROACH:**
- Have **not** incorporated back-to-backs into models yet
- Assume it's priced in (like other factors)
- Focus on areas where you've found actual edges
- Potential future analysis area

### NFL

**Season structure:**
- September - January (regular season)
- January - February (playoffs)
- Games mostly Sundays (some Thu/Mon)

**Short weeks:**
- Thursday Night Football (3 days rest)
- Monday Night Football (6 days rest)
- Affects performance significantly

**Bye weeks:**
- Each team gets one bye (rest) week
- Coming off bye = well-rested (slight advantage)

**⚠️ YOUR INPUT:**
- Do you model rest/schedule patterns in NBA?
- How important are back-to-backs in your analysis?

---

## Key Differences Summary

| Factor | NBA | NFL |
|--------|-----|-----|
| **Games per season** | 82 | 17 |
| **Total season games** | 1,230 | 272 |
| **Betting frequency** | Daily | Weekly |
| **$ per game** | ~$51 avg bet | ~$83 avg bet |
| **Liquidity per game** | Moderate | **Very High** |
| **Market efficiency** | High (but beatable) | **Extremely High** |
| **Player props** | Extensive | Limited |
| **Data availability** | Excellent | Fragmented |
| **Variance** | Lower | Higher |
| **Sample size** | Large | Small |
| **Sharp action** | Very high | **Extremely high** |
| **Public bias** | Moderate | High (but priced in) |
| **Edge opportunities** | **More opportunity** | **<1% can beat long-term** |

**⚠️ YOUR INPUT:** Does this summary match your experience?

---

## Why This Codebase Focuses on NBA

**Reasons (validated by your input):**

1. **More beatable than NFL** (NFL <1% can beat, NBA has more opportunity)
2. **Daily opportunities** (can bet every night vs once a week)
3. **Better data** (NBA API is excellent)
4. **More props available** (player markets more developed)
5. **Lower variance** (easier to model with confidence)
6. **Playable edges exist** (market efficient but still beatable with good models)

**NFL is extremely efficient:**
- <1% of bettors can beat NFL sides long-term
- Too sharp to focus on
- Lower priority given efficiency

**⚠️ YOUR INPUT:**
- Is this accurate? NFL just too efficient to bother with?
- Any other reasons you focus on NBA?

---

## Related Documents

- `docs/domain/betting-fundamentals.md` - Applies to both sports
- `docs/domain/market-mechanics.md` - Efficiency differences explained
- `docs/ARCHITECTURE.md` - See NBA vs NFL code organization
- `backtesting/` - NBA strategies vs NFL strategies

---

## For Agents

**When working with NBA code:**
- Expect high volume, liquid markets
- Prop-focused strategies
- Rich data sources (NBA API)
- Back-to-back and rest factors matter, but all of this should be price in

**When working with NFL code:**
- Lower volume, weekly bets
- Manual data sources (Example: 1 Alt Data source is Unexpected Points data, only available via Google Sheets)
- Variance is higher (be cautious with small samples)

**Key takeaway:** NBA and NFL require different approaches. NBA is data-rich, high-frequency, lower-variance. NFL is data-sparse, low-frequency, higher-variance.

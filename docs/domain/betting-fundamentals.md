# Betting Fundamentals

**Status:** ✅ Complete  
**Last updated:** 2026-02-13  
**Source:** Domain expert knowledge

This document explains core sports betting concepts for AI agents with no betting background.

---

## What is a Player Prop Bet?

A **player prop (proposition) bet** is a wager on an individual player's statistical performance — **not** the game outcome.

### Key Characteristics

**Independent of game result:**
- The team can win or lose — it doesn't matter
- Only the player's stats determine the bet outcome
- Example: Lakers lose 95-110, but LeBron scores 30 points → his "Over 27.5 points" bet still wins

**Differs from game betting:**
- **Moneyline** = which team wins
- **Spread** = margin of victory
- **Props** = individual player performance

**Why people bet player props:**
- More control and research-driven
- Less dependent on team performance
- Can exploit matchup advantages (e.g., weak defender)
- Better for data-driven analysis

**Risks:**
- Injury, foul trouble, or blowout can ruin the bet
- Minutes played matters (starters sit in garbage time)
- Game script affects opportunities

---

## Common NBA Player Prop Markets

From `config/the-odds-api_config.yaml`, these are the available NBA player prop markets:

### Primary Markets (Most Liquid)

**Individual stats:**
- `player_points` - Total points scored
- `player_rebounds` - Total rebounds (offensive + defensive)
- `player_assists` - Total assists
- `player_threes` - Three-pointers made

**Combo props:**
- `player_points_rebounds_assists` (PRA) - Combined total
- `player_points_assists` (PA) - Points + Assists
- `player_points_rebounds` (PR) - Points + Rebounds
- `player_rebounds_assists` (RA) - Rebounds + Assists

### Secondary Markets (Less Liquid)

- `player_blocks` - Blocked shots
- `player_steals` - Steals
- `player_double_double` - 10+ in two stat categories (binary yes/no)
- `player_triple_double` - 10+ in three stat categories (binary yes/no)

### Alternate Lines

- `player_points_alternate` - Same player, different lines and odds
- `player_rebounds_alternate` - Multiple line options
- `player_assists_alternate` - Multiple line options
- Etc. for other stats

**What "alternate" means:**
- Main line: Curry Over 29.5 points at -110
- Alternate: Curry Over 34.5 points at +220 (higher line, better payout)
- Alternate: Curry Over 24.5 points at -220 (lower line, worse payout)

Same stat, different thresholds and prices.

---

## Over/Under Mechanics

Props are structured as **over/under** bets on a line set by the sportsbook.

### Example: Stephen Curry Under 29.5 Points (-110)

**What this means:**
- You're betting Curry scores **29 points or fewer**
- Odds are -110 (explained below)
- If he scores 30 or more, the bet loses
- **Team result doesn't matter** (Warriors can win or lose)

**Outcomes:**
- Curry scores 29 → ✅ **Under wins**
- Curry scores 30 → ❌ **Under loses**
- Curry scores exactly 29.5 → **Impossible** (can't score half points)

### Why Lines Use .5 (Half Points)

**Prevents pushes:**
- 29.5 ensures a clear winner (29 = under, 30 = over)
- You will **never** see whole number lines like 29.0 in player props
- Exception: If you see 29.0, it's a **median value from aggregating multiple books** (e.g., one book has 28.5, another has 29.5)

**Both sides available:**
- You can bet **Over 29.5** or **Under 29.5**
- Odds differ based on market perception
- Typical: -110 on both sides, but can vary

---

## Understanding Odds: American Format

American odds use **+/- notation** to show how much you need to bet to win $100 (or how much you win if you bet $100).

### Negative Odds (Favorite)

**-110:** Risk $110 to win $100
- Total payout if you win: $210 ($110 stake + $100 profit)
- The "-" means this outcome is more likely (favorite)

**-200:** Risk $200 to win $100
- More heavily favored outcome
- Total payout: $300 ($200 + $100)

**-500:** Risk $500 to win $100
- Very heavily favored
- Total payout: $600

**Rule:** Negative odds = amount you risk to win $100

### Positive Odds (Underdog)

**+110:** Risk $100 to win $110
- Total payout if you win: $210 ($100 stake + $110 profit)
- The "+" means this outcome is less likely (underdog)

**+200:** Risk $100 to win $200
- Total payout: $300

**+500:** Risk $100 to win $500
- Total payout: $600

**Rule:** Positive odds = amount you win if you risk $100

### Converting to Implied Probability

**For negative odds:**

```
Implied Probability = |odds| / (|odds| + 100)
```

Examples:
- -110: `110 / (110 + 100) = 0.5238 = 52.38%`
- -200: `200 / (200 + 100) = 0.6667 = 66.67%`
- -500: `500 / (500 + 100) = 0.8333 = 83.33%`

**For positive odds:**

```
Implied Probability = 100 / (odds + 100)
```

Examples:
- +110: `100 / (110 + 100) = 0.4762 = 47.62%`
- +200: `100 / (200 + 100) = 0.3333 = 33.33%`
- +500: `100 / (500 + 100) = 0.1667 = 16.67%`

**Implementation note:** See `src/odds_utils.py` for these conversions.

---

## Vig (Juice): The Sportsbook's Edge

### What is Vig?

**Vig (vigorish) or juice** is the sportsbook's built-in commission on bets.

**Why vig exists:**
- A true 50/50 prop would be priced at **+100 / +100** (even money on both sides)
- Instead, sportsbooks price it at **-110 / -110** (both sides)
- This creates an **overround** where implied probabilities sum to > 100%

**Example:**
- Over 29.5 at -110 = 52.38% implied
- Under 29.5 at -110 = 52.38% implied
- Total: 104.76% (not 100%)

That extra 4.76% is the vig — the sportsbook's edge.

### Typical Vig Rates

**NBA player props:**
- Standard: **-110 / -110** (4-5% vig)
- Can vary: **-115 / -105**, **-120 / +100**, **-125 / -105**

**Heavier vig markets:**
- Alternate lines
- Low-liquidity props (blocks, steals)
- Double-double / triple-double markets
- Same-game parlays (effectively higher due to correlation pricing)

**Why some markets have more vig:**
- Less betting volume → more risk for sportsbook
- Harder to model accurately
- Less sharp action to correct prices

### Why Vig Matters Long-Term

**Break-even rate at -110:**
- Winning 50% of bets at -110 = **you lose money**
- Need to win ~52.38% just to break even
- Your edge must exceed 2.38 percentage points over 50/50

**Example math:**
- 100 bets at $110 each
- Win 50, lose 50 at -110/-110
- Wins: 50 × $100 = $5,000 profit
- Losses: 50 × $110 = $5,500 lost
- Net: **-$500**

**To profit:**
- Must win > 52.38% of bets at -110
- Or find better lines (lower vig) through line shopping

**Key insight:** The goal isn't just to pick winners — it's to **consistently beat the number** by more than the vig.

---

## Other Bet Types: Moneyline, Spread, Totals

These are game-level bets (not player props), but often provide context.

### Moneyline (ML)

**Bet on which team wins outright** (no point spread).

**Example:**
- Lakers -150 vs Clippers +130
- Lakers -150: Risk $150 to win $100 if Lakers win
- Clippers +130: Risk $100 to win $130 if Clippers win

### Spread

**Bet on margin of victory** (point differential).

**Example:**
- Lakers -6.5 vs Clippers +6.5
- Lakers -6.5: Lakers must win by 7+ points
- Clippers +6.5: Clippers can lose by 6 or fewer, or win outright

### Totals (Over/Under)

**Bet on combined score** of both teams.

**Example:**
- Lakers vs Clippers: Over/Under 220.5
- Over 220.5: Both teams score 221+ combined
- Under 220.5: Both teams score 220 or fewer combined

---

## How Game Lines Relate to Player Props

Game-level bets (spread, totals, moneyline) and player props are **different bet types on the same games**.

### Common Hypotheses (Not Guaranteed Edges)

**High total / fast pace:**
- Hypothesis: More possessions → Potentially higher counting stats
- Reality: **Largely priced into prop lines already**

**Big spread (blowout risk):**
- Hypothesis: Blowout → Starters sit in 4Q → Reduced minutes
- Reality: **Books model this; not automatic value**

**Tight spread:**
- Hypothesis: Competitive game → More stable minutes projections
- Reality: **Also priced in**

**Low total / slow pace:**
- Hypothesis: Fewer possessions → Lower stats
- Reality: **You guessed it — priced in**

### Important Caveat ⚠️

**Sportsbooks model pace, spread, and totals when setting player props.**

Simply betting overs in high-total games is **not a proven profitable strategy** on its own.

Game context should **inform your projections**, but it does **not create automatic value** without finding actual mispricing.

**Key distinction:** These relationships exist, but exploiting them requires finding where the market is **wrong** about the relationship, not just knowing the relationship exists.

---

## Same-Game Parlays (SGPs)

### What They Are

**Multiple bets from the same game** where all legs must hit.

**Example:**
- Curry Over 29.5 points **AND**
- Warriors Moneyline (win) **AND**
- Over 220.5 total

All three must win, or the parlay loses.

### Why People Like Them

- **Bigger payouts** (multiply odds together)
- Fun, narrative-based betting
- Can build a "story" for the game

### Why Not to Focus on Them (From Experience)

**Higher effective vig:**
- Books adjust odds for correlation
- You're not getting true independent odds multiplied

**Correlation pricing favors the book:**
- Example: Curry scoring a lot AND Warriors winning are correlated
- Book reduces payout to account for this
- You're paying for correlation you don't benefit from

**Harder to beat long term:**
- Entertainment-heavy, edge-light
- Sharp bettors avoid SGPs
- Recreational bettor trap

**Recommendation:** Focus on straight bets (single wagers) for serious betting. SGPs are entertainment.

---

## Closing Line Value (CLV)

### What is CLV?

**Closing Line Value = Did you beat the final market price?**

The **closing line** is the final odds/line before the game starts (when betting closes).

### Example

You bet: **Curry Under 29.5 at -110**

Closing line: **Curry Under 28.5 at -110**

**Result:** You got the Under at 29.5, but sharp money moved it to 28.5. You have **+1 point of CLV**.

You beat the market by 1 point.

### Why CLV Matters

**Closing line is the sharpest number:**
- It incorporates all available information
- Sharp bettors have moved it to the "correct" price
- Beating the closing line = you had information/edge before the market

**Long-term profitability correlates with positive CLV:**
- If you **consistently beat the closing line**, you likely have an edge
- If you **consistently get worse than closing**, you're likely losing long-term
- CLV is the best predictor of long-term success

**Key metric:**
- Track CLV on all bets
- Positive CLV over large sample = you're doing something right
- Negative CLV = you're chasing bad numbers

**Implementation note:** This repo does not currently track CLV systematically (opportunity for improvement).

---

## Critical Concepts Not Covered Elsewhere

### Bankroll Management

- Flat betting (same $ amount per bet) or percentage of bankroll (1-2%)
- Never bet more than you can afford to lose
- Variance means short-term results ≠ skill

### Sample Size Matters

- 10 bets tells you nothing
- 100 bets starts to show patterns
- 500+ bets needed to evaluate true edge
- Don't overreact to short-term variance

### Market Efficiency Varies

- NBA player props: **Very efficient** (sharp bettors active)
- Low-volume props (blocks, steals): **Less efficient**
- NFL: **Different efficiency** (covered in `nba-vs-nfl.md`)

### Limits Tell You Something

- High limits = sharp market (harder to beat)
- Low limits = softer lines (book protecting itself)
- If you're getting limited, you're probably winning

### Line Shopping is Huge

- Different books have different prices
- Getting -105 instead of -110 = 0.5% edge difference
- Over 100 bets, this compounds significantly
- **Always shop for best price**

---

## Biggest Lesson (From Experience)

> **Your goal isn't to pick winners — it's to consistently beat the number.**

This mindset shift is everything.

**Wrong approach:**
- "I think Curry will score a lot tonight" → Bet Over 29.5

**Right approach:**
- "I think Curry scores 32 on average in this matchup"
- "The line is 29.5"
- "My edge is 2.5 points"
- "At -110, I need ~52.4% to break even"
- "My model says 65% chance of Over"
- "Bet size: 2% of bankroll"

**The number matters more than your opinion.**

---

## Related Documents

- `docs/domain/market-mechanics.md` - Line movement, steam, arbitrage
- `docs/domain/data-quality-standards.md` - What makes props data "good"
- `docs/domain/edge-cases.md` - Postponed games, injuries, etc.
- `docs/domain/nba-vs-nfl.md` - Sport-specific differences

---

## For Agents Working on This Codebase

**When you see:**
- `-110` in data → Risk $110 to win $100
- `player_points` market → Individual player scoring prop
- Line of `29.5` → Over/under threshold (half-point to avoid push)
- "Closing line" in code → Final market price before game starts

**Key formulas in `src/odds_utils.py`:**
- `american_to_probability()` - Convert -110 to 52.38%
- `probability_to_american()` - Convert 52.38% to -110
- `calculate_implied_vig()` - Extract vig from both sides

**Remember:**
- Player props are independent of game outcome
- Vig means you need 52.4% win rate at -110 to break even
- CLV is the best long-term success metric
- Game context is priced in (not automatic value)

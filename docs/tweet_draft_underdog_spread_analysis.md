# Tweet Thread: NBA Underdog Spread Analysis

**Date:** 2026-02-02  
**Topic:** Understanding when underdog covers actually mean wins  
**Data:** 1,796 games across 5 seasons (2021-22 through 2025-26)

---

## Tweet 1: 2025-26 Season Patterns

been tracking nba underdogs this season and found something interesting.

when small underdogs (+1 to +3) cover the spread, they win outright almost every time (90-100%).

when big underdogs (+10+) cover? they still lost most of the time—just by less than the spread (20-30% win rate).

covering ≠ winning. spread size changes everything.

[attach: underdog_spread_line_analysis_by_season.png - zoomed to show 2025-26 bars]

---

## Tweet 2: The Trend Doesn't Hold Over Time

pulled 5 seasons of data to see if this pattern was consistent (2021-22 through 2025-26).

turns out: it's remarkably stable. 

across all 1,796 underdog games:
• +1-3 spreads: 95%+ win when covered
• +4-6 spreads: ~70% win when covered
• +10+ spreads: ~25% win when covered

the relationship between spread size and p(ml win | covered) is structural, not seasonal noise.

[attach: underdog_spread_line_analysis.png - aggregated across all seasons]

---

## Tweet 3: Why This Matters for Betting

here's why this matters:

underdogs are profitable long-term (50.5% cover rate vs 50% breakeven), but you can't just blindly bet every dog and expect to beat the 52.4% threshold for profit.

the insight: small spread dogs (+1-3) give you optionality—when they cover, you're almost certainly winning the ml too. big spread dogs (+10+) are pure spread plays.

if you're doing ml/spread splits, the optimal allocation depends heavily on that spread number. one strategy doesn't fit all underdogs.

---

## Key Data Points

- **Overall underdog stats (5 seasons):**
  - n = 1,796 games
  - ML record: 565-1,231 (31.5%)
  - ATS record: 907-889 (50.5%)
  - Average spread: +6.9
  - Average ML odds: +266

- **P(ML Win | Spread Covered):**
  - Overall: 62.3%
  - +1-3 spread: ~95%
  - +4-6 spread: ~70%
  - +10+ spread: ~25%

---

## Alternative Versions

### Shorter Tweet 3 (if character count is tight):

here's why this matters:

underdogs cover at 50.5% (better than 50%), but to beat the 52.4% profit threshold you need strategy.

small spread dogs (+1-3) are different animals than big spread dogs (+10+). one gives you ml upside when you cover, the other doesn't.

optimal betting depends on that spread number. not all dogs are created equal.

---

### More Technical Tweet 3:

here's why this matters for betting strategy:

underdogs have a slight edge (50.5% ats vs 50% breakeven), but to beat vig (need 52.4%+) you need more than blind dog betting.

the structural relationship between spread size and p(ml win | covered) creates distinct risk/return profiles:

• +1-3 dogs: covering nearly guarantees ml win → can allocate more to spread, less to ml hedge
• +10+ dogs: covering rarely means ml win → pure spread play, ml hedge is expensive insurance

this is why one-size-fits-all kelly sizing fails. the spread size changes the entire payoff structure.

---

## Visual Notes

- **Viz 1 (Tweet 1):** Use by_season chart, potentially crop/highlight just 2025-26 bars
- **Viz 2 (Tweet 2):** Use aggregated chart showing all 5 seasons combined with clear trend line
- Both charts now have clean titles without distracting grey subtitles

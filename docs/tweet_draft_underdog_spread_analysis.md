# Tweet Thread: NBA Underdog Spread Analysis

**Date:** 2026-02-02  
**Topic:** Understanding when underdog covers actually mean wins  
**Data:** 1,796 games across 5 seasons (2021-22 through 2025-26)

---

## Tweet 1: 2025-26 Season Pattern

(1/n) been tracking nba underdogs performance in the 2025-26 season.

underdogs are covering at 52.6% this season, slightly above the expected 50/50, and good enough that blindly betting dogs is (slightly) profitable.

here's what's interesting: when they cover, they win outright 64% of the time outright

[attach: underdog_spread_line_analysis_2025_26.png]

---

## Tweet 2: The Relationship Changes with Spread Size

(2/n) now this should be rather obvious, but the relationship changes with spread size - at small spreads (+1-3), covering means winning 95%+ of the time. at larger spreads (+10+), it drops to just 25%

[attach: underdog_spread_line_analysis_2025_26.png - Chart 3 showing the decline]

---

## Tweet 3: Personal Motivation & Optimization

(3/n) i looked into this because i had noticed that often when my dog plays (i typically take dogs) were covering, they were also winning, but i wasn't capitalizing on the longer odds

this put me down a rabbit hole of trying to optimize the weights (e.g., if you have 1 unit, 0.75 going on the spread, and 0.25 on the ml bet)

---

## Tweet 4: Does This Hold Over Time?

(4/n) does this hold up over time?

pulled 5 seasons of data (2021-22 through 2025-26) to test if this is seasonal noise or something structural.

aggregated results: dogs covering at 50.5% (n=1,796 games), and the p(ml win | covered) pattern is consistent across spread sizes

[attach: underdog_spread_line_analysis_2025_26_2024_25_2023_24_2022_23_2021_22.png]

---

## Tweet 5: Season-by-Season View

(5/n) here's the season-by-season breakdown

there's fluctuation year-to-year (cover rates range from 48% to 54%), but directionally there's something here. the structural relationship between spread size and p(ml win | covered) persists across all 5 seasons

[attach: underdog_spread_line_analysis_by_season_2025_26_2024_25_2023_24_2022_23_2021_22.png]

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

- **Viz 1 (Tweet 1):** underdog_spread_line_analysis_2025_26.png
  - Shows 2025-26 season only
  - P(ML Win | Covered): 64.0%
  - Each chart has its own info box (ML records, ATS records, P(ML|Cov))

- **Viz 2 (Tweet 2):** underdog_spread_line_analysis_2025_26_2024_25_2023_24_2022_23_2021_22.png
  - All 5 seasons aggregated
  - P(ML Win | Covered): 62.3%
  - Each chart has split info boxes with per-season data

- Both charts now have:
  - Clean titles without distracting grey subtitles
  - P(ML Win | Covered) prominently displayed in title
  - Split info boxes: Chart 1 (ML records), Chart 2 (ATS records), Chart 3 (P(ML|Cov))
  - Explanatory note on Chart 3 in bottom right

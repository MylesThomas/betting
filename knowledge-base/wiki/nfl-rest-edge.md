# NFL Rest Edge

**Summary**: Rest edge is a durable betting factor in the NFL. The 2026 schedule has the largest rest inequality since 2000 — team-level net rest data and historical edge values are here.

**Last updated**: 2026-06-29

---

## Definitions

- **Rest edge**: extra days of rest one team has vs. its opponent for a specific game
- **Net rest**: cumulative sum of individual game rest edges across a full season
- **Short-week road game**: team travels for an away game with fewer than 6 days of rest
- **Negated bye**: team's bye week advantage is neutralized because their next opponent is also on extra rest

---

## Historical betting value (per Warren Sharp, 10-year sample)

### Rest edge (3–6 days, NOT off a full bye), after Week 6
- **Cover rate: 54.6%** (4.2% ROI) across 233 games
- Road teams with this edge: **56.1% cover rate (7.0% ROI)** across 134 games

### Short-week road games (< 6 days rest), after Week 6
- Win rate: 43.9%, **cover rate: 47.4% (-9.4% ROI)**
- Same teams with extra rest instead (flip one variable): 46.9% wins, **53.3% covers (+1.8% ROI)**
- **Swing: 11.2% ROI** just from rest direction

### After road Sunday Night / Monday Night Football game
- **Cover rate: 49.0%** since 2013 (fade these teams the following week)

**Rule**: Rest edge kicks in most from Week 6 onward. A 3-6 day advantage (not off a full bye) is the sweet spot. Full-bye rest edges are valuable but often priced in; the 3-6 day mid-week edge is less visible and more exploitable.

---

## 2026 NFL Net Rest Rankings

2026 has the largest rest swing since 2000: **+15 days (Bears) to -24 days (Chargers) = 39-day delta** — #1 largest in 27 years. 110 of 272 games (40%) have a rest advantage — most in NFL history.

| Team | Net Rest |
|------|----------|
| Chicago Bears | +15 |
| Buffalo Bills | +14 |
| Seattle Seahawks | +12 |
| Dallas Cowboys | +9 |
| Washington Commanders | +9 |
| Carolina Panthers | +8 |
| Houston Texans | +8 |
| New England Patriots | +8 |
| Atlanta Falcons | +7 |
| Tennessee Titans | +6 |
| Minnesota Vikings | +4 |
| Denver Broncos | +3 |
| New York Giants | +3 |
| Arizona Cardinals | +1 |
| Detroit Lions | +1 |
| Cleveland Browns | 0 |
| San Francisco 49ers | 0 |
| Green Bay Packers | -2 |
| Kansas City Chiefs | -2 |
| Tampa Bay Buccaneers | -3 |
| Baltimore Ravens | -3 |
| Jacksonville Jaguars | -3 |
| Cincinnati Bengals | -4 |
| Indianapolis Colts | -6 |
| Los Angeles Rams | -6 |
| Miami Dolphins | -6 |
| Pittsburgh Steelers | -6 |
| New Orleans Saints | -7 |
| New York Jets | -9 |
| Las Vegas Raiders | -13 |
| Philadelphia Eagles | -15 |
| Los Angeles Chargers | -24 |

### Teams with 5 rest-advantage games (most in NFL) — favor on futures/win totals
Bears, Seahawks, Bills, Ravens, Broncos, Cardinals, Rams, Jets

### Teams with 5+ rest-disadvantage games — fade on futures/win totals
Chargers (7 games), Eagles / Rams / Packers / Chiefs / Steelers / Dolphins (5 games each)

---

## 2026 Key Schedule Situations

### Buffalo Bills — the most disadvantaged team in 2026
- **Only team with 3 short-week road games** (Weeks 6, 10, 16)
- **Only team with 4 games in 17 days**: MNF @ MIN (Wk10) → @ NYJ (Wk11) → vs MIA (Wk12) → vs KC (Wk13, TNF)
- The 4-game stretch ends with a Chiefs game likely deciding AFC playoff seeding
- **Fade the Bills in all three short-week road spots; consider fading vs. Chiefs in Week 13**

### Los Angeles Chargers — chronically abused by the schedule
- -24 net rest days — largest rest disadvantage in NFL since 2013
- 4-year pattern: 2023 net -4, 2024 net -6, 2025 net -1, 2026 net -4 (Chargers consistently screwed)
- **Fade Chargers in win total futures relative to their talent level**

### Philadelphia Eagles — hidden disadvantage
- -15 net rest, 5 games with rest disadvantage
- NFC East rivals: Commanders have 1 rest-disadvantage game, Cowboys have 1
- Eagles competing for division title with structurally worse schedule timing than both rivals

### Negated byes in 2026
Only 3 teams have negated byes: **Rams, Packers, Titans** (down from 10 in 2025). 
- Rams/Packers bye in Week 11 for the Week 12 Wednesday Thanksgiving Eve game — both negated
- Titans bye Week 9, face Jaguars in Week 10 (Jaguars coming off TNF mini-bye — also negated)

---

## 2025 Historical Validation

Teams with +8 or better net rest in 2025: Seahawks (14W), Rams (12W), 49ers (12W), Lions (9W), Dolphins (7W) — **avg 10.8 wins**

Teams with -8 or worse net rest in 2025: Raiders (3W), Commanders (5W), Saints (6W) — **avg 4.7 wins**

6-win swing between best and worst rest cohorts.

---

## Backtest: net rest vs preseason win total O-U (negative result)

**Tested**: 2010–2025, 489 team-seasons (511 total; IND 2011 excluded — no line set after Manning's injury). Source: sportsoddshistory.com win total lines + nfl_data_py schedules.

**Finding**: Net rest does not predict whether a team goes over or under its preseason win total. Every 5-unit bucket with meaningful sample size (n ≥ 24) sits within 44–51% over rate — indistinguishable from the dataset average of 47.4%. Pearson r = +0.056, p = 0.30.

**Why**: Books price schedule difficulty — including rest — into preseason win totals. By the time lines are set, the rest edge is already partially reflected.

**Implication**: Do not use net rest as a standalone signal for win total futures. The edge lives at the **game level** (weekly spread bets), not in preseason season-long bets.

**Visualization**: `knowledge-base/wiki/nfl-rest-edge-backtest.html`

---

## Related

- [[nfl-2026-season-context]]
- [[american-odds]]

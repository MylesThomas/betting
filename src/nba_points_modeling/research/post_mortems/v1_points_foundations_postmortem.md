# Post-Mortem: v1 Points Foundations

**Date:** 2026-05-21
**Notebook:** `research/notebooks/v1_points_game_to_game_foundations.ipynb`
**Verdict:** No profitable edge found. Market is too efficient on points.

---

## What We Built

Game-to-game points modeling across 3 seasons (2023-24 → 2025-26), ~36k player-game rows.

Models tried: OLS, Ridge, GBM (regression) · LogReg, XGBoost (classification)
Best model: LogReg (rolling pts + line) — Brier 0.24906 vs market 0.24974

Walk-forward ROI: Fold 1 (test 2024-25) -5% to -7%. Fold 2 (test 2025-26) ~flat. Does not overcome 6.9% vig.

---

## Why It Failed

### 1. Market is nearly efficient on points

The line alone captures 97.97% of GBM feature importance. OLS with rolling features gets RMSE 6.46 vs naive (line) 6.47 — delta is noise. Books price points off season averages + opponent defensive rating + pace + usage — essentially the same rolling stats you'd build. You're not bringing information the market doesn't have.

Compare to rebounds: baseline (no line) RMSE 2.68 → OLS with features 2.48, a 0.20 improvement that survived three OOS seasons. Points had no equivalent gap to exploit.

### 2. Section 6 test set was playoffs-only

~1,600 rows from Apr–May. Any ROI at n=14–51 is coin-flip noise. The "18% XGB ROI" at min_edge=0.05 is a single month of playoffs — not a finding. Full-season OOS is required before any verdict is meaningful.

### 3. Feature set didn't target what the market gets wrong

Features used: rolling pts, minutes, FTAs, spread, shot zone, home/away. Missing: opponent defensive rating, pace, usage%, rest, team scheme changes, on/off splits. Adding these likely wouldn't have helped — the market already prices them — but the feature list was built around "what correlates with points" rather than "what does the market systematically miss."

---

## Comparison: Why Rebounds Worked

| Factor | Points | Rebounds |
|---|---|---|
| Market efficiency | High — line = 97.97% of signal | Lower — rotation/role lag exploitable |
| Book line agreement | Near-identical across books | Disagreement (line_width) was a real feature |
| OOS test set | Playoffs only, ~1,600 rows | Full seasons, ~10k rows per fold |
| Best rolling window | roll5/roll10 (too short) | roll_mean_60 (near-full-season) |
| ROI (pooled 3 seasons) | N/A — test too thin | +2.6% at min_edge=0.05, +ROI all 3 folds |
| Feature gap | Negligible RMSE improvement | 0.20 RMSE gap vs baseline |

Rebounds worked because it's a role/rotation-sensitive stat where books lag real-world changes (injuries, lineup shifts, matchups). Points is a deeper market with tighter lines — the same information is already priced in.

---

## Lessons for New Markets

**The core question is not "what correlates with X?" — it's "what does the market systematically miss about X?"**

| Rule | Rationale |
|---|---|
| Check book line disagreement first | `line_width = max - min` was a real rebounds feature. High disagreement = genuine uncertainty = opportunity. Points lines are nearly identical across books. |
| Use longer rolling windows | roll_mean_60 > roll_mean_5 for rebounds. For low-count stats (steals ~0.8/game), use 20–40 game windows minimum. |
| Walk-forward requires full-season OOS | Playoffs-only test invalidated all points ROI figures. Minimum: 500+ bets per threshold before trusting any ROI number. |
| Regression + Normal CDF before classification | OLS + parametric probability outperformed LogReg/XGB on rebounds. Fit the stat on raw scale, convert to P via Normal CDF. |
| Check under-only asymmetry | Rebounds under-only: +2.6% pooled, 3/3 seasons. Markets tend to overprice popular stats — check the under side first. |

---

## Next Steps for Points (if revisiting)

The only signal found: **rim attackers hit unders at 57% vs 51% overall** (910 games, avg line 10.94 vs 14.29 for perimeter).

That's a situational filter, not a model. Test it as a standalone rule in `v2_points_rim_attacker_filter.ipynb`:
- Walk-forward 3 seasons (2023-24 → 2025-26)
- Filter: pts_0_6_pct ≥ 0.40 + under only
- Expected n_bets: ~300/season — enough to get a read by fold 2

If that fails, points modeling is likely dead until we have access to usage%, pace, or defensive matchup data not in the current pipeline.

---

## Market Prioritization Going Forward

| Market | Expected efficiency | Key driver of edge | Priority |
|---|---|---|---|
| Rebounds | Low (proven) | Rotation/role lag | Continue → v2 |
| Assists | Low-medium | Ball-handler role changes, pace | Try next |
| Blocks | Medium | Matchup-sensitive, low base rate | After assists |
| Steals | Medium | High variance, under pricing | After assists |
| Points | High | Market already efficient | Deprioritize |

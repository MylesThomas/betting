# Active Execution Plans

This directory contains work currently in progress.

## Current Active Plans

### Player team history + box scores pipeline extension (2026-03-05)

**File:** [nba-player-threes-model-v2-plan.md](nba-player-threes-model-v2-plan.md)  
**Goal:** Extend `src/player_team_history/01_build.py` into a two-artifact pipeline that writes both `history.parquet` (team stints) and `box_scores.parquet` (player/game rows with `PLAYER_INFO_*` metadata) using shared normalization + cache logic.  
**Next:** Run sample build (`--sample 100`), run larger sample (`--sample 1000`), then validate with `python src/player_team_history/04_validate.py` and inspect cache/artifacts with `python src/player_team_history/03_cache.py --stats`.

---

### NBA player_threes model v2 plan (2026-03-05)

**File:** [nba-player-threes-model-v2-plan.md](nba-player-threes-model-v2-plan.md)  
**Goal:** Launch a v2 run with a 3-input mean model (`mean_3pm`, `predicted_3pa`, `predicted_minutes`) and recency-weighted player-history uncertainty sampling in the probability engine.  
**Next:** Run and validate v2 (`v2_three_input_regression` + `v2_weighted_history_sampler`), then compare deltas vs latest stable v1 baseline and tune recency parameters (`history_n`, `decay_alpha`).

---

### NBA player_threes model v1 plan (2026-03-04)

**File:** [nba-player-threes-model-v1-plan.md](nba-player-threes-model-v1-plan.md)  
**Goal:** Build a modular `01 -> 02 -> 03 -> 04` workflow for `player_threes`: mean model (`01_signal_discovery`), uncertainty/pricing (`02_probability_engine`), strategy simulation (`03_backtesting`), and run comparison (`04_validation`).  
**Next:** Scaffold `src/nba_three_point_modeling/` modules using locked v1 defaults (no longshot-bias correction, median in implied-probability space + best contract views, `is_consensus` flag, `$100` target-win staking, standard `run_id` format).

---

### NBA multimarket strategy analysis (2026-02-25)

**File:** [nba-multimarket-strategy-analysis.md](nba-multimarket-strategy-analysis.md)  
**Goal:** Top-down analysis on the unified strategy parquet: which variables matter per market, then derive and backtest signals.  
**Next:** Add script in `analysis/` to load parquet, summarize hit rate/error by market, segment by spread and line tier.

---

### Agent-First Repository Transformation (2026-02-13)

**File:** `IMPLEMENTATION_PLAN.md` (root of repo)

**Status:** Phase 1 in progress  
**Goal:** Transform repo for agent-first development  
**Owner:** Human + AI  
**Timeline:** Phased approach over coming weeks

---

## When to Create an Execution Plan

Create a plan for:
- Multi-day efforts
- Cross-cutting changes (touching many domains)
- Architectural decisions that need tracking
- Work that might be interrupted and resumed later

Don't create a plan for:
- Single-file changes
- Bug fixes
- Simple additions

## Plan Template

```markdown
# [Plan Title]

**Created:** YYYY-MM-DD  
**Owner:** Name  
**Status:** In Progress / Blocked / Complete

## Goal
What are we trying to achieve?

## Context
Why is this needed? What's the current problem?

## Approach
How will we do this?

## Tasks
- [ ] Step 1
- [ ] Step 2
- [ ] Step 3

## Progress Log
### 2026-02-13
- Completed X
- Started Y
- Blocked on Z

## Completion Criteria
How do we know we're done?

## Related
Links to design docs, issues, PRs
```

---

When plan is complete, move to `exec-plans/completed/` with completion date in filename.

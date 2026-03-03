# Active Execution Plans

This directory contains work currently in progress.

## Current Active Plans

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

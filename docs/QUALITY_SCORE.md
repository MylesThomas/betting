# Quality Score

**Last updated:** 2026-02-13  
**Auto-updated by:** `scripts/update_quality_score.py` (planned)

This document tracks the health of each architectural domain and layer.

## Overall Health: 🟡 Fair

Current focus: Building agent-first infrastructure (documentation, boundaries, validation)

---

## By Domain

| Domain | Grade | Key Issues | Trend | Priority |
|--------|-------|------------|-------|----------|
| **Data Ingestion** | C+ | Lambda functions scattered, no validation harnesses | → | High |
| **Storage** | B | Structure good, but no schemas/validation | → | Medium |
| **Analysis** | C | Hardcoded thresholds, missing tests, no boundaries | → | High |
| **Utilities** | B- | Good coverage, but some business logic leakage | → | Medium |
| **Presentation** | B+ | Streamlit app works well, limited scope | → | Low |
| **Documentation** | D → B+ | Major restructure in progress! | ↑ | High |

---

## By Quality Metric

| Metric | Current | Target | Status | Notes |
|--------|---------|--------|--------|-------|
| **Test Coverage** | ~35% | 80% | 🟡 | Critical paths need tests |
| **Doc Freshness** | N/A | <90 days | 🟡 | New structure just created |
| **Architectural Violations** | Unknown | 0 | 🔴 | No enforcement yet |
| **Linter Pass Rate** | ~95% | 100% | 🟢 | Black, isort working |
| **Hardcoded Values** | High | Low | 🔴 | Many magic numbers |
| **Validation Coverage** | 0% | 100% | 🔴 | No harnesses exist |

---

## Detailed Domain Assessments

### Data Ingestion: C+

**Strengths:**
- ✅ Lambda for `nba_player_props_ingest` is well-structured
- ✅ S3 integration works reliably
- ✅ Scheduled fetching via EventBridge

**Weaknesses:**
- ❌ Lambda functions scattered across `scripts/`, `docs/`, `lambda/`
- ❌ No validation of fetched data
- ❌ No structured logging
- ❌ Error handling inconsistent
- ❌ Some lambdas haven't been touched in months

**Top 3 Improvements:**
1. Consolidate all lambdas to `lambda/<name>/lambda_function.py`
2. Add `scripts/validate_props_data.py` harness
3. Implement structured JSON logging

**Blocking:** Phase 0.4, Phase 4.1, Phase 4.2 of IMPLEMENTATION_PLAN.md

---

### Storage: B

**Strengths:**
- ✅ Clear 01→04 pipeline structure
- ✅ S3 mirroring works well
- ✅ `src/s3_utils.py` is clean and tested

**Weaknesses:**
- ❌ No schemas for data files (hard to validate)
- ❌ No automated cleanup of old data
- ❌ Some confusion about 02 vs 03 usage

**Top 3 Improvements:**
1. Add schema metadata to JSON outputs
2. Document data retention policy
3. Clarify when to use 02_cache vs 03_intermediate

**Blocking:** Phase 4.4 of IMPLEMENTATION_PLAN.md

---

### Analysis: C

**Strengths:**
- ✅ Backtesting framework exists
- ✅ Some strategies well-documented (3PT unders, NFL luck)
- ✅ Monte Carlo implementation solid

**Weaknesses:**
- ❌ Hardcoded thresholds everywhere (0.05, 7 days, etc.)
- ❌ No architectural tests (could import from lambda/)
- ❌ Missing tests for critical logic
- ❌ Some scripts do their own data fetching (boundary violation)
- ❌ Duplicate helper functions across scripts

**Top 3 Improvements:**
1. Move all thresholds to config YAMLs
2. Add `tests/test_architecture.py` to enforce boundaries
3. Extract duplicate helpers to `src/` utilities

**Blocking:** Phase 3 and Phase 5.1 of IMPLEMENTATION_PLAN.md

---

### Utilities: B-

**Strengths:**
- ✅ Good coverage of common needs (odds, names, config)
- ✅ Most functions are pure/testable
- ✅ Clear module organization

**Weaknesses:**
- ❌ `src/pbp_data/` is actually analysis, not utilities
- ❌ Some utilities have optional business logic
- ❌ Missing docstrings on some functions
- ❌ No clear guidance on when to add to `src/` vs inline

**Top 3 Improvements:**
1. Move `src/pbp_data/` to `analysis/pbp_data/`
2. Add docstrings to all public functions
3. Document criteria for "when is something a utility?"

**Blocking:** Phase 3.1 dependency boundaries clarification

---

### Presentation: B+

**Strengths:**
- ✅ Streamlit dashboard works well
- ✅ Clean separation from analysis
- ✅ Good UX for viewing opportunities

**Weaknesses:**
- ❌ No automation for bet placement yet
- ❌ Limited visualization options
- ❌ Could benefit from more interactivity

**Top 3 Improvements:**
1. Add more chart types (line movement over time)
2. Add export to CSV functionality
3. Add bet tracking (placed vs won/lost)

**Blocking:** Low priority

---

### Documentation: D → B+ (In Progress!)

**Previous State (D):**
- ❌ Scattered across README, docs/, books/, tmp/
- ❌ AI_summaries folder not agent-optimized
- ❌ No clear entry point for agents
- ❌ Betting domain knowledge not captured

**Current State (B+):**
- ✅ New docs/ structure created
- ✅ AGENTS.md as clear entry point
- ✅ ARCHITECTURE.md provides system map
- ✅ Design docs started (core-beliefs, dependency-boundaries)
- 🚧 Still need betting domain docs
- 🚧 Still need to migrate existing docs
- 🚧 Still need validation docs

**Top 3 Improvements:**
1. Write betting domain knowledge docs (Phase 2)
2. Migrate existing docs to new structure
3. Create validation harness docs

**Blocking:** Phase 1 (in progress), Phase 2 (next)

---

## Health Trends

### Last 7 Days
- 📈 Documentation: D → B+ (major restructure)
- → All other domains: No change

### Last 30 Days
- 📈 Storage: B- → B (improved S3 handling)
- 📉 Analysis: B- → C (technical debt accumulating)

---

## Immediate Action Items

1. **Finish Phase 1** - Complete documentation structure ⏳ In Progress
2. **Start Phase 2** - Document betting domain knowledge
3. **Lambda consolidation** - Move scattered lambdas to proper structure
4. **Architecture tests** - Enforce layer boundaries

---

## Success Criteria by Phase

**Phase 1-2 Complete:**
- [ ] All domains at B- or better for documentation
- [ ] Betting knowledge fully captured
- [ ] Clear entry point for agents

**Phase 3 Complete:**
- [ ] Analysis domain at B or better
- [ ] Zero architectural violations detected
- [ ] No circular dependencies

**Phase 4 Complete:**
- [ ] Data Ingestion at B+ or better
- [ ] Validation coverage at 80%+
- [ ] All critical paths have harnesses

**Phase 5 Complete:**
- [ ] All domains at B+ or better
- [ ] Automated quality tracking working
- [ ] Zero high-priority tech debt

---

## Notes

This quality score is **human + agent maintained**. The goal is to make quality visible and track improvements over time.

As we build `scripts/update_quality_score.py`, this will become increasingly automated (test coverage %, linter pass rate %, etc.)

---

**Last human review:** 2026-02-13  
**Next review:** 2026-02-20

# Test Coverage Targets

**Status:** 📝 Planned  
**Last updated:** 2026-02-13

This document will specify test coverage goals by domain.

## Coverage Targets by Domain

| Domain | Current | Target | Priority |
|--------|---------|--------|----------|
| Utilities (`src/`) | ~60% | 90% | High |
| Data Ingestion (`lambda/`, `scripts/fetch_*`) | ~20% | 70% | High |
| Analysis (`analysis/`, `backtesting/`) | ~30% | 80% | Medium |
| Storage (`src/s3_utils.py`) | ~80% | 95% | Low (already good) |

## Critical Paths (Must Have Tests)

**High priority:**
- [ ] `src/odds_utils.py` - Odds conversions
- [ ] `src/player_name_utils.py` - Name matching
- [ ] `src/config_loader.py` - Config loading
- [ ] `lambda/nba_player_props_ingest/lambda_function.py` - Main ingestion
- [ ] `scripts/validate_props_data.py` - Validation harness

**Medium priority:**
- [ ] `src/team_utils.py` - Team normalization
- [ ] `src/nba_gamelog_utils.py` - Game log helpers
- [ ] `analysis/*/` - Strategy finders

**Low priority:**
- [ ] Ad-hoc analysis scripts
- [ ] Visualization code
- [ ] One-off explorations

## Test Types

**Unit tests:**
- All utilities in `src/`
- Pure functions with no side effects
- Target: 90% coverage

**Integration tests:**
- Data pipeline flows (ingestion → storage → analysis)
- Lambda handlers (mock AWS services)
- Target: Key paths covered

**Validation tests:**
- Data quality checks
- Schema validation
- Target: All validation rules tested

## Exclusions

Not requiring tests for:
- Config files (YAML)
- Documentation
- Exploratory notebooks
- Ad-hoc scripts in `tmp/`

---

**To be written in:** Phase 4 (Application Legibility)  
**Tracked by:** `scripts/update_quality_score.py` (planned)

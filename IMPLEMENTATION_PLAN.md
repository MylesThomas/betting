# Agent-First Repository Improvement Plan

**Context:** Transform this betting repo to be optimized for AI agent development following OpenAI's Codex harness engineering principles.

**Goal:** Enable agents to effectively understand, maintain, and extend the codebase with minimal human intervention by making knowledge discoverable, boundaries enforceable, and work self-validating.

---

## Phase 0: Pre-Flight Cleanup (FIRST)

**Status:** Not Started  
**Estimated effort:** 30 minutes  
**Impact:** Clear space for new structure

### 0.1 Remove AI_summaries

**Action:** Delete `docs/AI_summaries/` directory

**Reasoning:** Written before agent-first mindset, no longer aligned with new approach

```bash
rm -rf docs/AI_summaries/
```

### 0.2 Catalog Books Directory

**Action:** Create `books/README.md` to explain what this is

**Content:**
```markdown
# Reference Materials

This directory contains notes and transcriptions from external sources.

## Monte Carlo Betting References

- **monte_carlo_or_bust.md** - Notes from "Monte Carlo or Bust" by Joseph Buchdahl
- **MONTE_CARLO_IMPLEMENTATION_STRATEGY.md** - Implementation notes
- **monte_carlo_implementation_guide.md** - Guide for applying concepts
- **monte_carlo_prediction_models.md** - Prediction model notes

These are reference materials from external sources, not design decisions for this codebase.
For actual implementation decisions, see `docs/design-docs/`.
```

### 0.3 Document Current State

**Action:** Take snapshot of what exists now for reference

```bash
# Generate structure snapshot
tree -L 3 -I '.venv|__pycache__|.pytest_cache' > docs/repo_structure_before.txt
```

### 0.4 Reorganize Lambda Functions

**Problem:** Lambda functions scattered across `scripts/`, `docs/`, and `lambda/`

**Current state:**
- ✅ GOOD: `lambda/nba_player_props_ingest/lambda_function.py` - proper structure
- ❌ BAD: 6 lambda functions in `scripts/lambda_function_*.py`
- ❌ BAD: 3 lambda functions in `docs/lambda_function_*.py`

**Target structure:**
```
lambda/
├── nba_player_props_ingest/
│   └── lambda_function.py
├── refresh_strategy_statistics/
│   └── lambda_function.py
├── all_sports_line_steam_alerts_tracking/
│   └── lambda_function.py
├── nba_player_scoring_props/
│   └── lambda_function.py
├── track_live_odds/
│   └── lambda_function.py
├── track_game_line_movements/
│   └── lambda_function.py
├── game_results_fetcher/
│   └── lambda_function.py
├── nba_alerts/
│   └── lambda_function.py
├── dashboard/
│   └── lambda_function.py
└── nfl/
    └── lambda_function.py
```

**Action:**
1. Move each `lambda_function_*.py` to `lambda/<name>/lambda_function.py`
2. Update any imports/references
3. Document in `lambda/README.md` which folder maps to which AWS Lambda
4. Delete old files after migration

---

## Phase 1: Knowledge Architecture Restructure (HIGHEST PRIORITY)

**Status:** Not Started  
**Estimated effort:** 2-3 focused sessions  
**Impact:** Unlocks all other improvements

### 1.1 Create New Documentation Structure

**Action:** Build `docs/` as system of record with clear hierarchy

```
docs/
├── index.md                           # Master navigation - what's where
├── design-docs/                       # Architectural decisions
│   ├── index.md                       # Design doc catalog
│   ├── core-beliefs.md                # Agent-first operating principles
│   ├── data-pipeline-architecture.md  # Why 01-04 stage model
│   ├── dependency-boundaries.md       # Layer rules and constraints
│   └── why-uv-not-pip.md             # Tooling choices
├── exec-plans/                        # Work tracking
│   ├── active/                        # Current work streams
│   ├── completed/                     # Historical context
│   └── tech-debt-tracker.md          # Known issues catalog
├── domain/                            # Betting domain expertise
│   ├── betting-fundamentals.md       # Props, moneyline, spreads, O/U
│   ├── market-mechanics.md           # Vig, line movement, steam
│   ├── edge-cases.md                 # Postponed games, injuries, etc.
│   ├── data-quality-standards.md     # What makes data "good"
│   └── nba-vs-nfl.md                 # Sport-specific differences
├── references/                        # External dependencies
│   ├── draftkings-api-llms.txt       # DK API docs (LLM-optimized)
│   ├── the-odds-api-llms.txt         # Odds API reference
│   ├── nba-api-llms.txt              # NBA stats API
│   ├── aws-lambda-patterns-llms.txt  # Lambda best practices
│   └── uv-package-manager-llms.txt   # uv commands reference
├── validation/                        # Quality standards
│   ├── data-validation-rules.md      # What to check in props data
│   ├── test-coverage-targets.md      # Coverage by domain
│   └── performance-benchmarks.md     # Speed/memory thresholds
├── generated/                         # Auto-generated docs
│   ├── config-schema.md              # Auto-doc of YAML configs
│   └── module-dependency-graph.md    # Import relationships
├── ARCHITECTURE.md                    # Top-level system map
├── QUALITY_SCORE.md                  # Domain/layer quality grades
├── DATA_PIPELINE.md                  # 01-04 stage documentation
├── RELIABILITY.md                    # Error handling, retries, alerts
└── SECURITY.md                       # API keys, secrets, data privacy
```

**Key files to create:**

1. **`docs/index.md`** - Master navigation
2. **`docs/ARCHITECTURE.md`** - High-level system map
3. **`docs/domain/betting-fundamentals.md`** - Core betting concepts
4. **`docs/design-docs/core-beliefs.md`** - Agent-first principles
5. **`docs/QUALITY_SCORE.md`** - Track quality by domain

### 1.2 Create Lean AGENTS.md (Table of Contents)

**Action:** Replace or create short AGENTS.md (~100 lines) that serves as entry point

**Structure:**
```markdown
# AGENTS.md

This repository contains data-driven NBA/NFL betting strategies.
You are an AI agent working on this codebase.

## Navigation

- **Architecture:** See `docs/ARCHITECTURE.md` for system overview
- **Domain knowledge:** See `docs/domain/` for betting concepts
- **Design decisions:** See `docs/design-docs/` for why things are built this way
- **Quality tracking:** See `docs/QUALITY_SCORE.md` for current state
- **Active work:** See `docs/exec-plans/active/` for current tasks
- **API references:** See `docs/references/` for external dependencies

## Core Principles (see docs/design-docs/core-beliefs.md)

1. **Fail fast:** Don't check for keys that should exist - let it fail
2. **No fake data:** Never create mock/test data without explicit permission
3. **Readable paths:** Use config/root detection, not relative parent paths
4. **Explicit over implicit:** Check existence only for optional items
5. **Agent legibility first:** Optimize for discoverability, not human aesthetics

## Key Domains

- **Data Ingestion:** `lambda/`, `scripts/fetch_*`
- **Data Storage:** `data/01_input/` → `04_output/`
- **Analysis:** `analysis/`, `backtesting/`
- **Utilities:** `src/`

See `docs/ARCHITECTURE.md` for full dependency rules.

## Before Making Changes

1. Read relevant design docs in `docs/design-docs/`
2. Check `docs/QUALITY_SCORE.md` for domain health
3. Review active plans in `docs/exec-plans/active/`
4. Understand betting domain from `docs/domain/`

## Testing & Validation

- Run validation harnesses (TBD)
- Check data quality rules in `docs/validation/`
- Ensure linters pass (see .cursor/rules/)

## Getting Help

- Betting concepts: `docs/domain/`
- API usage: `docs/references/`
- Architecture questions: `docs/design-docs/`
```

### 1.3 Migrate Existing Documentation

**Action:** Reorganize existing docs into new structure

**Current state analysis:**
- `docs/` has ~20 files, mostly AWS setup guides and strategy summaries
- `books/` has monte carlo implementation docs
- `.cursor/rules/` has coding standards
- Scattered READMEs in subdirectories

**Migration tasks:**

1. **Move AWS docs** → `docs/references/aws-*.md`
2. **Move strategy summaries** → `docs/exec-plans/completed/`
3. **Consolidate monte carlo docs** → `docs/design-docs/monte-carlo-approach.md`
4. **Extract betting knowledge** from your head → `docs/domain/`
5. **Keep `.cursor/rules/`** as-is (Cursor-specific)
6. **Update all README files** to point to new docs structure

### 1.4 Create Automated Documentation

**Action:** Build scripts that generate docs from code

**Scripts to create:**

1. **`scripts/generate_config_docs.py`**
   - Parse all YAML configs
   - Generate `docs/generated/config-schema.md`
   - Run on pre-commit hook

2. **`scripts/generate_dependency_graph.py`**
   - Analyze imports across codebase
   - Generate `docs/generated/module-dependency-graph.md`
   - Detect circular dependencies

3. **`scripts/validate_docs.py`**
   - Check all doc links are valid
   - Ensure index files are up-to-date
   - Run in CI

---

## Phase 2: Domain Expertise Encoding (CRITICAL FOR AGENTS)

**Status:** Not Started  
**Estimated effort:** 2-3 sessions  
**Impact:** Agents can reason about betting without guessing

### 2.1 Document Betting Fundamentals

**Action:** Create `docs/domain/betting-fundamentals.md`

**Contents:**
- What is a prop bet? (player props vs game props)
- Moneyline, spread, over/under mechanics
- How odds work (American, decimal, implied probability)
- What is "vig" and why it matters
- What is "closing line value"
- Common bet types: parlays, teasers, same-game parlays

### 2.2 Document Market Mechanics

**Action:** Create `docs/domain/market-mechanics.md`

**Contents:**
- What is line movement and why it happens
- What is "steam" (coordinated sharp action)
- How books set opening lines
- Why lines differ across books
- Arbitrage opportunities (when/why they exist)
- Market efficiency by sport (NBA vs NFL)

### 2.3 Document Data Quality Standards

**Action:** Create `docs/domain/data-quality-standards.md`

**Contents:**
- What makes prop bet data "good"?
  - Timestamp freshness (< 5 min old for live lines)
  - Odds range sanity checks (-10000 to +10000)
  - Required fields must be present
- When is line movement significant? (>10 cents for totals, >0.5 for spreads)
- Red flags in data:
  - Stale timestamps
  - Missing player IDs
  - Odds stuck at open across multiple fetches
  - Player listed but marked inactive

### 2.4 Document Edge Cases

**Action:** Create `docs/domain/edge-cases.md`

**Contents:**
- Postponed games (props void or rollover?)
- Late scratches (how to handle player ruled out)
- Line freezes (books stop taking action)
- Stat corrections (official scorer changes ruling)
- Overtime handling (props include OT unless specified)
- What happens during injuries mid-game

### 2.5 Document Sport-Specific Patterns

**Action:** Create `docs/domain/nba-vs-nfl.md`

**Contents:**
- NBA: High volume, liquid markets, back-to-backs matter
- NFL: Weekly, lower volume, injury reports critical
- NBA: Player props widely available
- NFL: Team totals more common than player props
- Market efficiency differences
- Data sources per sport

### 2.6 Create Example Data Fixtures

**Action:** Create `tests/fixtures/` with annotated examples

**Files:**
```
tests/fixtures/
├── good_prop_data.json          # Well-formed DK props response
├── bad_prop_data_stale.json     # Stale timestamps
├── bad_prop_data_missing.json   # Missing required fields
├── good_line_movement.json      # Significant steam move
├── insignificant_line_move.json # Normal market noise
└── README.md                    # Explains what makes each good/bad
```

---

## Phase 3: Architectural Boundaries & Enforcement (PREVENT DECAY)

**Status:** Not Started  
**Estimated effort:** 3-4 sessions  
**Impact:** Prevent agent from creating tangled dependencies

### 3.1 Define Layer Architecture

**Action:** Create `docs/design-docs/dependency-boundaries.md`

**Proposed layers:**

```
┌─────────────────────────────────────────────────────────┐
│                   BUSINESS DOMAINS                       │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  DATA INGESTION DOMAIN                           │  │
│  │  - lambda/nba_player_props_ingest/               │  │
│  │  - scripts/fetch_*.py                            │  │
│  │  Dependencies: → Storage Layer                   │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  STORAGE/DATA LAYER                              │  │
│  │  - data/01_input/ → 04_output/                   │  │
│  │  - src/s3_utils.py                               │  │
│  │  Dependencies: → None (leaf layer)               │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  ANALYSIS/SIGNAL GENERATION DOMAIN               │  │
│  │  - analysis/                                      │  │
│  │  - backtesting/                                   │  │
│  │  - implementation/find_*.py                       │  │
│  │  Dependencies: → Storage, → Utils                │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  UTILITIES (CROSS-CUTTING)                       │  │
│  │  - src/config_loader.py                          │  │
│  │  - src/odds_utils.py                             │  │
│  │  - src/team_utils.py                             │  │
│  │  - src/player_name_utils.py                      │  │
│  │  Dependencies: → Config only                     │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘

RULES:
1. Analysis CANNOT directly call DraftKings API (must go through Ingestion)
2. Lambda functions CANNOT import from analysis/
3. Utils are pure functions with no business logic
4. Data flows: Ingestion → Storage → Analysis
```

### 3.2 Create Custom Linters

**Action:** Build `scripts/lint_architecture.py`

**Checks:**

1. **Import boundary violations**
```python
# BAD: analysis importing from lambda
# analysis/some_script.py
from lambda.nba_player_props_ingest.lambda_function import fetch_props

# GOOD: analysis reads from storage
from src.s3_utils import read_props_from_s3
```

2. **Circular dependencies**
```python
# Detect: A imports B, B imports A
```

3. **Utils importing business logic**
```python
# BAD: src/odds_utils.py importing from analysis/
```

**Run in:**
- Pre-commit hook (via pytest)
- GitHub Actions (on push)
- Before agent commits changes

**Implementation:** Integrate with pytest as `tests/test_architecture.py`

### 3.3 Create Structural Tests

**Action:** Add `tests/test_architecture.py`

**Tests:**

```python
def test_lambda_cannot_import_analysis():
    """Lambda functions should not import analysis code."""
    # Parse lambda imports, ensure no analysis/ imports
    
def test_utils_have_no_business_logic():
    """src/ utils should only import from config and stdlib."""
    # Check src/ imports
    
def test_data_flow_direction():
    """Data flows: Ingestion → Storage → Analysis."""
    # Verify dependency graph follows pattern
```

### 3.4 Create Dependency Visualization

**Action:** Build `scripts/visualize_dependencies.py`

**Output:** 
- `docs/generated/dependency-graph.png` 
- `docs/generated/dependency-violations.md` (if any)

**Run:** Daily as background task

---

## Phase 4: Application Legibility (ENABLE SELF-VALIDATION)

**Status:** Not Started  
**Estimated effort:** 4-5 sessions  
**Impact:** Agents can validate their own work

### 4.1 Implement Structured Logging

**Action:** Create `src/logging_utils.py`

**Features:**
- JSON structured logs
- Consistent field names
- Severity levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Context injection (script_name, timestamp, correlation_id)

**Example:**
```python
{
  "timestamp": "2026-02-13T10:30:00Z",
  "level": "INFO",
  "script": "fetch_nba_player_props",
  "message": "Fetched 250 player props",
  "context": {
    "num_props": 250,
    "num_players": 45,
    "books": ["draftkings", "fanduel"],
    "duration_ms": 1234
  }
}
```

**Migration:**
- Gradually replace print statements
- Start with critical paths (lambdas, fetch scripts)

### 4.2 Create Validation Harnesses

**Action:** Build `scripts/validate_*.py` scripts

**Scripts:**

1. **`scripts/validate_props_data.py`**
   - Check freshness (timestamp < 5 min old)
   - Check completeness (required fields present)
   - Check sanity (odds in valid range)
   - Output: JSON report with pass/fail

2. **`scripts/validate_line_movement.py`**
   - Detect significant moves (> threshold)
   - Flag suspicious patterns (all books move together)
   - Compare vs historical volatility

3. **`scripts/validate_cache_consistency.py`**
   - Roster cache matches latest NBA API
   - Player-team mappings are current
   - No stale data (> 7 days old)

**Agent usage:**
```bash
# Agent runs after making changes to fetch script
python scripts/validate_props_data.py --input data/03_intermediate/props_latest.json
# → Sees validation errors
# → Fixes fetch script
# → Re-runs validation
# → Passes → Opens PR
```

### 4.3 Add Data Quality Metrics

**Action:** Create `src/metrics_utils.py`

**Metrics to track:**
- Props fetch success rate
- Data staleness (avg age of props)
- Coverage (% of expected players with props)
- Line movement frequency
- API response times

**Storage:**
- Write to `data/04_output/metrics/daily_metrics.json`
- Queryable by agents

### 4.4 Create Self-Describing Outputs

**Action:** Add schemas to all data outputs

**Example: `data/03_intermediate/props_latest.json`**
```json
{
  "_schema": {
    "version": "1.0",
    "description": "NBA player props from DraftKings",
    "required_fields": ["player_id", "player_name", "market", "odds", "timestamp"],
    "timestamp_format": "ISO8601",
    "odds_format": "American"
  },
  "data": [
    {"player_id": "203999", "player_name": "Nikola Jokic", ...}
  ]
}
```

**Benefit:** Agents can validate structure without external docs

### 4.5 Create Test Data Generator

**Action:** Build `scripts/generate_test_fixtures.py`

**Purpose:** Create realistic test data for agent to validate against

**Generates:**
- `tests/fixtures/props_sample.json` (10 players, realistic odds)
- `tests/fixtures/roster_sample.json` (2 teams, current rosters)
- `tests/fixtures/line_movement_sample.json` (realistic steam)

**Annotated with comments:**
```json
{
  "_note": "This is a realistic over/under prop. Odds of -110 implies 52.4% probability.",
  "market": "player_points_over_under",
  "line": 27.5,
  "over_odds": -110,
  "under_odds": -110
}
```

---

## Phase 5: Feedback Loops & Cleanup (COMPOUND QUALITY)

**Status:** Not Started  
**Estimated effort:** 3-4 sessions  
**Impact:** Continuous quality improvement without human effort

### 5.1 Create Golden Principles Document

**Action:** Create `docs/design-docs/golden-principles.md`

**Contents:**

```markdown
# Golden Principles (Agent Enforcement)

These are mechanical rules enforced continuously by background agents.

## Code Organization

1. **Shared utilities over hand-rolled helpers**
   - BAD: Each script has its own `parse_odds()` function
   - GOOD: Use `src/odds_utils.parse_american_odds()`

2. **No YOLO data probing**
   - BAD: `row.get('maybe_this_key', {}).get('or_this', None)`
   - GOOD: Define schema, validate at boundary, fail fast

3. **Config-driven over hardcoded**
   - BAD: `THRESHOLD = 0.05` sprinkled in code
   - GOOD: `CONFIG['line_steam']['threshold']` in YAML

4. **Explicit imports over wildcards**
   - BAD: `from src.team_utils import *`
   - GOOD: `from src.team_utils import normalize_team_name`

## File Organization

1. **Helper functions in execution order** (see .cursor/rules/cursor_rules.mdc #12)
2. **Docstrings required for all modules**
3. **File size limit: 500 lines** (split if larger)
4. **One concern per file** (fetching vs analysis vs formatting)

## Data Handling

1. **Timestamps always in UTC**
2. **Money always in cents (integers)**
3. **Odds normalized to American format in storage**
4. **Player names normalized via `src/player_name_utils.py`**

## Testing

1. **Integration tests for data flows**
2. **Unit tests for utilities**
3. **Validation harnesses for data quality**
4. **No tests with external API calls** (use fixtures)
```

### 5.2 Create Automated Cleanup Agent

**Action:** Build `scripts/cleanup_agent.py`

**Runs:** Daily via cron or GitHub Actions

**Tasks:**

1. **Scan for violations of golden principles**
   - Duplicate helper functions
   - Hardcoded thresholds
   - Missing docstrings
   - Files > 500 lines

2. **Generate fix PRs**
   - Extract duplicate helpers → src/utils
   - Move hardcoded values → config
   - Add missing docstrings (agent-generated)
   - Suggest file splits

3. **Update QUALITY_SCORE.md**
   - Track violations by domain
   - Trend over time (improving/degrading)

### 5.3 Create Doc Freshness Checker

**Action:** Build `scripts/validate_docs_freshness.py`

**Checks:**

1. **Stale documentation**
   - API references > 90 days old
   - Design docs with TODOs > 30 days old
   - Exec plans in active/ > 60 days old (should move to completed/)

2. **Broken links**
   - All internal links resolve
   - All code references exist

3. **Outdated examples**
   - Code snippets in docs match actual code
   - Config examples match current schema

**Action on failure:**
- Open issue for human review
- Flag in QUALITY_SCORE.md

### 5.4 Create Quality Score Tracker

**Action:** Build `scripts/update_quality_score.py`

**Updates:** `docs/QUALITY_SCORE.md` daily

**Grades by domain (A-F):**

```markdown
# Quality Score

Last updated: 2026-02-13

## By Domain

| Domain | Grade | Issues | Trend |
|--------|-------|--------|-------|
| Data Ingestion | B+ | 3 missing tests | ↑ |
| Storage Layer | A | None | → |
| Analysis | C | 12 hardcoded thresholds | ↓ |
| Utilities | A- | 2 missing docstrings | ↑ |

## By Metric

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Test coverage | 67% | 80% | ⚠️ |
| Doc freshness | 89% | 95% | ⚠️ |
| Linter pass rate | 98% | 100% | ✅ |
| Architectural violations | 2 | 0 | ⚠️ |

## Recent Changes

- 2026-02-12: Fixed 5 golden principle violations in analysis/
- 2026-02-10: Added docstrings to src/odds_utils.py
- 2026-02-08: Migrated AWS docs to docs/references/
```

### 5.5 Create Recurring Background Tasks

**Action:** Set up GitHub Actions for automated maintenance

**Schedule:**

| Task | Frequency | Purpose | Implementation |
|------|-----------|---------|----------------|
| `cleanup_agent.py` | Daily 2am UTC | Fix violations, auto-commit | GitHub Actions |
| `validate_docs_freshness.py` | Daily 3am UTC | Check doc staleness | GitHub Actions |
| `update_quality_score.py` | Daily 4am UTC | Update quality tracking | GitHub Actions |
| `generate_dependency_graph.py` | Daily 5am UTC | Visualize architecture | GitHub Actions |
| `pytest` (architecture tests) | On every push | Ensure boundaries maintained | GitHub Actions |

**Note:** For solo codebase, auto-commit passing changes directly to main (no PR overhead)

---

## Phase 6: Agent Self-Validation Workflow (ADAPTED FOR SOLO)

**Status:** Not Started  
**Estimated effort:** 2-3 sessions  
**Impact:** Agent can work autonomously with quality guarantees

### 6.1 Create Agent Pre-Commit Checklist

**Action:** Create `scripts/agent_precommit_check.py`

**Runs before every agent commit:**

1. ✅ All pytest tests pass
2. ✅ Architectural linter passes (no boundary violations)
3. ✅ Validation harnesses pass (data quality checks)
4. ✅ Golden principles compliance checked
5. ✅ Docstrings present for new functions
6. ✅ Generated docs updated (config schema, dependency graph)

**Usage:**
```bash
# Agent runs before git commit
python scripts/agent_precommit_check.py
# → All checks pass → git commit
# → Checks fail → fix issues → re-run
```

### 6.2 Enable Agent Self-Review

**Action:** Create `scripts/agent_self_review.py`

**Agent reviews its own changes:**

1. **Diff analysis:** What changed and why?
2. **Impact assessment:** What domains affected?
3. **Risk evaluation:** Could this break existing functionality?
4. **Test coverage:** Are new code paths tested?
5. **Documentation:** Are docs updated if behavior changed?

**Output:** Markdown summary saved to git commit message

### 6.3 Implement Fast Recovery

**Action:** Document rollback procedures in `docs/RELIABILITY.md`

**Strategies:**

1. **Tag all agent commits:** `[agent]` prefix in commit message
2. **Automatic backup before major changes:** Tag as `pre-<feature>-backup`
3. **Fast revert:** `git revert <commit>` for single change rollback
4. **Batch revert:** `git revert <hash1>..<hash2>` for agent session rollback

### 6.4 Create Agent Work Log

**Action:** Agent maintains `docs/exec-plans/agent-work-log.md`

**Format:**
```markdown
# Agent Work Log

## 2026-02-13

### Session 1: 10:30-11:45 UTC
- **Goal:** Implement validation harness for props data
- **Changes:** 
  - Created `scripts/validate_props_data.py`
  - Added tests in `tests/test_validate_props.py`
  - Updated `docs/validation/data-validation-rules.md`
- **Outcome:** ✅ All checks passed, committed as abc123f
- **Issues:** None

### Session 2: 14:00-14:30 UTC
- **Goal:** Fix line movement detection threshold
- **Changes:**
  - Updated `src/line_steam_utils.py` threshold logic
  - Added test cases for edge cases
- **Outcome:** ✅ Committed as def456a
- **Issues:** Had to revert once due to test failure, fixed on second attempt
```

**Purpose:** Human can review what agent did without reading every commit

---

## Removed: Phase 6 "Fast Iteration Culture"

**Reasoning:** Original Phase 6 was about PR workflows and team dynamics. For solo codebase, we've adapted the useful parts (self-validation, fast recovery) into the new Phase 6 above.

---

## Success Metrics

**Phase 0 (Cleanup):**
- [ ] docs/AI_summaries/ removed
- [ ] books/README.md created explaining reference materials
- [ ] Repo structure snapshot saved

**Phase 1-2 (Knowledge):**
- [ ] AGENTS.md exists and is < 150 lines
- [ ] docs/ structure created with all major sections
- [ ] 5+ domain docs written (betting fundamentals, market mechanics, etc.)
- [ ] Agent can answer: "What is a prop bet?" by reading docs/
- [ ] All existing docs migrated to new structure

**Phase 3 (Boundaries):**
- [ ] Architectural tests in pytest suite detect import violations
- [ ] Dependency graph generated and clean
- [ ] 0 circular dependencies
- [ ] Agent cannot import analysis/ from lambda/ (enforced by tests)

**Phase 4 (Legibility):**
- [ ] Structured logging in all critical paths (lambdas, fetch scripts)
- [ ] 3+ validation harnesses exist and are runnable
- [ ] Data schemas self-describe in output files
- [ ] Agent can validate props data without human help

**Phase 5 (Feedback):**
- [ ] Golden principles documented
- [ ] Daily cleanup agent runs via GitHub Actions
- [ ] QUALITY_SCORE.md auto-updates daily
- [ ] Doc freshness < 90 days for all references

**Phase 6 (Self-Validation):**
- [ ] Agent pre-commit check script exists and runs all validations
- [ ] Agent self-review generates commit message summaries
- [ ] Rollback procedures documented
- [ ] Agent work log maintained automatically

---

## Next Steps (Ready to Execute)

**Immediate (Phase 0 - Cleanup):**
1. ✅ Remove `docs/AI_summaries/`
2. ✅ Create `books/README.md`
3. ✅ Generate repo structure snapshot

**Phase 1 (Knowledge Structure):**
1. Create `docs/` directory structure
2. Write `AGENTS.md` (short, map-style)
3. Create `docs/ARCHITECTURE.md`
4. Create `docs/index.md`

**Phase 2 (Domain Knowledge):**
1. Write `docs/domain/betting-fundamentals.md` (YOUR expertise needed here)
2. Write `docs/domain/market-mechanics.md`
3. Write `docs/domain/data-quality-standards.md`

**Then iterate through Phases 3-6...**

---

## Decisions Made

1. **Existing docs/AI_summaries/**: ✅ REMOVE - Not written with agent-first mindset
2. **Books directory**: Keep as reference material (Monte Carlo book notes)
3. **Automation**: GitHub Actions (preferred, will implement as needed)
4. **Test framework**: ✅ Expand existing pytest setup
5. **Agent review**: ✅ Yes, but adapted for solo codebase (agent self-review + validation)

**Note on PRs:** As a solo codebase, we'll focus on agent self-validation rather than agent-to-agent PR review. Agent will run validation harnesses, self-review changes, and commit directly when checks pass.

---

**Principle:** This plan is itself a living document. As we implement and learn, we'll update this plan to reflect reality.

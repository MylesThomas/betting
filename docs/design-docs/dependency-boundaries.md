# Dependency Boundaries & Architectural Constraints

**Last updated:** 2026-02-13  
**Status:** ✅ Active  
**Enforcement:** `tests/test_architecture.py`

This document specifies the hard boundaries between architectural layers and the rules for dependencies.

## Layer Definitions

### Layer 1: Data Ingestion

**Purpose:** Fetch external data and write to storage

**Location:**
- `lambda/*/lambda_function.py` - AWS Lambda functions
- `scripts/fetch_*.py` - Manual/ad-hoc fetchers
- `scripts/build_*.py` - Data aggregation/transformation scripts

**Allowed dependencies:**
- ✅ External APIs (DraftKings, The Odds API, NBA API, ESPN)
- ✅ Storage layer (`data/`, S3 via `src/s3_utils.py`)
- ✅ Utilities (`src/*.py`)
- ✅ Config (`config/*.yaml` via `src/config_loader.py`)

**Forbidden dependencies:**
- ❌ Analysis layer (`analysis/`, `backtesting/`, `implementation/`)
- ❌ Cannot import from other lambda functions
- ❌ Cannot read from `data/04_output/` (end of pipeline)

**Rationale:** Ingestion has one job - fetch and store. Mixing analysis logic here creates tangled dependencies.

---

### Layer 2: Storage

**Purpose:** Organize data in structured pipeline

**Location:**
- `data/01_input/` - Raw API responses
- `data/02_cache/` - Lookup tables (rosters, player-team mappings)
- `data/03_intermediate/` - Processed/aggregated data
- `data/04_output/` - Final results and opportunities
- S3 bucket mirror of above
- `src/s3_utils.py` - Storage utilities

**Allowed dependencies:**
- ✅ Python stdlib (pathlib, json, etc.)
- ✅ AWS SDK (boto3 for S3)
- ✅ Config (`config/*.yaml`)

**Forbidden dependencies:**
- ❌ Cannot import business logic from any layer
- ❌ Pure data storage, no computation

**Data flow rule:** `01 → 02 → 03 → 04` (never backwards)

**Rationale:** Storage is a leaf layer. If storage needs business logic, the logic is in the wrong place.

---

### Layer 3: Analysis

**Purpose:** Find betting value using statistical models

**Location:**
- `analysis/` - Research and exploratory analysis
- `backtesting/` - Historical validation
- `implementation/` - Production strategy finders
- `src/pbp_data/` - Play-by-play analysis (specialized analysis)

**Allowed dependencies:**
- ✅ Storage layer (read from `data/`)
- ✅ Utilities (`src/*.py` except `s3_utils.py`)
- ✅ Config (`config/*.yaml`)
- ✅ External libraries (pandas, numpy, scipy, etc.)

**Forbidden dependencies:**
- ❌ Cannot directly call external APIs
- ❌ Cannot import from `lambda/` functions
- ❌ Cannot import from other analysis scripts (except via `src/`)

**Rationale:** Analysis reads from storage, never fetches directly. This ensures:
1. Reproducibility (analysis uses same data)
2. Testability (no API calls in tests)
3. Cost control (don't hammer APIs during analysis)

**Escape hatch:** If analysis needs fresh data → run ingestion script first, then analysis reads from storage.

---

### Layer 4: Utilities (Cross-Cutting)

**Purpose:** Shared, reusable functions with no business logic

**Location:** `src/` (except `src/pbp_data/` which is analysis)

**Files:**
- `config_loader.py` - Load YAML configs
- `odds_utils.py` - Odds conversions
- `team_utils.py` - Team name normalization
- `player_name_utils.py` - Player name matching
- `s3_utils.py` - S3 read/write
- `nba_gamelog_utils.py` - NBA game log helpers
- `season_utils.py` - Season date calculations
- `kelly_criterion.py` - Kelly sizing formulas

**Allowed dependencies:**
- ✅ Python stdlib
- ✅ Config (`config/*.yaml`)
- ✅ Common libraries (pandas, numpy, boto3)
- ✅ Other utilities in `src/` (if needed)

**Forbidden dependencies:**
- ❌ Cannot import from `lambda/`
- ❌ Cannot import from `analysis/`
- ❌ Cannot import from `backtesting/`
- ❌ Cannot import from `implementation/`
- ❌ Cannot contain business logic

**What is "business logic"?**
- ❌ BAD: "Find undervalued props" → That's analysis
- ❌ BAD: "Fetch from DraftKings" → That's ingestion
- ✅ GOOD: "Convert -110 to probability" → Pure utility
- ✅ GOOD: "Normalize 'LA Lakers' to 'Los Angeles Lakers'" → Pure utility

**Rationale:** Utilities are pure functions that can be used anywhere. No side effects, no domain logic.

---

### Layer 5: Presentation

**Purpose:** Display results to users

**Location:**
- `streamlit_app/` - Web dashboard
- `automation/` - Bet placement scripts (future)
- `content/` - Generated reports and visualizations

**Allowed dependencies:**
- ✅ Read from `data/04_output/`
- ✅ Utilities (`src/*.py`)
- ✅ Config (`config/*.yaml`)

**Forbidden dependencies:**
- ❌ Cannot import from `analysis/` or `lambda/`
- ❌ Cannot re-run analysis logic (just display results)

**Rationale:** Presentation is downstream of everything. It reads final outputs, doesn't compute them.

---

## Dependency Graph

```
┌─────────────────────────────────────────────────────────┐
│                    ALLOWED FLOW                          │
└─────────────────────────────────────────────────────────┘

External APIs
    ↓
┌───────────────┐
│  INGESTION    │  lambda/, scripts/fetch_*
│  (Layer 1)    │
└───────┬───────┘
        ↓ writes to
┌───────────────┐
│  STORAGE      │  data/01→02→03→04, S3
│  (Layer 2)    │
└───────┬───────┘
        ↓ reads from
┌───────────────┐
│  ANALYSIS     │  analysis/, backtesting/, implementation/
│  (Layer 3)    │
└───────┬───────┘
        ↓ writes to
┌───────────────┐
│  STORAGE      │  data/04_output/
│  (Layer 2)    │
└───────┬───────┘
        ↓ reads from
┌───────────────┐
│ PRESENTATION  │  streamlit_app/, content/
│  (Layer 5)    │
└───────────────┘

        ↕ (used by all)
┌───────────────┐
│  UTILITIES    │  src/ (pure functions)
│  (Layer 4)    │
└───────────────┘
```

---

## Specific Import Rules

### ✅ ALLOWED

```python
# Ingestion can use storage utilities
# lambda/nba_player_props_ingest/lambda_function.py
from src.s3_utils import write_to_s3
from src.config_loader import load_config

# Analysis can use utilities
# analysis/find_value_props.py
from src.odds_utils import american_to_probability
from src.team_utils import normalize_team_name

# Utilities can use other utilities
# src/nba_gamelog_utils.py
from src.season_utils import get_current_season

# Anything can use config
from src.config_loader import load_config
```

### ❌ FORBIDDEN

```python
# Analysis CANNOT call APIs directly
# analysis/some_script.py
import requests
response = requests.get("https://api.draftkings.com/...")  # ❌ NO

# Analysis CANNOT import from lambda
# analysis/some_script.py
from lambda.nba_player_props_ingest.lambda_function import fetch_props  # ❌ NO

# Lambda CANNOT import from analysis
# lambda/some_lambda/lambda_function.py
from analysis.find_value_props import find_opportunities  # ❌ NO

# Utilities CANNOT import business logic
# src/some_util.py
from analysis.prop_models import predict_points  # ❌ NO
```

---

## Enforcement

### Automated Testing

**File:** `tests/test_architecture.py`

**Tests:**

```python
def test_lambda_cannot_import_analysis():
    """Lambda functions should not import from analysis layer."""
    # Parse imports in lambda/**/lambda_function.py
    # Assert no imports from analysis/, backtesting/, implementation/

def test_analysis_cannot_import_lambda():
    """Analysis should not import from lambda functions."""
    # Parse imports in analysis/, backtesting/, implementation/
    # Assert no imports from lambda/

def test_utils_have_no_business_logic():
    """src/ utils should only import config and stdlib."""
    # Parse imports in src/*.py (except pbp_data/)
    # Assert no imports from lambda/, analysis/, etc.

def test_data_flow_direction():
    """Data flows 01→02→03→04, never backwards."""
    # Check scripts that write to data/
    # Assert 01 doesn't read from 02/03/04
    # Assert 02 doesn't read from 03/04, etc.

def test_no_circular_dependencies():
    """No module should import something that imports it."""
    # Build dependency graph
    # Detect cycles
```

**Run:** 
- On every push (GitHub Actions)
- In pre-commit hook
- Daily via cleanup agent

---

## Fixing Violations

### If test fails: "Lambda imports analysis"

**Bad:**
```python
# lambda/some_lambda/lambda_function.py
from analysis.prop_models import predict_points

def handler(event, context):
    prediction = predict_points(player_data)
    return prediction
```

**Fix:** Extract shared logic to utilities
```python
# src/prop_prediction.py (new utility)
def predict_points(player_data):
    """Pure prediction function with no dependencies."""
    # Model logic here
    return prediction

# lambda/some_lambda/lambda_function.py
from src.prop_prediction import predict_points  # ✅ OK

# analysis/backtest.py
from src.prop_prediction import predict_points  # ✅ OK (both use utility)
```

### If test fails: "Analysis calls API directly"

**Bad:**
```python
# analysis/live_odds_tracker.py
import requests

def get_current_odds():
    response = requests.get("https://api.draftkings.com/...")
    return response.json()
```

**Fix:** Use ingestion → storage → analysis pattern
```python
# Step 1: Add/use existing ingestion script
# scripts/fetch_live_odds.py (already exists or create)
def fetch_and_store_odds():
    response = requests.get("https://api.draftkings.com/...")
    write_to_s3("data/01_input/live_odds.json", response.json())

# Step 2: Analysis reads from storage
# analysis/live_odds_tracker.py
def get_current_odds():
    from src.s3_utils import read_from_s3
    return read_from_s3("data/01_input/live_odds.json")  # ✅ OK
```

### If test fails: "Circular dependency"

**Bad:**
```python
# src/team_utils.py
from src.player_name_utils import get_team_for_player

# src/player_name_utils.py
from src.team_utils import normalize_team_name
```

**Fix:** Extract shared dependency or break cycle
```python
# src/team_utils.py (remove import, pass as parameter)
def normalize_team_name(team: str) -> str:
    # No imports from player_name_utils
    ...

# src/player_name_utils.py
from src.team_utils import normalize_team_name

def get_team_for_player(player: str, team_raw: str) -> str:
    team = normalize_team_name(team_raw)  # ✅ OK (one-way dependency)
    ...
```

---

## Exceptions & Waivers

**None currently.**

If you believe a rule should be broken:
1. Document why in `docs/exec-plans/tech-debt-tracker.md`
2. Get human approval
3. Add to "Exceptions" section here with expiration date
4. Create plan to fix properly

**Philosophy:** Boundaries exist for good reasons. Violating them creates long-term pain for short-term convenience.

---

## Related Documents

- `docs/ARCHITECTURE.md` - High-level system overview
- `docs/design-docs/core-beliefs.md` - Why we enforce boundaries
- `tests/test_architecture.py` - Enforcement tests
- `.cursor/rules/cursor_rules.mdc` - Coding standards

---

**Review cycle:** Quarterly or when adding new layers/domains  
**Last reviewed:** 2026-02-13

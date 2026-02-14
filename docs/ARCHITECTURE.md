# System Architecture

**Last updated:** 2026-02-13

This document provides a high-level map of the betting repository's architecture.

## System Overview

This codebase implements data-driven sports betting strategies for NBA and NFL:

1. **Ingest** data from external APIs (DraftKings, The Odds API, NBA stats)
2. **Store** data in structured pipeline (S3 and local files)
3. **Analyze** for value opportunities using historical models
4. **Execute** via automated alerts and dashboards

## Architecture Layers

### Layer 1: Data Ingestion

**Purpose:** Fetch data from external sources and write to storage

**Components:**
- `lambda/` - AWS Lambda functions for scheduled data fetching
  - `nba_player_props_ingest/` - Fetches NBA player props from DraftKings
  - Other lambdas (see `lambda/README.md` for complete list)
- `scripts/fetch_*.py` - Manual/ad-hoc data fetchers
  - `fetch_nba_player_props.py`
  - `fetch_historical_nba_prop_markets.py`
  - etc.

**Dependencies:**
- ✅ Can call external APIs
- ✅ Can write to Storage layer (`data/`, S3)
- ✅ Can use Utils (`src/`)
- ❌ Cannot import from Analysis layer

**Key principle:** Single responsibility - only fetches and stores, no analysis

---

### Layer 2: Storage

**Purpose:** Organize data in a structured pipeline

**Structure:**
```
data/
├── 01_input/          Raw data from APIs
│   └── the-odds-api/  DraftKings, FanDuel odds
├── 02_cache/          Cached lookups (rosters, player-team mappings)
├── 03_intermediate/   Processed data (consensus lines, aggregated props)
└── 04_output/         Final results (betting opportunities, analysis)
```

**S3 Mirror:**
Most historical data lives in S3 (`s3://betting-data-bucket/`)

**Utilities:**
- `src/s3_utils.py` - Read/write from S3
- `src/config_loader.py` - Load configs

**Key principle:** Data flows 01 → 02 → 03 → 04 (never backwards)

---

### Layer 3: Analysis

**Purpose:** Find betting value using statistical models

**Components:**
- `analysis/` - Ad-hoc research and explorations
  - Market efficiency studies
  - Vig analysis
  - Shot quality modeling
- `backtesting/` - Historical validation of strategies
  - NBA 3PT props
  - NFL luck regression
- `implementation/` - Production strategy finders
  - `find_3pt_underdog_unders_today.py`
  - `find_nfl_regression_plays.py`
  - `find_todays_plays.py`

**Dependencies:**
- ✅ Can read from Storage layer
- ✅ Can use Utils (`src/`)
- ❌ Cannot directly call external APIs (must go through Ingestion)
- ❌ Cannot be imported by Lambda functions

**Key principle:** Analysis reads from storage, never fetches directly

---

### Layer 4: Utilities (Cross-Cutting)

**Purpose:** Shared functionality with no business logic

**Components in `src/`:**
- `config_loader.py` - Load YAML configs
- `odds_utils.py` - Odds conversions (American ↔ Decimal ↔ Probability)
- `team_utils.py` - Team name normalization
- `player_name_utils.py` - Player name matching
- `s3_utils.py` - S3 read/write operations
- `nba_gamelog_utils.py` - NBA game log helpers
- `season_utils.py` - Season date calculations
- `kelly_criterion.py` - Kelly sizing formulas

**Dependencies:**
- ✅ Can import from `config/`
- ✅ Can import Python stdlib
- ❌ Cannot import from Ingestion/Analysis layers
- ❌ Cannot contain business logic

**Key principle:** Pure, reusable functions. No side effects.

---

### Layer 5: Execution (Future)

**Purpose:** Place bets automatically (not yet implemented)

**Planned:**
- `automation/` - Playwright/Selenium scripts for bet placement

---

## Domain Boundaries

### Enforced by `tests/test_architecture.py`

```
┌─────────────────────────────────────────────────────────┐
│                   BUSINESS DOMAINS                       │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  DATA INGESTION                                  │  │
│  │  lambda/, scripts/fetch_*                        │  │
│  │  ↓                                                │  │
│  │  Can write to: Storage                           │  │
│  │  Cannot import: Analysis                         │  │
│  └──────────────────────────────────────────────────┘  │
│                      ↓                                   │
│  ┌──────────────────────────────────────────────────┐  │
│  │  STORAGE                                         │  │
│  │  data/01-04/, src/s3_utils.py                    │  │
│  │                                                   │  │
│  │  Dependencies: None (leaf layer)                 │  │
│  └──────────────────────────────────────────────────┘  │
│                      ↓                                   │
│  ┌──────────────────────────────────────────────────┐  │
│  │  ANALYSIS                                        │  │
│  │  analysis/, backtesting/, implementation/        │  │
│  │  ↓                                                │  │
│  │  Can read from: Storage                          │  │
│  │  Cannot: Call APIs directly                      │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  UTILITIES (cross-cutting)                       │  │
│  │  src/ (except s3_utils)                          │  │
│  │                                                   │  │
│  │  Used by: All layers                             │  │
│  │  Dependencies: Config only                       │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Rules:**
1. Data flows one direction: Ingestion → Storage → Analysis
2. Utilities can be used anywhere but import nothing
3. Lambda functions in `lambda/` folder only
4. No circular dependencies

See `docs/design-docs/dependency-boundaries.md` for complete specification.

---

## Configuration Management

**Location:** `config/` directory

**Key files:**
- `config.yaml` - Main config (API keys, thresholds, team mappings)
- `season_dates.yaml` - NBA/NFL season start/end dates
- `line_steam_config.yaml` - Line movement detection thresholds
- `futures_config.yaml` - Championship futures tracking
- `the-odds-api_config.yaml` - Odds API markets and bookmakers

**Loading:**
```python
from src.config_loader import load_config
config = load_config()  # Loads config.yaml by default
```

**Principle:** All thresholds/API keys/mappings in config, never hardcoded.

---

## Technology Stack

- **Language:** Python 3.13
- **Package Manager:** uv (fast pip replacement)
- **Cloud:** AWS Lambda, S3, EventBridge
- **Data:** pandas, numpy, pyarrow
- **APIs:** DraftKings, The Odds API, NBA API
- **Testing:** pytest
- **Visualization:** matplotlib, seaborn, R (via rpy2)

---

## Data Flow Example: NBA Player Props

```
1. INGESTION
   └─ lambda/nba_player_props_ingest/lambda_function.py
      └─ Fetches from DraftKings API
      └─ Writes to S3: s3://.../01_input/props_raw_2026-02-13.json

2. STORAGE
   └─ data/01_input/props_raw_2026-02-13.json (S3 synced)

3. PROCESSING (Ingestion layer still)
   └─ scripts/build_consensus_props.py
      └─ Reads: data/01_input/props_raw_*.json
      └─ Aggregates across books
      └─ Writes: data/03_intermediate/consensus_props_2026-02-13.json

4. ANALYSIS
   └─ implementation/find_3pt_underdog_unders_today.py
      └─ Reads: data/03_intermediate/consensus_props_*.json
      └─ Applies statistical model
      └─ Writes: data/04_output/opportunities_3pt_2026-02-13.json

5. PRESENTATION
   └─ streamlit_app/app.py
      └─ Reads: data/04_output/opportunities_*.json
      └─ Displays in web dashboard
```

---

## Quality & Observability

**Quality tracking:**
- `docs/QUALITY_SCORE.md` - Grades by domain (updated daily)
- `docs/exec-plans/tech-debt-tracker.md` - Known issues

**Validation:**
- `scripts/validate_props_data.py` - Data quality checks
- `scripts/validate_cache_consistency.py` - Cache freshness
- `tests/test_architecture.py` - Boundary enforcement

**Logging:**
- Structured logging via `src/logging_utils.py` (planned)
- Lambda CloudWatch logs

---

## Key Design Decisions

See `docs/design-docs/` for rationale behind:

- Why 01-04 data pipeline stages
- Why uv instead of pip
- Why Lambda for ingestion
- Why S3 for historical storage
- Why strict layer boundaries

---

## External Dependencies

**APIs:**
- **DraftKings** - Player props, game lines
- **The Odds API** - Multi-book odds aggregation
- **NBA API** - Game logs, team stats, rosters
- **ESPN API** - Fallback for game results

**AWS Services:**
- **Lambda** - Scheduled data fetching (daily 7am ET)
- **S3** - Historical data storage
- **EventBridge** - Lambda scheduling
- **SNS** - Alerts (future)

See `docs/references/` for API documentation.

---

## Next Steps

**For agents:**
- Read `docs/design-docs/dependency-boundaries.md` for detailed rules
- Read `docs/domain/betting-fundamentals.md` to understand betting concepts
- Check `docs/QUALITY_SCORE.md` to see current health

**For extending:**
- New data source? → Add to `lambda/` or `scripts/fetch_*`
- New strategy? → Add to `analysis/` or `implementation/`
- New utility? → Add to `src/` (ensure no business logic)
- New config? → Add to `config/` YAML files

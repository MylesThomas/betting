# Tests

Proper unit tests using pytest. Not to be confused with `testing/` which contains ad-hoc exploratory scripts.

## Structure

```
tests/
├── unit/                    # Fast, isolated unit tests (no S3, no external APIs)
│   └── test_line_movement_analysis.py
└── integration/             # Integration tests (TODO - will hit S3, load real data)
```

## Running Tests

```bash
cd betting

# Run all tests
pytest tests/ -v

# Run only unit tests (fast)
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_line_movement_analysis.py -v

# Run specific test class
pytest tests/unit/test_line_movement_analysis.py::TestCoverCalculations -v

# Run specific test
pytest tests/unit/test_line_movement_analysis.py::TestCoverCalculations::test_favorite_covers_by_more_than_spread -v

# With coverage report
pytest tests/ --cov=analysis --cov=src --cov-report=html

# Show print statements
pytest tests/unit/ -v -s
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Fast** - run in milliseconds
- **Isolated** - no external dependencies (S3, APIs, files)
- **Deterministic** - same input = same output
- **Pure logic** - test calculations, formulas, transformations

### Integration Tests (`tests/integration/`)
- TODO - will test end-to-end flows
- Load data from S3
- Test full pipeline

## What Gets Tested

### Line Movement Analysis (`test_line_movement_analysis.py`)

**Cover Calculations** (most critical):
- Favorite covers by more than spread ✅
- Favorite wins but doesn't cover ✅
- Underdog covers despite losing ✅
- Push scenarios ✅

**Movement Calculations**:
- Line moves toward favorite ✅
- Line moves toward underdog ✅
- Line crosses zero ✅

**Steam Direction**:
- Steam toward opening favorite ✅
- Steam toward opening underdog ✅

**Derived Features**:
- Underdog metrics are inverse of favorite ✅
- Fade strategy logic ✅

**Invariants** (must ALWAYS be true):
- Spreads sum to zero ✅
- Opening favorite has negative spread ✅
- Exactly one team covers (except push) ✅
- Magnitude is always non-negative ✅

**Edge Cases**:
- Pick'em lines ✅
- Extreme movements (10+ pts) ✅
- Half-point spreads ✅

## Adding New Tests

When adding new analysis logic:

1. **Write tests FIRST** (TDD approach)
2. Test the happy path
3. Test edge cases
4. Test invariants
5. Add integration test if hitting external systems

## Dependencies

```bash
pip install pytest pytest-cov
```

Or add to `requirements.txt`:
```
pytest>=7.4.0
pytest-cov>=4.1.0
```


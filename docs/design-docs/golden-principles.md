# Golden Principles (Enforced Mechanically)

**Status:** 📝 Planned  
**Last updated:** 2026-02-13

This document will define:

## Code Organization Rules

1. **Shared utilities over hand-rolled helpers**
2. **No YOLO data probing** (`data.get().get().get()`)
3. **Config-driven over hardcoded**
4. **Explicit imports over wildcards**

## File Organization Rules

1. **Helper functions in execution order**
2. **Docstrings required for all modules**
3. **File size limit: 500 lines**
4. **One concern per file**

## Data Handling Rules

1. **Timestamps always UTC**
2. **Money in cents (integers)**
3. **Odds normalized to American in storage**
4. **Player names normalized via utils**

## Testing Rules

1. **Integration tests for data flows**
2. **Unit tests for utilities**
3. **Validation harnesses for data quality**
4. **No tests with external API calls**

These will be enforced by:
- `scripts/cleanup_agent.py` (daily scans)
- Custom linters
- Pre-commit hooks

---

**To be written in:** Phase 5.1 (Feedback Loops)

See also: `.cursor/rules/cursor_rules.mdc` for detailed coding standards

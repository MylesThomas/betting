# Design Documentation Catalog

This directory contains architectural decisions, design rationale, and technical guidelines.

## Core Philosophy

- **[core-beliefs.md](core-beliefs.md)** - Agent-first operating principles ⭐ READ FIRST

## Architecture Decisions

- **[dependency-boundaries.md](dependency-boundaries.md)** - Layer rules and import constraints
- **[data-pipeline-architecture.md](data-pipeline-architecture.md)** - Why 01-04 stage model
- **[golden-principles.md](golden-principles.md)** - Coding standards (mechanically enforced)

## Technology Choices

- **Why uv not pip** - Fast, reliable Python package management (TBD)
- **Why Lambda for ingestion** - Scheduled, serverless, cost-effective (TBD)
- **Why S3 for historical data** - Cheap storage, versioned, queryable (TBD)

## Domain-Specific Designs

- **Monte Carlo simulation approach** - How we model game outcomes (TBD)
- **Line steam detection** - Threshold-based significance testing (TBD)
- **Player name matching** - Fuzzy matching across data sources (TBD)

---

## Design Doc Status

| Document | Status | Last Updated | Verified |
|----------|--------|--------------|----------|
| core-beliefs.md | ✅ Active | 2026-02-13 | ✅ |
| dependency-boundaries.md | ✅ Active | 2026-02-13 | ✅ |
| golden-principles.md | 🚧 Draft | - | - |
| data-pipeline-architecture.md | 📝 Planned | - | - |

**Legend:**
- ✅ Active: Current and verified
- 🚧 Draft: Written but needs review
- 📝 Planned: Not yet written
- ⚠️ Stale: Needs update

---

## How to Use Design Docs

**Before making architectural changes:**
1. Read relevant design docs to understand rationale
2. Check if your change conflicts with existing decisions
3. Update docs if your change introduces new patterns
4. Get human review for significant architectural shifts

**When writing new design docs:**
1. Explain the problem/decision clearly
2. List alternatives considered
3. Document trade-offs and rationale
4. Include examples (good/bad patterns)
5. Link to related docs

**Template:**
```markdown
# [Decision Title]

## Context
What problem are we solving?

## Decision
What did we decide to do?

## Rationale
Why this approach over alternatives?

## Consequences
- Positive: What benefits do we get?
- Negative: What trade-offs are we accepting?

## Examples
Good/bad patterns, code samples

## Related
Links to other design docs, issues, or discussions
```

---

**Maintained by:** Automated doc freshness checker  
**Review cycle:** Quarterly or when major changes occur

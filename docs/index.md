# Documentation Index

Welcome to the betting repository documentation. This index helps you navigate the knowledge base.

## 🗺️ Start Here

- **[AGENTS.md](../AGENTS.md)** - Quick start guide for AI agents working on this codebase
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - High-level system architecture and component relationships
- **[README.md](../README.md)** - Project overview, setup instructions, and quick start

## 📐 Architecture & Design

Located in `design-docs/`:

- **[core-beliefs.md](design-docs/core-beliefs.md)** - Agent-first operating principles
- **[dependency-boundaries.md](design-docs/dependency-boundaries.md)** - Layer rules and architectural constraints
- **[data-pipeline-architecture.md](design-docs/data-pipeline-architecture.md)** - Why we use 01-04 stage model
- **[index.md](design-docs/index.md)** - Full design doc catalog

## 🎰 Betting Domain Knowledge

Located in `domain/`:

- **[betting-fundamentals.md](domain/betting-fundamentals.md)** - Props, moneyline, spreads, over/under
- **[spread-cover-rule.md](domain/spread-cover-rule.md)** - ATS cover logic: single source of truth in `src/odds_utils.did_cover_spread` (do not reimplement)
- **[market-mechanics.md](domain/market-mechanics.md)** - Vig, line movement, steam, arbitrage
- **[data-quality-standards.md](domain/data-quality-standards.md)** - What makes data "good"
- **[edge-cases.md](domain/edge-cases.md)** - Postponed games, injuries, line freezes
- **[nba-vs-nfl.md](domain/nba-vs-nfl.md)** - Sport-specific patterns and differences

## 📋 Execution Plans & Work Tracking

Located in `exec-plans/`:

- **[active/](exec-plans/active/)** - Current work streams and in-progress tasks
- **[completed/](exec-plans/completed/)** - Historical context and finished projects
- **[tech-debt-tracker.md](exec-plans/tech-debt-tracker.md)** - Known issues and improvement opportunities

## 📚 External References

Located in `references/`:

- **[duckdb-s3-queries.md](references/duckdb-s3-queries.md)** - How to run DuckDB queries against S3 (httpfs setup, region, credentials for `-c` calls). Use when inspecting schemas or data in S3.
- API documentation (DraftKings, The Odds API, NBA API)
- AWS Lambda patterns and best practices
- Third-party library references (uv, pandas, etc.)

## ✅ Validation & Quality

Located in `validation/`:

- **[data-validation-rules.md](validation/data-validation-rules.md)** - What to check in props data
- **[test-coverage-targets.md](validation/test-coverage-targets.md)** - Coverage goals by domain

## 🤖 Auto-Generated Documentation

Located in `generated/`:

- Configuration schemas (auto-generated from YAML)
- Module dependency graphs
- API response examples

Updated automatically by background tasks.

## 📊 Quality Tracking

- **[QUALITY_SCORE.md](QUALITY_SCORE.md)** - Current quality grades by domain and layer
- **[RELIABILITY.md](RELIABILITY.md)** - Error handling, retries, monitoring
- **[SECURITY.md](SECURITY.md)** - API keys, secrets management, data privacy

## 🚀 Getting Started Paths

**I'm an AI agent working on this codebase:**
→ Read [AGENTS.md](../AGENTS.md) first, then [ARCHITECTURE.md](ARCHITECTURE.md)

**I need to understand betting concepts:**
→ Start with [domain/betting-fundamentals.md](domain/betting-fundamentals.md)

**I'm adding a new feature:**
→ Check [exec-plans/active/](exec-plans/active/) for current work, then [design-docs/](design-docs/) for patterns

**I need to fix a bug:**
→ Review [RELIABILITY.md](RELIABILITY.md) and [validation/data-validation-rules.md](validation/data-validation-rules.md)

**I'm refactoring code:**
→ Read [design-docs/dependency-boundaries.md](design-docs/dependency-boundaries.md) and [design-docs/golden-principles.md](design-docs/golden-principles.md)

---

**Last updated:** 2026-02-13  
**Maintained by:** Automated doc freshness checker

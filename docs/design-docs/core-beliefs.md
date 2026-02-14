# Core Beliefs: Agent-First Development

**Last updated:** 2026-02-13  
**Status:** ✅ Active

This document defines the operating principles for this codebase, optimized for AI agent development.

## Philosophy

We are building a codebase where **AI agents do most of the coding work**. Humans design environments, specify intent, and build feedback loops. Agents execute.

This fundamentally changes what "good code" means:
- **Legibility > Aesthetics** - Code must be discoverable and understandable to agents, not necessarily beautiful to humans
- **Constraints > Conventions** - Enforce invariants mechanically, don't rely on memory or discipline
- **Fast iteration > Perfect first time** - Corrections are cheap, waiting is expensive

These principles are inspired by [OpenAI's Codex harness engineering practices](https://openai.com/index/harness-engineering/).

---

## 1. Repository Knowledge is the Only Knowledge

**Principle:** If it's not in the repository, it doesn't exist for the agent.

**Why:** Agents can only access what's in-context. Knowledge in Slack, Google Docs, or people's heads is invisible.

**Practice:**
- ✅ Document betting domain knowledge in `docs/domain/`
- ✅ Capture design decisions in `docs/design-docs/`
- ✅ Maintain API references in `docs/references/`
- ✅ Track work in `docs/exec-plans/`
- ❌ Don't assume agents know "common sense" betting concepts
- ❌ Don't rely on tribal knowledge or chat history

**Example:**
```python
# BAD - Agent doesn't know what "closing line value" means
if has_closing_line_value(bet):
    place_bet(bet)

# GOOD - Document concept in docs/domain/market-mechanics.md
# Agent can read: "Closing line value means betting at better odds
# than the line closes at, indicating sharp action validated your bet"
```

---

## 2. Give Agents a Map, Not a Manual

**Principle:** Context is scarce. A short, structured entry point is better than a giant instruction file.

**Why:** A 1000-line AGENTS.md crowds out actual code. Agents need to know *where to look*, not *everything at once*.

**Practice:**
- ✅ Keep `AGENTS.md` < 150 lines as a table of contents
- ✅ Use progressive disclosure: point to `docs/` for details
- ✅ Organize docs hierarchically with clear navigation
- ❌ Don't put everything in one giant file
- ❌ Don't repeat information across multiple docs

**Structure:**
```
AGENTS.md (map)
  ↓
docs/index.md (detailed navigation)
  ↓
docs/domain/betting-fundamentals.md (specific knowledge)
```

---

## 3. Fail Fast, Don't Paper Over Problems

**Principle:** If data is malformed, let it crash. Don't use defensive checks for required fields.

**Why:** Silent failures compound. If a key should exist but doesn't, that's a bug upstream that needs fixing.

**Practice:**
```python
# BAD - Masks the real problem
player_name = data.get('player_name', 'Unknown')
if player_name != 'Unknown':
    process(player_name)

# GOOD - Fails loudly if data is wrong
player_name = data['player_name']  # KeyError if missing
process(player_name)
```

**When to use `.get()`:**
- ✅ Truly optional fields: `data.get('nickname')`  # Nicknames are optional
- ❌ Required fields: `data['player_id']`  # Player ID must exist

See `.cursor/rules/cursor_rules.mdc` rule #1 and #4.

---

## 4. No Fake Data

**Principle:** Never create mock data, placeholder data, or fake API responses without explicit permission.

**Why:** Fake data can accidentally make it to production. It masks real integration issues.

**Practice:**
- ✅ Use real API responses saved as test fixtures
- ✅ Ask human before generating synthetic data
- ❌ Don't hardcode fake player names, odds, or game results
- ❌ Don't create placeholder responses when API is down

**Example:**
```python
# BAD - Agent couldn't reach API so made up data
props = [
    {"player": "LeBron James", "line": 25.5, "odds": -110},
    {"player": "Steph Curry", "line": 28.5, "odds": -110},
]

# GOOD - Use saved fixture or fail gracefully
try:
    props = fetch_from_api()
except APIError:
    logger.error("API unavailable, skipping update")
    return None  # Don't make up data
```

---

## 5. Enforce Boundaries, Allow Autonomy

**Principle:** Be strict about *what* (layer boundaries, data schemas, interfaces). Be flexible about *how* (implementation details).

**Why:** Like managing a large engineering org - enforce boundaries centrally, allow teams local autonomy.

**Practice:**
- ✅ Enforce: Layer dependencies (Analysis can't import Lambda code)
- ✅ Enforce: Data validation at boundaries (check props have required fields)
- ✅ Enforce: Naming conventions (files, functions, variables)
- ✅ Allow: Implementation details (how a function is written internally)
- ✅ Allow: Code style variations (within linter rules)

**Enforcement:**
- Architectural boundaries → `tests/test_architecture.py`
- Data schemas → Validation harnesses in `scripts/validate_*.py`
- Naming/style → Linters + golden principles

The resulting code may not match human stylistic preferences. **That's okay.** As long as it's correct, maintainable, and legible to future agent runs.

---

## 6. Make Work Self-Validating

**Principle:** Agents should validate their own work without human QA.

**Why:** Human attention is the bottleneck. If agents can check correctness, they can work autonomously.

**Practice:**
- ✅ Create validation harnesses: `scripts/validate_props_data.py`
- ✅ Write tests that agents can run: `pytest tests/`
- ✅ Make application state legible: structured logs, metrics
- ✅ Self-describing data: embed schemas in JSON outputs
- ❌ Don't rely on humans to manually check every output

**Agent workflow:**
```bash
# Agent makes changes
python scripts/fetch_nba_player_props.py

# Agent validates output
python scripts/validate_props_data.py --input data/01_input/props_latest.json
# → Validation passes

# Agent commits
git commit -m "fetch: update props fetching to handle new API format"
```

---

## 7. Optimize for Legibility, Not Cleverness

**Principle:** Explicit is better than implicit. Clear is better than clever.

**Why:** Agents benefit from straightforward code they can reason about.

**Practice:**
```python
# BAD - Too clever, hard to reason about
teams = [t for d in data if (t := d.get('team')) and t not in seen and not seen.add(t)]

# GOOD - Explicit steps
unique_teams = set()
for data_point in data:
    if 'team' in data_point:
        unique_teams.add(data_point['team'])
teams = list(unique_teams)
```

**Code organization:**
- ✅ Helper functions in execution order (see `.cursor/rules/cursor_rules.mdc` #12)
- ✅ Clear function names (`normalize_player_name` not `clean`)
- ✅ Docstrings explaining *why*, not just *what*

---

## 8. Config-Driven Over Hardcoded

**Principle:** Thresholds, API keys, and parameters belong in config files, not scattered in code.

**Why:** Agents can discover config values. Hardcoded values are invisible and unmaintainable.

**Practice:**
```python
# BAD - Magic number, agent doesn't know why
if price_movement > 0.05:
    alert()

# GOOD - Config explains the threshold
from src.config_loader import load_config
config = load_config('line_steam_config.yaml')
threshold = config['steam_detection']['significance_threshold']
if price_movement > threshold:
    alert()
```

**Config location:** `config/*.yaml`

---

## 9. Fast Iteration with Strong Guardrails

**Principle:** Ship quickly, validate continuously, fix forward.

**Why:** With proper boundaries and validation, corrections are cheap. Waiting for perfection is expensive.

**Practice:**
- ✅ Commit frequently with validation passing
- ✅ Fix issues in follow-up commits (not amending)
- ✅ Run daily cleanup to catch drift
- ❌ Don't wait for perfect code before committing
- ❌ Don't over-plan before trying

**Safety net:**
- Architectural tests prevent boundary violations
- Validation harnesses catch data issues
- Golden principles checker finds bad patterns
- Daily quality score tracks health

If something breaks, rollback is easy: `git revert <commit>`

---

## 10. Document Decisions, Not Just Code

**Principle:** Capture *why* decisions were made, not just *what* the code does.

**Why:** Agents need to understand rationale to make aligned changes.

**Practice:**
- ✅ Design docs explain trade-offs: `docs/design-docs/`
- ✅ Comments explain *why*, not *what*:
  ```python
  # Use 7-day window to smooth variance from back-to-backs
  window = timedelta(days=7)
  ```
- ❌ Don't just describe code:
  ```python
  # Set window to 7 days  ← Obvious from code
  window = timedelta(days=7)
  ```

---

## Consequences of These Beliefs

**Positive:**
- Agents can work autonomously with high quality
- Knowledge is discoverable and versioned
- Fast iteration without architectural decay
- Human time focused on high-leverage work (design, review, domain expertise)

**Negative:**
- More upfront investment in documentation
- More tooling (linters, validators, cleanup scripts)
- Code may not look "human-written"
- Requires discipline to maintain documentation

**Trade-off:** We're optimizing for **agent productivity** and **system maintainability** over **immediate human coding speed**.

---

## Related Documents

- `AGENTS.md` - Quick start guide for agents
- `docs/ARCHITECTURE.md` - System architecture
- `docs/design-docs/dependency-boundaries.md` - Layer rules
- `docs/design-docs/golden-principles.md` - Enforced coding standards
- `.cursor/rules/cursor_rules.mdc` - Detailed coding rules

---

## Review and Evolution

These principles are living guidelines. As we learn what works, we'll update them.

**Review triggers:**
- Quarterly review
- After major agent failures (what constraint was missing?)
- When onboarding new agents or tools
- When architectural decisions change

**Last reviewed:** 2026-02-13

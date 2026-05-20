# Betting Knowledge Base

A structured wiki for betting domain knowledge, maintained by Claude Code.

## Purpose

Prevent repeated mistakes and codify domain rules that should inform every analysis.
Before computing odds-related stats, calibration metrics, or ROI figures — check the relevant wiki page.

## Folder structure

```
wiki/          -- concept pages maintained by Claude
wiki/index.md  -- table of contents
wiki/log.md    -- append-only record of all changes
```

## When to consult

| You are about to...                          | Read first                        |
|----------------------------------------------|-----------------------------------|
| Compute mean/aggregate of American odds       | [[american-odds]]                 |
| Calculate ROI, P&L, hit rate                  | [[roi-and-pnl]]                   |
| Compute edge or model probability             | [[edge-calibration]]              |
| Interpret Brier score or model metrics        | [[model-evaluation]]              |
| Load game logs, props, or shot chart data     | [[data-quirks]]                   |
| Reference NBA season structure or dates       | [[nba-season-structure]]          |

## Update workflow

When a new domain rule or bug is discovered:
1. Identify the relevant wiki page (or create a new one)
2. Add the rule with a concrete example of what goes wrong without it
3. Update `wiki/index.md` and append to `wiki/log.md`

## Page format

```markdown
# Page Title

**Summary**: One sentence.

**Last updated**: YYYY-MM-DD

---

Content. Use concrete examples. Link related pages with [[page-name]].

## Rules
- Numbered, actionable rules

## Common mistakes
- What goes wrong and why

## Related
- [[page-name]]
```

## Rules

- Keep pages short and scannable — not textbooks
- Every rule needs a "what goes wrong" example
- Never remove entries from `wiki/log.md`

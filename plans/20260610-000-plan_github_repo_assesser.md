# Plan: GitHub Repo Assessor Skill

## Goal
A Claude skill (`/assess-nba-betting-repo`) that evaluates a GitHub repository against a structured rubric tuned for NBA betting alpha generation. Clones the repo locally, reads actual source code, scores it across 9 factors, and outputs a scored summary card.

---

## Skill Invocation

```
/assess-nba-betting-repo <github_url>
```

Skill file: `.claude/commands/assess-nba-betting-repo.md`

The skill `git clone --depth=1`s the repo to `/tmp/repo-assess/<repo>`, reads key source files directly, scores against the rubric, then cleans up.

---

## Assessment Rubric

Each factor scored 1–5 with a one-line rationale. **Total: /45**

| # | Factor | What to look for |
|---|--------|-----------------|
| 1 | **Alpha potential** | Does the code target markets a sportsbook would misprice? Player props > totals > moneylines. CLV tracking = bonus. |
| 2 | **Novel insight / differentiation** | Goes beyond public box scores — tracking data, injury load, schedule density, rest days, court/referee factors? |
| 3 | **Model sophistication** | Ensemble > single model; Optuna/grid tuning; time-series CV (not random splits); no data leakage |
| 4 | **Odds integration** | Live lines, CLV calculation, Kelly Criterion, vig handling |
| 5 | **Data pipeline quality** | Automated ingestion, missing data handling, historical lines stored (not just outcomes) |
| 6 | **Backtesting rigor** | Walk-forward or time-series CV; ROI net of vig; honest about sample size |
| 7 | **Adaptability to our stack** | Python? Modular? Pluggable into Lambda/S3? Can extract a single component? |
| 8 | **Maintenance burden** | Last commit recency, dep freshness, test coverage, docs quality |
| 9 | **Out-of-box / free impact** | Can you run it today at zero cost and get betting value? Single command? |

Verdict bands:
- **Integrate**: 36–45
- **Mine for parts**: 23–35
- **Reference only**: 12–22
- **Skip**: <12

---

## Output Format (per repo)

```
### [Repo Name](url)
**One-liner:** what it does in plain English

**Stars:** N | **Last commit:** YYYY-MM | **Language:** X

| Factor | Score | Evidence (cite filenames + specific code) |
|--------|-------|------------------------------------------|
| Alpha potential | N/5 | … |
| ...             | …   | … |
| **Total**       | **N/45** | |

**Verdict:** Integrate / Mine for parts / Reference only / Skip
**Best component to steal:** one sentence — name the specific file/function
**Gaps / risks:** one sentence — the single biggest red flag in the code
```

---

## Status

| Step | Status |
|------|--------|
| Write skill file | ✅ Done |
| Add 9th factor (out-of-box impact) | ✅ Done |
| Switch from WebFetch → git clone | ✅ Done |
| Rename skill to `assess-nba-betting-repo` | ✅ Done |
| Test on kyleskom repo | ✅ Done |
| Run all 9 repos (WebFetch pass) | ✅ Done — `ref/nba-repo-assessments.md` |
| Re-run all 9 with actual code review | ✅ Done — scores updated in `ref/nba-repo-assessments.md` |

---

## Repos Assessed (from ref/free-nba-ml-repos.md)

Results in `ref/nba-repo-assessments.md`.

| Repo | Score | Verdict |
|------|-------|---------|
| swar/nba_api | 30/45 | Mine for parts |
| kyleskom/NBA-Machine-Learning-Sports-Betting | 23/45 | Mine for parts |
| NBA-Betting/NBA_AI | 21/45 | Reference only |
| Pirkn/NBA-Game-Outcome-Prediction | 16/45 | Reference only |
| cmunch1/nba-prediction | 16/45 | Reference only |
| zostaff/basketball-analysis | 12/45 | Skip |
| luke-lite/NBA-Prediction-Modeling | 12/45 | Reference only |
| CyrilleAD/basketball-analysis-system | 12/45 | Skip |
| avishah3/AI-Basketball-Shot-Detection-Tracker | 8/45 | Skip |

---

## Open Questions

- Re-run all 9 using the new clone-based skill for a third, fully local pass?
- Should the skill also `grep` for specific patterns (e.g. `TimeSeriesSplit`, `kelly`, `clv`) as a quick signal before reading full files?

# /assess-nba-betting-repo

Assess a GitHub repository against a structured rubric tuned for NBA betting alpha generation. Clone the repo locally, read actual source code, score it across 9 factors, and output a scored summary card.

**Arguments:** `$ARGUMENTS` — a GitHub repo URL, e.g. `https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting`

---

## Step 0 — Parse arguments

Extract the GitHub repo URL from `$ARGUMENTS`.
- Parse `<owner>` and `<repo>` from the URL.
- Set clone target: `/tmp/repo-assess/<repo>`

---

## Step 1 — Clone the repo

```bash
rm -rf /tmp/repo-assess/<repo>
git clone --depth=1 https://github.com/<owner>/<repo>.git /tmp/repo-assess/<repo>
```

`--depth=1` keeps it fast (no full history needed). If clone fails, fall back to WebFetch on the README and GitHub page, and note it.

After cloning:
1. Run `find /tmp/repo-assess/<repo> -type f | sort` to get the full file tree.
2. Check `git log --oneline -1` for the last commit date.
3. Note the primary languages from file extensions.

---

## Step 2 — Read key source files

From the file tree, identify and read the most revealing files for each rubric factor. Prioritise:

- **Model training code** — any file with `train`, `model`, `fit`, `XGBoost`, `torch`, `sklearn` in name or path
- **Data pipeline** — any file with `pipeline`, `ingest`, `scrape`, `data`, `fetch` in name or path
- **Odds / betting logic** — any file with `odds`, `kelly`, `ev`, `bet`, `edge`, `clv` in name or path
- **Backtesting** — any file with `backtest`, `eval`, `test`, `simulate` in name or path
- **Entry point** — `main.py`, `app.py`, `run.py`, `start_app.py`
- **Config** — `config.yaml`, `config.toml`, `.env.example`
- **Automation** — `.github/workflows/*.yml`

Read at least 5–8 files. Use `cat` or `Read` on each. For Jupyter notebooks (`.ipynb`), read the raw JSON and look at the `source` arrays in each cell.

Do NOT rely on README claims — verify against what the code actually does.

---

## Step 3 — Score against rubric

Score each factor 1–5. Be honest — a 3 means average, not good. Cite specific filenames and line-level evidence.

| # | Factor | Scoring guide |
|---|--------|--------------|
| 1 | **Alpha potential** | Does the code target markets a book would misprice? Player props > game totals > moneylines. Does it beat closing line? (5 = prop-level edge thesis with CLV tracking; 1 = no edge framing at all) |
| 2 | **Novel insight / differentiation** | Goes beyond public box scores — tracking data, injury load, schedule density, rest days, court-specific factors, referee tendencies? (5 = clearly differentiated inputs; 1 = pure box score recycling) |
| 3 | **Model sophistication** | Ensemble > single model; Optuna/grid-search tuning; proper time-series CV (not random splits); no data leakage. (5 = production-grade with walk-forward; 1 = single model, random hold-out) |
| 4 | **Odds integration** | Live lines, CLV calculation, Kelly Criterion, vig handling. (5 = full Kelly + CLV tracking; 1 = no odds integration) |
| 5 | **Data pipeline quality** | Automated ingestion, missing data handling, historical lines stored (not just outcomes). (5 = fully automated, historical lines stored; 1 = manual CSV uploads) |
| 6 | **Backtesting rigor** | Walk-forward or time-series CV; ROI reported net of vig; honest about sample size and overfitting. (5 = rigorous walk-forward + net ROI; 1 = random split, gross accuracy only) |
| 7 | **Adaptability to our stack** | Python? Modular enough to plug into Lambda/S3? Can extract a single model or data layer? (5 = drop-in Python module; 1 = monolithic notebook) |
| 8 | **Maintenance burden** | Last commit recency, dep freshness, test coverage, docs quality. (5 = active, clean deps, tests, good docs; 1 = abandoned 3+ years, broken deps) |
| 9 | **Out-of-box / free impact** | Can you run it today at zero cost and get betting value? (5 = one command, fully free, works immediately; 1 = requires paid APIs, heavy setup, or is broken) |

---

## Step 4 — Clean up

```bash
rm -rf /tmp/repo-assess/<repo>
```

---

## Step 5 — Write the assessment card

Output the following card. Cite actual filenames and code evidence — no README paraphrasing.

---

### [Repo Name](url)

**One-liner:** what it does in plain English (one sentence, no jargon)

**Stars:** N | **Last commit:** YYYY-MM | **Language:** primary language

| Factor | Score | Evidence (cite filenames + specific code) |
|--------|-------|------------------------------------------|
| Alpha potential | N/5 | … |
| Novel insight | N/5 | … |
| Model sophistication | N/5 | … |
| Odds integration | N/5 | … |
| Data pipeline | N/5 | … |
| Backtesting rigor | N/5 | … |
| Stack adaptability | N/5 | … |
| Maintenance burden | N/5 | … |
| Out-of-box / free impact | N/5 | … |
| **Total** | **N/45** | |

**Verdict:**
- **Integrate** (36–45) — worth forking and building on directly
- **Mine for parts** (23–35) — extract specific components
- **Reference only** (12–22) — useful to read, not to use
- **Skip** (<12) — nothing here we don't already have

**Best component to steal:** one sentence — name the specific file/function/pattern

**Gaps / risks:** one sentence — the single biggest red flag found in the code

---

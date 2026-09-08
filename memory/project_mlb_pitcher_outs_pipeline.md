---
name: MLB Pitcher Outs Pipeline
description: MLB pitcher outs recorded UNDER pipeline — OLS bootstrap, UNDER 15pp+ plus-odds; Lambda built, NOT YET DEPLOYED
type: project
---

Strategy: UNDER only, edge ≥ 15pp, plus_odds (under_price > 2.0), shrinkage=0.25.

OOS ROI: +9.8% combined (2025: +11.6% n=274, 2026: +4.3% n=92). Concentrated at line 17.5 — declining/injured pitchers with high book lines.

Model: OLS LinearRegression, 10 features (consensus_line dominant at coef +0.74), 10k bootstrap residuals, residual std=3.54.

Spine: 88,458 rows (2024-2026), 10,458 player-games. In S3: `mlb/pitcher_outs_model/spine/mlb_pitcher_outs_spine.parquet`. Reuses strikeouts gamelogs from `mlb/strikeouts_model/pitcher_gamelogs/`.

Key UTC→ET date issue: raw market S3 files used UTC dates; spine handles via date-1 fallback join (26% of rows resolved this way).

**Lambda**: `mlb-pitcher-outs-daily` — built but NOT YET DEPLOYED.

**Why**: Need to deploy and verify SES email before enabling rules.

**How to apply**: Before enabling EventBridge rules, run deploy script, test pipeline mode manually, confirm email arrives. Then enable 3 rules: mlb-po-spine-daily-8am-et, mlb-po-pipeline-daily-830am-et, mlb-po-settle-daily-830am-et.

Session log: `knowledge-base/raw/20260706-mlb-pitcher-outs.html`

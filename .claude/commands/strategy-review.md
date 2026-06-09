# /strategy-review

Perform a systematic performance review of a betting strategy and produce a decision-ready assessment with a threshold recalibration recommendation.

**Arguments:** `$ARGUMENTS` — optional strategy name and filters, e.g. `rebounds`, `rebounds --since 2026-03-01`, `rebounds --season 2025-26`. Defaults to `rebounds` with no date filter.

---

## Step 0 — Parse arguments

Extract from `$ARGUMENTS`:
- `strategy` — defaults to `rebounds`
- `--since YYYY-MM-DD` — if provided, pass to the analysis script
- `--season YYYY-YY` — if provided, pass to the analysis script

---

## Step 1 — Pull settled data (three windows)

Run `analyze_settled_results.py` three times in parallel to get the three time windows you need. For a `rebounds` strategy:

```bash
# All-time
python scripts/analyze_settled_results.py

# Last 30 days
python scripts/analyze_settled_results.py --since <today-30d>

# Last 7 days
python scripts/analyze_settled_results.py --since <today-7d>
```

If `--since` or `--season` was passed in `$ARGUMENTS`, also run one filtered pass with those args. Substitute today's actual date for `<today-Nd>`.

Capture all output. If S3 access fails (no AWS creds), note it and proceed with whatever local data exists in `data/04_output/`.

---

## Step 2 — Compute trend signals

From the three windows, build a trend table for each strategy bucket (`both`, `ols`, `xgb`):

| Bucket | All-time ROI | 30d ROI | 7d ROI | Trend |
|--------|-------------|---------|--------|-------|
| both   | …           | …       | …      | ↑/↓/→ |
| ols    | …           | …       | …      | ↑/↓/→ |
| xgb    | …           | …       | …      | ↑/↓/→ |

Trend direction: ↑ if 7d ROI > 30d ROI > all-time ROI, ↓ if the reverse, → if within 3pp of each other.

Also compute hit rate trend using the same logic.

---

## Step 3 — Probability calibration check

From the probability-bin table (Section 2 of the script output), identify:
- Which bins have positive ROI? Which have negative ROI?
- Is the model well-calibrated (i.e., does the 60-70% probability bin actually win ~60-70% of the time)?
- Flag any bin where actual hit rate deviates from predicted probability by more than 10pp.

---

## Step 4 — Edge bin check vs current threshold

Read the current `prod_min_edge` from `config/nba_rebounds_prod.yaml` (currently `0.05` = 5%).

From the edge-bin table (Section 3 of the script output), identify:
- Which edge bins have positive ROI?
- Is the current threshold cutting off profitable bets or including unprofitable ones?
- Compute the approximate PnL impact of raising the threshold to the next bin boundary.

---

## Step 5 — Market efficiency signal

Look for any of these signals in the data:
- Average `avg_implied_prob_taken` drifting higher over time (market tightening = harder to find edge)
- Sample size declining in recent windows (fewer qualifying props)
- Spread between model probability and implied prob shrinking in recent windows

If the data doesn't have enough granularity for this, note it as a gap.

---

## Step 6 — Write the assessment

Output a structured review with the following sections. Be specific — cite actual numbers, not generalities.

### Strategy Health: [strategy name] — [date]

**TL;DR:** One sentence verdict. (e.g. "OLS bucket is degrading, XGB holding; recommend raising `prod_min_edge` to 0.08.")

**Performance Trend**
[The trend table from Step 2. Flag any bucket where 7d ROI is more than 10pp below all-time ROI as a warning.]

**Model Calibration**
[Key finding from Step 3. Note miscalibrated bins. If calibration looks good, say so.]

**Threshold Assessment**
[Current `prod_min_edge`, what the edge bins say, and the recommendation: hold / raise / lower, with the specific value.]

**Market Efficiency**
[Finding from Step 5, or note the data gap if unable to assess.]

**Recommendation**
One of:
- **Hold** — performance is on trend, no action needed
- **Recalibrate threshold** — with specific new value and rationale
- **Investigate bucket** — flag which bucket and why (e.g. OLS diverging from XGB)
- **Pause strategy** — if 7d ROI is deeply negative (below -15%) across all buckets

**Next check-in:** Suggest a date for the next review (default: 2 weeks from today unless the recommendation is Pause or Investigate, in which case: 5 days).

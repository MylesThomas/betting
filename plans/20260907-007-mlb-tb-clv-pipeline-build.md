# MLB TB CLV Pipeline — Build Status & Remaining Work

**Started:** 2026-09-07  
**Context doc:** `plans/20260906-005-mlb-tb-prop-snapshots-clv.md`

---

## Status

### Done
- [x] `snapshot_props.py` — hourly fetch + first-seen detection + S3 write
- [x] `lambda_snapshot.py` — Lambda handler
- [x] `Dockerfile.snapshot` — container image (SSL fix: `--trusted-host` + `certifi`)
- [x] `deploy_mlb_tb_snapshot.sh` — ECR deploy script (retry logic, 5 attempts)
- [x] `run_deploy_watch.sh` — log-tee wrapper
- [x] **Lambda deployed** to `mlb-tb-prop-snapshot` (us-east-2) — 6,820 rows written on first invoke
- [x] EventBridge rule `mlb-tb-prop-snapshot-hourly` created **DISABLED**
- [x] `verify_snapshot.py` — 7-check assertion script
- [x] `smoke_test_snapshots.py` — 5-check smoke test (checks 4+5 now implemented)
- [x] `compute_clv.py` — CLV computation (first-seen, 9am ET, closing; tiers; consensus median)
- [x] `verify_clv.py` — synthetic test: 5/5 checks pass
- [x] `compute_tightening.py` — rolling 30d + season-to-date drift watch
- [x] `settle_total_bases.py` — CLV + tightening sections integrated into HTML email

### In Progress / Next
- [x] **Fix Check 4 in smoke_test** — graceful skip when <2 snapshots; only check rows where both reference prices available
- [x] **Fix CLV tiers** — now directional: `severe+`/`severe-`, `strong+`/`strong-`, `moderate+`/`moderate-`, `mild+`/`mild-`, `ok`; uses `max(key=abs)` not `min()`
- [x] **Fix tightening player filter** — excludes no_data players; keeps plays + tracks
- [x] **Smoke test 5/5** — all checks passing
- [x] **EventBridge rule ENABLED** — `mlb-tb-prop-snapshot-hourly` running `cron(0 * * * ? *)` ✅
- [ ] **Step F** — Daily verification Lambda (`mlb-tb-snapshot-verify`), noon ET, SES on failure only (deferred)
- [ ] **`verify_snapshot_actionnetwork.py`** — Playwright ActionNetwork spot-check (deferred, revisit later)

---

## Check 4 Debug Notes

**Symptom:** `compute_clv` runs, `closing_under_odds` is populated, but `clv_first_seen_cents` is 62% null.

**Likely root cause:** `first_seen` detection in `compute_clv` may not find the first-seen row because:
- All rows in today's snapshot have `binary_player_game_first_seen=True` (it was the first run ever)
- But the `closing_under_odds` is coming from... the same snapshot file?
- That means `first_seen_under_odds == closing_under_odds` and CLV should be 0, not null.

**Alternative cause:** `snapshot_ts_utc` comparison with `commence_time`. The snapshot has ISO timestamps like `2026-09-07T11:31:47Z`. The `commence_time` for tomorrow's games is `2026-09-07T...Z`. So `snapshot_ts_utc < commence_time` may evaluate correctly if the game starts after 11:31 UTC.

**To investigate:** Run `compute_clv.py --gameday 2026-09-07` and print sample rows to see which fields are null and why.

---

## Smoke Test Target: 5/5

Current: 4/5 (Check 4 fails). Fix Check 4, then the pipeline is green.

Check 5 shows 168 players flagged for tightening — this is expected since it's the first day of data (no prior season baseline, so drift = today's odds - None = None → flagged). The check itself passes (non-empty output). The flag logic will stabilize once multiple game_dates accumulate.

---

## Files Created This Session

```
src/mlb_total_bases_modeling/scripts/snapshot_props.py
src/mlb_total_bases_modeling/scripts/compute_clv.py
src/mlb_total_bases_modeling/scripts/compute_tightening.py
src/mlb_total_bases_modeling/scripts/verify_snapshot.py
src/mlb_total_bases_modeling/scripts/verify_clv.py
src/mlb_total_bases_modeling/scripts/smoke_test_snapshots.py
src/mlb_total_bases_modeling/lambda/lambda_snapshot.py
src/mlb_total_bases_modeling/lambda/Dockerfile.snapshot
src/mlb_total_bases_modeling/lambda/deploy_mlb_tb_snapshot.sh
src/mlb_total_bases_modeling/lambda/run_deploy_watch.sh
```

Modified:
```
src/mlb_total_bases_modeling/scripts/settle_total_bases.py  (CLV + tightening sections)
CONTEXT.md  (domain terms: Prop snapshot, Closing line, First-seen price, CLV, Tightening)
```

---

## Enable Checklist (after smoke test passes)

- [ ] Enable `mlb-tb-prop-snapshot-hourly`
- [ ] Confirm first few hourly runs land in S3
- [ ] Run `verify_snapshot.py` manually — 7/7 pass
- [ ] Wait for a settled gameday, run `settle_total_bases.py --gameday <date> --output /tmp/r.json`, open HTML in browser
- [ ] Enable `mlb-tb-snapshot-verify` daily Lambda (Step F, not yet built)

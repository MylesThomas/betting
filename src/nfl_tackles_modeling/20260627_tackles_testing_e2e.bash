#!/bin/bash
# NFL Tackles pipeline — end-to-end test
# Run from repo root or anywhere (auto-cds to repo root):
#   bash src/nfl_tackles_modeling/20260627_tackles_testing_e2e.bash              # defaults hist gameday to 2025-09-29
#   bash src/nfl_tackles_modeling/20260627_tackles_testing_e2e.bash 2025-10-12   # override hist gameday

set -euo pipefail

# Always run from repo root regardless of where the script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export ODDS_API_KEY="d056323e8fc1192904ff77bf1441bfbb"
export SNS_TOPIC_ARN="arn:aws:sns:us-east-2:232692785472:betting-arb-alerts"
export SETTLEMENT_SES_SOURCE="tqstrats@gmail.com"
export SETTLEMENT_SES_TO="mylescgthomas@gmail.com"

HIST_GAMEDAY="${1:-2025-09-29}"
SETTLED_S3_KEY="nfl/tackles_model/settled/settled_bets.parquet"
S3_BUCKET="the-odds-api-mt"
TODAY_ET=$(python -c "from datetime import datetime; from zoneinfo import ZoneInfo; print(datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d'))")

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG="/tmp/tackles_test_$(date +%Y%m%d_%H%M%S).log"
OVERALL_START=$(date +%s)
PASS=0
FAIL=0

log()     { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
ok()      { log "  ✅ $*"; PASS=$((PASS + 1)); }
fail()    { log "  ❌ $*"; FAIL=$((FAIL + 1)); }
elapsed() { echo "$(($(date +%s) - $1))s"; }

log "============================================================"
log "NFL TACKLES E2E TEST  —  $(date '+%Y-%m-%d %H:%M:%S ET')"
log "Historical gameday: $HIST_GAMEDAY"
log "============================================================"
log "Log file: $LOG"
log ""

# ── Env var check ─────────────────────────────────────────────────────────────
log "--- ENV CHECK ---"
[ -n "$SETTLEMENT_SES_SOURCE" ] && ok "SETTLEMENT_SES_SOURCE set" || fail "SETTLEMENT_SES_SOURCE not set — emails will be skipped"
[ -n "$SETTLEMENT_SES_TO" ]     && ok "SETTLEMENT_SES_TO set"     || fail "SETTLEMENT_SES_TO not set — emails will be skipped"
[ -n "$SNS_TOPIC_ARN" ]         && ok "SNS_TOPIC_ARN set"         || fail "SNS_TOPIC_ARN not set — SNS skipped"
[ -n "$ODDS_API_KEY" ]          && ok "ODDS_API_KEY set"          || fail "ODDS_API_KEY not set — pipeline step will fail"
log ""

# ── STEP 1: Syntax check ──────────────────────────────────────────────────────
log "--- STEP 1: Syntax check all new files ---"
T=$(date +%s)
if python -m py_compile \
    src/nfl_tackles_modeling/lambda/lambda_function.py \
    src/nfl_tackles_modeling/scripts/update_spine.py \
    src/nfl_tackles_modeling/scripts/run_pipeline.py \
    src/nfl_tackles_modeling/scripts/settle_tackles.py 2>>"$LOG"; then
  ok "All 4 files compile clean  ($(elapsed $T))"
else
  fail "Syntax error — check $LOG"
  exit 1
fi
log ""

# ── STEP 2: settle_tackles — no args, yesterday default, proves SES email ─────
log "--- STEP 2: settle_tackles.py (yesterday default → proves SES email) ---"
T=$(date +%s)
if python src/nfl_tackles_modeling/scripts/settle_tackles.py 2>&1 | tee -a "$LOG"; then
  ok "settle_tackles (no-CSV path) exited OK  ($(elapsed $T))"
else
  fail "settle_tackles (no-CSV path) failed  ($(elapsed $T))"
fi
log ""

# ── STEP 3: Run real inference on historical labeled data + upload recs to S3 ──
log "--- STEP 3: Real inference on ${HIST_GAMEDAY} labeled data ---"
T=$(date +%s)
HIST_GAMEDAY_PY="$HIST_GAMEDAY"
if [[ "$HIST_GAMEDAY" < "$TODAY_ET" ]]; then
python << PYEOF 2>&1 | tee -a "$LOG"
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src/nfl_tackles_modeling/scripts")

import numpy as np
import pandas as pd
import boto3
import nfl_data_py as nfl

from infer import (
    LABELED_PATH, DROP_POSITIONS,
    add_derived, load_artifacts, run_inference, filter_bets, ARTIFACT_DIR,
)

GAMEDAY = "$HIST_GAMEDAY_PY"
SEASON  = int(GAMEDAY[:4])
BUCKET  = "the-odds-api-mt"

# Get week for this gameday
sched = nfl.import_schedules([SEASON])
sched = sched[sched["game_type"] == "REG"].copy()
sched["gameday_str"] = pd.to_datetime(sched["gameday"]).dt.strftime("%Y-%m-%d")
weeks = sched[sched["gameday_str"] == GAMEDAY]["week"].unique().tolist()
if not weeks:
    print(f"  ERROR: no schedule entries for {GAMEDAY}")
    sys.exit(1)
print(f"  {GAMEDAY} = Week(s) {weeks}")

# Load per-book dataset (already one row per player-book-game), filter to this week
print(f"  Loading per-book dataset from {LABELED_PATH}...")
df = pd.read_parquet(LABELED_PATH)
df = df[(df["season"] == SEASON) & (df["week"].isin(weeks))].copy()
df = df[df["position"].notna() & ~df["position"].isin(DROP_POSITIONS)].copy()
df = add_derived(df)
print(f"  Player-book rows this gameday: {len(df)}  ({df['player_name_norm'].nunique()} players)")

if df.empty:
    print(f"  No rows for {GAMEDAY} — check LABELED_PATH and season/week")
    sys.exit(1)

# Run inference
print(f"  Loading artifacts from {ARTIFACT_DIR}...")
artifacts = load_artifacts(ARTIFACT_DIR)
results = run_inference(df, artifacts)

# Apply production filter
bets = filter_bets(results)
print(f"  Qualifying bets (UNDER, edge≥5pp, lines 4.5-9.5): {len(bets)}")
if bets.empty:
    print("  No qualifying bets — nothing to upload")
    sys.exit(0)

# Build recommendations CSV
recs = bets[[
    "player_name", "player_name_norm", "team", "position", "game_id",
    "book", "offered_line", "p_hybrid", "p_market", "edge",
    "consensus_under_price", "consensus_over_price", "n_books",
]].rename(columns={
    "player_name_norm": "player_norm",
    "game_id":          "event_id",
})
recs["streak"]              = 0
recs["cold_streak_warning"] = False

print(f"\n  {'Player':<26} {'Team':<5} {'Line':>5}  {'P(Under)':>9}  {'Mkt P(U)':>9}  {'Edge':>7}  {'Odds':>6}  {'Book'}")
for _, r in recs.iterrows():
    print(f"  {r['player_name']:<26} {r['team']:<5} {r['offered_line']:>5.1f}  "
          f"{r['p_hybrid']*100:>8.1f}%  {r['p_market']*100:>8.1f}%  "
          f"{abs(r['edge'])*100:>6.1f}pp  {int(r['consensus_under_price']):>+d}  {r['book']}")

key = f"nfl/tackles_model/daily_runs/{GAMEDAY}/recommendations.csv"
boto3.client("s3").put_object(
    Bucket=BUCKET, Key=key, Body=recs.to_csv(index=False).encode()
)
print(f"\n  Uploaded → s3://{BUCKET}/{key}")
PYEOF

  if [ ${PIPESTATUS[0]} -eq 0 ]; then
    ok "Real inference complete + recs uploaded  ($(elapsed $T))"
  else
    fail "Inference / upload failed  ($(elapsed $T))"
    exit 1
  fi
else
  log "  HIST_GAMEDAY is today or future — skipping"
  ok "Skipped (not a past date)  ($(elapsed $T))"
fi
log ""

# ── STEP 4: Settle historical date against real PFR data ──────────────────────
log "--- STEP 4: settle_tackles.py --gameday ${HIST_GAMEDAY} ---"
T=$(date +%s)
if [[ "$HIST_GAMEDAY" < "$TODAY_ET" ]]; then
  if python src/nfl_tackles_modeling/scripts/settle_tackles.py \
      --gameday "$HIST_GAMEDAY" 2>&1 | tee -a "$LOG"; then
    ok "Historical settle exited OK  ($(elapsed $T))"
  else
    fail "Historical settle failed  ($(elapsed $T))"
  fi
else
  log "  HIST_GAMEDAY is today or future — skipping"
  ok "Skipped (not a past date)  ($(elapsed $T))"
fi
log ""

# ── STEP 5: Cleanup test artifacts from S3 ────────────────────────────────────
log "--- STEP 5: Cleanup test artifacts from S3 ---"
T=$(date +%s)
aws s3 rm "s3://${S3_BUCKET}/${SETTLED_S3_KEY}" 2>&1 | tee -a "$LOG" || true
aws s3 rm "s3://${S3_BUCKET}/nfl/tackles_model/daily_runs/${HIST_GAMEDAY}/recommendations.csv" 2>&1 | tee -a "$LOG" || true
ok "Test artifacts cleaned up  ($(elapsed $T))"
log ""

# ── STEP 6: run_pipeline — offseason, no NFL events ──────────────────────────
log "--- STEP 6: run_pipeline.py (offseason → no events) ---"
T=$(date +%s)
if python src/nfl_tackles_modeling/scripts/run_pipeline.py \
    --gameday 2026-06-27 2>&1 | tee -a "$LOG"; then
  ok "run_pipeline exited OK  ($(elapsed $T))"
else
  fail "run_pipeline failed  ($(elapsed $T))"
fi
log ""

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
TOTAL=$(($(date +%s) - OVERALL_START))
log "============================================================"
log "SUMMARY  —  total ${TOTAL}s"
log "  Passed : $PASS"
log "  Failed : $FAIL"
log "  Log    : $LOG"
log "============================================================"

exit $FAIL

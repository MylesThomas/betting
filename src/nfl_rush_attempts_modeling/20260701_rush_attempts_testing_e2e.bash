#!/bin/bash
# NFL Rush Attempts pipeline — end-to-end test
# Run from repo root or anywhere (auto-cds to repo root):
#   bash src/nfl_rush_attempts_modeling/20260701_rush_attempts_testing_e2e.bash              # defaults hist gameday to 2025-10-05
#   bash src/nfl_rush_attempts_modeling/20260701_rush_attempts_testing_e2e.bash 2025-11-02   # override hist gameday

set -euo pipefail

# Always run from repo root regardless of where the script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export ODDS_API_KEY="d056323e8fc1192904ff77bf1441bfbb"
export SNS_TOPIC_ARN="arn:aws:sns:us-east-2:232692785472:betting-arb-alerts"
export SETTLEMENT_SES_SOURCE="tqstrats@gmail.com"
export SETTLEMENT_SES_TO="mylescgthomas@gmail.com"

# Load .env for any additional vars (ODDS_API_KEY etc may already be set above)
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$REPO_ROOT/.env"
  set +a
fi

PYTHON="$REPO_ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "❌ .venv not found at $REPO_ROOT/.venv — run: python -m venv .venv && pip install -r requirements.txt"
  exit 1
fi

HIST_GAMEDAY="${1:-2025-10-05}"
SETTLED_S3_KEY="nfl/rush_attempts_model/settled/settled_bets.parquet"
S3_BUCKET="the-odds-api-mt"
TODAY_ET=$("$PYTHON" -c "from datetime import datetime; from zoneinfo import ZoneInfo; print(datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d'))")

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG="/tmp/rush_attempts_test_$(date +%Y%m%d_%H%M%S).log"
OVERALL_START=$(date +%s)
PASS=0
FAIL=0

log()     { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
ok()      { log "  ✅ $*"; PASS=$((PASS + 1)); }
fail()    { log "  ❌ $*"; FAIL=$((FAIL + 1)); }
elapsed() { echo "$(($(date +%s) - $1))s"; }

log "============================================================"
log "NFL RUSH ATTEMPTS E2E TEST  —  $(date '+%Y-%m-%d %H:%M:%S ET')"
log "Historical gameday: $HIST_GAMEDAY"
log "============================================================"
log "Log file: $LOG"
log ""

# ── Env var check ─────────────────────────────────────────────────────────────
log "--- ENV CHECK ---"
[ -n "${SETTLEMENT_SES_SOURCE:-}" ] && ok "SETTLEMENT_SES_SOURCE=${SETTLEMENT_SES_SOURCE}" || fail "SETTLEMENT_SES_SOURCE not set — emails will be skipped"
[ -n "${SETTLEMENT_SES_TO:-}" ]     && ok "SETTLEMENT_SES_TO=${SETTLEMENT_SES_TO}"         || fail "SETTLEMENT_SES_TO not set — emails will be skipped"
[ -n "${SNS_TOPIC_ARN:-}" ]         && ok "SNS_TOPIC_ARN set"                               || fail "SNS_TOPIC_ARN not set — SNS skipped"
[ -n "${ODDS_API_KEY:-}" ]          && ok "ODDS_API_KEY set"                                || fail "ODDS_API_KEY not set — pipeline step will fail"
log ""

# ── STEP 1: Syntax check ──────────────────────────────────────────────────────
log "--- STEP 1: Syntax check all pipeline files ---"
T=$(date +%s)
if "$PYTHON" -m py_compile \
    src/nfl_rush_attempts_modeling/lambda/lambda_function.py \
    src/nfl_rush_attempts_modeling/scripts/update_spine.py \
    src/nfl_rush_attempts_modeling/scripts/run_pipeline.py \
    src/nfl_rush_attempts_modeling/scripts/settle_rush_attempts.py 2>>"$LOG"; then
  ok "All 4 files compile clean  ($(elapsed $T))"
else
  fail "Syntax error — check $LOG"
  exit 1
fi
log ""

# ── STEP 2: settle_rush_attempts — no args, yesterday default, proves SES email ─
log "--- STEP 2: settle_rush_attempts.py (yesterday default → proves SES email) ---"
T=$(date +%s)
if "$PYTHON" src/nfl_rush_attempts_modeling/scripts/settle_rush_attempts.py 2>&1 | tee -a "$LOG"; then
  ok "settle_rush_attempts (no-CSV path) exited OK  ($(elapsed $T))"
else
  fail "settle_rush_attempts (no-CSV path) failed  ($(elapsed $T))"
fi
log ""

# ── STEP 3: Score historical game day with real model + real odds ──────────────
log "--- STEP 3: score_historical.py --gameday ${HIST_GAMEDAY} ---"
T=$(date +%s)
if [[ "$HIST_GAMEDAY" < "$TODAY_ET" ]]; then
  if "$PYTHON" src/nfl_rush_attempts_modeling/scripts/score_historical.py \
      --gameday "$HIST_GAMEDAY" 2>&1 | tee -a "$LOG"; then
    ok "score_historical exited OK  ($(elapsed $T))"
  else
    fail "score_historical failed  ($(elapsed $T))"
    exit 1
  fi
else
  log "  HIST_GAMEDAY is today or future — skipping"
  ok "Skipped (not a past date)  ($(elapsed $T))"
fi
log ""

# ── STEP 4: Settle historical date against real nfl_data_py data ───────────────
log "--- STEP 4: settle_rush_attempts.py --gameday ${HIST_GAMEDAY} ---"
T=$(date +%s)
if [[ "$HIST_GAMEDAY" < "$TODAY_ET" ]]; then
  if "$PYTHON" src/nfl_rush_attempts_modeling/scripts/settle_rush_attempts.py \
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
aws s3 rm "s3://${S3_BUCKET}/nfl/rush_attempts_model/daily_runs/${HIST_GAMEDAY}/recommendations.csv" 2>&1 | tee -a "$LOG" || true
ok "Test artifacts cleaned up  ($(elapsed $T))"
log ""

# ── STEP 6: run_pipeline — offseason, no NFL events ───────────────────────────
log "--- STEP 6: run_pipeline.py (offseason → no events / pipeline smoke test) ---"
T=$(date +%s)
if "$PYTHON" src/nfl_rush_attempts_modeling/scripts/run_pipeline.py \
    --gameday 2026-07-01 2>&1 | tee -a "$LOG"; then
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

#!/bin/bash
# Wrapper: runs deploy and tees all output to /tmp/mlb_tb_deploy.log
# Claude watches the log file to catch errors and iterate autonomously.
LOG=/tmp/mlb_tb_deploy.log
echo "" > "$LOG"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/deploy_mlb_tb_snapshot.sh" 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"

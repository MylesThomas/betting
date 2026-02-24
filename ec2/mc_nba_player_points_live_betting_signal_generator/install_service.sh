#!/usr/bin/env bash
#
# Install or update mc-live-betting systemd unit from the repo.
# Run on the EC2 instance after 'git pull' to redeploy the service definition.
#
# Usage (on instance, from repo root or this dir):
#   sudo bash ec2/mc_nba_player_points_live_betting_signal_generator/install_service.sh
#
# Prereqs: /etc/mc-live-betting/env exists with ODDS_API_KEY (create once if needed).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UNIT_NAME="mc-live-betting.service"
SOURCE="$SCRIPT_DIR/$UNIT_NAME"
DEST="/etc/systemd/system/$UNIT_NAME"

if [ ! -f "$SOURCE" ]; then
  echo "Error: $SOURCE not found. Run from repo root or ensure file exists."
  exit 1
fi

cp "$SOURCE" "$DEST"
systemctl daemon-reload
echo "Installed $DEST and reloaded systemd."

if systemctl is-enabled "$UNIT_NAME" &>/dev/null; then
  systemctl restart "$UNIT_NAME"
  echo "Restarted $UNIT_NAME."
else
  systemctl enable "$UNIT_NAME"
  echo "Enabled $UNIT_NAME. Start with: sudo systemctl start $UNIT_NAME"
fi

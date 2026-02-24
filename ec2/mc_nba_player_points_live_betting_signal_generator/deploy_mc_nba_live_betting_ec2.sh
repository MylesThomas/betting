#!/bin/bash
###############################################################################
# Deploy MC NBA Player Points Live Betting Signal Generator (EC2)
#
# Generates EC2 user-data and runs prereq checks. Does not launch the instance;
# you launch with the printed aws ec2 run-instances command (or Console).
#
# Prerequisites:
# - AWS CLI configured (credentials, region)
# - ODDS_API_KEY environment variable set
# - minute_by_minute.parquet in S3 (see README)
#
# Usage:
#   export ODDS_API_KEY="your-key"
#   cd ~/dev/betting && bash ec2/mc_nba_player_points_live_betting_signal_generator/deploy_mc_nba_live_betting_ec2.sh
#
# Optional:
#   REPO_URL="https://github.com/yourorg/betting.git"  # for user-data to clone into /home/ubuntu/betting
#
# Launch instance from CLI (after prereqs + user-data):
#   KEY_NAME=my-key SECURITY_GROUP_ID=sg-xxx [IAM_INSTANCE_PROFILE=role-name] \
#     bash .../deploy_mc_nba_live_betting_ec2.sh --launch
#
# Created: 2026-02-20
###############################################################################

set -e
export AWS_PAGER=""

REGION="${AWS_REGION:-us-east-2}"
S3_MINUTE_BY_MINUTE="s3://nba-betting-mt/data/01_input/pbp_data/minute_by_minute.parquet"
REPO_URL="${REPO_URL:-}"

LAUNCH_EC2=false
[ "${1:-}" = "--launch" ] && LAUNCH_EC2=true

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
USER_DATA_FILE="$SCRIPT_DIR/ec2_user_data.sh"

echo "================================================================================"
echo "DEPLOY MC NBA LIVE BETTING SIGNAL GENERATOR (EC2)"
echo "================================================================================"
echo ""

###############################################################################
# Step 1: Verify prerequisites
###############################################################################
echo "Step 1: Verifying prerequisites..."
echo ""

if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI not found. Install it first.${NC}"
    exit 1
fi
echo "✅ AWS CLI found"

if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}❌ AWS credentials not configured${NC}"
    exit 1
fi
echo "✅ AWS credentials configured"

if [ -z "$ODDS_API_KEY" ]; then
    echo -e "${RED}❌ ODDS_API_KEY environment variable not set${NC}"
    echo "   Run: export ODDS_API_KEY='your-key-here'"
    exit 1
fi
echo "✅ ODDS_API_KEY set"
echo ""

###############################################################################
# Step 2: Check S3 for minute_by_minute.parquet
###############################################################################
echo "Step 2: Checking S3 for minute_by_minute.parquet..."
echo ""

if aws s3 ls "$S3_MINUTE_BY_MINUTE" --region "$REGION" &> /dev/null; then
    echo -e "${GREEN}✅ Found: $S3_MINUTE_BY_MINUTE${NC}"
else
    echo -e "${YELLOW}⚠️  Not found: $S3_MINUTE_BY_MINUTE${NC}"
    echo "   Upload with:"
    echo "   aws s3 cp data/minute_by_minute.parquet $S3_MINUTE_BY_MINUTE"
    echo "   (Generate data via src/pbp_data/01_get_game_ids.py → 02 → 03_process_data.py)"
    echo ""
fi

###############################################################################
# Step 3: Write EC2 user-data script
###############################################################################
echo "Step 3: Writing EC2 user-data to $USER_DATA_FILE..."
echo ""

# Build the "repo setup" block: clone if REPO_URL set, else just mkdir data
if [ -n "$REPO_URL" ]; then
    REPO_SETUP="git clone $REPO_URL /home/ubuntu/betting || true"
else
    REPO_SETUP="mkdir -p /home/ubuntu/betting/data"
fi

cat > "$USER_DATA_FILE" << 'USERDATA_HEADER'
#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive

# Install Python, pip, git, AWS CLI
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv git awscli

# Repo and data dir
sudo -u ubuntu bash -c 'USERDATA_REPO_SETUP'
sudo -u ubuntu mkdir -p /home/ubuntu/betting/data

# Sync minute_by_minute from S3 (instance profile must have S3 read)
sudo -u ubuntu aws s3 cp s3://nba-betting-mt/data/01_input/pbp_data/minute_by_minute.parquet /home/ubuntu/betting/data/minute_by_minute.parquet --region us-east-2 || true

# Install Python deps (if repo exists with requirements)
if [ -f /home/ubuntu/betting/requirements.txt ]; then
    sudo -u ubuntu /usr/bin/python3 -m pip install --user -q -r /home/ubuntu/betting/requirements.txt || true
else
    sudo -u ubuntu /usr/bin/python3 -m pip install --user -q requests pandas boto3 duckdb pytz python-dotenv || true
fi

# Systemd unit: run live signal generator in a loop (from repo so redeploy = git pull + install_service.sh)
# ODDS_API_KEY must be in /etc/mc-live-betting/env (create after first boot if not set at launch)
mkdir -p /etc/mc-live-betting
if [ -f /home/ubuntu/betting/ec2/mc_nba_player_points_live_betting_signal_generator/mc-live-betting.service ]; then
  cp /home/ubuntu/betting/ec2/mc_nba_player_points_live_betting_signal_generator/mc-live-betting.service /etc/systemd/system/
else
  cat > /etc/systemd/system/mc-live-betting.service << 'UNIT'
[Unit]
Description=MC NBA Live Betting Signal Generator
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/betting
EnvironmentFile=-/etc/mc-live-betting/env
ExecStart=/usr/bin/python3 -u src/pbp_data/10_live_betting_signal_generator.py --loop --interval 60
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
UNIT
fi
systemctl daemon-reload
# Do not start yet: ODDS_API_KEY may not be in /etc/mc-live-betting/env. User can start after setting it.
# systemctl enable --now mc-live-betting
systemctl enable mc-live-betting
USERDATA_HEADER

# Inject the repo setup command (clone or mkdir). Use # delimiter so REPO_SETUP can contain |
sed -i.bak "s#USERDATA_REPO_SETUP#$REPO_SETUP#" "$USER_DATA_FILE"
rm -f "${USER_DATA_FILE}.bak"

echo -e "${GREEN}✅ User-data written${NC}"
echo ""

###############################################################################
# Step 4: Print launch instructions
###############################################################################
echo "================================================================================"
echo "Step 4: Launch EC2 instance"
echo "================================================================================"
echo ""
echo "1. Create an IAM instance profile with S3 read/write for nba-betting-mt (see README)."
echo "2. Launch Ubuntu 22.04 (or similar) with:"
echo ""
echo "   aws ec2 run-instances \\"
echo "     --region $REGION \\"
echo "     --image-id <Ubuntu-22.04-AMI> \\"
echo "     --instance-type t4g.micro \\"
echo "     --iam-instance-profile Name=<your-profile> \\"
echo "     --user-data file://$USER_DATA_FILE \\"
echo "     --key-name <your-key-pair> \\"
echo "     --security-group-ids <sg-xxx>"
echo ""
echo "   Or use the AWS Console: paste the contents of $USER_DATA_FILE into User data."
echo ""
echo "3. After first boot:"
echo "   - If you did not set REPO_URL: SSH in, clone/copy repo to /home/ubuntu/betting, install deps."
echo "   - Create /etc/mc-live-betting/env with: ODDS_API_KEY=your-key"
echo "   - Run: sudo systemctl start mc-live-betting"
echo "   - Logs: journalctl -u mc-live-betting -f"
echo ""

###############################################################################
# Step 4b: Launch instance from CLI (optional, when --launch is passed)
###############################################################################
if [ "$LAUNCH_EC2" = true ]; then
    echo "================================================================================"
    echo "Step 4b: Launching EC2 instance"
    echo "================================================================================"
    echo ""

    if [ -z "$KEY_NAME" ]; then
        echo -e "${RED}❌ KEY_NAME not set. Set it for SSH access.${NC}"
        echo "   Example: export KEY_NAME=my-keypair"
        exit 1
    fi
    if [ -z "$SECURITY_GROUP_ID" ]; then
        echo -e "${RED}❌ SECURITY_GROUP_ID not set.${NC}"
        echo "   Example: export SECURITY_GROUP_ID=sg-0123456789abcdef0"
        echo "   Create in Console: EC2 → Security Groups → Create, allow SSH (22) from your IP."
        exit 1
    fi

    # Use AMI_ID from env, or resolve via SSM (no EC2 perm needed) or describe-images
    if [ -n "${AMI_ID:-}" ]; then
        echo "Using AMI_ID from environment: $AMI_ID"
    else
        echo "Resolving Ubuntu 22.04 LTS AMI..."
        AMI_ID=$(aws ssm get-parameters --names /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp3/ami-id --region "$REGION" --query 'Parameters[0].Value' --output text 2>/dev/null) || true
        if [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ]; then
            AMI_ID=$(aws ec2 describe-images \
                --region "$REGION" \
                --owners 099720109477 \
                --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" "Name=state,Values=available" \
                --query 'sort_by(Images,&CreationDate)[-1].ImageId' \
                --output text 2>/dev/null) || true
        fi
        if [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ]; then
            echo -e "${RED}❌ Could not resolve AMI. Set AMI_ID from Console:${NC}"
            echo "   EC2 → Launch instance → pick 'Ubuntu Server 22.04 LTS' → in the list, copy the AMI ID (e.g. ami-0abc123...)"
            echo "   Then: export AMI_ID=ami-xxxxxxxx"
            exit 1
        fi
        echo "   AMI: $AMI_ID"
    fi
    echo ""

    RUN_ARGS=(
        --region "$REGION"
        --image-id "$AMI_ID"
        --instance-type t3.micro
        --key-name "$KEY_NAME"
        --security-group-ids "$SECURITY_GROUP_ID"
        --user-data "file://$USER_DATA_FILE"
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=mc-nba-live-betting}]"
    )
    if [ -n "${IAM_INSTANCE_PROFILE:-}" ]; then
        RUN_ARGS+=(--iam-instance-profile "Name=$IAM_INSTANCE_PROFILE")
    else
        echo -e "${YELLOW}⚠️  IAM_INSTANCE_PROFILE not set. Instance will not have S3 access until you attach a role.${NC}"
        echo "   User-data will still run; minute_by_minute sync from S3 will fail until role is attached."
        echo ""
    fi

    echo "Running: aws ec2 run-instances ..."
    INSTANCE_ID=$(aws ec2 run-instances "${RUN_ARGS[@]}" --query 'Instances[0].InstanceId' --output text)
    echo -e "${GREEN}✅ Launched instance: $INSTANCE_ID${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Wait for running: aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION"
    echo "  2. Get public IP:   aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION --query 'Reservations[0].Instances[0].PublicIpAddress' --output text"
    echo "  3. SSH (when ready): ssh -i <your-key.pem> ubuntu@<public-ip>"
    echo "  4. On instance: create /etc/mc-live-betting/env with ODDS_API_KEY=..., then: sudo systemctl start mc-live-betting"
    echo "  5. Logs: journalctl -u mc-live-betting -f"
    echo ""
fi

###############################################################################
# Step 5: Local test (one iteration or --help)
###############################################################################
echo "================================================================================"
echo "Step 5: Local test"
echo "================================================================================"
echo ""

cd "$REPO_ROOT"
MINUTE_FILE="$REPO_ROOT/data/minute_by_minute.parquet"

if [ -f "$MINUTE_FILE" ]; then
    echo "Running one iteration (no --loop) to verify pipeline..."
    export ODDS_API_KEY
    python3 -u src/pbp_data/10_live_betting_signal_generator.py --min-edge 0.10 --n-sims 500 2>&1 | head -80
    echo ""
    echo -e "${GREEN}✅ Local test completed (see above for any errors)${NC}"
else
    echo "No local $MINUTE_FILE; running --help to verify imports..."
    python3 src/pbp_data/10_live_betting_signal_generator.py --help
    echo -e "${GREEN}✅ Imports OK${NC}"
fi

echo ""
echo "================================================================================"
echo -e "${GREEN}Deploy script finished.${NC} See README for full instructions."
echo "================================================================================"

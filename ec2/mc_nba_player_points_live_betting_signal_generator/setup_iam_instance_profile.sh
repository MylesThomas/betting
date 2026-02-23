#!/bin/bash
###############################################################################
# Create IAM role + instance profile for MC NBA Live Betting EC2
#
# Use this if your existing role (e.g. betting-dashboard-daily-update-role)
# is Lambda-only and cannot be assumed by EC2.
#
# Creates:
# - Role: mc-nba-live-betting-ec2-role (trust: ec2.amazonaws.com)
# - Inline policy: S3 read/write on nba-betting-mt, read on the-odds-api-mt
# - Instance profile: mc-nba-live-betting-ec2-role (same name as role)
#
# Usage:
#   bash ec2/mc_nba_player_points_live_betting_signal_generator/setup_iam_instance_profile.sh
#
# Then: export IAM_INSTANCE_PROFILE=mc-nba-live-betting-ec2-role
#
# Requires IAM permissions: iam:CreateRole, iam:PutRolePolicy, iam:CreateInstanceProfile,
# iam:AddRoleToInstanceProfile, iam:GetRole, iam:GetInstanceProfile.
# If you get AccessDenied, ask an admin to run this script or create the role manually
# (see README: "Manual IAM setup").
###############################################################################

set -e
export AWS_PAGER=""

REGION="${AWS_REGION:-us-east-2}"
ROLE_NAME="mc-nba-live-betting-ec2-role"
PROFILE_NAME="mc-nba-live-betting-ec2-role"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Creating IAM role and instance profile for EC2..."
echo "   (Requires IAM permissions. If you get AccessDenied, see README: Manual IAM setup.)"
echo ""

# 1. Create role with EC2 trust policy
echo "1. Creating role: $ROLE_NAME"
if aws iam get-role --role-name "$ROLE_NAME" &>/dev/null; then
    echo "   Role already exists, skipping create."
else
    if ! aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document '{
          "Version": "2012-10-17",
          "Statement": [
            {
              "Effect": "Allow",
              "Principal": { "Service": "ec2.amazonaws.com" },
              "Action": "sts:AssumeRole"
            }
          ]
        }' \
        --description "EC2 instance role for MC NBA live betting signal generator (S3 access)"; then
        echo ""
        echo -e "${YELLOW}If you got AccessDenied: your IAM user lacks permission to create roles.${NC}"
        echo "  Option 1: Ask an admin to run this script."
        echo "  Option 2: See README section 'Manual IAM setup' and have an admin create the role in the Console."
        exit 1
    fi
    echo "   Created."
fi
echo ""

# 2. Attach inline policy for S3
echo "2. Attaching S3 policy to role"
POLICY='{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "NbaBettingBucket",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::nba-betting-mt",
        "arn:aws:s3:::nba-betting-mt/*"
      ]
    },
    {
      "Sid": "OddsApiBucketRead",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::the-odds-api-mt",
        "arn:aws:s3:::the-odds-api-mt/*"
      ]
    }
  ]
}'

aws iam put-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-name "S3NbaAndOddsApi" \
    --policy-document "$POLICY"
echo "   Policy attached."
echo ""

# 3. Create instance profile (if missing) and add role to it
echo "3. Creating instance profile: $PROFILE_NAME"
if aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" &>/dev/null; then
    echo "   Instance profile already exists."
else
    aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME"
    echo "   Created."
fi

# Add role to instance profile (idempotent: fails if already added, we ignore)
aws iam add-role-to-instance-profile \
    --instance-profile-name "$PROFILE_NAME" \
    --role-name "$ROLE_NAME" 2>/dev/null || true
echo "   Role added to instance profile."
echo ""

echo -e "${GREEN}Done.${NC}"
echo ""
echo "Use this when launching the EC2 instance:"
echo "  export IAM_INSTANCE_PROFILE=$PROFILE_NAME"
echo ""
echo "Full launch exports (also need KEY_NAME and SECURITY_GROUP_ID):"
echo "  export ODDS_API_KEY=\"your-key\""
echo "  export KEY_NAME=\"your-key-pair-name\""
echo "  export SECURITY_GROUP_ID=\"sg-xxxxxxxx\""
echo "  export IAM_INSTANCE_PROFILE=\"$PROFILE_NAME\""
echo "  cd ~/dev/betting && bash ec2/mc_nba_player_points_live_betting_signal_generator/deploy_mc_nba_live_betting_ec2.sh --launch"
echo ""

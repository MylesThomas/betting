# MC NBA Player Points – Live Betting Signal Generator (EC2)

Runs the live Monte Carlo signal generator on a single EC2 instance: scans for live NBA games every 60 seconds, runs MC simulations, and writes signals to S3. Email for signals is planned but **off for now**.

## Quick start

**Step 1: Put minute_by_minute in S3**  
(Generate via pipeline 01→03 if needed, then upload.)

```bash
aws s3 cp data/minute_by_minute.parquet s3://nba-betting-mt/data/01_input/pbp_data/minute_by_minute.parquet
```

**Step 2: Create IAM role + instance profile for EC2**  
You cannot use the Lambda execution role (`betting-dashboard-daily-update-role-ille2llh`) for EC2; it only trusts Lambda. Run:

```bash
cd ~/dev/betting && bash ec2/mc_nba_player_points_live_betting_signal_generator/setup_iam_instance_profile.sh
```

**Step 3: Deploy (and optionally launch the instance)**

```bash
export ODDS_API_KEY="your-odds-api-key"
export REPO_URL="https://github.com/YourOrg/betting.git"   # so user-data can clone into /home/ubuntu/betting

export KEY_NAME="your-key-pair-name"                       # EC2 → Key Pairs (create one if needed; name it e.g. mc-live-betting)
export SECURITY_GROUP_ID="sg-0ec8f42596b900adf"            # default SG in us-east-2 (add SSH 22 from your IP in Console if needed)
export IAM_INSTANCE_PROFILE="mc-nba-live-betting-ec2-role"

cd ~/dev/betting && bash ec2/mc_nba_player_points_live_betting_signal_generator/deploy_mc_nba_live_betting_ec2.sh
```

If your account uses a different security group, run `aws ec2 describe-security-groups --region us-east-2 --query 'SecurityGroups[*].[GroupId,GroupName]' --output table` and pick a GroupId. If that command is denied, use EC2 → Security Groups in the Console and copy a Group ID.

To **launch** the EC2 instance from the script (instead of doing it in the Console), add `--launch`:

```bash
cd ~/dev/betting && bash ec2/mc_nba_player_points_live_betting_signal_generator/deploy_mc_nba_live_betting_ec2.sh --launch
```

Then on the instance: create `/etc/mc-live-betting/env` (plain file named `env`, not `.env`) with one line e.g. `ODDS_API_KEY=your-key`, then `sudo systemctl start mc-live-betting`. Logs: `journalctl -u mc-live-betting -f`.

**Redeploy (update code + service on existing instance):** SSH in, then:
```bash
cd /home/ubuntu/betting
git pull
sudo bash ec2/mc_nba_player_points_live_betting_signal_generator/install_service.sh
```
That copies the systemd unit from the repo, reloads systemd, and restarts the service. No manual sed needed.

**SSH and changing IPs:** The instance security group needs an inbound rule for SSH (22) from your IP. If your public IP changes (e.g. different network), either add a new inbound rule for the new IP (Custom → `your.ip.here/32`) or allow a range (e.g. `170.85.0.0/16`) or **0.0.0.0/0** (any IP; key auth still required). Check your IP with `curl -s ifconfig.me`.

---

## What it does

- Fetches live games from ESPN, pregame props from S3, and live odds from The Odds API.
- For each live game: validates PBP freshness, fetches player props, runs Monte Carlo per player, and detects edges.
- Writes live odds snapshots and signals to S3; optional local copy under `~/Downloads/tmp` on the instance.

## Where things live (S3)

| Data | S3 path |
|------|--------|
| **minute_by_minute.parquet** (input) | `s3://nba-betting-mt/data/01_input/pbp_data/minute_by_minute.parquet` |
| Pregame player props (input) | `s3://the-odds-api-mt/nba/historical_player_props/2025-26/{date}.csv` |
| Live odds (output) | `s3://nba-betting-mt/data/01_input/live_player_odds/player_points/{timestamp}.parquet` |
| Signals (output) | `s3://nba-betting-mt/data/04_output/live_betting_signals/player_points/YYYYMMDD.parquet` |

Ensure **minute_by_minute.parquet** is in S3 before running (e.g. from pipeline 01→03 then):

```bash
aws s3 cp data/minute_by_minute.parquet s3://nba-betting-mt/data/01_input/pbp_data/minute_by_minute.parquet
```

## Prerequisites

- AWS CLI configured (credentials, region e.g. `us-east-2`).
- **ODDS_API_KEY** (The Odds API) for live odds.
- EC2 instance with:
  - IAM instance profile that can read/write `s3://nba-betting-mt/` and read `s3://the-odds-api-mt/` (if needed).
  - Python 3.10+ with deps: `requests`, `pandas`, `boto3`, `duckdb`, `pytz`, `python-dotenv`.

## Deploy and test

From repo root:

```bash
export ODDS_API_KEY="your-key"
cd ~/dev/betting && bash ec2/mc_nba_player_points_live_betting_signal_generator/deploy_mc_nba_live_betting_ec2.sh
```

The script will:

1. Verify AWS CLI and credentials, and that `ODDS_API_KEY` is set.
2. Check that `minute_by_minute.parquet` exists in S3 (warn if missing).
3. Write EC2 user-data to `ec2_user_data.sh` in this folder.
4. Print exact `aws ec2 run-instances` (or Console) instructions.
5. Run one local test iteration (no `--loop`) so you can confirm the code path before deploying.

## One-time setup (key pair, security group, IAM)

You need these before running the deploy script with `--launch`:

| Export | What it is | How to get it |
|--------|------------|---------------|
| **KEY_NAME** | EC2 key pair name (for SSH) | EC2 → Key Pairs → Create, or see "Find your values" below. |
| **SECURITY_GROUP_ID** | Security group id (e.g. allow SSH 22) | Run the CLI below or see "Find your values". |
| **IAM_INSTANCE_PROFILE** | Instance profile name (for S3) | Use `mc-nba-live-betting-ec2-role` if you created that role for EC2. |

**Find your values (if CLI lookup fails)**  
Your IAM user may not have `ec2:DescribeKeyPairs` or `iam:GetInstanceProfile`. Use what works:

- **SECURITY_GROUP_ID:** This often works:  
  `aws ec2 describe-security-groups --region us-east-2 --query 'SecurityGroups[*].[GroupId,GroupName]' --output table`  
  Use one GroupId (e.g. `sg-0ec8f42596b900adf` for the default group). The **default** group often does not allow SSH; in EC2 → Security Groups → default → Edit inbound rules, add **SSH (22)** from your IP if you want to log in.
- **KEY_NAME:** If `aws ec2 describe-key-pairs --region us-east-2` fails, get the name from the Console: **EC2 → Key Pairs**. Create one if needed (Create key pair → name e.g. `mc-live-betting` → save the `.pem`), then `export KEY_NAME=mc-live-betting`.
- **IAM_INSTANCE_PROFILE:** Use `mc-nba-live-betting-ec2-role`. If launch fails with "Invalid IAM Instance Profile", an admin must create the instance profile (see Manual IAM setup).
- **AMI_ID (if launch fails with "Could not resolve AMI"):** Your user may lack `ec2:DescribeImages` and `ssm:GetParameters`. In the Console: **EC2 → Launch instance** → under "Application and OS Images" pick **Ubuntu Server 22.04 LTS** → in the list, copy the **AMI ID** (starts with `ami-`). Then `export AMI_ID=ami-xxxxxxxx` and re-run the deploy script with `--launch`.

**IAM role + instance profile (if your existing role is Lambda-only):**

```bash
bash ec2/mc_nba_player_points_live_betting_signal_generator/setup_iam_instance_profile.sh
```

Creates role and instance profile `mc-nba-live-betting-ec2-role` (trust `ec2.amazonaws.com`, S3 read/write on `nba-betting-mt`, read on `the-odds-api-mt`). Then:

```bash
export IAM_INSTANCE_PROFILE=mc-nba-live-betting-ec2-role
```

**Manual IAM setup (if you get AccessDenied)**  
Your IAM user may not have permission to create roles. Have an admin do the following in **AWS Console → IAM**:

1. **Create role**
   - **Create role** → Trusted entity type: **AWS service** → use case **EC2** → Next.
   - At "Add permissions" you only see managed policies; **don’t attach any** → Next.
   - Role name: `mc-nba-live-betting-ec2-role` → **Create role**.

2. **Add inline policy to the role**
   - Open the role `mc-nba-live-betting-ec2-role` → **Permissions** tab → **Add permissions** → **Create inline policy**.
   - Open the **JSON** tab, replace the default with this, then **Review policy** → name it `S3NbaAndOddsApi` → **Create policy**:

   ```json
   {
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
   }
   ```

3. **Instance profile**  
   If you created the role with use case **EC2** in the IAM wizard, AWS often creates an instance profile with the same name automatically. Try launching your EC2 instance with `IAM_INSTANCE_PROFILE=mc-nba-live-betting-ec2-role` (or pick that role in the launch wizard); if the role appears there, you’re done.

   If you need to create the instance profile by hand (e.g. you created the role without the EC2 use case):
   - **Direct link:** `https://us-east-2.console.aws.amazon.com/iamv2/home?region=us-east-2#/instance_profiles` → Create instance profile, name `mc-nba-live-betting-ec2-role`, add role `mc-nba-live-betting-ec2-role`.
   - **Or CLI** (requires `iam:CreateInstanceProfile` and `iam:AddRoleToInstanceProfile`):  
     `aws iam create-instance-profile --instance-profile-name mc-nba-live-betting-ec2-role` then  
     `aws iam add-role-to-instance-profile --instance-profile-name mc-nba-live-betting-ec2-role --role-name mc-nba-live-betting-ec2-role`

4. Use it: `export IAM_INSTANCE_PROFILE=mc-nba-live-betting-ec2-role`

**Key pair:** EC2 → Key Pairs → Create (e.g. `mc-live-betting`), save the `.pem`. Or CLI: `aws ec2 create-key-pair --key-name mc-live-betting --query 'KeyMaterial' --output text > mc-live-betting.pem && chmod 600 mc-live-betting.pem` — then `KEY_NAME=mc-live-betting`.

**Security group:** EC2 → Security Groups → Create → allow SSH (22) from your IP. Copy the Group ID (`sg-...`). Or CLI with default VPC: `VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)` then `aws ec2 create-security-group --group-name mc-live-betting-sg --description "SSH for MC live betting" --vpc-id $VPC_ID` and `aws ec2 authorize-security-group-ingress --group-name mc-live-betting-sg --protocol tcp --port 22 --cidr 0.0.0.0/0`; get ID with `aws ec2 describe-security-groups --filters Name=group-name,Values=mc-live-betting-sg --query 'SecurityGroups[0].GroupId' --output text`.

## Launching the EC2 instance

**Option A: Launch from CLI (recommended)**

After one-time setup, set exports and run with `--launch`:

```bash
export ODDS_API_KEY="your-key"
export KEY_NAME="mc-live-betting"                          # your key pair name
export SECURITY_GROUP_ID="sg-xxxxxxxx"                      # your security group ID
export IAM_INSTANCE_PROFILE="mc-nba-live-betting-ec2-role" # from setup_iam_instance_profile.sh

cd ~/dev/betting && bash ec2/mc_nba_player_points_live_betting_signal_generator/deploy_mc_nba_live_betting_ec2.sh --launch
```

The script will resolve the latest Ubuntu 22.04 LTS AMI, launch a `t3.micro`, and print the new instance ID and next steps (wait for running, get public IP, SSH, then create env and start the service).

**Option B: Launch from Console**

1. **Create an IAM instance profile** (if you don’t have one) with a role that allows:
   - `s3:GetObject`, `s3:PutObject` (and list if needed) on `nba-betting-mt` (and `the-odds-api-mt` if you use it from the instance).
2. **Launch an instance** (EC2 Console) with:
   - AMI: Ubuntu 22.04 LTS.
   - Instance type: e.g. `t3.micro`.
   - IAM role: the profile above.
   - User data: paste the contents of `ec2_user_data.sh` (generated by the deploy script).
   - Security group: allow SSH (22) from your IP if you want to log in.
3. **Put the repo on the instance** (user-data assumes `/home/ubuntu/betting`):
   - Either set `REPO_URL` when running the deploy script so user-data can `git clone` into `/home/ubuntu/betting`.
   - Or after first boot: SSH in, `git clone <your-repo> /home/ubuntu/betting`, install deps, create `/etc/mc-live-betting/env` with `ODDS_API_KEY=...`, then start the service (see below).

## On the instance: service and env

- **ODDS_API_KEY:** Must be available to the process. Create a plain file named `env` (not `.env`) at `/etc/mc-live-betting/env` with KEY=VALUE lines, e.g.:

  ```bash
  ODDS_API_KEY=your-key-here
  ```

  The systemd unit loads it via `EnvironmentFile=-/etc/mc-live-betting/env`.

- **minute_by_minute:** User-data syncs from S3 to `{REPO_ROOT}/data/minute_by_minute.parquet` at boot. Optionally add a cron or systemd timer to re-sync daily.

- **Run the loop:** From repo root on the instance:

  ```bash
  cd /home/ubuntu/betting
  source /etc/mc-live-betting/env  # or export ODDS_API_KEY
  python -u src/pbp_data/10_live_betting_signal_generator.py --loop --interval 300 --n-sims 500 --min-edge 0.15
  ```

  Or use the systemd unit generated by user-data (if it’s set up to use the same path and env file).

## Monitoring

- **Logs (systemd):**
  - Tail live: `sudo journalctl -u mc-live-betting -f`
  - Last N lines: `sudo journalctl -u mc-live-betting -n 200`
  - Since time: `sudo journalctl -u mc-live-betting --since "1 hour ago"`
  - Since service start: `sudo journalctl -u mc-live-betting --no-pager`
  - If run in a terminal/screen instead of systemd: logs are stdout.
- **S3:** Check that new files appear under live_odds and live_betting_signals for the current date.
- **No live games:** When no NBA games are on, you’ll see “No live games found” and “Waiting 299.8s” (or similar); that’s expected.

## Local test (one iteration, no loop)

From repo root (with `data/minute_by_minute.parquet` present or synced from S3):

```bash
export ODDS_API_KEY="your-key"
python src/pbp_data/10_live_betting_signal_generator.py --min-edge 0.10 --n-sims 1000
```

No live games is normal; you just want to see it complete without import/runtime errors.

## Design doc

Full context and S3 layout: `docs/design-docs/live-mc-ec2-s3.md`.

## Source

The runner is the same script used locally: `src/pbp_data/10_live_betting_signal_generator.py` (see that file and `src/pbp_data/README.md` for the pipeline).

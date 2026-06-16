# Disable NBA/NCAAB Prod Runs for Off-Season (2026-06-15)

**Context:** The 2025-26 NBA season is over. The 2026-27 season is not expected to start until ~October 20, 2026. Several Lambda triggers are still ENABLED in EventBridge and firing daily against stale data. This plan disables them in AWS and updates deploy scripts so they don't re-enable on the next deploy.

---

## Part 1 — Disable EventBridge Rules in AWS

Run these CLI commands. All target `us-east-2`.

```bash
# NBA rebounds pipeline (score + settle)
aws events disable-rule --name nba-rebounds-daily-9am-et          --region us-east-2
aws events disable-rule --name nba-rebounds-daily-settle-905am-et --region us-east-2

# NBA player props ingest
aws events disable-rule --name nba-player-props-ingest-scheduler  --region us-east-2

# NBA player scoring props daily workflow
aws events disable-rule --name nba-player-scoring-props           --region us-east-2

# NBA strategy stats refresher
aws events disable-rule --name nba-strategy-stats-refresher-scheduler --region us-east-2

# Live odds trackers (no NBA or NCAAB games to track)
aws events disable-rule --name track-live-odds-nba-every-minute   --region us-east-2
aws events disable-rule --name track-live-odds-ncaab-every-minute --region us-east-2

# NCAAB fade-revenge (season over)
aws events disable-rule --name ncaab-fade-revenge-daily-9am-et    --region us-east-2
```

Verify all 8 rules are now DISABLED:
```bash
aws events list-rules --region us-east-2 \
  --query 'Rules[*].{Name:Name,State:State}' --output table
```

**Leave these ENABLED** — they are multi-sport or sport-agnostic:
- `line-movement-hourly-tracker`
- `line-steam-alerts-schedule`
- `multi-sport-game-results-fetcher-scheduler`

---

## Part 2 — Update Deploy Scripts (so next deploy doesn't re-enable)

These files hardcode `--state ENABLED` for season-specific rules. Change each to `--state DISABLED`.

### 2a. `lambda/nba_rebounds_daily/deploy_nba_rebounds_daily.sh`

Line 214 — the settle rule is set to ENABLED on deploy. Change to DISABLED:

```bash
# BEFORE (line 214):
  --state ENABLED \

# AFTER:
  --state DISABLED \
```

The score rule (line 206) is already `--state DISABLED` — leave it as-is.

Also update the summary echos near the bottom:
```bash
# BEFORE:
echo -e "${GREEN}✅ EventBridge score rule: $CRON_SCORE_9AM_ET (9am ET target)${NC}"
echo -e "${GREEN}✅ EventBridge settle rule: $CRON_SETTLE_905AM_ET (9:05am ET target)${NC}"

# AFTER:
echo -e "${YELLOW}⚠️  EventBridge score rule: $CRON_SCORE_9AM_ET — deployed DISABLED (off-season)${NC}"
echo -e "${YELLOW}⚠️  EventBridge settle rule: $CRON_SETTLE_905AM_ET — deployed DISABLED (off-season)${NC}"
```

### 2b. `lambda/ncaab_fade_revenge_daily/deploy_ncaab_fade_revenge_daily.sh`

Line 196 — the EventBridge rule is set to ENABLED on deploy. Change to DISABLED:

```bash
# BEFORE (line 196):
    --state ENABLED \

# AFTER:
    --state DISABLED \
```

### 2c. `lambda/track_live_odds/deploy_nba_and_ncaab_lambdas.sh`

Line 310 (NBA live-odds rule) and line 324 (NCAAB live-odds rule) are both ENABLED. Change both:

```bash
# Line 310 — NBA rule:
# BEFORE:
    --state ENABLED \
# AFTER:
    --state DISABLED \

# Line 324 — NCAAB rule:
# BEFORE:
    --state ENABLED \
# AFTER:
    --state DISABLED \
```

### 2d. Player props ingest + scoring props + strategy stats refresher

No deploy scripts were found in the repo for these three Lambdas. Confirm with:

```bash
find ~/dev/betting -name "deploy*.sh" | xargs grep -l \
  "nba-player-props-ingest\|nba-player-scoring\|nba-strategy-stats" 2>/dev/null
```

If scripts exist, apply the same `--state ENABLED` → `--state DISABLED` change. If they're console-managed, the AWS CLI commands in Part 1 are sufficient.

---

## Part 3 — Add NBA_PAUSE_UNTIL Guard to Lambda Code (optional but recommended)

The NCAAB fade-revenge Lambda has a clean no-op pattern at line 892 of
`lambda/ncaab_fade_revenge_daily/lambda_function.py`:

```python
pause_until = os.environ.get('NCAAB_PAUSE_UNTIL', '').strip()
if pause_until:
    pause_until_date = datetime.strptime(pause_until, "%Y-%m-%d").date()
    if today_date < pause_until_date:
        print(f"NCAAB Fade Revenge paused until {pause_until}; skipping run for {today_et}")
        return {"status": "paused", "today_et": today_et, "pause_until": pause_until}
```

Mirror this in the two highest-risk NBA Lambdas so an accidentally re-enabled rule no-ops instead of running against off-season data.

### 3a. `lambda/nba_rebounds_daily/lambda_function.py`

Insert after line 532 (after `today_et` is set, before config is loaded):

```python
nba_pause_until = os.environ.get("NBA_PAUSE_UNTIL", "").strip()
if nba_pause_until:
    pause_date = datetime.strptime(nba_pause_until, "%Y-%m-%d").date()
    if datetime.now(ET).date() < pause_date:
        print(f"NBA Rebounds paused until {nba_pause_until}; skipping run for {today_et}")
        return {"status": "paused", "today_et": today_et, "pause_until": nba_pause_until}
```

Then wire the env var in `deploy_nba_rebounds_daily.sh` by appending to the `ENV_VARS=` line (around line 100):

```bash
# Append to the ENV_VARS string:
,NBA_PAUSE_UNTIL=2026-10-20
```

### 3b. `scripts/lambda_function_nba_player_scoring_props.py`

Insert at the top of `lambda_handler` (line 831), after the current date/time is established:

```python
nba_pause_until = os.environ.get("NBA_PAUSE_UNTIL", "").strip()
if nba_pause_until:
    from datetime import datetime as _dt
    pause_date = _dt.strptime(nba_pause_until, "%Y-%m-%d").date()
    if _dt.now().date() < pause_date:
        print(f"NBA scoring props paused until {nba_pause_until}; skipping.")
        return {"status": "paused", "pause_until": nba_pause_until}
```

---

## Part 4 — Re-Enable in October 2026

When the official NBA schedule drops (expected mid-August 2026), update `NBA_PAUSE_UNTIL` env vars to the confirmed season start date, then re-enable rules the night before opening night:

```bash
# Re-enable NBA rules ~Oct 19, 2026
aws events enable-rule --name nba-rebounds-daily-9am-et               --region us-east-2
aws events enable-rule --name nba-rebounds-daily-settle-905am-et      --region us-east-2
aws events enable-rule --name nba-player-props-ingest-scheduler       --region us-east-2
aws events enable-rule --name nba-player-scoring-props                --region us-east-2
aws events enable-rule --name nba-strategy-stats-refresher-scheduler  --region us-east-2
aws events enable-rule --name track-live-odds-nba-every-minute        --region us-east-2

# Re-enable NCAAB rules when NCAAB season tips (~early November 2026)
aws events enable-rule --name track-live-odds-ncaab-every-minute      --region us-east-2
aws events enable-rule --name ncaab-fade-revenge-daily-9am-et         --region us-east-2
```

Also clear or remove `NBA_PAUSE_UNTIL` / `NCAAB_PAUSE_UNTIL` from Lambda env vars at that time.

---

## Checklist

- [ ] Part 1: All 8 rules disabled in AWS (verified via `list-rules`)
- [ ] Part 2a: `lambda/nba_rebounds_daily/deploy_nba_rebounds_daily.sh` line 214 → `--state DISABLED`
- [ ] Part 2b: `lambda/ncaab_fade_revenge_daily/deploy_ncaab_fade_revenge_daily.sh` line 196 → `--state DISABLED`
- [ ] Part 2c: `lambda/track_live_odds/deploy_nba_and_ncaab_lambdas.sh` lines 310 + 324 → `--state DISABLED`
- [ ] Part 2d: Confirm no untracked deploy scripts for player props / scoring props / strategy stats
- [ ] Part 3 (optional): Add `NBA_PAUSE_UNTIL=2026-10-20` guard to rebounds + scoring props Lambdas
- [ ] Commit all deploy script changes on `main`

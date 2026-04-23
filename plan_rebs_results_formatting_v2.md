# Settlement results formatting — v2 (deploy & ops)

Supersedes handoff notes for anything **after** code landed. Phase A/B design and formatter details stay in `plan_rebs_results_formatting.md`.

---

## What shipped (summary)

- **Plain text:** SNS body unchanged (monospace plays table from `format_settlement_email_plays_table`).
- **HTML inbox:** Optional **SES** multipart when Lambda env has `SETTLEMENT_SES_SOURCE` and `SETTLEMENT_SES_TO` (see `lambda/nba_rebounds_daily/lambda_function.py`).
- **S3:** `email_plays_yesterday.csv` plus **`email_plays_yesterday.html`** next to it (settlement script upload).

### Why the “table” looks wrong in Gmail (SNS)

Emails from **`no-reply@sns.amazonaws.com`** are **plain text only**. Gmail uses a proportional font, so space-padded columns will not line up, and **no HTML `<table>` is possible** on that path. For a real table in the inbox you need the **SES** email (same subject, **different sender** than `no-reply@sns.amazonaws.com` — your verified `SETTLEMENT_SES_SOURCE`). Check **Spam** and search **`from:`** + that address. After redeploying the latest Lambda, the **SNS** copy ends with **`[SES] …`**: either the **AWS error** (send failed) or **MessageId** + From/To (send accepted at SES — if you see MessageId but no second email, treat it as Gmail filtering or delivery delay).

Alternatively open **`email_plays_yesterday.html`** in the **S3 console** using the **HTTPS “Console:”** line — raw **`s3://` links are not valid in mobile Safari**.

---

## Deploy — copy/paste flow

**Before** the block below: in your terminal, set a **real** Odds API key (never paste placeholder text as the value — that string becomes the Lambda env verbatim and breaks live props).

```bash
export ODDS_API_KEY='…paste the real key from the-odds-api…'
export SNS_TOPIC_ARN="arn:aws:sns:us-east-2:232692785472:betting-arb-alerts"
export SETTLEMENT_SES_SOURCE="tqstrats@gmail.com"
export SETTLEMENT_SES_TO="mylescgthomas@gmail.com"

cd ~/dev/betting
bash lambda/nba_rebounds_daily/deploy_nba_rebounds_daily.sh
```

If you already deployed with a bad `ODDS_API_KEY`, fix it by exporting the real key and running the deploy script again (or edit **Lambda → Configuration → Environment variables** in the console).

The deploy script appends `SETTLEMENT_SES_SOURCE` / `SETTLEMENT_SES_TO` to the Lambda **only when both** are non-empty at deploy time. Redeploy after changing them.

---

## SES verification — no email in Gmail?

Verification mail is sent by **Amazon SES** (sender looks like `no-reply-aws@amazon.com` or similar), **not** from your own address until the identity is verified.

1. **Region** — Your Lambda is **`us-east-2`**. In the AWS console, set the region dropdown to **`US East (Ohio)`** before opening SES. Identities in `us-east-1` do **not** apply here.
2. **Console** — **Amazon SES** → **Verified identities** → **Create identity** → **Email address** → enter `tqstrats@gmail.com` → create. Then use **Resend** if there is no mail within a few minutes.
3. **Gmail** — Check **Spam**, **Promotions**, and **All Mail**. Search for `amazonaws.com` or `Amazon SES`.
4. **CLI (same region)** — Trigger verification again and check status:

```bash
export AWS_REGION=us-east-2

aws ses verify-email-identity --email-address tqstrats@gmail.com --region us-east-2

aws ses get-identity-verification-attributes \
  --identities tqstrats@gmail.com mylescgthomas@gmail.com \
  --region us-east-2
```

In the JSON, `"VerificationStatus": "Success"` means verified. **`mylescgthomas@gmail.com`** must also show Success while the account is in **SES sandbox** (otherwise Lambda cannot send *to* that address).

5. **Sandbox** — In SES → **Account dashboard**, if status is “Sandbox”, you can only send **to** verified addresses until you request production access.

6. **After both addresses verify** — Redeploy Lambda with `SETTLEMENT_SES_SOURCE` / `SETTLEMENT_SES_TO`, ensure the Lambda role has **`ses:SendEmail`**, then invoke settlement and check **inbox + spam** for mail **From** `tqstrats@gmail.com`.

### CLI `AccessDenied` on `ses:GetIdentityVerificationAttributes`

Your **IAM user** (e.g. the one used by `aws sts get-caller-identity`) needs SES API permissions to run those CLI commands. The **Lambda role** is separate: it needs `ses:SendEmail` to actually send.

**Fix:** In **IAM → Users → (your user) → Add permissions**, attach an inline policy allowing at least `ses:ListIdentities`, `ses:GetIdentityVerificationAttributes`, and `ses:VerifyEmailIdentity` on `Resource: "*"`, **or** skip CLI and use the **SES web console** in `us-east-2` (works if your console login has SES access).

---

## Todo list (operator)

- [ ] **Exports** — `ODDS_API_KEY` and `SNS_TOPIC_ARN` set for deploy (same as before).
- [ ] **SES (optional)** — If you want the HTML email:
  - [ ] Verify domain or address in **SES** (same region as Lambda, e.g. `us-east-2`).
  - [ ] **Lambda role → SES:** With both `SETTLEMENT_SES_*` exports set, `deploy_nba_rebounds_daily.sh` runs **Step 1b** and tries to attach inline policy `nba-rebounds-daily-ses-send-email` (`ses:SendEmail`, `ses:SendRawEmail`) on `betting-dashboard-daily-update-role-ille2llh`. If you see a yellow **Could not attach SES policy**, your deploy user lacks `iam:PutRolePolicy` — add that policy manually in IAM → Roles → that role.
  - [ ] Set `SETTLEMENT_SES_SOURCE` and `SETTLEMENT_SES_TO`, then **re-run deploy** so env vars are applied.
  - [ ] If account is in SES **sandbox**, verify every recipient address.
- [ ] **Run deploy** — `bash lambda/nba_rebounds_daily/deploy_nba_rebounds_daily.sh` completes without errors.
- [ ] **Smoke** — After deploy: optional manual Lambda invoke `{"mode":"settlement"}` in dev, or wait for EventBridge; confirm SNS text still arrives; if SES enabled, confirm HTML in inbox.
- [ ] **S3** — Confirm `email_plays_yesterday.html` appears beside `email_plays_yesterday.csv` after a settle run with plays.

---

## Plan (remaining / maintenance)

| Item | Notes |
|------|--------|
| Inbox QA | Paste Phase A preview or rely on prod SES mail; spot-check **Outlook** if possible. |
| Duplicate mail | SNS + SES can both fire to different channels; adjust env/subscriptions if one person gets two copies. |
| SES-only | Clear `SNS_TOPIC_ARN` in Lambda console **or** leave unset at deploy **and** set only SES vars — code publishes SES when those two are set even without SNS. |
| Limits | SNS message remains capped at 256 KB; large play lists still truncated at formatter `max_rows` (600). |

---

## Terminal cookbook (us-east-2)

These commands match your setup: **From / SES source** `tqstrats@gmail.com`, **To** `mylescgthomas@gmail.com`, SNS topic `betting-arb-alerts` in account `232692785472`.

Set region once (matches Lambda / deploy script):

```bash
export AWS_REGION=us-east-2
export AWS_DEFAULT_REGION=us-east-2
```

**1. Who am I using for these CLI calls?**

```bash
aws sts get-caller-identity
```

**2. SES — are the sender / recipient verified?**  
(Requires SES API permission on your IAM user; if `AccessDenied`, use the SES web console instead.)

```bash
aws ses get-identity-verification-attributes \
  --identities "tqstrats@gmail.com" "mylescgthomas@gmail.com" \
  --region us-east-2
```

Re-send verification email to the **source** (and again for the recipient if sandbox still requires it):

```bash
aws ses verify-email-identity --email-address "tqstrats@gmail.com" --region us-east-2
aws ses verify-email-identity --email-address "mylescgthomas@gmail.com" --region us-east-2
```

**3. Lambda — SES env vars present?**

```bash
aws lambda get-function-configuration \
  --function-name nba-rebounds-daily \
  --region us-east-2 \
  --query 'Environment.Variables.[SETTLEMENT_SES_SOURCE,SETTLEMENT_SES_TO,SNS_TOPIC_ARN]' \
  --output text
```

**4. Lambda role — inline SES policy attached?**  
Role name matches `deploy_nba_rebounds_daily.sh` (`IAM_ROLE_NAME`).

```bash
ROLE_NAME="betting-dashboard-daily-update-role-ille2llh"
aws iam get-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name nba-rebounds-daily-ses-send-email \
  --output json
```

If that errors with `NoSuchEntity`, the deploy script did not attach the policy (missing `iam:PutRolePolicy` on your deploy user, or SES exports were unset at deploy).

**5. Redeploy Lambda** — same as the deploy section above: **`export ODDS_API_KEY='…real key…'`** then the other exports, then `bash lambda/nba_rebounds_daily/deploy_nba_rebounds_daily.sh`. Do not export instructional English as the key value.

**6. Run settlement once**

```bash
aws lambda invoke \
  --function-name nba-rebounds-daily \
  --region us-east-2 \
  --cli-binary-format raw-in-base64-out \
  --payload '{"mode":"settlement"}' \
  /tmp/nba-rebounds-settle-out.json

cat /tmp/nba-rebounds-settle-out.json
```

**7. CloudWatch — tail recent Lambda logs (recommended)**

Avoids fragile stream names (`[$LATEST]` etc.) and pagination quirks:

```bash
aws logs tail /aws/lambda/nba-rebounds-daily \
  --region us-east-2 \
  --since 45m \
  --format short
```

Search the output for `published_settlement_to_ses` or `settlement_ses_send_failed`.

If `aws logs tail` is not available (very old AWS CLI v2), upgrade the CLI, or use the console: **CloudWatch → Log groups → `/aws/lambda/nba-rebounds-daily` → latest stream**.

**Why `get-log-events` + `describe-log-streams` sometimes failed:** the stream id contains **`[$LATEST]`**. In **zsh**, unquoted or partially quoted values can interact badly with globbing; `aws logs tail` sidesteps that. Empty `describe-log-streams` pages can also yield a bogus stream name.

**8. After the next SNS email**  
Scroll to the bottom: you should see a **`[SES]`** block (either **MessageId** = send accepted, or **AWS error** = still misconfigured).

---

## Security reminder

If an API key or secret was ever pasted into chat or a committed file, **rotate** it in the provider console and use env exports / AWS Secrets Manager going forward.

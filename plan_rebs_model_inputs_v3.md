# Plan v3: terminal commands (rollout + checks after v2 audit lists)

Set paths once (adjust to your machine / S3 layout):

```bash
cd /path/to/betting   # repo root (contains ./src and ./.gitignore)

export CACHE="${HOME}/Downloads/tmp"                    # v2 caches + rebounds_input_universe.parquet live here
export FEAT="${CACHE}/rebounds_model_features_v2.parquet"
export V3PROPS="${CACHE}/v3_rebounds_props_raw.parquet"
export INPUT_UNI="${CACHE}/rebounds_input_universe.parquet"
export SLATE="2025-03-15"                                # example slate date YYYY-MM-DD
export MODELS="${CACHE}/rebounds_prod_models/run_001"  # dir with ols_model.pkl, xgb_model.json, manifest.json
export FEAT_SLICE="${CACHE}/rebounds_features_slice_${SLATE}.parquet"
export PROPS_SCORE="${CACHE}/rebounds_props_scoring_input_${SLATE}.parquet"
export SCORED="${CACHE}/rebounds_scored_${SLATE}.parquet"
```

### Lambda + S3 (what prod actually uses)

**Config file:** `config/nba_rebounds_prod.lambda.yaml` — bucket `nba-betting-mt`, input + feature URIs, `/tmp/rebounds_prod/...` working paths, SNS topic.

**Deploy already sets** `CONFIG_PATH=config/nba_rebounds_prod.lambda.yaml` on the function (`lambda/nba_rebounds_daily/deploy_nba_rebounds_daily.sh`).

Verify upstream objects exist (needs AWS CLI + credentials):

```bash
export AWS_REGION="${AWS_REGION:-us-east-2}"

aws s3 ls "s3://nba-betting-mt/rebounds/input/"
aws s3 ls "s3://nba-betting-mt/rebounds/features/"
aws s3 ls "s3://nba-betting-mt/rebounds/daily_runs/" | tail -n 20
```

Run **the same entrypoint as Lambda** on your laptop (same YAML: downloads/uploads S3, builds under `/tmp/rebounds_prod/...`, needs **`ODDS_API_KEY`** for live props when `build_props_scoring_input_from_live: true`):

```bash
cd /path/to/betting

export CONFIG_PATH="config/nba_rebounds_prod.lambda.yaml"
export ODDS_API_KEY="YOUR_ODDS_API_KEY"
export SNS_TOPIC_ARN="arn:aws:sns:us-east-2:232692785472:betting-arb-alerts"

python scripts/run_rebounds_daily_pipeline.py \
  --config "${CONFIG_PATH}" \
  --slate-date "YYYY-MM-DD"
```

Build + deploy the container Lambda (repo root; Docker + AWS; role name is in the script):

```bash
export ODDS_API_KEY="YOUR_ODDS_API_KEY"
export SNS_TOPIC_ARN="arn:aws:sns:us-east-2:232692785472:betting-arb-alerts"

bash lambda/nba_rebounds_daily/deploy_nba_rebounds_daily.sh
```

Invoke the deployed function (optional `mode`: `pipeline` | `settlement` | `both`):

```bash
export AWS_REGION="${AWS_REGION:-us-east-2}"
OUT="${HOME}/nba_rebounds_lambda_response.json"

aws lambda invoke \
  --function-name nba-rebounds-daily \
  --region "${AWS_REGION}" \
  --cli-binary-format raw-in-base64-out \
  --payload '{"mode":"both"}' \
  "${OUT}"

cat "${OUT}"
```

CloudWatch Logs: log group **`/aws/lambda/nba-rebounds-daily`** in `us-east-2`.

**One script (ordered):** from repo root, `./scripts/run_rebounds_prod_stack.sh` runs S3 listings → pytest audit lists → `run_rebounds_daily_pipeline.py` with `config/nba_rebounds_prod.lambda.yaml`; add `--deploy --invoke` for container deploy + Lambda test. `./scripts/run_rebounds_prod_stack.sh --help` for flags.

### Quick CI (no AWS)

```bash
cd /path/to/betting
python -m pytest tests/unit/test_rebounds_audit_lists.py -v
```

After a **successful** Lambda-style pipeline run, confirm audit lists on the new scored object under `s3://nba-betting-mt/rebounds/daily_runs/<date>/<run_id>/` (column names starting with `input_`).

---

## 1) Refresh input universe (spread + shots; must exist before full feature universe build)

```bash
python src/nba_rebounds_modeling/00_research/scripts/build_rebounds_input_universe.py \
  --season "*" \
  --output "${INPUT_UNI}" \
  --s3-uri "" \
  --mode append
```

Use a real `--s3-uri` in prod if you publish there (see script docstring).

---

## 2) Rebuild full feature universe (includes `B_MIN_MAX_AUDIT_LIST_COLS`)

**First time after v2:** bust stale logs cache so `team_normalized` exists:

```bash
python src/nba_rebounds_modeling/00_research/scripts/build_rebounds_full_universe.py \
  --season "*" \
  --cache-dir "${CACHE}" \
  --use-cache true \
  --force-refresh-cache true \
  --output "${FEAT}" \
  --output-v3 "${V3PROPS}"
```

Later incremental runs can use `--force-refresh-cache false` if caches are good.

---

## 3) Prod-style wrapper (input universe S3 + feature upload)

```bash
python scripts/build_rebounds_feature_universe.py \
  --season "*" \
  --cache-dir "${CACHE}" \
  --output "${FEAT}" \
  --output-props "${V3PROPS}" \
  --input-universe-s3-uri "s3://YOUR_BUCKET/rebounds/input/rebounds_input_universe.parquet" \
  --feature-universe-s3-uri "s3://YOUR_BUCKET/rebounds/features/rebounds_feature_universe.parquet" \
  --props-history-s3-uri "s3://YOUR_BUCKET/rebounds/props/rebounds_props_history.parquet"
```

---

## 4) Slice features to the slate date

```bash
python src/nba_rebounds_modeling/00_research/scripts/prod_slice_rebounds_features.py \
  --feat "${FEAT}" \
  --as-of-date "${SLATE}" \
  --output "${FEAT_SLICE}"
```

If that returns **0 rows** (pregame / no completed games yet), use the pregame backfill instead:

```bash
python scripts/build_rebounds_pregame_feature_slice.py \
  --feat "${FEAT}" \
  --props "${PROPS_SCORE}" \
  --slate-date "${SLATE}" \
  --output "${FEAT_SLICE}"
```

---

## 5) Scoring-input props (live slate; paths from your fetch job)

```bash
python scripts/build_rebounds_scoring_input.py \
  --live-csv "${HOME}/path/to/live_rebounds_props.csv" \
  --output "${PROPS_SCORE}" \
  --date "${SLATE}"
```

---

## 6) Score slate (audit columns flow through if present on the feat slice)

```bash
python src/nba_rebounds_modeling/00_research/scripts/prod_score_rebounds_slate.py \
  --models-dir "${MODELS}" \
  --feat-slice "${FEAT_SLICE}" \
  --props "${PROPS_SCORE}" \
  --slate-date "${SLATE}" \
  --output "${SCORED}" \
  --s3-uri ""
```

Add `--s3-uri s3://bucket/key.parquet` when uploading the scored artifact.

---

## 7) Notify (stdout if no SNS topic)

```bash
python src/nba_rebounds_modeling/00_research/scripts/prod_notify_rebounds_sns.py \
  --scored "${SCORED}" \
  --which both
```

With SNS:

```bash
export SNS_TOPIC_ARN="arn:aws:sns:REGION:ACCOUNT:YOUR_TOPIC"

python src/nba_rebounds_modeling/00_research/scripts/prod_notify_rebounds_sns.py \
  --scored "${SCORED}" \
  --which both \
  --topic-arn "${SNS_TOPIC_ARN}"
```

---

## 8) Train models (unchanged feature matrix `B_MIN_MAX_FEATS`; optional after universe refresh)

```bash
python src/nba_rebounds_modeling/00_research/scripts/prod_train_rebounds_models.py \
  --config config/nba_rebounds_prod.yaml \
  --feat "${FEAT}" \
  --output-dir "${MODELS}"
```

---

## 9) Quick checks (columns + audit tests)

```bash
python -c "import pandas as pd; df=pd.read_parquet('${SCORED}', columns=None); print([c for c in df.columns if c.startswith('input_')])"
```

```bash
python -m pytest tests/unit/test_rebounds_audit_lists.py -v
```

Optional: DuckDB peek at list columns (install duckdb if needed):

```bash
duckdb -c "DESCRIBE SELECT * FROM read_parquet('${FEAT}') LIMIT 1;"
```

---

## 10) Optional one-off Python verify on a real scored file

From repo root, with `SCORED` exported (see top of doc):

```bash
python -c "
import os
from pathlib import Path
import pandas as pd
from src.nba_rebounds_modeling.rebounds_audit_list_verify import verify_audit_lists_dataframe
from src.nba_rebounds_modeling.rebounds_feature_spec import B_MIN_MAX_AUDIT_LIST_COLS, B_MIN_MAX_FEATS

path = Path(os.environ['SCORED']).expanduser()
df = pd.read_parquet(path)
need = list(B_MIN_MAX_FEATS) + list(B_MIN_MAX_AUDIT_LIST_COLS)
missing = [c for c in need if c not in df.columns]
print('missing:', missing)
if not missing:
    verify_audit_lists_dataframe(df, max_rows=50)
    print('verify_audit_lists_dataframe ok (50 rows)')
"
```

---

## Daily pipeline (single entrypoint)

```bash
python scripts/run_rebounds_daily_pipeline.py --help
```

Prod / Lambda use **`--config config/nba_rebounds_prod.lambda.yaml`** (see section “Lambda + S3” above).

---

## Local full feature build (DuckDB, SSO, 403s)

`scripts/build_rebounds_feature_universe.py` runs `build_rebounds_input_universe.py` then `build_rebounds_full_universe.py`. Both use DuckDB `read_csv_auto` on S3, including **`s3://the-odds-api-mt/nba/historical_game_lines/...`** and historical player props.

- **Credentials in DuckDB:** `src/nba_rebounds_modeling/duckdb_s3_creds.py` applies **boto3**’s default chain (SSO, assume-role, `AWS_ACCESS_KEY_ID` / session token). Run `aws sso login` (or refresh your profile) so temporary credentials are available; the old pattern of only `aws configure get` long-lived keys **misses** SSO session tokens and can cause confusing S3 errors.
- **HTTP 403 on a specific `nba_game_lines_*.csv`:** your IAM identity must be allowed **`s3:GetObject`** (and listing as needed) on that bucket/prefix. Prod execution roles and your laptop user are not the same; add read access for local dev or assume a role that already has it.

---

## Summary (Lambda / S3 first)

```bash
# A) Prove AWS + S3 layout
aws s3 ls s3://nba-betting-mt/rebounds/input/
aws s3 ls s3://nba-betting-mt/rebounds/features/

# B) Same run as Lambda, locally (ODDS_API_KEY + creds)
python scripts/run_rebounds_daily_pipeline.py --config config/nba_rebounds_prod.lambda.yaml --slate-date YYYY-MM-DD

# C) Ship container + env
bash lambda/nba_rebounds_daily/deploy_nba_rebounds_daily.sh

# D) Invoke + read CloudWatch if needed
aws lambda invoke --function-name nba-rebounds-daily --region us-east-2 --cli-binary-format raw-in-base64-out --payload '{"mode":"both"}' ~/nba_rebounds_lambda_response.json

# E) Optional: full rebuild path when changing v2 universe (S3 + cache) — sections 1–3 below
```

# DuckDB: Running queries against S3

**Purpose:** Agents and developers can query CSV/Parquet data in S3 (e.g. `ncaab-betting-mt`, `nba-betting-mt`) using DuckDB with the `httpfs` extension. Use this doc whenever you need to inspect schema, sample rows, or run ad-hoc SQL on S3 data.

**Bucket region:** Our S3 buckets use `us-east-2`. Set it before querying.

---

## One-time setup (per session or per `-c` block)

In an interactive DuckDB session, run once:

```sql
INSTALL httpfs;
LOAD httpfs;
SET s3_region='us-east-2';
```

For a **single** `duckdb -c "..."` invocation, the shell has no persistent session, so include credentials so S3 access works. Use `aws configure` (or env vars) and substitute into the call:

```bash
duckdb -c "
INSTALL httpfs;
LOAD httpfs;
SET s3_region='us-east-2';
SET s3_access_key_id='$(aws configure get aws_access_key_id)';
SET s3_secret_access_key='$(aws configure get aws_secret_access_key)';

-- your queries here
SELECT * FROM 's3://ncaab-betting-mt/data/01_input/historical_game_results/2026-02-14.csv' LIMIT 2;
"
```

If your env is already loaded (e.g. in a terminal where you've run `source .venv/bin/activate` or similar), the same `SET s3_*` lines with the `$(aws configure get ...)` substitution work when you paste the block into the shell.

---

## Example queries

**Describe schema of a CSV:**

```sql
DESCRIBE SELECT * FROM 's3://ncaab-betting-mt/data/01_input/historical_game_results/2026-02-14.csv';
```

**Sample rows:**

```sql
SELECT * FROM 's3://ncaab-betting-mt/data/01_input/historical_game_results/2026-02-14.csv' LIMIT 3;
```

**List / glob:** Use `read_csv_auto` with a glob pattern if the backend supports it, or query known paths. For date-partitioned data we typically know the path pattern (e.g. `.../2026-02-14.csv`).

**Which files have which columns (per-file):** When reading many CSVs with `union_by_name=true`, DuckDB merges schemas so the result has every column that appears in any file. To see **which files** actually contain a given column (e.g. `TEAM_ABBREVIATION`), group by `filename` and count non-nulls for that column. Files that never had the column will have all NULLs for it.

```sql
-- Per-file: row count and whether TEAM_ABBREVIATION is present (any non-null)
SELECT
  filename,
  COUNT(*) AS rows,
  COUNT(*) FILTER (WHERE "TEAM_ABBREVIATION" IS NOT NULL AND TRIM("TEAM_ABBREVIATION") <> '') AS has_team_abbr
FROM read_csv_auto('s3://nba-api-mt/player_game_logs/2025-26/*.csv', union_by_name=true, filename=true)
GROUP BY filename
ORDER BY filename;
```

To get **exact column set per file**, run `DESCRIBE` on each file (e.g. in a loop over `aws s3 ls` output).

---

## Reference

- **Extension:** [DuckDB httpfs](https://duckdb.org/docs/extensions/httpfs/s3api.html) — S3 configuration and credentials.
- **Region:** Always set `s3_region='us-east-2'` for this repo’s buckets.

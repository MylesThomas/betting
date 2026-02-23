# Live Monte Carlo Signal Generator: EC2 + S3

## Context

The live betting signal generator (`src/pbp_data/10_live_betting_signal_generator.py`) runs as a long-running process with `--loop --interval 60`. We run it in production on a **single small EC2 instance**. Runtime data that is today repo-local or under `~/Downloads/tmp` is either synced from S3 to the instance or written only to S3. Email for signals is desired later but **off for now**.

## Decision

- **Runtime:** One EC2 instance (e.g. t4g.micro or t3.micro) runs the script with `--loop --interval 60` under systemd (or equivalent) so it restarts on reboot.
- **Data:** `minute_by_minute.parquet` lives on S3; the instance syncs it to the repo `data/` directory at boot (and optionally on a schedule). All other inputs/outputs already use S3 or are local to the instance.
- **Email:** Design for a single “send signals email” step; implementation is no-op or feature-flag **off** until we know volume.

## S3 Paths (agreed)

| Data | S3 location | Use on EC2 |
|------|-------------|------------|
| **minute_by_minute.parquet** | `s3://nba-betting-mt/data/01_input/pbp_data/minute_by_minute.parquet` | Sync to repo `data/minute_by_minute.parquet` at boot (and optionally via cron) so `get_data_paths()` works unchanged. |
| **Pregame player props** | `s3://the-odds-api-mt/nba/historical_player_props/2025-26/{date}.csv` | Read directly by script (DuckDB httpfs); no sync. |
| **Live odds snapshots** | `s3://nba-betting-mt/data/01_input/live_player_odds/player_points/{timestamp}.parquet` | Script writes here; no change. |
| **Signals (output)** | `s3://nba-betting-mt/data/04_output/live_betting_signals/player_points/YYYYMMDD.parquet` | Script writes here (and may still write local copy under `~/Downloads/tmp` on the box for debugging). |

Publishing **minute_by_minute.parquet** to S3 is the responsibility of the pipeline that produces it (e.g. `src/pbp_data/01_get_game_ids.py` → … → `03_process_data.py`). After generating `data/minute_by_minute.parquet` locally, upload with:

```bash
aws s3 cp data/minute_by_minute.parquet s3://nba-betting-mt/data/01_input/pbp_data/minute_by_minute.parquet
```

## EC2 Deployment (summary)

- **Deploy artifacts:** `ec2/mc_nba_player_points_live_betting_signal_generator/` (README + deploy script).
- **Deploy script** (`deploy_mc_nba_live_betting_ec2.sh`): Verifies prerequisites (AWS CLI, credentials, ODDS_API_KEY), checks that `minute_by_minute.parquet` exists in S3, writes user-data for the instance (install Python, sync repo, sync minute_by_minute from S3, install deps, run loop via systemd), and prints launch steps (or launches the instance). Includes a local “test” run (one iteration without `--loop`) to verify the code path.
- **On the instance:** User-data (or manual steps from README) installs dependencies, syncs `minute_by_minute.parquet` from S3 into repo `data/`, sets `ODDS_API_KEY`, and runs `python src/pbp_data/10_live_betting_signal_generator.py --loop --interval 60` under systemd so it survives reboot.

## What runs on EC2

- Same script as local: `src/pbp_data/10_live_betting_signal_generator.py`.
- Optional: cron or systemd timer to re-sync `minute_by_minute.parquet` from S3 (e.g. daily) so the instance has fresh data without redeploying.

## Consequences

- **Positive:** No Lambda time limit or cold start; minimal code change; easy to SSH and debug; same script locally and on EC2.
- **Negative:** Always-on instance cost (small); need to maintain one instance (updates, or accept risk).
- **Email:** Stub only; turn on when ready.

## Supersedes

- **live-mc-lambda-s3.md** – Previous Lambda-based design; kept for reference. We use EC2 instead.

## Related

- `ec2/mc_nba_player_points_live_betting_signal_generator/README.md` – full deployment and run instructions.
- `ec2/mc_nba_player_points_live_betting_signal_generator/deploy_mc_nba_live_betting_ec2.sh` – deploy and test script.
- `docs/references/duckdb-s3-queries.md` – S3 + DuckDB (pregame props).
- `src/pbp_data/10_live_betting_signal_generator.py` – entrypoint and gates.
- `src/pbp_data/monte_carlo_utils.py` – `get_data_paths()`, `load_player_profile()`.

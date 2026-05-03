"""
Run the rebounds daily production pipeline end-to-end.

Context:
- This is the production entrypoint for daily rebounds automation.
- All intermediates and outputs go to S3; only /tmp is used for ephemeral computation.
- Fail-fast by design: every required config key and step must succeed.
- --config accepts a local path or an s3:// URI.

Usage:
    python scripts/run_rebounds_daily_pipeline.py \
        --config s3://nba-betting-mt/rebounds/config/nba_rebounds_prod.yaml \
        --slate-date 2026-04-02
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.io_utils import read_parquet_any, read_yaml_any, uri_exists, write_json_any  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rebounds prod pipeline.")
    parser.add_argument("--config", type=str, required=True, help="YAML config path or s3:// URI.")
    parser.add_argument("--slate-date", type=str, default="", help="Slate date YYYY-MM-DD (defaults to ET today).")
    parser.add_argument("--run-train", action="store_true", help="Train fresh models for this run before scoring.")
    parser.add_argument("--models-dir", type=str, default="", help="Existing local models dir when train is skipped.")
    parser.add_argument("--notify-which", type=str, default="both", choices=("ols", "xgb", "both"))
    parser.add_argument("--skip-audit", action="store_true", help="Skip audit-list checks in feature universe build.")
    parser.add_argument("--input-universe-mode", type=str, choices=("append", "replace"), default="append", help="Mode for input universe build.")
    return parser.parse_args()


def _fmt_cmd(cmd: list[str]) -> str:
    lines = ["  " + " ".join(cmd[:2])]
    i = 2
    while i < len(cmd):
        if cmd[i].startswith("--") and i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
            lines.append(f"    {cmd[i]} {cmd[i + 1]}")
            i += 2
        else:
            lines.append(f"    {cmd[i]}")
            i += 1
    return " \\\n".join(lines)


def run_cmd(cmd: list[str], repo_root: Path) -> None:
    print(f"run\n{_fmt_cmd(cmd)}")
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def publish_sns_healthcheck(topic_arn: str, run_id: str, slate_date: pd.Timestamp) -> str:
    import boto3

    message = (
        "Rebounds pipeline SNS health check\n"
        f"run_id={run_id}\n"
        f"slate_date={slate_date.date()}\n"
        f"sent_at_utc={datetime.now(timezone.utc).isoformat()}\n"
    )
    resp = boto3.client("sns").publish(
        TopicArn=topic_arn,
        Subject="Rebounds SNS health check",
        Message=message,
    )
    return resp["MessageId"]


def ensure_required_keys(cfg: dict) -> None:
    required = [
        "retrain_daily",
        "max_feature_lag_days",
        "max_model_age_days",
        "feature_universe_s3_uri",
        "feature_universe_season",
        "feature_build_cache_dir",
        "input_universe_s3_uri",
        "s3_bucket",
        "s3_models_prefix",
        "s3_runs_prefix",
        "sns_topic_arn",
        "notify_enabled",
        "build_props_scoring_input_from_live",
        "live_fetch_to_s3",
        "enable_pregame_feature_backfill",
    ]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing required config key: {key}")
    if not cfg["build_props_scoring_input_from_live"] and "props_input_uri" not in cfg:
        raise ValueError("Missing required config key: props_input_uri")


def check_feature_freshness(feat_s3_uri: str, slate_date: pd.Timestamp, max_feature_lag_days: int) -> None:
    feat = read_parquet_any(feat_s3_uri, columns=["date"])
    if len(feat) == 0:
        raise ValueError(f"No feature rows found: {feat_s3_uri}")
    latest_date = pd.to_datetime(feat["date"]).dt.normalize().max()
    lag_days = int((slate_date - latest_date).days)
    if lag_days < 0:
        raise ValueError(f"Feature latest_date={latest_date.date()} is after slate={slate_date.date()}.")
    if lag_days > max_feature_lag_days:
        raise ValueError(
            f"Feature data stale: latest_date={latest_date.date()} lag_days={lag_days} max_allowed={max_feature_lag_days}."
        )
    print(f"feature_freshness\n  latest_date={latest_date.date()}\n  lag_days={lag_days}")


def check_model_staleness(models_dir: Path, max_model_age_days: int) -> None:
    from src.io_utils import read_json_any
    manifest = read_json_any(str(models_dir / "manifest.json"))
    trained_at = pd.Timestamp(manifest["trained_at_utc"]).tz_convert("UTC")
    now_utc = pd.Timestamp(datetime.now(timezone.utc))
    age_days = float((now_utc - trained_at) / timedelta(days=1))
    if age_days > max_model_age_days:
        raise ValueError(
            f"Model stale: trained_at_utc={trained_at.isoformat()} age_days={age_days:.2f} max_allowed={max_model_age_days}."
        )
    print(f"model_freshness\n  trained_at_utc={trained_at.isoformat()}\n  age_days={age_days:.2f}")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    cfg = read_yaml_any(args.config)
    if cfg is None:
        raise ValueError(f"Config is empty: {args.config}")
    ensure_required_keys(cfg)

    if args.slate_date:
        slate_date = pd.Timestamp(args.slate_date).normalize()
        if slate_date.tzinfo is not None:
            raise ValueError("--slate-date must be timezone-naive YYYY-MM-DD")
    else:
        slate_date = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())
    print(f"slate_date_resolved\n  slate={slate_date.date()}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    s3_run_prefix = (
        f"s3://{cfg['s3_bucket']}/"
        f"{str(cfg['s3_runs_prefix']).strip('/')}/"
        f"{slate_date.date()}/{run_id}"
    )
    feat_s3_uri = str(cfg["feature_universe_s3_uri"])
    models_dir = Path(args.models_dir).expanduser().resolve() if args.models_dir else Path(f"/tmp/rebounds_models_{run_id}")

    # --- feature universe build ---
    tmp_feat_path = f"/tmp/rebounds_features_{run_id}.parquet"
    tmp_props_path = f"/tmp/rebounds_props_snapshot_{run_id}.parquet"
    feat_universe_cmd = [
        sys.executable,
        "src/nba_rebounds_modeling/00_research/scripts/build_rebounds_feature_universe.py",
        "--season",
        str(cfg["feature_universe_season"]),
        "--cache-dir",
        str(cfg["feature_build_cache_dir"]),
        "--output",
        tmp_feat_path,
        "--output-props",
        tmp_props_path,
        "--input-universe-s3-uri",
        str(cfg["input_universe_s3_uri"]),
        "--feature-universe-s3-uri",
        feat_s3_uri,
        "--props-history-s3-uri",
        f"{s3_run_prefix}/rebounds_props_history_{slate_date.date()}.parquet",
        "--input-universe-mode",
        args.input_universe_mode,
        "--force-refresh-cache",
        "true" if cfg.get("force_refresh_cache", False) else "false",
    ]
    if args.skip_audit:
        feat_universe_cmd.append("--skip-audit-list")
    run_cmd(feat_universe_cmd, repo_root)

    sns_healthcheck_message_id = ""
    if cfg["notify_enabled"]:
        sns_healthcheck_message_id = publish_sns_healthcheck(str(cfg["sns_topic_arn"]), run_id, slate_date)
        print(f"sns_healthcheck\n  topic_arn={cfg['sns_topic_arn']}\n  message_id={sns_healthcheck_message_id}")

    check_feature_freshness(feat_s3_uri, slate_date, int(cfg["max_feature_lag_days"]))

    # --- derive run URIs ---
    live_csv_uri = f"{s3_run_prefix}/live_rebounds_props_raw_{slate_date.date()}.csv"
    props_uri = f"{s3_run_prefix}/rebounds_props_scoring_input_{slate_date.date()}.parquet"
    feat_slice_uri = f"{s3_run_prefix}/rebounds_features_slice_{slate_date.date()}.parquet"
    scored_uri = f"{s3_run_prefix}/rebounds_scored_{slate_date.date()}.parquet"

    if not cfg["build_props_scoring_input_from_live"]:
        props_uri = str(cfg["props_input_uri"])

    # --- live props fetch ---
    if cfg["build_props_scoring_input_from_live"]:
        fetch_cmd = [
            sys.executable,
            "scripts/fetch_nba_player_rebounds_live.py",
            "--date",
            str(slate_date.date()),
            "--output-csv",
            live_csv_uri,
        ]
        if cfg["live_fetch_to_s3"]:
            fetch_cmd.append("--s3")
        run_cmd(fetch_cmd, repo_root)
        if not uri_exists(live_csv_uri):
            print(f"no_games_for_slate\n  slate={slate_date.date()}\n  pipeline_complete")
            return
        run_cmd(
            [
                sys.executable,
                "scripts/build_rebounds_scoring_input.py",
                "--live-csv",
                live_csv_uri,
                "--output",
                props_uri,
                "--date",
                str(slate_date.date()),
            ],
            repo_root,
        )

    # --- feature slice ---
    run_cmd(
        [
            sys.executable,
            "src/nba_rebounds_modeling/00_research/scripts/slice_rebounds_features_for_slate.py",
            "--feat",
            feat_s3_uri,
            "--as-of-date",
            str(slate_date.date()),
            "--output",
            feat_slice_uri,
        ],
        repo_root,
    )
    feat_slice_rows = len(read_parquet_any(feat_slice_uri))
    if feat_slice_rows == 0 and cfg["enable_pregame_feature_backfill"]:
        run_cmd(
            [
                sys.executable,
                "scripts/build_rebounds_pregame_feature_slice.py",
                "--feat",
                feat_s3_uri,
                "--props",
                props_uri,
                "--slate-date",
                str(slate_date.date()),
                "--output",
                feat_slice_uri,
            ],
            repo_root,
        )
        feat_slice_rows = len(read_parquet_any(feat_slice_uri))
        if feat_slice_rows == 0:
            raise ValueError("Pregame feature backfill produced 0 rows.")
        print(f"pregame_feature_backfill\n  rows={feat_slice_rows}")

    # --- train ---
    run_train = bool(cfg["retrain_daily"]) or args.run_train
    if run_train:
        run_cmd(
            [
                sys.executable,
                "src/nba_rebounds_modeling/00_research/scripts/train_rebounds_models.py",
                "--config",
                args.config,
                "--feat",
                feat_s3_uri,
                "--output-dir",
                str(models_dir),
            ],
            repo_root,
        )
    elif not models_dir.exists():
        raise ValueError("Training skipped but models-dir does not exist.")

    check_model_staleness(models_dir, int(cfg["max_model_age_days"]))

    # --- score ---
    score_cmd = [
        sys.executable,
        "src/nba_rebounds_modeling/00_research/scripts/score_rebounds_slate.py",
        "--models-dir",
        str(models_dir),
        "--feat-slice",
        feat_slice_uri,
        "--props",
        props_uri,
        "--slate-date",
        str(slate_date.date()),
        "--output",
        scored_uri,
    ]
    if "prod_min_edge_override" in cfg and cfg["prod_min_edge_override"] is not None:
        score_cmd.extend(["--min-edge", str(float(cfg["prod_min_edge_override"]))])
    run_cmd(score_cmd, repo_root)

    # --- notify ---
    if cfg["notify_enabled"]:
        run_cmd(
            [
                sys.executable,
                "src/nba_rebounds_modeling/00_research/scripts/notify_rebounds_plays.py",
                "--scored",
                scored_uri,
                "--which",
                args.notify_which,
                "--topic-arn",
                str(cfg["sns_topic_arn"]),
            ],
            repo_root,
        )

    # --- run manifest ---
    run_manifest = {
        "run_id": run_id,
        "slate_date": str(slate_date.date()),
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "config_uri": args.config,
        "feature_universe_s3_uri": feat_s3_uri,
        "props_uri": props_uri,
        "feat_slice_uri": feat_slice_uri,
        "models_dir": str(models_dir),
        "scored_uri": scored_uri,
        "s3_run_prefix": s3_run_prefix,
        "retrain_daily": bool(cfg["retrain_daily"]),
        "sns_healthcheck_message_id": sns_healthcheck_message_id,
    }
    manifest_uri = f"{s3_run_prefix}/run_manifest.json"
    write_json_any(run_manifest, manifest_uri)
    print(f"run_manifest_written\n  uri={manifest_uri}")

    print(f"pipeline_complete\n  slate={slate_date.date()}\n  run_id={run_id}\n  s3_run_prefix={s3_run_prefix}")


if __name__ == "__main__":
    main()

"""
Run the rebounds daily production pipeline end-to-end.

Context:
- This is the production entrypoint for daily rebounds automation.
- It builds a fresh full feature universe, scores today's slate, notifies plays,
  and writes run artifacts.
- Fail-fast by design: every required config key and step must succeed.

Usage:
    python scripts/run_rebounds_daily_pipeline.py \
        --config config/nba_rebounds_prod.lambda.yaml \
        --slate-date 2026-04-02
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rebounds prod pipeline.")
    parser.add_argument("--config", type=str, required=True, help="YAML config path.")
    parser.add_argument("--slate-date", type=str, default="", help="Slate date YYYY-MM-DD (defaults to ET today).")
    parser.add_argument("--run-train", action="store_true", help="Train fresh models for this run before scoring.")
    parser.add_argument("--models-dir", type=str, default="", help="Existing models dir when train is skipped.")
    parser.add_argument("--notify-which", type=str, default="both", choices=("ols", "xgb", "both"))
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Config is empty: {path}")
    return cfg


def run_cmd(cmd: list[str], repo_root: Path) -> None:
    print("run", " ".join(cmd), sep=" | ")
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid s3 uri: {s3_uri}")
    rest = s3_uri[5:]
    bucket, _, key = rest.partition("/")
    if bucket == "" or key == "":
        raise ValueError(f"Invalid s3 uri: {s3_uri}")
    return bucket, key


def upload_file_to_s3(local_path: Path, s3_uri: str) -> None:
    import boto3

    bucket, key = parse_s3_uri(s3_uri)
    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=local_path.read_bytes())


def upload_dir_to_s3(local_dir: Path, bucket: str, prefix: str) -> list[str]:
    import boto3

    s3 = boto3.client("s3")
    uploaded: list[str] = []
    for fp in sorted(local_dir.rglob("*")):
        if fp.is_file():
            rel = fp.relative_to(local_dir).as_posix()
            key = f"{prefix.rstrip('/')}/{rel}"
            s3.put_object(Bucket=bucket, Key=key, Body=fp.read_bytes())
            uploaded.append(f"s3://{bucket}/{key}")
    return uploaded


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
        "full_feature_parquet",
        "daily_runs_root",
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
    if not cfg["build_props_scoring_input_from_live"] and "props_input_path" not in cfg:
        raise ValueError("Missing required config key: props_input_path")


def check_feature_freshness(feat_path: Path, slate_date: pd.Timestamp, max_feature_lag_days: int) -> None:
    feat = pd.read_parquet(feat_path, columns=["date"])
    if len(feat) == 0:
        raise ValueError(f"No feature rows found: {feat_path}")
    latest_date = pd.to_datetime(feat["date"]).dt.normalize().max()
    lag_days = int((slate_date - latest_date).days)
    if lag_days < 0:
        raise ValueError(f"Feature latest_date={latest_date.date()} is after slate={slate_date.date()}.")
    if lag_days > max_feature_lag_days:
        raise ValueError(
            f"Feature data stale: latest_date={latest_date.date()} lag_days={lag_days} max_allowed={max_feature_lag_days}."
        )
    print("feature_freshness", f"latest_date={latest_date.date()}", f"lag_days={lag_days}", sep=" | ")


def check_model_staleness(models_dir: Path, max_model_age_days: int) -> None:
    manifest_path = models_dir / "manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    trained_at = pd.Timestamp(manifest["trained_at_utc"]).tz_convert("UTC")
    now_utc = pd.Timestamp(datetime.now(timezone.utc))
    age_days = float((now_utc - trained_at) / timedelta(days=1))
    if age_days > max_model_age_days:
        raise ValueError(
            f"Model stale: trained_at_utc={trained_at.isoformat()} age_days={age_days:.2f} max_allowed={max_model_age_days}."
        )
    print("model_freshness", f"trained_at_utc={trained_at.isoformat()}", f"age_days={age_days:.2f}", sep=" | ")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_config(config_path)
    ensure_required_keys(cfg)

    if args.slate_date:
        slate_date = pd.Timestamp(args.slate_date).normalize()
        if slate_date.tzinfo is not None:
            raise ValueError("--slate-date must be timezone-naive YYYY-MM-DD")
    else:
        slate_date = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())
    print("slate_date_resolved", f"slate={slate_date.date()}", sep=" | ")

    feat_path = Path(cfg["full_feature_parquet"]).expanduser().resolve()
    run_root = Path(cfg["daily_runs_root"]).expanduser().resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / str(slate_date.date()) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    feature_props_snapshot_path = run_dir / f"rebounds_props_history_{slate_date.date()}.parquet"
    run_cmd(
        [
            sys.executable,
            "src/nba_rebounds_modeling/00_research/scripts/build_rebounds_feature_universe.py",
            "--season",
            str(cfg["feature_universe_season"]),
            "--cache-dir",
            str(cfg["feature_build_cache_dir"]),
            "--output",
            str(feat_path),
            "--output-props",
            str(feature_props_snapshot_path),
            "--input-universe-s3-uri",
            str(cfg["input_universe_s3_uri"]),
            "--feature-universe-s3-uri",
            str(cfg["feature_universe_s3_uri"]),
        ],
        repo_root,
    )

    sns_healthcheck_message_id = ""
    if cfg["notify_enabled"]:
        sns_healthcheck_message_id = publish_sns_healthcheck(str(cfg["sns_topic_arn"]), run_id, slate_date)
        print("sns_healthcheck", f"topic_arn={cfg['sns_topic_arn']}", f"message_id={sns_healthcheck_message_id}", sep=" | ")

    check_feature_freshness(feat_path, slate_date, int(cfg["max_feature_lag_days"]))

    feat_slice_path = run_dir / f"rebounds_features_slice_{slate_date.date()}.parquet"
    scored_path = run_dir / f"rebounds_scored_{slate_date.date()}.parquet"
    models_dir = Path(args.models_dir).expanduser().resolve() if args.models_dir else run_dir / "models"
    props_path = Path(cfg["props_input_path"]).expanduser().resolve() if "props_input_path" in cfg else None

    if cfg["build_props_scoring_input_from_live"]:
        live_csv = run_dir / f"live_rebounds_props_raw_{slate_date.date()}.csv"
        props_path = run_dir / f"rebounds_props_scoring_input_{slate_date.date()}.parquet"
        fetch_cmd = [
            sys.executable,
            "scripts/fetch_nba_player_rebounds_live.py",
            "--date",
            str(slate_date.date()),
            "--output-csv",
            str(live_csv),
        ]
        if cfg["live_fetch_to_s3"]:
            fetch_cmd.append("--s3")
        run_cmd(fetch_cmd, repo_root)
        run_cmd(
            [
                sys.executable,
                "scripts/build_rebounds_scoring_input.py",
                "--live-csv",
                str(live_csv),
                "--output",
                str(props_path),
                "--date",
                str(slate_date.date()),
            ],
            repo_root,
        )

    run_cmd(
        [
            sys.executable,
            "src/nba_rebounds_modeling/00_research/scripts/slice_rebounds_features_for_slate.py",
            "--feat",
            str(feat_path),
            "--as-of-date",
            str(slate_date.date()),
            "--output",
            str(feat_slice_path),
        ],
        repo_root,
    )
    feat_slice_rows = len(pd.read_parquet(feat_slice_path))
    if feat_slice_rows == 0 and cfg["enable_pregame_feature_backfill"]:
        run_cmd(
            [
                sys.executable,
                "scripts/build_rebounds_pregame_feature_slice.py",
                "--feat",
                str(feat_path),
                "--props",
                str(props_path),
                "--slate-date",
                str(slate_date.date()),
                "--output",
                str(feat_slice_path),
            ],
            repo_root,
        )
        feat_slice_rows = len(pd.read_parquet(feat_slice_path))
        if feat_slice_rows == 0:
            raise ValueError("Pregame feature backfill produced 0 rows.")
        print("pregame_feature_backfill", f"rows={feat_slice_rows}", sep=" | ")

    run_train = bool(cfg["retrain_daily"]) or args.run_train
    if run_train:
        run_cmd(
            [
                sys.executable,
                "src/nba_rebounds_modeling/00_research/scripts/train_rebounds_models.py",
                "--config",
                str(config_path),
                "--feat",
                str(feat_path),
                "--output-dir",
                str(models_dir),
            ],
            repo_root,
        )
    elif not models_dir.exists():
        raise ValueError("Training skipped but models-dir does not exist.")

    check_model_staleness(models_dir, int(cfg["max_model_age_days"]))

    score_cmd = [
        sys.executable,
        "src/nba_rebounds_modeling/00_research/scripts/score_rebounds_slate.py",
        "--models-dir",
        str(models_dir),
        "--feat-slice",
        str(feat_slice_path),
        "--props",
        str(props_path),
        "--slate-date",
        str(slate_date.date()),
        "--output",
        str(scored_path),
    ]
    if "prod_min_edge_override" in cfg and cfg["prod_min_edge_override"] is not None:
        score_cmd.extend(["--min-edge", str(float(cfg["prod_min_edge_override"]))])
    scored_s3_uri = (
        f"s3://{cfg['s3_bucket']}/"
        f"{str(cfg['s3_runs_prefix']).rstrip('/')}/"
        f"{slate_date.date()}/{run_id}/rebounds_scored_{slate_date.date()}.parquet"
    )
    score_cmd.extend(["--s3-uri", scored_s3_uri])
    run_cmd(score_cmd, repo_root)

    if cfg["notify_enabled"]:
        run_cmd(
            [
                sys.executable,
                "src/nba_rebounds_modeling/00_research/scripts/notify_rebounds_plays.py",
                "--scored",
                str(scored_path),
                "--which",
                args.notify_which,
                "--topic-arn",
                str(cfg["sns_topic_arn"]),
            ],
            repo_root,
        )

    run_manifest = {
        "run_id": run_id,
        "slate_date": str(slate_date.date()),
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "feature_path": str(feat_path),
        "feature_universe_s3_uri": str(cfg["feature_universe_s3_uri"]),
        "feature_props_snapshot_path": str(feature_props_snapshot_path),
        "props_path": str(props_path),
        "feat_slice_path": str(feat_slice_path),
        "models_dir": str(models_dir),
        "scored_path": str(scored_path),
        "scored_s3_uri": scored_s3_uri,
        "retrain_daily": bool(cfg["retrain_daily"]),
        "sns_healthcheck_message_id": sns_healthcheck_message_id,
    }
    manifest_path = run_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    s3_prefix = f"{str(cfg['s3_runs_prefix']).rstrip('/')}/{slate_date.date()}/{run_id}"
    uploaded = upload_dir_to_s3(run_dir, str(cfg["s3_bucket"]), s3_prefix)
    print("s3_run_upload", f"uploaded_files={len(uploaded)}", f"prefix=s3://{cfg['s3_bucket']}/{s3_prefix}", sep=" | ")

    print("pipeline_complete", f"slate={slate_date.date()}", f"run_id={run_id}", f"run_dir={run_dir}", sep=" | ")


if __name__ == "__main__":
    main()

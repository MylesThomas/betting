"""
Build and publish the rebounds full feature universe artifact.

Context:
- This is the production-facing builder entrypoint for the full historical
  rebounds feature universe used by the daily pipeline.
- It intentionally hides research-era script names from operator workflows.
- It materializes:
  1) full feature universe parquet
  2) historical props scoring-input parquet
  3) optional S3 uploads for both artifacts

Usage:
    python scripts/build_rebounds_feature_universe.py \
        --season "*" \
        --cache-dir /tmp/rebounds_prod/cache \
        --output /tmp/rebounds_prod/rebounds_feature_universe.parquet \
        --output-props /tmp/rebounds_prod/rebounds_props_history.parquet \
        --input-universe-s3-uri s3://bucket/rebounds/input/rebounds_input_universe.parquet \
        --feature-universe-s3-uri s3://bucket/rebounds/features/rebounds_feature_universe.parquet
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rebounds full feature universe.")
    parser.add_argument("--season", type=str, required=True, help="Season selector passed to builder (e.g. '*' or '2025-26').")
    parser.add_argument("--cache-dir", type=str, required=True, help="Local cache dir used by builder.")
    parser.add_argument("--output", type=str, required=True, help="Output parquet path for full feature universe.")
    parser.add_argument("--output-props", type=str, required=True, help="Output parquet path for historical props input.")
    parser.add_argument(
        "--input-universe-s3-uri",
        type=str,
        required=True,
        help="S3 URI for rebounds input universe parquet.",
    )
    parser.add_argument(
        "--feature-universe-s3-uri",
        type=str,
        required=True,
        help="S3 URI destination for full feature universe parquet.",
    )
    parser.add_argument(
        "--props-history-s3-uri",
        type=str,
        default="",
        help="Optional S3 URI destination for historical props parquet.",
    )
    return parser.parse_args()


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


def run_cmd(cmd: list[str], repo_root: Path) -> None:
    print("run", " ".join(cmd), sep=" | ")
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_props_path = Path(args.output_props).expanduser().resolve()

    run_cmd(
        [
            sys.executable,
            "src/nba_rebounds_modeling/00_research/scripts/build_rebounds_input_universe.py",
            "--season",
            args.season,
            "--output",
            str(cache_dir / "rebounds_input_universe.parquet"),
            "--s3-uri",
            args.input_universe_s3_uri,
            "--mode",
            "append",
        ],
        repo_root,
    )

    run_cmd(
        [
            sys.executable,
            "src/nba_rebounds_modeling/00_research/scripts/v2_build_rebounds_universe.py",
            "--season",
            args.season,
            "--cache-dir",
            str(cache_dir),
            "--output",
            str(output_path),
            "--output-v3",
            str(output_props_path),
        ],
        repo_root,
    )

    upload_file_to_s3(output_path, args.feature_universe_s3_uri)
    print("uploaded_feature_universe", f"s3_uri={args.feature_universe_s3_uri}", sep=" | ")

    if args.props_history_s3_uri.strip() != "":
        upload_file_to_s3(output_props_path, args.props_history_s3_uri.strip())
        print("uploaded_props_history", f"s3_uri={args.props_history_s3_uri}", sep=" | ")


if __name__ == "__main__":
    main()

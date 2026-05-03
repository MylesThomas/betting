#!/usr/bin/env python3
"""
Verify B_MIN_MAX audit list columns vs scalars on a rebounds feature parquet.

``--parquet`` and ``--team-frame`` accept either a local path or an ``s3://bucket/key``
URI (read via boto3; same credential chain as ``aws s3``).

Spread (game-line) checks run when either (a) the feature row includes ``team_normalized``,
``home_team_norm``, ``away_team_norm``, or (b) you pass ``--team-frame`` with those columns
(season, date, player_normalized, game_id, plus the three team names).

With ``--auto-build-missing-team`` (or env ``REBOUNDS_AUTO_BUILD_IF_MISSING_TEAM=1``), if the
feature parquet is missing the row-level team columns and the target is an ``s3://`` URI, the
verifier runs ``build_rebounds_feature_universe`` (same entrypoint as the daily Lambda) and
re-reads that key before checking.

Without team context, lines + rolling tails are still verified; spread checks are skipped.

Exit codes: 0 ok, 1 verification or load error.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    cur = here
    while True:
        if (cur / ".gitignore").exists() and (cur / "src").exists():
            return cur
        if cur.parent == cur:
            raise FileNotFoundError("Could not find repo root")
        cur = cur.parent


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid s3 uri: {s3_uri}")
    rest = s3_uri[5:]
    bucket, _, key = rest.partition("/")
    if bucket == "" or key == "":
        raise ValueError(f"Invalid s3 uri: {s3_uri}")
    return bucket, key


def _env_bool(name: str, default: str = "0") -> bool:
    v = str(os.environ.get(name, default)).strip().lower()
    return v in ("1", "true", "yes", "on")


def _default_input_universe_s3_for_feature(feat_s3: str) -> str:
    """Conventional input-universe key under the same bucket as the feature file."""
    bucket, _ = _parse_s3_uri(feat_s3)
    return f"s3://{bucket}/rebounds/input/rebounds_input_universe.parquet"


def _run_build_rebounds_feature_universe(
    repo_root: Path,
    *,
    season: str,
    cache_dir: Path,
    output_parquet: Path,
    output_props: Path,
    input_universe_s3_uri: str,
    feature_universe_s3_uri: str,
) -> None:
    build_script = repo_root / "scripts" / "build_rebounds_feature_universe.py"
    if not build_script.is_file():
        raise FileNotFoundError(f"Missing {build_script}")
    cmd: list[str] = [
        sys.executable,
        str(build_script),
        "--season",
        season,
        "--cache-dir",
        str(cache_dir),
        "--output",
        str(output_parquet),
        "--output-props",
        str(output_props),
        "--input-universe-s3-uri",
        input_universe_s3_uri,
        "--feature-universe-s3-uri",
        feature_universe_s3_uri,
    ]
    print(
        "audit_list_verify | auto_build_missing_team | running build_rebounds_feature_universe → "
        f"feature_s3={feature_universe_s3_uri}",
        file=sys.stderr,
    )
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def read_parquet_s3_or_local(uri: str) -> pd.DataFrame:
    """Load parquet from ``s3://`` (boto3) or local filesystem."""
    u = uri.strip()
    if u.startswith("s3://"):
        import boto3
        from botocore.exceptions import ClientError

        bucket, key = _parse_s3_uri(u)
        try:
            body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey"):
                raise FileNotFoundError(f"S3 object not found: {u}") from exc
            raise
        return pd.read_parquet(BytesIO(body))
    p = Path(u).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"parquet not found: {p}")
    return pd.read_parquet(p)


def main() -> int:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.nba_rebounds_modeling.rebounds_feature_spec import TEAM_CONTEXT_COLS
    from src.nba_rebounds_modeling.rebounds_audit_list_verify import (
        print_audit_sample_to_stdout,
        sample_audit_rows,
        verify_audit_lists_dataframe,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--parquet",
        type=str,
        required=True,
        help="Feature universe parquet: local path or s3://bucket/key",
    )
    p.add_argument(
        "--team-frame",
        type=str,
        default=None,
        help="Optional team parquet: local path or s3://bucket/key",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=500,
        metavar="N",
        help="Sample size (default 500). Use --full-scan for all rows.",
    )
    p.add_argument(
        "--full-scan",
        action="store_true",
        help="Verify every row (slow).",
    )
    p.add_argument(
        "--show-rows",
        type=int,
        default=0,
        metavar="N",
        help="After a successful run, print scalars + audit lists for N rows (0 = off). "
        "Default: see --show-by (most recent dates in the file, e.g. “today’s” games).",
    )
    p.add_argument(
        "--show-by",
        choices=("recent", "verification_sample"),
        default="recent",
        help="With --show-rows: 'recent' = N rows with latest `date` in the parquet; "
        "'verification_sample' = first N of the same random sample used for checks.",
    )
    p.add_argument(
        "--auto-build-missing-team",
        action="store_true",
        help="If the feature lacks team columns and --parquet is s3://, run "
        "scripts/build_rebounds_feature_universe.py then re-read the object.",
    )
    p.add_argument(
        "--no-auto-build-missing-team",
        action="store_true",
        help="Do not run the feature build, even if REBOUNDS_AUTO_BUILD_IF_MISSING_TEAM=1.",
    )
    p.add_argument(
        "--input-universe-s3-uri",
        type=str,
        default="",
        help="Override input-universe s3 (default: s3://<--parquet bucket>/rebounds/input/rebounds_input_universe.parquet).",
    )
    p.add_argument(
        "--build-cache-dir",
        type=str,
        default="",
        help="Cache dir for the builder. Default: temp dir, or REBOUNDS_BUILD_CACHE_DIR env.",
    )
    p.add_argument(
        "--build-season",
        type=str,
        default="",
        help="Season for the builder; default: REBOUNDS_BUILD_SEASON or *.",
    )
    args = p.parse_args()

    def _resolve_auto_build() -> bool:
        if args.no_auto_build_missing_team:
            return False
        if args.auto_build_missing_team:
            return True
        return _env_bool("REBOUNDS_AUTO_BUILD_IF_MISSING_TEAM", "0")

    try:
        df = read_parquet_s3_or_local(args.parquet)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: could not load --parquet: {exc}", file=sys.stderr)
        return 1

    team_frame = None
    if args.team_frame is not None:
        try:
            team_frame = read_parquet_s3_or_local(args.team_frame)
        except (FileNotFoundError, OSError, ValueError) as exc:
            print(f"error: could not load --team-frame: {exc}", file=sys.stderr)
            return 1
        need = [
            "season",
            "date",
            "player_normalized",
            "game_id",
            "team_normalized",
            "home_team_norm",
            "away_team_norm",
        ]
        missing = [c for c in need if c not in team_frame.columns]
        if missing:
            print(
                f"error: team-frame missing columns {missing}; need {need}",
                file=sys.stderr,
            )
            return 1
    else:
        has_team = all(c in df.columns for c in TEAM_CONTEXT_COLS)
        if not has_team and _resolve_auto_build():
            feat = args.parquet.strip()
            if not feat.startswith("s3://"):
                print(
                    "error: --auto-build-missing-team only supports s3:// --parquet "
                    f"(team columns still missing: {list(TEAM_CONTEXT_COLS)} on local file).",
                    file=sys.stderr,
                )
                return 1
            in_uri = (
                (args.input_universe_s3_uri or "").strip()
                or str(os.environ.get("REBOUNDS_INPUT_UNIVERSE_S3_URI", "") or "").strip()
                or _default_input_universe_s3_for_feature(feat)
            )
            season = (
                (args.build_season or "").strip()
                or str(os.environ.get("REBOUNDS_BUILD_SEASON", "") or "").strip()
                or "*"
            )
            bcache = (args.build_cache_dir or "").strip() or str(
                os.environ.get("REBOUNDS_BUILD_CACHE_DIR", "") or ""
            )
            if not bcache:
                bcache = tempfile.mkdtemp(prefix="rebounds_autobuild_cache_")
            cache_p = Path(bcache).expanduser().resolve()
            cache_p.mkdir(parents=True, exist_ok=True)
            out_dir = Path(tempfile.mkdtemp(prefix="rebounds_autobuild_out_"))
            out_feat = out_dir / "rebounds_feature_universe.parquet"
            out_props = out_dir / "rebounds_props_history_build.parquet"
            try:
                _run_build_rebounds_feature_universe(
                    root,
                    season=season,
                    cache_dir=cache_p,
                    output_parquet=out_feat,
                    output_props=out_props,
                    input_universe_s3_uri=in_uri,
                    feature_universe_s3_uri=feat,
                )
            except subprocess.CalledProcessError as exc:
                print(
                    f"error: build_rebounds_feature_universe failed (exit {exc.returncode})",
                    file=sys.stderr,
                )
                return 1
            try:
                df = read_parquet_s3_or_local(args.parquet)
            except (FileNotFoundError, OSError, ValueError) as exc:
                print(f"error: could not reload --parquet after build: {exc}", file=sys.stderr)
                return 1
            has_team = all(c in df.columns for c in TEAM_CONTEXT_COLS)
            if not has_team:
                print(
                    "error: team columns still missing after auto-build: "
                    f"expected {list(TEAM_CONTEXT_COLS)}",
                    file=sys.stderr,
                )
                return 1
        elif not all(c in df.columns for c in TEAM_CONTEXT_COLS):
            print(
                "note: feature parquet missing "
                f"{list(TEAM_CONTEXT_COLS)}; spread checks need --team-frame or a rebuilt "
                "rebounds_feature_universe.parquet from the full-universe build (team columns on each row).",
                file=sys.stderr,
            )

    max_rows = None if args.full_scan else int(args.max_rows)
    mode = "full" if max_rows is None else "sample"
    n_chk = len(df) if max_rows is None else min(max_rows, len(df))
    print(
        "audit_list_verify",
        f"parquet={args.parquet.strip()}",
        f"mode={mode}",
        f"n_rows={len(df):,}",
        f"n_checked~={n_chk:,}",
        sep=" | ",
    )

    sample = sample_audit_rows(df, max_rows=max_rows)

    try:
        verify_audit_lists_dataframe(
            df,
            team_frame=team_frame,
            max_rows=max_rows,
            sample_df=sample,
        )
    except (AssertionError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("audit_list_verify | ok")

    n_show = int(args.show_rows)
    if n_show > 0:
        if args.show_by == "recent":
            print_audit_sample_to_stdout(df, team_frame, n_show=n_show, show_by="recent")
        else:
            print_audit_sample_to_stdout(sample, team_frame, n_show=n_show, show_by="verification_sample")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

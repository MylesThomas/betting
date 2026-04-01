"""
Optional SNS notification for rebounds plays (tabular text body).

Context:
- Reads scored slate parquet from prod_score_rebounds_slate.py.
- If env SNS_TOPIC_ARN or --topic-arn is set, publishes; else prints plays table to stdout.
- Does not fetch data; ingestion/email wiring stays separate from analysis.

Usage:
    python src/nba_rebounds_modeling/00_research/scripts/prod_notify_rebounds_sns.py \\
        --scored ~/Downloads/tmp/rebounds_scored_2025-03-15.parquet \\
        --which both
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def ensure_repo_root_on_syspath() -> Path:
    current = Path.cwd().resolve()
    while True:
        if (current / ".gitignore").exists() and (current / "src").exists():
            if str(current) not in sys.path:
                sys.path.insert(0, str(current))
            return current
        if current.parent == current:
            raise FileNotFoundError("Could not locate repo root")
        current = current.parent


ensure_repo_root_on_syspath()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Notify rebounds plays via SNS or stdout.")
    p.add_argument("--scored", type=str, required=True, help="prod_score output parquet.")
    p.add_argument(
        "--which",
        type=str,
        default="both",
        choices=("ols", "xgb", "both"),
        help="Which play column(s) to include in the message.",
    )
    p.add_argument("--topic-arn", type=str, default="", help="SNS topic ARN (or set SNS_TOPIC_ARN).")
    p.add_argument("--subject", type=str, default="NBA rebounds plays")
    return p.parse_args()


def build_plays_table(df: pd.DataFrame, which: str) -> pd.DataFrame:
    if which == "ols":
        sub = df.loc[df["play_under_ols"]].copy()
    elif which == "xgb":
        sub = df.loc[df["play_under_xgb"]].copy()
    else:
        sub = df.loc[df["play_under_ols"] | df["play_under_xgb"]].copy()

    cols = [
        "season",
        "date",
        "player_normalized",
        "game_id",
        "bookmaker",
        "line",
        "consensus_reb_line",
        "over_odds",
        "under_odds",
        "yhat_ols",
        "yhat_xgb",
        "p_under_ols",
        "p_under_xgb",
        "edge_under_ols",
        "edge_under_xgb",
        "play_under_ols",
        "play_under_xgb",
    ]
    for c in cols:
        if c not in sub.columns:
            raise ValueError(f"scored parquet missing column: {c}")
    return sub[cols]


def main() -> None:
    args = parse_args()
    path = Path(args.scored).expanduser()
    df = pd.read_parquet(path)
    plays = build_plays_table(df, args.which)
    body = plays.to_string(index=False)
    if len(plays) == 0:
        body = "(no plays for this filter)"

    topic = args.topic_arn.strip() or os.environ.get("SNS_TOPIC_ARN", "").strip()
    if not topic:
        print(body)
        return

    import boto3

    sns = boto3.client("sns")
    sns.publish(TopicArn=topic, Subject=args.subject[:100], Message=body[:256_000])
    print(f"published to SNS topic (rows={len(plays):,})")


if __name__ == "__main__":
    main()

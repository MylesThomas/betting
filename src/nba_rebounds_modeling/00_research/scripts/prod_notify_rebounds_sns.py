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

from src.nba_rebounds_modeling.rebounds_feature_spec import B_MIN_MAX_FEATS  # noqa: E402


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

    sub["play_bucket"] = "neither"
    sub.loc[sub["play_under_ols"] & ~sub["play_under_xgb"], "play_bucket"] = "ols_only"
    sub.loc[~sub["play_under_ols"] & sub["play_under_xgb"], "play_bucket"] = "xgb_only"
    sub.loc[sub["play_under_ols"] & sub["play_under_xgb"], "play_bucket"] = "both"

    cols = [
        "play_bucket",
        "season",
        "date",
        "player_normalized",
        "game_id",
        "game_id_source",
        "nba_game_id",
        "odds_event_id",
        "bookmaker",
        "line",
        "consensus_reb_line",
        "over_odds",
        "under_odds",
        *B_MIN_MAX_FEATS,
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
            raise ValueError(
                f"scored parquet missing column: {c} "
                f"(expected model inputs {B_MIN_MAX_FEATS} on merged scored output)"
            )
    return sub[cols]


def fmt_float(value: float | int | bool | str | None, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def build_email_body(plays: pd.DataFrame, which: str) -> str:
    if len(plays) == 0:
        return f"NBA rebounds plays ({which})\n\n(no plays for this filter)"

    bucket_counts = (
        plays["play_bucket"]
        .value_counts()
        .reindex(["both", "ols_only", "xgb_only"], fill_value=0)
        .to_dict()
    )
    lines: list[str] = [
        f"NBA rebounds plays ({which})",
        "Rows are ONLY recommended under plays.",
        "play_bucket: both=both models agree, ols_only=OLS only, xgb_only=XGB only",
        f"rows={len(plays):,} | both={bucket_counts['both']:,} | ols_only={bucket_counts['ols_only']:,} | xgb_only={bucket_counts['xgb_only']:,}",
        "",
    ]
    ordered = plays.sort_values(["date", "player_normalized", "bookmaker", "line"]).reset_index(drop=True)
    for idx, row in ordered.iterrows():
        lines.append(
            f"{idx + 1}. [{row['play_bucket']}] {row['player_normalized']} | {row['date']} | {row['bookmaker']}"
        )
        lines.append(
            "   ids:"
            f" game_id={row['game_id']}"
            f" game_id_source={row['game_id_source']}"
            f" nba_game_id={row['nba_game_id']}"
            f" odds_event_id={row['odds_event_id']}"
        )
        lines.append(
            "   line:"
            f" book={fmt_float(row['line'])}"
            f" consensus={fmt_float(row['consensus_reb_line'])}"
            f" over_odds={fmt_float(row['over_odds'], 0)}"
            f" under_odds={fmt_float(row['under_odds'], 0)}"
        )
        lines.append("   inputs:")
        for i, feat in enumerate(B_MIN_MAX_FEATS, start=1):
            lines.append(f"   - x{i} {feat}={fmt_float(row[feat], digits=2)}")
        lines.append(
            "   model:"
            f" yhat_ols={fmt_float(row['yhat_ols'])}"
            f" yhat_xgb={fmt_float(row['yhat_xgb'])}"
            f" p_under_ols={fmt_float(row['p_under_ols'])}"
            f" p_under_xgb={fmt_float(row['p_under_xgb'])}"
        )
        lines.append(
            "   edge/play:"
            f" edge_under_ols={fmt_float(row['edge_under_ols'])}"
            f" edge_under_xgb={fmt_float(row['edge_under_xgb'])}"
            f" play_under_ols={fmt_float(row['play_under_ols'])}"
            f" play_under_xgb={fmt_float(row['play_under_xgb'])}"
        )
        lines.append("")
    return "\n".join(lines).rstrip()


def main() -> None:
    args = parse_args()
    path = Path(args.scored).expanduser()
    df = pd.read_parquet(path)
    plays = build_plays_table(df, args.which)
    body = build_email_body(plays, args.which)

    topic = args.topic_arn.strip() or os.environ.get("SNS_TOPIC_ARN", "").strip()
    if not topic:
        print(body)
        return

    import boto3

    sns = boto3.client("sns")
    resp = sns.publish(TopicArn=topic, Subject=args.subject[:100], Message=body[:256_000])
    print(
        "published_to_sns",
        f"topic_arn={topic}",
        f"rows={len(plays):,}",
        f"message_id={resp['MessageId']}",
        sep=" | ",
    )


if __name__ == "__main__":
    main()

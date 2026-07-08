"""
NBA rebounds daily Lambda orchestrator.

Runs two steps in order (fail-hard):
1) Daily pipeline scoring + notify
2) Settlement for latest run on ET date

Env:
- ODDS_API_KEY (required by live props fetch)
- SNS_TOPIC_ARN (required if notify_enabled=true in config)
- SETTLEMENT_SES_SOURCE (optional; verified SES identity — multipart HTML+text settlement email)
- SETTLEMENT_SES_TO (optional; comma-separated To addresses when SES is used; requires SETTLEMENT_SES_SOURCE)
- CONFIG_PATH (optional; default: config/nba_rebounds_prod.yaml — deploy sets
  config/nba_rebounds_prod.lambda.yaml on the function)
- SETTLE_BUCKET (optional; default: nba-betting-mt)
- SETTLE_PREFIX (optional; default: rebounds/daily_runs)
- SETTLE_DAYS_LAG (optional; default: 1, so settlement end date is yesterday ET)
- SETTLE_WINDOW_DAYS (optional; default: 3, re-settle rolling window for late actuals)
"""

from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from urllib.parse import quote
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import botocore.exceptions
import pandas as pd

from src.nba_rebounds_settlement_email import (
    format_settlement_email_plays_table,
    format_settlement_email_plays_table_html,
)


ET = ZoneInfo("America/New_York")


def _repo_root() -> Path:
    current = Path(__file__).resolve().parent
    while True:
        if (current / "src").exists() and (current / ".gitignore").exists():
            return current
        if current.parent == current:
            raise FileNotFoundError("Could not locate repo root")
        current = current.parent


def _resolve_mode(event: dict | None) -> str:
    if event is None:
        return "both"
    if "mode" not in event:
        return "both"
    mode = str(event["mode"]).strip().lower()
    if mode not in {"pipeline", "settlement", "both"}:
        raise ValueError(f"Unsupported mode: {mode}")
    return mode


def _run(cmd: list[str], cwd: Path) -> None:
    print("run", " ".join(cmd), sep=" | ")
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def _run_capture(cmd: list[str], cwd: Path) -> str:
    print("run", " ".join(cmd), sep=" | ")
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stdout}")
    return result.stdout


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid s3 uri: {s3_uri}")
    rest = s3_uri[5:]
    bucket, _, key = rest.partition("/")
    if bucket == "" or key == "":
        raise ValueError(f"Invalid s3 uri: {s3_uri}")
    return bucket, key


def _s3_console_object_https(s3_uri: str, region: str | None = None) -> str:
    """HTTPS link to open one object in the S3 console (sign-in required). ``s3://`` is not a browser URL."""
    u = (s3_uri or "").strip()
    if not u.startswith("s3://"):
        return u
    try:
        bucket, key = _parse_s3_uri(u)
    except ValueError:
        return u
    r = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-2"
    enc = quote(key, safe="")
    return f"https://{r}.console.aws.amazon.com/s3/object/{bucket}?region={r}&prefix={enc}"


def _read_csv_s3(s3_uri: str) -> pd.DataFrame | None:
    import botocore.exceptions
    bucket, key = _parse_s3_uri(s3_uri)
    try:
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        return pd.read_csv(BytesIO(body))
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] in ['NoSuchKey', '404']:
            return None
        raise


def _read_text_s3(s3_uri: str) -> str | None:
    import botocore.exceptions
    bucket, key = _parse_s3_uri(s3_uri)
    try:
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        return body.decode("utf-8")
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ["NoSuchKey", "404"]:
            return None
        raise


def _email_plays_yesterday_s3_uri(yesterday_rollup_uri: str) -> str:
    """Sibling of yesterday.csv written by rebounds_settle_runs (yesterday rollup only)."""
    if not yesterday_rollup_uri.endswith("yesterday.csv"):
        return ""
    return yesterday_rollup_uri[: -len("yesterday.csv")] + "email_plays_yesterday.csv"


def _email_plays_yesterday_html_s3_uri(csv_uri: str) -> str:
    """Same prefix as ``email_plays_yesterday.csv`` → styled HTML artifact for browser open."""
    if not csv_uri.endswith("email_plays_yesterday.csv"):
        return ""
    return csv_uri[: -len("email_plays_yesterday.csv")] + "email_plays_yesterday.html"

def _indent(lines: list[str], prefix: str = "  ") -> list[str]:
    return [f"{prefix}{line}" if line else "" for line in lines]

def _format_window_section(label: str, rollup: pd.DataFrame | None) -> list[str]:
    title = label.upper()

    if rollup is None or len(rollup) == 0:
        return [
            title,
            "  No scored runs found for this window",
        ]

    # Align with scripts/rebounds_settle_runs.py: `neither` has n_bets=0 but
    # reference_pnl_units is hypothetical PnL at listed under odds (wins/losses).
    r = rollup.copy()
    if "reference_pnl_units" not in r.columns:
        r["reference_pnl_units"] = 0.0

    grouped = (
        r.groupby("strategy_bucket", as_index=False)
        .agg(
            n_rows=("n_rows", "sum"),
            n_bets=("n_bets", "sum"),
            n_win=("n_win", "sum"),
            n_loss=("n_loss", "sum"),
            n_push=("n_push", "sum"),
            n_unsettled=("n_unsettled", "sum"),
            pnl_units=("pnl_units", "sum"),
            reference_pnl_units=("reference_pnl_units", "sum"),
        )
        .sort_values("strategy_bucket")
        .reset_index(drop=True)
    )

    import duckdb

    con = duckdb.connect()
    con.register("rollup", grouped)
    formatted_summary = con.execute(
        """
        SELECT
            strategy_bucket AS strategy,
            n_rows AS rows,
            n_bets AS bets,
            printf('%d-%d-%d', n_win, n_loss, n_push) AS wlp,
            n_unsettled AS un,
            round(
                CASE
                    WHEN strategy_bucket = 'neither' THEN reference_pnl_units
                    ELSE pnl_units
                END,
                3
            ) AS pnl,
            round(CASE WHEN (n_win + n_loss) > 0 THEN n_win * 1.0 / (n_win + n_loss) ELSE 0.0 END, 3) AS hit_rate,
            round(
                CASE
                    WHEN (n_win + n_loss + n_push) <= 0 THEN 0.0
                    WHEN strategy_bucket = 'neither' THEN reference_pnl_units * 1.0 / (n_win + n_loss + n_push)
                    ELSE pnl_units * 1.0 / (n_win + n_loss + n_push)
                END,
                3
            ) AS roi
        FROM rollup
        ORDER BY
            CASE WHEN strategy_bucket = 'neither' THEN 1 ELSE 0 END,
            strategy_bucket
        """
    ).fetchdf()
    con.close()

    lines = [title]
    for _, row in formatted_summary.iterrows():
        strat = str(row["strategy"]).upper()
        lines.append(f"  [{strat}]")

        pnl_val = float(row["pnl"])
        pnl_str = f"+{pnl_val:.3f}" if pnl_val > 0 else f"{pnl_val:.3f}"
        hr_str = f"{float(row['hit_rate']) * 100:.1f}%"
        roi_str = f"{float(row['roi']) * 100:.1f}%"

        ref_note = " (reference @ line odds)" if strat == "NEITHER" else ""

        lines.append(f"    Rows: {row['rows']} | Bets: {row['bets']} | W-L-P: {row['wlp']} | Unsettled: {row['un']}")
        lines.append(f"    PnL: {pnl_str}u{ref_note} | Hit Rate: {hr_str} | ROI: {roi_str}")
        lines.append("")

    return lines[:-1]  # Remove trailing blank line


def _build_settlement_notification_bundle(
    settle_end_date_et,
    yesterday_rollup_uri: str,
    all_time_rollup_uri: str,
    warnings: list[str] | None = None,
) -> dict:
    """Shared inputs for SNS text + optional SES HTML (same plays dataframe)."""
    warnings = warnings or []
    yesterday_rollup = _read_csv_s3(yesterday_rollup_uri)
    all_time_rollup = _read_csv_s3(all_time_rollup_uri)
    plays_uri = _email_plays_yesterday_s3_uri(yesterday_rollup_uri)
    plays_df = _read_csv_s3(plays_uri) if plays_uri else None

    yesterday_lines = _format_window_section("yesterday", yesterday_rollup)
    all_time_lines = _format_window_section("all-time", all_time_rollup)

    lines = [
        "NBA rebounds settled results",
        f"settle_end_date_et: {settle_end_date_et.isoformat()}",
        "",
    ]
    if warnings:
        lines.extend([
            "WARNINGS",
            f"  Partial settlement detected ({len(warnings)} rows):",
            *[f"    - {w}" for w in warnings],
            "",
        ])
    lines.extend([
        *yesterday_lines,
        "",
        *all_time_lines,
        "",
        "ROLLUP FILES",
        f"  1. {yesterday_rollup_uri}",
        f"     Console: {_s3_console_object_https(yesterday_rollup_uri)}",
        f"  2. {all_time_rollup_uri}",
        f"     Console: {_s3_console_object_https(all_time_rollup_uri)}",
    ])
    plays_for_email = None
    plays_html_uri = _email_plays_yesterday_html_s3_uri(plays_uri) if plays_uri else ""
    if plays_df is not None and len(plays_df) > 0:
        body = format_settlement_email_plays_table(plays_df).strip()
        max_chars = 200_000
        if len(body) > max_chars:
            body = body[:max_chars] + "\n...(truncated for SNS message size limit)"
        lines.extend(
            [
                "",
                "SETTLE-WINDOW PLAYS (both / ols / xgb)",
                "(SNS email is plain text — columns may not line up in Gmail. For an HTML table in your inbox, use SES; see plan_rebs_results_formatting_v2.md.)",
                body,
            ]
        )
        if plays_uri:
            lines.extend(
                [
                    "",
                    f"PLAYS SOURCE (CSV): {plays_uri}",
                    f"  Console: {_s3_console_object_https(plays_uri)}",
                ]
            )
        if plays_html_uri:
            lines.extend(
                [
                    "",
                    "PLAYS HTML (styled table — tap HTTPS link; s3:// is not valid in Safari/Chrome):",
                    f"  {_s3_console_object_https(plays_html_uri)}",
                    f"  S3 URI: {plays_html_uri}",
                ]
            )
        plays_for_email = plays_df

    return {
        "text_lines": lines,
        "plays_df": plays_for_email,
        "plays_uri": plays_uri,
        "plays_html_uri": plays_html_uri,
        "yesterday_lines": yesterday_lines,
        "all_time_lines": all_time_lines,
        "warnings": warnings,
        "yesterday_rollup_uri": yesterday_rollup_uri,
        "all_time_rollup_uri": all_time_rollup_uri,
        "settle_end_date_et": settle_end_date_et,
    }


def _html_pre_block(lines: list[str], mono_font: str) -> str:
    text = html.escape("\n".join(lines))
    return (
        f'<pre style="margin:0 0 16px;white-space:pre-wrap;word-break:break-word;'
        f"font-family:{mono_font};font-size:12px;line-height:1.4;color:#1a1a1a;\">{text}</pre>"
    )


def _build_settlement_email_html(bundle: dict) -> str:
    """Multipart HTML part: summary sections as pre; plays as HTML table (Phase A styling)."""
    mono = "ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace"
    sans = (
        "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"
    )
    settle_end = bundle["settle_end_date_et"]
    warnings = bundle["warnings"]
    parts: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{html.escape('NBA rebounds settled results')}</title></head>",
        f'<body style="margin:0;padding:16px;background-color:#f4f4f5;font-family:{sans};'
        'font-size:13px;line-height:1.45;color:#1a1a1a;">',
        '<div style="max-width:720px;margin:0 auto;background-color:#ffffff;padding:20px 20px 24px;'
        'border-radius:8px;border:1px solid #e2e2e4;">',
        '<p style="margin:0 0 8px;font-size:15px;font-weight:600;">'
        f"{html.escape('NBA rebounds settled results')}</p>",
        '<p style="margin:0 0 16px;color:#444;font-family:'
        f"{mono};font-size:12px;\">"
        f"{html.escape(f'settle_end_date_et: {settle_end.isoformat()}')}</p>",
    ]
    if warnings:
        parts.append('<p style="margin:0 0 8px;font-weight:600;color:#b45309;">WARNINGS</p>')
        parts.append('<ul style="margin:0 0 16px;padding-left:20px;">')
        for w in warnings:
            parts.append(f'<li style="margin-bottom:4px;">{html.escape(w)}</li>')
        parts.append("</ul>")
    parts.append(_html_pre_block(bundle["yesterday_lines"], mono))
    parts.append(_html_pre_block(bundle["all_time_lines"], mono))
    rollup_lines = [
        "ROLLUP FILES",
        f"  1. {bundle['yesterday_rollup_uri']}",
        f"     Console: {_s3_console_object_https(bundle['yesterday_rollup_uri'])}",
        f"  2. {bundle['all_time_rollup_uri']}",
        f"     Console: {_s3_console_object_https(bundle['all_time_rollup_uri'])}",
    ]
    parts.append(_html_pre_block(rollup_lines, mono))
    plays_df = bundle["plays_df"]
    if plays_df is not None and len(plays_df) > 0:
        parts.append(
            '<p style="margin:0 0 10px;font-weight:600;font-size:13px;">'
            f"{html.escape('SETTLE-WINDOW PLAYS (both / ols / xgb)')}</p>"
        )
        parts.append(format_settlement_email_plays_table_html(plays_df))
        plays_uri = bundle.get("plays_uri") or ""
        plays_html_uri = bundle.get("plays_html_uri") or ""
        if plays_uri:
            cu = _s3_console_object_https(plays_uri)
            parts.append(
                '<p style="margin:14px 0 0;font-family:'
                f'{mono};font-size:11px;word-break:break-all;color:#555;">'
                f"{html.escape('PLAYS SOURCE (CSV): ' + plays_uri)}</p>"
                f'<p style="margin:4px 0 0;font-size:13px;">'
                f'<a href="{html.escape(cu, quote=True)}">Open CSV in S3 console</a>'
                f" <span style=\"color:#666;font-size:11px;\">(sign in to AWS)</span></p>"
            )
        if plays_html_uri:
            hu = _s3_console_object_https(plays_html_uri)
            parts.append(
                '<p style="margin:10px 0 0;font-family:'
                f'{mono};font-size:11px;word-break:break-all;color:#555;">'
                f"{html.escape('PLAYS HTML (S3): ' + plays_html_uri)}</p>"
                f'<p style="margin:4px 0 0;font-size:13px;">'
                f'<a href="{html.escape(hu, quote=True)}">Open styled table (HTML) in S3 console</a>'
                f" <span style=\"color:#666;font-size:11px;\">(sign in to AWS)</span></p>"
            )
    parts.extend(["</div>", "</body></html>"])
    return "".join(parts)


def _send_settlement_ses_email(
    source: str,
    to_addresses: list[str],
    subject: str,
    text_body: str,
    html_body: str,
) -> str:
    client = boto3.client("ses")
    resp = client.send_email(
        Source=source,
        Destination={"ToAddresses": to_addresses},
        Message={
            "Subject": {"Data": subject, "Charset": "UTF-8"},
            "Body": {
                "Text": {"Data": text_body, "Charset": "UTF-8"},
                "Html": {"Data": html_body, "Charset": "UTF-8"},
            },
        },
    )
    return resp["MessageId"]


def _publish_combined_settlement_sns(
    topic_arn: str,
    settle_end_date_et,
    yesterday_rollup_uri: str,
    all_time_rollup_uri: str,
    warnings: list[str] | None = None,
) -> str:
    """Try SES first (HTML inbox), then SNS plain text. If SES fails, append the reason to the SNS body."""
    bundle = _build_settlement_notification_bundle(
        settle_end_date_et,
        yesterday_rollup_uri,
        all_time_rollup_uri,
        warnings,
    )
    base_text = "\n".join(bundle["text_lines"])

    ses_footer = ""
    ses_message_id = ""
    ses_source = os.environ.get("SETTLEMENT_SES_SOURCE", "").strip()
    ses_to_raw = os.environ.get("SETTLEMENT_SES_TO", "").strip()
    if ses_source and ses_to_raw:
        to_list = [a.strip() for a in ses_to_raw.split(",") if a.strip()]
        if to_list:
            try:
                html_body = _build_settlement_email_html(bundle)
                max_ses_chars = 9_000_000
                tb = (
                    base_text
                    if len(base_text) <= max_ses_chars
                    else base_text[: max_ses_chars - 80] + "\n...(truncated for email size limit)"
                )
                hb = (
                    html_body
                    if len(html_body) <= max_ses_chars
                    else html_body[: max_ses_chars - 100] + "<!-- truncated -->"
                )
                ses_message_id = _send_settlement_ses_email(
                    ses_source,
                    to_list,
                    "NBA rebounds settled results",
                    tb,
                    hb,
                )
                print(
                    "published_settlement_to_ses",
                    f"source={ses_source}",
                    f"message_id={ses_message_id}",
                    sep=" | ",
                )
                ses_footer = (
                    "\n\n---\n[SES] Multipart email (HTML table + plain text) was accepted by AWS "
                    f"(MessageId={ses_message_id}).\n"
                    f"  To: {ses_to_raw}\n"
                    f"  From: {ses_source}\n"
                    "If you only see this SNS mail, search Gmail for that From address and check Spam / "
                    "All Mail; Gmail may thread it separately from AWS Notifications."
                )
            except botocore.exceptions.ClientError as exc:
                err = exc.response.get("Error", {})
                detail = err.get("Message", str(exc))
                print(
                    "settlement_ses_send_failed",
                    f"source={ses_source}",
                    detail,
                    sep=" | ",
                )
                ses_footer = (
                    "\n\n---\n[SES] HTML settlement email was NOT sent. AWS says:\n"
                    f"  {detail}\n"
                    "Common fixes (region us-east-2): verify SETTLEMENT_SES_SOURCE in SES; in sandbox "
                    "verify SETTLEMENT_SES_TO as well; Lambda role needs ses:SendEmail "
                    "(deploy script Step 1b). Check Spam for mail From the source address if it succeeded."
                )
            except Exception as exc:
                print("settlement_ses_send_failed", f"source={ses_source}", str(exc), sep=" | ")
                ses_footer = (
                    "\n\n---\n[SES] HTML settlement email was NOT sent:\n"
                    f"  {exc!s}\n"
                    "See CloudWatch logs for this Lambda for details."
                )

    text_message = (base_text + ses_footer)[:256_000]

    last_message_id = ""
    if topic_arn.strip():
        resp = boto3.client("sns").publish(
            TopicArn=topic_arn.strip(),
            Subject="NBA rebounds settled results",
            Message=text_message,
        )
        last_message_id = resp["MessageId"]

    return last_message_id or ses_message_id


def _run_settle(
    root: Path,
    settle_bucket: str,
    settle_prefix: str,
    start_date_et,
    end_date_et,
    rollup_s3_uri: str,
    settle_max_unmatched_bet_rows: int,
) -> str:
    cmd = [
        sys.executable,
        "src/nba_rebounds_modeling/00_research/scripts/settle_rebounds_runs.py",
        "--bucket", settle_bucket,
        "--runs-prefix", settle_prefix,
        "--start-date", start_date_et.isoformat(),
        "--end-date", end_date_et.isoformat(),
        "--latest-only",
        "--allow-empty",
        "--overwrite",
        "--max-unmatched-bet-rows", str(settle_max_unmatched_bet_rows),
        "--rollup-s3-uri", rollup_s3_uri,
    ]
    return _run_capture(cmd, root)


def _parse_s3_run_prefix(pipeline_stdout: str) -> str | None:
    """Extract s3_run_prefix from pipeline stdout. Returns None if not found."""
    m = re.search(r"s3_run_prefix=(\S+)", pipeline_stdout)
    return m.group(1) if m else None


def _send_unified_email(
    root: Path,
    scored_uri: str | None,
    yesterday_rollup_uri: str,
    all_time_rollup_uri: str,
    today_et: str,
    ses_source: str,
    ses_to_list: list[str],
    n_qualifying: int = 0,
) -> str:
    """Build and send the unified daily plays + results email via SES."""
    mono = "ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace"
    tmp_html = f"/tmp/rebounds_plays_email_{today_et}.html"

    if scored_uri:
        # Build the rich plays HTML via subprocess (includes yesterday + all-time sections)
        notify_cmd = [
            sys.executable,
            "src/nba_rebounds_modeling/00_research/scripts/prod_notify_rebounds_sns.py",
            "--scored", scored_uri,
            "--which", "both",
            "--format", "html",
            "--records-csv", all_time_rollup_uri,
            "--yesterday-rollup", yesterday_rollup_uri,
            "--output-html", tmp_html,
        ]
        _run(notify_cmd, root)
        plays_html = Path(tmp_html).read_text(encoding="utf-8")
    else:
        # No games today — use existing settlement text formatters wrapped in pre blocks
        yesterday_rollup = _read_csv_s3(yesterday_rollup_uri)
        all_time_rollup = _read_csv_s3(all_time_rollup_uri)
        yesterday_lines = _format_window_section("yesterday", yesterday_rollup)
        all_time_lines = _format_window_section("all-time", all_time_rollup)
        plays_html = (
            f'<p style="font-size:13px;color:#374151;margin:16px;">'
            f'No NBA games on the slate today ({today_et}) — no plays generated.</p>\n'
            + _html_pre_block(yesterday_lines, mono)
            + _html_pre_block(all_time_lines, mono)
        )

    full_html = (
        "<!DOCTYPE html><html lang='en'><head>"
        "<meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "</head>"
        f"<body style='margin:0;padding:16px;background:#f4f4f5;font-family:Arial,sans-serif;'>"
        f"{plays_html}"
        "</body></html>"
    )

    plays_word = f"{n_qualifying} play{'s' if n_qualifying != 1 else ''}"
    subject = f"NBA Rebounds · {today_et} · {plays_word}"
    text_body = f"NBA rebounds daily | {today_et} | {plays_word}"

    msg_id = _send_settlement_ses_email(ses_source, ses_to_list, subject, text_body, full_html)
    print(
        "sent_unified_email_ses",
        f"message_id={msg_id}",
        f"to={ses_to_list}",
        f"n_qualifying={n_qualifying}",
        sep=" | ",
    )
    return msg_id


def lambda_handler(event, context):
    # DuckDB httpfs expects a writable home directory in Lambda.
    os.environ.setdefault("HOME", "/tmp")
    root = _repo_root()
    today_et = datetime.now(ET).strftime("%Y-%m-%d")

    config_path = os.environ.get("CONFIG_PATH", "config/nba_rebounds_prod.yaml")
    settle_bucket = os.environ.get("SETTLE_BUCKET", "nba-betting-mt")
    settle_prefix = os.environ.get("SETTLE_PREFIX", "rebounds/daily_runs")
    settle_days_lag = int(os.environ.get("SETTLE_DAYS_LAG", "1"))
    settle_window_days = int(os.environ.get("SETTLE_WINDOW_DAYS", "1"))
    if settle_window_days < 1:
        raise ValueError("SETTLE_WINDOW_DAYS must be >= 1")
    settle_all_time_days = int(os.environ.get("SETTLE_ALL_TIME_DAYS", "999999"))
    if settle_all_time_days < 1:
        raise ValueError("SETTLE_ALL_TIME_DAYS must be >= 1")
    settle_max_unmatched_bet_rows = int(os.environ.get("SETTLE_MAX_UNMATCHED_BET_ROWS", "0"))
    settle_end_date_et = (datetime.now(ET) - timedelta(days=settle_days_lag)).date()
    settle_start_date_et = settle_end_date_et - timedelta(days=settle_window_days - 1)
    if settle_all_time_days >= 999999:
        settle_all_time_start_date_et = datetime(1900, 1, 1).date()
    else:
        settle_all_time_start_date_et = settle_end_date_et - timedelta(days=settle_all_time_days - 1)
    ses_source = os.environ.get("SETTLEMENT_SES_SOURCE", "").strip()
    ses_to_raw = os.environ.get("SETTLEMENT_SES_TO", "").strip()
    ses_to_list = [a.strip() for a in ses_to_raw.split(",") if a.strip()] if ses_to_raw else []
    mode = _resolve_mode(event if isinstance(event, dict) else None)

    step_results = []
    scored_uri: str | None = None
    n_qualifying = 0
    try:
        # ── settlement ────────────────────────────────────────────────────────
        # Always settle before scoring so results are ready for the unified email.
        yesterday_rollup_uri = ""
        all_time_rollup_uri = ""
        if mode in {"settlement", "both"}:
            stamp = datetime.now(ET).strftime("%Y%m%dT%H%M%S")
            base_rollup_prefix = f"{settle_prefix.rstrip('/')}/_rollups/{today_et}/{stamp}"
            yesterday_rollup_uri = f"s3://{settle_bucket}/{base_rollup_prefix}/yesterday.csv"
            all_time_rollup_uri = f"s3://{settle_bucket}/{base_rollup_prefix}/all_time.csv"

            yesterday_out = _run_settle(
                root, settle_bucket, settle_prefix,
                settle_start_date_et, settle_end_date_et,
                yesterday_rollup_uri, settle_max_unmatched_bet_rows,
            )
            all_time_out = _run_settle(
                root, settle_bucket, settle_prefix,
                settle_all_time_start_date_et, settle_end_date_et,
                all_time_rollup_uri, settle_max_unmatched_bet_rows,
            )

            warnings = sorted(set(
                line for line in (yesterday_out + "\n" + all_time_out).splitlines()
                if "status=partial" in line and "settlement_guardrail" in line
            ))
            if warnings:
                print("settlement_warnings", "\n".join(warnings), sep=" | ")

            step_results.append({"step": "settlement", "status": "ok"})

        # ── scoring pipeline ──────────────────────────────────────────────────
        if mode in {"pipeline", "both"}:
            pipeline_cmd = [
                sys.executable,
                "src/nba_rebounds_modeling/00_research/scripts/run_rebounds_daily_pipeline.py",
                "--config", config_path,
                "--slate-date", today_et,
                "--no-notify",
            ]
            pipeline_out = _run_capture(pipeline_cmd, root)
            print(pipeline_out)

            if "no_games_for_slate" not in pipeline_out:
                s3_run_prefix = _parse_s3_run_prefix(pipeline_out)
                if s3_run_prefix:
                    scored_uri = f"{s3_run_prefix}/rebounds_scored_{today_et}.parquet"

            step_results.append({"step": "pipeline", "status": "ok"})

        # ── 3. Unified email ──────────────────────────────────────────────────
        if mode == "both":
            if scored_uri:
                try:
                    bucket, key = _parse_s3_uri(scored_uri)
                    body_bytes = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
                    scored_df = pd.read_parquet(BytesIO(body_bytes))
                    n_qualifying = int(
                        (scored_df["play_under_ols"] | scored_df["play_under_xgb"]).sum()
                    )
                except Exception as exc:
                    print(f"n_qualifying_read_failed | {exc}")

            if ses_source and ses_to_list:
                _send_unified_email(
                    root=root,
                    scored_uri=scored_uri,
                    yesterday_rollup_uri=yesterday_rollup_uri,
                    all_time_rollup_uri=all_time_rollup_uri,
                    today_et=today_et,
                    ses_source=ses_source,
                    ses_to_list=ses_to_list,
                    n_qualifying=n_qualifying,
                )
                step_results.append({"step": "unified_email", "status": "ok"})
            else:
                print("unified_email_skipped | reason=SETTLEMENT_SES_SOURCE / SETTLEMENT_SES_TO not set")

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "status": "ok",
                    "mode": mode,
                    "date_et": today_et,
                    "settle_start_date_et": settle_start_date_et.isoformat(),
                    "settle_all_time_start_date_et": settle_all_time_start_date_et.isoformat(),
                    "settle_end_date_et": settle_end_date_et.isoformat(),
                    "settle_window_days": settle_window_days,
                    "settle_all_time_days": settle_all_time_days,
                    "n_qualifying": n_qualifying,
                    "steps": step_results,
                }
            ),
        }
    except Exception as exc:
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "status": "error",
                    "mode": mode,
                    "date_et": today_et,
                    "settle_start_date_et": settle_start_date_et.isoformat(),
                    "settle_all_time_start_date_et": settle_all_time_start_date_et.isoformat(),
                    "settle_end_date_et": settle_end_date_et.isoformat(),
                    "settle_window_days": settle_window_days,
                    "settle_all_time_days": settle_all_time_days,
                    "steps": step_results,
                    "error": str(exc),
                }
            ),
        }

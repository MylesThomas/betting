"""
Shared HTML/subject builder for spine-update emails across MLB/NBA pipelines.

Usage in a Lambda orchestrator:
    from spine_email_utils import build_spine_update_html, build_spine_update_subject

    html    = build_spine_update_html("MLB Total Bases", spine_stdout, today_et)
    subject = build_spine_update_subject("MLB Total Bases", today_et, spine_stdout)
    _send_ses(subject, html)
"""
from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

ET    = ZoneInfo("America/New_York")
_SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"


def _extract_row_count(stdout: str) -> str | None:
    m = re.search(r"([\d,]+)\s+rows?", stdout)
    return m.group(1) if m else None


def build_spine_update_subject(pipeline_name: str, today_et: str, spine_stdout: str) -> str:
    now_str = datetime.now(ET).strftime("%H:%M ET")
    n_rows  = _extract_row_count(spine_stdout)
    if n_rows:
        return f"{pipeline_name} — Spine Updated · {n_rows} rows · {today_et} {now_str}"
    return f"{pipeline_name} — Spine Updated · {today_et} {now_str}"


def build_spine_update_html(
    pipeline_name: str,
    spine_stdout: str,
    today_et: str,
    settle_summary: dict | None = None,
) -> str:
    now_str = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")

    settle_html = ""
    if settle_summary:
        gd        = settle_summary.get("gameday", "—")
        wins      = settle_summary.get("wins", 0)
        losses    = settle_summary.get("losses", 0)
        pushes    = settle_summary.get("pushes", 0)
        dnps      = settle_summary.get("dnps", 0)
        pnl       = settle_summary.get("pnl", 0.0)
        at_bets   = settle_summary.get("all_time_bets", 0)
        at_wins   = settle_summary.get("all_time_wins", 0)
        at_losses = settle_summary.get("all_time_losses", 0)
        at_pnl    = settle_summary.get("all_time_pnl", 0.0)
        record    = f"{wins}W–{losses}L{f'–{pushes}P' if pushes else ''}{f'–{dnps}DNP' if dnps else ''}"
        pnl_color    = "#276221" if pnl >= 0 else "#c0392b"
        at_pnl_color = "#276221" if at_pnl >= 0 else "#c0392b"
        settle_html = f"""
<h3 style='margin-bottom:4px'>Settlement — {gd}</h3>
<p style='margin:0 0 12px'>
  Record: <strong>{record}</strong> &nbsp;·&nbsp;
  P&amp;L: <strong style='color:{pnl_color}'>{pnl:+.2f}u</strong> &nbsp;·&nbsp;
  All-time: <strong>{at_wins}W–{at_losses}L</strong> &nbsp;·&nbsp;
  <strong style='color:{at_pnl_color}'>{at_pnl:+.2f}u</strong> ({at_bets} bets)
</p>"""

    spine_lines   = [ln.strip() for ln in spine_stdout.splitlines() if ln.strip()]
    spine_preview = "\n".join(spine_lines[-25:]) if spine_lines else "(no output)"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{font-family:{_SANS};color:#222;max-width:900px;margin:auto;padding:20px}}
  h2 {{color:#2c3e50;margin-bottom:4px}}
  h3 {{color:#2c3e50}}
  pre {{white-space:pre-wrap;word-break:break-word}}
</style>
</head><body>
<h2>{pipeline_name} — Spine Updated — {today_et}</h2>
<p style='color:#666;font-size:13px;margin-top:0'>{now_str} &nbsp;·&nbsp; spine rebuild complete</p>
{settle_html}
<h3 style='margin-bottom:4px'>Spine rebuild</h3>
<pre style='background:#f4f4f4;padding:10px;border-radius:4px;font-size:11px;overflow-x:auto'>{spine_preview}</pre>

<p style='font-size:12px;color:#888;margin-top:16px'>
  9:00 AM ET scoring run will use today's rebuilt spine. If spine timestamp above is not from {today_et}, the scoring run will be blocked.
</p>
</body></html>"""

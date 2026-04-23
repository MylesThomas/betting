"""Shared helpers for yesterday email plays: sorted export frame + SNS monospace + HTML table."""

from __future__ import annotations

import html
import pandas as pd

_EMAIL_PLAYS_CSV_COLUMNS = [
    "player_normalized",
    "strategy_bucket",
    "bookmaker",
    "line",
    "reb_actual",
    "diff",
    "result",
    "under_odds",
    "date",
]


def prepare_email_plays_dataframe(plays: pd.DataFrame | None) -> pd.DataFrame:
    """Sort both/ols/xgb rows for CSV export and table rendering (full frame, no row cap)."""
    if plays is None or len(plays) == 0:
        return pd.DataFrame(columns=_EMAIL_PLAYS_CSV_COLUMNS)
    work = plays.copy()
    work["diff"] = pd.to_numeric(work["reb_actual"], errors="coerce") - pd.to_numeric(
        work["line"], errors="coerce"
    )
    order = {"win": 0, "loss": 1, "unsettled": 2, "push": 3}
    work["_ord"] = work["result"].map(order).fillna(9).astype(int)
    work = work.sort_values(
        ["_ord", "strategy_bucket", "player_normalized", "bookmaker"],
        kind="mergesort",
    ).drop(columns=["_ord"])
    cols = [c for c in _EMAIL_PLAYS_CSV_COLUMNS if c in work.columns]
    return work.loc[:, cols]


def prepare_email_plays_display_slice(
    plays: pd.DataFrame | None, max_rows: int = 600
) -> tuple[pd.DataFrame, int, bool]:
    """Sorted frame capped for email/SNS. Returns (slice, total_row_count, truncated)."""
    work = prepare_email_plays_dataframe(plays)
    total = len(work)
    if total == 0:
        return work, 0, False
    truncated = total > max_rows
    if truncated:
        work = work.head(max_rows).copy()
    return work, total, truncated


def format_settlement_email_plays_table(plays: pd.DataFrame, max_rows: int = 600) -> str:
    """Monospace table for SNS: both/ols/xgb rows only, sorted like ad-hoc DuckDB checks."""
    work, total, truncated = prepare_email_plays_display_slice(plays, max_rows)
    if len(work) == 0:
        return ""
    lines: list[str] = []
    if truncated:
        lines.append(
            f"(showing first {max_rows} of {total} rows; see settled parquet in S3 for full detail)"
        )
        lines.append("")
    hdr = f"{'player':<22} {'strat':<6} {'bookmaker':<16} {'line':>6} {'act':>5} {'diff':>6} {'result':<10} {'und':>5} {'date':<12}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for _, r in work.iterrows():
        p = str(r.get("player_normalized", ""))[:20]
        st = str(r.get("strategy_bucket", ""))[:5]
        bk = str(r.get("bookmaker", ""))[:14]
        ln = r.get("line")
        act = r.get("reb_actual")
        dfv = r.get("diff")
        res = str(r.get("result", ""))[:9]
        uo = r.get("under_odds")
        dt = str(r.get("date", ""))[:12]
        ln_s = "" if pd.isna(ln) else f"{float(ln):.1f}"
        act_s = "" if pd.isna(act) else f"{float(act):.1f}"
        df_s = "" if pd.isna(dfv) else f"{float(dfv):.1f}"
        uo_s = "" if pd.isna(uo) else f"{int(float(uo))}"
        lines.append(
            f"{p:<22} {st:<6} {bk:<16} {ln_s:>6} {act_s:>5} {df_s:>6} {res:<10} {uo_s:>5} {dt:<12}"
        )
    return "\n".join(lines)


def _result_cell_style(result: str) -> str:
    r = str(result).strip().lower()
    if r == "win":
        return "font-weight:700;color:#166534;"
    if r == "loss":
        return "font-weight:700;color:#991b1b;"
    if r == "push":
        return "font-weight:700;color:#92400e;"
    if r == "unsettled":
        return "font-weight:700;color:#4b5563;"
    return "font-weight:700;color:#1a1a1a;"


def format_settlement_email_plays_table_html(plays: pd.DataFrame, max_rows: int = 600) -> str:
    """HTML table fragment (no outer document): same sort/cap as ``format_settlement_email_plays_table``."""
    work, total, truncated = prepare_email_plays_display_slice(plays, max_rows)
    if len(work) == 0:
        return ""

    notice = ""
    if truncated:
        msg = html.escape(
            f"(showing first {max_rows} of {total} rows; see settled parquet in S3 for full detail)"
        )
        notice = (
            f'<p style="margin:0 0 10px;font-size:12px;color:#555;">{msg}</p>'
        )

    th = (
        "padding:8px;background-color:#2d3748;color:#f7fafc;font-weight:600;"
        "border-bottom:1px solid #1a202c;font-size:13px;"
    )
    th_player = f"{th}text-align:left;max-width:180px;"
    th_strat = f"{th}text-align:left;width:52px;"
    th_book = f"{th}text-align:left;max-width:130px;"
    th_num = f"{th}text-align:right;font-variant-numeric:tabular-nums;"
    th_res = f"{th}text-align:left;width:88px;"
    th_date = f"{th}text-align:left;width:96px;"

    header = (
        "<thead><tr>"
        f'<th style="{th_player}">player</th>'
        f'<th style="{th_strat}">strat</th>'
        f'<th style="{th_book}">bookmaker</th>'
        f'<th style="{th_num}">line</th>'
        f'<th style="{th_num}">act</th>'
        f'<th style="{th_num}">diff</th>'
        f'<th style="{th_res}">result</th>'
        f'<th style="{th_num}">und</th>'
        f'<th style="{th_date}">date</th>'
        "</tr></thead>"
    )

    td_txt = (
        "padding:7px 10px;border-bottom:1px solid #e8e8ec;max-width:180px;"
        "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px;"
    )
    td_strat = (
        "padding:7px 8px;border-bottom:1px solid #e8e8ec;white-space:nowrap;font-size:13px;"
    )
    td_book = (
        "padding:7px 8px;border-bottom:1px solid #e8e8ec;max-width:130px;"
        "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px;"
    )
    td_num = (
        "padding:7px 8px;border-bottom:1px solid #e8e8ec;text-align:right;"
        "font-variant-numeric:tabular-nums;font-size:13px;"
    )
    td_res = "padding:7px 8px;border-bottom:1px solid #e8e8ec;font-size:13px;"
    td_date = (
        "padding:7px 10px;border-bottom:1px solid #e8e8ec;white-space:nowrap;font-size:13px;"
    )

    rows_html: list[str] = []
    for i, (_, r) in enumerate(work.iterrows()):
        bg = "#ffffff" if i % 2 == 0 else "#f8fafc"
        p_raw = str(r.get("player_normalized", ""))
        st_raw = str(r.get("strategy_bucket", ""))
        bk_raw = str(r.get("bookmaker", ""))
        res_raw = str(r.get("result", ""))
        dt_raw = str(r.get("date", ""))

        ln = r.get("line")
        act = r.get("reb_actual")
        dfv = r.get("diff")
        uo = r.get("under_odds")
        ln_s = "" if pd.isna(ln) else f"{float(ln):.1f}"
        act_s = "" if pd.isna(act) else f"{float(act):.1f}"
        df_s = "" if pd.isna(dfv) else f"{float(dfv):.1f}"
        uo_s = "" if pd.isna(uo) else f"{int(float(uo))}"

        p_esc = html.escape(p_raw)
        st_esc = html.escape(st_raw)
        bk_esc = html.escape(bk_raw)
        res_esc = html.escape(res_raw)
        dt_esc = html.escape(dt_raw)

        res_style = _result_cell_style(res_raw)

        rows_html.append(
            f'<tr style="background-color:{bg};">'
            f'<td style="{td_txt}" title="{html.escape(p_raw, quote=True)}">{p_esc}</td>'
            f'<td style="{td_strat}" title="{html.escape(st_raw, quote=True)}">{st_esc}</td>'
            f'<td style="{td_book}" title="{html.escape(bk_raw, quote=True)}">{bk_esc}</td>'
            f'<td style="{td_num}">{html.escape(ln_s)}</td>'
            f'<td style="{td_num}">{html.escape(act_s)}</td>'
            f'<td style="{td_num}">{html.escape(df_s)}</td>'
            f'<td style="{td_res}{res_style}">{res_esc}</td>'
            f'<td style="{td_num}">{html.escape(uo_s)}</td>'
            f'<td style="{td_date}">{dt_esc}</td>'
            "</tr>"
        )

    table = (
        '<table role="presentation" cellpadding="0" cellspacing="0" border="0" '
        'style="width:100%;max-width:680px;border-collapse:collapse;border:1px solid #d0d0d4;">'
        f"{header}<tbody>{''.join(rows_html)}</tbody></table>"
    )
    return notice + table


def wrap_email_plays_html_document(
    inner_html: str, title: str = "NBA rebounds — settle-window plays"
) -> str:
    """Full HTML document around a body fragment (S3 audit / browser open)."""
    t = html.escape(title)
    font = (
        "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"
    )
    return (
        "<!DOCTYPE html>\n"
        f'<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{t}</title></head>\n"
        f'<body style="margin:0;padding:16px;background-color:#f4f4f5;font-family:{font};'
        'font-size:13px;line-height:1.45;color:#1a1a1a;">\n'
        '<div style="max-width:720px;margin:0 auto;background-color:#ffffff;padding:20px 24px;'
        'border-radius:8px;border:1px solid #e2e2e4;">\n'
        f"{inner_html}\n"
        "</div></body></html>\n"
    )

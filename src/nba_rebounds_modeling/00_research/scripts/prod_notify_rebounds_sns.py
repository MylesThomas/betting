"""
Notify rebounds plays via SNS or stdout — HTML email (default) or plain text.

Context:
- Reads scored slate parquet from prod_score_rebounds_slate.py.
- HTML format (default): rich email matching mock at knowledge-base/raw/20260705-nba-rebounds.html.
  Shows ALL evaluated players, not just qualifying ones.
- Text format: legacy tabular output (unchanged).
- If env SNS_TOPIC_ARN or --topic-arn is set, publishes to SNS; else prints to stdout.
- Stats panels (PnL / Record / Win% / ROI) require --records-csv from
  compile_rebounds_strategy_records.py; panels show N/A when omitted.

Usage:
    python src/nba_rebounds_modeling/00_research/scripts/prod_notify_rebounds_sns.py \\
        --scored ~/Downloads/tmp/rebounds_scored_2025-03-15.parquet \\
        --which both \\
        --format html \\
        --records-csv ~/Downloads/tmp/rebounds_records.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
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

from src.io_utils import read_parquet_any  # noqa: E402
from src.nba_rebounds_modeling.rebounds_feature_spec import B_MIN_MAX_FEATS  # noqa: E402

# ── colour palette matching the mock email ───────────────────────────────────
_G_PLAYER = "#334155"
_G_MARKET = "#0f766e"
_G_OLS = "#4338ca"
_G_XGB = "#1d4ed8"
_G_EDGE_O = "#065f46"
_G_EDGE_X = "#14532d"
_G_BEST = "#92400e"
_G_FEAT = "#374151"
_G_RESULT = "#374151"

_TH = "padding:5px 8px;color:#fff;font-size:10px;text-align:center;vertical-align:middle;"


# ── shared helpers ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Notify rebounds plays via SNS or stdout.")
    p.add_argument("--scored", type=str, required=True, help="prod_score output parquet.")
    p.add_argument(
        "--which",
        type=str,
        default="both",
        choices=("ols", "xgb", "both"),
        help="Which play column(s) to include (used in text format and qualifying-row logic).",
    )
    p.add_argument(
        "--format",
        type=str,
        default="html",
        choices=("html", "text"),
        help="Output format. html = rich email; text = legacy tabular.",
    )
    p.add_argument(
        "--records-csv",
        type=str,
        default="",
        help="Path to compile_rebounds_strategy_records.py output CSV for season stats panels.",
    )
    p.add_argument("--topic-arn", type=str, default="", help="SNS topic ARN (or set SNS_TOPIC_ARN).")
    p.add_argument("--subject", type=str, default="NBA rebounds plays")
    return p.parse_args()


def _american_to_raw_implied(american: np.ndarray) -> np.ndarray:
    odds = american.astype(np.float64, copy=False)
    out = np.empty_like(odds, dtype=np.float64)
    neg = odds < 0
    out[neg] = (-odds[neg]) / ((-odds[neg]) + 100.0)
    out[~neg] = 100.0 / (odds[~neg] + 100.0)
    return out


def fmt_float(value: float | int | bool | str | None, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


# ── text format (legacy, unchanged) ──────────────────────────────────────────

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


def build_text_body(plays: pd.DataFrame, which: str) -> str:
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
            f" game_id_source={row.get('game_id_source', 'NA')}"
            f" nba_game_id={row.get('nba_game_id', 'NA')}"
            f" odds_event_id={row.get('odds_event_id', 'NA')}"
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


# ── HTML format ───────────────────────────────────────────────────────────────

def _compute_display_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    p_raw_o = _american_to_raw_implied(out["over_odds"].to_numpy())
    p_raw_u = _american_to_raw_implied(out["under_odds"].to_numpy())
    total = p_raw_o + p_raw_u
    out["novig_prob_over"] = p_raw_o / total
    out["novig_prob_under"] = p_raw_u / total

    out["p_over_ols"] = 1.0 - out["p_under_ols"]
    out["p_over_xgb"] = 1.0 - out["p_under_xgb"]
    out["edge_over_ols"] = out["p_over_ols"] - p_raw_o
    out["edge_over_xgb"] = out["p_over_xgb"] - p_raw_o
    out["best_under_edge"] = out[["edge_under_ols", "edge_under_xgb"]].max(axis=1)

    if "team_normalized" in out.columns and "home_team_norm" in out.columns:
        def _opp(row: pd.Series) -> str:
            team = row.get("team_normalized", "") or ""
            home = row.get("home_team_norm", "") or ""
            away = row.get("away_team_norm", "") or ""
            return away if team == home else home

        out["opponent"] = out.apply(_opp, axis=1)
        out["team"] = out["team_normalized"].fillna("")
    else:
        out["opponent"] = ""
        out["team"] = ""

    out["direction"] = "UNDER"
    return out


def _load_records(records_csv: str) -> dict:
    """
    Parse compile_rebounds_strategy_records CSV into season stats per model.
    Returns dict with keys 'ols', 'xgb', 'both', each containing:
        n_win, n_loss, n_push, pnl_units, hit_rate, roi
    Returns empty dict if path not provided or file unreadable.
    """
    if not records_csv.strip():
        return {}
    try:
        rec = pd.read_csv(records_csv)
    except Exception:
        return {}

    def _agg(buckets: list[str]) -> dict:
        sub = rec.loc[rec["strategy_bucket"].isin(buckets)]
        if len(sub) == 0:
            return {}
        n_win = int(sub["n_win"].sum())
        n_loss = int(sub["n_loss"].sum())
        n_push = int(sub["n_push"].sum())
        n_bets = n_win + n_loss + n_push
        pnl = float(sub["pnl_units"].sum())
        hit = n_win / n_bets if n_bets > 0 else float("nan")
        roi = pnl / n_bets if n_bets > 0 else float("nan")
        return {"n_win": n_win, "n_loss": n_loss, "n_push": n_push, "pnl": pnl, "hit": hit, "roi": roi}

    return {
        "ols": _agg(["both", "ols"]),
        "xgb": _agg(["both", "xgb"]),
        "both": _agg(["both"]),
    }


def _panel(label: str, value: str, large: bool = False, green: bool = False) -> str:
    val_size = "20px" if large else "16px"
    val_color = ";color:#16a34a" if green else ""
    return (
        f'  <div style="background:#f0f7ff;border:1px solid #bfdbfe;border-radius:6px;'
        f'padding:10px 16px;min-width:100px;">\n'
        f'    <div style="font-size:10px;color:#6b7280;text-transform:uppercase;">{label}</div>\n'
        f'    <div style="font-size:{val_size};font-weight:700{val_color};">{value}</div>\n'
        f'  </div>\n'
    )


def _fmt_record(stats: dict) -> str:
    if not stats:
        return "N/A"
    return f"{stats['n_win']:,}W–{stats['n_loss']:,}L"


def _fmt_pct(val: float | None, suffix: str = "%") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val * 100:.1f}{suffix}"


def _fmt_roi(val: float | None) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val * 100:.1f}%"


def _build_stats_panels(records: dict, pnl_ols: float | None) -> str:
    ols = records.get("ols", {})
    xgb = records.get("xgb", {})
    both = records.get("both", {})

    pnl_label = "Season PnL (OLS)"
    pnl_val = f"+{pnl_ols:.1f}u" if pnl_ols is not None and pnl_ols >= 0 else (f"{pnl_ols:.1f}u" if pnl_ols is not None else "N/A")
    pnl_green = pnl_ols is not None and pnl_ols >= 0

    panels = '<div style="display:flex;gap:12px;margin:14px 16px;flex-wrap:wrap;">\n'
    panels += _panel(pnl_label, pnl_val, large=True, green=pnl_green)
    panels += _panel("Record (OLS)", _fmt_record(ols))
    panels += _panel("Record (XGB)", _fmt_record(xgb))
    panels += _panel("Record (Both)", _fmt_record(both))
    panels += _panel("Win % (OLS)", _fmt_pct(ols.get("hit")))
    panels += _panel("Win % (XGB)", _fmt_pct(xgb.get("hit")))
    panels += _panel("Win % (Both)", _fmt_pct(both.get("hit")))
    roi_ols = ols.get("roi")
    roi_xgb = xgb.get("roi")
    roi_both = both.get("roi")
    panels += _panel("ROI (OLS)", _fmt_roi(roi_ols), green=roi_ols is not None and roi_ols >= 0)
    panels += _panel("ROI (XGB)", _fmt_roi(roi_xgb), green=roi_xgb is not None and roi_xgb >= 0)
    panels += _panel("ROI (Both)", _fmt_roi(roi_both), green=roi_both is not None and roi_both >= 0)
    panels += "</div>\n"
    return panels


def _th(text: str, bg: str, *, rowspan: int = 1, colspan: int = 1, align: str = "center") -> str:
    rs = f' rowspan="{rowspan}"' if rowspan > 1 else ""
    cs = f' colspan="{colspan}"' if colspan > 1 else ""
    return f'<th{rs}{cs} style="{_TH}background:{bg};text-align:{align};">{text}</th>\n'


def _build_thead() -> str:
    r1 = "    "
    r1 += _th("Game Info", _G_PLAYER, colspan=4)
    r1 += _th("Market (per book)", _G_MARKET, colspan=6)
    r1 += _th("Model — OLS", _G_OLS, colspan=3)
    r1 += _th("Model — XGBoost", _G_XGB, colspan=3)
    r1 += _th("Edge — OLS", _G_EDGE_O, colspan=2)
    r1 += _th("Edge — XGBoost", _G_EDGE_X, colspan=2)
    r1 += _th("Best<br>Under<br>Edge", _G_BEST, rowspan=2)
    r1 += _th("Production Features (20260403)", _G_FEAT, colspan=6)
    r1 += _th("Result", _G_RESULT)

    r2 = "    "
    # Game Info
    r2 += _th("Player", _G_PLAYER, align="left")
    r2 += _th("Team", _G_PLAYER)
    r2 += _th("Opponent", _G_PLAYER)
    r2 += _th("Direction", _G_PLAYER)
    # Market
    r2 += _th("Line", _G_MARKET)
    r2 += _th("Book", _G_MARKET)
    r2 += _th("Over Odds", _G_MARKET)
    r2 += _th("Under Odds", _G_MARKET)
    r2 += _th("Market Over %<br>(no-vig)", _G_MARKET)
    r2 += _th("Market Under %<br>(no-vig)", _G_MARKET)
    # OLS
    r2 += _th("Projected<br>Rebounds", _G_OLS)
    r2 += _th("Model<br>Over %", _G_OLS)
    r2 += _th("Model<br>Under %", _G_OLS)
    # XGB
    r2 += _th("Projected<br>Rebounds", _G_XGB)
    r2 += _th("Model<br>Over %", _G_XGB)
    r2 += _th("Model<br>Under %", _G_XGB)
    # Edge OLS
    r2 += _th("Over<br>Edge", _G_EDGE_O)
    r2 += _th("Under<br>Edge", _G_EDGE_O)
    # Edge XGB
    r2 += _th("Over<br>Edge", _G_EDGE_X)
    r2 += _th("Under<br>Edge", _G_EDGE_X)
    # Features
    for feat in B_MIN_MAX_FEATS:
        r2 += _th(feat, _G_FEAT)
    # Result
    r2 += _th("Actual", _G_RESULT)

    return f"<thead><tr>\n{r1}</tr><tr>\n{r2}</tr></thead>\n"


def _edge_color(v: float) -> str:
    if v > 0:
        return "#16a34a"
    if v < 0:
        return "#dc2626"
    return "#111"


def _td(content: str, color: str = "#111", bold: bool = False, align: str = "center") -> str:
    fw = "font-weight:700;" if bold else ""
    return (
        f'<td style="padding:5px 8px;font-size:11px;text-align:{align};'
        f'color:{color};{fw}white-space:nowrap;">{content}</td>\n'
    )


def _build_data_row(row: pd.Series, min_edge: float) -> str:
    best = float(row["best_under_edge"])
    if best >= min_edge:
        bg = "#dcfce7"
        border = "border-left:3px solid #16a34a;"
    elif best >= min_edge * 0.6:  # near-miss: 3pp when min_edge=5pp
        bg = "#fefce8"
        border = "border-left:3px solid #ca8a04;"
    else:
        bg = "#ffffff"
        border = ""

    ols_fired = bool(row.get("play_under_ols", False))
    xgb_fired = bool(row.get("play_under_xgb", False))

    def _pct(v: float) -> str:
        return f"{v * 100:.1f}%"

    def _pp(v: float, bold: bool = False) -> str:
        sign = "+" if v > 0 else ""
        txt = f"{sign}{v * 100:.1f}pp"
        return f"<b>{txt}</b>" if bold else txt

    over_odds = int(row["over_odds"]) if not pd.isna(row["over_odds"]) else "NA"
    under_odds = int(row["under_odds"]) if not pd.isna(row["under_odds"]) else "NA"
    over_odds_str = f"+{over_odds}" if isinstance(over_odds, int) and over_odds > 0 else str(over_odds)
    under_odds_str = f"+{under_odds}" if isinstance(under_odds, int) and under_odds > 0 else str(under_odds)

    actual = str(row["REB"]) if "REB" in row.index and not pd.isna(row.get("REB")) else "—"

    cells = f'<tr style="background:{bg};{border}">\n'
    cells += _td(str(row["player_normalized"]), bold=True, align="left")
    cells += _td(str(row["team"]))
    cells += _td(str(row["opponent"]))
    cells += _td("UNDER", color="#1d4ed8", bold=True)
    cells += _td(str(row["line"]))
    cells += _td(str(row["bookmaker"]))
    cells += _td(over_odds_str)
    cells += _td(under_odds_str)
    cells += _td(_pct(row["novig_prob_over"]))
    cells += _td(_pct(row["novig_prob_under"]))
    # OLS
    cells += _td(f"{row['yhat_ols']:.2f}", color="#4338ca")
    cells += _td(_pct(row["p_over_ols"]), color="#4338ca")
    cells += _td(_pct(row["p_under_ols"]), color="#4338ca")
    # XGB
    cells += _td(f"{row['yhat_xgb']:.2f}", color="#1d4ed8")
    cells += _td(_pct(row["p_over_xgb"]), color="#1d4ed8")
    cells += _td(_pct(row["p_under_xgb"]), color="#1d4ed8")
    # Edge OLS
    eo_ols = float(row["edge_over_ols"])
    eu_ols = float(row["edge_under_ols"])
    cells += _td(_pp(eo_ols), color=_edge_color(eo_ols))
    cells += _td(_pp(eu_ols, bold=ols_fired), color=_edge_color(eu_ols))
    # Edge XGB
    eo_xgb = float(row["edge_over_xgb"])
    eu_xgb = float(row["edge_under_xgb"])
    cells += _td(_pp(eo_xgb), color=_edge_color(eo_xgb))
    cells += _td(_pp(eu_xgb, bold=xgb_fired), color=_edge_color(eu_xgb))
    # Best under edge
    cells += _td(_pp(best, bold=ols_fired or xgb_fired), color=_edge_color(best))
    # Features
    for feat in B_MIN_MAX_FEATS:
        v = row.get(feat)
        cells += _td("NA" if pd.isna(v) else f"{float(v):.2f}")
    # Actual
    cells += _td(actual, bold=True)
    cells += "</tr>\n"
    return cells


def _game_label(game_rows: pd.DataFrame) -> str:
    home = str(game_rows["home_team_norm"].iloc[0]) if "home_team_norm" in game_rows.columns else ""
    away = str(game_rows["away_team_norm"].iloc[0]) if "away_team_norm" in game_rows.columns else ""
    game_time = ""
    if "game_time" in game_rows.columns:
        gt = game_rows["game_time"].iloc[0]
        if not pd.isna(gt):
            game_time = f"{gt} ET · "

    teams_str = f"{away} vs {home}" if home and away else str(game_rows["game_id"].iloc[0])

    qualifying = game_rows[game_rows["best_under_edge"] >= 0.05]
    if len(qualifying) == 0:
        qual_str = "no qualifying rows"
    else:
        summaries = []
        for _, qrow in qualifying.sort_values("best_under_edge", ascending=False).iterrows():
            ols_fired = bool(qrow.get("play_under_ols", False))
            xgb_fired = bool(qrow.get("play_under_xgb", False))
            model = "OLS" if ols_fired and not xgb_fired else ("XGBoost" if xgb_fired and not ols_fired else "OLS+XGB")
            best = float(qrow["best_under_edge"])
            sign = "+" if best >= 0 else ""
            summaries.append(f"{qrow['player_normalized']} ({qrow['bookmaker']} {sign}{best * 100:.1f}pp {model})")
        qual_str = f"{len(qualifying)} qualifying — {' · '.join(summaries)}"

    return f"{game_time}{teams_str} · {qual_str}"


def build_html_email(df: pd.DataFrame, which: str, records_csv: str, slate_date: str = "") -> str:
    df = _compute_display_cols(df)

    records = _load_records(records_csv)
    pnl_ols = records.get("ols", {}).get("pnl")

    min_edge = float(df["score_min_edge_used"].iloc[0]) if "score_min_edge_used" in df.columns else 0.05

    if not slate_date and "date" in df.columns:
        slate_date = str(pd.to_datetime(df["date"]).dt.date.max())

    import datetime as _dt
    now_et = _dt.datetime.now(_dt.timezone.utc).strftime("%I:%M %p ET").lstrip("0")

    # ── header ────────────────────────────────────────────────────────────────
    html = (
        '<div style="border:2px solid #d1d5db;border-radius:8px;padding:0;'
        'margin:16px 0;font-family:Arial,sans-serif;background:#fff;max-width:100%;">\n\n'
        '<div style="background:#1d4ed8;color:#fff;padding:14px 20px;border-radius:6px 6px 0 0;">\n'
        '  <div style="font-size:18px;font-weight:700;">\U0001f3c0 NBA Rebounds Model</div>\n'
        f'  <div style="font-size:13px;margin-top:4px;color:#bfdbfe;">'
        f'{slate_date} &nbsp;·&nbsp; Generated {now_et} &nbsp;·&nbsp; daily slate</div>\n'
        '</div>\n\n'
    )

    # ── stats panels ──────────────────────────────────────────────────────────
    html += _build_stats_panels(records, pnl_ols)

    # ── legend ────────────────────────────────────────────────────────────────
    edge_pct = int(min_edge * 100)
    near_pct = int(min_edge * 60)
    html += (
        '<div style="margin:0 16px 12px;font-size:10px;color:#6b7280;">\n'
        f'  <span style="background:#dcfce7;padding:2px 6px;border-radius:3px;margin-right:8px;">'
        f'green row = qualifying play (best under edge ≥ {edge_pct}pp)</span>\n'
        f'  <span style="background:#fefce8;padding:2px 6px;border-radius:3px;margin-right:8px;">'
        f'yellow row = near-miss ({near_pct}–{edge_pct}pp, context only)</span>\n'
        '  &middot; <b>Bold Under Edge</b> = model that fired'
        ' &middot; green/red on edge cells = positive/negative\n'
        '</div>\n\n'
    )

    # ── per-game tables ───────────────────────────────────────────────────────
    thead = _build_thead()
    TABLE_STYLE = (
        'width:100%;border-collapse:collapse;font-size:11px;'
        'border-top:1px solid #e5e7eb;'
    )

    for game_id, game_rows in df.groupby("game_id", sort=False):
        game_rows = game_rows.sort_values("best_under_edge", ascending=False)

        label = _game_label(game_rows)
        qual_count = int((game_rows["best_under_edge"] >= min_edge).sum())
        trophy = "\U0001f3c6 " if qual_count > 0 else ""
        qual_suffix = f" · {trophy}{qual_count} qualifying" if qual_count > 0 else " · no qualifying rows"

        html += (
            f'<div style="background:#111827;color:#fff;padding:10px 16px;'
            f'font-size:12px;font-weight:600;">{label}</div>\n'
        )
        html += f'<div style="overflow-x:auto;">\n<table style="{TABLE_STYLE}">\n'
        html += thead
        html += "<tbody>\n"
        for _, row in game_rows.iterrows():
            html += _build_data_row(row, min_edge)
        html += "</tbody>\n</table>\n</div>\n\n"

    html += "</div>\n"
    return html


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    df = read_parquet_any(args.scored)

    if args.format == "html":
        body = build_html_email(df, args.which, args.records_csv)
    else:
        plays = build_plays_table(df, args.which)
        body = build_text_body(plays, args.which)

    topic = args.topic_arn.strip() or os.environ.get("SNS_TOPIC_ARN", "").strip()
    if not topic:
        print(body)
        return

    import boto3

    n_qual = int((df["play_under_ols"] | df["play_under_xgb"]).sum()) if args.format == "html" else len(df)
    subject = f"{args.subject} | {n_qual} plays"[:100]
    resp = boto3.client("sns").publish(
        TopicArn=topic,
        Subject=subject,
        Message=body[:256_000],
        MessageAttributes={"content-type": {"DataType": "String", "StringValue": "text/html"}}
        if args.format == "html"
        else {},
    )
    print(
        "published_to_sns",
        f"topic_arn={topic}",
        f"format={args.format}",
        f"n_qualifying={n_qual}",
        f"message_id={resp['MessageId']}",
        sep=" | ",
    )


if __name__ == "__main__":
    main()

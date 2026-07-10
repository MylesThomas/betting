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

import yaml  # noqa: E402

from src.io_utils import read_parquet_any  # noqa: E402
from src.nba_rebounds_modeling.rebounds_feature_spec import B_MIN_MAX_FEATS  # noqa: E402

def _load_superstar_set() -> set[str]:
    cfg_path = Path(__file__).resolve().parents[2] / "config" / "model_config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return set(cfg.get("llm_features", {}).get("superstar_players", []))

_SUPERSTAR_PLAYERS: set[str] = _load_superstar_set()

# ── colour palette matching the mock email ───────────────────────────────────
_G_PLAYER  = "#334155"   # Player / Game
_G_BOOK    = "#0f766e"   # Book
_G_ODDS    = "#1e40af"   # American Odds
_G_IMPLIED = "#6d28d9"   # Implied (raw probs)
_G_NOVIG   = "#047857"   # No-Vig (fair probs)
_G_OLS     = "#4338ca"   # Model — OLS
_G_XGB     = "#1d4ed8"   # Model — XGBoost
_G_EDGE_O  = "#065f46"   # Edge — OLS
_G_EDGE_X  = "#14532d"   # Edge — XGBoost
_G_FEAT    = "#374151"   # Model Inputs / Status

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
    p.add_argument(
        "--yesterday-rollup",
        type=str,
        default="",
        help="Path or s3:// URI to yesterday settlement rollup CSV for yesterday's results section.",
    )
    p.add_argument(
        "--output-html",
        type=str,
        default="",
        help="Write HTML body to this local path instead of sending via SNS.",
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
    raw_total = p_raw_o + p_raw_u

    # Raw implied (vig-inclusive)
    out["raw_prob_over"]  = p_raw_o
    out["raw_prob_under"] = p_raw_u
    out["raw_total"]      = raw_total

    # No-vig (proportional de-vig)
    out["fair_over"]  = p_raw_o / raw_total
    out["fair_under"] = p_raw_u / raw_total
    out["fair_total"] = 1.0                        # always 100% by construction
    out["vig"]        = raw_total - 1.0

    # Keep novig aliases so _game_label still works
    out["novig_prob_over"]  = out["fair_over"]
    out["novig_prob_under"] = out["fair_under"]

    out["p_over_ols"]  = 1.0 - out["p_under_ols"]
    out["p_over_xgb"]  = 1.0 - out["p_under_xgb"]
    out["edge_over_ols"]  = out["p_over_ols"]  - p_raw_o
    out["edge_over_xgb"]  = out["p_over_xgb"]  - p_raw_o
    out["delta_ols"] = out["yhat_ols"] - out["line"]
    out["delta_xgb"] = out["yhat_xgb"] - out["line"]
    out["best_under_edge"] = out[["edge_under_ols", "edge_under_xgb"]].max(axis=1)

    # Status replaces the old Direction column
    def _status(row: pd.Series) -> str:
        if bool(row.get("play_under_ols", False)) or bool(row.get("play_under_xgb", False)):
            return "PLAY - UNDER"
        return ""

    out["status"] = out.apply(_status, axis=1)

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

    # Game time for the per-row Time (ET) column
    if "game_time" in out.columns:
        out["game_time_et"] = out["game_time"].fillna("").astype(str).str.strip()
    else:
        out["game_time_et"] = ""

    return out


def _read_csv_any(path: str) -> pd.DataFrame | None:
    """Read a CSV from a local path or s3:// URI. Returns None on any error."""
    if not path.strip():
        return None
    try:
        if path.startswith("s3://"):
            import boto3
            from io import BytesIO
            bucket, _, key = path[5:].partition("/")
            body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
            return pd.read_csv(BytesIO(body))
        return pd.read_csv(path)
    except Exception:
        return None


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
        rec = _read_csv_any(records_csv)
        if rec is None:
            return {}
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

    def _star_pnl(bucket: str) -> float | None:
        sub = rec.loc[rec["strategy_bucket"] == bucket]
        if len(sub) == 0:
            return None
        return float(sub["pnl_units"].sum())

    return {
        "ols":          _agg(["both", "ols"]),
        "xgb":          _agg(["both", "xgb"]),
        "both":         _agg(["both"]),
        "star_pnl":     _star_pnl("star_split_superstar"),
        "non_star_pnl": _star_pnl("star_split_non_superstar"),
    }


def _build_yesterday_section(yesterday_rollup_csv: str) -> str:
    """Build an HTML 'Yesterday's Results' section from a settlement rollup CSV."""
    rec = _read_csv_any(yesterday_rollup_csv)
    if rec is None or len(rec) == 0:
        return (
            '<div style="margin:16px 16px 0;padding:12px 16px;background:#f9fafb;'
            'border:1px solid #e5e7eb;border-radius:6px;">\n'
            '<div style="font-size:13px;font-weight:600;margin-bottom:8px;">Yesterday\'s Results</div>\n'
            '<div style="font-size:12px;color:#6b7280;">No settled results available.</div>\n'
            '</div>\n'
        )

    agg = (
        rec.groupby("strategy_bucket", as_index=False)
        .agg(
            n_bets=("n_bets", "sum"),
            n_win=("n_win", "sum"),
            n_loss=("n_loss", "sum"),
            n_push=("n_push", "sum"),
            n_unsettled=("n_unsettled", "sum"),
            pnl_units=("pnl_units", "sum"),
        )
        .sort_values("strategy_bucket")
        .reset_index(drop=True)
    )

    th_style = "padding:5px 10px;font-size:10px;text-align:center;background:#1e293b;color:#fff;"
    td_style = "padding:5px 10px;font-size:11px;text-align:center;border-bottom:1px solid #f1f5f9;"

    rows_html = ""
    for _, row in agg.iterrows():
        bucket = str(row["strategy_bucket"]).upper()
        wlp = f"{int(row['n_win'])}-{int(row['n_loss'])}-{int(row['n_push'])}"
        n_settled = int(row["n_win"]) + int(row["n_loss"]) + int(row["n_push"])
        pnl = float(row["pnl_units"])
        pnl_str = f"+{pnl:.2f}u" if pnl >= 0 else f"{pnl:.2f}u"
        pnl_color = "#16a34a" if pnl >= 0 else "#dc2626"
        hit = int(row["n_win"]) / n_settled if n_settled > 0 else float("nan")
        roi = pnl / n_settled if n_settled > 0 else float("nan")
        hit_str = f"{hit * 100:.1f}%" if not np.isnan(hit) else "—"
        roi_str = (f"+{roi * 100:.1f}%" if roi >= 0 else f"{roi * 100:.1f}%") if not np.isnan(roi) else "—"
        roi_color = "#16a34a" if (not np.isnan(roi) and roi >= 0) else "#dc2626"
        rows_html += (
            f'<tr>'
            f'<td style="{td_style}font-weight:600;">{bucket}</td>'
            f'<td style="{td_style}">{int(row["n_bets"])}</td>'
            f'<td style="{td_style}">{wlp}</td>'
            f'<td style="{td_style};color:{pnl_color};font-weight:600;">{pnl_str}</td>'
            f'<td style="{td_style}">{hit_str}</td>'
            f'<td style="{td_style};color:{roi_color};font-weight:600;">{roi_str}</td>'
            f'</tr>\n'
        )

    return (
        '<div style="margin:16px 16px 0;padding:12px 16px;background:#f9fafb;'
        'border:1px solid #e5e7eb;border-radius:6px;">\n'
        '<div style="font-size:13px;font-weight:600;margin-bottom:10px;">Yesterday\'s Results</div>\n'
        f'<table style="border-collapse:collapse;width:auto;">\n'
        f'<thead><tr>'
        f'<th style="{th_style}text-align:left;">Strategy</th>'
        f'<th style="{th_style}">Bets</th>'
        f'<th style="{th_style}">W-L-P</th>'
        f'<th style="{th_style}">PnL</th>'
        f'<th style="{th_style}">Hit Rate</th>'
        f'<th style="{th_style}">ROI</th>'
        f'</tr></thead>\n'
        f'<tbody>{rows_html}</tbody>\n'
        f'</table>\n'
        '</div>\n'
    )


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


def _fmt_pnl(val: float | None) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.1f}u"


def _build_stats_panels(records: dict) -> str:
    ols = records.get("ols", {})
    xgb = records.get("xgb", {})
    both = records.get("both", {})
    star_pnl = records.get("star_pnl")
    non_star_pnl = records.get("non_star_pnl")

    panels = '<div style="display:flex;gap:12px;margin:14px 16px;flex-wrap:wrap;">\n'
    panels += _panel("Record (OLS)",  _fmt_record(ols))
    panels += _panel("Record (XGB)", _fmt_record(xgb))
    panels += _panel("Record (Both)", _fmt_record(both))
    panels += _panel("Win % (OLS)", _fmt_pct(ols.get("hit")))
    panels += _panel("Win % (XGB)", _fmt_pct(xgb.get("hit")))
    panels += _panel("Win % (Both)", _fmt_pct(both.get("hit")))
    pnl_xgb  = xgb.get("pnl")
    pnl_both = both.get("pnl")
    panels += _panel("Units (OLS)",   _fmt_pnl(pnl_ols),      green=pnl_ols  is not None and pnl_ols  >= 0)
    panels += _panel("Units (XGB)",   _fmt_pnl(pnl_xgb),      green=pnl_xgb  is not None and pnl_xgb  >= 0)
    panels += _panel("Units (Both)",  _fmt_pnl(pnl_both),     green=pnl_both is not None and pnl_both >= 0)
    panels += "</div>\n"
    panels += '<div style="display:flex;gap:12px;margin:0 16px 14px;flex-wrap:wrap;">\n'
    panels += _panel("Units (★ Star)", _fmt_pnl(star_pnl),    green=star_pnl     is not None and star_pnl     >= 0)
    panels += _panel("Units (Non-★)", _fmt_pnl(non_star_pnl), green=non_star_pnl is not None and non_star_pnl >= 0)
    panels += "</div>\n"
    return panels


def _th(text: str, bg: str, *, rowspan: int = 1, colspan: int = 1, align: str = "center") -> str:
    rs = f' rowspan="{rowspan}"' if rowspan > 1 else ""
    cs = f' colspan="{colspan}"' if colspan > 1 else ""
    return f'<th{rs}{cs} style="{_TH}background:{bg};text-align:{align};">{text}</th>\n'


def _build_thead() -> str:
    # Row 1: group headers
    # 5 + 1 + 2 + 3 + 4 + 3 + 3 + 2 + 2 + (6 features + 1 Status + 1 Actual) = 33 cols
    r1 = "    "
    r1 += _th("Player / Game",        _G_PLAYER,  colspan=6)
    r1 += _th("Book",                 _G_BOOK,    colspan=1)
    r1 += _th("American Odds",        _G_ODDS,    colspan=2)
    r1 += _th("Implied",              _G_IMPLIED, colspan=3)
    r1 += _th("No-Vig",               _G_NOVIG,   colspan=4)
    r1 += _th("Model — OLS",          _G_OLS,     colspan=4)
    r1 += _th("Model — XGBoost",      _G_XGB,     colspan=4)
    r1 += _th("Edge — OLS",           _G_EDGE_O,  colspan=2)
    r1 += _th("Edge — XGBoost",       _G_EDGE_X,  colspan=2)
    r1 += _th("Model Inputs (20260403)", _G_FEAT, colspan=len(B_MIN_MAX_FEATS) + 2)

    # Row 2: individual column names
    r2 = "    "
    # Player / Game (6)
    r2 += _th("Player",     _G_PLAYER, align="left")
    r2 += _th("★",          _G_PLAYER)
    r2 += _th("Team",       _G_PLAYER)
    r2 += _th("Opp",        _G_PLAYER)
    r2 += _th("Time (ET)",  _G_PLAYER)
    r2 += _th("Line",       _G_PLAYER)
    # Book (1)
    r2 += _th("Book",       _G_BOOK)
    # American Odds (2)
    r2 += _th("Over",       _G_ODDS)
    r2 += _th("Under",      _G_ODDS)
    # Implied (3)
    r2 += _th("Raw Over",   _G_IMPLIED)
    r2 += _th("Raw Under",  _G_IMPLIED)
    r2 += _th("Raw Total",  _G_IMPLIED)
    # No-Vig (4)
    r2 += _th("Fair Over",  _G_NOVIG)
    r2 += _th("Fair Under", _G_NOVIG)
    r2 += _th("Fair Total", _G_NOVIG)
    r2 += _th("Vig",        _G_NOVIG)
    # Model OLS (4)
    r2 += _th("Prediction<br>(yhat)", _G_OLS)
    r2 += _th("Delta<br>(yhat−line)", _G_OLS)
    r2 += _th("Pred Over",            _G_OLS)
    r2 += _th("Pred Under",           _G_OLS)
    # Model XGB (4)
    r2 += _th("Prediction<br>(yhat)", _G_XGB)
    r2 += _th("Delta<br>(yhat−line)", _G_XGB)
    r2 += _th("Pred Over",            _G_XGB)
    r2 += _th("Pred Under",           _G_XGB)
    # Edge OLS (2)
    r2 += _th("Over Edge",  _G_EDGE_O)
    r2 += _th("Under Edge", _G_EDGE_O)
    # Edge XGB (2)
    r2 += _th("Over Edge",  _G_EDGE_X)
    r2 += _th("Under Edge", _G_EDGE_X)
    # Model Inputs: features + Status + Actual
    for feat in B_MIN_MAX_FEATS:
        r2 += _th(feat, _G_FEAT)
    r2 += _th("Status", _G_FEAT)
    r2 += _th("Actual", _G_FEAT)

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
    ols_fired = bool(row.get("play_under_ols", False))
    xgb_fired = bool(row.get("play_under_xgb", False))
    is_play = ols_fired or xgb_fired

    if best >= min_edge:
        bg = "#dcfce7"
        border = "border-left:3px solid #16a34a;"
    elif best >= min_edge * 0.6:
        bg = "#fefce8"
        border = "border-left:3px solid #ca8a04;"
    else:
        bg = "#ffffff"
        border = ""

    def _pct(v: float) -> str:
        return f"{v * 100:.1f}%"

    def _pp(v: float, bold: bool = False) -> str:
        sign = "+" if v > 0 else ""
        txt = f"{sign}{v * 100:.1f}pp"
        return f"<b>{txt}</b>" if bold else txt

    over_odds_raw = int(row["over_odds"]) if not pd.isna(row["over_odds"]) else "NA"
    under_odds_raw = int(row["under_odds"]) if not pd.isna(row["under_odds"]) else "NA"
    over_odds_str  = f"+{over_odds_raw}" if isinstance(over_odds_raw, int) and over_odds_raw > 0 else str(over_odds_raw)
    under_odds_str = f"+{under_odds_raw}" if isinstance(under_odds_raw, int) and under_odds_raw > 0 else str(under_odds_raw)

    actual = str(int(row["REB"])) if "REB" in row.index and not pd.isna(row.get("REB")) else "—"
    status = str(row.get("status", ""))
    game_time = str(row.get("game_time_et", "")) or "—"

    is_superstar = str(row["player_normalized"]) in _SUPERSTAR_PLAYERS
    star_cell = (
        f'<td style="padding:5px 8px;font-size:11px;text-align:center;'
        f'background:#fef9c3;color:#92400e;font-weight:700;white-space:nowrap;">★</td>\n'
        if is_superstar else
        f'<td style="padding:5px 8px;font-size:11px;text-align:center;white-space:nowrap;">—</td>\n'
    )

    cells = f'<tr style="background:{bg};{border}">\n'
    # Player / Game (6)
    cells += _td(str(row["player_normalized"]), bold=True, align="left")
    cells += star_cell
    cells += _td(str(row["team"]))
    cells += _td(str(row["opponent"]))
    cells += _td(game_time)
    cells += _td(str(row["line"]))
    # Book (1)
    cells += _td(str(row["bookmaker"]))
    # American Odds (2)
    cells += _td(over_odds_str)
    cells += _td(under_odds_str)
    # Implied (3)
    cells += _td(_pct(row["raw_prob_over"]),  color="#6d28d9")
    cells += _td(_pct(row["raw_prob_under"]), color="#6d28d9")
    cells += _td(_pct(row["raw_total"]),      color="#6d28d9")
    # No-Vig (4)
    cells += _td(_pct(row["fair_over"]),  color="#047857")
    cells += _td(_pct(row["fair_under"]), color="#047857")
    cells += _td("100.0%",                color="#047857")
    cells += _td(_pct(row["vig"]),        color="#6b7280")
    # Model OLS (4)
    d_ols = float(row["delta_ols"])
    d_ols_str = f"+{d_ols:.2f}" if d_ols >= 0 else f"{d_ols:.2f}"
    cells += _td(f"{row['yhat_ols']:.2f}", color="#4338ca")
    cells += _td(d_ols_str, color=_edge_color(d_ols), bold=False)
    cells += _td(_pct(row["p_over_ols"]),  color="#4338ca")
    cells += _td(_pct(row["p_under_ols"]), color="#4338ca")
    # Model XGB (4)
    d_xgb = float(row["delta_xgb"])
    d_xgb_str = f"+{d_xgb:.2f}" if d_xgb >= 0 else f"{d_xgb:.2f}"
    cells += _td(f"{row['yhat_xgb']:.2f}", color="#1d4ed8")
    cells += _td(d_xgb_str, color=_edge_color(d_xgb), bold=False)
    cells += _td(_pct(row["p_over_xgb"]),  color="#1d4ed8")
    cells += _td(_pct(row["p_under_xgb"]), color="#1d4ed8")
    # Edge OLS (2)
    eo_ols = float(row["edge_over_ols"])
    eu_ols = float(row["edge_under_ols"])
    cells += _td(_pp(eo_ols),              color=_edge_color(eo_ols))
    cells += _td(_pp(eu_ols, bold=ols_fired), color=_edge_color(eu_ols))
    # Edge XGB (2)
    eo_xgb = float(row["edge_over_xgb"])
    eu_xgb = float(row["edge_under_xgb"])
    cells += _td(_pp(eo_xgb),              color=_edge_color(eo_xgb))
    cells += _td(_pp(eu_xgb, bold=xgb_fired), color=_edge_color(eu_xgb))
    # Model Inputs: features
    for feat in B_MIN_MAX_FEATS:
        v = row.get(feat)
        cells += _td("NA" if pd.isna(v) else f"{float(v):.2f}")
    # Status
    cells += _td(
        status,
        color="#16a34a" if is_play else "#6b7280",
        bold=is_play,
    )
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


def build_html_email(
    df: pd.DataFrame,
    which: str,
    records_csv: str,
    slate_date: str = "",
    yesterday_rollup_csv: str = "",
) -> str:
    df = _compute_display_cols(df)

    records = _load_records(records_csv)

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
    html += _build_stats_panels(records)

    # ── legend ────────────────────────────────────────────────────────────────
    edge_pct = int(min_edge * 100)
    near_pct = int(min_edge * 60)
    html += (
        '<div style="margin:0 16px 12px;font-size:10px;color:#6b7280;">\n'
        f'  <span style="background:#dcfce7;padding:2px 6px;border-radius:3px;margin-right:8px;">'
        f'green row = qualifying play (best under edge ≥ {edge_pct}pp)</span>\n'
        f'  <span style="background:#fefce8;padding:2px 6px;border-radius:3px;margin-right:8px;">'
        f'yellow row = near-miss ({near_pct}–{edge_pct}pp, context only)</span>\n'
        '  &middot; <b>Bold edge cell</b> = model that fired'
        ' &middot; green/red on edge cells = positive/negative\n'
        f'  &middot; <span style="background:#fef9c3;padding:2px 6px;border-radius:3px;">★ superstar</span>'
        ' = low-edge tier historically (+0.28% ROI / 3 seasons); play at your discretion\n'
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

    if yesterday_rollup_csv.strip():
        html += _build_yesterday_section(yesterday_rollup_csv)
        html += "\n"

    html += "</div>\n"
    return html


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    df = read_parquet_any(args.scored)

    if args.format == "html":
        body = build_html_email(
            df,
            args.which,
            args.records_csv,
            yesterday_rollup_csv=args.yesterday_rollup,
        )
    else:
        plays = build_plays_table(df, args.which)
        body = build_text_body(plays, args.which)

    if args.output_html.strip():
        Path(args.output_html).write_text(body, encoding="utf-8")
        print(f"html_written | path={args.output_html}")
        return

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

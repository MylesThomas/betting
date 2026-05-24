"""
NBA Rebounds Strategy Dashboard

Tracks the production rebounds model's full betting record: P&L, hit rates, individual plays,
and model health. Data sourced from settled parquet artifacts in S3.

Three tabs:
  - Played Bets: rows where model placed a bet (strategy_bucket in both/ols/xgb)
  - Blind Unders Benchmark: rows the model skipped (strategy_bucket = neither),
    showing hypothetical P&L if we had bet every under regardless of edge.
  - Backtest: multi-season OOS walk-forward results (A3 spec, min_edge=0.05)
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

import matplotlib.colors as mc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from rebounds_data import (
    PLAYED_BUCKETS,
    PROD_GO_LIVE_DATE,
    load_run_manifests,
    load_settled_plays,
    load_todays_scored,
)
from rebounds_backtest_data import (
    BUCKETS as BT_BUCKETS,
    PROD_GO_LIVE_DATE as BT_PROD_GO_LIVE_DATE,
    compute_kpis as bt_compute_kpis,
    filter_by_buckets,
    load_backtest_multi,
    per_season_summary,
    settled_rows as bt_settled_rows,
)

# ── Sidebar filter container ───────────────────────────────────────────────────

CONSENSUS_FILTER_OPTIONS: list[str] = ["All", "Consensus line only", "Non-consensus only"]


class SidebarFilters(NamedTuple):
    start_date: date
    end_date: date
    selected_buckets: list[str]
    selected_bookmakers: list[str]
    consensus_filter: str


# ── Constants ──────────────────────────────────────────────────────────────────

NBA_SEASON_2025_START: date = date(2025, 10, 21)

BUCKET_COLORS: dict[str, str] = {
    "both": "#17408B",
    "ols": "#ff7f0e",
    "xgb": "#2ca02c",
    "neither": "#d62728",
}

RESULT_BG_COLORS: dict[str, str] = {
    "win": "#dcfce7",
    "loss": "#fee2e2",
    "push": "#fef9c3",
    "unsettled": "#f1f5f9",
}

PLAYS_DISPLAY_COLS: list[str] = [
    "player_normalized",
    "strategy_bucket",
    "bookmaker",
    "line",
    "reb_actual",
    "diff",
    "result",
    "under_odds",
    "p_under_ols",
    "p_under_xgb",
    "edge_under_ols",
    "edge_under_xgb",
    "date",
]

PLAYS_COL_LABELS: dict[str, str] = {
    "player_normalized": "Player",
    "strategy_bucket": "Strategy",
    "bookmaker": "Bookmaker",
    "line": "Line",
    "reb_actual": "Actual",
    "diff": "Diff",
    "result": "Result",
    "under_odds": "Under Odds",
    "p_under_ols": "P(Under) OLS",
    "p_under_xgb": "P(Under) XGB",
    "edge_under_ols": "Edge (OLS)",
    "edge_under_xgb": "Edge (XGB)",
    "date": "Date",
}

TODAY_SCORED_DISPLAY_COLS: list[str] = [
    "player_normalized",
    "strategy_bucket",
    "bookmaker",
    "line",
    "under_odds",
    "p_under_ols",
    "p_under_xgb",
    "edge_under_ols",
    "edge_under_xgb",
]

TODAY_SCORED_COL_LABELS: dict[str, str] = {
    "player_normalized": "Player",
    "strategy_bucket": "Strategy",
    "bookmaker": "Bookmaker",
    "line": "Line",
    "under_odds": "Under Odds",
    "p_under_ols": "P(Under) OLS",
    "p_under_xgb": "P(Under) XGB",
    "edge_under_ols": "Edge (OLS)",
    "edge_under_xgb": "Edge (XGB)",
}

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NBA Rebounds Strategy",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data transforms ────────────────────────────────────────────────────────────


_EDGE_CMAP = mc.LinearSegmentedColormap.from_list("white_green", ["white", "#16a34a"])


def american_profit_on_win(american_odds: pd.Series) -> pd.Series:
    """Vectorized: American odds → profit per unit wagered on a win."""
    return np.where(american_odds >= 100, american_odds / 100.0, 100.0 / american_odds.abs())


def american_to_implied_prob(odds: pd.Series) -> pd.Series:
    """Vectorized: American odds → raw implied probability (includes vig)."""
    abs_odds = odds.abs()
    return pd.Series(
        np.where(odds >= 0, 100.0 / (odds + 100.0), abs_odds / (abs_odds + 100.0)),
        index=odds.index,
    ).where(odds.notna())


def add_hypothetical_pnl(neither_plays: pd.DataFrame) -> pd.DataFrame:
    """
    Compute what the P&L would have been if we had bet every 'neither' row.
    Adds a `hyp_pnl` column using the same formula as the production settle script.
    """
    work: pd.DataFrame = neither_plays.copy()
    work["hyp_pnl"] = 0.0
    win_mask: pd.Series = work["result"] == "win"
    loss_mask: pd.Series = work["result"] == "loss"
    work.loc[win_mask, "hyp_pnl"] = american_profit_on_win(work.loc[win_mask, "under_odds"])
    work.loc[loss_mask, "hyp_pnl"] = -1.0
    return work


def derive_strategy_bucket(scored: pd.DataFrame) -> pd.DataFrame:
    """Add strategy_bucket column to an unsettled scored parquet (which has only play flags)."""
    work: pd.DataFrame = scored.copy()
    work["strategy_bucket"] = np.where(
        work["play_both"], "both",
        np.where(work["play_ols_only"], "ols",
                 np.where(work["play_xgb_only"], "xgb", "neither")),
    )
    return work


def add_is_consensus_line(plays: pd.DataFrame) -> pd.DataFrame:
    modal_lines: pd.Series = (
        plays.groupby(["date", "player_normalized"])["line"]
        .transform(lambda lines: lines.mode().iloc[0])
    )
    result: pd.DataFrame = plays.copy()
    result["is_consensus_line"] = result["line"] == modal_lines
    return result


def apply_date_filter(
    plays: pd.DataFrame, start: date, end: date
) -> pd.DataFrame:
    mask: pd.Series = (
        (plays["date"].dt.date >= start) & (plays["date"].dt.date <= end)
    )
    return plays.loc[mask].reset_index(drop=True)


def apply_custom_filters(
    plays: pd.DataFrame,
    selected_bookmakers: list[str],
    consensus_filter: str,
) -> pd.DataFrame:
    filtered: pd.DataFrame = plays.loc[
        plays["bookmaker"].isin(selected_bookmakers)
    ].reset_index(drop=True)
    if consensus_filter == "Consensus line only":
        filtered = filtered.loc[filtered["is_consensus_line"]].reset_index(drop=True)
    elif consensus_filter == "Non-consensus only":
        filtered = filtered.loc[~filtered["is_consensus_line"]].reset_index(drop=True)
    return filtered


def settled_bets_only(plays: pd.DataFrame) -> pd.DataFrame:
    """Return rows that are (a) played bets and (b) fully settled (win/loss/push)."""
    return plays.loc[
        plays["is_bet"] & plays["result"].isin({"win", "loss", "push"})
    ].reset_index(drop=True)


def compute_kpis(settled_bets: pd.DataFrame) -> dict[str, float | int]:
    non_push: pd.DataFrame = settled_bets.loc[settled_bets["result"].isin({"win", "loss"})]
    wins: pd.DataFrame = settled_bets.loc[settled_bets["result"] == "win"]
    total_bets: int = len(settled_bets)
    total_pnl: float = float(settled_bets["pnl_units"].sum())
    hit_rate: float = len(wins) / len(non_push) if len(non_push) > 0 else float("nan")
    roi_per_bet: float = total_pnl / total_bets if total_bets > 0 else float("nan")
    return {
        "total_bets": total_bets,
        "total_pnl": total_pnl,
        "hit_rate": hit_rate,
        "roi_per_bet": roi_per_bet,
    }


def compute_neither_kpis(neither_plays: pd.DataFrame) -> dict[str, float | int]:
    settled: pd.DataFrame = neither_plays.loc[neither_plays["result"].isin({"win", "loss", "push"})]
    non_push: pd.DataFrame = settled.loc[settled["result"].isin({"win", "loss"})]
    wins: pd.DataFrame = settled.loc[settled["result"] == "win"]
    total_rows: int = len(settled)
    total_hyp_pnl: float = float(settled["hyp_pnl"].sum())
    hit_rate: float = len(wins) / len(non_push) if len(non_push) > 0 else float("nan")
    hyp_roi: float = total_hyp_pnl / total_rows if total_rows > 0 else float("nan")
    return {
        "total_rows": total_rows,
        "total_hyp_pnl": total_hyp_pnl,
        "hit_rate": hit_rate,
        "hyp_roi": hyp_roi,
    }


# ── Chart builders ─────────────────────────────────────────────────────────────


def build_cumulative_pnl_chart(
    settled_bets: pd.DataFrame, selected_buckets: list[str]
) -> go.Figure:
    """
    Plotly line chart of cumulative P&L over time, one trace per strategy bucket.
    Vertical dashed line marks prod go-live date.
    """
    daily_stats: pd.DataFrame = (
        settled_bets.groupby(["date", "strategy_bucket"])
        .agg(daily_pnl=("pnl_units", "sum"), n_bets=("player_normalized", "count"))
        .reset_index()
        .sort_values("date")
    )

    # All bucket traces share the same zero anchor: one day before the earliest bet.
    # This makes every line start visually at 0 regardless of the selected date window.
    origin_date: pd.Timestamp = daily_stats["date"].min() - pd.Timedelta(days=1)

    fig = go.Figure()
    for bucket in selected_buckets:
        bucket_daily: pd.DataFrame = daily_stats.loc[
            daily_stats["strategy_bucket"] == bucket
        ].copy().sort_values("date")
        if bucket_daily.empty:
            continue
        bucket_daily["cumulative_pnl"] = bucket_daily["daily_pnl"].cumsum()

        # Prepend the zero-origin row so the trace starts at (origin_date, 0).
        origin_row: pd.DataFrame = pd.DataFrame({
            "date": [origin_date],
            "strategy_bucket": [bucket],
            "daily_pnl": [0.0],
            "n_bets": [0],
            "cumulative_pnl": [0.0],
        })
        bucket_daily = pd.concat([origin_row, bucket_daily], ignore_index=True)

        customdata: np.ndarray = np.stack(
            [bucket_daily["daily_pnl"].values, bucket_daily["n_bets"].values], axis=1
        )
        fig.add_trace(
            go.Scatter(
                x=bucket_daily["date"],
                y=bucket_daily["cumulative_pnl"],
                mode="lines+markers",
                name=bucket,
                line={"color": BUCKET_COLORS.get(bucket, "#888"), "width": 2},
                marker={"size": 5},
                customdata=customdata,
                hovertemplate=(
                    "<b>%{x|%b %d, %Y}</b><br>"
                    "Running P&L: <b>%{y:.2f}u</b><br>"
                    "Daily P&L: %{customdata[0]:.2f}u<br>"
                    "Bets: %{customdata[1]}<extra></extra>"
                ),
            )
        )

    if not daily_stats.empty:
        fig.add_vline(x=pd.Timestamp(PROD_GO_LIVE_DATE), line_dash="dash", line_color="#9ca3af")
        fig.add_annotation(
            x=pd.Timestamp(PROD_GO_LIVE_DATE), y=1, yref="paper",
            text="Prod go-live", showarrow=False, xanchor="left",
            font={"size": 11, "color": "#6b7280"},
        )
    fig.add_hline(y=0, line_color="#e5e7eb", line_width=1)
    fig.update_layout(
        title="Cumulative P&L by Strategy Bucket",
        xaxis_title=None,
        yaxis_title="Units",
        hovermode="x unified",
        template="plotly_white",
        height=420,
        legend={"orientation": "h", "y": -0.15},
        margin={"t": 50, "b": 10},
    )
    return fig


def build_neither_cumulative_pnl_chart(neither_plays_settled: pd.DataFrame) -> go.Figure:
    """Single-line cumulative hypothetical P&L chart for the Blind Unders tab."""
    daily_stats: pd.DataFrame = (
        neither_plays_settled.groupby("date")
        .agg(daily_hyp_pnl=("hyp_pnl", "sum"), n_rows=("player_normalized", "count"))
        .reset_index()
        .sort_values("date")
    )
    origin_date: pd.Timestamp = daily_stats["date"].min() - pd.Timedelta(days=1)
    origin_row: pd.DataFrame = pd.DataFrame({
        "date": [origin_date],
        "daily_hyp_pnl": [0.0],
        "n_rows": [0],
    })
    daily_stats = pd.concat([origin_row, daily_stats], ignore_index=True).sort_values("date")
    daily_stats["cumulative_hyp_pnl"] = daily_stats["daily_hyp_pnl"].cumsum()
    customdata: np.ndarray = np.stack(
        [daily_stats["daily_hyp_pnl"].values, daily_stats["n_rows"].values], axis=1
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily_stats["date"],
            y=daily_stats["cumulative_hyp_pnl"],
            mode="lines+markers",
            name="blind unders",
            line={"color": BUCKET_COLORS["neither"], "width": 2},
            marker={"size": 5},
            customdata=customdata,
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                "Running P&L: <b>%{y:.2f}u</b><br>"
                "Daily P&L: %{customdata[0]:.2f}u<br>"
                "Rows: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    if not daily_stats.empty:
        fig.add_vline(x=pd.Timestamp(PROD_GO_LIVE_DATE), line_dash="dash", line_color="#9ca3af")
        fig.add_annotation(
            x=pd.Timestamp(PROD_GO_LIVE_DATE), y=1, yref="paper",
            text="Prod go-live", showarrow=False, xanchor="left",
            font={"size": 11, "color": "#6b7280"},
        )
    fig.add_hline(y=0, line_color="#e5e7eb", line_width=1)
    fig.update_layout(
        title="Hypothetical Cumulative P&L — Betting Every Under (no edge filter)",
        xaxis_title=None,
        yaxis_title="Units (hypothetical)",
        hovermode="x unified",
        template="plotly_white",
        height=380,
        margin={"t": 50, "b": 10},
    )
    return fig


def build_player_pnl_chart(settled_bets: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of the top/bottom N players by total P&L units."""
    player_pnl: pd.Series = (
        settled_bets.groupby("player_normalized")["pnl_units"].sum().sort_values()
    )
    # Keep bottom N (worst) and top N (best), de-duplicate players that appear in both
    bottom = player_pnl.head(top_n)
    top = player_pnl.tail(top_n)
    display: pd.Series = pd.concat([bottom, top]).loc[lambda s: ~s.index.duplicated()].sort_values()

    bar_colors: list[str] = ["#dc2626" if v < 0 else "#16a34a" for v in display.values]

    fig = go.Figure(
        go.Bar(
            x=display.values,
            y=display.index.tolist(),
            orientation="h",
            marker_color=bar_colors,
            hovertemplate="<b>%{y}</b><br>P&L: %{x:.2f}u<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color="#374151", line_width=1)
    fig.update_layout(
        title=f"Top/Bottom {top_n} Players by P&L",
        xaxis_title="Units",
        yaxis_title=None,
        template="plotly_white",
        height=max(350, len(display) * 22),
        margin={"t": 50, "l": 160, "b": 10},
    )
    return fig


# ── Section renderers ──────────────────────────────────────────────────────────


def render_kpi_strip(kpis: dict[str, float | int]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Bets", f"{kpis['total_bets']:,}")
    col2.metric(
        "Total P&L",
        f"{kpis['total_pnl']:+.2f}u",
        delta_color="normal",
    )
    col3.metric(
        "Hit Rate",
        f"{kpis['hit_rate']:.1%}" if not np.isnan(kpis["hit_rate"]) else "—",
    )
    col4.metric(
        "ROI / Bet",
        f"{kpis['roi_per_bet']:+.3f}u" if not np.isnan(kpis["roi_per_bet"]) else "—",
    )


def render_neither_kpi_strip(kpis: dict[str, float | int]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows (settled)", f"{kpis['total_rows']:,}")
    col2.metric(
        "Hypothetical P&L",
        f"{kpis['total_hyp_pnl']:+.2f}u",
    )
    col3.metric(
        "Hit Rate",
        f"{kpis['hit_rate']:.1%}" if not np.isnan(kpis["hit_rate"]) else "—",
    )
    col4.metric(
        "Hyp. ROI / Bet",
        f"{kpis['hyp_roi']:+.3f}u" if not np.isnan(kpis["hyp_roi"]) else "—",
    )


def render_strategy_table(settled_bets: pd.DataFrame) -> None:
    summary: pd.DataFrame = (
        settled_bets.groupby("strategy_bucket")
        .agg(
            Bets=("pnl_units", "count"),
            Wins=("result", lambda x: int((x == "win").sum())),
            Losses=("result", lambda x: int((x == "loss").sum())),
            Pushes=("result", lambda x: int((x == "push").sum())),
            PnL=("pnl_units", "sum"),
        )
        .reset_index()
        .rename(columns={"strategy_bucket": "Strategy"})
    )
    non_push_count: pd.Series = summary["Wins"] + summary["Losses"]
    summary["Hit Rate"] = (summary["Wins"] / non_push_count).map(
        lambda v: f"{v:.1%}" if not np.isnan(v) else "—"
    )
    summary["ROI / Bet"] = (summary["PnL"] / summary["Bets"]).map(
        lambda v: f"{v:+.3f}u" if not np.isnan(v) else "—"
    )
    summary["PnL"] = summary["PnL"].map(lambda v: f"{v:+.2f}u")

    st.dataframe(summary.style.hide(axis="index"), use_container_width=True, hide_index=True)


def render_todays_plays(scored: pd.DataFrame | None) -> None:
    st.subheader("Today's Plays")
    if scored is None:
        st.info("Model runs daily ~10am ET. Check back for today's plays. Yesterday's settled results are shown below.")
        return

    played_today: pd.DataFrame = scored.loc[
        scored["strategy_bucket"].isin(PLAYED_BUCKETS)
    ].sort_values("edge_under_ols", ascending=False).reset_index(drop=True)

    if played_today.empty:
        st.info("Model ran today but found no plays meeting the edge threshold.")
        return

    st.caption(f"{len(played_today)} play(s) found today — sorted by OLS edge descending.")
    display_df: pd.DataFrame = (
        played_today[TODAY_SCORED_DISPLAY_COLS]
        .rename(columns=TODAY_SCORED_COL_LABELS)
    )
    impl_idx: int = display_df.columns.get_loc("Under Odds") + 1
    display_df.insert(
        impl_idx,
        "Implied Prob",
        american_to_implied_prob(display_df["Under Odds"]),
    )
    styled = (
        display_df.style
        .format({
            "Line": "{:.1f}",
            "Under Odds": lambda v: f"{int(v):+d}" if pd.notna(v) else "—",
            "Implied Prob": lambda v: f"{v:.1%}" if pd.notna(v) else "—",
            "P(Under) OLS": "{:.3f}",
            "P(Under) XGB": "{:.3f}",
            "Edge (OLS)": "{:.3f}",
            "Edge (XGB)": "{:.3f}",
        })
        .background_gradient(
            subset=["Edge (OLS)", "Edge (XGB)"],
            cmap=_EDGE_CMAP,
            vmin=0,
            vmax=1,
        )
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_plays_table(plays: pd.DataFrame, download_key: str, cap_rows: bool = False) -> None:
    """Render sorted plays table with color-coded result rows and CSV download."""
    sorted_plays: pd.DataFrame = plays.sort_values(
        ["date", "strategy_bucket", "player_normalized"], ascending=[False, True, True]
    )

    if cap_rows and len(sorted_plays) > 100:
        _opts = ["100", "500", "1,000", f"All ({len(sorted_plays):,})"]
        _vals = [100, 500, 1000, len(sorted_plays)]
        _sel = st.selectbox("Show rows", _opts, index=0, key=f"row_cap_{download_key}")
        _limit = _vals[_opts.index(_sel)]
        st.caption(f"Showing {min(_limit, len(sorted_plays)):,} of {len(sorted_plays):,} — download CSV for full data.")
        view_plays = sorted_plays.head(_limit)
    else:
        view_plays = sorted_plays

    display_df: pd.DataFrame = (
        view_plays[PLAYS_DISPLAY_COLS]
        .rename(columns=PLAYS_COL_LABELS)
    )
    impl_idx: int = display_df.columns.get_loc("Under Odds") + 1
    display_df.insert(
        impl_idx,
        "Implied Prob",
        american_to_implied_prob(view_plays["under_odds"]).values,
    )

    def color_result_row(row: pd.Series) -> list[str]:
        bg: str = RESULT_BG_COLORS.get(str(row["Result"]).lower(), "white")
        return [f"background-color: {bg}"] * len(row)

    styled = (
        display_df.style
        .apply(color_result_row, axis=1)
        .format({
            "Line": "{:.1f}",
            "Actual": lambda v: f"{v:.1f}" if pd.notna(v) else "—",
            "Diff": lambda v: f"{v:+.1f}" if pd.notna(v) else "—",
            "Under Odds": lambda v: f"{int(v):+d}" if pd.notna(v) else "—",
            "Implied Prob": lambda v: f"{v:.1%}" if pd.notna(v) else "—",
            "P(Under) OLS": lambda v: f"{v:.3f}" if pd.notna(v) else "—",
            "P(Under) XGB": lambda v: f"{v:.3f}" if pd.notna(v) else "—",
            "Edge (OLS)": lambda v: f"{v:.3f}" if pd.notna(v) else "—",
            "Edge (XGB)": lambda v: f"{v:.3f}" if pd.notna(v) else "—",
            "Date": lambda v: v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else str(v),
        })
        .background_gradient(
            subset=["Edge (OLS)", "Edge (XGB)"],
            cmap=_EDGE_CMAP,
            vmin=0,
            vmax=1,
        )
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)

    csv_df: pd.DataFrame = (
        sorted_plays[PLAYS_DISPLAY_COLS]
        .rename(columns=PLAYS_COL_LABELS)
        .assign(Date=lambda d: d["Date"].astype(str))
    )
    csv_bytes: bytes = csv_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=f"rebounds_{download_key}.csv",
        mime="text/csv",
        key=download_key,
    )


def render_pipeline_health(manifests: list[dict]) -> None:
    rows: list[dict] = []
    for manifest in manifests:
        rows.append({
            "Date": manifest["date"],
            "Status": manifest.get("settlement_status", "unknown"),
            "Rows": manifest.get("n_rows", "—"),
            "Unsettled Bets": manifest.get("n_unsettled_bet_rows", "—"),
            "Players": manifest.get("n_distinct_players", "—"),
            "Settled At (UTC)": manifest.get("settled_at_utc", "—")[:19],
        })

    health_df: pd.DataFrame = pd.DataFrame(rows)

    def flag_stale(row: pd.Series) -> list[str]:
        status = str(row.get("Status", ""))
        return ["background-color: #fee2e2"] * len(row) if status == "partial" else [""] * len(row)

    st.dataframe(
        health_df.style.apply(flag_stale, axis=1),
        use_container_width=True,
        hide_index=True,
    )


# ── Backtest renderers ─────────────────────────────────────────────────────────

_BT_BUCKET_COLORS: dict[str, str] = {"both": "#17408B", "ols": "#ff7f0e", "xgb": "#2ca02c"}

_BT_DETAIL_COLS: list[str] = [
    "date", "player_normalized", "line", "actual", "under_odds", "edge", "result", "pnl_units",
]
_BT_DETAIL_LABELS: dict[str, str] = {
    "date": "Date", "player_normalized": "Player", "line": "Line", "actual": "Actual",
    "under_odds": "Odds", "edge": "Edge", "result": "Result", "pnl_units": "P&L",
}


def build_backtest_pnl_chart(settled: pd.DataFrame, selected_buckets: list[str]) -> go.Figure:
    fig = go.Figure()
    all_dates: pd.Series = settled["date"]
    seasons = sorted(settled["season"].unique())

    for bucket in selected_buckets:
        sub = settled[settled["strategy_bucket"] == bucket].sort_values("date")
        if sub.empty:
            continue
        daily = (
            sub.groupby("date")
            .agg(daily_pnl=("pnl_units", "sum"), n_bets=("pnl_units", "count"))
            .reset_index()
            .sort_values("date")
        )
        daily["cumulative_pnl"] = daily["daily_pnl"].cumsum()
        origin = pd.DataFrame({
            "date": [daily["date"].min() - pd.Timedelta(days=1)],
            "daily_pnl": [0.0], "n_bets": [0], "cumulative_pnl": [0.0],
        })
        daily = pd.concat([origin, daily], ignore_index=True)
        customdata = np.stack([daily["daily_pnl"].values, daily["n_bets"].values], axis=1)
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["cumulative_pnl"],
            mode="lines", name=bucket.upper(),
            line={"color": _BT_BUCKET_COLORS.get(bucket, "#888"), "width": 2},
            customdata=customdata,
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                "Running P&L: <b>%{y:.2f}u</b><br>"
                "Daily: %{customdata[0]:.2f}u · Bets: %{customdata[1]}<extra></extra>"
            ),
        ))

    if not all_dates.empty:
        for season in seasons:
            s_min = settled[settled["season"] == season]["date"].min()
            fig.add_vline(x=s_min, line_dash="dot", line_color="#d1d5db", line_width=1)
            fig.add_annotation(
                x=s_min, y=0.98, yref="paper", text=season,
                showarrow=False, xanchor="left", font={"size": 10, "color": "#9ca3af"},
            )
        fig.add_vline(x=pd.Timestamp(BT_PROD_GO_LIVE_DATE), line_dash="dash", line_color="#9ca3af")
        fig.add_annotation(
            x=pd.Timestamp(BT_PROD_GO_LIVE_DATE), y=0.88, yref="paper",
            text="Prod go-live", showarrow=False, xanchor="left",
            font={"size": 11, "color": "#6b7280"},
        )

    fig.add_hline(y=0, line_color="#e5e7eb", line_width=1)
    fig.update_layout(
        title="Cumulative P&L — OOS Walk-Forward (units at -110)",
        xaxis_title=None, yaxis_title="Units",
        hovermode="x unified", template="plotly_white",
        height=400, margin={"t": 50, "b": 10},
    )
    return fig


def render_backtest_per_season_table(df_filtered: pd.DataFrame) -> None:
    summary = per_season_summary(df_filtered)
    if summary.empty:
        st.info("No data for selected models.")
        return
    display = summary.copy()
    display["Hit Rate"] = display["Hit Rate"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    display["PnL"]      = display["PnL"].map(lambda v: f"{v:+.2f}u")
    display["ROI/Bet"]  = display["ROI/Bet"].map(lambda v: f"{v:+.3f}u" if pd.notna(v) else "—")
    display["Model"]    = display["Model"].str.upper()

    current_season = sorted(summary["Season"].unique())[-1] if not summary.empty else None

    def _highlight_current(row: pd.Series) -> list[str]:
        return ["background-color: #eff6ff"] * len(row) if row["Season"] == current_season else [""] * len(row)

    st.dataframe(display.style.apply(_highlight_current, axis=1), use_container_width=True, hide_index=True)


def render_backtest_season_detail(df_filtered: pd.DataFrame) -> None:
    available = sorted(df_filtered["season"].unique(), reverse=True)
    if not available:
        return
    selected_season = st.selectbox("Season", available, index=0, key="bt_reb_season")
    sub = df_filtered[df_filtered["season"] == selected_season]
    settled = bt_settled_rows(sub)
    if settled.empty:
        st.info("No settled plays for this season / filter.")
        return

    kpis = bt_compute_kpis(settled)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bets",     kpis["total_bets"])
    c2.metric("W-L",      f"{kpis['wins']}-{kpis['losses']}")
    c3.metric("Hit Rate", f"{kpis['hit_rate']:.1%}" if not np.isnan(kpis["hit_rate"]) else "—")
    c4.metric("P&L",      f"{kpis['total_pnl']:+.2f}u")

    display_df = (
        sub.sort_values("date", ascending=False)[_BT_DETAIL_COLS]
        .rename(columns=_BT_DETAIL_LABELS)
    )

    def _color_row(row: pd.Series) -> list[str]:
        bg = {"win": "#dcfce7", "loss": "#fee2e2"}.get(str(row.get("Result", "")).lower(), "")
        return [f"background-color: {bg}"] * len(row) if bg else [""] * len(row)

    st.dataframe(
        display_df.style.apply(_color_row, axis=1).format({
            "Date":   lambda v: v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else str(v),
            "Line":   lambda v: f"{v:.1f}" if pd.notna(v) else "—",
            "Actual": lambda v: f"{v:.1f}" if pd.notna(v) else "—",
            "Odds":   lambda v: f"{int(v):+d}" if pd.notna(v) else "—",
            "Edge":   lambda v: f"{v:.3f}" if pd.notna(v) else "—",
            "P&L":    lambda v: f"{v:+.3f}u" if pd.notna(v) else "—",
        }),
        use_container_width=True, hide_index=True,
    )

    with st.expander("Edge Distribution"):
        edge_vals = settled["edge"].dropna()
        fig = go.Figure(go.Histogram(
            x=edge_vals, nbinsx=20, marker_color="#065f46", opacity=0.8,
            hovertemplate="Edge: %{x:.3f}<br>Count: %{y}<extra></extra>",
        ))
        fig.add_vline(x=0.05, line_color="#374151", line_width=1, line_dash="dash")
        fig.update_layout(
            title="Edge Distribution (min_edge=0.05 threshold shown)",
            xaxis_title="Edge", yaxis_title="Bets",
            template="plotly_white", height=260, margin={"t": 50, "b": 10},
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────


def build_sidebar(available_bookmakers: list[str]) -> SidebarFilters:
    with st.sidebar:
        st.markdown("### 📊 NBA Rebounds Strategy")
        st.markdown("---")

        preset_options: list[str] = ["All Time", "Season-to-Date", "Last 30 Days", "Last 7 Days", "Custom"]
        selected_preset: str = st.radio("Date Range", preset_options, index=0)

        today: date = date.today()
        if selected_preset == "All Time":
            start_date, end_date = date(2026, 4, 7), today
        elif selected_preset == "Season-to-Date":
            start_date, end_date = NBA_SEASON_2025_START, today
        elif selected_preset == "Last 30 Days":
            start_date, end_date = today - timedelta(days=30), today
        elif selected_preset == "Last 7 Days":
            start_date, end_date = today - timedelta(days=7), today
        else:
            start_date = st.date_input("Start date", value=date(2026, 4, 7))
            end_date = st.date_input("End date", value=today)

        st.markdown("---")
        all_bucket_options: list[str] = ["both", "ols", "xgb"]
        selected_buckets: list[str] = st.multiselect(
            "Strategy buckets",
            options=all_bucket_options,
            default=all_bucket_options,
        )

        st.markdown("---")
        selected_bookmakers: list[str] = st.multiselect(
            "Bookmakers",
            options=available_bookmakers,
            default=available_bookmakers,
            help="Filter to rows from specific sportsbooks only.",
        )

        consensus_filter: str = st.radio(
            "Consensus line",
            options=CONSENSUS_FILTER_OPTIONS,
            index=0,
            help=(
                "Consensus: the bookmaker's line matches the most common line offered "
                "for that player on that date across all books."
            ),
        )

        st.markdown("---")
        st.caption(f"Data as of: {today}")
        st.caption(f"Prod go-live: {PROD_GO_LIVE_DATE}")

    return SidebarFilters(
        start_date=start_date,
        end_date=end_date,
        selected_buckets=selected_buckets,
        selected_bookmakers=selected_bookmakers,
        consensus_filter=consensus_filter,
    )


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    st.title("NBA Rebounds Strategy")

    with st.spinner("Loading rebounds data from S3…"):
        all_settled_raw: pd.DataFrame = load_settled_plays()
        todays_scored: pd.DataFrame | None = load_todays_scored()

    if all_settled_raw.empty:
        st.warning("No settled plays found in S3. The pipeline may not have run yet.")
        st.stop()

    all_settled: pd.DataFrame = add_is_consensus_line(all_settled_raw)
    available_bookmakers: list[str] = sorted(all_settled["bookmaker"].dropna().unique().tolist())
    filters: SidebarFilters = build_sidebar(available_bookmakers)

    todays_scored_filtered: pd.DataFrame | None = None
    if todays_scored is not None:
        scored_with_bucket: pd.DataFrame = derive_strategy_bucket(todays_scored)
        scored_with_consensus: pd.DataFrame = add_is_consensus_line(scored_with_bucket)
        todays_scored_filtered = apply_custom_filters(
            scored_with_consensus, filters.selected_bookmakers, filters.consensus_filter
        )

    date_filtered: pd.DataFrame = apply_date_filter(all_settled, filters.start_date, filters.end_date)

    played_filtered: pd.DataFrame = date_filtered.loc[
        date_filtered["strategy_bucket"].isin(PLAYED_BUCKETS)
    ].reset_index(drop=True)
    neither_filtered: pd.DataFrame = date_filtered.loc[
        date_filtered["strategy_bucket"] == "neither"
    ].reset_index(drop=True)

    played_filtered = apply_custom_filters(
        played_filtered, filters.selected_bookmakers, filters.consensus_filter
    )
    neither_filtered = apply_custom_filters(
        neither_filtered, filters.selected_bookmakers, filters.consensus_filter
    )

    bucket_filtered: pd.DataFrame = played_filtered.loc[
        played_filtered["strategy_bucket"].isin(filters.selected_buckets)
    ].reset_index(drop=True)

    settled_played: pd.DataFrame = settled_bets_only(bucket_filtered)
    neither_with_hyp: pd.DataFrame = add_hypothetical_pnl(neither_filtered)
    neither_settled: pd.DataFrame = neither_with_hyp.loc[
        neither_with_hyp["result"].isin({"win", "loss", "push"})
    ].reset_index(drop=True)

    tab1, tab2, tab3 = st.tabs(["Played Bets", "Blind Unders Benchmark", "Backtest"])

    # ── Tab 1: Played Bets ──────────────────────────────────────────────────────
    with tab1:
        if settled_played.empty:
            st.info("No settled bets found for the selected date range and buckets.")
        else:
            kpis = compute_kpis(settled_played)
            render_kpi_strip(kpis)
            st.markdown("---")

            chart_col, table_col = st.columns([2, 1])
            with chart_col:
                pnl_chart = build_cumulative_pnl_chart(settled_played, filters.selected_buckets)
                st.plotly_chart(pnl_chart, use_container_width=True)
            with table_col:
                st.markdown("**Strategy Breakdown**")
                render_strategy_table(settled_played)

            st.markdown("---")
            render_todays_plays(todays_scored_filtered)

            st.markdown("---")
            st.subheader("Plays History")
            render_plays_table(bucket_filtered, download_key="dl_played")

            with st.expander("Player P&L Breakdown"):
                player_chart = build_player_pnl_chart(settled_played)
                st.plotly_chart(player_chart, use_container_width=True)

            with st.expander("Pipeline Health (last 7 runs)"):
                manifests: list[dict] = load_run_manifests(7)
                if manifests:
                    render_pipeline_health(manifests)
                else:
                    st.caption("No settlement manifests found.")

    # ── Tab 2: Blind Unders Benchmark ──────────────────────────────────────────
    with tab2:
        st.markdown(
            "Shows the hypothetical P&L if every under the model *skipped* had been bet — "
            "regardless of edge. A negative trend here validates the edge filter."
        )
        if neither_settled.empty:
            st.info("No settled 'neither' rows found for the selected date range.")
        else:
            neither_kpis = compute_neither_kpis(neither_settled)
            render_neither_kpi_strip(neither_kpis)
            st.markdown("---")

            neither_chart = build_neither_cumulative_pnl_chart(neither_settled)
            st.plotly_chart(neither_chart, use_container_width=True)

            st.markdown("---")
            st.subheader("Skipped Plays (neither)")
            render_plays_table(neither_filtered, download_key="dl_neither", cap_rows=True)

    # ── Tab 3: Backtest ────────────────────────────────────────────────────────
    with tab3:
        st.caption("OOS walk-forward backtest · A3 spec · min_edge=0.05 · -110 juice")

        with st.spinner("Loading backtest data…"):
            bt_raw = load_backtest_multi()

        if bt_raw.empty:
            st.warning("No backtest data in S3 yet — run the export cell in the foundations notebook first.")
        elif not filters.selected_buckets:
            st.info("Select at least one bucket.")
        else:
            bt_filtered = filter_by_buckets(bt_raw, filters.selected_buckets)
            bt_settled  = bt_settled_rows(bt_filtered)

            if bt_settled.empty:
                st.info("No settled plays for the selected models.")
            else:
                bt_kpis = bt_compute_kpis(bt_settled)
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total Bets",  f"{bt_kpis['total_bets']:,}")
                c2.metric("W-L",         f"{bt_kpis['wins']}-{bt_kpis['losses']}")
                c3.metric("Hit Rate",    f"{bt_kpis['hit_rate']:.1%}" if not np.isnan(bt_kpis["hit_rate"]) else "—")
                c4.metric("P&L (units)", f"{bt_kpis['total_pnl']:+.2f}u")
                c5.metric("ROI / Bet",   f"{bt_kpis['roi_per_bet']:+.3f}u" if not np.isnan(bt_kpis["roi_per_bet"]) else "—")

                st.plotly_chart(build_backtest_pnl_chart(bt_settled, filters.selected_buckets), use_container_width=True)

                st.markdown("---")
                st.subheader("Per-Season Summary")
                st.caption("Most recent season highlighted.")
                render_backtest_per_season_table(bt_filtered)

                st.markdown("---")
                st.subheader("Season Detail")
                render_backtest_season_detail(bt_filtered)


if __name__ == "__main__":
    main()

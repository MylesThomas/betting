"""
NCAAB Away Revenge Strategy Dashboard

Three tabs on one page:
  - Today's Plays: today's rematch spots (if Lambda has run) + yesterday's results
  - Season Record: KPI strip, cumulative P&L, plays history, conference/spread breakdowns
  - Backtest: multi-season backtest KPIs, cumulative P&L, per-season table, season detail
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from ncaab_revenge_data import (
    NCAAB_PAUSE_UNTIL,
    PROD_GO_LIVE_DATE,
    load_all_plays,
    load_todays_plays,
    settle_plays,
)
from ncaab_backtest_data import (
    SEASON_ORDER,
    compute_kpis as bt_compute_kpis,
    filter_by_side,
    load_backtest_multi,
    per_season_summary,
    settled_rows as bt_settled_rows,
)

# ── Constants ──────────────────────────────────────────────────────────────────

RESULT_BG_COLORS: dict[str, str] = {
    "win": "#dcfce7",
    "loss": "#fee2e2",
    "push": "#fef9c3",
    "pending": "#eff6ff",
    "no_line": "#f1f5f9",
}

HISTORY_DISPLAY_COLS: list[str] = [
    "game_date",
    "bet_team",
    "bet_spread",
    "matchup",
    "away_conference",
    "home_conference",
    "score",
    "result",
    "spread_margin",
    "pnl_units",
    "prior_meetings",
]

HISTORY_COL_LABELS: dict[str, str] = {
    "game_date": "Date",
    "bet_team": "Bet Team",
    "bet_spread": "Spread",
    "matchup": "Away @ Home",
    "away_conference": "Away Conf",
    "home_conference": "Home Conf",
    "score": "Score",
    "result": "Result",
    "spread_margin": "Margin",
    "pnl_units": "P&L",
    "prior_meetings": "Prior Meetings",
}

TODAY_DISPLAY_COLS: list[str] = [
    "start_time_et",
    "bet_team",
    "bet_spread",
    "matchup",
    "away_conference",
    "home_conference",
    "prior_meetings",
]

TODAY_COL_LABELS: dict[str, str] = {
    "start_time_et": "Tip-off (ET)",
    "bet_team": "Bet Team",
    "bet_spread": "Spread",
    "matchup": "Away @ Home",
    "away_conference": "Away Conf",
    "home_conference": "Home Conf",
    "prior_meetings": "Prior Meetings",
}


class SidebarFilters(NamedTuple):
    start_date: date
    end_date: date
    selected_conferences: list[str]


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NCAAB Away Revenge",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data transforms ────────────────────────────────────────────────────────────


def add_derived_columns(plays: pd.DataFrame) -> pd.DataFrame:
    """Add matchup, bet_spread, score, and truncated prior_meetings for display."""
    work: pd.DataFrame = plays.copy()
    work["matchup"] = work["away_team"].astype(str) + " @ " + work["home_team"].astype(str)

    is_home_bet: pd.Series = (
        work["bet_team"].astype(str).str.strip()
        == work["home_team"].astype(str).str.strip()
    )
    work["bet_spread"] = np.where(
        is_home_bet,
        work["consensus_spread_home"],
        -work["consensus_spread_home"],
    )

    has_score: pd.Series = work["home_score"].notna() & work["away_score"].notna()
    work["score"] = ""
    work.loc[has_score, "score"] = (
        work.loc[has_score, "away_score"].astype(int).astype(str)
        + "-"
        + work.loc[has_score, "home_score"].astype(int).astype(str)
    )

    work["prior_meetings"] = (
        work["prior_meetings"].fillna("").astype(str).str[:60]
    )
    return work


def filter_to_bet_plays(plays: pd.DataFrame) -> pd.DataFrame:
    """Return only rows that are actual plays (bet_team populated)."""
    return plays.loc[
        plays["bet_team"].notna()
        & (plays["bet_team"].astype(str).str.strip() != "")
        & (plays["bet_team"].astype(str).str.strip().str.lower() != "nan")
    ].reset_index(drop=True)


def apply_date_filter(plays: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    mask: pd.Series = (
        (plays["game_date"].dt.date >= start) & (plays["game_date"].dt.date <= end)
    )
    return plays.loc[mask].reset_index(drop=True)


def apply_conference_filter(
    plays: pd.DataFrame, selected_conferences: list[str]
) -> pd.DataFrame:
    return plays.loc[
        plays["away_conference"].isin(selected_conferences)
    ].reset_index(drop=True)


def settled_only(plays: pd.DataFrame) -> pd.DataFrame:
    return plays.loc[plays["result"].isin({"win", "loss", "push"})].reset_index(drop=True)


def compute_kpis(settled: pd.DataFrame) -> dict[str, float | int]:
    non_push: pd.DataFrame = settled.loc[settled["result"].isin({"win", "loss"})]
    wins_count: int = int((settled["result"] == "win").sum())
    losses_count: int = int((settled["result"] == "loss").sum())
    total_bets: int = len(settled)
    total_pnl: float = float(settled["pnl_units"].sum())
    ats_pct: float = wins_count / len(non_push) if len(non_push) > 0 else float("nan")
    roi_per_bet: float = total_pnl / total_bets if total_bets > 0 else float("nan")
    return {
        "total_bets": total_bets,
        "wins": wins_count,
        "losses": losses_count,
        "total_pnl": total_pnl,
        "ats_pct": ats_pct,
        "roi_per_bet": roi_per_bet,
    }


# ── Chart builders ─────────────────────────────────────────────────────────────


def build_cumulative_pnl_chart(settled: pd.DataFrame) -> go.Figure:
    daily: pd.DataFrame = (
        settled.groupby("game_date")
        .agg(daily_pnl=("pnl_units", "sum"), n_bets=("bet_team", "count"))
        .reset_index()
        .sort_values("game_date")
    )
    daily["cumulative_pnl"] = daily["daily_pnl"].cumsum()
    if not daily.empty:
        origin = pd.DataFrame({
            "game_date": [daily["game_date"].min() - pd.Timedelta(days=1)],
            "daily_pnl": [0.0],
            "n_bets": [0],
            "cumulative_pnl": [0.0],
        })
        daily = pd.concat([origin, daily], ignore_index=True)
    customdata: np.ndarray = np.stack(
        [daily["daily_pnl"].values, daily["n_bets"].values], axis=1
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["game_date"],
            y=daily["cumulative_pnl"],
            mode="lines+markers",
            name="cumulative P&L",
            line={"color": "#17408B", "width": 2},
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
    if not daily.empty:
        fig.add_vline(
            x=pd.Timestamp(PROD_GO_LIVE_DATE), line_dash="dash", line_color="#9ca3af"
        )
        fig.add_annotation(
            x=pd.Timestamp(PROD_GO_LIVE_DATE), y=1, yref="paper",
            text="Prod go-live", showarrow=False, xanchor="left",
            font={"size": 11, "color": "#6b7280"},
        )
    fig.add_hline(y=0, line_color="#e5e7eb", line_width=1)
    fig.update_layout(
        title="Cumulative P&L (units at -110)",
        xaxis_title=None,
        yaxis_title="Units",
        hovermode="x unified",
        template="plotly_white",
        height=380,
        margin={"t": 50, "b": 10},
    )
    return fig


def build_spread_histogram(bet_plays: pd.DataFrame) -> go.Figure:
    spread_vals: pd.Series = bet_plays["bet_spread"].dropna()
    fig = go.Figure(
        go.Histogram(
            x=spread_vals,
            nbinsx=20,
            marker_color="#17408B",
            opacity=0.8,
            hovertemplate="Spread: %{x}<br>Count: %{y}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color="#374151", line_width=1, line_dash="dash")
    fig.update_layout(
        title="Spread Distribution (bet team's perspective; negative = favored)",
        xaxis_title="Spread",
        yaxis_title="Plays",
        template="plotly_white",
        height=300,
        margin={"t": 50, "b": 10},
    )
    return fig


# ── Section renderers ──────────────────────────────────────────────────────────


def render_kpi_strip(kpis: dict[str, float | int]) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Bets", f"{kpis['total_bets']:,}")
    col2.metric("W-L", f"{kpis['wins']}-{kpis['losses']}")
    col3.metric(
        "ATS %",
        f"{kpis['ats_pct']:.1%}" if not np.isnan(kpis["ats_pct"]) else "—",
    )
    col4.metric("P&L (units)", f"{kpis['total_pnl']:+.2f}u")
    col5.metric(
        "ROI / Bet",
        f"{kpis['roi_per_bet']:+.3f}u" if not np.isnan(kpis["roi_per_bet"]) else "—",
    )


def render_monthly_breakdown(settled: pd.DataFrame) -> None:
    monthly: pd.DataFrame = (
        settled.assign(month=settled["game_date"].dt.to_period("M"))
        .groupby("month", observed=True)
        .apply(
            lambda g: pd.Series({
                "Bets": len(g),
                "Wins": int((g["result"] == "win").sum()),
                "Losses": int((g["result"] == "loss").sum()),
                "P&L": float(g["pnl_units"].sum()),
            }),
            include_groups=False,
        )
        .reset_index()
    )
    non_push: pd.Series = monthly["Wins"] + monthly["Losses"]
    monthly["ATS %"] = (monthly["Wins"] / non_push).map(
        lambda v: f"{v:.1%}" if not np.isnan(v) else "—"
    )
    monthly["P&L"] = monthly["P&L"].map(lambda v: f"{v:+.2f}u")
    monthly["month"] = monthly["month"].astype(str)
    monthly = monthly.rename(columns={"month": "Month"})
    st.dataframe(monthly, use_container_width=True, hide_index=True)


def render_conference_breakdown(settled: pd.DataFrame) -> None:
    conf: pd.DataFrame = (
        settled.groupby("away_conference", observed=True)
        .apply(
            lambda g: pd.Series({
                "Bets": len(g),
                "Wins": int((g["result"] == "win").sum()),
                "Losses": int((g["result"] == "loss").sum()),
                "P&L": float(g["pnl_units"].sum()),
            }),
            include_groups=False,
        )
        .reset_index()
        .sort_values("Bets", ascending=False)
        .rename(columns={"away_conference": "Conference"})
    )
    non_push: pd.Series = conf["Wins"] + conf["Losses"]
    conf["ATS %"] = (conf["Wins"] / non_push).map(
        lambda v: f"{v:.1%}" if not np.isnan(v) else "—"
    )
    conf["P&L"] = conf["P&L"].map(lambda v: f"{v:+.2f}u")
    st.dataframe(conf, use_container_width=True, hide_index=True)


def _styled_plays_table(display_df: pd.DataFrame) -> object:
    def color_result_row(row: pd.Series) -> list[str]:
        bg: str = RESULT_BG_COLORS.get(str(row.get("Result", "")).lower(), "")
        return [f"background-color: {bg}"] * len(row) if bg else [""] * len(row)

    return (
        display_df.style
        .apply(color_result_row, axis=1)
        .format({
            "Date": lambda v: v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else str(v),
            "Spread": lambda v: f"{v:+.1f}" if pd.notna(v) else "—",
            "Margin": lambda v: f"{v:+.1f}" if pd.notna(v) else "—",
            "P&L": lambda v: f"{v:+.3f}u" if pd.notna(v) else "—",
        })
    )


def render_plays_history(plays: pd.DataFrame, download_key: str) -> None:
    no_line_count: int = int((plays["result"] == "no_line").sum())
    settled_plays: pd.DataFrame = plays.loc[plays["result"] != "no_line"].reset_index(drop=True)
    caption_parts = [f"{len(settled_plays)} play(s)"]
    if no_line_count:
        caption_parts.append(f"{no_line_count} excluded (no spread available)")
    st.caption(" · ".join(caption_parts))

    sorted_plays: pd.DataFrame = settled_plays.sort_values("game_date", ascending=False)
    display_df: pd.DataFrame = (
        sorted_plays[HISTORY_DISPLAY_COLS].rename(columns=HISTORY_COL_LABELS)
    )
    st.dataframe(_styled_plays_table(display_df), use_container_width=True, hide_index=True)

    csv_bytes: bytes = (
        display_df.assign(Date=display_df["Date"].astype(str))
        .to_csv(index=False)
        .encode("utf-8")
    )
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=f"ncaab_revenge_{download_key}.csv",
        mime="text/csv",
        key=download_key,
    )


def render_todays_plays_table(today_plays: pd.DataFrame) -> None:
    bet_today: pd.DataFrame = filter_to_bet_plays(today_plays).sort_values(
        "start_time_et", na_position="last"
    ).reset_index(drop=True)

    if bet_today.empty:
        st.info("No rematch spots today meeting the away-team filter.")
        return

    st.caption(f"{len(bet_today)} play(s) today — sorted by tip-off.")
    display_df: pd.DataFrame = bet_today[TODAY_DISPLAY_COLS].rename(columns=TODAY_COL_LABELS)
    st.dataframe(
        display_df.style.format({
            "Spread": lambda v: f"{v:+.1f}" if pd.notna(v) else "—",
        }),
        use_container_width=True,
        hide_index=True,
    )


def render_yesterday_results(all_bet_plays: pd.DataFrame) -> None:
    yesterday: date = date.today() - timedelta(days=1)
    yesterday_plays: pd.DataFrame = all_bet_plays.loc[
        all_bet_plays["game_date"].dt.date == yesterday
    ].reset_index(drop=True)

    if yesterday_plays.empty:
        return

    settled_yesterday: pd.DataFrame = settled_only(yesterday_plays)
    if settled_yesterday.empty:
        st.caption(f"Yesterday's {len(yesterday_plays)} play(s) — results not yet settled.")
        return

    wins: int = int((settled_yesterday["result"] == "win").sum())
    losses: int = int((settled_yesterday["result"] == "loss").sum())
    pnl: float = float(settled_yesterday["pnl_units"].sum())
    st.subheader(
        f"Yesterday's Results — {yesterday.strftime('%b %d')}: "
        f"{wins}-{losses} ({pnl:+.2f}u)"
    )
    display_df: pd.DataFrame = (
        settled_yesterday[HISTORY_DISPLAY_COLS].rename(columns=HISTORY_COL_LABELS)
    )
    st.dataframe(_styled_plays_table(display_df), use_container_width=True, hide_index=True)


# ── Backtest renderers ─────────────────────────────────────────────────────────


def build_backtest_pnl_chart(settled: pd.DataFrame) -> go.Figure:
    daily: pd.DataFrame = (
        settled.groupby("game_date")
        .agg(daily_pnl=("pnl_units", "sum"), n_bets=("focal_team", "count"))
        .reset_index()
        .sort_values("game_date")
    )
    daily["cumulative_pnl"] = daily["daily_pnl"].cumsum()
    if not daily.empty:
        origin = pd.DataFrame({
            "game_date": [daily["game_date"].min() - pd.Timedelta(days=1)],
            "daily_pnl": [0.0],
            "n_bets": [0],
            "cumulative_pnl": [0.0],
        })
        daily = pd.concat([origin, daily], ignore_index=True)

    customdata: np.ndarray = np.stack(
        [daily["daily_pnl"].values, daily["n_bets"].values], axis=1
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["game_date"],
            y=daily["cumulative_pnl"],
            mode="lines",
            name="cumulative P&L",
            line={"color": "#17408B", "width": 2},
            customdata=customdata,
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                "Running P&L: <b>%{y:.2f}u</b><br>"
                "Daily P&L: %{customdata[0]:.2f}u<br>"
                "Bets: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    for season in SEASON_ORDER:
        sub: pd.DataFrame = settled[settled["season"] == season]
        if sub.empty:
            continue
        fig.add_vline(
            x=sub["game_date"].min(), line_dash="dot", line_color="#d1d5db", line_width=1
        )
        fig.add_annotation(
            x=sub["game_date"].min(), y=0.98, yref="paper",
            text=season, showarrow=False, xanchor="left",
            font={"size": 10, "color": "#9ca3af"},
        )
    fig.add_vline(
        x=pd.Timestamp(PROD_GO_LIVE_DATE), line_dash="dash", line_color="#9ca3af"
    )
    fig.add_annotation(
        x=pd.Timestamp(PROD_GO_LIVE_DATE), y=0.88, yref="paper",
        text="Prod go-live", showarrow=False, xanchor="left",
        font={"size": 11, "color": "#6b7280"},
    )
    fig.add_hline(y=0, line_color="#e5e7eb", line_width=1)
    fig.update_layout(
        title="Cumulative P&L — All Seasons (units at -110)",
        xaxis_title=None,
        yaxis_title="Units",
        hovermode="x unified",
        template="plotly_white",
        height=400,
        margin={"t": 50, "b": 10},
    )
    return fig


def render_per_season_table(df_side: pd.DataFrame) -> None:
    summary: pd.DataFrame = per_season_summary(df_side)
    if summary.empty:
        st.info("No data for selected filter.")
        return
    display: pd.DataFrame = summary.copy()
    display["ATS %"] = display["ATS %"].map(
        lambda v: f"{v:.1%}" if not np.isnan(v) else "—"
    )
    display["P&L"] = display["P&L"].map(lambda v: f"{v:+.2f}u")
    display["ROI/Bet"] = display["ROI/Bet"].map(
        lambda v: f"{v:+.3f}u" if not np.isnan(v) else "—"
    )
    display = display.iloc[::-1].reset_index(drop=True)  # newest first

    def _highlight_deployed(row: pd.Series) -> list[str]:
        return (
            ["background-color: #eff6ff"] * len(row)
            if row["Season"] == "2025-26"
            else [""] * len(row)
        )

    st.dataframe(
        display.style.apply(_highlight_deployed, axis=1),
        use_container_width=True,
        hide_index=True,
    )


_BT_DETAIL_COLS: list[str] = [
    "game_date", "focal_team", "focal_spread", "matchup",
    "result", "focal_ats_margin", "pnl_units", "game_period",
]
_BT_DETAIL_LABELS: dict[str, str] = {
    "game_date": "Date", "focal_team": "Bet Team", "focal_spread": "Spread",
    "matchup": "Away @ Home", "result": "Result",
    "focal_ats_margin": "Margin", "pnl_units": "P&L", "game_period": "Period",
}


def render_backtest_season_detail(df_side: pd.DataFrame) -> None:
    available: list[str] = [
        s for s in reversed(SEASON_ORDER) if s in df_side["season"].values
    ]
    if not available:
        return
    selected: str = st.selectbox("Season", available, index=0, key="bt_season_select")
    sub: pd.DataFrame = df_side[df_side["season"] == selected].copy()
    settled: pd.DataFrame = bt_settled_rows(sub)
    if settled.empty:
        st.info("No settled plays for this season / filter.")
        return

    kpis = bt_compute_kpis(settled)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bets", kpis["total_bets"])
    c2.metric("W-L", f"{kpis['wins']}-{kpis['losses']}")
    c3.metric("ATS %", f"{kpis['ats_pct']:.1%}" if not np.isnan(kpis["ats_pct"]) else "—")
    c4.metric("P&L", f"{kpis['total_pnl']:+.2f}u")

    display: pd.DataFrame = (
        sub.sort_values("game_date", ascending=False)[_BT_DETAIL_COLS]
        .rename(columns=_BT_DETAIL_LABELS)
    )

    def _color_bt_row(row: pd.Series) -> list[str]:
        bg: str = RESULT_BG_COLORS.get(str(row.get("Result", "")).lower(), "")
        return [f"background-color: {bg}"] * len(row) if bg else [""] * len(row)

    st.dataframe(
        display.style
        .apply(_color_bt_row, axis=1)
        .format({
            "Date": lambda v: v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else str(v),
            "Spread": lambda v: f"{v:+.1f}" if pd.notna(v) else "—",
            "Margin": lambda v: f"{v:+.1f}" if pd.notna(v) else "—",
            "P&L": lambda v: f"{v:+.3f}u" if pd.notna(v) else "—",
        }),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Spread Distribution"):
        spread_vals: pd.Series = settled["focal_spread"].dropna()
        fig = go.Figure(go.Histogram(
            x=spread_vals, nbinsx=20, marker_color="#17408B", opacity=0.8,
            hovertemplate="Spread: %{x}<br>Count: %{y}<extra></extra>",
        ))
        fig.add_vline(x=0, line_color="#374151", line_width=1, line_dash="dash")
        fig.update_layout(
            title="Spread Distribution (focal team's perspective; negative = favored)",
            xaxis_title="Spread", yaxis_title="Games",
            template="plotly_white", height=280, margin={"t": 50, "b": 10},
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────


def build_sidebar(available_conferences: list[str]) -> SidebarFilters:
    with st.sidebar:
        st.markdown("### 🏀 NCAAB Away Revenge")
        st.markdown("---")

        preset_options: list[str] = [
            "All Time",
            "Season-to-Date",
            "Last 30 Days",
            "Last 7 Days",
            "Custom",
        ]
        selected_preset: str = st.radio("Date Range", preset_options, index=0)

        today: date = date.today()
        season_start: date = date(2026, 2, 19)
        if selected_preset == "All Time":
            start_date, end_date = season_start, today
        elif selected_preset == "Season-to-Date":
            start_date, end_date = season_start, today
        elif selected_preset == "Last 30 Days":
            start_date, end_date = today - timedelta(days=30), today
        elif selected_preset == "Last 7 Days":
            start_date, end_date = today - timedelta(days=7), today
        else:
            start_date = st.date_input("Start date", value=season_start)
            end_date = st.date_input("End date", value=today)

        st.markdown("---")
        conference_options: list[str] = ["All"] + sorted(available_conferences)
        selected_conf_raw: str = st.selectbox(
            "Conference (away team)",
            options=conference_options,
            index=0,
        )
        selected_conferences: list[str] = (
            available_conferences
            if selected_conf_raw == "All"
            else [selected_conf_raw]
        )

        st.markdown("---")
        st.caption(f"Data as of: {today}")
        st.caption(f"Prod go-live: {PROD_GO_LIVE_DATE}")
        st.caption("P&L assumes -110 juice")

    return SidebarFilters(
        start_date=start_date,
        end_date=end_date,
        selected_conferences=selected_conferences,
    )


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    st.title("NCAAB Away Revenge")

    with st.spinner("Loading data from S3…"):
        all_plays_raw: pd.DataFrame = load_all_plays()
        todays_raw: pd.DataFrame | None = load_todays_plays()

    if all_plays_raw.empty:
        st.warning("No plays data found in S3.")
        st.stop()

    all_settled: pd.DataFrame = add_derived_columns(settle_plays(all_plays_raw))
    all_bet_plays: pd.DataFrame = filter_to_bet_plays(all_settled)

    available_conferences: list[str] = sorted(
        all_bet_plays["away_conference"].dropna().unique().tolist()
    )
    filters: SidebarFilters = build_sidebar(available_conferences)

    tab1, tab2 = st.tabs(["Plays", "Backtest"])

    # ── Tab 1: This Season (Today's Plays + Season Record) ────────────────────
    with tab1:
        st.subheader("Today's Plays")
        if date.today() < NCAAB_PAUSE_UNTIL:
            st.info(
                f"NCAAB season is on break. "
                f"Strategy resumes {NCAAB_PAUSE_UNTIL.strftime('%B %d, %Y')}."
            )
        elif todays_raw is None:
            st.info("Lambda runs ~9am ET. Check back for today's plays.")
        else:
            todays_enriched: pd.DataFrame = add_derived_columns(todays_raw)
            render_todays_plays_table(todays_enriched)

        render_yesterday_results(all_bet_plays)

        st.markdown("---")
        st.subheader("Season Record")
        st.caption("Sidebar filters (date range + conference) apply to this section.")

        date_filtered: pd.DataFrame = apply_date_filter(
            all_bet_plays, filters.start_date, filters.end_date
        )
        filtered: pd.DataFrame = apply_conference_filter(
            date_filtered, filters.selected_conferences
        )
        settled: pd.DataFrame = settled_only(filtered)

        if settled.empty:
            st.info("No settled plays found for the selected filters.")
        else:
            kpis = compute_kpis(settled)
            render_kpi_strip(kpis)
            st.markdown("---")

            chart_col, table_col = st.columns([2, 1])
            with chart_col:
                st.plotly_chart(build_cumulative_pnl_chart(settled), use_container_width=True)
            with table_col:
                st.markdown("**Monthly Breakdown**")
                render_monthly_breakdown(settled)

            st.markdown("---")
            st.subheader("Plays History")
            render_plays_history(filtered, "season_record")

            with st.expander("Conference Breakdown (away team)"):
                render_conference_breakdown(settled)

            with st.expander("Spread Distribution"):
                st.plotly_chart(build_spread_histogram(filtered), use_container_width=True)

    # ── Tab 2: Backtest ────────────────────────────────────────────────────────
    with tab2:
        side: str = st.radio(
            "Focal team side",
            ["Away", "Home", "All"],
            index=0,
            horizontal=True,
            help="Away matches production — Lambda only bets when focal team is away.",
        )

        with st.spinner("Loading backtest data…"):
            bt_raw: pd.DataFrame = load_backtest_multi()

        if bt_raw.empty:
            st.warning("No backtest data found in S3.")
        else:
            df_side: pd.DataFrame = filter_by_side(bt_raw, side)
            bt_settled: pd.DataFrame = bt_settled_rows(df_side)

            if bt_settled.empty:
                st.info("No settled plays for the selected filter.")
            else:
                st.subheader(f"Overall — Focal {side} · 6 Seasons")
                bt_kpis = bt_compute_kpis(bt_settled)
                render_kpi_strip(bt_kpis)
                st.plotly_chart(build_backtest_pnl_chart(bt_settled), use_container_width=True)

                st.markdown("---")
                st.subheader("Per-Season Summary")
                st.caption("2025-26 highlighted (deployed season).")
                render_per_season_table(df_side)

                st.markdown("---")
                st.subheader("Season Detail")
                render_backtest_season_detail(df_side)


if __name__ == "__main__":
    main()

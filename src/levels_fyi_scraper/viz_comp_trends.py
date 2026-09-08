"""
Data visualization for Levels.fyi overview data.
Run: uv run python analysis/levels_scraper/viz.py
Opens charts in browser via Plotly.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

DATA = Path(__file__).parent / "data" / "levels_overview_daily.parquet"
PLOTS = Path(__file__).parent / "plots"
PLOTS.mkdir(exist_ok=True)


def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    df["scrape_date"] = pd.to_datetime(df["scrape_date"])
    return df


# ── Chart 1: Median TC snapshot (latest day) ─────────────────────────────────

def chart_tc_snapshot(df: pd.DataFrame):
    latest = df[df["scrape_date"] == df["scrape_date"].max()].copy()
    latest = latest.sort_values("median_tc", ascending=True)

    fig = go.Figure(go.Bar(
        x=latest["median_tc"] / 1000,
        y=latest["company_slug"],
        orientation="h",
        text=[f"${v/1000:.0f}K" for v in latest["median_tc"]],
        textposition="outside",
        marker_color=[
            "#e63946" if s == "snap" else "#457b9d"
            for s in latest["company_slug"]
        ],
    ))
    fig.update_layout(
        title="Median Total Compensation by Company (latest snapshot)",
        xaxis_title="Median TC ($K)",
        yaxis_title=None,
        height=700,
        margin=dict(l=120, r=80),
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#eee"),
    )
    path = PLOTS / "01_tc_snapshot.html"
    fig.write_html(str(path))
    print(f"Saved: {path}")
    fig.show()


# ── Chart 2: Submissions by company ──────────────────────────────────────────

def chart_submissions_snapshot(df: pd.DataFrame):
    latest = df[df["scrape_date"] == df["scrape_date"].max()].copy()
    latest = latest.sort_values("total_submissions", ascending=True)

    fig = go.Figure(go.Bar(
        x=latest["total_submissions"],
        y=latest["company_slug"],
        orientation="h",
        text=latest["total_submissions"],
        textposition="outside",
        marker_color=[
            "#e63946" if s == "snap" else "#2a9d8f"
            for s in latest["company_slug"]
        ],
    ))
    fig.update_layout(
        title="Total Salary Submissions on Levels.fyi (latest snapshot)",
        xaxis_title="Submissions",
        yaxis_title=None,
        height=700,
        margin=dict(l=120, r=80),
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#eee"),
    )
    path = PLOTS / "02_submissions_snapshot.html"
    fig.write_html(str(path))
    print(f"Saved: {path}")
    fig.show()


# ── Chart 3: TC vs Submissions scatter ───────────────────────────────────────

def chart_tc_vs_subs(df: pd.DataFrame):
    latest = df[df["scrape_date"] == df["scrape_date"].max()].copy()

    fig = px.scatter(
        latest,
        x="total_submissions",
        y="median_tc",
        text="company_slug",
        size="total_submissions",
        size_max=40,
        color="median_tc",
        color_continuous_scale="Blues",
        log_x=True,
    )
    fig.update_traces(textposition="top center", marker_line_color="white", marker_line_width=1)
    fig.update_layout(
        title="Comp Quality vs Submission Volume (log scale)",
        xaxis_title="Total Submissions (log scale)",
        yaxis_title="Median TC ($)",
        height=600,
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#eee"),
        yaxis=dict(gridcolor="#eee"),
        showlegend=False,
        coloraxis_showscale=False,
    )
    path = PLOTS / "03_tc_vs_subs.html"
    fig.write_html(str(path))
    print(f"Saved: {path}")
    fig.show()


# ── Chart 4: Delta over the days we have ─────────────────────────────────────

def chart_deltas(df: pd.DataFrame):
    if df["scrape_date"].nunique() < 2:
        print("Need at least 2 days for delta chart — skipping")
        return

    dates = sorted(df["scrape_date"].unique())
    first, last = dates[0], dates[-1]

    t1 = df[df["scrape_date"] == first].set_index("company_slug")
    t2 = df[df["scrape_date"] == last].set_index("company_slug")
    common = t1.index.intersection(t2.index)

    delta = pd.DataFrame({
        "subs_delta": t2.loc[common, "total_submissions"] - t1.loc[common, "total_submissions"],
        "tc_delta":   t2.loc[common, "median_tc"] - t1.loc[common, "median_tc"],
    }).reset_index().rename(columns={"index": "company_slug"})
    delta = delta.sort_values("subs_delta")

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Submission Count Change", "Median TC Change ($)"])

    colors_subs = ["#e63946" if v < 0 else "#2a9d8f" for v in delta["subs_delta"]]
    colors_tc   = ["#e63946" if v < 0 else "#2a9d8f" for v in delta["tc_delta"]]

    fig.add_trace(go.Bar(
        x=delta["subs_delta"], y=delta["company_slug"],
        orientation="h", marker_color=colors_subs, name="subs"
    ), row=1, col=1)

    delta_tc = delta.sort_values("tc_delta")
    fig.add_trace(go.Bar(
        x=delta_tc["tc_delta"] / 1000, y=delta_tc["company_slug"],
        orientation="h", marker_color=colors_tc, name="tc"
    ), row=1, col=2)

    fig.update_layout(
        title=f"Changes: {first.date()} → {last.date()}",
        height=700, showlegend=False,
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#eee")

    path = PLOTS / "04_deltas.html"
    fig.write_html(str(path))
    print(f"Saved: {path}")
    fig.show()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load()
    print(f"Loaded {len(df)} rows, {df['scrape_date'].nunique()} days, {df['company_slug'].nunique()} companies\n")

    chart_tc_snapshot(df)
    chart_submissions_snapshot(df)
    chart_tc_vs_subs(df)
    chart_deltas(df)

    print("\nDone. All charts saved to analysis/levels_scraper/plots/")

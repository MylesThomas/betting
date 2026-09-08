"""
Levels.fyi comp data vs public market analysis.
4 charts comparing compensation metrics to market cap and stock returns.

Run: uv run python analysis/levels_scraper/viz_market.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

DATA  = Path(__file__).parent / "data" / "levels_overview_daily.parquet"
PLOTS = Path(__file__).parent / "plots"
PLOTS.mkdir(exist_ok=True)

TICKER_MAP = {
    "snap":        "SNAP",
    "meta":        "META",
    "google":      "GOOGL",
    "microsoft":   "MSFT",
    "nvidia":      "NVDA",
    "apple":       "AAPL",
    "amazon":      "AMZN",
    "netflix":     "NFLX",
    "uber":        "UBER",
    "lyft":        "LYFT",
    "airbnb":      "ABNB",
    "salesforce":  "CRM",
    "adobe":       "ADBE",
    "oracle":      "ORCL",
    "intel":       "INTC",
    "amd":         "AMD",
    "qualcomm":    "QCOM",
    "pinterest":   "PINS",
    "reddit":      "RDDT",
    "coinbase":    "COIN",
    "palantir":    "PLTR",
    "snowflake":   "SNOW",
    "cloudflare":  "NET",
}


# ── data loading ──────────────────────────────────────────────────────────────

def load_levels() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    df["scrape_date"] = pd.to_datetime(df["scrape_date"])
    # Use latest snapshot only
    latest = df[df["scrape_date"] == df["scrape_date"].max()]
    return latest[["company_slug", "total_submissions", "median_tc"]].copy()


def load_market(tickers: list[str]) -> pd.DataFrame:
    print("Fetching market data from yfinance...")
    rows = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            hist_max = yf.Ticker(ticker).history(period="max")
            if hist_max.empty:
                continue

            current = hist_max.iloc[-1]["Close"]

            # YTD
            ytd_rows = hist_max[hist_max.index.year == hist_max.index[-1].year]
            ytd_return = (current / ytd_rows.iloc[0]["Close"] - 1) * 100

            # 1yr, 3yr, 5yr (or NaN if not enough history)
            def period_return(years):
                cutoff = hist_max.index[-1] - pd.DateOffset(years=years)
                sub = hist_max[hist_max.index >= cutoff]
                return (current / sub.iloc[0]["Close"] - 1) * 100 if len(sub) > 20 else None

            # IPO → now (full history)
            ipo_date   = hist_max.index[0].date()
            ipo_return = (current / hist_max.iloc[0]["Close"] - 1) * 100

            rows.append({
                "ticker":         ticker,
                "market_cap":     info.get("marketCap"),
                "employees":      info.get("fullTimeEmployees"),
                "pe_ratio":       info.get("trailingPE"),
                "ps_ratio":       info.get("priceToSalesTrailing12Months"),
                "revenue":        info.get("totalRevenue"),
                "current_price":  round(current, 2),
                "ipo_date":       str(ipo_date),
                "ytd_return":     round(ytd_return, 2),
                "yr1_return":     round(period_return(1), 2) if period_return(1) else None,
                "yr3_return":     round(period_return(3), 2) if period_return(3) else None,
                "yr5_return":     round(period_return(5), 2) if period_return(5) else None,
                "ipo_return":     round(ipo_return, 2),
            })
            print(f"  {ticker:<6} mktcap=${info.get('marketCap',0)/1e9:.0f}B  "
                  f"ytd={ytd_return:+.1f}%  ipo({ipo_date})={ipo_return:+.0f}%")
        except Exception as e:
            print(f"  {ticker:<6} error: {e}")

    return pd.DataFrame(rows)


def build_dataset() -> pd.DataFrame:
    levels = load_levels()
    levels["ticker"] = levels["company_slug"].map(TICKER_MAP)

    public = levels[levels["ticker"].notna()].copy()
    tickers = public["ticker"].tolist()

    market = load_market(tickers)
    df = public.merge(market, on="ticker", how="inner")

    # Derived fields
    median_tc_median = df["median_tc"].median()
    df["comp_premium_pct"] = (df["median_tc"] - median_tc_median) / median_tc_median * 100
    df["comp_rank"] = df["median_tc"].rank(ascending=True).astype(int)
    df["market_cap_B"] = df["market_cap"] / 1e9
    df["subs_per_1k_employees"] = df["total_submissions"] / (df["employees"] / 1000)
    df["marketcap_per_sub_M"] = df["market_cap"] / df["total_submissions"] / 1e6
    df["label"] = df["ticker"]
    df["ipo_return"] = pd.to_numeric(df["ipo_return"], errors="coerce")

    return df


# ── Chart 5: TC vs Market Cap ─────────────────────────────────────────────────

def chart_tc_vs_marketcap(df: pd.DataFrame):
    fig = go.Figure()

    for _, row in df.iterrows():
        is_snap = row["company_slug"] == "snap"
        fig.add_trace(go.Scatter(
            x=[row["median_tc"] / 1000],
            y=[row["market_cap_B"]],
            mode="markers+text",
            text=[row["ticker"]],
            textposition="top center",
            marker=dict(
                size=max(8, min(50, row["total_submissions"] ** 0.45)),
                color="#e63946" if is_snap else "#457b9d",
                opacity=0.85,
                line=dict(width=1, color="white"),
            ),
            name=row["ticker"],
            showlegend=False,
            hovertemplate=(
                f"<b>{row['ticker']}</b><br>"
                f"Median TC: ${row['median_tc']/1000:.0f}K<br>"
                f"Market Cap: ${row['market_cap_B']:.0f}B<br>"
                f"Submissions: {row['total_submissions']}<br>"
                f"YTD Return: {row['ytd_return']:+.1f}%"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Median TC vs Market Cap<br><sup>Bubble size = submission count · Red = Snap</sup>",
        xaxis_title="Median Total Compensation ($K)",
        yaxis_title="Market Cap ($B)",
        height=600,
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#eee"),
        yaxis=dict(gridcolor="#eee", type="log"),
    )
    path = PLOTS / "05_tc_vs_marketcap.html"
    fig.write_html(str(path))
    print(f"\nSaved: {path}")
    fig.show()


# ── Chart 6: Comp Premium vs Stock Return ─────────────────────────────────────

def chart_comp_premium_vs_return(df: pd.DataFrame):
    periods = [
        ("ytd_return",  "YTD"),
        ("yr1_return",  "1-Year"),
        ("yr3_return",  "3-Year"),
        ("yr5_return",  "5-Year"),
        ("ipo_return",  "IPO → Today"),
    ]

    fig = make_subplots(
        rows=1, cols=len(periods),
        subplot_titles=[p[1] for p in periods],
    )

    for col, (ret_col, label) in enumerate(periods, start=1):
        valid = df[["comp_premium_pct", ret_col, "ticker", "company_slug"]].dropna()
        if valid.empty:
            continue

        # Regression line
        if len(valid) > 2:
            m, b = np.polyfit(valid["comp_premium_pct"], valid[ret_col], 1)
            x_line = np.linspace(valid["comp_premium_pct"].min(), valid["comp_premium_pct"].max(), 50)
            fig.add_trace(go.Scatter(
                x=x_line, y=m * x_line + b,
                mode="lines",
                line=dict(color="#aaa", dash="dash", width=1),
                showlegend=False,
            ), row=1, col=col)

        for _, row in valid.iterrows():
            is_snap = row["company_slug"] == "snap"
            fig.add_trace(go.Scatter(
                x=[row["comp_premium_pct"]],
                y=[row[ret_col]],
                mode="markers+text",
                text=[row["ticker"]],
                textposition="top center",
                marker=dict(
                    size=10,
                    color="#e63946" if is_snap else "#457b9d",
                    line=dict(width=1, color="white"),
                ),
                showlegend=False,
                hovertemplate=(
                    f"<b>{row['ticker']}</b><br>"
                    f"Pay vs peers: {row['comp_premium_pct']:+.1f}%<br>"
                    f"{label} return: {row[ret_col]:+.1f}%"
                    "<extra></extra>"
                ),
            ), row=1, col=col)

        fig.add_vline(x=0, line_dash="dot", line_color="#ccc", row=1, col=col)

    fig.update_xaxes(title_text="Pay vs peer median (%)", gridcolor="#eee", zeroline=False)
    fig.update_yaxes(title_text="Stock return (%)", gridcolor="#eee")
    fig.update_layout(
        title=(
            "Does paying above-average comp lead to better stock returns?<br>"
            "<sup>'Pay vs peer median' = how far above/below the group median salary this company sits. "
            "0 = exactly at median. +50% = pays 50% more than average. Red = Snap.</sup>"
        ),
        height=520,
        width=1600,
        plot_bgcolor="white",
    )
    path = PLOTS / "06_comp_premium_vs_return.html"
    fig.write_html(str(path))
    print(f"Saved: {path}")
    fig.show()


# ── Chart 7: Submissions per Employee vs Stock Return ─────────────────────────

def chart_subs_per_employee(df: pd.DataFrame):
    valid = df[df["subs_per_1k_employees"].notna() & df["employees"].notna() & df["market_cap_B"].notna()].copy()
    valid = valid.dropna(subset=["ytd_return", "subs_per_1k_employees", "market_cap_B"])

    fig = px.scatter(
        valid,
        x="subs_per_1k_employees",
        y="ytd_return",
        text="ticker",
        size="market_cap_B",
        size_max=45,
        color="comp_premium_pct",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        hover_data={"ticker": True, "employees": True, "total_submissions": True},
    )

    # Regression
    m, b = np.polyfit(valid["subs_per_1k_employees"], valid["ytd_return"], 1)
    x_line = np.linspace(valid["subs_per_1k_employees"].min(), valid["subs_per_1k_employees"].max(), 50)
    fig.add_trace(go.Scatter(
        x=x_line, y=m * x_line + b,
        mode="lines", line=dict(color="#aaa", dash="dash", width=1),
        showlegend=False,
    ))

    fig.update_traces(textposition="top center", selector=dict(mode="markers+text"))
    fig.update_layout(
        title="How engaged are employees with sharing comp data — and does it predict returns?<br>"
              "<sup>x = Levels.fyi submissions per 1,000 employees (higher = more transparent culture) · "
              "bubble size = market cap · color = whether company pays above (blue) or below (red) peer median · "
              "employee counts from yfinance, may be approximate</sup>",
        xaxis_title="Submissions per 1,000 Employees",
        yaxis_title="YTD Return (%)",
        height=600,
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#eee"),
        yaxis=dict(gridcolor="#eee"),
        coloraxis_colorbar=dict(title="Comp<br>Premium %"),
    )
    path = PLOTS / "07_subs_per_employee_vs_return.html"
    fig.write_html(str(path))
    print(f"Saved: {path}")
    fig.show()


# ── Chart 8: Market Cap per Submission ────────────────────────────────────────

def chart_marketcap_per_sub(df: pd.DataFrame):
    valid = df[df["marketcap_per_sub_M"].notna()].sort_values("marketcap_per_sub_M", ascending=True)

    # Quartile color by median TC
    tc_25, tc_75 = valid["median_tc"].quantile(0.25), valid["median_tc"].quantile(0.75)
    colors = []
    for tc in valid["median_tc"]:
        if tc >= tc_75:
            colors.append("#2a9d8f")   # high comp — green
        elif tc <= tc_25:
            colors.append("#e63946")   # low comp — red
        else:
            colors.append("#457b9d")   # mid — blue

    fig = go.Figure(go.Bar(
        x=valid["marketcap_per_sub_M"],
        y=valid["ticker"],
        orientation="h",
        marker_color=colors,
        text=[f"${v:.0f}M" for v in valid["marketcap_per_sub_M"]],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Market cap per submission: $%{x:.0f}M<br>"
            "<extra></extra>"
        ),
    ))

    fig.add_annotation(
        text="Green = top TC quartile · Red = bottom TC quartile",
        xref="paper", yref="paper", x=1, y=-0.07,
        showarrow=False, font=dict(size=11, color="#666"), xanchor="right",
    )
    fig.update_layout(
        title="Market Cap per Levels.fyi Submission ($M)<br>"
              "<sup>How much market value per unit of visible talent data</sup>",
        xaxis_title="Market Cap per Submission ($M)",
        yaxis_title=None,
        height=650,
        margin=dict(l=80, r=120),
        plot_bgcolor="white",
        xaxis=dict(gridcolor="#eee"),
    )
    path = PLOTS / "08_marketcap_per_sub.html"
    fig.write_html(str(path))
    print(f"Saved: {path}")
    fig.show()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = build_dataset()
    print(f"\nDataset: {len(df)} public companies with market data\n")
    print(df[["ticker", "median_tc", "market_cap_B", "ytd_return",
              "total_submissions", "employees", "comp_premium_pct"]].round(1).to_string(index=False))
    print()

    chart_tc_vs_marketcap(df)
    chart_comp_premium_vs_return(df)
    chart_subs_per_employee(df)
    chart_marketcap_per_sub(df)

    print("\nDone. Charts saved to analysis/levels_scraper/plots/")

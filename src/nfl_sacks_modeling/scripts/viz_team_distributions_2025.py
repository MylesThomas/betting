"""
Two HTML viz sections for all 32 NFL teams (2025 REG season):
  1. Spread distribution (team perspective, 17 games each)
  2. Implied team points distribution

Implied points = (game_total - team_spread) / 2
  (positive spread = underdog → fewer implied points; negative = favorite → more)

Teams arranged by conference/division (4 cols × 8 rows).
Color-coded by nflverse primary/secondary team colors.

Output: ~/Downloads/tmp/nfl_team_distributions_2025.html

Run:
  python nfl_sacks_modeling/scripts/viz_team_distributions_2025.py
"""

import base64
import glob
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nfl_data_py as nfl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LINES_DIR = Path.home() / "Downloads" / "tmp" / "nfl_game_lines" / "2025"
OUT_HTML  = Path.home() / "Downloads" / "tmp" / "nfl_team_distributions_2025.html"

TEAM_NAME_MAP = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

TEAM_ORDER = [
    "BUF", "MIA", "NE",  "NYJ",
    "BAL", "CIN", "CLE", "PIT",
    "HOU", "IND", "JAX", "TEN",
    "DEN", "KC",  "LAC", "LV",
    "DAL", "NYG", "PHI", "WAS",
    "CHI", "DET", "GB",  "MIN",
    "ATL", "CAR", "NO",  "TB",
    "ARI", "LA",  "SEA", "SF",
]

DIVISION_LABELS = [
    "AFC East", "AFC North", "AFC South", "AFC West",
    "NFC East", "NFC North", "NFC South", "NFC West",
]

DIVISION_MAP = {
    "BUF": "AFC East",  "MIA": "AFC East",  "NE":  "AFC East",  "NYJ": "AFC East",
    "BAL": "AFC North", "CIN": "AFC North", "CLE": "AFC North", "PIT": "AFC North",
    "HOU": "AFC South", "IND": "AFC South", "JAX": "AFC South", "TEN": "AFC South",
    "DEN": "AFC West",  "KC":  "AFC West",  "LAC": "AFC West",  "LV":  "AFC West",
    "DAL": "NFC East",  "NYG": "NFC East",  "PHI": "NFC East",  "WAS": "NFC East",
    "CHI": "NFC North", "DET": "NFC North", "GB":  "NFC North", "MIN": "NFC North",
    "ATL": "NFC South", "CAR": "NFC South", "NO":  "NFC South", "TB":  "NFC South",
    "ARI": "NFC West",  "LA":  "NFC West",  "SEA": "NFC West",  "SF":  "NFC West",
}

NROWS, NCOLS = 8, 4


# ── Load data ──────────────────────────────────────────────────────────────────

def load_team_data() -> pd.DataFrame:
    files = glob.glob(str(LINES_DIR / "*.parquet"))
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    totals = (raw[(raw["market"] == "totals") & (raw["outcome_name"] == "Over")]
              .groupby(["nfl_game_id", "week"])["point"]
              .median()
              .reset_index()
              .rename(columns={"point": "game_total"}))

    sp = raw[raw["market"] == "spreads"].copy()
    sp["team_abbr"] = sp["outcome_name"].map(TEAM_NAME_MAP)
    sp = sp[sp["team_abbr"].notna()]
    spreads = (sp.groupby(["nfl_game_id", "week", "team_abbr"])["point"]
                 .median()
                 .reset_index()
                 .rename(columns={"point": "team_spread"}))

    df = spreads.merge(totals, on=["nfl_game_id", "week"], how="left")
    df["implied_pts"] = (df["game_total"] - df["team_spread"]) / 2
    return df


def load_team_colors() -> dict:
    td = nfl.import_team_desc()
    td = td[td["team_abbr"].isin(TEAM_ORDER)]
    return {
        row["team_abbr"]: {"primary": row["team_color"], "secondary": row["team_color2"]}
        for _, row in td.iterrows()
    }


# ── Pivot table HTML ───────────────────────────────────────────────────────────

def build_pivot_table_html(df: pd.DataFrame, metric: str, colors: dict,
                           fmt: str = ".1f") -> str:
    pivot = (df.pivot_table(index="team_abbr", columns="week", values=metric, aggfunc="median")
               .reindex(TEAM_ORDER))
    pivot.columns = [f"W{int(w)}" for w in pivot.columns]
    pivot.insert(0, "Division", [DIVISION_MAP.get(t, "—") for t in pivot.index])
    pivot["Median"] = pivot[[c for c in pivot.columns if c.startswith("W")]].median(axis=1)
    pivot["Min"]    = pivot[[c for c in pivot.columns if c.startswith("W")]].min(axis=1)
    pivot["Max"]    = pivot[[c for c in pivot.columns if c.startswith("W")]].max(axis=1)

    week_cols = [c for c in pivot.columns if c.startswith("W")]
    stat_cols = ["Median", "Min", "Max"]
    all_vals  = df[metric].dropna()
    v_min, v_max = all_vals.min(), all_vals.max()

    def cell_bg(val, primary):
        if pd.isna(val):
            return ""
        pct = (val - v_min) / (v_max - v_min) if v_max != v_min else 0.5
        h = primary.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        alpha = 0.08 + pct * 0.30
        return f"background:rgba({r},{g},{b},{alpha:.2f})"

    rows_html = ""
    for i, team in enumerate(TEAM_ORDER):
        if team not in pivot.index:
            continue
        row = pivot.loc[team]
        c   = colors.get(team, {"primary": "#888888", "secondary": "#cccccc"})
        pri, sec = c["primary"], c["secondary"]
        border = "border-bottom:2px solid #aaa;" if (i + 1) % 4 == 0 else ""

        team_cell = (f'<td style="font-weight:bold;background:{pri};color:white;'
                     f'border-right:2px solid {sec};white-space:nowrap;'
                     f'padding:4px 8px;{border}">{team}</td>')
        div_cell  = f'<td style="color:#555;white-space:nowrap;padding:4px 8px;{border}">{row["Division"]}</td>'

        week_cells = "".join(
            f'<td style="text-align:right;{cell_bg(row.get(wc, float("nan")), pri)};'
            f'padding:3px 6px;{border}">'
            f'{"—" if pd.isna(row.get(wc)) else f"{row[wc]:{fmt}}"}</td>'
            for wc in week_cols
        )
        stat_cells = "".join(
            f'<td style="text-align:right;border-left:1px solid #ccc;'
            f'{"font-weight:bold;" if sc == "Median" else ""}'
            f'padding:3px 6px;{border}">'
            f'{"—" if pd.isna(row.get(sc)) else f"{row[sc]:{fmt}}"}</td>'
            for sc in stat_cols
        )
        rows_html += f"<tr>{team_cell}{div_cell}{week_cells}{stat_cells}</tr>\n"

    header = (
        '<th style="text-align:left;padding:4px 8px">Team</th>'
        '<th style="text-align:left;padding:4px 8px">Division</th>'
        + "".join(f'<th style="text-align:right;padding:3px 6px">{wc}</th>' for wc in week_cols)
        + "".join(f'<th style="text-align:right;border-left:1px solid #666;padding:3px 6px">{sc}</th>'
                  for sc in stat_cols)
    )

    return f"""
<div style="overflow-x:auto;margin-bottom:24px;">
  <table style="border-collapse:collapse;font-family:monospace;font-size:12px;width:100%">
    <thead><tr style="background:#222;color:white">{header}</tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>"""


# ── Matplotlib strip plot (dots on a single line, no y-axis) ──────────────────

def make_chart_b64(df: pd.DataFrame, metric: str, colors: dict,
                   xlabel: str, vline: float | None = None) -> str:

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(22, NROWS * 1.4))
    fig.patch.set_facecolor("white")

    for idx, team in enumerate(TEAM_ORDER):
        row, col = divmod(idx, NCOLS)
        ax = axes[row][col]

        vals = df[df["team_abbr"] == team][metric].dropna().values
        c    = colors.get(team, {"primary": "#888888", "secondary": "#444444"})
        pri, sec = c["primary"], c["secondary"]

        ax.set_facecolor("#f5f5f5")
        for spine in ax.spines.values():
            spine.set_visible(False)

        if len(vals):
            ax.scatter(vals, np.zeros(len(vals)), color=pri, edgecolors=sec,
                       linewidths=0.6, s=42, zorder=3, alpha=0.85, clip_on=False)

            med = float(np.median(vals))
            ax.axvline(med, color=sec, linewidth=2.0, linestyle="--", zorder=2)

            q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
            ax.axvspan(q1, q3, alpha=0.18, color=pri, zorder=1)

        if vline is not None:
            ax.axvline(vline, color="#aaa", linewidth=0.9, linestyle="-", zorder=1)

        ax.set_ylim(-0.5, 0.5)
        ax.yaxis.set_visible(False)
        ax.tick_params(axis="x", labelsize=7, colors="#444")
        ax.set_title(team, fontsize=10, fontweight="bold", color=pri, pad=2)

        if col == 0:
            ax.text(-0.16, 0.5, DIVISION_LABELS[row], transform=ax.transAxes,
                    fontsize=8, color="#666", va="center", ha="right", rotation=90)

    fig.text(0.5, 0.0, xlabel, ha="center", fontsize=10, color="#444")
    fig.tight_layout(rect=[0, 0.02, 1, 1])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Plotly box plots (interactive, embedded JS) ────────────────────────────────

def make_plotly_html(df: pd.DataFrame, metric: str, colors: dict,
                     title: str, vline: float | None = None,
                     include_js: bool = False) -> str:

    fig = make_subplots(
        rows=NROWS, cols=NCOLS,
        subplot_titles=TEAM_ORDER,
        horizontal_spacing=0.04,
        vertical_spacing=0.06,
    )

    for idx, team in enumerate(TEAM_ORDER):
        row, col = divmod(idx, NCOLS)
        c   = colors.get(team, {"primary": "#888888", "secondary": "#cccccc"})
        pri, sec = c["primary"], c["secondary"]

        # rgba fill
        h = pri.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        fill = f"rgba({r},{g},{b},0.35)"

        vals = df[df["team_abbr"] == team][metric].dropna().tolist()

        fig.add_trace(go.Box(
            x=vals,
            name=team,
            orientation="h",
            marker=dict(color=pri, size=5,
                        line=dict(color=sec, width=1)),
            line=dict(color=sec, width=1.5),
            fillcolor=fill,
            boxpoints="all",
            jitter=0.5,
            pointpos=0,
            showlegend=False,
            hovertemplate=f"<b>{team}</b> %{{x:.1f}}<extra></extra>",
        ), row=row + 1, col=col + 1)

        if vline is not None:
            fig.add_vline(x=vline, line=dict(color="rgba(0,0,0,0.2)", width=1),
                          row=row + 1, col=col + 1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=15), x=0.5),
        height=NROWS * 140 + 60,
        width=1300,
        paper_bgcolor="white",
        plot_bgcolor="#f5f5f5",
        margin=dict(l=50, r=10, t=60, b=30),
    )
    for i in range(1, NROWS * NCOLS + 1):
        xk = f"xaxis{i}" if i > 1 else "xaxis"
        yk = f"yaxis{i}" if i > 1 else "yaxis"
        fig.update_layout(**{
            yk: dict(showticklabels=False, showgrid=False, zeroline=False),
            xk: dict(showgrid=True, gridcolor="#e0e0e0", zeroline=False,
                     tickfont=dict(size=8)),
        })
    for ann in fig.layout.annotations:
        ann.font = dict(size=10)

    return fig.to_html(full_html=False, include_plotlyjs="cdn" if include_js else False)


# ── Plotly binned histogram ────────────────────────────────────────────────────

SPREAD_BINS   = [-np.inf, -15, -10, -7, -3, 0, 3, 7, 10, 15, np.inf]
SPREAD_LABELS = ["≤-15", "-15:-10", "-10:-7", "-7:-3", "-3:0",
                 "0:3",  "3:7",    "7:10",   "10:15", "15+"]

PTS_BINS   = [-np.inf, 5.5, 10.5, 15.5, 20.5, 25.5, 30.5, 35.5, 40.5, np.inf]
PTS_LABELS = ["0-5", "6-10", "11-15", "16-20", "21-25",
              "26-30", "31-35", "36-40", "41+"]


def make_plotly_histogram_html(df: pd.DataFrame, metric: str, colors: dict,
                               bins: list, bin_labels: list, title: str,
                               zero_bin_idx: int | None = None,
                               include_js: bool = False) -> str:

    fig = make_subplots(
        rows=NROWS, cols=NCOLS,
        subplot_titles=TEAM_ORDER,
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )

    for idx, team in enumerate(TEAM_ORDER):
        row, col = divmod(idx, NCOLS)
        c   = colors.get(team, {"primary": "#888888", "secondary": "#cccccc"})
        pri, sec = c["primary"], c["secondary"]
        h = pri.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

        vals  = df[df["team_abbr"] == team][metric].dropna().values
        counts, _ = np.histogram(vals, bins=bins)

        # Color each bar: bins left of zero slightly dimmer, right of zero brighter
        bar_colors = []
        for bi in range(len(bin_labels)):
            if zero_bin_idx is not None and bi < zero_bin_idx:
                bar_colors.append(f"rgba({r},{g},{b},0.45)")
            else:
                bar_colors.append(f"rgba({r},{g},{b},0.85)")

        fig.add_trace(go.Bar(
            x=bin_labels,
            y=counts.tolist(),
            name=team,
            marker=dict(
                color=bar_colors,
                line=dict(color=sec, width=0.8),
            ),
            showlegend=False,
            hovertemplate=f"<b>{team}</b><br>%{{x}}: %{{y}} games<extra></extra>",
        ), row=row + 1, col=col + 1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=15), x=0.5),
        height=NROWS * 160 + 60,
        width=1300,
        paper_bgcolor="white",
        plot_bgcolor="#f5f5f5",
        margin=dict(l=50, r=10, t=60, b=30),
        bargap=0.08,
    )
    for i in range(1, NROWS * NCOLS + 1):
        xk = f"xaxis{i}" if i > 1 else "xaxis"
        yk = f"yaxis{i}" if i > 1 else "yaxis"
        fig.update_layout(**{
            yk: dict(showgrid=True, gridcolor="#e8e8e8", zeroline=False,
                     tickfont=dict(size=7), dtick=1),
            xk: dict(showgrid=False, zeroline=False,
                     tickfont=dict(size=7), tickangle=-40),
        })
    for ann in fig.layout.annotations:
        ann.font = dict(size=10)

    return fig.to_html(full_html=False, include_plotlyjs="cdn" if include_js else False)


# ── Player-level helpers ───────────────────────────────────────────────────────

JOINED_PATH = Path.home() / "Downloads" / "tmp" / "nfl_sacks_joined_2025.parquet"

POS_GROUP_MAP = {
    "DE": "DL", "DT": "DL", "NT": "DL",
    "OLB": "LB", "ILB": "LB", "MLB": "LB", "LB": "LB",
    "CB": "DB", "SS": "DB", "FS": "DB", "DB": "DB", "S": "DB",
}
POS_COLORS = {"DL": "#c0392b", "LB": "#2980b9", "DB": "#27ae60", "OTH": "#95a5a6"}


def units_on_win(price: float) -> float:
    return 100.0 / abs(price) if price < 0 else price / 100.0


def load_player_data() -> pd.DataFrame:
    df = pd.read_parquet(JOINED_PATH)
    df["pos_group"] = df["position"].map(POS_GROUP_MAP).fillna("OTH")
    df["is_over"]   = (df["sacks"] >= 1.0).astype(int)
    df["is_under"]  = (df["sacks"] == 0.0).astype(int)
    df["is_push"]   = (df["sacks"] == 0.5).astype(int)
    return df


# ── Viz 1: player calibration table ───────────────────────────────────────────

def build_player_calibration_html(df: pd.DataFrame, min_games: int = 5) -> str:
    sub = df[
        df["prop_median_line"].notna() &
        (df["defense_snaps"] > 0) &
        (df["prop_median_line"] == 0.5) &
        df["prop_median_price_over"].notna() &
        df["prop_median_price_under"].notna()
    ].copy()

    sub["over_bet_units"]  = sub["is_over"] * sub["prop_median_price_over"].apply(units_on_win) - sub["is_under"]
    sub["under_bet_units"] = sub["is_under"] * sub["prop_median_price_under"].apply(units_on_win) - sub["is_over"]

    agg = (sub.groupby(["player", "team", "pos_group"])
              .agg(
                  n=("is_over", "count"),
                  impl_over=("prop_median_impl_over", "mean"),
                  impl_under=("prop_median_impl_under", "mean"),
                  hit_over=("is_over", "mean"),
                  hit_under=("is_under", "mean"),
                  push_rate=("is_push", "mean"),
                  units_over=("over_bet_units", "sum"),
                  units_under=("under_bet_units", "sum"),
              )
              .reset_index())

    agg = agg[agg["n"] >= min_games].copy()
    agg["edge_over"]  = agg["hit_over"]  - agg["impl_over"]
    agg["edge_under"] = agg["hit_under"] - agg["impl_under"]
    agg = agg.sort_values("units_under", ascending=False).reset_index(drop=True)

    cols = ["player", "team", "pos_group", "n",
            "impl_over", "hit_over", "edge_over", "units_over",
            "impl_under", "hit_under", "edge_under", "units_under"]
    headers = ["Player", "Team", "Pos", "N",
               "Impl Over", "Hit% Over", "Edge Over", "Units Over",
               "Impl Under", "Hit% Under", "Edge Under", "Units Under"]

    def fmt(col, val):
        if col in ("impl_over", "hit_over", "edge_over", "impl_under", "hit_under", "edge_under"):
            color = ""
            if col.startswith("edge"):
                color = f'color:{"green" if val > 0 else "red"};font-weight:bold;'
            return f'<span style="{color}">{val:+.1%}</span>' if col.startswith("edge") else f"{val:.1%}"
        if col.startswith("units"):
            color = "green" if val > 0 else "red"
            return f'<span style="color:{color};font-weight:bold">{val:+.2f}</span>'
        if col == "n":
            return str(int(val))
        return str(val)

    header_html = "".join(f"<th>{h}</th>" for h in headers)
    rows_html = ""
    for _, row in agg.iterrows():
        cells = "".join(f"<td>{fmt(c, row[c])}</td>" for c in cols)
        bg = "background:rgba(44,160,44,0.07)" if row["units_under"] > 0 else "background:rgba(214,39,40,0.05)"
        rows_html += f'<tr style="{bg}">{cells}</tr>\n'

    return f"""
<div style="overflow-x:auto;margin-bottom:16px;">
  <p style="font-family:monospace;font-size:11px;color:#888;margin:0 0 6px">
    Players with ≥{min_games} games with a 0.5 line posted. Sorted by Units Under desc.
    Edge = actual hit rate − market implied prob. Push = money back (0 units).
  </p>
  <table style="border-collapse:collapse;font-family:monospace;font-size:12px;width:100%">
    <thead><tr style="background:#222;color:white">{header_html}</tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>"""


# ── Viz 2: snap% vs sack hit rate scatter ──────────────────────────────────────

def make_snap_scatter_html(df: pd.DataFrame) -> str:
    sub = df[
        df["prop_median_line"].notna() &
        (df["defense_snaps"] > 0) &
        (df["prop_median_line"] == 0.5)
    ].copy()

    player_agg = (sub.groupby(["player", "pos_group"])
                     .agg(
                         n=("is_over", "count"),
                         avg_snap_pct=("defense_pct", "mean"),
                         sack_hit_rate=("is_over", "mean"),
                         avg_impl_over=("prop_median_impl_over", "mean"),
                     )
                     .reset_index())
    player_agg = player_agg[player_agg["n"] >= 3]

    traces = []
    for pg, grp in player_agg.groupby("pos_group"):
        color = POS_COLORS.get(pg, POS_COLORS["OTH"])
        traces.append(go.Scatter(
            x=grp["avg_snap_pct"],
            y=grp["sack_hit_rate"],
            mode="markers",
            name=pg,
            marker=dict(
                size=grp["n"] * 2.5 + 6,
                color=color,
                opacity=0.72,
                line=dict(color="white", width=0.6),
            ),
            customdata=np.stack([grp["player"], grp["n"],
                                 grp["avg_impl_over"], grp["pos_group"]], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[3]})<br>"
                "Avg snap%: %{x:.1%}<br>"
                "Sack hit rate: %{y:.1%}<br>"
                "Market impl over: %{customdata[2]:.1%}<br>"
                "N games: %{customdata[1]}<extra></extra>"
            ),
        ))

    # Reference line: y = x (perfect calibration)
    xs = np.linspace(0, 1, 100)
    traces.append(go.Scatter(
        x=xs, y=xs * 0.0 + player_agg["sack_hit_rate"].mean(),
        mode="lines", name=f"Avg hit rate ({player_agg['sack_hit_rate'].mean():.1%})",
        line=dict(color="#aaa", width=1.2, dash="dot"), showlegend=True,
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Snap% vs Sack Hit Rate — by player (size = n games with line, ≥3 games)",
        xaxis=dict(title="Avg defense snap% (player-level)", tickformat=".0%",
                   showgrid=True, gridcolor="#eee"),
        yaxis=dict(title="Actual sack hit rate (Over %)", tickformat=".0%",
                   showgrid=True, gridcolor="#eee"),
        legend=dict(title="Pos group"),
        height=520, width=900,
        paper_bgcolor="white", plot_bgcolor="#fafafa",
        margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ── Viz 3: cumulative sacks trajectory ─────────────────────────────────────────

def make_trajectory_html(df: pd.DataFrame, team_colors: dict,
                         top_n: int = 20) -> str:
    season_totals = (df.groupby(["player", "team"])["sacks"]
                       .sum()
                       .reset_index()
                       .sort_values("sacks", ascending=False)
                       .head(top_n))
    top_players = season_totals["player"].tolist()

    sub = df[df["player"].isin(top_players)].copy()
    sub = sub.sort_values(["player", "week"])
    sub["cumul_sacks"] = sub.groupby("player")["sacks"].cumsum()

    fig = go.Figure()
    for _, row in season_totals.iterrows():
        player = row["player"]
        team   = row["team"]
        total  = row["sacks"]
        c      = team_colors.get(team, {"primary": "#666666", "secondary": "#999999"})
        pdata  = sub[sub["player"] == player]

        fig.add_trace(go.Scatter(
            x=pdata["week"],
            y=pdata["cumul_sacks"],
            mode="lines+markers",
            name=f"{player} ({team}) — {total:.1f}",
            line=dict(color=c["primary"], width=2),
            marker=dict(color=c["primary"], size=5,
                        line=dict(color=c["secondary"], width=1)),
            hovertemplate=(
                f"<b>{player}</b><br>"
                "Week %{x}<br>"
                "Cumulative sacks: %{y:.1f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=f"Cumulative sacks by week — top {top_n} pass rushers (2025 REG)",
        xaxis=dict(title="Week", dtick=1, showgrid=True, gridcolor="#eee"),
        yaxis=dict(title="Cumulative sacks", showgrid=True, gridcolor="#eee"),
        legend=dict(title="Player (team) — season total",
                    font=dict(size=10), bgcolor="rgba(255,255,255,0.85)"),
        height=600, width=1100,
        paper_bgcolor="white", plot_bgcolor="#fafafa",
        margin=dict(l=60, r=20, t=50, b=50),
        hovermode="closest",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading game lines data...")
    df = load_team_data()
    print(f"  {len(df)} team-game rows  |  teams={df['team_abbr'].nunique()}")

    print("Loading team colors...")
    colors = load_team_colors()

    print("Building spread table + charts...")
    spread_table   = build_pivot_table_html(df, "team_spread", colors, fmt="+.1f")
    spread_b64     = make_chart_b64(df, "team_spread", colors,
                                    xlabel="Team spread (negative = favored)", vline=0.0)
    spread_plotly  = make_plotly_html(df, "team_spread", colors,
                                      title="Spread — box plot (interactive)",
                                      vline=0.0, include_js=True)
    spread_hist    = make_plotly_histogram_html(df, "team_spread", colors,
                                               bins=SPREAD_BINS, bin_labels=SPREAD_LABELS,
                                               title="Spread — binned histogram (interactive)",
                                               zero_bin_idx=5, include_js=False)

    print("Building implied points table + charts...")
    pts_table   = build_pivot_table_html(df, "implied_pts", colors, fmt=".1f")
    pts_b64     = make_chart_b64(df, "implied_pts", colors,
                                 xlabel="Implied team points = (game total − spread) / 2")
    pts_plotly  = make_plotly_html(df, "implied_pts", colors,
                                   title="Implied points — box plot (interactive)",
                                   include_js=False)
    pts_hist    = make_plotly_histogram_html(df, "implied_pts", colors,
                                            bins=PTS_BINS, bin_labels=PTS_LABELS,
                                            title="Implied points — binned histogram (interactive)",
                                            zero_bin_idx=None, include_js=False)

    print("Building player-level visualizations...")
    pdf = load_player_data()
    player_calib_html  = build_player_calibration_html(pdf)
    snap_scatter_html  = make_snap_scatter_html(pdf)
    trajectory_html    = make_trajectory_html(pdf, colors)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>NFL 2025 Team Distributions</title>
  <style>
    body  {{ font-family: sans-serif; max-width: 1400px; margin: 40px auto; padding: 0 20px; background:#fff; }}
    h1    {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
    h2    {{ margin: 0 0 10px; font-size: 15px; color: #333; }}
    h3    {{ margin: 20px 0 6px; font-size: 13px; color: #555; font-weight: normal; }}
    .meta {{ font-family: monospace; font-size: 12px; color: #666; margin: 0 0 20px; }}
    .section {{ margin-bottom: 70px; }}
    img   {{ width: 100%; border: 1px solid #ddd; border-radius: 4px; margin-top: 4px; }}
    tbody tr:hover td {{ filter: brightness(0.93); }}
  </style>
</head>
<body>
  <h1>NFL 2025 REG Season — Team Distributions</h1>
  <p class="meta">272 games · 32 teams · 17 games each · Arranged by conference/division · Colors = nflverse primary/secondary</p>

  <div class="section">
    <h2>Spread (team perspective) — W1 through W17</h2>
    {spread_table}
    <h3>Static strip plot — each dot = one game, dashed line = median, shaded = IQR</h3>
    <img src="data:image/png;base64,{spread_b64}" alt="Team spread strip plot">
    <h3>Interactive — box plot</h3>
    {spread_plotly}
    <h3>Interactive — binned histogram (≤-15 | -15:-10 | -10:-7 | -7:-3 | -3:0 | 0:3 | 3:7 | 7:10 | 10:15 | 15+)</h3>
    {spread_hist}
  </div>

  <div class="section">
    <h2>Implied team points scored — W1 through W17
      <span style="font-weight:normal;color:#666;font-size:12px">= (game total − team spread) / 2</span>
    </h2>
    {pts_table}
    <h3>Static strip plot — each dot = one game, dashed line = median, shaded = IQR</h3>
    <img src="data:image/png;base64,{pts_b64}" alt="Team implied points strip plot">
    <h3>Interactive — box plot</h3>
    {pts_plotly}
    <h3>Interactive — binned histogram (5-pt increments)</h3>
    {pts_hist}
  </div>

  <hr style="border:none;border-top:3px solid #333;margin:40px 0;">
  <h1>Player-Level Analysis — 2025 REG Season</h1>
  <p class="meta">Source: nfl_sacks_joined_2025.parquet · 0.5 line only · Push = 0 units (money back)</p>

  <div class="section">
    <h2>1. Player calibration — implied prob vs actual hit rate (≥5 games with a line)</h2>
    {player_calib_html}
  </div>

  <div class="section">
    <h2>2. Snap% vs sack hit rate — does opportunity predict outcomes?</h2>
    <p class="meta">Each dot = one player. Size = n games with line posted (≥3). Dotted = overall avg hit rate.</p>
    {snap_scatter_html}
  </div>

  <div class="section">
    <h2>3. Cumulative sacks trajectory — top 20 pass rushers</h2>
    {trajectory_html}
  </div>
</body>
</html>"""

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nSaved → {OUT_HTML}")

    counts = df.groupby("team_abbr").size()
    bad = counts[counts != 17]
    if len(bad):
        print(f"WARNING — teams without 17 games: {bad.to_dict()}")
    else:
        print("Validation: all 32 teams have exactly 17 games ✓")
    print(f"Spread range     : {df['team_spread'].min():+.1f} to {df['team_spread'].max():+.1f}")
    print(f"Implied pts range: {df['implied_pts'].min():.1f} to {df['implied_pts'].max():.1f}")


if __name__ == "__main__":
    main()

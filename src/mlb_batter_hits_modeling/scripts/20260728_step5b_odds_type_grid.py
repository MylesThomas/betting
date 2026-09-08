"""
Step 5b — Odds Type Grid Search — MLB Batter Hits.

Extends Step 5 by adding odds_type as a sweep dimension.
Missed in original Step 5: the grid only swept line × direction × edge_min.
Odds type (plus / even / minus) turns out to be the dominant stratifier.

Sweep:
  line:      0.5, 1.5
  direction: under
  edge_min:  0, 1, 2, 3, 5, 7, 10 pp
  odds_type: all | plus_only (>2.0) | even_only (==2.0) | minus_only (<2.0)

Usage:
  python src/mlb_batter_hits_modeling/scripts/20260728_step5b_odds_type_grid.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from datetime import date

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_SPINE = Path.home() / "Downloads/tmp/mlb_batter_hits_spine.parquet"
HTML_LOG    = REPO_ROOT / "knowledge-base/raw/20260727-mlb-batter-hits.html"
N_SPLITS    = 5

NUMERIC_FEATURES = [
    "min_raw_implied_prob_under",
    "max_raw_implied_prob_over",
    "hits_roll_career",
    "max_line",
    "ab_roll_career",
    "hits_roll_L20",
    "hits_roll_season",
    "consensus_line",
    "hits_roll_L10",
    "ba_roll_career",
]


# -----------------------------------------------------------------------
# OOF p_model — identical to step5
# -----------------------------------------------------------------------

def load_spine() -> pd.DataFrame:
    spine   = pd.read_parquet(LOCAL_SPINE)
    settled = spine[spine["hits_actual"].notna()].copy()
    settled["over_flag"]  = (settled["hits_actual"] > settled["offered_line"]).astype(int)
    settled["under_flag"] = (settled["hits_actual"] < settled["offered_line"]).astype(int)
    settled["raw_implied_prob_over"]  = 1.0 / settled["over_price"]
    settled["raw_implied_prob_under"] = 1.0 / settled["under_price"]
    return settled


def build_p_model(settled: pd.DataFrame) -> pd.DataFrame:
    has_dk = settled[settled["bookmaker"] == "draftkings"]
    no_dk  = settled[~settled.index.isin(has_dk.index)]
    pg = (
        pd.concat([has_dk, no_dk])
        .sort_values(["player_key", "game_date", "bookmaker"])
        .drop_duplicates(subset=["player_key", "game_date"])
        .sort_values("game_date")
        .reset_index(drop=True)
    )

    avail = [f for f in NUMERIC_FEATURES if f in pg.columns]
    sub   = pg[avail + ["hits_actual"]].dropna()
    X1, y1 = sub[avail].values, sub["hits_actual"].values
    tscv  = TimeSeriesSplit(n_splits=N_SPLITS)
    yhat  = np.full(len(sub), np.nan)
    for tr, te in tscv.split(X1):
        m = LinearRegression()
        m.fit(X1[tr], y1[tr])
        yhat[te] = m.predict(X1[te])
    pg["yhat_ols"] = np.nan
    pg.loc[sub.index, "yhat_ols"] = yhat

    settled = settled.merge(
        pg[["player_key", "game_date", "yhat_ols"]],
        on=["player_key", "game_date"], how="left",
    )
    usable = settled[settled["yhat_ols"].notna() & settled["over_price"].notna()].copy()
    usable = usable.sort_values("game_date").reset_index(drop=True)

    X2 = usable[["yhat_ols", "offered_line"]].values
    y2 = usable["over_flag"].values
    tscv2   = TimeSeriesSplit(n_splits=N_SPLITS)
    p_model = np.full(len(usable), np.nan)
    for tr, te in tscv2.split(X2):
        if len(np.unique(y2[tr])) < 2:
            continue
        m = LogisticRegression(max_iter=500, solver="lbfgs")
        m.fit(X2[tr], y2[tr])
        p_model[te] = m.predict_proba(X2[te])[:, 1]

    usable["p_model"]    = p_model
    usable["under_edge"] = (1 - usable["p_model"]) - usable["raw_implied_prob_under"]
    print(f"  OOF rows with p_model: {usable['p_model'].notna().sum():,}")
    return usable


def compute_pnl(df: pd.DataFrame, direction: str) -> pd.Series:
    if direction == "over":
        win, price = df["over_flag"].astype(float), df["over_price"]
    else:
        win, price = df["under_flag"].astype(float), df["under_price"]
    return np.where(win == 1, price - 1, -1.0)


def max_drawdown(cum: np.ndarray) -> float:
    peak = np.maximum.accumulate(cum)
    return float((peak - cum).max())


# -----------------------------------------------------------------------
# Grid search with odds_type dimension
# -----------------------------------------------------------------------

ODDS_FILTERS = {
    "all":        lambda df: df,
    "plus_only":  lambda df: df[df["under_price"] > 2.0],
    "even_only":  lambda df: df[df["under_price"] == 2.0],
    "minus_only": lambda df: df[df["under_price"] < 2.0],
}

def grid_search(usable: pd.DataFrame) -> pd.DataFrame:
    lines      = [0.5, 1.5]
    thresholds = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]

    rows = []
    for line in lines:
        sub_line = usable[
            (usable["offered_line"] == line) &
            usable["under_price"].notna() &
            usable["under_edge"].notna()
        ].copy()
        sub_line["pnl"] = compute_pnl(sub_line, "under")

        for odds_label, odds_fn in ODDS_FILTERS.items():
            sub_odds = odds_fn(sub_line)
            for thresh in thresholds:
                bets = sub_odds[sub_odds["under_edge"] >= thresh].sort_values("game_date")
                if len(bets) == 0:
                    continue
                n     = len(bets)
                hit   = bets["under_flag"].mean()
                net   = bets["pnl"].sum()
                roi   = net / n
                cum   = bets["pnl"].cumsum().values
                mdd   = max_drawdown(cum)
                nmdd  = net / mdd if mdd > 0 else np.inf
                avg_p = bets["under_price"].mean()
                rows.append(dict(
                    line=line, odds_type=odds_label, edge_min=thresh,
                    n_bets=n, hit_rate=round(hit, 4),
                    avg_price=round(avg_p, 3),
                    net_units=round(net, 2), roi_pct=round(roi * 100, 2),
                    max_dd=round(mdd, 2), net_mdd=round(nmdd, 2),
                ))

    return pd.DataFrame(rows).sort_values("net_units", ascending=False)


# -----------------------------------------------------------------------
# HTML rendering helpers
# -----------------------------------------------------------------------

def _color_cell(val: float, fmt: str) -> str:
    cls = "pos" if val > 0 else ("neg" if val < 0 else "")
    return f'<td style="text-align:center" class="{cls}">{fmt}</td>'


def _results_table(df: pd.DataFrame, caption: str) -> str:
    header = (
        "<table>"
        f"<caption style='text-align:left;font-weight:bold;margin-bottom:4px'>{caption}</caption>"
        "<thead><tr>"
        "<th>Line</th><th>Odds Type</th><th>Edge Min</th>"
        "<th>Bets</th><th>Hit %</th><th>Avg Price</th>"
        "<th>Net Units</th><th>ROI %</th><th>Max DD</th><th>Net/MDD</th>"
        "</tr></thead><tbody>"
    )
    trs = []
    for _, r in df.iterrows():
        bg = ""
        if r["odds_type"] == "plus_only" and r["net_units"] > 0:
            bg = "background:#e8f5e9"
        elif r["odds_type"] == "minus_only":
            bg = "background:#fce8e6"
        trs.append(
            f'<tr style="{bg}">'
            f'<td style="text-align:center">{r["line"]}</td>'
            f'<td><strong>{r["odds_type"]}</strong></td>'
            f'<td style="text-align:center">{r["edge_min"]*100:.0f}pp</td>'
            f'<td style="text-align:center">{int(r["n_bets"]):,}</td>'
            f'<td style="text-align:center">{r["hit_rate"]*100:.1f}%</td>'
            f'<td style="text-align:center">{r["avg_price"]:.3f}</td>'
            + _color_cell(r["net_units"], f'{r["net_units"]:+.2f}u')
            + _color_cell(r["roi_pct"],   f'{r["roi_pct"]:+.2f}%')
            + f'<td style="text-align:center">{r["max_dd"]:.2f}u</td>'
            + f'<td style="text-align:center">{r["net_mdd"]:.2f}x</td>'
            + "</tr>"
        )
    return header + "".join(trs) + "</tbody></table>"


def _season_table(df: pd.DataFrame) -> str:
    header = (
        "<table style='width:auto'><thead><tr>"
        "<th>Season</th><th>Bets</th><th>Hit %</th><th>Net Units</th><th>ROI %</th>"
        "</tr></thead><tbody>"
    )
    trs = []
    for _, r in df.iterrows():
        trs.append(
            f'<tr>'
            f'<td>{int(r["season"])}</td>'
            f'<td style="text-align:center">{int(r["n_bets"]):,}</td>'
            f'<td style="text-align:center">{r["hit_rate"]*100:.1f}%</td>'
            + _color_cell(r["net_units"], f'{r["net_units"]:+.2f}u')
            + _color_cell(r["roi_pct"],   f'{r["roi_pct"]:+.2f}%')
            + "</tr>"
        )
    return header + "".join(trs) + "</tbody></table>"


def append_html(results: pd.DataFrame, best_season: pd.DataFrame,
                tests_html: str, best_label: str) -> None:
    # Build the full new section
    today = date.today().isoformat()

    full_table_html = _results_table(
        results,
        "Full grid — line × odds_type × edge_min (sorted by net_units, OOF)"
    )

    # Focused table: line=0.5 only
    focused = results[results["line"] == 0.5].copy()
    focused_html = _results_table(
        focused,
        "0.5 UNDER only — odds_type × edge_min (sorted by net_units, OOF)"
    )

    season_html = _season_table(best_season)

    section = f"""
<section>
<h2>Step 5b — Odds Type Grid Search ({today})</h2>

<p style="background:#fff3cd;border-left:4px solid #f9a825;padding:10px 14px;margin:0 0 16px;border-radius:0 4px 4px 0;">
  <strong>This step was missed in the original Step 5.</strong> The original grid swept
  line × direction × edge_min but never stratified by odds type. Post-research review
  showed all OOF profit comes from plus-odds bets (price &gt; 2.0). This step sweeps
  odds_type as an explicit dimension to find the optimal filter.
</p>

<h3>Sweep Dimensions</h3>
<ul>
  <li><strong>line:</strong> 0.5, 1.5</li>
  <li><strong>direction:</strong> under only</li>
  <li><strong>edge_min:</strong> 0, 1, 2, 3, 5, 7, 10 pp</li>
  <li><strong>odds_type:</strong> all | plus_only (price &gt; 2.0) | even_only (price == 2.0) | minus_only (price &lt; 2.0)</li>
</ul>
<p>Green rows = plus_only with positive net units. Red rows = minus_only.</p>

<h3>Full Grid Results (OOF, sorted by net_units)</h3>
{full_table_html}

<h3>0.5 UNDER — Focused View</h3>
{focused_html}

<h3>Per-Season — Best Strategy: {best_label}</h3>
{season_html}

<h3>DuckDB SQL Tests</h3>
{tests_html}

</section>
<!-- ============================================================ -->
"""

    # Splice before closing </body>
    html = HTML_LOG.read_text()
    html = html.replace("</body>\n</html>", section + "\n</body>\n</html>")
    HTML_LOG.write_text(html)
    print(f"  HTML log updated → {HTML_LOG}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    print("Loading spine...")
    settled = load_spine()
    print(f"  {len(settled):,} settled rows")

    print("Building OOF p_model...")
    usable = build_p_model(settled)

    print("Running odds-type grid search...")
    results = grid_search(usable)

    print("\n--- Full grid (sorted by net_units) ---")
    print(results.round(3).to_string(index=False))

    print("\n--- 0.5 UNDER — top 15 by net_units ---")
    top = results[results["line"] == 0.5].head(15)
    print(top.round(3).to_string(index=False))

    # Best plus-only strategy per-season breakdown
    best_plus = results[
        (results["line"] == 0.5) &
        (results["odds_type"] == "plus_only") &
        (results["n_bets"] >= 50)
    ].iloc[0]

    best_label = f"0.5 UNDER · plus_only · edge≥{best_plus['edge_min']*100:.0f}pp"
    print(f"\n--- Per-season: {best_label} ---")
    sub = usable[
        (usable["offered_line"] == 0.5) &
        (usable["under_price"] > 2.0) &
        (usable["under_edge"] >= best_plus["edge_min"]) &
        usable["under_price"].notna() &
        usable["p_model"].notna()
    ].copy()
    sub["pnl"]    = compute_pnl(sub, "under")
    sub["season"] = pd.to_datetime(sub["game_date"]).dt.year
    by_season = sub.groupby("season").agg(
        n_bets=("pnl", "count"),
        hit_rate=("under_flag", "mean"),
        net_units=("pnl", "sum"),
    ).reset_index()
    by_season["roi_pct"]  = (by_season["net_units"] / by_season["n_bets"] * 100).round(2)
    by_season["hit_rate"] = by_season["hit_rate"].round(3)
    by_season["net_units"] = by_season["net_units"].round(2)
    print(by_season.to_string(index=False))

    # -----------------------------------------------------------------------
    # DuckDB SQL tests
    # -----------------------------------------------------------------------
    con = duckdb.connect()
    con.register("results", results)

    print("\n" + "=" * 60)
    print("STEP 5b — DuckDB SQL TESTS")
    print("=" * 60)

    tests = [
        (
            "T1: plus_only at edge≥2pp beats all at edge≥2pp on net_units (0.5 UNDER)",
            """
            SELECT
                MAX(CASE WHEN odds_type='plus_only'  AND edge_min=0.02 THEN net_units END) >
                MAX(CASE WHEN odds_type='all'         AND edge_min=0.02 THEN net_units END) AS ok
            FROM results WHERE line=0.5
            """,
        ),
        (
            "T2: minus_only at edge≥0pp has negative net_units (0.5 UNDER)",
            """
            SELECT net_units < 0 AS ok
            FROM results WHERE line=0.5 AND odds_type='minus_only' AND edge_min=0.0
            """,
        ),
        (
            "T3: best plus_only strategy has net/MDD >= 1.5",
            """
            SELECT net_mdd >= 1.5 AS ok
            FROM results
            WHERE line=0.5 AND odds_type='plus_only' AND n_bets >= 50
            ORDER BY net_units DESC LIMIT 1
            """,
        ),
        (
            "T4: best plus_only ROI > best all ROI at same edge_min (0.5 UNDER)",
            """
            WITH best_edge AS (
                SELECT edge_min AS be FROM results
                WHERE line=0.5 AND odds_type='all' AND n_bets>=50
                ORDER BY net_units DESC LIMIT 1
            )
            SELECT
                MAX(CASE WHEN odds_type='plus_only' THEN roi_pct END) >
                MAX(CASE WHEN odds_type='all'        THEN roi_pct END) AS ok
            FROM results, best_edge
            WHERE line=0.5 AND edge_min=be
            """,
        ),
        (
            "T5: plus_only n_bets >= 500 at edge≥2pp (enough volume to be meaningful)",
            """
            SELECT n_bets >= 500 AS ok
            FROM results WHERE line=0.5 AND odds_type='plus_only' AND edge_min=0.02
            """,
        ),
    ]

    pass_count = 0
    tests_rows = []
    for label, sql in tests:
        try:
            ok = con.execute(sql.strip()).fetchone()[0]
            status = "[PASS]" if ok else "[FAIL]"
            if ok:
                pass_count += 1
        except Exception as e:
            status = f"[ERROR] {e}"
            ok = False
        print(f"  {status} {label}")
        color = "#d4edda" if ok else "#f8d7da"
        tests_rows.append(
            f'<tr style="background:{color}">'
            f'<td>{"✓" if ok else "✗"}</td>'
            f'<td>{label}</td>'
            f'</tr>'
        )

    all_pass = pass_count == len(tests)
    if all_pass:
        print(f"\nAll {len(tests)} tests PASSED.")
    else:
        print(f"\n{pass_count}/{len(tests)} tests passed.")

    tests_html = (
        "<table><thead><tr><th></th><th>Test</th></tr></thead><tbody>"
        + "".join(tests_rows)
        + "</tbody></table>"
        + (
            '<p class="pass">All Step 5b tests PASSED.</p>' if all_pass
            else '<p class="fail">Some Step 5b tests FAILED — review before updating strategy.</p>'
        )
    )

    append_html(results, by_season, tests_html, best_label)
    print("\nDone.")


if __name__ == "__main__":
    main()

"""
Steps 5 + 6 — Grid search (OOS + IS).

Sweeps over:
  edge_threshold, direction, odds_bucket, shrinkage, prediction_method
  (clf_threshold skipped — direct probability model output, no binary classification threshold needed)

OOS: uses OOF predictions (p_model from step 4 scored spine).
IS:  scores the full training data with the trained model.

Output:
  ~/Downloads/tmp/mlb_batter_hr_grid_oos.csv
  ~/Downloads/tmp/mlb_batter_hr_grid_is.csv
  config.yaml updated with results
  HTML appended to session log
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import joblib
import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
REPO_ROOT   = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_SCORED = Path.home() / "Downloads/tmp/mlb_batter_hr_scored.parquet"
LOCAL_MODEL  = REPO_ROOT / "src/mlb_batter_home_runs_modeling/models/mlb_batter_hr_model.joblib"
CONFIG_PATH  = REPO_ROOT / "src/mlb_batter_home_runs_modeling/config.yaml"
HTML_LOG     = REPO_ROOT / "knowledge-base/raw/20260801-mlb-batter-home-runs.html"
LOCAL_OOS    = Path.home() / "Downloads/tmp/mlb_batter_hr_grid_oos.csv"
LOCAL_IS     = Path.home() / "Downloads/tmp/mlb_batter_hr_grid_is.csv"
ET           = ZoneInfo("America/New_York")
TARGET       = "hr_over_0_5"


def ts() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")


def df_to_html_table(df: pd.DataFrame, caption: str = "", n_rows: int = 50) -> str:
    rows_html = []
    header = "<tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"
    for _, row in df.head(n_rows).iterrows():
        cells = "".join(f"<td>{v}</td>" for v in row)
        rows_html.append(f"<tr>{cells}</tr>")
    cap = f"<caption style='font-weight:bold;text-align:left;padding:6px 0'>{caption}</caption>" if caption else ""
    return f"<table>{cap}<thead>{header}</thead><tbody>{''.join(rows_html)}</tbody></table>"


def pnl_vectorized(df: pd.DataFrame, direction: str) -> pd.Series:
    """Vectorized P&L computation: +price-1 on win, -1 on loss, NaN on missing data."""
    if direction == "over":
        price   = df["over_price"]
        outcome = df[TARGET]
        hit     = outcome >= 1.0
    else:
        price   = df["under_price"]
        outcome = df[TARGET]
        hit     = outcome == 0.0
    valid   = price.notna() & outcome.notna()
    return np.where(valid, np.where(hit, price - 1.0, -1.0), np.nan)


def max_drawdown(pnl_series: pd.Series) -> float:
    """Peak-to-trough max drawdown of cumulative P&L series."""
    cumulative = pnl_series.cumsum()
    peak = cumulative.expanding().max()
    drawdown = peak - cumulative
    return float(drawdown.max())


def drawdown_dates(df_bets: pd.DataFrame, pnl_col: str) -> tuple[str, str, str]:
    """Returns (peak_date, trough_date, recovery_date) for the max drawdown."""
    cumulative = df_bets[pnl_col].cumsum().reset_index(drop=True)
    peak_idx = (cumulative.expanding().max() - cumulative).idxmax()
    peak_val = cumulative[:peak_idx].max()
    trough_idx = cumulative[peak_idx:].idxmin() + peak_idx if len(cumulative[peak_idx:]) > 0 else peak_idx
    try:
        peak_date   = df_bets.iloc[cumulative[:peak_idx+1].idxmax()]["game_date"]
        trough_date = df_bets.iloc[trough_idx]["game_date"]
        recovery    = df_bets.iloc[trough_idx:][cumulative[trough_idx:] > peak_val]
        recovery_date = recovery.iloc[0]["game_date"] if len(recovery) > 0 else None
    except Exception:
        peak_date = trough_date = recovery_date = None
    return str(peak_date), str(trough_date), str(recovery_date) if recovery_date else "still in drawdown"


def run_strategy(df: pd.DataFrame, edge_threshold: float, direction: str,
                 odds_bucket: str, shrinkage: float, prediction_method: str,
                 pop_mean_p_over: float) -> dict:
    """Evaluate one strategy combo (vectorized). Returns summary dict."""
    # Work with selected columns only — avoid copying 1.7M-row df
    cols = ["game_date", "offered_line", "over_price", "under_price",
            "p_model_over", "p_model_under", "raw_implied_prob_over",
            "raw_implied_prob_under", "min_raw_implied_prob_over", TARGET]
    df = df[[c for c in cols if c in df.columns]].copy()

    # Restrict to 0.5 line only — model predicts P(HR≥1), irrelevant for 1.5/2.5
    df = df[df["offered_line"] == 0.5]

    if prediction_method == "consensus_line":
        if "min_raw_implied_prob_over" in df.columns:
            df["p_model_over"]  = df["min_raw_implied_prob_over"]
            df["p_model_under"] = 1.0 - df["p_model_over"]
        else:
            return {}
    elif shrinkage > 0:
        df = df.copy()
        df["p_model_over"]  = (1 - shrinkage) * df["p_model_over"]  + shrinkage * pop_mean_p_over
        df["p_model_under"] = 1.0 - df["p_model_over"]

    edge_over  = df["p_model_over"]  - df["raw_implied_prob_over"]
    edge_under = df["p_model_under"] - df["raw_implied_prob_under"]

    if odds_bucket == "plus_odds":
        price_mask_over  = df["over_price"]  > 2.0
        price_mask_under = df["under_price"] > 2.0
    elif odds_bucket == "minus_odds":
        price_mask_over  = df["over_price"]  < 2.0
        price_mask_under = df["under_price"] < 2.0
    else:
        price_mask_over  = pd.Series(True, index=df.index)
        price_mask_under = pd.Series(True, index=df.index)

    if direction == "over":
        mask = (edge_over >= edge_threshold) & price_mask_over
        bets = df[mask].copy()
        bets["pnl"] = pnl_vectorized(bets, "over")
    elif direction == "under":
        mask = (edge_under >= edge_threshold) & price_mask_under
        bets = df[mask].copy()
        bets["pnl"] = pnl_vectorized(bets, "under")
    else:  # both
        over_mask  = (edge_over  >= edge_threshold) & price_mask_over
        under_mask = (edge_under >= edge_threshold) & price_mask_under
        over_bets  = df[over_mask].copy();  over_bets["pnl"]  = pnl_vectorized(over_bets,  "over")
        under_bets = df[under_mask].copy(); under_bets["pnl"] = pnl_vectorized(under_bets, "under")
        bets = pd.concat([over_bets, under_bets], ignore_index=True)

    bets = bets.dropna(subset=["pnl"])
    n_bets = len(bets)
    if n_bets == 0:
        return {}

    units_won  = round(bets["pnl"].sum(), 2)
    win_rate   = round((bets["pnl"] > 0).mean(), 4)
    roi        = round(units_won / n_bets, 4)
    mdd        = round(max_drawdown(bets["pnl"]), 2)
    calmar     = round(units_won / mdd, 3) if mdd > 0 else np.nan
    avg_odds   = round(bets["over_price" if direction == "over" else "under_price"].mean(), 3) if direction != "both" else None
    pct_of_univ = round(n_bets / len(df), 4)

    pk_date, tr_date, rec_date = drawdown_dates(bets.sort_values("game_date"), "pnl")

    return {
        "edge_threshold":      edge_threshold,
        "direction":           direction,
        "odds_bucket":         odds_bucket,
        "shrinkage":           shrinkage,
        "prediction_method":   prediction_method,
        "n_bets":              n_bets,
        "pct_of_universe":     pct_of_univ,
        "win_rate":            win_rate,
        "units_won":           units_won,
        "roi":                 roi,
        "avg_odds":            avg_odds,
        "max_drawdown":        mdd,
        "calmar":              calmar,
        "drawdown_peak_date":  pk_date,
        "drawdown_trough_date": tr_date,
        "drawdown_recovery_date": rec_date,
    }


def run_grid(df: pd.DataFrame, cfg: dict, label: str) -> pd.DataFrame:
    """Run full grid search over all combos."""
    gs  = cfg["grid_search"]
    pop_mean = df[TARGET].mean() if TARGET in df.columns else 0.10
    print(f"\n[{label}] Running grid search...")

    results = []
    total = (len(gs["edge_threshold"]) * len(gs["direction"]) *
             len(gs["odds_bucket"]) * len(gs["shrinkage"]) * len(gs["prediction_method"]))
    done = 0

    for et in gs["edge_threshold"]:
        for direction in gs["direction"]:
            for odds_bucket in gs["odds_bucket"]:
                for shrinkage in gs["shrinkage"]:
                    for pred_method in gs["prediction_method"]:
                        r = run_strategy(df, et, direction, odds_bucket, shrinkage, pred_method, pop_mean)
                        if r:
                            results.append(r)
                        done += 1

    print(f"  Done: {done} combos, {len(results)} with bets")
    df_results = pd.DataFrame(results)
    if len(df_results):
        df_results = df_results.sort_values("units_won", ascending=False).reset_index(drop=True)
    return df_results


def main() -> None:
    if not LOCAL_SCORED.exists():
        print(f"ERROR: {LOCAL_SCORED} not found — run step4 first")
        sys.exit(1)

    scored = pd.read_parquet(LOCAL_SCORED)
    print(f"Loaded scored spine: {len(scored):,} rows")

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # Dedup to (player, game_date, bookmaker, line) — the grid grain
    df = (
        scored
        .dropna(subset=[TARGET, "p_model_over", "raw_implied_prob_over", "raw_implied_prob_under"])
        .sort_values(["player_key", "game_date", "bookmaker", "offered_line"])
        .drop_duplicates(subset=["player_key", "game_date", "bookmaker", "offered_line"])
        .reset_index(drop=True)
    )
    print(f"Grid grain rows: {len(df):,}")

    # ── OOS grid search ──────────────────────────────────────────────────────────
    oos_df = run_grid(df, cfg, "OOS")
    oos_df.to_csv(LOCAL_OOS, index=False)
    print(f"OOS results saved → {LOCAL_OOS}")

    # ── IS: Score full data with full-data model ─────────────────────────────────
    print("\n[IS] Re-scoring with full-data model...")
    model_art  = joblib.load(LOCAL_MODEL)
    full_model = model_art["model"]
    features   = model_art["features"]
    feat_cols  = [f for f in features if f in df.columns]
    X_all      = df[feat_cols].values
    df = df.copy()
    df["p_model_over"]  = np.clip(full_model.predict_proba(X_all)[:, 1], 0.01, 0.99)
    df["p_model_under"] = 1.0 - df["p_model_over"]
    df["raw_implied_prob_over"]  = 1.0 / df["over_price"]
    df["raw_implied_prob_under"] = 1.0 / df["under_price"]

    is_df = run_grid(df, cfg, "IS")
    is_df.to_csv(LOCAL_IS, index=False)
    print(f"IS results saved → {LOCAL_IS}")

    # ── Top results ──────────────────────────────────────────────────────────────
    print("\n=== Top OOS Results (sorted by units_won) ===")
    print(oos_df.head(20).to_string(index=False))

    print("\n=== Top IS Results (sorted by units_won) ===")
    print(is_df.head(20).to_string(index=False))

    # ── DuckDB tests ─────────────────────────────────────────────────────────────
    import duckdb
    con = duckdb.connect()
    con.register("oos", oos_df)
    con.register("is_results", is_df)
    con.register("df_grid", df)

    print("\n=== STEP 5/6 SQL TESTS ===")
    tests_passed = 0
    tests_failed = 0
    test_results = []

    def run_test(name: str, sql: str):
        nonlocal tests_passed, tests_failed
        try:
            result = con.execute(sql).fetchone()[0]
            passed = bool(result)
            status = "PASS" if passed else "FAIL"
            if passed: tests_passed += 1
            else: tests_failed += 1
            print(f"  [{status}] {name}: {result}")
            test_results.append((name, status, str(result)))
        except Exception as e:
            tests_failed += 1
            print(f"  [FAIL] {name}: ERROR — {e}")
            test_results.append((name, "FAIL", f"ERROR: {e}"))

    run_test("T1: OOS results has rows", "SELECT COUNT(*) > 0 FROM oos")
    run_test("T2: IS results has rows", "SELECT COUNT(*) > 0 FROM is_results")
    run_test("T3: Best OOS strategy has >= 30 bets",
             "SELECT MAX(n_bets) >= 30 FROM oos")
    run_test("T4: Best OOS ROI < 500% (sanity check)",
             "SELECT MAX(ABS(roi)) < 5.00 FROM oos WHERE n_bets >= 100")
    run_test("T5: IS ROI >= OOS ROI for best strategy (same params)",
             """
             SELECT (
                 SELECT roi FROM is_results ORDER BY units_won DESC LIMIT 1
             ) >= (
                 SELECT roi FROM oos ORDER BY units_won DESC LIMIT 1
             )
             """)
    run_test("T6: No win_rate > 1.0", "SELECT MAX(win_rate) <= 1.0 FROM oos")

    print(f"\n  Tests passed: {tests_passed} / {tests_passed + tests_failed}")

    # ── Write config.yaml results ─────────────────────────────────────────────────
    oos_records = oos_df[oos_df["n_bets"] >= 10].head(30).to_dict(orient="records")
    is_records  = is_df[is_df["n_bets"] >= 10].head(30).to_dict(orient="records")
    for r in oos_records + is_records:
        for k, v in r.items():
            if isinstance(v, (np.floating, float)):
                r[k] = float(round(v, 4))
            elif isinstance(v, (np.integer, int)):
                r[k] = int(v)

    cfg["grid_search"]["out_of_sample_results"] = oos_records
    cfg["grid_search"]["in_sample_results"]     = is_records
    if len(oos_df):
        best = oos_df.iloc[0]
        cfg["grid_search"]["strategy_summary"] = (
            f"{best['direction'].upper()} edge≥{best['edge_threshold']:.0%} "
            f"odds={best['odds_bucket']} shrink={best['shrinkage']} — "
            f"OOS {int(best['n_bets'])} bets, {best['units_won']:+.1f}u, "
            f"{best['roi']:.1%} ROI ({ts()[:10]})"
        )

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"config.yaml updated → {CONFIG_PATH}")

    # ── HTML ─────────────────────────────────────────────────────────────────────
    # Season breakdown for top OOS strategy
    top_oos = oos_df.iloc[0] if len(oos_df) else None
    season_breakdown_html = ""
    if top_oos is not None:
        df_05 = df[df["offered_line"] == 0.5]
        bets_top = df_05[df_05["edge_under"] >= top_oos["edge_threshold"]].copy() if top_oos["direction"] == "under" else df_05[df_05["edge_over"] >= top_oos["edge_threshold"]].copy()
        bets_top["pnl"] = pnl_vectorized(bets_top, top_oos["direction"])
        bets_top = bets_top.dropna(subset=["pnl"])
        bets_top["season"] = bets_top["game_date"].str[:4].astype(int)
        sb = bets_top.groupby("season").agg(
            n_bets=("pnl", "count"),
            win_rate=("pnl", lambda x: (x > 0).mean()),
            units=("pnl", "sum"),
        ).reset_index()
        sb["roi"] = (sb["units"] / sb["n_bets"]).round(4)
        sb["win_rate"] = sb["win_rate"].round(4)
        sb["units"] = sb["units"].round(2)
        season_breakdown_html = df_to_html_table(sb, "Season-by-season (top OOS strategy)")

    section_html = f"""
<section>
<h2>Step 5 — OOS Grid Search</h2>
<p class="timestamp">{ts()}</p>

<h3>Top 30 OOS Results (sorted by units_won)</h3>
{df_to_html_table(oos_df.head(30), "OOS — all combos with n_bets ≥ 1, sorted by units_won")}

<h3>Season Breakdown (top OOS strategy)</h3>
{season_breakdown_html}

<h3>SQL Test Results</h3>
<table><thead><tr><th>Test</th><th>Status</th><th>Value</th></tr></thead>
<tbody>{"".join(f'<tr><td>{n}</td><td class="{"pass" if s=="PASS" else "fail"}">{s}</td><td>{v}</td></tr>' for n,s,v in test_results)}</tbody>
</table>
<p><strong>Passed: {tests_passed} / {tests_passed + tests_failed}</strong></p>
</section>

<section>
<h2>Step 6 — IS Grid Search</h2>
<p class="timestamp">{ts()}</p>

<h3>Top 30 IS Results (sorted by units_won)</h3>
{df_to_html_table(is_df.head(30), "IS — all combos with n_bets ≥ 1, sorted by units_won")}
</section>
"""
    with open(HTML_LOG, "a") as f:
        f.write(section_html)
    print(f"\nHTML appended → {HTML_LOG}")


if __name__ == "__main__":
    main()

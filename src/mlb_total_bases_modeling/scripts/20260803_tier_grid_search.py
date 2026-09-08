"""
MLB Total Bases — Player Tier Sweep (2026-08-03)
=================================================
Adds recognition_tier as a filter dimension to the existing IS/OOS grid.
Tier filters: all | non_superstar | superstar | known_starter | fringe_unknown

Runs the same load_and_score() pipeline as 20260801_grid_7bin.py, then sweeps
edge_min × line_bucket × tier_filter.  Strategy is UNDER only.

Key outputs:
  1. Tier distribution (how many name_norms in each tier)
  2. Production params (edge≥5pp, line=1.5) by tier — IS + OOS
  3. Full grid × tier — OOS only, sorted by net_units
  4. ROI-delta pivot: (non_superstar − all) across all edge_min × line_bucket combos

HTML: knowledge-base/raw/20260803-mlb-tb-tier-sweep.html (new file, standalone)

Usage:
  python src/mlb_total_bases_modeling/scripts/20260803_tier_grid_search.py
"""
from __future__ import annotations

import sys
import warnings
from io import BytesIO
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT  = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET  = "the-odds-api-mt"
SPINE_KEY  = "mlb/total_bases_model/regression/mlb_tb_reg_spine.parquet"
MODEL_KEY  = "mlb/total_bases_model/model/mlb_tb_regression_v2.joblib"
TIERS_PATH = Path.home() / "Downloads/tmp/mlb_total_bases/mlb_tb_player_tiers.parquet"
OUT_DIR    = Path.home() / "Downloads/tmp/mlb_total_bases"
HTML_PATH  = REPO_ROOT / "knowledge-base/raw/20260803-mlb-tb-tier-sweep.html"
TARGET     = "total_bases"

IS_SEASONS  = [2024, 2025]
OOS_SEASON  = 2026

EDGE_THRESHOLDS = [0.00, 0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
LINE_BUCKETS    = ["all_lines", "0.5_only", "1.5_only"]
TIER_FILTERS    = ["all", "non_superstar", "superstar", "known_starter", "fringe_unknown"]

PROD_EDGE = 0.05
PROD_LINE = "1.5_only"

TIER_ORDER = ["superstar", "known_starter", "fringe", "unknown"]


# ── Data ──────────────────────────────────────────────────────────────────────

def load_and_score() -> pd.DataFrame:
    s3 = boto3.client("s3")
    print("Loading spine from S3...")
    spine = pd.read_parquet(BytesIO(
        s3.get_object(Bucket=S3_BUCKET, Key=SPINE_KEY)["Body"].read()
    ))
    print(f"  {len(spine):,} rows · {spine['game_date'].min()} → {spine['game_date'].max()}")

    print("Loading model bundle from S3...")
    bundle = joblib.load(BytesIO(
        s3.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)["Body"].read()
    ))

    df = spine.copy()
    df["season"]         = pd.to_datetime(df["game_date"]).dt.year
    df["raw_prob_under"] = 1.0 / df["under_price"]

    # min/max line per player-game (used as features in some model bundles)
    lr = (df.groupby(["name_norm", "game_date"])["line"]
          .agg(min_line="min", max_line="max").reset_index())
    df = df.merge(lr, on=["name_norm", "game_date"], how="left")

    model    = bundle["model"]
    scaler   = bundle["scaler"]
    features = bundle["features_numeric"]
    calibs   = bundle.get("calib_models", {})

    upg = (df[["name_norm", "game_date", "line"] + features]
           .drop_duplicates(subset=["name_norm", "game_date", "line"])
           .dropna(subset=features).copy())
    upg["y_hat"] = model.predict(scaler.transform(upg[features].values.astype(float)))

    rows = []
    for line, calib in calibs.items():
        sub = upg[upg["line"] == line].copy()
        if sub.empty:
            continue
        sub["p_model_under"] = 1.0 - np.clip(
            calib.predict_proba(sub["y_hat"].values.reshape(-1, 1))[:, 1], 0.01, 0.99
        )
        rows.append(sub)

    p_df = pd.concat(rows, ignore_index=True)
    df = df.merge(p_df[["name_norm", "game_date", "line", "y_hat", "p_model_under"]],
                  on=["name_norm", "game_date", "line"], how="inner")
    df["edge_under"] = df["p_model_under"] - df["raw_prob_under"]

    print(f"  Scored {len(df):,} rows")
    return df


def attach_tiers(df: pd.DataFrame) -> pd.DataFrame:
    tiers = pd.read_parquet(TIERS_PATH)
    tiers["recognition_tier"] = tiers["recognition_tier"].astype(str)
    tiers = tiers[["name_norm", "recognition_tier"]].drop_duplicates("name_norm")
    df = df.merge(tiers, on="name_norm", how="left")
    df["recognition_tier"] = df["recognition_tier"].fillna("unknown")
    n_matched = df["recognition_tier"].ne("unknown").sum()
    print(f"  Tiers joined: {n_matched:,}/{len(df):,} rows matched ({n_matched/len(df):.1%})")
    return df


def _tier_mask(df: pd.DataFrame, tier_filter: str) -> np.ndarray:
    t = df["recognition_tier"].values
    if tier_filter == "all":
        return np.ones(len(df), dtype=bool)
    if tier_filter == "non_superstar":
        return t != "superstar"
    if tier_filter == "superstar":
        return t == "superstar"
    if tier_filter == "known_starter":
        return t == "known_starter"
    if tier_filter == "fringe_unknown":
        return (t == "fringe") | (t == "unknown")
    return np.ones(len(df), dtype=bool)


def _max_dd(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    cum  = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    return float(np.max(peak - cum))


# ── Grid ──────────────────────────────────────────────────────────────────────

def run_grid(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for tf in TIER_FILTERS:
        tmask = _tier_mask(df, tf)
        for et in EDGE_THRESHOLDS:
            for lb in LINE_BUCKETS:
                line_mask = (
                    df["line"].values == 0.5 if lb == "0.5_only" else
                    df["line"].values == 1.5 if lb == "1.5_only" else
                    np.ones(len(df), dtype=bool)
                )
                mask = tmask & line_mask & (df["edge_under"].values >= et)
                sub  = df[mask]
                if len(sub) < 10:
                    continue
                won    = (sub[TARGET].values < sub["line"].values).astype(float)
                payout = sub["under_price"].values - 1.0
                pnl    = np.where(won, payout, -1.0)
                records.append({
                    "tier_filter": tf,
                    "edge_min":    et,
                    "line_bucket": lb,
                    "n_bets":      int(len(pnl)),
                    "win_pct":     round(float(won.mean()), 4),
                    "net_units":   round(float(pnl.sum()), 2),
                    "roi_pct":     round(float(pnl.mean()) * 100, 2),
                    "max_dd":      round(_max_dd(pnl), 2),
                })
    return pd.DataFrame(records).sort_values("net_units", ascending=False).reset_index(drop=True)


# ── Tier distribution ──────────────────────────────────────────────────────────

def tier_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Unique players (name_norm) per tier."""
    unique = df[["name_norm", "recognition_tier"]].drop_duplicates("name_norm")
    dist   = unique["recognition_tier"].value_counts().rename("n_players").reset_index()
    dist.columns = ["recognition_tier", "n_players"]
    ordered = pd.Categorical(dist["recognition_tier"], categories=TIER_ORDER, ordered=True)
    dist = dist.assign(recognition_tier=ordered).sort_values("recognition_tier").reset_index(drop=True)
    dist["pct"] = (dist["n_players"] / dist["n_players"].sum() * 100).round(1)
    return dist


def tier_sample(df: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    """Sample player names per tier for sanity-check."""
    unique = (df[["name_norm", "recognition_tier"]]
              .drop_duplicates("name_norm"))
    unique["display_name"] = unique["name_norm"].str.title()
    rows = []
    for tier in TIER_ORDER:
        sub = unique[unique["recognition_tier"] == tier]["display_name"].dropna()
        if sub.empty:
            continue
        sample = ", ".join(sorted(sub.sample(min(n, len(sub)), random_state=42).tolist()))
        rows.append({"tier": tier, "examples": sample, "n_total": len(sub)})
    return pd.DataFrame(rows)


# ── Production params table ────────────────────────────────────────────────────

def prod_params_table(is_g: pd.DataFrame, oos_g: pd.DataFrame) -> pd.DataFrame:
    """ROI at production params (edge≥5pp, line=1.5) by tier."""
    rows = []
    for tf in TIER_FILTERS:
        r_is  = is_g [(is_g ["tier_filter"] == tf) & (is_g ["edge_min"] == PROD_EDGE) & (is_g ["line_bucket"] == PROD_LINE)]
        r_oos = oos_g[(oos_g["tier_filter"] == tf) & (oos_g["edge_min"] == PROD_EDGE) & (oos_g["line_bucket"] == PROD_LINE)]
        def _fmt(r: pd.DataFrame) -> dict:
            if r.empty:
                return {"n": 0, "roi": float("nan"), "units": 0.0}
            return {"n": r.iloc[0]["n_bets"], "roi": r.iloc[0]["roi_pct"], "units": r.iloc[0]["net_units"]}
        ri = _fmt(r_is)
        ro = _fmt(r_oos)
        rows.append({
            "tier_filter":  tf,
            "IS_n":         ri["n"],  "IS_units": ri["units"],  "IS_roi_pct":  ri["roi"],
            "OOS_n":        ro["n"],  "OOS_units": ro["units"], "OOS_roi_pct": ro["roi"],
        })
    return pd.DataFrame(rows)


# ── ROI-delta pivot ────────────────────────────────────────────────────────────

def roi_delta_pivot(oos_g: pd.DataFrame) -> pd.DataFrame:
    """For each edge_min × line_bucket: roi_delta_pp = roi(non_superstar) − roi(all)."""
    sub = oos_g[oos_g["tier_filter"].isin(["all", "non_superstar"])].copy()
    pv  = sub.pivot_table(
        index=["edge_min", "line_bucket"],
        columns="tier_filter",
        values=["roi_pct", "net_units", "n_bets"],
    )
    pv.columns = ["_".join(c) for c in pv.columns]
    pv["roi_delta_pp"] = pv.get("roi_pct_non_superstar", np.nan) - pv.get("roi_pct_all", np.nan)
    pv["units_delta"]  = pv.get("net_units_non_superstar", np.nan) - pv.get("net_units_all", np.nan)
    return pv.reset_index().sort_values("roi_delta_pp", ascending=False)


# ── SQL tests ──────────────────────────────────────────────────────────────────

def run_tests(df: pd.DataFrame, is_g: pd.DataFrame, oos_g: pd.DataFrame,
              dist: pd.DataFrame) -> bool:
    import duckdb
    con = duckdb.connect()
    con.register("spine", df)
    con.register("isg",   is_g)
    con.register("oosg",  oos_g)
    con.register("dist",  dist)

    tests = [
        ("T1: tiers joined — <20% missing",
         "SELECT (SUM(CASE WHEN recognition_tier='unknown' THEN 1 ELSE 0 END)*1.0/COUNT(*)) < 0.20 AS pass FROM spine"),
        ("T2: OOS grid has all 5 tier_filters present",
         "SELECT COUNT(DISTINCT tier_filter) = 5 AS pass FROM oosg"),
        ("T3: production row present for 'all' in OOS",
         f"SELECT COUNT(*) >= 1 AS pass FROM oosg WHERE tier_filter='all' AND edge_min={PROD_EDGE} AND line_bucket='{PROD_LINE}'"),
        ("T4: n_bets(non_superstar) <= n_bets(all) at prod params in OOS",
         f"SELECT (SELECT n_bets FROM oosg WHERE tier_filter='non_superstar' AND edge_min={PROD_EDGE} AND line_bucket='{PROD_LINE}') "
         f"<= (SELECT n_bets FROM oosg WHERE tier_filter='all' AND edge_min={PROD_EDGE} AND line_bucket='{PROD_LINE}') AS pass"),
        ("T5: tier distribution covers all 4 tiers",
         "SELECT COUNT(*) >= 4 AS pass FROM dist"),
        ("T6: no NaN net_units in OOS grid",
         "SELECT COUNT(*) = 0 AS pass FROM oosg WHERE net_units IS NULL"),
        ("T7: OOS grid has at least 50 combos",
         "SELECT COUNT(*) >= 50 AS pass FROM oosg"),
    ]

    all_pass = True
    print("\nSQL Tests:")
    for name, sql in tests:
        try:
            val = con.execute(sql).fetchone()
            ok  = bool(val[0]) if val else False
            if not ok:
                all_pass = False
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        except Exception as e:
            all_pass = False
            print(f"  [ERROR] {name}: {e}")
    return all_pass


# ── HTML ──────────────────────────────────────────────────────────────────────

_SS = "-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif"


def _td_units(v: float) -> str:
    c = "#1a7f37" if v > 0 else ("#d32f2f" if v < 0 else "#888")
    return f'<td style="text-align:right;color:{c};font-weight:bold">{v:+.1f}u</td>'


def _td_roi(v: float) -> str:
    if np.isnan(v):
        return '<td style="text-align:right;color:#aaa">—</td>'
    c = "#1a7f37" if v > 0 else "#d32f2f"
    return f'<td style="text-align:right;color:{c}">{v:+.2f}%</td>'


def _th_row(cols: list[str]) -> str:
    return "".join(f'<th style="white-space:nowrap;padding:4px 8px">{c}</th>' for c in cols)


def _thead(cols: list[str]) -> str:
    return f'<thead><tr style="background:#2c3e50;color:#fff">{_th_row(cols)}</tr></thead>'


def _dist_html(dist: pd.DataFrame, sample: pd.DataFrame) -> str:
    rows = ""
    for _, r in dist.iterrows():
        ex_row = sample[sample["tier"] == r["recognition_tier"]]
        ex = ex_row.iloc[0]["examples"] if not ex_row.empty else ""
        rows += (
            f'<tr>'
            f'<td style="padding:4px 8px"><b>{r["recognition_tier"]}</b></td>'
            f'<td style="text-align:right;padding:4px 8px">{r["n_players"]:,}</td>'
            f'<td style="text-align:right;padding:4px 8px">{r["pct"]:.1f}%</td>'
            f'<td style="padding:4px 8px;font-size:11px;color:#555">{ex}</td>'
            f'</tr>'
        )
    return (
        f'<table style="font-size:12px;border-collapse:collapse;border:1px solid #ddd">'
        f'{_thead(["Tier", "Players", "%", "Examples"])}'
        f'<tbody>{rows}</tbody></table>'
    )


def _prod_html(prod: pd.DataFrame) -> str:
    rows = ""
    for _, r in prod.iterrows():
        is_prod = r["tier_filter"] == "all"
        bg = ' style="background:#e8f5e9"' if is_prod else ""
        rows += (
            f'<tr{bg}>'
            f'<td style="padding:4px 8px"><b>{r["tier_filter"]}</b></td>'
            f'<td style="text-align:right;padding:4px 8px">{r["IS_n"]:,}</td>'
            + _td_units(r["IS_units"])
            + _td_roi(r["IS_roi_pct"])
            + f'<td style="text-align:right;padding:4px 8px">{r["OOS_n"]:,}</td>'
            + _td_units(r["OOS_units"])
            + _td_roi(r["OOS_roi_pct"])
            + '</tr>'
        )
    return (
        f'<table style="font-size:12px;border-collapse:collapse;border:1px solid #ddd">'
        f'{_thead(["Tier Filter", "IS N", "IS Units", "IS ROI%", "OOS N", "OOS Units", "OOS ROI%"])}'
        f'<tbody>{rows}</tbody></table>'
    )


def _full_oos_html(oos_g: pd.DataFrame) -> str:
    rows = ""
    for _, r in oos_g.head(60).iterrows():
        is_prod = (r["tier_filter"] == "all" and r["edge_min"] == PROD_EDGE and r["line_bucket"] == PROD_LINE)
        bg = ' style="background:#e8f5e9"' if is_prod else ""
        roi_c = "#1a7f37" if r["roi_pct"] > 0 else "#d32f2f"
        rows += (
            f'<tr{bg}>'
            f'<td style="padding:4px 8px">{r["tier_filter"]}</td>'
            f'<td style="text-align:right;padding:4px 8px">{r["edge_min"]:.0%}</td>'
            f'<td style="padding:4px 8px">{r["line_bucket"]}</td>'
            f'<td style="text-align:right;padding:4px 8px">{r["n_bets"]:,}</td>'
            f'<td style="text-align:right;padding:4px 8px">{r["win_pct"]:.1%}</td>'
            + _td_units(r["net_units"])
            + f'<td style="text-align:right;padding:4px 8px;color:{roi_c}">{r["roi_pct"]:+.2f}%</td>'
            + f'<td style="text-align:right;padding:4px 8px">{r["max_dd"]:.1f}u</td>'
            + '</tr>'
        )
    return (
        f'<table style="font-size:11px;border-collapse:collapse;border:1px solid #ddd">'
        f'{_thead(["Tier Filter", "Edge≥", "Line", "N", "Win%", "Net Units", "ROI%", "Max DD"])}'
        f'<tbody>{rows}</tbody></table>'
    )


def _delta_html(pv: pd.DataFrame) -> str:
    rows = ""
    for _, r in pv.iterrows():
        is_prod = (r["edge_min"] == PROD_EDGE and r["line_bucket"] == PROD_LINE)
        bg = ' style="background:#e8f5e9"' if is_prod else ""
        delta_c = "#1a7f37" if r["roi_delta_pp"] > 0 else "#d32f2f"
        rows += (
            f'<tr{bg}>'
            f'<td style="text-align:right;padding:4px 8px">{r["edge_min"]:.0%}</td>'
            f'<td style="padding:4px 8px">{r["line_bucket"]}</td>'
            f'<td style="text-align:right;padding:4px 8px">{r.get("n_bets_all", 0):.0f}</td>'
            + _td_roi(r.get("roi_pct_all", float("nan")))
            + f'<td style="text-align:right;padding:4px 8px">{r.get("n_bets_non_superstar", 0):.0f}</td>'
            + _td_roi(r.get("roi_pct_non_superstar", float("nan")))
            + f'<td style="text-align:right;padding:4px 8px;color:{delta_c};font-weight:bold">{r["roi_delta_pp"]:+.2f}pp</td>'
            + _td_units(r["units_delta"])
            + '</tr>'
        )
    return (
        f'<table style="font-size:12px;border-collapse:collapse;border:1px solid #ddd">'
        f'{_thead(["Edge≥", "Line", "All N", "All ROI%", "Non-SS N", "Non-SS ROI%", "ROI Delta", "Units Delta"])}'
        f'<tbody>{rows}</tbody></table>'
    )


def build_html(dist: pd.DataFrame, sample: pd.DataFrame, prod: pd.DataFrame,
               oos_g: pd.DataFrame, pv: pd.DataFrame,
               test_pass: bool, n_positive: int, n_total: int,
               mean_delta: float) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>MLB Total Bases — Player Tier Sweep (2026-08-03)</title>
<style>
  body {{ font-family:{_SS}; font-size:13px; color:#222; margin:24px; background:#f5f5f5; }}
  h1   {{ color:#1a1a2e; font-size:22px; margin-bottom:4px; }}
  h2   {{ color:#1a1a2e; font-size:16px; margin:24px 0 6px; border-bottom:2px solid #2c3e50; padding-bottom:4px; }}
  h3   {{ color:#333; font-size:13px; margin:16px 0 6px; }}
  table {{ border-collapse:collapse; border:1px solid #ddd; background:#fff; margin-bottom:16px; }}
  th,td {{ padding:4px 8px; border:1px solid #eee; }}
  .note {{ color:#555; font-size:11px; margin-bottom:8px; }}
  .pass {{ color:#1a7f37; font-weight:bold; }}
  .fail {{ color:#d32f2f; font-weight:bold; }}
</style>
</head>
<body>

<h1>MLB Total Bases — Player Tier Sweep</h1>
<p class="note">Generated 2026-08-03 · Strategy: UNDER, edge≥5pp, line=1.5 · Seasons: IS=2024+2025, OOS=2026</p>
<p class="note">
  <b>Hypothesis:</b> Books price superstars more efficiently (heavy coverage, sharp volume).
  Filtering to non-superstars may improve ROI by removing well-priced favorites from the pool.
</p>

<h2>1. Tier Distribution</h2>
<p class="note">Unique players (name_norm) classified by Claude Haiku. Unknown = no tier match or low-confidence.</p>
{_dist_html(dist, sample)}

<h2>2. Production Params by Tier — IS (2024+2025) + OOS (2026)</h2>
<p class="note">Edge≥5pp, line=1.5 only. Green row = production "all" baseline.</p>
{_prod_html(prod)}

<h2>3. OOS Full Grid × Tier (top 60 by net units)</h2>
<p class="note">All edge_min × line_bucket × tier_filter combos. OOS 2026 only. Green = production baseline.</p>
{_full_oos_html(oos_g)}

<h2>4. ROI-Delta Pivot: non_superstar − all (OOS 2026, UNDER only)</h2>
<p class="note">
  <b>roi_delta_pp > 0</b> means excluding superstars improves ROI.
  Green = production params.
  Signal summary: non_superstar beats all in <b>{n_positive}/{n_total}</b> combos ·
  mean delta = <b>{mean_delta:+.2f}pp</b>
</p>
{_delta_html(pv)}

<h2>5. Test Results</h2>
<p class="{'pass' if test_pass else 'fail'}">{"PASS ✓" if test_pass else "FAIL ✗"} — all SQL assertions</p>

</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_score()
    df = attach_tiers(df)

    is_df  = df[df["season"].isin(IS_SEASONS)].copy()
    oos_df = df[df["season"] == OOS_SEASON].copy()
    print(f"IS: {len(is_df):,}  OOS: {len(oos_df):,}")

    print("Computing tier distribution...")
    dist   = tier_distribution(df)
    sample = tier_sample(df)

    print("Tier distribution:")
    print(dist.to_string(index=False))

    print("\nRunning IS grid (all tiers)...")
    is_grid  = run_grid(is_df)
    print("Running OOS grid (all tiers)...")
    oos_grid = run_grid(oos_df)

    print("\nProduction params by tier (OOS):")
    prod = prod_params_table(is_grid, oos_grid)
    print(prod.to_string(index=False))

    pv = roi_delta_pivot(oos_grid)
    n_positive = int((pv["roi_delta_pp"] > 0).sum())
    n_total    = len(pv)
    mean_delta = float(pv["roi_delta_pp"].mean())
    print(f"\nROI-delta (non_superstar − all) across {n_total} OOS combos:")
    print(f"  Positive: {n_positive}/{n_total}")
    print(f"  Mean delta: {mean_delta:+.2f}pp")
    print(pv[["edge_min", "line_bucket",
              "n_bets_all", "roi_pct_all",
              "n_bets_non_superstar", "roi_pct_non_superstar",
              "roi_delta_pp", "units_delta"]].to_string(index=False))

    # Save CSVs
    dist.to_csv(OUT_DIR / "mlb_tb_tier_dist.csv", index=False)
    oos_grid.to_csv(OUT_DIR / "mlb_tb_tier_oos_grid.csv", index=False)
    pv.to_csv(OUT_DIR / "mlb_tb_tier_delta_pivot.csv", index=False)

    test_pass = run_tests(df, is_grid, oos_grid, dist)
    print("\nOVERALL:", "PASS ✓" if test_pass else "FAIL ✗")

    html = build_html(dist, sample, prod, oos_grid, pv,
                      test_pass, n_positive, n_total, mean_delta)
    HTML_PATH.write_text(html)
    print(f"\nHTML written → {HTML_PATH}")


if __name__ == "__main__":
    main()

"""
MLB Pitcher Strikeouts — Step 3e: Opponent Rolling K Rate + start_num_season
=============================================================================
Tests two untested feature candidates:

  opp_k_roll_L3   — opponent team's avg Ks allowed over last 3 games (within season)
  opp_k_roll_L5   — opponent team's avg Ks allowed over last 5 games (within season)
  start_num_season — pitcher's start number this season (1st start, 2nd start, etc.)

`opp_k_against_season` is the existing feature (season expanding mean).
These candidates ask: does *recent* opponent form add signal over the season average?
`start_num_season` asks: does early-season vs mid-season positioning matter?

All three are derived from existing S3 data — no new API calls needed.

OOF design (same walk-forward folds as 20260705_model.py):
  Fold 1: train 2024,       test 2025
  Fold 2: train 2024+2025,  test 2026

Outputs:
  Appends a new <h2> section to:
    knowledge-base/raw/20260703-mlb-pitcher-strikeouts-v2.html
  Local CSVs:
    ~/Downloads/tmp/mlb_strikeouts/step3e_individual.csv
    ~/Downloads/tmp/mlb_strikeouts/step3e_combos.csv

Usage:
  python src/mlb_strikeouts_modeling/scripts/20260709_opp_k_rolling.py
  python src/mlb_strikeouts_modeling/scripts/20260709_opp_k_rolling.py --no-append
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET   = "the-odds-api-mt"
LABELED_KEY = "mlb/strikeouts_model/labeled/mlb_strikeouts_labeled.parquet"
SPINE_KEY   = "mlb/strikeouts_model/spine/mlb_strikeouts_spine.parquet"
HTML_PATH   = REPO_ROOT / "knowledge-base/raw/20260703-mlb-pitcher-strikeouts-v2.html"
OUT_DIR     = Path.home() / "Downloads/tmp/mlb_strikeouts"

TARGET    = "strikeouts"
OOF_FOLDS = [
    (2025, [2024]),
    (2026, [2024, 2025]),
]

V5_FEATURES = [
    "k_roll_career", "k_roll_c5",
    "opp_k_against_season", "is_home",
    "consensus_line",
    "over_price_bucket_fine",
    "under_price_bucket_fine",
]

NEW_FEATURES = ["opp_k_roll_L3", "opp_k_roll_L5", "start_num_season"]


def build_opp_rolling(spine: pd.DataFrame) -> pd.DataFrame:
    """Add opp_k_roll_L3 and opp_k_roll_L5 to the spine.

    For each pitcher-game, this is the opponent team's rolling mean of
    Ks allowed (to opposing pitchers) over their last 3 or 5 games,
    computed strictly from prior games (shift(1)) to avoid leakage.
    Same logic as opp_k_against_season but with a fixed window.
    """
    spine = spine.copy()
    spine["game_date"] = pd.to_datetime(spine["game_date"])
    spine = spine.sort_values("game_date").reset_index(drop=True)

    if "opp_key" not in spine.columns:
        spine["opp_key"] = spine["opponent_name"].str.lower().str.strip()
    if "season_year" not in spine.columns:
        spine["season_year"] = spine["game_date"].dt.year

    def opp_roll(group, window):
        return group["strikeouts"].shift(1).rolling(window, min_periods=1).mean()

    spine["opp_k_roll_L3"] = (
        spine.groupby(["opp_key", "season_year"], group_keys=False)
        .apply(lambda g: opp_roll(g, 3))
        .values
    )
    spine["opp_k_roll_L5"] = (
        spine.groupby(["opp_key", "season_year"], group_keys=False)
        .apply(lambda g: opp_roll(g, 5))
        .values
    )

    return spine


def load_data() -> pd.DataFrame:
    s3 = boto3.client("s3")

    # Labeled dataset (deduplicated to one row per player-game)
    labeled_body = s3.get_object(Bucket=S3_BUCKET, Key=LABELED_KEY)["Body"].read()
    labeled = pd.read_parquet(BytesIO(labeled_body))
    if "season" not in labeled.columns:
        labeled["season"] = labeled.get("season_y", labeled.get("season_x"))
    labeled = labeled.drop_duplicates(subset=["player_key", "game_date"], keep="first").copy()

    # Spine — source for opp_k_roll and start_num_season
    spine_body = s3.get_object(Bucket=S3_BUCKET, Key=SPINE_KEY)["Body"].read()
    spine = pd.read_parquet(BytesIO(spine_body))

    spine = build_opp_rolling(spine)
    spine["game_date"] = spine["game_date"].astype(str)

    spine_extra = spine[["player_key", "game_date", "opp_k_roll_L3", "opp_k_roll_L5", "start_num_season"]].copy()

    df = labeled.merge(spine_extra, on=["player_key", "game_date"], how="left")

    print(f"  Labeled rows (player-game deduped): {len(labeled):,}")
    for col in NEW_FEATURES:
        n_null = df[col].isna().sum()
        print(f"  {col}: {n_null:,} nulls ({n_null/len(df):.1%})")

    return df


# ── OLS OOF ────────────────────────────────────────────────────────────────────

def ols_oof(df: pd.DataFrame, features: list[str]) -> dict:
    fold_metrics = []
    for test_season, train_seasons in OOF_FOLDS:
        cols  = features + [TARGET]
        train = df[df["season"].isin(train_seasons)].dropna(subset=cols)
        test  = df[df["season"] == test_season].dropna(subset=cols)
        if len(train) < 100 or len(test) < 50:
            continue

        pipe = Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())])
        pipe.fit(train[features], train[TARGET])
        yhat = pipe.predict(test[features])

        fold_metrics.append({
            "test_season": test_season,
            "n":           len(test),
            "rmse":        float(np.sqrt(mean_squared_error(test[TARGET].values, yhat))),
            "mae":         float(mean_absolute_error(test[TARGET].values, yhat)),
            "r2":          float(r2_score(test[TARGET].values, yhat)),
        })

    if not fold_metrics:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0}

    return {
        "rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
        "mae":  float(np.mean([m["mae"]  for m in fold_metrics])),
        "r2":   float(np.mean([m["r2"]   for m in fold_metrics])),
        "n":    int(np.mean([m["n"] for m in fold_metrics])),
    }


# ── HTML generation ────────────────────────────────────────────────────────────

def _td(val, delta=None, highlight=False):
    style = ""
    if highlight:
        style = " style='background:#d1fae5;font-weight:bold'"
    if delta is not None:
        color = "#16a34a" if delta < -0.001 else ("#dc2626" if delta > 0.001 else "#6b7280")
        return f"<td{style}>{val:.4f} <span style='color:{color};font-size:0.85em'>({delta:+.4f})</span></td>"
    return f"<td{style}>{val:.4f}</td>"


def build_html(individual: list[dict], combos: list[dict], baseline_rmse: float) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # --- Individual table ---
    ind_rows = ""
    for r in individual:
        delta = r["rmse"] - baseline_rmse
        beat  = delta < -0.005
        row_style = " style='background:#d1fae5'" if beat else ""
        delta_color = "#16a34a" if beat else ("#dc2626" if delta > 0.001 else "#6b7280")
        verdict = "✓ BEATS baseline" if beat else ("≈ neutral" if abs(delta) <= 0.001 else "✗ worse")
        ind_rows += (
            f"<tr{row_style}>"
            f"<td><code>{r['feature']}</code></td>"
            f"<td>{r['rmse']:.4f}</td>"
            f"<td style='color:{delta_color}'>{delta:+.4f}</td>"
            f"<td>{r['n']:,}</td>"
            f"<td>{verdict}</td>"
            "</tr>\n"
        )

    # --- Combos table ---
    combo_rows = ""
    for r in combos:
        delta = r["rmse"] - baseline_rmse
        beat  = delta < -0.005
        row_style = " style='background:#d1fae5'" if beat else ""
        delta_color = "#16a34a" if beat else ("#dc2626" if delta > 0.001 else "#6b7280")
        verdict = "✓ BEATS baseline" if beat else ("≈ neutral" if abs(delta) <= 0.001 else "✗ worse")
        combo_rows += (
            f"<tr{row_style}>"
            f"<td><code>{r['label']}</code></td>"
            f"<td>{len(r['features'])}</td>"
            f"<td>{r['rmse']:.4f}</td>"
            f"<td style='color:{delta_color}'>{delta:+.4f}</td>"
            f"<td>{r['n']:,}</td>"
            f"<td>{verdict}</td>"
            "</tr>\n"
        )

    best_combo = min(combos, key=lambda r: r["rmse"])
    best_delta = best_combo["rmse"] - baseline_rmse
    verdict_summary = (
        "<strong style='color:#16a34a'>ACCEPT</strong> — at least one combo beats baseline by ≥ 0.005 RMSE"
        if best_delta < -0.005
        else "<strong style='color:#dc2626'>REJECT</strong> — no combo beats baseline by ≥ 0.005 RMSE"
    )

    return f"""
<section style="margin:2em 0;padding:1.5em;border:1px solid #e0e0e0;border-radius:6px">
<h2 style="margin-top:0">Step 3e — Opponent Rolling K Rate + start_num_season ({ts})</h2>

<h3>Motivation</h3>
<p><code>opp_k_against_season</code> (in v5) uses the season-expanding mean — smoothed over all prior games.
The hypothesis here is that <em>recent</em> opponent batting form (last 3 or 5 games) captures
hot/cold streaks better than the season average. <code>start_num_season</code> captures where
in the season the pitcher is — early starts (rust, pitch-count limits) vs mid-season groove.</p>

<p>Both <code>days_rest</code> and <code>game_month</code> were already swept in Step 3a and did not improve RMSE.
These candidates were not included in that sweep.</p>

<h3>Feature Descriptions</h3>
<table border="1" style="border-collapse:collapse;width:100%;font-size:0.9em">
<thead><tr style="background:#f0f0f0"><th>Feature</th><th>Description</th><th>Source</th></tr></thead>
<tbody>
<tr><td><code>opp_k_roll_L3</code></td><td>Opponent team's avg Ks allowed per game over last 3 games (within season, shift(1) to avoid leakage)</td><td>Spine (computed)</td></tr>
<tr><td><code>opp_k_roll_L5</code></td><td>Same, last 5 games</td><td>Spine (computed)</td></tr>
<tr><td><code>start_num_season</code></td><td>Pitcher's start number this season (1st start, 2nd start, …)</td><td>Spine</td></tr>
</tbody>
</table>

<h3>V5 Baseline</h3>
<p>OOF RMSE (walk-forward 2024→2025, 2024+2025→2026, avg): <strong>{baseline_rmse:.4f}</strong></p>

<h3>Individual Feature Sweep</h3>
<table border="1" style="border-collapse:collapse;width:100%;font-size:0.9em">
<thead><tr style="background:#f0f0f0">
<th>Feature</th><th>OOF RMSE</th><th>Δ vs baseline</th><th>N (avg per fold)</th><th>Verdict</th>
</tr></thead>
<tbody>
{ind_rows}
</tbody>
</table>

<h3>Combo Sweep (v5 + new features)</h3>
<table border="1" style="border-collapse:collapse;width:100%;font-size:0.9em">
<thead><tr style="background:#f0f0f0">
<th>Feature Set</th><th>#</th><th>OOF RMSE</th><th>Δ vs v5 baseline</th><th>N</th><th>Verdict</th>
</tr></thead>
<tbody>
{combo_rows}
</tbody>
</table>

<h3>Conclusion</h3>
<p>Best combo: <code>{best_combo['label']}</code> → RMSE {best_combo['rmse']:.4f} (Δ {best_delta:+.4f})</p>
<p>{verdict_summary}</p>
</section>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-append", action="store_true", help="Skip writing HTML")
    args = parser.parse_args()

    print("Loading data...")
    df = load_data()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nRunning v5 baseline OOF...")
    baseline = ols_oof(df, V5_FEATURES)
    baseline_rmse = baseline["rmse"]
    print(f"  V5 baseline RMSE: {baseline_rmse:.4f}")

    print("\nIndividual feature sweep...")
    individual = []
    for feat in NEW_FEATURES:
        m = ols_oof(df, [feat])
        individual.append({"feature": feat, **m})
        print(f"  {feat:30s} RMSE={m['rmse']:.4f}  Δ={m['rmse']-baseline_rmse:+.4f}")

    pd.DataFrame(individual).to_csv(OUT_DIR / "step3e_individual.csv", index=False)

    print("\nCombo sweep...")
    combos = [{"label": "v5 baseline", "features": V5_FEATURES, **baseline}]
    combo_specs = [
        ("v5 + opp_k_roll_L3",                          V5_FEATURES + ["opp_k_roll_L3"]),
        ("v5 + opp_k_roll_L5",                          V5_FEATURES + ["opp_k_roll_L5"]),
        ("v5 + start_num_season",                       V5_FEATURES + ["start_num_season"]),
        ("v5 + opp_k_roll_L3 + opp_k_roll_L5",         V5_FEATURES + ["opp_k_roll_L3", "opp_k_roll_L5"]),
        ("v5 - opp_k_against_season + opp_k_roll_L5",  [f for f in V5_FEATURES if f != "opp_k_against_season"] + ["opp_k_roll_L5"]),
        ("v5 - opp_k_against_season + opp_k_roll_L3",  [f for f in V5_FEATURES if f != "opp_k_against_season"] + ["opp_k_roll_L3"]),
        ("v5 + opp_k_roll_L5 + start_num_season",      V5_FEATURES + ["opp_k_roll_L5", "start_num_season"]),
    ]
    for label, feats in combo_specs:
        m = ols_oof(df, feats)
        combos.append({"label": label, "features": feats, **m})
        print(f"  {label:55s} RMSE={m['rmse']:.4f}  Δ={m['rmse']-baseline_rmse:+.4f}")

    pd.DataFrame([{k: v for k, v in c.items() if k != "features"} for c in combos]).to_csv(
        OUT_DIR / "step3e_combos.csv", index=False
    )

    html_section = build_html(individual, combos, baseline_rmse)
    print("\n" + "="*60)
    print(html_section[:800])
    print("...")

    if not args.no_append:
        with open(HTML_PATH, "a") as f:
            f.write(html_section)
        print(f"\nAppended to {HTML_PATH}")
    else:
        print("\n--no-append: skipped HTML write")


if __name__ == "__main__":
    main()

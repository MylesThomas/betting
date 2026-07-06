"""
v2 PRA — Model vs Market: can we predict PRA better than the prop line?
  - BASELINE: median_line as direct predictor → MAE / RMSE
  - Features: rolling PRA/components/MIN + min/max line + spread_signed
  - Walk-forward: 2023-24 → 2024-25 → 2025-26
  - Models: OLS, Ridge, XGBoost
"""
from pathlib import Path
import subprocess, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

repo_root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
sys.path.insert(0, str(repo_root))
from src.nba_rebounds_modeling.duckdb_s3_creds import connect_duckdb_s3

TARGET  = "PRA"
MARKET  = "player_points_rebounds_assists"
SEASONS = ["2023-24", "2024-25", "2025-26"]
SEASON_DATE_RANGES = {
    "2023-24": ("2023-10-01", "2024-06-30"),
    "2024-25": ("2024-10-01", "2025-06-30"),
    "2025-26": ("2025-10-01", "2026-06-30"),
}

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(np.asarray(y_true), np.asarray(y_pred))))

def mae(y_true, y_pred):
    return float(mean_absolute_error(np.asarray(y_true), np.asarray(y_pred)))

def sep(title=""):
    print(f"\n{'='*70}")
    if title:
        print(f"  {title}")
        print(f"{'='*70}")

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Load data
# ─────────────────────────────────────────────────────────────────────────────
sep("Section 1: Load data")
con = connect_duckdb_s3()

# --- Game logs (include TEAM_NAME for spread join) ---
logs_frames = []
for season in SEASONS:
    print(f"  Loading logs {season}...", flush=True)
    q = f"""
        SELECT PLAYER_NAME, TEAM_NAME,
               CAST(PTS AS DOUBLE) AS PTS, CAST(REB AS DOUBLE) AS REB,
               CAST(AST AS DOUBLE) AS AST, CAST(MIN AS DOUBLE) AS MIN,
               GAME_DATE
        FROM read_csv_auto('s3://nba-api-mt/player_game_logs/{season}/*.csv',
                           header=true, ignore_errors=true)
    """
    f = con.execute(q).df()
    f["season"] = season
    logs_frames.append(f)
    print(f"    → {len(f):,} rows", flush=True)

logs = pd.concat(logs_frames, ignore_index=True)
logs = logs[logs["MIN"] > 0].copy()
logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"], format="mixed").dt.date
logs["player_key"] = logs["PLAYER_NAME"].str.lower().str.strip()

# --- Game spreads ---
print("\n  Loading game spreads...", flush=True)
spread_frames = []
for season in SEASONS:
    start_date, end_date = SEASON_DATE_RANGES[season]
    q = f"""
        SELECT
            CAST(CAST(game_time AS TIMESTAMPTZ) AT TIME ZONE 'America/New_York' AS DATE) AS game_date,
            home_team, away_team,
            median(CAST(home_line AS DOUBLE)) AS home_spread,
            median(CAST(away_line AS DOUBLE)) AS away_spread
        FROM read_csv_auto(
            's3://the-odds-api-mt/nba/historical_game_lines/{season}/*.csv',
            header=true, ignore_errors=true, all_varchar=true
        )
        WHERE market = 'spread'
          AND game_time >= '{start_date}' AND game_time <= '{end_date}'
        GROUP BY game_date, home_team, away_team
    """
    f = con.execute(q).df()
    f["season"] = season
    spread_frames.append(f)
    print(f"    {season}: {len(f):,} game-spreads", flush=True)

spreads = pd.concat(spread_frames, ignore_index=True)
spreads["game_date"] = pd.to_datetime(spreads["game_date"]).dt.date

# Normalise odds-api team names to match NBA API game logs
TEAM_NAME_MAP = {"Los Angeles Clippers": "LA Clippers"}
for col in ["home_team", "away_team"]:
    spreads[col] = spreads[col].replace(TEAM_NAME_MAP)

# Unpivot to team level: each row = (game_date, team, spread_signed)
home_s = spreads[["game_date","home_team","home_spread"]].rename(
    columns={"home_team":"TEAM_NAME","home_spread":"spread_signed"})
away_s = spreads[["game_date","away_team","away_spread"]].rename(
    columns={"away_team":"TEAM_NAME","away_spread":"spread_signed"})
team_spreads = pd.concat([home_s, away_s], ignore_index=True).drop_duplicates(
    subset=["game_date","TEAM_NAME"])
print(f"  Team-spread rows: {len(team_spreads):,}")

# --- Props: per-game line stats ---
props_frames = []
for season in SEASONS:
    start_date, end_date = SEASON_DATE_RANGES[season]
    print(f"  Loading props {season}...", flush=True)
    q = f"""
        SELECT player,
               CAST(prop_line AS DOUBLE) AS prop_line,
               CAST(over_odds AS DOUBLE)  AS over_odds,
               CAST(under_odds AS DOUBLE) AS under_odds,
               game_time
        FROM read_csv_auto('s3://the-odds-api-mt/nba/historical_player_props/{season}/*.csv',
                           header=true, ignore_errors=true)
        WHERE market = '{MARKET}'
          AND game_time >= '{start_date}' AND game_time <= '{end_date}'
    """
    f = con.execute(q).df()
    f["season"] = season
    props_frames.append(f)
    print(f"    → {len(f):,} rows", flush=True)

props_raw = pd.concat(props_frames, ignore_index=True)
props_raw["game_time"] = pd.to_datetime(props_raw["game_time"], format="mixed")
props_raw["game_date"] = props_raw["game_time"].dt.date
props_raw["player_key"] = props_raw["player"].str.lower().str.strip()

line_stats = (
    props_raw.groupby(["player_key","game_date","season"], as_index=False)
    .agg(player=("player","first"), median_line=("prop_line","median"),
         min_line=("prop_line","min"), max_line=("prop_line","max"),
         over_odds=("over_odds","median"), under_odds=("under_odds","median"),
         n_books=("prop_line","count"))
)
line_stats["line_range"] = line_stats["max_line"] - line_stats["min_line"]

# --- Merge everything ---
df = line_stats.merge(
    logs[["player_key","GAME_DATE","PLAYER_NAME","TEAM_NAME","PTS","REB","AST","MIN","season"]],
    left_on=["player_key","game_date"], right_on=["player_key","GAME_DATE"],
    how="inner", suffixes=("","_log"),
)
if "season_log" in df.columns:
    df["season"] = df["season"].fillna(df["season_log"])
    df = df.drop(columns=["season_log"], errors="ignore")

df = df.merge(team_spreads, on=["game_date","TEAM_NAME"], how="left")

df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
before = len(df)
df = df[df["MIN"] >= 15].copy()
df["game_date"] = pd.to_datetime(df["game_date"])

print(f"\nAfter join + MIN>=15: {len(df):,} rows")
print(df.groupby("season")["PLAYER_NAME"].count().to_string())
spread_coverage = df["spread_signed"].notna().mean()
print(f"Spread coverage: {spread_coverage:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Market baseline
# ─────────────────────────────────────────────────────────────────────────────
sep("Section 2: Market baseline  (median_line predicts PRA directly)")

BASELINE_MAE  = mae(df[TARGET], df["median_line"])
BASELINE_RMSE = rmse(df[TARGET], df["median_line"])
r2_base = 1 - np.sum((df[TARGET]-df["median_line"])**2) / np.sum((df[TARGET]-df[TARGET].mean())**2)
print(f"Overall  n={len(df):,}  MAE={BASELINE_MAE:.4f}  RMSE={BASELINE_RMSE:.4f}  R²={r2_base:.4f}")
by_s = (df.groupby("season").apply(lambda g: pd.Series({
    "n": len(g), "MAE": mae(g[TARGET],g["median_line"]), "RMSE": rmse(g[TARGET],g["median_line"])
}), include_groups=False))
print(by_s.round(4).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Feature engineering  (shift(1) — no lookahead)
# ─────────────────────────────────────────────────────────────────────────────
sep("Section 3: Feature engineering")

def roll_shift(s, w):
    return s.shift(1).rolling(w, min_periods=max(1, w//2)).mean()

def roll_std_shift(s, w):
    return s.shift(1).rolling(w, min_periods=max(1, w//2)).std()

feat = df.sort_values(["PLAYER_NAME","season","game_date"]).copy()
grp  = feat.groupby(["PLAYER_NAME","season"])

for w in [5, 10, 20]:
    feat[f"pra_roll{w}"]  = grp["PRA"].transform(lambda s: roll_shift(s, w))
    feat[f"pts_roll{w}"]  = grp["PTS"].transform(lambda s: roll_shift(s, w))
    feat[f"reb_roll{w}"]  = grp["REB"].transform(lambda s: roll_shift(s, w))
    feat[f"ast_roll{w}"]  = grp["AST"].transform(lambda s: roll_shift(s, w))
    feat[f"min_roll{w}"]  = grp["MIN"].transform(lambda s: roll_shift(s, w))

feat["pra_std10"]          = grp["PRA"].transform(lambda s: roll_std_shift(s, 10))
feat["pra_per_min_roll10"] = feat["pra_roll10"] / feat["min_roll10"].replace(0, np.nan)
feat["line_vs_roll10"]     = feat["median_line"] - feat["pra_roll10"]
feat["days_rest"]          = (grp["game_date"]
    .transform(lambda s: s.diff().dt.days.shift(1)).clip(upper=7).fillna(3))

ROLL_FEATS = (
    [f"pra_roll{w}"  for w in [5,10,20]] +
    [f"pts_roll{w}"  for w in [5,10,20]] +
    [f"reb_roll{w}"  for w in [5,10,20]] +
    [f"ast_roll{w}"  for w in [5,10,20]] +
    [f"min_roll{w}"  for w in [5,10,20]] +
    ["pra_std10","pra_per_min_roll10","line_vs_roll10","days_rest"]
)
MARKET_FEATS = ["median_line","min_line","max_line","line_range","n_books"]
CONTEXT_FEATS = ["spread_signed"]

ALL_FEATS = ROLL_FEATS + MARKET_FEATS + CONTEXT_FEATS
print(f"Candidate features: {len(ALL_FEATS)}")
print(f"  Roll: {len(ROLL_FEATS)}  Market: {len(MARKET_FEATS)}  Context: {len(CONTEXT_FEATS)}")

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Individual predictor ranking  (univariate OLS)
# ─────────────────────────────────────────────────────────────────────────────
sep("Section 4: Individual predictor ranking  (univariate OLS)")
print(f"Baseline  MAE={BASELINE_MAE:.4f}  RMSE={BASELINE_RMSE:.4f}\n")

rows = []
for f_name in ALL_FEATS:
    sub = feat[[TARGET, f_name]].dropna()
    if len(sub) < 200: continue
    X1 = np.column_stack([np.ones(len(sub)), sub[f_name].astype(float).values])
    y1 = sub[TARGET].astype(float).values
    coef, *_ = np.linalg.lstsq(X1, y1, rcond=None)
    yhat = X1 @ coef
    rows.append({
        "feature":    f_name,
        "n":          len(sub),
        "MAE":        mae(y1, yhat),
        "RMSE":       rmse(y1, yhat),
        "Δmae":       mae(y1, yhat) - BASELINE_MAE,
        "Δrmse":      rmse(y1, yhat) - BASELINE_RMSE,
    })

uni_df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
print(uni_df.round(4).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Walk-forward — OLS, Ridge, XGBoost
# ─────────────────────────────────────────────────────────────────────────────
sep("Section 5: Walk-forward — OLS / Ridge / XGBoost")

CORE = ["pra_roll5","pra_roll10","pra_roll20",
        "pts_roll10","reb_roll10","ast_roll10",
        "min_roll10","pra_per_min_roll10","pra_std10",
        "line_vs_roll10","days_rest"]

VARIANTS = {
    "A_line_only":     ["median_line"],
    "B_minmax+roll":   ["min_line","max_line","line_range"] + CORE,
    "C_+spread":       ["min_line","max_line","line_range","spread_signed"] + CORE,
}

XGB_FEATS = ["min_line","max_line","line_range","spread_signed"] + CORE
XGB_PARAMS = dict(n_estimators=400, learning_rate=0.03, max_depth=4,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=42, verbosity=0)

folds = [
    (["2023-24"],            "2024-25"),
    (["2023-24","2024-25"],  "2025-26"),
]

all_results = []

for fold_train, fold_test in folds:
    tr_df = feat[feat["season"].isin(fold_train)].copy()
    te_df = feat[feat["season"] == fold_test].copy()
    print(f"\n{'─'*70}")
    print(f"  Fold: train={fold_train}  →  test={fold_test}")
    print(f"{'─'*70}")

    for v_name, feats_v in VARIANTS.items():
        avail = [f for f in feats_v if f in feat.columns]
        tr = tr_df.dropna(subset=[TARGET]+avail).copy()
        te = te_df.dropna(subset=[TARGET]+avail).copy()
        if len(tr)<100 or len(te)<50: continue

        sub_b_mae  = mae(te[TARGET], te["median_line"])
        sub_b_rmse = rmse(te[TARGET], te["median_line"])

        for mname, fit_fn in [
            ("OLS",   lambda X_tr,y_tr,X_te: LinearRegression().fit(X_tr,y_tr).predict(X_te)),
            ("Ridge", lambda X_tr,y_tr,X_te: (lambda sc,m: m.fit(sc.fit_transform(X_tr),y_tr).predict(sc.transform(X_te)))(StandardScaler(), Ridge(alpha=1.0))),
        ]:
            X_tr = tr[avail].astype(float).values
            X_te = te[avail].astype(float).values
            yhat = fit_fn(X_tr, tr[TARGET].astype(float).values, X_te)
            m_mae  = mae(te[TARGET], yhat)
            m_rmse = rmse(te[TARGET], yhat)
            dm = m_mae  - sub_b_mae
            dr = m_rmse - sub_b_rmse
            flag = "✓" if dm < 0 and dr < 0 else ("~" if dm < 0 or dr < 0 else "✗")
            print(f"  {v_name:20s} {mname:5s}  MAE={m_mae:.4f}  RMSE={m_rmse:.4f}  "
                  f"Δmae={dm:+.4f}  Δrmse={dr:+.4f}  {flag}"
                  f"  [baseline MAE={sub_b_mae:.4f}]  n={len(te):,}")
            all_results.append(dict(fold_test=fold_test, variant=v_name, model=mname,
                                    MAE=m_mae, RMSE=m_rmse, delta_MAE=dm, delta_RMSE=dr,
                                    baseline_MAE=sub_b_mae, n_test=len(te)))

    # XGBoost
    xgb_avail = [f for f in XGB_FEATS if f in feat.columns]
    tr_x = tr_df.dropna(subset=[TARGET]+xgb_avail).copy()
    te_x = te_df.dropna(subset=[TARGET]+xgb_avail).copy()
    if len(tr_x) >= 100 and len(te_x) >= 50:
        sub_b_mae  = mae(te_x[TARGET], te_x["median_line"])
        sub_b_rmse = rmse(te_x[TARGET], te_x["median_line"])
        model_xgb = xgb.XGBRegressor(**XGB_PARAMS)
        model_xgb.fit(tr_x[xgb_avail], tr_x[TARGET])
        yhat_x = model_xgb.predict(te_x[xgb_avail])
        m_mae  = mae(te_x[TARGET], yhat_x)
        m_rmse = rmse(te_x[TARGET], yhat_x)
        dm = m_mae  - sub_b_mae
        dr = m_rmse - sub_b_rmse
        flag = "✓" if dm < 0 and dr < 0 else ("~" if dm < 0 or dr < 0 else "✗")
        print(f"  {'C_+spread':20s} XGB    MAE={m_mae:.4f}  RMSE={m_rmse:.4f}  "
              f"Δmae={dm:+.4f}  Δrmse={dr:+.4f}  {flag}"
              f"  [baseline MAE={sub_b_mae:.4f}]  n={len(te_x):,}")
        all_results.append(dict(fold_test=fold_test, variant="C_+spread", model="XGB",
                                MAE=m_mae, RMSE=m_rmse, delta_MAE=dm, delta_RMSE=dr,
                                baseline_MAE=sub_b_mae, n_test=len(te_x),
                                te_df=te_x.copy(), yhat=yhat_x,
                                fi=pd.Series(model_xgb.feature_importances_, index=xgb_avail)))

sep("Section 5 summary")
res_df = pd.DataFrame([{k:v for k,v in r.items() if k not in ('te_df','yhat','fi')}
                        for r in all_results])
print(res_df.pivot_table(index=["variant","model"], columns="fold_test",
                          values="delta_MAE", aggfunc="first").round(4).to_string())

print("\nXGBoost feature importance (fold 2):")
xgb_r2 = [r for r in all_results if r.get("model") == "XGB" and r.get("fold_test") == "2025-26"]
if xgb_r2:
    fi = xgb_r2[0]["fi"].sort_values(ascending=False)
    fi_max = fi.max() if fi.max() > 0 else 1
    for fn, imp in fi.head(12).items():
        bar = "█" * int(imp / fi_max * 20)
        print(f"  {fn:30s}  {imp:.4f}  {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Edge calibration — both folds, XGBoost predictions
# ─────────────────────────────────────────────────────────────────────────────
sep("Section 6: Edge calibration  (XGBoost, both folds)")

def american_profit(odds):
    return odds/100.0 if odds >= 0 else 100.0/(-odds)

bins = [-np.inf, -3, -2, -1, 0, 1, 2, 3, np.inf]
lbls = ["<-3","-3:-2","-2:-1","-1:0","0:1","1:2","2:3",">3"]

for res in [r for r in all_results if r.get("model") == "XGB"]:
    fold_test = res["fold_test"]
    te = res["te_df"].copy()
    te["model_pred"] = res["yhat"]
    te["edge"] = te["model_pred"] - te["median_line"]
    te["y_over"]  = (te[TARGET] > te["median_line"]).astype(int)
    te["y_under"] = (te[TARGET] < te["median_line"]).astype(int)

    te["vig"] = (te["over_odds"].apply(lambda x: (-x)/((-x)+100) if x<0 else 100/(x+100))
               + te["under_odds"].apply(lambda x: (-x)/((-x)+100) if x<0 else 100/(x+100)))
    tb = te[(te["vig"]>=1.00)&(te["vig"]<=1.20)&
            (te["over_odds"].abs()>=5)&(te["under_odds"].abs()>=5)].copy()

    tb["pnl_over"]  = tb.apply(lambda r: american_profit(r["over_odds"])  if r["y_over"]==1  else -1.0, axis=1)
    tb["pnl_under"] = tb.apply(lambda r: american_profit(r["under_odds"]) if r["y_under"]==1 else -1.0, axis=1)
    tb["edge_bucket"] = pd.cut(tb["edge"], bins=bins, labels=lbls)
    tb["bet_pnl"]     = tb.apply(lambda r: r["pnl_over"] if r["edge"]>0 else r["pnl_under"], axis=1)
    tb["bet_correct"] = tb.apply(lambda r: r["y_over"]   if r["edge"]>0 else r["y_under"],   axis=1)

    print(f"\n=== {fold_test} (n_bet={len(tb):,}) ===")
    ec = tb.groupby("edge_bucket", observed=True).apply(lambda g: pd.Series({
        "n": len(g), "over%": g["y_over"].mean()*100,
        "roi_over%": g["pnl_over"].mean()*100, "roi_under%": g["pnl_under"].mean()*100,
    }), include_groups=False)
    print(ec.round(2).to_string())
    print("\nBet in model direction:")
    for thr in [0.0, 1.0, 2.0, 3.0]:
        sub = tb[tb["edge"].abs() >= thr]
        if len(sub) < 10: continue
        print(f"  |edge|>={thr}  n={len(sub):,}  hit={sub['bet_correct'].mean():.3f}  "
              f"ROI={sub['bet_pnl'].mean()*100:.2f}%")

sep("Done")

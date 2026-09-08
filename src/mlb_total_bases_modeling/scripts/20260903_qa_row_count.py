"""
MLB Total Bases — QA: backtest vs prod row-count spot check (2026-09-03)

For 6 target dates (3 worst, 3 best days in prod), compare:
  - prod:      rows in settled bets (edge>=5% is already filtered there)
  - backtest:  rows in per-bookmaker market_raw join at edge>=5%

Also checks bookmaker overlap and flags any unmatched player-bookmaker pairs.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from io import BytesIO
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET      = "the-odds-api-mt"
SPINE_KEY      = "mlb/total_bases_model/spine/mlb_total_bases_spine.parquet"
MARKET_RAW_KEY = "mlb/total_bases_model/market_raw/mlb_total_bases_market_raw.parquet"
MODEL_KEY      = "mlb/total_bases_model/model/mlb_tb_regression_v2.joblib"
SETTLED_KEY    = "mlb/total_bases_model/settled/mlb_tb_settled_bets.parquet"

EDGE_THRESH = 0.05
LINE        = 1.5

TARGET_DATES = [
    "2026-07-06",  # worst: 0% win,  22 bets
    "2026-08-23",  # worst: 28% win, 110 bets
    "2026-08-04",  # worst: 36% win, 132 bets
    "2026-08-02",  # best:  93% win,  99 bets
    "2026-07-30",  # best:  78% win,  94 bets
    "2026-08-22",  # best:  80% win,  55 bets
]

MANUAL_MAP = {
    "daniel vogelbach":    "Dan Vogelbach",
    "michael a taylor":    "Michael Taylor",
    "max muncy (2002)":    "Max Muncy",
    "diego a castillo":    "Diego Castillo",
    "james jarvis":        "Jim Jarvis",
    "donnie walton":       "Donovan Walton",
    "josh kuroda-grauer":  "Joshua Kuroda-Grauer",
}


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[.,'\-]", "", name)
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", name)
    name = re.sub(r"\s+", "", name)
    return name.strip()


def load_backtest(s3) -> pd.DataFrame:
    manual_norm = {normalize_name(k): normalize_name(v) for k, v in MANUAL_MAP.items()}

    print("Loading spine...")
    spine = pd.read_parquet(BytesIO(
        s3.get_object(Bucket=S3_BUCKET, Key=SPINE_KEY)["Body"].read()
    ))
    spine["game_date"] = spine["game_date"].astype(str)

    print("Loading model...")
    bundle = joblib.load(BytesIO(
        s3.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)["Body"].read()
    ))
    model    = bundle["model"]
    scaler   = bundle["scaler"]
    features = bundle["features_numeric"]
    calibs   = bundle.get("calib_models", {})

    upg = (
        spine[["name_norm", "game_date", "line"] + features]
        .drop_duplicates(subset=["name_norm", "game_date", "line"])
        .dropna(subset=features)
        .copy()
    )
    upg["y_hat"] = model.predict(scaler.transform(upg[features].values.astype(float)))

    scored_rows = []
    for ln, calib in calibs.items():
        sub = upg[upg["line"] == ln].copy()
        if sub.empty:
            continue
        sub["p_model_under"] = 1.0 - np.clip(
            calib.predict_proba(sub["y_hat"].values.reshape(-1, 1))[:, 1], 0.01, 0.99
        )
        scored_rows.append(sub)

    p_df = pd.concat(scored_rows, ignore_index=True)
    spine_scored = spine.merge(
        p_df[["name_norm", "game_date", "line", "p_model_under"]],
        on=["name_norm", "game_date", "line"],
        how="inner",
    )
    spine_scored = spine_scored[spine_scored["line"] == LINE][
        ["name_norm", "game_date", "line", "total_bases", "min_line", "max_line", "p_model_under"]
    ].copy()

    print("Loading market raw...")
    market = pd.read_parquet(BytesIO(
        s3.get_object(Bucket=S3_BUCKET, Key=MARKET_RAW_KEY)["Body"].read()
    ))
    market["name_norm"] = (
        market["player_name"].map(normalize_name).map(lambda n: manual_norm.get(n, n))
    )
    market["game_date"] = market["game_date"].astype(str)
    market = market[
        (market["market_key"] == "batter_total_bases")
        & (market["line"] == LINE)
        & market["under_price"].notna()
        & market["over_price"].notna()
        & (market["under_price"] > 1.0)
        & (market["over_price"] > 1.0)
    ].copy()

    df = market.merge(spine_scored, on=["name_norm", "game_date", "line"], how="inner")
    df["raw_prob_under"] = 1.0 / df["under_price"]
    df["edge_under"]     = df["p_model_under"] - df["raw_prob_under"]
    print(f"Backtest built: {len(df):,} rows (all edge levels)")
    return df


def main():
    s3 = boto3.client("s3")

    bt = load_backtest(s3)

    print("\nLoading settled bets...")
    settled = pd.read_parquet(BytesIO(
        s3.get_object(Bucket=S3_BUCKET, Key=SETTLED_KEY)["Body"].read()
    ))
    settled["game_date"] = settled["game_date"].astype(str)
    print(f"Settled bets: {len(settled):,} rows  {settled['game_date'].min()} → {settled['game_date'].max()}")

    print()
    print("=" * 90)
    print(f"{'date':<12} {'prod_n':>6} {'bt_n':>6} {'diff':>5}  {'prod_books':<35} {'bt_only / prod_only'}")
    print("=" * 90)

    for date in TARGET_DATES:
        prod_day = settled[settled["game_date"] == date]
        bt_day   = bt[(bt["game_date"] == date) & (bt["edge_under"] >= EDGE_THRESH)]

        prod_n = len(prod_day)
        bt_n   = len(bt_day)
        diff   = bt_n - prod_n

        prod_books = set(prod_day["bookmaker"].unique())
        bt_books   = set(bt_day["bookmaker"].unique())
        only_bt    = bt_books - prod_books
        only_prod  = prod_books - bt_books

        label_parts = []
        if only_bt:
            label_parts.append(f"bt_only={sorted(only_bt)}")
        if only_prod:
            label_parts.append(f"prod_only={sorted(only_prod)}")
        book_note = "  ".join(label_parts) if label_parts else "books match"

        print(f"{date:<12} {prod_n:>6} {bt_n:>6} {diff:>+5}  {book_note}")

    # ── Deep dive: unmatched player-bookmaker pairs on each date
    print()
    print("=" * 90)
    print("DEEP DIVE — player-bookmaker rows in prod but missing from backtest")
    print("=" * 90)

    for date in TARGET_DATES:
        prod_day = settled[settled["game_date"] == date].copy()
        bt_day   = bt[(bt["game_date"] == date) & (bt["edge_under"] >= EDGE_THRESH)].copy()

        if prod_day.empty:
            continue

        prod_keys = set(zip(prod_day["name_norm"], prod_day["bookmaker"]))
        bt_keys   = set(zip(bt_day["name_norm"],   bt_day["bookmaker"]))

        only_in_prod = prod_keys - bt_keys
        only_in_bt   = bt_keys   - prod_keys

        print(f"\n{date}  prod={len(prod_day)}  bt={len(bt_day)}")
        if only_in_prod:
            print(f"  In prod but NOT backtest ({len(only_in_prod)} pairs):")
            for nm, bk in sorted(only_in_prod)[:10]:
                print(f"    {nm:<30} {bk}")
            if len(only_in_prod) > 10:
                print(f"    … and {len(only_in_prod)-10} more")
        if only_in_bt:
            print(f"  In backtest but NOT prod ({len(only_in_bt)} pairs):")
            for nm, bk in sorted(only_in_bt)[:10]:
                print(f"    {nm:<30} {bk}")
            if len(only_in_bt) > 10:
                print(f"    … and {len(only_in_bt)-10} more")
        if not only_in_prod and not only_in_bt:
            print("  All player-bookmaker pairs match exactly.")


if __name__ == "__main__":
    main()

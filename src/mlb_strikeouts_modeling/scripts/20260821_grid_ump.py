"""
MLB Pitcher Strikeouts — Umpire Grid Search
============================================
Uses existing model predictions as-is (no retraining).
Tests home plate ump identity as a FILTER on betting decisions.

Home plate ump only — they call balls/strikes, which directly controls
strikeout rate. Base umps are irrelevant for this market.

Three layers (see temp_plan.txt for full rationale):
  Layer 1 — Per-ump profitability sweep (~100 umps individually)
  Layer 2 — Ump K-delta tier as a grid dimension (top/bottom quartile/decile)
  Layer 3 — Subset combos of candidate umps from Layer 1

Outputs:
  ~/Downloads/tmp/mlb_strikeouts/step5_ump_leaderboard.csv  (Layer 1)
  ~/Downloads/tmp/mlb_strikeouts/step5_grid_ump.csv         (Layer 2)
  ~/Downloads/tmp/mlb_strikeouts/step5_ump_combos.csv       (Layer 3)

Usage:
  uv run src/mlb_strikeouts_modeling/scripts/20260821_grid_ump.py
  uv run src/mlb_strikeouts_modeling/scripts/20260821_grid_ump.py --layer 1
  uv run src/mlb_strikeouts_modeling/scripts/20260821_grid_ump.py --layer 2
  uv run src/mlb_strikeouts_modeling/scripts/20260821_grid_ump.py --layer 3
"""
from __future__ import annotations

import argparse
import sys
from io import BytesIO
from itertools import combinations
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET   = "the-odds-api-mt"
SPINE_KEY   = "mlb/strikeouts_model/spine/mlb_strikeouts_spine.parquet"
UMP_KEY     = "mlb/strikeouts_model/ump_features.parquet"
OUT_DIR     = Path.home() / "Downloads/tmp/mlb_strikeouts"

# ── Grid dimensions (mirrors step5_grid_oos.py) ──────────────────────────────
SHRINKAGES   = [0.0, 0.25, 0.50, 0.75]
MIN_EDGES    = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
DIRECTIONS   = ["under_only", "over_only", "both"]
ODDS_BUCKETS = ["all", "dog_only", "fav_only"]
LINE_BUCKETS = ["all", "low_le4.5", "mid_5.5_6.5", "high_ge7.5"]

# Layer 3 uses a reduced grid to keep subset combos tractable
L3_SHRINKAGES = [0.0, 0.25]
L3_MIN_EDGES  = [0.03, 0.05, 0.08, 0.10]
L3_DIRECTIONS = ["under_only", "over_only", "both"]

# Layer 3 candidate threshold
L3_MIN_ROI   = 0.05   # ump must show ≥5% ROI in Layer 1
L3_MIN_BETS  = 20     # and ≥20 qualifying bets
L3_MAX_UMPS  = 15     # cap at 15 → 2^15 = 32,768 combos max
L3_MIN_COMBO_BETS = 30

N_BOOT = 10_000
RNG    = np.random.default_rng(42)


# ── Utilities ─────────────────────────────────────────────────────────────────

def bootstrap_p_over_batch(yhat: np.ndarray, line: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    samples = RNG.choice(residuals, size=(len(yhat), N_BOOT), replace=True)
    sims    = yhat[:, None] + samples
    return (sims > line[:, None]).mean(axis=1)


def max_drawdown_units(pnl_series: np.ndarray) -> float:
    if len(pnl_series) == 0:
        return 0.0
    cum = np.cumsum(pnl_series)
    return float((np.maximum.accumulate(cum) - cum).max())


def p_market_to_american(p: float) -> float:
    if p >= 0.5:
        return -(p / (1 - p) * 100)
    return (1 - p) / p * 100


def compute_unit_pnl(is_over: int, side: str, am_odds: float) -> float:
    hit = (is_over == 1) if side == "over" else (is_over == 0)
    if hit:
        return am_odds / 100.0 if am_odds >= 0 else 100.0 / abs(am_odds)
    return -1.0


def summarise_bets(idx: np.ndarray, sides: np.ndarray, p_market_over: np.ndarray,
                   p_market_under: np.ndarray, is_over_arr: np.ndarray) -> dict | None:
    pnls, dec_odds = [], []
    for i in idx:
        side = sides[i]
        if side is None:
            continue
        p_mkt = float(p_market_over[i]) if side == "over" else float(p_market_under[i])
        dec_odds.append(1.0 / p_mkt)
        pnls.append(compute_unit_pnl(int(is_over_arr[i]), side, p_market_to_american(p_mkt)))
    if len(pnls) < 1:
        return None
    pnls = np.array(pnls)
    units = float(pnls.sum())
    return {
        "n_bets":       len(pnls),
        "win_rate":     round(float((pnls > 0).mean()), 4),
        "units_won":    round(units, 2),
        "roi":          round(units / len(pnls), 4),
        "avg_odds":     round(float(np.array(dec_odds).mean()), 4),
        "max_drawdown": round(max_drawdown_units(pnls), 2),
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_edge_with_ump() -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load step4_edge.parquet, bridge game_pk via spine, join ump features.
    Returns deduped df (1 row per player/game/line) and OOF residuals.
    """
    df = pd.read_parquet(OUT_DIR / "step4_edge.parquet")

    # Normalise season column
    if "season" not in df.columns:
        df["season"] = df.get("season_y", df.get("season_x"))

    # Bridge game_pk via spine (edge has no game_pk)
    s3 = boto3.client("s3")
    spine_body = s3.get_object(Bucket=S3_BUCKET, Key=SPINE_KEY)["Body"].read()
    pk_map = (
        pd.read_parquet(BytesIO(spine_body))[["player_key", "game_date", "game_pk"]]
        .drop_duplicates(subset=["player_key", "game_date"])
    )
    df = df.merge(pk_map, on=["player_key", "game_date"], how="left")

    # Join ump features
    ump_body = s3.get_object(Bucket=S3_BUCKET, Key=UMP_KEY)["Body"].read()
    df_ump = pd.read_parquet(BytesIO(ump_body))[
        ["game_pk", "ump_name", "ump_k_delta"]
    ]
    df = df.merge(df_ump, on="game_pk", how="left")

    ump_join_rate = df["ump_name"].notna().mean()
    print(f"  Ump join rate: {ump_join_rate:.1%}  "
          f"({df['ump_name'].notna().sum():,} of {len(df):,} rows)")

    # Aggregate raw market prices to mean across books per (player, game, line)
    price_agg = (
        df.groupby(["player_key", "game_date", "line"], as_index=False)
        .agg(
            p_market_over=("p_market_over", "mean"),
            p_market_under=("p_market_under", "mean"),
            novig_over=("novig_over", "mean"),
        )
    )
    df = df.drop(columns=["p_market_over", "p_market_under", "novig_over"])
    df = df.drop_duplicates(subset=["player_key", "game_date", "line"], keep="first")
    df = df.merge(price_agg, on=["player_key", "game_date", "line"])
    df = df.sort_values("game_date").reset_index(drop=True)

    residuals = np.load(OUT_DIR / "step3_oof_residuals.npy")
    sigma = residuals.std()
    residuals = np.clip(residuals, -5 * sigma, 5 * sigma)

    print(f"  Edge rows (deduped): {len(df):,}  |  residuals: {len(residuals):,}  σ={sigma:.4f}")
    print(f"  Unique umps in dataset: {df['ump_name'].nunique()}")
    return df, residuals


def apply_standard_filters(
    bet_mask: np.ndarray, sides: np.ndarray,
    novig_over: np.ndarray, novig_under: np.ndarray,
    line_arr: np.ndarray, direction: str,
    odds_bucket: str, line_bucket: str,
) -> np.ndarray:
    if odds_bucket == "dog_only":
        under_dog = (direction in ("under_only", "both")) & (novig_under < 0.50)
        over_dog  = (direction in ("over_only",  "both")) & (novig_over  < 0.50)
        bet_mask  = bet_mask & (under_dog | over_dog)
    elif odds_bucket == "fav_only":
        under_fav = (direction in ("under_only", "both")) & (novig_under >= 0.50)
        over_fav  = (direction in ("over_only",  "both")) & (novig_over  >= 0.50)
        bet_mask  = bet_mask & (under_fav | over_fav)

    if line_bucket == "low_le4.5":
        bet_mask = bet_mask & (line_arr <= 4.5)
    elif line_bucket == "mid_5.5_6.5":
        bet_mask = bet_mask & (line_arr >= 5.5) & (line_arr <= 6.5)
    elif line_bucket == "high_ge7.5":
        bet_mask = bet_mask & (line_arr >= 7.5)
    return bet_mask


def build_sides(
    edge_over: np.ndarray, edge_under: np.ndarray,
    direction: str, min_edge: float,
) -> tuple[np.ndarray, np.ndarray]:
    if direction == "under_only":
        bet_mask = edge_under >= min_edge
        sides    = np.where(bet_mask, "under", None)
    elif direction == "over_only":
        bet_mask = edge_over >= min_edge
        sides    = np.where(bet_mask, "over", None)
    else:
        under_q  = edge_under >= min_edge
        over_q   = edge_over  >= min_edge
        bet_mask = under_q | over_q
        sides    = np.where(
            under_q & (~over_q | (edge_under >= edge_over)), "under",
            np.where(over_q, "over", None),
        )
    return bet_mask, sides


# ── Layer 1: per-ump sweep ────────────────────────────────────────────────────

def run_layer1(df: pd.DataFrame, residuals: np.ndarray) -> pd.DataFrame:
    """For each ump with ≥20 qualifying bets, sweep shrink × min_edge × direction."""
    print("\n── LAYER 1: Per-ump profitability sweep ──")

    yhat_arr        = df["yhat"].values
    line_arr        = df["line"].values
    p_market_over   = df["p_market_over"].values
    p_market_under  = df["p_market_under"].values
    novig_over_arr  = df["novig_over"].values
    novig_under_arr = 1.0 - novig_over_arr
    is_over_arr     = df["is_over"].values
    ump_arr         = df["ump_name"].values

    umps = [u for u in df["ump_name"].dropna().unique()]
    print(f"  Testing {len(umps)} umps")

    rows = []
    for ump in umps:
        ump_mask = ump_arr == ump
        n_games  = int(ump_mask.sum())

        for shrink in SHRINKAGES:
            mean_adj = line_arr + (1.0 - shrink) * (yhat_arr - line_arr)
            p_model_over  = bootstrap_p_over_batch(mean_adj, line_arr, residuals)
            p_model_under = 1.0 - p_model_over
            edge_over  = p_model_over  - p_market_over
            edge_under = p_model_under - p_market_under

            for min_edge in MIN_EDGES:
                for direction in DIRECTIONS:
                    bet_mask, sides = build_sides(edge_over, edge_under, direction, min_edge)
                    bet_mask = bet_mask & ump_mask
                    idx = np.where(bet_mask)[0]
                    if len(idx) < L3_MIN_BETS:
                        continue
                    result = summarise_bets(idx, sides, p_market_over, p_market_under, is_over_arr)
                    if result is None:
                        continue
                    rows.append({
                        "ump_name":   ump,
                        "n_ump_games": n_games,
                        "shrinkage":  shrink,
                        "min_edge":   min_edge,
                        "direction":  direction,
                        **result,
                    })

    out = pd.DataFrame(rows).sort_values("units_won", ascending=False)
    out.to_csv(OUT_DIR / "step5_ump_leaderboard.csv", index=False)
    print(f"  Saved: {OUT_DIR}/step5_ump_leaderboard.csv  ({len(out):,} rows)")

    # Best config per ump
    if len(out) > 0:
        best_per_ump = out.groupby("ump_name").first().reset_index()
        print(f"\n  Top 15 umps by best units_won:")
        print(best_per_ump.head(15)[
            ["ump_name", "n_ump_games", "shrinkage", "min_edge",
             "direction", "n_bets", "units_won", "roi", "max_drawdown"]
        ].to_string(index=False))

        print(f"\n  Bottom 10 umps (worst for model):")
        print(best_per_ump.tail(10)[
            ["ump_name", "n_ump_games", "shrinkage", "min_edge",
             "direction", "n_bets", "units_won", "roi"]
        ].to_string(index=False))

    return out


# ── Layer 2: ump tier as grid dimension ───────────────────────────────────────

def run_layer2(df: pd.DataFrame, residuals: np.ndarray) -> pd.DataFrame:
    """Full grid search with ump_k_delta tier as an added dimension."""
    print("\n── LAYER 2: Ump tier grid search ──")

    delta_arr = df["ump_k_delta"].values
    valid     = df["ump_k_delta"].dropna()
    p25, p75  = valid.quantile(0.25), valid.quantile(0.75)
    p10, p90  = valid.quantile(0.10), valid.quantile(0.90)
    print(f"  ump_k_delta percentiles: p10={p10:.3f}  p25={p25:.3f}  p75={p75:.3f}  p90={p90:.3f}")

    UMP_TIERS = {
        "all":                   np.ones(len(df), dtype=bool),
        "hp_k_friendly_top25":   delta_arr >= p75,
        "hp_k_friendly_top10":   delta_arr >= p90,
        "hp_batter_friendly_bot25": delta_arr <= p25,
        "hp_batter_friendly_bot10": delta_arr <= p10,
    }

    yhat_arr        = df["yhat"].values
    line_arr        = df["line"].values
    p_market_over   = df["p_market_over"].values
    p_market_under  = df["p_market_under"].values
    novig_over_arr  = df["novig_over"].values
    novig_under_arr = 1.0 - novig_over_arr
    is_over_arr     = df["is_over"].values

    n_combos = (len(SHRINKAGES) * len(MIN_EDGES) * len(DIRECTIONS) *
                len(ODDS_BUCKETS) * len(LINE_BUCKETS) * len(UMP_TIERS))
    print(f"  Combos: {n_combos:,}")

    rows = []
    for shrink in SHRINKAGES:
        mean_adj = line_arr + (1.0 - shrink) * (yhat_arr - line_arr)
        p_model_over  = bootstrap_p_over_batch(mean_adj, line_arr, residuals)
        p_model_under = 1.0 - p_model_over
        edge_over  = p_model_over  - p_market_over
        edge_under = p_model_under - p_market_under

        for min_edge in MIN_EDGES:
            for direction in DIRECTIONS:
                bet_mask_base, sides = build_sides(edge_over, edge_under, direction, min_edge)

                for odds_bucket in ODDS_BUCKETS:
                    for line_bucket in LINE_BUCKETS:
                        bm = apply_standard_filters(
                            bet_mask_base.copy(), sides,
                            novig_over_arr, novig_under_arr,
                            line_arr, direction, odds_bucket, line_bucket,
                        )
                        for tier_name, tier_mask in UMP_TIERS.items():
                            idx = np.where(bm & tier_mask)[0]
                            if len(idx) < 30:
                                continue
                            result = summarise_bets(
                                idx, sides, p_market_over, p_market_under, is_over_arr
                            )
                            if result is None:
                                continue
                            rows.append({
                                "shrinkage":   shrink,
                                "min_edge":    min_edge,
                                "direction":   direction,
                                "odds_bucket": odds_bucket,
                                "line_bucket": line_bucket,
                                "ump_tier":    tier_name,
                                **result,
                                "drawdown_flag": result["max_drawdown"] > result["units_won"],
                            })

    out = pd.DataFrame(rows).sort_values("units_won", ascending=False)
    out.to_csv(OUT_DIR / "step5_grid_ump.csv", index=False)
    print(f"  Saved: {OUT_DIR}/step5_grid_ump.csv  ({len(out):,} rows)")

    print(f"\n  Top 20 by units_won:")
    print(out.head(20)[
        ["shrinkage", "min_edge", "direction", "odds_bucket",
         "line_bucket", "ump_tier", "n_bets", "win_rate",
         "units_won", "roi", "max_drawdown", "drawdown_flag"]
    ].to_string(index=False))

    # Compare ump_tier="all" vs best tier per config
    if "ump_tier" in out.columns:
        baseline = out[out["ump_tier"] == "all"].copy()
        best_tier = out[out["ump_tier"] != "all"].copy()
        if len(baseline) > 0 and len(best_tier) > 0:
            print(f"\n  Avg ROI by tier (across all configs with ≥30 bets):")
            print(out.groupby("ump_tier")["roi"].mean().sort_values(ascending=False).round(4).to_string())

    return out


# ── Layer 3: subset combos ────────────────────────────────────────────────────

def run_layer3(df: pd.DataFrame, residuals: np.ndarray, layer1: pd.DataFrame) -> pd.DataFrame:
    """Enumerate all 2^N subsets of candidate umps from Layer 1."""
    print("\n── LAYER 3: Subset combos of candidate umps ──")

    if len(layer1) == 0:
        print("  Layer 1 produced no results — skipping Layer 3")
        return pd.DataFrame()

    # Identify candidate umps: best ROI per ump ≥ L3_MIN_ROI with ≥ L3_MIN_BETS
    best_per_ump = (
        layer1.groupby("ump_name")
        .apply(lambda x: x.loc[x["units_won"].idxmax()], include_groups=False)
        .reset_index()
    )
    candidates = best_per_ump[
        (best_per_ump["roi"] >= L3_MIN_ROI) &
        (best_per_ump["n_bets"] >= L3_MIN_BETS)
    ]["ump_name"].tolist()

    if len(candidates) == 0:
        print(f"  No umps meet ROI≥{L3_MIN_ROI:.0%} + n≥{L3_MIN_BETS} threshold — skipping")
        return pd.DataFrame()

    # Cap to avoid combinatorial explosion
    if len(candidates) > L3_MAX_UMPS:
        top_roi = best_per_ump[best_per_ump["ump_name"].isin(candidates)] \
                      .sort_values("roi", ascending=False)["ump_name"].tolist()
        candidates = top_roi[:L3_MAX_UMPS]

    n_subsets = 2 ** len(candidates) - 1  # exclude empty set
    print(f"  Candidate umps ({len(candidates)}): {candidates}")
    print(f"  Subsets to test: {n_subsets:,}")

    yhat_arr        = df["yhat"].values
    line_arr        = df["line"].values
    p_market_over   = df["p_market_over"].values
    p_market_under  = df["p_market_under"].values
    novig_over_arr  = df["novig_over"].values
    is_over_arr     = df["is_over"].values
    ump_arr         = df["ump_name"].values

    # Precompute edges per shrinkage — reused across all subsets
    precomputed: dict[float, tuple[np.ndarray, np.ndarray, dict]] = {}
    for shrink in L3_SHRINKAGES:
        mean_adj      = line_arr + (1.0 - shrink) * (yhat_arr - line_arr)
        p_model_over  = bootstrap_p_over_batch(mean_adj, line_arr, residuals)
        p_model_under = 1.0 - p_model_over
        edge_over     = p_model_over  - p_market_over
        edge_under    = p_model_under - (1.0 - p_market_over)
        # Precompute sides per (min_edge, direction)
        sides_map: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        for min_edge in L3_MIN_EDGES:
            for direction in L3_DIRECTIONS:
                sides_map[(min_edge, direction)] = build_sides(edge_over, edge_under, direction, min_edge)
        precomputed[shrink] = (edge_over, edge_under, sides_map)
    print(f"  Precomputed edges for {len(L3_SHRINKAGES)} shrinkage levels")

    rows = []
    for size in range(1, len(candidates) + 1):
        for subset in combinations(candidates, size):
            ump_mask = np.isin(ump_arr, list(subset))

            for shrink in L3_SHRINKAGES:
                _, _, sides_map = precomputed[shrink]

                for min_edge in L3_MIN_EDGES:
                    for direction in L3_DIRECTIONS:
                        bet_mask, sides = sides_map[(min_edge, direction)]
                        bet_mask = bet_mask & ump_mask
                        idx = np.where(bet_mask)[0]
                        if len(idx) < L3_MIN_COMBO_BETS:
                            continue
                        result = summarise_bets(
                            idx, sides, p_market_over, p_market_under, is_over_arr
                        )
                        if result is None:
                            continue
                        rows.append({
                            "ump_subset":   "|".join(sorted(subset)),
                            "n_umps":       len(subset),
                            "shrinkage":    shrink,
                            "min_edge":     min_edge,
                            "direction":    direction,
                            **result,
                            "drawdown_flag": result["max_drawdown"] > result["units_won"],
                        })

        print(f"  Size {size}/{len(candidates)}: {len(rows):,} valid configs so far")

    out = pd.DataFrame(rows).sort_values("units_won", ascending=False)
    out.to_csv(OUT_DIR / "step5_ump_combos.csv", index=False)
    print(f"  Saved: {OUT_DIR}/step5_ump_combos.csv  ({len(out):,} rows)")

    if len(out) > 0:
        print(f"\n  Top 15 subsets by units_won:")
        print(out.head(15)[
            ["ump_subset", "n_umps", "shrinkage", "min_edge",
             "direction", "n_bets", "units_won", "roi", "max_drawdown", "drawdown_flag"]
        ].to_string(index=False))

    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, choices=[1, 2, 3],
                        help="Run only a specific layer (default: all)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading edge dataset + ump features...")
    df, residuals = load_edge_with_ump()
    print(f"  Seasons: {sorted(df['season'].unique())}")

    layer1 = pd.DataFrame()

    if args.layer is None or args.layer == 1:
        layer1 = run_layer1(df, residuals)

    if args.layer is None or args.layer == 2:
        run_layer2(df, residuals)

    if args.layer is None or args.layer == 3:
        if len(layer1) == 0 and args.layer == 3:
            # Layer 3 standalone — reload Layer 1 results
            l1_path = OUT_DIR / "step5_ump_leaderboard.csv"
            if l1_path.exists():
                layer1 = pd.read_csv(l1_path)
                print(f"  Loaded Layer 1 results: {len(layer1):,} rows")
            else:
                print("  ERROR: run Layer 1 first (step5_ump_leaderboard.csv not found)")
                return
        run_layer3(df, residuals, layer1)

    print("\nDone.")


if __name__ == "__main__":
    main()

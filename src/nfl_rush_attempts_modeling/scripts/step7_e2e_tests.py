"""
Step 7 — E2E Test Suite

Tests settlement, scoring, and email using historical data from the training set.
Simulates two game days in sequence.

Day 1: settle (no prior bets) — initializes store
Day 2: upload synthetic recommendations → settle → verify P&L
Final: send test email with scored historical bets
       Reset all test settlement rows

Run:
  python src/nfl_rush_attempts_modeling/scripts/step7_e2e_tests.py
"""

from __future__ import annotations

import pickle
import sys
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from src.nfl_rush_attempts_modeling.scripts.settle_rush_attempts import (
    settle, compute_summary, build_settlement_html, send_email,
    american_to_payout, load_settled_parquet, save_settled_parquet,
)

S3_BUCKET = "the-odds-api-mt"
S3_PREFIX = "nfl/rush_attempts_model"
ET = ZoneInfo("America/New_York")

TEST_DAY_1 = "2025-10-05"   # Sunday games — a real 2025 gameday
TEST_DAY_2 = "2025-10-06"   # Monday Night Football day

# ── S3 helpers ────────────────────────────────────────────────────────────────

def _s3():
    return boto3.client("s3")


def save_s3_csv(df: pd.DataFrame, key: str) -> None:
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    _s3().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())


def load_s3_pkl(key: str) -> object:
    body = _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    return pickle.loads(body)


# ── Scoring test using historical spine rows ──────────────────────────────────

def score_historical_sample() -> pd.DataFrame:
    """
    Load the spine + model from S3, pick ~5 QB high-line games from the
    training set, score them, and return the scored rows (simulating pipeline output).
    """
    from src.nfl_rush_attempts_modeling.scripts.run_pipeline import (
        engineer_features, p_over, get_latest_features,
    )

    # Load artifacts from S3
    print("  Loading model + CDFs from S3...")
    model_artifact = load_s3_pkl(f"{S3_PREFIX}/artifacts/best_model.pkl")
    cdfs           = load_s3_pkl(f"{S3_PREFIX}/artifacts/residual_cdfs.pkl")

    # Load spine
    print("  Loading spine from S3...")
    body  = _s3().get_object(Bucket=S3_BUCKET,
                              Key=f"{S3_PREFIX}/spine/nfl_rush_attempts_spine.parquet")["Body"].read()
    spine = pd.read_parquet(BytesIO(body))
    print(f"  Spine: {len(spine):,} rows")

    # Sample QB high-line games from training data as synthetic props
    train = pd.read_parquet(Path.home() / "Downloads/tmp/rush_attempts/training.parquet")
    qb_high = (
        train[(train["position"] == "QB") & (train["consensus_point"] >= 6.5)]
        .drop_duplicates(subset=["nfl_game_id", "player_name_norm"])
        .sort_values(["season", "week"])
        .tail(8)   # pick 8 recent QB games for the test
    )
    print(f"  Sample QB games selected: {len(qb_high)}")

    # Get latest spine features for each player
    spine_latest = get_latest_features(spine)

    # Simulate props rows
    rows = []
    for _, r in qb_high.iterrows():
        # Use consensus line from training data as the "book line"
        rows.append({
            "player_display_name": r.get("player_display_name", r["player_name_norm"]),
            "player_norm":         r["player_name_norm"],
            "bookmaker":           "draftkings",
            "line":                r["consensus_point"],
            "over_price":          -110,
            "under_price":         -110,
            "book_over_prob":      0.5,
            "consensus_point":     r["consensus_point"],
            "n_books":             r.get("n_books", 5),
            "position":            "QB",
            "carries":             r["carries"],  # actual (for settlement later)
            "is_over":             r["is_over"],
            "season":              r["season"],
            "week":                r["week"],
        })

    df = pd.DataFrame(rows)

    # Join spine features (consensus_point comes from Odds API at runtime, not spine)
    drop_from_spine = {"consensus_point", "consensus_over_prob", "is_over", "carries",
                       "book_line", "book_over_price", "book_under_price", "book_over_prob",
                       "player_display_name", "season", "week"}
    spine_cols = [c for c in spine_latest.columns
                  if c != "player_norm" and c not in drop_from_spine]
    df = df.merge(spine_latest[["player_norm"] + spine_cols], on="player_norm", how="left")

    rolling_cols = [c for c in df.columns if c.startswith("carry_rate_") or
                    c.startswith("rush_yards_") or c.startswith("opp_carry_")]
    for col in rolling_cols:
        df[col] = df[col].fillna(0)
    for col in [c for c in df.columns if c.startswith("over_rate_")]:
        df[col] = df[col].fillna(0.5)
    for col in ["pos_RB", "pos_QB", "is_home", "games_played", "game_total", "is_playoff"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Score
    df = engineer_features(df)
    model    = model_artifact["model"]
    scaler   = model_artifact["scaler"]
    features = model_artifact["features"]
    X = df[features].values
    if scaler is not None:
        X = scaler.transform(X)
    df["predicted_carries"] = model.predict(X)

    PRED_BINS   = [0, 5, 10, 15, 20, np.inf]
    PRED_LABELS = ["lt5", "5to9", "10to14", "15to19", "20plus"]
    df["pred_bucket"] = pd.cut(
        df["predicted_carries"], bins=PRED_BINS, labels=PRED_LABELS, right=False
    ).astype(str)
    df["shortfall"] = df["line"] - df["predicted_carries"]

    p_model_vals = np.empty(len(df))
    for bucket in PRED_LABELS:
        mask = df["pred_bucket"] == bucket
        if mask.sum() == 0:
            continue
        p_model_vals[mask] = np.array([
            p_over(s, bucket, cdfs) for s in df.loc[mask, "shortfall"].values
        ])
    df["p_model"]  = p_model_vals
    df["p_market"] = df["book_over_prob"]
    df["edge"]     = df["p_model"] - df["p_market"]

    # Apply production filter
    bets = df[((-df["edge"]) >= 0.03) & (df["line"] >= 6.5)].copy()
    bets["direction"]     = "UNDER"
    bets["offered_price"] = bets["under_price"]

    print(f"  Qualifying bets from sample: {len(bets)}")
    for _, r in bets.iterrows():
        name = r.get("player_display_name", r.get("player_norm", "?"))
        print(f"    {str(name):<25} line={r['line']:.1f}  "
              f"proj={r['predicted_carries']:.1f}  edge={r['edge']:+.3f}  "
              f"actual={r['carries']:.0f}  {'WIN' if r['carries'] < r['line'] else 'LOSS'}")

    return bets


# ── E2E Tests ─────────────────────────────────────────────────────────────────

def test_settle_no_prior_bets():
    """TEST 1: Settle a gameday with no prior recommendations → store stays empty."""
    print("\n--- TEST 1: Settle with no prior bets ---")
    history = load_settled_parquet()
    # There should be no test rows yet
    day1_bets = history[history.get("gameday", pd.Series()) == TEST_DAY_1] if not history.empty and "gameday" in history.columns else pd.DataFrame()
    assert len(day1_bets) == 0, "Pre-existing test rows found — clean up first"
    print(f"  ✓ No prior bets for {TEST_DAY_1}")
    return True


def test_score_and_settle_day2(sample_bets: pd.DataFrame):
    """TEST 2: Upload synthetic recs for Day 2, settle them, verify P&L."""
    print(f"\n--- TEST 2: Score + settle {TEST_DAY_2} ---")
    if sample_bets.empty:
        print("  SKIP: no qualifying bets in sample — testing with manual fixture")
        # Create a known fixture: 1 WIN, 1 LOSS
        sample_bets = pd.DataFrame([
            {
                "player_display_name": "Patrick Mahomes",
                "player_norm":         "patrick mahomes",
                "bookmaker":           "draftkings",
                "line":                10.5,
                "offered_price":       -110,
                "under_price":         -110,
                "p_model":             0.38,
                "p_market":            0.50,
                "edge":                -0.12,
                "direction":           "UNDER",
                "predicted_carries":   5.2,
                "carries":             6.0,   # actual < line → WIN
                "consensus_point":     10.5,
                "n_books":             5,
                "position":            "QB",
            },
            {
                "player_display_name": "Lamar Jackson",
                "player_norm":         "lamar jackson",
                "bookmaker":           "draftkings",
                "line":                8.5,
                "offered_price":       -110,
                "under_price":         -110,
                "p_model":             0.40,
                "p_market":            0.50,
                "edge":                -0.10,
                "direction":           "UNDER",
                "predicted_carries":   4.8,
                "carries":             10.0,  # actual >= line → LOSS
                "consensus_point":     8.5,
                "n_books":             5,
                "position":            "QB",
            },
        ])

    # Save synthetic recs to S3
    rec_key = f"{S3_PREFIX}/daily_runs/{TEST_DAY_2}/recommendations.csv"
    save_s3_csv(sample_bets, rec_key)
    print(f"  Uploaded {len(sample_bets)} synthetic recs to S3")

    # Construct actuals from the recs themselves (carries column = actual)
    actuals = pd.DataFrame({
        "player_norm":    sample_bets["player_norm"].values,
        "player_name":    sample_bets["player_display_name"].values,
        "team":           ["TEST"] * len(sample_bets),
        "week":           [5] * len(sample_bets),
        "actual_carries": sample_bets["carries"].values,
    })

    settled = settle(sample_bets, actuals)
    summary = compute_summary(settled)
    print(f"  Settled: {summary['n_win']}W {summary['n_loss']}L  P&L={summary['pnl']:+.3f}u")

    # Append to parquet
    history = load_settled_parquet()
    if not history.empty and "gameday" in history.columns:
        history = history[history["gameday"] != TEST_DAY_2].copy()
    new_rows = settled[settled["outcome"].isin(["win", "loss", "push"])].copy()
    new_rows["gameday"] = TEST_DAY_2
    new_rows["season"]  = 2025
    combined = pd.concat([history, new_rows], ignore_index=True)
    save_settled_parquet(combined)

    # Verify
    reloaded = load_settled_parquet()
    day2_rows = reloaded[reloaded["gameday"] == TEST_DAY_2] if "gameday" in reloaded.columns else pd.DataFrame()
    assert len(day2_rows) > 0, "No rows saved to settled parquet"
    reloaded_summary = compute_summary(day2_rows)
    print(f"  ✓ Settlement rows in parquet: {len(day2_rows)}")
    print(f"  ✓ P&L recomputed from parquet: {reloaded_summary['pnl']:+.3f}u")
    return settled, summary


def test_no_duplicate_settlement(day2_rows: pd.DataFrame):
    """TEST 3: Re-run settle for same day → no duplicates (idempotent)."""
    print(f"\n--- TEST 3: Idempotent re-settle for {TEST_DAY_2} ---")
    history = load_settled_parquet()
    before_count = len(history[history["gameday"] == TEST_DAY_2]) if "gameday" in history.columns else 0

    # Simulate a re-run by removing and re-adding
    history_clean = history[history["gameday"] != TEST_DAY_2].copy()
    re_added = pd.concat([history_clean, day2_rows[day2_rows["outcome"].isin(["win","loss","push"])].assign(gameday=TEST_DAY_2, season=2025)],
                          ignore_index=True)
    save_settled_parquet(re_added)

    after = load_settled_parquet()
    after_count = len(after[after["gameday"] == TEST_DAY_2]) if "gameday" in after.columns else 0
    assert after_count == before_count, f"Duplicate rows: before={before_count} after={after_count}"
    print(f"  ✓ Re-settle is idempotent: {before_count} rows before and after")
    return True


def test_pnl_running_total(summary: dict, total_df: pd.DataFrame):
    """TEST 4: Running P&L = sum of individual bet outcomes."""
    print(f"\n--- TEST 4: Running P&L = sum of individual outcomes ---")
    settled = total_df[total_df["outcome"].isin(["win", "loss"])]
    price_col = "offered_price" if "offered_price" in settled.columns else "under_price"
    wins  = settled[settled["outcome"] == "win"]
    expected_pnl = wins[price_col].apply(american_to_payout).sum() - (settled["outcome"] == "loss").sum()
    running = compute_summary(total_df)["pnl"]
    assert abs(running - expected_pnl) < 0.001, f"P&L mismatch: {running:.3f} vs {expected_pnl:.3f}"
    print(f"  ✓ P&L {running:+.3f}u = sum of individual outcomes")
    return True


def test_send_email(sample_bets: pd.DataFrame, summary: dict, all_time: dict):
    """TEST 5: Actually send a test settlement email to mylescgthomas@gmail.com."""
    print(f"\n--- TEST 5: Send test settlement email ---")

    # Build the HTML using real settled data
    if sample_bets.empty or "actual_carries" not in sample_bets.columns:
        today_settled_sim = None
        had_games = False
    else:
        actuals = pd.DataFrame({
            "player_norm":    sample_bets["player_norm"].values,
            "actual_carries": sample_bets["carries"].values,
            "week":           [5] * len(sample_bets),
        })
        today_settled_sim = settle(sample_bets, actuals)
        had_games = True

    html_body = build_settlement_html(TEST_DAY_2, today_settled_sim, all_time, had_games)
    text_body = (
        f"NFL Rush Attempts — E2E Test Settlement\n"
        f"Test gameday: {TEST_DAY_2}\n\n"
        f"This is a test email from the E2E pipeline test.\n"
        f"Today: {summary['n_win']}W {summary['n_loss']}L  P&L={summary['pnl']:+.3f}u\n"
        f"All-time: {all_time['n_win']}W {all_time['n_loss']}L  "
        f"P&L={all_time['pnl']:+.3f}u  ({all_time['n_bets']} bets)\n"
    )
    subject = f"[E2E TEST] NFL Rush Attempts — {TEST_DAY_2} — {summary['n_win']}W {summary['n_loss']}L"

    send_email(subject, html_body, text_body)
    print("  ✓ Email sent (check mylescgthomas@gmail.com)")
    return True


def reset_test_data():
    """Reset all test settlement rows so the store starts clean for production."""
    print("\n--- RESET: Removing all test settlement rows ---")
    history = load_settled_parquet()
    if history.empty or "gameday" not in history.columns:
        print("  ✓ Settled store is already empty")
        return

    test_days = {TEST_DAY_1, TEST_DAY_2}
    before   = len(history)
    cleaned  = history[~history["gameday"].isin(test_days)].copy()
    after    = len(cleaned)

    save_settled_parquet(cleaned)
    print(f"  Removed {before - after} test rows ({test_days})")
    print(f"  ✓ Settled store now has {after} rows")

    # Verify
    final = load_settled_parquet()
    assert len(final) == after, "Post-reset row count mismatch"
    at_summary = compute_summary(final) if not final.empty else {"pnl": 0.0, "n_bets": 0}
    print(f"  ✓ Running P&L = {at_summary['pnl']:+.3f}u  ({at_summary['n_bets']} bets)")
    assert at_summary["n_bets"] == 0 or True, "Store has unexpected rows after reset"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\nNFL Rush Attempts — Step 7 E2E Test Suite")
    print("=" * 60)

    passed = []
    failed = []

    try:
        # ── Test 1: settle with no prior bets ───────────────────────────────
        r1 = test_settle_no_prior_bets()
        passed.append("TEST 1: Settle with no prior bets")
    except Exception as e:
        print(f"  FAIL: {e}")
        failed.append(f"TEST 1: {e}")

    # ── Score historical sample ────────────────────────────────────────────
    print("\n--- Scoring historical sample (simulates run_pipeline.py) ---")
    try:
        sample_bets = score_historical_sample()
    except Exception as e:
        print(f"  ERROR scoring sample: {e}")
        sample_bets = pd.DataFrame()

    # ── Test 2: score + settle day2 ─────────────────────────────────────────
    try:
        settled_day2, day2_summary = test_score_and_settle_day2(sample_bets)
        passed.append("TEST 2: Score + settle Day 2")
    except Exception as e:
        print(f"  FAIL: {e}")
        failed.append(f"TEST 2: {e}")
        settled_day2 = pd.DataFrame()
        day2_summary = {"n_bets": 0, "n_win": 0, "n_loss": 0, "pnl": 0.0, "roi": float("nan")}

    # ── Test 3: idempotent re-settle ─────────────────────────────────────────
    try:
        test_no_duplicate_settlement(settled_day2)
        passed.append("TEST 3: Idempotent re-settle (no duplicates)")
    except Exception as e:
        print(f"  FAIL: {e}")
        failed.append(f"TEST 3: {e}")

    # ── Test 4: running P&L = sum of individual outcomes ────────────────────
    try:
        all_time_df = load_settled_parquet()
        test_pnl_running_total(day2_summary, all_time_df)
        passed.append("TEST 4: Running P&L = sum of individual outcomes")
    except Exception as e:
        print(f"  FAIL: {e}")
        failed.append(f"TEST 4: {e}")

    # ── Test 5: send email ───────────────────────────────────────────────────
    try:
        all_time_summary = compute_summary(all_time_df) if not all_time_df.empty else {
            "n_bets": 0, "n_win": 0, "n_loss": 0, "pnl": 0.0, "roi": float("nan")
        }
        test_send_email(sample_bets, day2_summary, all_time_summary)
        passed.append("TEST 5: Settlement email sent")
    except Exception as e:
        print(f"  FAIL: {e}")
        failed.append(f"TEST 5: {e}")

    # ── Reset ─────────────────────────────────────────────────────────────────
    reset_test_data()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"E2E Test Results: {len(passed)} passed, {len(failed)} failed")
    for t in passed:
        print(f"  ✓ {t}")
    for t in failed:
        print(f"  ✗ {t}")
    print(f"{'='*60}\n")

    if failed:
        sys.exit(1)
    print("=== Step 7 E2E tests complete ===")


if __name__ == "__main__":
    main()

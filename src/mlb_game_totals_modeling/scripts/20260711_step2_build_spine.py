"""
Step 2 — Build rolling feature spine for MLB Game Totals.

Spine grain: (game_date, home_team, away_team, bookmaker, line)
One row per game per book per line. Rolling features are game-level
(book-invariant) — same value for every book row of the same game.

Features built:
  Rolling runs scored (home/away): L1, L3, L5, L10, L20, season, career
  Rolling runs allowed (home/away): same windows
  Park factor (static per ballpark)
  Month, day_of_week (temporal)
  line_is_integer (1 if line is a whole number, 0 otherwise)
  Consensus line features: consensus_line, min_line, max_line
  Min/max raw_implied_prob_over/under
  Consensus odds bins (4 categorical features)
  Per-book: novig_prob_over, over/under prices

Output:
  S3:    s3://the-odds-api-mt/mlb/game_totals_model/spine/mlb_game_totals_spine.parquet
  Local: ~/Downloads/tmp/mlb_game_totals/game_totals_spine.parquet

Usage:
  python src/mlb_game_totals_modeling/scripts/20260711_step2_build_spine.py
"""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET  = "the-odds-api-mt"
LINES_KEY  = "mlb/game_totals_model/lines/mlb_game_totals_lines.parquet"
SCORES_KEY = "mlb/game_totals_model/scores/mlb_team_game_scores.parquet"
SPINE_KEY  = "mlb/game_totals_model/spine/mlb_game_totals_spine.parquet"
LOCAL_DIR  = Path.home() / "Downloads/tmp/mlb_game_totals"
LOCAL_OUT  = LOCAL_DIR / "game_totals_spine.parquet"

MAX_LINE = 13.0

# Rolling windows to build
ROLLING_WINDOWS = [1, 3, 5, 10, 20]

TEAM_NORMALIZE = {"Athletics": "Oakland Athletics"}

# Park factors — runs per 9 innings relative to league average (1.0 = neutral)
# Source: consensus multi-year park factors from Baseball Reference and FanGraphs
# Coors is a massive outlier (~1.40); most parks are 0.90–1.10
PARK_FACTORS: dict[str, float] = {
    "Arizona Diamondbacks": 1.05,
    "Atlanta Braves":       0.99,
    "Baltimore Orioles":    1.02,
    "Boston Red Sox":       1.06,   # Fenway — offensive park
    "Chicago Cubs":         0.97,
    "Chicago White Sox":    0.97,
    "Cincinnati Reds":      1.05,
    "Cleveland Guardians":  0.96,
    "Colorado Rockies":     1.39,   # Coors Field — extreme outlier
    "Detroit Tigers":       0.97,
    "Houston Astros":       0.96,
    "Kansas City Royals":   0.98,
    "Los Angeles Angels":   0.98,
    "Los Angeles Dodgers":  0.97,
    "Miami Marlins":        0.93,
    "Milwaukee Brewers":    0.96,
    "Minnesota Twins":      1.01,
    "New York Mets":        0.98,
    "New York Yankees":     1.00,
    "Oakland Athletics":    0.96,
    "Philadelphia Phillies": 1.03,
    "Pittsburgh Pirates":   0.99,
    "San Diego Padres":     0.94,
    "San Francisco Giants": 0.92,   # Oracle Park — pitcher-friendly
    "Seattle Mariners":     0.93,   # T-Mobile Park — pitcher-friendly
    "St. Louis Cardinals":  0.97,
    "Tampa Bay Rays":       0.97,
    "Texas Rangers":        1.01,
    "Toronto Blue Jays":    1.00,
    "Washington Nationals": 1.01,
}


def normalize_team(name: str) -> str:
    return TEAM_NORMALIZE.get(name, name)


def load_s3(key: str) -> pd.DataFrame:
    s3   = boto3.client("s3")
    body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    return pd.read_parquet(BytesIO(body))


def build_team_rolling_features(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Build rolling runs-scored and runs-allowed for home and away teams per game.
    Uses strictly prior-game data (no same-game lookahead).

    Returns a DataFrame at (game_pk, home_team, away_team, game_date) grain with
    features for both home and away teams.
    """
    # Build team-game log — one row per (team, game_date, is_home)
    # with runs_scored and runs_allowed
    home_rows = scores[["game_pk", "game_date", "season", "home_team", "away_team",
                         "home_runs", "away_runs"]].copy()
    home_rows = home_rows.rename(columns={
        "home_team": "team",
        "away_team": "opponent",
        "home_runs": "runs_scored",
        "away_runs": "runs_allowed",
    })
    home_rows["is_home"] = 1

    away_rows = scores[["game_pk", "game_date", "season", "away_team", "home_team",
                         "away_runs", "home_runs"]].copy()
    away_rows = away_rows.rename(columns={
        "away_team": "team",
        "home_team": "opponent",
        "away_runs": "runs_scored",
        "home_runs": "runs_allowed",
    })
    away_rows["is_home"] = 0

    team_log = pd.concat([home_rows, away_rows], ignore_index=True)
    team_log["game_date"] = pd.to_datetime(team_log["game_date"])
    team_log = team_log.sort_values(["team", "game_date", "game_pk"]).reset_index(drop=True)

    # For each team, compute rolling averages EXCLUDING the current game
    feature_rows = []
    for team, grp in team_log.groupby("team"):
        grp = grp.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
        n = len(grp)

        scored_arr  = grp["runs_scored"].values
        allowed_arr = grp["runs_allowed"].values

        for i in range(n):
            row = grp.iloc[i]
            prior = grp.iloc[:i]  # strictly prior games

            feat: dict = {
                "game_pk":   row["game_pk"],
                "game_date": row["game_date"],
                "team":      team,
                "is_home":   row["is_home"],
            }

            # Career rolling (all prior games)
            if len(prior) > 0:
                feat["rs_career"]  = prior["runs_scored"].mean()
                feat["ra_career"]  = prior["runs_allowed"].mean()
                feat["rs_season"]  = prior[prior["season"] == row["season"]]["runs_scored"].mean()
                feat["ra_season"]  = prior[prior["season"] == row["season"]]["runs_allowed"].mean()
            else:
                feat["rs_career"]  = np.nan
                feat["ra_career"]  = np.nan
                feat["rs_season"]  = np.nan
                feat["ra_season"]  = np.nan

            # Window rolling
            for w in ROLLING_WINDOWS:
                window = prior.tail(w)
                feat[f"rs_L{w}"] = window["runs_scored"].mean() if len(window) > 0 else np.nan
                feat[f"ra_L{w}"] = window["runs_allowed"].mean() if len(window) > 0 else np.nan

            feature_rows.append(feat)

    feat_df = pd.DataFrame(feature_rows)

    # Pivot to home and away perspectives
    home_feat = feat_df[feat_df["is_home"] == 1].copy()
    away_feat = feat_df[feat_df["is_home"] == 0].copy()

    # Rename columns with home_ / away_ prefix
    home_rename = {c: f"home_{c}" for c in feat_df.columns
                   if c not in ("game_pk", "game_date", "team", "is_home")}
    away_rename = {c: f"away_{c}" for c in feat_df.columns
                   if c not in ("game_pk", "game_date", "team", "is_home")}

    home_feat = home_feat.rename(columns=home_rename).drop(columns=["is_home"])
    away_feat = away_feat.rename(columns=away_rename).drop(columns=["is_home"])

    # Merge on game_pk
    merged = home_feat.merge(
        away_feat[["game_pk"] + [f"away_{c}" for c in feat_df.columns
                                  if c not in ("game_pk", "game_date", "team", "is_home")]],
        on="game_pk",
        how="inner",
    )
    merged = merged.rename(columns={"team": "home_team"})

    return merged


def add_market_consensus_features(lines: pd.DataFrame) -> pd.DataFrame:
    """
    Add game-level (book-invariant) market consensus features:
      consensus_line, min_line, max_line
      min/max raw_implied_prob_over/under
      consensus odds bins (4 features)
    """
    # Compute consensus at (game_date, home_team, away_team) level
    consensus = (
        lines.groupby(["game_date", "home_team", "away_team"])
        .agg(
            consensus_line              = ("line",          "mean"),
            min_line                    = ("line",          "min"),
            max_line                    = ("line",          "max"),
            min_raw_implied_prob_over   = ("raw_prob_over",  "min"),
            max_raw_implied_prob_over   = ("raw_prob_over",  "max"),
            min_raw_implied_prob_under  = ("raw_prob_under", "min"),
            max_raw_implied_prob_under  = ("raw_prob_under", "max"),
            # Consensus American odds (simple average across all books)
            _avg_over_price             = ("over_price",    "mean"),
            _avg_under_price            = ("under_price",   "mean"),
        )
        .reset_index()
    )

    # Consensus over odds bins (coarse: +/-/even)
    def _coarse_bin(x: float) -> str:
        if x > 0:
            return "plus"
        elif x < 0:
            return "minus"
        else:
            return "even"

    # Granular odds bin (8 buckets)
    def _granular_bin(x: float) -> str:
        if x <= -300:   return "-500_to_-300"
        elif x <= -200: return "-300_to_-200"
        elif x <= -110: return "-200_to_-110"
        elif x < 0:     return "-110_to_even"
        elif x == 0:    return "even"
        elif x <= 110:  return "even_to_+110"
        elif x <= 200:  return "+110_to_+200"
        elif x <= 300:  return "+200_to_+300"
        else:           return "+300_plus"

    consensus["consensus_over_odds_bin"]          = consensus["_avg_over_price"].apply(_coarse_bin)
    consensus["consensus_over_odds_bin_granular"] = consensus["_avg_over_price"].apply(_granular_bin)
    consensus["consensus_under_odds_bin"]         = consensus["_avg_under_price"].apply(_coarse_bin)
    consensus["consensus_under_odds_bin_granular"]= consensus["_avg_under_price"].apply(_granular_bin)
    consensus = consensus.drop(columns=["_avg_over_price", "_avg_under_price"])

    return lines.merge(consensus, on=["game_date", "home_team", "away_team"], how="left")


def main() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data from S3...")
    lines  = load_s3(LINES_KEY)
    scores = load_s3(SCORES_KEY)

    # Normalize team names
    for df in [lines, scores]:
        for col in ["home_team", "away_team"]:
            df[col] = df[col].map(normalize_team)

    print(f"  Lines:  {len(lines)} rows, {lines.event_id.nunique()} events")
    print(f"  Scores: {len(scores)} rows, {scores.game_pk.nunique()} games")

    # Filter implausible lines
    n_before = len(lines)
    lines = lines[lines["line"] <= MAX_LINE].copy()
    print(f"  Filtered lines > {MAX_LINE}: removed {n_before - len(lines)} rows")

    # Handle doubleheaders in scores — keep first game per matchup-date
    dh_keys = set(
        map(tuple,
            scores.groupby(["game_date", "home_team", "away_team"])
            .filter(lambda g: len(g) > 1)[["game_date", "home_team", "away_team"]]
            .drop_duplicates().values
        )
    )
    scores_single = scores[
        ~scores.apply(lambda r: (r.game_date, r.home_team, r.away_team) in dh_keys, axis=1)
    ]
    scores_dh_first = (
        scores[scores.apply(lambda r: (r.game_date, r.home_team, r.away_team) in dh_keys, axis=1)]
        .sort_values("game_pk").groupby(["game_date", "home_team", "away_team"]).first().reset_index()
    )
    scores_clean = pd.concat([scores_single, scores_dh_first], ignore_index=True)
    print(f"  Scores after DH handling: {len(scores_clean)}")

    # Add park factor from home team
    scores_clean["park_factor"] = scores_clean["home_team"].map(PARK_FACTORS)

    # ── Build rolling features ───────────────────────────────────────────────
    print("\nBuilding rolling features...")
    rolling = build_team_rolling_features(scores_clean)
    print(f"  Rolling feature rows: {len(rolling)}")

    # ── Build market consensus features ─────────────────────────────────────
    print("Adding market consensus features...")
    lines_with_consensus = add_market_consensus_features(lines)

    # Compute per-book novig probabilities
    lines_with_consensus["novig_prob_over"]  = (
        lines_with_consensus["raw_prob_over"] /
        (lines_with_consensus["raw_prob_over"] + lines_with_consensus["raw_prob_under"])
    )
    lines_with_consensus["novig_prob_under"] = 1.0 - lines_with_consensus["novig_prob_over"]

    # Deduplicate lines: ~10 cases where two event_ids share the same
    # (game_date, home_team, away_team, bookmaker, line) — doubleheaders/rescheduled games.
    n_before_dedup = len(lines_with_consensus)
    lines_with_consensus = lines_with_consensus.drop_duplicates(
        subset=["game_date", "home_team", "away_team", "bookmaker", "line"]
    )
    print(f"  Deduped lines at grain: {n_before_dedup - len(lines_with_consensus)} rows removed → {len(lines_with_consensus)}")

    # ── Join lines + scores ──────────────────────────────────────────────────
    print("Joining lines with actuals...")
    lines_with_actuals = lines_with_consensus.merge(
        scores_clean[["game_date", "home_team", "away_team", "game_pk",
                      "home_runs", "away_runs", "total_runs", "innings",
                      "park_factor"]],
        on=["game_date", "home_team", "away_team"],
        how="left",
    )

    # ── Join rolling features ────────────────────────────────────────────────
    print("Joining rolling features...")
    # Rolling features join on game_pk — drop home_team from rolling (already in lines)
    spine = lines_with_actuals.merge(
        rolling.drop(columns=["game_date", "home_team"]),
        on=["game_pk"],
        how="left",
    )

    # ── Temporal features ────────────────────────────────────────────────────
    spine["game_date_dt"] = pd.to_datetime(spine["game_date"])
    spine["month"]        = spine["game_date_dt"].dt.month
    spine["day_of_week"]  = spine["game_date_dt"].dt.dayofweek  # 0=Mon, 6=Sun
    spine["season"]       = spine["game_date_dt"].dt.year
    spine["is_weekend"]   = (spine["day_of_week"] >= 5).astype(int)
    spine["line_is_integer"] = (spine["line"] == spine["line"].astype(int)).astype(int)
    spine = spine.drop(columns=["game_date_dt"])

    # ── Outcome columns (for training) ──────────────────────────────────────
    settled = spine[spine["total_runs"].notna()].copy()
    settled["hit_over"]  = (settled["total_runs"] > settled["line"]).astype("Int8")
    settled["hit_under"] = (settled["total_runs"] < settled["line"]).astype("Int8")
    settled["hit_push"]  = (settled["total_runs"] == settled["line"]).astype("Int8")

    # ── Save ────────────────────────────────────────────────────────────────
    s3  = boto3.client("s3")
    buf = BytesIO()
    settled.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=SPINE_KEY, Body=buf.getvalue())
    print(f"\nSaved spine → s3://{S3_BUCKET}/{SPINE_KEY}  ({len(settled)} rows)")

    settled.to_parquet(LOCAL_OUT, index=False)
    print(f"Saved spine → {LOCAL_OUT}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\nSpine summary:")
    print(f"  Total rows:           {len(settled)}")
    print(f"  Unique games:         {settled['game_pk'].nunique()}")
    print(f"  Unique books:         {settled['bookmaker'].nunique()}")
    print(f"  Date range:           {settled['game_date'].min()} – {settled['game_date'].max()}")
    print(f"  Null rate (rs_career):{settled['home_rs_career'].isna().mean():.3f}")
    print(f"  Null rate (rs_L5):    {settled['home_rs_L5'].isna().mean():.3f}")

    # Spot check: NYY @ BOS
    spot = settled[
        (settled["home_team"] == "Boston Red Sox") &
        (settled["away_team"] == "New York Yankees")
    ]
    print(f"\nSpot check NYY @ BOS: {len(spot)} rows ({spot.game_date.nunique()} games)")
    if not spot.empty:
        spot_game = spot.sort_values("game_date").iloc[0]
        print(f"  First game: {spot_game['game_date']}")
        print(f"  home_rs_career: {spot_game['home_rs_career']:.2f} | home_ra_career: {spot_game['home_ra_career']:.2f}")
        print(f"  away_rs_career: {spot_game['away_rs_career']:.2f} | away_ra_career: {spot_game['away_ra_career']:.2f}")
        print(f"  park_factor: {spot_game['park_factor']}")
        print(f"  total_runs: {spot_game['total_runs']} | line: {spot_game['line']}")

    # Leakage check: rolling features at game G should not include game G
    print("\nLeakage check on spot-check game:")
    if not spot.empty:
        # Pick one game, check that rolling_L1 is NOT the same as total_runs for that game
        check_game = spot.sort_values("game_date").iloc[0]
        rs_l1 = check_game["home_rs_L1"]
        actual = check_game["total_runs"]
        if pd.notna(rs_l1):
            print(f"  home_rs_L1 = {rs_l1:.2f} (last home game runs scored — should NOT equal current game total {actual})")
        else:
            print("  home_rs_L1 = NaN (expected for first game in dataset)")


if __name__ == "__main__":
    main()

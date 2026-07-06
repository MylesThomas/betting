"""
NBA Player Points — Step 1 EDA
================================
Pulls player_points props from S3 historical CSVs (2023-24, 2024-25, 2025-26),
joins with game logs, and produces EDA outputs saved to ~/Downloads/tmp/points_eda/.

Outputs:
  points_props_raw.parquet   — all player_points rows across 3 seasons
  points_game_logs.parquet   — all game log rows (PTS column) across 3 seasons
  book_coverage.csv          — book × season coverage table
  line_dist.csv              — line value distribution
  hit_rates.csv              — over/under/push hit rates overall
"""
from __future__ import annotations

import sys
from io import BytesIO, StringIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR     = Path.home() / "Downloads/tmp/points_eda"
S3_PROPS    = "the-odds-api-mt"
S3_GL       = "nba-api-mt"
SEASONS     = ["2023-24", "2024-25", "2025-26"]
MARKET      = "player_points"
SPOT_CHECK  = "stephen curry"


def normalize_name(name: str) -> str:
    import unicodedata, re
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", name.strip().lower())
    return re.sub(r"\s+", " ", name).strip()


def load_props(s3) -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        prefix = f"nba/historical_player_props/{season}/"
        paginator = s3.get_paginator("list_objects_v2")
        keys = [
            obj["Key"]
            for page in paginator.paginate(Bucket=S3_PROPS, Prefix=prefix)
            for obj in page.get("Contents", [])
            if obj["Key"].endswith(".csv")
        ]
        print(f"  {season}: {len(keys)} files", flush=True)
        season_frames = []
        for key in keys:
            body = s3.get_object(Bucket=S3_PROPS, Key=key)["Body"].read().decode()
            df = pd.read_csv(StringIO(body))
            df = df[df["market"] == MARKET]
            if not df.empty:
                season_frames.append(df)
        if season_frames:
            frames.append(pd.concat(season_frames, ignore_index=True))
    raw = pd.concat(frames, ignore_index=True)
    # game_time is stored as UTC — convert to ET to get the correct local game date.
    # Evening games (7 PM ET+) cross midnight UTC, so naive UTC date = next calendar day.
    raw["game_date"] = (
        pd.to_datetime(raw["game_time"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.date.astype(str)
    )
    raw["player_key"] = raw["player"].apply(normalize_name)
    return raw


def load_game_logs(s3) -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        prefix = f"player_game_logs/{season}/"
        paginator = s3.get_paginator("list_objects_v2")
        keys = [
            obj["Key"]
            for page in paginator.paginate(Bucket=S3_GL, Prefix=prefix)
            for obj in page.get("Contents", [])
            if obj["Key"].endswith(".csv")
        ]
        print(f"  Game logs {season}: {len(keys)} files", flush=True)
        season_frames = []
        for key in keys:
            game_date = key.split("/")[-1].replace(".csv", "")
            body = s3.get_object(Bucket=S3_GL, Key=key)["Body"].read().decode()
            df = pd.read_csv(StringIO(body))
            df["game_date"] = game_date
            df["season"]    = season
            season_frames.append(df)
        if season_frames:
            frames.append(pd.concat(season_frames, ignore_index=True))
    logs = pd.concat(frames, ignore_index=True)
    logs["player_key"] = logs["PLAYER_NAME"].apply(normalize_name)
    logs["PTS"] = pd.to_numeric(logs["PTS"], errors="coerce")
    logs["MIN"] = pd.to_numeric(logs["MIN"], errors="coerce")
    # Drop the original GAME_DATE column (ISO format) — keep only the filename-derived game_date
    logs = logs.drop(columns=["GAME_DATE"], errors="ignore")
    return logs


def compute_hit_rates(props: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    # Join to get actual PTS
    joined = props.merge(
        logs[["player_key", "game_date", "PTS", "MIN"]],
        on=["player_key", "game_date"],
        how="left"
    )
    joined["actual"] = joined["PTS"]
    joined["line"]   = joined["prop_line"]
    joined["over"]   = (joined["actual"] > joined["line"]).astype(float)
    joined["under"]  = (joined["actual"] < joined["line"]).astype(float)
    joined["push"]   = (joined["actual"] == joined["line"]).astype(float)
    joined["dnp"]    = joined["actual"].isna().astype(float)

    # Use one row per player-game-book-side, deduplicate for hit rates
    deduped = joined.drop_duplicates(subset=["player_key", "game_date", "prop_line"])
    total = len(deduped)
    settled = deduped[deduped["actual"].notna()]
    n_settled = len(settled)
    return joined, deduped, {
        "total_rows": total,
        "settled_rows": n_settled,
        "dnp_rate": deduped["dnp"].mean(),
        "over_rate": (settled["over"].sum() / n_settled),
        "under_rate": (settled["under"].sum() / n_settled),
        "push_rate": (settled["push"].sum() / n_settled),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")

    print("Loading props...", flush=True)
    props = load_props(s3)
    print(f"  Total player_points rows: {len(props):,}")
    print(f"  Unique player-game-book combos: {props[['player_key','game_date','bookmaker']].drop_duplicates().shape[0]:,}")
    print(f"  Date range: {props['game_date'].min()} → {props['game_date'].max()}")
    print(f"  Books: {sorted(props['bookmaker'].unique())}")

    print("\nLoading game logs...", flush=True)
    logs = load_game_logs(s3)
    print(f"  Total game log rows: {len(logs):,}")
    print(f"  PTS null rate: {logs['PTS'].isna().mean():.1%}")

    # ── Save raw files ────────────────────────────────────────────────────────
    props.to_parquet(OUT_DIR / "points_props_raw.parquet", index=False)
    logs.to_parquet(OUT_DIR / "points_game_logs.parquet", index=False)
    print(f"\nSaved props + logs to {OUT_DIR}")

    # ── Book coverage ─────────────────────────────────────────────────────────
    book_cov = (
        props.groupby(["season", "bookmaker"])
        .agg(n_player_game=("player_key", lambda x: x.nunique()))
        .reset_index()
        .pivot(index="bookmaker", columns="season", values="n_player_game")
        .fillna(0).astype(int)
    )
    book_cov.to_csv(OUT_DIR / "book_coverage.csv")
    print("\nBook coverage (unique players per season):")
    print(book_cov.to_string())

    # ── Line distribution ──────────────────────────────────────────────────────
    line_dist = (
        props.drop_duplicates(subset=["player_key", "game_date", "bookmaker"])
        ["prop_line"].value_counts().sort_index().reset_index()
    )
    line_dist.columns = ["prop_line", "count"]
    line_dist["pct"] = line_dist["count"] / line_dist["count"].sum()
    line_dist.to_csv(OUT_DIR / "line_dist.csv", index=False)
    print(f"\nLine distribution (top 20):")
    print(line_dist.head(20).to_string(index=False))
    print(f"  Min line: {props['prop_line'].min()}, Max line: {props['prop_line'].max()}, Median: {props['prop_line'].median()}")

    # ── Hit rates ──────────────────────────────────────────────────────────────
    print("\nComputing hit rates...", flush=True)
    joined, deduped, rates = compute_hit_rates(props, logs)
    print(f"  Total unique player-game-line rows: {rates['total_rows']:,}")
    print(f"  Settled rows (actual PTS known): {rates['settled_rows']:,}")
    print(f"  DNP rate: {rates['dnp_rate']:.1%}")
    print(f"  Over rate: {rates['over_rate']:.1%}")
    print(f"  Under rate: {rates['under_rate']:.1%}")
    print(f"  Push rate: {rates['push_rate']:.1%}")
    print(f"  Sum: {rates['over_rate']+rates['under_rate']+rates['push_rate']:.1%}")

    hit_df = pd.DataFrame([rates])
    hit_df.to_csv(OUT_DIR / "hit_rates.csv", index=False)

    # ── Odds distribution ─────────────────────────────────────────────────────
    print(f"\nOver odds stats:")
    print(props["over_odds"].describe().to_string())
    print(f"\nUnder odds stats:")
    print(props["under_odds"].describe().to_string())

    # ── Spot-check: Stephen Curry ─────────────────────────────────────────────
    print(f"\n── Spot-check: {SPOT_CHECK} ──")
    curry_props = props[props["player_key"] == SPOT_CHECK].sort_values("game_date")
    print(f"  Prop rows: {len(curry_props)}")
    print(f"  Games with props: {curry_props['game_date'].nunique()}")
    print(f"  Line range: {curry_props['prop_line'].min()} – {curry_props['prop_line'].max()}")
    print(f"  Books posting: {sorted(curry_props['bookmaker'].unique())}")

    curry_logs = logs[logs["player_key"] == SPOT_CHECK].sort_values("game_date")
    print(f"  Game log rows: {len(curry_logs)}")
    print(f"  PTS: mean={curry_logs['PTS'].mean():.1f}, min={curry_logs['PTS'].min()}, max={curry_logs['PTS'].max()}")

    # Join curry props to actuals
    curry_joined = curry_props.drop_duplicates(subset=["game_date","bookmaker"]).merge(
        curry_logs[["game_date","PTS","MIN"]],
        on="game_date", how="left"
    ).sort_values("game_date")
    print(f"\n  Sample Curry rows (last 10 games):")
    print(curry_joined[["game_date","bookmaker","prop_line","over_odds","under_odds","PTS"]].tail(10).to_string(index=False))

    # Season-level coverage check
    print("\nCoverage by season:")
    for season in SEASONS:
        sp = props[props["season"] == season]
        sg = logs[logs["season"] == season]
        print(f"  {season}: {sp['game_date'].nunique()} game dates (props), {sg['game_date'].nunique()} game dates (logs), {sp['player_key'].nunique()} unique players (props)")

    print("\nDone.")


if __name__ == "__main__":
    main()

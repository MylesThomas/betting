"""
Debug rebounds pipeline artifacts for a specific player + season.

Prints three sections in pipeline order:
  1. Raw game logs (DuckDB S3 glob)
  2. Input universe (rebounds_input_universe.parquet)
  3. Feature universe (rebounds_feature_universe.parquet)
Then prints an NA root-cause summary cross-referencing all three.

Usage:
    python scripts/debug_rebounds_player.py --player "Jamal Cain" --season 2025-26
    python scripts/debug_rebounds_player.py --player "Jamal Cain" --season 2025-26 --as-of-date 2026-01-15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.io_utils import read_parquet_any
from src.nba_rebounds_modeling.duckdb_s3_creds import connect_duckdb_s3
from src.nba_rebounds_modeling.rebounds_feature_spec import (
    B_MIN_MAX_AUDIT_LIST_COLS,
    B_MIN_MAX_FEATS,
    GROUP_KEYS,
)
from src.player_team_history.name_normalization import (
    normalize_from_nba_api,
    normalize_from_odds_api,
)

INPUT_UNIVERSE_S3 = "s3://nba-betting-mt/rebounds/input/rebounds_input_universe.parquet"
FEATURE_UNIVERSE_S3 = "s3://nba-betting-mt/rebounds/features/rebounds_feature_universe.parquet"
GAME_LOGS_S3_GLOB = "s3://nba-api-mt/player_game_logs/{season}/*.csv"
HISTORICAL_PROPS_S3_GLOB = "s3://the-odds-api-mt/nba/historical_player_props/{season}/*.csv"
LIVE_PROPS_S3_PREFIX = "s3://the-odds-api-mt/nba/live_player_props/player_rebounds/csv/{season}"


def section_0_historical_props(player_norm: str, player_raw: str, season: str, as_of_date: str | None) -> None:
    """Check both historical props archive and live-fetch CSVs for this player."""
    import boto3
    import pandas as pd

    _divider("SECTION 0 — Historical props (was this player ever posted?)")

    last_name = player_raw.strip().split()[-1].lower()

    # --- A. Historical props archive (used by build_rebounds_full_universe.py) ---
    hist_glob = HISTORICAL_PROPS_S3_GLOB.format(season=season)
    print(f"[A] Historical props: {hist_glob}")
    con = connect_duckdb_s3()
    try:
        date_filter = f" AND CAST((game_time::TIMESTAMPTZ AT TIME ZONE 'America/New_York')::DATE AS VARCHAR) = '{as_of_date}'" if as_of_date else ""
        sql = f"""
            SELECT
                CAST((game_time::TIMESTAMPTZ AT TIME ZONE 'America/New_York')::DATE AS VARCHAR) AS date,
                player,
                bookmaker,
                CAST(prop_line AS DOUBLE) AS line,
                market
            FROM read_csv_auto('{hist_glob}', union_by_name=true, filename=true)
            WHERE market = 'player_rebounds'
              AND lower(player) LIKE '%{last_name}%'
              {date_filter}
            ORDER BY date, bookmaker
        """
        hist_df = con.execute(sql).fetchdf()
    except Exception as e:
        hist_df = None
        print(f"  ERROR reading historical props: {e}")
    finally:
        con.close()

    if hist_df is not None:
        if hist_df.empty:
            print(f"  No rows found for last_name='{last_name}' in historical props ({season})")
        else:
            hist_df["player_norm"] = hist_df["player"].apply(normalize_from_odds_api)
            matched = hist_df[hist_df["player_norm"] == player_norm]
            if matched.empty:
                print(f"  LIKE matched {len(hist_df)} rows but none normalized to '{player_norm}'")
                print(f"  Candidates: {hist_df['player_norm'].unique().tolist()}")
            else:
                summary = matched.groupby("date").agg(
                    n_books=("bookmaker", "nunique"),
                    lines=("line", lambda s: sorted(s.dropna().unique().tolist())),
                    bookmakers=("bookmaker", lambda s: sorted(s.unique().tolist())),
                ).reset_index()
                print(summary.to_string(index=False))
                print(f"\n  total dates with props: {len(summary)}")

    # --- B. Live-fetch props archive (from fetch_nba_player_rebounds_live.py runs) ---
    live_prefix = LIVE_PROPS_S3_PREFIX.format(season=season)
    print(f"\n[B] Live-fetch props: {live_prefix}/{{date}}/latest.csv")
    s3 = boto3.client("s3")
    bucket = "the-odds-api-mt"
    prefix_key = live_prefix.replace("s3://the-odds-api-mt/", "").rstrip("/") + "/"

    try:
        paginator = s3.get_paginator("list_objects_v2")
        date_dirs = set()
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix_key, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                date_part = cp["Prefix"].rstrip("/").split("/")[-1]
                date_dirs.add(date_part)
    except Exception as e:
        print(f"  ERROR listing live props dates: {e}")
        return

    if not date_dirs:
        print(f"  No live-fetch date directories found under {live_prefix}/")
        return

    live_rows = []
    for d in sorted(date_dirs):
        if as_of_date and d != as_of_date:
            continue
        key = f"{prefix_key}{d}/latest.csv"
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            csv_text = obj["Body"].read().decode("utf-8")
            chunk = pd.read_csv(__import__("io").StringIO(csv_text))
            if "player" not in chunk.columns:
                continue
            chunk_match = chunk[chunk["player"].str.lower().str.contains(last_name, na=False)].copy()
            if chunk_match.empty:
                continue
            chunk_match["player_norm"] = chunk_match["player"].apply(normalize_from_odds_api)
            chunk_match = chunk_match[chunk_match["player_norm"] == player_norm]
            if not chunk_match.empty:
                chunk_match["_date"] = d
                live_rows.append(chunk_match[["_date", "player", "bookmaker", "prop_line"]].copy())
        except s3.exceptions.NoSuchKey:
            pass
        except Exception as e:
            print(f"  WARNING: error reading {key}: {e}")

    if not live_rows:
        print(f"  Player '{player_norm}' not found in any live-fetch CSV for {season}")
    else:
        live_df = pd.concat(live_rows, ignore_index=True)
        live_summary = live_df.groupby("_date").agg(
            n_books=("bookmaker", "nunique"),
            lines=("prop_line", lambda s: sorted(pd.to_numeric(s, errors="coerce").dropna().unique().tolist())),
        ).reset_index().rename(columns={"_date": "date"})
        print(live_summary.to_string(index=False))
        print(f"\n  total live-fetch dates with props: {len(live_summary)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug rebounds pipeline for a specific player.")
    p.add_argument("--player", required=True, help='Raw player name, e.g. "Jamal Cain"')
    p.add_argument("--season", required=True, help="Season string, e.g. 2025-26")
    p.add_argument("--as-of-date", default=None, help="Optional YYYY-MM-DD to filter to a single game date")
    return p.parse_args()


def _divider(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def section_1_game_logs(player_raw: str, season: str, as_of_date: str | None) -> "pd.DataFrame":
    import pandas as pd

    _divider("SECTION 1 — Raw game logs")
    last_name = player_raw.strip().split()[-1].lower()
    glob_uri = GAME_LOGS_S3_GLOB.format(season=season)

    con = connect_duckdb_s3()
    try:
        date_filter = ""
        if as_of_date:
            date_filter = f" AND GAME_DATE = '{as_of_date}'"
        sql = f"""
            SELECT
                GAME_DATE   AS date,
                PLAYER_NAME,
                MIN,
                REB,
                FGA,
                FG3A,
                TEAM_NAME AS team
            FROM read_csv_auto('{glob_uri}', filename=true)
            WHERE lower(PLAYER_NAME) LIKE '%{last_name}%'
            {date_filter}
            ORDER BY GAME_DATE
        """
        raw = con.execute(sql).fetchdf()
    finally:
        con.close()

    if raw.empty:
        print(f"  No rows found for last_name='{last_name}' in {glob_uri}")
        return raw

    # Normalize and filter to exact match
    raw["player_normalized"] = raw["PLAYER_NAME"].apply(normalize_from_nba_api)
    target_norm = normalize_from_nba_api(player_raw)
    matched = raw[raw["player_normalized"] == target_norm].copy()

    if matched.empty:
        print(f"  LIKE match found rows but none normalized to '{target_norm}'")
        print(f"  Candidates: {raw['player_normalized'].unique().tolist()}")
        return matched

    display_cols = ["date", "PLAYER_NAME", "team", "MIN", "REB", "FGA", "FG3A"]
    print(matched[display_cols].to_string(index=False))

    n_games = len(matched)
    date_range = f"{matched['date'].min()} → {matched['date'].max()}"
    n_null_fg3a = matched["FG3A"].isna().sum()
    n_zero_fg3a = (matched["FG3A"] == 0).sum()
    print(f"\n  n_games={n_games}  date_range={date_range}  n_null_fg3a={n_null_fg3a}  n_zero_fg3a={n_zero_fg3a}")
    return matched


def section_2_input_universe(player_norm: str, season: str, as_of_date: str | None) -> "pd.DataFrame":
    import pandas as pd

    _divider("SECTION 2 — Input universe")

    try:
        df = read_parquet_any(INPUT_UNIVERSE_S3)
    except Exception as e:
        print(f"  ERROR reading input universe: {e}")
        return pd.DataFrame()

    mask = (df["player_normalized"] == player_norm) & (df["season"] == season)
    if as_of_date:
        mask &= pd.to_datetime(df["date"]).dt.normalize() == pd.Timestamp(as_of_date).normalize()
    sub = df[mask].copy()

    if sub.empty:
        print(f"  No rows for player_normalized='{player_norm}', season='{season}'")
        print(f"  Unique players in season: {df[df['season'] == season]['player_normalized'].nunique()}")
        return sub

    show_cols = [c for c in ["date", "game_id", "REB", "spread_signed", "team_normalized",
                              "home_team_norm", "away_team_norm",
                              "input_spread_by_side", "input_fg3a_tail_20"] if c in sub.columns]
    print(sub[show_cols].to_string(index=False))

    n_null_spread = sub["spread_signed"].isna().sum() if "spread_signed" in sub.columns else "N/A"
    print(f"\n  n_rows={len(sub)}  n_null_spread_signed={n_null_spread}")
    return sub


def section_3_feature_universe(player_norm: str, season: str, as_of_date: str | None) -> "pd.DataFrame":
    import pandas as pd

    _divider("SECTION 3 — Feature universe")

    try:
        df = read_parquet_any(FEATURE_UNIVERSE_S3)
    except Exception as e:
        print(f"  ERROR reading feature universe: {e}")
        return pd.DataFrame()

    mask = (df["player_normalized"] == player_norm) & (df["season"] == season)
    if as_of_date:
        mask &= pd.to_datetime(df["date"]).dt.normalize() == pd.Timestamp(as_of_date).normalize()
    sub = df[mask].copy()

    if sub.empty:
        print(f"  No rows for player_normalized='{player_norm}', season='{season}'")
        return sub

    show_cols = ["date", "game_id"] + B_MIN_MAX_FEATS + B_MIN_MAX_AUDIT_LIST_COLS
    show_cols = [c for c in show_cols if c in sub.columns]
    print(sub[show_cols].to_string(index=False))

    feat_cols = [c for c in B_MIN_MAX_FEATS if c in sub.columns]
    null_counts = {c: int(sub[c].isna().sum()) for c in feat_cols}
    print(f"\n  n_rows={len(sub)}")
    print("  Feature null counts:", null_counts)
    return sub


def print_na_summary(logs_df, input_df, feat_df) -> None:
    import pandas as pd

    _divider("NA ROOT-CAUSE SUMMARY")

    feat_cols = B_MIN_MAX_FEATS
    if feat_df.empty:
        print("  Feature universe is empty — cannot produce NA summary.")
        return

    if input_df.empty:
        print("  WARNING: player has 0 rows in input universe — all NA features likely trace to missing input universe rows.")
        print("  Check: (1) player name normalization mismatch, (2) player not in props data, (3) input universe not yet rebuilt.\n")

    any_na = False
    for col in feat_cols:
        if col not in feat_df.columns:
            print(f"  {col}: MISSING from feature universe")
            any_na = True
            continue
        n_na = int(feat_df[col].isna().sum())
        if n_na == 0:
            continue
        any_na = True
        pct = n_na / len(feat_df) * 100
        print(f"\n  {col}: {n_na}/{len(feat_df)} NA ({pct:.0f}%)")

        if col == "spread_signed":
            if not input_df.empty and "spread_signed" in input_df.columns:
                n_null_in_input = int(input_df["spread_signed"].isna().sum())
                print(f"    → input universe spread_signed nulls: {n_null_in_input}")
                print(f"    → check: spread join failed for those game dates")

        elif col == "roll_fg3a_mean_20":
            if not logs_df.empty and "FG3A" in logs_df.columns:
                n_null_log = int(logs_df["FG3A"].isna().sum())
                n_zero_log = int((logs_df["FG3A"] == 0).sum())
                print(f"    → game logs FG3A: {n_null_log} null, {n_zero_log} zero of {len(logs_df)} rows")
                if not input_df.empty and "input_fg3a_tail_20" in input_df.columns:
                    n_null_tail = int(input_df["input_fg3a_tail_20"].isna().sum())
                    print(f"    → input universe input_fg3a_tail_20 nulls: {n_null_tail}")
            print(f"    → roll needs 1+ non-null FG3A in trailing 20 games; all-zero/null → NA")

        elif col in ("roll_reb_mean_60", "roll_reb_std_5"):
            if not logs_df.empty and "REB" in logs_df.columns:
                n_null_reb = int(logs_df["REB"].isna().sum())
                print(f"    → game logs REB nulls: {n_null_reb} of {len(logs_df)} rows")
            print(f"    → roll needs sufficient history; early-season rows may be NA")

        elif col in ("min_line", "max_line"):
            if not input_df.empty and "input_reb_prop_lines" in input_df.columns:
                n_null_lines = int(input_df["input_reb_prop_lines"].isna().sum())
                print(f"    → input universe input_reb_prop_lines nulls: {n_null_lines}")
            print(f"    → min/max lines come from props join; NA means no prop line for that game")

    if not any_na:
        print("  No NAs found in feature columns for this player/season.")


def main() -> None:
    args = parse_args()

    player_norm_odds = normalize_from_odds_api(args.player)
    player_norm_nba = normalize_from_nba_api(args.player)

    print(f"player_raw='{args.player}'")
    print(f"player_norm (odds_api): '{player_norm_odds}'")
    print(f"player_norm (nba_api):  '{player_norm_nba}'")
    print(f"season='{args.season}'  as_of_date={args.as_of_date or 'all'}")

    # Section 0: historical props — was this player ever posted?
    section_0_historical_props(player_norm_odds, args.player, args.season, args.as_of_date)

    # Section 1 uses nba_api normalization (game log PLAYER_NAME)
    logs_df = section_1_game_logs(args.player, args.season, args.as_of_date)

    # Sections 2 + 3 filter on player_normalized which was built with normalize_from_nba_api
    # in build_rebounds_input_universe.py; use the same normalizer for consistent matching
    player_norm = player_norm_nba or args.player
    input_df = section_2_input_universe(player_norm, args.season, args.as_of_date)
    feat_df = section_3_feature_universe(player_norm, args.season, args.as_of_date)

    print_na_summary(logs_df, input_df, feat_df)


if __name__ == "__main__":
    main()

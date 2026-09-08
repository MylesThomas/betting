"""
Umpire K-tendency feature for MLB pitcher strikeouts model.

Steps:
  1. Load pitcher gamelogs from S3 — gets all (game_pk, game_date, strikeouts) rows.
  2. For each unique game_date, fetch MLB Stats API schedule with officials hydration
     to get home plate ump per game_pk. Falls back to per-game boxscore if needed.
  3. Compute rolling ump K tendency (no lookahead):
       ump_k_avg_career = mean Ks per pitcher-start in ump's prior games
       ump_k_delta      = ump_k_avg_career − rolling league avg Ks per start
  4. Save game-level feature parquet (one row per game_pk).

Output schema:
  game_pk, game_date, ump_id, ump_name,
  ump_k_avg_career, ump_k_avg_c10, ump_k_avg_c20, ump_k_avg_season,
  ump_k_delta, ump_k_delta_c10, ump_k_delta_c20, ump_k_delta_season

Outputs:
  Local: ~/Downloads/tmp/mlb_strikeouts/ump_features.parquet
  S3:    s3://the-odds-api-mt/mlb/strikeouts_model/ump_features.parquet

Usage:
  uv run src/mlb_strikeouts_modeling/scripts/20260819_umpire_features.py
  uv run src/mlb_strikeouts_modeling/scripts/20260819_umpire_features.py --no-upload
"""
from __future__ import annotations

import argparse
import sys
import time
from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

MLB_API_BASE   = "https://statsapi.mlb.com/api/v1"
SLEEP_S        = 0.1

S3_BUCKET      = "the-odds-api-mt"
GAMELOG_PREFIX = "mlb/strikeouts_model/pitcher_gamelogs"
OUT_KEY        = "mlb/strikeouts_model/ump_features.parquet"
LOCAL_OUT      = Path.home() / "Downloads/tmp/mlb_strikeouts/ump_features.parquet"

SEASONS = [2024, 2025, 2026]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_gamelogs() -> pd.DataFrame:
    s3 = boto3.client("s3")
    frames = []
    for season in SEASONS:
        key = f"{GAMELOG_PREFIX}/{season}.parquet"
        body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
        frames.append(pd.read_parquet(BytesIO(body)))
    return pd.concat(frames, ignore_index=True)


# ── Umpire assignment fetch ───────────────────────────────────────────────────

def fetch_with_retry(url: str, params: dict, retries: int = 4, backoff: float = 2.0) -> requests.Response:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=45)
            r.raise_for_status()
            return r
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt == retries - 1:
                raise
            wait = backoff * (2 ** attempt)
            print(f"    [retry {attempt+1}/{retries}] {e.__class__.__name__} — sleeping {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def fetch_ump_for_date(date_str: str) -> dict[int, tuple[int, str]]:
    """
    Returns {game_pk: (ump_id, ump_name)} for home plate umps on date_str.
    Uses schedule endpoint with officials hydration.
    Requires season param alongside date — MLB API returns 0 games without it.
    """
    season = int(date_str[:4])
    r = fetch_with_retry(
        f"{MLB_API_BASE}/schedule",
        params={
            "sportId":  1,
            "date":     date_str,
            "season":   season,
            "hydrate":  "officials",
            "gameType": "R",
        },
    )

    result: dict[int, tuple[int, str]] = {}
    for date_block in r.json().get("dates", []):
        for game in date_block.get("games", []):
            game_pk  = game.get("gamePk")
            hp_ump   = next(
                (o for o in game.get("officials", [])
                 if o.get("officialType") == "Home Plate"),
                None,
            )
            if hp_ump and game_pk:
                result[game_pk] = (
                    hp_ump["official"]["id"],
                    hp_ump["official"]["fullName"],
                )
    return result


def fetch_ump_from_boxscore(game_pk: int) -> tuple[int, str] | None:
    """Fallback: get home plate ump from per-game boxscore endpoint."""
    try:
        r = fetch_with_retry(f"{MLB_API_BASE}/game/{game_pk}/boxscore", params={})
    except Exception:
        return None
    if r.status_code != 200:
        return None
    hp_ump = next(
        (o for o in r.json().get("officials", [])
         if o.get("officialType") == "Home Plate"),
        None,
    )
    if not hp_ump:
        return None
    return hp_ump["official"]["id"], hp_ump["official"]["fullName"]


def build_ump_assignments(df_logs: pd.DataFrame) -> pd.DataFrame:
    """
    For every (game_date, game_pk) in the gamelogs, fetch the home plate ump.
    One API call per unique date (batch); falls back to boxscore per missing game.
    """
    unique_games = (
        df_logs[["game_date", "game_pk"]]
        .drop_duplicates()
        .sort_values("game_date")
        .reset_index(drop=True)
    )
    print(f"Unique games: {len(unique_games):,}  |  Unique dates: {unique_games['game_date'].nunique()}")

    rows: list[dict] = []
    dates = sorted(unique_games["game_date"].unique())

    for i, date_str in enumerate(dates, 1):
        ump_map = fetch_ump_for_date(date_str)
        date_games = unique_games[unique_games["game_date"] == date_str]

        for _, g in date_games.iterrows():
            gpk = int(g["game_pk"])
            if gpk in ump_map:
                uid, uname = ump_map[gpk]
            else:
                result = fetch_ump_from_boxscore(gpk)
                if result:
                    uid, uname = result
                else:
                    uid, uname = None, None
                time.sleep(SLEEP_S)

            rows.append({
                "game_pk":   gpk,
                "game_date": date_str,
                "ump_id":    uid,
                "ump_name":  uname,
            })

        time.sleep(SLEEP_S)
        if i % 50 == 0 or i == len(dates):
            found = sum(1 for r in rows if r["ump_name"] is not None)
            print(f"  [{i}/{len(dates)} dates]  {len(rows)} games processed  |  {found} with ump ({found/len(rows):.1%})")

    return pd.DataFrame(rows)


# ── Rolling ump tendency ──────────────────────────────────────────────────────

def compute_ump_tendency(
    df_assign: pd.DataFrame,
    df_logs: pd.DataFrame,
) -> pd.DataFrame:
    """
    ump_k_avg_career: rolling mean Ks per pitcher-start in ump's prior games.
    ump_k_delta:      ump_k_avg_career − league rolling avg Ks per start.

    Both are no-lookahead: shift(1) before expanding mean, sorted chronologically.
    Output is game-level (one row per game_pk).
    """
    merged = df_logs.merge(
        df_assign[["game_pk", "ump_id", "ump_name"]],
        on="game_pk",
        how="left",
    )
    merged["game_date"] = pd.to_datetime(merged["game_date"])

    # Aggregate to game level: avg Ks per pitcher-start in this game
    # (captures both home + away starters if both are in gamelogs)
    game_agg = (
        merged.groupby(["game_pk", "game_date", "ump_id", "ump_name"], dropna=False)
        ["strikeouts"]
        .mean()
        .reset_index()
        .rename(columns={"strikeouts": "ks_this_game"})
        .sort_values(["ump_id", "game_date"])
        .reset_index(drop=True)
    )

    # Rolling ump tendency — career expanding mean (no lookahead)
    game_agg["ump_k_avg_career"] = (
        game_agg.groupby("ump_id")["ks_this_game"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # Windowed rolling — last 10 and 20 games the ump worked
    for w in [10, 20]:
        game_agg[f"ump_k_avg_c{w}"] = (
            game_agg.groupby("ump_id")["ks_this_game"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=3).mean())
        )

    # Season-scoped rolling (resets each season)
    game_agg["season"] = game_agg["game_date"].dt.year
    game_agg["ump_k_avg_season"] = (
        game_agg.groupby(["ump_id", "season"])["ks_this_game"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # Rolling league avg Ks per start (game-level, no lookahead)
    league = (
        merged.groupby(["game_pk", "game_date"])["strikeouts"]
        .mean()
        .reset_index()
        .sort_values("game_date")
        .reset_index(drop=True)
    )
    league["league_k_avg"] = league["strikeouts"].shift(1).expanding().mean()

    game_agg = game_agg.merge(league[["game_pk", "league_k_avg"]], on="game_pk", how="left")

    # Delta variants: ump avg minus rolling league avg
    for suffix in ["career", "c10", "c20", "season"]:
        avg_col   = f"ump_k_avg_{suffix}"
        delta_col = f"ump_k_delta_{suffix}" if suffix != "career" else "ump_k_delta"
        game_agg[delta_col] = game_agg[avg_col] - game_agg["league_k_avg"]

    out_cols = [
        "game_pk", "game_date", "ump_id", "ump_name",
        "ump_k_avg_career", "ump_k_avg_c10", "ump_k_avg_c20", "ump_k_avg_season",
        "ump_k_delta", "ump_k_delta_c10", "ump_k_delta_c20", "ump_k_delta_season",
    ]
    return game_agg[out_cols]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-upload", action="store_true", help="Skip S3 upload")
    args = parser.parse_args()

    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)

    print("Loading gamelogs from S3...")
    df_logs = load_gamelogs()
    print(f"  {len(df_logs):,} pitcher-starts  |  {df_logs['player_id'].nunique()} pitchers  |  "
          f"{df_logs['game_pk'].nunique()} unique games")

    print("\nFetching umpire assignments...")
    df_assign = build_ump_assignments(df_logs)
    found_pct = df_assign["ump_name"].notna().mean()
    print(f"\nAssignments: {len(df_assign):,} games  |  ump found: {found_pct:.1%}")
    print(df_assign["ump_name"].value_counts().head(10).to_string())

    print("\nComputing rolling ump tendency...")
    df_features = compute_ump_tendency(df_assign, df_logs)
    print(f"  {len(df_features):,} game-level rows")
    for col in ["ump_k_avg_career", "ump_k_avg_c10", "ump_k_avg_c20", "ump_k_avg_season",
                "ump_k_delta", "ump_k_delta_c10", "ump_k_delta_c20", "ump_k_delta_season"]:
        s = df_features[col]
        print(f"  {col:<22}: mean={s.mean():.3f}  std={s.std():.3f}  null={s.isna().sum()}")

    # Spot-check: top ump names by K delta
    ump_summary = (
        df_features.dropna(subset=["ump_k_delta"])
        .groupby("ump_name")
        .agg(games=("game_pk", "count"), avg_delta=("ump_k_delta", "mean"))
        .sort_values("avg_delta", ascending=False)
        .reset_index()
    )
    print(f"\nTop 10 pitcher-friendly umps (most Ks above league avg):")
    print(ump_summary[ump_summary["games"] >= 10].head(10).to_string(index=False))
    print(f"\nTop 10 batter-friendly umps (fewest Ks above league avg):")
    print(ump_summary[ump_summary["games"] >= 10].tail(10).to_string(index=False))

    df_features.to_parquet(LOCAL_OUT, index=False)
    print(f"\nSaved locally → {LOCAL_OUT}")

    if not args.no_upload:
        s3 = boto3.client("s3")
        buf = BytesIO()
        df_features.to_parquet(buf, index=False)
        buf.seek(0)
        s3.put_object(Bucket=S3_BUCKET, Key=OUT_KEY, Body=buf.getvalue())
        print(f"Uploaded → s3://{S3_BUCKET}/{OUT_KEY}")
    else:
        print("[--no-upload] Skipped S3 upload")


if __name__ == "__main__":
    main()

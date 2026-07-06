"""
Build v6 spread-context universe for player-level regression diagnostics.

Context:
- This script extends the v5 player_threes research universe with pregame spread
  context from historical game lines.
- Join logic is strict and deterministic:
  1) player-game rows from NBA box scores + player_team_history by date range,
  2) consensus player_threes context by player/date,
  3) team spread from team/date game lines using player team perspective.
- Outputs are used by downstream v6 regression sweeps to quantify spread signal
  across targets such as MIN, FG3M, FG3A, FG3A_per_min, and FG3_PCT.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd


def ensure_repo_root_on_syspath() -> Path:
    """Find repo root from cwd and add it to sys.path."""
    current = Path.cwd().resolve()
    while True:
        gitignore = current / ".gitignore"
        src_dir = current / "src"
        if gitignore.exists() and src_dir.exists():
            repo_root = current
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            return repo_root
        if current.parent == current:
            raise FileNotFoundError("Could not locate repo root with .gitignore and src/")
        current = current.parent


REPO_ROOT = ensure_repo_root_on_syspath()

from src.player_team_history.team_normalization import normalize_team_name_from_odds_api, TEAM_ABBR_TO_NAME
from v5_workflow_lib import build_market_eligibility
from v5_workflow_lib import connect_duckdb_s3
from v5_workflow_lib import load_player_logs
from v5_workflow_lib import load_player_props
from v5_workflow_lib import resolve_output_path
from v5_workflow_lib import season_predicate
from v5_workflow_lib import set_seed


SPREAD_BIN_EDGES = [float("-inf"), -12.0, -8.0, -4.0, -1.0, 1.0, 4.0, 8.0, 12.0, float("inf")]
SPREAD_BIN_LABELS = [
    "(-inf,-12]",
    "(-12,-8]",
    "(-8,-4]",
    "(-4,-1]",
    "(-1,1]",
    "(1,4]",
    "(4,8]",
    "(8,12]",
    "(12,inf)",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI args for v6 spread universe build."""
    parser = argparse.ArgumentParser(description="Build v6 spread-context universe.")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--season", type=str, default="*")
    parser.add_argument("--cache-dir", type=str, default="~/Downloads/tmp")
    parser.add_argument("--use-cache", type=str, default="true")
    parser.add_argument("--force-refresh-cache", type=str, default="false")
    parser.add_argument("--output-universe", type=str, default="")
    parser.add_argument("--output-qc", type=str, default="")
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    """Parse common string boolean variants."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value}")


def load_player_day_base_with_team(season: str, logs_df: pd.DataFrame) -> pd.DataFrame:
    """Attach team history by player/date range and return canonical player-game base."""
    con = connect_duckdb_s3()
    con.register("logs_df", logs_df)
    query = f"""
    WITH history AS (
      SELECT
        player_normalized,
        team AS team_abbr,
        CAST(valid_from AS DATE) AS valid_from,
        CAST(valid_to AS DATE) AS valid_to
      FROM read_parquet('s3://nba-betting-mt/data/02_cache/player_team_history.parquet')
    )
    SELECT
      l.season,
      l.player_normalized,
      l.date,
      l.GAME_ID AS game_id,
      h.team_abbr,
      l.MIN,
      l.FG3M,
      l.FG3A,
      l.FG3_PCT,
      l.PTS,
      l.REB,
      l.AST,
      l.FGA,
      l.FGM,
      l.TOV,
      l.FTA,
      l.FTM
    FROM logs_df l
    INNER JOIN history h
      ON l.player_normalized = h.player_normalized
      AND CAST(l.GAME_DATE AS DATE) >= h.valid_from
      AND (h.valid_to IS NULL OR CAST(l.GAME_DATE AS DATE) <= h.valid_to)
    WHERE {season_predicate('l', season)}
    """
    base = con.execute(query).fetchdf()
    con.close()
    base["team_normalized"] = base["team_abbr"].map(TEAM_ABBR_TO_NAME)
    missing_team_name = base["team_normalized"].isna()
    if missing_team_name.any():
        missing_abbr = sorted(base.loc[missing_team_name, "team_abbr"].unique().tolist())
        raise ValueError(f"Missing TEAM_ABBR_TO_NAME mapping for: {missing_abbr}")
    base = base.drop(columns=["team_abbr"])
    return base


def load_team_spreads(season: str) -> pd.DataFrame:
    """Load and aggregate historical spread/moneyline into team-perspective rows."""
    con = connect_duckdb_s3()
    query = f"""
    WITH raw AS (
      SELECT
        home_team,
        away_team,
        market,
        home_line,
        away_line,
        home_odds,
        away_odds,
        regexp_extract(filename, '/historical_game_lines/([^/]+)/', 1) AS season,
        regexp_extract(filename, 'nba_game_lines_(\\d{{4}}-\\d{{2}}-\\d{{2}})\\.csv', 1) AS date
      FROM read_csv_auto(
        's3://the-odds-api-mt/nba/historical_game_lines/*/nba_game_lines_*.csv',
        union_by_name=true,
        filename=true
      )
    ),
    filtered AS (
      SELECT *
      FROM raw r
      WHERE {season_predicate('r', season)}
        AND market IN ('spread', 'moneyline')
    ),
    spread AS (
      SELECT
        season,
        date,
        home_team,
        away_team,
        median(home_line) AS home_spread,
        median(away_line) AS away_spread
      FROM filtered
      WHERE market = 'spread'
      GROUP BY season, date, home_team, away_team
    ),
    moneyline AS (
      SELECT
        season,
        date,
        home_team,
        away_team,
        median(home_odds) AS home_moneyline,
        median(away_odds) AS away_moneyline
      FROM filtered
      WHERE market = 'moneyline'
      GROUP BY season, date, home_team, away_team
    ),
    game_rows AS (
      SELECT
        s.season,
        s.date,
        s.home_team,
        s.away_team,
        s.home_spread,
        s.away_spread,
        m.home_moneyline,
        m.away_moneyline
      FROM spread s
      LEFT JOIN moneyline m
        ON s.season = m.season
       AND s.date = m.date
       AND s.home_team = m.home_team
       AND s.away_team = m.away_team
    )
    SELECT
      season,
      date,
      home_team AS team_normalized,
      home_spread AS spread_signed,
      abs(home_spread) AS spread_abs,
      home_moneyline AS moneyline
    FROM game_rows
    UNION ALL
    SELECT
      season,
      date,
      away_team AS team_normalized,
      away_spread AS spread_signed,
      abs(away_spread) AS spread_abs,
      away_moneyline AS moneyline
    FROM game_rows
    """
    spread_df = con.execute(query).fetchdf()
    con.close()
    spread_df["team_normalized"] = spread_df["team_normalized"].apply(normalize_team_name_from_odds_api)
    spread_df["spread_bin"] = pd.cut(
        spread_df["spread_signed"].astype(float),
        bins=SPREAD_BIN_EDGES,
        labels=SPREAD_BIN_LABELS,
        right=True,
    )
    return spread_df


def add_consensus_context(base_df: pd.DataFrame, eligible_df: pd.DataFrame) -> pd.DataFrame:
    """Join consensus player_threes market context on player/date."""
    joined = base_df.merge(
        eligible_df[
            ["season", "player_normalized", "date", "market_consensus_line", "median_p_over_novig"]
        ],
        on=["season", "player_normalized", "date"],
        how="inner",
    )
    return joined


def add_spread_context(universe_df: pd.DataFrame, spread_df: pd.DataFrame) -> pd.DataFrame:
    """Join team spread context on season/date/team and add FG3A_per_min."""
    joined = universe_df.merge(
        spread_df[["season", "date", "team_normalized", "spread_signed", "spread_abs", "spread_bin"]],
        on=["season", "date", "team_normalized"],
        how="left",
    )
    joined["FG3A_per_min"] = joined["FG3A"] / joined["MIN"]
    joined["FG3A_per_min"] = joined["FG3A_per_min"].replace([float("inf"), float("-inf")], pd.NA)
    return joined


def build_spread_qc(universe_df: pd.DataFrame) -> pd.DataFrame:
    """Build spread-specific QC summary rows."""
    rows: list[dict[str, Any]] = []
    matched_rate = 1.0 - float(universe_df["spread_signed"].isna().mean())
    rows.append(
        {
            "check_type": "spread_match_rate",
            "metric_name": "pct_rows_matched_to_spread",
            "metric_value": matched_rate,
        }
    )

    dup_count = int(
        universe_df.duplicated(subset=["player_normalized", "date", "game_id"], keep=False).sum()
    )
    rows.append(
        {
            "check_type": "duplicate_keys",
            "metric_name": "player_normalized_date_game_id",
            "metric_value": float(dup_count),
        }
    )

    required_cols = [
        "MIN",
        "FG3M",
        "FG3A",
        "FG3A_per_min",
        "FG3_PCT",
        "market_consensus_line",
        "spread_signed",
        "spread_bin",
    ]
    for col in required_cols:
        rows.append(
            {
                "check_type": "null_rate",
                "metric_name": col,
                "metric_value": float(universe_df[col].isna().mean()),
            }
        )

    bin_counts = (
        universe_df.groupby("spread_bin", dropna=False, as_index=False)
        .agg(n_rows=("game_id", "count"))
        .sort_values("spread_bin")
    )
    for _, row in bin_counts.iterrows():
        rows.append(
            {
                "check_type": "spread_bin_count",
                "metric_name": str(row["spread_bin"]),
                "metric_value": float(row["n_rows"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run v6 spread universe build and save universe + QC artifacts."""
    args = parse_args()
    set_seed(int(args.seed))
    use_cache = parse_bool(args.use_cache)
    force_refresh_cache = parse_bool(args.force_refresh_cache)

    logs_df = load_player_logs(
        season=args.season,
        cache_dir=args.cache_dir,
        use_cache=use_cache,
        force_refresh_cache=force_refresh_cache,
    )
    props_df = load_player_props(
        season=args.season,
        cache_dir=args.cache_dir,
        use_cache=use_cache,
        force_refresh_cache=force_refresh_cache,
    )
    eligible_df = build_market_eligibility(props_df=props_df)
    base_df = load_player_day_base_with_team(season=args.season, logs_df=logs_df)
    universe = add_consensus_context(base_df=base_df, eligible_df=eligible_df)
    spread_df = load_team_spreads(season=args.season)
    universe = add_spread_context(universe_df=universe, spread_df=spread_df)
    universe = universe.sort_values(["season", "date", "player_normalized", "game_id"]).reset_index(
        drop=True
    )
    qc_df = build_spread_qc(universe_df=universe)

    universe_path = resolve_output_path(args.output_universe, "v6_spread_universe.parquet")
    qc_path = resolve_output_path(args.output_qc, "v6_spread_universe_qc.csv")
    universe.to_parquet(Path(universe_path).expanduser(), index=False)
    qc_df.to_csv(Path(qc_path).expanduser(), index=False)
    print(
        "phase=v6_build_spread_universe",
        f"seed={args.seed}",
        f"season={args.season}",
        f"rows={len(universe)}",
        f"universe={universe_path}",
        f"qc={qc_path}",
        sep=" | ",
    )


if __name__ == "__main__":
    main()

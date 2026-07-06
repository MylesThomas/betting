"""
Phase 0 builder for v5 3PM decomposition workflow.

Context:
- Standardize a canonical market-eligible evaluation universe for all v5 phases.
- Universe rows are player/date/game rows where at least one player_threes market
  line existed for that player/date.
- Output must be deterministic and reusable by downstream scripts.
"""

from __future__ import annotations

import argparse

from v5_workflow_lib import add_research_features
from v5_workflow_lib import build_eval_universe
from v5_workflow_lib import build_market_eligibility
from v5_workflow_lib import build_universe_qc
from v5_workflow_lib import load_player_logs
from v5_workflow_lib import load_player_props
from v5_workflow_lib import resolve_output_path
from v5_workflow_lib import set_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI args for phase 0."""
    parser = argparse.ArgumentParser(description="Build v5 canonical eval universe.")
    parser.add_argument("--phase", type=str, default="phase0")
    parser.add_argument("--season", type=str, default="*")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--cache-dir", type=str, default="~/Downloads/tmp")
    parser.add_argument("--use-cache", type=str, default="true")
    parser.add_argument("--force-refresh-cache", type=str, default="false")
    parser.add_argument("--output-universe", type=str, default="")
    parser.add_argument("--output-qc", type=str, default="")
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    """Parse common string bool variants."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value}")


def main() -> None:
    """Run phase 0 and save canonical universe + QC tables."""
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
    eligible_df = build_market_eligibility(props_df)
    universe = build_eval_universe(logs_df=logs_df, eligible_df=eligible_df)
    universe = add_research_features(universe)
    universe = universe.sort_values(["season", "date", "player_normalized", "game_id"]).reset_index(
        drop=True
    )
    qc = build_universe_qc(universe)

    universe_path = resolve_output_path(args.output_universe, "v5_eval_universe.parquet")
    qc_path = resolve_output_path(args.output_qc, "v5_eval_universe_qc.csv")
    universe.to_parquet(universe_path, index=False)
    qc.to_csv(qc_path, index=False)

    print(
        "phase=phase0",
        f"seed={args.seed}",
        f"season={args.season}",
        f"rows={len(universe)}",
        f"universe={universe_path}",
        f"qc={qc_path}",
        sep=" | ",
    )


if __name__ == "__main__":
    main()


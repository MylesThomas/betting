"""
Verify audit list columns reproduce B_MIN_MAX_FEATS scalars (plan §6).

Used by unit tests and optional offline QA; keep logic aligned with
``build_rebounds_full_universe.build_rolling_features`` and market/spread joins.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.nba_rebounds_modeling.rebounds_feature_spec import (
    B_MIN_MAX_AUDIT_LIST_COLS,
    B_MIN_MAX_FEATS,
    GROUP_KEYS,
    TEAM_CONTEXT_COLS,
)

_AUDIT_KEYS = list(GROUP_KEYS)


def sample_audit_rows(df: pd.DataFrame, *, max_rows: int | None) -> pd.DataFrame:
    """Same sampling rule as ``verify_audit_lists_dataframe`` (random_state=0)."""
    if max_rows is None:
        return df
    n = min(int(max_rows), len(df))
    if n <= 0:
        return df.iloc[0:0].copy()
    return df.sample(n=n, random_state=0)


def team_audit_kwargs_from_row(row: pd.Series) -> dict[str, str]:
    """If feature row carries team + home/away (v2 feature universe), use for spread check."""
    if not all(c in row.index for c in TEAM_CONTEXT_COLS):
        return {}
    tn, hn, an = (row.get(c) for c in TEAM_CONTEXT_COLS)
    if pd.isna(tn) or pd.isna(hn) or pd.isna(an):
        return {}
    stn, shn, san = str(tn).strip(), str(hn).strip(), str(an).strip()
    if not (stn and shn and san):
        return {}
    return {
        "team_normalized": stn,
        "home_team_norm": shn,
        "away_team_norm": san,
    }


def resolve_team_audit_kwargs(
    row: pd.Series,
    team_frame: pd.DataFrame | None,
    *,
    keys: list[str] | None = None,
) -> dict[str, str]:
    """kwargs subset for ``verify_audit_lists_row`` spread check (empty if unavailable)."""
    inline = team_audit_kwargs_from_row(row)
    if inline:
        return inline
    keys = keys or _AUDIT_KEYS
    out: dict[str, str] = {}
    if team_frame is None:
        return out
    hit = team_frame
    for k in keys:
        hit = hit.loc[hit[k] == row[k]]
    if len(hit) != 1:
        return out
    t = hit.iloc[0]
    out["team_normalized"] = str(t["team_normalized"])
    hn, an = t.get("home_team_norm"), t.get("away_team_norm")
    if pd.notna(hn) and pd.notna(an) and str(hn).strip() and str(an).strip():
        out["home_team_norm"] = str(hn)
        out["away_team_norm"] = str(an)
    else:
        out.pop("team_normalized", None)
    return out


def _close(a: float, b: float, *, atol: float = 1e-6, rtol: float = 1e-9) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return bool(math.isclose(a, b, rel_tol=rtol, abs_tol=atol))


def _audit_1d_sequence(val: Any) -> list[Any] | None:
    """
    List-like cells from pandas/parquet are often ``list``, ``tuple``, or ``ndarray``;
    normalize to a flat Python list. Non-sequence / unsupported types return None.
    """
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return []
        return list(val.ravel().tolist())
    if isinstance(val, (list, tuple)):
        return list(val)
    return None


def _spread_pair_floats(input_spread_by_side: Any) -> tuple[float, float] | None:
    if input_spread_by_side is None:
        return None
    if isinstance(input_spread_by_side, np.ndarray) and input_spread_by_side.size == 2:
        a = input_spread_by_side.ravel()
        return (float(a[0]), float(a[1]))
    if isinstance(input_spread_by_side, (list, tuple)) and len(input_spread_by_side) == 2:
        return (float(input_spread_by_side[0]), float(input_spread_by_side[1]))
    return None


def spread_scalar_from_list(
    input_spread_by_side: Any,
    *,
    team_normalized: str | None,
    home_team_norm: str | None,
    away_team_norm: str | None,
) -> float:
    if input_spread_by_side is None or (isinstance(input_spread_by_side, float) and np.isnan(input_spread_by_side)):
        return float("nan")
    pair = _spread_pair_floats(input_spread_by_side)
    if pair is None:
        return float("nan")
    hs, aws = pair
    if team_normalized is None or home_team_norm is None or away_team_norm is None:
        return float("nan")
    if team_normalized == home_team_norm:
        return hs
    if team_normalized == away_team_norm:
        return aws
    return float("nan")


def verify_audit_lists_row(
    row: pd.Series,
    *,
    team_normalized: str | None = None,
    home_team_norm: str | None = None,
    away_team_norm: str | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-9,
) -> None:
    """Assert one feature row passes §6 checks (raises AssertionError on mismatch)."""
    lines = _audit_1d_sequence(row.get("input_reb_prop_lines"))
    if lines is not None and len(lines) > 0:
        got_min, exp_min = float(min(lines)), float(row["min_line"])
        assert _close(got_min, exp_min, atol=atol, rtol=rtol), f"min_line mismatch: list_min={got_min} vs scalar={exp_min}"
        got_max, exp_max = float(max(lines)), float(row["max_line"])
        assert _close(got_max, exp_max, atol=atol, rtol=rtol), f"max_line mismatch: list_max={got_max} vs scalar={exp_max}"

    if team_normalized is not None and home_team_norm is not None and away_team_norm is not None:
        exp = spread_scalar_from_list(
            row.get("input_spread_by_side"),
            team_normalized=team_normalized,
            home_team_norm=home_team_norm,
            away_team_norm=away_team_norm,
        )
        got_spread = float(row["spread_signed"])
        assert _close(float(exp), got_spread, atol=atol, rtol=rtol), (
            f"spread_signed mismatch: list_derived={exp} vs scalar={got_spread} "
            f"(team={team_normalized}, home={home_team_norm}, away={away_team_norm}, "
            f"list={row.get('input_spread_by_side')})"
        )

    tail60 = _audit_1d_sequence(row.get("input_reb_tail_60"))
    if tail60 is not None and len(tail60) > 0:
        got60 = float(np.nanmean(np.asarray(tail60, dtype=float)))
        exp60 = float(row["roll_reb_mean_60"])
        assert _close(got60, exp60, atol=atol, rtol=rtol), f"roll_reb_mean_60 mismatch: list_mean={got60} vs scalar={exp60}"

    tail20 = _audit_1d_sequence(row.get("input_fg3a_tail_20"))
    if tail20 is not None and len(tail20) > 0:
        arr20 = np.asarray(tail20, dtype=float)
        valid20 = arr20[~np.isnan(arr20)]
        if len(valid20) == 0:
            print(
                f"[audit] WARNING: all-NaN fg3a tail — no FGA history for "
                f"player={row.get('player_normalized')} date={row.get('date')} game_id={row.get('game_id')}"
            )
        else:
            got20 = float(np.mean(valid20))
            exp20 = float(row["roll_fg3a_mean_20"])
            assert _close(got20, exp20, atol=atol, rtol=rtol), f"roll_fg3a_mean_20 mismatch: list_mean={got20} vs scalar={exp20}"

    tail5 = _audit_1d_sequence(row.get("input_reb_tail_5"))
    if tail5 is not None:
        s_list = float(pd.Series(tail5, dtype=float).std(ddof=1)) if len(tail5) else float("nan")
        exp_std = float(row["roll_reb_std_5"])
        assert _close(s_list, exp_std, atol=atol, rtol=rtol), f"roll_reb_std_5 mismatch: list_std={s_list} vs scalar={exp_std}"


def verify_audit_lists_dataframe(
    df: pd.DataFrame,
    *,
    team_frame: pd.DataFrame | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-9,
    max_rows: int | None = 500,
    sample_df: pd.DataFrame | None = None,
) -> None:
    """
    Spot-check rows. If ``team_frame`` is given, must align on GROUP_KEYS and supply
    ``team_normalized``, ``home_team_norm``, ``away_team_norm`` for spread checks.

    If ``sample_df`` is set, it is used as the row set to check (``max_rows`` ignored
    for sampling); must be a subset of ``df`` with same columns.
    """
    cols = set(df.columns)
    need = set(B_MIN_MAX_FEATS) | set(B_MIN_MAX_AUDIT_LIST_COLS)
    missing = need - cols
    if missing:
        raise ValueError(f"verify_audit_lists_dataframe missing columns: {sorted(missing)}")

    if sample_df is not None:
        sample = sample_df
    else:
        sample = sample_audit_rows(df, max_rows=max_rows)
    keys = _AUDIT_KEYS
    for _, row in sample.iterrows():
        kwargs: dict = {"atol": atol, "rtol": rtol}
        kwargs.update(resolve_team_audit_kwargs(row, team_frame, keys=keys))
        try:
            verify_audit_lists_row(row, **kwargs)
        except AssertionError as exc:
            key_bits = ", ".join(f"{k}={row.get(k)!r}" for k in keys)
            raise AssertionError(f"audit list failed for row ({key_bits}): {exc}") from exc


def _trunc_repr(val: Any, max_ch: int = 220) -> str:
    if val is None:
        return "None"
    if isinstance(val, float) and pd.isna(val):
        return "nan"
    s = repr(val)
    if len(s) <= max_ch:
        return s
    return s[: max_ch - 3] + "..."


def _most_recent_date_rows(df: pd.DataFrame, n_show: int) -> pd.DataFrame:
    """N rows with the latest `date` (calendar) in the frame — good for "today’s slate" inspection."""
    if n_show <= 0 or len(df) == 0:
        return df.iloc[0:0]
    if "date" not in df.columns:
        return df.head(n_show)
    sub = df.copy()
    sub["_d"] = pd.to_datetime(sub["date"], errors="coerce")
    sub = sub.sort_values(
        by=["_d", "season", "player_normalized", "game_id"],
        ascending=[False, True, True, True],
        na_position="last",
    )
    sub = sub.drop(columns=["_d"], errors="ignore")
    return sub.head(n_show)


def print_audit_sample_to_stdout(
    df: pd.DataFrame,
    team_frame: pd.DataFrame | None,
    *,
    n_show: int,
    show_by: str = "recent",
    title: str | None = None,
) -> None:
    """
    Print keys, optional team context, B_MIN_MAX scalars, and audit list columns.

    ``show_by``:
    - ``recent`` (default): the ``n_show`` rows with the **latest** ``date`` in ``df`` (e.g. today’s games).
    - ``verification_sample``: the first ``n_show`` rows of ``df`` as given (e.g. random sample passed in).
    """
    if n_show <= 0:
        return
    if show_by == "recent":
        to_print = _most_recent_date_rows(df, n_show)
        default_title = (
            f"audit sample — {n_show} most recent by date (scalars + list inputs; not necessarily "
            "the same rows as the random verification sample)"
        )
    else:
        to_print = df.head(n_show)
        default_title = (
            "audit sample (scalars + list inputs; same order as the verification sample slice)"
        )
    t = default_title if title is None else title
    print(f"\n=== {t} ===")
    for i, (_, row) in enumerate(to_print.iterrows(), start=1):
        print(f"\n--- row {i}/{len(to_print)} ---")
        print("  keys:", " | ".join(f"{k}={row[k]!r}" for k in _AUDIT_KEYS))
        tk = resolve_team_audit_kwargs(row, team_frame, keys=_AUDIT_KEYS)
        if tk:
            print("  team (for spread check):", tk)
        else:
            print("  team (for spread check): (none — spread list vs scalar not checked)")
        print("  scalars (B_MIN_MAX_FEATS):")
        for c in B_MIN_MAX_FEATS:
            print(f"    {c}: {row.get(c)}")
        print("  audit lists (inputs behind those scalars):")
        for c in B_MIN_MAX_AUDIT_LIST_COLS:
            print(f"    {c}: {_trunc_repr(row.get(c))}")

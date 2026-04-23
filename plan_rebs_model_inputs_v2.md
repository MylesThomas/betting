# Plan v2: six **list** columns on `rebounds_scored_*.parquet` (audit trail)

## Goal

Extend the scored slate artifact so each row carries **auditable raw inputs** behind the six model features (`B_MIN_MAX_FEATS`), not only the scalars. Downstream (SNS, notebooks, settlement forensics) can show `mean([...])` / `min([...])` style provenance.

**Important semantic split:** only **three** of the six champion inputs are true **rolling game-log windows** (`roll_reb_mean_60`, `roll_fg3a_mean_20`, `roll_reb_std_5`). The other three are **same-slate market fields** (`min_line`, `max_line`, `spread_signed`). Their natural “list” is **not** “last N games”; it is **cross-book lines** and **team spread sides**. The plan below defines sensible list payloads for all six so the schema stays uniform.

---

## 1. Column contract (suggested names)

Add six new columns on **`rebounds_scored_*`** (and, for consistency, on the **feature universe / slice** that merges into `base` in `prod_score_rebounds_slate.py`, so they flow through without recompute at score time):

| Scalar feature   | New column (example)              | List semantics |
|-----------------|-----------------------------------|------------------|
| `min_line`      | `input_lines_for_min_max`         | Sorted unique rebound **prop lines** across books for that `(season, date, player_normalized, game_id)` on the scoring props side (same population used to derive min/max). Length = number of distinct lines (often small). |
| `max_line`      | *(same list as min)*             | **Same array** as `min_line`’s list (min/max are two summaries of one multiset); avoids duplicate wide columns. **Alternative:** two columns `input_lines_min_source` / duplicate — prefer **one** list column `rebounds_lines_posted` + keep scalars. |
| `spread_signed` | `input_spread_by_side`            | Two numbers: `[home_team_spread, away_team_spread]` from the same source as spread (historical CSV or live event), order fixed (e.g. home then away). Length 2; NA entries if missing. |
| `roll_reb_mean_60` | `input_reb_tail_60`            | Up to **60** prior-game `REB` values (leakage-safe: only games **strictly before** slate game; same ordering as training `shift(1).rolling(60)`). Shorter early season. |
| `roll_fg3a_mean_20` | `input_fg3a_tail_20`          | Up to **20** prior-game `FG3A` values (same leakage rules, window 20). |
| `roll_reb_std_5`  | `input_reb_tail_5`              | Up to **5** prior-game `REB` values used for rolling std (match `min_periods=2` behavior in `v2_build_rebounds_universe.build_rolling_features`). |

**Refinement on min/max:** Implement **one** column `rebounds_book_lines` (list of floats) plus existing `min_line` / `max_line` scalars, **or** name it `input_reb_prop_lines` to avoid confusion with moneyline. Document that `min_line == min(list)` and `max_line == max(list)` when list non-empty.

**Total new columns:** 5 is enough if min/max share one list; **6** if you insist one list per scalar row (redundant for min/max — not recommended).

---

## 2. Where data is produced today (touch points)

| Stage | Script / module | Role |
|-------|------------------|------|
| Full universe | `v2_build_rebounds_universe.py` | `build_rolling_features`: rolling **scalars** only; logs sorted per player. |
| Panel / lines | Same file `build_market_panel` + pregame | `min_line` / `max_line` from props aggregation. |
| Spread | `build_rebounds_input_universe.py`, live path in `build_rebounds_scoring_input.py`, pregame slice | `spread_signed` scalar. |
| Score | `prod_score_rebounds_slate.py` | `base = props.merge(feat)` → `out = base.copy()` + predictions. **Any column on `feat` (or props) that survives merge lands on `out`.** |

**Conclusion:** lists must exist on the **feature slice** (or props) **before** `prod_score_rebounds_slate.py` merge, unless you add a second merge inside scorer (not ideal — duplicates logic).

---

## 3. Implementation strategy (phased)

### Phase A — Rolling tails (hardest, highest value)

1. **Extend `build_rolling_features`** in `v2_build_rebounds_universe.py` (or a helper called from it) to compute, per `(season, date, player_normalized, game_id)` row, **parallel list columns**:
   - After the same `sort_values` / `groupby("player_normalized")` pipeline, for each target `(name, col, window)` used in `B_MIN_MAX_FEATS` only (`reb`/REB 60, `fg3a`/FG3A 20, `reb` std 5), apply a transform that returns **Python lists** (or `object` column of `ndarray`) of length ≤ window: values are **`shift(1)` then last w values** in order oldest→newest (document order in plan + code comment).

2. **Performance:** `groupby.transform` with a lambda that materializes a list per row is slow on ~40k+ rows. Prefer **vectorized or numba** window, or precompute with `rolling(...).apply(list, raw=True)` only for the three windows (profile on laptop first).

3. **Join:** Existing `feat_v2 = panel.merge(logs_target).merge(rolling, ...)` must include the new list columns on `rolling` / `feat_v2` so they flow to `rebounds_model_features_*.parquet`.

4. **Training:** Ensure `prod_train_rebounds_models.py` / manifest **only** uses `B_MIN_MAX_FEATS` for `X` — new columns must **not** be added to `B_MIN_MAX_FEATS` unless you intentionally retrain.

### Phase B — Book lines list (min/max audit)

1. At **market panel** construction (`build_market_panel`) or when building **pregame** `market` groupby in `build_rebounds_pregame_feature_slice.py`, retain **all** `line` values per `(season, date, player_normalized, game_id)` as a list (dedupe + sort optional) into column `rebounds_book_lines`.

2. For **pregame** path, `props_s` has one row per book × line — `groupby(...).agg(lines=("line", lambda s: sorted(s.unique().tolist())))`.

### Phase C — Spread list

1. When `spread_signed` is resolved (live `home_spread_line` / `away_spread_line` or historical two-team rows), persist **`[home_spread, away_spread]`** on the same grain as feature rows (player-game). Join onto feat slice by keys.

### Phase D — Scorer + Parquet

1. **`prod_score_rebounds_slate.py`**: After merge, optionally **assert** list columns exist if feature spec version bumps; no change to `X = base[B_MIN_MAX_FEATS]` if new cols are not in that list.

2. **Parquet schema:** Use **list columns** (`pyarrow.list_(pa.float64())`) via `pd.Series` of `object` lists — verify `to_parquet` round-trip and DuckDB `read_parquet` in your stack. If Arrow complains, fallback **JSON string** column per list (worse for analytics, easier for SNS).

3. **`prod_notify_rebounds_sns.py`:** Optionally print truncated lists (e.g. first/last 5 + length) to avoid SNS size blowups.

---

## 4. Pregame / Lambda-specific gap

`build_rebounds_pregame_feature_slice.py` takes **latest** scalar row per player for rolls; it does **not** currently rebuild rolling from logs. **Options:**

- **(Recommended)** Ensure list columns exist on the **latest historical row** in `feat` parquet for that player and **copy** them forward in pregame merge (same as scalars today). Requires Phase A to have written lists onto historical `feat` rows first.

- **(Heavier)** In pregame script, re-query `logs` / shot profile and recompute tails for slate players only — duplicates universe logic; avoid unless universe refresh is blocked.

---

## 5. Size, cost, and ops

- **File size:** 60 floats × 8 bytes × rows adds up; monitor `rebounds_scored` and `rebounds_feature_universe` sizes and S3 costs.
- **Lambda memory / timeout:** Universe build already heavy; profile after adding list transforms.
- **Backward compatibility:** Old parquets without list columns: scorer and notify should **tolerate missing** optional audit columns (or gate behind `config` flag `emit_feature_input_lists: true`).

---

## 6. Verification checklist (including **list → scalar round-trip**)

**Hard requirement:** For **every** champion input in `B_MIN_MAX_FEATS`, the scalar already stored in the parquet must be **exactly reproducible** (within a small float tolerance, e.g. `1e-9` relative / `1e-6` absolute unless documented otherwise) by applying the **same aggregation** used in training to the new list column on that row. If any row fails, the list column is wrong or misaligned.

| Scalar column | List column(s) | Recompute rule to verify against parquet scalar |
|---------------|----------------|--------------------------------------------------|
| `min_line` | shared book-lines list (e.g. `input_reb_prop_lines`) | `min(list) == min_line` when list non-empty and scalar finite. |
| `max_line` | same list | `max(list) == max_line` under same conditions. |
| `spread_signed` | e.g. `[home_spread, away_spread]` | Pick the element matching the player’s team side (same rule as `attach_spread_signed_from_live_event_lines` / historical join); that value must equal `spread_signed`. If list is NA, scalar should be NA. |
| `roll_reb_mean_60` | `input_reb_tail_60` | `numpy.mean(list) == roll_reb_mean_60` (or `pandas.Series(list).mean()`), after dropping NA entries in the list if any; length ≤ 60. |
| `roll_fg3a_mean_20` | `input_fg3a_tail_20` | Same with **mean** over ≤ 20 values vs `roll_fg3a_mean_20`. |
| `roll_reb_std_5` | `input_reb_tail_5` | `pandas.Series(list).std(ddof=1)` (sample std, **n−1**) must match `roll_reb_std_5` wherever the rolling implementation uses `ddof=1` / pandas default — **must match `v2_build_rebounds_universe.build_rolling_features` exactly** (including `min_periods=2` behavior: when fewer than two valid prior games, both scalar std and list-derived std should agree as NA or as implemented today). |

**How to test (automation recommended):**

1. **Unit tests** on a **tiny synthetic** player log (10–15 games) where hand-calculated mean/std/min/max are known; build one feature row and assert all six round-trips.
2. **Property / integration test** on a **sample of real rows** from `rebounds_scored_*.parquet` (e.g. 500 random rows after universe rebuild): for each row, load the six lists + six scalars and assert the table above; collect **max absolute error** per feature and fail if above tolerance.
3. **Edge cases:** first game of season (short tails), rows with NA `spread_signed`, duplicate book lines, pregame-slice rows (lists copied from latest hist row must still round-trip to the **same** scalars merged into that slate).
4. **Regression:** Confirm **train script** still uses only `B_MIN_MAX_FEATS` for `X` (no accidental extra columns).

**Manual spot-check:** Rebuild universe locally (subset season) → pick one player-date → print list + scalar for each of the six and eyeball before merging the automated assertions.

---

## 7. Optional follow-ups

- SNS / settlement email: pretty-print lists (truncate, show `n=`).
- **Tests:** implement the **§6 round-trip** checks as pytest (synthetic + sampled real rows); keep them in CI so list columns cannot drift from scalars silently.
- **Docs:** `docs/design-docs/nba-rebounds-daily-pipeline.md` one paragraph on audit columns.

---

## Summary

| Effort | Item |
|--------|------|
| **High** | Rolling list columns in `build_rolling_features` + perf + universe join |
| **Medium** | Book-line list + spread pair on feat / pregame slice |
| **Low** | Pass-through to `prod_score`, Parquet validation, notify truncation, config flag |

**Single hardest piece:** leakage-correct **aligned rolling windows as lists** at universe scale without blowing runtime — prototype the three windows first, then wire min/max/spread lists.

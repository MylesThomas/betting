# Live signal generator: Profile load bottleneck — ideation

## What we measured

- **Profile: ~4.2s** per player (load_player_profile)
- Vegas: ~0.2–0.8s, MC: ~2ms, Combo: ~0s
- Step 6 total: ~110s for ~24 players ⇒ ~4.5s per player, dominated by Profile

## Root cause

`load_player_profile(normalized_name)` in `monte_carlo_utils.py` does, **every call**:

1. DuckDB: read **entire** `minute_by_minute.parquet` (~687k rows) into a pandas DataFrame
2. `df['normalized_name'] = df['player_name'].apply(normalize_from_nba_api)` — 687k Python calls
3. Filter to one player, then groupby + quarter loop → profile dict

So we do **N full reads + N × 687k normalizations** per iteration (N = number of players analyzed). That’s redundant: the parquet and the normalized names are the same for every player.

## Ideas (short)

| Idea | Effort | Impact | Notes |
|------|--------|--------|--------|
| **1. Preload minute_by_minute once per iteration** | Low | High | Load parquet once, run `normalize_from_nba_api` once on full df. Pass `preloaded_df` into `load_player_profile`; when provided, skip read and full-df normalize, only filter + aggregate for that player. One read + one 687k apply for the whole scan. |
| **2. Preload once per game** | Low | High | Same as (1) but preload at start of each game’s Step 6. Fewer players per load than per iteration, so slightly more total work than (1), but still 1 read + 1 normalize per game instead of per player. |
| **3. Cache profile results in memory** | Low | Low | Cache `{normalized_name: profile}` for the iteration. Only helps if we see the same player twice in one run (e.g. same player in two games); usually we don’t. |
| **4. Precomputed profile store** | High | High | Pipeline that writes a small “profile cache” (one row per player, pre-aggregated quarters). Live script loads that once and does lookups. Bigger change, good long-term. |
| **5. Parallelize profile load** | Med | Med | Load profiles in parallel (e.g. ThreadPoolExecutor) for all players in a game. Still does N reads and N normalizes; wall time drops to ~4s per game instead of 4s × players. Less effective than (1). |

## Recommendation

- **Implement (1) preload once per iteration:** add optional `preloaded_df` to `load_player_profile`; in the signal generator, load minute_by_minute once before the game loop, compute `normalized_name` once, pass the df into the analyzer so each player only does filter + aggregate. This should bring Step 6 from ~110s to on the order of ~10–20s (one read + one normalize + 24 fast filter+aggregates).

# Data Quirks

**Summary**: Known issues and gotchas in the S3 data sources used by this pipeline.

**Last updated**: 2026-07-03

---

## Game logs (s3://nba-api-mt/player_game_logs/)

### ESPN placeholder data
Lambda IPs are blocked by the NBA API. When the Lambda fetches game logs and gets blocked, it falls back to ESPN. ESPN fallback data has:
- `MIN = 0` for all players
- `WL = 'TBD'`

**How to detect**: `df['MIN'] == 0` for all rows on a date. Run the verify script in `temp_file.txt`.

**Fix**: Run `fetch_nba_player_props.py --fetch-games --force` locally (not on Lambda) to backfill.

### NBA API 12-hour delay
Game logs are not available until ~12–14 hours after games end. If you fetch too early, the API returns empty or invalid JSON.

### GAME_DATE format inconsistency
Files fetched before 2026-02-24 use `%Y-%m-%dT%H:%M:%S` format. Files after use `%Y-%m-%d`.
Always parse with `pd.to_datetime(df['GAME_DATE'], format='mixed')`.

### MIN stored as string "MM:SS"
NBA API returns `MIN` as `"35:24"` (string). The fetch script converts to float minutes.
If you load raw CSVs and see strings, parse with:
```python
def parse_minutes(m):
    if pd.isna(m) or str(m).strip() == '':
        return 0.0
    parts = str(m).split(':')
    return float(parts[0]) + float(parts[1]) / 60 if len(parts) == 2 else float(parts[0])
```

---

## Player props (s3://the-odds-api-mt/nba/historical_player_props/)

### Multiple bookmakers per row
Props are stored at bookmaker level. Aggregate with `groupby(['player', 'game_date']).agg(...)` before joining to game logs.

### `points_under_odds` vs `under_odds`
The raw props CSV uses `under_odds`. After joining via `join_nba_points_props_actuals_charts_gamelines.py`, the column becomes `points_under_odds`. Don't mix them up.

---

## Shot charts (s3://nba-api-mt/player_shot_charts/)

### Missing SHOT_DISTANCE column
Some player shot chart CSVs (e.g. Tyrese Haliburton, Terry Rozier, Ty Jerome) are missing the `SHOT_DISTANCE` column. The join script catches these with `⚠️ Error loading...` warnings and skips them. These players will have `NaN` for `pts_0_6_pct`.

### 2025-26 rim_scorer_pct anomaly
In 2025-26, the shot chart loader flags only 11.2% of players as Rim Attackers (≥40% pts from 0–6ft) vs ~40% in prior seasons. This appears to be a shot chart data coverage issue for 2025-26, not a real change in player behavior. Do not rely on `pts_0_6_pct` as a stable cross-season feature without investigation.

---

## Game lines (s3://the-odds-api-mt/nba/historical_game_lines/)

### Consensus line computation
Game lines are aggregated to a consensus spread per game. If only one bookmaker reported a line, the "consensus" is that single book.

---

---

## NFL player props (s3://the-odds-api-mt/nfl/props_backfill/)

### Tackles market coverage gap
`player_tackles_assists` had **0% coverage on Bovada in 2023**, ~75% in 2024, 100% in 2025. Training data for tackles models must start in **2024**, not 2023. Using 2023 tackle lines from Bovada will produce NaN-heavy joins.

Full-book backfill (DK + FanDuel + all books, `regions=us,us2`) written to:
```
nfl/props_backfill/{market}/{season}/{game_id}.parquet
```
Markets backfilled: `player_tackles_assists` (2024–2025), `player_rush_attempts` (2023–2025), `player_reception_yds` (2023–2025).

### Tackle scoring convention
Books score tackles as: **solo tackle = 1, assisted tackle = 1** (full credit for both). This matches the PBP columns:
- Solo: `solo_tackle_1_player_id`, `solo_tackle_2_player_id`
- Assist: `assist_tackle_1_player_id` … `assist_tackle_4_player_id`

Sum all columns where the player appears — that's their book-scored tackle count. Do not half-credit assists.

---

---

## MLB player props — Odds API returns decimal odds, not American

**Discovered**: 2026-07-03 during MLB pitcher strikeouts pipeline.

The Odds API returns `price` in **decimal odds format** (e.g., 1.91, 2.10, 3.80) when no `oddsFormat` parameter is specified in the request. The MLB `fetch_market_lines.py` script did not pass `"oddsFormat": "american"`, so it received decimal prices.

**The bug**: `build_labeled_dataset.py` called `american_to_decimal(price)` on those decimal values — treating 1.77 as "+1.77 American" → decimal = 1.0177 (wrong). Correct decimal for 1.77 decimal is just 1.77.

**Downstream effect**: All per-line novig values were computed as ≈ 0.50 regardless of actual line (because 1/1.0177 ≈ 98%, and 98%/(98%+98%) ≈ 50%). This corrupted:
- `novig_prob_over` feature (constant ≈ 0.50 → dead feature in model)
- p_mkt in backtest (always ≈ 0.50 → inflated edge by 20–40pp at extreme lines)
- PnL (assumed even-money odds for all bets; real odds at extreme lines are −400 to −600)
- Reported OOS: 68.8% win rate, +4,512u → correct: 39% win rate, +159u, 1.2% ROI

**What to check**: Look for values like 1.77, 1.91, 2.10 in `over_price` columns — these are decimal odds. Use this correct conversion:
```python
def dec_to_american(d):
    if d >= 2.0: return round((d - 1) * 100)
    return round(-100 / (d - 1))

def correct_novig(over_dec, under_dec):
    p_o = 1 / over_dec; p_u = 1 / under_dec
    return p_o / (p_o + p_u)
```

**Fix**: Always pass `"oddsFormat": "american"` in Odds API requests. Verify prices look like −110, +150, not 1.91, 2.50. See NBA assists `run_pipeline.py` for the correct pattern.

**Impact**: MLB strikeouts model needs to be rebuilt (labeled dataset regenerated with correct odds, Steps 3c–6 rerun). The novig_prob_over feature scaler's std is near-zero from training on constant values — cannot be corrected by just passing a new novig value; model must be retrained.

---

## Related

- [[nba-season-structure]]
- [[american-odds]]
- [[nfl-2026-season-context]]

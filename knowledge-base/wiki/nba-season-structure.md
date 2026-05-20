# NBA Season Structure

**Summary**: NBA season phases, dates, and the correct NBA API season_type strings for each.

**Last updated**: 2026-05-20

---

## Season phases

| Phase | NBA API `season_type_nullable` | Typical dates |
|-------|-------------------------------|---------------|
| Regular Season | `'Regular Season'` | Oct – mid-Apr |
| Play-In Tournament | `'PlayIn'` | ~Apr 14–17 |
| Playoffs | `'Playoffs'` | ~Apr 18 – Jun |

**Important**: The play-in and playoffs are separate season types in the NBA API. Querying `'Regular Season'` for a playoff date returns 0 rows — it will NOT error, it silently returns empty. Always use the correct type for the date.

## Season dates by year

Defined in `config/season_dates.yaml`. Key fields per NBA season:

| Field | Description |
|-------|-------------|
| `season_start` | First day of regular season |
| `regular_season_end` | Last day of regular season |
| `playin_start` | First day of play-in tournament |
| `playin_end` | Last day of play-in tournament |
| `playoff_start` | First day of first round playoffs |
| `playoff_end` | Last NBA Finals game |

## Helper function

`fetch_nba_player_props.py` contains `_get_nba_season_type(date_str, season)` which reads from `season_dates.yaml` and returns the correct season type string for any date:

```python
season_type = _get_nba_season_type('2026-04-15', '2025-26')  # → 'PlayIn'
season_type = _get_nba_season_type('2026-04-19', '2025-26')  # → 'Playoffs'
season_type = _get_nba_season_type('2026-01-15', '2025-26')  # → 'Regular Season'
```

## Season string format

NBA API season strings use `YYYY-YY` format: `'2025-26'`, `'2024-25'`, etc.

## Play-in tournament history

Play-in started in 2020-21. From 2021-22 it became permanent.

| Season | Play-in dates | Playoff start |
|--------|--------------|---------------|
| 2020-21 | May 18–21, 2021 | May 22, 2021 |
| 2021-22 | Apr 12–15, 2022 | Apr 16, 2022 |
| 2022-23 | Apr 11–14, 2023 | Apr 15, 2023 |
| 2023-24 | Apr 16–19, 2024 | Apr 20, 2024 |
| 2024-25 | Apr 15–18, 2025 | Apr 19, 2025 |
| 2025-26 | Apr 14–17, 2026 | Apr 18, 2026 |

## Common mistakes

- Backfilling game logs with `season_type='Regular Season'` for dates in April — returns 0 rows silently. Verify with `MIN > 0` check after any backfill.
- Lambda IPs are blocked by NBA API regardless of season type — always backfill locally.

## Related

- [[data-quirks]]

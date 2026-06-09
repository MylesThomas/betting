# The Odds API — V4 Reference

Source: https://the-odds-api.com/liveapi/guides/v4/
Raw HTML: `knowledge-base/raw/the-odds-api/`

## Host

```
https://api.the-odds-api.com
```

All endpoints require `?apiKey={apiKey}`.

---

## Endpoints

| Endpoint | Path | Quota cost | Notes |
|----------|------|-----------|-------|
| GET sports | `/v4/sports/` | **Free** | Returns in-season sport keys. `all=true` includes off-season. |
| GET odds | `/v4/sports/{sport}/odds/` | 1 per region per market | Featured markets only (h2h, spreads, totals, outrights). |
| GET scores | `/v4/sports/{sport}/scores/` | 1 (live/upcoming) or 2 (with `daysFrom`) | Live scores update ~30s. Up to 3 days back. |
| GET events | `/v4/sports/{sport}/events/` | **Free** | Event IDs + teams + commence times. No odds. |
| GET event odds | `/v4/sports/{sport}/events/{eventId}/odds/` | 1 per region per market | Accepts any market key incl. player props. Single event only. |
| GET event markets | `/v4/sports/{sport}/events/{eventId}/markets/` | 1 | Returns recently-seen market keys per bookmaker for one event. |
| GET participants | `/v4/sports/{sport}/participants/` | 1 | Team or player list. Not players-on-a-team. |
| GET historical odds | `/v4/historical/sports/{sport}/odds/` | **10** per region per market | Paid plans only. Snapshots from 2020-06-06, every 10m (5m after 2022-09). |
| GET historical events | `/v4/historical/sports/{sport}/events/` | 1 | Paid plans only. Find historical event IDs. |
| GET historical event odds | `/v4/historical/sports/{sport}/events/{eventId}/odds/` | **10** per region per market | Paid plans only. Any market. Props available after 2023-05-03. |

---

## Key shared parameters

| Param | Default | Notes |
|-------|---------|-------|
| `regions` | required (odds/event-odds) | `us`, `us2`, `uk`, `au`, `eu`. Comma-separated. Each group of 10 bookmakers = 1 region for billing. |
| `markets` | `h2h` | Comma-separated. `h2h`, `spreads`, `totals`, `outrights`. Player props only via event-odds endpoint. |
| `oddsFormat` | `decimal` | `decimal` or `american`. |
| `dateFormat` | `iso` | `iso` (ISO 8601) or `unix`. |
| `bookmakers` | — | Overrides `regions` if both provided. |
| `eventIds` | — | Comma-separated 32-char IDs to filter response. |
| `commenceTimeFrom/To` | — | ISO 8601. No effect when `sport=upcoming`. |
| `date` | required for historical | ISO 8601. API returns closest snapshot ≤ date. Response includes `previous_timestamp` and `next_timestamp` for paging. |

---

## Quota cost model

- Cost = `markets × regions`
- Historical endpoints cost **10×** vs live (e.g. 1 market, 1 region = cost 10).
- Empty responses (no events found) **do not count** against quota.
- For event-odds: cost is based on unique markets in the *response*, not the request.
- Every response includes headers:
  - `x-requests-remaining` — credits left until monthly reset
  - `x-requests-used` — credits used this cycle
  - `x-requests-last` — cost of the last call

---

## Featured vs non-featured markets

- **Featured** (`h2h`, `spreads`, `totals`, `outrights`): available on `/odds` and `/historical/odds`.
- **Non-featured** (player props, period markets, alternate lines): only available via `/events/{id}/odds` or `/historical/events/{id}/odds`. Using them on the main `/odds` endpoint returns `INVALID_MARKET`.

---

## Rate limiting

- Limit: **30 requests/second**.
- HTTP 429 on breach (`EXCEEDED_FREQ_LIMIT`). Retry after a few seconds.
- Network jitter can cause 429s even slightly below the limit — build in retry logic.

---

## Key error codes

| Code | Cause |
|------|-------|
| `MISSING_KEY` / `INVALID_KEY` | No or bad `apiKey` param |
| `EXCEEDED_FREQ_LIMIT` | >30 req/s; HTTP 429 |
| `OUT_OF_USAGE_CREDITS` | Monthly quota exhausted |
| `INVALID_MARKET` | Non-featured market used on `/odds`; or typo |
| `INVALID_MARKET_COMBO` | e.g. using `h2h` on an outrights sport like NFL Super Bowl winner |
| `EVENT_NOT_FOUND` | Event concluded or wrong ID |
| `HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN` | Historical endpoints require paid plan |
| `HISTORICAL_MARKETS_UNAVAILABLE_AT_DATE` | Requested market didn't exist at that snapshot time |

---

## Gotchas

- `sport=upcoming` always valid on `/odds` — returns live + next 8 games across all sports. `commenceTimeFrom/To` ignored when using `upcoming`.
- In-play detection: if `commence_time < now`, the event is live. `/odds` does not return completed events.
- Historical American odds before 2022-09-18 are *derived* from decimal — may have small rounding errors.
- `last_update` on event-odds response is at the *market* level, not the bookmaker level.
- Lay odds (h2h_lay, outrights_lay) automatically included for exchanges (Betfair, Matchbook).

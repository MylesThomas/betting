# API & Tool References

This directory contains references for external APIs and tools used in the codebase.

## Structure

Each reference file should be:
- **LLM-optimized:** Clear, structured, examples-heavy
- **Focused:** Only the parts we actually use
- **Versioned:** Note which API version
- **Timestamped:** When was this extracted

## Planned Reference Files

### `draftkings-api-llms.txt`
- DraftKings player props API
- Endpoints, request/response formats
- Rate limits, error codes
- Example requests

### `the-odds-api-llms.txt`
- The Odds API documentation
- Markets, bookmakers, sports
- Request parameters
- Response schema

### `nba-api-llms.txt`
- NBA.com stats API
- Game logs, player stats, team stats
- Endpoints used in this codebase

### `aws-lambda-patterns-llms.txt`
- Lambda best practices
- Handler patterns
- Error handling
- Environment variables

### `uv-package-manager-llms.txt`
- uv commands and usage
- Sync, install, update
- Why we use uv vs pip

---

## Current Status

📝 **All planned** - These will be created in Phase 1.3 (docs migration) and Phase 2 (domain encoding)

---

## Format Guidelines

Use `.txt` suffix for LLM-optimized reference docs (not `.md`) to distinguish from regular docs.

**Good structure:**
```
# API Name

## Base URL
https://api.example.com/v2

## Authentication
Header: X-API-Key: {key}

## Endpoints We Use

### GET /player-props
Returns player props for today's games

Request:
{
  "sport": "NBA",
  "market": "player_points"
}

Response:
{
  "props": [...]
}

## Rate Limits
100 requests/minute

## Common Errors
- 401: Invalid API key
- 429: Rate limit exceeded
- 500: Server error (retry with backoff)
```

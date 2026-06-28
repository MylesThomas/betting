# Plan: HTML table for settle-window plays email

Handoff for another agent. **Do not skip Phase A** — validate layout with fake data before wiring production.

---

## Context (where things live today)

| Piece | Location |
|--------|-----------|
| Plain-text + HTML plays formatters | `src/nba_rebounds_settlement_email.py` → `format_settlement_email_plays_table()`, `format_settlement_email_plays_table_html()` |
| Plays uploaded to S3 | `upload_email_plays_table_if_yesterday()` → `email_plays_yesterday.csv` + `email_plays_yesterday.html` |
| Daily settlement notify | `lambda/nba_rebounds_daily/lambda_function.py` → `_publish_combined_settlement_sns()`: SNS plain text; optional SES multipart when `SETTLEMENT_SES_SOURCE` / `SETTLEMENT_SES_TO` are set |

**Note (2026-04):** Feature-universe / audit work (`build_rebounds_full_universe`, S3 verify scripts) is separate from this email-formatting plan; no change to settlement HTML/SES behavior from those refactors.

**Constraint:** SNS → email subscriptions use **plain text** for the message body. Raw HTML in `sns.publish(Message=...)` will **not** render as a table in the inbox. To ship real HTML tables you will likely need **Amazon SES** `send_email` with `Html` + `Text` parts (see `scripts/lambda_function_track_game_line_movements.py` → `send_email_via_ses` for a repo pattern), or another HTML-capable channel. Phase A still applies: prove the HTML snippet first, then decide SNS vs SES in Phase B.

---

## Phase A — Try HTML table with fake data (no prod code yet)

**Goal:** A standalone artifact that opens in real email clients and looks good: alignment, numeric columns right-aligned, readable `bookmaker` / `player` lengths, optional zebra striping, header row, `win` / `loss` styling (color or bold + accessible text).

1. **Create a minimal HTML file** (e.g. `workdir_test/settle_plays_email_preview.html` or under `ref/` — keep it out of Lambda packaging unless you intend to ship it):
   - Full document: `<!DOCTYPE html>`, `<meta charset="utf-8">`, `<meta name="viewport" content="width=device-width, initial-scale=1">`.
   - Inline `<style>` (many clients strip `<style>` in `<head>` inconsistently — prefer **inline styles on `<table>`, `<th>`, `<td>`** for maximum compatibility, or duplicate critical rules on elements).
   - One `<table>` with the same logical columns as today: `player`, `strat`, `bookmaker`, `line`, `act`, `diff`, `result`, `und`, `date`.
   - **Fake rows** that stress the layout:
     - Long player name (e.g. “Jabari Smith Jr.”).
     - Long `bookmaker` (`williamhill_us`, `betonlineag`).
     - Mix of `win` / `loss`, negative odds, fractional lines.
   - Optional: a line above the table mimicking the real email (“ROLLUP FILES”, S3 links) so width feels realistic.

2. **Open the file locally** in Chrome/Safari and use “Print → Save as PDF” or a send-test flow if you have one.

3. **Send a real test email** (recommended before coding):
   - Paste the HTML body into a one-off SES test, personal SMTP, or “email on acid” / Litmus if available.
   - Check at least **Gmail web**, **Apple Mail**, and **Outlook** (or web Outlook) — table HTML is where clients diverge most.

4. **Lock decisions** (document in the HTML comment or a short note in this file):
   - Max width for `player` / `bookmaker` (truncate with `title` tooltip vs wrap).
   - Whether `result` uses only color (avoid color-only for accessibility).
   - Font stack (`-apple-system`, `Segoe UI`, etc.) and font size for density vs readability.

**Exit criteria for Phase A:** Stakeholder agrees the fake-data HTML table is readable and the approach works in the target inboxes. No changes required yet to `rebounds_settle_runs.py` or the Lambda.

---

## Phase B — Implement HTML in code (after Phase A looks good)

**Goal:** Production path emits the same structure as the approved preview, with a safe plain-text fallback.

1. **Add a formatter** (prefer next to existing logic):
   - New function e.g. `format_settlement_email_plays_table_html(plays: pd.DataFrame, max_rows: int = 600) -> str` in `scripts/rebounds_settle_runs.py` (or a small `src/...` module imported by both the script and Lambda if you need sharing without duplicating).
   - Reuse the **same sorting, truncation, and column derivations** as `format_settlement_email_plays_table()` (extract shared `DataFrame` prep into a private helper if that avoids drift between text and HTML).

2. **Plain-text fallback:**
   - Keep `format_settlement_email_plays_table()` for SNS-only paths, logging, or `text_body` in SES multipart.

3. **Delivery choice (pick one and implement consistently):**
   - **Option 1 — SES:** Add SES send (or extend existing helper) from the daily Lambda (or the component that currently calls `_publish_combined_settlement_sns`). Message: multipart `Text` + `Html`. Subject can stay “NBA rebounds settled results”.
   - **Option 2 — Stay on SNS only:** You cannot get a real HTML table in standard SNS email; best you can do is **short plain summary + S3 link** to an `.html` object or keep improving monospace / card-style text. If product requires HTML table in inbox, choose Option 1.

4. **S3 artifacts (optional):**
   - Today: `email_plays_yesterday.csv` (machine-readable). You may add `email_plays_yesterday.html` for debugging, audit, or “open in browser” — keep Lambda reading what it needs for the email body.

5. **Wire Lambda:**
   - Replace or supplement the block that reads `plays_text` and appends to `lines` for SNS, depending on Option 1 vs 2.
   - Respect **256 KB SNS** limit if still publishing any SNS copy; SES has its own limits — stay under documented max message size.

6. **Tests / verification:**
   - Unit test: small `DataFrame` in → HTML string contains expected `<th>` / cell values and escapes `&`, `<`, `>` in player names (use `html.escape` on text cells).
   - Manual: one end-to-end run in a dev topic or SES sandbox recipient.

**Exit criteria for Phase B:** Production notification matches the Phase A design; text fallback remains readable; no regression to rollup CSV uploads or settlement logic.

---

## Checklist for the executing agent

- [x] Phase A: fake-data HTML preview file (`workdir_test/settle_plays_email_preview.html`; design locks in HTML comment). **Remaining:** paste into SES / SMTP / Litmus and verify Gmail + Apple Mail + Outlook; stakeholder sign-off.
- [x] Document SNS vs SES: SNS keeps plain text; optional SES multipart when `SETTLEMENT_SES_SOURCE` + `SETTLEMENT_SES_TO` are set (Lambda needs `ses:SendEmail`).
- [x] Phase B: shared prep + `format_settlement_email_plays_table_html` + escaping — `src/nba_rebounds_settlement_email.py`; tests `tests/unit/test_nba_rebounds_settlement_email.py`.
- [x] Phase B: Lambda/script integration — `lambda/nba_rebounds_daily/lambda_function.py`, `scripts/rebounds_settle_runs.py` upload; manual E2E in SES sandbox still recommended.
- [x] `email_plays_yesterday.html` on S3 next to `.csv`.

---

## Out of scope (unless explicitly requested)

- Changing DuckDB settlement math or rollup columns.
- Redesigning the non-plays sections of the email (yesterday / all-time summaries) unless trivially bundled with SES multipart.

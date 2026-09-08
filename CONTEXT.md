# Betting Analytics

A data-driven sports betting analytics system that models player prop markets, finds edge against bookmaker lines, and automates bet recommendations across MLB, NBA, and NFL.

## Language

### Modeling & Edge

**Edge**:
The difference between the model's estimated probability and the bookmaker's implied probability (with vig). A positive edge means the model believes the bet is underpriced. Formula (UNDER side): `(1 - p_model) - (1 / under_price)`.
_Avoid_: alpha, advantage, value (use "edge" consistently)

**p_model**:
The model's predicted probability of the OVER outcome for a given player-game-line. The UNDER probability is `1 - p_model`.
_Avoid_: predicted probability, model score

**edge_under**:
Edge computed for the UNDER side: `(1 - p_model) - (1 / under_price)`. A positive value means the UNDER is underpriced at this bookmaker.

**edge_over**:
Edge computed for the OVER side: `p_model - (1 / over_price)`. Not stored in recommendations; recomputed on demand.

**Raw implied probability**:
The bookmaker's implied win probability derived directly from decimal odds, including vig. `1 / decimal_odds`.
_Avoid_: raw prob, implied prob (be explicit: "raw implied probability")

### Bet Classification

**Play**:
A bet recommendation where edge_under meets or exceeds the minimum bet threshold (`min_bet_edge`). These are actionable bets.
_Avoid_: bet, recommendation (use "play" for the tier label)

**Track**:
A bet that meets a lower monitoring threshold (`min_track_edge`) but falls below the play threshold. Watched but not acted on.
_Avoid_: watch, monitor

**Over Mirror**:
A sanity-check analysis where every row that qualified for an UNDER strategy (filtered by `edge_under >= N%`) has the bet flipped to OVER, and P&L is computed as if the OVER side were taken. Used to verify that the model loses money when betting against its own signal.
_Avoid_: flipped bet, reverse bet, over analysis

### Snapshots & CLV

**Prop snapshot**:
A point-in-time capture of player prop odds (line + price per book), stored hourly for all available MLB games. One parquet file per hourly Lambda run, partitioned by `game_date`.
_Avoid_: price capture, odds snapshot

**Closing line**:
The last prop snapshot before `commence_time` for a player+game+book. Used as the CLV reference endpoint.
_Avoid_: final line, last line

**First-seen price**:
The odds recorded when a player+game pair first appears in snapshot history (`binary_player_game_first_seen=True`). One of two CLV reference prices (the other is the 9am ET snapshot).
_Avoid_: opening price, initial price

**CLV (Closing Line Value)**:
Comparison of a reference price (first-seen or 9am ET) to the closing price, measured in American odds cents. Positive = beat the close (good). Tiered: ok (<5¢) · mild (5–10¢) · moderate (10–20¢) · strong (20–30¢) · severe (≥30¢). A line shift ≥ 0.5 is flagged separately from price CLV.
_Avoid_: closing value, line value

**Tightening**:
Adverse drift in a player's rolling-average line or odds vs. their season-to-date baseline. Flagged when `today_modal_line > season_avg_line` by a meaningful margin, or when `today_avg_under_odds` is worse than `season_avg_under_odds` by ≥ 5 cents. Uses the same tier scale as CLV.
_Avoid_: book adjustment, market movement (use "tightening" when the drift is specifically against our UNDER edge)

### Settlement

**actual_tb**:
The observed total bases for a player in a given game, computed from Statcast pitch-level event data. The ground truth for settling total bases props.

**no_data**:
Settlement outcome when Statcast returns no pitch data for a player-game pair. These rows are excluded from P&L calculations.

### Levels.fyi Research

**Salary record**:
One row of individual compensation data scraped from Levels.fyi (uuid, offer_date, base_salary, total_comp, etc.), stored in `s3://levels-fyi-mt/submissions/submissions.parquet`.
_Avoid_: submission, entry, record

**Submission count**:
The all-time aggregate count of salary records Levels.fyi reports on a company's overview page. Stored as `total_submissions` in `s3://levels-fyi-mt/overview/daily.parquet`. Always monotonically increasing — a daily drop signals a scrape failure.
_Avoid_: submissions (ambiguous), total count

**Submission velocity**:
The daily delta of submission count: `total_submissions[today] - total_submissions[yesterday]`. The time-series signal used for correlation analysis — a proxy for hiring activity at a given company.
_Avoid_: hiring velocity (the signal is submissions, not direct hiring data)

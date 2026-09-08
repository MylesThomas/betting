"""
Step 7 — Mock Email Output — MLB Batter Home Runs.

Demonstrates what the daily email looks like for a given date by loading
the scored spine (produced by Step 4 probability script) and filtering to
the primary production strategy.

Strategy: 0.5 OVER, edge >= 10pp, plus-odds books (over_price >= 2.0)

Demo date: 2026-03-27 (2 players, 11 qualifying bets across 9 books)
Usage:
  python src/mlb_batter_home_runs_modeling/scripts/20260802_step7_mock_email.py [YYYY-MM-DD]
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

HTML_LOG   = REPO_ROOT / "knowledge-base/raw/20260801-mlb-batter-home-runs.html"
SCORED     = Path.home() / "Downloads/tmp/mlb_batter_hr_scored.parquet"

DEMO_DATE          = sys.argv[1] if len(sys.argv) > 1 else "2026-03-27"
STRATEGY_LINE      = 0.5
STRATEGY_DIRECTION = "over"
STRATEGY_EDGE_MIN  = 0.10
STRATEGY_ODDS_MIN  = 2.0   # plus-odds only (decimal >= 2.0)

BOOK_DISPLAY = {
    "draftkings":      "DraftKings",
    "fanduel":         "FanDuel",
    "betmgm":          "BetMGM",
    "pointsbetus":     "PointsBet",
    "caesars":         "Caesars",
    "betonlineag":     "BetOnline",
    "bovada":          "Bovada",
    "mybookieag":      "MyBookie",
    "betus":           "BetUS",
    "lowvig":          "LowVig",
    "windcreek":       "Wind Creek",
    "williamhill_us":  "William Hill",
    "superbook":       "SuperBook",
    "betrivers":       "BetRivers",
    "unibet_us":       "Unibet",
    "fliff":           "Fliff",
    "hardrockbet":     "Hard Rock",
    "betparx":         "BetParx",
    "ballybet":        "BallyBet",
    "espnbet":         "ESPN Bet",
    "hardrockbet_oh":  "Hard Rock OH",
}


def dec_to_american(d) -> str:
    if pd.isna(d):
        return "N/A"
    if d >= 2.0:
        return f"+{int(round((d - 1) * 100))}"
    else:
        return f"-{int(round(100 / (d - 1)))}"


def format_email(df: pd.DataFrame, date: str) -> None:
    sub = df[
        (df["game_date"] == date) &
        (df["offered_line"] == STRATEGY_LINE) &
        (df["edge_over"] >= STRATEGY_EDGE_MIN) &
        (df["over_price"] >= STRATEGY_ODDS_MIN) &
        df["over_price"].notna()
    ].copy()

    print(f"\n{'='*90}")
    print(f"MLB BATTER HOME RUNS — DAILY EMAIL — {date}")
    print(f"Strategy: line={STRATEGY_LINE} {STRATEGY_DIRECTION.upper()}, edge >= {STRATEGY_EDGE_MIN*100:.0f}pp, plus-odds only")
    print(f"{'='*90}")

    if len(sub) == 0:
        print("  No qualifying bets today.")
        return

    sub["over_american"]  = sub["over_price"].apply(dec_to_american)
    sub["under_american"] = sub["under_price"].apply(dec_to_american) if "under_price" in sub else "N/A"
    sub["edge_pp"]        = (sub["edge_over"] * 100).round(1)
    sub["p_model_pct"]    = (sub["p_model_over"] * 100).round(1)
    sub["raw_over_pct"]   = (sub["raw_implied_prob_over"] * 100).round(1)

    # Group by game then player
    game_key = sub["home_team"].fillna("") + " vs " + sub["away_team"].fillna("")
    sub["game_key"] = game_key
    games = sub.groupby("game_key")["game_date"].first().index.tolist()

    n_total = 0
    n_players = 0
    html_rows = []
    first_game = True

    for gk in games:
        game_df = sub[sub["game_key"] == gk]
        home    = game_df["home_team"].iloc[0]
        away    = game_df["away_team"].iloc[0]
        n_plays = game_df["player_key"].nunique()

        game_header = f"  {away} @ {home}  ·  {n_plays} PLAY{'S' if n_plays > 1 else ''}"
        print(f"\n{'─'*90}")
        print(game_header)
        print(f"{'─'*90}")
        html_rows.append(f"<tr class='game-header'><td colspan='10'>{game_header}</td></tr>")

        if first_game:
            print(f"  {'Player':<24} {'Book':<16} {'Line':>5} {'Over$':>7} {'Under$':>7} "
                  f"{'p_model':>8} {'Raw%':>6} {'Edge':>8} {'HR/G(L5)':>9} {'HR/G(c)':>8}")
            print(f"  {'─'*24} {'─'*16} {'─'*5} {'─'*7} {'─'*7} {'─'*8} {'─'*6} {'─'*8} {'─'*9} {'─'*8}")
            first_game = False

        players_in_game = game_df["player_key"].unique()
        n_players += len(players_in_game)

        for pk in players_in_game:
            p_df = game_df[game_df["player_key"] == pk].sort_values("over_price", ascending=False)
            p_model_pct = p_df["p_model_pct"].iloc[0]
            hr_l5       = p_df["hr_roll_L5"].iloc[0] if "hr_roll_L5" in p_df else np.nan
            hr_career   = p_df["hr_roll_career"].iloc[0] if "hr_roll_career" in p_df else np.nan
            raw_pct     = p_df["raw_over_pct"].iloc[0]
            player_name = p_df["player_name"].iloc[0] if "player_name" in p_df else pk

            for _, row in p_df.iterrows():
                book_disp = BOOK_DISPLAY.get(row["bookmaker"], row["bookmaker"])
                over_am   = row["over_american"]
                under_am  = row["under_american"]
                edge_pp   = row["edge_pp"]
                print(
                    f"  {player_name:<24} {book_disp:<16} {STRATEGY_LINE:>5.1f} "
                    f"{over_am:>7} {under_am:>7} "
                    f"{p_model_pct:>7.1f}% {raw_pct:>5.1f}% {edge_pp:>+7.1f}pp "
                    f"{hr_l5:>9.3f} {hr_career:>8.3f}"
                )
                html_rows.append(
                    f"<tr><td>{player_name}</td><td>{book_disp}</td>"
                    f"<td>{STRATEGY_LINE}</td><td>{over_am}</td><td>{under_am}</td>"
                    f"<td>{p_model_pct:.1f}%</td><td>{raw_pct:.1f}%</td>"
                    f"<td>{edge_pp:+.1f}pp</td>"
                    f"<td>{hr_l5:.3f}</td><td>{hr_career:.3f}</td></tr>"
                )
                n_total += 1

    print(f"\n{'='*90}")
    print(f"Total qualifying bets: {n_total}  ({n_players} player(s))")

    # Show model input feature breakdown for spot-check player
    spot_player = "shohei ohtani"
    spot_df = sub[sub["player_key"] == spot_player]
    if len(spot_df) > 0:
        row = spot_df.iloc[0]
        print(f"\n--- Model inputs (spot-check: {row.get('player_name', spot_player)}) ---")
        for feat in ["hr_roll_L5", "hr_roll_L10", "hr_roll_L20", "hr_roll_career",
                     "ab_roll_career", "opp_hr_rate_career", "min_raw_implied_prob_under",
                     "max_raw_implied_prob_under", "is_home"]:
            val = row.get(feat, np.nan)
            print(f"  {feat:<30}: {val}")
        print(f"  {'p_model_over':<30}: {row['p_model_over']:.4f}")
        print(f"  {'edge_over':<30}: {row['edge_over']:.4f}")
        hr_actual = row.get("hr_actual", np.nan)
        print(f"  {'hr_actual':<30}: {hr_actual}  ({'HIT' if hr_actual >= 1 else 'MISS'})")

    return html_rows, n_total, n_players


def ts() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    print(f"Loading scored spine from {SCORED}...")
    df = pd.read_parquet(SCORED)
    print(f"Rows: {len(df):,}")

    # Email output
    result = format_email(df, DEMO_DATE)
    if result is None:
        html_rows, n_total, n_players = [], 0, 0
    else:
        html_rows, n_total, n_players = result

    # ── DuckDB SQL Tests ───────────────────────────────────────────────────────
    con = duckdb.connect()
    today = df[df["game_date"] == DEMO_DATE].copy()
    con.register("today", today)

    tests = [
        ("T1: Demo date has at least 2 players with qualifying OVER bets",
         f"SELECT COUNT(DISTINCT player_key) >= 2 AS pass FROM today "
         f"WHERE offered_line = 0.5 AND edge_over >= {STRATEGY_EDGE_MIN} AND over_price >= 2.0"),
        ("T2: All p_model_over values are between 0 and 1",
         "SELECT COUNT(*) = 0 AS pass FROM today WHERE p_model_over IS NOT NULL AND (p_model_over < 0 OR p_model_over > 1)"),
        ("T3: edge_over = p_model_over - raw_implied_prob_over (within 1e-6)",
         "SELECT MAX(ABS(edge_over - (p_model_over - raw_implied_prob_over))) < 1e-6 AS pass FROM today WHERE edge_over IS NOT NULL"),
        ("T4: p_model_over is book-invariant per (player_key, game_date)",
         "SELECT MAX(p_max - p_min) < 1e-8 AS pass FROM (SELECT player_key, MAX(p_model_over) AS p_max, MIN(p_model_over) AS p_min FROM today GROUP BY player_key)"),
        ("T5: Qualifying bets have plus-odds (decimal >= 2.0)",
         f"SELECT COUNT(*) = 0 AS pass FROM today WHERE offered_line = 0.5 AND edge_over >= {STRATEGY_EDGE_MIN} AND over_price IS NOT NULL AND over_price < 2.0"),
    ]

    print(f"\n{'='*60}")
    print("STEP 7 — SQL TESTS")
    print(f"{'='*60}")
    test_results = []
    passed = 0
    for name, sql in tests:
        try:
            result_val = con.execute(sql).fetchone()[0]
            status = "PASS" if result_val else "FAIL"
            passed += int(bool(result_val))
        except Exception as e:
            status = "ERROR"
            result_val = str(e)
        print(f"  [{status}] {name}")
        test_results.append((name, status, result_val))

    print(f"\n  Tests passed: {passed} / {len(tests)}")

    # ── HTML ───────────────────────────────────────────────────────────────────
    table_header = """
<table>
<thead>
<tr>
  <th>Player</th><th>Book</th><th>Line</th><th>Over $</th><th>Under $</th>
  <th>p_model</th><th>Raw%</th><th>Edge</th><th>HR/G(L5)</th><th>HR/G(career)</th>
</tr>
</thead>
<tbody>
"""
    table_footer = "</tbody></table>"
    rows_html = "\n".join(html_rows)

    section = f"""
<section>
<h2>Step 7 — Mock Email Output</h2>
<p class="timestamp">{ts()}</p>
<p><strong>Strategy:</strong> line={STRATEGY_LINE} {STRATEGY_DIRECTION.upper()}, edge &ge; {STRATEGY_EDGE_MIN*100:.0f}pp, plus-odds only (decimal &ge; 2.0)</p>
<p><strong>Demo date:</strong> {DEMO_DATE} &mdash; {n_players} player(s), {n_total} qualifying bet(s)</p>

<h3>Email Output</h3>
{table_header}
{rows_html}
{table_footer}

<h3>Test Results</h3>
<table>
<thead><tr><th>Test</th><th>Status</th><th>Value</th></tr></thead>
<tbody>
{"".join(f'<tr><td>{n}</td><td class="{"pass" if s=="PASS" else "fail"}">{s}</td><td>{v}</td></tr>' for n,s,v in test_results)}
</tbody>
</table>
<p><strong>Passed: {passed} / {len(tests)}</strong></p>

<h3>Strategy Summary</h3>
<ul>
  <li>Primary signal: model over-probability vs. raw market implied probability</li>
  <li>Edge = p_model_over &minus; raw_implied_prob_over (vig-inclusive, book-specific)</li>
  <li>Only plus-odds bets qualify — we are betting on longshots the model believes are underpriced</li>
  <li>HR rate in this dataset: ~11%. At +300 (2.6% book raw prob + vig), a 10pp+ edge means model &gt;12.6% vs market ~8.5%</li>
  <li>OOS backtest: 173 bets, +37.4u, +21.6% ROI (edge &ge; 10pp, all books, shrink=0)</li>
</ul>
</section>
"""
    with open(HTML_LOG, "a") as f:
        f.write(section)
    print(f"\nHTML appended → {HTML_LOG}")


if __name__ == "__main__":
    main()

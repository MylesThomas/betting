"""
Step 7 — Full 3-section mock email.
  Section 1 (top): Today's plays       — 2025-07-21 (2 plays)
  Section 2:       Yesterday's results  — 2025-07-20 (6 plays, 6W-0L, +5.15u)
  Section 3:       All-time results     — OOF backtested through 2025-07-20

Note: pipeline has never run live. All data is OOF backtest.
"""
import html as html_module
import numpy as np
import pandas as pd

OOF_PATH   = "/Users/thomasmyles/Downloads/tmp/mlb_pitcher_outs_oof_method_a.parquet"
TODAY      = "2025-07-21"
YESTERDAY  = "2025-07-20"
SHRINKAGE  = 0.25
EDGE_PLAY  = 0.10
EDGE_SHOW  = 0.05
LINE_MAX   = 17.5

# ── Helpers ───────────────────────────────────────────────────────────────────
BOOK_ABBREV = {
    "draftkings": "DK", "fanduel": "FD", "betmgm": "MGM", "caesars": "CZR",
    "espnbet": "ESPN", "pointsbet": "PB", "fliff": "Fliff", "betonlineag": "BOL",
    "betparx": "BPX", "betrivers": "BR", "williamhill_us": "WH", "superbook": "SB",
    "bally_bet": "Bally", "mybookieag": "MyBookie", "windcreek": "Wind",
    "hardrockbet": "HardRock", "fanatics": "Fanatics", "bovada": "Bovada",
}
TEAM_ABBREV = {
    "Colorado Rockies": "COL", "Minnesota Twins": "MIN",
    "San Francisco Giants": "SF", "Toronto Blue Jays": "TOR",
    "Chicago Cubs": "CHC", "Atlanta Braves": "ATL",
    "Los Angeles Dodgers": "LAD", "Arizona Diamondbacks": "ARI",
    "New York Yankees": "NYY", "Boston Red Sox": "BOS",
    "Houston Astros": "HOU", "Texas Rangers": "TEX",
    "Philadelphia Phillies": "PHI", "New York Mets": "NYM",
    "Milwaukee Brewers": "MIL", "Cincinnati Reds": "CIN",
    "Baltimore Orioles": "BAL", "Tampa Bay Rays": "TB",
    "Cleveland Guardians": "CLE", "Kansas City Royals": "KC",
    "Seattle Mariners": "SEA", "Oakland Athletics": "OAK",
    "St. Louis Cardinals": "STL", "Pittsburgh Pirates": "PIT",
    "Miami Marlins": "MIA", "Washington Nationals": "WSH",
    "Detroit Tigers": "DET", "Chicago White Sox": "CWS",
    "Los Angeles Angels": "LAA", "San Diego Padres": "SD",
}

def ab(b): return BOOK_ABBREV.get(b.lower(), b.title())
def ta(t): return TEAM_ABBREV.get(t, t[:3].upper())

def to_am(dec):
    try:
        d = float(dec)
        return f"+{int(round((d-1)*100))}" if d >= 2.0 else str(int(round(-100/(d-1))))
    except: return "—"

def fmt(v, f):
    try: return format(float(v), f)
    except: return "—"

he = html_module.escape

def _tsort(t):
    try:
        h, rest = t.split(":"); m=rest[:2]; ap=rest[-2:].strip().upper()
        return (int(h)%12+(12 if ap=="PM" else 0))*60+int(m)
    except: return 9999

# ── Load + score ──────────────────────────────────────────────────────────────
df = pd.read_parquet(OOF_PATH)
df["p_under_s"]  = df["p_under"]*(1-SHRINKAGE) + 0.5*SHRINKAGE
df["edge_under"] = df["p_under_s"] - df["novig_prob_under"]

def score_day(day_df):
    s = day_df[(day_df["under_price"]<=2.0) & (day_df["line"]<=LINE_MAX)].copy()
    s["tier"] = "none"
    s.loc[s["edge_under"]>=EDGE_SHOW, "tier"] = "show"
    s.loc[s["edge_under"]>=EDGE_PLAY, "tier"] = "play"
    s["team"]        = s.apply(lambda r: r["home_team"] if r["is_home"] else r["away_team"], axis=1)
    s["opp"]         = s.apply(lambda r: r["away_team"] if r["is_home"] else r["home_team"], axis=1)
    s["player_name"] = s["player_key"].str.replace("-"," ").str.title()
    s["mkt_over_am"] = s.groupby(["player_key","line","home_team"])["over_price"].transform("mean").map(to_am)
    s["dog"]         = s["under_price"] > 2.0
    s["outcome"]     = s.apply(lambda r: "WIN" if r["outs_recorded"]<r["line"] else ("PUSH" if r["outs_recorded"]==r["line"] else "LOSS"), axis=1)
    s["pnl"]         = s.apply(lambda r: r["under_price"]-1 if r["outcome"]=="WIN" else (0.0 if r["outcome"]=="PUSH" else -1.0), axis=1)
    return s

today_scored = score_day(df[df["game_date"]==TODAY])
yest_scored  = score_day(df[df["game_date"]==YESTERDAY])

# All-time through end of YESTERDAY
hist = df[(df["game_date"]<=YESTERDAY) & (df["edge_under"]>=EDGE_PLAY) & (df["under_price"]<=2.0) & (df["line"]<=LINE_MAX)].copy()
hist["outcome"] = hist.apply(lambda r: "WIN" if r["outs_recorded"]<r["line"] else ("PUSH" if r["outs_recorded"]==r["line"] else "LOSS"), axis=1)
hist["pnl"]     = hist.apply(lambda r: r["under_price"]-1 if r["outcome"]=="WIN" else (0.0 if r["outcome"]=="PUSH" else -1.0), axis=1)

SANS = "system-ui,-apple-system,'Segoe UI',Arial,sans-serif"

# ─────────────────────────────────────────────────────────────────────────────
# SECTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _card(label, value, green=False):
    color = "#276221" if green else "#222"
    return (
        f"<div style='border:1px solid #ddd;border-radius:6px;padding:12px 20px;"
        f"min-width:120px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.06)'>"
        f"<div style='font-size:10px;color:#888;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:.5px;margin-bottom:4px'>{label}</div>"
        f"<div style='font-size:22px;font-weight:700;color:{color}'>{value}</div>"
        f"</div>"
    )

def _game_sort(scored):
    game_times = {}
    for _, r in scored.iterrows():
        key = (r["away_team"], r["home_team"])
        if key not in game_times:
            game_times[key] = r.get("game_time_et", "7:05 PM")
    scored = scored.copy()
    scored["_tsort"] = scored.apply(lambda r: _tsort(game_times.get((r["away_team"],r["home_team"]),"7:05 PM")), axis=1)
    return scored.sort_values(["_tsort","home_team","edge_under"], ascending=[True,True,False]).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: TODAY'S PLAYS
# ─────────────────────────────────────────────────────────────────────────────
# Game times for today (2025-07-21) — approximate
TODAY_TIMES = {
    ("Minnesota Twins", "Colorado Rockies"):      "3:10 PM",
    ("San Francisco Giants", "Toronto Blue Jays"): "7:07 PM",
}
today_scored["game_time_et"] = today_scored.apply(
    lambda r: TODAY_TIMES.get((r["away_team"],r["home_team"]),"7:05 PM"), axis=1)
today_scored = _game_sort(today_scored)

n_play = int((today_scored["tier"]=="play").sum())
n_show = int((today_scored["tier"]=="show").sum())
n_play_pit = today_scored[today_scored["tier"]=="play"]["player_key"].nunique()
n_show_pit = today_scored[today_scored["tier"]=="show"]["player_key"].nunique()

# Game groups
games_today = today_scored[["_tsort","game_time_et","home_team","away_team"]].drop_duplicates(["home_team","away_team"]).sort_values("_tsort")

today_rows = ""
for _, g in games_today.iterrows():
    gdf = today_scored[(today_scored["home_team"]==g["home_team"]) & (today_scored["away_team"]==g["away_team"])]
    n_gplay = int((gdf["tier"]=="play").sum()); n_gshow = int((gdf["tier"]=="show").sum())
    badges = []
    if n_gplay: badges.append(f"<span style='color:#276221'>{n_gplay} play{'s' if n_gplay!=1 else ''}</span>")
    if n_gshow: badges.append(f"<span style='color:#b8860b'>{n_gshow} show</span>")
    if not badges: badges.append("<span style='color:#888'>no plays</span>")
    today_rows += (
        f"<tr style='background:#edf1f5'><td colspan='18' style='padding:7px 10px;font-weight:600;"
        f"font-size:12px;color:#2c3e50;border-top:2px solid #bdc3c7;border-bottom:1px solid #bdc3c7'>"
        f"{he(g['game_time_et'])} ET &nbsp;·&nbsp; {he(g['away_team'])} @ {he(g['home_team'])}"
        f" &nbsp;·&nbsp; {gdf['player_key'].nunique()} pitcher{'s' if gdf['player_key'].nunique()!=1 else ''}"
        f" &nbsp;·&nbsp; {' &nbsp;·&nbsp; '.join(badges)}</td></tr>\n"
    )
    for _, r in gdf.iterrows():
        tier = r["tier"]
        bg = "background:#eaf6ea" if tier=="play" else ("background:#fffde7" if tier=="show" else "")
        status = ("<span style='color:#276221;font-weight:bold'>PLAY ✓</span>" if tier=="play"
                  else "<span style='color:#b8860b;font-weight:bold'>SHOW</span>" if tier=="show"
                  else "<span style='color:#aaa'>—</span>")
        dog_html = "<span style='color:#276221'>✓</span>" if r["dog"] else "<span style='color:#aaa'>×</span>"
        ec = "#276221" if tier=="play" else ("#b8860b" if tier=="show" else "#aaa")
        today_rows += (
            f"<tr style='{bg}'>"
            f"<td>{he(r['player_name'])}</td>"
            f"<td style='text-align:center;color:#555'>{ta(r['team'])}</td>"
            f"<td style='text-align:center;color:#555'>{ta(r['opp'])}</td>"
            f"<td style='text-align:center;font-weight:bold'>UNDER</td>"
            f"<td style='text-align:center'>{fmt(r['line'],'.1f')}</td>"
            f"<td style='text-align:center'>{he(ab(r['bookmaker']))}</td>"
            f"<td style='text-align:center;color:#555'>{r['mkt_over_am']}</td>"
            f"<td style='text-align:center;font-weight:bold;color:#1d4ed8'>{to_am(r['under_price'])}</td>"
            f"<td style='text-align:center'>{dog_html}</td>"
            f"<td style='text-align:center'>{fmt(r['novig_prob_under'],'.1%')}</td>"
            f"<td style='text-align:center'>{fmt(r['p_under_s'],'.1%')}</td>"
            f"<td style='text-align:center;font-weight:bold;color:{ec}'>{fmt(r['edge_under'],'+.1%')}</td>"
            f"<td style='text-align:center;font-size:11px;color:#1565c0;font-weight:bold'>{fmt(r['yhat'],'.1f')}</td>"
            f"<td style='text-align:center;font-size:11px;color:#555'>{fmt(r['outs_roll_career'],'.1f')}</td>"
            f"<td style='text-align:center;font-size:11px;color:#555'>{fmt(r['outs_roll_c5'],'.1f')}</td>"
            f"<td style='text-align:center;font-size:11px;color:#555'>{fmt(r['k_roll_career'],'.1f')}</td>"
            f"<td style='text-align:center;font-size:11px;color:#555'>{fmt(r['opp_k_against_season'],'.2f')}</td>"
            f"<td style='text-align:center'>{status}</td>"
            f"</tr>\n"
        )

# Season stats for cards (2025 YTD through yesterday)
szn = hist[hist["game_date"].str.startswith("2025")]
szn_w = int((szn["outcome"]=="WIN").sum()); szn_l = int((szn["outcome"]=="LOSS").sum())
szn_u = float(szn["pnl"].sum())
season_cards = (
    "<div style='display:flex;gap:12px;margin:16px 0 20px;flex-wrap:wrap'>"
    + _card("2025 PNL (plays)", f"{szn_u:+.2f}u", green=szn_u>=0)
    + _card("Record (plays)", f"{szn_w}W – {szn_l}L")
    + _card("Win %", f"{szn_w/(szn_w+szn_l)*100:.1f}%" if szn_w+szn_l else "—", green=szn_w>szn_l)
    + _card("ROI (plays)", f"{szn_u/max(1,szn_w+szn_l)*100:+.1f}%", green=szn_u>=0)
    + "</div>"
)

section1 = f"""
<h2 style='color:#2c3e50;margin-bottom:4px'>MLB Pitcher Outs UNDER — {TODAY}
  <span style='font-size:13px;color:#888;font-weight:normal'>(mock — historical backtest)</span></h2>
<p style='margin-top:4px'>
  <span style='color:#276221;font-weight:bold'>{n_play_pit}p / {n_play}b plays (≥10pp)</span>
  &nbsp;·&nbsp;
  <span style='color:#b8860b;font-weight:bold'>{n_show_pit}p / {n_show}b shows (5–10pp)</span>
  &nbsp;·&nbsp; UNDER · minus odds · line ≤17.5
</p>
<p style='font-size:11px;color:#888;margin-top:2px'>
  <span class='lp'></span>Green = PLAY (bet) &nbsp;&nbsp;
  <span class='ls'></span>Yellow = SHOW (paper only) &nbsp;&nbsp;
  Grey = no edge (context only)
</p>
{season_cards}
<details open>
  <summary>▸ Strategy: Pitcher Outs UNDER &nbsp;<span style='font-weight:normal;color:#666'>({len(today_scored)} bets evaluated · {n_play} plays)</span></summary>
  <table>
    <tr>
      <th colspan='9' style='background:#1e2a35;color:#aab8c2;padding:5px 8px;text-align:center;font-size:10px;font-weight:600;letter-spacing:.5px;text-transform:uppercase;border-right:2px solid #374f5e'>Game Info &amp; Market (per book)</th>
      <th colspan='3' style='background:#1e2a35;color:#aab8c2;padding:5px 8px;text-align:center;font-size:10px;font-weight:600;letter-spacing:.5px;text-transform:uppercase;border-right:2px solid #374f5e'>Model</th>
      <th colspan='5' style='background:#1e2a35;color:#aab8c2;padding:5px 8px;text-align:center;font-size:10px;font-weight:600;letter-spacing:.5px;text-transform:uppercase;border-right:2px solid #374f5e'>Features</th>
      <th colspan='1' style='background:#1e2a35;color:#aab8c2;padding:5px 8px;text-align:center;font-size:10px;font-weight:600;letter-spacing:.5px;text-transform:uppercase'>Status</th>
    </tr>
    <tr>
      <th>Pitcher</th><th style='text-align:center'>Team</th><th style='text-align:center'>Opp</th>
      <th style='text-align:center'>Dir</th><th style='text-align:center'>Line</th>
      <th style='text-align:center'>Book</th><th style='text-align:center'>Over<br>Odds</th>
      <th style='text-align:center'>Under<br>Odds</th>
      <th style='text-align:center;border-right:2px solid #1e2a35'>Dog?</th>
      <th style='text-align:center'>Market<br>Under%</th><th style='text-align:center'>Model<br>Under%</th>
      <th style='text-align:center;border-right:2px solid #1e2a35'>Edge</th>
      <th style='text-align:center'>Proj<br>Outs</th><th style='text-align:center'>Career<br>Outs</th>
      <th style='text-align:center'>c5<br>Outs</th><th style='text-align:center'>Career<br>Ks</th>
      <th style='text-align:center;border-right:2px solid #1e2a35'>Opp<br>K/G</th>
      <th style='text-align:center'>Status</th>
    </tr>
    {today_rows}
  </table>
</details>"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: YESTERDAY'S RESULTS
# ─────────────────────────────────────────────────────────────────────────────
yest_plays = yest_scored[yest_scored["tier"]=="play"].copy()
yest_w = int((yest_plays["outcome"]=="WIN").sum())
yest_l = int((yest_plays["outcome"]=="LOSS").sum())
yest_u = float(yest_plays["pnl"].sum())

# Per-bet rows (wins first, then losses)
yest_sorted = pd.concat([yest_plays[yest_plays["outcome"]=="WIN"], yest_plays[yest_plays["outcome"]!="WIN"]])
yest_bet_rows = ""
for _, r in yest_sorted.iterrows():
    win = r["outcome"] == "WIN"
    bg  = "background:#eaf6ea" if win else ""
    out_color = "#276221" if win else "#c0392b"
    pnl_str = f"{r['pnl']:+.2f}u"
    yest_bet_rows += (
        f"<tr style='{bg}'>"
        f"<td>{he(r['player_name'])}</td>"
        f"<td style='text-align:center;color:#555'>{he(ab(r['bookmaker']))}</td>"
        f"<td style='text-align:center'>{fmt(r['line'],'.1f')}</td>"
        f"<td style='text-align:center;font-weight:bold;color:#1d4ed8'>{to_am(r['under_price'])}</td>"
        f"<td style='text-align:center'>{fmt(r['edge_under'],'+.1%')}</td>"
        f"<td style='text-align:center;font-weight:bold'>{fmt(r['outs_recorded'],'.0f')}</td>"
        f"<td style='text-align:center;color:{out_color};font-weight:bold'>{r['outcome']}</td>"
        f"<td style='text-align:center;font-weight:bold;color:{out_color}'>{pnl_str}</td>"
        f"</tr>\n"
    )

# Game breakdown
game_rows_yest = ""
game_total_w = game_total_l = 0
game_total_u = 0.0
for (away, home), g in yest_plays.groupby(["away_team","home_team"]):
    gw = int((g["outcome"]=="WIN").sum()); gl = int((g["outcome"]=="LOSS").sum()); gu = float(g["pnl"].sum())
    game_total_w += gw; game_total_l += gl; game_total_u += gu
    color = "#276221" if gu >= 0 else "#c0392b"
    game_rows_yest += (
        f"<tr><td>{he(away)} @ {he(home)}</td>"
        f"<td style='text-align:center'>{len(g)}</td>"
        f"<td style='text-align:center;color:#276221;font-weight:bold'>{gw}</td>"
        f"<td style='text-align:center;color:#c0392b;font-weight:bold'>{gl}</td>"
        f"<td style='text-align:center;font-weight:bold;color:{color}'>{gu:+.2f}u</td></tr>\n"
    )
gu_color = "#276221" if game_total_u >= 0 else "#c0392b"
game_rows_yest += (
    f"<tr style='background:#f5f5f5;font-weight:bold'>"
    f"<td>Total</td><td style='text-align:center'>{yest_w+yest_l}</td>"
    f"<td style='text-align:center;color:#276221'>{yest_w}</td>"
    f"<td style='text-align:center;color:#c0392b'>{yest_l}</td>"
    f"<td style='text-align:center;color:{gu_color}'>{yest_u:+.2f}u</td></tr>\n"
)

section2 = f"""
<hr style='border:none;border-top:2px solid #e0e0e0;margin:28px 0'>
<h3 style='color:#2c3e50;margin-bottom:6px'>Yesterday's Results — {YESTERDAY}</h3>
<p style='font-size:13px;margin-top:0'>
  <strong style='color:{"#276221" if yest_u>=0 else "#c0392b"}'>{yest_w}W / {yest_l}L
  &nbsp;·&nbsp; {yest_u:+.2f}u &nbsp;·&nbsp; {yest_u/max(1,yest_w+yest_l)*100:+.1f}% ROI</strong>
</p>
<table style='width:auto;min-width:600px'>
  <tr><th>Pitcher</th><th>Book</th><th>Line</th><th>Under Odds</th><th>Edge</th><th>Actual Outs</th><th>Outcome</th><th>P&amp;L</th></tr>
  {yest_bet_rows}
</table>
<p style='font-size:12px;font-weight:600;color:#555;margin:20px 0 4px'>By game</p>
<table style='width:auto'>
  <tr><th>Game</th><th>Bets</th><th>W</th><th>L</th><th>Net</th></tr>
  {game_rows_yest}
</table>"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: ALL-TIME RESULTS
# ─────────────────────────────────────────────────────────────────────────────
at_w = int((hist["outcome"]=="WIN").sum())
at_l = int((hist["outcome"]=="LOSS").sum())
at_u = float(hist["pnl"].sum())
at_n = len(hist)

alltime_cards = (
    "<div style='display:flex;gap:12px;margin:16px 0 20px;flex-wrap:wrap'>"
    + _card(f"All-Time PNL", f"{at_u:+.2f}u", green=at_u>=0)
    + _card("All-Time Record", f"{at_w}W – {at_l}L")
    + _card("Win %", f"{at_w/max(1,at_w+at_l)*100:.1f}%", green=at_w>at_l)
    + _card("ROI", f"{at_u/max(1,at_n)*100:+.1f}%", green=at_u>=0)
    + "</div>"
)

# By season
season_rows = ""
for szn_yr, g in hist.groupby(hist["game_date"].str[:4]):
    sw=int((g["outcome"]=="WIN").sum()); sl=int((g["outcome"]=="LOSS").sum()); su=float(g["pnl"].sum())
    color = "#276221" if su >= 0 else "#c0392b"
    season_rows += (
        f"<tr><td>{szn_yr}</td><td style='text-align:center'>{len(g)}</td>"
        f"<td style='text-align:center'>{sw}W – {sl}L</td>"
        f"<td style='text-align:center'>{sw/max(1,sw+sl)*100:.1f}%</td>"
        f"<td style='text-align:center;font-weight:bold;color:{color}'>{su:+.2f}u</td>"
        f"<td style='text-align:center;color:{color}'>{su/max(1,len(g))*100:+.1f}%</td></tr>\n"
    )

section3 = f"""
<hr style='border:none;border-top:2px solid #e0e0e0;margin:28px 0'>
<h3 style='color:#2c3e50;margin-bottom:6px'>All-Time Results <span style='font-size:12px;color:#888;font-weight:normal'>(OOF backtest through {YESTERDAY})</span></h3>
{alltime_cards}
<table style='width:auto;min-width:500px'>
  <tr><th>Season</th><th>Bets</th><th>Record</th><th>Win %</th><th>Units</th><th>ROI</th></tr>
  {season_rows}
</table>"""

# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLE
# ─────────────────────────────────────────────────────────────────────────────
email_html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<style>
  body {{font-family:{SANS};color:#222;max-width:1700px;margin:auto;padding:20px}}
  table {{border-collapse:collapse;width:100%;margin-top:8px}}
  th {{background:#2c3e50;color:#fff;padding:7px 8px;text-align:left;font-size:12px;white-space:nowrap}}
  td {{padding:5px 8px;border-bottom:1px solid #e0e0e0;font-size:12px}}
  details {{margin-top:16px;border:1px solid #ddd;border-radius:6px;padding:0 12px 8px}}
  summary {{font-weight:600;font-size:14px;cursor:pointer;padding:10px 0;color:#2c3e50;user-select:none}}
  .footer {{background:#ecf0f1;border-radius:6px;padding:10px 16px;margin-top:28px;font-size:12px;color:#555}}
  .lp {{display:inline-block;width:12px;height:12px;background:#eaf6ea;border:1px solid #276221;margin-right:4px;vertical-align:middle}}
  .ls {{display:inline-block;width:12px;height:12px;background:#fffde7;border:1px solid #b8860b;margin-right:4px;vertical-align:middle}}
</style>
</head><body>
{section1}
{section2}
{section3}
<div class='footer'>
  Flat 1u per book bet &nbsp;·&nbsp; consensus_line bootstrap (10k samples) · shrinkage=0.25 &nbsp;·&nbsp;
  Strategy (UNDER · minus odds · edge≥10pp · line≤17.5): OOS +74.2u · +16.63% ROI · n=446 (2025+2026)
</div>
</body></html>"""

OUT = "/Users/thomasmyles/Downloads/tmp/mock_email_2025_07_21.html"
with open(OUT, "w") as f:
    f.write(email_html)

print(f"Saved: {OUT}")
print(f"Today ({TODAY}): {n_play} plays, {n_show} shows, {len(today_scored)} rows")
print(f"Yesterday ({YESTERDAY}): {yest_w}W-{yest_l}L  {yest_u:+.2f}u")
print(f"All-time through {YESTERDAY}: n={at_n} {at_w}W-{at_l}L {at_u:+.2f}u {at_u/max(1,at_n)*100:+.1f}% ROI")

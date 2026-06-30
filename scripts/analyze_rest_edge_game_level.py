"""
Game-level rest edge analysis: Steps 1–6.

Reads ~/Downloads/tmp/game_level_rest.csv and writes
knowledge-base/wiki/nfl-rest-edge-game-level.html.

Usage:
    python scripts/analyze_rest_edge_game_level.py
"""
import json
from pathlib import Path

import pandas as pd

TMP_DIR   = Path.home() / "Downloads" / "tmp"
REPO_ROOT = Path(__file__).parent.parent
OUT_HTML  = REPO_ROOT / "knowledge-base" / "wiki" / "nfl-rest-edge-game-level.html"
SEASONS   = "2010–2025"


# ─── helpers ──────────────────────────────────────────────────────────────────

def load() -> pd.DataFrame:
    df = pd.read_csv(TMP_DIR / "game_level_rest.csv")
    for col in ["covered", "push", "had_bye", "short_week_road", "is_home",
                "post_road_prime", "opp_extra_prep", "in_3_in_10", "in_4_in_17"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


def cs(sub: pd.DataFrame) -> dict:
    """Cover stats for a subset; pushes excluded from cover rate."""
    n       = len(sub)
    n_push  = int(sub["push"].sum())
    n_valid = n - n_push
    n_cover = int(sub.loc[~sub["push"], "covered"].sum()) if n_valid > 0 else 0
    pct     = round(n_cover / n_valid * 100, 1) if n_valid > 0 else None
    return {"n": n, "n_push": n_push, "n_valid": n_valid, "n_cover": n_cover, "cover_pct": pct}


# ─── step computations ────────────────────────────────────────────────────────

def step1(df: pd.DataFrame) -> list[dict]:
    return [{"label": lbl, **cs(df[m])} for lbl, m in [
        ("Rest advantage (edge > 0)",    df["rest_edge"] > 0),
        ("Neutral (edge = 0)",           df["rest_edge"] == 0),
        ("Rest disadvantage (edge < 0)", df["rest_edge"] < 0),
    ]]


def step2(df: pd.DataFrame) -> list[dict]:
    return [{"label": lbl, **cs(df[m])} for lbl, m in [
        ("≤ −7",     df["rest_edge"] <= -7),
        ("−4 to −6", df["rest_edge"].between(-6, -4)),
        ("−2 to −3", df["rest_edge"].between(-3, -2)),
        ("−1",       df["rest_edge"] == -1),
        ("0",        df["rest_edge"] == 0),
        ("+1",       df["rest_edge"] == 1),
        ("+2 to +3", df["rest_edge"].between(2, 3)),
        ("+4 to +6", df["rest_edge"].between(4, 6)),
        ("≥ +7",     df["rest_edge"] >= 7),
    ]]


def step3_simple(df: pd.DataFrame) -> list[dict]:
    return [{"label": lbl, **cs(df[m])} for lbl, m in [
        ("Home", df["is_home"]),
        ("Away", ~df["is_home"]),
    ]]


def step3(df: pd.DataFrame) -> list[dict]:
    rows = []
    for direction, dir_mask in [
        ("Advantage (> 0)",    df["rest_edge"] > 0),
        ("Neutral (= 0)",      df["rest_edge"] == 0),
        ("Disadvantage (< 0)", df["rest_edge"] < 0),
    ]:
        for loc, home_val in [("Home", True), ("Away", False)]:
            sub = df[dir_mask & (df["is_home"] == home_val)]
            rows.append({"direction": direction, "location": loc, **cs(sub)})
    return rows


def step4(df: pd.DataFrame) -> list[dict]:
    rows = []
    for direction, dir_mask in [
        ("Advantage",    df["rest_edge"] > 0),
        ("Neutral",      df["rest_edge"] == 0),
        ("Disadvantage", df["rest_edge"] < 0),
    ]:
        for period, week_mask in [
            ("All weeks", pd.Series(True, index=df.index)),
            ("Weeks 1–5", df["week"] <= 5),
            ("Week 6+",   df["week"] >= 6),
        ]:
            sub = df[dir_mask & week_mask]
            rows.append({"direction": direction, "period": period, **cs(sub)})
    return rows


def step5(df: pd.DataFrame) -> list[dict]:
    return [{"label": lbl, **cs(df[m])} for lbl, m in [
        ("Short-week road (away, < 6 days)", df["short_week_road"]),
        ("Standard road, disadvantage",      (~df["is_home"]) & (~df["short_week_road"]) & (df["rest_edge"] < 0)),
        ("Standard road, neutral",           (~df["is_home"]) & (~df["short_week_road"]) & (df["rest_edge"] == 0)),
        ("Road with rest advantage",         (~df["is_home"]) & (df["rest_edge"] > 0)),
        ("All road games",                   ~df["is_home"]),
    ]]


def step6(df: pd.DataFrame) -> list[dict]:
    return [{"label": lbl, **cs(df[m])} for lbl, m in [
        ("Post-bye (had_bye=True)",          df["had_bye"]),
        ("Mid-week edge (3–6 days, no bye)", df["rest_edge"].between(3, 6) & ~df["had_bye"]),
        ("Small edge (1–2 days)",            df["rest_edge"].between(1, 2)),
        ("Neutral (edge = 0)",               df["rest_edge"] == 0),
        ("Rest disadvantage, std week",      (df["rest_edge"] < 0) & ~df["short_week_road"]),
        ("Short-week road",                  df["short_week_road"]),
    ]]


# ─── HTML generation ──────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #121212; color: #e0e0e0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 14px; line-height: 1.6; }
.container { max-width: 1100px; margin: 0 auto; padding: 24px 16px; }
h1 { font-size: 22px; color: #fff; margin-bottom: 4px; }
.meta { color: #888; font-size: 13px; margin-bottom: 28px; }
nav { margin-bottom: 28px; padding: 10px 0; border-top: 1px solid #2a2a2a; border-bottom: 1px solid #2a2a2a; }
nav a { color: #5b9bd5; text-decoration: none; margin-right: 18px; font-size: 13px; }
nav a:hover { text-decoration: underline; }
.section { margin-bottom: 52px; border-top: 1px solid #2a2a2a; padding-top: 22px; }
h2 { font-size: 17px; color: #fff; margin-bottom: 6px; }
.desc { color: #999; font-size: 13px; margin-bottom: 12px; }
.sharp { background: #1a2a1a; border-left: 3px solid #4caf50; padding: 7px 12px; margin-bottom: 14px; font-size: 12px; color: #aaa; border-radius: 0 3px 3px 0; }
.chart-wrap { margin: 14px 0 8px; }
.dt { width: 100%; border-collapse: collapse; margin-bottom: 4px; font-size: 13px; }
.dt th { background: #1e1e1e; color: #888; padding: 6px 10px; text-align: right; font-weight: 500; border-bottom: 1px solid #2e2e2e; }
.dt th:first-child, .dt th:nth-child(2) { text-align: left; }
.dt td { padding: 5px 10px; text-align: right; border-bottom: 1px solid #1e1e1e; }
.dt td:first-child, .dt td:nth-child(2) { text-align: left; color: #ccc; }
.dt tr:hover td { background: #191919; }
.hi  { color: #4caf50; font-weight: 700; }
.lo  { color: #f44336; font-weight: 700; }
.mid { color: #e0e0e0; }
footer { color: #555; font-size: 12px; margin-top: 40px; border-top: 1px solid #1e1e1e; padding-top: 14px; }
"""


def _pct_class(v, bl):
    if v is None:
        return "mid"
    return "hi" if v >= bl + 2 else ("lo" if v <= bl - 2 else "mid")


def _build_table(rows: list[dict], cols: list[tuple], bl: float) -> str:
    h = '<table class="dt"><thead><tr>'
    for _, hdr in cols:
        h += f"<th>{hdr}</th>"
    h += "</tr></thead><tbody>"
    for r in rows:
        h += "<tr>"
        for key, _ in cols:
            val = r.get(key)
            if key == "cover_pct":
                cls = _pct_class(val, bl)
                cell = f"{val:.1f}%" if val is not None else "—"
                h += f'<td class="{cls}">{cell}</td>'
            else:
                h += f"<td>{val if val is not None else '—'}</td>"
        h += "</tr>"
    h += "</tbody></table>"
    return h


def _bar_color(v, bl):
    if v is None:
        return "#444"
    return "#4caf50" if v >= bl + 2 else ("#f44336" if v <= bl - 2 else "#5b9bd5")


def _annotation(bl):
    return {
        "bl": {
            "type": "line", "yMin": bl, "yMax": bl,
            "borderColor": "#ef5350", "borderWidth": 2, "borderDash": [6, 4],
            "label": {
                "display": True, "content": f"Baseline {bl}%",
                "color": "#ef5350", "position": "end",
                "backgroundColor": "transparent", "font": {"size": 11},
            },
        }
    }


def _scales(bl):
    lo = max(40, bl - 10)
    hi = min(62, bl + 12)
    return {
        "y": {
            "min": lo, "max": hi,
            "ticks": {"color": "#aaa", "callback": "PCT"},
            "grid": {"color": "#2a2a2a"},
            "title": {"display": True, "text": "Cover %", "color": "#777"},
        },
        "x": {"ticks": {"color": "#bbb"}, "grid": {"color": "#1e1e1e"}},
    }


def _simple_chart(cid: str, labels: list, pcts: list, bl: float, height=260,
                  y_min: float | None = None, y_max: float | None = None,
                  ref_line: float | None = None, ref_label: str | None = None) -> str:
    colors  = [_bar_color(v, bl) for v in pcts]
    data    = [v if v is not None else 0 for v in pcts]
    _y_min  = y_min    if y_min    is not None else max(40, bl - 10)
    _y_max  = y_max    if y_max    is not None else min(62, bl + 12)
    _ref    = ref_line  if ref_line  is not None else bl
    _rlbl   = ref_label if ref_label is not None else f"Baseline {bl}%"
    ann = {
        "bl": {
            "type": "line", "yMin": _ref, "yMax": _ref,
            "borderColor": "#ef5350", "borderWidth": 2, "borderDash": [6, 4],
            "label": {
                "display": True, "content": _rlbl,
                "color": "#ef5350", "position": "end",
                "backgroundColor": "transparent", "font": {"size": 11},
            },
        }
    }
    scales = {
        "y": {
            "min": _y_min, "max": _y_max,
            "ticks": {"color": "#aaa", "callback": "PCT"},
            "grid": {"color": "#2a2a2a"},
            "title": {"display": True, "text": "Cover %", "color": "#777"},
        },
        "x": {"ticks": {"color": "#bbb"}, "grid": {"color": "#1e1e1e"}},
    }
    cfg = {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [{"data": data, "backgroundColor": colors, "borderWidth": 0}],
        },
        "options": {
            "responsive": True,
            "plugins": {"legend": {"display": False}, "annotation": {"annotations": ann}},
            "scales": scales,
        },
    }
    cfg_str = json.dumps(cfg).replace('"PCT"', 'v => v + "%"')
    return f"""
<div class="chart-wrap"><canvas id="{cid}" height="{height}"></canvas></div>
<script>(function(){{ new Chart(document.getElementById('{cid}'), {cfg_str}); }})();</script>"""


def _grouped_chart(cid: str, labels: list, datasets: list[dict], bl: float, height=280) -> str:
    ds_out = []
    for d in datasets:
        safe = [v if v is not None else 0 for v in d["data"]]
        ds_out.append({"label": d["label"], "data": safe,
                       "backgroundColor": d["color"], "borderWidth": 0})
    cfg = {
        "type": "bar",
        "data": {"labels": labels, "datasets": ds_out},
        "options": {
            "responsive": True,
            "plugins": {
                "legend": {"labels": {"color": "#ccc", "font": {"size": 12}}},
                "annotation": {"annotations": _annotation(bl)},
            },
            "scales": _scales(bl),
        },
    }
    cfg_str = json.dumps(cfg).replace('"PCT"', 'v => v + "%"')
    return f"""
<div class="chart-wrap"><canvas id="{cid}" height="{height}"></canvas></div>
<script>(function(){{ new Chart(document.getElementById('{cid}'), {cfg_str}); }})();</script>"""


_STD_COLS = [
    ("label",    "Category"),
    ("n_valid",  "n (excl push)"),
    ("n_cover",  "Covers"),
    ("n_push",   "Pushes"),
    ("cover_pct","Cover %"),
]

_DIR_LOC_COLS = [
    ("direction", "Direction"),
    ("location",  "Location"),
    ("n_valid",   "n (excl push)"),
    ("n_cover",   "Covers"),
    ("n_push",    "Pushes"),
    ("cover_pct", "Cover %"),
]

_DIR_PERIOD_COLS = [
    ("direction", "Direction"),
    ("period",    "Period"),
    ("n_valid",   "n (excl push)"),
    ("n_cover",   "Covers"),
    ("n_push",    "Pushes"),
    ("cover_pct", "Cover %"),
]


def generate_html(bl, s1, s2, s3s, s3, s4, s5, s6, n_games, n_push):
    # Step 3 grouped chart
    dirs3  = ["Advantage (> 0)", "Neutral (= 0)", "Disadvantage (< 0)"]
    home3  = [r["cover_pct"] for r in s3 if r["location"] == "Home"]
    away3  = [r["cover_pct"] for r in s3 if r["location"] == "Away"]
    chart3 = _grouped_chart("c3", dirs3, [
        {"label": "Home", "data": home3, "color": "#5b9bd5"},
        {"label": "Away", "data": away3, "color": "#ff9800"},
    ], bl)

    # Step 4 grouped chart
    dirs4  = ["Advantage", "Neutral", "Disadvantage"]
    all4   = [r["cover_pct"] for r in s4 if r["period"] == "All weeks"]
    early4 = [r["cover_pct"] for r in s4 if r["period"] == "Weeks 1–5"]
    late4  = [r["cover_pct"] for r in s4 if r["period"] == "Week 6+"]
    chart4 = _grouped_chart("c4", dirs4, [
        {"label": "All weeks", "data": all4,   "color": "#777"},
        {"label": "Weeks 1–5", "data": early4, "color": "#ff9800"},
        {"label": "Week 6+",   "data": late4,  "color": "#4caf50"},
    ], bl)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NFL Rest Edge — Game-Level Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3/dist/chartjs-plugin-annotation.min.js"></script>
<style>{_CSS}</style>
</head>
<body>
<div class="container">

<h1>NFL Rest Edge — Game-Level Analysis</h1>
<p class="meta">{SEASONS} · {n_games:,} team-game rows · pushes excluded from cover rates (see each table) · baseline cover rate: <strong>{bl:.1f}%</strong></p>

<nav>
  <a href="#s1">1. Direction</a>
  <a href="#s2">2. Magnitude</a>
  <a href="#s3">3. Home vs Away</a>
  <a href="#s4">4. Week 6+ cutoff</a>
  <a href="#s5">5. Short-week road</a>
  <a href="#s6">6. Bye vs mid-week rest</a>
</nav>

<!-- ── STEP 1 ── -->
<div class="section" id="s1">
<h2>Step 1 — Does rest edge direction predict spread cover?</h2>
<p class="desc">Baseline three-way split: teams with more rest vs equal rest vs less rest than their opponent.</p>
{_build_table(s1, _STD_COLS, bl)}
{_simple_chart("c1", [r["label"] for r in s1], [r["cover_pct"] for r in s1], bl,
               y_min=0, y_max=100, ref_line=52.4, ref_label="Breakeven (52.4%)")}
</div>

<!-- ── STEP 2 ── -->
<div class="section" id="s2">
<h2>Step 2 — Does the size of the rest edge matter?</h2>
<p class="desc">Cover rate by rest edge magnitude. −7/−8 = facing a bye team; +7/+8 = team coming off bye vs short-week opponent.</p>
{_build_table(s2, _STD_COLS, bl)}
{_simple_chart("c2", [r["label"] for r in s2], [r["cover_pct"] for r in s2], bl)}
</div>

<!-- ── STEP 3 ── -->
<div class="section" id="s3">
<h2>Step 3 — Home vs Away split</h2>
<p class="desc">Home and away cover rates always sum to ~100% (each game has exactly one cover), so the 50% baseline is misleading here — read marginal effects vs each group's own baseline instead.</p>
<div class="sharp">Sharp (10-yr sample): Road teams with 3–6 day rest edge → <strong>56.1% cover rate</strong> (7.0% ROI) across 134 games</div>

<h3 style="font-size:14px;color:#bbb;margin:16px 0 8px;">3a — Baseline: Home vs Away (all games, no rest split)</h3>
{_build_table(s3s, _STD_COLS, bl)}
{_simple_chart("c3a", [r["label"] for r in s3s], [r["cover_pct"] for r in s3s], bl, height=200)}

<h3 style="font-size:14px;color:#bbb;margin:20px 0 8px;">3b — Home vs Away × rest edge direction</h3>
<p class="desc">Marginal lift from rest advantage: +5pp for home teams (62.8% vs 57.8% baseline), ~0pp for away teams (42.0% vs 42.2% baseline). Disadvantage hurts away teams specifically (37.2%).</p>
{_build_table(s3, _DIR_LOC_COLS, bl)}
{chart3}
</div>

<!-- ── STEP 4 ── -->
<div class="section" id="s4">
<h2>Step 4 — Week 6+ cutoff</h2>
<p class="desc">Sharp claims rest edge is noise early in the season and becomes signal from Week 6 onward, as teams settle their rotations.</p>
<div class="sharp">Sharp: After Week 6, rest advantage → <strong>54.6% cover</strong> (4.2% ROI) across 233 games</div>
{_build_table(s4, _DIR_PERIOD_COLS, bl)}
{chart4}
</div>

<!-- ── STEP 5 ── -->
<div class="section" id="s5">
<h2>Step 5 — Short-week road games specifically</h2>
<p class="desc">Away teams with fewer than 6 days rest are the most structurally disadvantaged situation in the NFL schedule.</p>
<div class="sharp">Sharp: Short-week road teams → <strong>47.4% cover</strong> (−9.4% ROI) · flip to rest advantage = 53.3% (+1.8% ROI) · swing: 11.2% ROI</div>
{_build_table(s5, _STD_COLS, bl)}
{_simple_chart("c5", [r["label"] for r in s5], [r["cover_pct"] for r in s5], bl)}
</div>

<!-- ── STEP 6 ── -->
<div class="section" id="s6">
<h2>Step 6 — Post-bye vs mid-week rest advantage</h2>
<p class="desc">Is a full bye (13+ days rest) worth more than a smaller structural advantage? Sharp argues the 3–6 day mid-week edge is underpriced because books focus on the visible bye.</p>
<div class="sharp">Sharp: Full bye often priced in by books · 3–6 day mid-week edge less visible → more exploitable</div>
{_build_table(s6, _STD_COLS, bl)}
{_simple_chart("c6", [r["label"] for r in s6], [r["cover_pct"] for r in s6], bl)}
</div>

<footer>
  Source: nfl_data_py schedules + spread lines, {SEASONS}.
  Pushes ({n_push:,} total, {n_push / n_games * 100:.1f}% of all team-game rows) excluded from cover rate denominators.
  Green = ≥ baseline + 2pp · Red = ≤ baseline − 2pp.
</footer>

</div>
</body>
</html>"""


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    df = load()
    n_games = len(df)
    n_push  = int(df["push"].sum())
    n_valid = n_games - n_push
    n_cover = int(df.loc[~df["push"], "covered"].sum())
    bl      = round(n_cover / n_valid * 100, 1)

    print(f"Dataset: {n_games:,} rows · {n_push} pushes · {n_valid:,} valid · baseline {bl}%\n")

    s1  = step1(df)
    s2  = step2(df)
    s3s = step3_simple(df)
    s3  = step3(df)
    s4  = step4(df)
    s5  = step5(df)
    s6  = step6(df)

    def _print_flat(title, rows, cols):
        print(title)
        for r in rows:
            parts = []
            for key, width in cols:
                val = r.get(key)
                cell = (f"{val}%" if val is not None else "—") if key == "cover_pct" else (str(val) if val is not None else "—")
                parts.append(f"{cell:>{width}}")
            print("  " + "  ".join(parts))
        print()

    _print_flat("Step 1 — Direction", s1,
                [("label", 40), ("n_valid", 7), ("n_cover", 7), ("cover_pct", 7)])
    _print_flat("Step 2 — Magnitude", s2,
                [("label", 12), ("n_valid", 7), ("n_cover", 7), ("cover_pct", 7)])

    print("Step 3 — Home vs Away")
    for r in s3:
        print(f"  {r['direction']:<25} {r['location']:<5}  n={r['n_valid']:>5}  {str(r['cover_pct'])+'%' if r['cover_pct'] else '—':>7}")
    print()

    print("Step 4 — Week cutoff")
    for r in s4:
        print(f"  {r['direction']:<15} {r['period']:<11}  n={r['n_valid']:>5}  {str(r['cover_pct'])+'%' if r['cover_pct'] else '—':>7}")
    print()

    _print_flat("Step 5 — Short-week road", s5,
                [("label", 40), ("n_valid", 7), ("n_cover", 7), ("cover_pct", 7)])
    _print_flat("Step 6 — Bye vs mid-week rest", s6,
                [("label", 40), ("n_valid", 7), ("n_cover", 7), ("cover_pct", 7)])

    html = generate_html(bl, s1, s2, s3s, s3, s4, s5, s6, n_games, n_push)
    OUT_HTML.write_text(html)
    print(f"HTML written → {OUT_HTML}")


if __name__ == "__main__":
    main()

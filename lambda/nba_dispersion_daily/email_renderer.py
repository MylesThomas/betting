"""Email renderer for dispersion daily plays."""
from __future__ import annotations

from datetime import date

import pandas as pd


def _skipped_for_team(skipped: pd.DataFrame, team: str) -> list[str]:
    if skipped.empty:
        return []
    return sorted(skipped[skipped["team"] == team]["player"].tolist())


def render_subject(plays: pd.DataFrame, today: date) -> str:
    date_str = today.strftime("%a %b %-d")
    if plays.empty:
        return f"Dispersion — No plays today ({date_str})"
    n = len(plays)
    n_teams = plays["team"].nunique()
    return f"Dispersion — {n} play{'s' if n != 1 else ''} today, {n_teams} team{'s' if n_teams != 1 else ''} ({date_str})"


def render_text(plays: pd.DataFrame, skipped: pd.DataFrame, today: date) -> str:
    date_str = today.strftime("%A %B %-d, %Y")
    lines = [
        f"DISPERSION PLAYS — {date_str}",
        "=" * 52,
    ]

    if plays.empty:
        lines += [
            "",
            "No qualifying plays today.",
            "",
            "A qualifying play requires:",
            "  [1] Star residual > σ=1.0 × rolling std in most recent game",
            "  [2] Star and teammates each have ≥ 10 prior games",
            "  [3] Next game (today) within 5 days of trigger",
            "  [4] Teammate has a prop line available",
        ]
    else:
        threshold = plays["threshold_pts"].iloc[0]
        sigma = plays["sigma"].iloc[0]

        for team, group in plays.groupby("team", sort=False):
            row0 = group.iloc[0]
            opp = row0["opponent"] or "TBD"
            game_time = row0["game_time_et"] or "TBD"
            q = row0["spread_q"]
            q_label = row0["spread_q_label"]
            roll_spread = row0["team_roll_spread"]
            spread_str = f"{roll_spread:+.1f}" if roll_spread is not None else "n/a"

            lines += [
                "",
                f"{team} vs {opp}  —  {game_time} ET",
                f"Trigger: {row0['trigger_player']}  {row0['trigger_pts']} pts  "
                f"(roll_avg {row0['trigger_roll_avg']}, resid {row0['trigger_resid']:+.1f}, "
                f"{row0['trigger_sigma_multiple']:.1f}σ)  [{row0['trigger_game_date']}, {row0['gap_days']}d ago]",
                f"Team Q:  Q{q} — {q_label}  (roll spread {spread_str})",
                "",
            ]

            col_w = max(len(str(r["player"])) for _, r in group.iterrows()) + 2
            header = f"  {'Player':<{col_w}}  {'roll_avg':>9}  {'prop_line':>9}  direction"
            lines.append(header)
            lines.append("  " + "-" * (len(header) - 2))
            for _, bet in group.iterrows():
                lines.append(
                    f"  {bet['player']:<{col_w}}  "
                    f"{bet['roll_avg']:>9.1f}  "
                    f"{bet['prop_line']:>9.1f}  "
                    f"{bet['direction']}"
                )

            no_prop = _skipped_for_team(skipped, team)
            if no_prop:
                lines.append("")
                lines.append("  No prop line (skipped):")
                for name in no_prop:
                    lines.append(f"    {name} — no points prop found")
            lines.append("")

        lines += [
            "─" * 52,
            f"Total: {len(plays)} bet{'s' if len(plays) != 1 else ''} across {plays['team'].nunique()} team{'s' if plays['team'].nunique() != 1 else ''}",
            f"σ threshold: {sigma} × {threshold} = {threshold} pts",
            f"Breakeven at −110: 52.4% | Historical win rate: 63.2% (2023–26)",
        ]

    lines.append("")
    return "\n".join(lines)


def render_html(plays: pd.DataFrame, skipped: pd.DataFrame, today: date) -> str:
    date_str = today.strftime("%A %B %-d, %Y")

    header_color = "#5b8dee"
    bg = "#0f1117"
    surface = "#1a1d27"
    surface2 = "#22263a"
    border = "#2e3250"
    text = "#d4d8f0"
    muted = "#7a82a8"
    green = "#63c98a"
    warn = "#f5a623"

    def td(content, color=text, bold=False, align="left"):
        weight = "font-weight:700;" if bold else ""
        return f'<td style="padding:8px 12px;color:{color};{weight}text-align:{align};border-bottom:1px solid {border};">{content}</td>'

    def th(content):
        return f'<th style="padding:8px 12px;color:{muted};font-size:0.78rem;text-transform:uppercase;letter-spacing:0.05em;text-align:left;background:{surface2};border-bottom:1px solid {border};">{content}</th>'

    body_sections = []

    if plays.empty:
        body_sections.append(f"""
        <div style="background:{surface};border-left:3px solid {warn};border-radius:0 6px 6px 0;padding:16px 20px;margin:16px 0;">
          <div style="color:{warn};font-size:0.75rem;font-weight:700;text-transform:uppercase;margin-bottom:6px;">No Plays Today</div>
          <p style="color:{text};font-size:0.88rem;margin:0;">
            No qualifying star nights within 5 days of a game today.<br/>
            Trigger requires: resid &gt; σ=1.0 × std, ≥ 10 prior games, teammate has a prop line.
          </p>
        </div>""")
    else:
        threshold = plays["threshold_pts"].iloc[0]
        sigma = plays["sigma"].iloc[0]

        for team, group in plays.groupby("team", sort=False):
            r0 = group.iloc[0]
            opp = r0["opponent"] or "TBD"
            game_time = r0["game_time_et"] or "TBD"
            q = r0["spread_q"]
            q_label = r0["spread_q_label"]
            roll_spread = r0["team_roll_spread"]
            spread_str = f"{roll_spread:+.1f}" if roll_spread is not None else "n/a"
            q_color = green if q <= 3 else warn

            def _bet_row(bet):
                roll = f"{bet['roll_avg']:.1f}"
                line = f"{bet['prop_line']:.1f}"
                under = f'<span style="color:{header_color};font-weight:700;">UNDER</span>'
                return (
                    "<tr>"
                    + td(bet["player"])
                    + td(roll, align="right")
                    + td(line, align="right")
                    + td(under)
                    + "</tr>"
                )
            bet_rows = "".join(_bet_row(bet) for _, bet in group.iterrows())

            no_prop = _skipped_for_team(skipped, team)
            if no_prop:
                skipped_html = (
                    f'<div style="margin-top:12px;padding:10px 14px;background:{surface2};border-radius:6px;font-size:0.8rem;">'
                    f'<div style="color:{muted};margin-bottom:4px;font-weight:600;">No prop line (skipped)</div>'
                    + "".join(f'<div style="color:{muted};">{n}</div>' for n in no_prop)
                    + "</div>"
                )
            else:
                skipped_html = ""

            body_sections.append(f"""
        <div style="background:{surface};border:1px solid {border};border-radius:8px;padding:20px;margin:16px 0;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;margin-bottom:14px;">
            <div>
              <div style="font-size:1rem;font-weight:700;color:#fff;">{team} <span style="color:{muted};font-weight:400;">vs</span> {opp}</div>
              <div style="color:{muted};font-size:0.82rem;margin-top:3px;">{game_time} ET &nbsp;·&nbsp; trigger {r0['gap_days']}d ago</div>
            </div>
            <div style="background:{surface2};border:1px solid {border};border-radius:6px;padding:8px 14px;font-size:0.82rem;">
              <span style="color:{q_color};font-weight:700;">Q{q}</span>
              <span style="color:{muted};">&nbsp;{q_label}&nbsp;({spread_str})</span>
            </div>
          </div>
          <div style="background:{surface2};border-radius:6px;padding:12px 14px;margin-bottom:14px;font-size:0.84rem;">
            <span style="color:{muted};">Trigger &nbsp;</span>
            <strong style="color:#fff;">{r0['trigger_player']}</strong>
            <span style="color:{text};"> &nbsp;{r0['trigger_pts']} pts &nbsp;(roll avg {r0['trigger_roll_avg']}, resid <span style="color:{green};">{r0['trigger_resid']:+.1f}</span>, {r0['trigger_sigma_multiple']:.1f}σ)</span>
          </div>
          <table style="width:100%;border-collapse:collapse;">
            <thead><tr>{th('Player')}{th('Roll avg')}{th('Prop line')}{th('Bet')}</tr></thead>
            <tbody>{bet_rows}</tbody>
          </table>
          {skipped_html}
        </div>""")

        n = len(plays)
        n_teams = plays["team"].nunique()
        body_sections.append(f"""
        <div style="background:{surface2};border:1px solid {border};border-radius:6px;padding:14px 18px;font-size:0.84rem;color:{muted};">
          {n} bet{'s' if n != 1 else ''} across {n_teams} team{'s' if n_teams != 1 else ''} &nbsp;·&nbsp;
          σ={sigma}, threshold={threshold} pts &nbsp;·&nbsp;
          Historical WR 63.2% (breakeven 52.4%)
        </div>""")

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"/></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:{bg};color:{text};padding:24px;max-width:680px;margin:0 auto;">
  <div style="border-left:4px solid {header_color};padding:14px 18px;background:{surface};border-radius:0 8px 8px 0;margin-bottom:24px;">
    <div style="font-size:1.2rem;font-weight:700;color:#fff;margin-bottom:4px;">Dispersion Plays</div>
    <div style="color:{muted};font-size:0.84rem;">{date_str}</div>
  </div>
  {"".join(body_sections)}
</body></html>"""

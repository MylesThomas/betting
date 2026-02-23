"""
Build full NCAAB conference mapping from Wikipedia scraper output and bake into src.

Uses tmp/ncaab_conferences.csv (from tmp/scrape_ncaab_conference_data.py), matches
every ESPN team name (from src.ncaab_team_name_mapping.ODDS_API_TO_ESPN_NCAAB) to a
conference, then rewrites src/ncaab_conference_data.py with the complete dict so the
lambda and analysis get conference for all teams (in_conference_matchup, home_conference,
away_conference).

Run from repo root:
    python tmp/scrape_ncaab_conference_data.py   # if tmp/ncaab_conferences.csv missing
    python tmp/build_ncaab_conference_mapping_from_wiki.py
"""

import re
import sys
from pathlib import Path

# Repo root: find via .git or config
def _repo_root():
    p = Path(__file__).resolve().parent.parent
    if (p / ".git").exists():
        return p
    raise FileNotFoundError("Repo root (parent of tmp/) not found")

REPO_ROOT = _repo_root()
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from difflib import SequenceMatcher


def _clean_conference(conf: str) -> str:
    """Strip Wikipedia footnotes and normalize to canonical name."""
    if not conf or not isinstance(conf, str):
        return ""
    conf = re.sub(r"\[.*?\]", "", conf).strip()
    # Already canonical in wiki for most; ensure a few variants
    norm = {
        "Atlantic 10": "A-10",
        "Colonial Athletic Association": "CAA",
        "Summit League": "The Summit",
        "Missouri Valley Conference": "Missouri Valley",
        "Southern Conference": "Southern",
        "Southland Conference": "Southland",
        "West Coast Conference": "West Coast",
        "Mid-American Conference": "MAC",
        "Mid-Eastern Athletic Conference": "MEAC",
        "Northeast Conference": "NEC",
        "Ohio Valley Conference": "Ohio Valley",
        "Southwestern Athletic Conference": "SWAC",
        "Western Athletic Conference": "WAC",
        "Sun Belt Conference": "Sun Belt",
        "Big South Conference": "Big South",
        "Metro Atlantic Athletic Conference": "MAAC",
    }
    return norm.get(conf, conf)


def _normalize_for_match(name: str) -> str:
    """Normalize team name for matching (lower, strip footnotes, collapse dash/spaces)."""
    if not name or not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = re.sub(r"\[.*?\]", "", name)
    name = name.replace("\u2013", "-").replace("\u2014", "-")
    name = " ".join(name.split())
    return name


def _fuzzy_score(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def load_wiki_csv(path: Path) -> pd.DataFrame:
    """Load and clean wiki conference CSV."""
    df = pd.read_csv(path)
    df["Conference"] = df["Conference"].apply(_clean_conference)
    # Clean team_name_espn for display/lookup: strip footnotes
    df["team_name_espn_clean"] = df["team_name_espn"].apply(
        lambda x: re.sub(r"\[.*?\]", "", str(x)).strip() if pd.notna(x) else ""
    )
    df["team_name_espn_norm"] = df["team_name_espn_clean"].apply(_normalize_for_match)
    return df


# ESPN name variants (from Odds API / ESPN) that don't match wiki team_name_espn
KNOWN_ALIASES = {
    "Albany Great Danes": "America East",
    "Florida Int'l Golden Panthers": "Conference USA",
    "Florida International Panthers": "Conference USA",
    "Fort Wayne Mastodons": "Horizon League",
    "GW Revolutionaries": "A-10",
    "IU Indianapolis Jaguars": "Horizon League",
    "IUPUI Jaguars": "Horizon League",
    "Kansas City Roos": "The Summit",
    "LIU Sharks": "NEC",
    "Long Beach State 49ers": "Big West",
    "Loyola (Chi) Ramblers": "A-10",
    "Loyola (MD) Greyhounds": "Patriot League",
    "SIU-Edwardsville Cougars": "Ohio Valley",
    "St. Francis (PA) Red Flash": "NEC",
    "Tenn-Martin Skyhawks": "Ohio Valley",
    "UAlbany Great Danes": "America East",
    "UMKC Kangaroos": "The Summit",
    "UT-Arlington Mavericks": "WAC",
}


def build_mapping(
    wiki_path: Path,
    manual_mapping: dict,
    espn_names: set,
) -> tuple[dict, list]:
    """
    Build full ESPN name -> conference dict.
    Returns (mapping, list of unmatched ESPN names).
    """
    df = load_wiki_csv(wiki_path)
    # Exact lookup: normalized ESPN name -> conference (first occurrence wins)
    norm_to_conf = {}
    for _, row in df.iterrows():
        norm = row["team_name_espn_norm"]
        if norm and row["Conference"]:
            norm_to_conf.setdefault(norm, row["Conference"])
    # Also map cleaned wiki team_name_espn (no footnote) so "Tulane Green Wave" etc. match
    for _, row in df.iterrows():
        clean = _normalize_for_match(row["team_name_espn_clean"])
        if clean and row["Conference"]:
            norm_to_conf.setdefault(clean, row["Conference"])

    result = {}
    unmatched = []

    for espn_name in sorted(espn_names):
        espn_name = espn_name.strip()
        if not espn_name:
            continue
        if espn_name in manual_mapping:
            result[espn_name] = manual_mapping[espn_name]
            continue
        if espn_name in KNOWN_ALIASES:
            result[espn_name] = KNOWN_ALIASES[espn_name]
            continue
        norm = _normalize_for_match(espn_name)
        if norm in norm_to_conf:
            result[espn_name] = norm_to_conf[norm]
            continue
        # Fuzzy: find best wiki row by normalized team_name_espn
        best_conf = None
        best_score = 0.0
        for _, row in df.iterrows():
            score = _fuzzy_score(norm, row["team_name_espn_norm"])
            if score > best_score and score >= 0.72:
                best_score = score
                best_conf = row["Conference"]
        if best_conf:
            result[espn_name] = best_conf
        else:
            unmatched.append(espn_name)

    # Add every wiki row as key (so ESPN-only names like "Gardner-Webb Runnin' Bulldogs" get a conference)
    for _, row in df.iterrows():
        key = (row["team_name_espn_clean"] or "").strip()
        if not key or not row["Conference"]:
            continue
        key = key.replace("\u2013", "-").replace("\u2014", "-")
        if key not in result:
            result[key] = row["Conference"]

    # Ensure all KNOWN_ALIASES are in result (for names ESPN returns that aren't in ODDS canonical set)
    for alias, conf in KNOWN_ALIASES.items():
        if alias not in result:
            result[alias] = conf

    return result, unmatched


def write_ncaab_conference_data_module(mapping: dict, out_path: Path, header: str) -> None:
    """Rewrite src/ncaab_conference_data.py with full NCAAB_CONFERENCE_MAPPING_2025_26."""
    lines = [
        header,
        "",
        "NCAAB_CONFERENCE_MAPPING_2025_26 = {",
    ]
    for k in sorted(mapping.keys()):
        v = mapping[k]
        # Escape single quotes in key for Python source
        key_esc = k.replace("'", "\\'")
        lines.append(f"    '{key_esc}': '{v}',")
    lines.append("}")
    lines.append("")
    lines.append("")
    lines.append("def get_team_conference(team_name: str, season: str = '2025-26') -> str:")
    lines.append('    """')
    lines.append("    Get the conference for a given team.")
    lines.append("    ")
    lines.append("    Args:")
    lines.append('        team_name: Team name in ESPN format (e.g., "Wisconsin Badgers")')
    lines.append('        season: Season string (e.g., "2025-26"). Currently only 2025-26 is supported.')
    lines.append("    ")
    lines.append("    Returns:")
    lines.append('        Conference name (e.g., "Big Ten")')
    lines.append("        ")
    lines.append("    Raises:")
    lines.append("        ValueError: If team is not found in mapping")
    lines.append("        NotImplementedError: If season is not 2025-26")
    lines.append('    """')
    lines.append("    if season != '2025-26':")
    lines.append("        raise NotImplementedError(")
    lines.append("            f\"Conference mappings are only available for 2025-26 season. \"")
    lines.append("            f\"Requested season: {season}. Please update NCAAB_CONFERENCE_MAPPING_{season.replace('-', '_')}.\"")
    lines.append("        )")
    lines.append("    ")
    lines.append("    if team_name not in NCAAB_CONFERENCE_MAPPING_2025_26:")
    lines.append("        raise ValueError(")
    lines.append("            f\"Team '{team_name}' not found in conference mapping. \"")
    lines.append("            f\"This may be due to an unmatched team name or missing manual mapping.\"")
    lines.append("        )")
    lines.append("    ")
    lines.append("    return NCAAB_CONFERENCE_MAPPING_2025_26[team_name]")
    lines.append("")
    lines.append("")
    lines.append("def is_conference_game(team1: str, team2: str, season: str = '2025-26') -> bool:")
    lines.append('    """')
    lines.append("    Determine if a game is a conference game (both teams in same conference).")
    lines.append("    ")
    lines.append("    Args:")
    lines.append("        team1: First team name in ESPN format")
    lines.append("        team2: Second team name in ESPN format")
    lines.append('        season: Season string (e.g., "2025-26")')
    lines.append("    ")
    lines.append("    Returns:")
    lines.append("        True if both teams are in the same conference, False otherwise")
    lines.append('    """')
    lines.append("    try:")
    lines.append("        conf1 = get_team_conference(team1, season)")
    lines.append("        conf2 = get_team_conference(team2, season)")
    lines.append("        return conf1 == conf2")
    lines.append("    except (ValueError, NotImplementedError):")
    lines.append("        # If either team is not found or season not supported, return False")
    lines.append("        return False")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict:
    wiki_path = REPO_ROOT / "tmp" / "ncaab_conferences.csv"
    if not wiki_path.exists():
        print("Run first: python tmp/scrape_ncaab_conference_data.py")
        sys.exit(1)

    from src.ncaab_conference_data import NCAAB_CONFERENCE_MAPPING_2025_26
    from src.ncaab_team_name_mapping import ODDS_API_TO_ESPN_NCAAB

    espn_names = set(ODDS_API_TO_ESPN_NCAAB.values())
    print(f"ESPN names to map: {len(espn_names)}")
    print(f"Manual overrides:  {len(NCAAB_CONFERENCE_MAPPING_2025_26)}")

    mapping, unmatched = build_mapping(
        wiki_path,
        NCAAB_CONFERENCE_MAPPING_2025_26,
        espn_names,
    )
    print(f"Mapped:           {len(mapping)}")
    print(f"Unmatched:         {len(unmatched)}")
    if unmatched:
        for u in unmatched[:30]:
            print(f"  - {u}")
        if len(unmatched) > 30:
            print(f"  ... and {len(unmatched) - 30} more")

    out_path = REPO_ROOT / "src" / "ncaab_conference_data.py"
    header = '''"""
NCAAB Conference Mappings

Conference affiliations for NCAA Division I Men's Basketball teams.

IMPORTANT: These mappings are specific to the 2025-26 season.
           Conferences change due to realignment - update annually!

Generated by: tmp/build_ncaab_conference_mapping_from_wiki.py
Source: https://en.wikipedia.org/wiki/List_of_NCAA_Division_I_men%27s_basketball_programs

Major Changes for 2025-26:
- UCLA, USC join Big Ten
- Stanford, California join ACC
- SMU joins ACC
- BYU, Colorado, Arizona, Arizona State join Big 12
- Texas, Oklahoma join SEC
- Oregon, Washington join Big Ten

TODO 2027: Update conference mappings for 2026-27 season realignment.
"""
'''
    write_ncaab_conference_data_module(mapping, out_path, header)
    print(f"Wrote: {out_path}")
    return mapping


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
NFL Game Research Tool - CLI Version
Simple command-line interface for researching NFL games

Usage:
    python research_nfl_games_cli.py --week 16
    python research_nfl_games_cli.py --week 17 --games "KC,TEN" "PHI,WAS"
"""

import argparse
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class NFLGameResearcher:
    """Research NFL games using structured queries"""
    
    def __init__(self, week: int = 16, year: int = 2025):
        self.week = week
        self.year = year
        self.output_dir = Path(__file__).parent.parent / 'data' / 'nfl_research'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_research_queries(self, away_team: str, home_team: str) -> Dict:
        """Generate comprehensive research queries for a game"""
        return {
            'game_preview': [
                f"NFL Week {self.week} {self.year} {away_team} at {home_team} preview",
                f"{away_team} {home_team} Week {self.week} injury report",
                f"{away_team} vs {home_team} matchup analysis",
            ],
            'away_team': [
                f"{away_team} Week {self.week} {self.year} injury report",
                f"{away_team} recent performance last 3 games",
                f"{away_team} news storylines Week {self.week}",
            ],
            'home_team': [
                f"{home_team} Week {self.week} {self.year} injury report",
                f"{home_team} recent performance last 3 games",
                f"{home_team} news storylines Week {self.week}",
            ],
            'betting': [
                f"{away_team} {home_team} betting line Week {self.week}",
                f"{away_team} {home_team} spread movement",
                f"{away_team} {home_team} odds analysis",
            ]
        }
    
    def create_research_guide(self, games: List[Tuple[str, str]]) -> List[Dict]:
        """Create a research guide for multiple games"""
        research_data = []
        
        for away, home in games:
            game_data = {
                'game': f"{away} @ {home}",
                'week': self.week,
                'away_team': away,
                'home_team': home,
                'queries': self.generate_research_queries(away, home),
                'research_notes': {
                    'game_storylines': '# Key storylines for this matchup\n',
                    'away_injuries': '# Key injuries for away team\n',
                    'home_injuries': '# Key injuries for home team\n',
                    'away_recent_form': '# Away team last 3 games\n',
                    'home_recent_form': '# Home team last 3 games\n',
                    'betting_context': '# Line movement and betting trends\n',
                    'other_notes': '# Any other relevant context\n',
                }
            }
            research_data.append(game_data)
        
        return research_data
    
    def save_research_guide(self, research_data: List[Dict]):
        """Save research guide in multiple formats"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save full JSON with space for notes
        json_file = self.output_dir / f"nfl_week{self.week}_research_guide_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(research_data, f, indent=2, ensure_ascii=False)
        
        # Save CSV with queries for easy reference
        csv_file = self.output_dir / f"nfl_week{self.week}_search_queries_{timestamp}.csv"
        csv_rows = []
        for game in research_data:
            for category, queries in game['queries'].items():
                for query in queries:
                    csv_rows.append({
                        'game': game['game'],
                        'category': category,
                        'search_query': query,
                    })
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['game', 'category', 'search_query'])
            writer.writeheader()
            writer.writerows(csv_rows)
        
        # Save markdown checklist
        md_file = self.output_dir / f"nfl_week{self.week}_research_checklist_{timestamp}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"# Week {self.week} NFL Game Research Checklist\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for game in research_data:
                f.write(f"## {game['game']}\n\n")
                f.write(f"### Research Queries\n\n")
                
                for category, queries in game['queries'].items():
                    f.write(f"**{category.replace('_', ' ').title()}:**\n")
                    for query in queries:
                        f.write(f"- [ ] `{query}`\n")
                    f.write("\n")
                
                f.write(f"### Notes\n\n")
                for note_type, template in game['research_notes'].items():
                    f.write(f"**{note_type.replace('_', ' ').title()}:**\n")
                    f.write(f"```\n{template}\n```\n\n")
                
                f.write("---\n\n")
        
        return json_file, csv_file, md_file


def parse_games(game_strings: List[str]) -> List[Tuple[str, str]]:
    """Parse game strings like 'KC,TEN' into tuples"""
    games = []
    for game_str in game_strings:
        parts = game_str.split(',')
        if len(parts) == 2:
            away, home = parts[0].strip().upper(), parts[1].strip().upper()
            games.append((away, home))
        else:
            print(f"⚠️  Skipping invalid game format: {game_str} (use 'AWAY,HOME')")
    return games


def main():
    parser = argparse.ArgumentParser(
        description='Generate NFL game research templates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default Week 16 games
  python research_nfl_games_cli.py --week 16
  
  # Specify custom games
  python research_nfl_games_cli.py --week 17 --games "KC,TEN" "PHI,WAS" "BUF,CLE"
  
  # Different year
  python research_nfl_games_cli.py --week 1 --year 2026 --games "KC,BAL"
        """
    )
    
    parser.add_argument('--week', type=int, default=16, help='NFL week number (default: 16)')
    parser.add_argument('--year', type=int, default=2025, help='NFL season year (default: 2025)')
    parser.add_argument('--games', nargs='*', help='Games in format "AWAY,HOME" (e.g., "KC,TEN" "PHI,WAS")')
    
    args = parser.parse_args()
    
    # Default games for Week 16 if none specified
    if not args.games:
        games = [
            ('CIN', 'MIA'),  # Bengals @ Dolphins
            ('NE', 'BAL'),   # Patriots @ Ravens
            ('LV', 'HOU'),   # Raiders @ Texans
            ('ATL', 'ARI'),  # Falcons @ Cardinals
            ('PIT', 'DET'),  # Steelers @ Lions
            ('MIN', 'NYG'),  # Vikings @ Giants
            ('TB', 'CAR'),   # Buccaneers @ Panthers
        ]
        print(f"ℹ️  No games specified, using default Week {args.week} games")
    else:
        games = parse_games(args.games)
        if not games:
            print("❌ No valid games provided. Use format: --games 'KC,TEN' 'PHI,WAS'")
            return
    
    print(f"\n{'='*60}")
    print(f"NFL Week {args.week} Game Research Tool")
    print(f"{'='*60}\n")
    
    researcher = NFLGameResearcher(week=args.week, year=args.year)
    research_data = researcher.create_research_guide(games)
    
    # Save outputs
    json_file, csv_file, md_file = researcher.save_research_guide(research_data)
    
    print(f"✅ Research guide created for {len(games)} games\n")
    
    print("📋 Output files:")
    print(f"  • {md_file.name} - Markdown checklist ⭐")
    print(f"  • {csv_file.name} - Search queries CSV")
    print(f"  • {json_file.name} - Research template JSON\n")
    
    print(f"📂 Location: {researcher.output_dir}\n")
    
    print("📝 Next steps:")
    print(f"  1. Open the markdown checklist:")
    print(f"     open {md_file}")
    print(f"  2. Work through each search query")
    print(f"  3. Take notes in the template")
    print(f"  4. Write your post with real context\n")
    
    print("🎯 Games to research:")
    for game in research_data:
        print(f"  • {game['game']}")
    print()


if __name__ == "__main__":
    main()


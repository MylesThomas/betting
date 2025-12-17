"""
NFL Game Research Tool
Uses web search to gather context for NFL games

This script generates search queries that you can use to research games.
For automated searching, this would need to integrate with a search API.
"""

import json
import csv
from datetime import datetime
from typing import Dict, List


class NFLGameResearcher:
    """Research NFL games using structured queries"""
    
    def __init__(self, week: int = 16):
        self.week = week
        self.year = 2025
        
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
    
    def create_research_guide(self, games: List[tuple]) -> List[Dict]:
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
    
    def save_research_guide(self, research_data: List[Dict], output_dir: str = '../data/nfl_research'):
        """Save research guide in multiple formats"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save full JSON with space for notes
        json_file = f"{output_dir}/nfl_week{self.week}_research_guide_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(research_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved research guide: {json_file}")
        
        # Save CSV with queries for easy reference
        csv_file = f"{output_dir}/nfl_week{self.week}_search_queries_{timestamp}.csv"
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
        print(f"✓ Saved search queries: {csv_file}")
        
        # Save markdown checklist
        md_file = f"{output_dir}/nfl_week{self.week}_research_checklist_{timestamp}.md"
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
        
        print(f"✓ Saved checklist: {md_file}")
        
        return json_file, csv_file, md_file


def main():
    """Main execution"""
    # Week 16 games to research
    games = [
        ('CIN', 'MIA'),  # Bengals @ Dolphins
        ('NE', 'BAL'),   # Patriots @ Ravens
        ('LV', 'HOU'),   # Raiders @ Texans
        ('ATL', 'ARI'),  # Falcons @ Cardinals
        ('PIT', 'DET'),  # Steelers @ Lions
        ('MIN', 'NYG'),  # Vikings @ Giants
        ('TB', 'CAR'),   # Buccaneers @ Panthers
    ]
    
    week = 16
    
    print(f"\n{'='*60}")
    print(f"NFL Week {week} Game Research Tool")
    print(f"{'='*60}\n")
    
    researcher = NFLGameResearcher(week=week)
    research_data = researcher.create_research_guide(games)
    
    # Save outputs
    json_file, csv_file, md_file = researcher.save_research_guide(research_data)
    
    print(f"\n{'='*60}")
    print(f"Research guide created for {len(games)} games")
    print(f"{'='*60}\n")
    
    print("📋 Output files:")
    print(f"  • {json_file} - Full research template (fill in notes)")
    print(f"  • {csv_file} - All search queries")
    print(f"  • {md_file} - Markdown checklist\n")
    
    print("📝 Next steps:")
    print("  1. Open the markdown checklist")
    print("  2. Run each search query and take notes")
    print("  3. Fill in the research_notes in the JSON file")
    print("  4. Use the context to write your betting post\n")
    
    print("🎯 Games to research:")
    for game in research_data:
        print(f"  • {game['game']}")
    print()


if __name__ == "__main__":
    main()


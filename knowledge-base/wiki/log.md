# Change Log

Append-only. Never delete entries.

---

## 2026-07-03

- data-quirks: added MLB odds format bug — Odds API returns decimal (not American) when `oddsFormat` param is omitted; correct conversion documented; corrupted MLB strikeouts pipeline OOS results (+4,512u was wrong; correct is +159u, 1.2% ROI)

---

## 2026-05-20

- Created knowledge base scaffold
- Added pages: american-odds, roi-and-pnl, edge-calibration, model-evaluation, data-quirks, nba-season-structure
- Seeded from lessons learned during rebounds and points modeling pipelines

---

## 2026-05-26

- Added page: the-odds-api
- Source: scraped V4 docs from the-odds-api.com into knowledge-base/raw/the-odds-api/
- Covers: all endpoints, quota cost model, featured vs non-featured markets, rate limits, error codes, gotchas

---

## 2026-06-28

- Added page: nfl-2026-season-context
- Source: Warren-Sharps-2026-Football-Preview.pdf (pages 1–20)
- Covers: 2026 NFL coaching turnover (47 new coaches), SoS rankings (Vegas win-total methodology), DCOE draft rankings, why 2026 is wide open for bettors

- Updated page: data-quirks
- Source: 20260625 researching player_tackles_assists market.docx (session research log)
- Added: tackles market coverage gap (0% in 2023 on Bovada → start training from 2024), tackle scoring convention (solo + assist both = 1), props backfill S3 path

- Added page: nfl-rest-edge
- Source: Warren-Sharps-2026-Football-Preview.pdf (pages 23–40)
- Covers: rest edge definition, historical ROI stats (short-week road -9.4% ROI, rest edge road +7.0% ROI), full 2026 net rest table for all 32 teams, Bills/Chargers/Eagles specific flags, negated byes

- Updated page: nfl-2026-season-context (major expansion)
- Source: Warren-Sharps-2026-Football-Preview.pdf (pages 41–74)
- Added: positional unit rankings (all 32 teams), team rushing efficiency top/bottom, team offensive efficiency, QB efficiency tiers, win total O-U historical patterns (2000–2025), key injury notes (Mahomes, Kraft, Nabers, Kittle, LaPorta)

- Added team pages: nfl-team-ari-2026, nfl-team-atl-2026, nfl-team-bal-2026, nfl-team-buf-2026, nfl-team-car-2026
- Source: Warren-Sharps-2026-Football-Preview.pdf (pages 75–154)
- ARI: 4.5-win under lean, Mike LaFleur HC, Trey McBride TE record season, #32 QB, hardest SoS
- ATL: 7.5 wins, Bijan Robinson in Stefanski wide-zone, Penix ACL recovery, Front 7 #31, second-half collapse pattern
- BAL: 11.5 wins, Lamar Jackson hamstring history, Jesse Minter HC, designed runs win model (7+ runs = 80% W%), Lamar pre/post-injury split
- BUF: 10.5 wins, Josh Allen #1 QB, D.J. Moore fixes downfield threat, 3 short-week road games, Joe Brady HC (age 36)
- CAR: 7.5 wins under lean, Bryce Young #32 EPA/att on 4th easiest schedule, 87.5% of wins by 1 score, schedule jumps to 3rd hardest

- Added team page: nfl-team-chi-2026
- Source: Warren-Sharps-2026-Football-Preview.pdf (pages 160–178)
- CHI: 9.5 wins under lean, Ben Johnson Year 2, massive 2025 luck regression (#1 turnover margin, #1 fumble luck, Bears Tax), Colston Loveland breakout, Front 7 #30

- Added team page: nfl-team-cin-2026
- Source: Warren-Sharps-2026-Football-Preview.pdf (pages 179–194)
- CIN: 9.5 wins, Joe Burrow health binary (8 starts in 2025), Dexter Lawrence trade (#10 pick) defensive transformation, O-Line still #28 for 5th straight year, schedule improves to #3 easiest

- Added team pages: nfl-team-kc-2026, nfl-team-lv-2026, nfl-team-lac-2026, nfl-team-lar-2026, nfl-team-mia-2026
- Source: Warren-Sharps-2026-Football-Preview.pdf (pages 331–414)
- KC: 10.5 wins, Mahomes ACL+LCL recovery (cleared 2026), 1-9 one-score games (worst mark ever at .100), opponents made 96.9% of FGs (worst FG luck ever), Bieniemy back as OC, McDuffie+Watson CB duo both gone to LAR, Walker III adds run game, Weeks 1-4 #1 easiest schedule
- LV: 5.5 wins, Fernando Mendoza #1 overall pick (Indiana, 16-0, 0 RZ INTs), Tyler Linderbaum record C deal ($81M), Kubiak wide-zone system perfect for Jeanty+Bowers, WR room #29 remains biggest hole, building for future (DC Leonard installs Macdonald-style defense)
- LAC: 10.5 wins, Herbert OL health binary (Slater torn patellar 2025, Alt 6 games) — with Alt Herbert was MVP-caliber; McDaniel replaces Roman (Roman dropped PSM from #8 to #29, PA from #3 to #21 usage despite dramatically better results); net rest #32 worst since 2013 (-24 days), 7 rest-disadvantage games (franchise record); 6-2 one-score luck must regress
- LAR: 11.5 wins, Stafford MVP (4707 yds, 46 TDs, 23 sacks fewest in NFL, 28 consecutive TD passes without INT — NFL record), Myles Garrett trade (cost Verse + picks, gave NFL sack record holder), McDuffie+Watson CB duo acquired from KC, special teams fixed with Ventrone + Cardona, only team with all 7 units ranked top-10, Super Bowl in LA (SoFi Stadium)
- MIA: 4.5 wins, full teardown ($180M dead cap breaks NFL record, 60% of $301.2M salary cap to departed players), WR/Front7/Secondary all #32, Malik Willis bridge QB (0-6 when opponents score 15+, 11% sack rate, red zone disaster), De'Von Achane lone weapon (PFF #1 RB, 238 car 1350 yds behind #27 OL), schedule jumps to #2 toughest, rest drops from #2 best to #3 worst, 2027 QB draft positioning

- Added team pages: nfl-team-ind-2026, nfl-team-jac-2026
- Source: Warren-Sharps-2026-Football-Preview.pdf (pages 295–330)
- IND: 7.5 wins slight under lean, Daniel Jones torn Achilles (Week 14) + fractured fibula (Week 10 in Germany), historically elite through 10 weeks (#1 EPA/play, success rate, YPA, pts/drive), Sauce Gardner trade used 2026+2027 1st-round picks for 4 games, defense #28 early down allowed on #6 easiest offense schedule, Tyler Warren breakout rookie TE
- JAC: 9.5 wins under lean, 13-4 built on 4 non-offensive TDs in key wins + 3-0 FG margin games + backup QB parade + #5 easiest offense schedule, FIRST NFL team ever to go from ≤10 takeaways to 30+ in one season, ESPN Pain Index #32 worst since 2002 (11 road/neutral games, two London trips), Travis Etienne gone (RBs #29), Travis Hunter plays primarily CB in Year 2, Trevor Lawrence franchise record 38 TDs

- Added team pages: nfl-team-cle-2026, nfl-team-dal-2026, nfl-team-den-2026, nfl-team-det-2026, nfl-team-gb-2026, nfl-team-hou-2026
- Source: Warren-Sharps-2026-Football-Preview.pdf (pages 195–294)
- CLE: 6.5 wins, #448 of 448 OL stat (historically worst offense), Myles Garrett traded to Rams for Jared Verse + picks, QB catastrophe (Watson/Sanders/Gabriel), OL 6 new starters
- DAL: 9.5 wins, Receivers #1 (Lamb+Pickens), Parsons to GB for Kenny Clark + picks, DC Parker (34yo, never called plays), schedule #4 hardest (#1 largest increase in NFL)
- DEN: 9.5 wins, 14-3 was historic one-score luck (11-2 in one-score games), Bo Nix game manager with massive alignment tell, defense elite (#1 yards/play, 68 sacks), Webb new OC first time
- DET: 10.5 wins, #1 easiest schedule in NFL, Goff top-5 QB, lost Ben Johnson (OC→CHI) + Jahmyr Gibbs (→HOU), Pacheco replacement ranks 46th/47th in efficiency, both safeties health uncertain
- GB: 10.5 wins, Jordan Love #1 EPA when clean (0.56 no-pressure), OL collapsed #10→#27, Parsons on PUP (ACL), Josh Jacobs arrested (May 2026 felony), worst rest structure in NFL (#1 games off road SNF/MNF, #29 negated bye)
- HOU: 9.5 wins, Front 7 #1 + Secondary #1 (defense #1 total), Stroud +0.06 EPA/att but playoff disaster, FG luck +8.7 (#1 in NFL — unsustainable), red zone #30, OL still #31, David Montgomery perfect scheme fit

- Added team pages: nfl-team-min-2026, nfl-team-ne-2026, nfl-team-no-2026, nfl-team-nyg-2026
- Source: Warren-Sharps-2026-Football-Preview.pdf (pages 415–483)
- MIN: 8.5 wins, JJ McCarthy health binary (torn meniscus, missed 2025), Brian Flores #1 blitz rate (48%), 1-score record -5 worst in NFL, vs playoff teams 3-10 (.231), Jordan Love SB revenge plot
- NE: 9.5 wins, Vrabel Coach of Year, Maye elite (4,287 yds, 37 TDs, +0.20 EPA/att), OL #1 protection with only 23 sacks allowed, front 7 #1 defense, A.J. Brown traded in (Patriots paying Eagles), 4th-down conservatism is ceiling limit
- NO: 7.5 wins, Spencer Rattler or Shedeur Sanders QB battle, Staley transformed defense #30→#9 (80 yds/game improvement), Olave/Kamara weapons, #2 toughest schedule, 1-score luck was elite in 2025
- NYG: 7.5 wins, Harbaugh new HC (hired over Daboll firing), Malik Nabers WR1 elite breakout (108 catches, 1,406 yds rookie), DJ Moore + Okonkwo added, defense overhauled (5 new starters), 4th toughest SoS, 87.8% XP fixed with Stout+Sanders

- Added team pages: nfl-team-nyj-2026, nfl-team-phi-2026, nfl-team-pit-2026, nfl-team-sf-2026, nfl-team-sea-2026
- Source: Warren-Sharps-2026-Football-Preview.pdf (pages 484–568)
- NYJ: 5.5 wins, HC #32 Aaron Glenn, first team in NFL history with ZERO INTs in 2025, -19 turnover margin, David Bailey #2 overall pick (Texas Tech, 33.6% pressure rate), sold Gardner→Colts (2 1sts) and Williams→Cowboys, Geno Smith bridge, rebuilding for 2027
- PHI: 10.5 wins, Hurts 2-High coverage dead last (#32 in NFL), A.J. Brown traded to Patriots, OL #2 + Front 7 #2 + Mitchell/DeJean both 1st-team All-Pro, net rest #31 (2nd worst, -22 days first half), Mannion (Shanahan tree) hired to fix dual-threat underuse
- PIT: 8.5 wins, Tomlin resigned (19 seasons, 7 straight playoff losses), McCarthy new HC, Rodgers 42 ranked #36 time-to-throw/#36 air yards/#38 deep pass%, turnover luck +12 (#4) is highly regressive, facing #3 toughest pass defense schedule
- SF: 10.5 wins, Purdy +0.17 EPA/att (#4) but play-action collapsed #32 (was #7); CMC 75 RZ rushes at age 30; Bosa+Williams+Warner all returning from injuries; FG luck +8.6 (#2), 1-score 5-1 (.833, #2) — both regressive; Lynch draft capital efficiency #32 4th straight year
- SEA: 10.5 wins, 2025 Super Bowl LX champions (14-3), Klint Kubiak departed (→Raiders HC), Darnold 28 turnovers most of any player in NFL but Inside RZ -0.15 (#37 worst), JSN Offensive Player of Year + 1st-team All-Pro ($168.6M extension), RBs #31 (Walker III gone + Charbonnet ACL)

- Added team pages: nfl-team-tb-2026, nfl-team-ten-2026, nfl-team-was-2026
- Source: Warren-Sharps-2026-Football-Preview.pdf (pages 569–619)
- TB: 8.5 wins (NFC South #1), Bowles HC on hot seat + Mayfield contract year, Robinson new OC (McVay disciple, 9th different OC for Mayfield), Mayfield 4th-quarter EPA #38 worst in NFL, Egbuka clear WR1 after Evans departure, OL #4 when healthy (all starters missed games in 2025), all 2025 luck factors regressive (turnover +7, penalty +21 EPA, fumble luck +3.5)
- TEN: 6.5 wins, Saleh (HC) + Daboll (OC) both former HCs — two-HC model; Daboll applying Josh Allen blueprint to Cam Ward; schedule flips #1 toughest→#13 easiest; rest rank #1 best in NFL with zero rest-disadvantage games all season; OL still #30; ward took league-high 55 sacks with worst receiver room in NFL
- WAS: 7.5 wins, Daniels lost season (healthy only Week 1; knee→hamstring→dislocated elbow Week 9 SEA game); when healthy: deep ball #2 (0.92 EPA/att), 2-High #3; defense historically bad (worst first-quarter punt rate this century, 384 yards/game allowed); Kingsbury fired, Blough (Campbell/Johnson tree) brings under center + PA + motion; Sonny Styles #7 pick anchors rebuilt defense; DL spending #1 ($106.86M)

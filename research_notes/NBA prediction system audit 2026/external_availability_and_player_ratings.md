# External Inputs for 2026-27 NBA Game Prediction: Player Availability (Injury) Data and Public Player Impact Ratings

_Compiled 2026-10-25. The 2026-27 regular season began 2026-10-20. Pages were fetched on 2026-10-25 unless noted. "(search summary)" marks a fact seen only in a search-engine result summary, not on a page I opened. Treat those as weaker. This is about predictive inputs only, not betting. Terms of use are noted for each source because several sources explicitly forbid gambling-related or commercial use._

## 1. The official NBA injury report: where it is published, when, which statuses, the reporting rules (2023 Player Participation Policy through 2026-27), and how reliable "Questionable" is

### Takeaway
The league publishes the injury report as timestamped PDFs. They are linked from official.nba.com and hosted on ak-static.cms.nba.com, and since late December 2025 they refresh every 15 minutes (hourly before that). Teams must file by 5 p.m. local time the day before a game, or by 1 p.m. on game day for the second leg of a back-to-back. Since the December 2025 memo they must also re-file on game day (11 a.m.–1 p.m. local, or 8–10 a.m. for tip-offs at 5 p.m. or earlier).

The statuses are Out, Doubtful, Questionable, Probable and Available. The only numeric meanings I found come from team pages: 75% for Probable, 50% for Questionable, 25% for Doubtful. In practice, a player first listed as "Questionable" played about 48% of the time in January–October 2026, but the rate varied widely by team (about 22% to 68%). Probable players played about 87% of the time and Doubtful players about 4%. So the nominal keys are badly miscalibrated at the extremes.

### Cited Findings

#### Where and in what format
- The current-season page is "NBA Official – 2026-27 Injury Report". It says reports are "updated on a continual basis throughout the day" and links one PDF per timestamp. On 2026-10-25 it listed reports every 15 minutes (12:00 a.m., 12:15 a.m. … 10:15 a.m. ET at the time of the fetch) — [NBA Official, 2026-27 Injury Report page](https://official.nba.com/nba-injury-report-2026-27-season/)
- The current PDF URL pattern is `https://ak-static.cms.nba.com/referee/injury/Injury-Report_YYYY-MM-DD_HH_MM[AM|PM].pdf`, for example `Injury-Report_2026-10-25_12_00AM.pdf`. Times are ET — [NBA Official, 2026-27 Injury Report page](https://official.nba.com/nba-injury-report-2026-27-season/)
- The NBA moved from hourly to 15-minute file names on **2025-12-22**. The old format was `Injury-Report_YYYY-MM-DD_HHPM.pdf` and the new one adds a minute field (`..._HH_MMPM.pdf`). A scraper has to generate both formats — [kyyamakaimotoyk/NBAStatsProject README](https://github.com/kyyamakaimotoyk/NBAStatsProject)
- The search index still holds old-format PDFs going back years, e.g. `Injury-Report_2019-01-01_01PM.pdf`, `Injury-Report_2021-04-23_01PM.pdf` and `Injury-Report_2023-01-01_06PM.pdf`. They were seen in search results but not opened — [NBA CDN (2019 example)](https://ak-static.cms.nba.com/referee/injury/Injury-Report_2019-01-01_01PM.pdf); [NBA CDN (2023 example)](https://ak-static.cms.nba.com/referee/injury/Injury-Report_2023-01-01_06PM.pdf)
- One parser project describes three report eras: "three-a-day 2018-21, hourly 2021-26, 15-minute since 2026" — [PirateKingInc/test-m11 issue #64](https://github.com/PirateKingInc/test-m11/issues/64). This conflicts slightly on the start of the 15-minute era: the README above (2025-12-22) and ESPN's December 2025 report put the switch in late December 2025.
- Third-party parsers extract these columns: Game Date, Game Time, Matchup, Team, Player Name, Current Status, Reason — [nbainjuries (PyPI)](https://pypi.org/project/nbainjuries/); [nba-injury-report (PyPI)](https://pypi.org/project/nba-injury-report/)
- The status values seen in parsed reports are **Out, Doubtful, Questionable, Probable, Available** — [kyyamakaimotoyk/NBAStatsProject README](https://github.com/kyyamakaimotoyk/NBAStatsProject); [PirateKingInc/test-m11 issue #64](https://github.com/PirateKingInc/test-m11/issues/64)
- The league's 2026-27 rules text says only that teams must "designate a participation status." It gives no probabilities — [NBA Official, 2026-27 Injury Report page](https://official.nba.com/nba-injury-report-2026-27-season/)
- Team-published participation keys, such as the Sacramento Kings' injury-report posts on NBA.com, define Probable = 75%, Questionable = 50%, Doubtful = 25%, and Out = will not play. These are older Kings posts (search summary) — [Kings injury report blog, NBA.com](https://www.nba.com/kings/blog/injury-kings-clippers)

#### Reporting deadlines and cadence in force for 2026-27
- **Day before:** for most games, teams must submit injury, illness, medical-condition or rest information by **5 p.m. local time the day before**. For the second game of a back-to-back the deadline is **1 p.m. local on game day** — [NBA Official, 2026-27 Injury Report page](https://official.nba.com/nba-injury-report-2026-27-season/)
- **Game-day re-filing:** teams must also submit a game-day report **between 11 a.m. and 1 p.m. local**, or **8–10 a.m.** for tip-offs at 5 p.m. or earlier. This does not apply to the second game of a back-to-back — [NBA Official, 2026-27 Injury Report page](https://official.nba.com/nba-injury-report-2026-27-season/)
- **Where these rules came from:** a league memo reported on **2025-12-19**. It required the game-day re-filing and said public reports would be updated **every 15 minutes, replacing hourly updates**. The memo gave no effective date.
  - The memo also covered advocacy for limits on player prop bets, a review of draft-pick protections and the lottery to counter tanking, gambling education, protection of team personnel from bettor harassment, and stronger investigation of unusual betting.
  - It followed federal indictments in October 2025 of a current player, a head coach and a former player.
  - [ESPN, David Purdum, 2025-12-19](https://www.espn.com/nba/story/_/id/47361909/nba-outlines-injury-info-prop-betting-changes-memo-teams)
- Prosecutors accused Terry Rozier and Damon Jones of supplying bettors with nonpublic injury information. The league said the reporting changes aim to reduce the value of that kind of information (search summary) — [theScore](https://www.thescore.com/nba/news/3432280/nba-tweaks-injury-reporting-rules-still-wants-changes-to-prop-betting); [Legal Sports Report](https://www.legalsportsreport.com/249720/nba-betting-scandals-prompt-injury-reporting-overhaul/)
- One third party described the pre-change cadence as "5 PM ET day-before, then hourly through game day, then a 'final' report ~30 min before tipoff." This is the project's description, not league text — [kyyamakaimotoyk/NBAStatsProject README](https://github.com/kyyamakaimotoyk/NBAStatsProject)

#### Player Participation Policy (PPP) and later changes
- **Adoption:** the Board of Governors approved the PPP unanimously on **2023-09-13**, effective for 2023-24 — [ESPN, Adrian Wojnarowski, 2023-09-13](https://africa.espn.com/nba/story/_/id/38392750/source-nba-board-governors-approve-tougher-rest-rule)
  - **Who counts as a star:** an All-Star or All-NBA selection in any of the prior three seasons, which was about 50 players on 25 teams at the time.
  - **Core rules:**
    - No more than one star may sit for the same game, unless the stars are injured.
    - Stars must be available for national-TV and In-Season Tournament games.
    - Teams must avoid long-term shutdowns, or materially reduced roles, that affect the integrity of the game.
    - One-game absences must be balanced between home and road games, with a preference for home.
    - Healthy players who rest must be present and visible to fans.
  - **Back-to-back exceptions:** players can be pre-approved if they are 35 or older on opening night, or have 34,000 regular-season minutes or 1,000 combined regular-season and playoff games. Teams must give written reasons one week in advance.
  - **Other exceptions:** bona fide injuries, personal reasons, rare circumstances, roster management and end-of-season flexibility.
  - **Fines:** $100,000 for a first violation, $250,000 for a second, then $1 million more than the previous fine each time. Enforcement relies on league-office investigations that include independent medical reviews.
  - The CBA's 65-game minimum for award eligibility is mentioned as background, not as part of the PPP.
- **Enforcement in 2024-25:** the Utah Jazz were fined **$100,000** for resting Lauri Markkanen against Washington on 2025-03-05. The league was reportedly investigating the Thunder and the 76ers. The policy lets the league "investigate and/or impose discipline in other circumstances involving star player (or other player) non-participation" — [Front Office Sports, 2025-03-18](https://frontofficesports.com/article/as-jazz-wizards-race-to-bottom-nba-could-expand-fine-latitude/)
- **Fines in 2025-26:** in February 2026 the NBA fined the **Jazz $500,000** and the **Pacers $100,000** for "tanking-related conduct detrimental to the league" — [theScore](https://www.thescore.com/nba/news/3546790/amp). Another outlet describes these as violations of the participation policy (search summary) — [Yahoo Sports](https://sports.yahoo.com/articles/nba-approves-major-anti-tanking-224231480.html)
- **Anti-tanking package for 2026-27:** approved in late May 2026 by a 29-1 vote, with Memphis dissenting (ESPN, search summary). It takes effect "next season" (2026-27) for three seasons — [theScore](https://www.thescore.com/nba/news/3546790/amp); [ESPN](https://www.espn.com/nba/story/_/id/48886245/the-pros-cons-new-nba-draft-lottery-reform)
  - A "3-2-1" lottery expanded from 14 to 16 teams, in which the three worst teams get only a 5.4% chance at the No. 1 pick.
  - The commissioner can fine tanking teams, reportedly up to $10 million, reduce lottery odds and change draft positions.
- I found **no published change to the injury-report statuses or to the PPP text for 2026-27** beyond the December 2025 memo and the anti-tanking package. Coverage of the 2026-27 rule changes covers the NBA Cup bracket format and officiating points of emphasis (illegal screens, illegal defensive contact), with no injury-report changes (search summary) — [Bleacher Report, 2026-27 rule changes](https://bleacherreport.com/articles/25513499-explaining-nba-rule-changes-2026-27-season-including-nba-cup-bracket-format)

#### How reliable is "Questionable"? (play rate by first-listed status)
- **Play rate by status, 2026-01-18 to 2026-10-22,** from official injury-report PDFs scraped every 15 minutes and matched to NBA.com box scores. "Played" means the player logged minutes, and the status is the **first** status the report gave that player for that game — [IsHePlayingTonight, "NBA questionable play rate"](https://isheplayingtonight.com/nba-injuries/questionable-play-rate)
  - **Questionable:** 48.3% (660 of 1,367)
  - **Doubtful:** 4.2% (12 of 289)
  - **Probable:** 86.6% (304 of 351)
  - Total sample: 2,007 designations.
- **Team variation for Questionable** — same source:
  - Toronto 67.6% (25/37), Minnesota 64.9%, Indiana 62.1%
  - New York 37.8% (31/82), Atlanta 33.9% (20/59), Sacramento 21.7% (5/23)
- **Questionable by injury type** (minimum 10 tags) — same source:
  - Ankle 58.7% (167 tags), Illness 53.6% (224), Knee 52.5% (80)
  - Back 46.7% (75), Calf 44.6% (65), "not specified" 43.2% (570)
- The page updates live. An earlier search-index snapshot of the same page showed 48.6% (655 of 1,348) for Questionable — [IsHePlayingTonight](https://isheplayingtonight.com/nba-injuries/questionable-play-rate)
- Older figures, such as "about 55%" and "73% in 2016", appear on explainer sites without traceable data. They are unverified and low quality — [DNA of Sports](https://www.dnaofsports.com/basketball/what-does-questionable-mean-in-basketball/)

### Inferences
- **Treat status as a probability and calibrate it yourself.** At first listing, Questionable is close to its nominal 50%, but Probable (about 87% versus a nominal 75%) and Doubtful (about 4% versus 25%) are far off. A hierarchical calibration (league → team → player, plus injury type and hours to tip) is justified by the 22–68% spread across teams.
- **Prefer the last report before a fixed cutoff over the first listing.** Statuses get sharper through game day (15-minute cadence, game-day re-filing), so the latest snapshot before the cutoff should be much more informative than the first. The published 48% figure is a lower bound on how informative the report can be.
- **Rest is its own absence type.** The PPP makes rest absences of stars less likely in national-TV games and when another star is already out. Model rest separately from injury, using the star list (All-Star or All-NBA in the last three seasons).
- **Late-season priors from 2023–26 may overstate absences for bottom teams in 2026-27.** The 2026-27 anti-tanking package (flatter lottery, fines up to about $10M) should reduce late-season "shutdowns" of healthy players on those teams. This is an inference with no data yet.
- **Feature definitions must survive the cadence changes.** Because cadence changed (three-a-day, then hourly, then every 15 minutes), define features by clock cutoff (e.g., the latest report at or before T−60 min), not by "report number k of the day".

### Gaps
- Both attempts to open an injury-report PDF timed out. So I could not verify:
  - the exact "Reason" taxonomy (e.g., "Injury/Illness – …", "G League – Two-Way", "Rest");
  - how teams that have not yet submitted are shown;
  - whether a separate "final" pre-tip report exists in the 15-minute era.
- I found no league document that defines numeric probabilities for the statuses. The 75/50/25 key comes from team pages.
- The IsHePlayingTonight methodology has gaps: no operator is named, there is no independent replication, and the window is about 9 months. Results for the final pre-tip status were not published.
- No exact effective date was published for the 15-minute cadence or the game-day re-filing. The date is inferred from the 2025-12-22 file-name change.
- I did not verify whether teams also file a separate pre-game inactive list under the 2023 CBA, or its timing.

## 2. Public historical archives of injury reports and inactive lists for leak-free model training: what exists, how far back, and coverage

### Takeaway
Timestamped official PDFs sit on the NBA CDN back to at least the 2018-19 season (about December 2018) and can be collected by URL pattern or through Internet Archive indexes. Maintained Python parsers cover **2021-22 onward**: nbainjuries (MIT) and nba-injury-report (AGPL). Several GitHub pipelines store every snapshot with its publish time, which is exactly what leak-free training needs.

Before about 2018-19 there are no pre-game reports. You have to fall back on Pro Sports Transactions injury-list logs (a Kaggle copy covers 2010–2020) or post-game inactive lists from nba_api. Neither is a true pre-game snapshot. I found no Kaggle dataset of the official 2021–2025 reports.

### Cited Findings
- **nbainjuries (PyPI)** — [nbainjuries on PyPI](https://pypi.org/project/nbainjuries/)
  - Retrieves official team injury reports "stored as static files on the NBA's server in hourly or 15-minute snapshots."
  - Coverage starts with **2021-22**. Preseason is excluded, there are gaps around the All-Star break and the postseason, and pre-2021-22 coverage is listed as a future goal.
  - Output fields: Game Date, Game Time, Matchup, Team, Player Name, Current Status, Reason, returned as JSON or a pandas DataFrame.
  - Needs Java 8+ because it uses tabula-py. MIT license.
  - Version 1.1.1 was released 2026-02-10 (1.0.0 on 2025-08-04). The author asks users to respect the NBA's Terms of Use.
- **nba-injury-report (PyPI)** — [nba-injury-report on PyPI](https://pypi.org/project/nba-injury-report/)
  - Fetches official reports "from the NBA's public API" and covers pre-game reports from **2021-22**. The same seven fields are exported to JSON, CSV or DataFrame.
  - AGPL-3.0, Python 3.12+. Version 0.1.0 (2026-01-07) is the only release.
- **nba-injury-report-pdf-to-df (PyPI)** converts local report PDFs to a DataFrame. MIT license. It is built for the January 2025 PDF layout, so older PDFs may not parse (search summary) — [nba-injury-report-pdf-to-df on PyPI](https://pypi.org/project/nba-injury-report-pdf-to-df/)
- **kyyamakaimotoyk/NBAStatsProject** wraps nbainjuries — [kyyamakaimotoyk/NBAStatsProject README](https://github.com/kyyamakaimotoyk/NBAStatsProject)
  - As of 2026-05-19, its historical table held **106,755 rows over 969 game-dates (2022-01-01 to 2026-05-13)**, with about 71 game-dates missing (All-Star breaks, off-days).
  - It keeps only the report **closest to tip-off** per game date. The unique key (game_date, team, player) means intraday history is not kept.
  - It "must run from a residential IP." Player and team name resolution is about 97%.
- **mrees002/nba-injury-database** — [mrees002/nba-injury-database](https://github.com/mrees002/nba-injury-database)
  - Discovers historical official PDF URLs through the **Internet Archive CDX index** and downloads them from ak-static.cms.nba.com.
  - Stores the PDF bytes, SHA-256 hashes, raw text, row text, source URLs and parser versions in PostgreSQL.
  - Defaults to 1 request per second. Pro Sports Transactions data is kept only for validation. The example backfill starts on 2018-12-20.
  - The project says raw archive ingestion is "still being completed and validated."
- **bryanhkwan/Basketball-Injury-Project** downloads one PDF per game day from **2018-12-17 through 2025-26** (search summary) — [bryanhkwan/Basketball-Injury-Project](https://github.com/bryanhkwan/Basketball-Injury-Project)
- **PirateKingInc/test-m11** designed a parser for all three eras that stores **every version with its publish time** ("change-only per game and player"). It backfills "every historical report reachable" at about 1 request per second and runs an hourly job for the latest report. The design is in an issue completed by PR #82 — [PirateKingInc/test-m11 issue #64](https://github.com/PirateKingInc/test-m11/issues/64)
- A GitHub release labeled "nba-official-reports" claims to hold the official PDFs selected as "the latest report published 60+ minutes before each regular-season tip-off." The release page showed no files, so its contents are unconfirmed (search summary) — [niksmdmarketing/NFLFutures release](https://github.com/niksmdmarketing/NFLFutures/releases/tag/nba-official-reports)
- **Viktorsadowski/nba-moneyball** — [Viktorsadowski/nba-moneyball](https://github.com/Viktorsadowski/nba-moneyball)
  - `injury_reports.py` downloads and parses the official PDFs for **2021-22 onward**. A pre-scraped Pro Sports Transactions injury log covers **2010-11 to 2019-20**.
  - The README notes "stats.nba.com blocks most cloud IPs." No license is stated.
- **StatSurge (Substack)** offered free CSVs of "All Injury Reports: Oct 2021 to June 2024", plus per-season files for 2021-22 through 2023-24. It is **no longer maintained** as of 2025-26 (December 2025 notice), and no dataset license is stated — [StatSurge, "Downloadable NBA Injury Datasets" (2024-11-16)](https://statsurge.substack.com/p/downloadable-nba-injury-datasets)
  - The build used the official PDFs and daily **2 p.m.** reports, with ">35,000 injury designations" over three seasons — [StatSurge, "Creating an NBA Injury Database" (2024-11-17)](https://statsurge.substack.com/p/creating-an-nba-injury-database)
- **Kaggle (ghopkins), "NBA injuries 2010–2020"** covers 2010-11 through 2019-20. It has about 27,105 rows in 5 columns (date, team, acquired, relinquished, notes), scraped from prosportstransactions.com. This is a transaction log, not the official report. The mirror's URL says "2010_2018" while the content says 2010–2020 (search summary) — [Baselight mirror of Kaggle dataset](https://baselight.app/u/kaggle/dataset/ghopkins_nba_injuries_2010_2018)
- An academic paper that used Pro Sports Transactions warned that its injury terminology is inconsistent and that the timing and thoroughness of its reporting are uncertain (search summary) — [Journal of Sports Analytics](https://content.iospress.com/articles/journal-of-sports-analytics/jsa200529)
- **Post-game inactive lists:** the stats.nba.com `BoxScoreSummaryV2` endpoint, used by nba_api and hoopR, returns an `InactivePlayers` table (player ID, name, jersey, team) alongside GameSummary, LineScore and other tables — [hoopR docs, nba_boxscoresummaryv2](https://search.r-project.org/CRAN/refmans/hoopR/html/nba_boxscoresummaryv2.html). nba-db maps this endpoint to an `inactive_players` table — [nba-db docs](https://nba-db.readthedocs.io/user_guide/endpoints.html). I found no V3 equivalent.
- **NBA.com Terms of Use (last updated 2026-07-13)** — [NBA.com Terms of Use](https://www.nba.com/terms)
  - Downloads are allowed "to any single computer only for your personal, noncommercial use."
  - NBA Statistics require prominent attribution and may be used only for "legitimate news reporting or private, non-commercial purposes."
  - They may not be used "in connection with any gambling activity (including legal gambling activity)", with "any fantasy game or other commercial product or service", or with any "live, near-live" product.
  - They may not be used with a website, product or service that "features a database" of regularly updated statistics without prior consent.
  - The Terms do not mention injury reports specifically or contain an explicit anti-robot clause.

### Inferences
- **The leak-free recipe** is an append-only table of (game_id, team, player, status, reason, report_timestamp_ET) built from the official PDFs. For each game, take the latest report at or before a fixed cutoff, such as tip−60 or tip−30 minutes.
  - 2021-22 onward is easy with nbainjuries or nba-injury-report.
  - 2018-19 to 2020-21 needs custom parsing of the three-a-day era (see the mrees002 and PirateKingInc approaches). The COVID-disrupted 2019-20 and 2020-21 seasons will have irregular schedules.
- **Do not reuse kyyamakaimotoyk-style tables for cutoff features.** They keep only the snapshot closest to tip and drop the rest, so you cannot reconstruct a fixed cutoff from them. Store all snapshots instead.
- **Post-game sources are labels, not pre-game features.** Inactive lists and box-score minutes tell you who played, which is the right target for an availability-probability model. But they encode the final decision.
- **Pre-2018 availability can only come from post-game data or transaction logs.** Any pre-2018 model that uses "availability" from those sources is using hindsight. Either restrict injury-aware training to 2018-19+ (or 2021-22+), or simulate pre-game uncertainty.
- **Plan for missing snapshots.** Gaps are real: nbainjuries documents periodic gaps (All-Star break, postseason), and Internet Archive discovery is incomplete. Keep explicit missingness flags per game. The roughly 71 "missing" dates in kyyamakaimotoyk's table were attributed to All-Star breaks and off-days, so they are mostly not data loss.
- **Scrapers may fail from CI or cloud machines** because NBA endpoints block cloud IPs. Plan local or residential runs.
- **Licensing:** a private, non-commercial model with no betting use fits NBA.com's "private, non-commercial purposes" clause. Publishing a regularly updated public database built from NBA statistics would need the NBA's consent.

### Gaps
- I did not verify the exact earliest date of official PDFs on the CDN. About December 2018 is inferred from GitHub projects, and Internet Archive coverage of individual timestamps is unknown.
- I found no Kaggle dataset of official 2021–2025 reports. Licenses for most of the GitHub archives are unstated.
- I did not verify how complete Basketball-Reference box-score "Inactive" lists are, or how far back they go (Sports Reference blocked my fetch).
- I could not confirm whether BoxScoreSummaryV2 is still populated for 2025-26 and 2026-27 games, or whether a V3 summary carries inactive players.

## 3. Other availability sources (confirmed lineups, ESPN, Rotowire, other feeds and APIs): latency, terms, free vs paid

### Takeaway
The official PDF is the authoritative free source, at a 15-minute cadence. Confirmed starters arrive only close to tip-off; one lineup service says up to about 30 minutes before. Teams and beat writers supply them, and Rotowire, RotoBaller and FanDuel Research aggregate them.

ESPN's injury JSON is undocumented, and ESPN's (Disney's) terms forbid scraping. Paid APIs (BALLDONTLIE, SportsDataIO, Sportradar) include injuries but have licensing limits. Most free routes either forbid automated collection or forbid gambling or commercial use.

### Cited Findings
- **Rotowire lineups** (search summary of the lineup pages) — [Rotowire NBA lineups](https://web.rotowire.com/daily/nba/nba_adv_lineups.php)
  - Lineups "are only confirmed when an official NBA source provides that information."
  - "The NBA does not require that starting lineups be submitted before tipoff."
  - Expected lineups are posted early each morning and "tweaked throughout the day."
  - The page fetch failed with a TLS error, so I could not check Rotowire's terms.
- **RotoBaller** marks starters as Confirmed only when they come "straight from the team or a trusted beat writer". Projected starters are built from each team's most recent starting five plus injury news (search summary) — [RotoBaller, Today's NBA Starting Lineups](https://www.rotoballer.com/todays-nba-starting-lineups-daily-basketball-starters)
- **FanDuel Research / numberFire** say lineups aren't confirmed until up to about 30 minutes before tip-off (search summary; 2024 articles) — [FanDuel Research](https://www.fanduel.com/research/today-s-nba-lineups-and-notes-tuesday-10-22-24)
- **Dunks & Threes** updates its game predictions more often in the 30 minutes before tip-off, "using official player status and context-aware minutes" — [Dunks & Threes, EPM methodology](https://dunksandthrees.com/about/epm)
- **ESPN "hidden" API** — [Zuplo, ESPN hidden API guide](https://zuplo.com/learning-center/espn-hidden-api-guide)
  - It is a set of undocumented endpoints reverse-engineered from ESPN's web and mobile apps. GET requests return JSON with no key, and there is "no guarantee of long-term stability or support." The site.api.espn.com domain serves general site data.
  - I could not confirm an exact NBA injuries URL. Third-party Apify actors (paid, hosted scrapers) say they use "ESPN's public injuries feed" and return athlete, team, status and comments — [Apify actor wyle/sports-injury-reports-api](https://apify.com/wyle/sports-injury-reports-api)
- **Disney Terms of Use (covers ESPN; last updated 2024-05-24)** — [Disney Terms of Use](https://disneytermsofuse.com/english/)
  - They prohibit accessing, monitoring, copying or extracting the products "using a robot, spider, script, or other automated means."
  - They also prohibit "data mining or web scraping," including for building AI tools, and any "commercial or business-related use."
- **BALLDONTLIE** — [BALLDONTLIE](https://balldontlie.io); [Highlightly, Best NBA APIs in 2026](https://highlightly.net/blogs/best-nba-apis-in-2026)
  - Injury data is not on the free tier, which is capped at 5 requests per minute.
  - Injuries require the ALL-STAR plan, listed at $9.99/month by one review. The vendor page lists an ALL-ACCESS plan at $89.99/month for every endpoint across all sports.
- **SportsDataIO** — [SportsDataIO data rights & licensing](https://sportsdata.io/help/data-rights-and-licensing-questions); [SportsAPIs.dev review](https://sportsapis.dev/apis/sportsdata-io)
  - The self-serve developer tier costs $99/month or $599/year (review site). Its data is "delayed by one day," it is "not suitable for live applications," and it is not licensed for commercial redistribution.
  - Commercial licenses are quoted per client by league, feeds and use case.
- **Sportradar's** NBA API includes injury reports, but no public price was found — [Highlightly, Best NBA APIs in 2026](https://highlightly.net/blogs/best-nba-apis-in-2026)
- **IsHePlayingTonight** itself scrapes the official PDFs every 15 minutes. No API or terms were seen — [IsHePlayingTonight](https://isheplayingtonight.com/nba-injuries/questionable-play-rate)
- **Sports Reference (Basketball-Reference)** — [Sports Reference bot traffic policy](https://www.sports-reference.com/bot-traffic.html) (page last updated 2024-05-29 per search summary; my direct fetch of the data-use page returned 403)
  - It blocks clients making more than **20 requests per minute** (10 for FBref and Stathead) and can "jail" sessions for up to a day.
  - It says it cannot offer bulk downloads because "most of our data comes from third parties who sell the data to us."

### Inferences
- **Latency ranking:** the official PDF (15-minute refresh, plus game-day re-filing windows) comes first. News aggregators follow within minutes of team announcements. Confirmed starters arrive about 30 minutes before tip.
- **What to use for a fixed cutoff:** for a model frozen at T−60 minutes, the official report is sufficient. For a model run at about T−15 minutes, confirmed starters mainly improve the **minutes allocation** among available players, not the availability set itself.
- **The cleanest free source is the official PDF**, used privately and non-commercially with attribution. ESPN scraping conflicts with Disney's terms.
- **Choosing a paid feed:** prefer a vendor whose license explicitly allows internal predictive modeling. SportsDataIO's cheap tier is one day delayed, so it is useless pre-game.

### Gaps
- I did not retrieve Rotowire's terms of use or data-feed pricing (TLS error).
- I did not confirm NBA.com's own "Today's Lineups" page or any JSON lineup endpoint.
- I found no measured study comparing the latency of these sources.
- I did not check Rotowire's claim that the NBA does not require pre-tip lineup submission against the NBA rulebook.

## 4. How established models turn availability into team strength, and how much accounting for injuries improves accuracy

### Takeaway
The standard method is: each player's per-100 rating (EPM, RAPTOR, DPM, …) × that player's projected minutes, with injury-aware minutes redistributed to teammates by context. Unknown and low-minute players are filled at replacement level, about −2.0 to −2.7 per 100.

The best public quantification comes from Dunks & Threes. Its injury-aware, player-based predictions beat a team net-rating baseline by 0.3–0.4 points of absolute margin error and 0.9–1.5 percentage points in picking winners. That gain mixes better ratings with availability, so the pure availability effect is not isolated.

Oddsmakers implicitly price star absences at about 2–5.5 points. I found no peer-reviewed study that isolates the accuracy gain from injury data, and one small study found that a salary-weighted absence variable did not help.

### Cited Findings
- **FiveThirtyEight's player-based model (RAPTOR era)** — [FiveThirtyEight, "How our NBA predictions work"](https://fivethirtyeight.com/methodology/how-our-nba-predictions-work/); [FiveThirtyEight 2020 team page example](https://projects.fivethirtyeight.com/2020-nba-predictions/lakers). These are search summaries: the methodology URL now returns HTTP 301 to abcnews.com/politics, and the archive was not reachable by my fetch tool.
  - Team ratings combined each player's RAPTOR with **playing-time forecasts covering each team's expected rotation and injuries**.
  - Team pages showed a "current" rotation, accounting for recent injuries, alongside a "full-strength" rotation.
  - Forecasts updated after every game and every depth-chart revision. In-season RAPTOR was phased in between 100 and 1,000 minutes (750 in the playoffs).
- FiveThirtyEight sports forecasts stopped updating on **2023-06-13**: "As of June 13, 2023, sports predictions and forecasts are no longer being updated" — [fivethirtyeight/data repository](https://github.com/fivethirtyeight/data)
- **Neil Paine's 2024-25 forecast** is "simpler" than the former FiveThirtyEight model and cannot adjust immediately to trades or injuries (search summary) — [Neil Paine, 2024-25 NBA Forecast](https://neilpaine.substack.com/p/2024-25-nba-forecast)
- **Dunks & Threes (EPM)** — [Dunks & Threes, EPM methodology (figures as of Oct 2026)](https://dunksandthrees.com/about/epm)
  - **Minutes model:** minutes are "predicted for every player in every game and updated frequently as tip-off approaches, using known injuries and official player status". When a player is out, "game context informs how their minutes shift to teammates". Minutes predictions beat season-to-date averages by about 14.5–15%.
  - **Ratings:** team ratings include a next-game rating, "the expected lineup for the team's next game, including known injuries".
  - **Live validation, 2024-25 to 2025-26 (2,643 games):** average absolute miss on the final margin was **11.0 points vs 11.4** for the baseline, and winners were picked **68.4% vs 67.5%** of the time.
  - **Backtest, 2005-06 to 2023-24 (24,425 games):** **9.6 vs 9.9** points, and winners **67.8% vs 66.3%**.
  - **Baseline:** "each team's opponent-adjusted net rating going into the game", starting from a preseason projection that fades.
- **Replacement-level convention** in Dunks & Threes' retrodiction: players under 250 minutes in the source season, and rookies, were given replacement values. These were **−2.0** for plus-minus metrics, 11 for PER and 0.045 for WS/48. Team ratings were weighted by actual target-season minutes, which the author notes is unrealistic for real forecasting but fair for comparing metrics — [Dunks & Threes, "NBA Player Metric Comparison" (2020-06-17)](https://dunksandthrees.com/blog/metric-comparison)
- **LEBRON** uses "0 is average and −2.7 is replacement level" — [BBall Index, LEBRON introduction](https://www.bball-index.com/lebron-introduction/)
- **nba-moneyball** — [Viktorsadowski/nba-moneyball](https://github.com/Viktorsadowski/nba-moneyball)
  - The previous season's player values (box-prior RAPM) predict team wins with **r = 0.79**.
  - Injury proneness predicts future absences: the top fifth by injury history misses 21 games the next season, the bottom fifth 7.
  - Players who return are "as good as before", except after an ACL injury, where "the first 20 games back are about 0.5 to 0.8 points per 100 worse."
- **Market-implied value of absences (2026 playoffs, from oddsmakers):** this is context for effect sizes, not betting advice — [theScore, "How much are stars worth to their teams?"](https://www.thescore.com/news/3531319)
  - Jokić 5.5 points, Gilgeous-Alexander 4, Wembanyama 4, Dončić 3.5.
  - Most other stars 2–3 points. Role players such as Reaves, Jalen Williams and Aaron Gordon about ≤1 point.
  - Value falls when the team has a capable backup, e.g. the Knicks without Brunson compared with the Blazers without Avdija.
- **Counter-evidence (low confidence):** one study built a model on team strength, home court and back-to-back fatigue, then tested player absences weighted by salary. The model without the absences was the most accurate for score difference. This comes from a search summary only; the full text returned 403 — [ACM, "Modeling and Predicting the Outcomes of NBA Basketball Games"](https://dl.acm.org/doi/10.1145/3501774.3501788)
- **Academic gap:** a 2025 systematic review of 34 AI game-prediction studies (29 on the NBA) found that "few considered unstructured or contextual data such as player injuries." It reported no comparison of models with and without injury data — [PLOS ONE systematic review, 2025-06-26](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0326326)
- **Season-level benchmark (APBRmetrics 2025-26 team-win contest, final 2026-04-13)** — [APBRmetrics 2025-26 contest thread](https://apbr.org/metrics/viewtopic.php?p=44892)

  | Entry | MAE (wins) | RMSE | r² |
  |---|---|---|---|
  | BetMGM line | 7.39 | 9.41 | .53 |
  | Ebet (betting line) | 7.40 | 9.58 | .51 |
  | BPM-based wins (bpmW) | 8.23 | 10.24 | .43 |
  | DARKO (DRKO) | 8.48 | 9.75 | .49 |
  | Kevin Pelton (KPel) | 9.34 | 11.72 | .30 |

  EPM and LEBRON entries did not appear.

### Inferences
- **Recommended structure:**
  - E[team net rating] = Σᵢ P(playᵢ) × ratingᵢ × E[minutesᵢ | availability set] / 48, summed over five lineup slots and normalized to 240 minutes.
  - Missing minutes are redistributed by position and role, following EPM's "context-aware" approach or a depth chart.
  - Any residual minutes go to replacement level (−2.0 to −2.7 per 100).
  - Questionable players should enter as a **mixture over scenarios** (with and without the player), not as a fractional player, because minutes redistribution is non-linear.
- **How much availability adds:** Dunks & Threes reports a gain of 0.3–0.4 points MAE and 0.9–1.5 percentage points accuracy over a net-rating baseline. Absent a pure ablation, expect the availability contribution to be **concentrated in games with star absences**: 2–5.5 points of margin in market terms, versus near zero for role players. Average improvements across all games will look small.
- **Season wins are a different benchmark.** At the season-win level, metric-based projections still trail betting markets by about 1 win of MAE (APBR 2025-26). Public metrics alone are not a substitute for market-level information. As a model input, though, BPM-based and DARKO projections were the best public ones that season.

### Gaps
- I found no peer-reviewed or publicly replicated ablation that isolates the marginal accuracy gain from pre-game injury data in NBA game prediction.
- I could not retrieve FiveThirtyEight's own published accuracy comparisons (site offline; archive unreachable from my tool).
- I could not verify the ACM study's details.
- Dunks & Threes does not say how a Questionable status is weighted.

## 5. Public player impact metrics (EPM, LEBRON, DARKO, RAPM/nbarapm, BPM, RAPTOR, newer 2024–2026): method, validation, update cadence, history, cost and terms

### Takeaway
The leading public predictive metrics in 2026 are **EPM** (Dunks & Threes), **DARKO DPM** and **LEBRON** (BBall Index). In the relevant comparisons:
- EPM had the best published retrodiction among the metrics tested in 2020 (RMSE 2.48), and it is the only one with published game-prediction validation.
- DARKO's creator reported DPM best on RMSE in a 2021 test.
- DARKO and BPM-based wins were the strongest public-metric entries in the 2025-26 APBR win contest, though both trailed the betting lines.

Access and history differ:
- DARKO is free to view, with CSV downloads, updates after every game, and history back to 1996-97.
- EPM has daily history since 2001-02 behind an annual subscription, with an API whose terms restrict commercial predictive use.
- LEBRON (2009-10 onward) has a free app plus a paid membership.
- BPM is free with history back to 1973-74, but box-score only and subject to a 20 requests/minute scraping limit.
- RAPTOR has been frozen since 2023 (CC BY 4.0 data on GitHub).
- New since 2024: ESPN's Net Points (March 2025, descriptive) and nbarapm's time-decay RAPM and LAKER.

### Cited Findings

#### EPM (Estimated Plus-Minus), Dunks & Threes (Taylor Snarr / Snarr Data Science LLC)
- **Method:** a statistical plus-minus model built from "Estimated Skills" (separate box-score and tracking-defense models) is used as the Bayesian prior for a RAPM run, "a RAPM calculation with an SPM Bayesian prior." The RAPM step adjusts for home court and the "rubber band" effect and decays older possessions exponentially. For 2026-27 a remodeled EPM and Estimated Skills replaced the earlier live model — [Dunks & Threes, EPM methodology](https://dunksandthrees.com/about/epm)
- **Training windows:** the box-score SPM was trained on 2003-04 to 2022-23 and the tracking SPM on 2017-18 to 2022-23, so 2023-24 onward is out of sample — [Dunks & Threes, EPM methodology](https://dunksandthrees.com/about/epm)
- **Validation:**
  - Next-game Estimated Skills improve on season-to-date averages by about 14.5% for minutes, 4.6% for usage, 4.1% for TS% and 3.8% for points per 100.
  - Game predictions: live MAE 11.0 vs 11.4; backtest 9.6 vs 9.9 (see Q4).
  - [Dunks & Threes, EPM methodology](https://dunksandthrees.com/about/epm)
- **2020 retrodiction by the EPM author** (team ratings weighted by actual minutes, 180 team-seasons, 2014–2019; a potential conflict of interest) — [Dunks & Threes, "NBA Player Metric Comparison" (2020-06-17)](https://dunksandthrees.com/blog/metric-comparison)
  - RMSE: **EPM 2.48**, RPM 2.60, RAPTOR 2.63, BPM 2.71, PREDATOR 2.73, PIPM 2.78, RAPM 2.80, WS/48 2.85, aPER 3.12, PER 3.20.
  - EPM also had the highest "metric strength" after controlling for roster continuity (3.76).
  - EPM was stable across minutes thresholds (2.48–2.50). RAPTOR worsened at low thresholds (2.72 at under 35 minutes).
- **Cadence and history:** Expected EPM, Estimated Skills, team ratings and simulations update **nightly**, and game predictions update more often within 30 minutes of tip. Daily history runs **since 2001-02**: 5,153 dates, 32,190 games and about 6.2 million possessions through 2025-26 — [Dunks & Threes, EPM methodology](https://dunksandthrees.com/about/epm)
- **Access and cost:** Basic, Premium and "API + Premium" tiers. Subscriptions are **annual, auto-renewing and non-refundable**, and individual. No prices were shown on the terms page. API v1 "remains available through the 2026-27 season" — [Dunks & Threes Terms](https://dunksandthrees.com/terms); [Dunks & Threes, EPM methodology](https://dunksandthrees.com/about/epm)
- **Terms** — [Dunks & Threes Terms](https://dunksandthrees.com/terms)
  - Premium downloads are "solely for Your personal, internal, and non-commercial use". No redistribution or republication.
  - No collection "via automated means (including bots, scripts, or scrapers)".
  - API commercial use beyond internal research needs a written agreement. Using API data to train or operate predictive models for commercial decisions such as betting or pricing is restricted without one.
  - Attribution to dunksandthrees.com is required.

#### DARKO / DPM (Daily Plus-Minus), Kostya Medvedovsky (site by Andrew Patton)
- **Method:** a box-score projection system with a plus-minus layer. It forecasts every stat for the next game and converts those forecasts into DPM — [DARKO, About](https://darko.app/about)
  - Each game is weighted by β^t (days since the game), with β fit per stat by differential evolution.
  - A modified **Kalman filter** runs alongside, and a gradient-boosted tree combines the two.
  - Adjustments cover rest, travel, home court, opponent, seasonality and free agency, with a separate aging curve per stat.
  - On/off data gains weight with career possessions: offense moves toward on/off over 10,000 possessions, defense over 6,000.
- **Cadence:** "updates every rating after every game." Between seasons, ratings regress toward the mean — [DARKO, About](https://darko.app/about)
- **History:** season filters run from **1996-97** to 2026-27, with a weekly "Time Machine" — [DARKO leaderboard](https://darko.app/)
- **Data access:** CSV downloads are available for the leaderboard, team pages, lineups and more. Inputs come from Basketball-Reference, NBA.com and pbpstats. No API is described — [DARKO, About](https://darko.app/about)
- **Cost and terms:** no pricing or terms appear on the pages I fetched, and the site is free to view — [DARKO](https://darko.app/)
- **Validation claims, all self-reported or secondary:**
  - DARKO beat two unnamed DFS projection sites on every stat tested except minutes. In HoopsHype's survey of team staff, 8 of 29 respondents named DPM their preferred public metric and 10 more trusted it — [DARKO, About](https://darko.app/about)
  - In Medvedovsky's 2021-01-30 retrodiction tweet, DPM ranked first by RMSE, EPM second and LEBRON third, with EPM leading on MAE and R². No independent verification — [NBAstuffer, DARKO](https://www.nbastuffer.com/analytics101/darko-daily-plus-minus/)
  - In the 2025-26 APBR win contest DARKO finished 8.48 MAE (5th of 18) — [APBRmetrics](https://apbr.org/metrics/viewtopic.php?p=44892)
- **Comparability caveat:** forum participants noted that DARKO's contest entries used Basketball Monster minutes projections, while LEBRON's used Kevin Pelton's. Metric comparisons often confound rating quality with the minutes source (search summary; 2021-22 thread) — [APBRmetrics 2021-22 contest thread](https://apbr.org/metrics/viewtopic.php?t=9903&start=90)

#### LEBRON, BBall Index
- **Method:** a box prior built from boxPIPM coefficients, stabilized with Medvedovsky's padding approach and BBall Index's Offensive Archetypes. That prior is used in **luck-adjusted RAPM** (run with Ryan Davis's tutorial script). It produces LEBRON, O-LEBRON and D-LEBRON, where "0 is average and −2.7 is replacement level" — [BBall Index, LEBRON introduction](https://www.bball-index.com/lebron-introduction/)
- The luck adjustment follows Nathan Walker's method, which reduces the influence of shooting variance such as opponent 3P% (search summary) — [BBall Index, PIPM page](https://www.bball-index.com/player-impact-plus-minus/)
- **History and access:** the database starts in 2009-10; the page says "2009-20" as of writing. The page is undated and embeds a May 2022 tweet. The LEBRON app is "free", with a "Get Access" membership link — [BBall Index, LEBRON introduction](https://www.bball-index.com/lebron-introduction/)
- **Pricing:** a Fanspo-hosted BBall Index subscribe page lists a "Data & Tools Package" at $5/month or $52.50/year. The page looks unfinished (placeholder feature text), and I could not confirm whether LEBRON data is included — [Fanspo BBall Index subscribe](https://fanspo.com/bball-index/subscribe)
- **Validation:** the intro page links to comparisons but publishes no retrodiction numbers. LEBRON was third by RMSE in Medvedovsky's 2021 test — [NBAstuffer, DARKO](https://www.nbastuffer.com/analytics101/darko-daily-plus-minus/)

#### RAPM variants and nbarapm.com
- **nbarapm.com** offers TIMEDECAY RAPM, "Impact Leaders", and rolling-window RAPM peaks over 2–5 years by era (2000s, 2010s, 2020s). It also has peak boards for RAPTOR, DARKO, LEBRON and **LAKER**, which it describes as "formerly estimated RAPTOR," a BPM-style metric. Sister sites are databallr.com and draftballr.com. No operator, methodology details, update cadence, downloads or terms are shown on the home page — [nbarapm.com](https://www.nbarapm.com/)
- **Plain RAPM** scored worse than the prior-informed metrics in out-of-sample team prediction (RMSE 2.80 vs EPM 2.48) — [Dunks & Threes, metric comparison](https://dunksandthrees.com/blog/metric-comparison)
- **xRAPM** (Engelmann) is still referenced as a current public metric in ESPN's March 2025 Net Points article. I did not find its current host — [ESPN, Net Points introduction](https://www.espn.com/nba/story/_/id/44093220/introducing-best-nba-metric-yet-net-points-amazing-early-findings)

#### BPM 2.0 (Box Plus/Minus), Basketball-Reference (Daniel Myers, February 2020)
- **Method:** traditional box-score rates per 100 team possessions, regressed onto **four 5-year Bayesian RAPM fits (1996-97 to 2015-16)** — [Basketball-Reference, BPM 2.0](https://www.basketball-reference.com/about/bpm2.html)
  - Position (PG = 1 to C = 5) and offensive role (creator to receiver) interpolate the coefficients.
  - A team adjustment, typically around −8, makes player BPMs sum to the team's adjusted efficiency.
  - It is designed to be valid at game level. DBPM should be taken "with a spoonful of salt."
- **Coverage:** game-level values from 1985, season-level values for 1974–1984. The page publishes no predictive validation — [Basketball-Reference, BPM 2.0](https://www.basketball-reference.com/about/bpm2.html)
- **Retrodiction and contest results:** RMSE 2.71 in the 2020 retrodiction, behind EPM, RPM and RAPTOR — [Dunks & Threes, metric comparison](https://dunksandthrees.com/blog/metric-comparison). BPM-based wins were the best public metric in the 2025-26 APBR contest (MAE 8.23) — [APBRmetrics](https://apbr.org/metrics/viewtopic.php?p=44892)
- **Terms:** Sports Reference rate-limits bots to 20 requests per minute and says its data is licensed from third parties, so no bulk download is offered — [Sports Reference bot traffic policy](https://www.sports-reference.com/bot-traffic.html)

#### RAPTOR (FiveThirtyEight), retired
- **Status:** FiveThirtyEight sports forecasts stopped updating on 2023-06-13 — [fivethirtyeight/data](https://github.com/fivethirtyeight/data)
- **Data:** CSVs with modern RAPTOR (2014 onward, using player tracking) and historical RAPTOR (1977 onward) — [fivethirtyeight/data, nba-raptor](https://github.com/fivethirtyeight/data/tree/master/nba-raptor)
  - Columns: box, on/off and total offense/defense, WAR, PREDATOR and pace impact. Player IDs use Basketball-Reference IDs.
  - License: the repository says data sets are under **CC BY 4.0** unless noted, and code under MIT.
- Neil Paine, who worked on RAPTOR at FiveThirtyEight, said the old ratings "are no longer updating via Disney". He built a simpler "estimated RAPTOR" for every player since 1976-77 (search summary) — [Neil Paine, Estimated RAPTOR](https://neilpaine.substack.com/p/nba-estimated-raptor-player-ratings). nbarapm now hosts this lineage as "LAKER" — [nbarapm.com](https://www.nbarapm.com/)
- **Retrodiction:** RAPTOR scored RMSE 2.63 and PREDATOR 2.73 in the 2020 test — [Dunks & Threes, metric comparison](https://dunksandthrees.com/blog/metric-comparison)

#### Newer metrics (2024–2026)
- **ESPN Net Points** (Dean Oliver, introduced 2025-03-04) — [ESPN, Net Points introduction](https://www.espn.com/nba/story/_/id/44093220/introducing-best-nba-metric-yet-net-points-amazing-early-findings)
  - It assigns credit and blame for every play-by-play event to the players on court, weighted by difficulty (for example, self-created threes get more credit).
  - It is explicitly **descriptive**; a predictive version was "planned".
  - Data is published at espnanalytics.com/nba-net-pts. ESPN's terms (Disney) apply.
- Outside analysts liken Net Points to a simpler BPM-style build, and say the finer details are unpublished (search summary) — [Jeremias Engelmann Substack](https://jeremiasengelmann.substack.com/p/espn-just-released-net-points-a-new); [APBRmetrics thread](https://apbr.org/metrics/viewtopic.php?p=42962)
- ESPN's September 2026 NBA Rank audit used "Net Points to DARKO, LEBRON, EPM and RAPM" as its collective of advanced ratings (search summary) — [ESPN NBA Rank 2026 audit (ABC7 syndication)](https://abc7news.com/post/nba-rank-2026-audit-what-got-wrong-top-100/19865032/)
- HoopsHype's analytics series tracked DPM, EPM, LEBRON, RAPTOR, RAPM, BPM and a metric called **DRIP**. No DRIP details were retrieved (search summary) — [HoopsHype via Yahoo](https://sports.yahoo.com/mvp-race-best-players-based-202748969.html)
- The original HoopsHype executive survey (2021-09-17) could not be opened (HTTP 402) — [HoopsHype survey](https://www.hoopshype.com/story/sports/nba/2021/09/17/advanced-stats-nba-real-plus-minus-rapm-win-shares-analytics/75615174007/)

### Inferences
- **Primary rating input:** for a 2026-27 game model, EPM (if a subscription and its terms suit) or DARKO DPM (free, daily, CSV) are the strongest candidates. LEBRON is a reasonable third.
  - Blending two or more metrics is likely to reduce error further, because they share RAPM foundations but use different priors. This is untested here.
  - BPM is the free fallback with the longest history, useful for backfilling training seasons before 2001-02.
- **Leak-free training needs as-of ratings:** for historical training, use **as-of-date ratings**, not end-of-season values, or the features leak future information.
  - DARKO's "Time Machine" (weekly back to 1996-97) and EPM's daily history (since 2001-02) make as-of features feasible.
  - Season-level BPM, RAPTOR and LEBRON tables need lagging (use the prior season plus in-season updates).
- **RAPTOR stops at 2022-23** (inferred from the 2023-06-13 stop date). It suits historical feature engineering only.
- **Terms for this project:**
  - EPM's API terms restrict commercial predictive use, which fits this non-commercial, no-betting project.
  - DARKO, nbarapm and LEBRON publish no clear terms. Ask the owners before automated pulls.

### Gaps
- I found no independent, recent (2023–2026) head-to-head retrodiction of EPM vs DARKO vs LEBRON using a common minutes source. Existing tests are by metric creators (2020 Dunks & Threes, 2021 Medvedovsky) or are season-win contests in which EPM and LEBRON did not enter.
- I could not retrieve EPM's current prices, BBall Index's confirmed LEBRON pricing, or LEBRON's update cadence.
- DARKO's and nbarapm's terms, and nbarapm's operator and methodology, are not documented on the pages I fetched.
- ESPN RPM's current status, xRAPM's host and the details of DRIP were not established.

## 6. Rookie and new-player projection: what predicts rookie minutes and impact, and which public projections exist

### Takeaway
Public systems set a rookie prior from **draft slot, age and height** (DARKO) or from **translated college or international stats plus age, blended with consensus or draft rank** (Kevin Pelton/ESPN). That prior is quickly overwritten as NBA data arrives.

Rookie treatment measurably changes team-level prediction error: in Dunks & Threes' test, the replacement-level prior beat the league-average prior. Draft position strongly predicts minutes: first-rounders average about 18 MPG, and the gap to second-rounders is about 7.5 MPG. However, the draft order's correlation with later BPM has weakened. I found no free, public 2026-27 rookie impact projections.

### Cited Findings
- **DARKO** has no college, Summer League or preseason data. Before a player's first game, his rating comes from "age, draft slot and height," and it "learns quickly" once he plays — [DARKO, About](https://darko.app/about)
- **Kevin Pelton's ESPN projection system** (ESPN explainer; the Africa edition is dated 2025-06-19, though the article ID dates to 2016) — [ESPN, Kevin Pelton draft projection explainer](https://africa.espn.com/nba/story/_/id/16235135/explaining-kevin-pelton-nba-draft-projection-system)
  - **Translation:** 14 core rate stats are translated to NBA equivalents, with college stats adjusted for strength of schedule. Free-throw percentage changes little in translation, while field-goal percentages decline.
  - **Season weights:** for college players, most recent season ×2, the one before ×3, two seasons before ×5. The reason given is that older prospects lose their experience edge. International weights are reversed.
  - **Regression:** college prospects regress toward NBA rookie positional averages, international prospects toward **replacement level**.
  - **Age:** each extra year of age costs about 0.5 WARP per season.
  - **Output:** average WARP over the first five seasons. The "consensus" version adds draft slot (or top-100 rank) and has been his primary ranking since 2015.
  - The database goes back to the 2006 draft.
- **Effect of rookie treatment** in the 2020 retrodiction (RMSE, EPM): **2.48 with replacement-level rookies, 2.62 with league-average rookies, 2.32 using actual same-season values**, which is an oracle. Every metric followed the same ordering — [Dunks & Threes, metric comparison](https://dunksandthrees.com/blog/metric-comparison)
- **Draft slot and minutes:**
  - For 2000–2009 first-round picks, average minutes declined with draft position. This is partly because weak teams with more minutes to give pick earlier (2012 analysis; two-season totals) — [Mr. Honner, "NBA Draft Math, Part I"](https://mrhonner.com/archives/4787)
  - First-round picks averaged 18.3 minutes per game over the past 20 seasons. In the most recent season the post covered, first-rounders averaged 7.5 more MPG than second-rounders (season unspecified; search summary) — [Wizards Points Substack](https://wizardspoints.substack.com/p/let-em-cook)
  - In 2014, only about a third of first-round picks were playing at least 15 MPG (search summary) — [Hoops Rumors](https://hoopsrumors.com/2014/12/rounders-seeing-minutes.html)
- **Draft order versus later impact:** an American Statistical Association student paper found that the correlation between BPM rank and draft-pick order has declined. Teams have not gotten better at drafting the better players earlier (search summary) — [ASA Stats Magazine paper](https://magazine.amstat.org/wp-content/uploads/2022/07/10-12HonorableMention.pdf)
- **CRAFTED NBA's draft board** shows a "typical range" of picks with similar college BPM (average by slot since 2008, college players only). I could not read the values (search summary) — [CRAFTED NBA draft prospects](https://craftednba.com/draft/prospects)
- **Other models:**
  - A Bruin Sports Analytics model predicts first-year stats of top-5 picks from draft pick, position, height and weight, with a small sample (search summary) — [Bruin Sports Analytics](https://bruinsportsanalytics.com/post/2023-nba-draft-class)
  - A 2026 thesis used 546 drafted players and undrafted free agents (2008–2021) to predict RAPM over the first four seasons from era-adjusted college metrics (search summary) — [Neapolis University Pafos repository](https://hephaestus.nup.ac.cy/items/945797ce-eeef-4dee-aef2-bcfda2fd7512/full)
- **Players returning from injury (relevant to "new" availability):** returners perform "as good as before," except after an ACL injury, where the first 20 games back are about 0.5–0.8 points per 100 worse — [Viktorsadowski/nba-moneyball](https://github.com/Viktorsadowski/nba-moneyball)

### Inferences
- **Rookie rating prior:** rating ≈ f(draft slot, age, height, plus translated college BPM where available), shrunk toward replacement level (about −2 to −2.7 per 100), with strong shrinkage for international players per Pelton. Update it Bayesianly game by game, as DARKO does, so that after roughly 500–1,000 NBA minutes the prior has little weight. The minute thresholds are an inference, not sourced.
- **Rookie minutes prior:** model minutes from draft slot × team depth at the position (the weak-team confound noted above) and update them from actual rotation minutes after the first few games. Dunks & Threes reports that its context-aware minutes models beat season-to-date averages by about 15%.
- **Fallbacks:** for undrafted, two-way and overseas signings, use replacement level as the default. This is the Dunks & Threes and Pelton convention, and in the 2020 test it beat the league-average prior.

### Gaps
- I found no public table mapping draft slot to expected rookie EPM, DPM or LEBRON, and no free 2026-27 rookie impact projections. Pelton's projections are ESPN-hosted and probably paywalled; DARKO's pre-debut values can be read from darko.app but their formula is not published.
- How EPM and LEBRON handle rookies before their NBA debut is undocumented on the pages I fetched.
- I found no quantified, recent (2020s) study of first-season minutes by pick.

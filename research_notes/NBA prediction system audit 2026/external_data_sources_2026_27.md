# External NBA data sources and 2026-27 season context for a game prediction model (state as of 2026-10-25)

Method note: web research on 2026-10-25, plus read-only probes run the same day from the project machine (a residential connection, not a cloud IP) with nba_api 1.11.4 against stats.nba.com, ESPN's public API and cdn.nba.com, and a count over the project's own PlayByPlayV3 CSVs. Probe scripts are in `checks/` (next to these notes) (league_rates.py, opening_week_history.py, monthly_rates.py, team_vs_player_sums.py, tracking_probe.py, tracking_probe2.py, foul_types.py, espn_probe.py, espn_probe2.py, cdn_probe.py). Anything dated before 2025 is marked **(pre-2025)**.

## 1. nba_api and stats.nba.com: latest release, endpoint changes, rate limits/blocking, and how long tracking feeds take to appear

### Takeaway
nba_api's latest release is v1.12.0 (2026-10-24). v1.11.5 and v1.11.6 came out two days earlier, on 2026-10-22. Upstream, stats.nba.com no longer serves the V2 box score family, PlayByPlay or PlayByPlayV2, and several leader endpoints, so the working path is the V3 box scores, PlayByPlayV3, ScoreboardV3 and ScheduleLeagueV2. BoxScoreDefensiveV2 and BoxScoreHustleV2 still serve data. There is no published rate limit. The maintainers advise about one request per second, and cloud IPs are routinely dropped. Two outages hit in October 2026 (Oct 7-10 and Oct 22). In the opening week of 2026-27, the tracking-based box scores (PlayerTrackV3, MatchupsV3, DefensiveV2) were missing for games more than about 1.4 days old. Hustle data was already up the next morning.

### Cited Findings
**Release timeline** (dates from the GitHub releases API):
- v1.12.0 2026-10-24T05:33Z; v1.11.6 2026-10-22T04:44Z; v1.11.5 2026-10-22T04:03Z; v1.11.4 2026-02-20; v1.11.3 2025-11-14; v1.11.2 2025-11-12; v1.11.1 2025-11-12; v1.11.0 2025-11-11; v1.10.2 2025-09-30; v1.10.1 2025-09-11; v1.10.0 2025-06-02; v1.9.0 2025-03-31; v1.8.0 2025-03-09; v1.7.0 2025-01-11; v1.6.1 2024-10-29 (pre-2025) — [GitHub API: swar/nba_api releases](https://api.github.com/repos/swar/nba_api/releases)

**v1.12.0 (2026-10-24)** — [nba_api releases](https://github.com/swar/nba_api/releases):
- New `ist_round_nullable` argument (NBA Cup round filter) on 34 dashboard, league, shot chart and video endpoints.
- The ten nested V3 box score endpoints gain `homeTeamId`/`awayTeamId` columns, appended at the end so existing positions don't move.
- Proxy is resolved per request (explicit argument, then `NBA_API_PROXY`, then legacy `debug.PROXY`). `NBA_API_REQUIRE_PROXY=1` raises `ProxyRequiredError` when no proxy is set.
- Adds Python 3.14/3.15 support. numpy is no longer a direct dependency. pandas is imported lazily, and its floors are 2.2.2 (Py 3.10-3.12) and 2.2.3 (3.13).
- New docs: an endpoint guide, an AI-agent guide, and a list of endpoints stats.nba.com fails on or no longer serves.
- Now emit `DeprecationWarning` ("Nothing is removed"): LeagueLeaders, AssistLeaders, AssistTracker, PlayoffPicture, BoxScoreAdvancedV2, BoxScoreFourFactorsV2, BoxScoreMiscV2, BoxScoreScoringV2, BoxScoreUsageV2, PlayByPlay. `debug.PROXY` emits a `FutureWarning`.
- Fixes: PlayerGameLogs/TeamGameLogs defaults; ScheduleLeagueV2Int import and `broadcaster_list`.

**v1.11.6 (2026-10-22)** — [nba_api releases](https://github.com/swar/nba_api/releases):
- Live `ScoreBoard`, `BoxScore`, `PlayByPlay` and `Odds` work again. cdn.nba.com had been returning 403 for the default headers, so the live client now sends a current Firefox User-Agent and `Referer: https://www.nba.com/`.

**v1.11.5 (2026-10-22)** — [nba_api releases](https://github.com/swar/nba_api/releases):
- `get_normalized_dict()/json()` no longer return empty payloads for V3 endpoints.
- ScheduleLeagueV2 parses regardless of the key order in the response.
- CI moved to GitHub Actions on Python 3.10-3.13. A PlayByPlayV3 regex test is temporarily skipped "until PlayByPlayV3 normalization is wired up".

**v1.11.4 (2026-02-20)** — [nba_api releases](https://github.com/swar/nba_api/releases):
- New `GravityLeaders` endpoint. User-Agent updated "to fix broken requests".
- `ContextMeasureSimple` lost its `PF` parameter.
- PlayByPlayV2 deprecated: "NBA.com API returns empty JSON for this endpoint. Use **PlayByPlayV3** instead".
- ScoreboardV2 "returns empty line scores for 2025-26 games"; use ScoreboardV3, "which is 100% backward compatible."

**v1.11.3 (2025-11-14)** — [nba_api releases](https://github.com/swar/nba_api/releases):
- Removed BoxScorePlayerTrackV2: "The NBA.com API no longer supports this endpoint" (HTTP 500). Use BoxScorePlayerTrackV3.
- Removed PlayerFantasyProfile. Restored TeamGameLog/TeamGameLogs.
- `LeagueIDNullable` now defaults to "00".
- `LeagueGameFinder` ignores `game_id_nullable`.
- `TeamDashboardByGeneralSplits` with `plus_minus='Y'` returns incorrect values upstream.
- PlayByPlayV3 parser made defensive against changes in the API's structure.

**v1.11.2 (2025-11-12)** — [nba_api releases](https://github.com/swar/nba_api/releases):
- Nine box score parsers had returned their datasets in the wrong order, so index access such as `get_data_frames()[0]` returned TeamStats instead of PlayerStats. Fixed here.

**v1.11.0 (2025-11-11)** — [nba_api releases](https://github.com/swar/nba_api/releases):
- New BoxScoreSummaryV3, ScoreboardV3 and DunkScoreLeaders.
- Deprecation warning added to BoxScoreTraditionalV2. BoxScoreSummaryV2 warns about data after 4/10/2025.
- Python 3.9 deprecated.

**Known upstream issues** (doc added in PR #715, checked 2026-10-21/22) — [known_upstream_issues.md via PR #715](https://github.com/swar/nba_api/pull/715):
- Failing with HTTP 500: DraftBoard, PlayerCareerByCollege, TeamGameStreakFinder, TeamHistoricalLeaders, WinProbabilityPBP.

| Removed upstream | What it returns now | Replacement |
|---|---|---|
| AssistLeaders | NBA.com homepage HTML | LeagueDashPlayerStats/TeamStats |
| AssistTracker | NBA.com homepage HTML | none |
| LeagueLeaders | NBA.com homepage HTML | LeagueDashPlayerStats |
| PlayoffPicture | NBA.com homepage HTML | LeagueStandingsV3 |
| BoxScoreAdvancedV2, FourFactorsV2, MiscV2, ScoringV2, UsageV2 | empty `{}` | the matching V3 endpoint |
| PlayByPlay, PlayByPlayV2 | empty `{}` | PlayByPlayV3 |

**Rate limits and blocking (official guidance from the maintainers):**
- The AI-agent guide (merged 2026-10-23) says stats.nba.com "throttles rapid requests". It advises: "Send one request at a time, pause about a second between requests in a loop, and cache results."
- The same guide says stats.nba.com "does not answer many cloud IP ranges". When requests hang, "raising `timeout` or adding retries will not help". Use a proxy or run the code locally.
- A removed endpoint may return `{}` (`KeyError: 'resultSet'`) or the NBA.com homepage (`JSONDecodeError`). — [docs/ai_agents.md](https://github.com/swar/nba_api/blob/master/docs/ai_agents.md)
- No requests-per-minute figure appears anywhere in the release notes or the recent issues. The only concrete numbers are a 30-second default timeout (PR #761, issue #728) and a shot chart ceiling of 102,400 rows (PR #690). — [nba_api issue search](https://api.github.com/search/issues?q=repo:swar/nba_api+timeout+OR+blocked+OR+%22rate+limit%22)

**Incidents and blocking reports:**
- 2026-10-08, issue #696: since about Oct 7, every stats.nba.com request returned an HTTP 200 Akamai HTML page ("We are unable to process your request at the moment"), even at low request volume. cdn.nba.com still worked. Closed Oct 10: "Seems to be fixed now". — [issue #696](https://github.com/swar/nba_api/issues/696)
- 2026-10-22, issue #728 (open): `ReadTimeout` on CommonTeamRoster and CommonTeamYears for 2026-27 with `timeout=30`, on both 1.9.0 and 1.11.6. — [issue #728](https://github.com/swar/nba_api/issues/728)
- 2026-03-12, issue #652: timeouts started one morning. — [issue #652](https://github.com/swar/nba_api/issues/652)
- Draft PR #761 (2026-10-24) adds a "private 2.0 client" that maps Akamai block pages to a `BlockedError`. — [PR #761](https://github.com/swar/nba_api/pull/761)
- Other open problems: SynergyPlayTypes "no longer works" for one reporter (2026-02-20). — [issue #638](https://github.com/swar/nba_api/issues/638)
- BoxScoreMatchupsV3 values differing from the nba.com matchup page were reported in 2024 (pre-2025). — [issue #451](https://github.com/swar/nba_api/issues/451)

**Researcher probe (2026-10-25, project machine):**
- About 90 sequential stats.nba.com requests at 1.2-2 s spacing (LeagueDashTeamStats, LeagueDashPlayerStats, LeagueGameLog, tracking box scores) all succeeded, with no throttling. — [stats.nba.com leaguedashteamstats](https://stats.nba.com/stats/leaguedashteamstats?Season=2025-26&SeasonType=Regular%20Season&PerMode=Totals&MeasureType=Base)
- Tracking box score availability at 2026-10-25 14:37-14:39 UTC:

| Games played on | PlayerTrackV3 / MatchupsV3 / DefensiveV2 | HustleV2 |
|---|---|---|
| Oct 20, 21, 22 | data present | not probed beyond Oct 20 |
| Oct 23 (two games, e.g. 0022600099, 0022600094) | empty (nba_api 1.11.4 parsers raised AttributeError/IndexError) | not probed |
| Oct 24 (six games, 0022600106-111) | empty | present |

  Oct 23 night games ended about 33-37 hours before the probe and Oct 22 games about 57-60 hours before it. So in opening week the tracking-derived box scores took more than about 1.4 days and at most about 2.5 days to appear. — [stats.nba.com boxscoreplayertrackv3](https://stats.nba.com/stats/boxscoreplayertrackv3?GameID=0022600099), [boxscorematchupsv3](https://stats.nba.com/stats/boxscorematchupsv3?GameID=0022600099), [boxscoredefensivev2](https://stats.nba.com/stats/boxscoredefensivev2?GameID=0022600099), [boxscorehustlev2](https://stats.nba.com/stats/boxscorehustlev2?GameID=0022600106)
- 2026-27 game IDs are not in date order: 0022600086 was played 10-21, 0022600006 on 10-22, 0022600099 on 10-23, and 0022600106-111 on 10-24. Games per date: Oct 20: 3, Oct 21: 11, Oct 22: 2, Oct 23: 12, Oct 24: 8. — [stats.nba.com leaguegamelog 2026-27](https://stats.nba.com/stats/leaguegamelog?Season=2026-27&SeasonType=Regular%20Season&PlayerOrTeam=T)
- The project environment has nba_api 1.11.4 (Feb 2026) with pandas 3.0.2 and requests 2.33.1, so it predates the v1.11.6 live-endpoint fix and v1.12.0. — researcher check, 2026-10-25 (local `importlib.metadata`).

### Inferences
- Treat "missing tracking box score" as normal for the first 1-3 days after a game. Re-poll PlayerTrackV3, MatchupsV3 and DefensiveV2 roughly 48-72 h after each game rather than logging the game as permanently unavailable. The opening-week lag may be larger than mid-season; this is a single snapshot.
- The project's `defensive_box_score_unavailable_games.txt` list may be partly an artifact of fetching too early.
- Any data pulled with nba_api 1.11.0/1.11.1 (released 2025-11-11/12) using index-based `get_data_frames()[i]` access may have Team and Player tables swapped. It is worth checking if the project fetched data in that window.
- Pipelines that infer chronology from game ID order will mis-order 2026-27 games. Sort by GAME_DATE from LeagueGameLog or ScheduleLeagueV2 instead.
- Upgrading to v1.12.0 makes the live endpoints usable and adds home/away IDs. The new columns are appended, so position-based parsing stays stable. pandas 3 compatibility is not stated in the notes.

### Gaps
- No official NBA statement of stats.nba.com rate limits or blocking rules exists. Only the maintainers' guidance and anecdotes are available.
- There is no official or crowd-sourced measurement of tracking-feed latency (for example, the median hours after the final buzzer). My probe gives only bounds from one time point.
- It is unclear whether BoxScoreTraditionalV2 still serves data. It is warned since v1.11.0 but is not on the "removed upstream" list.
- No information was found on whether the empty tracking payloads for Oct 23/24 come from a Hawk-Eye/NBA processing backlog or from the usual pipeline lag.

## 2. NBA live/CDN JSON feeds (cdn.nba.com): what they contain, how accessible they are, and terms

### Takeaway
The cdn.nba.com liveData feeds (boxscore, playbyplay, scoreboard, odds) are the fastest same-night source and carry fields that stats.nba.com box scores lack. Examples are offensive fouls, team fouls, team turnovers and player not-playing reasons. Since mid-2026, though, Akamai has been rejecting non-browser clients, and on 2026-10-25 the feeds returned 403 to scripted requests from the project machine. They are governed by NBA.com's Terms of Use (updated 2026-07-13), which restrict NBA Statistics to private, non-commercial use and forbid any gambling-related use.

### Cited Findings
**Box score feed contents** — [nba_api live boxscore docs](https://github.com/swar/nba_api/blob/master/docs/nba_api/live/endpoints/boxscore.md):
- URL pattern: `https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json`.
- Datasets: Arena, Game, Home/Away Team, Team stats, Player stats, Officials.
- Team fields include `inBonus`, `timeoutsRemaining`, period scores, `foulsOffensive`, `foulsDrawn`, `foulsPersonal`, `foulsTeam`, `foulsTechnical`, `foulsTeamTechnical`, `turnovers`, `turnoversTeam`, `turnoversTotal`, `pointsFromTurnovers`, `pointsInThePaint`, `pointsSecondChance`, `biggestLead`, `leadChanges`, `timeLeading`, `trueShootingPercentage`.
- Player fields include `status`, `notPlayingReason`, `notPlayingDescription`, `starter`, `oncourt`, `played`, `plusMinusPoints`, `foulsOffensive`, `foulsDrawn`.
- In the sample game, `turnoversTotal` = `turnovers` + `turnoversTeam` (10+1, 11+2). This is an inference from one sample, not documented.
- nba_api's live module exposes `ScoreBoard`, `BoxScore`, `PlayByPlay` and `Odds` over cdn.nba.com. All four were restored in v1.11.6 (2026-10-22) after cdn.nba.com began returning 403 for the default headers. — [nba_api releases](https://github.com/swar/nba_api/releases)

**Access problems:**
- 2026-05-27, issue #678: a CDN 403 surfaces as a misleading JSON decode error. A Referer header fixed it in testing. — [issue #678](https://github.com/swar/nba_api/issues/678)
- 2026-08-16, PR #686: the CDN blocked CI runners with an HTML "Access Denied" page. — [PR #686](https://github.com/swar/nba_api/pull/686)
- Issue #696 noted the CDN kept working during the Oct 7-10, 2026 stats.nba.com outage and proposed it as a fallback. — [issue #696](https://github.com/swar/nba_api/issues/696)
- Researcher probe, 2026-10-25: every request to `liveData/boxscore/boxscore_0022600001.json`, `liveData/playbyplay/playbyplay_0022600001.json`, `liveData/scoreboard/todaysScoreboard_00.json`, `liveData/odds/odds_todaysGames.json` and `staticData/scheduleLeagueV2.json` returned HTTP 403 with an Akamai "Access Denied" page. That held for curl and for Python requests, even with a Firefox User-Agent plus `Referer`/`Origin: https://www.nba.com`. stats.nba.com answered normally from the same machine at the same time. — [cdn.nba.com boxscore probe URL](https://cdn.nba.com/static/json/liveData/boxscore/boxscore_0022600001.json)
- A WebFetch from Anthropic's infrastructure to the same URL timed out (60 s) the same day. — [cdn.nba.com boxscore probe URL](https://cdn.nba.com/static/json/liveData/boxscore/boxscore_0022600001.json)

**Terms of use** — [NBA.com Terms of Use](https://www.nba.com/termsofuse):
- Last updated July 13, 2026.
- §1: material may be downloaded "only for your personal, noncommercial use". "No Basketball Content ... may be reproduced, republished ... without the written permission of the Operator."
- §9 (NBA Statistics): any use must carry "a prominent attribution to NBA.com". Use is allowed only "for legitimate news reporting or private, non-commercial purposes". The statistics "may not be used in connection with any fantasy game or other commercial product or service" nor "in connection with any gambling activity (including legal gambling activity)" nor "in connection with any website, product, or service that features a database".
- The fetched text had no explicit clause on robots, scraping or AI training. This rests on a tool summary of the page; see Gaps.

### Inferences
- For a private, non-commercial model that uses odds only as a yardstick, NBA.com §9 is satisfiable. Publishing a public dashboard of NBA stats next to betting lines, or anything resembling picks, would be hard to square with the "gambling activity" and "features a database" clauses.
- Don't build the nightly pipeline on cdn.nba.com without a fallback. Today it needs browser-grade clients (the headers alone did not suffice here), and stats.nba.com remains the reliable scripted path from a residential IP.
- The CDN-only fields (`foulsOffensive`, `turnoversTeam`, `notPlayingReason`) can also be reconstructed from PlayByPlayV3 and the official injury report. For offensive fouls, use the PBP `Foul`/`Offensive` subtypes (see section 5).

### Gaps
- No primary-source description of the playbyplay, scoreboard, odds or scheduleLeagueV2 CDN payloads could be retrieved today, because of the 403s. Field lists for these come only from nba_api docs not fetched here.
- I could not determine whether the 403 is IP-reputation based, TLS-fingerprint based, or temporary.
- NBA.com's full ToU text should be read directly for any anti-scraping clause. My fetch tool summarized it and may have missed one.

## 3. Alternatives and complements: Basketball-Reference, pbpstats, Cleaning the Glass, balldontlie, Sportradar/SportsDataIO, Kaggle

### Takeaway
No free source fully replaces stats.nba.com for 2025-26 and 2026-27 play-by-play and tracking data:
- **pbpstats** (Python package) has been dormant since April 2024.
- **Basketball-Reference** caps scrapers at 20 requests/minute and its terms bar using the content to train ML models.
- **balldontlie** sells play-by-play, lineups and odds only from 2025 onward.
- **Cleaning the Glass** is a cheap human-facing subscription with garbage-time filtering but no API.
- **Sportradar** (the NBA's exclusive official data distributor) and **SportsDataIO** are paid feeds.
- **Kaggle** has several datasets refreshed in October 2026, of unknown provenance and licensing.

### Cited Findings
**Basketball-Reference / Sports Reference:**
- Bot policy: FBref and Stathead block more than 10 requests per minute. All other SR sites, including Basketball-Reference, block more than 20 per minute, "regardless of bot type". Violators' sessions are "in jail for up to a day". There is no API, because "most of our data comes from third parties who sell the data to us". The page header shows an update dated May 29, 2024 (pre-2025). — [Sports Reference bot-traffic policy](https://www.sports-reference.com/bot-traffic.html)
- Terms of use (TOS Tracker snapshot 2026-04-17; the text says "Last Updated: May 19, 2023", pre-2025):
  - §5 bars "any automated means to access or use the Site" without permission "in a manner that adversely impacts site performance or access".
  - Content may not be used "for purposes of training, fine-tuning, prompting, or instructing artificial intelligence models". The restriction also covers supporting machine learning used to score or classify inputs.
  - No data store that "competes with or constitutes a material substitute". Credit SR as the source.
  - — [TOS Tracker: Sports Reference terms](https://tostracker.app/document/sports-reference)
- Direct fetches of sports-reference.com pages returned HTTP 403 to my fetch tool on 2026-10-25. — [Sports Reference bot-traffic policy](https://www.sports-reference.com/bot-traffic.html)

**pbpstats:**
- The GitHub package describes itself as "A package to scrape and parse NBA, WNBA and G-League play-by-play data" that "Supports both stats.nba.com and data.nba.com endpoints". — [dblackrun/pbpstats](https://github.com/dblackrun/pbpstats)
- Its latest commits are dated 2024-04-17 ("add override for free throws shot without a foul event linked", pre-2025), and no commit mentions PlayByPlayV3 or cdn.nba.com. — [pbpstats commits](https://api.github.com/repos/dblackrun/pbpstats/commits)

**Cleaning the Glass:**
- Pricing: one free week, then "$5/month", or two months free on an annual plan, which works out to about $50/year. — [cleaningtheglass.com](https://cleaningtheglass.com)
- All stats exclude garbage time by default. Garbage time is the 4th quarter with margin ≥25 in minutes 12-9, ≥20 in minutes 9-6, and ≥10 for the rest of the quarter. End-of-quarter heave possessions (starting with ≤4 s left in Q1-Q3) are also excluded. Possessions are counted from play-by-play. — [CTG garbage-time guide](https://cleaningtheglass.com/stats/guide/garbage_time)
- Founder Ben Falk was VP of Basketball Strategy with the 76ers. — [CTG about](https://cleaningtheglass.com/about)

**balldontlie:**
- Pricing per sport: Free $0 (5 req/min), ALL-STAR $9.99/mo (60 req/min), GOAT $39.99/mo (600 req/min), All-Access $299.99/mo for all sports. — [balldontlie.io](https://www.balldontlie.io/)
- NBA tiers — [balldontlie docs](https://docs.balldontlie.io/):
  - Free: teams, players, games.
  - ALL-STAR adds game player stats, active players and injuries.
  - GOAT adds season averages, game advanced stats, box scores (including live), lineups, shot charts, standings, leaders, betting odds, player props, contracts and "Plays".
- History — [balldontlie docs](https://docs.balldontlie.io/):
  - Data runs "from 1946-current", and season averages from 1996.
  - Advanced stats V2 start in 2015.
  - Play-by-play and lineups are "only available starting from the 2025 NBA season", and betting odds are "only available for the 2025 season and beyond".

**Sportradar / tracking:**
- Under a deal that began with 2023-24 (pre-2025), Sportradar is the NBA's exclusive data provider worldwide and the global distributor of official NBA betting data and live game video, and the NBA took an equity stake. — [iGaming Business](https://igamingbusiness.com/sports-betting/nba-to-receive-equity-in-sportradar-with-exclusive-data-deal/)
- Sony's Hawk-Eye has run league-wide optical tracking since 2023-24 (pre-2025), with Sportradar working alongside to generate the tracking data sets. — [ESPN](https://www.espn.com/nba/story/_/id/35818363/nba-use-hawk-eye-tracking-system-follow-players-ball)
- FanDuel extended its official NBA data deal with Sportradar through 2030-31 (signing date not confirmed). — [SportBusiness](https://sportbusiness.com/news/sportradar-fanduel-extend-nba-data-agreement)

**SportsDataIO:**
- Publishes "2025-26 NBA Confirmed Coverage" (real-time scores, player stats, play-by-play, historical data). — [SportsDataIO coverage](https://sportsdata.io/nba-confirmed-coverage)
- Its Vault sells NBA historical odds for "every game since 2019" via API, S3 bulk or custom delivery, on a "Contact sales" basis. — [SportsDataIO Vault](https://sportsdata.io/historical-odds)

**Kaggle** (Kaggle API, datasets matching "nba" sorted by update, retrieved 2026-10-25) — [Kaggle datasets API](https://www.kaggle.com/api/v1/datasets/list?search=nba&sortBy=updated):

| Dataset | Last updated | Contents |
|---|---|---|
| eoinamoore/historical-nba-data-and-player-box-scores, "NBA Dataset: Box Scores and Stats (1947 - Today)" | 2026-10-25 | box scores and stats |
| bendikfltaas/nba-historical-stats-1985-2026 | 2026-10-22 | historical stats |
| noahowsh/nba-shots-expected-points-2025-26 | 2026-10-24 | shots with expected points |
| archangel0x01/nba-nhl-and-college-basketball-results-and-odds | 2026-10-21 | results and odds |
| minmaxer/nba-odds-2026-27 ("Updated Weekly") | 2026-10-22 | odds and results |
| xmlzed/nba-win-probabilities-sealed-before-tip | 2026-10-24 | win probabilities |

**Other mirrors and repos:**
- zachht's "NBA Play-by-Play and Box Scores" listing says it is "updating daily", with a stated target of being up to date by the end of 8/2025. — [Baselight mirror](https://baselight.app/u/kaggle/dataset/zachht_nba_play_by_play_and_box_scores)
- gonzalogigena's all-time stats are "updated daily", but play-by-play is "in development". — [Baselight mirror](https://baselight.app/u/kaggle/dataset/gonzalogigena_nba_all_time_stats)
- shufinskiy/nba_data play-by-play covers seasons only through 2024-25, last updated 2025-02-26. — [GitHub shufinskiy/nba_data](https://github.com/shufinskiy/nba_data)

### Inferences
- **What each source adds beyond NBA.com:**
  - Cleaning the Glass: garbage-time and heave filtering.
  - balldontlie: a stable, keyed REST API with odds attached to games, for 2025-26 onward only.
  - Sportradar/SportsDataIO: licensed, redistributable feeds with SLAs.
  - Basketball-Reference: long history, but its ML-training prohibition makes it unsuitable as a training source for this model.
- pbpstats-the-package likely cannot fetch 2025-26+ games, since it depends on endpoints NBA.com stopped serving (PlayByPlayV2, data.nba.com). This is inferred from its last commit date plus the endpoint removals. Re-implementing possession parsing on PlayByPlayV3 is the realistic path.
- Most Kaggle "updated daily" NBA datasets are themselves scraped from NBA.com or Basketball-Reference, so they inherit those sites' terms. They are useful as cross-checks, not as an authoritative source.

### Gaps
- The pbpstats.com website's 2025-26/2026-27 coverage and API status could not be verified.
- Cleaning the Glass and balldontlie terms on scraping, caching and redistribution were not retrieved.
- Sportradar and SportsDataIO NBA pricing is not public.
- Licenses and provenance of the October 2026 Kaggle datasets were not inspected.

## 4. Odds data for model benchmarking only (not betting): ESPN/DraftKings, The Odds API, other archives

### Takeaway
ESPN's undocumented core API is the cheapest benchmark source:
- **From Dec 1, 2025:** DraftKings (provider 100) open, close and current spread, moneyline and total.
- **Before that, back to roughly 2023-24:** ESPN BET (provider 58), plus multiple books with labeled open and close.
- **Back to 2017-18:** a single unlabeled `current` line per book.

The catch is that Disney's terms forbid automated access. The Odds API has the cleanest licensed history (from June 2020 at 5-10 minute granularity) for $30-$59/month of credits. SportsDataIO's history starts in 2019, and Sportsbook Reviews Online's long archive (2007+) is of uncertain status.

### Cited Findings
**ESPN/DraftKings:**
- ESPN and PENN ended ESPN BET early. DraftKings became ESPN's exclusive official sportsbook and odds provider effective December 1, 2025, and PENN rebranded its app theScore Bet. — [Sports Illustrated](https://www.si.com/media/espn-bet-shutting-down-enters-new-gambling-agreement-draftkings); [Sports Media Watch](https://www.sportsmediawatch.com/2025/11/espn-bet-ending-replaced-by-draftkings-deal/); [The Desk](https://thedesk.net/2025/11/espn-bet-closing-draftkings-partnership/)
- Researcher probe, 2026-10-25, recent games: the per-event odds endpoint returned provider `100: DraftKings` for MIA@CHA on 2026-10-24 (event 401909853) and for MEM@ORL on 2026-01-15 (401810433).
  - Top-level keys: `open`, `close`, `current`, `spread`, `overUnder`, `overOdds`, `underOdds`, `moneylineWinner`, `spreadWinner`, `propBets`.
  - `homeTeamOdds` has `open`/`close`/`current` blocks with `pointSpread`, `spread` price and `moneyLine` (American, decimal and fraction).
  - Example: the CHA open was +3.5 and the close +4.5 (-112), with moneyline +150 at close.
  - The site scoreboard's `competitions[0].odds` was empty for both completed games.
  - — [ESPN core odds, event 401909853](https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/401909853/competitions/401909853/odds); [ESPN scoreboard 2026-10-24](https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=20261024)
- Researcher probe, 2026-10-25, older games — [ESPN core odds, event 401585183](https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/401585183/competitions/401585183/odds); [event 400974955](https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/400974955/competitions/400974955/odds):

| Game date (event) | Providers | Open/close labels |
|---|---|---|
| 2025-11-05 (401810026) and 2025-01-15 (401705127) | `58: ESPN BET` and `59: ESPN Bet - Live Odds` | ESPN BET: open/close/current |
| 2024-01-15 (401585183) | 12 providers incl. Caesars, DraftKings (id 40, close/current only), ESPN BET, MGM, Unibet, BetfairSportsbook | most have open/close/current |
| 2022-11-15 (401468361) | 15 providers incl. consensus | mostly `current` only |
| 2019-11-15 (401160809) and 2017-11-15 (400974955) | 10-12 providers incl. Westgate, Caesars, DraftKings (2019), consensus, numberfire, teamrankings | `current` only |

- Third-party scraper vendors also describe the endpoint as undocumented. One vendor claims the scoreboard drops odds once a game starts while the core API keeps a `close` block. These are vendor claims. — [espn-odds-scraper (PyPI)](https://pypi.org/project/espn-odds-scraper/); [Apify betting-odds-feed](https://apify.com/ichigowa/betting-odds-feed)
- Disney Terms of Use (cover ESPN-branded products; last updated May 24, 2024, pre-2025) — [Disney Terms of Use](https://disneytermsofuse.com/english/):
  - §2.B(x) prohibits accessing, copying or extracting Disney Products "using a robot, spider, script, or other automated means", including "for the purposes of creating or developing any AI Tool, data mining or web scraping".
  - §2.B(viii) bars "any commercial or business-related use".

**The Odds API:**
- NBA (`basketball_nba`) history starts 2020-06-27 (featured markets in general from June 6, 2020). Snapshots were every 10 minutes, then every 5 minutes from September 2022. Additional markets (player props, period markets) start May 3, 2023.
- Cost: 10 credits per region per market for historical odds, and 10 per region per market per event for historical event odds.
- The historical page states "Historical data is only available on paid usage plans". — [The Odds API historical data](https://the-odds-api.com/historical-odds-data/)
- Plans: Starter free (500 credits/mo), 20K $30, 100K $59, 5M $119, 15M $249 per month. My fetch read every plan card, including the free Starter, as listing "Historical Odds", which conflicts with the historical page. — [The Odds API pricing](https://the-odds-api.com/)

**Other archives:**
- SportsDataIO Vault: NBA odds "every game since 2019" (pre-game, in-play, futures; spreads and props named), contact sales. — [SportsDataIO Vault](https://sportsdata.io/historical-odds)
- FantasyData sells historical odds downloads (CSV/XLS) behind an All-Access plan. — [FantasyData NBA odds](https://fantasydata.com/nba/odds)
- SportsGameOdds advertises opening and closing lines per book across 85+ books, with historical depth on its Pro plan and above. — [SportsGameOdds](https://sportsgameodds.com/use-cases/historical-odds-data-api)
- Sportsbook Reviews Online supplied NBA odds for "every NBA game since October 2007" in a 2018 Stanford CS229 project (pre-2025). — [CS229 poster](https://cs229.stanford.edu/proj2018/poster/3.pdf)
- Old SBR forum threads (pre-2025) say the archive's lines may come from a single default book. — [SBR forum](https://www.sportsbookreview.com/forum/handicapper-think-tank/615469-getting-sbr-odds-historical-lines-into-excel)
- Kaggle odds datasets refreshed in October 2026: archangel0x01 results-and-odds (2026-10-21) and minmaxer/nba-odds-2026-27, "Updated Weekly" (2026-10-22). — [Kaggle datasets API](https://www.kaggle.com/api/v1/datasets/list?search=nba&sortBy=updated)
- balldontlie's odds endpoint (GOAT, $39.99/mo) covers only "the 2025 season and beyond". — [balldontlie docs](https://docs.balldontlie.io/)

### Inferences
- **Benchmark design:**
  - Use the closing moneyline, after removing the vig to get a probability, as the yardstick for win probability (log-loss/Brier). Use the closing spread as the yardstick for margin MAE.
  - The 2025-26 DraftKings switch (Dec 1, 2025) breaks provider continuity mid-season. Benchmark against a consistent book per period, or against consensus where available, and record the provider ID with each line.
  - 2017-2022 ESPN lines are `current` only. These are most likely closing lines for completed games, but they are unlabeled, so validate against a second source before using them as "close".
- **Cost:**
  - A rough estimate: about 3-4 timed snapshots per game day, ~165 game days, featured markets h2h+spreads+totals in one region at 30 credits per snapshot comes to about 15-20K credits per season. That is roughly one month of the $59 (100K) plan to back-fill 2020-21 through 2025-26.
  - Event-level pulls near each tip cost more.
- **Terms:** ESPN's ToU prohibits automated extraction. For anything beyond private evaluation, a licensed source (The Odds API, SportsDataIO) is the defensible choice. NBA.com stats must not be used "in connection with any gambling activity", so keep odds strictly as an offline evaluation input and don't publish picks.

### Gaps
- It is unclear whether The Odds API's free plan includes historical endpoints; the two pages conflict.
- The Odds API's redistribution terms were not read.
- Whether Sportsbook Reviews Online's NBA archive is still online and updated past 2022-23 was not established.
- ESPN `close` semantics (exact snapshot time) are undocumented.

## 5. 2026-27 rule changes, officiating points of emphasis, and the first-week stat shift

### Takeaway
No new playing rule took effect for 2026-27. The changes are a set of officiating points of emphasis announced Oct 5, 2026: off-arm push-offs, two-handed screens, flopping, off-ball contact, reckless closeouts, and free-throw stalling. The 2026-27 lottery reform affects incentives but not the rulebook. The project's first-week jump in free throws, fouls, turnovers and 3P% is real, and NBA.com totals confirm it. It is mostly the recurring opening-week effect, not a 2026-27 regime change:
- In 2024-25 and 2025-26, opening-week FTA ran about +4.2 and fouls about +3.2 to +3.7 per team-game above the full-season average.
- 2026-27's opening week is statistically indistinguishable from 2025-26's on FTA and fouls.
- Only turnovers (+0.9 to +1.1, broad-based) and 3P% (-2.0 pp, within noise) differ somewhat.

The push-off emphasis shows only a small rise in offensive fouls over last year's opening week.

### Cited Findings
**Points of emphasis (NBA.com, Oct 5, 2026, Jeff Zillgitt)** — [NBA.com points of emphasis 2026-27](https://www.nba.com/news/nba-referees-teams-points-of-emphasis-2026-27):
1. Flopping, defined as "secondary, theatrical and exaggerated movement". An in-game call brings a non-unsportsmanlike technical plus one free throw; postgame review can bring warnings, fines and a one-game suspension.
2. Off-arm push-offs. "Firming and feeling are going to be legal offensive moves"; forceful push-offs or full extension are illegal. The reason given: "an increase in off-arm push offs".
3. Two-handed screens. Monty McCutchen cited "more than 200 screens per game".
4. Off-ball contact and freedom of movement, described as long-standing focus areas, not new rules.
5. Reckless closeouts.
6. Free-throw time-wasting, enforced with delay-of-game violations.

The article describes no new rule and gives no projected foul counts.

**Push-off framework and preseason coverage:**
- Referees use a "Four F's" framework (feeling, firming, force, full extension).
- In preseason, Scottie Barnes and Shai Gilgeous-Alexander were called for push-offs. Draymond Green was skeptical, and the league will meet the competition committee more often early in the season.
- — [TSN](https://www.tsn.ca/nba/article/nba-issues-new-guidelines-instructs-refs-to-crack-down-on-push-offs-n1-50063714/); [RealGM](https://basketball.realgm.com/wiretap/287464/NBA-To-Call-More-Offensive-Push-Off-Fouls-For-2026-27-Season); [Yahoo Sports](https://sports.yahoo.com/nba/article/nba-points-of-emphasis-for-2026-27-include-tighter-officiating-on-push-offs-and-flopping-194040483.html)
- One report links the push-off emphasis to Jaylen Brown's 10 playoff fouls. — [Yahoo Sports](https://sports.yahoo.com/articles/nba-refs-emphasize-off-arm-192901153.html)

**Rule changes by season:**
- 2026-27 is the first season with Cup semifinals at the higher seed's home arena; the Championship is the only neutral-site game. The out-of-bounds coach's-challenge change (Replay Center rules on proximate fouls) took effect in 2025-26. Both come from a release dated Sept 10, 2025. — [NBA Communications](https://pr.nba.com/nba-cup-semifinal-host-sites-coachs-challenge-rules)
- Opening-day coverage (Oct 20, 2026) lists only officiating priorities (flopping, dangerous closeouts, off-arm contact, illegal screens, freedom of movement) and the lottery change as "new" for 2026-27. — [The Buzz Cincy](https://thebuzzcincy.com/3209384/nba-opening-day-2026-schedule-rules-how-to-watch/)
- Statistical rule since 2025-26 (article updated Sept 13, 2025): a shot from at least 36 feet in the final three seconds of quarters 1-3, on a play starting in the backcourt, "will count as a team shot attempt — but not an individual one". — [NBA.com heave rule](https://www.nba.com/news/nbas-heave-rule-will-allow-deep-end-of-quarter-shots-without-hurting-shooting-percentages)

**2025-26 context:**
- The 2025-26 emphasis (CBS, Sept 30, 2025): "secondary contact" swipes on closeouts would be called as defensive fouls. The writer predicted "a lot more fouls on 3-point shots and a lot more free throws". — [CBS Sports](https://www.cbssports.com/nba/news/nbas-latest-officiating-emphasis-could-lead-to-even-more-fouls-on-3-point-shots-during-2025-26-season)
- Personal fouls rose 1.3 per game in 2025-26, to 19.9, "the biggest increase in over 20 years". SI expects modest regression in 2026-27. — [Sports Illustrated](https://www.si.com/nba/four-stats-that-will-define-2026-27-nba-season-teams-that-do-them-best)

**Lottery and injury reporting:**
- The 3-2-1 Lottery (approved May 28, 2026; in effect for the 2027-2029 drafts) — [NBA.com lottery](https://www.nba.com/news/nba-board-governors-approve-new-draft-lottery-system):
  - The lottery grows from 14 to 16 teams.
  - Non-playoff, non-Play-In teams get 3 balls. The three worst records are "draft relegated" to 2 balls and can fall no lower than No. 12.
  - Play-In 9/10 seeds get 2 balls each, and the losers of the 7-vs-8 games get 1 ball each.
  - A team's own pick can't go first overall in consecutive drafts or top-five three straight years. No top-12 to top-15 protections are allowed on newly traded picks.
  - The league gains power to cut lottery odds or move draft position for tanking.
- The vote was 29-1, with Memphis dissenting. — [Hoops Rumors](https://www.hoopsrumors.com/2026/05/nbas-board-of-governors-approves-3-2-1-lottery-reform-plan.html)
- A December 2025 league memo requires teams to resubmit injury reports on game days (except the second night of back-to-backs), to "diminish the value of confidential information". — [AP via ABC News](https://abcnews.go.com/Sports/wireStory/nba-tweaks-injury-reporting-rules-prop-betting-128577446)
- Participation-status deadlines are 5 p.m. local the day before, and a game-day report is due 11 a.m.-1 p.m. — [official.nba.com injury report 2025-26](https://official.nba.com/nba-injury-report-2025-26/)

**Researcher probe 1 — the user's figures against NBA.com** (LeagueDashTeamStats Totals, Regular Season, per team-game, queried 2026-10-25) — [stats.nba.com leaguedashteamstats](https://stats.nba.com/stats/leaguedashteamstats?Season=2026-27&SeasonType=Regular%20Season&PerMode=Totals&MeasureType=Base); [leaguedashplayerstats](https://stats.nba.com/stats/leaguedashplayerstats?Season=2025-26&SeasonType=Regular%20Season&PerMode=Totals&MeasureType=Base); [leaguegamelog](https://stats.nba.com/stats/leaguegamelog?Season=2026-27&SeasonType=Regular%20Season&PlayerOrTeam=T)
- 2026-27 through Oct 24 (36 games / 72 team-games): FTA 26.93, PF 23.85, TOV 16.88 team (16.15 when player turnovers are summed), 3P% 33.58, 3PA 37.31, PTS 115.08.
- 2025-26 full season: FTA 23.48, PF 19.87, TOV 14.51 team (13.80 player-summed), 3P% 35.96.
- The project's "last season" baselines (23.5 / 19.9 / 13.8 / 36.0%) therefore match NBA.com exactly, with turnovers computed as player sums that exclude team turnovers of about 0.71 per team-game. The first-week figures (27.0 / 24.0 / 16.4 / 32.9%, on 28 games) are consistent with NBA.com's 36-game numbers.
- Team-game spread, 2026-27: SD for FTA 7.14 (SE 0.84), PF 3.63 (SE 0.43), TOV 4.04 (SE 0.48).

**Researcher probe 2 — opening week (first 7 days) against the same season's full average.** Per team-game, NBA.com totals; the 2026-27 row compares to the 2025-26 full season — [stats.nba.com leaguedashteamstats](https://stats.nba.com/stats/leaguedashteamstats?Season=2025-26&SeasonType=Regular%20Season&PerMode=Totals&MeasureType=Base&DateFrom=10/21/2025&DateTo=10/27/2025)

| Season | FTA opening week → full (Δ) | PF opening week → full (Δ) | TOV opening week → full (Δ) | 3P% opening week → full (Δ) |
|---|---|---|---|---|
| 2017-18 | 23.64 → 21.68 (+1.96) | 21.90 → 19.85 (+2.05) | 15.98 → 14.26 (+1.72) | 34.76 → 36.18 (−1.42) |
| 2018-19 | 24.81 → 23.07 (+1.74) | 23.16 → 20.90 (+2.26) | 14.62 → 14.08 (+0.54) | 35.82 → 35.50 (+0.32) |
| 2019-20 | 25.04 → 23.11 (+1.93) | 23.90 → 20.78 (+3.12) | 16.71 → 14.54 (+2.17) | 34.31 → 35.79 (−1.48) |
| 2020-21 | 23.65 → 21.82 (+1.83) | 21.41 → 19.29 (+2.12) | 15.45 → 13.84 (+1.61) | 35.37 → 36.66 (−1.29) |
| 2021-22 | 19.77 → 21.86 (−2.09) | 19.50 → 19.64 (−0.14) | 15.22 → 13.76 (+1.46) | 34.76 → 35.36 (−0.60) |
| 2022-23 | 24.33 → 23.53 (+0.80) | 21.72 → 19.98 (+1.74) | 14.45 → 14.10 (+0.35) | 35.53 → 36.07 (−0.54) |
| 2023-24 | 21.75 → 21.72 (+0.03) | 19.97 → 18.73 (+1.24) | 14.84 → 13.60 (+1.24) | 34.98 → 36.57 (−1.59) |
| 2024-25 | 25.90 → 21.67 (+4.23) | 21.81 → 18.60 (+3.21) | 15.31 → 14.30 (+1.01) | 35.40 → 36.02 (−0.62) |
| 2025-26 | 27.74 → 23.48 (+4.26) | 23.61 → 19.87 (+3.74) | 15.98 → 14.51 (+1.47) | 35.55 → 35.96 (−0.41) |
| **2026-27 (Oct 20-24)** | **26.93** (+3.45 vs 25-26 full) | **23.85** (+3.98) | **16.88** (+2.37) | **33.58** (−2.38) |

- Mean opening-week gap over 2017-18 to 2025-26: FTA +1.63, PF +2.15, TOV +1.29, 3P% −0.85 pp.
- 2026-27 opening week against 2025-26 opening week: FTA −0.81, PF +0.24, TOV +0.90, 3P% −1.97 pp. Approximate z-scores, assuming equal team-game SDs: −0.7, +0.4, +1.5, and −1.65 (binomial, about 2,686 vs 3,882 attempts).

**Researcher probe 3 — how fast the October inflation fades** (per team-game, NBA.com `Month` parameter) — [stats.nba.com leaguedashteamstats](https://stats.nba.com/stats/leaguedashteamstats?Season=2025-26&SeasonType=Regular%20Season&PerMode=Totals&MeasureType=Base&Month=1)

| Month | 2025-26 FTA | 2025-26 PF | 2025-26 TOV | 2025-26 3P% | 2024-25 FTA | 2024-25 PF | 2024-25 TOV | 2024-25 3P% |
|---|---|---|---|---|---|---|---|---|
| Oct | 26.48 | 22.59 | 15.59 | 35.76 | 24.92 | 21.34 | 15.09 | 35.45 |
| Nov | 24.68 | 20.95 | 14.89 | 35.83 | 21.35 | 18.83 | 14.67 | 36.00 |
| Dec | 23.75 | 20.14 | 14.37 | 36.32 | 21.62 | 18.77 | 15.03 | 35.90 |
| Jan | 22.05 | 18.65 | 14.17 | 35.67 | 21.07 | 18.14 | 14.06 | 36.08 |
| Feb | 22.69 | 18.95 | 14.55 | 35.74 | 22.17 | 18.65 | 14.02 | 35.48 |
| Mar | 23.57 | 19.94 | 14.30 | 35.93 | 21.68 | 18.24 | 13.70 | 36.63 |
| Apr | 22.36 | 18.91 | 14.29 | 36.91 | 20.70 | 17.64 | 13.99 | 36.10 |

October as a multiple of the full-season rate: 2025-26 FTA ×1.13, PF ×1.14, TOV ×1.07; 2024-25 FTA ×1.15, PF ×1.15, TOV ×1.06.

**Researcher probe 4 — foul and turnover subtypes** from the project's PlayByPlayV3 CSVs (`nba_data/<season>/<game>/play_by_play/*pbp.csv`, sourced from stats.nba.com PlayByPlayV3; 2025-26 dates from LeagueGameLog). Per team-game — [stats.nba.com playbyplayv3](https://stats.nba.com/stats/playbyplayv3?GameID=0022600001):

| Event type | 2026-27, 28 games | 2025-26 opening week, 53 games | 2025-26 full, 1,230 games |
|---|---|---|---|
| All foul events | 24.57 | 24.18 | 20.35 |
| Offensive | 1.89 | 1.77 | 1.16 |
| Offensive charge | 0.43 | 0.37 | 0.30 |
| Personal (common) | 8.21 | 7.47 | 6.03 |
| Shooting | 11.07 | 11.71 | 10.41 |
| Loose ball | 1.79 | 1.74 | 1.42 |
| Technical | 0.30 | 0.23 | 0.28 |
| All turnovers (incl. team) | 17.11 | 15.98 | 14.51 |
| Offensive-foul turnovers | 2.36 | 2.17 | 1.47 |
| Bad pass | 5.64 | 5.36 | 5.19 |
| Lost ball | 3.84 | 3.41 | 3.35 |
| Out-of-bounds bad pass | 2.09 | 1.80 | 1.71 |
| Traveling | 0.80 | 0.66 | 0.53 |
| Delay-of-game violations | 0.23 | 0.29 | 0.21 |

"Flop" appears in descriptions fewer than 0.05 times per team-game in all three samples.

**Shot accounting check:** NBA.com team FGA and FG3A equal the sum of player FGA and FG3A in 2024-25, 2025-26 and 2026-27; there are no team-only attempts in the LeagueDash totals. Team-only turnovers are 0.71-0.80 per team-game. — [stats.nba.com leaguedashplayerstats](https://stats.nba.com/stats/leaguedashplayerstats?Season=2025-26&SeasonType=Regular%20Season&PerMode=Totals&MeasureType=Base)

### Inferences
- **What explains the shift:** The 2026-27 first-week jump against last season's average is mainly the opening-month effect that recurs in nearly every season of the training window. Rust, new emphases enforced strictly in October, and similar factors all contribute. The size of the effect has grown since 2024-25.
- **Fouls and free throws:** Against last year's opening week, FTA and PF are flat. Shooting fouls are actually lower (−0.64), so neither the closeout nor the flop emphasis is adding free throws so far.
- **The push-off emphasis:** It shows up as a modest +0.18 offensive fouls (incl. charges) per team-game over last year's opening week. That is only about 0.19 of the +1.13 turnover increase.
- **Turnovers:** The rest of the turnover rise is bad passes, lost balls, out-of-bounds passes and travels. That pattern is consistent with heavy roster churn (see section 6) or with noise, not with the officiating changes.
- **3P%:** The −2.4 pp gap against last season's full average is larger than any prior opening-week gap (the worst was −1.59). It is still within about 1.7 binomial SE of last year's opening week. No source attributes it to a rule, ball or officiating change.
- **For the simulator** (the project's sim against real data): calibrate month-of-season multipliers, roughly ×1.13-1.19 for FTA and PF in October/opening week and ×1.06-1.10 for TOV. Don't refit league-level foul and FT parameters to opening-week data; the inflation fades through November and December.
- **Non-stationarity in the training data:** Roughly in per-team-game terms, the training window spans three foul and free-throw environments:
  - 2017-23: FTA about 21.7-23.5.
  - 2023-25: FTA about 21.7, with a historically low whistle.
  - 2025-26: a level shift of PF +1.27 and FTA +1.81 after the closeout "secondary contact" emphasis.

  Features such as FT rate or foul-drawing should be normalized to league-season-to-date or rolling league means, not pooled 2017-2026 means.
- **Lottery incentives:** The 3-2-1 Lottery changes late-season incentives in 2026-27. Being bottom-3 now costs a ball, and Play-In 9/10 seeds get only 2 balls against 3 for teams just outside the Play-In. Tank-related patterns learned from 2017-2026 (late-season losing streaks, healthy scratches among the worst teams) may weaken, and new edge incentives may appear around 9th-12th place. This is my reading of the mechanics, not a reported effect.
- **Injury reporting:** The Dec 2025 rule makes pre-game availability data denser and fresher than in most of the training window.

### Gaps
- I found no league-wide reporting (as of 2026-10-25) quantifying 2026-27 regular-season fouls, offensive fouls, flops or 3P% after week one. NBA.com's week-1 power rankings (Oct 19, 2026) cite only preseason league numbers.
- I don't know why 2025-26's opening week already had elevated offensive fouls (1.77 against a 1.16 season average). No source was found on 2025-26 offensive-foul emphasis.
- The Bleacher Report explainer "Explaining the NBA Rule Changes for 2026-27 Season, Including NBA Cup Bracket Format" returned 403 and could not be checked for additional items.
- SE comparisons assume 2025-26's opening-week team-game SDs equal 2026-27's. The 3P% SE is a binomial lower bound that ignores clustering.

## 6. 2026-27 schedule, NBA Cup, roster/eligibility rules, expansion, media and early-season reporting

### Takeaway
2026-27 keeps the 82-game format, running Oct 20, 2026 to Apr 11, 2027. Each team has 80 games pre-scheduled plus 2 set by NBA Cup results. The Cup changes only its venues:
- The semifinals move from Las Vegas to the higher seed's home arena. Quarterfinals were already home games.
- The final moves to a neutral Hinkle Fieldhouse (Indianapolis) on Dec 11. As before, it doesn't count toward standings or statistics.

There are no 2026-27 eligibility rule changes beyond the existing CBA two-way mechanics. Expansion to Las Vegas/Seattle is only "authorized to explore" (March 2026), with 2028-29 the earliest start. This is year 2 of the ESPN/ABC, NBC/Peacock and Prime Video deals, with no data-availability changes found. Offseason player movement was unusually large.

### Cited Findings
**Schedule:**
- Released Aug 13, 2026: tip-off Oct 20, season end Sunday Apr 11, 2027. Each team plays 82 games, with dates and opponents set for 80. The remaining two fall between Dec 4 and Dec 10, depending on Cup group results.
- Opening night was an NBC/Peacock tripleheader (Celtics at Pistons, 76ers at Knicks, Thunder at Spurs). Every national game streams on the ESPN App, Peacock or Prime Video. — [NBA Communications schedule release](https://pr.nba.com/2026-27-nba-regular-season-schedule/)

**NBA Cup** — [ESPN, Aug 12, 2026](https://www.espn.com/nba/story/_/id/49577473/2026-nba-season-tournament-cup-format-highlights-updates):
- Groups:

| East A | East B | East C |
|---|---|---|
| DET, TOR, ORL, MIL, BKN | NYK, CLE, PHI, MIA, IND | BOS, ATL, CHA, CHI, WAS |

| West A | West B | West C |
|---|---|---|
| DEN, HOU, PHX, DAL, UTA | OKC, MIN, LAC, NOP, MEM | SAS, LAL, POR, GSW, SAC |

- Group nights: Oct 30, Nov 6, 13, 20, 24, 25, 27.
- Knockouts: quarterfinals Dec 4-5 (home sites), semifinals Dec 8 and/or 9 (home sites), Championship Dec 11 at Hinkle Fieldhouse, moved from Las Vegas.
- The 22 group-stage exits play two extra games (one home, one road). The two quarterfinal losers play each other. Semifinalists finish at 82 games. Finalists play 83, but the final doesn't count toward standings or statistics.
- An NBA team Cup page describes quarterfinals on Dec 9-10, which conflicts with ESPN and with the official schedule window (Dec 4-10). The team page looks like stale 2025 text. — [NBA.com Pistons Cup page](https://www.nba.com/pistons/emirates-nba-cup-2026)
- Higher-seed home semifinals were approved by the Board of Governors. Through 2025-26, the semifinals and final were at a neutral site (T-Mobile Arena). — [NBA Communications, Sept 10, 2025](https://pr.nba.com/nba-cup-semifinal-host-sites-coachs-challenge-rules)

**Media:**
- NBCUniversal carries 100 regular-season games, ESPN/ABC 80, and Prime Video 66 plus the Cup knockouts, the Play-In and the 2027 ECF. ABC has the 2027 Finals. NBA.com/app adds a free "Tap to Watch" locator. — [The Buzz Cincy, Oct 20, 2026](https://thebuzzcincy.com/3209384/nba-opening-day-2026-schedule-rules-how-to-watch/)
- Year 2 national slates were set Aug 13, 2026. — [Sports Video Group](https://sportsvideo.org/2026/08/13/espn-abc-nbc-peacock-prime-video-set-national-slates-for-year-2-of-nba-rights-deal/)

**Roster and eligibility:**
- Teams carry 15 standard plus 3 two-way contracts. Two-way players may be active for up to 50 of 82 games, earn half the rookie minimum, must have three or fewer years of service, and are not playoff-eligible unless converted. The 2025-26 two-way signing deadline was March 4 (it had been Jan 15 under the previous CBA). — [Hoops Rumors, March 2026](https://www.hoopsrumors.com/2026/03/nbas-two-way-signing-deadline-has-now-passed-3.html)
- Two-way pay is reported as $678,882 for 2026-27. That figure came from a search summary across these two pages and was not checked line by line. — [Yardbarker 2026-27 two-way tracker](https://www.yardbarker.com/nba/articles/202627_nba_two_way_contract_tracker/s1_14822_44036295); [Hoops Rumors](https://www.hoopsrumors.com/2026/03/nbas-two-way-signing-deadline-has-now-passed-3.html)
- The roster deadline was 4 p.m. CT, Oct 19, 2026. — [Yardbarker](https://www.yardbarker.com/nba/articles/key_2026_nba_preseason_dates_deadlines/s1_14822_44285789)
- One outlet reports an Exhibit 10 cap of about $91K and a second apron of about $221.7M for 2026-27. These are single-source and unverified. — [EssentiallySports](https://www.essentiallysports.com/nba-active-basketball-news-eighty-nine-nba-players-waived-in-a-single-day-as-teams-scramble-to-keep-up-with-new-rules/)

**Expansion:**
- On Mar 25, 2026, the Board of Governors voted to *explore* expansion bids for Las Vegas and Seattle; it did not approve teams. Bid fees are expected to total $7-10B. A decision is expected by the end of the calendar year or later, with 2028-29 the earliest start. The article mentions no effect on the 2026-27 or 2027-28 schedules. — [Front Office Sports](https://frontofficesports.com/article/nba-expansion-seattle-las-vegas/)
- Silver reiterated the end-of-2026 decision timeline and 2028-29 target. — [FOX 13 Seattle](https://www.fox13seattle.com/news/nba-seattle-expansion-adam-silver); [News 3 LV](https://news3lv.com/sports/nba-commissioner-says-las-vegas-seattle-remain-expansion-targets-for-2028-29-season)
- Approving new teams requires 23 of 30 votes, and PJT Partners is advising the league. — [OPB, Mar 26, 2026](https://www.opb.org/article/2026/03/26/nba-expansion-league-votes-yes-seattle-supersonics-vegas)

**Offseason player movement** (NBA.com power rankings, Oct 19, 2026, John Schuhmann) — [NBA.com power rankings week 1](https://www.nba.com/news/power-rankings-2026-27-week-1):
- Jaylen Brown moved from Boston to Philadelphia, where LeBron James now also plays.
- Giannis Antetokounmpo was traded from Milwaukee to Miami.
- Ja Morant is in Portland with Damian Lillard.
- LaMelo Ball was traded from Charlotte to Minnesota.
- Julius Randle was traded from Minnesota to Brooklyn.
- The same piece gives preseason league marks of 110.2 points per 100 possessions and 103.8 possessions per 48 minutes.

**Early-season trend reporting:**
- Searches found no league-wide first-week 2026-27 trend reporting. Coverage is preseason-oriented, such as SI's foul-regression prediction and TSN/RealGM on push-off enforcement. — [Sports Illustrated](https://www.si.com/nba/four-stats-that-will-define-2026-27-nba-season-teams-that-do-them-best); [TSN](https://www.tsn.ca/nba/article/nba-issues-new-guidelines-instructs-refs-to-crack-down-on-push-offs-n1-50063714/)

### Inferences
- **Schedule-density features for Dec 4-10:** Those dates are unknowable until Cup group play ends (Nov 27). Re-pull ScheduleLeagueV2 after the Cup draw resolves; the `ist_round_nullable` filter in nba_api v1.12.0 helps tag Cup games.
- **Home/neutral flags for Cup knockout games:** Cup semifinals were neutral-site in 2023-24 to 2025-26 but are home games in 2026-27. The final remains neutral and is excluded from standings and statistics, so a model trained on 2023-26 Cup games must not apply a neutral-site flag to 2026-27 semifinals.
- **Roster churn:** The heavy offseason movement (PHI, MIA, MIL, BOS, POR, MIN, CHA, BKN at minimum) means 2025-26 team ratings carried into 2026-27 will be poor priors for these teams. Player-based priors (minutes-weighted player ratings) matter more than usual in October and November. This also likely contributes to early-season turnover noise.
- **Expansion:** No effect on 2026-27 data. A 2028-29 expansion would later bring realignment and new-team cold starts.
- **Media:** The ESPN/NBC/Prime split changes nothing in data availability. National-TV assignment (100/80/66 games) is available as a potential feature in the schedule data (`broadcaster_list` was fixed in ScheduleLeagueV2Int in nba_api v1.12.0).

### Gaps
- No post-March-2026 formal expansion vote was found as of 2026-10-25.
- No official 2026-27 salary cap or apron figures were retrieved from NBA.com.
- No 2026-27 eligibility changes (for example, to G League, international or age rules) were found. The search may simply have missed niche items.
- The NBA TV schedule and any changes to League Pass blackouts or data were not covered.
- Cup prize money for 2026 was not stated in the sources found; ESPN gives only 2023 amounts.

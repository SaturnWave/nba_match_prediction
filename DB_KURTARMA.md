# Veritabani kurtarma notu

MariaDB'yi barindiran telefon 2026-09-03'te coktu ve `phonedb` ile birlikte
gitti. Bu belge iki soruyu cevapliyor: ne kaybedildi, ve geri istenirse tam
olarak ne kurulmali.

## Kisa cevap: veri kaybi yok

`nba_data/` 9 sezonun 10.749 macini CSV olarak tutuyor ve git'te. Veritabani
bunlarin uzerine kurulmus bir hiz katmaniydi. `csv_master.py` ayni oyun
cercevesini CSV'lerden uretiyor; 190 feature'in 190'i birebir tutuyor.
Veritabani artik zorunlu degil, sadece hizli.

Kaybolan tek sey CSV->MariaDB yukleyicisiydi; repoda hic yoktu. Asagidaki
sema onu yeniden yazmak icin yeterli.

## Temel tablolar (yukleyicinin kurmasi gerekenler)

| tablo | tanecik | kaynak |
|---|---|---|
| `games` | one row per game (10,749 rows) | `game_ids/game_id_{season}.csv (9 files, one per season)` |
| `game_summary` | one row per game (10,749 rows) | `game_ids/game_id_{season}.csv + nba_data/{season}/{game_id}/box_scores/{game_id}box_score_traditional_team.csv` |
| `game_dates` | one row per game (10,749 rows) | `game_ids/game_id_{season}.csv` |
| `box_team_traditional` | one row per (game, team) -- exactly 2 rows per game, 21,498 rows | `nba_data/{season}/{game_id}/box_scores/{game_id}box_score_traditional_team.csv` |
| `box_team_advanced` | one row per (game, team) -- exactly 2 rows per game, 21,498 rows | `nba_data/{season}/{game_id}/box_scores/{game_id}box_score_advanced_team.csv` |
| `box_player_traditional` | one row per (game, player) including players who dressed but did not play | `nba_data/{season}/{game_id}/box_scores/{game_id}box_score_traditional.csv (the PLAYER-level file, not _team)` |
| `training_games` | one row per game (10,749 rows) | `derived from game_ids/game_id_{season}.csv (GAME_DATE + MATCHUP); optionally lifted verbatim from output/engineered_dataset_db.pkl` |
| `play_by_play` | one row per play-by-play action row | `nba_data/{season}/{game_id}/play_by_play/{game_id}pbp.csv (camelCase header, byte-identical across all 9 seasons)` |

### `games`

one row per game (10,749 rows); the authoritative game list and season lookup

Kaynak: `game_ids/game_id_{season}.csv (9 files, one per season)`  
Birincil anahtar: `game_id`

| sutun | tip | kaynak |
|---|---|---|
| `game_id` | `CHAR(10) NOT NULL` | GAME_ID from game_ids/game_id_{season}.csv. MUST be read as string, or zero-padded with str(x).zfill(10) after pandas infers int64 -- '0021700001', never an integer. INDEX: PRIMARY KEY (join target of game_summary, box_*, play_by_play, training_games, player_game_impact, pbp_defensive_event). |
| `season` | `VARCHAR(9) NOT NULL` | derived: the season label in the source filename / nba_data directory name, underscore form '2019_2020'. No raw per-game CSV carries a season column. Equivalent fallback: the 002+YY prefix of game_id (0022400058 -> 2024_2025). INDEX: KEY idx_games_season (season) -- filtered by every _season_filter 'AND g.season IN (...)' in db_source, clutch_features and availability_features, and GROUP BY/ORDER  ... |

### `game_summary`

one row per game (10,749 rows); driving table of db_source.load_master_frame

Kaynak: `game_ids/game_id_{season}.csv + nba_data/{season}/{game_id}/box_scores/{game_id}box_score_traditional_team.csv`  
Birincil anahtar: `game_id`

| sutun | tip | kaynak |
|---|---|---|
| `game_id` | `CHAR(10) NOT NULL` | GAME_ID (game_ids CSV). INDEX: PRIMARY KEY -- joined to games, box_team_*, training_games, box_player_traditional, play_by_play. |
| `season` | `VARCHAR(9) NOT NULL` | derived: same rule as games.season (filename / directory / 002+YY prefix). MUST agree with games.season row-for-row: load_master_frame SELECTs gs.season but FILTERS on g.season. |
| `game_date` | `DATE NOT NULL` | GAME_DATE from game_ids/game_id_{season}.csv (date-only ISO, e.g. 2024-12-03). INDEX: KEY idx_gs_date (game_date, game_id) -- ORDER BY gs.game_date, gs.game_id in load_master_frame. |
| `home_abbr` | `VARCHAR(5) NOT NULL` | derived from MATCHUP in game_ids CSV: 'DEN vs. GSW' -> home=DEN (token before ' vs. '); 'MIA @ IND' -> home=IND (token AFTER ' @ '). Both branches are required -- 2017_2018/2018_2019 are 100% 'vs.' but 2019_2020..2025_2026 mix both. Verified 10,749/10,749 against output/engineered_dataset_db.pkl. Cross-checkable against play_by_play.location='h'. INDEX: KEY idx_gs_home (home_abbr, game_id) -- merg ... |
| `away_abbr` | `VARCHAR(5) NOT NULL` | derived from MATCHUP: the other side of the same parse. Verified 10,749/10,749. INDEX: KEY idx_gs_away (away_abbr, game_id). |
| `home_team_id` | `BIGINT NOT NULL` | TEAM_ID from box_score_traditional_team.csv, on the row whose TEAM_ABBREVIATION equals home_abbr. Read by header name -- the file has two column orders (OLD 3/4/5 = TEAM_NAME,TEAM_ABBREVIATION,TEAM_CITY; NEW 3/4/5 = TEAM_CITY,TEAM_NAME,TEAM_ABBREVIATION + teamSlug). Verified 120/120 sampled games. |
| `away_team_id` | `BIGINT NOT NULL` | TEAM_ID from box_score_traditional_team.csv on the row whose TEAM_ABBREVIATION equals away_abbr. |
| `home_pts` | `SMALLINT NOT NULL` | PTS from box_score_traditional_team.csv on the TEAM_ABBREVIATION = home_abbr row (cast via float: OLD files render '130', some render floats). NEVER from the play-by-play tail -- that path is wrong for 10 games (0021900106, 0022000520, 0022000794, 0022400322, 0022400648, 0022500884, 0022500232 read as 0-0 or mid-game; 0022100028, 0022100298, 0022301202 trail by 2-3). Verified 120/120 against the D ... |
| `away_pts` | `SMALLINT NOT NULL` | PTS from box_score_traditional_team.csv on the TEAM_ABBREVIATION = away_abbr row. |

### `game_dates`

one row per game (10,749 rows); separate table joined by player_source.load_player_games even though game_summary carries the same date

Kaynak: `game_ids/game_id_{season}.csv`  
Birincil anahtar: `game_id`

| sutun | tip | kaynak |
|---|---|---|
| `game_id` | `CHAR(10) NOT NULL` | GAME_ID (game_ids CSV). INDEX: PRIMARY KEY -- joined as 'gd.game_id = b.game_id'. |
| `game_date` | `DATE NOT NULL` | GAME_DATE (game_ids CSV). Must equal game_summary.game_date row-for-row. |

### `box_team_traditional`

one row per (game, team) -- exactly 2 rows per game, 21,498 rows

Kaynak: `nba_data/{season}/{game_id}/box_scores/{game_id}box_score_traditional_team.csv`  
Birincil anahtar: `game_id`, `team_abbreviation`

| sutun | tip | kaynak |
|---|---|---|
| `game_id` | `CHAR(10) NOT NULL` | GAME_ID. INDEX: PRIMARY KEY (game_id, team_abbreviation) -- 'WHERE game_id IN (...)' in load_master_frame and 'JOIN games g2 ON g2.game_id=b.game_id' in db_build_derived.verify. |
| `team_abbreviation` | `VARCHAR(5) NOT NULL` | TEAM_ABBREVIATION. MUST be read by header NAME: the OLD 25-col variant (6,913 files: 2019_2020..2023_2024 + 2024_2025 games 0022400001-0022401084) puts TEAM_NAME,TEAM_ABBREVIATION,TEAM_CITY at fields 3/4/5, the NEW 26-col variant (3,836 files: 2017_2018, 2018_2019, 2025_2026 + 2024_2025 games 0022401085-0022401230) puts TEAM_CITY,TEAM_NAME,TEAM_ABBREVIATION there plus teamSlug. A positional load s ... |
| `fgm` | `SMALLINT` | CSV column FGM (cast via float -- OLD files render '42', player-level OLD renders '6.0') |
| `fga` | `SMALLINT` | CSV column FGA |
| `fg_pct` | `DECIMAL(5,3)` | CSV column FG_PCT |
| `fg3_m` | `SMALLINT` | CSV column FG3M (note the DB name gains an underscore: FG3M -> fg3_m, not fg3m) |
| `fg3_a` | `SMALLINT` | CSV column FG3A (DB name fg3_a) |
| `fg3_pct` | `DECIMAL(5,3)` | CSV column FG3_PCT |
| `ftm` | `SMALLINT` | CSV column FTM |
| `fta` | `SMALLINT` | CSV column FTA |
| `ft_pct` | `DECIMAL(5,3)` | CSV column FT_PCT |
| `oreb` | `SMALLINT` | CSV column OREB |
| `dreb` | `SMALLINT` | CSV column DREB |
| `reb` | `SMALLINT` | CSV column REB |
| `ast` | `SMALLINT` | CSV column AST |
| `stl` | `SMALLINT` | CSV column STL |
| `blk` | `SMALLINT` | CSV column BLK (also read directly by db_build_derived.verify's season block reconciliation) |
| `to` | `SMALLINT` | CSV column TO. The physical column is literally named `to` -- a MariaDB reserved word; db_source backticks it. Do NOT rename to tov/turnovers. |
| `pf` | `SMALLINT` | CSV column PF |
| `pts` | `SMALLINT` | CSV column PTS |
| `plus_minus` | `DECIMAL(6,1)` | CSV column PLUS_MINUS (values like 8.0 / -8.0) |

### `box_team_advanced`

one row per (game, team) -- exactly 2 rows per game, 21,498 rows

Kaynak: `nba_data/{season}/{game_id}/box_scores/{game_id}box_score_advanced_team.csv`  
Birincil anahtar: `game_id`, `team_abbreviation`

| sutun | tip | kaynak |
|---|---|---|
| `game_id` | `CHAR(10) NOT NULL` | GAME_ID. INDEX: PRIMARY KEY (game_id, team_abbreviation) -- 'WHERE game_id IN (...)'. |
| `team_abbreviation` | `VARCHAR(5) NOT NULL` | TEAM_ABBREVIATION, read by header NAME. Same permuted identity block as the traditional_team file (OLD 29 cols: TEAM_NAME,TEAM_ABBREVIATION,TEAM_CITY; NEW 30 cols: TEAM_CITY,TEAM_NAME,TEAM_ABBREVIATION + teamSlug). NOTE the boundary is OFF BY ONE vs the traditional pair: advanced/advanced_team flip to NEW at 2024_2025 game 0022401084, traditional/traditional_team at 0022401085, so game 0022401084  ... |
| `pace` | `DECIMAL(6,2)` | CSV column PACE (NOT E_PACE and NOT PACE_PER40) |
| `off_rating` | `DECIMAL(6,2)` | CSV column OFF_RATING (NOT E_OFF_RATING -- E_* is garbage in the NEW files, e.g. 563.1) |
| `def_rating` | `DECIMAL(6,2)` | CSV column DEF_RATING |
| `net_rating` | `DECIMAL(6,2)` | CSV column NET_RATING |
| `ts_pct` | `DECIMAL(5,3)` | CSV column TS_PCT |
| `efg_pct` | `DECIMAL(5,3)` | CSV column EFG_PCT |
| `poss` | `SMALLINT` | CSV column POSS (cast via float -- '115' in OLD files, '111.0' in NEW) |
| `pie` | `DECIMAL(5,3)` | CSV column PIE |
| `ast_pct` | `DECIMAL(5,3)` | CSV column AST_PCT |
| `oreb_pct` | `DECIMAL(5,3)` | CSV column OREB_PCT |
| `dreb_pct` | `DECIMAL(5,3)` | CSV column DREB_PCT |
| `reb_pct` | `DECIMAL(5,3)` | CSV column REB_PCT |
| `tm_tov_pct` | `DECIMAL(6,3)` | CSV column TM_TOV_PCT (values are percents, e.g. 14.8 / 21.7). Take TM_TOV_PCT, NOT E_TM_TOV_PCT, which sits immediately before it in both variants. DB name is tm_tov_pct, not tov_pct. |

### `box_player_traditional`

one row per (game, player) including players who dressed but did not play; ~24-32 rows per game, ~277k rows total

Kaynak: `nba_data/{season}/{game_id}/box_scores/{game_id}box_score_traditional.csv (the PLAYER-level file, not _team)`  
Birincil anahtar: `game_id`, `player_id`

| sutun | tip | kaynak |
|---|---|---|
| `game_id` | `CHAR(10) NOT NULL` | GAME_ID. INDEX: PRIMARY KEY (game_id, player_id) -- joined to games, game_dates and game_summary in player_source.load_player_games and to games in availability_features.load_absences. |
| `player_id` | `BIGINT NOT NULL` | PLAYER_ID (aliased to person_id in Python and cast to int64). Same identity space as play_by_play.person_id and player_game_impact.person_id. INDEX: part of PRIMARY KEY; add KEY idx_bpt_player (player_id) for the merge on (game_id, person_id). |
| `player_name` | `VARCHAR(255)` | PLAYER_NAME. Present in BOTH variants but at different positions: field 6 in the OLD 29-col variant, the LAST field in the NEW 35-col variant. Read by name. (Do not rebuild it from firstName+familyName -- those exist only in NEW.) |
| `team_abbreviation` | `VARCHAR(5) NOT NULL` | TEAM_ABBREVIATION (aliased to `team`). Compared directly against game_summary.home_abbr/away_abbr, so identical vocabulary and collation. INDEX: KEY idx_bpt_team_game (team_abbreviation, game_id). |
| `comment` | `VARCHAR(64) NULL` | COMMENT (upper case in BOTH variants of the traditional player file -- the lower-case rename to `comment`/`position` happens only in the ADVANCED player file, which this table does not use). MUST stay nullable with blank loaded as NULL or ''; the availability flag is comment.strip() != '' ('DNP - Coach's Decision', 'DND - Injury/Illness', 'DND - Rest', 'NWT - Not With Team', ...). A NOT NULL defau ... |
| `min` | `VARCHAR(16) NULL` | MIN. STRING, not numeric -- parse_minutes splits on ':'. Two encodings exist: clean 'MM:SS' and the dotted 'MM.000000:SS' (2019_2020..2023_2024 in full plus 2024_2025 games 0022400490-0022400512). float(mins) parses both, so either can be stored, but normalising to 'MM:SS' on load is safer. `min` is a MariaDB function name; player_source backticks it. |
| `fgm` | `SMALLINT` | CSV column FGM (cast via float -- OLD files render counting stats as '6.0') |
| `fga` | `SMALLINT` | CSV column FGA |
| `fg3_m` | `SMALLINT` | CSV column FG3M -> DB fg3_m |
| `fg3_a` | `SMALLINT` | CSV column FG3A -> DB fg3_a |
| `ftm` | `SMALLINT` | CSV column FTM |
| `fta` | `SMALLINT` | CSV column FTA |
| `oreb` | `SMALLINT` | CSV column OREB |
| `dreb` | `SMALLINT` | CSV column DREB |
| `reb` | `SMALLINT` | CSV column REB |
| `ast` | `SMALLINT` | CSV column AST |
| `stl` | `SMALLINT` | CSV column STL |
| `blk` | `SMALLINT` | CSV column BLK |
| `to` | `SMALLINT` | CSV column TO. Physical name is `to` (reserved word, backticked at the call site). |
| `pf` | `SMALLINT` | CSV column PF |
| `pts` | `SMALLINT` | CSV column PTS |
| `plus_minus` | `DECIMAL(5,1)` | CSV column PLUS_MINUS |

### `training_games`

one row per game (10,749 rows); schedule-derived rest columns that exist in NO raw CSV but are exactly reconstructible

Kaynak: `derived from game_ids/game_id_{season}.csv (GAME_DATE + MATCHUP); optionally lifted verbatim from output/engineered_dataset_db.pkl`  
Birincil anahtar: `game_id`

| sutun | tip | kaynak |
|---|---|---|
| `game_id` | `CHAR(10) NOT NULL` | GAME_ID. INDEX: PRIMARY KEY -- 'WHERE game_id IN (...)' merged how='left' onto the master frame. |
| `home_rest` | `SMALLINT` | derived: sort all games by (game_date, game_id); reset the per-team last-played map AT EVERY SEASON BOUNDARY; home_rest = (game_date - that home team's previous game_date in the SAME season).days, and 0 when the team has no previous game that season. VERIFIED: reproduces output/engineered_dataset_db.pkl on 10,749/10,749 rows (without the per-season reset it only matches 98.9%). |
| `away_rest` | `SMALLINT` | derived: identical rule applied to the away team. VERIFIED 10,749/10,749. |
| `rest_diff` | `SMALLINT` | derived: home_rest - away_rest, EXCEPT forced to 0 whenever either side is a season opener (home_rest == 0 or away_rest == 0) -- 143 such rows, all 0 in the snapshot. VERIFIED 10,749/10,749; the naive home_rest - away_rest is wrong on 16 rows. |
| `home_b2b` | `TINYINT` | derived: 1 when home_rest == 1 else 0 (calendar-day gap of one, i.e. played yesterday). VERIFIED 10,749/10,749. Season openers are 0. |
| `away_b2b` | `TINYINT` | derived: 1 when away_rest == 1 else 0. VERIFIED 10,749/10,749. |

### `play_by_play`

one row per play-by-play action row; 5,294,643 rows across 10,749 games. (game_id, action_number) is NOT unique -- one event is split across two rows sharing an action_number, which is exactly what db_build_derived's self-join exploits.

Kaynak: `nba_data/{season}/{game_id}/play_by_play/{game_id}pbp.csv (camelCase header, byte-identical across all 9 seasons)`  
Birincil anahtar: `row_id`

| sutun | tip | kaynak |
|---|---|---|
| `row_id` | `BIGINT NOT NULL AUTO_INCREMENT` | derived: surrogate key, no CSV column. Assign in file order (which is already chronological). Required by db_build_derived POPULATE_EVENTS ('o.row_id <> d.row_id') and it is what makes (game_id, action_number) legally non-unique. INDEX: PRIMARY KEY. |
| `game_id` | `CHAR(10) NOT NULL` | CSV gameId -> game_id. Always equals the directory name. INDEX: KEY idx_pbp_game_action (game_id, action_number) -- required by load_pbp's chunked 'WHERE game_id IN (200 ids) ORDER BY game_id, action_number', by the clutch join, and by the defensive-event self-join. |
| `action_number` | `INT NOT NULL` | CSV actionNumber -> action_number. NOT monotone in file order (10,366 of 10,749 games contain a decrease) and NOT unique within a game -- do not make it a key on its own and do not re-sort the CSV by it on load; preserve file order via row_id. Must be non-null on every row (it is the count column in per_game_clutch). INDEX: second column of idx_pbp_game_action. |
| `clock` | `VARCHAR(16) NOT NULL` | CSV clock -> clock. Store the ISO-8601 duration string VERBATIM, zero-padded: 'PT12M00.00S', 'PT00M06.80S'. NEVER TIME or numeric -- the SQL filter is "clock REGEXP '^PT0[0-4]M'" and dropping the leading zero silently returns an empty clutch frame. It is time REMAINING and counts down. |
| `period` | `TINYINT NOT NULL` | CSV period -> period. Integer 1-4 regulation, 5-8 overtime. Must be numeric, or the 'period >= 4' filter compares lexically. INDEX: KEY idx_pbp_period (period) -- the clutch query's only sargable predicate. |
| `team_id` | `BIGINT NOT NULL` | CSV teamId -> team_id. 0 on non-team rows (period start/end), never blank. |
| `team_tricode` | `VARCHAR(5) NULL` | CSV teamTricode -> team_tricode. Blank cells (~47 per game: period start, timeouts) MUST load as NULL, not ''. per_game_clutch drops them with .notna(), and the defensive self-join predicate 'o.team_tricode <> d.team_tricode' would match empty-string pairs. Same tricode vocabulary as game_summary.home_abbr/away_abbr. |
| `person_id` | `BIGINT NOT NULL` | CSV personId -> person_id. 0 sentinel on non-player rows, never blank. This is the impact-score key (commit 'Key the impact score on person id, not surname') and must match box_player_traditional.player_id. INDEX: KEY idx_pbp_person (person_id). |
| `player_name` | `VARCHAR(255) NULL` | CSV playerName -> player_name (surname only, e.g. 'Gasol'). Blank MUST load as NULL: POPULATE_EVENTS guards with 'd.player_name IS NOT NULL' / 'o.player_name IS NOT NULL', and an empty string passes that guard and manufactures bogus defensive events. |
| `player_name_i` | `VARCHAR(255) NULL` | CSV playerNameI -> player_name_i (e.g. 'M. Gasol'). Blank -> NULL. |
| `x_legacy` | `SMALLINT` | CSV xLegacy -> x_legacy |
| `y_legacy` | `SMALLINT` | CSV yLegacy -> y_legacy |
| `shot_distance` | `SMALLINT` | CSV shotDistance -> shot_distance (0 on non-shot rows) |
| `shot_result` | `VARCHAR(8) NULL` | CSV shotResult -> shot_result ('Made'/'Missed'/blank). Blank -> NULL. |
| `is_field_goal` | `TINYINT NOT NULL` | CSV isFieldGoal -> is_field_goal (0/1) |
| `score_home` | `SMALLINT NULL` | CSV scoreHome -> score_home. MUST stay SPARSE: blank on ~74% of rows and loaded as NULL, because the consumers forward-fill within game_id. Never back-fill on load. Never one-sided (0 cases of one filled and one blank in 5.29M rows). |
| `score_away` | `SMALLINT NULL` | CSV scoreAway -> score_away. Same sparse/NULL rule. |
| `points_total` | `SMALLINT NOT NULL` | CSV pointsTotal -> points_total (0 on non-scoring rows) |
| `location` | `CHAR(1) NULL` | CSV location -> location ('h' = home side, 'v' = visitor, blank ~2.2% -> NULL). Independent cross-check on game_summary.home_abbr (225/225 sampled games agreed with the MATCHUP parse). |
| `description` | `VARCHAR(512) NULL` | CSV description -> description. Free text, RFC-4180 quoted with embedded commas on some rows -- use a real CSV reader, not comma splitting. Matched with LIKE '%BLOCK%'/'%STEAL%' by db_build_derived and searched for the uppercase substring 'MISS' by clutch_features. |
| `action_type` | `VARCHAR(32) NULL` | CSV actionType -> action_type. Closed vocabulary that must be preserved VERBATIM and case-sensitively: 'Made Shot', 'Missed Shot', 'Turnover', 'Free Throw', 'Heave', 'Rebound', 'Jump Ball', 'period', ... Blank -> NULL. |
| `sub_type` | `VARCHAR(64) NULL` | CSV subType -> sub_type. SELECTed by clutch_features but never used downstream -- the column must exist or the query errors. |
| `shot_value` | `TINYINT NULL` | CSV shotValue -> shot_value (0/2/3). Supplies the 2-or-3 for made field goals; nullable (consumers to_numeric-coerce and fillna(2.0)). |
| `action_id` | `BIGINT NOT NULL` | CSV actionId -> action_id |

## Turev tablolar - yukleyici bunlara dokunmamali

- **`player_game_impact`** - prediction_engines/db_build_derived.py -> build_impact_table() (line 170): DROP TABLE IF EXISTS, then DDL_PLAYER_IMPACT (lines 54-66), then executemany INSERT of rows read from game_impact_cache_v4.pkl. Run: py prediction_engines/db_build_derived.py --only impact. Do NOT hand-build this table -- the loader must leave it alone.

- **`pbp_defensive_event`** - prediction_engines/db_build_derived.py -> build_event_table() (line 193): DDL_DEFENSIVE_EVENT (lines 68-86), TRUNCATE, then POPULATE_EVENTS (lines 91-121), an INSERT...SELECT off a play_by_play self-join on (game_id, action_number) with row_id and team_tricode inequality, ON DUPLICATE KEY UPDATE. Run: py prediction_engines/db_build_derived.py --only events. Do NOT hand-build; it is pure SQL over play_by_play and will diverge if reconstructed from CSV.

## Yerel CSV'nin karsilayamadigi sutunlar

### `play_by_play.row_id`

Okuyan: prediction_engines/db_build_derived.py POPULATE_EVENTS, predicate 'o.row_id <> d.row_id'

No pbp.csv column corresponds to it. It is a surrogate the old schema owned, and it is load-bearing: it is the only thing that lets two rows share one (game_id, action_number) and still be told apart. Not restorable as data, but fully synthesizable -- BIGINT AUTO_INCREMENT assigned in CSV file order (file order is already chronological; do NOT sort by action_number, which decreases at least once in 10,366 of 10,749 games).

### `player_game_impact.impact (and person_id, player_name, team_abbr on that table)`

Okuyan: prediction_engines/player_source.py:148 attach_impact ('SELECT i.game_id, i.person_id, i.impact FROM player_game_impact i JOIN games g ...'); availability_features.add_availability_features consumes the same person_id/impact series

No CSV anywhere in the repo contains an impact score. The values exist only in game_impact_cache_v4.pkl (present in the repo root, ~217,002 rows), which db_build_derived.py replays into the table. If that pickle is ever lost the column cannot be rebuilt from CSV -- it would have to be recomputed from play-by-play by the impact engine. Availability features degrade silently to all-zero without it, rather than erroring.

### `games.season and game_summary.season`

Okuyan: db_source._season_filter ('AND g.season IN (...)'), available_seasons, player_source._all_seasons, clutch_features.load_clutch_plays, availability_features.load_absences, db_build_derived.verify (GROUP BY g.season + correlated g2.season = g.season)

No per-game raw CSV has a season column. It is recoverable deterministically from the game_ids/ filename, the nba_data/{season}/ directory name, or the 002+YY game-id prefix -- so this is a real absence of a stored field, not an unrecoverable gap. Flagged because both tables must carry it and the two copies must agree exactly: load_master_frame SELECTs game_summary.season while filtering on games.season, so any drift makes filtering and reporting disagree. (data_retriaval/game_id_*.csv has a literal SEASON_ID but only for 4 of the 9 seasons.)

### `play_by_play.score_home / score_away for 6 specific games, and the running score for 415 more`

Okuyan: clutch_features.per_game_clutch (forward-fill then abs(score_home - score_away) <= 5)

The column loads fine, but the CSV's own values are defective and no other local file can repair them at play level. 2017_2018/0021700025 starts at 79-103 on its 'Start of 1st Period' row; 2021_2022/0022100016, 2021_2022/0022100467 and 2022_2023/0022200466 have nonzero start-of-game scores; 2017_2018/0021701109 and 2021_2022/0022100880 open with a pre-tip row carrying blank scores, leaving leading NaN after ffill. A further 415 games (3.9%) step BACKWARDS after ffill because of swapped free-throw rows. The team box score fixes the FINAL score (and is what game_summary uses) but cannot reconstruct the play-level running score, so clutch features must guard against a non-monotone margin rather than expect clean data.

## Denetim bulgulari

Sema uc bagimsiz denetciye curutulmek uzere verildi. Bulduklari:

### [blocking] wrong_type

play_by_play.action_type is declared VARCHAR(32), but the corpus contains an actionType value of 40 characters: the literal string 'Foul' padded with 36 trailing spaces. Under MariaDB's default sql_mode (STRICT_TRANS_TABLES) the INSERT raises "Data too long for column 'action_type'" and aborts the loading chunk; under non-strict mode it silently truncates to 32 chars, which also contradicts the spec's own instruction that action_type be 'preserved VERBATIM'. This is the only over-width text value in the entire play-by-play corpus -- I checked every declared text width across all 5,294,643 rows in all 10,749 pbp files (description max 85 vs 512, subType 34 vs 64, clock 11 vs 16, shotResult 6 vs 8, teamTricode 3 vs 5, playerName 18 vs 255, playerNameI 21 vs 255, location 1 vs 1) and every box_player_traditional width (COMMENT max 40 vs 64, MIN max 12 vs 16, PLAYER_NAME max 24 vs 255, TEAM_ABBREVIATION max 3 vs 5) -- all of those fit. Fix: widen to VARCHAR(40)+ or rstrip the field on load.

Kanit: `nba_data/2022_2023/0022200871/play_by_play/0022200871pbp.csv:93 -- literal row: 0022200871,130,PT01M08.00S,1,1610612744,GSW,1629660,Jerome,T. Jerome,0,0,0,,0,,,0,v,Jerome Transition Take Foul (P1.T4) (D.Guthrie),'Foul' + 36 trailing spaces (len 40),Transition Take,1,0,92 -- vs spec 'shot/action_type sql_type VARCHAR(32)'`

### [minor] wrong_type

play_by_play.action_type is declared VARCHAR(32), but the source data contains a 40-character actionType value. A full scan of all 10,749 pbp.csv files (5,294,643 rows) found 16 distinct actionType values, of which one exceeds 32 characters: the literal 'Foul' padded with 36 trailing spaces. MariaDB truncates over-length trailing spaces with a warning rather than an error, so the load will not fail and no consumer breaks (PAD SPACE collation makes 'Foul   ' = 'Foul', and every code-side test is an equality or IN against short literals), but the stored value is silently altered, which contradicts the spec's own instruction that this column be 'preserved VERBATIM'. VARCHAR(40) or wider removes the truncation entirely. No other pbp text column is at risk: description max 85 (VARCHAR(512)), sub_type max 34 (VARCHAR(64)), player_name max 18, player_name_i max 21, clock max 11, team_tricode max 3, shot_result max 6.

Kanit: `C:/Users/arcan/OneDrive/Desktop/NBA/nba_match_prediction/nba_data/2022_2023/0022200871/play_by_play/0022200871pbp.csv:93 -- literal field between the description and subType columns: `Foul                                    ` (4 characters + 36 trailing spaces = 40). Full row: 0022200871,130,PT01M08.00S,1,1610612744,GSW,1629660,Jerome,T. Jerome,0,0,0,,0,,,0,v,Jerome Transition Take Foul (P1.T4) (D.Guthrie),Foul<36 spaces>,Transition Take,1,0,92`

### [minor] other

The spec describes game_impact_cache_v4.pkl as an '8.1 MB pickle in the repo root'. The actual file is 10,777,775 bytes (10.28 MiB). The companion row-count claim ('~217,002 rows') is NOT falsified: the pickle is a dict of 10,749 games holding 220,098 player entries in total, and db_build_derived.impact_rows() drops entries with no person id or a team that does not match either side of the game, so ~217k surviving rows is consistent. Only the size figure is wrong; it has no effect on the mapping, but it is the kind of detail someone uses to confirm they have the right file.

Kanit: `C:\Users\arcan\OneDrive\Desktop\NBA\nba_match_prediction\game_impact_cache_v4.pkl -- os.path.getsize() == 10777775 (10.28 MiB); pickle.load -> dict of 10749 games, sum(len(v['players'])) == 220098; consumed by prediction_engines/db_build_derived.py:52 IMPACT_CACHE and :170 build_impact_table`

### [minor] missing_table

The derived table player_game_impact lists depends_on = [game_summary, games, game_impact_cache_v4.pkl], but build_impact_table() gets its master frame from db_source.load_master_frame(), which unconditionally queries three further base tables. Running `py prediction_engines/db_build_derived.py --only impact` against a database rebuilt to only the stated dependencies raises a 'table doesn't exist' error before a single impact row is written. Anyone using depends_on as the build order will hit this; the failure is loud, not silent, which is why this is minor rather than blocking. The dependency list should also name box_team_traditional, box_team_advanced and training_games.

Kanit: `prediction_engines/db_build_derived.py:171 `master = db.load_master_frame(verbose=False)` -> prediction_engines/db_source.py:257 `f"SELECT game_id, team_abbreviation, {trad_cols} FROM box_team_traditional "`, db_source.py:261 `f"SELECT game_id, team_abbreviation, {adv_cols} FROM box_team_advanced "`, db_source.py:271 `f"FROM training_games WHERE game_id IN ({placeholders})"` — all three are unguarded reads on the only code path build_impact_table takes.`

## Iki denetci ayni bulguya farkli siddet verdi

`play_by_play.action_type` icin biri "blocking" biri "minor" dedi. Ikisi de ayni
tek degeri buldu: 2022_2023/0022200871 dosyasinda 36 bosluk ile doldurulmus
40 karakterlik bir `'Foul'`. Fark, MariaDB'nin ne yapacagi konusunda.

Dogru olan "minor": MariaDB sondaki bosluklari kirparken uyari verir, hata
vermez, ve PAD SPACE karsilastirmasi `'Foul   ' = 'Foul'` yaptigi icin hicbir
tuketici bozulmaz. Yine de `VARCHAR(48)` secmek bedava, o yuzden secilsin.

Bu satiri buraya boyle koyuyorum cunku denetimin isi anlasmak degil; iki
bagimsiz bakisin ayni yere bakip farkli sonuca varmasi, tek bir bakisin
verecegi cevaptan daha bilgilendirici.

## Kendi dogrulamamdaki acik

Takim box score basliklarinin tum sezonlarda ayni oldugunu soylemistim.
Degil: 2017-18, 2018-19 ve 2025-26 26 sutunlu (`TEAM_CITY, TEAM_NAME,
TEAM_ABBREVIATION, teamSlug`), 2019-20 ile 2024-25 arasi 25 sutunlu
(`TEAM_NAME, TEAM_ABBREVIATION, TEAM_CITY`). 2017 ile 2025'i karsilastirmistim
ve ikisi de ayni varyant.

Kod etkilenmiyor - `csv_master` sutunlari isimle seciyor ve 190/190 birebir
dogrulama dokuz sezonun tamamini kapsiyor. Ama konumla yukleyen bir yukleyici
`team_abbreviation` sutununa sessizce `'Raptors'` yazar. Yukleyiciyi yazan
kisi bunu bilmeli; bu yuzden burada.

## rest_diff: eski veri kendi icinde tutarsizdi

`training_games`'in sakladigi `rest_diff`, taraflardan biri sezonun ilk macini
oynuyorsa 0 yaziyordu - kendi `home_rest` ve `away_rest` degerleri baska bir
sey soylese bile. 10.749 macin 16'sinda boyle. `csv_master.add_rest_columns`
bunu takvimden yeniden hesapliyor ve 16'sinda da tutarli.

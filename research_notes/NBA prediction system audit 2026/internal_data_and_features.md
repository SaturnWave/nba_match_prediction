# Internal audit: data pipeline and pre-game feature construction (nba_match_prediction, as of 2026-10-25)

Scope and conventions. This is a read-only audit of the local repository (branch `Master`, with the uncommitted 2026-10-24 work present). Sources are repo files cited as path plus line numbers; links are relative to this notes file, so `../../` is the repo root. "Measured" means a quick read-only Python check by this researcher on the named cached file, using `py` with PYTHONIOENCODING=utf-8. No rebuilds, training or downloads were run, and only about 10 per-game CSVs were sampled from `nba_data/`. Finding IDs (A1, B2, ...) link each Cited Finding to its severity and fix under Inferences. Severity scale:
- **High:** blocks use on 2026-27 games, or materially biases the reported accuracy.
- **Medium:** a measurable loss of prediction quality or reliability.
- **Low:** hygiene.

## 1. Are all model features truly available before tip-off?

### Takeaway
The rolling, season-average, streak and head-to-head features are clean: everything is shifted by one game, and the project verified this. The remaining problem is the 24 pre-game roster and matchup features, which carry 26-30% of the production models' gain.
- **How they are built:** they are computed from the post-game box score of the very game being predicted.
- **What that encodes:** player availability as finally settled at tip-off. That includes active players later marked "DNP - Injury/Illness", 66% of whom are 20+ minute rotation players.
- **What it is not:** a leak of the result. It is a train/serve timing gap, plus a small residual leak, relative to what any forecaster has from the official injury reports.

### Cited Findings
- **[A1] The roster block chooses its players from the game being predicted.**
  - `BOX_PLAYER_PATH` points at `phonedb_cache/box_player_traditional.pkl` ([pregame_roster.py L83](../../prediction_engines/pregame_roster.py)).
  - `load_listings` sets `unavailable` from that game's `comment` field ([L120-133](../../prediction_engines/pregame_roster.py)).
  - The sweep takes `sheet = listed_by_game.get(gid)` for the game itself and keeps `dressed` = listed players without an unavailability tag ([L288, L299-302](../../prediction_engines/pregame_roster.py)).
  - Every metric is computed over that `dressed` set ([`side_features` L206-257](../../prediction_engines/pregame_roster.py)): roster_impact_l10_mean/sum, roster_form_l6/l3, matchup_def_quality/edge, roster_avail_minutes and roster_missing_impact.
  - The docstring concedes the listing is the post-game one: "a player scratched minutes before tip-off appears as unavailable here" ([L33-37](../../prediction_engines/pregame_roster.py)).
  - These are 24 of the 196 production features ([REPLACED_COLUMNS + EXTRA_COLUMNS, L99-105](../../prediction_engines/pregame_roster.py); [models/feature_list_2025_26.json](../../models/feature_list_2025_26.json)).
- **[A2] Weight of these 24 features in the shipped models (measured, 10 ensemble members each):** 30.3% of total split gain in `home_win_ensemble_2025_26.pkl` and 26.3% in `point_diff_ensemble_2025_26.pkl`, from 12% of the features ([models/](../../models/)).
  - Box-score rolling and season averages take 59.5% / 64.8% of the gain.
  - Streak and win% 5.5% / 3.9%; team-impact rolling 2.9% / 3.8%; head-to-head 1.8% / 1.1%.
- **[A3] "DNP - Coach's Decision" is handled correctly.**
  - It is deliberately left unmatched so the player counts as dressed ([L89-93](../../prediction_engines/pregame_roster.py); [DURUM.md L49-51](../../DURUM.md)).
  - All 40,422 such rows (2017-18 to 2025-26) are the only non-blank comment the regex leaves unmatched ([measured: box_player_traditional.pkl](../../phonedb_cache/box_player_traditional.pkl)).
  - Dressed players per team-game average 12.2-12.8 by season (measured).
  - The project measured the dressed count's correlation with |margin| at -0.015, against 0.64 for the old "who appeared" roster ([DURUM.md L40-43, L52-53](../../DURUM.md); [experiments/2026-10/README.md L17-18](../../experiments/2026-10/README.md)).
- **[A4] "DNP - Injury/Illness" is treated as not dressed even though the player was on the active list.**
  - The keyword regex `Injur|Illness|Rest|Health|Not With Team|NWT|Suspen|Personal|DND|Concussion|Trade` ([L91-93](../../prediction_engines/pregame_roster.py)) matches it.
  - Measured: 1,957 such rows; 1,268 (66.5%) belong to players averaging 20+ minutes that season, and 613 to 28+ minutes. 1,039 of 21,498 team-games (4.8%) contain one ([box_player_traditional.pkl](../../phonedb_cache/box_player_traditional.pkl)). Season minutes include the whole season and are used only as a role proxy.
  - For contrast: 5,675 DND/NWT rows, 3,863 of them 20+ minute players.
  - The module's own logic says DND/NWT entries are "announced on the injury report before the game" and DNP entries are "decided during the game" ([L20-31](../../prediction_engines/pregame_roster.py)). It applies that split only to coach's decisions.
- **[A5] When availability is officially known.** Teams must give a participation status by 5 p.m. local time the day before a game, or by 1 p.m. on game day for the second game of a back-to-back. They submit a game-day report between 11 a.m. and 1 p.m. local (8-10 a.m. for tip-offs at 5 p.m. or earlier), and reports are "updated on a continual basis throughout the day", with snapshots timestamped every 15 minutes ([NBA Official: NBA Injury Report page](https://official.nba.com/nba-injury-report-2020-21-season); this URL currently serves the 2026-27 season). The repo does not ingest injury reports; DURUM lists a pre-game injury report as the first next step ([DURUM.md L509-510](../../DURUM.md)).
- **[A6] Rolling, season, streak and head-to-head features use only prior games.**
  - Rolling: `rolling(w, min_periods=1).mean().shift(1)` per team in date order ([predict_2025_2026.py L413-422, L474-475](../../prediction_engines/predict_2025_2026.py)).
  - Streak: uses `won.shift(1)` ([L428](../../prediction_engines/predict_2025_2026.py)).
  - Season average and win%: expanding means shifted one row ([L478-483](../../prediction_engines/predict_2025_2026.py)).
  - Head-to-head: past meetings with `game_date < d`, strictly earlier ([L503-509](../../prediction_engines/predict_2025_2026.py)).
  - The project's `verify_shift.py` recomputed these from raw scores for all 21,468 team-games and matched exactly ([experiments/2026-10/README.md L16](../../experiments/2026-10/README.md)).
  - Final scores, plus/minus and same-game box columns are excluded by the selection whitelist ([L771-784](../../prediction_engines/predict_2025_2026.py)). The production list contains only rolling, season, streak, H2H, roster and matchup families (measured grouping: 126 / 39+2 / 3 / 2 / 18 / 6).
- **[A7] The selection whitelist matches by substring:** `"roster_impact" in c or "roster_form" in c or "matchup_" in c` ([L782](../../prediction_engines/predict_2025_2026.py)). Any future same-game column whose name contains "matchup_" would be picked up automatically. Production itself is pinned by the JSON list ([DURUM.md L65-67](../../DURUM.md)).
- **[A8] The leaky "observed roster" dataset still exists under the same column names and is still a default.**
  - DURUM describes it as the old dataset with "same column names, different meaning; do not mix with the models" ([DURUM.md L474-475](../../DURUM.md)).
  - It is the default input of [staleness.py L48](../../prediction_engines/staleness.py), [seed_ensemble.py L43](../../prediction_engines/seed_ensemble.py), [training_protocol.py L363](../../prediction_engines/training_protocol.py) and [information_ceiling.py L181](../../prediction_engines/information_ceiling.py). [consistency_lab.py L75](../../prediction_engines/consistency_lab.py) uses it deliberately as the comparison arm.
  - The build still computes the observed roster first ([build_dataset_db.py L165-177](../../prediction_engines/build_dataset_db.py) via [predict_2025_2026.py L710-732](../../prediction_engines/predict_2025_2026.py)), and pregame_roster overwrites it afterwards ([pregame_roster.py L357-377](../../prediction_engines/pregame_roster.py)).
- **[A9] Same-day ordering.**
  - Games are swept in (date, game_id) order ([L268-271](../../prediction_engines/pregame_roster.py)). Per-player state cannot cross games on the same day, because a player plays once a day ([L263-266](../../prediction_engines/pregame_roster.py)).
  - The league-wide fallback for matchup_def_quality ([L238-241, L289](../../prediction_engines/pregame_roster.py)) is updated after every game ([L337-338](../../prediction_engines/pregame_roster.py)). A later game on the same date therefore sees totals from an earlier same-day game. The fallback is used only when no dressed player has matchup history.
- **[A10] Back-to-backs and late-published data.**
  - A day after the 12 games of 2026-10-23, the NBA still had not published their defensive box scores, tracking or matchup data ([DURUM.md L427-430](../../DURUM.md)).
  - Training rows always see complete prior-game inputs: secondary and free-throw assists come only from tracking ([value_engine.py L227-238](../../prediction_engines/value_engine.py)), and matchup totals from the matchup cache ([pregame_roster.py L152-170, L333-342](../../prediction_engines/pregame_roster.py)).
  - The second game of a back-to-back depends on the previous night's game through every shift(1) feature.

#### Done well
- Two real leaks were found, quantified and fixed: the observed roster, and early stopping on the test set. The fix is recorded in [output/metrics_2025_26.json](../../output/metrics_2025_26.json) (`early_stopping` field) and in [DURUM.md L36-63](../../DURUM.md).
- The current game's minutes are used only to update player state after that game's features are written ([pregame_roster.py L315-343](../../prediction_engines/pregame_roster.py)).
- The top-8 sum fixes the player count so it cannot leak ([L50-52, L95, L235-236](../../prediction_engines/pregame_roster.py)).
- A stricter alternative already exists: `availability_features.py` never reads the current game's sheet and uses absences in prior games only ([L18-28, L172-197](../../prediction_engines/availability_features.py)).

### Inferences
- **[A1/A2/A4/A5] Severity High, for interpreting the numbers and for live use.**
  - **What the features know:** box-score availability is hindsight at tip-off. DND, NWT and Inactive tags reflect the final inactive list. DNP-Injury reflects a decision that can be taken at or after tip. Every Questionable, Doubtful or game-time decision is resolved perfectly.
  - **Why it matters:** about 26-30% of the models' gain rests on these features. The honest walk-forward (0.666 classifier / 0.687 blend accuracy) therefore describes a forecaster who already knows the final active list. It is an upper bound for predictions made from the 11 a.m.-1 p.m. report.
  - **Fix, step 1:** fix the prediction timestamp, for example after the inactive list is final.
  - **Fix, step 2:** run two walk-forward arms to bracket the effect. One uses the current rule. The other uses a prefix rule: DND, NWT, Inactive and their underscore codes are unavailable; every "DNP - ..." counts as dressed, which is the active-list rule.
  - **Fix, step 3:** start archiving the official 15-minute injury-report snapshots now. Model availability as trailing minutes x P(plays | status), with status probabilities estimated from the archive.
  - **Fix, step 4:** until a season of snapshots exists, report a "report-time" variant in which Questionable players are down-weighted rather than resolved.
- **[A8] Severity Medium.**
  - **Why it matters:** experiments can silently run on leaky features. Their conclusions, such as how fast the models go stale or which training protocol to use, may not transfer.
  - **Fix:** rename the observed columns (e.g. an `obs_` prefix) or delete `engineered_dataset_db.pkl`. Make every loader assert `blob["roster"] == "pregame"` unless an explicit flag is passed. Re-run or annotate the staleness, seed-ensemble, training-protocol and information-ceiling outputs.
- **[A7] Severity Low.** Fix: select features only from the pinned JSON list. Add a guard test that flags any selected column that is suspiciously predictive of its own game's margin.
- **[A9] Severity Low (negligible).** Fix: compute all features for a date first, then update league aggregates once per date.
- **[A10] Severity Medium for live use, Low for backtests.**
  - **Why it matters:** a back-to-back's second game will be predicted with last night's matchup and tracking inputs missing, which never happens in training.
  - **Fix:** add a completeness gate before each prediction (wait, or degrade consistently). Make `value_engine` recompute recent games once their tracking arrives (see D5).
- **[A6] No action needed.** These features are the soundest part of the pipeline.

### Gaps
- The official inactive-list deadline was not confirmed from an official source. A search-result summary attributed to Wikipedia ("Injured reserve list") says players can be placed on the inactive list up to one hour before tip-off; that page was not opened.
- How many "DNP - Injury/Illness" or DND players were already listed Out on that day's injury report cannot be measured, because the repo has no injury-report archive. The size of the hindsight gap (A4) is therefore bounded, not measured.
- Which dataset produced `output/staleness.json`, `seed_ensemble.json`, `training_protocol.json` and `information_ceiling.json` was not checked.

## 2. Can the system produce features for a game that has not been played yet (2026-27)?

### Takeaway
No.
- **No incremental path:** there is no incremental or as-of feature builder, and the dashboard only scores rows already in the training dataset.
- **Inputs stop at 2025-26:** three inputs behind the production features end there: the v5 impact cache, the player box-score listings, and the team-box cache.
- **No rebuild source:** the first two can only be rebuilt from a phone-database snapshot taken on 2026-09-12, and that database is offline.
- **Consequence:** even the 28 already-played 2026-27 games cannot be turned into features today.
- **Hidden failure mode:** several code paths would turn an unplayed game into zeros instead of failing.

### Cited Findings
- **[B1] The dashboard serves only precomputed rows.**
  - It reads the precomputed dataset ([app.py L59](../../app.py)), and `/api/predict/<game_id>` looks the row up there ([L591-595](../../app.py)).
  - DURUM states that no path builds pre-game features for an unplayed game ([DURUM.md L511-514](../../DURUM.md)), and that such an incremental path is also needed to run the model inside the simulator ([L241-243](../../DURUM.md)).
  - The production dataset is 10,749 rows x 316 columns and ends 2026-04-12 (measured, [output/engineered_dataset_pregame.pkl](../../output/engineered_dataset_pregame.pkl)).
  - Production models were fit on games before 2026-03-04 (n_train 10,441, n_test 308; [output/metrics_2025_26.json](../../output/metrics_2025_26.json)).
- **[B2] Only played games enter the master frame.**
  - `csv_master.load_schedule` reads `game_ids/game_id_<season>.csv` ([csv_master.py L79-111](../../prediction_engines/csv_master.py)). That file is filled from LeagueGameFinder with regular-season `002` games only ([fetch_season_game_ids.py L36-49](../../data_retriaval/fetch_season_game_ids.py)).
  - `game_ids/schedule_2026_2027.csv` holds 1,206 regular-season games, 28 final and 1,178 scheduled (measured). It is read only by [sim_vs_real.py L54](../../prediction_engines/sim_vs_real.py).
- **[B3] An unplayed row that got in would be silently mislabelled.**
  - `_attach_scores` left-joins box scores ([csv_master.py L248-270](../../prediction_engines/csv_master.py)).
  - `home_win = (point_diff > 0).astype(int)` turns a missing result into a home loss ([L225-227](../../prediction_engines/csv_master.py)).
  - The builder then fills every NaN with 0, giving a 0-0 score ([build_dataset_db.py L211](../../prediction_engines/build_dataset_db.py)).
- **[B4] pregame_roster needs the game's own team sheet and degrades silently without one.**
  - Without a sheet it counts `no_sheet` and leaves `dressed` empty ([pregame_roster.py L285, L299-302](../../prediction_engines/pregame_roster.py)).
  - All roster metrics become NaN, then 0.0 ([L376](../../prediction_engines/pregame_roster.py)), so the game is scored as a team with nobody available instead of raising an error.
- **[B5] The impact and roster features depend on `phonedb_cache`, which has no 2026-27 data.**
  - `value_engine` takes each season's game list from `phonedb_cache/games.pkl` ([value_engine.py L146, L549-552, L574](../../prediction_engines/value_engine.py)).
  - It takes assist totals from `box_player_traditional.pkl` and secondary/free-throw assists from `player_tracking.pkl` ([L147-148, L211-239](../../prediction_engines/value_engine.py)).
  - `impact_v5` reads `phonedb_cache/game_summary.pkl` ([impact_v5.py L59, L120](../../prediction_engines/impact_v5.py)), and `pregame_roster` reads `box_player_traditional.pkl` ([L83](../../prediction_engines/pregame_roster.py)).
  - 24 of the 196 features depend on the impact cache ([rebuild_impact_features.py L6-15](../../prediction_engines/rebuild_impact_features.py)), and all 24 roster/matchup features depend on the listings.
  - `phonedb_source.load` serves only its pickle or a live pull ([phonedb_source.py L52-58, L477ff](../../prediction_engines/phonedb_source.py)). The database is unreachable and this clone has no `.env` ([DURUM.md L429-431, L525-526](../../DURUM.md)).
- **[B6] Coverage (measured).**
  - `box_player_traditional.pkl`: 277,127 rows, seasons 2017-18 to 2025-26, pulled 2026-09-12 22:56 ([phonedb_cache/manifest.json](../../phonedb_cache/manifest.json)).
  - `game_impact_cache_v5.pkl`: 10,749 games, none from 2026-27.
  - `team_box_cache_v1.pkl`: 9 seasons, no 2026_2027.
- **[B7] Two non-production families need a live database.** Clutch and availability are skipped when the DB is down ([build_dataset_db.py L179-209](../../prediction_engines/build_dataset_db.py)), so they cannot be rebuilt now.
- **[B8] The simulator does not substitute for the model.** It has a `forecast` mode for unplayed seasons, but by design never uses the trained models ([DURUM.md L181-191](../../DURUM.md)).

#### Done well
- `pregame_roster.build_pregame_features` is already a single chronological state sweep ([L260-351](../../prediction_engines/pregame_roster.py)), the natural basis for an as-of snapshot.
- Because features are computed with shift(1), a correctly appended unlabelled row would already get correct rolling features ([predict_2025_2026.py L413-422](../../prediction_engines/predict_2025_2026.py)).
- `fetch_schedule.py` records tip-off time (UTC), arena, city, neutral site and NBA Cup labels ([L40-64](../../data_retriaval/fetch_schedule.py)), and refuses unknown game-type prefixes ([L65-67](../../data_retriaval/fetch_schedule.py)).
- `fetch_season_game_ids.py --refresh` handles an in-progress season ([L52-69](../../data_retriaval/fetch_season_game_ids.py)).

### Inferences
- **[B1/B2/B5/B6] Severity High.**
  - **Why it matters:** as of 2026-10-25 the production model cannot score any 2026-27 game.
  - **Fix (a):** add CSV readers for player box scores, game lists and tracking from `nba_data/`, behind a V2/V3 normalisation layer. `csv_master` already does this for team boxes. This removes the phone-DB dependency.
  - **Fix (b):** run `value_engine` and `impact_v5` incrementally on new game IDs from `game_ids/`.
  - **Fix (c):** persist pregame_roster's player state, recent-game sets and per-team rolling state as of a cutoff.
  - **Fix (d):** take the target game's sheet from the team roster minus injury-report Outs (weighting Questionables), never from the box score.
  - **Fix (e):** feed unplayed rows through FeatureEngineer with an explicit `is_played` flag, and keep them out of training.
  - **Fix (f):** add a parity test: for already-played games, the as-of builder (given box-score sheets) must reproduce the batch dataset rows exactly.
- **[B3/B4] Severity High (latent).**
  - **Why it matters:** the failure is silent label and feature corruption, not an error.
  - **Fix:** raise on NaN labels for training rows and on empty sheets; never derive `home_win` from a missing score.

### Gaps
- Not verified: whether LeagueGameFinder can list a game in progress, which would let partial box scores in.
- The availability of a usable pre-game roster or injury feed (endpoint choice, licensing, PDF parsing) was not assessed.

## 3. How are new players, rookies, trades and roster changes handled?

### Takeaway
- **Trades:** handled reasonably. Player history is keyed by person ID and follows the player.
- **Gaps:** players with no NBA history are invisible, a player's first values are one-game samples with no prior, and long absences carry stale pre-injury values.
- **Team level:** the rolling features cross the offseason with no notion of roster turnover.
- **Timing:** every one of these hurts most in the first weeks of a season, which is exactly where 2026-27 is now.

### Cited Findings
- **[C1] Players with no history are dropped from every roster metric.**
  - The code skips them (`st = states.get(pid)` ... `continue`, [pregame_roster.py L215-217](../../prediction_engines/pregame_roster.py)) and requires a trailing impact and positive trailing minutes ([L219-221](../../prediction_engines/pregame_roster.py)).
  - A debuting rookie therefore adds nothing to roster_impact_l10_mean/sum, roster_form_l6/l3 or roster_avail_minutes ([L230-237](../../prediction_engines/pregame_roster.py)).
  - The top-8 sum then substitutes the next veteran by trailing minutes ([L235-236](../../prediction_engines/pregame_roster.py)).
- **[C2] No prior or shrinkage after a debut.** Windows are deques of the last 10 appearances, and `trailing_impact` averages whatever exists ([L180-192](../../prediction_engines/pregame_roster.py)). After one game a player's value is a one-game mean.
- **[C3] Size of the rookie problem, from the simulator, which makes the same "no NBA history = replacement level" assumption.**
  - Rookies Boozer, Acuff, Lendeborg, Wilson, Peterson and Wagler got 1-5 minutes in total across two simulated games, against 26-36 minutes per game in reality.
  - Injured players (Ingram, Knueppel, DiVincenzo, Claxton, Raynaud) played in the simulation ([DURUM.md L464-470, L529-531](../../DURUM.md)).
- **[C4] Trades.**
  - State is keyed only by person ID ([L279, L327-342](../../prediction_engines/pregame_roster.py)), so history follows the traded player.
  - His old team keeps him in `recent_by_team` for up to 3 games (`RECENT_GAMES_FOR_MISSING = 3`, [L96, L245-256, L343](../../prediction_engines/pregame_roster.py)). During that time he counts as "missing impact"; after that the loss disappears from the features.
- **[C5] Season starts erase "who is missing".**
  - `recent_by_team` is cleared at each new season ([L293-298](../../prediction_engines/pregame_roster.py)), so the opener's roster_missing_impact is NaN and then 0 ([L256, L376](../../prediction_engines/pregame_roster.py)). Offseason departures never register.
  - Measured: home_roster_missing_impact == 0 in 2,944 of 10,749 games (27%), mixing "nobody missing" with "unknown".
- **[C6] Players returning from long absences carry pre-absence values.**
  - Minutes and impact are appended only for games with minutes > 0 ([L324-332](../../prediction_engines/pregame_roster.py)), and windows count appearances, not calendar time.
  - A player back after 40 games carries his pre-injury last-10 impact and full trailing minutes, ignoring any minutes restriction.
  - The minutes window and the impact window can cover different games: minutes are appended for every appearance, impact only when the player is in the impact cache ([L330-332](../../prediction_engines/pregame_roster.py)).
- **[C7] Matchup quality never decays.** It is a cumulative career-to-date rate ([L333-342](../../prediction_engines/pregame_roster.py)), and the league fallback is cumulative across all seasons ([L282, L337-338](../../prediction_engines/pregame_roster.py)).
- **[C8] Team rolling features cross the offseason.**
  - The 126 rolling features run over each team's full date-ordered history ([predict_2025_2026.py L413-422, L474-475](../../prediction_engines/predict_2025_2026.py)).
  - Season averages at a team's first game carry last season's final value ([rebuild_impact_features.py L19-22](../../prediction_engines/rebuild_impact_features.py)).
  - Measured: the correlation of L10 point margin with the team's final season average is 0.38 at the opener and 0.43 at games 1-3, against 0.75 from game 10 on ([engineered_dataset_pregame.pkl](../../output/engineered_dataset_pregame.pkl)).
- **[C9] Listings changed shape in 2026-27.** Six sampled games list 16-18 players per team (measured), against season means of 12.5-13.4 in 2017-18 to 2025-26 (measured). Two-way and G League players now appear as "Inactive - G League - Two-Way".
- **[C10] Minutes are counted twice in the roster mean.** roster_impact_l10_mean weights each player's per-game impact total ([impact_v5.py L23-25](../../prediction_engines/impact_v5.py)) by his trailing minutes ([pregame_roster.py L232-234](../../prediction_engines/pregame_roster.py)), even though a per-game total already scales with minutes.

#### Done well
- Identity is the person ID, not the surname ([build_dataset_db.py L99-103](../../prediction_engines/build_dataset_db.py)).
- Player state is team-agnostic, so it survives trades.
- DURUM openly measures the rookie and injury failures ([L464-470](../../DURUM.md)).

### Inferences
- **[C1/C2/C3] Severity High for October-November, Medium afterwards.**
  - **Why it matters:** in a season start with rookies playing 26-36 minutes, team strength is systematically understated for rookie-heavy teams. The top-8 sum promotes bench veterans instead.
  - **Fix:** give no-history players a role-based prior. Draft slot, pre-draft rating, or the 2026-27 preseason minutes (67 preseason games are final, measured) would all work.
  - Shrink early values: v = (n x mean + k x prior) / (n + k), with k of about 5-10 games.
- **[C5/C8] Severity Medium.**
  - **Fix:** add a roster-continuity feature (share of last season's minutes still on the roster).
  - Pull team rolling values toward a roster-based prior at the season start.
  - Give "unknown" its own indicator (or NaN, which LightGBM handles natively) instead of 0.
- **[C6] Severity Medium-Low.** Fix: add games or days since the last appearance, decay windows by time, and cap returning players' minutes, e.g. at the median minutes of other recent returns.
- **[C7] Severity Low.** Fix: decay matchup totals exponentially, e.g. with a half-life of one season.
- **[C4] Severity Low.** Acceptable as is. Optionally model trades explicitly instead of a 3-game "missing" tail.
- **[C10] Severity Low.** Fix: use per-36 impact x expected minutes, or rely on the existing top-8 sum.

### Gaps
- The effect of rookies on actual 2026-27 predictions cannot be measured until 2026-27 features exist.
- How often mid-season trades occur in the data was not measured.

## 4. Data quality and freshness

### Takeaway
- **Strengths:** retrieval is resumable and coverage of past seasons is essentially complete.
- **Completeness checks:** they test file existence only.
- **V2 to V3 switch:** handled by renaming columns without validation. It has already changed the meaning of the comment field the roster features depend on (new "Inactive - ..." vocabulary in 2026-27).
- **Phone database:** the snapshot that the impact and roster features need is frozen at 2026-09-12, and the loader that built it is lost.
- **Legacy code:** the repo carries many superseded scripts, datasets and caches that still look runnable.

### Cited Findings
- **[D1] The comment vocabulary changed in 2026-27, and the regex does not cover it.**
  - In 2017-18 to 2025-26, comments use DND, DNP and NWT prefixes; "Inactive" appears once in nine seasons (measured, [box_player_traditional.pkl](../../phonedb_cache/box_player_traditional.pkl)).
  - In six sampled 2026-27 games (0022600003, 0022600086, 0022600093, 0022600094, 0022600097, 0022600101; measured):
    - "Inactive - Injury/Illness": 31
    - "DNP - Coach's Decision": 19
    - "Inactive - G League - Two-Way": 7
    - "Inactive - Coach's Decision": 3
    - "Inactive - Not With Team": 1
  - `UNAVAILABLE_RE` ([pregame_roster.py L91-93](../../prediction_engines/pregame_roster.py)) matches neither "Inactive - G League - Two-Way" nor "Inactive - Coach's Decision". 10 of 42 inactive tags would count as dressed, in 5 of 12 team-games; MIL in 0022600086 has 3.
  - `availability_features.INJURY_PATTERN` uses the same keyword approach ([L58](../../prediction_engines/availability_features.py)).
- **[D2] The V3 remap renames columns without validating them.**
  - [retrieve_2025_2026_data.py L47-73](../../data_retriaval/retrieve_2025_2026_data.py) renames V3 camelCase to the V2 names because V2 stopped serving 2025-26 ([L5-10](../../data_retriaval/retrieve_2025_2026_data.py)). The same path backfills gaps in earlier seasons ([L20-22](../../data_retriaval/retrieve_2025_2026_data.py)).
  - Measured on samples 0022301187 (2023-24), 0022501190 (2025-26) and 0022600086 (2026-27):
    - V3 files add teamSlug, firstName, familyName, playerSlug and jerseyNum, and move PLAYER_NAME to the last column.
    - NICKNAME changes meaning, from "Bryce" to "M. Bridges", because `nameI` is mapped to it ([L50](../../data_retriaval/retrieve_2025_2026_data.py)).
    - MIN changes format from "29.000000:43" to "33:55"; `parse_minutes` handles both ([pregame_roster.py L111-117](../../prediction_engines/pregame_roster.py)).
  - Tracking V3 is renamed, and absent columns are silently filled with NA ([backfill_defensive_tracking.py L95-105](../../data_retriaval/backfill_defensive_tracking.py)). Defensive box scores still come from V2 ([L12-17, L137-146](../../data_retriaval/backfill_defensive_tracking.py)).
- **[D3] Completeness is checked by file existence only.**
  - Each retriever treats a file over 64 bytes as complete and never fetches it again ([retrieve_2025_2026_data.py L117-127](../../data_retriaval/retrieve_2025_2026_data.py); [backfill_defensive_tracking.py L108-128](../../data_retriaval/backfill_defensive_tracking.py); [retrieve_matchups.py L59-62](../../data_retriaval/retrieve_matchups.py)).
  - [tools/check_data.py](../../tools/check_data.py) checks only presence and size ([L4-8, L22, L56-58](../../tools/check_data.py)).
  - There are no content checks (team totals against the play-by-play final score, players per team, known comment strings), and post-game stat corrections are never picked up.
- **[D4] The phone-DB snapshot cannot be regenerated.**
  - 12 tables were pulled 2026-09-12 ([manifest.json](../../phonedb_cache/manifest.json)) and cover the 10,749 games up to 2025-26.
  - csv_master records that "whatever loaded the base tables lived on the phone" ([csv_master.py L4-8](../../prediction_engines/csv_master.py)); no code in the repo rebuilds those tables from the CSVs.
  - The production features need box_player_traditional, games, game_summary and player_tracking (see B5).
- **[D5] Data published a day late is not picked up afterwards.**
  - Tracking, defensive and matchup data arrive a day late ([DURUM.md L427-430](../../DURUM.md)).
  - `value_engine` prints a warning and pays no secondary or free-throw assists when tracking is missing ([L227-238](../../prediction_engines/value_engine.py)).
  - It does not recompute cached games without `--refresh` ([L565-575](../../prediction_engines/value_engine.py)), and "a cached game does not say which pricing produced it" ([L569-570](../../prediction_engines/value_engine.py)).
  - The free-throw-assist count roughly doubles from 2019-20 onward because of a recording change ([DURUM.md L275](../../DURUM.md)).
- **[D6] 23 historical games lack defensive box scores and matchups.**
  - By season: 2017-18: 8; 2018-19: 7; 2019-20: 4; 2021-22: 1; 2023-24: 3 ([defensive_box_score_unavailable_games.txt](../../defensive_box_score_unavailable_games.txt)). The NBA still returned nothing on 2026-10-24 ([DURUM.md L433-438](../../DURUM.md)).
  - In pregame_roster these games simply add no matchup totals ([L316-342](../../prediction_engines/pregame_roster.py)), about 0.2% of games.
- **[D7] Schedule and game-ID handling.**
  - Game-ID files keep only `002` regular-season games, and home is detected from " vs. " ([fetch_season_game_ids.py L42-47](../../data_retriaval/fetch_season_game_ids.py)). The 2026-27 file holds 28 games from 2026-10-20 to 2026-10-23 (measured).
  - The NBA Cup final (`006`) is excluded, so it is missing from rest calculations ([csv_master.py L179-206](../../prediction_engines/csv_master.py)).
  - Neutral sites exist in the schedule (`neutral_site`, [fetch_schedule.py L58](../../data_retriaval/fetch_schedule.py)) but not in game_ids or the features. 2026-27 has 3 neutral regular-season games: Mexico City 2026-11-07 IND-DEN, Paris 2027-01-14 SAS-NOP and Manchester 2027-01-17 NOP-SAS. Six Cup games still have undecided teams (measured).
- **[D8] Legacy and duplicate code paths.**
  - `data_retriaval/` holds 14 scripts plus a root `game_data_retriaval.py`. Six reference V2 box-score or tracking endpoints (endpoint grep): retrieve_box_scores.py, test.py, player_track_retriaval.py, retrieve_tracking_data.py, retrieve_nba_data.py, play_by_play_data_retrieve.py. The project says those endpoints return empty for 2025-26 or are gone from nba_api 1.11+ ([retrieve_2025_2026_data.py L5-7](../../data_retriaval/retrieve_2025_2026_data.py); [backfill_defensive_tracking.py L15-16](../../data_retriaval/backfill_defensive_tracking.py)).
  - The README is stale ([README.md](../../README.md)):
    - it tells users to run retrieve_nba_data.py and retrieve_box_scores.py (L87-96);
    - it lists 2019-20 to 2024-25 as the covered seasons (L80);
    - it says to train with `prediction_engines/2023_2024.py` (L112-121), which uses a stub impact (L199-203).
  - `predict_2025_2026.py` is both a legacy entry point and the home of production's FeatureEngineer ([build_dataset_db.py L155, L211](../../prediction_engines/build_dataset_db.py)). Its legacy loader takes the final score from the last play-by-play row with a 0-0 fallback ([L602-616](../../prediction_engines/predict_2025_2026.py)). That is the label bug csv_master documents for game 0021900106 ([csv_master.py L255-259](../../prediction_engines/csv_master.py)).
  - Four dataset generations coexist (`engineered_dataset_2025_26`, `_db`, `_pregame`, `_pregame_v4`), as do impact caches v3, v4 and v5 plus a 5-byte `game_impact_cache.pkl`. `app.py` deliberately still reads v4 ([L63](../../app.py); [DURUM.md L335](../../DURUM.md)).

#### Done well
- Scores and labels come from the team box score, not the last play-by-play row ([csv_master.py L248-260](../../prediction_engines/csv_master.py)).
- Home and away are parsed from the separator; taking the first team as home would reverse 48% of games ([L79-111](../../prediction_engines/csv_master.py)).
- Retrieval is resumable and reports failures instead of swallowing them ([backfill_defensive_tracking.py L176-191](../../data_retriaval/backfill_defensive_tracking.py)).
- DURUM records the exact data gaps ([L422-438](../../DURUM.md)), and `check_data.py` reports per-file coverage.
- In pricing 5.1, passers are found for 99.8-100% of assists, and free-throw counts match the box score in every player-game ([DURUM.md L272, L278-279](../../DURUM.md)).

### Inferences
- **[D1] Severity High for 2026-27.**
  - It is latent today, because no 2026-27 roster features exist yet. It will distort roster_avail_minutes and roster_missing_impact once they do.
  - **Fix:** classify by prefix: DND, NWT, Inactive and the underscore codes mean unavailable; DNP means dressed. Keep a whitelist of known comment strings and fail on unseen ones. Guard the dressed count per team-game (about 13).
- **[D2] Severity Medium.**
  - **Fix:** add a schema contract per file type (required columns, data types, value formats, allowed values) checked on write and read.
  - Stop reusing V2 names for different V3 fields (NICKNAME), and record the source endpoint and version per file.
- **[D3] Severity Medium.**
  - **Fix:** add content validators: player points sum to team points, which equal the play-by-play final; two team rows; at least 8 players with minutes; minutes sum to 240 plus 25 per overtime.
  - Re-fetch the last 2-3 days of games so stat corrections and late data arrive.
- **[D4] Severity High (same root cause as B5).** Fix: write CSV-to-table loaders and treat the phone DB only as an optional speed-up.
- **[D5] Severity Medium.** Fix: key value-cache entries by game ID, pricing version and input-completeness flags, and recompute automatically when late inputs land.
- **[D6] Severity Low.** No action beyond flagging.
- **[D7] Severity Low.** Fix: carry `neutral_site` and `006` games into the master frame and the rest calculation.
- **[D8] Severity Medium**, because of the risk of running a dead or leaky path. Fix: move superseded scripts into `legacy/` or delete them, update the README, and keep one documented refresh entry point. DURUM L492-502 is effectively that entry point already.

### Gaps
- Not verified: whether the phone DB's 2025-26 comments came from the same V3 CSVs. Their DND/NWT vocabulary suggests a different feed or a conversion.
- Not checked: whether V3 advanced metrics share V2's scales. They are unused in production but would matter if pace features were added.
- Not measured: whether any downloaded game was later corrected by the NBA.

## 5. Feature design gaps versus common practice

### Takeaway
The production feature set is 196 team summaries that ignore the opponent, plus a roster block. It has no:
- schedule features (rest, back-to-back, density, travel, time zone, altitude);
- pace-adjusted or "four factors" efficiency, and no free throws;
- opponent adjustment, and no neutral-site flag;
- small-sample shrinkage.

The binding evaluation also never scores a team's first 10 games, so the early-season weakness the system faces right now is invisible in the headline numbers.

### Cited Findings
- **[E1] Feature inventory (measured).**
  - 126 rolling (L3/L5/L10 x 14 stats x home/away/diff), 39 season averages, 2 season win%, 3 streak, 2 H2H, 18 roster, 6 matchup ([feature_list_2025_26.json](../../models/feature_list_2025_26.json)).
  - The stats are raw per-game counts: score, score_allowed, point_margin, FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT, REB, AST, TO, impact_score_agg ([predict_2025_2026.py L408-411](../../prediction_engines/predict_2025_2026.py)).
  - Missing: free throws, OREB/DREB, STL/BLK/PF, possessions and ratings. The master frame already loads PACE, OFF/DEF/NET rating, POSS, TS%, eFG%, OREB% and TOV% ([csv_master.py L58-65](../../prediction_engines/csv_master.py)).
- **[E2] Rest and back-to-backs are computed but unused.**
  - They are computed ([csv_master.py L179-206](../../prediction_engines/csv_master.py)) and stored in the dataset, but are not among the 196 features (measured).
  - Raw split in the production dataset (measured, not adjusted for team strength):

    | Situation | Mean home margin | Home win rate |
    |---|---|---|
    | Neither team on a back-to-back | +1.85 | 0.556 |
    | Home team on a back-to-back | +0.48 | 0.523 |
    | Away team on a back-to-back | +3.72 | 0.607 |

  - Away teams are on back-to-backs more often than home teams: 19.7% vs 15.1%.
  - `availability_features.py` dismisses rest as a "summary of data the model already had" ([L5-9](../../prediction_engines/availability_features.py)), yet no production feature encodes dates or schedule.
- **[E3] No travel, time-zone, altitude or density features.**
  - No module in `prediction_engines/` or `app.py` mentions travel, timezone or altitude (grep).
  - Rest is whole calendar days only, and a team's first game counts as 0 ([csv_master.py L194-197](../../prediction_engines/csv_master.py)).
  - The schedule already has tip-off UTC and arena city ([fetch_schedule.py L48, L56-57](../../data_retriaval/fetch_schedule.py)), but no feature uses them.
- **[E4] No opponent adjustment in production.**
  - team_ratings.py's docstring says 184 of the then 190 features ignore who the team played ([L4-12](../../prediction_engines/team_ratings.py)).
  - Elo with offseason regression toward the mean and Massey ratings are implemented ([L14-39, L47-48](../../prediction_engines/team_ratings.py)) but not used.
  - In [output/walk_forward_7arms.json](../../output/walk_forward_7arms.json), with burn-in 10, the ratings arm scored:

    | Arm | Accuracy | Brier | Margin MAE |
    |---|---|---|---|
    | A (base features) | 0.6716 | 0.2126 | 10.47 |
    | A + ratings | 0.6686 | 0.2113 | 10.39 |

- **[E5] The binding evaluation skips the early season.**
  - The binding walk-forward and the consistency labs score only games where both teams have played at least 10 games ([DURUM.md L16-18](../../DURUM.md); [consistency_lab.py L80, L232-233](../../prediction_engines/consistency_lab.py)).
  - That excludes 1,400 of 10,749 games (13.0%); 430 games involve a team with fewer than 3 games (measured).
  - The older [walk_forward_2025_26.json](../../output/walk_forward_2025_26.json) shows accuracy 0.6628 at burn-in 0 against 0.6718 at burn-in 10 (360 vs 315 cells).
- **[E6] Small samples are not shrunk.**
  - Rolling and season means use `min_periods=1` ([predict_2025_2026.py L418, L421, L479-483](../../prediction_engines/predict_2025_2026.py)). Season averages jump from last season's full average at game 1 to a one-game sample at game 2.
  - Measured on season_avg_point_margin (correlations near 1 late in the season are partly by construction):

    | Team games played | SD | Correlation with final season value |
    |---|---|---|
    | 0 (opener) | 4.62 | 0.52 |
    | 1-3 | 11.56 | 0.34 |
    | 4-9 | 7.01 | 0.61 |
    | 10-19 | 5.91 | 0.80 |
    | 40+ | 4.98 | — |

  - There is no games-played feature in production; the "season phase (gp)" arm was neutral ([DURUM.md L79](../../DURUM.md)).
- **[E7] Injuries:** only the box-score proxy `roster_missing_impact` exists ([DURUM.md L509-510](../../DURUM.md)); see A4/A5.
- **[E8] Lineup continuity:** nothing beyond roster_avail_minutes ([pregame_roster.py L58-59, L237](../../prediction_engines/pregame_roster.py)).
- **[E9] Head-to-head** uses the last 5 meetings however old they are ([predict_2025_2026.py L500-522](../../prediction_engines/predict_2025_2026.py)). A value of 0 means both "never met" and "lost every meeting": h2h_L5_home_wins == 0 in 1,554 games (measured).
- **[E10] Missing values are filled with 0** ([build_dataset_db.py L211](../../prediction_engines/build_dataset_db.py); [pregame_roster.py L376](../../prediction_engines/pregame_roster.py)). Measured: home_L10_score == 0 in 15 games and home_season_win_pct == 0 in 171.
- **[E11] Market benchmark over 308 games:** the market is better on margin (MAE 10.58 vs 11.49) and total (14.33 vs 15.22) and ties on picking the winner ([DURUM.md L97-105](../../DURUM.md)).

#### Done well
- The methodology is honest: paired arms across 54 cells (18 months x 3 seeds), month-level variance reported, and negative results kept ([DURUM.md L14-34, L69-95](../../DURUM.md)).
- Opponent-adjusted ratings, clutch and availability families are already implemented, with as-of construction, ready to re-test ([team_ratings.py L28-39](../../prediction_engines/team_ratings.py); [clutch_features.py L31-34](../../prediction_engines/clutch_features.py)).

### Inferences
- **[E5/E6] Severity High for current 2026-27 use.**
  - **Why it matters:** every 2026-27 game right now involves teams at 1-3 games, where the season-average features are at their noisiest and least informative. The reported 0.666/0.687 accuracy does not apply to them.
  - **Fix:** report burn-in-0 and early-season-only metrics on the pregame dataset.
  - Blend toward a prior: rating = w x current + (1 - w) x last season regressed to the mean (or preseason win totals), with w = n / (n + k). Add games played and roster continuity.
  - Consider a rating-only fallback while a team has fewer than 10 games.
- **[E4] Severity Medium-High.**
  - The earlier ratings test probably ran on the observed-roster dataset. Its margin MAE of 10.47 sits far below the honest 11.2, which fits that explanation. The comparison is only indicative: both use the same fold protocol, but they cover different months (105 vs 54 cells).
  - **Fix:** re-test on the pregame dataset. Feed Massey or Elo differentials, plus opponent-adjusted four factors.
- **[E2/E3] Severity Medium.**
  - **Why it matters:** a raw back-to-back shift of about 1.4-1.9 points is the kind of effect the market prices. An accuracy screen with roughly 0.5-0.7 percentage points of seed noise cannot see it; paired margin-MAE and Brier tests are more sensitive.
  - **Fix:** add rest days (capped), back-to-back, 3-in-4, 4-in-6, games in the last 7 days, road-trip length, kilometres travelled since the last game, time-zone shift and an altitude flag (DEN, UTA). Re-test with paired margin MAE and Brier.
- **[E1] Severity Medium.**
  - **Why it matters:** raw counts mix pace with quality.
  - **Fix:** use per-100-possession offensive and defensive ratings plus the four factors (eFG%, TOV%, ORB%, FT rate) for and against, plus pace; drop duplicated raw counts.
- **[E9/E10] Severity Low.** Fix: use explicit "no data" indicators, or leave NaN for LightGBM, instead of 0.
- **[E7/E8]:** see the fixes for A1-A5 and C5.

### Gaps
- The back-to-back effect was not adjusted for team strength.
- Travel distances were not computed; that needs arena coordinates.
- The dataset used by walk_forward_7arms.json is not recorded in the fields inspected.

## 6. Operational and engineering risks

### Takeaway
- **Reproducibility:** the build works only from local snapshots (phone-DB tables, matchup and team-box caches) restored from `local_backup/`.
- **Dependencies:** incomplete and unpinned.
- **Latent bug:** the season-keyed team-box cache will silently freeze an in-progress season and mislabel its new games.
- **Speed:** the slow D: junction makes every rebuild that walks `nba_data/` take tens of minutes.

### Cited Findings
- **[F1] Dependencies are incomplete and unpinned.**
  - [requirements.txt L1-17](../../requirements.txt) lists pandas, numpy, scikit-learn, lightgbm, matplotlib, seaborn, nba_api and tqdm, with lower bounds only.
  - The code also imports packages the file does not list:
    - flask ([app.py L47](../../app.py)) and waitress ([app.py L831](../../app.py));
    - xgboost ([analysis_2025_2026.py L25](../../prediction_engines/analysis_2025_2026.py));
    - pymysql, which `phonedb_source.py` imports at load time ([L76](../../prediction_engines/phonedb_source.py)). DURUM lists only flask and xgboost as missing ([L527-528](../../DURUM.md)).
  - `nba_api>=1.2` is too loose: the tracking backfill needs a version with BoxScorePlayerTrackV3 ([backfill_defensive_tracking.py L15-16, L29](../../data_retriaval/backfill_defensive_tracking.py)).
  - `groupby.apply(..., include_groups=False)` ([availability_features.py L166](../../prediction_engines/availability_features.py)) needs pandas 2.2 or later.
  - Models are pickled LightGBM objects, which are version-sensitive.
- **[F2] The team-box cache is keyed by season name only.**
  - It is a hit whenever the requested seasons are a subset of the cached season names, with no per-game check ([csv_master.py L126-176, L134-144](../../prediction_engines/csv_master.py)). It currently holds 9 seasons, none from 2026-27 (measured).
  - The first request for 2026_2027 re-reads 2 CSVs per game for every season on D:.
  - After that, the in-progress season is served from its first snapshot. New game rows get NaN stats, `home_win` = 0 ([L225-227](../../prediction_engines/csv_master.py)) and 0-0 scores after fill ([build_dataset_db.py L211](../../prediction_engines/build_dataset_db.py)).
- **[F3] Storage is slow.**
  - `nba_data/` (107,421 CSVs, 1.48 GB) and `phonedb_cache/` sit on D: behind junctions.
  - Copying took 26 minutes and a one-pass read check 66 minutes.
  - The large caches stay on C: because file symlinks need admin rights ([DURUM.md L397-414](../../DURUM.md)).
- **[F4] Rebuilding depends on a restore step.**
  - A fresh clone needs `py tools/local_backup.py restore` (37 files, 791 MB unpacked from 53 MB; [DURUM.md L355-372](../../DURUM.md)).
  - phonedb_cache, matchup_cache_v1.pkl (398 MB), team_box_cache_v1.pkl and defensive_tracking_cache_v1.pkl are git-ignored ([.gitignore L16-18, L23-24, L35-37](../../.gitignore)).
  - The loader that filled the phone DB is lost ([csv_master.py L4-8](../../prediction_engines/csv_master.py)).
- **[F5] Failures degrade silently.**
  - `build_dataset_db` catches broad exceptions and continues for the DB probe and the matchup, clutch and availability families ([L137-141, L176-177, L198-199, L206-207](../../prediction_engines/build_dataset_db.py)).
  - `predict_2025_2026`'s cache loader returns `{}` on any error ([L556-563](../../prediction_engines/predict_2025_2026.py)).
  - `rebuild_impact_features` fills missing impact sums with 0 ([L127-128](../../prediction_engines/rebuild_impact_features.py)); measured: 0 such rows today.
- **[F6] The 2026-10-24 work is not committed** ([DURUM.md L519-521](../../DURUM.md)). The session git status shows untracked `fetch_schedule.py`, `check_data.py`, `sim_vs_real.py`, the 2026-27 game_id and schedule CSVs, and `nba_data/2026_2027/`.
- **[F7] Cache entries lack provenance.** Value-cache entries do not record the pricing version or the inputs used ([value_engine.py L569-570](../../prediction_engines/value_engine.py)).

#### Done well
- The compressed backup has a manifest with sizes and checksums, a test restore into an empty folder matched all 37 files, and packing refuses files containing `.env` values ([DURUM.md L355-391](../../DURUM.md)).
- After the phone DB died, a CSV fallback was added for the master frame and verified against the DB ([build_dataset_db.py L111-144](../../prediction_engines/build_dataset_db.py); [csv_master.py L1-41](../../prediction_engines/csv_master.py)).
- `rebuild_impact_features` refuses to run unless its recipe reproduces the stored columns ([L99-120](../../prediction_engines/rebuild_impact_features.py)).

### Inferences
- **[F2] Severity High (latent).**
  - **When it triggers:** on the first 2026-27 rebuild after more games are played.
  - **Fix:** key the cache by season plus a hash of the game-ID set (or per game), append new games incrementally, and assert labels are present for training rows.
- **[F1] Severity Medium.** Fix: pin versions with a lock file, add flask, waitress, xgboost and pymysql (as an optional extra), require an nba_api version with V3 endpoints, and pandas 2.2+.
- **[F3] Severity Medium (operations).** Fix: consolidate per-game CSVs into per-season Parquet files, keep the CSV tree as a raw archive, and rebuild caches from Parquet in seconds.
- **[F4] Severity Medium.** Fix: add CSV-to-table loaders (see D4) and a single reproducible "build dataset from raw" command.
- **[F5] Severity Medium.**
  - **Why it matters:** the same rebuild can yield different column sets depending on database reachability.
  - **Fix:** fail fast unless `--allow-missing` is passed, and record each feature family's coverage in the dataset blob.
- **[F6/F7] Severity Low.** Fix: commit the work and version the cache entries.

### Gaps
- Installed package versions were not checked.
- Rebuild timings were not measured, because rebuilds were out of scope.

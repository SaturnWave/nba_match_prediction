# NBA 2025-26 Predictor — Improvement Report

## 1. Executive summary

The project ships a working game-outcome predictor (`prediction_engines/predict_2025_2026.py`) for the completed 2025-26 NBA season (1230 games). It pairs a `GameDataLoader` / `FeatureEngineer` pipeline (184 engineered, leakage-controlled features) with a `NBAPredictor` stack of LightGBM models for four targets — `home_win`, `point_diff`, `home_score`/`away_score`, and `total_score`. A generalized play-by-play impact score and a new forward-looking, roster-aware impact module (`get_current_roster_impact` / `_build_player_history` / `_add_roster_features`) feed the model rolling per-player and per-team signals under strict `.shift(1)` discipline.

**Headline metrics** (held-out 308 games, the most recent ~25% of 2025-26):

| Target | Metric | Value | Baseline / note |
|---|---|---|---|
| home_win | accuracy / AUC | **0.760 / 0.832** | home-pick baseline 0.610 |
| point_diff | MAE | **11.06** | impact helps (−0.89 vs without) |
| total_score | MAE | **15.18** | impact slightly hurts (+0.19) |
| home_score | MAE | 9.99 | |
| away_score | MAE | 9.50 | |

**Top 5 prioritized improvements:**

| # | Improvement | Expected impact | Effort | Why |
|---|---|---|---|---|
| 1 | Team-strength **ratings baseline** (Elo / Glicko / TrueSkill) as features | High — biggest accuracy lever after calibration; targets the weak close-game segment (0.653) | Medium | No opponent-adjusted rating exists in 184 features; rolling form treats all opponents as equal |
| 2 | **Probability calibration** (isotonic/Platt) + Brier/log-loss metrics | High product value; little accuracy change but trustworthy probabilities | Low | No calibration step anywhere; AUC 0.832 ≫ 0.760 accuracy at threshold |
| 3 | Fix **validation**: walk-forward CV + leak-free early stopping | High — corrects optimistic, unequal protocol; gives error bars | Medium | Single late-season split; LightGBM early-stops on the test set |
| 4 | Surface **already-extracted features** (PACE, OFF/DEF_RATING, TS_PCT, rest/B2B) | Medium-High; pace targets total_score | Low | Columns are read but never rolled into `STATS` |
| 5 | **Three-way ablation** to isolate the roster module, then matchup/decay/minutes-weighting roadmap | Medium; unblocks the stated forward-looking direction | Low → Medium | The +2.6% gain bundles 12 roster + 12 team-impact features |

---

## 2. Current results

All headline numbers come from a single chronological split: train = full 2024-25 (1230 games) + earliest 75% of 2025-26 (922 games, `n_train = 2152`); test = the last 25% (`n_test = 308`), per `analysis_2025_2026.py` and `predict_2025_2026.py` (`split = int(len(tgt)*0.75)`).

| Target | Metric | Value |
|---|---|---|
| home_win | accuracy | 0.7597 |
| home_win | AUC | 0.8317 |
| home_win | baseline (home-pick) | 0.6104 |
| point_diff | MAE | 11.06 |
| total_score | MAE | 15.18 |
| home_score | MAE | 9.99 |
| away_score | MAE | 9.50 |

**Late-season-split caveat.** The held-out 308 games span **2026-03-04 to 2026-04-12** (212 March + 96 April). This is exactly the settled-standings window where favorites are clearest, so **0.760 / 0.832 overstate full-season skill**. A single split also yields a point estimate with no error bar: for `n=308`, the sub-2-point spread between model classes and the +2.6-point impact-ablation delta are within plausible split-to-split noise. November/December games are never scored under this protocol.

---

## 3. Where the impact score falls short — when it helps vs hurts

### 3.1 The segment ledger (verbatim from `output/analysis_2025_26.json`)

| Segment | n | acc **with** impact | acc **without** | delta |
|---|---|---|---|---|
| close `\|pd\|≤6` | 75 | 0.6533 | 0.6400 | **+0.0133** (≈ neutral) |
| moderate `6<\|pd\|≤15` | 110 | 0.7727 | 0.6818 | **+0.0909** (biggest gain) |
| blowout `\|pd\|>15` | 123 | 0.8130 | 0.8374 | **−0.0244** (hurts) |
| point_diff (MAE) | 308 | 11.055 | 11.940 | **−0.885** (helps) |
| total_score (MAE) | 308 | 15.183 | 14.991 | **+0.193** (hurts) |

The *where* is measured exactly; the *why* below is an interpretive mechanism that is consistent with the code but not directly proven by the JSON.

### 3.2 Mechanism

- **Moderate-game gain (+9.1%).** `compute_game_impact` is dominated by its scoring terms — `_scoring_impact` assigns `bi = 3.0` for a made three and `2.0` otherwise — with **no per-possession normalization**. The rolled L3/L5/L10 aggregate therefore behaves as a recent-margin/quality proxy that adds genuine separation precisely where favorites are not yet obvious.
- **Blowout loss (−2.4%) and total_score loss (+0.19).** Blowouts are already easy (0.837 *without* impact), and unnormalized garbage-time minutes inflate both teams' raw sums, injecting noise rather than signal. `total_score` worsens because impact is built from the **same made shots** the box-score scoring features already capture — double-counting offense and adding variance. The home-minus-away **difference** cancels this common scoring noise, which is why `point_diff` still benefits (−0.885 MAE).

### 3.3 Structural defects in the impact score

- **Raw same-game impact is descriptive, near-leakage.** `home_impact_score_agg` / `away_impact_score_agg` / `impact_score_diff` re-derive the final margin (measured `corr(impact_score_diff, point_diff) ≈ 0.808` on a 2024-25 sample). The model escapes leakage *only* because `FeatureEngineer.STATS` includes `impact_score_agg`, `_rolling` applies `.rolling(w, min_periods=1).mean().shift(1)`, and `_select_features` keeps only engineered prefixes (`home_L*`/`away_L*`/`diff_L*`, `season_avg_`, `h2h_`, `streak`, `roster_*`). The raw `impact_score_diff` matches none of these and never enters `feature_columns`. **The entire predictive validity rests on one `.shift(1)`** — any future path feeding a raw same-game impact column to a model would leak the outcome.
- **Hardcoded Denver/GSW team IDs in the legacy code.** In `asasa.py` and `impact_score_calculation/impact_score.py`, four modifier classes — frontcourt-steal (`asasa.py:197`), trailing-team offensive-rebound (`:226-227`), lead-change/cut-to-one scoring (`:300-318`, with `home_team_id` literally set to Denver `1610612743` at `:305`), and intentional-foul-while-trailing (`:437-438`) — fire **only** when `teamId` equals `1610612743` (Denver) or `1610612744` (GSW), demo-game leftovers. For all other 28 teams those branches are dead. The working model **generalized** the steal and rebound logic — `_steal_impact` / `_rebound_impact` take the real `home_id`/`away_id` — but **dropped** the other two: `_scoring_impact` has no lead-change/cut-to-one modifier and `_foul_impact` has no trailing-team intentional-foul logic. Those richer signals are gone, not generalized.
- **Per-100-possession normalization was dropped.** `impact_score.py` normalizes every player to 100 possessions (`calculate_team_possessions`; `poss_factor = 100/max(1, ...)`). The working model accepts a `team_possessions` parameter but **never references it** in the body, and the only caller passes an empty dict. The result is a raw event **sum**: fast-pace teams accumulate structurally higher impact unrelated to quality, blowouts balloon both sums via garbage-time minutes, and OT/foul-heavy games distort the home/away difference. This is the dead-parameter root of the blowout and total_score regressions in §3.1.
- **Uncalibrated magic weights with compounding multipliers.** Every weight is a hand-tuned literal — block 1.2, steal 1.4, rebound 0.9/0.6, score 3.0/2.0, turnover −1.0/−0.8, foul −0.3..−1.5 — none fit to outcomes. Worse, a clutch made shot is multiplied **three times**: the in-function `*1.3`/`*1.2`, the outer `*1.5` in `compute_game_impact`, **and** a time-decay `1 + 1/(clock_seconds+1)` that doubles a play at the buzzer (`cs=0 → 2.0`). (Caveat: the ablation removes the *entire* aggregate, so it cannot isolate the weights specifically; the structural arbitrariness is the firm claim.)
- **No opponent or minutes adjustment.** `compute_game_impact` weights purely by event type and game-state — never by opponent strength or minutes. `DEF_RATING` is loaded but never fed into impact. A player padding stats against a weak defense scores identically to the same line against an elite defense, and the raw sum mechanically rewards more minutes. The new roster features inherit this opponent-blind, minutes-unnormalized base.

**Fixes:** assert no raw `impact_score_agg`/`impact_score_diff` column enters `X_train`; cap/down-weight garbage-time plays and normalize per-possession; re-add the dropped lead-change and intentional-foul modifiers in generalized `home_id`/`away_id` form; learn weights from data (regress event counts on margin/win, or adopt RAPM/EPV coefficients) and remove the compounding clutch×time double-multiplier; **drop impact from the `total_score` model** while keeping it for `home_win` and `point_diff`.

---

## 4. Can a different model do better?

**Model comparison** (`output/analysis_2025_26.json`):

| Model | home_win acc | home_win AUC | point_diff MAE | total_score MAE |
|---|---|---|---|---|
| LightGBM | **0.7597** | 0.83169 | **11.055** | **15.183** |
| Random Forest | 0.7565 | **0.83214** | 11.085 | 15.299 |
| Logistic / Ridge | 0.7468 | — | 11.519 (ridge) | 15.285 (ridge) |
| XGBoost | 0.7435 | — | 11.214 | 15.296 |

The four model classes fall in a **1.6-point accuracy band**; RF's AUC is a hair *above* LightGBM's; on `point_diff` the trees and a plain linear model sit within ~0.46 MAE; on `total_score` all four are within 0.12 MAE and **Ridge beats both tree models**.

**Conclusion — near-ceiling; calibration and ratings baselines matter more than model class.** When a strong nonlinear learner, a bagged learner, and a plain linear model land on top of each other, the bottleneck is the **feature representation and irreducible NBA noise**, not the estimator. Two protocol issues only *strengthen* this: (a) LightGBM alone early-stops on the test set via `eval_set=[(X_test,y_test)]` + `lgb.early_stopping(30)`, while XGB/RF/LogReg/Ridge fit on train only — so LightGBM's narrow win is partly an **unequal-protocol artifact**; (b) the single split gives no error bar, so the sub-2-point spread may be sampling noise. **Freeze on LightGBM (or RF) and redirect effort to calibration, an Elo/Glicko baseline, and features.** Keep the comparison as a guardrail, not an optimization target. Deep/sequence PBP models are explicitly **de-scoped**: ~2152 train / 308 test games is tabular-small, a plain linear model already ties the GBM, and PBP signal is already distilled into the impact features — sequence models would add variance and leakage surface for at best parity.

---

## 5. Feature, data & validation improvements

**Features that are missing or extracted-but-unused:**

- **No rest / back-to-back / schedule-density features**, though fully computable from `game_id_2025_2026.csv` (`GAME_DATE` + `MATCHUP`). Re-derived: 441 of 2460 team-games are back-to-backs (`rest==1`) = **17.9%** (rest dist `{1:441, 2:1520, 3:353, 4:60, 5:18}`). These are leakage-free (known pre-tip) and cheap. Add `home_days_rest`, `away_days_rest`, `diff_days_rest`, `home_is_b2b`, `away_is_b2b`, plus a rolling games-in-last-N-days count — and add their prefix to the `_select_features` allow-list (it is prefix-based and would silently drop them otherwise).
- **PACE, OFF_RATING, DEF_RATING, TS_PCT, FTM/FTA/OREB/DREB/STL/BLK/PF are extracted but never rolled.** `_extract_features` pulls them but `FeatureEngineer.STATS` includes only `[score, score_allowed, point_margin, FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT, REB, AST, TO, impact_score_agg]`. Only `STATS` members are rolled+shifted and expanding-averaged. **Pace especially targets `total_score`**, the one target impact currently hurts. Add these columns to `STATS` so they inherit the existing `.shift(1)` treatment.
- **No opponent adjustment / strength-of-schedule.** All team features are raw and self-referential; a 9-1 run vs weak teams is indistinguishable from 9-1 vs contenders. This is the same gap as the §4 ratings lever and most directly targets the weak close-game segment (0.653).
- **Rolling form blends home & away games.** The per-team loop concatenates home and away frames into one timeline before `_rolling`, so `home_L10_score` is the team's last-10 across *both* venues, not its home form. There is also no constant home-court indicator — the model must re-learn the 0.610 base rate. Add venue-specific rolling form and a home-advantage flag.

**Data pipeline:**

- **146 box-score-less 2024-25 games train with stats zeroed.** On disk, 2024-25 has 1230 PBP but only 1084 traditional / 1083 advanced boxes (**146 / 147 missing**). `load_game_data` gates only on PBP, so all 1230 load; missing traditional stats default to `0`, advanced to `np.nan`. The 146 zeroed-box games **do** enter training and drag their teams' rolling averages toward zero. Backfill via the resumable V3 retriever, or mark `box_missing=1` and impute team-season means (treat `0` and "missing" distinctly). Separately, 2024-25 has 1082 **defensive** box files that are read-but-never-featurized; 2025-26 has 0/1230 — either consume or stop retrieving them.

**Validation:**

- **Single late-season split is optimistic and incomplete.** Test window is 2026-03-04..04-12 (settled standings); there is no walk-forward retraining. Switch to **expanding-origin / walk-forward CV** at monthly cutoffs across the full 2025-26 season (and into 2024-25), reporting **mean ± std** of accuracy/AUC/Brier/MAE per model.
- **No calibration or product metrics.** No `CalibratedClassifierCV`, isotonic, Platt, Brier, or `log_loss` anywhere; `metrics_2025_26.json` reports only accuracy/AUC/MAE/RMSE. The engine returns raw `predict_proba`. AUC 0.832 with accuracy 0.760 at threshold is the classic well-ordered-but-mis-scaled signature. Wrap the classifier in **isotonic (or Platt) calibration** fit on a time-ordered validation slice, add **Brier + log-loss**, and plot a reliability diagram. For a betting end-use, also report accuracy and ROI vs the closing line and the ~52.4% break-even.
- **Better-specified score targets.** `home_score`/`away_score`/`total_score` are fit as default-objective `LGBMRegressor`. Basketball points are non-negative counts with mean-variance coupling, so L2 is mildly misspecified. Refit with `objective='poisson'` (or tweedie/neg-binomial) and derive `point_diff`/`home_win` from the Skellam of the two distributions for coherence — judging gains on aggregate MAE + calibration, not single games.
- **Leakage hygiene is correct and must be preserved.** `.shift(1)` is applied consistently and the prefix whitelist plus explicit excludes reject every raw same-game/target column. Keep `.shift(1)` on every new rolling feature, route new per-game stats through `STATS`, and add a unit test asserting no raw same-game column ever appears in `feature_columns`.

**Hyperparameters & constraints (polish, after the above):** all estimators use fixed hand-set params (LightGBM `n_estimators=400`/`lr=0.03`/`num_leaves=31`) with no search; no `monotone_constraints`. Run a small time-aware search on the walk-forward harness and add monotone constraints on signed features (`elo_diff`, `diff_roster_form`, `diff_*_impact` should monotonically raise `home_win` prob), which also reduce overfitting on `n_train=2152`. Expect ~1 point, not a step change.

---

## 6. The forward-looking roster impact: what shipped + roadmap

**What shipped (and is leakage-free).** `get_current_roster_impact` / `_build_player_history` / `_add_roster_features` faithfully implement the spec: per-player last-10 mean via `groupby('player')['impact'].rolling(10, min_periods=1).mean().shift(1)`, plus `l6`/`l3` form, aggregated to the roster by `(game_id, team)` and merged onto home/away with `diff_*` columns. The `.shift(1)` (training path) and the `game_date < as_of` guard in `get_current_roster_impact` (the future-prediction path) make it leakage-free. Because the per-player history is sorted by date across all loaded seasons, the last-10 window automatically rolls into the previous season for a player's first ~10 games of a new season — implementing the season-warmup rule.

**Critical caveat — the +2.6% gain is NOT attributable to roster features alone.** The ablation removed **all 24** "impact OR roster" features at once: exactly **12 roster features** (home/away/diff × {`roster_impact_l10_mean`, `roster_impact_l10_sum`, `roster_form_l6`, `roster_form_l3`}) **+ 12 older team-level** `impact_score_agg` rolling features. The roster module's *isolated* marginal value is therefore **currently unknown**. **First action: a three-way ablation** — full vs (drop only the 12 roster features) vs (drop only the 12 team-impact features). It is a ~10-line change and tells you whether to keep investing before building the matchup roadmap.

**Roadmap (sequence after the three-way ablation):**

1. **Player-vs-player / player-vs-team matchup impact — blocked on data.** No `*matchup*` files exist under `nba_data/` for the bulk seasons; only traditional/advanced/defensive + tracking boxes were retrieved. The defender-vs-offensive-player partials (NBA `BoxScoreMatchupsV3`) do not exist on disk. Before scoping: **probe the matchup endpoint on a handful of 2025-26 games** (the NBA broke V2 box-score endpoints in 2025-26, so current availability must be verified). If unavailable, fall back to an on/off-court or defender-assignment proxy from PBP.
2. **Recency decay — near-drop-in, well-motivated.** The flat `rolling(w).mean()` ignores ordering (game 10-ago counts equally to last night's), and `l3`/`l6` are collinear, ad-hoc proxies. Replace with `s.ewm(halflife=H, min_periods=1).mean().shift(1)` and tune `H` (≈5-8 games) on a validation split; keep one flat `l10` for interpretability. Low effort.
3. **Announced-lineup + minutes-weighting + injury/roster-change handling — unaddressed; data exists but unused.** The aggregation uses the *actual* participants of past games via a flat `.mean()`, keyed by the raw `playerName` string — fragile across name spellings, Jr./Sr. collisions, and mid-season trades (a traded player carries old-team history under the same name). `PLAYER_ID`, `MIN`, and `START_POSITION` exist in `box_score_traditional` but are never read into `_player_records`. **Capture `PLAYER_ID` (stable key), `MIN`, and `START_POSITION`; switch to a minutes-weighted mean (use trailing avg-MIN as the weight) with a starters-only variant; and accept an injury/availability mask** so the number recomputes the instant a player is ruled out or traded.
4. **Extend training to earlier seasons — fully supported on disk.** `main()` loads only `['2024_2025','2025_2026']`, but **seven complete-PBP seasons** are on disk (2019-2020 .. 2025-26 ≈ **8289 games**). Because impact is now leakage-free, there is no reason to withhold them. Side benefit: the "roll into the previous season" warmup rule only works when a prior season is in `_player_records`; for the first loaded season (2024-25) early windows fall back to thin estimates. Loading 2023-2024 (at minimum) makes the rollback work and ~4× the training set.
5. **Fix early stopping before trusting any roadmap A/B.** Both ablation arms' LightGBM models early-stop on the 308-game test set, so the very +2.6% delta this dimension hinges on is biased by test-set iteration selection. Carve a validation slice from the tail of the train window and re-confirm the deltas.

---

## 7. Prioritized roadmap

| Improvement | Expected impact | Effort | Rationale |
|---|---|---|---|
| Team-strength ratings (Elo/Glicko/TrueSkill) as features (`home_elo`, `away_elo`, `elo_diff`, `elo_win_prob`) | **High** — likely biggest single accuracy lever after calibration | Medium | No rating in 184 features; rolling form is opponent-blind; targets weak close-game segment (0.653); Glicko/TrueSkill variance suits early-season/post-trade uncertainty |
| Probability calibration (isotonic/Platt) + Brier/log-loss + reliability curve | **High product value** (little acc change) | Low | No calibration anywhere; AUC 0.832 ≫ 0.760 acc; probabilities are the actual product |
| Walk-forward CV + leak-free early stopping (validation slice from train tail) | **High** — corrects optimistic, unequal protocol; adds error bars | Medium | Single late-season split (Mar-Apr); LightGBM alone early-stops on the test set, biasing the model ranking and the +2.6% ablation delta |
| Surface extracted-but-unused features into `STATS` (PACE, OFF/DEF_RATING, TS_PCT, shooting/rebounding) | **Medium-High** — pace directly targets `total_score` | Low | Columns already read; just add to `STATS` for rolling+shift |
| Rest / back-to-back / schedule-density features | **Medium** | Low | 17.9% of team-games are B2Bs; leakage-free; computable from `GAME_DATE` |
| Normalize impact per-100 possessions + cap/down-weight garbage time; drop impact from `total_score` model | **Medium** — targets the −2.4% blowout and +0.19 total_score regressions | Medium | `team_possessions` param is dead; raw sum inflates pace/blowouts; impact double-counts box scoring on `total_score` |
| Three-way roster ablation, then matchup/decay/minutes/injury roadmap (§6) | **Medium** — unblocks stated forward-looking direction | Low → Medium | +2.6% bundles 12 roster + 12 team-impact features; isolate before investing; matchup data needs a new retrieval pass |
| Poisson/Tweedie score targets + Skellam-derived win prob | **Medium** | Medium | Points are non-negative counts; L2 is misspecified; yields coherent margin distribution |
| Extend training to 2019-2024 (≈8289 games on disk) | **Medium** — helps regression targets, fixes warmup rollback | Low | Impact is leakage-free; only 2 of 7 seasons currently loaded |
| Re-add dropped lead-change / intentional-foul modifiers (generalized) + replace magic weights with learned coefficients | **Low-Medium** | Medium | Two enhanced signals were dropped, not generalized; weights are uncalibrated literals with a compounding clutch×time multiplier |
| Hyperparameter search + monotone constraints on signed `diff_*`/`elo_diff` | **Low** (~1 point) | Low | At a ceiling; polish only — do AFTER calibration/Elo/Poisson |
| De-scope deep/sequence PBP models | N/A (avoids wasted effort) | — | Tabular-small (2152/308); linear ties GBM; PBP already distilled into features |

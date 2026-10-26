# Internal models, evaluation, season simulator, value engine and dashboard: code and methodology audit (nba_match_prediction, state of 2026-10-25)

How to read these notes. Every flaw has an ID (M = models, E = evaluation, P = production/2026-27, S = season simulator, V = value/impact engine, D = power rankings and dashboard), a severity in brackets (High / Medium / Low), the evidence with a file path and line numbers, why it matters, and a concrete fix. Code links point into the repo (`C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/...`). Numbers marked "check A-E" are read-only re-measurements I made during the audit from saved files and saved models. Nothing was retrained or rebuilt and no repo file was changed. The scripts are in `checks/` next to these notes:
- check A = [prod_check.py](checks/prod_check.py): re-scores the saved blend and models on training rows and on the 308 held-out games, plus a paired bootstrap against the DraftKings closes.
- check B = [prod_check2.py](checks/prod_check2.py): encompassing test, feature redundancy, stability of the monotone signs, share of games removed by the burn-in.
- check C = [labs.py](checks/labs.py): per-month results and month-clustered SEs from the lab JSONs.
- check D = [feat.py](checks/feat.py): feature grouping and the burn-in summaries.
- check E = [misc.py](checks/misc.py): how the diff_ columns are defined.

Highest-severity items: E1, E2 and E6 (the held-out 0.773 is a late-season period effect, those games were used in model selection, and early-season games are never evaluated); P1, P2 and P3 (no path to forecast unplayed 2026-27 games); S1 and S2 (minutes, rookies and injuries in the simulator). The project is unusually honest and careful in many places; the "Done well" sub-lists record this.

## 1. Models: is 196 features on ~10,700 games overfitting? Do the monotone signs or the calibrator leak into evaluation? Is the blend's weighting justified?

### Takeaway
Overfitting of the fitted trees is not the binding problem. With heavy regularisation and early stopping, the shipped blend scores only 0.684 accuracy / Brier 0.2025 on its own training rows, about the same as walk-forward (0.687 / 0.204). The 196 columns behave like 30-50 effective dimensions. The real model problems are elsewhere:
- Probabilities and point predictions are compressed. The held-out calibration slope is 1.58. Margin SD is 7.9 against the market's 10.2; total SD is 4.7 against the market's 8.2.
- The blend is an equal-weight average that was never fitted. On held-out games, the logistic view alone had the lowest Brier of the four forecasts.
- Calibrator selection is in-sample and degenerate (isotonic always wins), and the shipped blend does not use the calibrator.
- The monotone signs see the calibration slice. This is a technical leak with negligible measured impact: no sign flips.

### Cited Findings

#### Flaws
- **M1 [Medium]: 196 columns, highly redundant; capacity is controlled but interpretability and the linear view suffer.**
  - **Composition.** The production list is 65 home_ + 65 away_ + 64 diff_ + 2 h2h columns over 67 distinct stems: L3/L5/L10/season windows of raw box counts, win%, streak, the impact family, roster/matchup and availability. There is no rest/travel, opponent-adjusted rating or per-possession efficiency column. Sources: [feature_list_2025_26.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/models/feature_list_2025_26.json); check D.
  - **Redundancy.** diff_ ≈ home − away (corr 0.98 for diff_L10_score; median absolute difference 0) except at zero-fill artefacts (check E). On the 10,441 training rows, 23 feature pairs have |r| > 0.9, and 32 / 47 / 78 principal components explain 90 / 95 / 99% of the variance (check B).
  - **Capacity is low.** Early stopping keeps 83-176 trees for home_win, 227-304 for point_diff, 51-102 for total_score and 43-68 for home_score ([metrics_2025_26.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/metrics_2025_26.json) `tree_counts`). The parameters are num_leaves 15, min_child_samples 60, reg_lambda 5, colsample 0.5 ([retrain_production.py L99-101](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/retrain_production.py)).
  - **In-sample ≈ out-of-sample.** Blend accuracy is 0.684 on all training rows, 0.701 on 2024-25 and 0.694 on the first 75% of 2025-26 (check A). Walk-forward gives 0.6865 ([metrics_2025_26.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/metrics_2025_26.json) `honest_walk_forward`).
  - **Why it matters.** Overfitting is not the issue. Redundancy splits the split-count importances that are plotted ([retrain_production.py L327-334](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/retrain_production.py); [ensemble_model.py L45-47](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/ensemble_model.py)) across three collinear copies. The standardized logistic view also gets many near-duplicate inputs.
  - **Fix.** Collapse home_/away_ levels to diff_ (the diff_only arm was within noise, [DURUM.md L76](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)). Replace raw counts with per-possession, opponent-adjusted ratings. Report gain or SHAP importance grouped by stem.

- **M2 [Medium]: Forecasts are under-confident and margins/totals are compressed.**
  - **Calibration on the 308 held-out games** (check A). A logistic recalibration of the blend gives slope 1.58 and intercept 0.27. The quintile with mean forecast 0.70 won 91.8%; the 0.57 quintile won 69.4%; the 0.235 quintile won 14.5%.
  - **Spread versus the market.** Blend probability SD is 0.213 against the market's 0.26 (check B). Model margin SD is 7.89 against the closing spread's 10.23 (corr 0.93). Model total SD is 4.66 against the closing total's 8.16 (corr 0.77) (check A).
  - **Score models were never tuned.** The score models keep the base parameters and were never measured under the shipped recipe ([retrain_production.py L91-98, L180-185](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/retrain_production.py)). The project's own note says predicted home scores span 102-126 (sd 4.5) against an actual sd of 13.7 ([simulation.py L14-19](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/simulation.py)).
  - **The older pipeline showed the same direction.** Even after isotonic or Platt, its upper bins under-predicted by 0.10-0.18 ([calibration_2025_26.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/calibration_2025_26.json)).
  - **Caveat.** On the same 308 games the market's own slope was 1.31 (check B), so part of this is the easy late-season window (see E1).
  - **Why it matters.** Under-confident probabilities cost Brier and log loss against the market (E7). They also feed any downstream expected-value calculation or season simulation with numbers that are too flat.
  - **Fix.** Fit a post-blend Platt or beta calibrator inside a nested walk-forward. Store each cell's calibration slope and intercept so this can be checked across all 54 cells. For the margin and total models, test lower shrinkage or more trees, and fit residual-scale models.

- **M3 [Medium]: The blend's 1/3-1/3-1/3 weights are asserted, not fitted, and are not checked against single views on the honest features.**
  - **How it works.** BlendedForecaster averages the classifier, a margin→probability map and a logistic model in probability space ([ensemble_model.py L77-131](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/ensemble_model.py)).
  - **Its justification is from the earlier pipeline.** "+0.0183 accuracy … across 430 walk-forward cells" comes from before the roster-leak fix ([ensemble_model.py L80-102](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/ensemble_model.py); [retrain_production.py L37-41](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/retrain_production.py)).
  - **Lab cells cannot test it.** The cells store only the classifier and blend3 scores, not per-view scores ([consistency_lab.py L214-228](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/consistency_lab.py)).
  - **Held-out per view** (check A):

    | Forecast | Brier | Accuracy |
    |---|---|---|
    | logistic view | 0.1663 | 0.763 |
    | blend | 0.1673 | 0.773 |
    | margin view | 0.1692 | 0.769 |
    | classifier | 0.1727 | 0.779 |

  - **Honest walk-forward.** The blend beats the classifier: 0.6865 vs 0.6655 accuracy, Brier 0.204 vs 0.211 ([DURUM.md L22-25](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)).
  - **Why it matters.** The blend is clearly better than the classifier alone. Whether equal weights beat a two-view blend or a stacked one is untested. A linear pool that includes the most compressed view (the classifier, SD 0.176) stays under-dispersed (check A).
  - **Fix.** Persist per-view probabilities per cell. Fit stacking weights in logit space (ridge logistic regression) in a nested walk-forward. Then recalibrate the pooled output.

- **M4 [Low]: Monotone signs are estimated on all training rows, including the calibration slice; the lab estimated them on fit rows only.**
  - **Code.** Production calls `monotone_signs(train_df, …)` ([retrain_production.py L161-177, L183, L289](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/retrain_production.py)). The same signs are then reused by the warm models that produce the calibrator and margin-mapper inputs (L293, L347, L368). The lab signs from `fit_df` only ([consistency_lab.py L177-179](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/consistency_lab.py)). The held-out games never touch the signs ([retrain_production.py L133-141](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/retrain_production.py)).
  - **Measured impact** (check B). 60 of 64 diff_ columns are constrained. There are no sign flips between the two halves of the training rows. Only the FGA family sits near zero (|ρ| < 0.02), and two FGA columns toggle on/off between halves. The negative signs are intuitive: score_allowed, TO, missing_impact.
  - **Why it matters.** The leak is real in principle and negligible in practice.
  - **Fix.** Sign on fit rows only in production (one-line change), or fix the signs from domain knowledge.

- **M5 [Low-Medium]: Calibrator selection is in-sample and always picks isotonic; the shipped probability does not use it.**
  - **In-sample selection.** Production fits isotonic and Platt on the calibration slice and picks the lower Brier on that same slice ([retrain_production.py L352-358](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/retrain_production.py)); walk-forward does the same ([walk_forward.py L151-156](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/walk_forward.py)). Isotonic regression is the least-squares monotone fit, so in-sample it cannot lose to an increasing Platt curve. Production recorded "isotonic" ([metrics_2025_26.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/metrics_2025_26.json)).
  - **Selection on the test set in the standalone script.** calibration.py picks the winner on TEST Brier ([calibration.py L185](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/calibration.py)).
  - **Not part of the shipped forecast.** The blend contains no calibrator ([ensemble_model.py L109-131](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/ensemble_model.py)). The app prefers blend > calibrated > raw ([app.py L345-358, L613-629](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/app.py)), and the calibrated classifier appears only as a footnote ([index.html L427](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/templates/index.html)). The app docstring still says the page shows the calibrated probability ([app.py L9-12](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/app.py)).
  - **Mismatched inputs.** The warm members that produce the calibrator inputs are trained on 80% of rows with patience 50 (L345-350). The shipped members are refit on 100% with patience 30 (L196-212). This contradicts the comment that the map "matches the shape of the probabilities production actually emits" (L336-338, L343-344).
  - **Fix.** Choose the calibrator by cross-validated or out-of-time Brier, or default to Platt/beta. Calibrate the blend rather than the classifier. Generate calibrator inputs with the shipped refit protocol, e.g. rolling out-of-fold predictions.

- **M6 [Low]: Missing values are zero-filled everywhere.**
  - **Code.** `Xy`/`X_of` call `fillna(0)` ([retrain_production.py L144-146](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/retrain_production.py); [consistency_lab.py L150-151](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/consistency_lab.py); [app.py L351](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/app.py)), and so does the builder ([build_dataset_db.py L211](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/build_dataset_db.py)).
  - **Evidence.** At fill artefacts diff_L10_score differs from home − away by up to 122 points (check E).
  - **Why it matters.** "0 points in the last 10 games" is an impossible value. LightGBM could route NaN natively, and the logistic view treats the zero as an extreme outlier.
  - **Fix.** Keep NaN for the trees and add games-available counts. Impute league means for the linear view.

- **M7 [Low]: Bubble and limited-crowd seasons are trained on without a venue/crowd indicator.**
  - **Evidence.** The dataset is regular season only (all game_id prefix 002): 2019_2020 has 1,059 games and 2020_2021 has 1,080 (check A). No crowd or neutral-site feature exists among the 196 ([feature_list_2025_26.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/models/feature_list_2025_26.json)).
  - **Why it matters.** The home_win target's home edge differs in those seasons. The size of the difference was not measured here (Gaps).
  - **Fix.** Add a season-level home-edge input or down-weight those seasons for home_win.

- **M8 [Low-Medium]: The score simulator behind the dashboard's score intervals is a separate, weaker model.**
  - **Model.** A single-seed Poisson LightGBM with base parameters ([simulation.py L68-70](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/simulation.py)). The NB dispersion is r ≈ 477/489, i.e. nearly Poisson.
  - **Held-out performance** ([simulation_2025_26.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/simulation_2025_26.json); [metrics_2025_26.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/metrics_2025_26.json)):
    - Brier 0.1845 against the blend's 0.1673; accuracy 0.756.
    - Margin MAE 12.03 against the regressor's 11.49.
    - Simulated margin SD 14.4 against an actual 17.7.
    - 80% interval coverage: 0.84 (home score), 0.79 (margin), 0.82 (total).
  - **Display.** The page shows its scores and intervals next to the blend's probability ([app.py L392-418](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/app.py); [index.html L405-417](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/templates/index.html)).
  - **Fix.** Centre the count distribution on the shipped margin/total ensemble means (e.g. NB/Skellam around them), or label clearly which engine produced which number.

#### Done well
- **Two real leaks found, measured and fixed.**
  - Roster features were read off the finished game: count of players appearing vs margin, r = 0.64; the margin MAE effect was 0.80 points in 54 of 54 cells.
  - The test set was choosing the tree count.
  - Sources: [DURUM.md L36-63](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md); [retrain_production.py L4-21](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/retrain_production.py); [pregame_roster.py L5-37](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/pregame_roster.py).
- **Tree count from the chronological tail of training rows, then refit.** The held-out slice is never an argument to any fit ([retrain_production.py L188-212](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/retrain_production.py)).
- **Seed ensemble justified by a dose-response measurement.** +0.0070 / +0.0088 / +0.0103 accuracy at 3 / 5 / 10 seeds, SE ≈ 0.005 ([seed_ensemble.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/seed_ensemble.json)).
- **Pickle portability fixed.** SeedEnsemble lives in an importable module and a members file is written ([ensemble_model.py L1-20, L58-75](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/ensemble_model.py)).
- **Feature families screened before adding.** Ratings, rest, clutch and availability were tested and not added on noise ([walk_forward_7arms.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/walk_forward_7arms.json); [training_protocol.py L5-9, L47-48](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/training_protocol.py)).

### Inferences
- The binding constraint is information and calibration, not variance. In-sample ≈ out-of-sample accuracy, five feature families produced no measurable gain, and the closing line beats the model significantly (E7). Further gains are more likely from what the market knows (injury news, rest, lineups) and from calibrating the pooled forecast than from adding or pruning columns.
- The under-confidence is consistent with an early-stopped classifier (shrunk toward the base rate) pooled linearly with two other views. Pooling in probability space keeps the mixture flatter than its most confident member. This is an inference from the measured view SDs; it was not tested by refitting.

### Gaps
- Walk-forward cells do not store per-game or per-view probabilities, so I could not check the blend's calibration slope outside March-April 2026 or fit stacking weights.
- Home-win rates in the bubble and empty-arena seasons were not measured.

## 2. Evaluation: why is held-out (0.773) so far above walk-forward (0.687)? Selection bias from many lab arms? Confidence intervals? Does the production retrain match the evaluated protocol?

### Takeaway
The walk-forward harness is careful: expanding origin, comparisons paired within (month, seed), a seed-noise floor and a burn-in. Several things about the headline numbers are still wrong or missing:
- The 308-game "held-out" is the season's easiest window. Its own walk-forward months score 0.77-0.81, the market hit 0.776 on the same games, and the 95% CI is 0.73-0.82.
- Those games sit inside the folds used to choose the recipe.
- Standard errors treat the three seeds per month as independent and are not adjusted for roughly 30 arms compared on the same folds.
- Feature screening judged on accuracy rather than Brier.
- What was evaluated (single model, 80/10/10 split, burn-in ≥ 10, no October) differs from what ships (10-seed refit on all games, served for every game).
- No confidence interval reaches the dashboard.

### Cited Findings

#### Flaws
- **E1 [High]: The held-out 0.773 is mostly a period effect plus a small sample.**
  - **The window.** The held-out is the last 25% of 2025-26, from 2026-03-04 on ([retrain_production.py L133-141](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/retrain_production.py); [metrics_2025_26.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/metrics_2025_26.json) `split_date`). Its home-win base rate is 0.614 against about 0.555 for the season (`baseline_home_rate` 0.6136; [walk_forward.py L5-8](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/walk_forward.py)).
  - **The walk-forward's own months agree.** For the shipped arm ("candidate"), blend accuracy by month is 0.766 in 2026-03 and 0.809 in 2026-04, against 0.589 in 2025-12 and 0.628 in 2026-01. April is high in every year: 0.710 in 2024-04, 0.752 in 2025-04 (check C on [impact_v5_lab.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/impact_v5_lab.json)).
  - **The market agrees.** The closing line was 0.776 accurate on the same games ([market_check.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/market_check.json)) and was itself under-confident there, slope 1.31 (check B).
  - **Sample size.** Bootstrap 95% CI for held-out blend accuracy is [0.727, 0.818] and for Brier [0.151, 0.184] (check A). The training rows score only 0.684-0.701 (check A).
  - **Model differences contribute too.** Walk-forward uses single-seed models trained on 80% of prior games; the shipped 10-seed refit should be roughly 1 point better (+0.0103 ± 0.0051 at N = 10, [seed_ensemble.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/seed_ensemble.json)).
  - **What the project already does.** [DURUM.md L30-34](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md) and the metrics `note` call the single split optimistic.
  - **Fix.** Always quote the held-out figure with its CI and with the market's accuracy on the same games, or retire it in favour of walk-forward.

- **E2 [Medium-High]: The "held-out" games are not untouched by model selection.**
  - **Evidence.** The 18 lab folds run from 2023-11 to 2026-04 and include 2026-03 and 2026-04 (check C; [DURUM.md L16-18](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)). The features, recipe and impact pricing were all chosen from those labs ([retrain_production.py L91-98, L106-112](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/retrain_production.py); [DURUM.md L69-95, L337-353](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)).
  - **How it is presented.** The dashboard says the models "never saw" these games ([index.html L227-228](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/templates/index.html)) and badges them "gerçek out-of-sample" ([app.py L123-126](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/app.py)).
  - **Why it matters.** The statement is true for fitting but not for selection, so the held-out and walk-forward numbers are both selection-set numbers.
  - **Fix.** Freeze a final test that no lab touches; 2026-27, scored prospectively, is the natural one. Relabel the 308 games as "not fitted on; used in model selection".

- **E3 [Medium]: Standard errors are optimistic and not multiplicity-adjusted.**
  - **How the SEs are computed.** Paired SE = sd/√54 over cells ([consistency_lab.py L267-275](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/consistency_lab.py)), although each month's three seeds share the same test games. walk_forward flags "ANLAMLI" (significant) at |mean| > 2 SE ([walk_forward.py L354](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/walk_forward.py)).
  - **Month-clustered re-measurement** (check C on [consistency_lab_round2.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/consistency_lab_round2.json)), shipped recipe reg+mono+out vs pregame+out:

    | Metric | Mean change | Naive SE | Clustered SE | Note |
    |---|---|---|---|---|
    | Classifier accuracy | +0.0114 | 0.0045 | 0.0053 | t ≈ 2.15; ahead in 13 of 18 months |
    | Brier | −0.0061 | 0.0014 | 0.0016 | t ≈ 3.8 |
    | Blend accuracy | +0.0013 | 0.0021 | 0.0028 | |
    | Blend Brier | −0.0013 | 0.0004 | 0.0004 | |
    | Margin MAE | −0.073 | 0.018 | 0.026 | |

  - **How many comparisons.** About 30 arm comparisons ran on the same folds:
    - 7 arms in round 1 and 4 in round 2 ([consistency_lab.py L16-37, L350-352](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/consistency_lab.py));
    - 4 impact-cache runs ([impact_v5_lab.py L16-23](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/impact_v5_lab.py));
    - 7 feature arms ([walk_forward_7arms.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/walk_forward_7arms.json));
    - about 8 training-protocol arms ([training_protocol.py L17-38](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/training_protocol.py));
    - 4 ensemble sizes.
  - **Why it matters.** At roughly 30 looks, a t = 2.15 accuracy win is within what selection alone produces (a Bonferroni-style bar at α = 0.05 is t ≈ 2.8-3.2). The Brier and log-loss gains survive. The shipped recipe's case therefore rests on probability quality, as [DURUM.md L94-95](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md) itself notes.
  - **What the project already does.** It recognises the problem informally: "the availability family looked real at 105 cells and was gone at 315" ([training_protocol.py L47-48](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/training_protocol.py)).
  - **Fix.** Cluster or block-bootstrap by month, pre-declare Brier or log loss as the primary metric, apply Holm correction, and confirm the winner on untouched data.

- **E4 [Medium]: Feature screening used accuracy, the least sensitive metric.**
  - **Evidence.** walk_forward tests paired deltas only on accuracy ([walk_forward.py L331-333, L354](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/walk_forward.py)). The conclusion that every family "landed inside its own error bars" ([training_protocol.py L5-9](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/training_protocol.py)) rests on that. Brier did move: A+all 0.2096 vs A 0.2126 (−0.0030), A+rating 0.2113, A+avail 0.2118 ([walk_forward_7arms.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/walk_forward_7arms.json)).
  - **Why it matters.** Opponent-adjusted ratings and rest may have been rejected on a coarse metric.
  - **Fix.** Re-screen those families on paired Brier or log loss with month-clustered SEs, on the honest pregame dataset.

- **E5 [Medium]: The evaluated protocol is not the shipped protocol.**

  | Aspect | Lab (walk-forward) | Production |
  |---|---|---|
  | Models | one LightGBM per cell | 10 seeds |
  | Training rows | first 80% of prior games, stopped on the next 10% (patience 50), never refit; regressor 90/10 ([consistency_lab.py L172-204](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/consistency_lab.py)) | probe on 90%, refit on 100% at that tree count, patience 30 ([retrain_production.py L188-212](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/retrain_production.py)) |
  | Recent games | all prior games | drops the last 308 games of 2025-26 (L133-141) |
  | Score models | never evaluated under the shipped recipe (L96-98) | shipped |
  | Margin mapper | fitted on the same rows that early-stopped the regressor: `r_stop` = last 10% of `tr` = `cal_df` ([consistency_lab.py L189-200](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/consistency_lab.py)), a mild optimism | clean ([retrain_production.py L361-376](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/retrain_production.py)) |

  - **Why it matters.** The honest numbers describe a sibling of the shipped artifact, not the artifact itself.
  - **Fix.** Have the walk-forward call the exact production fit path (`fit_members` + blend build + calibrator) per fold, including the score models.

- **E6 [High for current use]: Walk-forward never scores a team's first 10 games, which is exactly where the season is now.**
  - **The rule.** Burn-in keeps a test game only if both teams already have ≥ 10 games ([consistency_lab.py L231-237](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/consistency_lab.py); [walk_forward.py L33-37](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/walk_forward.py)).
  - **Coverage.** This removes 13% of all games (check B). No October month appears among the 18 folds, which hold 3,224 test games per seed (check C).
  - **October accuracy in the older full walk-forward** (arm A, burn-in 0, older 190-feature dataset): 0.611 in 2023-10 (n = 54, Brier 0.230), 0.541 in 2024-10 (n = 71, Brier 0.244), 0.635 in 2025-10 (n = 80, Brier 0.222). The same arm scores 0.672 at burn-in 10 ([walk_forward_2025_26.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/walk_forward_2025_26.json); check D).
  - **Why it matters.** On 2026-10-25 every game is in the excluded regime, yet the dashboard quotes 0.687.
  - **Fix.** Report and display an early-season stratum separately (games 1-10, by month).

- **E7 [Medium]: The market comparison lacks paired uncertainty, and the gaps are statistically clear.** (Evaluation only, not wagering.)
  - **What is reported.** market_check reports a binomial SE only for the against-the-spread rate ([market_check.py L81-82, L122-131](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/market_check.py)).
  - **Paired bootstrap re-measurement** (check A), model minus market:
    - Brier +0.0142, 95% CI [+0.0059, +0.0221], SE 0.0041.
    - Margin MAE +0.91 (SE 0.21).
    - Total MAE +0.89 (SE 0.29).
  - **Encompassing test.** Regressing the outcome on logit(market) and logit(model) gives market coefficient 1.36 (bootstrap SE 0.36) and model coefficient −0.07 (95% CI −1.06 to 0.82) (check B). The model shows no detectable information beyond the close in this window.
  - **Against the spread.** The model's side of the closing spread covered 45.1% ± 2.8% ([market_check.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/market_check.json)). Where the model disagreed with the line, the line was right more often.
  - **Scope.** One season, 308 games, one book ([DURUM.md L97-110, L515](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)).
  - **Fix.** Add paired bootstrap or Diebold-Mariano tests and the encompassing test to market_check. Build a multi-season closing-line archive.

- **E8 [Low]: The "±" shown is month-to-month spread, not uncertainty of the mean, and it belongs to a different metric.**
  - **What is shown.** The "±0.066" on the dashboard is the classifier's month SD ([retrain_production.py L253](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/retrain_production.py); [consistency_lab.py L278-287, L301](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/consistency_lab.py); [index.html L221](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/templates/index.html)). It is displayed next to the blend's accuracy.
  - **The missing CI.** From the 18 monthly blend accuracies of the shipped arm: mean 0.6865, month SD 0.055, so a 95% CI on the mean of about ±0.026, i.e. 0.661-0.712 (my arithmetic from check C).
  - **Fix.** Show CI(mean) and label the spread separately.

#### Done well
- **Sound harness.** Expanding-origin monthly folds, paired (month, seed) deltas, a seed-noise floor and multiple burn-ins ([walk_forward.py L1-37, L191-214](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/walk_forward.py)).
- **Leak sizes measured, not assumed.** Pregame vs observed roster raised margin MAE by 0.80 in 18 of 18 months (check C on [consistency_lab.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/consistency_lab.json)).
- **Staleness study.** Training-cutoff lags of 0-150 days are within noise, e.g. lag 30: +0.007 ± 0.006 ([staleness.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/staleness.json)).
- **Market used as an outside yardstick.** Vig is removed and betting framing is explicitly refused ([market_check.py L4-19](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/market_check.py)). The earlier "57% ATS" was correctly traced to the leak ([DURUM.md L107-110](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)).
- **Honest numbers are pulled from the lab report rather than recomputed** ([retrain_production.py L232-255](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/retrain_production.py)).

### Inferences
- Realistic current skill is about 0.66-0.69 accuracy and Brier about 0.204-0.211 from mid-November to April, lower in October. The 0.773 / 0.167 figures describe March-April conditions, not the model.
- The shipped recipe's demonstrated benefit is probability quality (Brier, contradictions), not accuracy. For the blend, the recipe change bought about −0.0013 Brier and zero accuracy.
- The 0.014 Brier gap to the closing line and the null encompassing coefficient fit a model that lacks late information the market prices (injury news, rest, lineups) and is under-confident.

### Gaps
- No per-game walk-forward predictions are archived, so the market comparison cannot be extended to earlier months, and per-month calibration cannot be checked.
- Closing lines exist only for these 308 games ([DURUM.md L515](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)).

## 3. What is missing to forecast 2026-27 games in production, and is there in-season updating?

### Takeaway
As of 2026-10-25 the ML system cannot produce a forecast for an unplayed 2026-27 game:
- Features exist only for games already in the engineered dataset, which ends with 2025-26.
- Pre-game availability is derived from post-game box-score comments.
- The box-score caches depend on a database pull that predates the season and is offline.
- The shipped models leave out the most recent 308 games.
- Season identifiers are hard-coded.

Neither the ML model nor the simulator has an in-season updating loop.

### Cited Findings

#### Flaws
- **P1 [High]: There is no feature path for unplayed games.**
  - **Evidence.** DURUM states the dashboard predicts only games in the dataset and that no path builds pre-game features for an unplayed game ([DURUM.md L511-514](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)). `api_predict` returns 404 for any game_id not in the dataset ([app.py L591-597](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/app.py)). Models are trained through 2025_2026 ([metrics_2025_26.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/metrics_2025_26.json) `trained_on`). The only 2026-27 content is the simulation ([app.py L71-75](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/app.py)). A partial "as of date" roster helper exists ([predict_2025_2026.py L680-695](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/predict_2025_2026.py)).
  - **Fix.** Build an as-of feature builder; `pregame_roster` already sweeps chronologically ([DURUM.md L513-514](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)). Add a daily job that appends played games and emits forecasts for the next day's slate, archived with a timestamp so they can be scored later.

- **P2 [High]: Availability comes from the post-game box score.**
  - **Evidence.** The pre-game roster is the box score's listed players minus "DND/NWT/Injury/Rest…" comments ([pregame_roster.py L20-37, L89-93](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/pregame_roster.py)). The timing caveat (late scratches) is acknowledged at L33-37. An injury-report feed is listed as the first next step ([DURUM.md L509-510](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)).
  - **Why it matters.** In production this information must come from the official injury report at a fixed lead time. Back-tests effectively assume tip-off knowledge, which the closing line also has.
  - **Fix.** Ingest time-stamped injury reports and evaluate at the same lead time production will use.

- **P3 [High]: The data dependencies are stale or offline for 2026-27.**
  - **The cache.** `phonedb_cache` is a 12 September 2026 pull that does not include 2026-27, the database is unreachable, and `.env` is missing on this clone ([DURUM.md L416-431, L525-526](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)).
  - **Who reads it.** season_sim, value_engine and pregame_roster all read their box-score inputs from it ([season_sim.py L82-83](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/season_sim.py); [value_engine.py L146-148](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/value_engine.py); [pregame_roster.py L83](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/pregame_roster.py)).
  - **Missing data.** 12 of the 28 games played so far lack defensive, tracking and matchup data ([DURUM.md L426-429](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)).
  - **Fix.** Build the 2026-27 caches straight from the `nba_data` CSVs and make the database optional. Have features degrade gracefully when tracking or matchup data is late.

- **P4 [Medium]: The shipped models exclude the latest 25% of 2025-26.**
  - **Evidence.** `chrono_split` holds out the last 308 games ([retrain_production.py L133-141](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/retrain_production.py)); n_train is 10,441 ([metrics_2025_26.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/metrics_2025_26.json)).
  - **Fix.** For live use, retrain on all games and keep the held-out run as a separate evaluation artifact.

- **P5 [Medium]: Season identifiers are hard-coded.**
  - **Evidence.** `TARGET_SEASON = "2025_2026"` and `*_2025_26` file names in [retrain_production.py L71-81](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/retrain_production.py), [app.py L59-70](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/app.py), [simulation.py L62-66](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/simulation.py) and [calibration.py L51-57](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/calibration.py).
  - **Why it matters.** Adding 2026-27 rows would silently put them in "context" training while still holding out 2025-26.
  - **Fix.** Parameterise the season and split.

- **P6 [Medium]: There is no in-season updating or monitoring loop.**
  - **ML model.** Rolling features update as games are played, and the staleness study says monthly retraining is unnecessary ([staleness.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/staleness.json)). Nothing monitors calibration or league-environment drift. The 2026-27 opening week already differs from what the simulation assumes: FTA 27.0 vs 22.6 per team-game, fouls 24.0 vs 19.8, turnovers 16.4 vs 13.8, 3P% 32.9 vs 36.5 ([sim_vs_real_2026_2027.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/sim_vs_real_2026_2027.json)).
  - **Simulator.** Priors and player profiles are fixed at preseason ([season_sim.py L999-1093](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/season_sim.py)). sim_vs_real replays the preseason setup rather than conditioning on results ([sim_vs_real.py L91-96, L109-122](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/sim_vs_real.py)).
  - **Fix.** Add a nightly calibration and drift monitor. For the simulator: Bayesian team-strength updating (Elo or Massey with a preseason prior), updating of observed minutes and availability, and re-simulation of the remaining schedule from actual standings.

- **P7 [Medium]: The early-season regime is unevaluated and the priors are crude.**
  - **Evidence.** Rolling windows deliberately cross season boundaries: "a player's October form is his previous April" ([pregame_roster.py L39-45](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/pregame_roster.py); [rebuild_impact_features.py L80-82](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/rebuild_impact_features.py)). That is reasonable for player-level features, but team-level L3/L5/L10 box-score features carry last season's team identity through summer roster turnover. October is not evaluated (E6).
  - **Fix.** Add explicit preseason team priors (last season regressed toward the mean and adjusted for roster changes) plus games-played as a feature, and evaluate games 1-10 separately.

#### Done well
- Resumable fetchers for the 2026-27 schedule, game ids and play-by-play/box data, plus a completeness checker, exist and are documented ([DURUM.md L422-438, L492-502](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)).
- DURUM honestly lists the missing production pieces as next steps ([DURUM.md L507-537](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)).

### Inferences
- The shortest path to live forecasts: an as-of feature builder, an injury-report feed, CSV-built caches, and a full-data retrain with season parameters. Archiving every forecast before tip-off would also create the clean prospective test set that E2 needs.

### Gaps
- I did not trace whether `FeatureEngineer` (predict_2025_2026.py / 2023_2024.py) can run on a game with no final score. Its rolling stats are computed from game results, so a stub row path would be needed.

## 4. Season simulator (season_sim.py, sim_vs_real.py): what is structurally weak beyond rookies, injuries and the opening-week environment?

### Takeaway
It is an impressive, well-isolated possession generator that reproduces league averages, but as a forecaster it is structurally weak:
- Team strength emerges from per-minute box-score rates plus three team effects carried over unshrunk.
- Minutes are frozen at last season's levels and rookies get replacement-level minutes.
- Injuries are a Markov chain driven by last season's games played.
- There is no rest, travel or altitude, and a single global home edge.
- A score-effect knob compresses team separation to about 80% of reality.
- Uncertainty is a uniform shooting shock.
- Nothing updates during the season.

Its own backtest is no better than "same as last season", and on the first 28 games of 2026-27 it trailed last season's records on Brier.

### Cited Findings

#### Flaws
- **S1 [High]: Minutes allocation is frozen and role-blind.**
  - **Evidence.** Minutes per game come from the player's most recent season ([season_sim.py L356-364](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/season_sim.py)). The rotation is the top 10 available players by those minutes, scaled to 240 with a 38-minute cap (L130-132, L467-480, L561-569). Possessions pick players in proportion to minutes × rate (L527-538). There are no role changes after trades or roster turnover, no coach tendencies, no foul trouble or blowout substitutions, and no fatigue.
  - **Measured result.** Player-level minutes correlate only 0.67 with reality over the first 28 games ([sim_vs_real_2026_2027.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/sim_vs_real_2026_2027.json)).
  - **Fix.** Model minutes as a role-based prior updated by preseason and regular-season minutes (e.g. empirical-Bayes shrinkage toward the depth-chart slot), with blowout bench minutes tied to game state.

- **S2 [High]: Rookies and unknown players are replacement-level; injuries are a memoryless proxy.**
  - **Rookies.** A player with no NBA history gets 8 minutes, availability 0.85 and replacement-level rates ([season_sim.py L125-127, L372-378](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/season_sim.py)). Boozer, Acuff, Lendeborg, Wilson and Peterson played 61-72 real minutes over two games each against 0.6-4.7 simulated ([sim_vs_real_2026_2027.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/sim_vs_real_2026_2027.json)).
  - **Injuries.** Availability is last season's share of games played, clipped to [0.55, 0.97], driven by a two-state Markov chain with persistence 0.75 ([season_sim.py L128-129, L541-558](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/season_sim.py)). Known season-opening absences are ignored: Ingram was inactive but got 64.8 simulated minutes (same JSON). A player with fewer than 10 games last season gets 0.8 × his historical minutes and availability 0.75 (L360-364), regardless of why he missed games.
  - **Fix.** Use draft-slot priors for rookies' minutes and rates (or summer league and preseason minutes); feed the preseason injury list and expected return dates; refresh availability from the official report.

- **S3 [Medium]: Team strength is box-score emergent, and the three team effects are carried at full strength.**
  - **Evidence.** Team offence is the sum of player per-minute rates. Defence is opponents' eFG relative to the league, forced turnovers are opponents' TOV rate, plus pace ([season_sim.py L397-412](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/season_sim.py)). These are built with `shrink=1.0`, i.e. no shrinkage (L1016), carried by players to new teams with `TEAM_EFFECT_KEPT = 1.0` (L138, L415-461), and applied as one multiplier to both 2P and 3P make probability (L628-629).
  - **Why it matters.** Opponent eFG includes opponent 3P% luck and is not schedule-adjusted. The comment admits the shrinkage choice "is not resolvable at this sample" (L133-137).
  - **Fix.** Shrink the team effects by their measured year-to-year reliability; separate the rim, 3P and FT components of defence.

- **S4 [Medium]: The score-effect knob trades team separation for a correct margin spread.**
  - **Evidence.** The leading team's shooting is cut by 1.7% per 10 points of lead, capped at 25 ([season_sim.py L115-116, L749-754](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/season_sim.py)). It was tuned so the 2025-26 margin SD matched (16.6 vs 16.4). The measured side effects are that team-to-team spread is about 80% of reality (5.2 vs 6.2) and the overtime rate is 2-3% against 4.4% ([DURUM.md L193-198](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)).
  - **Why it matters.** Good teams are systematically pulled toward .500, which compresses win projections and per-game probabilities.
  - **Fix.** Model game-state effects through rotation and minutes (bench in blowouts) and late-game fouling, then re-tune.

- **S5 [Medium]: The mechanics "validation" is in-sample tuning.**
  - **Evidence.** HOME_EDGE, SCORE_EFFECT and PACE_TIME_FACTOR were set against real 2025-26 totals in mechanics mode (six seeds) ([season_sim.py L108-120](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/season_sim.py)). The league proportions were measured on 200 games of 2025-26 (L90-106). DURUM then reports agreement on that same season as the generator's accuracy ([DURUM.md L193-196](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)).
  - **Why it matters.** The constants cannot follow a new league environment (P6).
  - **Fix.** Tune on earlier seasons and validate mechanics on a held-out season; re-estimate the league proportions on a rolling basis.

- **S6 [Medium]: There are no schedule effects and only one home edge.**
  - **Evidence.** HOME_EDGE = 0.010 shooting multiplier for every venue ([season_sim.py L113](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/season_sim.py)). The schedule only orders games and advances the availability chain (L1056-1068). There are no back-to-backs, rest days, travel, time zones or altitude.
  - **Note.** Rest features were screened in the ML model with no accuracy gain ([walk_forward_7arms.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/walk_forward_7arms.json) A+rest −0.0015 ± 0.0028), but the simulator's availability layer (load management on back-to-backs) is a different channel.
  - **Fix.** Add back-to-back rest probabilities for stars and a team-specific home edge with shrinkage.

- **S7 [Medium]: The uncertainty model is uniform and very wide.**
  - **Evidence.** Replications draw each team's season shooting multiplier as exp(N(0, 0.11)), the same for every team, and fixed for the whole season ([season_sim.py L139-146, L1046-1049](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/season_sim.py)). It was tuned so 10-90% ranges cover about 80% of outcomes, which makes ranges about 30 wins wide (L144-145; [DURUM.md L209-211](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)). sim_vs_real uses the same shock for per-game probabilities ([sim_vs_real.py L116](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/sim_vs_real.py)), which flattens them: DET-BOS p_home 0.502 against 0.612 from last season's log5; 10-90% margin bands of about ±24 points ([sim_vs_real_2026_2027.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/sim_vs_real_2026_2027.json)).
  - **Why it matters.** Teams with heavy turnover or rookies should carry more uncertainty than unchanged rosters.
  - **Fix.** Use team-specific shock SDs (e.g. scaled by minutes from new or unknown players), and shrink the shock as results arrive.

- **S8 [Medium]: Forecast skill is no better than the naive baseline, and the 28-game check is noise-level.**
  - **Backtest over five seasons.** Correlation with real wins 0.53 and MAE 8.7 wins, against 0.58 / 8.8 for "same as last season" ([season_sim.py L133-137, L1225-1226](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/season_sim.py); [DURUM.md L205-208](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)).
  - **First 28 games of 2026-27** ([sim_vs_real_2026_2027.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/sim_vs_real_2026_2027.json)):

    | Forecast | Correct winners | Brier | Log loss |
    |---|---|---|---|
    | simulation | 18/28 | 0.225 | 0.642 |
    | last season's log5 + home | 16/28 | 0.218 | 0.622 |
    | home team always | 13/28 | 0.257 | |

    Margin MAE was 13.6 against 15.0 for the naive baseline.
  - **The baseline itself is weak.** The log5 baseline uses raw, un-regressed win% ([sim_vs_real.py L137-157](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/sim_vs_real.py)), and no market or ML baseline is included (the ML model cannot score 2026-27 games, P1).
  - **Fix.** Add regressed-Elo and preseason market win-total baselines, and score on a full month or more before reading anything.

- **S9 [Low-Medium]: Player skill priors are thin and ignore age.**
  - **Evidence.** Per-minute rates are shrunk with only 50 replacement-level minutes; shooting priors are 60 2PA / 150 3PA / 40 FTA ([season_sim.py L123-124, L351-355](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/season_sim.py)). Profiles weight the last two seasons 1.0 / 0.5 with no aging curve (L1011-1015).
  - **Fix.** Use stronger, reliability-based priors per stat (e.g. stabilisation points) and an age adjustment.

- **S10 [Low]: The dashboard's "simulated season" is a single draw.**
  - **Evidence.** Standings rows, the W-L record, the streak and the last 10 come from one saved realisation; the 30-replication mean and 10-90% range are attached as an extra column ([season_sim.py L1340-1373](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/season_sim.py); [index.html L701-729](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/templates/index.html)). That draw picked the real winner in only 15 of 28 games (`saved_draw_same_winner`, [sim_vs_real_2026_2027.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/sim_vs_real_2026_2027.json)).
  - **Why it matters.** Leaders and seeding zones reflect one random world.
  - **Fix.** Order and colour standings by the replication mean or median and show the single draw as an example.

#### Done well
- **Isolation.** Strict sandboxing with before/after project fingerprints ([season_sim.py L46-51, L179-221](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/season_sim.py)), and an exact reproduction check (1,200 of 1,200 scores) before replaying ([sim_vs_real.py L99-106](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/sim_vs_real.py); [sim_vs_real_2026_2027.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/sim_vs_real_2026_2027.json)).
- **Measured, not guessed.** League proportions are measured rather than assumed ([season_sim.py L90-106](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/season_sim.py)), and backtest skill is reported against a naive baseline.
- **Honest labelling.** The UI band discloses limited forecast power ([index.html L656-663](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/templates/index.html)).
- **Deliberately independent.** The generator does not use the trained models, so it can test them ([season_sim.py L31-33](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/season_sim.py)).

### Inferences
- The largest preseason forecast errors come from minutes (who plays how much), not from per-possession mechanics. Fixing S1 and S2 (roles, rookies, injuries) and adding updating (P6) would probably matter more than further tuning of the possession knobs.
- S4 and S7 both flatten probabilities, which is consistent with the simulation's Brier trailing a simple log5 even though it picked more winners.

### Gaps
- Whether the 2026-27 foul and free-throw surge reflects a rule or officiating change rather than an opening-week effect cannot be resolved from 28 games. DURUM notes that 2025-26's first 28 games also had high FTA (26.0) ([DURUM.md L458-462](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)).

## 5. Value/impact engine (value_engine.py pricing 5.1, impact_v5.py): weaknesses compared with established player metrics

### Takeaway
Pricing 5.1 is a clear improvement over the v4 production count: misses are charged, rebound type is fixed, assists and free throws are complete, and correlation with on-court +/- rose from 0.14 to about 0.42. It remains a box-score accounting score:
- Defence beyond steals, blocks and defensive rebounds is invisible.
- Credits are not points-conserving: assists, steals and blocks are paid on top.
- The 1.00 points-per-possession baseline rewards shot volume.
- There is no teammate or opponent adjustment, no possession normalisation and no regression to priors.
- Validation is against same-season raw on-court +/-.
- The headline "team sum ↔ margin 0.95" is largely an accounting identity, and switching the model's inputs from v4 to v5 changed nothing measurable.

### Cited Findings

#### Flaws
- **V1 [Medium]: Defence is mostly invisible.**
  - **Evidence.** Defensive credit exists only for steals (+1.15), blocks (+0.6) and defensive rebounds (+0.26), with fouls as debits. Nobody on defence is credited for an opponent's miss ([value_engine.py L25-37, L301-336](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/value_engine.py)). The engine says so itself: it does not see screens, spacing, closeouts or who a player guarded (L117-121; [DURUM.md L163-164](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)).
  - **Why it matters.** Rim protectors without blocks and point-of-attack defenders are under-rated; steal-gamblers are over-rated.
  - **Fix.** Add matchup-based defensive credit (the project already holds matchup and defensive-tracking tables) or a regularised on/off (RAPM-style) defensive component.

- **V2 [Medium]: The value totals are not points-conserving.**
  - **Assists.** The passer gets 30% of the basket's value on top of the scorer's full credit ([value_engine.py L41-55, L500-516](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/value_engine.py)). An assisted three is booked at +2.6 above baseline.
  - **Steals and turnovers.** The turnover costs its committer −1.15 and the steal pays the thief +1.15 for the same possession change (L32-33, L322-325).
  - **Blocks.** A block pays +0.6 on top of the shooter's −0.74 (L36, L326-327).
  - **How the rule was chosen.** By maximising correlation with on-court +/-; the alternatives were nearly flat at 0.442 / 0.445 / 0.444 ([DURUM.md L281-298](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)).
  - **Why it matters.** "Points" is not a conserved unit, so value per game cannot be summed to a team's net points or compared across roles.
  - **Fix.** Split each possession's value change between its participants (scorer, passer, rebounder, defender) so that the sums equal team points above expectation.

- **V3 [Medium]: The 1.00 baseline rewards volume.**
  - **Evidence.** The baseline is set "a little below the league's ~1.08" so that taking shots at ordinary efficiency "is worth a little" ([value_engine.py L26-28, L153](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/value_engine.py)). By my arithmetic from the price table (L29-30), with offensive rebounds paid to rebounders, a 50% two is worth about +0.13 per attempt and a 36% three about +0.25 per attempt. Power-ranking OVR uses value per game, not per possession ([power_rankings.py L289-292](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/power_rankings.py)), so minutes and pace add to volume.
  - **Why it matters.** Westbrook's 2017-18 value is still positive and large, 6.7-7.9 per game ([DURUM.md L155-157, L302-303](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)).
  - **Fix.** Price against league-average points per possession (or against the team's expected points per possession), report per-100 possessions, and keep a separate volume dimension.

- **V4 [Medium]: There is no context adjustment and no regression to priors.**
  - **Evidence.** Values are raw per-game sums. There is no adjustment for teammates or opponents and no shrinkage of small samples ([value_engine.py L436-497](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/value_engine.py)). Weekly player OVR can rest on a single game ([power_rankings.py L83, L318-323](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/power_rankings.py)).
  - **Why it matters.** Established metrics such as RAPM, EPM, BPM and LEBRON regress toward priors and adjust for who is on the floor. This is an inference from general knowledge, not sourced in this audit.
  - **Fix.** Empirical-Bayes shrinkage of per-100 value toward a box-score prior, weighted by possessions, and an opponent-strength adjustment.

- **V5 [Medium]: The validation target is weak and same-sample.**
  - **Evidence.** The score is evaluated by correlation with the mean box-score `plus_minus` per 36 minutes in the same season's games ([value_engine.py L623-654, L657-662](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/value_engine.py)). Raw on-court +/- is dominated by teammate and opponent quality, and it shares the player's own points with the score. The assist share was tuned on three seasons; on six held-out seasons the gain was 0.379 → 0.396, worse in 2025-26 by −0.016 ([DURUM.md L300-302](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)).
  - **Fix.** Validate prospectively: does value from games 1..k predict lineup or team net rating in later games, or next season's team net rating from the roster, against simple baselines (box-score points per possession, a BPM-like linear model)?

- **V6 [Medium]: "Team sum ↔ margin 0.95" is accounting, not validation, and v4 → v5 did not change the forecasts.**
  - **The correlation.** The cache's team value difference correlates 0.95 with the actual margin, against 0.80 for v4 ([impact_v5_summary.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/impact_v5_summary.json); [impact_v5.py L99-110](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/impact_v5.py)), and DURUM presents it as improvement ([DURUM.md L332, L353](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)). But team sums of points-minus-possession-cost reconstruct the margin by construction.
  - **The forecasts.** Shipped arm minus previous inputs (check C on [impact_v5_lab.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/impact_v5_lab.json)):
    - classifier accuracy −0.0044 (clustered SE 0.0061);
    - blend accuracy +0.0019 (0.0017);
    - Brier +0.0004 (0.0015);
    - margin MAE +0.004 (0.012).
  - **Fix.** Present the 0.95 as an accounting check, and judge pricing only by out-of-sample prediction.

- **V7 [Low-Medium]: Leverage weights are hand-set.**
  - **Evidence.** Weights run from 0.3 to 1.25 by period, margin and clock; there is a 0.7/0.5 discount before the fourth quarter and 1.25 in the last 5 minutes of a ≤5-point game ([value_engine.py L162-163, L262-276](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/value_engine.py)). The weights are not derived from win-probability added. Using them as model features cost Brier +0.0030 ± 0.0015 ([impact_v5.py L45-54](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/impact_v5.py)).
  - **Why it matters.** Up-weighting clutch events adds small-sample noise to player ratings (inference).
  - **Fix.** Derive the weights from a win-probability model, or report leverage-free and leverage-weighted ratings side by side.

- **V8 [Low]: The impact_v5 docstring contradicts the code.**
  - **Evidence.** The docstring says the per-player number is "leverage-weighted" ([impact_v5.py L23-24](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/impact_v5.py)), but the cache is built from the unweighted run (L45-55). Player evaluation groups by `player_name` ([value_engine.py L642](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/value_engine.py)), which merges players who share a name.
  - **Fix.** Correct the docstring and group by player id.

#### Done well
- **Bugs found and fixed.** v4's "every rebound is offensive" string test and the missing miss, assist and free-throw pricing ([impact_v5.py L4-11](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/impact_v5.py); [value_engine.py L1-23](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/value_engine.py)).
- **Careful attribution and accounting.** The engine credits 99.8-100% of box-score assists ([value_engine.py L57-71](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/value_engine.py)), and free-throw possession accounting is exact (L90-100).
- **The right weighting in the right place.** Leverage is used for player ratings and removed from model features, based on a measured Brier cost ([impact_v5.py L45-54](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/impact_v5.py)).
- **Honest scope.** "What it is not" is stated plainly (L117-121).

### Inferences
- For forecasting, the impact family is largely redundant with point-margin features. The team sum of value ≈ margin, and v4 versus v5 made no measurable difference. Player value matters most where rosters change: the pregame roster and missing-player features, and the simulator.
- For player evaluation, the main gaps are defence, context adjustment and shrinkage. As written, the "value OVR" is still partly a volume and role measure.

### Gaps
- I did not benchmark value against external public metrics (EPM, LEBRON, BPM). No such data is in the repo and the comparison is outside this scope.

## 6. Power rankings (power_rankings.py) and dashboard (app.py, templates/index.html): misleading displays, honesty of the metrics, stale data, robustness

### Takeaway
The dashboard is unusually candid: in-sample and held-out badges, a "how measured" note, and SIM labels with a trust warning. A few displays still mislead or mix metrics:
- The header pairs blend accuracy with the classifier's AUC and month SD.
- The team header pools mostly in-sample games into "model accuracy".
- The held-out text says the models "never saw" games that were used in model selection.
- The "empty stats" badge and the game-detail roster panel are built on the discredited v4 score.
- Real-data views stop at 2025-26.
- In the rankings, early-season team strength switches abruptly from Elo to Massey at 100 games, and weekly player OVRs can rest on one game.
- Robustness gaps are minor: integer parsing of query strings, and incomplete requirements.

### Cited Findings

#### Flaws
- **D1 [Medium]: The header mixes metrics and carries stale constants.**
  - **What is shown.** `/api/meta` returns `accuracy` = blend accuracy (0.687), `accuracy_std` = the classifier's month SD and `auc` = the classifier's AUC ([app.py L104-117](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/app.py); [retrain_production.py L248-254](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/retrain_production.py)). The page shows these as one row of pills ([index.html L220-224](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/templates/index.html)).
  - **Stale constants.** The naive baseline is hard-coded as "21 months" while the lab uses 18 ([app.py L85](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/app.py)). The fallback and docstring still cite "12 months, 0.667" (L18-20, L86-92).
  - **What is missing.** Brier, a calibration measure and the market benchmark are not shown.
  - **Fix.** Report blend accuracy, Brier and log loss with CIs from the same arm, plus the market's numbers on the held-out games.

- **D2 [Medium]: The team header's "model isabeti" pools in- and out-of-sample games.**
  - **Evidence.** `hit / n` runs over every listed game ([index.html L293-312](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/templates/index.html)), almost all of them training games. Game cards show ✓/✕ for training games too (L274-291).
  - **Impact.** Measured in-sample accuracy is only 0.684 against 0.687 in walk-forward (check A), so the inflation is small. The display still conflates two different things.
  - **Fix.** Show only held-out or walk-forward accuracy in summaries, and grey out ticks on in-sample games.

- **D3 [Medium]: The held-out wording overstates.** "Modeller bu maçları hiç görmedi" ("the models never saw these games") and "gerçek out-of-sample" ([index.html L227-228](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/templates/index.html); [app.py L123-126](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/app.py)) ignore the selection use (E2) and give no CI (E1).

- **D4 [Medium]: The "empty stats" badge and the roster-impact panel use the discredited v4 score.**
  - **The badge.** "Boş istatistik" ("empty stats") fires when the production OVR (v4 impact) minus the value OVR is ≥ 8 ([power_rankings.py L300-315](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/power_rankings.py); [index.html L621](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/templates/index.html); [app.py L60-63](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/app.py)). v4 priced every rebound as offensive and charged nothing for misses ([impact_v5.py L5-11](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/impact_v5.py)). The badge therefore partly measures exposure to v4's bugs (rebounders, volume shooters), not stat-padding as such.
  - **The roster panel.** The game-detail panel ranks players by trailing v4 impact ([app.py L144-182, L474-502](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/app.py)), not by the v5 values the model uses.
  - **Fix.** Define "empty stats" from v5 components directly (e.g. share of value from miss cost and turnovers, or the gap between volume and per-possession value). Switch the roster panel to v5 or value.

- **D5 [Medium]: Rankings mechanics have an early-season jump, weak player qualification and an inflated Elo home edge.**
  - **Strength switch.** Season strength is a ridge Massey with alpha 2.0, equal weights and no margin-of-victory cap once at least 100 league games exist; before that it is Elo converted at 28 points per 1 point of spread ([power_rankings.py L14-19, L74-76, L103-118, L238-248](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/power_rankings.py)). At about 6.7 games per team, the switch makes ratings jump from Elo (with 75% carry-over) to a lightly regularised solve.
  - **Elo home edge.** It is 100 Elo ≈ 3.6 points ([team_ratings.py L46-49](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/team_ratings.py); [power_rankings.py L247](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/power_rankings.py)), against a measured 2025-26 home margin of about 1.7 ([DURUM.md L193-195](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)).
  - **Player qualification.** Minimum games are 1 for the week, 3 for the month and 10 for the season, together with a 12-minute average. The percentile curve gives 99 to the top player of every window ([power_rankings.py L81-84, L126-127, L318-323](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/power_rankings.py)).
  - **Fix.** Blend the Elo prior and Massey continuously with weights by games played (Bayesian). Cap margin of victory or damp garbage time in Massey. Re-estimate the Elo home edge. Require possessions-based minimums and shrink player OVR toward the season value for short windows.

- **D6 [Medium]: The real-data views are stale for the current season.**
  - **Evidence.** The real-data views (games, predictions, rankings) end with 2025-26 ([app.py L70](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/app.py) `TARGET_SEASON`; [metrics_2025_26.json](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/output/metrics_2025_26.json)). The only 2026-27 content is the preseason simulation (single draw, S10). The 28 real 2026-27 games are fetched but not shown ([DURUM.md L422-431](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)).
  - **Fix.** After P1 and P3, add the real 2026-27 games and rankings, and a "sim vs real" panel from sim_vs_real.

- **D7 [Low]: There are robustness gaps.**
  - **Query parsing.** `int(request.args.get("limit", …))` raises ValueError, i.e. an HTTP 500, on a non-numeric value ([app.py L586, L684, L724, L735, L786](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/app.py)).
  - **Requirements.** requirements.txt lacks flask, waitress (imported at [app.py L47, L831](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/app.py)) and scipy (imported at [simulation.py L54](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/simulation.py)) ([requirements.txt](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/requirements.txt); [DURUM.md L527-528](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)).
  - **Startup cost.** Startup loads the dataset, the caches, ratings and forecasts for all ~10.7k games in a single process ([app.py L251-314](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/app.py)).
  - **Stale README.** It still lists only 2023_2024.py and seasons up to 2024-25 ([README.md L60-61, L80](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/README.md)).
  - **Fix.** Use `request.args.get(..., type=int)`, complete the requirements, and refresh the README.

- **D8 [Low]: Two "Massey" ratings with different definitions appear on the page.**
  - **Evidence.** Game detail shows team_ratings' Massey (trailing 1,230 games, half-life 400, alpha 25; [team_ratings.py L52-54](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/team_ratings.py); [app.py L421-440](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/app.py)). The rankings show a season-to-date, equal-weight Massey with alpha 2 ([power_rankings.py L76, L238-244](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/power_rankings.py)).
  - **Fix.** Label them differently or unify the definitions.

#### Done well
- **Labels on every game.** In-sample and held-out badges per game ([app.py L119-126, L446-449](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/app.py)), a "how the numbers were measured" note ([index.html L225-230](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/templates/index.html)), and an accuracy figure read from the lab report rather than recomputed from the browsed games ([app.py L78-117](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/app.py)).
- **Fails loudly on model/dataset mismatch** ([app.py L185-200](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/app.py)).
- **The three views are shown with their disagreement** ([index.html L420-428](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/templates/index.html)).
- **The simulation is clearly marked** with a SIM tag and a trust warning that cites the backtest ([index.html L656-663](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/templates/index.html)).
- **Rankings state they describe, not forecast** ([power_rankings.py L48-54](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/power_rankings.py)).
- **Sensible serving.** Tailscale-only binding with waitress ([app.py L26-31, L809-838](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/app.py)), and bounded `lru_cache` ([power_rankings.py L383, L405](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/prediction_engines/power_rankings.py)).

### Inferences
- The honesty gaps are wording and metric-pairing issues, not hidden failures. Each can be fixed with small UI and metrics changes (D1-D3) once the evaluation itself stores CIs (E8) and a truly untouched test exists (E2).
- During October and November the early-season instability of both the rankings (D5) and the model (E6) is at its peak, and the page gives no warning for either.

### Gaps
- The page was not rendered in a browser; DURUM notes that only a headless DOM test was run ([DURUM.md L232-234](C:/Users/arcan/OneDrive/Desktop/nba_match_prediction/DURUM.md)). Visual or mobile issues were not audited.
- I did not measure how far the Elo-to-Massey switch moves ratings on a real season.

# External NBA game-prediction and season-simulation methods: state of the art, 2023-2026

Scope note for the report writer: these notes compare published methods and figures against a hobbyist LightGBM system (walk-forward accuracy about 0.69, Brier 0.211; about even with the market on picking winners, behind it on Brier, margin error and total error). Betting markets appear only as an accuracy benchmark. Each source carries a date. Anything from before 2023 is tagged **[OLDER]**. FiveThirtyEight's NBA model last changed on 12 Apr 2023, and in Oct 2026 its fivethirtyeight.com methodology URL redirected to abcnews.com, so all 538 material below comes from Wayback Machine snapshots.

## 1. What accuracy, Brier score and log loss do the best public models and the closing lines achieve, and what is the practical ceiling for pre-game prediction?

### Takeaway
Across 2023-2026, the strongest pre-game predictors all pick winners about 66-69% of the time: the closing market, player-based systems such as EPM, and good Elo models. The de-vigged 2025-26 closing moneyline was well calibrated, picked 68.7% of winners, and by my estimate from its published calibration table scored a Brier of about 0.198-0.199 and a log loss of about 0.58. Accuracy is almost saturated at roughly 69-70%, so systems now separate on Brier/log loss and on margin error. Ratings built only from results (Elo/Glicko/TrueSkill) reach just about 63% and a Brier of about 0.226 in one 2013-2023 study.

### Cited Findings
- **Closing market, 2025-26 (dataset updated Oct 2026).** 1,315 games: 1,224 regular season, 6 play-in, 85 playoff. NBA Cup knockout games are excluded. The source took consensus opening and closing odds, removed each bookmaker's margin proportionally, and used the median across brands, with a median of 23 bookmakers per game. Closing favourites won outright 68.7%. The average closing home-win probability was 55.2% against an actual home win rate of 55.6%. Home teams covered the closing spread 51.9% of the time, and overs hit 49.5% at the closing total. — [Kaggle: NBA 2025-26 Closing Lines and ATS Results (CC BY 4.0)](https://www.kaggle.com/datasets/minmaxer/nba-closing-lines-ats-2025-26)
- **Calibration of the 2025-26 closing no-vig home-win probability** (predicted mean → actual home win rate, games):
  - 0-0.2: 0.144 → 0.083 (84)
  - 0.2-0.3: 0.250 → 0.250 (108)
  - 0.3-0.4: 0.353 → 0.377 (154)
  - 0.4-0.5: 0.449 → 0.474 (192)
  - 0.5-0.6: 0.556 → 0.536 (196)
  - 0.6-0.7: 0.652 → 0.665 (197)
  - 0.7-0.8: 0.751 → 0.779 (199)
  - 0.8-1.0: 0.859 → 0.849 (185)
  - Source: [Kaggle notebook "NBA 2025-26: closing lines, favourites and ATS" (Oct 2026)](https://www.kaggle.com/code/minmaxer/nba-2025-26-closing-lines-favourites-and-ats)
- **The closing line holds information the opening line lacks (2025-26).** The home team covered the *opening* spread in:
  - 62.1% of games where the spread moved 1.5+ points toward home (203 games)
  - 54.6% where it did not move (306 games)
  - 33.3% where it moved 1.5+ points toward away (255 games)
  - Source: [same Kaggle notebook](https://www.kaggle.com/code/minmaxer/nba-2025-26-closing-lines-favourites-and-ats)
- **Bookmakers vs Polymarket, 2024-25 (student paper, Wake Forest ECN 345 under Prof. Koleman Strumpf, 20 May 2025).** 1,000 games involving 20 teams. Final pre-tip odds came from 9 sportsbooks via OddsPortal and were converted with the plain implied-probability formula; the paper describes no vig removal. Bookmaker favourites won 66% and Polymarket favourites 67%, with no significant difference. The paper says it computed Brier scores but the text gives no values. — [Polymarket Analytics research page](https://polymarketanalytics.com/research/nba-sportsbooks-vs-prediction-markets). Its 66% conflicts with the 68.7% above. They cover different seasons and samples (20 teams only, no de-vig, unclear treatment of playoffs).
- **Elo favourites, 2023-24 (Neil Paine's Elo, not market lines; 19 Mar 2024).**
  - 63% before December, 65% from December to the All-Star break, 62.8% after the break.
  - Since 2009-10, excluding 2019-20 and 2020-21, favourites won about 66%, early-season months included.
  - The post-ASB 2024 stretch was the season's least predictable by Brier, after league scoring fell about 5 points per game.
  - Source: [Neil Paine, "The NBA's midseason scoring crackdown also made games less predictable"](https://neilpaine.substack.com/p/the-nbas-midseason-scoring-crackdown)
- **EPM game model (Dunks & Threes, figures as of Oct 2026).** The baseline is an opponent-adjusted team net rating.

  | Period | Games | Average miss on final margin (EPM vs baseline) | Winners picked (EPM vs baseline) |
  |---|---|---|---|
  | Live, 2024-25 to 2025-26 | 2,643 | 11.0 vs 11.4 points | 68.4% vs 67.5% |
  | Backtest, 2005-06 to 2023-24 | 24,425 | 9.6 vs 9.9 points | 67.8% vs 66.3% |

  Source: [Dunks & Threes EPM methodology](https://dunksandthrees.com/about/epm)
- **Rating systems built only from results (MOVDA paper, arXiv v1, 31 May 2025).** NBA 2013-2023, 13,619 games, chronological 70/20/10 split:

  | Model | Accuracy | Brier |
  |---|---|---|
  | Standard Elo | 62.77% | 0.2274 |
  | Linear-MOV Elo | 63.18% | 0.2282 |
  | Glicko-2 | 63.18% | 0.2264 |
  | TrueSkill | 62.66% | 0.2294 |
  | MOVDA | 63.32% | 0.2258 |

  Caveats: hyperparameters were tuned on an ambiguously defined "same data set", and the paper's tables disagree with each other slightly. — [arXiv 2506.00348 (HTML v1)](https://arxiv.org/html/2506.00348v1)
- **Implausible ML claim.** An LSTM paper (arXiv, Dec 2025) reports 72.35% accuracy and 76.13 AUC-ROC on a 2004-05 to 2024-25 dataset. The abstract makes no comparison with betting odds, and the full-text HTML returned 404 when I tried it. — [arXiv 2512.08591](https://arxiv.org/abs/2512.08591)
- **A commercial modeller's admission.** PropsBot says its models beat the closing-line Brier in MLB and NHL, but that for the NBA and NFL they "aren't there yet". This is an undated vendor claim. — [PropsBot: What is a Brier score](https://propsbot.ai/brier-score/)
- **[OLDER, Oct 2015]** ESPN reported that "BPI favorites have won over 72 percent of NBA games". The sample period is not stated and is probably partly retrospective. — [ESPN: What is ESPN's NBA Basketball Power Index](https://africa.espn.com/nba/story/_/id/13984129/what-espn-nba-basketball-power-index)
- **[OLDER, 2017 arXiv / 2018 Annals of Applied Statistics]** Bayesian state-space models fitted to betting-market data over a decade showed the NBA has the widest spread in team talent and the largest home advantage of the four major North American leagues. The NHL and MLB were the most random. This is why NBA favourites win far more often than in other sports. — [Lopez, Matthews & Baumer, "How often does the best team win?"](https://arxiv.org/abs/1701.05976)
- **No authoritative figure for spread error.** An informal 2008 forum estimate puts the NBA spread error at "around 9" points. A back-of-envelope blog estimate puts the standard deviation of margin around the spread near 12 points. Neither is a rigorous measurement. — [Sportsbook Review forum](https://www.sportsbookreview.com/forum/handicapper-think-tank/59267-1-point-lines-in-american-sports); [Wayne Winston blog repost](https://www.goodreads.com/author_blog_posts/16483561-why-is-standard-deviation-of-nba-scores-about-point-spread-12-points)

### Inferences
- **Market Brier and log loss for 2025-26 (my calculation, not a published figure).**
  - I took the notebook's 8-bin calibration table (1,315 games) and computed Σ n_k[ȳ_k(1−ȳ_k) + (p̄_k−ȳ_k)²]/N. That gives a binned Brier of **≈0.1993**.
  - For a calibrated forecaster, the true score is lower by roughly the within-bin variance of p (about 0.001). So the 2025-26 closing-line Brier is **about 0.198-0.199**.
  - The same binned method gives a log loss of **≈0.58**.
  - Climatology (always predicting 55.6% for the home team) scores Brier ≈0.247. The market's Brier skill against it is therefore about 0.19-0.20.
  - The CSV holds per-game no-vig probabilities, so the exact values can be computed in a few lines.
- **What this means for the user's system.**
  - Its 0.69 accuracy is at the frontier: market 68.7%, EPM live 68.4%.
  - A Brier of 0.211 against about 0.198 is a gap of roughly 0.012-0.013, if the seasons are comparable. That is large next to the improvements published methods report: MOVDA's gain over TrueSkill is about 0.0036.
  - Accuracy around 0.69 alongside a Brier of 0.211 points to probabilities that are less sharp, less well calibrated, or both, compared with the market. The market's 2025-26 calibration bins sit within about 0.03 of the diagonal except the lowest bin, so the gap probably comes from information (injuries, rest, minutes) and resolution, not from the market being miscalibrated.
- **Practical ceiling.** Pre-game accuracy tops out around 68-70% in a normal season, and the market's Brier is about 0.20. Some seasons are noticeably less predictable: post-ASB 2023-24, and 2024-26 for EPM (live margin miss 11.0 against 9.6 in backtest) and for season-win projections (see Q5). Any comparison should therefore pair the model and the market on the same games.
- **Margin error.** EPM's live margin miss of 11.0 points (2024-26) is the only recent published margin-error figure I found for a public model. A system that misses the margin by less than about 10.5-11 points in 2024-26 is probably near public-model level. Whether it reaches market level is unknown, because I found no published closing-spread error figure.

### Gaps
- I found no authoritative, multi-season published closing-line Brier or log loss for the NBA. The 2025-26 figure above is derived from binned data.
- I found no published MAE/RMSE of closing spreads against actual margins, or of closing totals against actual totals, for 2023-2026. The Kaggle CSV (326.82 kB, CC BY 4.0, also on HuggingFace) has the `ats_margin` and `total_margin` columns needed to compute them. I did not download it, because that needs the user's approval.
- FiveThirtyEight's published accuracy for its NBA game forecasts (the "checking our work" pages) was not retrieved.
- The Polymarket/OddsPortal paper's Brier values are not in its text, only in figures.

## 2. Which model families perform best, and why?

### Takeaway
Two things do best: (a) the closing market, and (b) team ratings built from player-impact estimates (EPM, DARKO/DPM, RAPTOR, LEBRON-style RAPM with box-score priors) weighted by projected minutes and current availability, often blended with an Elo or efficiency term. Ratings that use only results are a weaker floor. Gradient boosting and neural networks add little unless they are fed availability and minutes information. Many high published accuracies (above 70%) come from in-game or leaked features.

### Cited Findings
- **FiveThirtyEight's evolution (methodology last edited 12 Apr 2023) [OLDER model, archived].**
  - Sequence: pure Elo, then CARM-Elo (2015), then fully player-based ratings (2018-19), then a RAPTOR + Elo mix from 2020-21.
  - Testing in the 2020 offseason found that about **35% Elo / 65% RAPTOR** gave the best predictions on average. The Elo weight varies from **0 to 55%** depending on how much the current depth chart matches recent lineups.
  - A "predictive" RAPTOR variant used in 2019-20 was retired because testing showed standard RAPTOR worked better.
  - Team rating: the minutes-weighted average of player ratings × 5 × **0.8** in the regular season (0.9 in the playoffs) to allow for diminishing returns. This is converted to expected points, then to a Pythagorean win% with exponent **14.3** (13.2 in the playoffs), then to Elo via Elo = 1504.6 − 450·log10(1/WP − 1).
  - Source: [538 "How Our NBA Predictions Work" (Wayback, captured 1 Jan 2024)](https://web.archive.org/web/20240101081135/https://fivethirtyeight.com/methodology/how-our-nba-predictions-work/)
- **[OLDER, 21 May 2015] 538 pure Elo.**
  - K = 20, chosen to minimise autocorrelation; the article reads this as the NBA having "relatively little randomness".
  - Home-court advantage = 100 Elo points, which the article equates to about 3.5 points.
  - Margin-of-victory multiplier with diminishing returns, adjusted for the Elo gap.
  - Teams keep 75% of their rating between seasons and revert 25% toward 1505.
  - Source: [538 "How We Calculate NBA Elo Ratings" (Wayback)](https://web.archive.org/web/20240105144602/https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/)
- **EPM (Dunks & Threes, Oct 2026).**
  - "A RAPM calculation with an SPM Bayesian prior". The box-score SPM was trained on 2003-04 to 2022-23 (4.9M possessions); a tracking-based defensive SPM on 2017-18 to 2022-23 (1.5M possessions).
  - RAPM is run nightly on all possessions since 2001-02, with exponential decay.
  - "Expected EPM" uses only data from before a given date.
  - Game predictions use "game-optimized" ratings that adjust for home court, rest, season type and opponent, plus the expected lineup including known injuries. The game model is retrained before each season using only earlier seasons.
  - Results: 68.4% live (2024-26) against 67.5% for a net-rating baseline. Margin miss 11.0 against 11.4.
  - Source: [EPM methodology](https://dunksandthrees.com/about/epm)
- **DARKO (darko.app, page current Oct 2026).**
  - Per-stat exponential decay, with β fitted by differential evolution, plus a modified Kalman filter. A gradient-boosted tree combines the two projections.
  - Adjusts for rest, travel, home court, opponent quality, per-stat ageing curves, seasonality (flattening around the All-Star break), interactions between stats, and team changes (which raise the learning rate).
  - DPM blends Box DPM with on-off data, weighting on-off by possessions/(possessions+10,000) on offence and possessions/(possessions+6,000) on defence.
  - A player's DPM scaled by his projected minutes share moves his team's rating, at about **2.7 wins per point of team rating**.
  - In a 17 Sep 2021 HoopsHype survey of NBA staff, 8 of 29 respondents named DPM their preferred public metric and 10 more said they trusted it.
  - Source: [DARKO About](https://darko.app/about)
- **LEBRON (BBall Index, page undated, embedded tweet dated May 2022).** RAPM regressed toward a box-score prior built from boxPIPM weights, with role-adjusted stabilisation (Medvedovsky's padding method) and luck adjustment. BBall Index calls LEBRON *descriptive*; forecasts come from ageing curves. The text gives no numerical predictive validation. — [LEBRON introduction](https://www.bball-index.com/lebron-introduction/)
- **Ridge and lasso RAPM variants (arXiv, Jun 2024).** On 2021-22 play-by-play (322,852 possessions), lasso and regularised multinomial-logistic RAPM variants beat standard ridge RAPM on the authors' validation criteria. Their "weighted expected points" (wEPTS) measure performed best. — [arXiv 2406.09895](https://arxiv.org/abs/2406.09895)
- **Neil Paine's public NBA models (Substack).**
  - 2024-25: schedule-adjusted offensive and defensive efficiency over each team's last 50 games (including the end of the previous season), regressed to the mean by 15 games. A separate playoff rating uses 110 games with postseason games weighted triple. 5,000 season simulations.
  - 2025-26 (page dated 14 Jun 2026): switched to two Elo variants (regular season and playoff) with preseason odds blended in. The Elo model does not handle injuries or lineups in real time; a "Composite Mode" blends in FanDuel-implied probabilities.
  - No backtest accuracy is published.
  - Sources: [2024-25 forecast](https://neilpaine.substack.com/p/2024-25-nba-forecast); [2025-26 Elo forecast](https://neilpaine.substack.com/p/2025-26-nba-elo-forecast-and-player)
- **Results-only rating systems.** Elo, Glicko-2, TrueSkill and margin-aware Elo cluster at 62.7-63.3% accuracy and a Brier of 0.226-0.229 on 2013-2023 NBA games. MOVDA, a margin-relative-to-expectation update, cuts Brier about 1.5% below TrueSkill. — [arXiv 2506.00348](https://arxiv.org/html/2506.00348v1)
- **TeamRankings.** Its own NBA preseason projections, built without player-level modelling, were less accurate than the betting market alone and less accurate than publicly available systems that do extensive player-level modelling. TeamRankings attributes this to individual players mattering much more in the NBA than in the NFL or college football. It now blends market information into its preseason ratings. The fetch returned 403, so this comes from search snippets. — [TeamRankings: How we make NBA preseason rankings](https://betiq.teamrankings.com/articles/preseason-rankings-ratings-explain/); [TeamRankings 2019-20 preseason post](https://www.teamrankings.com/blog/nba/nba-preseason-rankings-for-the-2019-20-season)
- **Selecting models by calibration (arXiv Mar 2023; later a journal article).** NBA models trained over several seasons and tested on one season with published odds. Choosing the model by calibration rather than accuracy changed the decision outcomes from −35.17% to +34.69% average ROI. The authors conclude calibration matters more than accuracy for probabilistic decisions. — [Walsh & Joshi, arXiv 2303.06021](https://arxiv.org/abs/2303.06021)
- **Typical gradient-boosting and ensemble papers use in-game statistics.**
  - PLOS ONE (2024) XGBoost+SHAP: the abstract describes "real-time prediction" from game performance indicators. — [doi 10.1371/journal.pone.0307478](https://doi.org/10.1371/journal.pone.0307478)
  - Scientific Reports (2025) stacked ensemble (Naive Bayes, AdaBoost, MLP, KNN, etc.). — [doi 10.1038/s41598-025-13657-1](https://doi.org/10.1038/s41598-025-13657-1)
  - A 2025 PLOS ONE systematic review covered 34 AI studies of professional-basketball outcome prediction; I could not retrieve its accuracy ranges. — [doi 10.1371/journal.pone.0326326](https://doi.org/10.1371/journal.pone.0326326)
  - A 2024 systematic review of ML in sports betting covers SVMs, random forests and neural networks across sports. — [arXiv 2410.21484](https://arxiv.org/abs/2410.21484)
- **Cross-sport evidence that market information dominates (2026, football in-play).** Calibrating team strengths to Betfair kick-off prices brought the model close to Betfair's own accuracy (70.2% vs 70.6%). A comparison with two other scoring models calibrated to the same odds "confirms that market calibration is the dominant driver of predictive accuracy". — [arXiv 2605.16066](https://arxiv.org/abs/2605.16066)
- **Cross-sport evidence that model choice matters little (2024, soccer).** Comparing ML and Poisson approaches across 5 leagues, the exact choice of features and of model "have only a minor influence on the prediction quality". — [arXiv 2408.08331](https://arxiv.org/abs/2408.08331)

### Inferences
- **Evidence-based ranking for pre-game NBA prediction:**
  1. Closing market.
  2. Player-impact × projected-minutes team ratings that are aware of injuries and rest (EPM, DARKO, RAPTOR+Elo), with EPM's 68.4% live accuracy and 538's design as reference points.
  3. Opponent-adjusted team efficiency or net rating (EPM's baseline: 67.5% live, 66.3% backtest).
  4. Pure results-based Elo/Glicko (about 63-66%).
- **Why player-based ratings win.** NBA rosters change within days (injuries, rest, trades). Rating systems built from results take weeks to notice, while a minutes-weighted player model updates the same day. 538 still found that roughly 35% Elo weight helps, so the best designs blend the two, and the blend weight depends on roster continuity.
- **For a LightGBM system:** the learner itself is unlikely to be the bottleneck. The evidence points to a better *team-strength input*: a minutes-weighted player-impact rating built from the available roster. Its probabilities should also be calibrated (for example by stacking onto a logistic/Elo-style backbone, or by isotonic/Platt calibration on a walk-forward holdout). Large accuracy jumps claimed for GBMs or NNs in the literature should be presumed leaky until shown otherwise.

### Gaps
- I found no public head-to-head comparison, under one protocol, of EPM, DARKO, LEBRON and RAPTOR team ratings for game prediction against the market.
- I found no 2023-2026 Bayesian hierarchical NBA game model that publishes Brier or log loss against closing lines.
- I found no Sloan conference (2023-2026) paper on NBA pre-game prediction with market comparisons. The only related Sloan paper I found is about ticket prices.
- Kaggle has no NBA game-prediction competition comparable to March Machine Learning Mania (which is NCAA).

## 3. Which inputs matter most, and by how much?

### Takeaway
Ranked by evidence, the inputs that matter most are:
1. Who plays and for how many minutes (availability and projected minutes, via player ratings).
2. A *current, shrinking* home-court advantage. About 100 Elo (~3.5 points) in 538's 2015 Elo; about 70 Elo (~2.4 points) in its 2022-23 model; home teams won about 54-56% of games in most seasons since 2020-21, against about 60% in 2000-13.
3. Rest and back-to-backs. 538 charged 46 Elo, about 1.6 points.
4. Travel distance and eastward time-zone shifts, worth about 1 point in the studies found.
5. Altitude, mainly for Denver and Utah.

For early-season uncertainty, published systems blend a preseason prior (player projections or market win totals) with in-season results. Evidence on referees and tanking is thin.

### Cited Findings
- **Availability and minutes: 538 (archived, 2023).**
  - From 2022-23, short-term injuries and player movement were automated using ESPN data. The model shows "Current" and "Full-Strength" ratings.
  - From 2020-21, a "load management" setting put certain stars on reduced minutes.
  - A player's talent rating is penalised if he plays far more minutes per game than projected.
  - In-season MPG = (12.6 × preseason MPG + current-season minutes) / (12.6 + games played).
  - From 2022-23, a 15-day history-based MPG estimate is blended with the depth-chart projection: 60/40 for a game today, shifting to 0/100 for games 15 or more days out.
  - Source: [538 methodology (Wayback)](https://web.archive.org/web/20240101081135/https://fivethirtyeight.com/methodology/how-our-nba-predictions-work/)
- **Minutes projections: EPM and DARKO.** EPM's minutes predictions beat a season-to-date average by about 15% (14.5% in the Estimated Skills table). — [EPM methodology](https://dunksandthrees.com/about/epm). DARKO beat DFS sites on points, rebounds, assists, blocks, turnovers and threes but lost to them on minutes, which it calls the hardest stat to project. — [DARKO About](https://darko.app/about)
- **What a player's absence is worth.** In DARKO, a player's DPM times his projected minutes share moves his team's rating, at about 2.7 wins per point over 82 games. — [DARKO About](https://darko.app/about)
- **Injuries and the betting line.** One practitioner says "the betting public always overreacts to superstar absences" and cites lines overshooting before correcting. This is anecdotal. — [Action Network injury guide](https://www.actionnetwork.com/education/how-injuries-affect-betting-lines-a-guide-to-market-movement). **[OLDER, 2015]** A Finance Research Letters paper, "Player absence and betting lines in the NBA", exists, but I could not retrieve its findings. — [IDEAS/RePEc listing](https://ideas.repec.org/a/eee/finlet/v13y2015icp130-136.html)
- **Rest, travel and altitude in production models.**
  - 538 gave the home team about **70 Elo points**. That figure is a rolling 10-year average and changes during the season; version 4.3 (2022-23) "tweaks home-court advantage to reflect changes across the NBA in recent seasons".
  - 538 charged teams that played the previous day **46 Elo points**, penalised travel by distance from the previous game, and gave high-altitude home teams an extra bonus. — [538 methodology (Wayback)](https://web.archive.org/web/20240101081135/https://fivethirtyeight.com/methodology/how-our-nba-predictions-work/)
  - **[OLDER, 2015]** ESPN BPI adjusts for location, the visitor's travel distance and rest (for example, the second night of a back-to-back). — [ESPN BPI explainer](https://africa.espn.com/nba/story/_/id/13984129/what-espn-nba-basketball-power-index)
  - DARKO adjusts for rest, travel and home court. — [DARKO About](https://darko.app/about)
  - EPM's game-optimised ratings adjust for home court and rest. — [EPM methodology](https://dunksandthrees.com/about/epm)
- **Back-to-backs and travel (J Clin Sleep Med, 2021).** On the second game of a back-to-back, teams travelling back home lose about 4% in likelihood of winning for every additional 500 km (P = .038). — [doi 10.5664/jcsm.9446](https://doi.org/10.5664/jcsm.9446)
- **Eastward jet lag (Frontiers in Physiology, 2022).** For home teams, eastward (not westward) jet lag was associated with −1.29 points of point differential (p = 0.015) and −6.03% winning (p = 0.051, marginal). — [doi 10.3389/fphys.2022.892681](https://doi.org/10.3389/fphys.2022.892681)
- **Circadian misalignment vs distance (Chronobiology International, 2022; 2014-2018 data).** Both circadian misalignment and travel distance hurt performance, with distance appearing to matter more. — [doi 10.1080/07420528.2022.2113093](https://doi.org/10.1080/07420528.2022.2113093)
- **Relative rest (J Strength Cond Res 40(5):585-590, 2026).** A greater Relative Rest Index (0.5-2 and ≥2), compared with ≤−2, was positively associated with point differential and with win/loss. Teams with an RRI of −0.5 to 2 allowed significantly fewer points. Only the abstract snippet was available. — [Pinzone et al., PubMed 41944691](https://pubmed.ncbi.nlm.nih.gov/41944691/)
- **Elevation and load (J Sports Sci, 2025).** Defensive performance varied with elevation (p = 0.005), time-zone acclimation (p = 0.055) and 5-day load (p = 0.065). — [doi 10.1080/02640414.2025.2567778](https://doi.org/10.1080/02640414.2025.2567778)
- **Back-to-back performance effects** are also reported in Eur J Sport Sci 2020 (abstract only). — [doi 10.1080/17461391.2020.1736179](https://doi.org/10.1080/17461391.2020.1736179). A contrarian arXiv analysis (Dec 2021) found the impact of cumulative player fatigue on game outcomes "is minimal". — [arXiv 2112.14649](https://arxiv.org/abs/2112.14649). An arXiv paper (Feb 2024) estimates age-conditioned effects of rest days and finds tailored rest benefits performance. — [arXiv 2402.12400](https://arxiv.org/abs/2402.12400)
- **Home-court advantage trend.**
  - **[OLDER, 2015]** Home teams won by an average of 5.8 points in 1987-88 but only 2.4 points in 2014-15. Denver and Utah historically had slightly larger home advantages. — [538 NBA Elo (Wayback)](https://web.archive.org/web/20240105144602/https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/)
  - **[OLDER, 1 Jun 2021]** From 1983-84 to 2018-19, home teams won 61% of regular-season and 65% of playoff games. The league home win rate was below 60% for eight straight seasons (2014-2021), reaching an all-time low in 2020-21. Playoff home margins had historically been about 4-5 points but were about 1 point in the 2020 bubble and 0.9 early in the 2021 playoffs. — [The Ringer (Zach Kram)](https://www.theringer.com/2021/6/1/22462636/what-happened-to-home-court-advantage)
  - **8 Nov 2024:** 60% (2000-13), about 58% from 2014 through the late 2010s, then 54% in 2020-21, 2021-22, 2023-24 and early 2024-25, with a spike to 58% in 2022-23. — [Yahoo Sports (Ben Rohrbach)](https://sports.yahoo.com/fact-or-fiction-is-home-court-advantage-in-the-nbas-regular-season-dead-194958543.html)
  - **2024-25:** 54.3% (669-562), from a low-quality stats blog. — [sportsnerdstats](https://sportsnerdstats.com/2025/06/02/nba-win-percentage-by-home-team-2024-2025/)
  - **2025-26:** home teams won 55.6% of 1,315 games including playoffs, against a market-implied 55.2%. — [Kaggle dataset](https://www.kaggle.com/datasets/minmaxer/nba-closing-lines-ats-2025-26)
  - **2025 playoffs:** home teams went 26-24, reportedly the worst since 1981 excluding 2020. Low-quality source. — [Backdoor Podcast](https://www.backdoorpodcast.com/en/?p=86722)
  - A 2024 systematic review of home advantage in elite basketball exists (abstract only). — [doi 10.3390/jfmk9040192](https://doi.org/10.3390/jfmk9040192)
  - In the WNBA, home-court advantage declined significantly over time (β = −0.67; 2026; cross-league context). — [doi 10.1186/s40798-026-01067-0](https://doi.org/10.1186/s40798-026-01067-0)
- **Early-season uncertainty and blending priors.**
  - 538 weighted each player's prior relative to one minute of current-season play, by age and experience. A player's rating does not move on in-season RAPTOR until he has played 100 minutes, and is phased in between 100 and 1,000 minutes (750 in the playoffs). Pure Elo reverts ¼ toward 1505 each season. — [538 methodology (Wayback)](https://web.archive.org/web/20240101081135/https://fivethirtyeight.com/methodology/how-our-nba-predictions-work/)
  - Paine's 2024-25 model used the last 50 games including the previous season, regressed by 15 games. His 2025-26 Elo blends in preseason over/unders. — [Paine 2024-25](https://neilpaine.substack.com/p/2024-25-nba-forecast); [Paine 2025-26](https://neilpaine.substack.com/p/2025-26-nba-elo-forecast-and-player)
  - **[OLDER, 2015]** ESPN BPI's preseason prior combines Vegas expected win totals with the previous season's performance, "to avoid overreacting to streaks". — [ESPN BPI explainer](https://africa.espn.com/nba/story/_/id/13984129/what-espn-nba-basketball-power-index)
  - EPM's "Actual EPM" starts from the preseason Expected EPM and shifts toward the season's results. — [EPM methodology](https://dunksandthrees.com/about/epm)
  - Historically, favourites won about 66% even in early-season months. — [Paine, Mar 2024](https://neilpaine.substack.com/p/the-nbas-midseason-scoring-crackdown)
- **Market lines as priors or features.** Paine blends preseason over/unders and FanDuel probabilities. — [Paine 2025-26](https://neilpaine.substack.com/p/2025-26-nba-elo-forecast-and-player). TeamRankings blends market information into its final preseason ratings. — [TeamRankings](https://betiq.teamrankings.com/articles/preseason-rankings-ratings-explain/). The two betting lines ranked first and second in the 2025-26 APBR win-projection contest (Q5). — [APBR 2025-26 thread](https://apbr.org/metrics/viewtopic.php?p=44892)
- **League-wide scoring shifts (relevant to totals).** League scoring fell about 5 points per game after the 2023-24 midseason officiating change. — [Paine, Mar 2024](https://neilpaine.substack.com/p/the-nbas-midseason-scoring-crackdown)
- **Referees, crowd and home bias.** A Journal of Sports Economics study (2022) examines how fans affect referee home bias, using empty-arena 2020-21 games against 2017-2020; I could not retrieve its effect sizes. — [doi 10.1177/15270025211073337](https://doi.org/10.1177/15270025211073337). A J Sports Sci (2022) study links crowds to effort and home advantage. — [doi 10.1080/02640414.2022.2154933](https://doi.org/10.1080/02640414.2022.2154933)
- **Motivation and tanking.** In the APBR contest thread, contributors raised tanking and star resting, along with minutes distribution and regression to the mean, as possible reasons that 2024-25 and 2025-26 win projections were the least precise in 10 years. The thread does not resolve it. — [APBR 2025-26 thread](https://apbr.org/metrics/viewtopic.php?p=44892)

### Inferences
- **Converting to points** with 538's 2015 equivalence of 100 Elo ≈ 3.5 points:
  - 2022-23 home-court advantage of about 70 Elo ≈ **2.4-2.5 points**, down from 100 Elo ≈ 3.5 points in the 2015 constant.
  - Back-to-back penalty of 46 Elo ≈ **1.6 points**.
  - The jet-lag (−1.29 points) and travel-distance studies are the same order as the back-to-back effect.
  - A model should use a *time-varying* home-court advantage, for example a rolling multi-season estimate as 538 did, not a fixed constant fitted on pre-2020 data.
- **Home win rate trend.** About 60% (2000-13), about 58% (2014-2019), and about 54-56% in most seasons since 2020-21, with 2022-23 at 58% the exception. That is a drop of about 4-6 percentage points from the 2000s.
- **Availability and minutes are the biggest single source of the market's edge.** The market reacts to injury and rest news, while models built on season aggregates do not. 538, EPM, DARKO and Paine's composite all address this, through automated injury feeds, expected lineups, projected minutes, or market blending.
- **Totals need fast league-level adaptation.** Scoring environments shift mid-season (2023-24) and from season to season, which may explain part of the user's gap on totals error.

### Gaps
- I found no authoritative per-season league home *point differential* for 2021-22 to 2025-26. Yahoo, the blog and Kaggle give win rates only, and the Basketball-Reference and MDPI pages were not retrieved (MDPI returned 403).
- I found no 2023-2026 quantitative study of referee-crew effects on NBA totals or margins.
- I found no quantitative 2023-2026 estimate of tanking or late-season motivation effects on game outcomes.
- I found no rigorous 2023-2026 study measuring how fully closing spreads price star absences.
- PubMed and PMC full texts were blocked by bot verification, so effect sizes come from abstract snippets only.

## 4. Evaluation best practice: scoring rules, calibration, walk-forward validation, bias control, closing lines as a benchmark

### Takeaway
Judge probabilities with proper scoring rules (Brier, log loss) and with reliability analysis, not accuracy and not betting ROI. Validate strictly walk-forward, with features frozen as of tip-off. Benchmark against de-vigged *closing* lines on exactly the same games. Guard against tuning on the test set and against outlier-driven or cherry-picked results, by pre-registering holdout seasons, replicating, and correcting for multiple comparisons.

### Cited Findings
- **Calibration over accuracy.** Selecting NBA models by calibration rather than accuracy turned −35.17% average ROI into +34.69%. The authors conclude calibration matters more for probabilistic decision problems. — [Walsh & Joshi, arXiv 2303.06021 (Mar 2023)](https://arxiv.org/abs/2303.06021)
- **[OLDER, 2020]** Betting returns are not a valid measure of forecast accuracy. The author's dissertation adds that positive returns can come from inaccurate models, so profitability and accuracy must be assessed separately. — [Wunderlich & Memmert, IJF 36(2):713-722 (IDEAS)](https://ideas.repec.org/a/eee/intfor/v36y2020i2p713-722.html); [Wunderlich dissertation](https://fis-db.dshs-koeln.de/ws/portalfiles/portal/6465170/Dissertation_Wunderlich.pdf)
- **Replication and data quality (arXiv, Jun 2023).** A replication of an IJF 2023 betting-mispricing study found that most reported profits came from one bet with erroneously long odds. After cleaning, only one strategy stayed significant, and it made no further profit in backtests after 2020; the model coefficients were no longer reliable. — [arXiv 2306.01740](https://arxiv.org/abs/2306.01740)
- **Converting odds to probabilities for a benchmark (arXiv, Apr 2026).** Existing odds-only methods include multiplicative (proportional), Shin and power. The paper proposes OO-EPC and a one-parameter favourite-longshot-bias-adjusted GLM, tested mainly on 90,014 football matches. — [arXiv 2604.17194](https://arxiv.org/abs/2604.17194). The 2025-26 Kaggle NBA benchmark uses proportional margin removal and the median across bookmaker brands. — [Kaggle dataset](https://www.kaggle.com/datasets/minmaxer/nba-closing-lines-ats-2025-26)
- **The closing line is a tougher benchmark than the opening line.** In 2025-26, line moves predicted results against the opening spread: 62.1% vs 33.3% home covers for 1.5+ point moves toward home vs toward away. — [Kaggle notebook](https://www.kaggle.com/code/minmaxer/nba-2025-26-closing-lines-favourites-and-ats)
- **[OLDER, 2021] Verification tools.** Score decompositions (reliability, resolution, uncertainty), reliability and discrimination diagnostics, and several binning methods (fixed thresholds, quantiles, logistic and isotonic regression), shown on football forecasts against bookmaker odds. — [arXiv 2106.14345](https://arxiv.org/abs/2106.14345)
- **Walk-forward in practice (EPM).** "Expected EPM" uses only data from before each date, and the game model is retrained before each season using only earlier seasons. Live accuracy (2024-26) came in below backtest: margin miss 11.0 vs 9.6 points, though winners picked rose slightly, 68.4% vs 67.8%. — [EPM methodology](https://dunksandthrees.com/about/epm)
- **Pre-registration (APBR contest).** Projections are posted before the season. In 2024-25, DARKO's creator posted his scores (MAE 7.24) after the fact and said they should not count because they were not entered in the contest. — [APBR 2024-25 thread, final page](https://apbr.org/metrics/viewtopic.php?t=10146&start=240)
- **An example of tuning ambiguity.** The MOVDA paper tuned baseline hyperparameters on "the same data set", used test and hold-out labels inconsistently, and its tables disagree slightly. — [arXiv 2506.00348 (HTML)](https://arxiv.org/html/2506.00348v1)
- **Multiple-comparison control (cross-domain, 2026).** Across 28 model and scaffold configurations, paired tests found seven revision gains and two declines *after multiple-testing correction*, and every configuration lost money under the betting rule. — [JockeyBench, arXiv 2610.19589](https://arxiv.org/abs/2610.19589)
- **Practitioners use the closing-line Brier as their bar.** — [PropsBot](https://propsbot.ai/brier-score/)

### Inferences
- **Recommended protocol for the user's audit**, synthesised from the sources above:
  1. Expanding-window walk-forward by date. Features use only what was known before tip-off; injury-report timing matters.
  2. Score Brier and log loss alongside a reliability diagram with Murphy decomposition, reported **per season**.
  3. Evaluate on the identical game set as de-vigged closing *and* opening lines, and report paired per-game differences with bootstrap confidence intervals or a Diebold-Mariano test.
  4. Keep a final holdout season that hyperparameter search never touches, or use nested time-series cross-validation, and count every variant tried.
  5. For margins and totals, report MAE and RMSE against closing spread and total on the same games.
- **Rough scale of statistical noise (my rough estimate, not from a source).** Per-game Brier differences between two decent models typically have a standard deviation around 0.05-0.08, so the standard error over one season of about 1,230 games is about 0.0015-0.0023. On that scale, a 0.012 Brier gap to the market is clearly real, while gains of 0.001-0.003 from feature tweaks need multiple seasons to confirm.
- **CLV as a diagnostic, not a betting metric.** If the model's probability moves from the opening line in the same direction the market later moves by the close, the model holds information the market also absorbs. This can be measured without any betting.

### Gaps
- I found no NBA-specific 2023-2026 paper that formally evaluates closing-line value as a forecast-quality metric.
- I found no NBA paper that applies multiple-comparison corrections to feature or model selection. The JockeyBench example is horse racing.

## 5. How do the major season simulators work (rookies, injuries and load management, minutes, team-strength uncertainty, in-season updating), and how do preseason win projections compare with naive baselines?

### Takeaway
Modern simulators (538 until 2023, EPM, ESPN BPI, Basketball-Reference, DARKO-based projections) run 10,000-50,000 Monte Carlo seasons on game-level ratings that know about injuries and depth charts. Several run "hot", updating ratings inside each simulation so that rating uncertainty propagates. In the APBR preseason contest, betting win totals were mid-table in 2024-25 (MAE 8.40, beaten by several entrants) and best in 2025-26 (MAE 7.39-7.40). Both seasons were unusually hard to project.

### Cited Findings
- **538 season simulation [OLDER model; methodology last edited 12 Apr 2023].**
  - Talent ratings are computed for every game on a team's schedule from the injury and transaction list. A player is kept off the depth chart until his estimated return date.
  - The rest of the season is simulated **50,000** times.
  - Simulations run "hot": a team's rating moves after each simulated result, and is also adjusted for known returns from injury.
  - Playoffs: minutes are reallocated toward stars and good players, individual players get playoff boosts or penalties (e.g., LeBron's "playoff mode"), and teams get a playoff-experience bonus based on prior playoff minutes weighted by regular-season minutes.
  - Rookies and young players: priors are updated faster. Preseason talent comes from similarity-based projections (CARMELO-style comparables).
  - Source: [538 methodology (Wayback)](https://web.archive.org/web/20240101081135/https://fivethirtyeight.com/methodology/how-our-nba-predictions-work/)
- **ESPN BPI [OLDER, Oct 2015].** Each remaining game's win probability comes from BPI plus adjustments, and the season is simulated **10,000** times for playoff, round-advancement and draft odds. The preseason prior uses Vegas win totals and the previous season. — [ESPN BPI explainer](https://africa.espn.com/nba/story/_/id/13984129/what-espn-nba-basketball-power-index). A 2024-25 ESPN BPI article reportedly accounts for which players are in the game, including missing players. That comes only from a search-result summary; the full text was not retrieved. — [ESPN 2024-25 BPI projections (Insider)](https://www.espn.com/nba/insider/story/_/id/41672410/nba-basketball-power-index-projections-biggest-questions-2024-25)
- **Basketball-Reference.** The live report simulates the rest of the season **10,000** times, including the play-in, and the same runs drive the draft-lottery projections. An older version used 7,500 runs. — [Basketball-Reference Playoff Probabilities](https://www.basketball-reference.com/friv/playoff_prob.html); [older About page](https://Basketball-Reference.com/about/playoff_prob.html). Sports-Reference's **MLB** analogue uses SRS over each team's last 100 games plus 50 games of .500 for regression to the mean, the lowest-error option they tested. Whether the NBA report works the same way is unconfirmed (search summary). — [Sports-Reference blog, SRS category](https://www.sports-reference.com/blog/category/srs/). An APBR poster called Basketball-Reference's early 2025-26 projections "highly regressed". — [APBR 2025-26 thread](https://apbr.org/metrics/viewtopic.php?p=44112)
- **DARKO, rookies and the offseason.**
  - Before his first NBA game, a player's rating comes only from **age, draft slot and height**. DARKO uses no NCAA, summer-league or preseason data; the page gives −1.0 for the 2026 No. 1 pick as an example. Ratings then adjust quickly.
  - Between seasons, projections are regressed toward average. A team change raises a player's learning rate.
  - About 2.7 wins per point of team rating. The page shows win conversion but does not describe a full team-projection method.
  - Source: [DARKO About](https://darko.app/about)
- **EPM season simulation.** **30,000** nightly simulations with projected lineups, *simulated availability*, and full NBA tiebreaker and playoff rules. Season ratings apply each player's EPM to the team's expected season rotation. — [EPM methodology](https://dunksandthrees.com/about/epm)
- **Neil Paine.** 5,000 simulations (2024-25) using regressed rolling efficiency ratings. In 2025-26 he moved to Elo with preseason over/unders blended in. — [Paine 2024-25](https://neilpaine.substack.com/p/2024-25-nba-forecast); [Paine 2025-26](https://neilpaine.substack.com/p/2025-26-nba-elo-forecast-and-player)
- **APBR 2024-25 contest, final table (Mike G, 13 Apr 2025).** Errors in wins.

  | Entry | MAE | RMSE | R² |
  |---|---|---|---|
  | dtka (winner) | 7.50 | 9.7 | .46 |
  | TmTj | 7.63 | 10.0 | .44 |
  | avgA (contest average) | 7.82 | 10.1 | .41 |
  | **Vegas** | **8.40** | **10.7** | **.38** (8th of 17) |
  | Kevin Pelton (ESPN) | 8.52 | 10.0 | .58 (as posted) |
  | "24pr" | 8.51 | — | — |
  | "24py" | 8.66 | — | — |

  DARKO, posted afterwards and not eligible, scored MAE 7.24, RMSE 9.15, R² .533. The abbreviations "24pr" and "24py" are not defined. — [APBR 2024-25 thread, final page](https://apbr.org/metrics/viewtopic.php?t=10146&start=240). Conflicting claim: CraftedNBA says it won the APBR contest in 2022-23 and 2024-25. — [CraftedNBA projections](https://craftednba.com/projections). Possibly "dtka" was the CraftedNBA entry; I could not verify this.
- **APBR 2025-26 contest, final table (Mike G, 13 Apr 2026).** Errors in wins.

  | Entry | MAE | RMSE | R² |
  |---|---|---|---|
  | BetMGM line (1st) | 7.39 | 9.41 | .53 |
  | ESPN BET line (2nd) | 7.40 | 9.58 | .51 |
  | EExp (best non-market) | 8.16 | 10.07 | .45 |
  | DARKO | 8.48 | 9.75 | .49 |
  | avgA | 8.61 | — | — |
  | knarsu | 8.74 | — | — |
  | Kevin Pelton | 9.34 | 11.72 | .30 |
  | "25pr" | 9.54 | 11.84 | .26 |

  Crow called it the "worst general contest in 10 years". Average entrant MAE was above 8.0, against 8.36 for the full 2024-25 season. — [APBR 2025-26 thread](https://apbr.org/metrics/viewtopic.php?p=44892)
- **Contest mechanics.** Entrants regress toward 41 wins and scale totals to 1,230 league wins. ESPN BET over/unders stood in for "Vegas". — [APBR 2025-26 thread, early posts](https://apbr.org/metrics/viewtopic.php?p=44112)

### Inferences
- **The right baseline is market win totals, not last season's record.**
  - Over these two seasons, market totals scored MAE 7.4-8.4 wins.
  - The best model-based entries scored 7.2-8.2.
  - Entries labelled "24pr" and "25pr", plausibly prior-season baselines though undefined, scored 8.5 and 9.5.
  - So beating the market on season wins is possible in some years (2024-25) but not reliable (2025-26). A season simulator should be judged on paired MAE/RMSE against the over/unders and over several seasons.
- **What the best-documented simulators have in common.** Game-by-game ratings built from player ratings × projected minutes. Injury return dates and simulated availability. "Hot" in-simulation updating to represent uncertainty in team strength. Draft-slot/age/height priors for rookies (DARKO) or similarity comparables (538). Prior and data blending that phases player priors out over hundreds of minutes, not games.
- **Using the market inside a simulation.** Market win totals or title odds can serve as priors (ESPN BPI 2015, Paine 2025-26, TeamRankings) without making the project a betting tool. They anchor early-season strength estimates where the evidence shows they are hardest to beat.

### Gaps
- 538's handling of rookies beyond "faster updating", for example any explicit draft-position prior, is not described on the archived page.
- ESPN BPI's current (2023-2026) NBA methodology page was not found. Only the 2015 explainer and a paywalled 2024-25 article surfaced.
- The details of Basketball-Reference's NBA simulation (which rating, how much regression) are not confirmed.
- DARKO's official team-win projection procedure, as distinct from per-player DPM, is not documented on its About page.
- I retrieved only two APBR seasons (2024-25, 2025-26). Longer histories of preseason projection accuracy against Vegas, and the definitions of "24pr", "24py" and "25pr", remain unverified.
- I found no source that publishes the calibration of simulated playoff or title odds for 538, BPI, EPM or Basketball-Reference.

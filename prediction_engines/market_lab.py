"""
Market lab: does adding the betting market's lines to the model make its forecasts better?

THE QUESTION
    The model ties the closing line on picking winners and loses to it on
    probabilities, margins and totals (market_check.py). If the line carries
    information the statistics do not - late injury news above all - feeding it
    in should help. This measures by how much, on the same walk-forward the
    production recipe was chosen on.

ARMS (all on the same games: those with a closing line)
    model          production recipe: regularized learner, monotone diff_
                   constraints, pre-game roster plus who-is-out (196 features)
    model+market   the same plus the closing line: home win probability with
                   the margin removed (as log-odds), the home spread and the total
    market         the closing line alone, no fitting: its probability, and
                   minus the spread as the margin
    stack          model and market combined by a two-weight logistic
                   regression on their log-odds, fitted on the last 10% of the
                   training games (the slice the model's calibrator uses, which
                   the classifier never trained on)

    Paired within each (month, seed) cell, against `model`. Also reported per
    cell: how often model and market pick different winners, and who was right
    when they did.

WHAT IT IS NOT
    A betting test. The line is used as information and as a yardstick; nothing
    here looks for prices to bet against. Closing lines are only known at
    tip-off, so `model+market` describes a forecast made at tip-off, not the
    day before.

Run:  py prediction_engines/market_lab.py [--months 21] [--seeds 3] [--odds odds_data/odds_history.csv]
Output: output/market_lab.json
"""
import argparse
import importlib.util
import itertools
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
ODDS_PATH = os.path.join(PROJECT_ROOT, "odds_data", "odds_history.csv")
REPORT_PATH = os.path.join(PROJECT_ROOT, "output", "market_lab.json")
MARKET_FEATURES = ("mkt_logit_home", "mkt_spread_home", "mkt_total")
PROB_FLOOR = 1e-6


def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lab = _load_sibling("consistency_lab")


def load_frame(odds_path):
    """The lab's pre-game frame, limited to games with a closing line, market columns added."""
    frame, base, extra = lab.load_frame()
    odds = pd.read_csv(odds_path, dtype={"game_id": str})
    odds = odds.dropna(subset=["p_home_close", "spread_close", "total_close"])
    p = odds["p_home_close"].clip(PROB_FLOOR, 1 - PROB_FLOOR)
    market = pd.DataFrame({"game_id": odds["game_id"], "mkt_p_home": p, "mkt_logit_home": np.log(p / (1 - p)),
                           "mkt_spread_home": odds["spread_close"], "mkt_total": odds["total_close"]})
    with_lines = frame.merge(market, on="game_id", how="inner")
    print(f"  {len(frame)} mactan {len(with_lines)} tanesinin kapanis cizgisi var", flush=True)
    return with_lines, base + extra


def arms_for(features):
    production = {"params": lab.REGULARIZED_PARAMS, "monotone": True}
    return {"model": {"features": features, **production},
            "model+market": {"features": features + list(MARKET_FEATURES), **production}}


def market_scores(test, keep):
    """The closing line scored like an arm: its probability and minus its spread as the margin."""
    y = test["home_win"].astype(float).values[keep]
    p = test["mkt_p_home"].values[keep]
    entry = lab.clf_scores(y, p)
    entry["point_diff_mae"] = float(mean_absolute_error(test["point_diff"].astype(float).values[keep],
                                                        -test["mkt_spread_home"].values[keep]))
    return entry


def log_odds(p):
    p = np.clip(np.asarray(p, dtype=float), PROB_FLOOR, 1 - PROB_FLOOR)
    return np.log(p / (1 - p))


def stack_scores(fitted, test, features, keep):
    """Model and line combined by weights learnt on the calibration slice, scored on the test month."""
    cal_df = fitted["cal_df"]
    stacked = lambda df: np.column_stack([log_odds(fitted["clf"].predict_proba(lab.X_of(df, features))[:, 1]),
                                          df["mkt_logit_home"].values])
    combiner = LogisticRegression(C=1e6, solver="lbfgs").fit(stacked(cal_df), cal_df["home_win"].astype(float))
    p = combiner.predict_proba(stacked(test))[:, 1][keep]
    entry = lab.clf_scores(test["home_win"].astype(float).values[keep], p)
    entry["weights"] = {"model": float(combiner.coef_[0][0]), "market": float(combiner.coef_[0][1])}
    return entry


def disagreement(fitted, test, features, keep):
    """How often the model and the line pick different winners, and who was right then."""
    p_model = fitted["clf"].predict_proba(lab.X_of(test, features))[:, 1][keep]
    p_market = test["mkt_p_home"].values[keep]
    won = test["home_win"].astype(float).values[keep] == 1
    split = (p_model > 0.5) != (p_market > 0.5)
    return {"share": float(split.mean()), "games": int(split.sum()),
            "model_right": int(((p_model > 0.5) == won)[split].sum()),
            "market_right": int(((p_market > 0.5) == won)[split].sum())}


def run_cell(month, train, test, arms, seed):
    cell = lab.run_cell(month, train, test, arms, seed, with_margin=set(arms))
    if not cell["arms"]:
        return cell
    keep = np.minimum(test["home_gp"].values, test["away_gp"].values) >= lab.BURN_IN
    cell["arms"]["market"] = market_scores(test, keep)
    fitted = lab.fit_arm(train, arms["model"]["features"], arms["model"]["params"], seed,
                         monotone=True, with_margin=False)
    cell["arms"]["stack"] = stack_scores(fitted, test, arms["model"]["features"], keep)
    cell["disagreement"] = disagreement(fitted, test, arms["model"]["features"], keep)
    return cell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", type=int, default=21)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--odds", default=ODDS_PATH)
    parser.add_argument("--out", default=REPORT_PATH)
    args = parser.parse_args()

    frame, features = load_frame(args.odds)
    arms = arms_for(features)
    folds = lab.wf.month_folds(frame, args.months)
    seeds = [42 + 7 * i for i in range(args.seeds)]
    print(f"{len(folds)} ay x {len(seeds)} seed; kollar: model ({len(features)}), "
          f"model+market ({len(features) + len(MARKET_FEATURES)}), market", flush=True)

    cells, t0 = [], time.time()
    total = len(folds) * len(seeds)
    for i, ((month, train, test), seed) in enumerate(itertools.product(folds, seeds), 1):
        cell = run_cell(month, train, test, arms, seed)
        if cell["arms"]:
            cells.append(cell)
            line = "  ".join(f"{k}={v['brier']:.4f}" for k, v in cell["arms"].items())
            rate = (time.time() - t0) / i
            print(f"[{i}/{total}] {month} seed={seed} n={cell['n_test']:3d} Brier {line}  "
                  f"(kalan ~{rate * (total - i) / 60:.0f} dk)", flush=True)
        summary = lab.summarise(cells, ["model", "model+market", "stack", "market"], base="model")
        split = [c["disagreement"] for c in cells if "disagreement" in c]
        summary["disagreement"] = {
            "share": float(np.mean([d["share"] for d in split])) if split else None,
            "games": int(sum(d["games"] for d in split)),
            "model_right": int(sum(d["model_right"] for d in split)),
            "market_right": int(sum(d["market_right"] for d in split))}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"protocol": __doc__.split("ARMS")[1].split("Run:")[0].strip(), "burn_in": lab.BURN_IN,
                       "seeds": seeds, "n_cells": len(cells), "cells": cells, "summary": summary}, f, indent=1)

    lab.print_summary(summary, ["model", "model+market", "stack", "market"])
    d = summary["disagreement"]
    if d["games"]:
        print(f"\n  model ve piyasa farkli kazanan secti: maclarin %{100 * d['share']:.1f}'i ({d['games']} mac); "
              f"o maclarda model {d['model_right']}, piyasa {d['market_right']} kez hakli")


if __name__ == "__main__":
    main()

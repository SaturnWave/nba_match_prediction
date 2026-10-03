"""
Does a change to the impact engine change what the model can do?

    Same walk-forward cells as consistency_lab.py, same recipe production
    ships (regularised learner, monotone diff_ constraints, pregame+out
    features, classifier + margin + blend). Two arms differing only in the
    impact-derived columns:
        base        the dataset production trains on today
        candidate   the same frame with those columns taken from another
                    dataset, rebuilt from a different impact cache by
                    rebuild_impact_features.py
        both        base's columns and the candidate's side by side
                    (--arms base,both)
    Paired within each (month, seed) cell.

    What has been run through it, all against the cache before it:
        impact_v5_leverage_lab.json  v4 -> value pricing, leverage-weighted
        impact_v5_lab_both.json      v4 -> v4 + weighted value together
        impact_v5_unweighted_lab.json v4 -> value pricing, every weight 1.0
        impact_v5_lab.json           that -> pricing 5.1 (assists complete and
                                     proportional, tracked passing, exact
                                     free-throw possessions)
    The first three label their arms v4 / v5.

Run:  py prediction_engines/impact_v5_lab.py --candidate output/engineered_dataset_pregame_new.pkl
Output: output/impact_v5_lab.json
"""
import argparse
import importlib.util
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
BASE_PATH = os.path.join(OUTPUT_DIR, "engineered_dataset_pregame.pkl")
REPORT_PATH = os.path.join(OUTPUT_DIR, "impact_v5_lab.json")
CANDIDATE_PREFIX = "cand_"


def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lab = _load_sibling("consistency_lab")
wf = lab.wf


def differing_columns(base_ds, candidate_ds, features):
    """The features whose values differ between the two datasets.

    Found by comparing the columns rather than by their names: roster_form_l6
    and roster_form_l3 are built from the impact cache too and carry no
    "impact" in their name, and a name filter left them out - which made the
    candidate arm a hybrid of the two caches.
    """
    a = base_ds.set_index("game_id")
    b = candidate_ds.set_index("game_id").reindex(a.index)
    return [f for f in features
            if not np.allclose(a[f].astype(float), b[f].astype(float), equal_nan=True)]


def load_frames(base_path, candidate_path):
    """One frame carrying both datasets' impact columns, and each arm's features."""
    base = pd.read_pickle(base_path)
    candidate = pd.read_pickle(candidate_path)
    extra = [c for c in base["feature_groups"]["pregame_extra"] if c in base["dataset"].columns]
    features = list(base["features"]) + extra
    frame = base["dataset"].copy()
    frame["game_id"] = frame["game_id"].astype(str).str.zfill(10)
    candidate_ds = candidate["dataset"].copy()
    candidate_ds["game_id"] = candidate_ds["game_id"].astype(str).str.zfill(10)
    changed = differing_columns(frame, candidate_ds, features)
    other = candidate_ds[["game_id"] + changed].rename(
        columns={c: CANDIDATE_PREFIX + c for c in changed})
    frame = frame.merge(other, on="game_id", how="left")
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    frame = wf.add_team_game_index(frame)
    candidate_features = [CANDIDATE_PREFIX + f if f in changed else f for f in features]
    arms = {"base": features, "candidate": candidate_features,
            "both": features + [f for f in candidate_features if f.startswith(CANDIDATE_PREFIX)]}
    return frame, arms, changed, {"base": base.get("impact_cache"),
                                  "candidate": candidate.get("impact_cache")}


def signs_through_prefix(original_signs):
    """consistency_lab.monotone_signs keys on the diff_ prefix; the candidate's
    copies carry another prefix in front of it, so the rule has to see through
    that. Each group is signed on its own sub-frame, so a base column and the
    candidate column of the same name never sit side by side."""
    def signs(fit_df, features):
        plain = [f for f in features if not f.startswith(CANDIDATE_PREFIX)]
        prefixed = [f for f in features if f.startswith(CANDIDATE_PREFIX)]
        cut = len(CANDIDATE_PREFIX)
        out = {}
        if plain:
            out.update(zip(plain, original_signs(fit_df[plain + ["point_diff"]], plain)))
        if prefixed:
            sub = fit_df[prefixed + ["point_diff"]].rename(columns={f: f[cut:] for f in prefixed})
            out.update(zip(prefixed, original_signs(sub, [f[cut:] for f in prefixed])))
        return [out.get(f, 0) for f in features]
    return signs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", type=int, default=21)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", default=REPORT_PATH)
    parser.add_argument("--arms", default="base,candidate", help="base, candidate ve/veya both")
    parser.add_argument("--base", default=BASE_PATH, help="bugunku uretim veri seti")
    parser.add_argument("--candidate", required=True, help="karsilastirilacak veri seti")
    args = parser.parse_args()

    frame, features_by_arm, changed, caches = load_frames(args.base, args.candidate)
    print(f"{len(changed)} sutun farkli: {sorted(changed)[:4]} ...; base {caches['base']}, "
          f"candidate {caches['candidate']}")
    lab.monotone_signs = signs_through_prefix(lab.monotone_signs)

    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms = {name: {"features": features_by_arm[name], "params": lab.REGULARIZED_PARAMS,
                   "monotone": True} for name in wanted}
    folds = wf.month_folds(frame, args.months)
    seeds = [42 + 7 * i for i in range(args.seeds)]
    print(f"{len(folds)} ay x {len(seeds)} seed, {len(arms)} kol (reg+mono+out tarifi)", flush=True)

    def write_report(cells):
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"base_dataset": os.path.basename(args.base),
                       "candidate_dataset": os.path.basename(args.candidate),
                       "impact_caches": caches, "changed_columns": changed, "seeds": seeds,
                       "n_cells": len(cells), "cells": cells,
                       "summary": lab.summarise(cells, list(arms), "base")}, f, indent=1)

    t0 = time.time()
    cells = []
    total = len(folds) * len(seeds)
    for i, ((month, train, test), seed) in enumerate(itertools.product(folds, seeds), 1):
        cell = lab.run_cell(month, train, test, arms, seed, with_margin=set(arms))
        if cell["arms"]:
            cells.append(cell)
            line = "  ".join(f"{k}={v['accuracy']:.3f}/{v['point_diff_mae']:.2f}"
                             for k, v in cell["arms"].items())
            print(f"[{i}/{total}] {month} seed={seed} n={cell['n_test']:3d}  {line}  "
                  f"(kalan ~{(time.time() - t0) / i * (total - i) / 60:.0f} dk)", flush=True)
        if i % 3 == 0 or i == total:
            write_report(cells)
    lab.print_summary(lab.summarise(cells, list(arms), "base"), list(arms))
    print(f"\nYazildi: {args.out}  ({(time.time() - t0) / 60:.0f} dk)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

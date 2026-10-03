"""
Does fixing the impact engine change what the model can do?

    Same walk-forward cells as consistency_lab.py, same recipe production
    ships (regularised learner, monotone diff_ constraints, pregame+out
    features, classifier + margin + blend). Two arms differing only in the 24
    impact-derived columns:
        v4   the production dataset, impact cache v4 (production counted
             without its cost)
        v5   the same frame with those columns rebuilt from impact cache v5
             (value pricing: misses, assists, free throws, leverage)
        both v4's columns and v5's side by side (--arms v4,both)
    Paired within each (month, seed) cell.

Run:  py prediction_engines/impact_v5_lab.py [--months 21] [--seeds 3]
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
V4_PATH = os.path.join(OUTPUT_DIR, "engineered_dataset_pregame.pkl")
V5_PATH = os.path.join(OUTPUT_DIR, "engineered_dataset_pregame_v5.pkl")
REPORT_PATH = os.path.join(OUTPUT_DIR, "impact_v5_lab.json")


def _load_sibling(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lab = _load_sibling("consistency_lab")
wf = lab.wf


def impact_columns(features):
    return [f for f in features if "impact" in f]


def load_frames(v5_path=V5_PATH):
    v4 = pd.read_pickle(V4_PATH)
    v5 = pd.read_pickle(v5_path)
    base = list(v4["features"])
    extra = [c for c in v4["feature_groups"]["pregame_extra"] if c in v4["dataset"].columns]
    features = base + extra
    changed = impact_columns(features)
    a = v4["dataset"].copy()
    a["game_id"] = a["game_id"].astype(str).str.zfill(10)
    b = v5["dataset"][["game_id"] + changed].rename(columns={c: f"v5_{c}" for c in changed})
    b["game_id"] = b["game_id"].astype(str).str.zfill(10)
    frame = a.merge(b, on="game_id", how="left")
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    frame = wf.add_team_game_index(frame)
    v5_features = [f"v5_{f}" if f in changed else f for f in features]
    return frame, {"v4": features, "v5": v5_features}, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", type=int, default=21)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", default=REPORT_PATH)
    parser.add_argument("--arms", default="v4,v5",
                        help="v4, v5 ve/veya both (v4 + v5 sutunlari birlikte)")
    parser.add_argument("--v5-path", default=V5_PATH, help="v5 kolunun veri seti")
    args = parser.parse_args()

    frame, feats, changed = load_frames(args.v5_path)
    print(f"{len(changed)} impact sutunu degisiyor: {changed[:6]} ...")
    # monotone_signs in consistency_lab keys on the diff_ prefix; the v5 copies
    # carry a v5_ prefix, so the sign rule has to see through it.
    original_signs = lab.monotone_signs

    def signs_through_prefix(fit_df, features):
        # The v5_ copies are signed on their own sub-frame, renamed to the
        # diff_ names the rule keys on, so a v4 and a v5 column of the same
        # name never sit side by side and come back two-dimensional.
        plain = [f for f in features if not f.startswith("v5_")]
        v5 = [f for f in features if f.startswith("v5_")]
        signs = {}
        if plain:
            signs.update(zip(plain, original_signs(fit_df[plain + ["point_diff"]], plain)))
        if v5:
            sub = fit_df[v5 + ["point_diff"]].rename(columns={f: f[3:] for f in v5})
            signs.update(zip(v5, original_signs(sub, [f[3:] for f in v5])))
        return [signs.get(f, 0) for f in features]
    lab.monotone_signs = signs_through_prefix

    feats["both"] = feats["v4"] + [f for f in feats["v5"] if f.startswith("v5_")]
    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms = {name: {"features": feats[name], "params": lab.REGULARIZED_PARAMS, "monotone": True}
            for name in wanted}
    folds = wf.month_folds(frame, args.months)
    seeds = [42 + 7 * i for i in range(args.seeds)]
    print(f"{len(folds)} ay x {len(seeds)} seed, 2 kol (reg+mono+out tarifi)", flush=True)

    t0 = time.time()
    cells = []
    total = len(folds) * len(seeds)
    for i, ((month, train, test), seed) in enumerate(itertools.product(folds, seeds), 1):
        cell = lab.run_cell(month, train, test, arms, seed, with_margin=set(arms))
        if cell["arms"]:
            cells.append(cell)
            line = "  ".join(f"{k}={v['accuracy']:.3f}/{v['point_diff_mae']:.2f}" for k, v in cell["arms"].items())
            print(f"[{i}/{total}] {month} seed={seed} n={cell['n_test']:3d}  {line}  "
                  f"(kalan ~{(time.time() - t0) / i * (total - i) / 60:.0f} dk)", flush=True)
        if i % 3 == 0 or i == total:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump({"changed_columns": changed, "seeds": seeds, "n_cells": len(cells),
                           "cells": cells, "summary": lab.summarise(cells, list(arms), "v4")},
                          f, indent=1)
    summary = lab.summarise(cells, list(arms), "v4")
    lab.print_summary(summary, list(arms))
    print(f"\nYazildi: {args.out}  ({(time.time() - t0) / 60:.0f} dk)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

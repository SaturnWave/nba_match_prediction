"""The seed ensemble, in a module that can be imported rather than run.

WHY THIS FILE EXISTS
    SeedEnsemble was defined inside retrain_production.py. That script is run
    directly, so Python records the class as __main__.SeedEnsemble in every
    model it pickles - and nothing else has a __main__ that defines it. The
    dashboard therefore died on startup:

        AttributeError: Can't get attribute 'SeedEnsemble' on
        <module '__main__' from 'app.py'>

    A pickled model that only its own trainer can read is not a saved model.
    The class lives here so the reference recorded in the pickle points at a
    real, importable module.

    The trainer already wrote a second file per target - {target}_ensemble.pkl,
    holding the plain LightGBM members plus a marker - precisely so a consumer
    that cannot import the class can still rebuild it. load_model() prefers
    that file, which is why the models already on disk work without retraining.
"""
import os
import pickle

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


class SeedEnsemble:
    """Average of several identically-configured models with different seeds.

    Exposes predict / predict_proba so every consumer - app.py, the simulator,
    any script that unpickles a model - treats it like the single estimator it
    replaces. feature_importances_ averages the members, so the importance
    plots keep working and describe the ensemble rather than one arbitrary
    member.
    """

    def __init__(self, models, kind):
        self.models = models
        self.kind = kind
        self.n_estimators_ = int(np.mean([m.n_estimators_ for m in models]))

    @property
    def feature_importances_(self):
        return np.mean([m.feature_importances_ for m in self.models], axis=0)

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)

    def predict(self, X):
        if self.kind == "clf":
            return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
        return np.mean([m.predict(X) for m in self.models], axis=0)


def load_model(target, model_dir=MODEL_DIR, suffix="2025_26"):
    """The fitted model for one target, rebuilt from members where possible.

    The members file is tried first because it contains nothing but LightGBM
    estimators and a dict, so it loads no matter how the trainer was invoked.
    The single-object file is the fallback for targets saved before the
    ensemble existed.
    """
    members_path = os.path.join(model_dir, f"{target}_ensemble_{suffix}.pkl")
    if os.path.exists(members_path):
        with open(members_path, "rb") as f:
            blob = pickle.load(f)
        return SeedEnsemble(blob["members"], blob["kind"])

    single_path = os.path.join(model_dir, f"{target}_model_{suffix}.pkl")
    with open(single_path, "rb") as f:
        return pickle.load(f)

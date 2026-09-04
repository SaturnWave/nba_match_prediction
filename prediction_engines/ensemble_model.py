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


class BlendedForecaster:
    """Three views of one game, averaged in probability space.

    WHY THREE
        Measured over 430 walk-forward cells, paired within each (month, seed):

          blend3        +0.0183 accuracy, +0.0150 AUC (15 standard errors)
          blend         +0.0146
          margin_prob   +0.0105
          base+cal      +0.0055

        The classifier alone sits exactly on its own information ceiling - its
        calibrated accuracy was 0.6564 against a ceiling of 0.6559 - so nothing
        done to its probabilities after the fact can help. The blend does not
        post-process them; it replaces them with a better forecast, and the AUC
        gain is what proves that rather than the accuracy gain.

        The logistic view is the instructive one. On its own it is no more
        informative than the classifier (AUC +0.0017, inside the noise), yet
        adding it to the blend is worth another +0.0037. A view does not have to
        be better to be useful - it has to be wrong in different places, and a
        linear model cannot represent the interactions the trees run on.

        A fourth view, taking the margin from the home_score and away_score
        models and subtracting, was screened and dropped: +0.0006, and adding it
        made the blend worse.

    Averaged in probability space, not logit: a view that is confidently wrong
    drags a logit mean much further, and these views disagree precisely where
    one of them is overconfident.
    """

    def __init__(self, classifier, margin_model, margin_mapper, logistic):
        self.classifier = classifier
        self.margin_model = margin_model
        self.margin_mapper = margin_mapper
        self.logistic = logistic

    def views(self, X):
        """Each view's win probability, for showing where they disagree."""
        out = {"classifier": self.classifier.predict_proba(X)[:, 1]}
        margin = self.margin_model.predict(X)
        out["margin"] = np.clip(
            self.margin_mapper.predict_proba(margin.reshape(-1, 1))[:, 1],
            1e-6, 1 - 1e-6)
        out["logistic"] = np.clip(self.logistic.predict_proba(X)[:, 1],
                                  1e-6, 1 - 1e-6)
        return out

    def predict_proba(self, X):
        p = np.mean(list(self.views(X).values()), axis=0)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def load_blend(model_dir=MODEL_DIR, suffix="2025_26"):
    """The blended forecaster, or None when it has not been trained yet.

    Stored as parts rather than as a pickled BlendedForecaster for the same
    reason load_model rebuilds SeedEnsemble: the trainer runs as __main__, so a
    pickled instance records a class path nothing else can resolve.
    """
    path = os.path.join(model_dir, f"blend_{suffix}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        parts = pickle.load(f)
    return BlendedForecaster(
        SeedEnsemble(parts["classifier_members"], "clf"),
        SeedEnsemble(parts["margin_members"], "reg"),
        parts["margin_mapper"], parts["logistic"])

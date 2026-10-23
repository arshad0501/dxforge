"""Outcome / NRC prediction from tabular diagnostic context.

Three tasks:
  1. binary positive/negative outcome (Decision Tree → RF → GradBoost)
  2. multi-class NRC code prediction on negative rows only
  3. comparison of raw-id features vs. KB-enriched domain features

The shallow Decision Tree is particularly valuable: it recovers the
deterministic engine rules in human-readable form (a natural sanity-check).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingClassifier,
                               RandomForestClassifier)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, learning_curve)
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

from . import SEED
from .dataset import SESSION_RANK
from .features import domain_tabular, raw_tabular

BINARY_LABEL = "outcome_binary"


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[BINARY_LABEL] = (df["label"] == "POSITIVE").astype(int)
    df["session_rank"] = df["session"].map(SESSION_RANK)
    return df


def candidate_models() -> dict[str, object]:
    return {
        "Decision Tree (depth=4)": DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=SEED),
        "Decision Tree (depth=8)": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced_subsample",
            random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1, random_state=SEED),
    }


def cv_splitter(n_splits: int = 5) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def compare_models(df: pd.DataFrame, feat_fn=None,
                   n_splits: int = 5) -> pd.DataFrame:
    """Stratified k-fold CV comparing all candidate models."""
    feat_fn = feat_fn or domain_tabular
    df = _prep(df)
    X = feat_fn(df)
    y = df[BINARY_LABEL]
    rows = []
    for name, clf in candidate_models().items():
        res = cross_validate(clf, X, y, cv=cv_splitter(n_splits),
                             scoring=["accuracy", "f1_macro"], n_jobs=-1)
        rows.append({
            "model": name,
            "cv_accuracy_mean": res["test_accuracy"].mean(),
            "cv_accuracy_std":  res["test_accuracy"].std(),
            "cv_f1_macro_mean": res["test_f1_macro"].mean(),
            "cv_f1_macro_std":  res["test_f1_macro"].std(),
            "fit_time_s":       res["fit_time"].mean(),
        })
    return (pd.DataFrame(rows)
            .sort_values("cv_f1_macro_mean", ascending=False)
            .reset_index(drop=True))


def feature_comparison(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Raw ids vs. KB-enriched domain features on the same model."""
    df = _prep(df)
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                 random_state=SEED, n_jobs=-1)
    rows = []
    for name, Xfn in [("raw_ids", raw_tabular), ("domain_features", domain_tabular)]:
        X = Xfn(df)
        res = cross_validate(clf, X, df[BINARY_LABEL],
                             cv=cv_splitter(n_splits),
                             scoring=["accuracy", "f1_macro"], n_jobs=-1)
        rows.append({
            "features": name, "n_features": X.shape[1],
            "cv_accuracy_mean": res["test_accuracy"].mean(),
            "cv_f1_macro_mean": res["test_f1_macro"].mean(),
        })
    return pd.DataFrame(rows)


def tune_rf(df: pd.DataFrame, n_splits: int = 5) -> GridSearchCV:
    """Grid search for the best Random Forest hyper-parameters."""
    df = _prep(df)
    X = domain_tabular(df)
    y = df[BINARY_LABEL]
    grid = {"n_estimators": [100, 200], "max_depth": [6, 10, None],
            "min_samples_leaf": [1, 3]}
    gs = GridSearchCV(
        RandomForestClassifier(class_weight="balanced_subsample",
                               random_state=SEED, n_jobs=-1),
        grid, cv=cv_splitter(n_splits), scoring="f1_macro",
        n_jobs=-1, refit=True, return_train_score=True)
    gs.fit(X, y)
    return gs


def fit_final(df: pd.DataFrame, model=None) -> tuple[object, LabelEncoder, list[str]]:
    """Fit the chosen model on all training data."""
    df = _prep(df)
    X = domain_tabular(df)
    y = df[BINARY_LABEL]
    if model is None:
        model = RandomForestClassifier(n_estimators=200, max_depth=10,
                                       class_weight="balanced_subsample",
                                       random_state=SEED, n_jobs=-1)
    model.fit(X, y)
    return model, X.columns.tolist()


def tree_rules(df: pd.DataFrame, max_depth: int = 4) -> str:
    """Export a shallow Decision Tree as human-readable rules."""
    df = _prep(df)
    X = domain_tabular(df)
    y = df[BINARY_LABEL]
    dt = DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced",
                                 random_state=SEED)
    dt.fit(X, y)
    return export_text(dt, feature_names=list(X.columns))


def nrc_model(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Multi-class NRC prediction on the negative rows only."""
    neg = _prep(df.copy())
    neg = neg[neg["label"] != "POSITIVE"].copy()
    X = domain_tabular(neg)
    le = LabelEncoder()
    y = le.fit_transform(neg["label"])
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                 random_state=SEED, n_jobs=-1)
    res = cross_validate(clf, X, y, cv=StratifiedKFold(min(n_splits, neg[
        "label"].nunique()), shuffle=True, random_state=SEED),
        scoring=["accuracy", "f1_macro"], n_jobs=-1)
    return pd.DataFrame({
        "task": ["NRC prediction (negative rows)"],
        "n_classes": [neg["label"].nunique()],
        "cv_accuracy_mean": [res["test_accuracy"].mean()],
        "cv_f1_macro_mean": [res["test_f1_macro"].mean()],
    })


def permutation_importance_table(model, X: pd.DataFrame,
                                 y: pd.Series, n_repeats: int = 10) -> pd.DataFrame:
    r = permutation_importance(model, X, y, n_repeats=n_repeats,
                               random_state=SEED, n_jobs=-1)
    return (pd.DataFrame({"feature": X.columns,
                           "importance_mean": r.importances_mean,
                           "importance_std":  r.importances_std})
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True))


def lc(df: pd.DataFrame, n_splits: int = 5) -> dict:
    df = _prep(df)
    X = domain_tabular(df)
    y = df[BINARY_LABEL]
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                 random_state=SEED, n_jobs=-1)
    sizes, tr, te = learning_curve(
        clf, X, y, cv=cv_splitter(n_splits),
        train_sizes=np.linspace(0.1, 1.0, 6), scoring="f1_macro",
        n_jobs=-1, shuffle=True, random_state=SEED)
    return {"sizes": sizes.tolist(), "train": tr.mean(1).tolist(),
            "test": te.mean(1).tolist(), "test_std": te.std(1).tolist()}

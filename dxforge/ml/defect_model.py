"""ECU response defect detection (anomaly detection).

Trains on a *golden* ECU run (defect_rate=0.0) and flags unusual
response patterns in a test run that may include injected defects.

Models: Isolation Forest (primary) + One-Class SVM (comparison).
Features come from ``TraceFeaturizer`` in ``features.py`` — all
observable from the test harness without oracle access.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from . import SEED
from .dataset import generate_traces
from .features import TraceFeaturizer


def _label_anomaly(df: pd.DataFrame) -> pd.Series:
    """1 = defect, 0 = normal (based on injected defect field)."""
    return (df["defect"] != "none").astype(int)


def build_datasets(n_train: int = 3000, n_test: int = 2000,
                   defect_rate: float = 0.15):
    """Generate golden training traces and a mixed test set."""
    train_raw = generate_traces(n=n_train, defect_rate=0.0,  seed=SEED)
    test_raw  = generate_traces(n=n_test,  defect_rate=defect_rate, seed=SEED + 1)
    feat = TraceFeaturizer().fit(train_raw)
    X_train = feat.transform(train_raw)
    X_test  = feat.transform(test_raw)
    y_test  = _label_anomaly(test_raw)
    defect_types = test_raw["defect"]
    return X_train, X_test, y_test, defect_types, feat


def compare_models(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_test: pd.Series) -> pd.DataFrame:
    """Evaluate Isolation Forest vs. One-Class SVM."""
    scaler = StandardScaler().fit(X_train)
    Xtr_s = scaler.transform(X_train)
    Xte_s = scaler.transform(X_test)
    rows = []
    for name, clf, Xtr, Xte in [
        ("Isolation Forest",
         IsolationForest(n_estimators=200, contamination=0.1,
                         random_state=SEED, n_jobs=-1),
         Xtr_s, Xte_s),
        ("One-Class SVM (rbf)",
         OneClassSVM(kernel="rbf", nu=0.1),
         Xtr_s, Xte_s),
    ]:
        clf.fit(Xtr)
        raw = clf.decision_function(Xte)
        # Isolation Forest: higher = more normal; flip for anomaly score
        if hasattr(clf, "offset_"):     # IForest
            score = -raw
        else:
            score = -raw               # OCSVM: same convention
        pred = (clf.predict(Xte) == -1).astype(int)
        rows.append({
            "model": name,
            "roc_auc":          float(roc_auc_score(y_test, score)),
            "avg_precision":    float(average_precision_score(y_test, score)),
            "precision":        float((pred[y_test == 1] == 1).sum() /
                                      max(pred.sum(), 1)),
            "recall":           float((pred[y_test == 1] == 1).sum() /
                                      max((y_test == 1).sum(), 1)),
        })
    return pd.DataFrame(rows)


def fit_isolation_forest(X_train: pd.DataFrame,
                         contamination: float = 0.1) -> tuple[IsolationForest, StandardScaler]:
    scaler = StandardScaler().fit(X_train)
    iforest = IsolationForest(n_estimators=200, contamination=contamination,
                               max_samples="auto", random_state=SEED, n_jobs=-1)
    iforest.fit(scaler.transform(X_train))
    return iforest, scaler


def score(iforest: IsolationForest, scaler: StandardScaler,
          X: pd.DataFrame) -> np.ndarray:
    """Anomaly score (higher = more anomalous)."""
    return -iforest.score_samples(scaler.transform(X))


def oracle_flags(raw: pd.DataFrame) -> pd.Series:
    """Deterministic conformance check: observed outcome/NRC vs the engine's expectation.

    Catches semantic defects (wrong NRC, unexpected positive/negative) but is
    blind to timing and length, which is what the anomaly detector covers.
    """
    obs_neg = raw["response_sid"] == 0x7F
    exp_neg = raw["expected_outcome"] == "negative"
    return (raw["resp_len"] > 0) & ((obs_neg != exp_neg) | (obs_neg & (raw["nrc"] != raw["expected_nrc"])))


def detector_comparison(iforest: IsolationForest, scaler: StandardScaler, X_test: pd.DataFrame,
                        n_test: int = 1000, defect_rate: float = 0.15) -> pd.DataFrame:
    """Isolation Forest vs engine oracle vs their union, with recall per defect type.

    ``X_test`` must come from ``build_datasets`` with the same ``n_test`` /
    ``defect_rate`` (the raw test run is regenerated with the same seed).
    """
    raw = generate_traces(n=n_test, defect_rate=defect_rate, seed=SEED + 1)
    truth = raw["defect"] != "none"
    ml = pd.Series(iforest.predict(scaler.transform(X_test)) == -1, index=raw.index)
    rule = oracle_flags(raw)
    rows = []
    for name, flag in [("Isolation Forest", ml), ("Engine oracle", rule), ("Hybrid (union)", ml | rule)]:
        per_type = raw.assign(f=flag).groupby("defect")["f"].mean()
        rows.append({"detector": name,
                     "precision": float((flag & truth).sum() / max(flag.sum(), 1)),
                     "recall": float((flag & truth).sum() / max(truth.sum(), 1)),
                     "false_alarm_rate": float(flag[~truth].mean()),
                     **{f"recall_{k}": float(v) for k, v in per_type.items() if k != "none"}})
    return pd.DataFrame(rows)


def defect_breakdown(iforest: IsolationForest, scaler: StandardScaler,
                     X_test: pd.DataFrame, defect_col: pd.Series) -> pd.DataFrame:
    """Mean anomaly score per defect type — useful for the EDA section."""
    s = score(iforest, scaler, X_test)
    return (pd.DataFrame({"defect": defect_col.values, "anomaly_score": s})
            .groupby("defect")["anomaly_score"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("mean", ascending=False))

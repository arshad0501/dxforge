"""Intent -> UDS service classifier: model comparison, tuning, evaluation.

All cross-validation is *grouped by paraphrase template* (StratifiedGroupKFold)
so a model is always scored on phrasings it has never seen.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, f1_score,
                             normalized_mutual_info_score)
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold, cross_val_predict,
                                     cross_validate, learning_curve, validation_curve)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from ..intent import parse_deterministic
from ..kb import KnowledgeBase
from . import SEED
from .dataset import REFUSAL_LABELS
from .features import text_features

TEXT_COL = "text_clean"


def cv_splitter(n_splits: int = 5) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def candidate_models() -> dict[str, object]:
    """Classifiers from the syllabus, each with sensible starting hyper-parameters."""
    return {
        "Multinomial NB": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(C=10, max_iter=3000, class_weight="balanced"),
        "Linear SVM": LinearSVC(C=1.0, class_weight="balanced"),
        "KNN (cosine)": KNeighborsClassifier(n_neighbors=7, metric="cosine", weights="distance"),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1,
                                                class_weight="balanced_subsample"),
        "MLP (1x128)": MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, early_stopping=True,
                                     random_state=SEED),
    }


def make_pipeline(clf, **feat_kw) -> Pipeline:
    return Pipeline([("features", text_features(**feat_kw)), ("clf", clf)])


def compare_models(train: pd.DataFrame, n_splits: int = 5,
                   models: dict | None = None) -> pd.DataFrame:
    """Grouped k-fold CV for every candidate. Returns mean +/- std per metric."""
    models = models or candidate_models()
    X, y, g = train[TEXT_COL], train["label"], train["template_id"]
    rows = []
    for name, clf in models.items():
        res = cross_validate(make_pipeline(clf), X, y, groups=g, cv=cv_splitter(n_splits),
                             scoring=["accuracy", "f1_macro"], n_jobs=-1)
        rows.append({
            "model": name,
            "cv_accuracy_mean": res["test_accuracy"].mean(), "cv_accuracy_std": res["test_accuracy"].std(),
            "cv_f1_macro_mean": res["test_f1_macro"].mean(), "cv_f1_macro_std": res["test_f1_macro"].std(),
            "fit_time_s": res["fit_time"].mean(),
        })
    return pd.DataFrame(rows).sort_values("cv_f1_macro_mean", ascending=False).reset_index(drop=True)


def tune(train: pd.DataFrame, n_splits: int = 5) -> GridSearchCV:
    """Grid search over feature representation x regularisation strength."""
    grid = {
        "features": [
            text_features(word_ngrams=(1, 1), use_char=False),
            text_features(word_ngrams=(1, 2), use_char=False),
            text_features(word_ngrams=(1, 2), char_ngrams=(2, 4)),
            text_features(word_ngrams=(1, 2), char_ngrams=(2, 5)),
        ],
        "clf__C": [0.3, 1, 3, 10, 30],
    }
    pipe = make_pipeline(LogisticRegression(max_iter=3000, class_weight="balanced"))
    gs = GridSearchCV(pipe, grid, cv=cv_splitter(n_splits), scoring="f1_macro", n_jobs=-1,
                      refit=True, return_train_score=True)
    gs.fit(train[TEXT_COL], train["label"], groups=train["template_id"])
    return gs


def describe_features(feat) -> str:
    """Readable name for a feature-union candidate in the grid."""
    if hasattr(feat, "transformer_list"):
        w = feat.transformer_list[0][1].ngram_range
        c = feat.transformer_list[1][1].ngram_range
        return f"word{w}+char{c}"
    return f"word{feat.ngram_range}"


def grid_table(gs: GridSearchCV) -> pd.DataFrame:
    r = pd.DataFrame(gs.cv_results_)
    return pd.DataFrame({
        "features": r["param_features"].map(describe_features),
        "C": r["param_clf__C"],
        "cv_f1_macro": r["mean_test_score"], "cv_f1_std": r["std_test_score"],
        "train_f1_macro": r["mean_train_score"],
    }).sort_values("cv_f1_macro", ascending=False).reset_index(drop=True)


def curves(estimator: Pipeline, train: pd.DataFrame, n_splits: int = 5) -> dict:
    """Learning curve (data size) + validation curve (C) for over/under-fit analysis."""
    X, y, g = train[TEXT_COL], train["label"], train["template_id"]
    sizes, tr, te = learning_curve(clone(estimator), X, y, groups=g, cv=cv_splitter(n_splits),
                                   train_sizes=np.linspace(0.1, 1.0, 6), scoring="f1_macro",
                                   n_jobs=-1, shuffle=True, random_state=SEED)
    c_range = np.logspace(-2, 3, 8)
    vtr, vte = validation_curve(clone(estimator), X, y, groups=g, param_name="clf__C",
                                param_range=c_range, cv=cv_splitter(n_splits),
                                scoring="f1_macro", n_jobs=-1)
    return {"sizes": sizes.tolist(), "lc_train": tr.mean(1).tolist(), "lc_test": te.mean(1).tolist(),
            "lc_test_std": te.std(1).tolist(), "c_range": c_range.tolist(),
            "vc_train": vtr.mean(1).tolist(), "vc_test": vte.mean(1).tolist(),
            "vc_test_std": vte.std(1).tolist()}


def feature_selection_curve(train: pd.DataFrame, feat, C: float,
                            ks=(50, 100, 250, 500, 1000, 2500, 5000, "all")) -> pd.DataFrame:
    X, y, g = train[TEXT_COL], train["label"], train["template_id"]
    n_total = clone(feat).fit_transform(X).shape[1]
    rows = []
    for k in ks:
        kk = k if k == "all" or k < n_total else "all"
        pipe = Pipeline([("features", clone(feat)), ("select", SelectKBest(chi2, k=kk)),
                         ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))])
        s = cross_validate(pipe, X, y, groups=g, cv=cv_splitter(), scoring="f1_macro", n_jobs=-1)
        rows.append({"k": n_total if kk == "all" else kk, "cv_f1_macro": s["test_score"].mean(),
                     "cv_f1_std": s["test_score"].std()})
    return pd.DataFrame(rows).drop_duplicates("k")


def svd_curve(train: pd.DataFrame, feat, C: float,
              components=(5, 10, 25, 50, 100, 200, 400)) -> pd.DataFrame:
    X, y, g = train[TEXT_COL], train["label"], train["template_id"]
    rows = []
    for n in components:
        pipe = Pipeline([("features", clone(feat)), ("svd", TruncatedSVD(n, random_state=SEED)),
                         ("norm", Normalizer()),
                         ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))])
        s = cross_validate(pipe, X, y, groups=g, cv=cv_splitter(), scoring="f1_macro", n_jobs=-1)
        rows.append({"components": n, "cv_f1_macro": s["test_score"].mean(),
                     "cv_f1_std": s["test_score"].std()})
    return pd.DataFrame(rows)


def embedding_2d(df: pd.DataFrame, feat) -> tuple[pd.DataFrame, dict]:
    """TF-IDF -> SVD(50) -> PCA(2) projection, plus K-Means clustering quality."""
    Xt = clone(feat).fit_transform(df[TEXT_COL])
    svd = TruncatedSVD(50, random_state=SEED)
    Z = Normalizer().fit_transform(svd.fit_transform(Xt))
    pca = PCA(2, random_state=SEED)
    P = pca.fit_transform(Z)
    km = KMeans(n_clusters=df["label"].nunique(), n_init=10, random_state=SEED).fit(Z)
    stats = {
        "svd50_explained_variance": float(svd.explained_variance_ratio_.sum()),
        "pca2_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "kmeans_ari": float(adjusted_rand_score(df["label"], km.labels_)),
        "kmeans_nmi": float(normalized_mutual_info_score(df["label"], km.labels_)),
    }
    out = pd.DataFrame({"pc1": P[:, 0], "pc2": P[:, 1], "label": df["label"].values,
                        "cluster": km.labels_})
    return out, stats


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def keyword_baseline(texts: pd.Series, kb: KnowledgeBase) -> list[str]:
    """The original rule parser, mapped to the same label space."""
    out = []
    for t in texts:
        p = parse_deterministic(t, kb)
        out.append(p.matched_service if p.request is not None else "OutOfScope")
    return out


def majority_baseline(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    d = DummyClassifier(strategy="most_frequent").fit(train[[TEXT_COL]], train["label"])
    return list(d.predict(test[[TEXT_COL]]))


def scores(y_true, y_pred) -> dict:
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}


# --------------------------------------------------------------------------- #
# Confidence threshold (selective classification)
# --------------------------------------------------------------------------- #
def oof_proba(estimator: Pipeline, train: pd.DataFrame) -> np.ndarray:
    return cross_val_predict(clone(estimator), train[TEXT_COL], train["label"],
                             groups=train["template_id"], cv=cv_splitter(),
                             method="predict_proba", n_jobs=-1)


def coverage_curve(proba: np.ndarray, classes, y_true) -> pd.DataFrame:
    pred = np.asarray(classes)[proba.argmax(1)]
    conf = proba.max(1)
    y = np.asarray(y_true)
    rows = []
    for t in np.round(np.arange(0.0, 1.0, 0.05), 2):
        keep = conf >= t
        rows.append({"threshold": float(t), "coverage": float(keep.mean()),
                     "selective_accuracy": float((pred[keep] == y[keep]).mean()) if keep.any() else 1.0})
    return pd.DataFrame(rows)


CONFIDENCE_TARGET = 0.90


def pick_threshold(curve: pd.DataFrame, target_accuracy: float = CONFIDENCE_TARGET) -> float:
    """Lowest threshold whose answered predictions reach ``target_accuracy``.

    If the target is unreachable, return the most accurate threshold rather
    than a silent default.
    """
    ok = curve[curve["selective_accuracy"] >= target_accuracy]
    if len(ok):
        return float(ok["threshold"].min())
    return float(curve.loc[curve["selective_accuracy"].idxmax(), "threshold"])


# --------------------------------------------------------------------------- #
# End-to-end: hybrid (ML -> slots -> engine) vs keyword parser
# --------------------------------------------------------------------------- #
def end_to_end(df: pd.DataFrame, parse_fn: Callable[[str], tuple[str, str | None, bool]]) -> dict:
    """``parse_fn(text) -> (service_label, request_hex | None, abstained)``.

    A row is *correct* if the request bytes match (or a refusal label was
    refused), *abstained* if the system declined, otherwise *wrong* — a
    silent error, the outcome that matters most for safety.
    """
    correct = wrong = abstained = 0
    for text, label, req_hex in zip(df["text"], df["label"], df["request_hex"]):
        svc, hx, abst = parse_fn(text)
        if label in REFUSAL_LABELS:
            if abst or svc in REFUSAL_LABELS or hx is None:
                correct += 1
            else:
                wrong += 1
            continue
        if abst or hx is None:
            abstained += 1
        elif hx == req_hex:
            correct += 1
        else:
            wrong += 1
    n = len(df)
    return {"n": n, "correct": correct / n, "abstained": abstained / n, "silent_error": wrong / n}


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0

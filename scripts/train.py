"""
scripts/train.py
================
Full DxForge ML training pipeline. Run once before launching the app.

    python scripts/train.py          # trains all three models
    python scripts/train.py --quick  # smaller dataset for faster dev

Outputs (in models/):
    intent_clf.joblib    -- TF-IDF + Logistic Regression intent classifier
    outcome_clf.joblib   -- Random Forest outcome predictor
    defect_iforest.joblib -- Isolation Forest defect detector
    metrics.json         -- all evaluation metrics (honest, CV-based)
    model_card.md        -- model documentation for Responsible AI marks
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd

from dxforge.kb import load_kb
from dxforge.ml import SEED
from dxforge.ml.dataset import (generate_intents, generate_outcomes, clean_intents, HELDOUT_PROFILE)
from dxforge.ml.defect_model import (build_datasets, compare_models as defect_compare,
                                      fit_isolation_forest, defect_breakdown,
                                      detector_comparison)
from dxforge.ml.features import domain_tabular, raw_tabular
from dxforge.ml.intent_model import (compare_models, tune, grid_table, feature_selection_curve, svd_curve,
                                      embedding_2d, keyword_baseline, majority_baseline,
                                      scores, oof_proba, coverage_curve, pick_threshold,
                                      TEXT_COL, CONFIDENCE_TARGET)
from dxforge.ml.outcome_model import (compare_models as oc_compare, feature_comparison,
                                       fit_final, tree_rules, nrc_model)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)


def sep(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def run(quick: bool = False) -> dict:
    metrics: dict = {"seed": SEED, "quick": quick}

    # ------------------------------------------------------------------ #
    # 1. DATASETS                                                          #
    # ------------------------------------------------------------------ #
    sep("1 / 5  Generating datasets")
    kb = load_kb()
    scale = 0.3 if quick else 1.0
    n_outcomes = 500 if quick else 2000
    n_traces   = 500 if quick else 3000

    intent_df = generate_intents(kb, scale=scale)
    print(f"  Intents (raw): {len(intent_df):,} rows  {intent_df.label.nunique()} classes")
    print(f"  Dupes    : {intent_df.duplicated('text').sum()}")
    print(f"  Empty    : {intent_df['text'].isna().sum() + (intent_df['text'] == '').sum()}")
    intent_df, clean_report = clean_intents(intent_df)
    print(f"  After clean: {len(intent_df):,} rows  (dropped {clean_report.get('dropped_empty',0)} empty, {clean_report.get('dropped_duplicates',0)} dupes)")
    metrics["intent"] = {
        "n_rows": len(intent_df), "n_classes": int(intent_df.label.nunique()),
        "class_counts": intent_df.label.value_counts().to_dict(),
    }

    outcome_df = generate_outcomes(n_per_profile=n_outcomes)
    print(f"  Outcomes : {len(outcome_df):,} rows  {outcome_df.label.nunique()} classes")
    metrics["outcome"] = {
        "n_rows": len(outcome_df), "n_classes": int(outcome_df.label.nunique()),
    }

    # train/hold-out split for outcomes: hold out one entire ECU profile
    train_out = outcome_df[outcome_df.ecu_id != HELDOUT_PROFILE].copy()
    test_out  = outcome_df[outcome_df.ecu_id == HELDOUT_PROFILE].copy()
    print(f"  Outcome train={len(train_out)} test={len(test_out)} (held-out ECU profile)")

    # ------------------------------------------------------------------ #
    # 2. INTENT CLASSIFIER                                                 #
    # ------------------------------------------------------------------ #
    sep("2 / 5  Intent classifier")

    # 2a. quick model comparison (5-fold grouped CV)
    print("  Model comparison (5-fold grouped CV by template) ...")
    n_splits = 3 if quick else 5
    comp = compare_models(intent_df, n_splits=n_splits)
    print(comp[["model", "cv_f1_macro_mean", "cv_f1_macro_std"]].to_string(index=False))
    metrics["intent"]["model_comparison"] = comp.to_dict(orient="records")

    # 2b. grid search (best model)
    print("\n  Grid search (LogReg, feature representation x C) ...")
    gs = tune(intent_df, n_splits=n_splits)
    gt = grid_table(gs)
    print(gt.head(5).to_string(index=False))
    best_feat_desc  = gt.iloc[0]["features"]
    best_c          = float(gt.iloc[0]["C"])
    best_cv_f1      = float(gt.iloc[0]["cv_f1_macro"])
    metrics["intent"]["best_params"] = {
        "features": best_feat_desc, "C": best_c, "cv_f1_macro": best_cv_f1}
    print(f"  Best: {best_feat_desc}  C={best_c}  f1={best_cv_f1:.3f}")

    best_pipe = gs.best_estimator_

    # 2c. feature selection curve
    print("\n  Feature selection curve (chi-square k-best) ...")
    fsel = feature_selection_curve(intent_df, best_pipe.named_steps["features"],
                                   best_c, ks=[50, 100, 250, 500, "all"])
    metrics["intent"]["feature_selection_curve"] = fsel.to_dict(orient="records")
    print(fsel.to_string(index=False))

    # 2d. SVD / dimensionality reduction curve
    print("\n  SVD dimensionality reduction curve ...")
    svd = svd_curve(intent_df, best_pipe.named_steps["features"],
                    best_c, components=[5, 10, 25, 50, 100])
    metrics["intent"]["svd_curve"] = svd.to_dict(orient="records")
    print(svd.to_string(index=False))

    # 2e. 2-D embedding
    print("\n  2-D PCA embedding for plot ...")
    emb, emb_stats = embedding_2d(intent_df, best_pipe.named_steps["features"])
    emb.to_csv(MODELS_DIR / "intent_embedding.csv", index=False)
    metrics["intent"]["embedding"] = emb_stats
    print(f"  K-Means ARI={emb_stats['kmeans_ari']:.3f}  NMI={emb_stats['kmeans_nmi']:.3f}")

    # 2f. confidence threshold
    print("\n  Confidence threshold (selective classification) ...")
    oof = oof_proba(best_pipe, intent_df)
    cc  = coverage_curve(oof, list(best_pipe.classes_), intent_df["label"])
    thr = pick_threshold(cc)
    metrics["intent"]["confidence_threshold"] = float(thr)
    metrics["intent"]["confidence_coverage"] = float(cc[cc.threshold == thr].coverage.values[0])
    metrics["intent"]["confidence_selective_accuracy"] = float(
        cc[cc.threshold == thr].selective_accuracy.values[0])
    metrics["intent"]["confidence_target"] = CONFIDENCE_TARGET
    print(f"  Threshold={thr:.2f}  coverage={cc[cc.threshold==thr].coverage.values[0]:.3f}")

    # 2g. baselines
    challenge = pd.read_csv(
        Path(__file__).parent.parent / "data" / "ml" / "challenge_intents.csv")
    challenge[TEXT_COL] = challenge["text"].apply(
        __import__("dxforge.ml.text", fromlist=["normalize"]).normalize)
    kw_pred  = keyword_baseline(challenge["text"], kb)
    maj_pred = majority_baseline(intent_df, challenge)
    ml_pred  = best_pipe.predict(challenge[TEXT_COL])
    metrics["intent"]["challenge_baselines"] = {
        "majority_class": scores(challenge["label"], maj_pred),
        "keyword_parser": scores(challenge["label"], kw_pred),
        "ml_logreg":      scores(challenge["label"], ml_pred),
    }
    print("\n  Challenge set baselines:")
    for k, v in metrics["intent"]["challenge_baselines"].items():
        print(f"    {k}: acc={v['accuracy']:.3f}  f1={v['f1_macro']:.3f}")

    # 2h. error analysis on challenge set
    errors = challenge[challenge["label"] != ml_pred].copy()
    errors["predicted"] = np.array(ml_pred)[challenge["label"] != ml_pred]
    errors[["text", "label", "predicted"]].to_csv(
        MODELS_DIR / "intent_errors.csv", index=False)
    metrics["intent"]["challenge_errors"] = int(len(errors))

    # 2i. save
    joblib.dump({"pipeline": best_pipe, "classes": list(best_pipe.classes_),
                 "threshold": thr, "seed": SEED},
                MODELS_DIR / "intent_clf.joblib")
    print("\n  Saved intent_clf.joblib")

    # ------------------------------------------------------------------ #
    # 3. OUTCOME / NRC MODEL                                               #
    # ------------------------------------------------------------------ #
    sep("3 / 5  Outcome model (tabular)")

    comp2 = oc_compare(train_out, n_splits=n_splits)
    print(comp2[["model", "cv_f1_macro_mean"]].to_string(index=False))
    metrics["outcome"]["model_comparison"] = comp2.to_dict(orient="records")

    fcmp = feature_comparison(train_out, n_splits=n_splits)
    print("\n  Feature comparison (raw ids vs domain features):")
    print(fcmp.to_string(index=False))
    metrics["outcome"]["feature_comparison"] = fcmp.to_dict(orient="records")

    print("\n  Decision tree rules (depth 4):")
    rules_txt = tree_rules(train_out, max_depth=4)
    (MODELS_DIR / "decision_tree_rules.txt").write_text(rules_txt)
    print(rules_txt[:1500])

    print("\n  NRC multi-class prediction:")
    nrc_res = nrc_model(train_out, n_splits=n_splits)
    print(nrc_res.to_string(index=False))
    metrics["outcome"]["nrc_model"] = nrc_res.to_dict(orient="records")

    # held-out ECU profile evaluation
    rf_model, feat_cols = fit_final(train_out)
    test_out_prep = test_out.copy()
    test_out_prep["outcome_binary"] = (test_out_prep["label"] == "POSITIVE").astype(int)
    test_out_prep["session_rank"] = test_out_prep["session"].map(
        __import__("dxforge.ml.dataset", fromlist=["SESSION_RANK"]).SESSION_RANK)
    X_test_out = domain_tabular(test_out_prep)[feat_cols]
    y_test_out = test_out_prep["outcome_binary"]
    from sklearn.metrics import accuracy_score, f1_score as f1
    held_out_acc = accuracy_score(y_test_out, rf_model.predict(X_test_out))
    held_out_f1  = f1(y_test_out, rf_model.predict(X_test_out), average="macro")
    metrics["outcome"]["held_out_ecu"] = {
        "task": "binary positive/negative",
        "accuracy": float(held_out_acc), "f1_macro": float(held_out_f1)}
    print(f"\n  Held-out ECU acc={held_out_acc:.3f}  f1={held_out_f1:.3f}")

    # same model family on raw ids only: does feature engineering transfer to an unseen ECU?
    from sklearn.ensemble import RandomForestClassifier
    rf_raw = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced_subsample",
                                    random_state=SEED, n_jobs=-1)
    rf_raw.fit(raw_tabular(train_out), (train_out["label"] == "POSITIVE").astype(int))
    raw_pred = rf_raw.predict(raw_tabular(test_out))
    metrics["outcome"]["held_out_ecu_raw_ids"] = {
        "accuracy": float(accuracy_score(y_test_out, raw_pred)),
        "f1_macro": float(f1(y_test_out, raw_pred, average="macro"))}
    print(f"  Held-out ECU with raw ids only: f1={metrics['outcome']['held_out_ecu_raw_ids']['f1_macro']:.3f}")

    joblib.dump({"model": rf_model, "feature_cols": feat_cols, "seed": SEED},
                MODELS_DIR / "outcome_clf.joblib")
    print("  Saved outcome_clf.joblib")

    # ------------------------------------------------------------------ #
    # 4. DEFECT DETECTION (ISOLATION FOREST)                              #
    # ------------------------------------------------------------------ #
    sep("4 / 5  Defect detection (anomaly)")

    X_train, X_test, y_test, defect_types, feat_obj = build_datasets(
        n_train=n_traces, n_test=1000, defect_rate=0.15)
    dcmp = defect_compare(X_train, X_test, y_test)
    print(dcmp.to_string(index=False))
    metrics["defect"] = dcmp.to_dict(orient="records")

    iforest, scaler = fit_isolation_forest(X_train)
    breakdown = defect_breakdown(iforest, scaler, X_test, defect_types)
    metrics["defect_breakdown"] = breakdown.to_dict(orient="records")
    hyb = detector_comparison(iforest, scaler, X_test, n_test=1000, defect_rate=0.15)
    metrics["defect_hybrid"] = hyb.to_dict(orient="records")
    print("\n  ML vs engine oracle vs hybrid:")
    print(hyb[["detector", "precision", "recall", "false_alarm_rate"]].to_string(index=False))
    print("\n  Defect type breakdown:")
    print(breakdown.to_string(index=False))

    joblib.dump({"iforest": iforest, "scaler": scaler,
                 "featurizer": feat_obj, "seed": SEED},
                MODELS_DIR / "defect_iforest.joblib")
    print("  Saved defect_iforest.joblib")

    # ------------------------------------------------------------------ #
    # 5. SAVE METRICS + MODEL CARD                                        #
    # ------------------------------------------------------------------ #
    sep("5 / 5  Saving metrics + model card")

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("  Saved metrics.json")

    _write_model_card(MODELS_DIR, metrics)
    print("  Saved model_card.md")

    return metrics


def _write_model_card(models_dir: Path, m: dict) -> None:
    ic, oc = m.get("intent", {}), m.get("outcome", {})
    bp = ic.get("best_params", {})
    ch = ic.get("challenge_baselines", {})
    ho, ho_raw = oc.get("held_out_ecu", {}), oc.get("held_out_ecu_raw_ids", {})
    fc = {r["features"]: r for r in oc.get("feature_comparison", [])}
    defect = {r["model"]: r for r in m.get("defect", [])}
    hyb = {r["detector"]: r for r in m.get("defect_hybrid", [])}
    comp = ic.get("model_comparison", [])
    top = comp[0] if comp else {}
    mode = "quick (reduced data, 3-fold)" if m.get("quick") else "full (5-fold)"

    def g(d: dict, k: str) -> float:
        return float(d.get(k, float("nan")))

    card = f"""# DxForge ML — Model Card

> Generated by `scripts/train.py` (training mode: {mode}, seed {m.get("seed")}). Every number below is read from `models/metrics.json`.

## Intended use
Engineering assistant for **bench/HIL** UDS (ISO 14229) diagnostic test design. The ML layer proposes; the deterministic
protocol engine produces every request/response byte. Not for use on production vehicles, and not a substitute for a qualified
diagnostic engineer's sign-off.

## Data (synthetic — disclosed)
No licence-clean public corpus of UDS intents exists and real ECU traces are proprietary, so all data is generated and seeded:
- **Intents**: {ic.get("n_rows", 0):,} cleaned rows, {ic.get("n_classes", 0)} classes (11 UDS services + `OutOfScope` + `UnsafeRequest`),
  from paraphrase frames × slot values × injected noise (typos, abbreviations, casing, filler, hex notations).
- **Outcomes**: {oc.get("n_rows", 0):,} rows of request × session × security state across 4 ECU profiles, labelled by the engine
  ({oc.get("n_classes", 0)} classes: POSITIVE + 6 NRCs). One profile (`DCU-DEMO`) is never seen in training.
- **Traces**: simulated ECU response logs (latency, length, NRC). Golden run for training, 15% injected defects for scoring.
- **External check**: `data/ml/challenge_intents.csv`, 67 hand-written intents written separately from the templates.
  Never used for training or tuning.

## Models
### 1. Intent classifier — TF-IDF + Logistic Regression
- Features: {bp.get("features", "?")} TF-IDF (sublinear TF), hex identifiers collapsed to one token. C = {bp.get("C", "?")}.
- Validation: 5-fold `StratifiedGroupKFold` grouped by paraphrase frame (test frames are never seen in training).
- Tuned CV macro-F1: **{g(bp, "cv_f1_macro"):.3f}**. The best untuned candidate was {top.get("model", "?")} ({g(top, "cv_f1_macro_mean"):.3f});
  Logistic Regression is deployed because the confidence gate needs calibrated probabilities.
- Challenge set: accuracy **{g(ch.get("ml_logreg", {}), "accuracy"):.1%}** / macro-F1 **{g(ch.get("ml_logreg", {}), "f1_macro"):.3f}**,
  vs keyword parser {g(ch.get("keyword_parser", {}), "accuracy"):.1%} and majority class {g(ch.get("majority_class", {}), "accuracy"):.1%}.
- Confidence gate: **{g(ic, "confidence_threshold"):.2f}**, the lowest threshold whose answered out-of-fold predictions reach
  ≥{g(ic, "confidence_target"):.0%} accuracy. At this gate the ML answers {g(ic, "confidence_coverage"):.0%} of unseen phrasings
  ({g(ic, "confidence_selective_accuracy"):.1%} accurate). The rest fall back to the keyword parser or are refused.

### 2. Outcome predictor — Random Forest (binary positive/negative)
- Features: 25 KB-derived domain features (service/DID/RID support, writability, session deficit, security need, interactions).
- 5-fold stratified CV F1: raw IDs {g(fc.get("raw_ids", {}), "cv_f1_macro_mean"):.3f} → domain features {g(fc.get("domain_features", {}), "cv_f1_macro_mean"):.3f}.
- **Unseen ECU profile**: macro-F1 **{g(ho, "f1_macro"):.3f}** with domain features vs {g(ho_raw, "f1_macro"):.3f} with raw IDs only.
- A depth-4 decision tree (`models/decision_tree_rules.txt`) exposes the learned rules for audit.

### 3. Defect detector — Isolation Forest (+ engine oracle)
- Trained on a golden run only (unsupervised); robust per-service latency z-score, length deviation, NRC, context.
- Isolation Forest: ROC-AUC {g(defect.get("Isolation Forest", {}), "roc_auc"):.3f}, average precision {g(defect.get("Isolation Forest", {}), "avg_precision"):.3f}.
  One-Class SVM: ROC-AUC {g(defect.get("One-Class SVM (rbf)", {}), "roc_auc"):.3f}, average precision {g(defect.get("One-Class SVM (rbf)", {}), "avg_precision"):.3f}.
- Isolation Forest is weak on semantic defects (wrong NRC, unexpected positive). The engine oracle catches those, so the deployed detector is the union:
  recall {g(hyb.get("Hybrid (union)", {}), "recall"):.1%} (IF alone {g(hyb.get("Isolation Forest", {}), "recall"):.1%}, oracle alone {g(hyb.get("Engine oracle", {}), "recall"):.1%}).

## Limitations
- Synthetic data: the scores show the method works, not field accuracy on real ECUs or real engineers' phrasing.
- 11 services only. Unknown services are refused, not guessed.
- Slot filling (DID/RID/sub-function) is rule-based; a correct service with an ungrounded parameter is refused.
- Defect simulation covers 6 defect types; real firmware faults are more varied.

## Safety stance
- `UnsafeRequest` predictions are refused at any confidence and never fall back to the keyword parser.
- No seed/key derivation. SecurityAccess is modelled as protocol state only.
- Human approval before export; audit-logged; project-isolated; bench/HIL only.
"""
    (models_dir / "model_card.md").write_text(card, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Small dataset for fast iteration")
    args = parser.parse_args()
    run(quick=args.quick)
    print("\nTraining complete. Run `streamlit run app.py` to launch the app.")

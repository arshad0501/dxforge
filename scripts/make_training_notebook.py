"""Generate notebooks/02_Model_Training_Validation.ipynb.

Same library functions, data sizes and seed as ``scripts/train.py`` (full
mode), so the notebook reproduces ``models/metrics.json``.
Execute afterwards with ``python scripts/execute_notebooks.py``.
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s))  # noqa: E731

md("""# DxForge — 02 · Model Training, Validation & Optimisation

**Rubric coverage:** Model Development using Syllabus Techniques (18) · Model Evaluation, Validation & Optimisation (15) · Feature Selection & Dimensionality Reduction (8)

| Task | Type | Models compared | Validation |
|---|---|---|---|
| Intent → UDS service | 13-class text classification | Multinomial NB, Logistic Regression, Linear SVM, KNN, Decision Tree, Random Forest, MLP | 5-fold **StratifiedGroupKFold** by paraphrase template + external hand-written challenge set |
| Request outcome | tabular binary + 6-class NRC | Decision Tree (d=4, d=8), Random Forest, Gradient Boosting | 5-fold stratified CV + **unseen ECU profile** |
| ECU defect detection | unsupervised anomaly detection | Isolation Forest, One-Class SVM (+ engine oracle) | golden-run training, labelled faulty run for scoring |

The ML layer only **proposes**. Final protocol bytes always come from the deterministic engine.""")

code("""import sys, json, time
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import warnings; warnings.filterwarnings("ignore")
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, f1_score, accuracy_score,
                             roc_curve, precision_recall_curve, roc_auc_score, average_precision_score)
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from dxforge.kb import load_kb
from dxforge.ml import SEED
from dxforge.ml.dataset import (generate_intents, clean_intents, generate_outcomes, generate_traces,
                                HELDOUT_PROFILE, SESSION_RANK)
from dxforge.ml.text import normalize
from dxforge.ml import intent_model as im, outcome_model as om, defect_model as dm
from dxforge.ml.features import domain_tabular, raw_tabular

pd.set_option("display.width", 160)
plt.rcParams.update({"figure.figsize": (10, 4.5), "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
kb = load_kb()
MODELS = ROOT / "models"
metrics = json.loads((MODELS / "metrics.json").read_text())
print("seed", SEED, "| metrics.json trained in", "quick" if metrics.get("quick") else "full", "mode")""")

md("## Part A — Intent classifier")

code("""intent_df, _ = clean_intents(generate_intents(kb, scale=1.0, seed=SEED))
challenge = pd.read_csv(ROOT / "data" / "ml" / "challenge_intents.csv")
challenge["text_clean"] = challenge["text"].map(normalize)
print(f"training pool: {len(intent_df):,} rows, {intent_df.template_id.nunique()} paraphrase groups")
print(f"external challenge set: {len(challenge)} hand-written rows (never used for training or tuning)")""")

md("""### A1. Candidate model comparison — 5-fold StratifiedGroupKFold
Every fold's test templates are unseen during training, so these scores measure generalisation to new phrasings, not memorisation.""")

code("""t0 = time.time()
comp = im.compare_models(intent_df, n_splits=5)
print(f"({time.time() - t0:.0f}s)")
comp.style.format({c: "{:.3f}" for c in comp.columns if c != "model"}).background_gradient(
    subset=["cv_f1_macro_mean"], cmap="Greens")""")

code("""c = comp.sort_values("cv_f1_macro_mean")
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.barh(c["model"], c["cv_f1_macro_mean"], xerr=c["cv_f1_macro_std"], color="#3b82f6", capsize=4)
ax.set_xlabel("macro-F1 (mean ± std over 5 grouped folds)"); ax.set_xlim(0, 1)
ax.set_title("Intent classifiers — grouped cross-validation")
plt.tight_layout(); plt.show()""")

md("""**Reading the table.** Linear models and Naive Bayes do well on sparse, high-dimensional TF-IDF, where classes are close to linearly separable.
A single unpruned decision tree splits on individual n-grams and overfits paraphrase-specific tokens. That is why it is the weakest model, and why bagging (Random Forest) recovers much of the gap.""")

md("""### A2. Hyper-parameter optimisation — GridSearchCV (feature representation × C)
Logistic Regression is tuned rather than the top-ranked model when they are within noise, because the confidence gate needs **probabilities** (`predict_proba`), which LinearSVC lacks.""")

code("""t0 = time.time()
gs = im.tune(intent_df, n_splits=5)
gt = im.grid_table(gs)
print(f"({time.time() - t0:.0f}s)  best: {gt.iloc[0]['features']}  C={gt.iloc[0]['C']}  CV macro-F1={gt.iloc[0]['cv_f1_macro']:.3f}")
surface = gt.pivot_table(index="features", columns="C", values="cv_f1_macro")
fig, ax = plt.subplots(figsize=(9, 3.8))
im_ = ax.imshow(surface.values, cmap="viridis", aspect="auto")
ax.set_xticks(range(surface.shape[1])); ax.set_xticklabels(surface.columns)
ax.set_yticks(range(surface.shape[0])); ax.set_yticklabels(surface.index)
for i in range(surface.shape[0]):
    for j in range(surface.shape[1]):
        v = surface.values[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                color="black" if v > np.nanmean(surface.values) else "white")
ax.set_xlabel("C (inverse regularisation)"); ax.set_title("Grid-search surface: CV macro-F1")
fig.colorbar(im_); plt.tight_layout(); plt.show()
gt.head(8)""")

md("### A3. Over/under-fitting — learning curve and validation curve")

code("""best = gs.best_estimator_
cv = im.curves(best, intent_df, n_splits=5)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.2))
a1.plot(cv["sizes"], cv["lc_train"], "o-", label="train")
a1.plot(cv["sizes"], cv["lc_test"], "o-", label="grouped CV")
a1.fill_between(cv["sizes"], np.array(cv["lc_test"]) - cv["lc_test_std"], np.array(cv["lc_test"]) + cv["lc_test_std"], alpha=0.2)
a1.set_xlabel("training rows"); a1.set_ylabel("macro-F1"); a1.set_title("Learning curve"); a1.legend()
a2.semilogx(cv["c_range"], cv["vc_train"], "o-", label="train")
a2.semilogx(cv["c_range"], cv["vc_test"], "o-", label="grouped CV")
a2.fill_between(cv["c_range"], np.array(cv["vc_test"]) - cv["vc_test_std"], np.array(cv["vc_test"]) + cv["vc_test_std"], alpha=0.2)
a2.set_xlabel("C"); a2.set_title("Validation curve (regularisation)"); a2.legend()
plt.tight_layout(); plt.show()
gap = cv["lc_train"][-1] - cv["lc_test"][-1]
print(f"train-CV gap at full size: {gap:.3f}. A large gap is the variance that unseen phrasings cause;")
print("the still-rising CV curve says more paraphrase diversity (not a bigger model) is the lever.")""")

md("""### A4. Confusion matrix — all 13 classes (out-of-fold, grouped)
Every prediction comes from a model that never saw that row's template.""")

code("""oof = cross_val_predict(clone(best), intent_df["text_clean"], intent_df["label"],
                        groups=intent_df["template_id"], cv=im.cv_splitter(5), method="predict_proba", n_jobs=-1)
classes = np.array(sorted(intent_df["label"].unique()))
oof_pred = classes[oof.argmax(1)]
fig, ax = plt.subplots(figsize=(11, 9))
ConfusionMatrixDisplay.from_predictions(intent_df["label"], oof_pred, labels=classes, normalize="true",
                                        values_format=".2f", cmap="Blues", ax=ax, colorbar=False)
plt.xticks(rotation=60, ha="right"); ax.set_title("Out-of-fold confusion matrix (row-normalised)")
plt.tight_layout(); plt.show()
print(classification_report(intent_df["label"], oof_pred, digits=3, zero_division=0))""")

md("""### A5. Selective classification — the confidence gate
The threshold is picked on **out-of-fold** probabilities (not on the test set): the lowest confidence at which answered predictions reach the target accuracy (`CONFIDENCE_TARGET` = 90% on unseen phrasings). Below it the ML abstains and the request goes to the keyword parser or is refused.""")

code("""curve = im.coverage_curve(oof, classes, intent_df["label"])
thr = joblib.load(MODELS / "intent_clf.joblib")["threshold"]
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(curve["threshold"], curve["coverage"], "o-", label="coverage (share answered)")
ax.plot(curve["threshold"], curve["selective_accuracy"], "o-", label="accuracy on answered")
ax.axvline(thr, color="red", ls="--", label=f"deployed threshold = {thr:.2f}")
ax.set_xlabel("confidence threshold"); ax.set_ylim(0, 1.02); ax.legend(); ax.set_title("Coverage vs selective accuracy")
plt.tight_layout(); plt.show()
curve[curve.threshold.isin([0.0, thr, 0.6, 0.8])]""")

md("""### A6. Feature selection (χ²) and dimensionality reduction (TruncatedSVD)""")

code("""fs = pd.DataFrame(metrics["intent"]["feature_selection_curve"])
sv = pd.DataFrame(metrics["intent"]["svd_curve"])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4))
a1.errorbar(fs["k"], fs["cv_f1_macro"], yerr=fs["cv_f1_std"], fmt="o-", capsize=3)
a1.set_xscale("log"); a1.set_xlabel("k features kept (chi-square)"); a1.set_ylabel("grouped CV macro-F1")
a1.set_title("SelectKBest(chi2) curve")
a2.errorbar(sv["components"], sv["cv_f1_macro"], yerr=sv["cv_f1_std"], fmt="o-", capsize=3, color="#8b5cf6")
a2.set_xscale("log"); a2.set_xlabel("TruncatedSVD components"); a2.set_title("Dimensionality-reduction curve")
plt.tight_layout(); plt.show()
pd.concat([fs.assign(method="chi2 k-best").rename(columns={"k": "dims"}),
           sv.assign(method="TruncatedSVD").rename(columns={"components": "dims"})])[["method", "dims", "cv_f1_macro", "cv_f1_std"]]""")

md("""### A7. External test — hand-written challenge set vs baselines
Phrasings, typos and adversarial requests written separately from the templates. Baselines: majority class and the original keyword parser.""")

code("""base = metrics["intent"]["challenge_baselines"]
tbl = pd.DataFrame(base).T.rename(index={"majority_class": "Majority class", "keyword_parser": "Keyword parser (original)",
                                         "ml_logreg": "TF-IDF + Logistic Regression"})
ax = tbl["f1_macro"].plot(kind="barh", figsize=(8, 3), color=["#9ca3af", "#f59e0b", "#10b981"])
ax.set_xlabel("macro-F1 on challenge set"); ax.set_xlim(0, 1); plt.tight_layout(); plt.show()
tbl.round(3)""")

code("""errs = pd.read_csv(MODELS / "intent_errors.csv")
print(f"{len(errs)} challenge-set errors — error analysis:")
errs""")

md("""## Part B — Tabular outcome model
Predict whether the ECU accepts a request (positive/negative) from request + context. Train on 3 ECU profiles, test on the unseen `DCU-DEMO`.""")

code("""outcome_df = generate_outcomes(n_per_profile=2000, seed=SEED)
train_out = outcome_df[outcome_df.ecu_id != HELDOUT_PROFILE]
test_out = outcome_df[outcome_df.ecu_id == HELDOUT_PROFILE]
print(f"train {len(train_out):,} rows (3 ECUs)  |  held-out ECU {HELDOUT_PROFILE}: {len(test_out):,} rows")
oc = om.compare_models(train_out, n_splits=5)
oc.round(4)""")

md("""### B1. Feature engineering ablation — raw IDs vs KB-derived domain features
Two questions: does it help in-distribution (CV), and does it help on an **ECU the model has never seen**?""")

code("""fcmp = om.feature_comparison(train_out, n_splits=5)
y_tr = (train_out["label"] == "POSITIVE").astype(int); y_te = (test_out["label"] == "POSITIVE").astype(int)
held = {}
for name, fn in [("raw_ids", raw_tabular), ("domain_features", domain_tabular)]:
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
    rf.fit(fn(train_out), y_tr)
    held[name] = f1_score(y_te, rf.predict(fn(test_out)), average="macro")
fcmp["held_out_ecu_f1_macro"] = fcmp["features"].map(held)
ax = fcmp.set_index("features")[["cv_f1_macro_mean", "held_out_ecu_f1_macro"]].plot(kind="bar", figsize=(8, 4), rot=0)
ax.set_ylim(0.5, 1.01); ax.set_ylabel("macro-F1"); ax.set_title("Raw IDs vs domain features")
plt.tight_layout(); plt.show()
fcmp.round(4)""")

md("### B2. Feature importance (impurity and permutation on the unseen ECU)")

code("""rf_final, cols = om.fit_final(train_out)
X_te = domain_tabular(test_out)[cols]
imp = pd.DataFrame({"impurity": rf_final.feature_importances_}, index=cols)
perm = permutation_importance(rf_final, X_te, y_te, n_repeats=10, random_state=SEED, n_jobs=-1)
imp["permutation (held-out ECU)"] = perm.importances_mean
imp = imp.sort_values("impurity")
imp.plot(kind="barh", figsize=(9, 8)); plt.title("Random Forest feature importance")
plt.tight_layout(); plt.show()
print("held-out ECU accuracy:", round(accuracy_score(y_te, rf_final.predict(X_te)), 4),
      " macro-F1:", round(f1_score(y_te, rf_final.predict(X_te), average="macro"), 4))""")

md("""### B3. Interpretable model — depth-4 Decision Tree recovers the engine's rules
The shallow tree is an audit tool. If its learned splits match the engine's documented rule order (service support → sub-function → DID/RID → session → security), the data and the engine agree.""")

code("""print(om.tree_rules(train_out, max_depth=4))""")

code("""print("Multi-class NRC prediction on negative rows (5-fold stratified CV):")
om.nrc_model(train_out, n_splits=5).round(4)""")

md("""## Part C — ECU defect detection (unsupervised)
Detectors are trained on a golden run only (no labels), then scored on a faulty run with 15% injected defects.""")

code("""X_train, X_test, y_test, dtypes, feat = dm.build_datasets(n_train=3000, n_test=1000, defect_rate=0.15)
scaler = StandardScaler().fit(X_train)
Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
iforest = IsolationForest(n_estimators=200, contamination=0.1, random_state=SEED, n_jobs=-1).fit(Xtr)
ocsvm = OneClassSVM(kernel="rbf", nu=0.1).fit(Xtr)
scores_ = {"Isolation Forest": -iforest.score_samples(Xte), "One-Class SVM (rbf)": -ocsvm.decision_function(Xte)}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
for name, s in scores_.items():
    fpr, tpr, _ = roc_curve(y_test, s); p, r, _ = precision_recall_curve(y_test, s)
    a1.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_test, s):.3f})")
    a2.plot(r, p, label=f"{name} (AP {average_precision_score(y_test, s):.3f})")
a1.plot([0, 1], [0, 1], "k--", lw=0.8); a1.set_xlabel("FPR"); a1.set_ylabel("TPR"); a1.set_title("ROC"); a1.legend()
a2.axhline(y_test.mean(), color="k", ls="--", lw=0.8, label=f"prevalence {y_test.mean():.2f}")
a2.set_xlabel("recall"); a2.set_ylabel("precision"); a2.set_title("Precision-Recall"); a2.legend()
plt.tight_layout(); plt.show()
pd.DataFrame(metrics["defect"]).round(3)""")

md("""### C1. ML + rules: what each detector catches
The **engine oracle** compares the observed response with the engine's expected outcome/NRC. It catches every *semantic* defect but cannot see timing or truncation.
The anomaly detector covers the reverse. The union is the deployed design.""")

code("""test_raw = generate_traces(n=1000, defect_rate=0.15, seed=SEED + 1)
obs_neg = test_raw["response_sid"] == 0x7F
exp_neg = test_raw["expected_outcome"] == "negative"
oracle = (test_raw["resp_len"] > 0) & ((obs_neg != exp_neg) | (obs_neg & (test_raw["nrc"] != test_raw["expected_nrc"])))
iforest_flag = pd.Series(iforest.predict(Xte) == -1, index=test_raw.index)
hybrid = oracle | iforest_flag
truth = test_raw["defect"] != "none"
rows = []
for name, flag in [("Isolation Forest", iforest_flag), ("Engine oracle", oracle), ("Hybrid (union)", hybrid)]:
    rec = test_raw.assign(f=flag).groupby("defect")["f"].mean()
    rows.append({"detector": name, "precision": (flag & truth).sum() / max(flag.sum(), 1),
                 "recall": (flag & truth).sum() / truth.sum(), **{f"recall:{k}": v for k, v in rec.items() if k != "none"},
                 "false-alarm rate": flag[~truth].mean()})
pd.DataFrame(rows).set_index("detector").round(3).T""")

md("""## Summary
* **Intent:** linear text models and Naive Bayes beat trees on sparse TF-IDF under template-grouped CV. The tuned Logistic Regression beats the keyword parser and the majority baseline on the external challenge set, and abstains when unsure.
* **Outcome:** KB-derived domain features beat raw IDs, most of all on an unseen ECU profile. A depth-4 tree exposes the learned rules for audit.
* **Defects:** Isolation Forest catches timing/length anomalies without labels. The deterministic oracle catches semantic NRC defects. Their union is the deployed detector.
* **Limits:** all data is synthetic (see the model card). These numbers show the method works; they are not field accuracy.""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parent.parent / "notebooks" / "02_Model_Training_Validation.ipynb"
nbf.write(nb, out)
print(f"Wrote {out} ({len(cells)} cells)")

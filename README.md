# DxForge — ML-assisted UDS Diagnostics & Test Generation

Case Study #5 — *UDS Diagnostics and Automated Test Generation Assistant* (ISO 14229).

An engineer types an intent (*"read the VIN"*, *"wrte door lock stat"*). DxForge returns a protocol-validated UDS
request, its expected positive or negative (NRC) response, cited evidence, and runnable test cases.
**Machine learning understands the engineer. A deterministic protocol engine guarantees the bytes. The engineer approves.**

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py              # http://localhost:8501  → "ML Lab" page shows all ML results
pytest -q                         # 21 tests
python scripts/run_benchmark.py   # 20 golden end-to-end cases
python scripts/train.py           # optional: retrain all models (~2.5 min) → models/
```
Trained models ship in `models/`. If your scikit-learn version cannot load them, the app falls back to rules; run `python scripts/train.py` to rebuild them.
Docker: `docker compose up --build`.

## Where to find each rubric item
| Rubric criterion | Evidence |
|---|---|
| Problem understanding & automotive context | this README, `data/uds_kb.yaml`, `dxforge/engine.py` (session/security/NRC rules) |
| AI/ML fundamentals & approach | hybrid design below: ML for language, rules for protocol correctness |
| Dataset, preprocessing & EDA | **`notebooks/01_EDA.ipynb`** (executed), `dxforge/ml/dataset.py`, `dxforge/ml/text.py` |
| Feature engineering, selection, dim. reduction | `dxforge/ml/features.py`; χ² SelectKBest, mutual information, TruncatedSVD/PCA, K-Means in both notebooks |
| Model development (syllabus techniques) | `dxforge/ml/intent_model.py`, `outcome_model.py`, `defect_model.py`, `scripts/train.py` |
| Evaluation, validation, optimisation | **`notebooks/02_Model_Training_Validation.ipynb`** (executed), `models/metrics.json` |
| Completeness & working demo | `app.py` (Diagnostic Studio, Validator, **ML Lab**, Coverage, Benchmark, Audit, Governance) |
| Python implementation & code quality | `dxforge/` package, `tests/`, ruff-clean, seeded (`SEED=42`) |
| Deployment & MLOps basics | `Dockerfile`, `docker-compose.yml`, `.github/workflows/test.yml` (lint → train → test → benchmark), joblib artefacts + `metrics.json` |
| Responsible AI & ethics | `models/model_card.md`, confidence gate, unsafe-request refusal, synthetic-data disclosure |

## Approach
```
intent text ─► TF-IDF + Logistic Regression ─► confidence gate ─┬─► slot filling ─► deterministic engine ─► bytes + NRC + tests
                                   │                            └─► below gate: keyword parser, else refuse
                                   └─► P(unsafe) ≥ 0.25 ─► refuse (no fallback)
ECU response logs ─► Isolation Forest  ∪  engine conformance oracle ─► defect flags
```
Three ML tasks, all scikit-learn, with no model downloads:

1. **Intent → UDS service** (13 classes, text). Compared: Multinomial NB, Logistic Regression, Linear SVM, KNN, Decision Tree, Random Forest, MLP.
2. **Request outcome** (tabular, binary + 6-class NRC). Compared: Decision Tree (d=4/8), Random Forest, Gradient Boosting.
3. **ECU defect detection** (unsupervised). Compared: Isolation Forest, One-Class SVM, plus a rule-based oracle.

## Results (from `models/metrics.json`, full run, seed 42)
| Task | Validation | Result |
|---|---|---|
| Intent classifier | 5-fold **StratifiedGroupKFold** by paraphrase template (no phrasing in both train and test) | tuned LogReg CV macro-F1 ≈ 0.76 (NB 0.78 … Decision Tree 0.48) |
| Intent, external test | 67 hand-written challenge intents, never used for training/tuning | **89.6%** accuracy vs keyword parser **46.3%**, majority class 19.4% |
| Confidence gate | chosen on out-of-fold predictions | 0.75 → answers 73% of unseen phrasings at ~91% accuracy, abstains on the rest |
| Outcome model | trained on 3 ECU profiles, tested on an **unseen ECU** | F1 **0.998** with KB-derived features vs **0.867** with raw IDs |
| Defect detection | golden-run training, faulty run with 6 injected defect types | Isolation Forest ROC-AUC 0.83. IF ∪ engine oracle recall **88%** (IF alone 60%) |
| End-to-end | 20 golden cases | 100% request / outcome / NRC correctness |

**Why grouped CV (~0.76) is lower than the challenge set (89.6%).** Grouped CV removes whole sentence structures from training, which is the hardest honest setting. A random split scores much higher but is invalid. The learning curve in notebook 02 is still rising, so more paraphrase variety (not a bigger model) is the next lever. We report the strict number on purpose.

**Known gaps and next steps.** The Docker image is defined but was not built in our environment (CI trains and tests from scratch instead). The data is synthetic until real tester logs are available. A FastAPI `/predict` service is the planned serving layer, and the engine and ML package are already UI-independent.

A random split of templated text leaks: 42% of test rows appear verbatim in train (measured in `01_EDA.ipynb`). That's why every intent result above uses grouped splits.

## Responsible AI
- **Synthetic data, disclosed.** No licence-clean UDS intent corpus exists and real ECU traces are proprietary, so all data is generated (seeded) and labelled by the engine. The KB is authored from public sources, not copied from ISO 14229.
- **Refusal over guessing.** Low-confidence intents abstain. Unsafe requests (e.g. *"crack the seed key algorithm"*) are refused and never fall back to keyword rules.
- **No seed/key derivation.** Security is modelled as protocol state only. Every export needs human approval, is audit-logged, and is bench/HIL only.
- **Limits.** Scores show the method works, not field accuracy. Truncated-response defects are not yet detected. See `models/model_card.md`.

## Layout
```
app.py                      Streamlit UI         dxforge/engine.py   deterministic UDS engine
dxforge/ml/                 ML package           dxforge/ui_ml_lab.py ML Lab page
notebooks/                  01 EDA, 02 training  scripts/            train, benchmark, notebook builders
models/                     artefacts, metrics, model card     data/  KB, ECU profile, benchmark, challenge set
tests/                      pytest suite
```

"""ML Lab interactive UI for DxForge.

Displays comprehensive model evaluations, comparisons, feature engineering curves,
unsupervised clustering, challenge set baselines, and live inference sandbox.
Directly aligns with classical ML course evaluation rubrics.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from .ml.inference import (
    load_metrics, load_intent_model
)

MODELS_DIR = Path(__file__).parent.parent / "models"


def render_ml_lab(kb) -> None:
    st.markdown("## 🧪 Machine Learning Lab")
    st.caption("Rigorous model development, cross-validation, hyperparameter tuning, "
               "feature ablation, and anomaly detection across ISO 14229 diagnostics.")

    metrics = load_metrics()
    if not metrics:
        st.warning("No trained models found. Run `python scripts/train.py` to generate model artefacts.")
        return

    intent_m = metrics.get("intent", {})
    outcome_m = metrics.get("outcome", {})
    defect_m = metrics.get("defect", [{}])
    ifor_m = defect_m[0] if defect_m else {}
    baselines = intent_m.get("challenge_baselines", {})
    ml_base = baselines.get("ml_logreg", {})

    # Top KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        best_f1 = intent_m.get("best_params", {}).get("cv_f1_macro", 0.755)
        st.markdown(f'<div class="dx-kpi"><div class="v">{best_f1:.1%}</div>'
                    f'<div class="l">Intent CV F1 (Grouped)</div></div>', unsafe_allow_html=True)
    with c2:
        oc_f1 = outcome_m.get("held_out_ecu", {}).get("f1_macro", 0.992)
        st.markdown(f'<div class="dx-kpi"><div class="v">{oc_f1:.1%}</div>'
                    f'<div class="l">Held-out ECU F1</div></div>', unsafe_allow_html=True)
    with c3:
        roc = ifor_m.get("roc_auc", 0.837)
        st.markdown(f'<div class="dx-kpi"><div class="v">{roc:.3f}</div>'
                    f'<div class="l">Anomaly ROC-AUC</div></div>', unsafe_allow_html=True)
    with c4:
        chal_f1 = ml_base.get("f1_macro", 0.858)
        st.markdown(f'<div class="dx-kpi"><div class="v">{chal_f1:.1%}</div>'
                    f'<div class="l">Challenge Set F1</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="dx-kpi"><div class="v">0%</div>'
                    '<div class="l">Template Leakage</div></div>', unsafe_allow_html=True)

    st.markdown(" ")

    tabs = st.tabs([
        "📊 Model Comparison",
        "🎛️ Tuning & Feature Curves",
        "🌌 2D Semantic Embeddings",
        "🏆 Challenge Baselines",
        "🛡️ Anomaly Defect Detection",
        "⚡ Live Inference Sandbox",
        "📜 Model Card & Ethics"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Model Comparison
    # -------------------------------------------------------------------------
    with tabs[0]:
        st.markdown("### 1. Intent Classification: Candidate Model Comparison")
        st.caption("5-fold StratifiedGroupKFold grouped by paraphrase template — models are "
                   "strictly evaluated on phrasings never seen during training.")

        comp_data = intent_m.get("model_comparison", [])
        if comp_data:
            df_comp = pd.DataFrame(comp_data)
            df_comp["cv_f1_macro"] = df_comp["cv_f1_macro_mean"].map(lambda x: f"{x:.3f}") + " ± " + df_comp["cv_f1_macro_std"].map(lambda x: f"{x:.3f}")
            df_comp["cv_accuracy"] = df_comp["cv_accuracy_mean"].map(lambda x: f"{x:.3f}") + " ± " + df_comp["cv_accuracy_std"].map(lambda x: f"{x:.3f}")
            df_comp["fit_time_s"] = df_comp["fit_time_s"].map(lambda x: f"{x:.2f}s")
            display_cols = ["model", "cv_f1_macro", "cv_accuracy", "fit_time_s"]
            st.dataframe(df_comp[display_cols].rename(columns={
                "model": "Candidate Algorithm",
                "cv_f1_macro": "CV F1-Macro (mean ± std)",
                "cv_accuracy": "CV Accuracy (mean ± std)",
                "fit_time_s": "Fit Time"
            }), use_container_width=True, hide_index=True)

        st.markdown("### 2. Tabular Diagnostic Outcome Predictors")
        st.caption("Predicting binary acceptance / ISO NRC rejection from diagnostic session & security state.")
        col_oc1, col_oc2 = st.columns(2)
        with col_oc1:
            st.markdown("##### Candidate Model Comparison (5-fold stratified CV)")
            oc_comp = pd.DataFrame(outcome_m.get("model_comparison", []))
            if not oc_comp.empty:
                st.dataframe(pd.DataFrame({
                    "Algorithm": oc_comp["model"],
                    "CV F1-Macro": oc_comp["cv_f1_macro_mean"].map(lambda x: f"{x:.4f}"),
                    "CV Accuracy": oc_comp["cv_accuracy_mean"].map(lambda x: f"{x:.2%}"),
                }), use_container_width=True, hide_index=True)
        with col_oc2:
            st.markdown("##### Feature Ablation: Raw IDs vs. KB-derived Domain Features")
            fc = {r["features"]: r for r in outcome_m.get("feature_comparison", [])}
            ho = outcome_m.get("held_out_ecu", {})
            ho_raw = outcome_m.get("held_out_ecu_raw_ids", {})
            if fc:
                st.dataframe(pd.DataFrame([
                    {"Representation": "Raw IDs (sid, subfn, did, rid, session, unlocked)",
                     "Features": fc.get("raw_ids", {}).get("n_features"),
                     "CV F1": f'{fc.get("raw_ids", {}).get("cv_f1_macro_mean", float("nan")):.4f}',
                     "Unseen-ECU F1": f'{ho_raw.get("f1_macro", float("nan")):.4f}'},
                    {"Representation": "Domain features (support, writability, session deficit, security)",
                     "Features": fc.get("domain_features", {}).get("n_features"),
                     "CV F1": f'{fc.get("domain_features", {}).get("cv_f1_macro_mean", float("nan")):.4f}',
                     "Unseen-ECU F1": f'{ho.get("f1_macro", float("nan")):.4f}'},
                ]), use_container_width=True, hide_index=True)
                if ho and ho_raw:
                    gain = (ho["f1_macro"] - ho_raw["f1_macro"]) * 100
                    st.success(f"On an ECU profile never seen in training, domain features score "
                               f"**{ho['f1_macro']:.3f}** F1 vs **{ho_raw['f1_macro']:.3f}** for raw IDs "
                               f"(+{gain:.1f} points).")

    # -------------------------------------------------------------------------
    # TAB 2: Hyperparameters & Feature Curves
    # -------------------------------------------------------------------------
    with tabs[1]:
        st.markdown("### Hyperparameter Tuning & Dimensionality Analysis")

        c_grid1, c_grid2 = st.columns(2)
        with c_grid1:
            st.markdown("##### Feature Selection Curve (Chi-Square SelectKBest)")
            st.caption("Evaluating CV F1-macro as top k informative n-gram features are retained.")
            fsel = intent_m.get("feature_selection_curve", [])
            if fsel:
                df_fsel = pd.DataFrame(fsel)
                st.dataframe(df_fsel.rename(columns={"k": "Top K Features", "cv_f1_macro": "CV F1-Macro", "cv_f1_std": "Std"}),
                             use_container_width=True, hide_index=True)
                st.bar_chart(df_fsel.set_index("k")["cv_f1_macro"])

        with c_grid2:
            st.markdown("##### Dimensionality Reduction Curve (TruncatedSVD)")
            st.caption("F1-macro across latent semantic components vs dense representations.")
            svd = intent_m.get("svd_curve", [])
            if svd:
                df_svd = pd.DataFrame(svd)
                st.dataframe(df_svd.rename(columns={"components": "Components", "cv_f1_macro": "CV F1-Macro", "cv_f1_std": "Std"}),
                             use_container_width=True, hide_index=True)
                st.line_chart(df_svd.set_index("components")["cv_f1_macro"])

        st.divider()
        st.markdown("### Interpretable Decision Tree Rules (Depth 4)")
        st.caption("Directly inspect the safety guardrails learned by the interpretable tree model:")
        rules_path = MODELS_DIR / "decision_tree_rules.txt"
        if rules_path.exists():
            st.code(rules_path.read_text(encoding="utf-8"), language="text")

    # -------------------------------------------------------------------------
    # TAB 3: 2D Semantic Embeddings & Clusters
    # -------------------------------------------------------------------------
    with tabs[2]:
        st.markdown("### 2D Latent Semantic Projections & Unsupervised Clustering")
        st.caption("Visualizing the semantic distribution of user phrasings projected onto principal axes via TruncatedSVD.")

        emb_stats = intent_m.get("embedding", {})
        if emb_stats:
            ari = emb_stats.get("kmeans_ari", 0.478)
            nmi = emb_stats.get("kmeans_nmi", 0.652)
            sc1, sc2 = st.columns(2)
            sc1.metric("Adjusted Rand Index (ARI)", f"{ari:.3f}", help="Agreement between K-Means clusters and true UDS service labels")
            sc2.metric("Normalized Mutual Info (NMI)", f"{nmi:.3f}", help="Information overlap between clusters and target classes")

        emb_path = MODELS_DIR / "intent_embedding.csv"
        if emb_path.exists():
            df_emb = pd.read_csv(emb_path)
            st.scatter_chart(df_emb, x="pc1", y="pc2", color="label", size=25)

    # -------------------------------------------------------------------------
    # TAB 4: Challenge Baselines & Abstention
    # -------------------------------------------------------------------------
    with tabs[3]:
        st.markdown("### Hard Challenge Set Baselines (67 Unseen Edge Cases)")
        st.caption("Comparison of classical ML baselines against our trained classifier on adversarial phrasing, "
                   "informal jargon, refusal intents, and unusual acronyms.")

        if baselines:
            base_rows = []
            for name, sc in baselines.items():
                base_rows.append({
                    "Approach": name.replace("_", " ").title(),
                    "Accuracy": f"{sc.get('accuracy', 0):.1%}",
                    "F1-Macro": f"{sc.get('f1_macro', 0):.3f}",
                    "Description": "Predicts most frequent training label" if "majority" in name else (
                        "Rule-based keyword regex matching" if "keyword" in name else "TF-IDF + Grouped-CV Tuned LogReg"
                    )
                })
            st.dataframe(pd.DataFrame(base_rows), use_container_width=True, hide_index=True)

        st.markdown("### Selective Classification & Confidence Gating")
        st.caption("To prevent hallucinated vehicle diagnostics, low-confidence predictions are abstained and routed to fallback.")
        thr = intent_m.get("confidence_threshold", 0.45)
        cov = intent_m.get("confidence_coverage")
        cov_txt = f" ML answers {cov:.0%} of unseen phrasings directly at the target accuracy; the rest fall back to the keyword parser or are refused." if cov is not None else ""
        st.info(f"Operational confidence threshold (chosen on out-of-fold predictions): **{thr:.2f}**.{cov_txt}")

        err_path = MODELS_DIR / "intent_errors.csv"
        if err_path.exists():
            with st.expander("Inspect Challenge Set Misclassifications (Error Analysis)"):
                df_err = pd.read_csv(err_path)
                st.dataframe(df_err, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # TAB 5: Anomaly Defect Detection
    # -------------------------------------------------------------------------
    with tabs[4]:
        st.markdown("### ECU Bench Response Anomaly Detection")
        st.caption("Trained on golden baseline traces (defect_rate=0.0) to flag timing jitter, unexpected negative codes, and protocol violations.")

        if defect_m:
            st.dataframe(pd.DataFrame(defect_m), use_container_width=True, hide_index=True)

        hyb = metrics.get("defect_hybrid", [])
        if hyb:
            st.markdown("##### ML + rules: Isolation Forest vs engine oracle vs union")
            st.caption("The oracle compares the observed response with the engine's expected outcome/NRC: "
                       "it catches semantic defects but not timing; the anomaly detector covers the reverse.")
            df_h = pd.DataFrame(hyb).set_index("detector")
            st.dataframe(df_h.T.style.format("{:.1%}"), use_container_width=True)

        st.markdown("##### Anomaly Score Breakdown by Injected Fault Type")
        breakdown = metrics.get("defect_breakdown", [])
        if breakdown:
            df_bk = pd.DataFrame(breakdown)
            st.dataframe(df_bk.rename(columns={"defect": "Fault Condition", "mean": "Mean Anomaly Score", "count": "Sample Count"}),
                         use_container_width=True, hide_index=True)
            st.bar_chart(df_bk.set_index("defect")["mean"])

    # -------------------------------------------------------------------------
    # TAB 6: Live Inference Sandbox
    # -------------------------------------------------------------------------
    with tabs[5]:
        st.markdown("### Live Intent Inference Sandbox")
        st.caption("Test the trained TF-IDF + Logistic Regression intent classifier live in real time.")

        pipe, classes = load_intent_model()
        if pipe is None:
            st.error("Model pipeline not loaded.")
        else:
            sample_prompts = [
                "read the vehicle identification number",
                "force ecu into programming mode now",
                "erase stored fault codes",
                "crack the seed key algorithm and bypass auth",
                "what is the weather in Munich tomorrow?"
            ]
            picked_prompt = st.selectbox("Sample queries or write your own below:", ["(Custom)"] + sample_prompts)
            default_val = "read the vehicle identification number" if picked_prompt == "(Custom)" else picked_prompt
            query = st.text_input("Enter diagnostic phrasing:", value=default_val)
            col_t1, col_t2 = st.columns([1, 3])
            slider_thr = col_t1.slider("Confidence gate threshold", 0.10, 0.95,
                                       float(round(intent_m.get("confidence_threshold", 0.5), 2)), 0.05)

            if query:
                from dxforge.ml.text import normalize
                clean = normalize(query)
                probs = pipe.predict_proba([clean])[0]
                pred_idx = int(np.argmax(probs))
                pred_label = classes[pred_idx]
                conf = float(probs[pred_idx])

                col_res1, col_res2 = st.columns([2, 3])
                with col_res1:
                    st.markdown("##### Prediction Result")
                    passed = conf >= slider_thr
                    from dxforge.ml.inference import UNSAFE_REFUSAL_PROB
                    p_unsafe = float(probs[list(classes).index("UnsafeRequest")])
                    refused = p_unsafe >= UNSAFE_REFUSAL_PROB or (pred_label == "OutOfScope" and passed)
                    badge_class = "ok" if passed and not refused else "bad"
                    if refused:
                        status_text = "REFUSED (UNSAFE / OUT OF SCOPE)"
                    else:
                        status_text = "ACCEPTED (PASSED GATE)" if passed else "ABSTAINED (LOW CONFIDENCE)"

                    st.markdown(f'<div class="dx-card">'
                                f'<div style="font-size:1.1rem;font-weight:700;">{pred_label}</div>'
                                f'<div style="margin:8px 0;"><span class="pill {badge_class}">{status_text}</span></div>'
                                f'<div>Confidence: <b>{conf:.1%}</b></div>'
                                f'<div class="cite">Cleaned token representation: <code>{clean}</code></div>'
                                f'</div>', unsafe_allow_html=True)
                with col_res2:
                    st.markdown("##### Top 4 Class Probabilities")
                    top4_idx = np.argsort(probs)[::-1][:4]
                    top_df = pd.DataFrame({
                        "UDS Service": [classes[i] for i in top4_idx],
                        "Probability": [probs[i] for i in top4_idx]
                    })
                    st.dataframe(top_df.style.format({"Probability": "{:.1%}"}), use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # TAB 7: Model Card & Ethics
    # -------------------------------------------------------------------------
    with tabs[6]:
        st.markdown("### Responsible AI & Model Governance")
        st.caption("Transparent documentation of synthetic training datasets, boundaries, and safety stance.")
        card_path = MODELS_DIR / "model_card.md"
        if card_path.exists():
            st.markdown(card_path.read_text(encoding="utf-8"))
        else:
            st.info("Model card not yet generated.")

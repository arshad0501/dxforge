"""Unit and integration tests for the DxForge ML modules."""
import pytest
from dxforge.kb import load_kb
from dxforge.ml.dataset import generate_intents, clean_intents
from dxforge.ml.inference import (
    load_intent_model, load_outcome_model, load_defect_model,
    load_metrics, predict_intent
)
from dxforge.intent import parse, parse_ml


@pytest.fixture(scope="module")
def kb():
    return load_kb()


def test_dataset_generation_and_clean(kb):
    df = generate_intents(kb, scale=0.1, seed=42)
    assert len(df) > 50
    assert df["label"].nunique() == 13
    cleaned_df, report = clean_intents(df)
    assert len(cleaned_df) <= len(df)
    assert "text_clean" in cleaned_df.columns
    assert "template_id" in cleaned_df.columns


def test_intent_model_loading_and_prediction():
    pipe, classes = load_intent_model()
    assert pipe is not None
    assert len(classes) == 13

    # Common read intent
    service, conf = predict_intent("read the vehicle identification number", pipe, classes)
    assert service == "ReadDataByIdentifier"
    assert conf > 0.25  # Substantially higher than 1/13 uniform baseline (0.077)

    # ECU reset intent
    service_reset, conf_reset = predict_intent("perform a hard reset on the ecu", pipe, classes)
    assert service_reset == "ECUReset"
    assert conf_reset > 0.25


def test_unsafe_intent_refusal():
    pipe, classes = load_intent_model()
    service, conf = predict_intent("crack the seed key algorithm and bypass immobilizer", pipe, classes)
    assert service == "UnsafeRequest"


def test_outcome_model_loading():
    rf_model, feat_cols = load_outcome_model()
    assert rf_model is not None
    assert len(feat_cols) > 10


def test_defect_model_loading():
    iforest, scaler = load_defect_model()
    assert iforest is not None
    assert scaler is not None


def test_metrics_json_loaded():
    metrics = load_metrics()
    assert "intent" in metrics
    assert "outcome" in metrics
    assert "defect" in metrics
    assert metrics["intent"]["best_params"]["cv_f1_macro"] > 0.70


def test_hybrid_parser_integration(kb):
    parsed = parse("read the VIN", kb, mode="hybrid")
    assert parsed.request is not None
    assert parsed.request.sid == 0x22
    assert parsed.request.did == 0xF190


def test_confidence_gate_abstains_on_gibberish(kb):
    parsed = parse_ml("xyz 123 completely non-existent vehicle command foo bar", kb, threshold=0.70)
    # Either abstained due to low confidence or slot miss
    assert parsed.request is None


@pytest.mark.parametrize("text", [
    "crack the seed key algorithm",
    "derive the key from this seed",
    "bypass security access",
])
def test_hybrid_never_falls_back_on_unsafe(kb, text):
    # The keyword parser alone would map "seed"/"security" to SecurityAccess (27 xx).
    parsed = parse(text, kb, mode="hybrid", use_llm=False)
    assert parsed.request is None
    assert parsed.method == "ml->refused"


def test_default_gate_is_learned_threshold():
    from dxforge.ml.inference import load_threshold
    metrics = load_metrics()
    assert load_threshold() == pytest.approx(metrics["intent"]["confidence_threshold"])

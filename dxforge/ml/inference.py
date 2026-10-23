"""Model inference and hybrid parser.

Loads trained artefacts (joblib) and exposes:
  * ``predict_intent(text)``  -- service label + confidence
  * ``hybrid_parse(text, kb)`` -- ML proposes service; slots fill parameters;
                                   deterministic engine validates and produces bytes
  * ``is_confident(proba, threshold)`` -- selective classification
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from ..engine import DiagContext, evaluate
from ..kb import KnowledgeBase
from .text import normalize
from .slots import fill_request

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# Asymmetric cost: executing an unsafe request is far worse than refusing a
# legitimate one, so refuse once P(UnsafeRequest) reaches this bar even if it
# is not the top class. 0.25 costs ~1% false refusals on out-of-fold data.
UNSAFE_REFUSAL_PROB = 0.25


def _model_path(name: str) -> Path:
    return MODELS_DIR / name


def _load(name: str) -> Optional[dict]:
    """Load a joblib bundle; None if missing or unreadable (e.g. scikit-learn
    version mismatch). The app then falls back to the deterministic parser;
    ``python scripts/train.py`` rebuilds the artefacts for the local versions."""
    p = _model_path(name)
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None


@lru_cache(maxsize=1)
def load_intent_model():
    bundle = _load("intent_clf.joblib")
    if bundle is None:
        return None, None
    return bundle["pipeline"], bundle["classes"]


@lru_cache(maxsize=1)
def load_outcome_model():
    bundle = _load("outcome_clf.joblib")
    if bundle is None:
        return None, None
    return bundle["model"], bundle["feature_cols"]


@lru_cache(maxsize=1)
def load_defect_model():
    bundle = _load("defect_iforest.joblib")
    if bundle is None:
        return None, None
    return bundle["iforest"], bundle["scaler"]


@lru_cache(maxsize=1)
def load_threshold(default: float = 0.5) -> float:
    """Confidence gate chosen on out-of-fold predictions during training."""
    bundle = _load("intent_clf.joblib")
    return float(bundle.get("threshold", default)) if bundle else default


def load_metrics() -> dict:
    p = _model_path("metrics.json")
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def predict_intent(text: str, pipeline, classes) -> tuple[str, float]:
    """Return (predicted_service, confidence)."""
    clean = normalize(text)
    proba = pipeline.predict_proba([clean])[0]
    idx = int(np.argmax(proba))
    return classes[idx], float(proba[idx])


def unsafe_probability(text: str, pipeline, classes) -> float:
    if "UnsafeRequest" not in classes:
        return 0.0
    proba = pipeline.predict_proba([normalize(text)])[0]
    return float(proba[list(classes).index("UnsafeRequest")])


def is_confident(confidence: float, threshold: float = 0.50) -> bool:
    return confidence >= threshold


def hybrid_parse(text: str, kb: KnowledgeBase,
                 pipeline=None, classes=None,
                 ctx: Optional[DiagContext] = None,
                 threshold: float = 0.50) -> dict:
    """ML proposes service + slots; engine validates and produces bytes.

    Returns a dict with keys: service, confidence, abstained, request,
    engine_outcome, request_hex, method.
    """
    ctx = ctx or DiagContext()
    result: dict = {
        "text": text, "service": None, "confidence": 0.0,
        "abstained": False, "request": None,
        "engine_outcome": None, "request_hex": None,
        "method": "keyword_fallback",
    }

    if pipeline is not None and classes is not None:
        service, conf = predict_intent(text, pipeline, classes)
        result["service"] = service
        result["confidence"] = conf
        result["method"] = "ml"

        if not is_confident(conf, threshold):
            result["abstained"] = True
            return result

        req = fill_request(service, text, kb)
    else:
        # fallback to keyword parser
        from ..intent import parse_deterministic
        parsed = parse_deterministic(text, kb)
        if parsed.request is None:
            result["abstained"] = True
            return result
        result["service"] = parsed.matched_service
        result["confidence"] = 1.0
        req = parsed.request
        result["method"] = "keyword"

    if req is None:
        result["abstained"] = True
        return result

    result["request"] = req
    eng_res = evaluate(req, ctx, kb)
    result["engine_outcome"] = eng_res.outcome
    result["request_hex"] = eng_res.request_hex
    return result

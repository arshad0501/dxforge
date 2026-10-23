"""Feature engineering.

Text (intent classification)
  * word TF-IDF 1-2 grams        -- captures verbs + objects ("read vin", "clear dtc")
  * char_wb TF-IDF 2-5 grams     -- robust to typos ("ennable", "seend") and abbreviations
  * hex normalisation (``hexid``) -- done in ``text.normalize``: the class depends on the
                                     *presence* of an identifier, not its value

Tabular (outcome / NRC prediction)
  * raw ids: sid, sub-function, did, rid, session, unlocked
  * KB-derived "domain" features: joins the request against the KB and the ECU
    support matrix (is the DID known? writable? which session does it need?) plus
    interaction features such as ``locked_and_needs_security``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from ..kb import KnowledgeBase
from .dataset import SESSION_RANK, profile_kb


def text_features(word_ngrams: tuple[int, int] = (1, 2), char_ngrams: tuple[int, int] = (2, 5),
                  use_char: bool = True, min_df: int = 2) -> FeatureUnion | TfidfVectorizer:
    """Word (+ optional char) TF-IDF union. Input is already-normalised text."""
    word = TfidfVectorizer(analyzer="word", ngram_range=word_ngrams, min_df=min_df,
                           sublinear_tf=True)
    if not use_char:
        return word
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngrams, min_df=min_df,
                           sublinear_tf=True, max_features=20000)
    return FeatureUnion([("word", word), ("char", char)])


RAW_COLUMNS = ["sid", "subfunction", "did", "rid", "session_rank", "unlocked"]


def raw_tabular(df: pd.DataFrame) -> pd.DataFrame:
    """Raw identifiers only (what a naive model would see)."""
    out = df[["sid", "subfunction", "did", "rid"]].copy()
    out["session_rank"] = df["session"].map(SESSION_RANK)
    out["unlocked"] = df["unlocked"].astype(int)
    return out


def _row_features(row: pd.Series, kb: KnowledgeBase) -> dict:
    sid, sub, did, rid = int(row.sid), int(row.subfunction), int(row.did), int(row.rid)
    svc = kb.service(sid)
    known = svc is not None
    f: dict[str, float] = {
        "svc_known": float(known),
        "svc_supported": float(known and kb.ecu_supports_service(sid)),
        "subfn_required": float(bool(known and svc.get("has_subfunction"))),
        "subfn_present": float(sub >= 0),
        "subfn_valid": 0.0,
        "has_did": float(did >= 0), "did_known": 0.0, "did_supported": 0.0,
        "is_write": float(sid == 0x2E), "did_writable": 0.0,
        "has_rid": float(rid >= 0), "rid_known": 0.0, "rid_supported": 0.0,
        "required_session_rank": 0.0, "needs_security": 0.0,
    }
    if known and svc.get("has_subfunction") and sub >= 0:
        f["subfn_valid"] = float(sub in {s["value"] for s in svc.get("subfunctions", [])})
    req_session = svc.get("required_session", "any") if known else "any"
    if did >= 0:
        d = kb.did(did)
        f["did_known"] = float(d is not None)
        f["did_supported"] = float(d is not None and kb.ecu_supports_did(did))
        if d is not None and sid == 0x2E:
            f["did_writable"] = float("write" in d.get("access", []))
            req_session = d.get("write_session", "extendedDiagnosticSession")
            f["needs_security"] = float(bool(d.get("write_secure", False)))
    if rid >= 0:
        r = kb.rid(rid)
        f["rid_known"] = float(r is not None)
        f["rid_supported"] = float(r is not None and kb.ecu_supports_rid(rid))
        if r is not None:
            req_session = r.get("session", req_session)
            f["needs_security"] = float(bool(r.get("secure", False)))
    f["required_session_rank"] = float(SESSION_RANK.get(req_session, -1) if req_session != "any" else -1)
    have = SESSION_RANK[row.session]
    f["session_rank"] = float(have)
    f["session_deficit"] = float(max(0.0, f["required_session_rank"] - have))
    f["unlocked"] = float(bool(row.unlocked))
    # interaction features
    f["locked_and_needs_security"] = float(f["needs_security"] and not row.unlocked)
    f["write_to_readonly"] = float(f["is_write"] and f["did_known"] and not f["did_writable"])
    # redundant / weak features kept on purpose so feature selection has work to do
    for s in SESSION_RANK:
        f[f"session_{s}"] = float(row.session == s)
    f["request_len_bytes"] = float(1 + (sub >= 0) + 2 * (did >= 0) + 2 * (rid >= 0))
    f["sid_value"] = float(sid)
    return f


def domain_tabular(df: pd.DataFrame) -> pd.DataFrame:
    """KB + ECU-profile joined features. Works for any profile in ``ecu_id``."""
    kbs = {pid: profile_kb(pid) for pid in df["ecu_id"].unique()}
    recs = [_row_features(r, kbs[r.ecu_id]) for r in df.itertuples(index=False)]
    return pd.DataFrame.from_records(recs, index=df.index)


# --------------------------------------------------------------------------- #
# Defect-detection features (observable from the trace; no oracle columns)
# --------------------------------------------------------------------------- #
TRACE_FEATURES = ["log_latency", "latency_z", "resp_len", "len_dev", "is_negative",
                  "no_response", "nrc", "session_rank", "unlocked"]


class TraceFeaturizer:
    """Per-service robust statistics learned from a reference (golden) run.

    ``latency_z`` is a robust z-score (median / MAD) of log-latency per service,
    ``len_dev`` is deviation from the modal response length per (service, outcome).
    """

    def fit(self, ref: pd.DataFrame) -> "TraceFeaturizer":
        lg = np.log1p(ref["latency_ms"])
        self.med_ = lg.groupby(ref["sid"]).median().to_dict()
        mad = (lg - ref["sid"].map(self.med_)).abs().groupby(ref["sid"]).median()
        self.mad_ = (mad.replace(0, 1e-3) * 1.4826).to_dict()
        neg = (ref["response_sid"] == 0x7F)
        self.len_mode_ = ref.groupby([ref["sid"], neg])["resp_len"].agg(
            lambda s: s.mode().iloc[0]).to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        lg = np.log1p(df["latency_ms"])
        med = df["sid"].map(self.med_).fillna(lg.median())
        mad = df["sid"].map(self.mad_).fillna(1.0)
        neg = df["response_sid"] == 0x7F
        mode = [self.len_mode_.get((s, n), l) for s, n, l in zip(df["sid"], neg, df["resp_len"])]
        out = pd.DataFrame({
            "log_latency": lg,
            "latency_z": (lg - med) / mad,
            "resp_len": df["resp_len"].astype(float),
            "len_dev": (df["resp_len"] - np.array(mode)).abs().astype(float),
            "is_negative": neg.astype(float),
            "no_response": (df["resp_len"] == 0).astype(float),
            "nrc": df["nrc"].astype(float),
            "session_rank": df["session"].map(SESSION_RANK).astype(float),
            "unlocked": df["unlocked"].astype(float),
        }, index=df.index)
        return out[TRACE_FEATURES]

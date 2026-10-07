"""Deterministic UDS protocol engine.

This module contains ZERO AI. Given a structured request and a diagnostic
context (session + security state + ECU profile), it deterministically:

  1. validates the request structure,
  2. checks ECU support, session and security pre-conditions,
  3. produces the exact request bytes,
  4. produces the expected positive response, OR
  5. produces the most-specific negative response (NRC) plus the full set of
     NRCs that could apply.

Every decision is recorded as an applied rule with a KB citation, so the UI can
show *why* a result was reached. This is the "the protocol engine disposes"
half of DxForge: the LLM may propose an interpretation, but correctness of the
protocol artefact is guaranteed here, not by the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .kb import KnowledgeBase

# Session rank for "at least this session" comparisons.
_SESSION_RANK = {
    "defaultSession": 0,
    "extendedDiagnosticSession": 1,
    "programmingSession": 2,
    "any": -1,
}


@dataclass
class DiagContext:
    session: str = "defaultSession"
    unlocked: bool = False  # SecurityAccess state (protocol state only)


@dataclass
class Request:
    """Structured request produced by the intent layer or entered by hand."""
    sid: int
    subfunction: Optional[int] = None
    did: Optional[int] = None
    rid: Optional[int] = None
    data: list[int] = field(default_factory=list)


@dataclass
class Rule:
    id: str          # rule id, e.g. "R-SESSION"
    citation: str    # KB node id, e.g. "SVC-2E"
    detail: str      # human-readable explanation


@dataclass
class EngineResult:
    outcome: str                       # "positive" | "negative" | "invalid"
    request_hex: str
    positive_response_hex: Optional[str] = None
    nrc: Optional[int] = None
    nrc_name: Optional[str] = None
    candidate_nrcs: list[int] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    service_name: Optional[str] = None
    confidence: str = "deterministic"  # engine results are rule-guaranteed
    notes: list[str] = field(default_factory=list)


def _hex(*vals: int) -> str:
    return " ".join(f"{v:02X}" for v in vals)


def _did_bytes(did: int) -> list[int]:
    return [(did >> 8) & 0xFF, did & 0xFF]


def build_request_hex(req: Request) -> str:
    parts: list[int] = [req.sid]
    if req.subfunction is not None:
        parts.append(req.subfunction)
    if req.did is not None:
        parts.extend(_did_bytes(req.did))
    if req.rid is not None:
        parts.extend(_did_bytes(req.rid))
    parts.extend(req.data or [])
    return _hex(*parts)


def evaluate(req: Request, ctx: DiagContext, kb: KnowledgeBase) -> EngineResult:
    """Deterministically evaluate a request in a context. Pure function."""
    rules: list[Rule] = []
    req_hex = build_request_hex(req)
    svc = kb.service(req.sid)

    # Rule 0: service exists in the KB at all
    if svc is None:
        return EngineResult(
            outcome="negative", request_hex=req_hex, nrc=0x11,
            nrc_name=kb.nrc_name(0x11), candidate_nrcs=[0x11],
            rules=[Rule("R-SVC-UNKNOWN", "nrcs", f"SID {req.sid:#04x} not a known service.")],
            service_name=None,
        )
    rules.append(Rule("R-SVC", svc["id"], f"SID {req.sid:#04x} = {svc['name']}."))
    candidate = list(svc.get("applicable_nrcs", []))

    # Rule 1: ECU must support the service
    if not kb.ecu_supports_service(req.sid):
        return EngineResult(
            outcome="negative", request_hex=req_hex, nrc=0x11,
            nrc_name=kb.nrc_name(0x11), candidate_nrcs=[0x11],
            rules=rules + [Rule("R-ECU-SVC", "ECU-BCM-DEMO",
                               f"ECU does not list {svc['name']} as supported.")],
            service_name=svc["name"],
        )

    # Rule 2: sub-function validity (for services that take one)
    if svc.get("has_subfunction"):
        valid_subs = {s["value"] for s in svc.get("subfunctions", [])}
        if req.subfunction is None:
            return EngineResult(
                outcome="negative", request_hex=req_hex, nrc=0x13,
                nrc_name=kb.nrc_name(0x13), candidate_nrcs=[0x13],
                rules=rules + [Rule("R-SUBF-MISSING", svc["id"],
                                    f"{svc['name']} requires a sub-function.")],
                service_name=svc["name"],
            )
        if req.subfunction not in valid_subs:
            return EngineResult(
                outcome="negative", request_hex=req_hex, nrc=0x12,
                nrc_name=kb.nrc_name(0x12), candidate_nrcs=[0x12],
                rules=rules + [Rule("R-SUBF", svc["id"],
                                    f"Sub-function {req.subfunction:#04x} not valid for {svc['name']}.")],
                service_name=svc["name"],
            )
        rules.append(Rule("R-SUBF-OK", svc["id"], f"Sub-function {req.subfunction:#04x} valid."))

    # Rule 3: DID existence / support (for 0x22 / 0x2E)
    target_write_session = svc.get("required_session")
    target_write_secure = False
    if req.did is not None:
        did_rec = kb.did(req.did)
        if did_rec is None or not kb.ecu_supports_did(req.did):
            return EngineResult(
                outcome="negative", request_hex=req_hex, nrc=0x31,
                nrc_name=kb.nrc_name(0x31), candidate_nrcs=[0x31],
                rules=rules + [Rule("R-DID", "nrcs",
                                    f"DID {req.did:#06x} not supported -> requestOutOfRange.")],
                service_name=svc["name"],
            )
        rules.append(Rule("R-DID-OK", did_rec["id"], f"DID {req.did:#06x} = {did_rec['name']}."))
        if req.sid == 0x2E:  # write access checks
            if "write" not in did_rec.get("access", []):
                return EngineResult(
                    outcome="negative", request_hex=req_hex, nrc=0x31,
                    nrc_name=kb.nrc_name(0x31), candidate_nrcs=[0x31],
                    rules=rules + [Rule("R-DID-RO", did_rec["id"],
                                        f"DID {req.did:#06x} is read-only.")],
                    service_name=svc["name"],
                )
            target_write_session = did_rec.get("write_session", "extendedDiagnosticSession")
            target_write_secure = bool(did_rec.get("write_secure", False))

    # Rule 4: RID existence / support (for 0x31)
    rid_secure = False
    rid_session = svc.get("required_session")
    if req.rid is not None:
        rid_rec = kb.rid(req.rid)
        if rid_rec is None or not kb.ecu_supports_rid(req.rid):
            return EngineResult(
                outcome="negative", request_hex=req_hex, nrc=0x31,
                nrc_name=kb.nrc_name(0x31), candidate_nrcs=[0x31],
                rules=rules + [Rule("R-RID", "nrcs",
                                    f"RID {req.rid:#06x} not supported -> requestOutOfRange.")],
                service_name=svc["name"],
            )
        rules.append(Rule("R-RID-OK", rid_rec["id"], f"RID {req.rid:#06x} = {rid_rec['name']}."))
        rid_secure = bool(rid_rec.get("secure", False))
        rid_session = rid_rec.get("session", rid_session)

    # Rule 5: session pre-condition
    required_session = svc.get("required_session", "any")
    if req.sid == 0x2E:
        required_session = target_write_session
    if req.sid == 0x31:
        required_session = rid_session
    if required_session not in (None, "any"):
        need = _SESSION_RANK.get(required_session, 0)
        have = _SESSION_RANK.get(ctx.session, 0)
        if have < need:
            return EngineResult(
                outcome="negative", request_hex=req_hex, nrc=0x7F,
                nrc_name=kb.nrc_name(0x7F), candidate_nrcs=[0x7F, 0x22],
                rules=rules + [Rule("R-SESSION", svc["id"],
                                    f"{svc['name']} needs {required_session}; "
                                    f"active session is {ctx.session}.")],
                service_name=svc["name"],
            )
        rules.append(Rule("R-SESSION-OK", svc["id"],
                          f"Session {ctx.session} satisfies requirement ({required_session})."))

    # Rule 6: security pre-condition
    needs_secure = target_write_secure or rid_secure
    if needs_secure and not ctx.unlocked:
        return EngineResult(
            outcome="negative", request_hex=req_hex, nrc=0x33,
            nrc_name=kb.nrc_name(0x33), candidate_nrcs=[0x33],
            rules=rules + [Rule("R-SECURITY", "SEC-CONCEPT",
                                "Protected resource requested while security locked "
                                "-> securityAccessDenied.")],
            service_name=svc["name"],
            notes=["Engine models protocol lock state only; no seed/key derivation."],
        )
    if needs_secure and ctx.unlocked:
        rules.append(Rule("R-SECURITY-OK", "SEC-CONCEPT", "Security unlocked; protected access allowed."))

    # Positive path: build expected positive response
    pos_sid = svc["positive_response_sid"]
    pr_parts: list[int] = [pos_sid]
    if svc.get("has_subfunction") and req.subfunction is not None:
        # ECUReset/SessionControl/RoutineControl echo the sub-function
        pr_parts.append(req.subfunction)
    if req.did is not None:
        pr_parts.extend(_did_bytes(req.did))
    if req.rid is not None:
        if not (svc.get("has_subfunction") and req.subfunction is not None):
            pass
        pr_parts.extend(_did_bytes(req.rid))
    positive_hex = _hex(*pr_parts) + (" <data>" if req.sid in (0x22, 0x19, 0x31) else "")

    rules.append(Rule("R-POSITIVE", svc["id"],
                      f"All pre-conditions met -> positive response SID {pos_sid:#04x}."))
    return EngineResult(
        outcome="positive", request_hex=req_hex, positive_response_hex=positive_hex,
        candidate_nrcs=candidate, rules=rules, service_name=svc["name"],
        notes=svc.get("note") and [svc["note"]] or [],
    )

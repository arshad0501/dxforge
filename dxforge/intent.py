"""Natural-language intent -> structured UDS Request.

Two paths, chosen at runtime:
  * deterministic (default): keyword + pattern matching against the KB. Needs no
    model and is fully reproducible. This is what runs in the zero-dependency,
    clone-and-run configuration.
  * LLM-assisted (optional): if a provider is configured, the model proposes a
    structured request which is then CLAMPED to the KB (unknown SIDs/DIDs are
    dropped) before being handed to the deterministic engine. The model never
    produces the final protocol bytes — the engine does.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from . import llm
from .engine import Request
from .kb import KnowledgeBase


@dataclass
class ParsedIntent:
    request: Optional[Request]
    method: str          # "deterministic" | "llm" | "llm->clamped"
    rationale: str
    matched_service: Optional[str] = None


_HEX_RE = re.compile(r"0x([0-9a-fA-F]{1,4})")
_SESSION_WORDS = {
    "extended": "extendedDiagnosticSession",
    "programming": "programmingSession",
    "default": "defaultSession",
}


def _find_did(text: str, kb: KnowledgeBase) -> Optional[int]:
    # explicit 0x.... that matches a known DID
    for m in _HEX_RE.finditer(text):
        val = int(m.group(1), 16)
        if kb.did(val) is not None:
            return val
    # by name
    low = text.lower()
    for d in kb.raw.get("dids", []):
        name = d["name"].lower()
        core = name.split("(")[0].strip()
        if core and core in low:
            return d["did"]
    if "vin" in low:
        return 0xF190
    if "serial" in low:
        return 0xF18C
    if "software" in low or "sw version" in low:
        return 0xF189
    if "active diagnostic session" in low or "active session" in low:
        return 0xF186
    # explicit 0x.... with 3-4 hex digits -> treat as a DID (engine rejects unknowns with NRC 0x31)
    for m in _HEX_RE.finditer(text):
        if len(m.group(1)) >= 3:
            return int(m.group(1), 16)
    return None


def _find_rid(text: str, kb: KnowledgeBase) -> Optional[int]:
    for m in _HEX_RE.finditer(text):
        val = int(m.group(1), 16)
        if kb.rid(val) is not None:
            return val
    low = text.lower()
    for r in kb.raw.get("rids", []):
        if r["name"].lower() in low:
            return r["rid"]
    if "self test" in low or "self-test" in low:
        return 0x0301
    if "erase" in low:
        return 0x0203
    return None


def parse_deterministic(text: str, kb: KnowledgeBase) -> ParsedIntent:
    low = text.lower()

    # ControlDTCSetting (0x85) — must precede generic DTC read/clear
    if "dtc setting" in low or ("dtc" in low and any(w in low for w in ["turn off", "turn on", "stop", "resume", " on", " off"])):
        sub = 0x02 if ("off" in low or "stop" in low) else 0x01
        return ParsedIntent(Request(sid=0x85, subfunction=sub),
                            "deterministic", "ControlDTCSetting.", "ControlDTCSetting")

    # ClearDiagnosticInformation (0x14)
    if ("clear" in low or "erase" in low) and any(w in low for w in
                                                  ["diagnostic information", "dtc", "fault", "trouble code"]):
        return ParsedIntent(Request(sid=0x14, data=[0xFF, 0xFF, 0xFF]),
                            "deterministic", "Clear diagnostic information (group all).",
                            "ClearDiagnosticInformation")

    # ReadDTCInformation (0x19)
    if any(w in low for w in ["dtc", "fault code", "trouble code"]):
        return ParsedIntent(Request(sid=0x19, subfunction=0x02),
                            "deterministic", "Read DTC by status mask.", "ReadDTCInformation")

    # DiagnosticSessionControl
    if "session" in low and any(w in low for w in ["change", "enter", "switch", "go to", "set"]):
        sub = 0x03
        for word, name in _SESSION_WORDS.items():
            if word in low:
                sub = {"defaultSession": 0x01, "programmingSession": 0x02,
                       "extendedDiagnosticSession": 0x03}[name]
        return ParsedIntent(Request(sid=0x10, subfunction=sub),
                            "deterministic", "DiagnosticSessionControl.", "DiagnosticSessionControl")

    # ECUReset
    if "reset" in low or "reboot" in low or "restart" in low:
        sub = 0x01
        if "soft" in low:
            sub = 0x03
        elif "key off" in low or "keyoff" in low:
            sub = 0x02
        return ParsedIntent(Request(sid=0x11, subfunction=sub),
                            "deterministic", "ECUReset.", "ECUReset")

    # SecurityAccess
    if "security" in low or "seed" in low or ("unlock" in low and "door" not in low):
        sub = 0x02 if "key" in low else 0x01
        return ParsedIntent(Request(sid=0x27, subfunction=sub),
                            "deterministic", "SecurityAccess (protocol state only).", "SecurityAccess")

    # TesterPresent
    if "tester present" in low or "keep alive" in low or "heartbeat" in low:
        return ParsedIntent(Request(sid=0x3E, subfunction=0x00),
                            "deterministic", "TesterPresent.", "TesterPresent")

    # CommunicationControl
    if "communication" in low and any(w in low for w in ["disable", "enable", "silence"]):
        sub = 0x03 if "disable" in low or "silence" in low else 0x00
        return ParsedIntent(Request(sid=0x28, subfunction=sub, data=[0x01]),
                            "deterministic", "CommunicationControl.", "CommunicationControl")

    # RoutineControl
    if "routine" in low or "self test" in low or "self-test" in low or "erase memory" in low:
        rid = _find_rid(text, kb) or 0x0301
        sub = 0x02 if "stop" in low else (0x03 if "result" in low else 0x01)
        return ParsedIntent(Request(sid=0x31, subfunction=sub, rid=rid),
                            "deterministic", "RoutineControl.", "RoutineControl")

    # WriteDataByIdentifier
    if any(w in low for w in ["write", "set", "change", "update", "configure"]):
        did = _find_did(text, kb)
        if did is not None:
            return ParsedIntent(Request(sid=0x2E, did=did),
                                "deterministic", "WriteDataByIdentifier (data supplied at test time).",
                                "WriteDataByIdentifier")

    # ReadDataByIdentifier (default read path)
    if any(w in low for w in ["read", "get", "show", "retrieve", "value", "what is"]):
        did = _find_did(text, kb)
        if did is not None:
            return ParsedIntent(Request(sid=0x22, did=did),
                                "deterministic", "ReadDataByIdentifier.", "ReadDataByIdentifier")

    # last resort: any explicit DID -> read it
    did = _find_did(text, kb)
    if did is not None:
        return ParsedIntent(Request(sid=0x22, did=did),
                            "deterministic", "ReadDataByIdentifier (DID detected).",
                            "ReadDataByIdentifier")

    return ParsedIntent(None, "deterministic",
                        "Could not map the request to a known UDS service with confidence.")


def _clamp_llm(obj: dict, kb: KnowledgeBase) -> Optional[Request]:
    """Keep only KB-valid fields from an LLM proposal."""
    try:
        sid = int(obj.get("sid"))
    except (TypeError, ValueError):
        return None
    if kb.service(sid) is None:
        return None
    sub = obj.get("subfunction")
    did = obj.get("did")
    rid = obj.get("rid")
    def _asint(v):
        if v is None:
            return None
        return int(v, 16) if isinstance(v, str) and v.lower().startswith("0x") else int(v)
    try:
        sub = _asint(sub); did = _asint(did); rid = _asint(rid)
    except (TypeError, ValueError):
        sub = did = rid = None
    if did is not None and kb.did(did) is None:
        did = None
    if rid is not None and kb.rid(rid) is None:
        rid = None
    return Request(sid=sid, subfunction=sub, did=did, rid=rid, data=obj.get("data", []) or [])


_LLM_SYS = (
    "You translate an automotive diagnostic request into a UDS service call. "
    "Respond ONLY as JSON: {\"sid\": <int>, \"subfunction\": <int|null>, "
    "\"did\": <int|null>, \"rid\": <int|null>}. Use decimal integers. "
    "Do not invent identifiers; if unsure leave the field null."
)


def parse(text: str, kb: KnowledgeBase, use_llm: bool = True) -> ParsedIntent:
    """Parse intent. Deterministic first; LLM only to rescue a deterministic miss."""
    det = parse_deterministic(text, kb)
    if det.request is not None:
        return det
    if use_llm and llm.is_enabled():
        obj = llm.complete_json(_LLM_SYS, text)
        if obj:
            clamped = _clamp_llm(obj, kb)
            if clamped is not None:
                svc = kb.service(clamped.sid)
                return ParsedIntent(clamped, "llm->clamped",
                                    "LLM proposal, clamped to KB-valid identifiers.",
                                    svc["name"] if svc else None)
    return det

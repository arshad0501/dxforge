"""Slot filling for the hybrid parser.

The ML classifier decides *which service*; these rules fill the protocol
parameters (sub-function, DID, RID) from the same text. Name matching is fuzzy
(difflib) so typos like "vehcle identification" still resolve. Anything that
cannot be grounded in the KB stays ``None`` and the engine rejects it.
"""
from __future__ import annotations

import difflib
from typing import Optional

from ..engine import Request
from ..kb import KnowledgeBase
from .dataset import (COMM_OFF_WORDS, DID_SURFACES, DTC_OFF_WORDS, RID_SURFACES,
                      ROUTINE_STOP_WORDS, SERVICE_SID)
from .text import extract_hex_ids, normalize


def _fuzzy_lookup(text: str, surfaces: dict[int, list[str]], cutoff: float = 0.82) -> Optional[int]:
    norm = normalize(text)
    words = norm.split()
    best: tuple[float, Optional[int]] = (0.0, None)
    for ident, names in surfaces.items():
        for name in names:
            target = normalize(name)
            if target in norm:
                return ident
            n = len(target.split())
            for i in range(max(1, len(words) - n + 1)):
                window = " ".join(words[i:i + n])
                score = difflib.SequenceMatcher(None, window, target).ratio()
                if score > best[0]:
                    best = (score, ident)
    return best[1] if best[0] >= cutoff else None


def find_did(text: str, kb: KnowledgeBase) -> Optional[int]:
    ids = extract_hex_ids(text)
    for v in ids:
        if kb.did(v) is not None:
            return v
    named = _fuzzy_lookup(text, DID_SURFACES)
    if named is not None:
        return named
    big = [v for v in ids if v > 0xFF]
    return big[0] if big else None  # unknown id -> engine answers NRC 0x31


def find_rid(text: str, kb: KnowledgeBase) -> Optional[int]:
    ids = extract_hex_ids(text)
    for v in ids:
        if kb.rid(v) is not None:
            return v
    named = _fuzzy_lookup(text, RID_SURFACES)
    if named is not None:
        return named
    return ids[0] if ids else None


def _has(norm: str, *words: str) -> bool:
    return any(w in norm for w in words)


def fill_request(service: str, text: str, kb: KnowledgeBase) -> Optional[Request]:
    """Build a structured request for ``service`` from ``text``."""
    if service not in SERVICE_SID:
        return None
    sid = SERVICE_SID[service]
    n = normalize(text)
    if service == "ReadDataByIdentifier":
        did = find_did(text, kb)
        return Request(sid=sid, did=did) if did is not None else None
    if service == "WriteDataByIdentifier":
        did = find_did(text, kb)
        return Request(sid=sid, did=did) if did is not None else None
    if service == "DiagnosticSessionControl":
        sub = 0x03
        if _has(n, "programming", "flash", "bootloader"):
            sub = 0x02
        elif _has(n, "default", "normal", "standard"):
            sub = 0x01
        return Request(sid=sid, subfunction=sub)
    if service == "ECUReset":
        sub = 0x03 if _has(n, "soft", "warm") else 0x02 if _has(n, "key off", "ignition") else 0x01
        return Request(sid=sid, subfunction=sub)
    if service == "SecurityAccess":
        sub = 0x02 if ("key" in n and "seed key" not in n) else 0x01
        return Request(sid=sid, subfunction=sub)
    if service == "RoutineControl":
        rid = find_rid(text, kb)
        if rid is None:
            return None
        sub = 0x02 if _has(n, *ROUTINE_STOP_WORDS) else 0x03 if "result" in n else 0x01
        return Request(sid=sid, subfunction=sub, rid=rid)
    if service == "ReadDTCInformation":
        return Request(sid=sid, subfunction=0x0A if "supported" in n else 0x02)
    if service == "ClearDiagnosticInformation":
        return Request(sid=sid, data=[0xFF, 0xFF, 0xFF])
    if service == "TesterPresent":
        return Request(sid=sid, subfunction=0x00)
    if service == "CommunicationControl":
        off = _has(n, *COMM_OFF_WORDS)
        return Request(sid=sid, subfunction=0x03 if off else 0x00, data=[0x01])
    if service == "ControlDTCSetting":
        off = _has(n, *DTC_OFF_WORDS)
        return Request(sid=sid, subfunction=0x02 if off else 0x01)
    return None

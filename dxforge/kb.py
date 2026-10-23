"""Knowledge-base loader and lookup helpers.

Loads the authored UDS reference KB and the demo ECU profile from YAML and
exposes typed lookups by SID / DID / RID / NRC. Every record keeps its `id`
so the rest of the app can cite it (e.g. "SVC-22", "DID-F190", "NRC-31").
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import yaml

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


class KnowledgeBase:
    def __init__(self, kb: dict, ecu: dict):
        self.raw = kb
        self.ecu = ecu.get("ecu", {})
        self.services = {s["sid"]: s for s in kb.get("services", [])}
        self.dids = {d["did"]: d for d in kb.get("dids", [])}
        self.rids = {r["rid"]: r for r in kb.get("rids", [])}
        self.nrcs = {n["code"]: n for n in kb.get("nrcs", [])}
        self.sessions = {s["name"]: s for s in kb.get("sessions", [])}
        self.security = kb.get("security", {})

    # --- lookups -------------------------------------------------------------
    def service(self, sid: int) -> Optional[dict]:
        return self.services.get(sid)

    def service_by_name(self, name: str) -> Optional[dict]:
        for s in self.services.values():
            if s["name"].lower() == name.lower():
                return s
        return None

    def did(self, did: int) -> Optional[dict]:
        return self.dids.get(did)

    def rid(self, rid: int) -> Optional[dict]:
        return self.rids.get(rid)

    def nrc(self, code: int) -> dict:
        return self.nrcs.get(code, {"id": f"NRC-{code:02X}", "code": code, "name": "unknown"})

    def nrc_name(self, code: int) -> str:
        return self.nrc(code)["name"]

    def ecu_supports_service(self, sid: int) -> bool:
        return sid in self.ecu.get("supported_services", [])

    def ecu_supports_did(self, did: int) -> bool:
        return did in self.ecu.get("supported_dids", [])

    def ecu_supports_rid(self, rid: int) -> bool:
        return rid in self.ecu.get("supported_rids", [])

    def all_text_nodes(self) -> list[dict]:
        """Flatten the KB into citable text nodes for the retrieval index."""
        nodes: list[dict] = []
        for s in self.raw.get("services", []):
            subs = ", ".join(f"{x['value']:#04x} {x['name']}" for x in s.get("subfunctions", []))
            keywords = " ".join(s.get("intent_keywords", []))
            text = (
                f"Service {s['sid']:#04x} {s['name']}. Keywords: {keywords}. "
                f"Request format {s['request_format']}. "
                f"Positive response SID {s['positive_response_sid']:#04x}. "
                f"Required session {s.get('required_session')}. "
                f"Requires security {s.get('requires_security')}. "
                f"Sub-functions: {subs or 'none'}. "
                f"Applicable NRCs: {', '.join(f'{c:#04x}' for c in s.get('applicable_nrcs', []))}."
            )
            nodes.append({"id": s["id"], "kind": "service", "title": s["name"], "text": text})
        for d in self.raw.get("dids", []):
            nodes.append({
                "id": d["id"], "kind": "did", "title": d["name"],
                "text": f"DID {d['did']:#06x} {d['name']}, length {d['length']} bytes, access {d['access']}.",
            })
        for r in self.raw.get("rids", []):
            nodes.append({
                "id": r["id"], "kind": "rid", "title": r["name"],
                "text": f"Routine {r['rid']:#06x} {r['name']}, session {r['session']}, secure {r['secure']}.",
            })
        for n in self.raw.get("nrcs", []):
            nodes.append({
                "id": n["id"], "kind": "nrc", "title": n["name"],
                "text": f"NRC {n['code']:#04x} {n['name']}.",
            })
        return nodes


@lru_cache(maxsize=1)
def load_kb(kb_path: str | None = None, ecu_path: str | None = None) -> KnowledgeBase:
    kb_path = kb_path or os.path.join(_DATA_DIR, "uds_kb.yaml")
    ecu_path = ecu_path or os.path.join(_DATA_DIR, "ecu_profile_bcm.yaml")
    with open(kb_path, "r", encoding="utf-8") as f:
        kb = yaml.safe_load(f)
    with open(ecu_path, "r", encoding="utf-8") as f:
        ecu = yaml.safe_load(f)
    return KnowledgeBase(kb, ecu)

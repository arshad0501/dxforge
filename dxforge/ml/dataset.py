"""Synthetic, engine-labelled datasets for the DxForge ML layer.

SYNTHETIC DATA DISCLOSURE
-------------------------
No public, licence-clean corpus of UDS diagnostic intents exists, and real OEM
ECU traces are proprietary. Every row here is therefore generated:

  * intents   -- paraphrase templates x slot values x realistic noise
                 (typos, abbreviations, casing, filler, hex notations).
                 Ground-truth request is known from the template; outcome/NRC
                 labels come from running the deterministic engine.
  * outcomes  -- structured requests x diagnostic context x 4 ECU profiles,
                 labelled by the engine (one profile is held out entirely).
  * traces    -- simulated ECU response logs (latency, length, NRC) from a
                 golden ECU, plus a faulty-firmware run with injected defects.

Everything is seeded (``SEED``) and regenerable with ``scripts/make_dataset.py``.
Template ids are kept so evaluation can split by paraphrase group, which avoids
the leakage a random split over templated data would cause.
"""
from __future__ import annotations

import copy
import math
import random
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..engine import DiagContext, Request, evaluate
from ..kb import KnowledgeBase, load_kb
from . import SEED
from .text import add_noise, hex_surface

SESSIONS = ["defaultSession", "extendedDiagnosticSession", "programmingSession"]
SESSION_RANK = {"defaultSession": 0, "extendedDiagnosticSession": 1, "programmingSession": 2}

# --------------------------------------------------------------------------- #
# Slot vocabularies
# --------------------------------------------------------------------------- #
DID_SURFACES: dict[int, list[str]] = {
    0xF190: ["VIN", "vehicle identification number", "vin number", "chassis number"],
    0xF18C: ["ECU serial number", "serial number", "ecu serial", "module serial number"],
    0xF187: ["spare part number", "part number", "spare part no"],
    0xF189: ["software version", "software version number", "firmware version", "sw version"],
    0xF186: ["active diagnostic session", "current diagnostic session", "active session"],
    0x0105: ["interior temperature", "cabin temperature", "inside temperature"],
    0x4101: ["door lock state", "door lock status", "lock state of the doors"],
    0x4102: ["interior light level", "cabin light level", "dome light brightness"],
    0x4200: ["calibration set", "calibration data", "calibration dataset"],
}
WRITABLE_DIDS = [0x4101, 0x4102, 0x4200]
READONLY_DIDS = [0xF190, 0xF18C, 0xF187, 0xF189, 0xF186, 0x0105]
UNKNOWN_DIDS = [0x9999, 0x1234, 0xA0B0, 0x2201, 0xF1A0, 0x0200, 0x7777, 0xBEEF]

RID_SURFACES: dict[int, list[str]] = {
    0x0301: ["self test", "self-test", "built-in self test", "selftest"],
    0x0203: ["erase memory", "memory erase", "flash erase"],
    0x0400: ["check programming dependencies", "programming dependency check",
             "dependency check"],
}
UNKNOWN_RIDS = [0x0999, 0xFF00, 0x0500]

SESSION_SURFACES = {
    0x03: ["extended", "extended diagnostic", "ext diag", "extended diag"],
    0x02: ["programming", "flash", "bootloader", "reprogramming"],
    0x01: ["default", "normal", "standard"],
}
RESET_SURFACES = {0x01: ["hard", "full", "power-on"], 0x03: ["soft", "warm"],
                  0x02: ["key off on", "key-off", "ignition cycle"]}

# --------------------------------------------------------------------------- #
# Paraphrase templates. Each (label, index) pair is a paraphrase *group* (a
# sentence frame). ``[a|b|c]`` picks one alternative per sample, so verbs and
# objects are shared across frames while the frame itself can be held out.
# Placeholders: {did} {rid} {ses} {rst} {val}
# --------------------------------------------------------------------------- #
TEMPLATES: dict[str, list[str]] = {
    "ReadDataByIdentifier": [
        "[read|get|show|retrieve|fetch|display|query|pull|report|dump] the {did}",
        "[read|get|fetch|retrieve|query] {did}",
        "can you [read|get|show me|fetch|pull|check] the {did}",
        "[read|retrieve|fetch|get|pull] the {did} from the [ecu|module|controller|bcm]",
        "i need [the|to read the|to see the] {did}",
        "what is the [current |stored |]{did}",
        "[read|query|request] data identifier {did}",
        "[show|display|give] me the {did} [value|reading|]",
        "[check|look up|report|output] the [current |]{did}",
        "[read out|dump|pull out|print] the {did} [please|now|]",
        "[tell me|let me know] the {did}",
        "[get|read] the value of [the |]{did}",
    ],
    "WriteDataByIdentifier": [
        "[write|set|change|update|configure|modify] the {did} to {val}",
        "[write|set|update|program|overwrite] {did}",
        "[write|set|configure] {did} = {val}",
        "[overwrite|update|replace] the {did} with {val}",
        "can you [write|change|modify|adjust|set] the {did}",
        "[program|write|store|flash] a new {did}",
        "[write|set] data identifier {did} [to|value] {val}",
        "i want to [change|modify|adjust|update|write] the {did}",
        "[store|push|put|load] {val} [into|to] the {did}",
        "[adjust|alter|tweak|change] the {did} [setting|value|]",
    ],
    "DiagnosticSessionControl": [
        "[change|switch|go|move|transition] to [the |]{ses} session",
        "[enter|start|open|begin|request|activate] [the |a |]{ses} session",
        "[switch|change|put the ecu] into {ses} mode",
        "[go|switch|move] to the {ses} diagnostic session",
        "[activate|enable|start] {ses} diagnostics",
        "put the [module|ecu|controller] in {ses} session",
        "i [want|need] [a |the |]{ses} session",
        "[request|start] diagnostic session {ses}",
        "[open|establish] a {ses} diagnostic session [with the ecu|now|]",
        "session control [to|->] {ses}",
    ],
    "ECUReset": [
        "[reset|reboot|restart] the [ecu|module|controller|bcm]",
        "[do|perform|issue|trigger] a {rst} reset",
        "[perform|execute|run] {rst} reset on the [bcm|ecu|module]",
        "[power cycle|cycle|reboot] the [ecu|module]",
        "[trigger|issue|send] an ecu reset",
        "{rst} reset [please|now|]",
        "[reset|restart] [the ecu |]with a {rst} reset",
        "ecu reset {rst}",
        "[can you|please] [reset|reboot|restart] it",
        "[restart|reboot] [the ecu|the module|it] [now|immediately|]",
    ],
    "SecurityAccess": [
        "[unlock|open|start] security access",
        "request [a |the |][security |]seed",
        "[send|submit|transmit] the [security |]key",
        "[get|request|ask for] [the |a |]seed for level [1|one]",
        "[begin|start|initiate] [the |]security access [sequence|procedure|]",
        "[send|provide] key to unlock the [ecu|module]",
        "open security level [1|one]",
        "[perform|do|run] [the |a |]seed key handshake",
        "[authenticate|log in] [with|to] the ecu [using security access|]",
        "[unlock|elevate] [the ecu|access] [via|with] security access",
    ],
    "RoutineControl": [
        "[start|run|execute|launch|begin|trigger] the {rid} routine",
        "[run|execute|launch] {rid}",
        "[execute|start|trigger] routine {rid}",
        "[stop|abort|halt|cancel] the {rid} [routine|]",
        "[get|fetch|read|request] [the |]results of {rid}",
        "[kick off|fire up|initiate] the {rid}",
        "request routine results for {rid}",
        "routine control [start|stop] {rid}",
        "[can you|please] [run|start] {rid} [now|on the bench|]",
        "[abort|stop] routine {rid}",
    ],
    "ReadDTCInformation": [
        "[read|get|show|list|display] [the |all |]dtcs",
        "[show|list|read|display] [the |stored |active |]fault codes",
        "[get|read|retrieve] [the |]diagnostic trouble codes",
        "[list|show|report] [stored|active|current] faults",
        "what [errors|faults|codes] are [stored|logged|present]",
        "read dtc by status mask",
        "[check|scan] for [trouble codes|faults|dtcs]",
        "report supported dtcs",
        "[any|are there any] faults [logged|stored|present]",
        "[dump|pull|read] [the |][fault|error] memory",
        "list [active|current] malfunctions",
    ],
    "ClearDiagnosticInformation": [
        "[clear|erase|delete|remove|wipe] [all |the |]dtcs",
        "[clear|erase|delete|reset] [all |the |]fault codes",
        "[clear|erase] diagnostic information",
        "[wipe|clear|flush|empty] [the |]fault memory",
        "[delete|remove|clear] stored trouble codes",
        "reset the [fault|error] memory",
        "[clear|erase|wipe] [the |]error memory",
        "clear diagnostic information for all groups",
        "[flush|purge|clear] [all |]fault codes [now|please|]",
        "get rid of [all |the |]dtcs",
    ],
    "TesterPresent": [
        "send tester present",
        "keep the [session|connection|link] alive",
        "send [a |]heartbeat to [the |]ecu",
        "keep alive [message|ping|]",
        "stop the session from timing out",
        "ping the ecu to [stay in session|keep the session open]",
        "maintain [the |]diagnostic session",
        "keep the [connection|session] active",
        "prevent [the |]s3 timeout",
        "tester present [message|request|ping] [please|now|]",
    ],
    "CommunicationControl": [
        "[disable|stop|block] [normal |]communication",
        "[enable|resume|restore] rx and tx",
        "[silence|mute|quiet] the [bus|can bus|network]",
        "turn off normal communication",
        "[enable|resume|restore|re-enable] [normal |]communication",
        "stop [normal |application |]messages",
        "[mute|silence] ecu transmissions",
        "disable rx and tx for normal messages",
        "[switch|turn] [off|on] [normal |]can tx",
    ],
    "ControlDTCSetting": [
        "turn [off|on] dtc setting",
        "[stop|pause|suspend|halt] dtc [logging|recording|storage|setting]",
        "[disable|enable] dtc [recording|setting|logging]",
        "resume dtc [setting|logging|recording]",
        "freeze dtc storage",
        "suspend fault logging",
        "control dtc setting [off|on]",
        "[pause|stop] dtc detection",
        "dtc setting [off|on] [please|now|]",
    ],
    "OutOfScope": [
        "what is the weather [today|tomorrow|like]", "open the sunroof", "play [some|my] music",
        "book a service appointment", "tell me a joke", "how far is the next charging station",
        "translate this [to german|into french]", "order a pizza",
        "who won the [match|game] yesterday", "set an alarm for 7 am", "summarise this email",
        "what time is it", "navigate to [home|the office]", "call my [manager|colleague]",
    ],
    "UnsafeRequest": [
        "crack the seed key algorithm", "bypass [security access|the immobilizer]",
        "brute force the security key", "disable the immobilizer on a customer car",
        "derive the key from this seed", "flash firmware on a moving vehicle",
        "turn off airbag diagnostics on the road", "hack the ecu without authorisation",
        "[reverse engineer|leak] the oem key algorithm",
        "unlock [the ecu|security] without [authorisation|the oem key]",
    ],
}
_ALT = re.compile(r"\[([^\]]*)\]")
COMM_OFF_WORDS = ("disable", "silence", "turn off", "switch off", "stop", "mute", "block", "quiet")
DTC_OFF_WORDS = ("off", "stop", "disable", "freeze", "suspend", "pause", "halt")
ROUTINE_STOP_WORDS = ("stop", "abort", "halt", "cancel")


def expand(template: str, rng: random.Random) -> str:
    """Resolve every ``[a|b|c]`` alternation with one random choice."""
    s = _ALT.sub(lambda m: rng.choice(m.group(1).split("|")), template)
    return re.sub(r"\s+", " ", s).strip()


# Relative class frequencies (deliberately imbalanced, like real tester logs:
# reads dominate, rare services are rare).
CLASS_WEIGHTS: dict[str, int] = {
    "ReadDataByIdentifier": 900, "WriteDataByIdentifier": 560, "DiagnosticSessionControl": 440,
    "RoutineControl": 420, "ReadDTCInformation": 380, "ECUReset": 280,
    "ClearDiagnosticInformation": 260, "SecurityAccess": 260, "TesterPresent": 180,
    "CommunicationControl": 170, "ControlDTCSetting": 170, "OutOfScope": 280,
    "UnsafeRequest": 110,
}
REFUSAL_LABELS = {"OutOfScope", "UnsafeRequest"}
SERVICE_SID = {
    "DiagnosticSessionControl": 0x10, "ECUReset": 0x11, "ReadDataByIdentifier": 0x22,
    "WriteDataByIdentifier": 0x2E, "SecurityAccess": 0x27, "RoutineControl": 0x31,
    "ReadDTCInformation": 0x19, "ClearDiagnosticInformation": 0x14, "TesterPresent": 0x3E,
    "CommunicationControl": 0x28, "ControlDTCSetting": 0x85,
}


@dataclass
class Filled:
    text: str
    request: Optional[Request]


def _did_mention(did: int, rng: random.Random) -> str:
    if did in DID_SURFACES and rng.random() < 0.72:
        return rng.choice(DID_SURFACES[did])
    return rng.choice(["DID ", "", "identifier ", "data id "]) + hex_surface(did, rng)


def _rid_mention(rid: int, rng: random.Random) -> str:
    if rid in RID_SURFACES and rng.random() < 0.75:
        return rng.choice(RID_SURFACES[rid])
    return rng.choice(["routine ", "rid ", ""]) + hex_surface(rid, rng)


def _fill(label: str, template: str, rng: random.Random) -> Filled:
    """Expand alternations, fill slots, and build the ground-truth request."""
    t = expand(template, rng)
    if label in REFUSAL_LABELS:
        return Filled(t, None)
    sid = SERVICE_SID[label]
    if label == "ReadDataByIdentifier":
        did = rng.choice(READONLY_DIDS + WRITABLE_DIDS if rng.random() < 0.88 else UNKNOWN_DIDS)
        return Filled(t.format(did=_did_mention(did, rng)), Request(sid=sid, did=did))
    if label == "WriteDataByIdentifier":
        r = rng.random()
        did = rng.choice(WRITABLE_DIDS if r < 0.75 else READONLY_DIDS if r < 0.92 else UNKNOWN_DIDS)
        val = rng.choice(["1", "0", "locked", "unlocked", "50", "0x01", "max", "off"])
        return Filled(t.format(did=_did_mention(did, rng), val=val), Request(sid=sid, did=did))
    if label == "DiagnosticSessionControl":
        sub = rng.choice([0x03, 0x03, 0x02, 0x01])
        return Filled(t.format(ses=rng.choice(SESSION_SURFACES[sub])), Request(sid=sid, subfunction=sub))
    if label == "ECUReset":
        if "{rst}" in t:
            sub = rng.choice([0x01, 0x03, 0x02])
            return Filled(t.format(rst=rng.choice(RESET_SURFACES[sub])), Request(sid=sid, subfunction=sub))
        return Filled(t, Request(sid=sid, subfunction=0x01))
    if label == "SecurityAccess":
        sub = 0x02 if "key" in t and "seed key" not in t else 0x01
        return Filled(t, Request(sid=sid, subfunction=sub))
    if label == "RoutineControl":
        rid = rng.choice(list(RID_SURFACES)) if rng.random() < 0.9 else rng.choice(UNKNOWN_RIDS)
        sub = 0x02 if any(w in t for w in ROUTINE_STOP_WORDS) else 0x03 if "result" in t else 0x01
        return Filled(t.format(rid=_rid_mention(rid, rng)), Request(sid=sid, subfunction=sub, rid=rid))
    if label == "ReadDTCInformation":
        sub = 0x0A if "supported" in t else 0x02
        return Filled(t, Request(sid=sid, subfunction=sub))
    if label == "ClearDiagnosticInformation":
        return Filled(t, Request(sid=sid, data=[0xFF, 0xFF, 0xFF]))
    if label == "TesterPresent":
        return Filled(t, Request(sid=sid, subfunction=0x00))
    if label == "CommunicationControl":
        off = any(w in t for w in COMM_OFF_WORDS)
        return Filled(t, Request(sid=sid, subfunction=0x03 if off else 0x00, data=[0x01]))
    if label == "ControlDTCSetting":
        off = any(w in t for w in DTC_OFF_WORDS)
        return Filled(t, Request(sid=sid, subfunction=0x02 if off else 0x01))
    raise ValueError(label)


def _sample_context(rng: random.Random) -> DiagContext:
    session = rng.choices(SESSIONS, weights=[0.5, 0.35, 0.15])[0]
    return DiagContext(session=session, unlocked=rng.random() < 0.3)


def generate_intents(kb: KnowledgeBase | None = None, seed: int = SEED,
                     scale: float = 1.0, inject_quality_issues: bool = True) -> pd.DataFrame:
    """Generate the raw (uncleaned) intent dataset.

    ``inject_quality_issues`` adds the defects a real export would have
    (empty rows, duplicates, stray whitespace) so the EDA has real cleaning to do.
    """
    kb = kb or load_kb()
    rng = random.Random(seed)
    rows: list[dict] = []
    for label, n in CLASS_WEIGHTS.items():
        templates = TEMPLATES[label]
        n = max(len(templates), int(n * scale))
        for i in range(n):
            t_idx = i % len(templates)
            filled = _fill(label, templates[t_idx], rng)
            text, flags = add_noise(filled.text, rng)
            ctx = _sample_context(rng)
            row = {
                "text": text, "label": label, "template_id": f"{label}#{t_idx:02d}",
                "session": ctx.session, "unlocked": ctx.unlocked, **{f"noise_{k}": v for k, v in flags.items()},
                "sid": None, "request_hex": None, "outcome": "refused", "nrc": None,
            }
            if filled.request is not None:
                res = evaluate(filled.request, ctx, kb)
                row.update(sid=filled.request.sid, request_hex=res.request_hex,
                           outcome=res.outcome, nrc=res.nrc)
            rows.append(row)
    df = pd.DataFrame(rows)
    if inject_quality_issues:
        df = _inject_quality_issues(df, rng)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _inject_quality_issues(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    empty_idx = rng.sample(range(n), k=int(0.015 * n))
    df.loc[empty_idx[: len(empty_idx) // 2], "text"] = None
    df.loc[empty_idx[len(empty_idx) // 2:], "text"] = "   "
    ws_idx = rng.sample(range(n), k=int(0.03 * n))
    df.loc[ws_idx, "text"] = df.loc[ws_idx, "text"].map(
        lambda s: f"  {s.replace(' ', '  ', 1)}  " if isinstance(s, str) else s)
    dup = df.sample(n=int(0.03 * n), random_state=7)
    return pd.concat([df, dup], ignore_index=True)


def clean_intents(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Cleaning pipeline. Returns (clean_df, report) — the report feeds the EDA."""
    from .text import normalize

    report = {"rows_in": len(raw)}
    df = raw.copy()
    df["text"] = df["text"].astype("string").str.strip()
    empty = df["text"].isna() | (df["text"].str.len() == 0)
    report["dropped_empty"] = int(empty.sum())
    df = df[~empty]
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)
    before = len(df)
    df = df.drop_duplicates(subset=["text", "label", "session", "unlocked"])
    report["dropped_duplicates"] = before - len(df)
    df["text_clean"] = df["text"].map(normalize)
    df = df[df["text_clean"].str.len() > 0]
    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report


def group_split(df: pd.DataFrame, test_frac: float = 0.2,
                seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole paraphrase templates (per class) for the test set.

    A random row split would put near-identical paraphrases on both sides and
    inflate accuracy to ~100%. Here every test template is unseen in training.
    Exact cleaned-text overlaps are then removed from test as a second guard.
    """
    rng = random.Random(seed)
    test_groups: set[str] = set()
    for label, g in df.groupby("label"):
        groups = sorted(g["template_id"].unique())
        k = max(1, round(len(groups) * test_frac))
        test_groups.update(rng.sample(groups, k))
    is_test = df["template_id"].isin(test_groups)
    train, test = df[~is_test], df[is_test]
    test = test[~test["text_clean"].isin(set(train["text_clean"]))]
    return train.reset_index(drop=True), test.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Outcome dataset (structured request + context -> engine outcome / NRC)
# --------------------------------------------------------------------------- #
ECU_PROFILES: dict[str, dict] = {
    "BCM-DEMO": {},  # filled from data/ecu_profile_bcm.yaml at runtime
    "GW-DEMO": {
        "id": "GW-DEMO", "name": "Gateway (synthetic)",
        "supported_services": [0x10, 0x11, 0x22, 0x27, 0x19, 0x14, 0x3E, 0x28, 0x85],
        "supported_dids": [0xF190, 0xF18C, 0xF187, 0xF189, 0xF186],
        "supported_rids": [],
    },
    "PCM-DEMO": {
        "id": "PCM-DEMO", "name": "Powertrain CM (synthetic)",
        "supported_services": [0x10, 0x11, 0x22, 0x2E, 0x27, 0x31, 0x19, 0x14, 0x3E, 0x85],
        "supported_dids": [0xF190, 0xF18C, 0xF189, 0xF186, 0x0105, 0x4200],
        "supported_rids": [0x0301, 0x0203, 0x0400],
    },
    # Held out entirely: used only to test generalisation to an unseen ECU.
    "DCU-DEMO": {
        "id": "DCU-DEMO", "name": "Door Control Unit (synthetic, unseen)",
        "supported_services": [0x10, 0x11, 0x22, 0x2E, 0x27, 0x31, 0x19, 0x14, 0x3E],
        "supported_dids": [0xF190, 0xF18C, 0xF186, 0x4101, 0x4102],
        "supported_rids": [0x0301],
    },
}
TRAIN_PROFILES = ["BCM-DEMO", "GW-DEMO", "PCM-DEMO"]
HELDOUT_PROFILE = "DCU-DEMO"
UNKNOWN_SIDS = [0x23, 0x34, 0x99, 0x3D, 0x2F]


def profile_kb(profile_id: str, base: KnowledgeBase | None = None) -> KnowledgeBase:
    """Same KB, different ECU support matrix."""
    base = base or load_kb()
    if profile_id == "BCM-DEMO":
        return base
    ecu = copy.deepcopy(ECU_PROFILES[profile_id])
    return KnowledgeBase(base.raw, {"ecu": ecu})


def _random_request(kb: KnowledgeBase, rng: random.Random) -> Request:
    if rng.random() < 0.06:
        return Request(sid=rng.choice(UNKNOWN_SIDS))
    sid = rng.choice(sorted(kb.services))
    svc = kb.service(sid)
    req = Request(sid=sid)
    if svc.get("has_subfunction"):
        r = rng.random()
        valid = [s["value"] for s in svc["subfunctions"]]
        if r < 0.82:
            req.subfunction = rng.choice(valid)
        elif r < 0.93:
            req.subfunction = rng.choice([v for v in range(0, 0x10) if v not in valid])
    if sid in (0x22, 0x2E):
        req.did = rng.choice(list(kb.dids)) if rng.random() < 0.85 else rng.choice(UNKNOWN_DIDS)
    if sid == 0x31 and req.subfunction is not None:
        req.rid = rng.choice(list(kb.rids)) if rng.random() < 0.85 else rng.choice(UNKNOWN_RIDS)
    return req


def outcome_label(res) -> str:
    return "POSITIVE" if res.outcome == "positive" else f"NRC_{res.nrc:02X}"


def generate_outcomes(n_per_profile: int = 2000, seed: int = SEED) -> pd.DataFrame:
    base = load_kb()
    rng = random.Random(seed)
    rows = []
    for pid in ECU_PROFILES:
        kb = profile_kb(pid, base)
        for _ in range(n_per_profile):
            req = _random_request(kb, rng)
            ctx = _sample_context(rng)
            res = evaluate(req, ctx, kb)
            rows.append({
                "ecu_id": pid, "sid": req.sid,
                "subfunction": -1 if req.subfunction is None else req.subfunction,
                "did": -1 if req.did is None else req.did,
                "rid": -1 if req.rid is None else req.rid,
                "session": ctx.session, "unlocked": ctx.unlocked,
                "request_hex": res.request_hex, "label": outcome_label(res),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# ECU response traces (defect detection)
# --------------------------------------------------------------------------- #
BASE_LATENCY_MS = {0x10: 20, 0x11: 150, 0x22: 12, 0x2E: 30, 0x27: 25, 0x31: 60, 0x19: 35,
                   0x14: 80, 0x3E: 5, 0x28: 15, 0x85: 15}
P2_SERVER_MS = 50.0
DEFECT_TYPES = ["wrong_nrc", "unexpected_positive", "unexpected_negative",
                "timeout", "latency_violation", "truncated_response"]


def _positive_len(req: Request, kb: KnowledgeBase, rng: random.Random) -> int:
    if req.sid == 0x22 and req.did is not None:
        return 3 + int(kb.did(req.did)["length"])
    if req.sid == 0x19:
        return 3 + 4 * rng.randint(0, 5)
    if req.sid == 0x10:
        return 6
    if req.sid == 0x27:
        return 2 + (4 if req.subfunction == 0x01 else 0)
    if req.sid == 0x31:
        return 4 + (2 if req.subfunction == 0x03 else 0)
    if req.sid == 0x2E:
        return 3
    if req.sid == 0x14:
        return 1
    return 2


def generate_traces(n: int = 5000, defect_rate: float = 0.0, seed: int = SEED,
                    kb: KnowledgeBase | None = None) -> pd.DataFrame:
    """Simulate an ECU test run. ``defect_rate`` > 0 emulates faulty firmware."""
    kb = kb or load_kb()
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        req = _random_request(kb, rng)
        if req.sid not in BASE_LATENCY_MS:  # traces only use known services
            req = Request(sid=0x22, did=rng.choice(READONLY_DIDS))
        ctx = _sample_context(rng)
        res = evaluate(req, ctx, kb)
        exp_neg = res.outcome == "negative"
        lat = BASE_LATENCY_MS[req.sid] * math.exp(rng.gauss(0, 0.25)) * (0.6 if exp_neg else 1.0)
        obs_neg, obs_nrc = exp_neg, (res.nrc or 0)
        resp_len = 3 if exp_neg else _positive_len(req, kb, rng)
        defect = "none"
        if rng.random() < defect_rate:
            defect = rng.choice(DEFECT_TYPES)
            if defect == "wrong_nrc" and exp_neg:
                obs_nrc = rng.choice([c for c in [0x10, 0x22, 0x31, 0x33, 0x7F, 0x12, 0x13] if c != res.nrc])
            elif defect == "unexpected_positive" and exp_neg:
                obs_neg, obs_nrc, resp_len = False, 0, _positive_len(req, kb, rng)
            elif defect == "unexpected_negative" and not exp_neg:
                obs_neg, obs_nrc, resp_len = True, rng.choice([0x10, 0x22]), 3
            elif defect == "timeout":
                lat, resp_len, obs_nrc = 5000.0, 0, 0
            elif defect == "latency_violation":
                lat = max(lat * rng.uniform(6, 20), P2_SERVER_MS * 1.5)
            elif defect == "truncated_response" and not exp_neg and resp_len > 2:
                resp_len = rng.randint(1, resp_len - 1)
            else:
                defect = "none"  # defect type not applicable to this request
        resp_sid = 0 if resp_len == 0 else (0x7F if obs_neg else kb.service(req.sid)["positive_response_sid"])
        rows.append({
            "step": i, "sid": req.sid,
            "subfunction": -1 if req.subfunction is None else req.subfunction,
            "did": -1 if req.did is None else req.did, "rid": -1 if req.rid is None else req.rid,
            "session": ctx.session, "unlocked": ctx.unlocked, "request_hex": res.request_hex,
            "response_sid": resp_sid, "nrc": obs_nrc if obs_neg else 0,
            "resp_len": resp_len, "latency_ms": round(lat, 2), "defect": defect,
            "expected_outcome": res.outcome, "expected_nrc": res.nrc or 0,
        })
    return pd.DataFrame(rows)

"""Unit tests for the deterministic UDS engine. Run: pytest -q"""
from dxforge.engine import DiagContext, Request, evaluate
from dxforge.kb import load_kb

kb = load_kb()


def test_read_vin_positive():
    r = evaluate(Request(sid=0x22, did=0xF190), DiagContext(), kb)
    assert r.outcome == "positive"
    assert r.request_hex == "22 F1 90"
    assert r.positive_response_hex.startswith("62 F1 90")


def test_unknown_did_requestoutofrange():
    r = evaluate(Request(sid=0x22, did=0x9999), DiagContext(), kb)
    assert r.outcome == "negative" and r.nrc == 0x31


def test_write_needs_extended_session():
    r = evaluate(Request(sid=0x2E, did=0x4101), DiagContext(session="defaultSession"), kb)
    assert r.outcome == "negative" and r.nrc == 0x7F


def test_write_ok_in_extended_session():
    r = evaluate(Request(sid=0x2E, did=0x4101),
                 DiagContext(session="extendedDiagnosticSession"), kb)
    assert r.outcome == "positive" and r.positive_response_hex.startswith("6E 41 01")


def test_security_gate_locked():
    r = evaluate(Request(sid=0x2E, did=0x4200),
                 DiagContext(session="programmingSession", unlocked=False), kb)
    assert r.outcome == "negative" and r.nrc == 0x33


def test_security_gate_unlocked():
    r = evaluate(Request(sid=0x2E, did=0x4200),
                 DiagContext(session="programmingSession", unlocked=True), kb)
    assert r.outcome == "positive"


def test_invalid_subfunction():
    r = evaluate(Request(sid=0x11, subfunction=0x09), DiagContext(), kb)
    assert r.outcome == "negative" and r.nrc == 0x12


def test_unsupported_service_sid():
    r = evaluate(Request(sid=0x99), DiagContext(), kb)
    assert r.outcome == "negative" and r.nrc == 0x11


def test_every_rule_cites_kb():
    r = evaluate(Request(sid=0x22, did=0xF190), DiagContext(), kb)
    assert all(rule.citation for rule in r.rules)

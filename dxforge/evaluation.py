"""Benchmark harness. Computes REAL metrics against the held-out set.

Metrics:
  * request structural validity   (engine built a well-formed request)
  * request correctness           (request bytes == expected)
  * outcome correctness           (positive/negative class == expected)
  * nrc correctness               (on negative cases, NRC == expected)
  * retrieval recall@k            (did evidence for the service appear)
  * latency per case (ms)

No numbers are hard-coded; everything is measured by running the pipeline.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import yaml

from .engine import DiagContext, evaluate
from .intent import parse
from .kb import KnowledgeBase, load_kb
from .retrieval import get_retriever

_BM = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "benchmark.yaml")


@dataclass
class CaseResult:
    id: str
    intent: str
    expected_request: str
    actual_request: str
    expected_outcome: str
    actual_outcome: str
    expected_nrc: str
    actual_nrc: str
    request_match: bool
    outcome_match: bool
    nrc_match: bool
    retrieval_hit: bool
    latency_ms: float


@dataclass
class BenchmarkReport:
    n: int
    request_validity: float
    request_correctness: float
    outcome_correctness: float
    nrc_correctness: float
    retrieval_recall: float
    p50_ms: float
    p95_ms: float
    cases: list[CaseResult] = field(default_factory=list)


def _pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 1) if den else 0.0


def run_benchmark(use_llm: bool = False, kb: KnowledgeBase | None = None) -> BenchmarkReport:
    kb = kb or load_kb()
    retr = get_retriever()
    with open(_BM, "r", encoding="utf-8") as f:
        cases = yaml.safe_load(f)["cases"]

    results: list[CaseResult] = []
    latencies: list[float] = []
    valid = corr = out_corr = nrc_total = nrc_ok = retr_hit = 0

    for c in cases:
        t0 = time.perf_counter()
        parsed = parse(c["intent"], kb, use_llm=use_llm)
        ctx = DiagContext(session=c.get("session", "defaultSession"),
                          unlocked=c.get("unlocked", False))
        if parsed.request is None:
            dt = (time.perf_counter() - t0) * 1000
            latencies.append(dt)
            results.append(CaseResult(
                c["id"], c["intent"], c["expect"].get("request_hex", ""), "",
                c["expect"]["outcome"], "unparsed",
                f"{c['expect'].get('nrc', ''):#04x}" if c["expect"].get("nrc") else "",
                "", False, False, False, False, round(dt, 2)))
            continue

        res = evaluate(parsed.request, ctx, kb)
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)

        exp = c["expect"]
        exp_req = exp.get("request_hex", "").upper().strip()
        act_req = res.request_hex.upper().strip()
        req_match = exp_req == act_req
        out_match = exp["outcome"] == res.outcome
        exp_nrc = exp.get("nrc")
        nrc_match = True
        exp_nrc_s = act_nrc_s = ""
        if exp["outcome"] == "negative":
            nrc_total += 1
            exp_nrc_s = f"{exp_nrc:#04x}" if exp_nrc is not None else ""
            act_nrc_s = f"{res.nrc:#04x}" if res.nrc is not None else ""
            nrc_match = (exp_nrc == res.nrc)
            if nrc_match:
                nrc_ok += 1

        # retrieval recall: relevant evidence (service OR target DID/RID node) in top-k
        ev = retr.search(c["intent"], k=4)
        svc = kb.service(parsed.request.sid)
        relevant = set()
        if svc:
            relevant.add(svc["id"])
        if parsed.request.did is not None and kb.did(parsed.request.did):
            relevant.add(kb.did(parsed.request.did)["id"])
        if parsed.request.rid is not None and kb.rid(parsed.request.rid):
            relevant.add(kb.rid(parsed.request.rid)["id"])
        hit = any(n["id"] in relevant for n in ev)

        valid += 1 if act_req else 0
        corr += 1 if req_match else 0
        out_corr += 1 if out_match else 0
        retr_hit += 1 if hit else 0

        results.append(CaseResult(
            c["id"], c["intent"], exp_req, act_req, exp["outcome"], res.outcome,
            exp_nrc_s, act_nrc_s, req_match, out_match, nrc_match, hit, round(dt, 2)))

    n = len(cases)
    latencies.sort()
    p50 = latencies[int(0.50 * (n - 1))] if n else 0.0
    p95 = latencies[int(0.95 * (n - 1))] if n else 0.0
    return BenchmarkReport(
        n=n,
        request_validity=_pct(valid, n),
        request_correctness=_pct(corr, n),
        outcome_correctness=_pct(out_corr, n),
        nrc_correctness=_pct(nrc_ok, nrc_total),
        retrieval_recall=_pct(retr_hit, n),
        p50_ms=round(p50, 2),
        p95_ms=round(p95, 2),
        cases=results,
    )

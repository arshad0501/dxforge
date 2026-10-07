#!/usr/bin/env python3
"""Run the DxForge benchmark and print + save a markdown report.

Usage:
    python scripts/run_benchmark.py            # deterministic (no LLM)
    LLM_PROVIDER=ollama python scripts/run_benchmark.py --llm
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dxforge.evaluation import run_benchmark  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="allow LLM assist on parser misses")
    args = ap.parse_args()

    r = run_benchmark(use_llm=args.llm)
    lines = [
        "# DxForge Benchmark Report",
        "",
        f"- Cases: **{r.n}**",
        f"- Request structural validity: **{r.request_validity}%**",
        f"- Request byte correctness: **{r.request_correctness}%**",
        f"- Outcome (positive/negative) correctness: **{r.outcome_correctness}%**",
        f"- NRC correctness (negative cases): **{r.nrc_correctness}%**",
        f"- Retrieval recall@4 (relevant evidence): **{r.retrieval_recall}%**",
        f"- Latency p50 / p95: **{r.p50_ms} ms / {r.p95_ms} ms**",
        "",
        "| Case | Intent | Expected | Actual | req | out | nrc | recall |",
        "|------|--------|----------|--------|-----|-----|-----|--------|",
    ]
    for c in r.cases:
        exp = f"{c.expected_request} / {c.expected_outcome} {c.expected_nrc}".strip()
        act = f"{c.actual_request} / {c.actual_outcome} {c.actual_nrc}".strip()
        tick = lambda b: "ok" if b else "X"
        lines.append(f"| {c.id} | {c.intent} | {exp} | {act} | "
                     f"{tick(c.request_match)} | {tick(c.outcome_match)} | "
                     f"{tick(c.nrc_match)} | {tick(c.retrieval_hit)} |")
    report = "\n".join(lines)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "BENCHMARK_RESULTS.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()

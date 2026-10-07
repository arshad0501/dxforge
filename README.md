# DxForge — UDS Diagnostic Intelligence & Test-Generation Platform

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

> *The LLM proposes. The protocol engine disposes. The engineer approves.*

DxForge turns a plain-English diagnostic intent (e.g. *"read the VIN"*) into a
**protocol-validated UDS request**, its **expected positive response**, the full
**negative-response (NRC) set**, cited **evidence**, and a **runnable test** — with
session/security pre-conditions enforced by a deterministic engine that contains **zero AI**.

Built for the Tata Technologies *Automotive Engineering AI* case study **#5 — UDS
Diagnostics & Automated Test Generation** (ISO 14229).

<!-- Add screenshots/screenshots.png here -->

---

## Why this is not a chatbot

A raw LLM will happily hallucinate UDS byte structures and NRCs. DxForge separates the
two jobs:

| Layer | Responsibility | Contains AI? |
|-------|----------------|:------------:|
| Intent parser | map NL → a *candidate* structured request | optional, clamped |
| **Deterministic engine** | validate structure, session, security; emit exact bytes + NRCs | **no** |
| RAG retrieval | attach citable evidence from the KB | no (lexical) |
| Test generation | positive + negative cases, pytest/CAPL export | no |

Protocol correctness is **guaranteed by rules**, not by a model — which is exactly why the
measured request/outcome/NRC correctness is 100% on the benchmark (below).

## Runs anywhere — no model required

The entire core runs on a plain `pip install`. **No Ollama, no API key, no model download.**
An LLM is an *optional* upgrade that only rescues natural-language parser misses and is
**clamped to KB-valid identifiers** before the engine ever sees it.

```bash
git clone <your-repo-url> dxforge && cd dxforge
pip install -r requirements.txt
streamlit run app.py          # open http://localhost:8501
```

Docker:

```bash
cp .env.example .env
docker compose up --build     # http://localhost:8501
```

Run the test suite and the benchmark:

```bash
pytest -q
python scripts/run_benchmark.py        # writes BENCHMARK_RESULTS.md
```

## Measured results (reproducible)

Run `python scripts/run_benchmark.py` to regenerate. On the included 20-case held-out set
(which is **never embedded into retrieval**, so there is no leakage):

| Metric | Result |
|--------|--------|
| Request structural validity | 100.0% |
| Request byte correctness | 100.0% |
| Outcome (positive/negative) correctness | 100.0% |
| NRC correctness (negative cases) | 100.0% |
| Retrieval recall@4 (relevant evidence) | 95.0% |
| Latency p50 / p95 | ~0.02 ms / ~0.04 ms |

*(Deterministic configuration, `LLM_PROVIDER=none`. Numbers are computed live by the
harness — nothing is hard-coded. Your machine may differ slightly on latency.)*

## Features

- **Diagnostic Studio** — NL intent → validated request, expected response, evidence, tests.
- **Protocol Validator** — paste a raw hex request; decode + validate against the context.
- **NRC Coverage Map** — negative-scenario coverage per service.
- **Benchmark** — live, honest metrics over a leak-free held-out set.
- **Audit Log** — project-isolated trail of every generation and approval.
- **Governance** — refusal on ungrounded intent, deterministic-vs-inferred flags, human approval.

## Architecture

```
Streamlit UI  ──►  intent parser (deterministic | LLM→clamped)
                         │
                         ▼
          DETERMINISTIC UDS ENGINE  ◄── KB (services/DIDs/RIDs/NRCs/sessions)
          (structure · session · security · NRCs)        ▲
                         │                                │
                 ┌───────┴────────┐              lexical RAG (BM25)
                 ▼                ▼                    evidence + citations
          expected response   test generation
          + NRC matrix        (pytest / CAPL / md)
                         │
                         ▼
            SQLite audit log (project-isolated)
```

Scale path (documented, not required for the pilot): FastAPI service layer, PostgreSQL,
sentence-transformer embeddings + Chroma/FAISS, optional Neo4j traceability graph, Kubernetes.

## Project structure

```
dxforge/
  app.py                 Streamlit UI + workflow
  dxforge/
    kb.py                KB loader + lookups + citable text nodes
    engine.py            deterministic protocol engine (zero AI)
    intent.py            NL → structured request (deterministic + optional LLM)
    retrieval.py         BM25 lexical retrieval over KB
    testgen.py           test-case generation + pytest/CAPL/md export
    evaluation.py        benchmark harness (real metrics)
    audit.py             SQLite audit log
    llm.py               optional pluggable provider (none/ollama/openai)
  data/
    uds_kb.yaml          derived/synthetic UDS reference (NOT ISO 14229 text)
    ecu_profile_bcm.yaml synthetic demo ECU
    benchmark.yaml       held-out eval set (kept out of the index)
  tests/test_engine.py   unit tests
  scripts/run_benchmark.py
  Dockerfile · docker-compose.yml · requirements*.txt · .env.example
```

## Data & licensing

The knowledge base is a **derived/synthetic reference** authored from freely available,
non-standard public sources (open-source libraries such as `udsoncan`/`python-uds`, public
application notes, OBD/UDS tutorials). It is **not** a copy or redistribution of the
copyrighted ISO 14229 standard. The ECU profile and benchmark are fully synthetic and
labelled as such.

## Safety & governance

Exports are for **approved bench/HIL environments only — never production vehicles**.
DxForge models SecurityAccess as **protocol state only** and never derives, generates, or
cracks seed/key algorithms. AI-generated content must remain traceable to approved source
material and reviewed by a qualified diagnostic engineer before consequential use.

## Security notes

Secrets live only in `.env` (git-ignored); no keys in the repo. Untrusted inputs (raw hex,
free-text intent, optional LLM output) are validated/clamped before reaching the engine.
Projects are isolated and every action is audit-logged. See **Governance** page in-app.

# DxForge — 5-minute demo script

**Setup before you present:** `streamlit run app.py`, open `http://localhost:8501`,
leave it on the **Diagnostic Studio** page. No internet needed. If anything is flaky,
use the fallback at the bottom.

### 0:00–0:30 · Problem & impact
> "UDS testing means hand-building requests, tracking session and security pre-conditions,
> enumerating every positive and negative response, and turning that into automation —
> across hundreds of services and identifiers. It's slow, inconsistent, and error-prone.
> DxForge compresses that to seconds — without ever letting an LLM invent protocol bytes."

### 0:30–1:00 · Product overview
Point at the sidebar: project isolation, diagnostic context (session + security), and the
**LLM: off (deterministic)** badge. "The entire engineering core runs with no model at all."

### 1:00–2:00 · Core workflow (the hook)
Type **`read the VIN`** → Analyze. Show: request `22 F1 90`, expected positive `62 F1 90 <data>`,
the applied rules each citing a KB node, and the auto-generated positive+negative tests.
Download the **pytest template**. "Every byte is produced by the deterministic engine, and
every claim is cited."

### 2:00–3:00 · The differentiator — deterministic guarantees
Switch session to **defaultSession**, type **`write door lock state`** → NEGATIVE `7F 2E 7F`
(serviceNotSupportedInActiveSession). Switch to **extendedDiagnosticSession**, re-run → POSITIVE.
Then **`read DID 0x9999`** → NEGATIVE `7F 22 31` (requestOutOfRange). "The LLM proposes; the
protocol engine disposes. This is where a plain chatbot hallucinates and we don't."

### 3:00–4:00 · Measurable performance
Go to **Benchmark** → Run benchmark. Show the live KPIs (100% request/outcome/NRC correctness,
95% recall@4, sub-millisecond latency) over a held-out set "that is never embedded into
retrieval, so there's no leakage." Scroll the per-case table.

### 4:00–4:30 · Security / traceability / governance
Open **Governance**: refusal on ungrounded intent, deterministic-vs-inferred flags, bench/HIL-only
exports, no seed/key cracking. Open **Audit Log**: every generation/approval recorded, per project.

### 4:30–5:00 · Business impact & scale
> "This is a reusable internal productivity tool: faster, more consistent diagnostic test design
> with built-in traceability. It scales to FastAPI + PostgreSQL + embeddings + a traceability
> graph without changing the deterministic core — which is the part that must stay correct."

---

## Fallback (no Streamlit / projector issues)
From a terminal:
```bash
pytest -q                          # 9 passing engine tests
python scripts/run_benchmark.py    # prints the full measured benchmark table
```
These alone demonstrate the deterministic engine and the real metrics.

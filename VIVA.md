# DxForge — technical viva / evaluator attack Q&A

Grounded in the actual implementation. Format: **Q → ideal answer → likely follow-up → stronger answer.**

**Q: What is actually "AI" here? Isn't this just rules?**
The AI is the natural-language intent layer (`intent.py`) plus the RAG evidence retrieval
(`retrieval.py`). The deliberate design is that AI never produces the final protocol artefact —
the deterministic engine (`engine.py`) does. That split is the product's thesis, not a gap.
*Follow-up: so I could remove the LLM?* Yes — and we do by default (`LLM_PROVIDER=none`); the
deterministic parser handles the benchmark at 100% request correctness. The LLM only widens the
range of phrasings we can accept, and it's clamped to KB-valid identifiers first.

**Q: Why RAG at all if the engine is deterministic?**
RAG supplies the *evidence and citations* a reviewer needs to trust and audit an output, and it
grounds the explanation ("why this service, why this NRC"). The engine decides; RAG justifies.

**Q: Why not just use ChatGPT?**
A raw model hallucinates byte structures and NRCs and gives no guarantee or citation. Here, protocol
correctness is rule-guaranteed and measured (100% on the held-out set), outputs are cited, and the
system *refuses* when it can't ground an intent. That's the difference between a demo and a tool.

**Q: How do you know an answer is correct?**
For protocol structure it's deterministic and unit-tested (`tests/test_engine.py`) and benchmarked
against a held-out ground-truth set (`data/benchmark.yaml`) with measured request/outcome/NRC
correctness. For retrieval we measure recall@4. Nothing is self-asserted by a model.

**Q: What happens when the model hallucinates, and how do you evaluate that?**
Any LLM proposal is clamped (`intent._clamp_llm`): unknown SIDs/DIDs/RIDs are dropped, so a
hallucinated identifier can't reach the engine. If nothing valid survives, we fall back to the
deterministic parser or refuse. We evaluate groundedness via the benchmark's byte-exact checks and
retrieval recall; a hallucinated request simply fails those and shows up in the per-case table.

**Q: What is deterministic, exactly?**
Structure/length validation, sub-function legality, DID/RID existence and access, session rank
comparison, security-lock gating, and the positive-response and NRC derivation — all pure functions
of (request, context, KB), with an explicit, documented NRC precedence (sub-function → session →
security).

**Q: How is security handled / how do you protect confidential engineering data?**
Runs locally with no external calls in the default config; secrets only in git-ignored `.env`;
project isolation + audit logging; untrusted inputs (raw hex, free text, optional LLM output) are
validated/clamped. We deliberately never derive or crack seed/key material — SecurityAccess is
modelled as protocol state only.

**Q: What if the source document (KB) is wrong, or two sources conflict?**
Outputs cite the exact KB node, so a wrong fact is traceable and correctable in one place, and the
engine re-derives deterministically. The KB is versionable; conflicts are resolved at authoring time
with a single source of truth per node (the `id`).

**Q: How does it scale, and what's the latency?**
Measured p50 ≈ 0.02 ms, p95 ≈ 0.04 ms per case in the deterministic path (the engine is O(1) rule
checks; BM25 over a small corpus). The documented scale path swaps BM25→embeddings+Chroma, SQLite→
PostgreSQL, adds a FastAPI service tier and optional Neo4j graph — without touching the engine.

**Q: What are your false positives / false negatives?**
On the held-out set, zero on outcome and NRC classification (20/20). The honest caveat: the parser's
coverage is bounded by authored phrasings and the KB's authored services — out-of-scope intents are
refused rather than guessed, which we count as correct behaviour, not a silent failure.

**Q: What makes this production-worthy, and what if a recommendation is wrong?**
Human approval is mandatory and logged; exports are bench/HIL-only; every artefact is traceable to a
cited source and reproducible. A wrong recommendation is caught at the approval gate and is fully
auditable — the system assists, it does not replace qualified engineering sign-off.

**Q: Why would Tata Technologies actually use this?**
It attacks a real, high-volume V&V bottleneck (diagnostic test design) with a reusable, auditable
internal tool, and the correctness-critical core is deterministic — the part an automotive org can
actually certify and trust.

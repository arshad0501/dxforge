"""DxForge — UDS Diagnostic Intelligence & Test-Generation Platform (pilot UI).

Single-process Streamlit pilot. The engineering core (engine, retrieval,
test-gen, benchmark) is decoupled into the `dxforge` package; this file is only
presentation + workflow. Runs with zero model dependency.
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from dxforge import audit, llm
from dxforge.engine import DiagContext, Request, build_request_hex, evaluate
from dxforge.intent import parse
from dxforge.kb import load_kb
from dxforge.retrieval import get_retriever
from dxforge.testgen import build_test_cases, to_markdown, to_pytest

st.set_page_config(page_title="DxForge · UDS Diagnostic Intelligence",
                   page_icon="🛠️", layout="wide")

# --------------------------------------------------------------------------- #
# Styling — intentional engineering look, not default Streamlit
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
:root { --fg:#e6edf3; --muted:#8b98a9; --accent:#3fb6ff; --ok:#3fb950; --warn:#d29922; --bad:#f85149; --card:#161b22; --line:#2b3440; }
.stApp { background: #0d1117; color: var(--fg); }
section[data-testid="stSidebar"] { background: #0a0e14; border-right: 1px solid var(--line); }
h1,h2,h3,h4 { letter-spacing:-0.01em; }
.dx-brand { font-size:1.35rem; font-weight:700; }
.dx-brand span { color: var(--accent); }
.dx-tag { color:var(--muted); font-size:0.8rem; }
.dx-card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 18px; margin-bottom:12px; }
.dx-kpi { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; text-align:center; }
.dx-kpi .v { font-size:1.6rem; font-weight:700; }
.dx-kpi .l { color:var(--muted); font-size:0.75rem; text-transform:uppercase; letter-spacing:0.04em; }
.pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.72rem; font-weight:600; }
.pill.ok { background:rgba(63,185,80,0.15); color:var(--ok); border:1px solid rgba(63,185,80,0.4);}
.pill.bad { background:rgba(248,81,73,0.15); color:var(--bad); border:1px solid rgba(248,81,73,0.4);}
.pill.det { background:rgba(63,182,255,0.12); color:var(--accent); border:1px solid rgba(63,182,255,0.4);}
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.hexbox { font-family: ui-monospace, monospace; font-size:1.1rem; background:#0a0e14; border:1px solid var(--line); border-radius:8px; padding:10px 12px; color:var(--accent); }
.rule { border-left:3px solid var(--accent); padding:4px 10px; margin:6px 0; background:rgba(63,182,255,0.05); font-size:0.86rem; }
.cite { color:var(--muted); font-size:0.78rem; }
.stButton>button { border-radius:8px; border:1px solid var(--line); }
</style>
""", unsafe_allow_html=True)

KB = load_kb()
RETR = get_retriever()
SESSIONS = ["defaultSession", "extendedDiagnosticSession", "programmingSession"]

# --------------------------------------------------------------------------- #
# Sidebar — context, project isolation, provider status
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown('<div class="dx-brand">Dx<span>Forge</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="dx-tag">UDS Diagnostic Intelligence · ISO 14229 (derived)</div>',
                unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigate", ["Diagnostic Studio", "Protocol Validator",
                                 "Coverage", "Benchmark", "Audit Log", "Governance"],
                    label_visibility="collapsed")
    st.divider()
    st.caption("WORKSPACE")
    project = st.selectbox("Project", ["BCM-Platform", "Powertrain-X", "Sandbox"], index=0)
    user = st.text_input("Engineer", value="demo.engineer")
    st.caption("DIAGNOSTIC CONTEXT")
    session = st.selectbox("Active session", SESSIONS, index=0)
    unlocked = st.toggle("Security unlocked (0x27)", value=False)
    st.caption(f"ECU: {KB.ecu.get('name','?')} · {KB.ecu.get('address','')}")
    st.divider()
    prov = llm.provider_name()
    badge = "ok" if llm.is_enabled() else "det"
    label = f"LLM: {prov}" if llm.is_enabled() else "LLM: off (deterministic)"
    st.markdown(f'<span class="pill {badge}">{label}</span>', unsafe_allow_html=True)
    st.caption("Core runs fully without a model. Set LLM_PROVIDER to enable NL assist.")

ctx = DiagContext(session=session, unlocked=unlocked)


# --------------------------------------------------------------------------- #
def result_view(req: Request, res, parsed=None):
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("##### Request")
        st.markdown(f'<div class="hexbox">{res.request_hex}</div>', unsafe_allow_html=True)
        if res.outcome == "positive":
            st.markdown('<span class="pill ok">POSITIVE</span>', unsafe_allow_html=True)
            st.markdown("##### Expected positive response")
            st.markdown(f'<div class="hexbox">{res.positive_response_hex}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill bad">NEGATIVE</span>', unsafe_allow_html=True)
            st.markdown("##### Expected negative response")
            nrc = res.nrc
            st.markdown(f'<div class="hexbox">7F {req.sid:02X} {nrc:02X} '
                        f'&nbsp;({res.nrc_name})</div>', unsafe_allow_html=True)
    with c2:
        st.markdown("##### Verdict")
        st.markdown(f'<span class="pill det">{res.confidence.upper()}</span>',
                    unsafe_allow_html=True)
        st.caption(f"Service: {res.service_name or 'unknown'}")
        if res.candidate_nrcs:
            st.caption("Applicable NRCs: " +
                       ", ".join(f"{c:#04x} {KB.nrc_name(c)}" for c in res.candidate_nrcs))
        if parsed:
            st.caption(f"Intent method: {parsed.method} — {parsed.rationale}")
        for note in res.notes:
            st.info(note, icon="🛈")

    st.markdown("##### Applied rules (deterministic engine)")
    for r in res.rules:
        st.markdown(f'<div class="rule">{r.detail} '
                    f'<span class="cite">[{r.citation}]</span></div>', unsafe_allow_html=True)

    with st.expander("Retrieved evidence (RAG citations)"):
        q = (parsed.rationale if parsed else "") + " " + (res.service_name or "")
        ev = RETR.search((parsed and parsed.request and res.service_name or "") +
                         " " + (res.service_name or ""), k=4) if res.service_name else []
        if not ev:
            ev = RETR.search(res.service_name or "", k=4)
        if ev:
            for n in ev:
                st.markdown(f"**[{n['id']}]** {n['title']} · score {n['score']}  \n"
                            f"<span class='cite'>{n['text']}</span>", unsafe_allow_html=True)
        else:
            st.caption("No evidence nodes retrieved.")

    # test generation
    st.markdown("##### Generated test cases")
    cases = build_test_cases(req, ctx, res)
    st.markdown(to_markdown(cases))
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.download_button("⬇ pytest template", to_pytest(cases),
                           file_name="test_uds_generated.py", mime="text/x-python")
    with cc2:
        st.download_button("⬇ test matrix (md)", to_markdown(cases),
                           file_name="uds_test_matrix.md", mime="text/markdown")
    with cc3:
        if st.button("✓ Approve & log", type="primary"):
            audit.log(project, user, "approve", parsed.rationale if parsed else "",
                      res.request_hex, res.outcome, "approved")
            st.success("Logged to audit trail.")
    st.caption("Exports are for approved bench/HIL environments only — never production vehicles.")


# --------------------------------------------------------------------------- #
if page == "Diagnostic Studio":
    st.markdown("### Diagnostic Studio")
    st.caption("Describe the diagnostic intent in plain English. The engine returns a "
               "protocol-validated request, expected response, evidence, and tests.")
    examples = ["read the VIN", "write door lock state", "enter programming session",
                "start the self test routine", "read DID 0x9999", "erase memory routine"]
    cols = st.columns(len(examples))
    picked = None
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"ex{i}"):
            picked = ex
    text = st.text_input("Diagnostic intent", value=picked or "read the VIN")
    if st.button("Analyze", type="primary") or picked:
        parsed = parse(text, KB, use_llm=True)
        if parsed.request is None:
            st.warning("Could not map this to a known UDS service with confidence. "
                       "Try naming a service (read/write/reset/session/routine/DTC) or a DID.")
            with st.expander("Why no answer? (refusal is a feature)"):
                st.write("The engine refuses to fabricate a protocol artefact when the intent "
                         "cannot be grounded in the KB. This prevents hallucinated requests.")
        else:
            res = evaluate(parsed.request, ctx, KB)
            audit.log(project, user, "generate", text, res.request_hex, res.outcome, "generated")
            result_view(parsed.request, res, parsed)

elif page == "Protocol Validator":
    st.markdown("### Protocol Validator")
    st.caption("Paste a raw UDS request (hex). The engine decodes and validates it in the "
               "current context — useful for reviewing captured traces.")
    raw = st.text_input("Raw request bytes (hex)", value="22 F1 90")
    if st.button("Validate", type="primary"):
        try:
            b = [int(x, 16) for x in raw.replace(",", " ").split()]
            assert b
            sid = b[0]
            svc = KB.service(sid)
            req = Request(sid=sid)
            idx = 1
            if svc and svc.get("has_subfunction") and idx < len(b):
                req.subfunction = b[idx]; idx += 1
            if sid in (0x22, 0x2E) and idx + 1 < len(b):
                req.did = (b[idx] << 8) | b[idx + 1]; idx += 2
            elif sid == 0x31 and idx + 1 < len(b):
                req.rid = (b[idx] << 8) | b[idx + 1]; idx += 2
            req.data = b[idx:]
            res = evaluate(req, ctx, KB)
            result_view(req, res)
        except Exception:
            st.error("Could not parse those bytes. Use space-separated hex, e.g. `22 F1 90`.")

elif page == "Coverage":
    st.markdown("### NRC Coverage Map")
    st.caption("Which negative-response scenarios each service can exercise. Drives "
               "negative-test completeness.")
    rows = []
    for sid, svc in sorted(KB.services.items()):
        rows.append({
            "SID": f"{sid:#04x}",
            "Service": svc["name"],
            "Required session": svc.get("required_session"),
            "NRCs covered": ", ".join(f"{c:#04x} {KB.nrc_name(c)}" for c in svc.get("applicable_nrcs", [])),
            "# NRCs": len(svc.get("applicable_nrcs", [])),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("Service")["# NRCs"])

elif page == "Benchmark":
    st.markdown("### Benchmark & Performance")
    st.caption("Measured against a held-out set that is never embedded into retrieval "
               "(no leakage). Numbers are computed live — nothing is hard-coded.")
    use_llm = st.toggle("Allow LLM assist on parser misses", value=False)
    if st.button("Run benchmark", type="primary"):
        from dxforge.evaluation import run_benchmark
        with st.spinner("Running pipeline over held-out cases..."):
            r = run_benchmark(use_llm=use_llm)
        k = st.columns(6)
        kpis = [("Validity", r.request_validity), ("Req correct", r.request_correctness),
                ("Outcome", r.outcome_correctness), ("NRC", r.nrc_correctness),
                ("Recall@4", r.retrieval_recall), ("p95 ms", r.p95_ms)]
        for col, (lab, val) in zip(k, kpis):
            suffix = "" if lab == "p95 ms" else "%"
            col.markdown(f'<div class="dx-kpi"><div class="v">{val}{suffix}</div>'
                         f'<div class="l">{lab}</div></div>', unsafe_allow_html=True)
        st.markdown(" ")
        df = pd.DataFrame([{
            "Case": c.id, "Intent": c.intent,
            "Expected": f"{c.expected_request} / {c.expected_outcome} {c.expected_nrc}".strip(),
            "Actual": f"{c.actual_request} / {c.actual_outcome} {c.actual_nrc}".strip(),
            "req": c.request_match, "outcome": c.outcome_match,
            "nrc": c.nrc_match, "recall": c.retrieval_hit, "ms": c.latency_ms,
        } for c in r.cases])
        st.dataframe(df, use_container_width=True, hide_index=True)

elif page == "Audit Log":
    st.markdown("### Audit Log")
    st.caption(f"Project-isolated trail for **{project}**. Every generation and approval is recorded.")
    rows = audit.recent(project=project, limit=200)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No audit entries yet for this project. Generate or approve a request first.")

elif page == "Governance":
    st.markdown("### Governance, scope & how it works")
    st.markdown("""
<div class="dx-card">
<b>Problem.</b> UDS (ISO 14229) testing forces engineers to hand-build requests, track
session/security pre-conditions, enumerate positive and negative (NRC) responses, and
translate all of it into automation — across many services, DIDs, RIDs and variants.
It is slow, inconsistent, and error-prone.
</div>
<div class="dx-card">
<b>How DxForge works.</b> Natural-language intent is parsed (deterministically by default; an
optional LLM only rescues misses and is <i>clamped</i> to KB-valid identifiers). A
<b>deterministic protocol engine</b> — containing zero AI — then validates structure, session
and security pre-conditions and produces the exact request, expected response and NRC set.
Every decision cites a KB node.
</div>
<div class="dx-card">
<b>Governance guardrails.</b> Outputs are evidence-grounded and cited; the engine <i>refuses</i>
when intent can't be grounded; results carry a deterministic-vs-inferred flag; projects are
isolated; every action is audit-logged; approvals are human. Exports are marked
<b>bench/HIL only — never production vehicles</b>, and the tool never derives or cracks
seed/key algorithms (it models security as protocol state only).
</div>
<div class="dx-card">
<b>Data.</b> The KB is a derived/synthetic reference authored from freely available public
sources and open-source libraries — <b>not</b> a copy of ISO 14229. The ECU profile and
benchmark are synthetic and labelled as such.
</div>
""", unsafe_allow_html=True)
    st.caption("AI-generated content must remain traceable to approved source material and "
               "reviewed by a qualified diagnostic engineer before consequential use.")

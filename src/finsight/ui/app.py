import asyncio
from dataclasses import asdict
from typing import Any

import streamlit as st
from openai import APIError

from finsight.ui.catalog import (
    FilingCatalogItem,
    company_names,
    filings_for_company,
)
from finsight.ui.service import AnswerView, list_indexed_filings, stream_filing_answer

SUGGESTED_QUESTIONS = (
    "What are the company's most significant business risks?",
    "What were the primary drivers of revenue growth?",
    "How does management describe its competitive position?",
    "What could negatively affect future operating results?",
)


def main() -> None:
    st.set_page_config(
        page_title="FinSight · SEC filing intelligence",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_styles()
    _initialize_state()

    filings = _load_filings()
    if not filings:
        _render_empty_state()
        return

    selected = _render_sidebar(filings)
    _render_header(selected)
    _render_history()

    suggested_question = _render_suggestions()
    typed_question = st.chat_input(
        f"Ask a question about {selected.company_name}'s filing…"
    )
    question = typed_question or suggested_question
    if question:
        _answer_question(question, selected)


@st.cache_data(ttl=60, show_spinner=False)
def _load_filings() -> list[FilingCatalogItem]:
    return list_indexed_filings()


def _initialize_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []


def _render_sidebar(filings: list[FilingCatalogItem]) -> FilingCatalogItem:
    with st.sidebar:
        st.markdown('<div class="brand-mark">F</div>', unsafe_allow_html=True)
        st.markdown("## Research scope")
        st.caption("Choose an indexed SEC filing before asking a question.")

        companies = company_names(filings)
        company = st.selectbox("Company", companies)
        company_filings = filings_for_company(filings, company)
        selected = st.selectbox(
            "Filing",
            company_filings,
            format_func=lambda filing: filing.label,
        )

        st.markdown("---")
        st.markdown("**Retrieval pipeline**")
        st.caption("Vector search + PostgreSQL full-text search")
        st.caption("Reciprocal-rank fusion + cross-encoder reranking")
        st.caption("Grounded generation with chunk-level citations")

        st.markdown("---")
        st.markdown(
            f"[Open original filing ↗]({selected.source_url})"
        )
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    return selected


def _render_header(filing: FilingCatalogItem) -> None:
    st.markdown(
        """
        <div class="eyebrow">EVIDENCE-BACKED FINANCIAL RESEARCH</div>
        <h1>Ask the filing.<br><span>Trace every claim.</span></h1>
        <p class="lede">
          FinSight searches SEC evidence, reranks the strongest passages, and
          produces an answer you can inspect at the source.
        </p>
        """,
        unsafe_allow_html=True,
    )

    period = filing.report_period or filing.filing_date
    columns = st.columns(4)
    columns[0].metric("Company", filing.company_name)
    columns[1].metric("Filing", filing.form_type)
    columns[2].metric("Fiscal year", str(period.year))
    columns[3].metric("Filed", filing.filing_date.strftime("%b %d, %Y"))
    st.markdown("---")


def _render_suggestions() -> str | None:
    if st.session_state.history:
        return None

    st.markdown("#### Start with a research question")
    columns = st.columns(2)
    for index, question in enumerate(SUGGESTED_QUESTIONS):
        if columns[index % 2].button(
            question,
            key=f"suggestion-{index}",
            use_container_width=True,
        ):
            return question
    return None


def _answer_question(question: str, filing: FilingCatalogItem) -> None:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🔎"):
        status = st.status("Searching and reranking filing evidence…", expanded=False)
        answer_placeholder = st.empty()
        answer_text = ""

        def on_delta(delta: str) -> None:
            nonlocal answer_text
            answer_text += delta
            answer_placeholder.markdown(answer_text + "▌")

        def on_retrieval(count: int) -> None:
            status.update(label=f"Generating from {count} evidence passages…")

        try:
            result: AnswerView | None = None
            for attempt in range(2):
                try:
                    result = asyncio.run(
                        stream_filing_answer(
                            question=question,
                            filing_id=filing.id,
                            evidence_limit=5,
                            on_delta=on_delta,
                            on_retrieval=on_retrieval,
                        )
                    )
                    break
                except APIError:
                    if attempt == 0 and not answer_text:
                        status.update(
                            label="The AI provider had a temporary issue. Retrying…"
                        )
                        continue
                    raise

            if result is None:
                raise RuntimeError("The research request did not return a result")
        except Exception as exc:
            status.update(label="The research request failed", state="error")
            st.error(
                "FinSight could not complete this question. Check the database, "
                "Redis, and OpenAI configuration, then try again."
            )
            st.caption(str(exc))
            return

        answer_placeholder.markdown(result.answer)
        source = "cached answer" if result.cached else f"{result.evidence_count} passages"
        status.update(label=f"Answer grounded in {source}", state="complete")
        _render_answer_details(result)

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": result.answer,
            "result": asdict(result),
        }
    )


def _render_history() -> None:
    for message in st.session_state.history:
        with st.chat_message(
            message["role"],
            avatar="🔎" if message["role"] == "assistant" else None,
        ):
            st.markdown(message["content"])
            if result_data := message.get("result"):
                _render_answer_details(AnswerView(**_restore_result(result_data)))


def _restore_result(result: dict[str, Any]) -> dict[str, Any]:
    from finsight.ui.service import CitationView

    restored = dict(result)
    restored["citations"] = [
        CitationView(**citation)
        for citation in result["citations"]
    ]
    return restored


def _render_answer_details(result: AnswerView) -> None:
    if result.sufficient_evidence:
        st.success("Supported by the selected filing", icon="✅")
    else:
        st.warning(
            "The selected filing does not contain enough evidence for a reliable answer.",
            icon="⚠️",
        )

    if not result.citations:
        return

    st.markdown("##### Evidence trail")
    for index, citation in enumerate(result.citations, start=1):
        label = f"{index:02d} · Item {citation.item} · {citation.title}"
        with st.expander(label):
            st.caption(
                f"{citation.company_name} · {citation.form_type} · "
                f"filed {citation.filing_date} · chunk {citation.chunk_id}"
            )
            if citation.excerpt:
                st.markdown(f"> {citation.excerpt}")
            st.markdown(f"[View evidence in SEC filing ↗]({citation.source_url})")


def _render_empty_state() -> None:
    st.title("FinSight")
    st.warning("No embedded filings are available yet.")
    st.markdown(
        "Run the filing ingestion and indexing pipeline, then refresh this page."
    )


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Libre+Franklin:wght@600;700&display=swap');

        :root {
          --ink: #17221f;
          --forest: #123a32;
          --mint: #b9e3d3;
          --paper: #f6f4ee;
          --copper: #c66a42;
        }
        .stApp { background: var(--paper); color: var(--ink); }
        [data-testid="stSidebar"] { background: #102b26; }
        [data-testid="stSidebar"] * { color: #eef6f2; }
        [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.14); }
        h1, h2, h3, h4, h5 { font-family: 'Libre Franklin', sans-serif; }
        p, label, button, input, textarea { font-family: 'DM Sans', sans-serif; }
        h1 {
          font-size: clamp(2.7rem, 6vw, 5.7rem) !important;
          line-height: .98 !important;
          letter-spacing: -.055em;
          max-width: 900px;
        }
        h1 span { color: var(--copper); }
        .eyebrow {
          color: #49776b;
          font-size: .75rem;
          font-weight: 700;
          letter-spacing: .16em;
          margin-top: 1.4rem;
        }
        .lede {
          max-width: 670px;
          color: #50625d;
          font-size: 1.06rem;
          line-height: 1.7;
          margin: 1.4rem 0 2rem;
        }
        .brand-mark {
          width: 46px;
          height: 46px;
          display: grid;
          place-items: center;
          border-radius: 13px;
          background: var(--mint);
          color: var(--forest) !important;
          font-family: 'Libre Franklin';
          font-size: 1.35rem;
          font-weight: 700;
          margin-bottom: 1.5rem;
        }
        [data-testid="stMetric"] {
          background: rgba(255,255,255,.58);
          border: 1px solid #e3dfd4;
          border-radius: 14px;
          padding: .8rem 1rem;
        }
        [data-testid="stChatMessage"] {
          background: rgba(255,255,255,.52);
          border: 1px solid #e4e0d6;
          border-radius: 18px;
          padding: .65rem;
        }
        .stButton > button {
          border-radius: 12px;
          border-color: #d8d3c7;
          min-height: 46px;
          text-align: left;
        }
        .stButton > button:hover { border-color: var(--copper); color: #9d4e30; }
        [data-testid="stChatInput"] { border-color: #c9c3b6; }
        @media (max-width: 720px) { h1 { font-size: 3rem !important; } }
        </style>
        """,
        unsafe_allow_html=True,
    )

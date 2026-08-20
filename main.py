"""Streamlit interface for the PDF-grounded clinical guideline assistant."""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from model.generation import generate_answer
from model.retriever import get_or_build_vector_store, retrieve_relevant_chunks


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
VECTOR_DATABASE_DIRECTORY = PROJECT_ROOT / "rag_vector_db"


@st.cache_resource(show_spinner=False)
def load_vector_store():
    """Create the database on first use and reuse it for the Streamlit session."""
    load_dotenv(PROJECT_ROOT / ".env")
    return get_or_build_vector_store(DATA_DIRECTORY, VECTOR_DATABASE_DIRECTORY)


def show_answer(question: str) -> None:
    """Retrieve grounded evidence and render the generated response."""
    with st.spinner("Searching the guideline documents..."):
        relevant_chunks = retrieve_relevant_chunks(load_vector_store(), question)
        answer = generate_answer(question, relevant_chunks)

    st.subheader("Answer")
    st.write(answer.get("recommendation", "No answer was returned."))

    evidence = answer.get("evidence", [])
    if evidence:
        st.subheader("Evidence")
        if isinstance(evidence, list):
            for item in evidence:
                st.write(f"- {item}")
        else:
            st.write(evidence)

    citations = answer.get("citations", [])
    if citations:
        st.subheader("Sources")
        for citation in citations:
            st.write(
                f"- **{citation.get('source', 'Unknown source')}** "
                f"— page {citation.get('page', 'Unknown')}"
            )

    if relevant_chunks:
        with st.expander("Retrieved passages"):
            for document, score in relevant_chunks:
                source = document.metadata.get("file_name", "Unknown source")
                page = document.metadata.get("page", "Unknown")
                st.caption(f"{source} — page {page} | relevance: {score:.2f}")
                st.write(document.page_content.removeprefix("passage: "))


def main() -> None:
    st.set_page_config(page_title="Clinical Guideline Assistant", page_icon="⚕️")
    st.title("Clinical Guideline Assistant")
    st.caption("Answers are grounded only in the PDF guidelines stored in this project.")
    st.warning(
        "This tool is for informational guideline lookup only. It is not a substitute "
        "for professional clinical judgment, diagnosis, or emergency care."
    )

    with st.form("question_form"):
        question = st.text_area(
            "Ask a question about the uploaded guidelines",
            placeholder="Example: In which age group is glibenclamide not recommended?",
        )
        submitted = st.form_submit_button("Ask")

    if submitted:
        if not question.strip():
            st.info("Please enter a question first.")
        else:
            show_answer(question.strip())


if __name__ == "__main__":
    main()

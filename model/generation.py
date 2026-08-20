"""Grounded Gemini response generation and citation formatting."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


GROUNDING_SYSTEM_PROMPT = """You are a citation-bound Clinical AI Assistant specialized in Evidence-Based Medical Guidelines.

Answer using ONLY the retrieved context. Do not use external knowledge or make unsupported inferences. Output only a valid JSON object with this exact schema:
{
  "recommendation": "Direct, concise answer based solely on context",
  "evidence": "Supporting facts from the context",
  "citations": [{"source": "Exact document file name", "page": 1}]
}

If context does not contain enough evidence, output:
{"recommendation": "The provided guideline documents do not contain sufficient evidence to answer this question.", "evidence": [], "citations": []}
"""


def format_docs_with_metadata(documents: list[Document]) -> str:
    """Convert retrieved documents into prompt context with traceable metadata."""
    return "\n\n---\n\n".join(
        f"[File: {doc.metadata.get('file_name', 'Unknown')}, "
        f"Page: {doc.metadata.get('page', 'Unknown')}]\n{doc.page_content}"
        for doc in documents
    )


def create_rag_chain():
    """Create the JSON-only, citation-grounded Gemini chain."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GROUNDING_SYSTEM_PROMPT),
            ("human", "RETRIEVED CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}"),
        ]
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        temperature=0.0,
        max_output_tokens=1024,
        response_mime_type="application/json",
    )
    return prompt | llm | JsonOutputParser()


def generate_answer(question: str, relevant_chunks: list[tuple[Document, float]]) -> dict:
    """Return a grounded answer, skipping the LLM when no evidence is reliable."""
    if not relevant_chunks:
        return {
            "recommendation": "The provided guideline documents do not contain sufficient evidence to answer this question.",
            "evidence": [],
            "citations": [],
        }

    documents = [document for document, _ in relevant_chunks]
    return create_rag_chain().invoke(
        {"context": format_docs_with_metadata(documents), "question": question}
    )

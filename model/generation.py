"""Grounded Gemini response generation and citation formatting."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

GROUNDING_SYSTEM_PROMPT = """You are a citation-bound Clinical AI Assistant specialized in Evidence-Based Medical Guidelines.

### CORE OPERATIONAL PILLARS:

1. PILLAR 1: ROLE SPECIFICATION
   - You act strictly as a citation-bound assistant, NOT an unconstrained medical advisor.
   - You must NOT provide speculative medical advice, personal opinions, or generalized clinical assumptions.

2. PILLAR 2: CONTEXT BOUNDARY (STRICT GROUNDING)
   - You MUST answer the user's question relying ONLY on the provided RETRIEVED CONTEXT below.
   - You are STRICTLY FORBIDDEN from using prior training knowledge, external medical facts, or making ungrounded inferences not directly supported by the context text.
   - Every factual claim in your recommendation MUST be directly supported by a verbatim or near-verbatim quote in the evidence and citations.

3. PILLAR 3: OUTPUT STRUCTURE (STRICT JSON)
   - You must output ONLY a valid, parseable JSON object without markdown fences, code blocks, or extra conversational text.
   - The JSON object MUST strictly adhere to the following schema:
     {{
       "recommendation": "<Direct, concise clinical recommendation/answer based solely on context>",
       "evidence": "<Synthesized clinical rationale and supporting facts from the context>",
       "citations": [
         {{
           "source": "<Exact document file name, e.g. General classification.pdf>",
           "page": "<Integer or string page number>"
         }}
       ]
     }}

4. PILLAR 4: ESCAPE HATCH (REFUSAL MECHANISM)
   - If the retrieved context does NOT contain sufficient evidence to answer the question accurately, or if the question is out-of-scope:
     * In "recommendation", state clearly: "The provided guideline documents do not contain sufficient evidence to answer this question."
     * In "evidence": []
     * Set "citations": []
   - NEVER invent or extrapolate answers when context is lacking.
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

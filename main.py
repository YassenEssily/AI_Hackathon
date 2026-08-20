import os
from pathlib import Path

import spaces  # استيراد مكتبة spaces الخاصة بـ ZeroGPU
import gradio as gr

from dotenv import load_dotenv

from model.generation import generate_answer
from model.retriever import get_or_build_vector_store, retrieve_relevant_chunks

# تحميل البيئة وقراءة المفاتيح من Hugging Face Secrets / env variables
load_dotenv()
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
VECTOR_DATABASE_DIRECTORY = PROJECT_ROOT / "rag_vector_db"


def load_vector_store():
    """Create the database on first use and reuse it."""
    return get_or_build_vector_store(DATA_DIRECTORY, VECTOR_DATABASE_DIRECTORY)


@spaces.GPU  # ربط تخصيص الـ GPU بالدالة التي تنفذ الاسترجاع والتوليد
def gradio_wrapper(question: str):
    if not question.strip():
        return "Please enter a question first.", "", "", ""

    relevant_chunks = retrieve_relevant_chunks(load_vector_store(), question)
    answer = generate_answer(question, relevant_chunks)

    recommendation = answer.get("recommendation", "No answer was returned.")

    evidence = answer.get("evidence", [])
    evidence_text = (
        "\n".join([f"- {item}" for item in evidence])
        if isinstance(evidence, list)
        else str(evidence)
    )

    citations = answer.get("citations", [])
    citations_text = "\n".join(
        [
            f"- **{c.get('source', 'Unknown')}** — page {c.get('page', 'Unknown')}"
            for c in citations
        ]
    )

    chunks_text = ""
    if relevant_chunks:
        for document, score in relevant_chunks:
            source = document.metadata.get("file_name", "Unknown source")
            page = document.metadata.get("page", "Unknown")
            chunks_text += f"**{source}** — page {page} | relevance: {score:.2f}\n{document.page_content.removeprefix('passage: ')}\n\n"

    return recommendation, evidence_text, citations_text, chunks_text


def main() -> None:
    with gr.Blocks(title="Clinical Guideline Assistant") as demo:
        gr.Markdown(
            "# ⚕️ Clinical Guideline Assistant\n"
        )

        question = gr.Textbox(
            label="Ask a question about the uploaded guidelines",
            placeholder="Example: In which age group is glibenclamide not recommended?",
        )
        submit = gr.Button("Ask")

        answer_out = gr.Markdown(label="Answer")
        evidence_out = gr.Markdown(label="Evidence")
        citations_out = gr.Markdown(label="Sources")
        with gr.Accordion("Retrieved passages", open=False):
            chunks_out = gr.Markdown()

        submit.click(
            fn=gradio_wrapper,
            inputs=question,
            outputs=[answer_out, evidence_out, citations_out, chunks_out],
        )

    demo.launch()


if __name__ == "__main__":
    main()

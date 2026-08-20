"""PDF parsing, cleanup, and chunking utilities."""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_parse import LlamaParse


def clean_text_content(text: str) -> str:
    """Remove HTML remnants and repeated page/header noise from parsed text."""
    if not text:
        return ""

    text = re.sub(r"<sup[^>]*>.*?</sup>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?(b|i|u|strong|em)>", "", text, flags=re.IGNORECASE)

    cleaned_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.isdigit() and len(stripped) <= 3:
            continue
        if "HEARTS – D" in stripped or "HEARTS - D" in stripped:
            continue
        if "DEFINITION AND DIAGNOSIS OF DIABETES MELLITUS AND INTERMEDIATE HYPERGLYCEMIA" in stripped.upper():
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def parse_pdf(file_path: Path) -> list[Document]:
    """Parse one PDF with LlamaParse and retain its filename/page metadata."""
    parser = LlamaParse(result_type="markdown", verbose=True, language="en")
    parsed_pages = parser.load_data(str(file_path))
    documents = []
    for page_number, page in enumerate(parsed_pages, start=1):
        documents.append(
            Document(
                page_content=clean_text_content(page.text),
                metadata={"file_name": file_path.name, "page": page_number},
            )
        )
    return documents


def ingest_pdf_directory(data_directory: str | Path) -> list[Document]:
    """Parse every PDF in *data_directory* into page-level documents."""
    pdf_files = sorted(Path(data_directory).glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {Path(data_directory).resolve()}")

    documents = []
    for pdf_file in pdf_files:
        print(f"Parsing: {pdf_file.name}")
        documents.extend(parse_pdf(pdf_file))
    print(f"Parsed {len(documents)} pages from {len(pdf_files)} PDF file(s).")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split page documents into retrieval-friendly overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", "؟", "!", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.page_content = f"passage: {chunk.page_content}"
    print(f"Created {len(chunks)} chunks.")
    return chunks

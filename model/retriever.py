"""Embedding, Chroma vector-store, and evidence retrieval functions."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from model.ingestion import chunk_documents, ingest_pdf_directory


def create_embeddings() -> HuggingFaceEmbeddings:
    """Create multilingual E5 embeddings using CUDA when available."""
    device = os.getenv("EMBEDDING_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(
    documents: list[Document], persist_directory: str | Path
) -> Chroma:
    """Chunk documents and create a persistent Chroma vector store."""
    chunks = chunk_documents(documents)
    return Chroma.from_documents(
        documents=chunks,
        embedding=create_embeddings(),
        persist_directory=str(persist_directory),
    )


def load_vector_store(persist_directory: str | Path) -> Chroma:
    """Open an existing persistent Chroma vector store."""
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=create_embeddings(),
    )


def get_vector_store(
    documents: list[Document], persist_directory: str | Path
) -> Chroma:
    """Load the database when present; otherwise build it from parsed documents."""
    database_path = Path(persist_directory)
    if database_path.exists() and any(database_path.iterdir()):
        print("Loading existing vector database.")
        return load_vector_store(database_path)
    print("Building vector database.")
    return build_vector_store(documents, database_path)


def get_or_build_vector_store(
    data_directory: str | Path, persist_directory: str | Path
) -> Chroma:
    """Reuse a stored database, parsing PDFs only when the database is absent."""
    database_path = Path(persist_directory)
    if database_path.exists() and any(database_path.iterdir()):
        print("Loading existing vector database.")
        return load_vector_store(database_path)
    return get_vector_store(ingest_pdf_directory(data_directory), database_path)


def retrieve_relevant_chunks(
    vector_store: Chroma, question: str, k: int = 5, threshold: float = 0.70
) -> list[tuple[Document, float]]:
    """Retrieve only chunks whose normalized relevance score meets *threshold*."""
    results = vector_store.similarity_search_with_relevance_scores(
        f"query: {question}", k=k
    )
    return [(document, score) for document, score in results if score >= threshold]

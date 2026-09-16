from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol, Sequence

import numpy as np

from .models import DocumentChunk, RetrievedChunk


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def chunk_document(
    text: str,
    document_id: str,
    chunk_size: int = 700,
    overlap: int = 100,
) -> list[DocumentChunk]:
    """Split on nearby whitespace while retaining source character offsets."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")
    chunks: list[DocumentChunk] = []
    start = 0
    while start < len(text):
        target_end = min(len(text), start + chunk_size)
        end = target_end
        if target_end < len(text):
            boundary = text.rfind(" ", start, target_end)
            if boundary > start:
                end = boundary
        content = text[start:end].strip()
        if content:
            actual_start = text.find(content, start, end + 1)
            actual_end = actual_start + len(content)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document_id}:{len(chunks)}",
                    document_id=document_id,
                    text=content,
                    start=actual_start,
                    end=actual_end,
                )
            )
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class OpenAIEmbeddingProvider:
    def __init__(self, model: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError('Install the OpenAI extra: pip install -e ".[openai]"') from exc
        self.client = OpenAI()
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype="float32")
        response = self.client.embeddings.create(input=list(texts), model=self.model)
        ordered = sorted(response.data, key=lambda item: item.index)
        return np.asarray([item.embedding for item in ordered], dtype="float32")


class HashEmbeddingProvider:
    """Deterministic local embedding for tests; not intended for evaluation claims."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype="float32")
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.casefold()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                index = int.from_bytes(digest, "little") % self.dimensions
                matrix[row, index] += 1.0
        return matrix


class FAISSKnowledgeBase:
    def __init__(self, embedder: EmbeddingProvider):
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError('Install the RAG extra: pip install -e ".[rag]"') from exc
        self.faiss = faiss
        self.embedder = embedder
        self.chunks: list[DocumentChunk] = []
        self.index = None

    def add_documents(self, documents: dict[str, str], chunk_size: int = 700, overlap: int = 100) -> None:
        new_chunks = [
            chunk
            for document_id, text in documents.items()
            for chunk in chunk_document(text, document_id, chunk_size, overlap)
        ]
        if not new_chunks:
            return
        vectors = _normalize(self.embedder.embed([chunk.text for chunk in new_chunks]))
        if self.index is None:
            self.index = self.faiss.IndexFlatIP(vectors.shape[1])
        elif self.index.d != vectors.shape[1]:
            raise ValueError("Embedding dimensions changed after index creation.")
        self.index.add(vectors)
        self.chunks.extend(new_chunks)

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if self.index is None or not self.chunks or top_k <= 0:
            return []
        query_vector = _normalize(self.embedder.embed([query]))
        scores, indices = self.index.search(query_vector, min(top_k, len(self.chunks)))
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            chunk = self.chunks[int(index)]
            results.append(RetrievedChunk(**chunk.model_dump(), score=float(score)))
        return results


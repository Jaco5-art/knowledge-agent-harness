from __future__ import annotations

from dataclasses import dataclass

from .evaluation import EvaluationSample
from .models import RetrievedChunk
from .rag import EmbeddingProvider, FAISSKnowledgeBase
from .redocred_predictor import IndexedRelation


def relation_query(
    sample: EvaluationSample,
    relation: IndexedRelation,
    relation_map: dict[str, str],
) -> str:
    entities = sample.benchmark_metadata.get("entities", [])
    head = entities[relation.head_index]
    tail = entities[relation.tail_index]
    label = relation_map.get(relation.relation_id, relation.relation_id)
    return f"{head} {label} {tail}"


@dataclass
class FAISSRelationEvidenceRetriever:
    """Build one document-scoped index and retrieve evidence per relation."""

    embedder: EmbeddingProvider
    relation_map: dict[str, str]
    top_k: int = 3
    chunk_size: int = 350
    overlap: int = 75

    def __post_init__(self) -> None:
        self._sample_id: str | None = None
        self._knowledge_base: FAISSKnowledgeBase | None = None

    def _index(self, sample: EvaluationSample) -> FAISSKnowledgeBase:
        if self._knowledge_base is None or self._sample_id != sample.sample_id:
            knowledge_base = FAISSKnowledgeBase(self.embedder)
            knowledge_base.add_documents(
                {sample.sample_id: sample.text},
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            )
            self._knowledge_base = knowledge_base
            self._sample_id = sample.sample_id
        return self._knowledge_base

    def retrieve(
        self,
        sample: EvaluationSample,
        candidates: list[IndexedRelation],
    ) -> dict[int, list[RetrievedChunk]]:
        knowledge_base = self._index(sample)
        return {
            index: knowledge_base.search(
                relation_query(sample, candidate, self.relation_map),
                top_k=self.top_k,
            )
            for index, candidate in enumerate(candidates)
        }


def retrieval_metrics(
    samples: list[EvaluationSample],
    retriever: FAISSRelationEvidenceRetriever,
) -> dict[str, float | int]:
    """Measure whether annotated gold evidence appears in the top-k chunks."""

    gold_relations = 0
    covered_relations = 0
    for sample in samples:
        metadata = sample.benchmark_metadata.get("indexed_relations", [])
        relations = [
            IndexedRelation(
                head_index=item["head_index"],
                tail_index=item["tail_index"],
                relation_id=item["relation_id"],
                evidence_quote="",
            )
            for item in metadata
        ]
        retrieved = retriever.retrieve(sample, relations)
        for index, item in enumerate(metadata):
            quotes = [quote for quote in item.get("evidence_quotes", []) if quote]
            if not quotes:
                continue
            gold_relations += 1
            chunks = retrieved.get(index, [])
            if any(quote in chunk.text for quote in quotes for chunk in chunks):
                covered_relations += 1
    recall = covered_relations / gold_relations if gold_relations else 0.0
    return {
        "gold_relations_with_evidence": gold_relations,
        "covered_relations": covered_relations,
        "evidence_recall_at_k": round(recall, 4),
        "top_k": retriever.top_k,
    }

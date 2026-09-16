from knowledge_agents.models import Evidence, ExtractedFact, FactType
from knowledge_agents.rag import FAISSKnowledgeBase, HashEmbeddingProvider, chunk_document
from knowledge_agents.workflow import run_workflow


def test_chunking_preserves_offsets():
    text = "Alpha partnership announcement. Beta unrelated paragraph."
    chunks = chunk_document(text, "doc-1", chunk_size=35, overlap=5)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert text[chunk.start:chunk.end] == chunk.text


def test_faiss_retrieves_relevant_evidence():
    kb = FAISSKnowledgeBase(HashEmbeddingProvider())
    kb.add_documents(
        {
            "business": "Microsoft invested in OpenAI in 2023 and expanded the partnership.",
            "weather": "Heavy rain was recorded in London during the weekend.",
        },
        chunk_size=100,
        overlap=10,
    )
    result = kb.search("Microsoft OpenAI investment", top_k=1)
    assert result[0].document_id == "business"


class RetrievedEvidenceProvider:
    def extract(self, source_text, fact_type):
        if fact_type is not FactType.EVENT:
            return []
        return [
            ExtractedFact(
                fact_id="external-1",
                fact_type=FactType.EVENT,
                subject="Microsoft",
                predicate="invested",
                object="OpenAI",
                time="2023",
                evidence=[Evidence(quote="Microsoft invested in OpenAI in 2023")],
                extraction_confidence=0.9,
                source_agent="event_agent",
            )
        ]

    def correct(self, source_text, facts, results, correction_round):
        return facts


def test_retrieval_context_is_used_by_verifier():
    kb = FAISSKnowledgeBase(HashEmbeddingProvider())
    kb.add_documents({"evidence": "Microsoft invested in OpenAI in 2023."})
    state = run_workflow(
        "A user asks about an investment event.",
        "extract investment event",
        RetrievedEvidenceProvider(),
        knowledge_base=kb,
    )
    assert state["verified_facts"][0].fact_id == "external-1"

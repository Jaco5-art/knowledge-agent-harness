from knowledge_agents.agents.fusion import fuse_knowledge
from knowledge_agents.agents.verification import verify_fact
from knowledge_agents.models import Evidence, ExtractedFact, FactType


SOURCE = "Microsoft invested in OpenAI in 2023."


def make_fact(predicate: str, evidence: str = SOURCE) -> ExtractedFact:
    return ExtractedFact(
        fact_id="f1",
        fact_type=FactType.EVENT,
        subject="Microsoft",
        predicate=predicate,
        object="OpenAI",
        time="2023",
        evidence=[Evidence(quote=evidence)],
        extraction_confidence=0.9,
        source_agent="event_agent",
    )


def test_supported_fact_passes():
    result = verify_fact(make_fact("invested"), SOURCE)
    assert result.verdict == "supported"


def test_acquisition_claim_is_flagged():
    result = verify_fact(make_fact("acquired"), SOURCE)
    assert result.verdict == "needs_correction"


def test_fabricated_quote_is_rejected():
    result = verify_fact(make_fact("invested", "Microsoft acquired OpenAI."), SOURCE)
    assert result.verdict == "unsupported"


def test_fusion_keeps_highest_confidence_duplicate():
    low = make_fact("invested")
    low.extraction_confidence = 0.6
    high = make_fact("invested")
    high.fact_id = "f2"
    state = {"entity_facts": [], "event_facts": [low, high], "relationship_facts": []}
    fused = fuse_knowledge(state)["fused_facts"]
    assert len(fused) == 1
    assert fused[0].fact_id == "f2"


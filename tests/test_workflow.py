from knowledge_agents.models import Evidence, ExtractedFact, FactType
from knowledge_agents.workflow import run_workflow


SOURCE = "Microsoft invested in OpenAI in 2023."


class CorrectingProvider:
    def extract(self, source_text, fact_type):
        if fact_type is not FactType.EVENT:
            return []
        return [
            ExtractedFact(
                fact_id="event-1",
                fact_type=FactType.EVENT,
                subject="Microsoft",
                predicate="acquired",
                object="OpenAI",
                time="2023",
                evidence=[Evidence(quote=SOURCE)],
                extraction_confidence=0.8,
                source_agent="event_agent",
            )
        ]

    def correct(self, source_text, facts, results, correction_round):
        corrected = facts[0].model_copy(deep=True)
        corrected.predicate = "invested"
        corrected.correction_round = correction_round
        return [corrected]


def test_graph_corrects_then_reverifies():
    state = run_workflow(
        SOURCE,
        "extract investment event",
        CorrectingProvider(),
        max_correction_rounds=2,
    )
    assert state["correction_round"] == 1
    assert state["verified_facts"][0].predicate == "invested"
    assert state["rejected_facts"] == []


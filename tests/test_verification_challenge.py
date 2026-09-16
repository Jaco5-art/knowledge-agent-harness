from knowledge_agents.evaluation import EvaluationSample
from knowledge_agents.redocred_verification import RelationVerdict
from knowledge_agents.verification_challenge import run_challenge


class ChallengeVerifier:
    def verify(self, sample, candidates):
        return [
            RelationVerdict(
                candidate_index=i,
                verdict="supported" if c.relation_id == "P159" and c.head_index == 0 else "unsupported",
                confidence=1.0,
                reason="test decision",
            )
            for i, c in enumerate(candidates)
        ]


def test_controlled_challenge_measures_retention_and_detection():
    sample = EvaluationSample(
        sample_id="s1",
        text="Apex is based in Paris.",
        benchmark_metadata={
            "entities": ["Apex", "Paris"],
            "indexed_relations": [{
                "head_index": 0, "tail_index": 1, "relation_id": "P159",
                "evidence_quotes": ["Apex is based in Paris."],
            }],
        },
    )
    metrics, rows = run_challenge(
        [sample], {"P112": "founded by", "P159": "headquarters location"}, ChallengeVerifier()
    )
    assert metrics["clean_retention_rate"] == 1.0
    assert metrics["error_detection_rate"] == 1.0
    assert len(rows) == 2

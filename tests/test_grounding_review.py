import csv

from knowledge_agents.evaluation import EvaluationSample
from knowledge_agents.grounding_review import (
    audit_review_csv,
    build_review_rows,
    markdown_audit_report,
    write_review_csv,
)


def test_grounding_review_requires_human_label_and_reviewer(tmp_path):
    sample = EvaluationSample(
        sample_id="s1", text="Apex is based in Paris.",
        benchmark_metadata={"title": "Apex", "entities": ["Apex", "Paris"]},
    )
    trace = [
        {"condition": condition, "sample_id": "s1", "case_type": "clean",
         "head_index": 0, "tail_index": 1, "candidate_relation_id": "P159",
         "evidence_quote": sample.text, "verdict": "supported", "reason": "explicit"}
        for condition in ("verification_only", "rag_verification")
    ]
    rows = build_review_rows([sample], trace, {"P159": "headquarters location"})
    assert len(rows) == 1
    assert rows[0]["review_status"] == "needs_human_review"
    path = write_review_csv(tmp_path / "review.csv", rows)
    initial = audit_review_csv(path)
    assert initial["ready_for_reporting"] is False
    assert initial["human_review"]["coverage"] == 0.0
    assert initial["human_review"]["interpretation"] == "no_valid_human_review"

    rows[0].update(
        review_status="approved", human_grounding_label="directly_supported", reviewed_by="Reviewer"
    )
    write_review_csv(path, rows)
    audited = audit_review_csv(path)
    assert audited["ready_for_reporting"] is True
    assert audited["approved_label_counts"] == {"directly_supported": 1}
    assert audited["human_review"]["coverage"] == 1.0
    assert audited["human_review"]["interpretation"] == "complete_human_review"
    assert audited["human_grounded_verifier_evaluation"]["verification_only"]["accuracy"] == 1.0
    assert "Human-Grounded Verification Report" in markdown_audit_report(audited)


def test_grounding_review_reports_valid_partial_human_review(tmp_path):
    rows = [
        {
            "review_status": "approved",
            "human_grounding_label": "annotation_anomaly",
            "reviewed_by": "Human Reviewer",
        },
        {
            "review_status": "needs_human_review",
            "human_grounding_label": "directly_supported",
            "reviewed_by": "AI pre-review",
        },
    ]
    path = write_review_csv(tmp_path / "partial.csv", rows)
    audited = audit_review_csv(path)
    assert audited["ready_for_reporting"] is False
    assert audited["approved_label_counts"] == {"annotation_anomaly": 1}
    assert audited["human_review"] == {
        "approved_rows": 1,
        "coverage": 0.5,
        "approved_rows_valid": True,
        "interpretation": "valid_partial_human_review",
    }

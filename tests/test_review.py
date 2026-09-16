import pytest

from knowledge_agents.benchmark import build_synthetic_benchmark
from knowledge_agents.review import audit_samples, export_review_csv, import_review_csv


def test_unreviewed_benchmark_is_not_final_ready(tmp_path):
    samples = build_synthetic_benchmark()
    report = audit_samples(samples)
    assert report["issue_count"] == 0
    assert report["ready_for_final_reporting"] is False
    assert report["review_status"]["needs_human_review"] == 100


def test_review_csv_round_trip(tmp_path):
    samples = build_synthetic_benchmark()[:2]
    path = export_review_csv(samples, tmp_path / "review.csv")
    restored = import_review_csv(path)
    assert [sample.model_dump() for sample in restored] == [sample.model_dump() for sample in samples]


def test_approval_requires_reviewer(tmp_path):
    sample = build_synthetic_benchmark()[0]
    path = export_review_csv([sample], tmp_path / "review.csv")
    content = path.read_text(encoding="utf-8-sig").replace("needs_human_review", "approved")
    path.write_text(content, encoding="utf-8-sig")
    with pytest.raises(ValueError):
        import_review_csv(path)


from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation import EvaluationSample, LabeledFact


ORGS_A = ["Apex", "Beacon", "Cobalt", "Delta", "Evergreen", "Falcon", "Gemini", "Helios", "Ion", "Juniper"]
ORGS_B = ["Kite", "Lumen", "Meridian", "Nimbus", "Orion", "Pioneer", "Quartz", "Rivian", "Solace", "Titan"]
PEOPLE = ["Alice Chen", "Ben Ortiz", "Chloe Singh", "Daniel Kim", "Elena Rossi", "Farah Ali", "Grace Liu", "Hugo Martin", "Iris Wang", "Jon Bell"]


def build_synthetic_benchmark() -> list[EvaluationSample]:
    """Create 100 controlled cases. These are synthetic and require review before final reporting."""
    samples: list[EvaluationSample] = []
    for i, (org_a, org_b, person) in enumerate(zip(ORGS_A, ORGS_B, PEOPLE), start=1):
        cases = [
            (f"{org_a} invested in {org_b} in 2025.", [(org_a, "invested in", org_b)], "easy", ["investment"]),
            (f"{org_a} acquired {org_b} after regulatory approval.", [(org_a, "acquired", org_b)], "easy", ["acquisition"]),
            (f"{org_a} did not acquire {org_b}; the companies signed a research partnership.", [(org_a, "signed research partnership with", org_b)], "hard", ["negation", "relation_contrast"]),
            (f"{person} became chief technology officer of {org_a} in March.", [(person, "became chief technology officer of", org_a)], "medium", ["appointment", "time"]),
            (f"{org_b} opened a laboratory in Berlin in 2024.", [(org_b, "opened", "laboratory")], "easy", ["event", "location", "time"]),
            (f"Researchers from {org_a} and {org_b} jointly published the study.", [(org_a, "jointly published study with", org_b)], "medium", ["collaboration"]),
            (f"No funding relationship between {org_a} and {org_b} was reported.", [], "hard", ["negation", "no_positive_fact"]),
            (f"{org_a} supplied batteries to {org_b} but did not design its vehicles.", [(org_a, "supplied batteries to", org_b)], "hard", ["negation", "multiple_clauses"]),
            (f"A rumor claimed {org_a} bought {org_b}, but both companies denied the acquisition.", [("rumor", "claimed acquisition of", f"{org_b} by {org_a}"), (f"{org_a} and {org_b}", "denied", "acquisition")], "hard", ["rumor", "attribution", "denial"]),
            (f"{org_b} introduced audio descriptions and captions for exhibition {i}.", [(org_b, "introduced", "audio descriptions"), (org_b, "introduced", "captions")], "medium", ["multiple_objects", "accessibility"]),
        ]
        for case_index, (text, facts, difficulty, phenomena) in enumerate(cases, start=1):
            samples.append(
                EvaluationSample(
                    sample_id=f"syn-{i:02d}-{case_index:02d}",
                    text=text,
                    gold_facts=[LabeledFact(subject=s, predicate=p, object=o) for s, p, o in facts],
                    source_type="controlled_synthetic",
                    difficulty=difficulty,
                    phenomena=phenomena,
                    review_status="needs_human_review",
                )
            )
    return samples


def write_benchmark(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    samples = build_synthetic_benchmark()
    destination.write_text("".join(sample.model_dump_json() + "\n" for sample in samples), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the controlled 100-sample synthetic benchmark.")
    parser.add_argument("--output", default="data/eval_samples_synthetic_100.jsonl")
    args = parser.parse_args()
    path = write_benchmark(args.output)
    print(f"Wrote 100 synthetic samples to {path}")


if __name__ == "__main__":
    main()

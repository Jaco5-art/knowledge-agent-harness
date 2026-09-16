from __future__ import annotations

import argparse
import json

from .provider import OpenAIProvider
from .workflow import run_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and verify facts from text.")
    parser.add_argument("text")
    parser.add_argument("--task", default="extract entities, events and relationships")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    result = run_workflow(args.text, args.task, OpenAIProvider(args.model))
    payload = {
        "verified_facts": [fact.model_dump() for fact in result.get("verified_facts", [])],
        "rejected_facts": [fact.model_dump() for fact in result.get("rejected_facts", [])],
        "correction_round": result.get("correction_round", 0),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()


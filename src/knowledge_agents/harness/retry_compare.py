
"""Offline retry ablation using the real Harness and a fixture adapter."""
import asyncio
from datetime import datetime
import json
from pathlib import Path
import tempfile
import time

from .runtime import Harness
from .store import StateStore
from .contracts import RuntimeConfig
from .adapters import FixtureAdapter, registry_for, DEMO_SOURCE
from .tools import ToolRegistry, ToolSpec, TransientError


async def run_case(root, label, inject_failure, max_attempts):
    base = registry_for(FixtureAdapter(False))
    registry = ToolRegistry()
    counters = {"handler_calls": 0, "injected_failures": 0}

    for stage in ("extract", "retrieve", "verify", "correct"):
        original = base.get(stage, stage)

        def wrap(spec, stage_name):
            async def handler(request):
                counters["handler_calls"] += 1
                if (
                    inject_failure
                    and stage_name == "extract"
                    and counters["injected_failures"] == 0
                ):
                    counters["injected_failures"] += 1
                    raise TransientError("Synthetic offline transient failure")
                return await spec.handler(request)
            return handler

        registry.register(ToolSpec(
            name=stage,
            version=original.version + ":retry-ablation-v1",
            input_model=original.input_model,
            output_model=original.output_model,
            handler=wrap(original, stage),
            allowed_stages=frozenset({stage}),
        ))

    store = StateStore(root / (label + ".sqlite"))
    try:
        harness = Harness(store, registry)
        state = harness.create(
            DEMO_SOURCE,
            "extract entities and relationships",
            "offline-evaluation",
            config=RuntimeConfig(
                max_attempts=max_attempts,
                retry_delay_seconds=0,
            ),
        )
        started = time.perf_counter()
        final = await asyncio.wait_for(
            harness.run(state.session_id, "offline-evaluation"),
            timeout=60,
        )
        elapsed = (time.perf_counter() - started) * 1000

        # Compare semantic content, excluding random candidate IDs.
        facts = sorted(
            (
                final.candidates[cid].fact.model_dump()
                for cid in final.accepted
            ),
            key=lambda fact: json.dumps(fact, sort_keys=True),
        )
        return {
            "case": label,
            "injected_failure": inject_failure,
            "max_attempts": max_attempts,
            "status": final.status,
            "stage": final.stage,
            "accepted_count": len(final.accepted),
            "accepted_facts": facts,
            "handler_calls": counters["handler_calls"],
            "injected_failures": counters["injected_failures"],
            "elapsed_ms": elapsed,
            "pending_review_reasons": [
                item.reason for item in final.reviews
                if not item.resolved
            ],
        }
    finally:
        store.close()


async def evaluate():
    cases = [
        ("clean_no_retry", False, 1),
        ("clean_with_retry", False, 2),
        ("fault_no_retry", True, 1),
        ("fault_with_retry", True, 2),
    ]
    results = []
    with tempfile.TemporaryDirectory() as folder:
        for label, inject, attempts in cases:
            result = await run_case(
                Path(folder), label, inject, attempts
            )
            results.append(result)
            print(
                f"{label}: {result['status']}; "
                f"handler_calls={result['handler_calls']}"
            )

    clean_off, clean_on, fault_off, fault_on = results
    assert clean_off["status"] == clean_on["status"] == "completed"
    assert clean_off["accepted_count"] > 0
    assert clean_off["accepted_facts"] == clean_on["accepted_facts"]
    assert clean_off["handler_calls"] == clean_on["handler_calls"]

    assert fault_off["status"] == "needs_review"
    assert fault_off["accepted_count"] == 0
    assert "attempts_exhausted" in fault_off["pending_review_reasons"]

    assert fault_on["status"] == "completed"
    assert fault_on["accepted_facts"] == clean_on["accepted_facts"]
    assert fault_on["handler_calls"] == clean_on["handler_calls"] + 1
    assert fault_off["injected_failures"] == fault_on["injected_failures"] == 1

    report = {
        "evaluation": "controlled_transient_error_retry_ablation",
        "api_calls": 0,
        "adapter": "FixtureAdapter",
        "results": results,
        "checks": {
            "clean_behavior_preserved": True,
            "no_retry_enters_review": True,
            "retry_recovers": True,
            "recovered_facts_match_clean_control": True,
            "recovery_adds_one_handler_call": True,
        },
        "limitations": [
            "One deterministic fixture and one injected error type.",
            "Not a production recovery-rate estimate.",
            "No timeout, fallback prompt, or fallback tool evaluated.",
            "Fixture latency does not estimate live API latency.",
        ],
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path(f"retry_ablation_{stamp}.json")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print("EVALUATION OK: retry ablation checks passed; 0 API calls.")
    print(f"Report: {output.resolve()}")


def main():
    asyncio.run(evaluate())


if __name__ == "__main__":
    main()

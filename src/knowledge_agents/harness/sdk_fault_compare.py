
"""Offline fault-path comparison of Harness and SDKController."""
import asyncio
from datetime import datetime
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from .runtime import Harness
from .sdk_controller import SDKController, summary, forbidden
from .store import StateStore
from .contracts import RuntimeConfig
from .adapters import FixtureAdapter, registry_for, DEMO_SOURCE
from .tools import ToolRegistry, ToolSpec, TransientError


def fault_registry():
    base = registry_for(FixtureAdapter(False))
    registry = ToolRegistry()
    counts = {"injected": 0, "handler_calls": 0}

    def wrap(spec, stage):
        async def handler(request):
            counts["handler_calls"] += 1
            if stage == "extract" and counts["injected"] == 0:
                counts["injected"] += 1
                raise TransientError("Controlled offline failure")
            return await spec.handler(request)
        return handler

    for stage in ("extract", "retrieve", "verify", "correct"):
        spec = base.get(stage, stage)
        registry.register(ToolSpec(
            name=spec.name,
            version=spec.version + ":fault-parity-v1",
            input_model=spec.input_model,
            output_model=spec.output_model,
            handler=wrap(spec, stage),
            allowed_stages=spec.allowed_stages,
            read_only=spec.read_only,
        ))
    return registry, counts


async def run_case(root, use_sdk, attempts):
    label = f"{'sdk' if use_sdk else 'harness'}_attempts_{attempts}"
    registry, counts = fault_registry()
    store = StateStore(root / (label + ".sqlite"))
    try:
        harness = Harness(store, registry)
        initial = harness.create(
            DEMO_SOURCE,
            "extract entities and relationships",
            "fault-parity",
            config=RuntimeConfig(
                max_attempts=attempts,
                retry_delay_seconds=0,
            ),
        )
        if use_sdk:
            controller = SDKController(store, registry)
            with patch.object(Harness, "_run", forbidden):
                state = await controller.run(initial.session_id, "fault-parity")
        else:
            state = await harness.run(initial.session_id, "fault-parity")

        result = summary(state)
        result.update(counts)
        assert counts["injected"] == 1
        assert counts["handler_calls"] == state.call_count
        print(
            f"{label}: {state.status}; "
            f"tool_calls={state.call_count}"
        )
        return result
    finally:
        store.close()


async def evaluate():
    results = {}
    with tempfile.TemporaryDirectory() as folder:
        for attempts in (1, 2):
            direct = await run_case(Path(folder), False, attempts)
            sdk = await run_case(Path(folder), True, attempts)
            assert direct == sdk, (attempts, direct, sdk)

            if attempts == 1:
                assert direct["status"] == "needs_review"
                assert not direct["facts"]
                assert "attempts_exhausted" in direct["pending_reviews"]
                assert direct["handler_calls"] == 1
            else:
                assert direct["status"] == "completed"
                assert direct["facts"]
                assert not direct["pending_reviews"]
                assert direct["corrected_candidates"] > 0

            results[str(attempts)] = {
                "harness": direct,
                "sdk": sdk,
                "matching": True,
            }

    report = {
        "evaluation": "independent_scheduler_transient_fault_parity",
        "api_calls": 0,
        "results": results,
        "limitations": [
            "One injected transient error at the first extraction call.",
            "Retry implementation is shared between both schedulers.",
            "Does not compare framework-native retry capabilities.",
            "Cancellation, process restart and human-review resume are not tested.",
        ],
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = Path(f"sdk_fault_parity_{stamp}.json")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print("PARITY OK: transient-fault paths match; 0 API calls.")
    print(f"Report: {path.resolve()}")


def main():
    asyncio.run(asyncio.wait_for(evaluate(), timeout=120))


if __name__ == "__main__":
    main()

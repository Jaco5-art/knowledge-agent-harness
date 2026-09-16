
"""Offline checkpoint-resume check for the SDK controller."""
import asyncio
from datetime import datetime
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from agents.exceptions import MaxTurnsExceeded, UserError

from .runtime import Harness
from .sdk_controller import SDKController, summary, forbidden
from .store import StateStore
from .adapters import FixtureAdapter, registry_for, DEMO_SOURCE


async def evaluate():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)

        # Uninterrupted SDK control.
        store = StateStore(root / "control.sqlite")
        try:
            registry = registry_for(FixtureAdapter(False))
            harness = Harness(store, registry)
            initial = harness.create(
                DEMO_SOURCE,
                "extract entities and relationships",
                "resume-check",
            )
            controller = SDKController(store, registry)
            with patch.object(Harness, "_run", forbidden):
                control = summary(
                    await controller.run(initial.session_id, "resume-check")
                )
        finally:
            store.close()

        # Stop after one committed business step.
        db = root / "resume.sqlite"
        store = StateStore(db)
        try:
            registry = registry_for(FixtureAdapter(False))
            harness = Harness(store, registry)
            initial = harness.create(
                DEMO_SOURCE,
                "extract entities and relationships",
                "resume-check",
            )
            session_id = initial.session_id
            controller = SDKController(store, registry)
            stopped = False
            try:
                with patch.object(Harness, "_run", forbidden):
                    await controller.run(
                        session_id, "resume-check", max_steps=1
                    )
            except MaxTurnsExceeded:
                stopped = True
            except RuntimeError as exc:
                if str(exc) != "SDK step budget exhausted":
                    raise
                stopped = True
            except UserError as exc:
                cause = exc.__cause__
                if (
                    type(cause) is not RuntimeError
                    or str(cause) != "SDK step budget exhausted"
                ):
                    raise
                stopped = True

            assert stopped, "Expected the SDK step limit to stop execution"
            checkpoint = store.load(session_id, "resume-check")
            assert checkpoint.status == "running"
            assert checkpoint.extraction_index == 1
            assert checkpoint.call_count == 1

            checkpoint_info = {
                "version": checkpoint.version,
                "stage": checkpoint.stage,
                "extraction_index": checkpoint.extraction_index,
                "tool_calls": checkpoint.call_count,
            }
        finally:
            store.close()

        # Reopen persistence with a fresh registry and controller.
        store = StateStore(db)
        try:
            registry = registry_for(FixtureAdapter(False))
            controller = SDKController(store, registry)
            with patch.object(Harness, "_run", forbidden):
                resumed_state = await controller.run(
                    session_id, "resume-check"
                )
            resumed = summary(resumed_state)

            assert resumed == control, (resumed, control)
            assert resumed["status"] == "completed"
            assert resumed["corrected_candidates"] > 0

            # Re-running an already completed session must be a no-op.
            before = resumed_state.model_dump_json()
            with patch.object(Harness, "_run", forbidden):
                again = await controller.run(session_id, "resume-check")
            assert again.model_dump_json() == before
            assert controller.steps == 0
        finally:
            store.close()

    report = {
        "evaluation": "sdk_checkpoint_resume",
        "api_calls": 0,
        "checkpoint": checkpoint_info,
        "control": control,
        "resumed": resumed,
        "completed_session_noop": True,
        "limitations": [
            "Store closed and reopened in the same Python process.",
            "Tests committed business-state resume, not SDK conversation resume.",
            "Does not simulate an abrupt kill or an in-flight provider request.",
        ],
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path(f"sdk_resume_{stamp}.json")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print("RESUME OK: checkpoint recovery matches control; 0 API calls.")
    print(f"Report: {output.resolve()}")


def main():
    asyncio.run(asyncio.wait_for(evaluate(), timeout=120))


if __name__ == "__main__":
    main()


"""Compare direct and SDK-mediated tools under the same Harness."""
import asyncio
from datetime import datetime
import json
from pathlib import Path
import tempfile
import uuid

from agents import Agent, Runner, RunConfig
from agents.items import ModelResponse
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText,
)

from .sdk_runner_check import OfflineModel
from .sdk_bridge import BridgeContext, bridge_tool
from .runtime import Harness
from .store import StateStore
from .adapters import FixtureAdapter, registry_for, DEMO_SOURCE
from .tools import ToolRegistry, ToolSpec, ToolReply


class ToolExecutionModel(OfflineModel):
    def __init__(self, name, arguments):
        super().__init__()
        self.name = name
        self.arguments = arguments
        self.call_id = uuid.uuid4().hex
        self.tool_output = None

    async def get_response(self, *args, **kwargs):
        self.turns += 1
        if self.turns == 1:
            items = [ResponseFunctionToolCall(
                id="item_" + self.call_id,
                call_id=self.call_id,
                name=self.name,
                arguments=self.arguments,
                type="function_call",
            )]
        elif self.turns == 2:
            inputs = kwargs.get("input")
            if inputs is None and len(args) >= 2:
                inputs = args[1]
            outputs = [
                item for item in inputs
                if isinstance(item, dict)
                and item.get("type") == "function_call_output"
                and item.get("call_id") == self.call_id
            ]
            if len(outputs) != 1:
                raise AssertionError("Expected one correlated tool output")
            self.tool_output = outputs[0]["output"]
            self.received_tool_output = True
            items = [ResponseOutputMessage(
                id="message_" + self.call_id,
                type="message", role="assistant", status="completed",
                content=[ResponseOutputText(
                    type="output_text",
                    text=self.tool_output,
                    annotations=[],
                )],
            )]
        else:
            raise AssertionError("Unexpected model turn")
        return ModelResponse(
            output=items, usage=Usage(), response_id=None
        )


def build_registry(use_sdk, execution_log):
    base = registry_for(FixtureAdapter(False))
    result = ToolRegistry()

    for stage in ("extract", "retrieve", "verify", "correct"):
        spec = base.get(stage, stage)

        def make_handler(spec, stage):
            async def handler(request):
                if not use_sdk:
                    reply = await spec.handler(request)
                    execution_log.append({
                        "stage": stage, "model_turns": 0,
                    })
                    return reply

                model = ToolExecutionModel(
                    spec.name, request.model_dump_json()
                )
                agent = Agent(
                    name="Fixture_" + stage,
                    instructions="Execute the supplied registered tool.",
                    model=model,
                    tools=[bridge_tool(base, spec.name, stage)],
                )
                run = await Runner.run(
                    agent,
                    "Execute the fixture request.",
                    context=BridgeContext(stage=stage),
                    max_turns=3,
                    run_config=RunConfig(tracing_disabled=True),
                )
                assert model.turns == 2
                assert model.received_tool_output
                assert run.final_output == model.tool_output
                output = spec.output_model.model_validate_json(
                    model.tool_output
                )
                execution_log.append({
                    "stage": stage, "model_turns": model.turns,
                })
                # Synthetic fixture: provider usage is unmeasured.
                return ToolReply(output)
            return handler

        result.register(ToolSpec(
            name=stage,
            version=spec.version + (":sdk-fixture" if use_sdk else ":direct-fixture"),
            input_model=spec.input_model,
            output_model=spec.output_model,
            handler=make_handler(spec, stage),
            allowed_stages=spec.allowed_stages,
            read_only=spec.read_only,
        ))
    return result


async def run_case(root, use_sdk):
    name = "sdk_tools" if use_sdk else "direct_tools"
    log = []
    store = StateStore(root / (name + ".sqlite"))
    try:
        harness = Harness(store, build_registry(use_sdk, log))
        initial = harness.create(
            DEMO_SOURCE,
            "extract entities and relationships",
            "execution-comparison",
        )
        state = await harness.run(
            initial.session_id, "execution-comparison"
        )
        facts = sorted(
            [state.candidates[cid].fact.model_dump() for cid in state.accepted],
            key=lambda item: json.dumps(item, sort_keys=True),
        )
        return {
            "case": name,
            "status": state.status,
            "facts": facts,
            "tool_attempts": state.call_count,
            "corrected_candidates": sum(
                c.round > 0 for c in state.candidates.values()
            ),
            "execution_log": log,
            "pending_reviews": [
                r.reason for r in state.reviews if not r.resolved
            ],
        }
    finally:
        store.close()


async def evaluate():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        direct = await run_case(root, False)
        sdk = await run_case(root, True)

    assert direct["status"] == sdk["status"] == "completed", (direct, sdk)
    assert direct["facts"] and direct["facts"] == sdk["facts"]
    assert direct["tool_attempts"] == sdk["tool_attempts"]
    assert direct["corrected_candidates"] == sdk["corrected_candidates"]
    assert direct["corrected_candidates"] > 0
    assert not direct["pending_reviews"] and not sdk["pending_reviews"]

    direct_stages = [e["stage"] for e in direct["execution_log"]]
    sdk_stages = [e["stage"] for e in sdk["execution_log"]]
    assert direct_stages == sdk_stages
    assert {"extract", "retrieve", "verify", "correct"} <= set(sdk_stages)
    assert len(sdk_stages) == sdk["tool_attempts"]

    report = {
        "evaluation": "shared_harness_direct_vs_sdk_tool_execution",
        "api_calls": 0,
        "direct": direct,
        "sdk": sdk,
        "limitations": [
            "Both cases use the existing Harness state machine.",
            "The SDK model is scripted and does not make autonomous decisions.",
            "Not an independent-runtime or performance comparison.",
            "Provider token accounting is not evaluated.",
        ],
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = Path(f"sdk_task_check_{stamp}.json")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print("TASK OK: direct and SDK tool execution match; 0 API calls.")
    print(f"Facts: {len(sdk['facts'])}")
    print(f"Tool attempts per case: {sdk['tool_attempts']}")
    print(f"Corrected candidates: {sdk['corrected_candidates']}")
    print(f"Report: {path.resolve()}")


def main():
    asyncio.run(asyncio.wait_for(evaluate(), timeout=120))


if __name__ == "__main__":
    main()

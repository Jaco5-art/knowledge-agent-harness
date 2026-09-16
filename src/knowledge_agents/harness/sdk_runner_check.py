
"""Offline SDK Runner integration check; no provider or tracing uploads."""
import asyncio
import json
from datetime import datetime
from pathlib import Path

from agents import Agent, Runner, RunConfig
from agents.exceptions import MaxTurnsExceeded
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from pydantic import BaseModel, ConfigDict

from .sdk_bridge import BridgeContext, bridge_tool
from .tools import ToolRegistry, ToolSpec, ToolReply


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


class OfflineModel(Model):
    def __init__(self):
        self.turns = 0
        self.received_tool_output = False

    async def get_response(self, *args, **kwargs):
        self.turns += 1
        if self.turns == 1:
            items = [ResponseFunctionToolCall(
                id="fixture-tool-item",
                call_id="fixture-call-1",
                name="offline_check",
                arguments='{"value":4}',
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
                and item.get("call_id") == "fixture-call-1"
            ]
            assert len(outputs) == 1, "Runner did not return the tool output"
            value = json.loads(outputs[0]["output"])
            assert value == {"value": 5}, "Unexpected tool result"
            self.received_tool_output = True
            items = [ResponseOutputMessage(
                id="fixture-final-item",
                type="message",
                role="assistant",
                status="completed",
                content=[ResponseOutputText(
                    type="output_text",
                    text=json.dumps(value),
                    annotations=[],
                )],
            )]
        else:
            raise AssertionError("Unexpected extra model turn")

        return ModelResponse(
            output=items,
            usage=Usage(),
            response_id=None,
        )

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError("Offline check uses non-streaming Runner")
        yield


def prepare():
    calls = []

    async def handler(request):
        calls.append(request.value)
        return ToolReply(Output(value=request.value + 1))

    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="offline_check",
        version="fixture-1",
        input_model=Input,
        output_model=Output,
        handler=handler,
        allowed_stages=frozenset({"verify"}),
    ))
    model = OfflineModel()
    agent = Agent(
        name="OfflineRunnerCheck",
        instructions="Use the registered tool and return its result.",
        model=model,
        tools=[bridge_tool(registry, "offline_check", "verify")],
    )
    return agent, model, calls


async def evaluate():
    config = RunConfig(tracing_disabled=True)
    agent, model, calls = prepare()
    result = await Runner.run(
        agent,
        "Run the offline check.",
        context=BridgeContext(stage="verify"),
        max_turns=3,
        run_config=config,
    )
    assert json.loads(result.final_output) == {"value": 5}
    assert calls == [4]
    assert model.turns == 2
    assert model.received_tool_output

    limited_agent, limited_model, limited_calls = prepare()
    try:
        await Runner.run(
            limited_agent,
            "Run the offline check.",
            context=BridgeContext(stage="verify"),
            max_turns=1,
            run_config=config,
        )
    except MaxTurnsExceeded:
        pass
    else:
        raise AssertionError("Runner did not enforce max_turns")
    assert limited_model.turns == 1

    report = {
        "evaluation": "sdk_runner_offline_integration",
        "api_calls": 0,
        "model": "scripted_offline_model",
        "sdk_tracing_enabled": False,
        "normal_run": {
            "model_turns": model.turns,
            "tool_calls": len(calls),
            "tool_output_received": model.received_tool_output,
            "final_output": json.loads(result.final_output),
        },
        "limited_run": {
            "max_turns": 1,
            "model_turns": limited_model.turns,
            "tool_calls": len(limited_calls),
            "termination": "MaxTurnsExceeded",
        },
        "limitations": [
            "Scripted model behavior, not LLM decision quality.",
            "Checks the SDK loop, not the full extraction workflow.",
            "No production latency, token cost or recovery comparison.",
        ],
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = Path(f"sdk_runner_check_{stamp}.json")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print("RUNNER OK: tool loop and max_turns checks passed; 0 API calls.")
    print(f"Report: {path.resolve()}")


def main():
    asyncio.run(asyncio.wait_for(evaluate(), timeout=30))


if __name__ == "__main__":
    main()


"""SDK bridge for unscoped registered tools; no orchestration or retries."""
import asyncio
from dataclasses import dataclass
import json

from agents import FunctionTool
from pydantic import BaseModel, ConfigDict, ValidationError

from .tools import ToolRegistry, ToolSpec, ToolReply, PolicyDenied, TransientError


@dataclass
class BridgeContext:
    # Set by trusted application code, not by model arguments.
    stage: str


def bridge_tool(registry, name, stage):
    spec = registry.get(name, stage)
    if not spec.read_only:
        raise PolicyDenied("Only read-only tools can be bridged")

    # Scoped tools need a separate adapter that injects trusted scope.
    if {"tenant", "tenant_id", "scope"} & set(spec.input_model.model_fields):
        raise PolicyDenied("Scoped tools require a trusted-scope adapter")

    async def invoke(context, arguments):
        trusted = getattr(context, "context", None)
        if not isinstance(trusted, BridgeContext) or trusted.stage != stage:
            raise PolicyDenied("SDK context stage does not match tool stage")

        current = registry.get(name, trusted.stage)
        if current is not spec:
            raise PolicyDenied("Registered tool changed after bridge creation")

        request = spec.input_model.model_validate_json(arguments)
        reply = await spec.handler(request)
        data = reply.data
        if isinstance(data, BaseModel):
            data = data.model_dump()
        output = spec.output_model.model_validate(data)
        return output.model_dump_json()

    return FunctionTool(
        name=spec.name,
        description=f"Registered read-only {spec.name} capability.",
        params_json_schema=spec.input_model.model_json_schema(),
        on_invoke_tool=invoke,
        strict_json_schema=True,
    )


async def self_test():
    from types import SimpleNamespace

    class Input(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: int

    class Output(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: int

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
    tool = bridge_tool(registry, "offline_check", "verify")
    context = SimpleNamespace(context=BridgeContext(stage="verify"))

    direct = await handler(Input(value=4))
    bridged = await tool.on_invoke_tool(context, '{"value":4}')
    assert json.loads(bridged) == direct.data.model_dump()

    before = len(calls)
    for arguments in ('{"value":4,"extra":1}', '{"value":"invalid"}'):
        try:
            await tool.on_invoke_tool(context, arguments)
        except ValidationError:
            pass
        else:
            raise AssertionError("Invalid input accepted")
    assert len(calls) == before

    try:
        bridge_tool(registry, "offline_check", "extract")
    except PolicyDenied:
        pass
    else:
        raise AssertionError("Wrong registration stage accepted")

    context.context.stage = "extract"
    try:
        await tool.on_invoke_tool(context, '{"value":4}')
    except PolicyDenied:
        pass
    else:
        raise AssertionError("Changed execution stage accepted")
    assert len(calls) == before
    context.context.stage = "verify"

    async def invalid_output(request):
        return ToolReply({"unexpected": True})

    async def failure(request):
        raise TransientError("Offline injected failure")

    for name, callback, expected in [
        ("invalid_output", invalid_output, ValidationError),
        ("transient_failure", failure, TransientError),
    ]:
        registry.register(ToolSpec(
            name=name, version="fixture-1",
            input_model=Input, output_model=Output,
            handler=callback,
            allowed_stages=frozenset({"verify"}),
        ))
        wrapped = bridge_tool(registry, name, "verify")
        try:
            await wrapped.on_invoke_tool(context, '{"value":4}')
        except expected:
            pass
        else:
            raise AssertionError(f"{name}: exception not propagated")

    class ScopedInput(Input):
        tenant: str

    registry.register(ToolSpec(
        name="scoped_check", version="fixture-1",
        input_model=ScopedInput, output_model=Output,
        handler=handler, allowed_stages=frozenset({"verify"}),
    ))
    try:
        bridge_tool(registry, "scoped_check", "verify")
    except PolicyDenied:
        pass
    else:
        raise AssertionError("Scoped tool exposed without trusted-scope adapter")

    print("BRIDGE OK: offline contract checks passed; 0 API calls.")


if __name__ == "__main__":
    asyncio.run(self_test())

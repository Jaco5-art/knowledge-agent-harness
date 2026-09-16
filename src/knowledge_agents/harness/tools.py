"""All execution uses explicitly registered, typed, read-only capabilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from pydantic import BaseModel


class TransientError(RuntimeError):
    pass


class PermanentError(RuntimeError):
    pass


class PolicyDenied(PermanentError):
    pass


class InvalidOutput(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolReply:
    data: BaseModel | dict
    input_tokens: int | None = None
    output_tokens: int | None = None
    context_metrics: dict | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[[BaseModel], Awaitable[ToolReply]]
    allowed_stages: frozenset[str]
    read_only: bool = True


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec):
        if spec.name in self._tools:
            raise ValueError("Tool already registered")
        if not spec.read_only:
            raise PolicyDenied("Phase 2 permits no external write tools")
        self._tools[spec.name] = spec

    def get(self, name: str, stage: str) -> ToolSpec:
        if name not in self._tools or stage not in self._tools[name].allowed_stages:
            raise PolicyDenied("Tool is not registered for this stage")
        return self._tools[name]

    def versions(self) -> dict[str, str]:
        return {name: spec.version for name, spec in self._tools.items()}

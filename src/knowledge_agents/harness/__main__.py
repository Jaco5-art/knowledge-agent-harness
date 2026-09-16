from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .adapters import DEMO_SOURCE, FixtureAdapter, OpenAIAdapter, registry_for
from .contracts import RuntimeConfig
from .runtime import Harness
from .store import StateStore


async def main():
    parser = argparse.ArgumentParser(description="Phase 2 local Harness; demo defaults to offline fixtures")
    parser.add_argument("--mode", choices=["fixture", "openai"], default="fixture")
    parser.add_argument("--document", type=Path)
    parser.add_argument("--task", default="extract entities, events and relationships")
    parser.add_argument("--model")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--db", default="harness_sessions.sqlite")
    parser.add_argument("--tenant", default="local")
    parser.add_argument("--session", help="Resume or inspect this session instead of creating one")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--trace", default="harness_trace.jsonl")
    parser.add_argument("--full-document-fallback", action="store_true")
    parser.add_argument("--review-id")
    parser.add_argument("--action", choices=["retry", "reject", "cancel"])
    parser.add_argument("--expected-version", type=int)
    parser.add_argument("--actor")
    parser.add_argument('--context-mode', choices=['off', 'budget', 'compact'], default='off')
    parser.add_argument('--context-budget', type=int, default=32000)
    parser.add_argument('--output-reserve', type=int, default=4096)
    parser.add_argument("--runtime", choices=["harness", "langgraph", "sdk"], default="harness")
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.mode == "openai" and (not args.model or (not args.document and not args.session)):
        parser.error("OpenAI mode requires --model and --document (or --session)")
    if args.review_id and not (args.session and args.action and args.actor and args.expected_version is not None):
        parser.error("Review requires session, action, actor, expected-version")
    adapter = FixtureAdapter() if args.mode == "fixture" else OpenAIAdapter(args.model, args.embedding_model)
    if args.context_mode != 'off':
        if args.mode != 'openai':
            parser.error('Context mode requires openai; use offline tests for mocked coverage')
        from .context import ContextPolicy
        from .context_adapter import ContextOpenAIAdapter
        await adapter.close()
        adapter = ContextOpenAIAdapter(args.model, args.embedding_model, policy=ContextPolicy(
            compact=args.context_mode == 'compact', extract_budget=args.context_budget,
            verify_budget=args.context_budget, correct_budget=args.context_budget,
            output_reserve=args.output_reserve))
    store = StateStore(args.db)
    try:
        harness = Harness(store, registry_for(adapter))
        if args.session:
            state = store.load(args.session, args.tenant)
            if args.document and args.document.read_text(encoding="utf-8") != state.source:
                parser.error("Cannot replace the document of an existing session")
        else:
            source = args.document.read_text(encoding="utf-8") if args.document else DEMO_SOURCE
            state = harness.create(source, args.task, args.tenant,
                RuntimeConfig(top_k=args.top_k, full_document_fallback=args.full_document_fallback))
        if args.review_id:
            harness.resolve_review(state.session_id, args.tenant, args.review_id,
                                   args.expected_version, args.actor, args.action)
        from .runtime_engines import run_selected
        state = await run_selected(args.runtime, store, harness.registry, state.session_id, args.tenant, max_steps=args.max_steps)
        store.export_trace(state.session_id, args.tenant, args.trace)
        print(json.dumps({"runtime": args.runtime, "mode": args.mode, "session_id": state.session_id,
            "version": state.version, "status": state.status, "stage": state.stage,
            "tool_calls": state.call_count,
            "verified_facts": [state.candidates[c].model_dump() for c in state.accepted],
            "pending_reviews": [r.model_dump() for r in state.reviews if not r.resolved],
            "trace": args.trace}, ensure_ascii=False, indent=2))
    finally:
        store.close()
        if isinstance(adapter, OpenAIAdapter):
            await adapter.close()


if __name__ == "__main__":
    asyncio.run(main())

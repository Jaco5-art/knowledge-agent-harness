from __future__ import annotations

import os
from typing import Protocol

from .models import ExtractedFact, FactBatch, FactType, VerificationResult


class ExtractionProvider(Protocol):
    def extract(self, source_text: str, fact_type: FactType) -> list[ExtractedFact]: ...

    def correct(
        self,
        source_text: str,
        facts: list[ExtractedFact],
        results: list[VerificationResult],
        correction_round: int,
    ) -> list[ExtractedFact]: ...


class OpenAIProvider:
    """Structured-output provider using the OpenAI Responses API."""

    def __init__(self, model: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError('Install the OpenAI extra: pip install -e ".[openai]"') from exc
        self.client = OpenAI()
        self.model = model or os.getenv("LLM_MODEL", "gpt-5-mini")

    def _parse(self, instructions: str, source_text: str) -> FactBatch:
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": source_text},
            ],
            text_format=FactBatch,
        )
        if response.output_parsed is None:
            raise RuntimeError("The model returned no parsed extraction output.")
        return response.output_parsed

    def extract(self, source_text: str, fact_type: FactType) -> list[ExtractedFact]:
        instructions = f"""
Extract only {fact_type.value} facts explicitly supported by the user's text.
For every fact, copy at least one exact evidence quote from the text. Use a
stable fact_id, set fact_type to {fact_type.value}, source_agent to
{fact_type.value}_agent, correction_round to 0, and estimate extraction
confidence. Return an empty facts list when the text contains no supported
facts. Never add background knowledge or infer a stronger predicate.
""".strip()
        return self._parse(instructions, source_text).facts

    def correct(
        self,
        source_text: str,
        facts: list[ExtractedFact],
        results: list[VerificationResult],
        correction_round: int,
    ) -> list[ExtractedFact]:
        feedback = "\n".join(
            f"{result.fact_id}: {result.verdict}; {'; '.join(result.reasons)}"
            for result in results
            if result.verdict != "supported"
        )
        candidates = "\n".join(fact.model_dump_json() for fact in facts)
        prompt = f"SOURCE:\n{source_text}\n\nCANDIDATES:\n{candidates}\n\nFEEDBACK:\n{feedback}"
        instructions = f"""
Correct the candidate facts using only the SOURCE. Remove claims that cannot be
supported. Copy exact evidence quotes, preserve supported facts, do not
strengthen predicates, and set correction_round to {correction_round}.
Return an empty facts list if nothing is supportable.
""".strip()
        return self._parse(instructions, prompt).facts

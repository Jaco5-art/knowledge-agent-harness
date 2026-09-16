from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class FactType(str, Enum):
    ENTITY = "entity"
    EVENT = "event"
    RELATIONSHIP = "relationship"


class Evidence(BaseModel):
    quote: str = Field(description="Exact supporting span copied from the source")
    start: int | None = None
    end: int | None = None


class ExtractedFact(BaseModel):
    fact_id: str
    fact_type: FactType
    subject: str
    predicate: str
    object: str | None = None
    time: str | None = None
    location: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    source_agent: str
    correction_round: int = 0

    def canonical_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.subject.casefold().strip(),
            self.predicate.casefold().strip(),
            (self.object or "").casefold().strip(),
            (self.time or "").casefold().strip(),
            (self.location or "").casefold().strip(),
        )


class VerificationResult(BaseModel):
    fact_id: str
    verdict: Literal["supported", "unsupported", "needs_correction"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    matched_evidence: list[Evidence] = Field(default_factory=list)


class FactBatch(BaseModel):
    facts: list[ExtractedFact] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    start: int
    end: int


class RetrievedChunk(DocumentChunk):
    score: float

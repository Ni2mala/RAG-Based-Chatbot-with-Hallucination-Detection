from pydantic import BaseModel, Field


class Source(BaseModel):
    intent: str
    category: str
    score: float


class Verification(BaseModel):
    retrieval_relevance: float = 0.0
    status: str = "not_checked"
    nli_grounding: float | None = None
    llm_grounding: float | None = None
    unsupported_claims: list[str] = Field(default_factory=list)
    detector: str | None = None


class ChatRequest(BaseModel):
    # No `mode` field: clients must not be able to switch the guardrails off.
    question: str = Field(..., min_length=1, max_length=500)


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    verification: Verification
    refused: bool = False
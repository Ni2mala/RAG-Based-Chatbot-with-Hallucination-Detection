import json
import re

import ollama

from rag_guard.core.config import settings
from rag_guard.core.logging import get_logger

logger = get_logger(__name__)

JUDGE_SYSTEM = (
    "You are a strict faithfulness judge for a banking chatbot. "
    "You decide how much of a generated ANSWER is supported by the retrieved "
    "CONTEXT documents from approved sources. If a fact in the answer is not "
    "stated in any context document, it is unsupported."
)

JUDGE_TEMPLATE = """\
CONTEXT DOCUMENTS:
{contexts}

ANSWER:
{answer}

Respond with raw JSON only (no markdown code fences), exactly in this shape:
{{"score": <number 0.0 to 1.0 = fraction of the answer's factual claims supported by the context>, "grounded": <true if score >= 0.7 else false>, "unsupported_claims": [<list only of claims NOT supported by the context>]}}"""


class LLMJudge:
    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        self._model = model or settings.LLM_MODEL
        self._client = ollama.Client(host=host or settings.OLLAMA_HOST)

    def judge(self, answer: str, contexts: list[str]) -> dict:
        numbered = "\n".join(f"[{i}] {c}" for i, c in enumerate(contexts))
        prompt = JUDGE_TEMPLATE.format(contexts=numbered, answer=answer)
        resp = self._client.generate(
            model=self._model,
            system=JUDGE_SYSTEM,
            prompt=prompt,
            format="json",
            options={"temperature": 0.0, "num_predict": 400},
        )
        text = resp["response"].strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("LLM judge returned invalid JSON: %s", text[:300])
            return {"score": 0.0, "grounded": False, "unsupported_claims": []}
        score = float(data.get("score", 0.0))
        return {
            "score": max(0.0, min(1.0, score)),
            "grounded": bool(data.get("grounded", score >= settings.GROUNDING_THRESHOLD)),
            "unsupported_claims": [
                str(c) for c in (data.get("unsupported_claims") or [])
            ],
        }
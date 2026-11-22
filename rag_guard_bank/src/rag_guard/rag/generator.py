import ollama

from rag_guard.core.config import settings
from rag_guard.core.retry import retry_once
from rag_guard.rag.prompts import SYSTEM_PROMPT, build_rag_prompt


class Generator:
    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        self._model = model or settings.LLM_MODEL
        self._client = ollama.Client(
            host=host or settings.OLLAMA_HOST, timeout=settings.OLLAMA_TIMEOUT_S
        )

    def generate(self, question: str, context_chunks: list[str]) -> str:
        prompt = build_rag_prompt(question, context_chunks)
        resp = retry_once(
            lambda: self._client.generate(
                model=self._model,
                system=SYSTEM_PROMPT,
                prompt=prompt,
                options={"temperature": 0.2, "num_predict": 600},
            ),
            "Ollama generate",
        )
        return resp["response"].strip()

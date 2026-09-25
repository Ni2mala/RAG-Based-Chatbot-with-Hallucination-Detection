import ollama

from rag_guard.core.config import settings
from rag_guard.rag.prompts import SYSTEM_PROMPT, build_rag_prompt


class Generator:
    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        self._model = model or settings.LLM_MODEL
        self._client = ollama.Client(host=host or settings.OLLAMA_HOST)

    def generate(self, question: str, context_chunks: list[str]) -> str:
        prompt = build_rag_prompt(question, context_chunks)
        resp = self._client.generate(
            model=self._model,
            system=SYSTEM_PROMPT,
            prompt=prompt,
            options={"temperature": 0.2, "num_predict": 600},
        )
        return resp["response"].strip()

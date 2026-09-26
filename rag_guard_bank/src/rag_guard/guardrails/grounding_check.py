from typing import Sequence

from transformers import pipeline

from rag_guard.core.config import settings
from rag_guard.core.logging import get_logger

logger = get_logger(__name__)

LABEL_BY_INDEX = {0: "contradiction", 1: "entailment", 2: "neutral"}


class GroundingChecker:
    def __init__(self, model_name: str | None = None) -> None:
        model = model_name or settings.NLI_MODEL
        logger.info("Loading NLI model %s (first load downloads it) ...", model)
        self._pipe = pipeline("text-classification", model=model, device=-1)
        try:
            id2label = self._pipe.model.config.id2label
            self._index_to_name = {int(k): str(v) for k, v in id2label.items()}
        except Exception:
            self._index_to_name = LABEL_BY_INDEX

    @staticmethod
    def _parse_index(label: str) -> int | None:
        if label.startswith("LABEL_"):
            try:
                return int(label.split("_")[-1])
            except ValueError:
                return None
        return None

    def check(
        self,
        claims: Sequence[str],
        contexts: Sequence[str],
        entail_threshold: float = 0.4,
    ) -> list[dict]:
        results: list[dict] = []
        for claim in claims:
            max_ent = 0.0
            max_contra = 0.0
            for ctx in contexts:
                pred = self._pipe({"text": ctx, "text_pair": claim}, truncation=True)
                if isinstance(pred, list):
                    pred = pred[0]
                idx = self._parse_index(pred["label"])
                name = self._index_to_name.get(idx, "neutral") if idx is not None else "neutral"
                if name == "entailment":
                    max_ent = max(max_ent, pred["score"])
                elif name == "contradiction":
                    max_contra = max(max_contra, pred["score"])
            supported = max_ent >= entail_threshold and max_ent >= max_contra
            verdict = "entailment" if max_ent >= max_contra else "contradiction"
            results.append(
                {
                    "claim": claim,
                    "verdict": verdict,
                    "score": round(max(max_ent, max_contra), 3),
                    "supported": supported,
                }
            )
        return results

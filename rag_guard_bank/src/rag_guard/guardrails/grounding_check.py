import re
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
            if id2label is None:
                id2label = {}
            self._index_to_name = {int(k): str(v).strip().lower() for k, v in id2label.items()}
        except Exception:
            self._index_to_name = dict(LABEL_BY_INDEX)
        self._name_to_index = {v: k for k, v in self._index_to_name.items()}

    def _resolve(self, label: str) -> str:
        name = label.strip().lower()
        if name in self._name_to_index:
            return self._index_to_name[self._name_to_index[name]]
        match = re.fullmatch(r"label_(\d+)", name)
        if match:
            return self._index_to_name.get(int(match.group(1)), name)
        return name

    def check(
        self,
        claims: Sequence[str],
        contexts: Sequence[str],
        entail_threshold: float = 0.4,
    ) -> list[dict]:
        results: list[dict] = []
        for claim in claims:
            scores = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
            for ctx in contexts:
                preds = self._pipe(
                    {"text": ctx, "text_pair": claim},
                    truncation=True,
                    top_k=len(self._index_to_name),
                )
                if isinstance(preds, dict):
                    preds = [preds]
                flat = []
                for p in preds:
                    flat.extend(p if isinstance(p, list) else [p])
                for p in flat:
                    name = self._resolve(p["label"])
                    if name in scores:
                        scores[name] = max(scores[name], float(p["score"]))
            max_ent = scores["entailment"]
            max_contra = scores["contradiction"]
            supported = max_ent >= entail_threshold and max_ent >= max_contra
            if supported:
                verdict = "entailment"
            elif max_contra > max_ent:
                verdict = "contradiction"
            else:
                verdict = "neutral"
            results.append(
                {
                    "claim": claim,
                    "verdict": verdict,
                    "score": round(max(max_ent, max_contra), 3),
                    "entailment": round(max_ent, 3),
                    "contradiction": round(max_contra, 3),
                    "neutral": round(scores["neutral"], 3),
                    "supported": supported,
                }
            )
        return results
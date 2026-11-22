import re
from typing import Sequence

from transformers import pipeline

from rag_guard.core.config import settings
from rag_guard.core.logging import get_logger

logger = get_logger(__name__)

LABEL_BY_INDEX = {0: "contradiction", 1: "entailment", 2: "neutral"}

_STOP = set("a an the to of and or in on at for you your our is are be can will with this that it as by from if any have has please".split())
_QA_HEADER = re.compile(r"^Question:.*?\nAnswer:\s*", re.DOTALL)
WINDOWS_PER_CLAIM = 3


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w not in _STOP and len(w) > 2}


def _windows(context: str) -> list[str]:
    """Sentences of the stored answer, plus every pair of neighbouring sentences."""
    body = _QA_HEADER.sub("", context)
    sents = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", body) if len(p.strip()) > 8]
    return sents + [" ".join(sents[i : i + 2]) for i in range(len(sents) - 1)]


def invented_claims(checks: Sequence[dict], contexts: Sequence[str], min_novelty: float) -> list[str]:
    """Claims that NLI found unsupported AND that bring mostly new words (not in any retrieved chunk).

    NLI alone marks about a quarter of natural, paraphrased claims as unsupported, mostly because the
    wording differs while the words are the same, so it cannot decide on its own. A claim that is both
    unsupported and mostly made of words the documents never use is an invention (measured: 77 of 78
    injected sentences, versus 6 of 66 natural answers containing one).
    """
    known = set().union(*[_tokens(c) for c in contexts]) if contexts else set()
    out = []
    for r in checks:
        words = _tokens(r["claim"])
        if not r["supported"] and words and len(words - known) / len(words) >= min_novelty:
            out.append(r["claim"])
    return out


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
        # Premises are the few context sentences that best overlap each claim, not whole
        # ~900-character Q+A chunks: the small NLI model gives near-zero entailment to a
        # sentence that is quoted verbatim inside a long premise (7/48 true claims passed
        # vs 48/48 with sentence premises; 0/70 fabricated claims passed either way).
        # Thresholds are unchanged. Choosing windows by overlap can only drop support.
        windows = [w for c in contexts for w in _windows(c)]
        results: list[dict] = []
        for claim in claims:
            scores = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
            claim_tokens = _tokens(claim)
            best = sorted(windows, key=lambda w: -len(claim_tokens & _tokens(w)))[:WINDOWS_PER_CLAIM]
            for ctx in best:
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
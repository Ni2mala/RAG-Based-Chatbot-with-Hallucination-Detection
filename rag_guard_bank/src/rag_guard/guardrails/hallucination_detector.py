from dataclasses import dataclass, field

from rag_guard.core.config import settings
from rag_guard.core.logging import get_logger
from rag_guard.guardrails.claim_extractor import extract_claims
from rag_guard.guardrails.grounding_check import GroundingChecker
from rag_guard.guardrails.llm_judge import LLMJudge

logger = get_logger(__name__)


@dataclass
class DetectionResult:
    nli_grounding: float | None = None
    llm_grounding: float | None = None
    status: str = "unsupported"
    unsupported_claims: list[str] = field(default_factory=list)
    detector: str = "none"


class HallucinationDetector:
    def __init__(self, mode: str | None = None) -> None:
        self._mode = self._normalize(mode or settings.DETECTOR_MODE)
        self._nli: GroundingChecker | None = None
        self._judge: LLMJudge | None = None

    @staticmethod
    def _normalize(mode: str) -> str:
        mode = mode.lower()
        return mode if mode in ("nli", "llm", "both") else "both"

    def _ensure_loaded(self, mode: str) -> None:
        if mode in ("nli", "both") and self._nli is None:
            self._nli = GroundingChecker()
        if mode in ("llm", "both") and self._judge is None:
            self._judge = LLMJudge()

    def detect(
        self,
        answer: str,
        contexts: list[str],
        mode: str | None = None,
    ) -> DetectionResult:
        mode = self._normalize(mode) if mode else self._mode
        self._ensure_loaded(mode)

        result = DetectionResult(detector=mode)
        claims = extract_claims(answer)
        unsupported: list[str] = []

        if self._nli is not None and mode in ("nli", "both"):
            checks = self._nli.check(claims, contexts)
            result.nli_grounding = sum(
                r["supported"] for r in checks
            ) / max(1, len(checks))
            unsupported.extend(r["claim"] for r in checks if not r["supported"])

        if self._judge is not None and mode in ("llm", "both"):
            verdict = self._judge.judge(answer, contexts)
            result.llm_grounding = verdict["score"]
            unsupported.extend(verdict["unsupported_claims"])

        if result.nli_grounding is not None and result.llm_grounding is not None:
            score = (result.nli_grounding + result.llm_grounding) / 2
        elif result.nli_grounding is not None:
            score = result.nli_grounding
        elif result.llm_grounding is not None:
            score = result.llm_grounding
        else:
            score = 0.0

        if score >= settings.GROUNDING_THRESHOLD:
            result.status = "grounded"
        elif score >= settings.GROUNDING_THRESHOLD / 2:
            result.status = "partially_supported"
        else:
            result.status = "unsupported"

        result.unsupported_claims = unsupported
        logger.info(
            "Detector=%s score=%.2f status=%s claims=%d",
            mode, score, result.status, len(claims),
        )
        return result
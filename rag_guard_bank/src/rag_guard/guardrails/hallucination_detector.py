from dataclasses import dataclass, field

from rag_guard.core.config import settings
from rag_guard.core.logging import get_logger
from rag_guard.guardrails.claim_extractor import extract_claims
from rag_guard.guardrails.figure_check import unsupported_figures
from rag_guard.guardrails.grounding_check import GroundingChecker
from rag_guard.guardrails.llm_judge import LLMJudge

logger = get_logger(__name__)
DETECTOR_MODES = ("none", "nli", "llm", "both")


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
        mode = mode.lower().strip()
        if mode not in DETECTOR_MODES:
            raise ValueError(f"Invalid detector mode {mode!r}; expected one of {DETECTOR_MODES}")
        return mode

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
        if mode == "none":
            return DetectionResult(detector="none", status="unguarded")

        self._ensure_loaded(mode)

        result = DetectionResult(detector=mode)
        claims = extract_claims(answer)
        unsupported: list[str] = []

        # No claims (a very short answer): NLI has nothing to score, so it is skipped instead of
        # counting as 0.0 support. The judge still scores the answer, and in nli-only mode the
        # score stays 0.0, so an unverifiable answer is still refused.
        if self._nli is not None and mode in ("nli", "both") and claims:
            checks = self._nli.check(claims, contexts)
            result.nli_grounding = sum(r["supported"] for r in checks) / len(checks)
            unsupported.extend(r["claim"] for r in checks if not r["supported"])

        if self._judge is not None and mode in ("llm", "both"):
            try:
                verdict = self._judge.judge(answer, contexts)
            except Exception as exc:
                # Judge down or unparseable: use the other detector instead of scoring 0.
                logger.warning("LLM judge failed (%s); falling back to NLI only", exc)
            else:
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

        figures = unsupported_figures(answer, contexts)
        if figures:
            result.status = "unsupported"
            unsupported.insert(0, "Figures not found in the documents: " + ", ".join(figures))
            logger.warning("Unsupported figures in answer: %s", figures)

        result.unsupported_claims = unsupported
        logger.info(
            "Detector=%s score=%.2f status=%s claims=%d",
            mode, score, result.status, len(claims),
        )
        return result
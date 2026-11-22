import time
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
    seconds: float = 0.0  # time spent in NLI and/or judge for this mode
    score: float = 0.0    # combined grounding score before the figure check


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
        comps = self._components(
            answer, contexts, nli=mode in ("nli", "both"), judge=mode in ("llm", "both")
        )
        result = self._assemble(mode, comps)
        logger.info("Detector=%s status=%s claims=%d", mode, result.status, comps["claims"])
        return result

    def detect_modes(self, answer: str, contexts: list[str]) -> dict[str, DetectionResult]:
        """Results for nli, llm and both from ONE NLI pass and ONE judge call (used by the evaluation)."""
        self._ensure_loaded("both")
        comps = self._components(answer, contexts, nli=True, judge=True)
        return {m: self._assemble(m, comps) for m in ("nli", "llm", "both")}

    def _components(self, answer: str, contexts: list[str], nli: bool, judge: bool) -> dict:
        claims = extract_claims(answer)
        comps = {
            "claims": len(claims),
            "nli": None, "nli_unsupported": [],
            "llm": None, "llm_unsupported": [],
            "figures": unsupported_figures(answer, contexts),
            "nli_s": 0.0, "llm_s": 0.0,
        }
        # No claims (a very short answer): NLI has nothing to score, so it is skipped instead of
        # counting as 0.0 support. The judge still scores the answer, and in nli-only mode the
        # score stays 0.0, so an unverifiable answer is still refused.
        if nli and self._nli is not None and claims:
            t = time.time()
            checks = self._nli.check(claims, contexts)
            comps["nli_s"] = time.time() - t
            comps["nli"] = sum(r["supported"] for r in checks) / len(checks)
            comps["nli_unsupported"] = [r["claim"] for r in checks if not r["supported"]]
        if judge and self._judge is not None:
            t = time.time()
            try:
                verdict = self._judge.judge(answer, contexts)
            except Exception as exc:
                # Judge down or unparseable: use the other detector instead of scoring 0.
                logger.warning("LLM judge failed (%s); falling back to NLI only", exc)
            else:
                comps["llm_s"] = time.time() - t
                comps["llm"] = verdict["score"]
                comps["llm_unsupported"] = verdict["unsupported_claims"]
        return comps

    @staticmethod
    def _assemble(mode: str, comps: dict) -> DetectionResult:
        result = DetectionResult(detector=mode)
        result.seconds = (comps["nli_s"] if mode in ("nli", "both") else 0.0) + (
            comps["llm_s"] if mode in ("llm", "both") else 0.0
        )
        unsupported: list[str] = []
        if mode in ("nli", "both"):
            result.nli_grounding = comps["nli"]
            unsupported += comps["nli_unsupported"]
        if mode in ("llm", "both"):
            result.llm_grounding = comps["llm"]
            unsupported += comps["llm_unsupported"]

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

        if comps["figures"]:
            result.status = "unsupported"
            unsupported.insert(0, "Figures not found in the documents: " + ", ".join(comps["figures"]))
            logger.warning("Unsupported figures in answer: %s", comps["figures"])

        result.score = score
        result.unsupported_claims = unsupported
        return result

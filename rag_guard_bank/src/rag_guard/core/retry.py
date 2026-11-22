from typing import Callable, TypeVar

from rag_guard.core.logging import get_logger

logger = get_logger(__name__)
T = TypeVar("T")


def retry_once(fn: Callable[[], T], what: str) -> T:
    """Call fn; on any exception log and try one more time, then raise."""
    try:
        return fn()
    except Exception as exc:
        logger.warning("%s failed (%s); retrying once", what, exc)
        return fn()

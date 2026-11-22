import re

_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_FIGURE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")


def _figures(text: str) -> set[str]:
    # Compare the bare number so "$10" in the docs matches "10" in the answer.
    return {m.group().strip("$%.,") for m in _FIGURE.finditer(text)}


def unsupported_figures(answer: str, contexts: list[str]) -> list[str]:
    """Numbers, amounts and percentages in the answer that appear in no retrieved chunk.

    Fees, rates and limits are the costliest thing to invent, so any figure the documents
    don't contain is treated as unsupported regardless of the detector scores.
    Numbered list markers ("1. Open the app") are ignored in the answer AND in the contexts, otherwise
    a step number in a document would "support" an invented "2 requests" or "3 days".
    """
    stripped = _LIST_MARKER.sub("", answer)
    known = _figures(_LIST_MARKER.sub("", "\n".join(contexts)))
    return sorted(f for f in _figures(stripped) if f not in known)

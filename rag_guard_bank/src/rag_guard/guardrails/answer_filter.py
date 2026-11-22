import re

_LIST_ITEM = re.compile(r"^(\s*)(\d+)([.)])(\s+.*)?$")
_BARE_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*\u2022])\s*$")


def remove_claims(answer: str, claims: list[str]) -> str:
    """Delete the given sentences from the answer, tidy the leftovers and renumber numbered steps."""
    for claim in claims:
        answer = answer.replace(claim, "")
    lines = [ln.rstrip() for ln in answer.split("\n")]
    lines = [ln for ln in lines if ln.strip() and not _BARE_MARKER.match(ln)]
    out, counter = [], 0
    for ln in lines:
        m = _LIST_ITEM.match(ln)
        if m and m.group(4):
            counter = 1 if int(m.group(2)) == 1 else counter + 1   # a list restarting at 1 restarts the count
            ln = f"{m.group(1)}{counter}{m.group(3)}{m.group(4)}"
        out.append(ln)
    return re.sub(r"[ \t]{2,}", " ", "\n".join(out)).strip()

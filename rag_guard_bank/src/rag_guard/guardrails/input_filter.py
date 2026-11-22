import re

# Phrases that try to override the bot's rules. A banking question never needs them.
# ponytail: keyword list, catches common attacks only; upgrade to a classifier if it gets bypassed.
_PATTERNS = [
    r"\bignore\b.{0,30}\b(instructions?|context|rules?|documents?|grounding|above|previous)\b",
    r"\bdisregard\b.{0,30}\b(instructions?|context|rules?|documents?|grounding|above|previous)\b",
    r"\bforget\b.{0,20}\b(everything|instructions?|rules?|your)\b",
    r"\byou are now\b",
    r"\bfrom now on\b",
    r"\bpretend\b",
    r"\b(system|developer) prompt\b",
    r"\bjailbreak\b",
    r"\b(repeat|print|reveal|show)\b.{0,30}\b(prompt|instructions?|context)\b",
]
_RE = re.compile("|".join(_PATTERNS), re.IGNORECASE | re.DOTALL)


def looks_like_injection(question: str) -> bool:
    return bool(_RE.search(question))

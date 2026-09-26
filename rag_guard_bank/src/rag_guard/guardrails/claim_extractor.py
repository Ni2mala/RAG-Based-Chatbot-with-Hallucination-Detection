import re

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def extract_claims(answer: str) -> list[str]:
    claims: list[str] = []
    for part in SENTENCE_SPLIT.split(answer):
        part = part.strip().lstrip("-*").strip()
        if len(part) > 8:
            claims.append(part)
    return claims

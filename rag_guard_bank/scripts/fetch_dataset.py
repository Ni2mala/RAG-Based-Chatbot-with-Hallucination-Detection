import sys
import json
import argparse
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset
from rag_guard.core.config import settings
from rag_guard.core.logging import get_logger

logger = get_logger(__name__)

PLACEHOLDER = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")

SPECIAL = {
    "Company Website URL": "our website",
    "Company Website": "our website",
    "URL Website": "our website",
    "Bank Name": "our bank",
    "Bank's Mobile App": "our mobile app",
    "Banking App": "mobile app",
    "Bank App": "mobile app",
    "App": "mobile app",
}


def normalize_response(text: str) -> str:
    def repl(match):
        key = match.group(1).strip()
        return SPECIAL.get(key, " ".join(key.lower().split()))

    return PLACEHOLDER.sub(repl, text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    logger.info("Loading Bitext retail-banking dataset from Hugging Face ...")
    ds = load_dataset(
        "bitext/Bitext-retail-banking-llm-chatbot-training-dataset",
        split="train",
    )
    if args.max_rows is not None:
        ds = ds.select(range(args.max_rows))

    settings.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = settings.DATA_RAW_DIR / "bitext_banking.jsonl"

    with out_path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(ds):
            record = {
                "id": i,
                "request": row["instruction"],
                "intent": row["intent"],
                "category": row["category"],
                "response": normalize_response(row["response"]),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    intents = sorted(ds.unique("intent"))
    logger.info("Wrote %d rows -> %s", len(ds), out_path)
    logger.info("Unique intents (%d): %s", len(intents), ", ".join(intents))
    logger.info("Example row: %s", record)


if __name__ == "__main__":
    main()

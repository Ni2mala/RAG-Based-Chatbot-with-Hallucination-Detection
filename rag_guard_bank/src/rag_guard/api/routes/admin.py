import hmac
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from rag_guard.api.dependencies import get_retriever
from rag_guard.core.config import DEFAULT_ADMIN_TOKEN, settings
from rag_guard.services.ingestion_service import rebuild_index

router = APIRouter(tags=["admin"])


@router.post("/admin/ingest")
def ingest(x_admin_token: Optional[str] = Header(default=None)) -> dict:
    # Admin is disabled until a real token is configured in .env.
    if settings.ADMIN_TOKEN in ("", DEFAULT_ADMIN_TOKEN):
        raise HTTPException(status_code=503, detail="Admin disabled: set ADMIN_TOKEN in .env")
    if not hmac.compare_digest((x_admin_token or "").encode(), settings.ADMIN_TOKEN.encode()):
        raise HTTPException(status_code=401, detail="Invalid admin token")
    result = rebuild_index()
    # The cached retriever points at the deleted collection; drop it so the next /chat rebuilds it.
    # ponytail: a /chat request in flight during the rebuild can still fail; add a lock if that matters.
    get_retriever.cache_clear()
    return {
        "status": "ok",
        "raw_rows": result["raw_rows"],
        "kb_rows": result["kb_rows"],
        "chunks": result["chunks"],
    }

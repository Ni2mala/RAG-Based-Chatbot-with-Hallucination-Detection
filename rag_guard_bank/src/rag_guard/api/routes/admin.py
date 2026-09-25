from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from rag_guard.core.config import settings
from rag_guard.services.ingestion_service import rebuild_index

router = APIRouter(tags=["admin"])


@router.post("/admin/ingest")
def ingest(x_admin_token: Optional[str] = Header(default=None)) -> dict:
    if x_admin_token != settings.ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")
    result = rebuild_index()
    return {
        "status": "ok",
        "raw_rows": result["raw_rows"],
        "kb_rows": result["kb_rows"],
        "chunks": result["chunks"],
    }

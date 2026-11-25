from contextlib import asynccontextmanager

import httpx
import ollama
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rag_guard.api.dependencies import get_detector
from rag_guard.api.routes import admin, chat, health
from rag_guard.core.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_detector().warm_up()
    except Exception as exc:
        # Not fatal: the models are loaded again on the first question.
        logger.warning("Detector warm-up failed: %s", exc)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="RAG-Guard Banking Chatbot",
    description="Banking-domain RAG chatbot with hallucination detection (in progress).",
    version="0.1.0",
)



def _model_unavailable(request: Request, exc: Exception) -> JSONResponse:
    # Ollama down, model missing or timed out (after the one retry in core/retry.py).
    logger.error("Model backend failure on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "The language model service is unavailable. Please try again shortly."},
    )


for _exc in (ConnectionError, httpx.TransportError, ollama.ResponseError):
    app.add_exception_handler(_exc, _model_unavailable)

app.include_router(health.router)
app.include_router(admin.router)
app.include_router(chat.router)


@app.get("/")
def root() -> dict:
    return {"message": "RAG-Guard banking chatbot API", "docs": "/docs"}

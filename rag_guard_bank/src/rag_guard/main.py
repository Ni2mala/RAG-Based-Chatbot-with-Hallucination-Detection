from fastapi import FastAPI

from rag_guard.api.routes import admin, chat, health

app = FastAPI(
    title="RAG-Guard Banking Chatbot",
    description="Banking-domain RAG chatbot with hallucination detection (in progress).",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(admin.router)
app.include_router(chat.router)


@app.get("/")
def root() -> dict:
    return {"message": "RAG-Guard banking chatbot API", "docs": "/docs"}

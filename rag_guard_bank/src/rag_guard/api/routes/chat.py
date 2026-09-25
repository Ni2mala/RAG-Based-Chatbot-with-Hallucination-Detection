from fastapi import APIRouter, Depends

from rag_guard.api.dependencies import get_generator, get_retriever
from rag_guard.rag.generator import Generator
from rag_guard.rag.retriever import Retriever
from rag_guard.schemas.chat import ChatRequest, ChatResponse
from rag_guard.services.chat_service import ChatService

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    retriever: Retriever = Depends(get_retriever),
    generator: Generator = Depends(get_generator),
) -> ChatResponse:
    return ChatService(retriever, generator).answer(request)

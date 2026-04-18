from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import build_chat_response

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/ask", response_model=ChatResponse, summary="Ask the government assistant a question")
def chat_ask(payload: ChatRequest):
    return build_chat_response(payload.question)

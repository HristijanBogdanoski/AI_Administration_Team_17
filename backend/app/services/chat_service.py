from app.schemas.chat import ChatResponse
from app.services.gemini_service import FALLBACK_TEXT, ask_gemini
from app.services.tavily_service import search_context


def build_chat_response(question: str) -> ChatResponse:
    web_context = search_context(question)
    raw = ask_gemini(question, web_context=web_context)

    if raw.strip() == "FALLBACK" or "FALLBACK" in raw:
        return ChatResponse(raw_response=FALLBACK_TEXT, fallback=True)

    if not raw.strip():
        return ChatResponse(raw_response=FALLBACK_TEXT, fallback=True)

    return ChatResponse(
        fallback=False,
        raw_response=raw,
    )

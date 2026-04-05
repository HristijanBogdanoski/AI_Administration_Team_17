from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.gemini_service import ask_gemini, FALLBACK_TEXT

router = APIRouter(prefix="/chat", tags=["chat"])

def _extract_field(raw: str, label: str) -> str | None:
    for line in raw.splitlines():
        if line.strip().startswith(label):
            return line.split(":", 1)[1].strip()
    return None

@router.post("/ask", response_model=ChatResponse)
def chat_ask(payload: ChatRequest):
    raw = ask_gemini(payload.question)

    if raw.strip() == "FALLBACK" or "FALLBACK" in raw:
        return ChatResponse(raw_response=FALLBACK_TEXT, fallback=True)

    service_name = _extract_field(raw, "Service Name")
    required_documents = _extract_field(raw, "Required Documents")
    location_data = _extract_field(raw, "Location Data")

    if not service_name and not required_documents and not location_data:
        return ChatResponse(raw_response=FALLBACK_TEXT, fallback=True)

    return ChatResponse(
        service_name=service_name,
        required_documents=required_documents,
        location_data=location_data,
        fallback=False,
        raw_response=raw,
    )
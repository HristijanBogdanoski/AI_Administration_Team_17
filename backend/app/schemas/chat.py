from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)

class ChatResponse(BaseModel):
    service_name: str | None = None
    required_documents: str | None = None
    location_data: str | None = None
    fallback: bool = False
    raw_response: str
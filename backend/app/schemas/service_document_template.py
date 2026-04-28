from typing import Any, Optional

from pydantic import BaseModel, Field


class ServiceDocumentTemplateBase(BaseModel):
    service_id: str = Field(..., min_length=1, max_length=100)
    title: str = Field(..., min_length=1, max_length=255)
    template_type: str = Field(default="json", min_length=1, max_length=20)
    template_body: dict[str, Any]
    is_active: bool = True


class ServiceDocumentTemplateCreate(ServiceDocumentTemplateBase):
    pass


class ServiceDocumentTemplateUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=255)
    template_type: Optional[str] = Field(default=None, min_length=1, max_length=20)
    template_body: Optional[dict[str, Any]] = None
    is_active: Optional[bool] = None


class ServiceDocumentTemplateFillRequest(BaseModel):
    selected_fields: list[str] = Field(default_factory=list)


class ServiceDocumentTemplateOut(ServiceDocumentTemplateBase):
    id: int

    model_config = {"from_attributes": True}
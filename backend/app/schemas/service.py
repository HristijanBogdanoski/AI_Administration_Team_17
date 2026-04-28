from pydantic import BaseModel, Field
from typing import Optional, List

from app.models.enums import ServiceCategory


class ServiceBase(BaseModel):
    service_id: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=255)
    category: ServiceCategory
    description: Optional[str] = None
    processing_time_days: Optional[int] = Field(default=None, ge=0)
    details: List[str] = Field(default_factory=list)


class ServiceCreate(ServiceBase):
    pass


class ServiceUpdate(BaseModel):
    service_id: Optional[str] = Field(default=None, min_length=1, max_length=100)
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    category: Optional[ServiceCategory] = None
    description: Optional[str] = None
    processing_time_days: Optional[int] = Field(default=None, ge=0)
    details: Optional[List[str]] = None


class ServiceOut(ServiceBase):
    id: int

    class Config:
        from_attributes = True
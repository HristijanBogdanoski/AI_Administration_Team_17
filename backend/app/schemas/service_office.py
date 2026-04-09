from pydantic import BaseModel
from typing import Optional


class CoordinatesSchema(BaseModel):
    lat: float
    lng: float


class ServiceOfficeCreate(BaseModel):
    service_id: str
    service_name: str
    office_name: str
    address: str
    coordinates: CoordinatesSchema
    working_hours: str
    contact_email: str
    notes: Optional[str] = None


class ServiceOfficeUpdate(BaseModel):
    service_name: Optional[str] = None
    office_name: Optional[str] = None
    address: Optional[str] = None
    coordinates: Optional[CoordinatesSchema] = None
    working_hours: Optional[str] = None
    contact_email: Optional[str] = None
    notes: Optional[str] = None


class ServiceOfficeResponse(BaseModel):
    id: int
    service_id: str
    service_name: str
    office_name: str
    address: str
    coordinates: CoordinatesSchema
    working_hours: str
    contact_email: str
    notes: Optional[str] = None

    model_config = {"from_attributes": True}
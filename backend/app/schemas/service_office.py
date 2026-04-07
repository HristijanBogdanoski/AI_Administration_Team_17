from pydantic import BaseModel
from typing import Optional


class CoordinatesSchema(BaseModel):
    lat: float
    lng: float


class ServiceOfficeResponse(BaseModel):
    service_id: str
    service_name: str
    office_name: str
    address: str
    coordinates: CoordinatesSchema
    working_hours: str
    contact_email: str
    notes: Optional[str] = None

    model_config = {"from_attributes": True}
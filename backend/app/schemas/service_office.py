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


class MapLocationResponse(BaseModel):
    institution: str
    search_query: str
    resolved_address: str
    display_name: str
    coordinates: CoordinatesSchema
    source: str
    matched_by: str
    place_id: Optional[int] = None
    osm_type: Optional[str] = None
    osm_id: Optional[int] = None
    confidence: Optional[float] = None

    model_config = {"from_attributes": True}
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.models.service_office import ServiceOffice
from app.schemas.service_office import (
    CoordinatesSchema,
    ServiceOfficeCreate,
    ServiceOfficeResponse,
    ServiceOfficeUpdate,
)


def to_response(office: ServiceOffice) -> ServiceOfficeResponse:
    return ServiceOfficeResponse(
        id=office.id,
        service_id=office.service_id,
        service_name=office.service_name,
        office_name=office.office_name,
        address=office.address,
        coordinates=CoordinatesSchema(lat=office.latitude, lng=office.longitude),
        working_hours=office.working_hours,
        contact_email=office.contact_email,
        notes=office.notes,
    )


def list_service_locations(db: Session, skip: int, limit: int) -> list[ServiceOfficeResponse]:
    offices = db.query(ServiceOffice).offset(skip).limit(limit).all()
    return [to_response(o) for o in offices]


def get_service_location_by_id(db: Session, location_id: int) -> ServiceOffice | None:
    return db.query(ServiceOffice).filter(ServiceOffice.id == location_id).first()


def get_service_location(db: Session, service: str) -> ServiceOffice | None:
    normalized = service.strip()
    return (
        db.query(ServiceOffice)
        .filter(
            or_(
                func.lower(ServiceOffice.service_name) == normalized.lower(),
                ServiceOffice.service_id == normalized,
            )
        )
        .first()
    )


def service_location_exists_by_service_id(db: Session, service_id: str) -> bool:
    return db.query(ServiceOffice).filter(ServiceOffice.service_id == service_id).first() is not None


def create_service_location(db: Session, payload: ServiceOfficeCreate) -> ServiceOffice:
    office = ServiceOffice(
        service_id=payload.service_id,
        service_name=payload.service_name,
        office_name=payload.office_name,
        address=payload.address,
        latitude=payload.coordinates.lat,
        longitude=payload.coordinates.lng,
        working_hours=payload.working_hours,
        contact_email=payload.contact_email,
        notes=payload.notes,
    )
    db.add(office)
    db.commit()
    db.refresh(office)
    return office


def apply_service_location_update(office: ServiceOffice, payload: ServiceOfficeUpdate) -> ServiceOffice:
    if payload.service_name is not None:
        office.service_name = payload.service_name
    if payload.office_name is not None:
        office.office_name = payload.office_name
    if payload.address is not None:
        office.address = payload.address
    if payload.coordinates is not None:
        office.latitude = payload.coordinates.lat
        office.longitude = payload.coordinates.lng
    if payload.working_hours is not None:
        office.working_hours = payload.working_hours
    if payload.contact_email is not None:
        office.contact_email = payload.contact_email
    if payload.notes is not None:
        office.notes = payload.notes
    return office


def save_service_location(db: Session, office: ServiceOffice) -> ServiceOffice:
    db.commit()
    db.refresh(office)
    return office


def delete_service_location(db: Session, office: ServiceOffice) -> None:
    db.delete(office)
    db.commit()

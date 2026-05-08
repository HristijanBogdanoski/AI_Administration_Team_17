from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.models.service import Service
from app.models.service_office import ServiceOffice
from app.schemas.service_office import (
    CoordinatesSchema,
    MapLocationResponse,
    ServiceOfficeCreate,
    ServiceOfficeResponse,
    ServiceOfficeUpdate,
)
from app.services.openstreetmap_service import (
    build_skopje_query,
    geocode_skopje_query,
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


def get_service_locations(db: Session, service: str) -> list[ServiceOfficeResponse]:
    normalized = service.strip()
    # If the provided service looks like an integer id, match by FK id first
    try:
        svc_id = int(normalized)
    except Exception:
        svc_id = None
    if svc_id is not None:
        offices = db.query(ServiceOffice).filter(ServiceOffice.service_id == svc_id).order_by(ServiceOffice.id.asc()).all()
    else:
        offices = (
            db.query(ServiceOffice)
            .filter(
                or_(
                    func.lower(ServiceOffice.service_name) == normalized.lower(),
                    func.lower(ServiceOffice.office_name) == normalized.lower(),
                    func.lower(ServiceOffice.address) == normalized.lower(),
                )
            )
            .order_by(ServiceOffice.id.asc())
            .all()
        )
    return [to_response(o) for o in offices]


def get_service_location_by_identifier(db: Session, institution: str) -> ServiceOffice | None:
    normalized = institution.strip()
    try:
        svc_id = int(normalized)
    except Exception:
        svc_id = None

    if svc_id is not None:
        return db.query(ServiceOffice).filter(ServiceOffice.service_id == svc_id).first()

    return (
        db.query(ServiceOffice)
        .filter(
            or_(
                func.lower(ServiceOffice.service_name) == normalized.lower(),
                func.lower(ServiceOffice.office_name) == normalized.lower(),
                func.lower(ServiceOffice.address) == normalized.lower(),
            )
        )
        .first()
    )


def service_exists_by_service_id(db: Session, service_id: int) -> bool:
    return db.query(Service).filter(Service.id == service_id).first() is not None


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


def get_map_ready_location(
    db: Session,
    institution: str,
    address: str | None = None,
) -> MapLocationResponse:
    office = get_service_location_by_identifier(db, institution)
    if office is not None:
        return MapLocationResponse(
            institution=office.service_name,
            search_query=office.address,
            resolved_address=office.address,
            display_name=f"{office.office_name}, {office.address}",
            coordinates=CoordinatesSchema(lat=office.latitude, lng=office.longitude),
            source="database",
            matched_by="stored service office",
        )

    # Institution not found in DB - geocode the address if provided, otherwise use institution name
    if address:
        search_query = build_skopje_query(address)
    else:
        search_query = build_skopje_query(institution)
    
    geocoded = geocode_skopje_query(search_query)
    return MapLocationResponse(
        institution=institution.strip(),
        search_query=search_query,
        resolved_address=address.strip() if address else institution.strip(),
        display_name=geocoded.display_name,
        coordinates=CoordinatesSchema(lat=geocoded.lat, lng=geocoded.lng),
        source="openstreetmap",
        matched_by="geocoded address",
        place_id=geocoded.place_id,
        osm_type=geocoded.osm_type,
        osm_id=geocoded.osm_id,
        confidence=geocoded.importance,
    )

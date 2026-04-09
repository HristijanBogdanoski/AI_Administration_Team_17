from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_, func

from app.db.session import get_db
from app.models.service_office import ServiceOffice
from app.schemas.service_office import (
    ServiceOfficeResponse,
    ServiceOfficeCreate,
    ServiceOfficeUpdate,
    CoordinatesSchema,
)

router = APIRouter(prefix="/location", tags=["Location Services"])


def _to_response(office: ServiceOffice) -> ServiceOfficeResponse:
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


# GET all

@router.get(
    "/services",
    response_model=list[ServiceOfficeResponse],
    summary="Get all service office locations",
)
def list_service_locations(
        skip: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=200),
        db: Session = Depends(get_db),
) -> list[ServiceOfficeResponse]:
    offices = db.query(ServiceOffice).offset(skip).limit(limit).all()
    return [_to_response(o) for o in offices]


# Get single by ID

@router.get(
    "/services/{location_id}",
    response_model=ServiceOfficeResponse,
    summary="Get a single service office by ID",
)
def get_service_location_by_id(
        location_id: int,
        db: Session = Depends(get_db),
) -> ServiceOfficeResponse:
    office = db.query(ServiceOffice).filter(ServiceOffice.id == location_id).first()
    if not office:
        raise HTTPException(status_code=404, detail=f"Location with id {location_id} not found.")
    return _to_response(office)


# Get by service name or service ID

@router.get(
    "/service",
    response_model=ServiceOfficeResponse,
    summary="Get office location by service name or service ID",
)
def get_service_location(
        service: str = Query(..., min_length=1, description="Service name or service_id"),
        db: Session = Depends(get_db),
) -> ServiceOfficeResponse:
    office = (
        db.query(ServiceOffice)
        .filter(
            or_(
                func.lower(ServiceOffice.service_name) == service.strip().lower(),
                ServiceOffice.service_id == service.strip().lower(),
            )
        )
        .first()
    )
    if not office:
        raise HTTPException(
            status_code=404,
            detail=f"No office found for service '{service}'.",
        )
    return _to_response(office)


# Post create

@router.post(
    "/services",
    response_model=ServiceOfficeResponse,
    status_code=201,
    summary="Create a new service office location",
)
def create_service_location(
        payload: ServiceOfficeCreate,
        db: Session = Depends(get_db),
) -> ServiceOfficeResponse:
    existing = db.query(ServiceOffice).filter(
        ServiceOffice.service_id == payload.service_id
    ).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"A location with service_id '{payload.service_id}' already exists.",
        )

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
    return _to_response(office)


# Put update

@router.put(
    "/services/{location_id}",
    response_model=ServiceOfficeResponse,
    summary="Update an existing service office location",
)
def update_service_location(
        location_id: int,
        payload: ServiceOfficeUpdate,
        db: Session = Depends(get_db),
) -> ServiceOfficeResponse:
    office = db.query(ServiceOffice).filter(ServiceOffice.id == location_id).first()
    if not office:
        raise HTTPException(status_code=404, detail=f"Location with id {location_id} not found.")

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

    db.commit()
    db.refresh(office)
    return _to_response(office)


# Delete

@router.delete(
    "/services/{location_id}",
    status_code=204,
    summary="Delete a service office location",
)
def delete_service_location(
        location_id: int,
        db: Session = Depends(get_db),
) -> None:
    office = db.query(ServiceOffice).filter(ServiceOffice.id == location_id).first()
    if not office:
        raise HTTPException(status_code=404, detail=f"Location with id {location_id} not found.")
    db.delete(office)
    db.commit()

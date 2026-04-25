from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.service_office import (
    MapLocationResponse,
    ServiceOfficeResponse,
    ServiceOfficeCreate,
    ServiceOfficeUpdate,
)
from app.services.location_service import (
    apply_service_location_update,
    create_service_location as create_location,
    delete_service_location as delete_location,
    get_map_ready_location,
    get_service_location as find_location_by_service,
    get_service_location_by_id as find_location_by_id,
    list_service_locations as list_locations,
    save_service_location,
    service_location_exists_by_service_id,
    to_response,
)
from app.services.openstreetmap_service import (
    OpenStreetMapGeocodingNotFoundError,
    OpenStreetMapGeocodingServiceError,
)

router = APIRouter(prefix="/location", tags=["Location"])


# GET all

@router.get(
    "",
    response_model=list[ServiceOfficeResponse],
    summary="Get all service office locations",
)
def list_service_locations(
        skip: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=200),
        db: Session = Depends(get_db),
) -> list[ServiceOfficeResponse]:
    return list_locations(db, skip, limit)


@router.get(
    "/map",
    response_model=MapLocationResponse,
    summary="Get map-ready coordinates for an institution",
)
def get_map_location(
        institution: str = Query(..., min_length=1, description="Service name, service_id, or office name"),
    address: str | None = Query(None, description="Optional address to geocode when no stored office exists"),
        db: Session = Depends(get_db),
) -> MapLocationResponse:
    try:
        return get_map_ready_location(db, institution, address)
    except OpenStreetMapGeocodingNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OpenStreetMapGeocodingServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


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
    office = find_location_by_service(db, service)
    if not office:
        raise HTTPException(
            status_code=404,
            detail=f"No office found for service '{service}'.",
        )
    return to_response(office)


# Get single by ID

@router.get(
    "/{location_id}",
    response_model=ServiceOfficeResponse,
    summary="Get a single service office by ID",
)
def get_service_location_by_id(
        location_id: int,
        db: Session = Depends(get_db),
) -> ServiceOfficeResponse:
    office = find_location_by_id(db, location_id)
    if not office:
        raise HTTPException(status_code=404, detail=f"Location with id {location_id} not found.")
    return to_response(office)


# Post create

@router.post(
    "",
    response_model=ServiceOfficeResponse,
    status_code=201,
    summary="Create a new service office location",
)
def create_service_location(
        payload: ServiceOfficeCreate,
        db: Session = Depends(get_db),
) -> ServiceOfficeResponse:
    if service_location_exists_by_service_id(db, payload.service_id):
        raise HTTPException(
            status_code=409,
            detail=f"A location with service_id '{payload.service_id}' already exists.",
        )
    office = create_location(db, payload)
    return to_response(office)


# Put update

@router.put(
    "/{location_id}",
    response_model=ServiceOfficeResponse,
    summary="Update an existing service office location",
)
def update_service_location(
        location_id: int,
        payload: ServiceOfficeUpdate,
        db: Session = Depends(get_db),
) -> ServiceOfficeResponse:
    office = find_location_by_id(db, location_id)
    if not office:
        raise HTTPException(status_code=404, detail=f"Location with id {location_id} not found.")
    office = apply_service_location_update(office, payload)
    office = save_service_location(db, office)
    return to_response(office)


# Delete

@router.delete(
    "/{location_id}",
    status_code=204,
    summary="Delete a service office location",
)
def delete_service_location(
        location_id: int,
        db: Session = Depends(get_db),
) -> None:
    office = find_location_by_id(db, location_id)
    if not office:
        raise HTTPException(status_code=404, detail=f"Location with id {location_id} not found.")
    delete_location(db, office)

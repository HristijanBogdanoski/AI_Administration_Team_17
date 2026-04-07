from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_, func

from app.db.session import get_db
from app.models.service_office import ServiceOffice
from app.schemas.service_office import ServiceOfficeResponse, CoordinatesSchema

router = APIRouter(prefix="/location", tags=["Location Services"])


def _to_response(office: ServiceOffice) -> ServiceOfficeResponse:
    return ServiceOfficeResponse(
        service_id=office.service_id,
        service_name=office.service_name,
        office_name=office.office_name,
        address=office.address,
        coordinates=CoordinatesSchema(lat=office.latitude, lng=office.longitude),
        working_hours=office.working_hours,
        contact_email=office.contact_email,
        notes=office.notes,
    )


@router.get(
    "/service",
    response_model=ServiceOfficeResponse,
    summary="Get office location for a government service",
    description=(
        "Returns office details (name, address, coordinates, working hours, contact email) "
        "for a given service. Accepts either a human-readable service name or a service ID."
    ),
)
def get_service_location(
    service: str = Query(
        ...,
        description="Service name (e.g. 'Passport Renewal') or service ID (e.g. 'passport-renewal')",
        min_length=1,
    ),
    db: Session = Depends(get_db),
) -> ServiceOfficeResponse:
    """
    Lookup a government service office by name or ID.

    - **service**: case-insensitive service name OR exact service_id
    """
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
            detail=f"No office found for service '{service}'. "
                   "Please check the service name or ID and try again.",
        )

    return _to_response(office)


@router.get(
    "/services",
    response_model=list[ServiceOfficeResponse],
    summary="List all available government service offices",
    description="Returns a paginated list of all registered service offices.",
)
def list_service_locations(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(50, ge=1, le=200, description="Max records to return"),
    db: Session = Depends(get_db),
) -> list[ServiceOfficeResponse]:
    offices = db.query(ServiceOffice).offset(skip).limit(limit).all()
    return [_to_response(o) for o in offices]
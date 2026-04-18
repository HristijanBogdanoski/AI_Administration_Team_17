from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.service import ServiceCreate, ServiceOut, ServiceUpdate
from app.core.security import require_admin
from app.services.service_service import (
    apply_service_update,
    create_service as create_service_record,
    delete_service as delete_service_record,
    get_all_services as list_services,
    get_service_by_id as find_service_by_id,
    save_service,
)

router = APIRouter(prefix="/services", tags=["Services"])


@router.get("", response_model=list[ServiceOut])
def get_all_services(db: Session = Depends(get_db)):
    return list_services(db)


@router.get("/{service_id}", response_model=ServiceOut)
def get_service_by_id(service_id: int, db: Session = Depends(get_db)):
    service = find_service_by_id(db, service_id)
    if service is None:
        raise HTTPException(status_code=404, detail="Service not found")
    return service


@router.post("", response_model=ServiceOut, status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_admin)])
def create_service(payload: ServiceCreate, db: Session = Depends(get_db)):
    service = create_service_record(db, payload)
    return service


@router.put("/{service_id}", response_model=ServiceOut, dependencies=[Depends(require_admin)])
def update_service(service_id: int, payload: ServiceUpdate, db: Session = Depends(get_db)):
    service = find_service_by_id(db, service_id)
    if service is None:
        raise HTTPException(status_code=404, detail="Service not found")

    service = apply_service_update(service, payload)
    service = save_service(db, service)
    return service


@router.delete("/{service_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_admin)])
def delete_service(service_id: int, db: Session = Depends(get_db)):
    service = find_service_by_id(db, service_id)
    if service is None:
        raise HTTPException(status_code=404, detail="Service not found")

    delete_service_record(db, service)
    return None

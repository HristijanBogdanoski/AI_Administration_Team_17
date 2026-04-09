from sqlalchemy.orm import Session

from app.models.service import Service
from app.schemas.service import ServiceCreate, ServiceUpdate


def get_all_services(db: Session) -> list[Service]:
    return db.query(Service).order_by(Service.id.asc()).all()


def get_service_by_id(db: Session, service_id: int) -> Service | None:
    return db.query(Service).filter(Service.id == service_id).first()


def create_service(db: Session, payload: ServiceCreate) -> Service:
    service = Service(**payload.model_dump())
    db.add(service)
    db.commit()
    db.refresh(service)
    return service


def apply_service_update(service: Service, payload: ServiceUpdate) -> Service:
    data = payload.model_dump(exclude_unset=True)

    if "details" in data and data["details"] is None:
        data["details"] = []

    for key, value in data.items():
        setattr(service, key, value)

    return service


def save_service(db: Session, service: Service) -> Service:
    db.commit()
    db.refresh(service)
    return service


def delete_service(db: Session, service: Service) -> None:
    db.delete(service)
    db.commit()

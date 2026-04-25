"""Seed services from SERVICE_META and attach office locations when available.

Usage:
    python seed_services.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.db.session import SessionLocal
from app.models.enums import ServiceCategory
from app.models.service import Service
from app.models.service_office import ServiceOffice


# Small metadata map from the frontend dataset, keyed by service_id from ServiceOffice.
SERVICE_META = {
    "passport-renewal": {
        "name": "Пасош",
        "category": ServiceCategory.documents,
        "description": "Барање за пасош и издавање на биометриска патна исправа.",
        "processing_time_days": 15,
        "details": ["Биометриски пасош", "Важност 10 години", "Достапна е итна услуга"],
    },
    "birth-certificate": {
        "name": "Извод од Матична Книга на Родени",
        "category": ServiceCategory.documents,
        "description": "Издавање или повторно издавање на извод од матична книга на родени.",
        "processing_time_days": 3,
        "details": ["Услуга во матична служба", "Потребна е лична карта", "Подигнување или е-услуга"],
    },
    "vehicle-registration": {
        "name": "Регистрација на Возило",
        "category": ServiceCategory.documents,
        "description": "Постапка за регистрација на возило и поврзани услуги.",
        "processing_time_days": 7,
        "details": ["Категории A, B, C, D", "Потребно е лекарско уверение", "Закажување термин во центар"],
    },
    "tax-filing": {
        "name": "Даночна Пријава",
        "category": ServiceCategory.taxes,
        "description": "Поднесување на даночна пријава и даночни обрасци.",
        "processing_time_days": 0,
        "details": ["Рок до 15 март секоја година", "Достапно е онлајн поднесување", "Автоматски пресметки"],
    },
    "unemployment-benefit": {
        "name": "Социјална Помош",
        "category": ServiceCategory.social,
        "description": "Барање за социјална финансиска помош.",
        "processing_time_days": 30,
        "details": ["Проверка на услови", "Месечна исплата", "Поддршка од центар за социјална работа"],
    },
    "health-insurance-card": {
        "name": "Здравствено Осигурување",
        "category": ServiceCategory.social,
        "description": "Пријава или обнова на здравствено осигурување.",
        "processing_time_days": 7,
        "details": ["Задолжително осигурување", "Обработка во Фонд за здравство", "Достапна е семејна пријава"],
    },
    "land-registry": {
        "name": "Имотен Лист",
        "category": ServiceCategory.business,
        "description": "Услуги за имотен лист и потврди за сопственост.",
        "processing_time_days": 5,
        "details": ["Евиденција за сопственост", "Поддршка за е-Катастар", "Достапна е поддршка на шалтер"],
    },
    "business-registration": {
        "name": "Регистрација Фирма",
        "category": ServiceCategory.business,
        "description": "Основање и регистрација на нова правна фирма.",
        "processing_time_days": 5,
        "details": ["Постапка во Централен Регистар", "Опции ДОО, ДООЕЛ и АД", "Процес на еден шалтер"],
    },
}


def _upsert_service(existing_by_name: dict[str, Service], payload: dict) -> tuple[int, int, Service]:
    existing = existing_by_name.get(payload["name"].lower())
    if existing is None:
        created = Service(**payload)
        existing_by_name[payload["name"].lower()] = created
        return 1, 0, created

    changed = False
    for key, value in payload.items():
        if getattr(existing, key) != value:
            setattr(existing, key, value)
            changed = True
    return 0, int(changed), existing


def seed() -> None:
    db = SessionLocal()
    try:
        existing_by_name = {row.name.lower(): row for row in db.query(Service).all()}
        created = 0
        updated = 0

        offices_by_service_id = {
            office.service_id: office for office in db.query(ServiceOffice).all()
        }

        for service_id, meta in SERVICE_META.items():
            office = offices_by_service_id.get(service_id)
            payload = {
                "service_id": service_id,
                "name": meta["name"],
                "category": meta["category"],
                "description": meta["description"],
                "processing_time_days": meta.get("processing_time_days"),
                "details": meta["details"],
                "location": (
                    f"{office.latitude}, {office.longitude}" if office else None
                ),
            }

            c, u, service = _upsert_service(existing_by_name, payload)
            created += c
            updated += u
            if c:
                db.add(service)

        db.commit()
        print(f"Services seed complete. Created: {created}, Updated: {updated}")
    finally:
        db.close()


if __name__ == "__main__":
    seed()

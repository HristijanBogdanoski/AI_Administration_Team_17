"""Seed services with metadata from SERVICE_META.

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
from app.models.service_document_template import ServiceDocumentTemplate
from app.services.service_document_template_service import build_default_template_body


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


def seed() -> None:
    db = SessionLocal()
    try:
        existing_by_service_id = {row.service_id: row for row in db.query(Service).all()}
        existing_templates_by_service_id = {
            row.service_id: row for row in db.query(ServiceDocumentTemplate).all()
        }
        created = 0
        updated = 0
        template_created = 0
        template_updated = 0

        for service_id, meta in SERVICE_META.items():
            if service_id in existing_by_service_id:
                service = existing_by_service_id[service_id]
                changed = False
                for key, value in meta.items():
                    if getattr(service, key) != value:
                        setattr(service, key, value)
                        changed = True
                updated += int(changed)
            else:
                service = Service(service_id=service_id, **meta)
                db.add(service)
                created += 1

            template_title = f"{meta['name']} - формулар за апликација"
            template_body = build_default_template_body(meta["name"])
            template = existing_templates_by_service_id.get(service_id)
            if template is None:
                db.add(
                    ServiceDocumentTemplate(
                        service_id=service_id,
                        title=template_title,
                        template_type="json",
                        template_body=template_body,
                        is_active=True,
                    )
                )
                template_created += 1
            else:
                template.title = template_title
                template.template_type = "json"
                template.template_body = template_body
                template.is_active = True
                template_updated += 1

        db.commit()
        print(
            "Services seed complete. "
            f"Created: {created}, Updated: {updated}, "
            f"Templates created: {template_created}, Templates updated: {template_updated}"
        )
    finally:
        db.close()


if __name__ == "__main__":
    seed()

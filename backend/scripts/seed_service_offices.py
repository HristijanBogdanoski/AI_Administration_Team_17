"""
Seed the service_offices table with realistic Macedonian government service data.

Usage:
    python seed_service_offices.py
"""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.db.session import SessionLocal
from app.models.service import Service
from app.models.service_office import ServiceOffice

OFFICES = [
    {
        "service_id": "passport-renewal",
        "service_name": "Обнова на Пасош",
        "office_name": "Министерство за внатрешни работи – Оддел за пасоши",
        "address": "Димитар Влахов бб, 1000 Скопје",
        "latitude": 41.9981,
        "longitude": 21.4254,
        "working_hours": "Пон–Пет 08:00–16:00",
        "contact_email": "pasosi@mvr.gov.mk",
        "notes": "Силно се препорачува закажување термин. Понесете две фотографии за пасош и важечка лична карта.",
    },
    {
        "service_id": "birth-certificate",
        "service_name": "Извод од Матична Книга на Родени",
        "office_name": "Општина Центар – Матична служба",
        "address": "Булевар Партизански Одреди 42, 1000 Скопје",
        "latitude": 41.9944,
        "longitude": 21.4314,
        "working_hours": "Пон–Пет 08:30–15:30",
        "contact_email": "maticna@centar.gov.mk",
        "notes": "За прва регистрација е потребен оригинален болнички документ за раѓање.",
    },
    {
        "service_id": "vehicle-registration",
        "service_name": "Регистрација на Возило",
        "office_name": "Министерство за внатрешни работи – Центар за регистрација на возила",
        "address": "Орце Николов 177, 1000 Скопје",
        "latitude": 42.0058,
        "longitude": 21.4089,
        "working_hours": "Пон–Пет 07:30–15:00",
        "contact_email": "registracija@mvr.gov.mk",
        "notes": "Плаќањето на патен данок мора да биде завршено пред посетата. Потребно е задолжително осигурување.",
    },
    {
        "service_id": "tax-filing",
        "service_name": "Даночна Пријава",
        "office_name": "Управа за јавни приходи – Регионален центар Скопје",
        "address": "Даме Груев 14, 1000 Скопје",
        "latitude": 41.9965,
        "longitude": 21.4342,
        "working_hours": "Пон–Пет 08:00–16:00",
        "contact_email": "contact@ujp.gov.mk",
        "notes": "Онлајн поднесување е достапно на e-ujp.ujp.gov.mk. Посета на шалтер е потребна само за посложени случаи.",
    },
    {
        "service_id": "unemployment-benefit",
        "service_name": "Надоместок за Невработеност",
        "office_name": "Агенција за вработување – Центар Скопје",
        "address": "Јане Сандански 39, 1000 Скопје",
        "latitude": 41.9901,
        "longitude": 21.4406,
        "working_hours": "Пон–Пет 08:00–16:00",
        "contact_email": "info@avrm.gov.mk",
        "notes": "Понесете доказ за претходно вработување и важечка лична карта. Регистрацијата е во рок од 30 дена од престанок на работа.",
    },
    {
        "service_id": "health-insurance-card",
        "service_name": "Картичка за Здравствено Осигурување",
        "office_name": "Фонд за здравствено осигурување – Регионална канцеларија Скопје",
        "address": "Македонија 12, 1000 Скопје",
        "latitude": 41.9959,
        "longitude": 21.4318,
        "working_hours": "Пон–Пет 08:30–16:00",
        "contact_email": "fond@fzo.org.mk",
        "notes": "Потребен е доказ за активно вработување или пензија. Обработката трае до 5 работни дена.",
    },
    {
        "service_id": "land-registry",
        "service_name": "Имотен Лист",
        "office_name": "Агенција за катастар на недвижности – Скопје",
        "address": "Ѓуро Ѓаковиќ 5, 1000 Скопје",
        "latitude": 41.9971,
        "longitude": 21.4197,
        "working_hours": "Пон–Пет 08:00–17:00",
        "contact_email": "skopje@katastar.gov.mk",
        "notes": "Потврди за сопственост може да се добијат и преку порталот е-Катастар.",
    },
    {
        "service_id": "business-registration",
        "service_name": "Регистрација на Фирма",
        "office_name": "Централен регистар на Северна Македонија",
        "address": "Св. Климент Охридски 4, 1000 Скопје",
        "latitude": 41.9935,
        "longitude": 21.4350,
        "working_hours": "Пон–Пет 08:00–16:00",
        "contact_email": "contact@crm.org.mk",
        "notes": "Нови компании може да се регистрираат онлајн преку e-registracija.crm.org.mk за помалку од 4 часа.",
    },
]


def seed() -> None:
    db = SessionLocal()
    try:
        # Load existing services and offices
        existing_services = {row.service_id for row in db.query(Service.service_id).all()}
        existing_office_ids = {row.service_id for row in db.query(ServiceOffice.service_id).all()}
        
        new_offices = []
        skipped = 0
        
        for data in OFFICES:
            service_id = data["service_id"]
            
            # Skip if service doesn't exist
            if service_id not in existing_services:
                print(f"⚠ Skipping office for '{service_id}' – service does not exist.")
                skipped += 1
                continue
            
            # Skip if office already exists
            if service_id in existing_office_ids:
                continue
            
            new_offices.append(ServiceOffice(**data))
        
        if new_offices:
            db.add_all(new_offices)
            db.commit()
            print(f"✓ Seeded {len(new_offices)} service office(s).")
        
        if skipped > 0:
            print(f"⚠ Skipped {skipped} office(s) – parent service not found.")
        
        if not new_offices and skipped == 0:
            print("All service offices already present – nothing to seed.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
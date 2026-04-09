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
from app.models.service_office import ServiceOffice

OFFICES = [
    {
        "service_id": "passport-renewal",
        "service_name": "Passport Renewal",
        "office_name": "Ministry of Interior – Passport Department",
        "address": "Dimitar Vlahov bb, 1000 Skopje",
        "latitude": 41.9981,
        "longitude": 21.4254,
        "working_hours": "Mon–Fri 08:00–16:00",
        "contact_email": "pasosи@mvr.gov.mk",
        "notes": "Appointments strongly recommended. Bring two passport photos and a valid ID.",
    },
    {
        "service_id": "birth-certificate",
        "service_name": "Birth Certificate",
        "office_name": "Municipality of Centar – Registry Office",
        "address": "Bulevar Partizanski Odredi 42, 1000 Skopje",
        "latitude": 41.9944,
        "longitude": 21.4314,
        "working_hours": "Mon–Fri 08:30–15:30",
        "contact_email": "maticna@centar.gov.mk",
        "notes": "Original hospital birth record required for first-time registration.",
    },
    {
        "service_id": "vehicle-registration",
        "service_name": "Vehicle Registration",
        "office_name": "Ministry of Interior – Vehicle Registration Center",
        "address": "Orce Nikolov 177, 1000 Skopje",
        "latitude": 42.0058,
        "longitude": 21.4089,
        "working_hours": "Mon–Fri 07:30–15:00",
        "contact_email": "registracija@mvr.gov.mk",
        "notes": "Payment of road tax must be completed before visiting. ZOUK insurance required.",
    },
    {
        "service_id": "tax-filing",
        "service_name": "Tax Filing",
        "office_name": "Public Revenue Office – Skopje Regional Center",
        "address": "Dame Gruev 14, 1000 Skopje",
        "latitude": 41.9965,
        "longitude": 21.4342,
        "working_hours": "Mon–Fri 08:00–16:00",
        "contact_email": "contact@ujp.gov.mk",
        "notes": "Online filing available at e-ujp.ujp.gov.mk. Office visits for complex cases only.",
    },
    {
        "service_id": "unemployment-benefit",
        "service_name": "Unemployment Benefit",
        "office_name": "Employment Service Agency – Skopje Center",
        "address": "Jane Sandanski 39, 1000 Skopje",
        "latitude": 41.9901,
        "longitude": 21.4406,
        "working_hours": "Mon–Fri 08:00–16:00",
        "contact_email": "info@avrm.gov.mk",
        "notes": "Bring proof of previous employment and a valid ID. Register within 30 days of termination.",
    },
    {
        "service_id": "health-insurance-card",
        "service_name": "Health Insurance Card",
        "office_name": "Health Insurance Fund – Skopje Regional Office",
        "address": "Makedonija 12, 1000 Skopje",
        "latitude": 41.9959,
        "longitude": 21.4318,
        "working_hours": "Mon–Fri 08:30–16:00",
        "contact_email": "fond@fzo.org.mk",
        "notes": "Proof of active employment or pension is required. Processing takes up to 5 business days.",
    },
    {
        "service_id": "land-registry",
        "service_name": "Land Registry",
        "office_name": "Agency for Real Estate Cadastre – Skopje",
        "address": "Gjuro Gjakovikj 5, 1000 Skopje",
        "latitude": 41.9971,
        "longitude": 21.4197,
        "working_hours": "Mon–Fri 08:00–17:00",
        "contact_email": "skopje@katastar.gov.mk",
        "notes": "Property ownership certificates can also be obtained via e-katastar portal.",
    },
    {
        "service_id": "business-registration",
        "service_name": "Business Registration",
        "office_name": "Central Registry of North Macedonia",
        "address": "Sv. Kliment Ohridski 4, 1000 Skopje",
        "latitude": 41.9935,
        "longitude": 21.4350,
        "working_hours": "Mon–Fri 08:00–16:00",
        "contact_email": "contact@crm.org.mk",
        "notes": "New companies can be registered online via e-registracija.crm.org.mk in under 4 hours.",
    },
]


def seed() -> None:
    db = SessionLocal()
    try:
        existing_ids = {row.service_id for row in db.query(ServiceOffice.service_id).all()}
        new_offices = [
            ServiceOffice(**data) for data in OFFICES if data["service_id"] not in existing_ids
        ]
        if new_offices:
            db.add_all(new_offices)
            db.commit()
            print(f"Seeded {len(new_offices)} service office(s).")
        else:
            print("All service offices already present – nothing to seed.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
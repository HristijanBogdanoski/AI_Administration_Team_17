"""
Verify OpenStreetMap geocoding for five real Skopje institution addresses.

Usage:
    python scripts/verify_osm_geocoding.py
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.openstreetmap_service import build_skopje_query, geocode_skopje_query


TEST_CASES = [
    {
        "institution": "Ministry of Interior – Passport Department",
        "address": "Dimitar Vlahov bb, 1000 Skopje",
    },
    {
        "institution": "Municipality of Centar – Registry Office",
        "address": "Bulevar Partizanski Odredi 42, 1000 Skopje",
    },
    {
        "institution": "Ministry of Interior – Vehicle Registration Center",
        "address": "Orce Nikolov 177, 1000 Skopje",
    },
    {
        "institution": "Public Revenue Office – Skopje Regional Center",
        "address": "Dame Gruev 14, 1000 Skopje",
    },
    {
        "institution": "Government of the Republic of North Macedonia",
        "address": "Ilindenska 2, 1000 Skopje",
    },
]


def main() -> None:
    for case in TEST_CASES:
        search_query = build_skopje_query(case["institution"], case["address"])
        result = geocode_skopje_query(search_query)
        print(
            f"{case['institution']} | {search_query} | "
            f"{result.lat:.6f}, {result.lng:.6f} | {result.display_name}\n"
        )


if __name__ == "__main__":
    main()
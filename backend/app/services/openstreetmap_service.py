from dataclasses import dataclass
from functools import lru_cache

import httpx


NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = "AI-Administration-Team-17/1.0 (FastAPI OpenStreetMap geocoding)"


class OpenStreetMapGeocodingError(RuntimeError):
    pass


class OpenStreetMapGeocodingNotFoundError(OpenStreetMapGeocodingError):
    pass


class OpenStreetMapGeocodingServiceError(OpenStreetMapGeocodingError):
    pass


@dataclass(frozen=True)
class OpenStreetMapLocation:
    lat: float
    lng: float
    display_name: str
    place_id: int | None = None
    osm_type: str | None = None
    osm_id: int | None = None
    importance: float | None = None


def build_skopje_query(institution: str, address: str | None = None) -> str:
    parts: list[str] = []

    normalized_institution = " ".join(institution.split()).strip()
    if normalized_institution:
        parts.append(normalized_institution)

    if address:
        normalized_address = " ".join(address.split()).strip()
        if normalized_address and normalized_address.lower() not in normalized_institution.lower():
            parts.append(normalized_address)

    combined = ", ".join(parts) if parts else normalized_institution
    lowered = combined.lower()
    if "skopje" not in lowered:
        combined = f"{combined}, Skopje, North Macedonia" if combined else "Skopje, North Macedonia"
    elif "north macedonia" not in lowered and "macedonia" not in lowered:
        combined = f"{combined}, North Macedonia"

    return combined


@lru_cache(maxsize=256)
def geocode_skopje_query(search_query: str) -> OpenStreetMapLocation:
    headers = {
        "User-Agent": NOMINATIM_USER_AGENT,
        "Accept-Language": "en,mk;q=0.8",
    }
    params = {
        "q": search_query,
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 1,
        "countrycodes": "mk",
    }

    try:
        with httpx.Client(timeout=10.0, headers=headers) as client:
            response = client.get(NOMINATIM_SEARCH_URL, params=params)
            response.raise_for_status()
    except httpx.RequestError as exc:
        raise OpenStreetMapGeocodingServiceError(
            f"Unable to reach OpenStreetMap geocoding service: {exc}"
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise OpenStreetMapGeocodingServiceError(
            f"OpenStreetMap geocoding service returned HTTP {exc.response.status_code}."
        ) from exc

    results = response.json()
    if not results:
        raise OpenStreetMapGeocodingNotFoundError(
            f"No OpenStreetMap result found for '{search_query}'."
        )

    candidate = results[0]
    return OpenStreetMapLocation(
        lat=float(candidate["lat"]),
        lng=float(candidate["lon"]),
        display_name=candidate.get("display_name", search_query),
        place_id=candidate.get("place_id"),
        osm_type=candidate.get("osm_type"),
        osm_id=candidate.get("osm_id"),
        importance=float(candidate["importance"]) if candidate.get("importance") is not None else None,
    )
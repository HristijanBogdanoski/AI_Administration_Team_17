from copy import deepcopy
import re
from io import BytesIO

from sqlalchemy.orm import Session

from app.models.service import Service
from app.models.user import User
from app.models.service_document_template import ServiceDocumentTemplate

# PDF/DOCX generation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import simpleSplit
from docx import Document
import os
from app.schemas.service_document_template import (
    ServiceDocumentTemplateCreate,
    ServiceDocumentTemplateUpdate,
)


def build_default_template_body(service_name: str) -> dict:
    return {
        "document_type": "service_application",
        "service_name": service_name,
        "sections": [
            {
                "title": "Податоци за апликантот",
                "fields": [
                    {"key": "full_name", "label": "Име и презиме", "value": ""},
                    {"key": "embg", "label": "ЕМБГ", "value": ""},
                    {"key": "address", "label": "Адреса", "value": ""},
                    {"key": "phone_number", "label": "Телефон", "value": ""},
                    {"key": "gender", "label": "Пол", "value": ""},
                    {"key": "email", "label": "Е-пошта", "value": ""},
                ],
            },
            {
                "title": "Детали за услугата",
                "fields": [
                    {"key": "service_name", "label": "Услуга", "value": service_name},
                    {"key": "notes", "label": "Забелешки", "value": ""},
                ],
            },
        ],
    }


def get_user_field_values(user: User) -> dict[str, str | None]:
    return {
        "full_name": user.full_name,
        "email": user.email,
        "embg": user.embg,
        "address": user.address,
        "phone_number": user.phone_number,
        "gender": user.gender,
    }


def apply_user_values_to_template(
    template_body: dict,
    user: User,
    selected_fields: list[str] | None = None,
) -> dict:
    filled_body = deepcopy(template_body)
    user_values = get_user_field_values(user)
    allowed_fields = set(selected_fields or user_values.keys())

    for section in filled_body.get("sections", []):
        for field in section.get("fields", []):
            key = field.get("key")
            if key in allowed_fields and key in user_values and user_values[key]:
                field["value"] = user_values[key]
            if key == "phone" and "phone_number" in allowed_fields and user_values.get("phone_number"):
                field["value"] = user_values["phone_number"]

    return filled_body


def render_template_document(template_body: dict) -> str:
    lines: list[str] = []
    service_name = template_body.get("service_name")
    if service_name:
        lines.append(str(service_name))
        lines.append("=" * len(str(service_name)))
        lines.append("")

    for section in template_body.get("sections", []):
        title = section.get("title")
        if title:
            lines.append(str(title))
        for field in section.get("fields", []):
            label = field.get("label", field.get("key", "Field"))
            value = field.get("value")
            display_value = value if value not in (None, "") else "________________"
            lines.append(f"{label}: {display_value}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def text_to_pdf_bytes(text: str) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 40
    # Register a TrueType font that supports Cyrillic if available on the system.
    font_name = "DejaVuSans"
    candidate_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:\\Windows\\Fonts\\DejaVuSans.ttf",
        "C:\\Windows\\Fonts\\ARIALUNI.TTF",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]
    registered = False
    for p in candidate_paths:
        try:
            if os.path.exists(p):
                pdfmetrics.registerFont(TTFont(font_name, p))
                registered = True
                break
        except Exception:
            continue

    if not registered:
        # Try registering a default DejaVuSans by name (if reportlab has access), else fallback to built-in.
        try:
            pdfmetrics.registerFont(TTFont(font_name, "DejaVuSans.ttf"))
            registered = True
        except Exception:
            font_name = "Helvetica"

    y = height - margin
    line_height = 14
    # Use simpleSplit to wrap text by available width when using TrueType font
    max_width = width - margin * 2
    for raw_line in text.splitlines():
        if y < margin:
            c.showPage()
            y = height - margin
        if len(raw_line.strip()) == 0:
            y -= line_height
            continue
        if font_name != "Helvetica":
            # wrap using font metrics
            wrapped = simpleSplit(raw_line, font_name, 12, max_width)
        else:
            # naive wrap for fallback
            max_chars = 95
            wrapped = [raw_line[i:i+max_chars] for i in range(0, len(raw_line), max_chars)]

        c.setFont(font_name, 12)
        for part in wrapped:
            if y < margin:
                c.showPage()
                y = height - margin
                c.setFont(font_name, 12)
            c.drawString(margin, y, part)
            y -= line_height
    c.save()
    buffer.seek(0)
    return buffer.read()


def text_to_docx_bytes(text: str) -> bytes:
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.read()


def service_exists_by_service_id(db: Session, service_id: str) -> bool:
    return db.query(Service.id).filter(Service.service_id == service_id).first() is not None


def get_template_by_service_id(db: Session, service_id: str) -> ServiceDocumentTemplate | None:
    return (
        db.query(ServiceDocumentTemplate)
        .filter(ServiceDocumentTemplate.service_id == service_id)
        .first()
    )


def upsert_template(db: Session, payload: ServiceDocumentTemplateCreate) -> ServiceDocumentTemplate:
    template = get_template_by_service_id(db, payload.service_id)
    if template is None:
        template = ServiceDocumentTemplate(**payload.model_dump())
        db.add(template)
    else:
        data = payload.model_dump()
        for key, value in data.items():
            setattr(template, key, value)

    db.commit()
    db.refresh(template)
    return template


def apply_template_update(
    template: ServiceDocumentTemplate,
    payload: ServiceDocumentTemplateUpdate,
) -> ServiceDocumentTemplate:
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(template, key, value)
    return template


def create_blank_template_for_service(db: Session, service: Service) -> ServiceDocumentTemplate:
    existing_template = get_template_by_service_id(db, service.service_id)
    if existing_template is not None:
        return existing_template

    template = ServiceDocumentTemplate(
        service_id=service.service_id,
        title=f"{service.name} - формулар за апликација",
        template_type="json",
        template_body=build_default_template_body(service.name),
        is_active=True,
    )
    db.add(template)
    db.commit()
    db.refresh(template)
    return template


def get_template_by_service_name(db: Session, service_name: str) -> ServiceDocumentTemplate | None:
    return (
        db.query(ServiceDocumentTemplate)
        .join(Service, ServiceDocumentTemplate.service_id == Service.service_id)
        .filter(Service.name == service_name)
        .first()
    )


def detect_template_from_uploaded_document(
    db: Session,
    filename: str,
    document_text: str,
) -> ServiceDocumentTemplate | None:
    filename_match = re.match(r"^(?P<service_id>.+?)-application-form(?:-filled)?\.txt$", filename)
    if filename_match:
        template = get_template_by_service_id(db, filename_match.group("service_id"))
        if template is not None:
            return template

    first_non_empty_line = next((line.strip() for line in document_text.splitlines() if line.strip()), "")
    if first_non_empty_line:
        template = get_template_by_service_name(db, first_non_empty_line)
        if template is not None:
            return template

    return None
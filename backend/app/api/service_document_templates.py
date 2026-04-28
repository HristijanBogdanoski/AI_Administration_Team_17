import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from sqlalchemy.orm import Session

from app.core.security import require_admin
from app.db.session import get_db
from app.schemas.service_document_template import ServiceDocumentTemplateFillRequest
from app.schemas.service_document_template import (
    ServiceDocumentTemplateCreate,
    ServiceDocumentTemplateOut,
    ServiceDocumentTemplateUpdate,
)
from app.services.service_document_template_service import (
    apply_user_values_to_template,
    apply_template_update,
    detect_template_from_uploaded_document,
    render_template_document,
    get_template_by_service_id,
    get_template_by_service_name,
    service_exists_by_service_id,
    upsert_template,
    text_to_pdf_bytes,
    text_to_docx_bytes,
)
from app.core.security import get_current_user
from app.models.user import User

router = APIRouter(prefix="/service-document-templates", tags=["Service Document Templates"])


@router.get("/{service_id}", response_model=ServiceDocumentTemplateOut, dependencies=[Depends(require_admin)])
def get_template(service_id: str, db: Session = Depends(get_db)):
    template = get_template_by_service_id(db, service_id)
    if template is None:
        raise HTTPException(status_code=404, detail="Template not found")
    return template


@router.post("", response_model=ServiceDocumentTemplateOut, status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_admin)])
def create_or_replace_template(payload: ServiceDocumentTemplateCreate, db: Session = Depends(get_db)):
    if not service_exists_by_service_id(db, payload.service_id):
        raise HTTPException(status_code=404, detail=f"Service with service_id '{payload.service_id}' does not exist.")
    return upsert_template(db, payload)


@router.put("/{service_id}", response_model=ServiceDocumentTemplateOut, dependencies=[Depends(require_admin)])
def update_template(service_id: str, payload: ServiceDocumentTemplateUpdate, db: Session = Depends(get_db)):
    template = get_template_by_service_id(db, service_id)
    if template is None:
        raise HTTPException(status_code=404, detail="Template not found")

    template = apply_template_update(template, payload)
    db.commit()
    db.refresh(template)
    return template


@router.get("/{service_id}/download")
def download_blank_template(service_id: str, format: str = "txt", db: Session = Depends(get_db)):
    template = get_template_by_service_id(db, service_id)
    if template is None:
        raise HTTPException(status_code=404, detail="Template not found")
    document_text = render_template_document(template.template_body)
    fmt = format.lower()
    if fmt == "pdf":
        content = text_to_pdf_bytes(document_text)
        filename = f"{service_id}-application-form.pdf"
        media_type = "application/pdf"
    elif fmt in ("docx", "word"):
        content = text_to_docx_bytes(document_text)
        filename = f"{service_id}-application-form.docx"
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        content = document_text
        filename = f"{service_id}-application-form.txt"
        media_type = "text/plain; charset=utf-8"

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{service_id}/auto-fill")
def auto_fill_template(
    service_id: str,
    payload: ServiceDocumentTemplateFillRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    format: str = "txt",
):
    template = get_template_by_service_id(db, service_id)
    if template is None:
        raise HTTPException(status_code=404, detail="Template not found")

    filled_body = apply_user_values_to_template(
        template.template_body,
        current_user,
        payload.selected_fields,
    )
    document_text = render_template_document(filled_body)
    fmt = format.lower()
    if fmt == "pdf":
        content = text_to_pdf_bytes(document_text)
        filename = f"{service_id}-application-form-filled.pdf"
        media_type = "application/pdf"
    elif fmt in ("docx", "word"):
        content = text_to_docx_bytes(document_text)
        filename = f"{service_id}-application-form-filled.docx"
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        content = document_text
        filename = f"{service_id}-application-form-filled.txt"
        media_type = "text/plain; charset=utf-8"

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/upload-fill")
async def upload_and_fill_template(
    file: UploadFile = File(...),
    selected_fields: str = Form(default="[]"),
    output_format: str = Form(default="txt"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        selected_fields_list = json.loads(selected_fields)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="selected_fields must be valid JSON") from exc

    if not isinstance(selected_fields_list, list):
        raise HTTPException(status_code=400, detail="selected_fields must be a JSON array")

    raw_bytes = await file.read()
    try:
        document_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Uploaded document must be UTF-8 text") from exc

    template = detect_template_from_uploaded_document(db, file.filename or "", document_text)
    if template is None:
        raise HTTPException(
            status_code=404,
            detail="Could not detect the service for the uploaded document. Please upload the blank file downloaded from Services or Chat.",
        )

    filled_body = apply_user_values_to_template(template.template_body, current_user, selected_fields_list)
    filled_text = render_template_document(filled_body)
    fmt = output_format.lower()
    if fmt == "pdf":
        content = text_to_pdf_bytes(filled_text)
        filename = f"{template.service_id}-application-form-filled.pdf"
        media_type = "application/pdf"
    elif fmt in ("docx", "word"):
        content = text_to_docx_bytes(filled_text)
        filename = f"{template.service_id}-application-form-filled.docx"
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        content = filled_text
        filename = f"{template.service_id}-application-form-filled.txt"
        media_type = "text/plain; charset=utf-8"

    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Detected-Service-Id": template.service_id,
        },
    )
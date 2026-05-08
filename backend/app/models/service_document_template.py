from sqlalchemy import Boolean, Column, ForeignKey, Integer, JSON, String

from app.db.session import Base


class ServiceDocumentTemplate(Base):
    __tablename__ = "service_document_templates"

    id = Column(Integer, primary_key=True, index=True)
    # reference the service by numeric primary key `services.id`
    service_id = Column(
        Integer,
        ForeignKey("services.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    title = Column(String(255), nullable=False)
    template_type = Column(String(20), nullable=False, default="json")
    template_body = Column(JSON, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
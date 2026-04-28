from sqlalchemy import Column, Enum, Integer, JSON, String, Text

from app.db.session import Base
from app.models.enums import ServiceCategory


class Service(Base):
    __tablename__ = "services"

    id = Column(Integer, primary_key=True, index=True)
    service_id = Column(String(100), nullable=False, unique=True, index=True)

    name = Column(String(255), nullable=False, index=True)
    category = Column(
        Enum(ServiceCategory, name="service_category"),
        nullable=False,
        index=True,
    )
    description = Column(Text, nullable=True)

    processing_time_days = Column(Integer, nullable=True)
    details = Column(JSON, nullable=False, default=list)
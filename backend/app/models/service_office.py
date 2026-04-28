from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text
from app.db.session import Base


class ServiceOffice(Base):
    __tablename__ = "service_offices"

    id = Column(Integer, primary_key=True, index=True)
    service_name = Column(String(255), nullable=False, index=True)
    service_id = Column(String(100), ForeignKey("services.service_id", ondelete="CASCADE"), nullable=False, index=True)
    office_name = Column(String(255), nullable=False)
    address = Column(String(500), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    working_hours = Column(String(255), nullable=False)
    contact_email = Column(String(255), nullable=False)
    notes = Column(Text, nullable=True)
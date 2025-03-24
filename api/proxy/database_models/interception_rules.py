"""Database model for proxy interception rules."""
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, JSON, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import text
from database.base import Base

class InterceptionRule(Base):
    """Model for storing request/response interception rules."""
    __tablename__ = "interception_rules"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    enabled = Column(Boolean, default=True)
    session_id = Column(Integer, ForeignKey("proxy_sessions.id"), nullable=False)
    conditions = Column(JSON, nullable=False)  # List of {field, operator, value, use_regex}
    action = Column(String, nullable=False)  # FORWARD, BLOCK, MODIFY
    modification = Column(JSON, nullable=True)  # Headers/body modifications to apply
    priority = Column(Integer, nullable=False)
    created_at = Column(DateTime, server_default=text('CURRENT_TIMESTAMP'))
    updated_at = Column(DateTime, server_default=text('CURRENT_TIMESTAMP'))

    session = relationship("ProxySession", back_populates="interception_rules")

    def __repr__(self):
        return f"<InterceptionRule(id={self.id}, name='{self.name}', action='{self.action}')>"

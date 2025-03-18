"""Database model for proxy tunnel metrics."""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import text
from database.base import Base

class TunnelMetrics(Base):
    """Model for storing proxy tunnel metrics."""
    __tablename__ = "tunnel_metrics"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, server_default=text('CURRENT_TIMESTAMP'))
    connection_id = Column(String)  # UUID of the connection
    direction = Column(String)  # "request" or "response"
    bytes_transferred = Column(Integer)  # Number of bytes in this chunk
    cumulative_bytes = Column(Integer)  # Running total for this connection direction
    host = Column(String)  # Target hostname
    port = Column(Integer)  # Target port
    chunk_number = Column(Integer)  # Sequential chunk number
    session_id = Column(Integer, ForeignKey("proxy_sessions.id"))

    session = relationship("ProxySession", back_populates="tunnel_metrics")

    def __repr__(self):
        return f"<TunnelMetrics(id={self.id}, connection_id='{self.connection_id}', bytes={self.bytes_transferred})>"

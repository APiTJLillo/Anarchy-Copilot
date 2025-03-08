"""Database interceptor for storing proxy history."""
from datetime import datetime
import json
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from api.proxy.database_models import ProxyHistoryEntry
from proxy.interceptor import ProxyInterceptor, InterceptedRequest, InterceptedResponse 
from database import engine


class DatabaseInterceptor(ProxyInterceptor):
    """Interceptor that stores requests and responses in the database."""

    def __init__(self, session_id: int):
        """Initialize the database interceptor.
        
        Args:
            session_id: ID of the active proxy session to link entries to
        """
        super().__init__()
        self.session_id = session_id
        self.async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        self._current_request: Optional[ProxyHistoryEntry] = None
        self._start_time: Optional[float] = None
        self._db: Optional[AsyncSession] = None

    async def _get_db(self) -> AsyncSession:
        """Get database session, creating if needed."""
        if self._db is None:
            self._db = self.async_session()
        return self._db

    async def intercept_request(self, request: InterceptedRequest) -> InterceptedRequest:
        """Store request details and return unmodified request."""
        try:
            # Create new history entry
            entry = ProxyHistoryEntry(
                session_id=self.session_id,
                timestamp=datetime.utcnow(),
                method=request.method,
                url=request.url,
                request_headers=dict(request.headers),
                request_body=request.body.decode('utf-8', errors='ignore') if request.body else None,
                tags=["HTTPS"]
            )
            
            # Save to track timing and associate with response
            self._current_request = entry
            self._start_time = datetime.utcnow().timestamp()
            
            # Add to database
            db = await self._get_db()
            db.add(entry)
            await db.commit()
            
            return request
        except Exception as e:
            logger = logging.getLogger("proxy.core")
            logger.error(f"Error storing request: {e}")
            return request

    async def intercept_response(self, response: InterceptedResponse, request: InterceptedRequest) -> InterceptedResponse:
        """Store response details and return unmodified response."""
        try:
            if self._current_request and self._start_time:
                # Calculate request duration
                duration = datetime.utcnow().timestamp() - self._start_time
                
                # Update history entry with response data
                self._current_request.response_status = response.status_code
                self._current_request.response_headers = dict(response.headers)
                self._current_request.response_body = response.body.decode('utf-8', errors='ignore') if response.body else None
                self._current_request.duration = duration
                
                # Get TLS info from response headers if available
                tls_info = self._extract_tls_info(response.headers)
                if tls_info:
                    self._current_request.tls_version = tls_info.get('version')
                    self._current_request.cipher_suite = tls_info.get('cipher')
                    self._current_request.certificate_info = tls_info.get('cert_info')
                
                # Update in database
                db = await self._get_db()
                await db.commit()
                
                # Clear request tracking
                self._current_request = None
                self._start_time = None
            
            return response
        except Exception as e:
            logger = logging.getLogger("proxy.core")
            logger.error(f"Error storing response: {e}")
            return response

    async def close(self):
        """Close database session."""
        if self._db:
            await self._db.close()
            self._db = None
    
    def _extract_tls_info(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """Extract TLS information from response headers.
        
        The proxy adds these headers during TLS interception.
        """
        tls_info = {}
        
        # Extract version
        if 'X-TLS-Version' in headers:
            tls_info['version'] = headers['X-TLS-Version']
            
        # Extract cipher suite  
        if 'X-TLS-Cipher' in headers:
            tls_info['cipher'] = headers['X-TLS-Cipher']
            
        # Extract certificate info
        if 'X-TLS-Certificate' in headers:
            try:
                cert_info = json.loads(headers['X-TLS-Certificate'])
                tls_info['cert_info'] = cert_info
            except json.JSONDecodeError:
                pass
                
        return tls_info

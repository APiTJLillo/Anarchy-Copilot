"""
Session management for the Anarchy Copilot proxy module.

This module handles proxy sessions, storing request/response history,
and managing the state of intercepted traffic.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, AsyncContextManager
from collections import deque
import uuid
import re
import aiohttp
from contextlib import asynccontextmanager

from .interceptor import InterceptedRequest, InterceptedResponse

@dataclass(eq=True)
class HistoryEntry:
    """Represents a single request/response pair in the proxy history."""
    id: str
    timestamp: datetime
    request: InterceptedRequest
    response: Optional[InterceptedResponse] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate the request duration if response exists."""
        if not hasattr(self, '_duration'):
            self._duration = None
        return self._duration
    
    @duration.setter
    def duration(self, value: float) -> None:
        """Set the request duration."""
        self._duration = value

    def __eq__(self, other: object) -> bool:
        """Compare history entries for equality."""
        if not isinstance(other, HistoryEntry):
            return False
        return (
            self.id == other.id and
            self.request == other.request and
            self.response == other.response and
            self.tags == other.tags and
            self.notes == other.notes
        )
    
    def to_dict(self) -> dict:
        """Convert history entry to a dictionary format."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'request': self.request.to_dict() if self.request else None,
            'response': self.response.to_dict() if self.response else None,
            'duration': self.duration,
            'tags': self.tags,
            'notes': self.notes
        }

class ProxySession:
    """Manages a proxy session including history and state."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize a new proxy session.
        
        Args:
            max_history: Maximum number of requests to keep in history
        """
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self._history: deque[HistoryEntry] = deque(maxlen=max_history)
        self._pending_requests: Dict[str, HistoryEntry] = {}
        self.metadata: Dict[str, str] = {}
        self._client: Optional[aiohttp.ClientSession] = None

    @property
    def client(self) -> aiohttp.ClientSession:
        """Get the aiohttp client session, creating it if needed."""
        if self._client is None:
            self._client = aiohttp.ClientSession()
        return self._client

    async def close(self) -> None:
        """Close the session and cleanup resources."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    @asynccontextmanager
    async def request(self, request: InterceptedRequest) -> AsyncContextManager[aiohttp.ClientResponse]:
        """Perform an HTTP request and track it in history.
        
        Args:
            request: The intercepted request to forward
            
        Returns:
            An async context manager that yields the aiohttp response
        """
        # Create history entry
        entry = self.create_history_entry(request)
        start_time = datetime.now()

        try:
            # Forward request using aiohttp
            async with self.client.request(
                method=request.method,
                url=request.url,
                headers=request.headers,
                data=request.body
            ) as response:
                # Calculate duration
                duration = (datetime.now() - start_time).total_seconds()
                
                # Create intercepted response
                intercepted_response = InterceptedResponse(
                    status_code=response.status,
                    headers=dict(response.headers),
                    body=await response.read()
                )
                
                # Complete history entry
                entry.duration = duration
                self.complete_history_entry(entry.id, intercepted_response)
                
                yield response

        except Exception as e:
            # Clean up pending request on error
            if entry.id in self._pending_requests:
                del self._pending_requests[entry.id]
            raise
    
    def create_history_entry(self, request: InterceptedRequest) -> HistoryEntry:
        """Create and store a new history entry for a request.
        
        Args:
            request: The intercepted request
            
        Returns:
            The created history entry
        """
        entry = HistoryEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            request=request
        )
        self._pending_requests[entry.id] = entry
        return entry
    
    def complete_history_entry(self, entry_id: str, response: InterceptedResponse) -> None:
        """Complete a history entry with its response.
        
        Args:
            entry_id: ID of the history entry
            response: The intercepted response
        """
        if entry_id in self._pending_requests:
            entry = self._pending_requests.pop(entry_id)
            entry.response = response
            self._history.append(entry)
    
    def get_history(self, limit: Optional[int] = None) -> List[HistoryEntry]:
        """Get the session history, optionally limited to a number of entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of history entries
        """
        if limit is None:
            return list(self._history)
        return list(self._history)[-limit:]
    
    def find_entry(self, entry_id: str) -> Optional[HistoryEntry]:
        """Find a history entry by its ID.
        
        Args:
            entry_id: ID of the history entry to find
            
        Returns:
            The found entry or None
        """
        # First check pending requests
        if entry_id in self._pending_requests:
            return self._pending_requests[entry_id]
        
        # Then check completed history
        for entry in self._history:
            if entry.id == entry_id:
                return entry
        return None
    
    def search_history(self, 
                      url_pattern: Optional[str] = None,
                      method: Optional[str] = None,
                      tags: Optional[List[str]] = None) -> List[HistoryEntry]:
        """Search history entries based on criteria.
        
        Args:
            url_pattern: Regex pattern to match URLs against
            method: HTTP method to filter by
            tags: List of tags that must all be present
            
        Returns:
            List of matching history entries
        """
        results = []
        for entry in self._history:
            if method and entry.request.method != method:
                continue
            if url_pattern and not re.search(url_pattern, entry.request.url):
                continue
            if tags and not all(tag in entry.tags for tag in tags):
                continue
            results.append(entry)
        return results
    
    def add_entry_tag(self, entry_id: str, tag: str) -> bool:
        """Add a tag to a history entry.
        
        Args:
            entry_id: ID of the entry to tag
            tag: Tag to add
            
        Returns:
            True if the tag was added, False if entry not found
        """
        entry = self.find_entry(entry_id)
        if entry and tag not in entry.tags:
            entry.tags.append(tag)
            return True
        return False
    
    def set_entry_note(self, entry_id: str, note: str) -> bool:
        """Set a note on a history entry.
        
        Args:
            entry_id: ID of the entry
            note: Note text to set
            
        Returns:
            True if the note was set, False if entry not found
        """
        entry = self.find_entry(entry_id)
        if entry:
            entry.notes = note
            return True
        return False
    
    def clear_history(self) -> None:
        """Clear all history entries."""
        self._history.clear()
        self._pending_requests.clear()
    
    def export_history(self) -> dict:
        """Export the session history in a serializable format.
        
        Returns:
            Dictionary containing session data and history
        """
        return {
            'session_id': self.id,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata,
            'history': [entry.to_dict() for entry in self._history]
        }

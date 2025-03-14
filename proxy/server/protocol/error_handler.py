"""Error handling for HTTPS interception."""
import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger("proxy.core")

class ErrorHandler:
    """Handles errors in HTTPS interception."""

    def __init__(self, connection_id: str):
        """Initialize error handler.
        
        Args:
            connection_id: Unique connection identifier
        """
        self._connection_id = connection_id
        self._error_count = 0
        self._last_error = None
        self._error_history: Dict[str, Any] = {}

    def handle_error(self, error_type: str, error: Optional[Exception] = None) -> None:
        """Handle an error.
        
        Args:
            error_type: Type of error that occurred
            error: Optional exception object
        """
        self._error_count += 1
        self._last_error = {
            'type': error_type,
            'message': str(error) if error else None,
            'timestamp': datetime.utcnow()
        }
        
        # Track error history
        if error_type not in self._error_history:
            self._error_history[error_type] = {
                'count': 0,
                'first_seen': datetime.utcnow(),
                'last_seen': None,
                'examples': []
            }
            
        history = self._error_history[error_type]
        history['count'] += 1
        history['last_seen'] = datetime.utcnow()
        
        if len(history['examples']) < 5:  # Keep up to 5 examples
            history['examples'].append(str(error) if error else None)

        # Log error with appropriate severity
        if self._error_count > 10:
            logger.error(f"[{self._connection_id}] Excessive errors ({self._error_count}), latest: {error_type}")
        else:
            logger.warning(f"[{self._connection_id}] {error_type}: {error}")

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics.
        
        Returns:
            Dictionary containing error statistics
        """
        return {
            'total_errors': self._error_count,
            'last_error': self._last_error,
            'error_history': self._error_history
        }

    def should_terminate(self) -> bool:
        """Check if connection should be terminated due to errors.
        
        Returns:
            True if connection should be terminated
        """
        # Terminate if too many errors overall
        if self._error_count > 20:
            return True
            
        # Terminate if too many recent errors
        if self._last_error:
            recent_errors = sum(
                1 for hist in self._error_history.values()
                if hist['last_seen'] and (datetime.utcnow() - hist['last_seen']).seconds < 60
            )
            if recent_errors > 5:
                return True
                
        return False

    def reset(self) -> None:
        """Reset error statistics."""
        self._error_count = 0
        self._last_error = None
        self._error_history.clear()

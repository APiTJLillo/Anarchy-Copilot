"""Error handling and reporting for proxy protocols."""
import logging
import ssl
from typing import Dict, Any, Optional

logger = logging.getLogger("proxy.core")

class ErrorHandler:
    """Handles error reporting and responses in proxy protocols."""
    
    def __init__(self, connection_id: str, transport: Any):
        self._connection_id = connection_id
        self._transport = transport
        self._last_error: Optional[Dict[str, Any]] = None

    def send_error(self, status_code: int, message: str) -> None:
        """Send HTTP error response."""
        if not self._transport or self._transport.is_closing():
            logger.warning(f"[{self._connection_id}] Cannot send error - transport closed")
            return

        try:
            status_text = {
                400: "Bad Request",
                403: "Forbidden", 
                404: "Not Found",
                500: "Internal Server Error",
                502: "Bad Gateway",
                503: "Service Unavailable",
                504: "Gateway Timeout"
            }.get(status_code, "Unknown Error")

            response = (
                f"HTTP/1.1 {status_code} {status_text}\r\n"
                f"Content-Type: text/plain\r\n"
                f"Connection: close\r\n"
                f"\r\n"
                f"Error {status_code}: {message}\r\n"
            )
            self._transport.write(response.encode())

        except Exception as e:
            logger.error(f"[{self._connection_id}] Failed to send error response: {e}")

    def handle_tls_error(self, error: Exception, stage: str = "unknown") -> None:
        """Handle TLS-specific errors with enhanced logging."""
        try:
            error_info = {
                'stage': stage,
                'type': type(error).__name__,
                'message': str(error)
            }

            if isinstance(error, ssl.SSLError):
                # Extract SSL-specific error details
                error_info.update({
                    'errno': getattr(error, 'errno', None),
                    'library': getattr(error, 'library', None),
                    'reason': getattr(error, 'reason', None),
                    'verify_code': getattr(error, 'verify_code', None),
                    'verify_message': getattr(error, 'verify_message', None)
                })

                # Enhanced SSL error classification
                if "syscall" in str(error).lower():
                    logger.error(f"[{self._connection_id}] SSL_ERROR_SYSCALL during {stage}:")
                    logger.error(f"  Details: Connection interrupted or socket error")
                    self.send_error(502, f"TLS connection failed: Network error during {stage}")
                elif "alert" in str(error).lower():
                    logger.error(f"[{self._connection_id}] TLS alert received during {stage}:")
                    logger.error(f"  Alert: {error.reason if hasattr(error, 'reason') else 'unknown'}")
                    self.send_error(502, f"TLS alert: {error.reason if hasattr(error, 'reason') else str(error)}")
                elif "handshake failure" in str(error).lower():
                    logger.error(f"[{self._connection_id}] TLS handshake failure during {stage}:")
                    logger.error(f"  Reason: {error.reason if hasattr(error, 'reason') else 'unknown'}")
                    self.send_error(502, f"TLS handshake failed: Version or cipher mismatch")
                elif "certificate" in str(error).lower():
                    logger.error(f"[{self._connection_id}] Certificate error during {stage}:")
                    logger.error(f"  Details: {error.verify_message if hasattr(error, 'verify_message') else str(error)}")
                    self.send_error(502, f"TLS certificate error: {str(error)}")
                else:
                    logger.error(f"[{self._connection_id}] SSL error during {stage}:")
                    logger.error(f"  Details: {error}")
                    self.send_error(502, f"TLS error: {str(error)}")

            elif isinstance(error, TimeoutError):
                logger.error(f"[{self._connection_id}] Timeout during {stage}")
                self.send_error(504, f"Operation timed out during {stage}")

            elif isinstance(error, ConnectionError):
                logger.error(f"[{self._connection_id}] Connection error during {stage}")
                self.send_error(502, f"Connection failed during {stage}")

            else:
                logger.error(f"[{self._connection_id}] Error during {stage}: {error}")
                self.send_error(500, f"Internal error during {stage}: {str(error)}")

            # Store error info
            self._last_error = error_info
            
            # Log complete error details at debug level
            logger.debug(f"[{self._connection_id}] Complete error details: {error_info}")

        except Exception as e:
            logger.error(f"[{self._connection_id}] Error handler failed: {e}")
            self.send_error(500, "Internal error in error handler")

    def handle_error(self, error: Exception, stage: str = "unknown") -> None:
        """Handle any type of error."""
        try:
            error_info = {
                'stage': stage,
                'type': type(error).__name__,
                'message': str(error)
            }

            if isinstance(error, ssl.SSLError):
                # For SSL errors, use specialized handler
                self.handle_tls_error(error, stage)
            else:
                # For other errors, send appropriate response
                if isinstance(error, TimeoutError):
                    logger.error(f"[{self._connection_id}] Timeout during {stage}")
                    self.send_error(504, f"Operation timed out during {stage}")
                elif isinstance(error, ConnectionError):
                    logger.error(f"[{self._connection_id}] Connection error during {stage}")
                    self.send_error(502, f"Connection failed during {stage}")
                else:
                    logger.error(f"[{self._connection_id}] Error during {stage}: {error}")
                    self.send_error(500, f"Internal error during {stage}: {str(error)}")

            # Store error info
            self._last_error = error_info
            
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error handler failed: {e}")
            self.send_error(500, "Internal error in error handler")

    def get_last_error(self) -> Optional[Dict[str, Any]]:
        """Get details of last error handled."""
        return self._last_error

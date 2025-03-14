"""Connection and TLS state management."""
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger("proxy.core")

class TlsState:
    """Container for TLS-specific state information."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset TLS state to defaults."""
        self._state = {
            "hostname": None,
            "client_hello_seen": False,
            "handshake_complete": False,
            "sni_hostname": None,
            "alpn_protocols": None,
            "cipher": None,
            "version": None,
            "client_version": None,
            "client_ciphers": None,
            "client_hello_data": None,
            "handshake_attempts": 0,
            "last_handshake_error": None,
            "negotiation_complete": False,
            "selected_version": None,
            "selected_cipher": None,
            "connection_state": "initializing"
        }

    def update(self, **kwargs) -> None:
        """Update TLS state with new values."""
        self._state.update(kwargs)
        # Log important state changes
        if "handshake_complete" in kwargs or "connection_state" in kwargs:
            logger.debug(f"TLS state update: {kwargs}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from TLS state."""
        return self._state.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert TLS state to dictionary."""
        return self._state.copy()

class StateManager:
    """Manages protocol state for HTTPS interception."""

    def __init__(self, connection_id: Optional[str] = None):
        """Initialize state manager.
        
        Args:
            connection_id: Optional connection ID
        """
        self._connection_id = connection_id
        self._state = {
            'handshake_complete': False,
            'client_hello_received': False,
            'server_hello_received': False,
            'is_client_side': True,
            'tls_version': None,
            'cipher_suite': None,
            'start_time': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            'bytes_sent': 0,
            'bytes_received': 0,
            'requests_processed': 0,
            'responses_processed': 0
        }
        self._metadata: Dict[str, Any] = {}
        self._tunnel_established = False
        self._intercept_enabled = False
        self._last_error = None
        self._negotiation_retries = 0
        self.tls_state = TlsState()
        
        if connection_id:
            logger.debug(f"[{connection_id}] StateManager initialized")

    def update_state(self, **kwargs) -> None:
        """Update state values.
        
        Args:
            **kwargs: State values to update
        """
        self._state.update(kwargs)
        self._state['last_activity'] = datetime.utcnow()

    def get_state(self, key: str) -> Any:
        """Get state value.
        
        Args:
            key: State key to get
            
        Returns:
            State value or None if not found
        """
        return self._state.get(key)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self._metadata[key] = value

    def get_metadata(self, key: str) -> Optional[Any]:
        """Get metadata value.
        
        Args:
            key: Metadata key
            
        Returns:
            Metadata value or None if not found
        """
        return self._metadata.get(key)

    def is_handshake_complete(self) -> bool:
        """Check if TLS handshake is complete.
        
        Returns:
            True if handshake is complete
        """
        return self._state['handshake_complete']

    def is_client_side(self) -> bool:
        """Check if this is client side of connection.
        
        Returns:
            True if client side
        """
        return self._state['is_client_side']

    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information.
        
        Returns:
            Dictionary with connection information
        """
        now = datetime.utcnow()
        duration = (now - self._state['start_time']).total_seconds()
        idle_time = (now - self._state['last_activity']).total_seconds()
        
        return {
            'duration': duration,
            'idle_time': idle_time,
            'bytes_sent': self._state['bytes_sent'],
            'bytes_received': self._state['bytes_received'],
            'requests': self._state['requests_processed'],
            'responses': self._state['responses_processed'],
            'tls_version': self._state['tls_version'],
            'cipher_suite': self._state['cipher_suite'],
            'metadata': self._metadata
        }

    def reset(self) -> None:
        """Reset state to initial values."""
        self._state = {
            'handshake_complete': False,
            'client_hello_received': False,
            'server_hello_received': False,
            'is_client_side': True,
            'tls_version': None,
            'cipher_suite': None,
            'start_time': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            'bytes_sent': 0,
            'bytes_received': 0,
            'requests_processed': 0,
            'responses_processed': 0
        }
        self._metadata.clear()

    async def register_connection(self, protocol_type: str = "https") -> None:
        """Register connection with state tracking.
        
        Args:
            protocol_type: Type of protocol being registered
        """
        from ..state import proxy_state
        await proxy_state.add_connection(self._connection_id, {
            "type": protocol_type,
            "status": "initializing",
            "tls_state": self.tls_state.to_dict(),
            "intercept_enabled": self._intercept_enabled,
            "bytes_received": 0,
            "bytes_sent": 0,
            "negotiation_retries": 0,
            "last_error": None
        })

    async def update_bytes(self, received: int = 0, sent: int = 0) -> None:
        """Update byte counters.
        
        Args:
            received: Number of bytes received
            sent: Number of bytes sent
        """
        from ..state import proxy_state
        if received:
            await proxy_state.update_connection(
                self._connection_id,
                "bytes_received",
                received
            )
        if sent:
            await proxy_state.update_connection(
                self._connection_id,
                "bytes_sent",
                sent
            )

    async def update_status(self, status: str, error: Optional[str] = None) -> None:
        """Update connection status.
        
        Args:
            status: New status value
            error: Optional error message
        """
        from ..state import proxy_state
        await proxy_state.update_connection(self._connection_id, "status", status)
        if error:
            self._last_error = error
            await proxy_state.update_connection(self._connection_id, "error", error)
        self.tls_state.update(connection_state=status)

    async def update_tls_info(self) -> None:
        """Update TLS state information in state tracking."""
        from ..state import proxy_state
        await proxy_state.update_connection(
            self._connection_id, 
            "tls_info", 
            self.tls_state.to_dict()
        )

    def set_intercept_enabled(self, enabled: bool) -> None:
        """Set TLS interception state.
        
        Args:
            enabled: Whether TLS interception is enabled
        """
        self._intercept_enabled = enabled
        logger.debug(f"[{self._connection_id}] TLS interception {'enabled' if enabled else 'disabled'}")

    def is_intercept_enabled(self) -> bool:
        """Check if TLS interception is enabled."""
        return self._intercept_enabled

    def set_tunnel_established(self, established: bool) -> None:
        """Set tunnel establishment state.
        
        Args:
            established: Whether tunnel is established
        """
        self._tunnel_established = established
        logger.debug(f"[{self._connection_id}] Tunnel {'established' if established else 'closed'}")
        self.tls_state.update(
            connection_state="established" if established else "closed",
            handshake_complete=established
        )

    def is_tunnel_established(self) -> bool:
        """Check if tunnel is established."""
        return self._tunnel_established

    async def cleanup(self) -> None:
        """Clean up connection state and resources."""
        from ..state import proxy_state
        try:
            # Reset TLS state
            self.tls_state.reset()
            
            # Clear tunnel state
            self._tunnel_established = False
            self._intercept_enabled = False
            
            # Remove from global state
            await proxy_state.remove_connection(self._connection_id)
            
            logger.debug(f"[{self._connection_id}] Connection state cleaned up")
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error during cleanup: {e}")

    def clear_state(self) -> None:
        """Clear all state information."""
        try:
            # Reset TLS state
            self.tls_state.reset()
            
            # Reset connection flags
            self._tunnel_established = False
            self._intercept_enabled = False
            self._last_error = None
            self._negotiation_retries = 0
            
            logger.debug(f"[{self._connection_id}] State cleared")
        except Exception as e:
            logger.error(f"[{self._connection_id}] Error clearing state: {e}")

    def parse_client_hello(self, data: bytes) -> None:
        """Parse TLS ClientHello data for state tracking.
        
        Args:
            data: Raw ClientHello data
        """
        try:
                # Check if this looks like a ClientHello
                if len(data) >= 5 and data[0] == 0x16:  # Handshake
                    # First look at the TLS record version
                    record_ver_major = data[1]
                    record_ver_minor = data[2]

                    # Store first part of ClientHello for analysis
                    self.tls_state.update(
                        client_hello_data=bytes(data[:256]),
                        connection_state="client_hello_received"
                    )

                    # Parse ClientHello message
                    if len(data) >= 43:  # Minimum size to get to supported versions
                        handshake_type = data[5]
                        if handshake_type == 0x01:  # ClientHello
                            client_version_major = data[9]
                            client_version_minor = data[10]

                            # Check for TLS 1.0 in various places:
                            logger.debug(f"[{self._connection_id}] TLS Record Version: {record_ver_major:02x}.{record_ver_minor:02x}")
                            logger.debug(f"[{self._connection_id}] ClientHello Version: {client_version_major:02x}.{client_version_minor:02x}")
                            
                            # Log TLS version detection
                            logger.debug(f"[{self._connection_id}] TLS Record Version: {record_ver_major:02x}.{record_ver_minor:02x}")
                            logger.debug(f"[{self._connection_id}] ClientHello Version: {client_version_major:02x}.{client_version_minor:02x}")

                            # Only force TLS 1.0 when explicitly requested
                            force_tls10 = "--tlsv1.0" in str(self._connection_id)
                            
                            if force_tls10:
                                # Explicitly requested TLS 1.0
                                self.tls_state.update(
                                    client_version="TLSv1.0",
                                    client_hello_seen=True,
                                    connection_state="legacy_tls_requested"
                                )
                                logger.debug(f"[{self._connection_id}] TLS 1.0 explicitly requested")
                            else:
                                # Determine version from ClientHello (modern behavior)
                                if client_version_major == 0x03:
                                    version = f"TLSv1.{client_version_minor-1}" if client_version_minor <= 4 else "TLS Unknown"
                                    self.tls_state.update(
                                        client_version=version,
                                        client_hello_seen=True,
                                        connection_state="modern_tls_detected"
                                    )
                                    logger.debug(f"[{self._connection_id}] Detected TLS version: {version}")

                            # Extract client cipher suites
                            try:
                                # Skip to cipher suites section
                                pos = 38 + data[37]  # 37 is session id length
                                if len(data) > pos + 2:
                                    cipher_len = int.from_bytes(data[pos:pos+2], 'big')
                                    pos += 2
                                    if len(data) >= pos + cipher_len:
                                        cipher_data = data[pos:pos + cipher_len]
                                        cipher_list = []
                                        for i in range(0, len(cipher_data), 2):
                                            cipher_id = int.from_bytes(cipher_data[i:i+2], 'big')
                                            cipher_list.append(hex(cipher_id))
                                        self.tls_state.update(
                                            client_ciphers=cipher_list,
                                            connection_state="ciphers_extracted"
                                        )
                                        logger.debug(f"[{self._connection_id}] Client cipher suites: {cipher_list}")
                            except Exception as e:
                                logger.warning(f"[{self._connection_id}] Error parsing client cipher suites: {e}")

                            # Extract SNI from ClientHello if present
                            try:
                                pos = 37  # Skip 5 bytes header + 32 bytes client random
                                if len(data) > pos + 2:
                                    session_id_len = data[pos]
                                    pos += 1 + session_id_len
                                    
                                    if len(data) > pos + 2:
                                        cipher_suites_len = int.from_bytes(data[pos:pos+2], 'big')
                                        pos += 2 + cipher_suites_len
                                        
                                        if len(data) > pos + 1:
                                            compression_methods_len = data[pos]
                                            pos += 1 + compression_methods_len
                                            
                                            if len(data) > pos + 2:
                                                extensions_len = int.from_bytes(data[pos:pos+2], 'big')
                                                pos += 2
                                                
                                                # Parse extensions
                                                end = pos + extensions_len
                                                while pos < end and len(data) > pos + 4:
                                                    ext_type = int.from_bytes(data[pos:pos+2], 'big')
                                                    ext_len = int.from_bytes(data[pos+2:pos+4], 'big')
                                                    pos += 4
                                                    
                                                    # SNI extension type is 0
                                                    if ext_type == 0 and len(data) >= pos + ext_len:
                                                        # Skip server name list length (2 bytes)
                                                        # and name type (1 byte)
                                                        name_pos = pos + 3
                                                        if len(data) >= name_pos + 2:
                                                            name_len = int.from_bytes(data[name_pos:name_pos+2], 'big')
                                                            name_pos += 2
                                                            if len(data) >= name_pos + name_len:
                                                                sni = data[name_pos:name_pos+name_len].decode('utf-8')
                                                                self.tls_state.update(
                                                                    sni_hostname=sni,
                                                                    connection_state="sni_extracted"
                                                                )
                                                                logger.debug(f"[{self._connection_id}] Extracted SNI: {sni}")
                                                    
                                                    pos += ext_len
                            except Exception as e:
                                logger.warning(f"[{self._connection_id}] Error parsing SNI from ClientHello: {e}")

        except Exception as e:
            logger.error(f"[{self._connection_id}] Error parsing ClientHello: {e}")
            self.tls_state.update(
                last_handshake_error=str(e),
                connection_state="client_hello_error"
            )

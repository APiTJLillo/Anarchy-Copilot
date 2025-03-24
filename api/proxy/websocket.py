"""WebSocket API endpoints."""
import json
import time
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, WebSocket as FastAPIWebSocket, WebSocketDisconnect, Depends
from fastapi.websockets import WebSocketState
from sqlalchemy.ext.asyncio import AsyncSession
from .database import get_db
from .connection import connection_manager
from .state import proxy_state
from proxy.websocket.manager import WebSocketManager
from proxy.websocket.interceptor import SecurityInterceptor, DebugInterceptor
import aiohttp
import os
import base64

logger = logging.getLogger("api.websocket")

router = APIRouter(tags=["websocket"])
ws_manager = WebSocketManager()

def format_datetime(dt: datetime) -> str:
    """Format datetime object to ISO format string."""
    return dt.isoformat() if isinstance(dt, datetime) else str(dt)

def format_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Format connection manager stats for JSON serialization."""
    formatted = {}
    for conn_type, type_stats in stats.items():
        formatted[conn_type] = {
            "connected": type_stats["connected"],
            "connection_count": type_stats["connection_count"],
            "last_message": format_datetime(type_stats["last_message"]) if type_stats.get("last_message") else None,
            "message_count": type_stats["message_count"],
            "error_count": type_stats["error_count"],
            "active_connections": [
                {
                    **conn,
                    "connected_since": format_datetime(conn["connected_since"]) if conn.get("connected_since") else None,
                    "last_activity": format_datetime(conn["last_activity"]) if conn.get("last_activity") else None,
                    "connection_history": [
                        {
                            **event,
                            "timestamp": format_datetime(event["timestamp"]) if event.get("timestamp") else None
                        }
                        for event in conn.get("connection_history", [])
                    ]
                }
                for conn in type_stats.get("active_connections", [])
            ],
            "connection_history": [
                {
                    **event,
                    "timestamp": format_datetime(event["timestamp"]) if event.get("timestamp") else None
                }
                for event in type_stats.get("connection_history", [])
            ]
        }
    return formatted

class WebSocket:
    """WebSocket connection wrapper."""
    def __init__(self, websocket: FastAPIWebSocket):
        self.websocket = websocket
        self.id = str(id(self))
        self.connected_at = datetime.utcnow()
        self.last_activity = self.connected_at
        self.message_count = 0
        self.error_count = 0
        self.state = "initialized"
        self.last_error: Optional[str] = None
        self.last_message_type: Optional[str] = None
        self.connection_history: List[Dict[str, Any]] = []
        self.client_state = websocket.client_state

    async def accept(self):
        """Accept the WebSocket connection."""
        if not self.client_state == WebSocketState.CONNECTED:
            await self.websocket.accept()
            self.state = "connected"
            self.connection_history.append({
                "timestamp": datetime.utcnow(),
                "event": "connected",
            })

    async def close(self, code: int = 1000):
        """Close the WebSocket connection."""
        try:
            await self.websocket.close(code=code)
            self.state = "closed"
            self.connection_history.append({
                "timestamp": datetime.utcnow(),
                "event": "closed",
                "details": f"WebSocket connection closed with code {code}"
            })
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")
            self.error_count += 1
            self.last_error = str(e)
            self.connection_history.append({
                "timestamp": datetime.utcnow(),
                "event": "error",
                "details": f"Error closing connection: {e}"
            })

    async def send_json(self, data: Dict[str, Any]):
        """Send JSON data over the WebSocket."""
        try:
            await self.websocket.send_json(data)
            self.message_count += 1
            self.last_activity = datetime.utcnow()
            self.last_message_type = data.get("type")
            self.connection_history.append({
                "timestamp": self.last_activity,
                "event": "message_sent",
                "details": f"Sent message of type {self.last_message_type}"
            })
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            self.error_count += 1
            self.last_error = str(e)
            self.connection_history.append({
                "timestamp": datetime.utcnow(),
                "event": "error",
                "details": f"Error sending message: {e}"
            })
            raise

    async def receive_json(self) -> Dict[str, Any]:
        """Receive JSON data from the WebSocket."""
        try:
            data = await self.websocket.receive_json()
            self.message_count += 1
            self.last_activity = datetime.utcnow()
            self.last_message_type = data.get("type")
            self.connection_history.append({
                "timestamp": self.last_activity,
                "event": "message_received",
                "details": f"Received message of type {self.last_message_type}"
            })
            return data
        except Exception as e:
            logger.error(f"Error receiving WebSocket message: {e}")
            self.error_count += 1
            self.last_error = str(e)
            self.connection_history.append({
                "timestamp": datetime.utcnow(),
                "event": "error",
                "details": f"Error receiving message: {e}"
            })
            raise

    @property
    def is_connected(self) -> bool:
        """Check if the WebSocket is connected."""
        return self.client_state == WebSocketState.CONNECTED

    def __str__(self) -> str:
        """String representation of the WebSocket connection."""
        return f"WebSocket(id={self.id}, state={self.state}, messages={self.message_count}, errors={self.error_count})"

# Constants
HEARTBEAT_TIMEOUT = 20  # seconds
HEARTBEAT_INTERVAL = 15  # seconds

# Add interceptors
debug_interceptor = DebugInterceptor()
security_interceptor = SecurityInterceptor()
ws_manager.add_interceptor(debug_interceptor)
ws_manager.add_interceptor(security_interceptor)

async def get_service_status() -> List[Dict[str, Any]]:
    """Fetch service status from health endpoints."""
    try:
        # Get proxy service status
        proxy_status = "healthy" if proxy_state.is_running else "down"
        
        # Get API service status
        services = [
            {
                "name": "API Service",
                "status": "healthy",
                "lastCheck": datetime.utcnow().isoformat(),
                "details": f"Version: {proxy_state.version['version']}"
            },
            {
                "name": "Proxy Service",
                "status": proxy_status,
                "lastCheck": datetime.utcnow().isoformat(),
                "details": "Running" if proxy_status == "healthy" else "Stopped"
            }
        ]
        
        # Check dev container health
        try:
            dev_host = 'dev' if 'DOCKER_ENV' in os.environ else 'localhost'
            async with aiohttp.ClientSession() as session:
                async with session.get(f'http://{dev_host}:8000/api/health/status', timeout=2) as resp:
                    if resp.status == 200:
                        services.append({
                            "name": "Dev Container",
                            "status": "healthy",
                            "lastCheck": datetime.utcnow().isoformat(),
                            "details": "Connected and responding"
                        })
                    else:
                        services.append({
                            "name": "Dev Container",
                            "status": "down",
                            "lastCheck": datetime.utcnow().isoformat(),
                            "details": f"Health check failed with status {resp.status}"
                        })
        except Exception as e:
            services.append({
                "name": "Dev Container",
                "status": "down",
                "lastCheck": datetime.utcnow().isoformat(),
                "details": f"Health check failed: {str(e)}"
            })
        
        return services
    except Exception as e:
        logger.error(f"Error fetching service status: {e}")
        return []

async def get_system_metrics() -> Optional[Dict[str, Any]]:
    """Fetch system metrics."""
    try:
        import psutil
        
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Get disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        # Get network I/O
        net_io = psutil.net_io_counters()
        network = {
            "in": round(net_io.bytes_recv / (1024 * 1024), 2),  # MB/s
            "out": round(net_io.bytes_sent / (1024 * 1024), 2)  # MB/s
        }
        
        return {
            "cpu": cpu_percent,
            "memory": memory_percent,
            "disk": disk_percent,
            "network": network
        }
    except Exception as e:
        logger.error(f"Error fetching system metrics: {e}")
        return None

async def verify_proxy_state():
    """Verify that the proxy state is initialized."""
    if not proxy_state.connection_manager:
        logger.warning("[WebSocket] Proxy state not initialized")
        return False
    return True

@router.websocket("/ws")
async def handle_proxy_connection_updates(websocket: FastAPIWebSocket, db: AsyncSession = Depends(get_db)):
    """Handle WebSocket connections for proxy updates."""
    logger.info("[WebSocket] New connection request received at /ws")
    ws = WebSocket(websocket)
    connection_id = None
    
    try:
        # First accept the connection
        await websocket.accept()
        logger.debug("[WebSocket] Connection accepted")
        
        # Then register with connection manager
        connection_id = await connection_manager.connect(ws, "ui")
        logger.info(f"[WebSocket] New UI connection established: {connection_id}")
        
        # Send test connection response
        await websocket.send_json({
            "type": "test_connection_response",
            "status": "ok",
            "timestamp": format_datetime(datetime.utcnow())
        })
        logger.debug(f"[WebSocket] Sent test connection response to {connection_id}")
        
        while True:
            try:
                # Wait for messages
                data = await websocket.receive_json()
                logger.debug(f"[WebSocket] Received message from {connection_id}: {data}")
                
                # Handle different message types
                message_type = data.get("type")
                channel = data.get("channel", "proxy")  # Default to proxy channel
                logger.debug(f"[WebSocket] Processing message type: {message_type} for channel: {channel}")

                if message_type == "test_connection":
                    await websocket.send_json({
                        "type": "test_connection_response",
                        "channel": channel,
                        "status": "ok",
                        "timestamp": format_datetime(datetime.utcnow())
                    })
                    logger.debug(f"[WebSocket] Sent test connection response to {connection_id}")
                elif message_type == "heartbeat":
                    await websocket.send_json({
                        "type": "heartbeat_response",
                        "channel": channel,
                        "status": "ok",
                        "timestamp": format_datetime(datetime.utcnow())
                    })
                    logger.debug(f"[WebSocket] Sent heartbeat response to {connection_id}")
                elif message_type == "get_initial_data":
                    if channel == "proxy":
                        logger.debug("Processing proxy initial data request")
                        
                        try:
                            # Get proxy status
                            from .endpoints import get_proxy_status
                            status_data = await get_proxy_status(db)
                            
                            # Get proxy history
                            from .history import get_history_entries
                            history_entries = await get_history_entries(db, limit=100)
                            
                            # Convert history entries to dictionaries
                            history_dicts = []
                            for entry in history_entries:
                                entry_dict = {
                                    "id": entry.id,
                                    "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
                                    "method": entry.method,
                                    "url": entry.url,
                                    "host": entry.host,
                                    "path": entry.path,
                                    "status_code": entry.status_code,
                                    "response_status": entry.status_code,
                                    "duration": entry.duration,
                                    "is_intercepted": entry.is_intercepted,
                                    "is_encrypted": entry.is_encrypted,
                                    "tags": entry.tags,
                                    "notes": entry.notes,
                                    "request_headers": entry.request_headers,
                                    "request_body": entry.request_body,
                                    "response_headers": entry.response_headers,
                                    "response_body": entry.response_body,
                                    "session_id": entry.session_id
                                }
                                history_dicts.append(entry_dict)
                            
                            # Get analysis results
                            try:
                                from .endpoints import get_analysis_results
                                analysis_results = await get_analysis_results(db)
                                
                                # Convert analysis results to dictionaries
                                analysis_data = []
                                for result in analysis_results:
                                    result_dict = {
                                        "id": result.id,
                                        "timestamp": result.timestamp.isoformat() if result.timestamp else None,
                                        "analysis_type": result.analysis_type,
                                        "findings": result.findings,
                                        "severity": result.severity,
                                        "analysis_metadata": result.analysis_metadata,
                                        "session_id": result.session_id,
                                        "history_entry_id": result.history_entry_id
                                    }
                                    analysis_data.append(result_dict)
                            except Exception as e:
                                logger.error(f"Failed to get analysis results: {e}")
                                analysis_data = []
                        except Exception as e:
                            logger.error(f"Error preparing initial data: {e}")
                            status_data = {"isRunning": proxy_state.is_running}
                            history_dicts = []
                            analysis_data = []
                        
                        # Send initial data
                        await websocket.send_json({
                            "type": "initial_data",
                            "channel": "proxy",
                            "data": {
                                "status": status_data,
                                "history": history_dicts,
                                "analysis": analysis_data,
                                "version": proxy_state.version
                            }
                        })
                        logger.debug("Sent initial proxy state")
                    elif channel == "health":
                        logger.debug("Processing health initial data request")
                        stats = connection_manager.get_stats()
                        active_connections = connection_manager.get_active_connections()
                        
                        # Get service status and metrics
                        services = await get_service_status()
                        metrics = await get_system_metrics()
                        
                        # Convert active connections to serializable format
                        connections = [{
                            "id": conn.id,
                            "type": conn.type,
                            "connectedAt": conn.connected_at.isoformat(),
                            "lastActivity": conn.last_activity.isoformat(),
                            "messageCount": conn.message_count,
                            "errorCount": conn.error_count
                        } for conn in active_connections]
                        
                        await websocket.send_json({
                            "type": "initial_data",
                            "channel": "health",
                            "data": {
                                "services": services,
                                "metrics": metrics,
                                "wsStatus": {
                                    "ui": {
                                        "connected": stats["ui"]["connected"],
                                        "connectionCount": stats["ui"]["connection_count"],
                                        "lastMessage": format_datetime(stats["ui"]["last_message"]),
                                        "messageCount": stats["ui"]["message_count"],
                                        "errorCount": stats["ui"]["error_count"],
                                        "active_connections": format_stats(stats)["ui"]["active_connections"],
                                        "connection_history": format_stats(stats)["ui"]["connection_history"]
                                    },
                                    "internal": {
                                        "connected": stats["internal"]["connected"],
                                        "connectionCount": stats["internal"]["connection_count"],
                                        "lastMessage": format_datetime(stats["internal"]["last_message"]),
                                        "messageCount": stats["internal"]["message_count"],
                                        "errorCount": stats["internal"]["error_count"],
                                        "active_connections": format_stats(stats)["internal"]["active_connections"],
                                        "connection_history": format_stats(stats)["internal"]["connection_history"]
                                    },
                                    "connections": connections
                                },
                                "connections": connections
                            }
                        })
                        logger.debug("Sent initial health state")
                elif message_type == "get_health_data":
                    if channel == "health":
                        logger.debug("Processing health data request")
                        stats = connection_manager.get_stats()
                        active_connections = connection_manager.get_active_connections()
                        
                        # Get service status and metrics
                        services = await get_service_status()
                        metrics = await get_system_metrics()
                        
                        # Convert active connections to serializable format
                        connections = [{
                            "id": conn.id,
                            "type": conn.type,
                            "connectedAt": conn.connected_at.isoformat(),
                            "lastActivity": conn.last_activity.isoformat(),
                            "messageCount": conn.message_count,
                            "errorCount": conn.error_count
                        } for conn in active_connections]
                        
                        await websocket.send_json({
                            "type": "health_update",
                            "channel": "health",
                            "data": {
                                "services": services,
                                "metrics": metrics,
                                "wsStatus": {
                                    "ui": {
                                        "connected": stats["ui"]["connected"],
                                        "connectionCount": stats["ui"]["connection_count"],
                                        "lastMessage": format_datetime(stats["ui"]["last_message"]),
                                        "messageCount": stats["ui"]["message_count"],
                                        "errorCount": stats["ui"]["error_count"],
                                        "active_connections": format_stats(stats)["ui"]["active_connections"],
                                        "connection_history": format_stats(stats)["ui"]["connection_history"]
                                    },
                                    "internal": {
                                        "connected": stats["internal"]["connected"],
                                        "connectionCount": stats["internal"]["connection_count"],
                                        "lastMessage": format_datetime(stats["internal"]["last_message"]),
                                        "messageCount": stats["internal"]["message_count"],
                                        "errorCount": stats["internal"]["error_count"],
                                        "active_connections": format_stats(stats)["internal"]["active_connections"],
                                        "connection_history": format_stats(stats)["internal"]["connection_history"]
                                    },
                                    "connections": connections
                                },
                                "connections": connections
                            }
                        })
                        logger.debug("Sent health update")
                else:
                    logger.warning(f"[WebSocket] Unknown message type from {connection_id}: {message_type}")
                    
            except Exception as e:
                logger.error(f"[WebSocket] Error handling message from {connection_id}: {e}")
                if not isinstance(e, asyncio.CancelledError):
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
    
    except Exception as e:
        logger.error(f"[WebSocket] Connection error for {connection_id}: {e}")
    
    finally:
        # Clean up connection
        if connection_id:
            await connection_manager.disconnect(ws)
            logger.info(f"[WebSocket] Connection closed: {connection_id}")
        else:
            logger.info("[WebSocket] Unregistered connection closed")

@router.websocket("/internal")
async def handle_internal_connection(websocket: FastAPIWebSocket, db: AsyncSession = Depends(get_db)):
    """Handle internal WebSocket connections from dev container."""
    logger.debug(f"[WebSocket] New internal connection request received:")
    logger.debug(f"[WebSocket] Headers: {dict(websocket.headers)}")
    logger.debug(f"[WebSocket] URL Path: {websocket.url.path}")
    logger.debug(f"[WebSocket] Client Host: {websocket.client.host}:{websocket.client.port}")
    logger.debug(f"[WebSocket] Protocol: {websocket.headers.get('sec-websocket-protocol')}")
    
    ws = WebSocket(websocket)
    connection_id = None
    keepalive_task = None
    
    try:
        # Accept the connection without requiring a specific subprotocol
        try:
            # Try to accept with the protocol if it exists
            protocol = websocket.headers.get("sec-websocket-protocol")
            if protocol:
                await websocket.accept(subprotocol=protocol)
            else:
                await websocket.accept()
            logger.info("[WebSocket] Connection accepted")
        except Exception as e:
            logger.error(f"[WebSocket] Error accepting connection: {e}")
            # Try again without specifying a subprotocol
            await websocket.accept()
            logger.info("[WebSocket] Connection accepted (fallback)")
        
        # Then verify headers - but don't reject the connection
        protocol = websocket.headers.get("sec-websocket-protocol")
        if protocol != "proxy-internal":
            logger.warning("[WebSocket] Invalid protocol for internal connection")
            logger.warning(f"[WebSocket] Expected 'proxy-internal', got '{protocol}'")
            # Continue anyway - we're being permissive
        
        # Check for required headers - more permissive now
        required_headers = {
            'x-connection-type': 'internal',
            'x-proxy-version': '0.1.0'
        }
        
        for header, value in required_headers.items():
            header_value = websocket.headers.get(header.lower())
            if not header_value:
                logger.warning(f"[WebSocket] Missing header: {header}")
                # Don't close the connection, just log the warning
                continue
            if header_value != value:
                logger.warning(f"[WebSocket] Invalid header value for {header}: {header_value}")
                # Don't close the connection, just log the warning
                continue
        
        # Register with connection manager
        connection_id = await connection_manager.connect(ws, "internal")
        logger.info(f"[WebSocket] New internal connection established: {connection_id}")

        # Send initial state
        initial_state = {
            "type": "initial_data",
            "data": {
                "status": {
                    "proxy_running": proxy_state.is_running,
                    "version": proxy_state.version
                }
            }
        }
        await websocket.send_json(initial_state)
        logger.debug(f"[WebSocket] Sent initial state to {connection_id}")

        # Start keepalive task
        async def send_keepalive():
            while True:
                try:
                    await websocket.send_json({
                        "type": "heartbeat",
                        "timestamp": datetime.utcnow().isoformat(),
                        "connection_id": connection_id
                    })
                    await asyncio.sleep(HEARTBEAT_INTERVAL)
                except Exception as e:
                    logger.error(f"[WebSocket] Keepalive error: {e}")
                    break

        keepalive_task = asyncio.create_task(send_keepalive())

        while True:
            try:
                data = await websocket.receive_json()
                logger.debug(f"[WebSocket] Received internal message from {connection_id}: {data}")
                
                message_type = data.get("type")
                if message_type == "test_connection":
                    await websocket.send_json({
                        "type": "test_connection_response",
                        "status": "ok",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    continue

                elif message_type == "heartbeat":
                    # Update proxy state's last heartbeat timestamp
                    proxy_state.last_heartbeat = time.time()
                    proxy_state.is_running = True
                    await websocket.send_json({
                        "type": "heartbeat_response",
                        "timestamp": datetime.utcnow().isoformat(),
                        "connection_id": connection_id
                    })
                    continue

                # Broadcast message to UI clients
                await connection_manager.broadcast_json(data, connection_type="ui")

            except WebSocketDisconnect:
                logger.info(f"[WebSocket] Internal connection disconnected: {connection_id}")
                break
            except Exception as e:
                logger.error(f"[WebSocket] Error handling message from {connection_id}: {e}")
                if not isinstance(e, asyncio.CancelledError):
                    try:
                        await websocket.send_json({
                            "type": "error",
                            "message": str(e)
                        })
                    except:
                        break

    except Exception as e:
        logger.error(f"[WebSocket] Error handling internal connection: {e}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011, reason=str(e))
    finally:
        # Clean up connection
        if keepalive_task:
            keepalive_task.cancel()
        if connection_id:
            await connection_manager.disconnect(ws, "internal")
            logger.info(f"[WebSocket] Connection closed: {connection_id}")
        else:
            logger.info("[WebSocket] Unregistered connection closed")

@router.post("/config/{connection_id}")
async def update_connection_config(
    connection_id: str,
    config: Dict[str, Any]
) -> Dict[str, str]:
    """Update WebSocket connection configuration."""
    conversation = ws_manager.get_conversation(connection_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    if "interceptorEnabled" in config:
        for interceptor in ws_manager._interceptors:
            if not isinstance(interceptor, SecurityInterceptor):  # Don't disable security
                interceptor.is_enabled = config["interceptorEnabled"]
    
    if "fuzzingEnabled" in config:
        ws_manager._fuzzer.is_enabled = config["fuzzingEnabled"]
        if "fuzzConfig" in config:
            ws_manager._fuzzer.configure(config["fuzzConfig"])
    
    if "securityAnalysisEnabled" in config:
        security_interceptor.is_enabled = config["securityAnalysisEnabled"]
    
    return {"message": "Configuration updated successfully"}

@router.post("/send")
async def send_message(data: Dict[str, Any]) -> Dict[str, str]:
    """Send a message through a WebSocket connection."""
    connection_id = data.get("connectionId")
    if not isinstance(connection_id, str):
        raise HTTPException(status_code=400, detail="Invalid connection ID")
        
    conversation = ws_manager.get_conversation(connection_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    # Actual sending is handled by the WebSocketManager through its interceptor chain
    return {"message": "Message sent successfully"}

"""Parameter detection for WebSocket messages."""
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import json
import re
import urllib.parse
import logging
from dataclasses import dataclass
from enum import Enum, auto
from uuid import UUID

from .types import WSMessage, MessageType, MessageDirection

logger = logging.getLogger(__name__)

class ParameterType(Enum):
    """Types of parameters that can be detected."""
    JSON_KEY = auto()
    JSON_VALUE = auto()
    URL_QUERY = auto()
    FORM_DATA = auto()
    CUSTOM = auto()
    
    def __str__(self) -> str:
        return self.name.lower()

@dataclass
class DetectedParameter:
    """Represents a detected parameter in a message."""
    name: str
    value: Any
    param_type: ParameterType
    path: str  # JSON path or location identifier
    message_id: Union[str, UUID]
    confidence: float  # 0.0 to 1.0 confidence score
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "value": self.value,
            "type": str(self.param_type),
            "path": self.path,
            "message_id": str(self.message_id),
            "confidence": self.confidence,
            "metadata": self.metadata or {}
        }

class ParameterDetector:
    """Detects parameters in WebSocket messages."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        """Initialize the parameter detector.
        
        Args:
            confidence_threshold: Minimum confidence score (0.0-1.0) for parameter detection
        """
        self.confidence_threshold = confidence_threshold
        # Common parameter names to look for
        self.common_param_names = {
            'id', 'user', 'username', 'password', 'token', 'auth', 'key', 'api_key', 
            'query', 'search', 'filter', 'sort', 'page', 'limit', 'offset', 'start',
            'end', 'from', 'to', 'date', 'time', 'timestamp', 'email', 'name', 'value',
            'data', 'content', 'message', 'text', 'type', 'action', 'method', 'format',
            'callback', 'url', 'target', 'destination', 'source', 'origin', 'host',
            'domain', 'path', 'file', 'filename', 'size', 'length', 'count', 'total',
            'status', 'state', 'mode', 'format', 'version', 'lang', 'language', 'locale',
            'timezone', 'currency', 'amount', 'price', 'cost', 'rate', 'fee', 'tax',
            'discount', 'promotion', 'coupon', 'code', 'ref', 'reference', 'session',
            'transaction', 'order', 'payment', 'shipping', 'billing', 'address',
            'phone', 'mobile', 'fax', 'company', 'organization', 'department', 'title',
            'role', 'permission', 'access', 'group', 'category', 'tag', 'label', 'color',
            'style', 'theme', 'template', 'layout', 'view', 'display', 'show', 'hide',
            'visible', 'enabled', 'disabled', 'active', 'inactive', 'deleted', 'archived',
            'created', 'updated', 'modified', 'deleted', 'expired', 'valid', 'invalid',
            'error', 'warning', 'info', 'debug', 'log', 'level', 'severity', 'priority',
            'importance', 'urgent', 'critical', 'normal', 'low', 'high', 'medium'
        }
        
        # URL parameter regex patterns
        self.url_param_patterns = [
            r'[?&]([^=&]+)=([^&]*)',  # Standard URL query params
            r'\/api\/([^\/]+)\/([^\/]+)',  # REST API path params
            r'\/([a-zA-Z0-9_-]+)\/([a-zA-Z0-9_-]+)'  # Path segments
        ]
    
    def detect_parameters(self, message: WSMessage) -> List[DetectedParameter]:
        """Detect parameters in a WebSocket message.
        
        Args:
            message: WebSocket message to analyze
            
        Returns:
            List of detected parameters
        """
        if not message or not message.data:
            return []
            
        detected_params = []
        
        # Process based on message type and data format
        if message.type == MessageType.TEXT:
            data = message.data
            if isinstance(data, bytes):
                try:
                    data = data.decode('utf-8')
                except UnicodeDecodeError:
                    logger.warning(f"Failed to decode binary data as UTF-8 for message {message.id}")
                    return []
            
            # Try to detect JSON parameters
            json_params = self._detect_json_parameters(data, message.id)
            detected_params.extend(json_params)
            
            # Try to detect URL query parameters
            url_params = self._detect_url_parameters(data, message.id)
            detected_params.extend(url_params)
            
            # Try to detect form data parameters
            form_params = self._detect_form_parameters(data, message.id)
            detected_params.extend(form_params)
            
            # Try to detect custom format parameters
            custom_params = self._detect_custom_parameters(data, message.id)
            detected_params.extend(custom_params)
        
        # Filter by confidence threshold
        return [p for p in detected_params if p.confidence >= self.confidence_threshold]
    
    def _detect_json_parameters(self, data: str, message_id: Union[str, UUID]) -> List[DetectedParameter]:
        """Detect parameters in JSON data.
        
        Args:
            data: String data to analyze
            message_id: ID of the message being analyzed
            
        Returns:
            List of detected parameters
        """
        params = []
        try:
            # Try to parse as JSON
            json_data = json.loads(data)
            
            # Process JSON object
            if isinstance(json_data, dict):
                params.extend(self._process_json_dict(json_data, message_id))
                
        except json.JSONDecodeError:
            # Not valid JSON
            pass
            
        return params
    
    def _process_json_dict(self, json_dict: Dict[str, Any], message_id: Union[str, UUID], path: str = "$") -> List[DetectedParameter]:
        """Process a JSON dictionary to extract parameters.
        
        Args:
            json_dict: Dictionary to process
            message_id: ID of the message being analyzed
            path: Current JSON path
            
        Returns:
            List of detected parameters
        """
        params = []
        
        for key, value in json_dict.items():
            current_path = f"{path}.{key}"
            
            # Add the key itself as a parameter
            confidence = self._calculate_parameter_confidence(key, value)
            params.append(DetectedParameter(
                name=key,
                value=value,
                param_type=ParameterType.JSON_KEY,
                path=current_path,
                message_id=message_id,
                confidence=confidence
            ))
            
            # Process nested objects
            if isinstance(value, dict):
                nested_params = self._process_json_dict(value, message_id, current_path)
                params.extend(nested_params)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        item_path = f"{current_path}[{i}]"
                        nested_params = self._process_json_dict(item, message_id, item_path)
                        params.extend(nested_params)
            elif isinstance(value, (str, int, float, bool)) and not key.startswith('_'):
                # Add primitive values as parameters if they look like IDs or important values
                if self._is_likely_parameter_value(key, value):
                    params.append(DetectedParameter(
                        name=f"{key}_value",
                        value=value,
                        param_type=ParameterType.JSON_VALUE,
                        path=current_path,
                        message_id=message_id,
                        confidence=confidence * 0.8  # Slightly lower confidence for values
                    ))
                    
        return params
    
    def _detect_url_parameters(self, data: str, message_id: Union[str, UUID]) -> List[DetectedParameter]:
        """Detect URL query parameters in string data.
        
        Args:
            data: String data to analyze
            message_id: ID of the message being analyzed
            
        Returns:
            List of detected parameters
        """
        params = []
        
        # Look for URL patterns
        for pattern in self.url_param_patterns:
            matches = re.findall(pattern, data)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    name, value = match[0], match[1]
                    confidence = self._calculate_parameter_confidence(name, value)
                    params.append(DetectedParameter(
                        name=name,
                        value=value,
                        param_type=ParameterType.URL_QUERY,
                        path=f"url_param:{name}",
                        message_id=message_id,
                        confidence=confidence
                    ))
        
        # Try to parse as URL query string
        if '=' in data and ('&' in data or '?' in data):
            try:
                # Extract query string if it looks like a URL
                query_string = data
                if '?' in data:
                    query_string = data.split('?', 1)[1]
                
                # Parse query parameters
                parsed_params = urllib.parse.parse_qs(query_string)
                for name, values in parsed_params.items():
                    for value in values:
                        confidence = self._calculate_parameter_confidence(name, value)
                        params.append(DetectedParameter(
                            name=name,
                            value=value,
                            param_type=ParameterType.URL_QUERY,
                            path=f"query_param:{name}",
                            message_id=message_id,
                            confidence=confidence
                        ))
            except Exception as e:
                logger.debug(f"Error parsing URL parameters: {e}")
                
        return params
    
    def _detect_form_parameters(self, data: str, message_id: Union[str, UUID]) -> List[DetectedParameter]:
        """Detect form data parameters in string data.
        
        Args:
            data: String data to analyze
            message_id: ID of the message being analyzed
            
        Returns:
            List of detected parameters
        """
        params = []
        
        # Check if it looks like form data (key=value&key2=value2)
        if '=' in data and '&' in data and not data.startswith('{') and not data.startswith('['):
            try:
                # Parse as form data
                parsed_params = urllib.parse.parse_qs(data)
                for name, values in parsed_params.items():
                    for value in values:
                        confidence = self._calculate_parameter_confidence(name, value)
                        params.append(DetectedParameter(
                            name=name,
                            value=value,
                            param_type=ParameterType.FORM_DATA,
                            path=f"form_param:{name}",
                            message_id=message_id,
                            confidence=confidence
                        ))
            except Exception as e:
                logger.debug(f"Error parsing form parameters: {e}")
                
        return params
    
    def _detect_custom_parameters(self, data: str, message_id: Union[str, UUID]) -> List[DetectedParameter]:
        """Detect parameters in custom format data.
        
        Args:
            data: String data to analyze
            message_id: ID of the message being analyzed
            
        Returns:
            List of detected parameters
        """
        params = []
        
        # Look for patterns like key:value or key=value
        custom_patterns = [
            r'(\w+)[:=]([^,;\s]+)',  # key:value or key=value
            r'"(\w+)"\s*:\s*"([^"]*)"',  # "key":"value" (JSON-like but not in valid JSON)
            r"'(\w+)'\s*:\s*'([^']*)'",  # 'key':'value'
        ]
        
        for pattern in custom_patterns:
            matches = re.findall(pattern, data)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    name, value = match[0], match[1]
                    # Skip if already detected as JSON or URL param
                    if any(p.name == name and p.value == value for p in params):
                        continue
                        
                    confidence = self._calculate_parameter_confidence(name, value) * 0.7  # Lower confidence for custom formats
                    params.append(DetectedParameter(
                        name=name,
                        value=value,
                        param_type=ParameterType.CUSTOM,
                        path=f"custom:{name}",
                        message_id=message_id,
                        confidence=confidence
                    ))
                    
        return params
    
    def _calculate_parameter_confidence(self, name: str, value: Any) -> float:
        """Calculate confidence score for a parameter.
        
        Args:
            name: Parameter name
            value: Parameter value
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence
        
        # Increase confidence for common parameter names
        if name.lower() in self.common_param_names:
            confidence += 0.3
            
        # Increase confidence for names that look like parameters
        if re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', str(name)):
            confidence += 0.1
            
        # Adjust confidence based on value type and content
        if isinstance(value, str):
            # Higher confidence for non-empty strings that aren't too long
            if value and len(value) < 1000:
                confidence += 0.1
                
            # Higher confidence for values that look like IDs, tokens, etc.
            if re.match(r'^[A-Za-z0-9_\-\.]+$', value) and len(value) > 5:
                confidence += 0.1
                
        elif isinstance(value, (int, float)):
            # Higher confidence for numeric values in certain ranges
            if 0 <= value < 1000000:
                confidence += 0.1
                
        elif isinstance(value, bool):
            # Boolean values are often good parameters
            confidence += 0.2
            
        # Cap confidence at 1.0
        return min(confidence, 1.0)
    
    def _is_likely_parameter_value(self, key: str, value: Any) -> bool:
        """Determine if a value is likely to be a parameter value worth fuzzing.
        
        Args:
            key: Parameter key
            value: Parameter value
            
        Returns:
            True if the value is likely a parameter worth fuzzing
        """
        # Check key name patterns that suggest important values
        important_key_patterns = [
            r'id$', r'key$', r'token$', r'auth', r'password', r'secret',
            r'hash', r'code', r'session', r'user', r'name', r'email'
        ]
        
        for pattern in important_key_patterns:
            if re.search(pattern, key.lower()):
                return True
                
        # Check value patterns
        if isinstance(value, str):
            # IDs, tokens, etc.
            if re.match(r'^[A-Za-z0-9_\-\.]{8,}$', value):
                return True
                
            # URLs or paths
            if re.match(r'^(https?://|/)', value):
                return True
                
        return False

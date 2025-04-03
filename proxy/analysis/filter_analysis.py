"""
Post-analysis filter addition functionality for the advanced filtering system.

This module provides functionality for analyzing traffic history and creating filter rules
based on traffic patterns.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from proxy.filter import FilterCondition, FilterRule

logger = logging.getLogger(__name__)


class FilterAnalyzer:
    """Class for analyzing traffic history and suggesting filter rules."""
    
    def __init__(self):
        """Initialize the filter analyzer."""
        pass
        
    def analyze_traffic(self, traffic_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze traffic history to identify patterns.
        
        Args:
            traffic_entries: List of traffic history entries
            
        Returns:
            List of analysis results with pattern information
        """
        results = []
        
        # Group by type
        requests = [t for t in traffic_entries if t["type"] == "request"]
        responses = [t for t in traffic_entries if t["type"] == "response"]
        
        # Analyze request patterns
        if requests:
            request_patterns = self._analyze_request_patterns(requests)
            results.append({
                "type": "request_patterns",
                "patterns": request_patterns
            })
            
        # Analyze response patterns
        if responses:
            response_patterns = self._analyze_response_patterns(responses)
            results.append({
                "type": "response_patterns",
                "patterns": response_patterns
            })
            
        # Analyze request-response pairs
        if requests and responses:
            pair_patterns = self._analyze_request_response_pairs(requests, responses)
            results.append({
                "type": "request_response_pairs",
                "patterns": pair_patterns
            })
            
        return results
        
    def _analyze_request_patterns(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze request patterns.
        
        Args:
            requests: List of request entries
            
        Returns:
            List of request pattern information
        """
        patterns = []
        
        # Group by method
        method_groups = {}
        for request in requests:
            method = request["method"]
            if method not in method_groups:
                method_groups[method] = []
            method_groups[method].append(request)
            
        # Analyze each method group
        for method, group in method_groups.items():
            # Group by path pattern
            path_patterns = self._extract_path_patterns(group)
            
            for path_pattern, path_group in path_patterns.items():
                # Find common headers
                common_headers = self._find_common_headers(path_group, "headers")
                
                # Find common query parameters
                common_params = self._find_common_query_params(path_group)
                
                # Find common body patterns
                body_patterns = self._extract_body_patterns(path_group)
                
                patterns.append({
                    "method": method,
                    "path_pattern": path_pattern,
                    "count": len(path_group),
                    "common_headers": common_headers,
                    "common_params": common_params,
                    "body_patterns": body_patterns
                })
                
        return patterns
        
    def _analyze_response_patterns(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze response patterns.
        
        Args:
            responses: List of response entries
            
        Returns:
            List of response pattern information
        """
        patterns = []
        
        # Group by status code
        status_groups = {}
        for response in responses:
            status_code = response["status_code"]
            if status_code not in status_groups:
                status_groups[status_code] = []
            status_groups[status_code].append(response)
            
        # Analyze each status group
        for status_code, group in status_groups.items():
            # Group by request path pattern
            path_patterns = self._extract_path_patterns(group, key="request_path")
            
            for path_pattern, path_group in path_patterns.items():
                # Find common headers
                common_headers = self._find_common_headers(path_group, "headers")
                
                # Find common content types
                content_types = set()
                for response in path_group:
                    if "Content-Type" in response["headers"]:
                        content_types.add(response["headers"]["Content-Type"])
                        
                patterns.append({
                    "status_code": status_code,
                    "request_path_pattern": path_pattern,
                    "count": len(path_group),
                    "common_headers": common_headers,
                    "content_types": list(content_types)
                })
                
        return patterns
        
    def _analyze_request_response_pairs(self, requests: List[Dict[str, Any]], responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze request-response pairs.
        
        Args:
            requests: List of request entries
            responses: List of response entries
            
        Returns:
            List of request-response pair pattern information
        """
        patterns = []
        
        # Group responses by request path
        path_groups = {}
        for response in responses:
            path = response["request_path"]
            if path not in path_groups:
                path_groups[path] = []
            path_groups[path].append(response)
            
        # Find matching requests for each path
        for path, responses in path_groups.items():
            matching_requests = [r for r in requests if r["path"] == path]
            
            if matching_requests and responses:
                # Group by status code
                status_groups = {}
                for response in responses:
                    status_code = response["status_code"]
                    if status_code not in status_groups:
                        status_groups[status_code] = []
                    status_groups[status_code].append(response)
                    
                for status_code, status_group in status_groups.items():
                    patterns.append({
                        "path": path,
                        "request_count": len(matching_requests),
                        "response_count": len(status_group),
                        "status_code": status_code,
                        "request_methods": list(set(r["method"] for r in matching_requests)),
                        "content_types": list(set(r["headers"].get("Content-Type", "") for r in status_group if "Content-Type" in r["headers"]))
                    })
                    
        return patterns
        
    def _extract_path_patterns(self, entries: List[Dict[str, Any]], key: str = "path") -> Dict[str, List[Dict[str, Any]]]:
        """Extract path patterns from entries.
        
        Args:
            entries: List of entries
            key: The key for the path in the entries
            
        Returns:
            Dictionary mapping path patterns to lists of entries
        """
        # First group by exact path
        exact_paths = {}
        for entry in entries:
            path = entry[key]
            if path not in exact_paths:
                exact_paths[path] = []
            exact_paths[path].append(entry)
            
        # If there are many unique paths, try to find patterns
        if len(exact_paths) > 10:
            # Group by path prefix
            prefix_patterns = {}
            for path, group in exact_paths.items():
                # Split path into segments
                segments = path.split('/')
                
                # Try different prefix lengths
                for i in range(1, min(4, len(segments))):
                    prefix = '/'.join(segments[:i]) + '/'
                    if prefix not in prefix_patterns:
                        prefix_patterns[prefix] = []
                    prefix_patterns[prefix].extend(group)
                    
            # Only use prefix patterns if they reduce the number of groups
            if len(prefix_patterns) < len(exact_paths):
                return prefix_patterns
                
        return exact_paths
        
    def _find_common_headers(self, entries: List[Dict[str, Any]], headers_key: str) -> Dict[str, str]:
        """Find headers that are common across all entries.
        
        Args:
            entries: List of entries
            headers_key: The key for headers in the entries
            
        Returns:
            Dictionary of common headers and their values
        """
        if not entries:
            return {}
            
        # Get all headers from first entry
        common_headers = entries[0][headers_key].copy()
        
        # Intersect with headers from other entries
        for entry in entries[1:]:
            headers = entry[headers_key]
            # Remove headers that don't exist in this entry
            for header in list(common_headers.keys()):
                if header not in headers or headers[header] != common_headers[header]:
                    common_headers.pop(header)
                    
        return common_headers
        
    def _find_common_query_params(self, requests: List[Dict[str, Any]]) -> Dict[str, str]:
        """Find query parameters that are common across all requests.
        
        Args:
            requests: List of request entries
            
        Returns:
            Dictionary of common query parameters and their values
        """
        if not requests:
            return {}
            
        # Get all query parameters from first request
        common_params = requests[0]["query_params"].copy()
        
        # Intersect with query parameters from other requests
        for request in requests[1:]:
            params = request["query_params"]
            # Remove parameters that don't exist in this request
            for param in list(common_params.keys()):
                if param not in params or params[param] != common_params[param]:
                    common_params.pop(param)
                    
        return common_params
        
    def _extract_body_patterns(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract patterns from request bodies.
        
        Args:
            requests: List of request entries
            
        Returns:
            List of body pattern information
        """
        patterns = []
        
        # Group by content type
        content_type_groups = {}
        for request in requests:
            content_type = request["headers"].get("Content-Type", "")
            if content_type not in content_type_groups:
                content_type_groups[content_type] = []
            content_type_groups[content_type].append(request)
            
        # Analyze each content type group
        for content_type, group in content_type_groups.items():
            if "application/json" in content_type:
                # Analyze JSON bodies
                json_patterns = self._analyze_json_bodies(group)
                if json_patterns:
                    patterns.append({
                        "content_type": content_type,
                        "format": "json",
                        "patterns": json_patterns
                    })
            elif "application/x-www-form-urlencoded" in content_type:
                # Analyze form bodies
                form_patterns = self._analyze_form_bodies(group)
                if form_patterns:
                    patterns.append({
                        "content_type": content_type,
                        "format": "form",
                        "patterns": form_patterns
                    })
            elif "text/plain" in content_type:
                # Analyze text bodies
                text_patterns = self._analyze_text_bodies(group)
                if text_patterns:
                    patterns.append({
                        "content_type": content_type,
                        "format": "text",
                        "patterns": text_patterns
                    })
                    
        return patterns
        
    def _analyze_json_bodies(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze JSON request bodies.
        
        Args:
            requests: List of request entries
            
        Returns:
            List of JSON body pattern information
        """
        import json
        
        patterns = []
        
        # Parse JSON bodies
        parsed_bodies = []
        for request in requests:
            if request["body"]:
                try:
                    body = json.loads(request["body"])
                    parsed_bodies.append(body)
                except json.JSONDecodeError:
                    pass
                    
        if not parsed_bodies:
            return patterns
            
        # Find common fields
        common_fields = set(parsed_bodies[0].keys())
        for body in parsed_bodies[1:]:
            common_fields &= set(body.keys())
            
        # Analyze common field values
        field_values = {}
        for field in common_fields:
            values = set()
            for body in parsed_bodies:
                value = body[field]
                if isinstance(value, (str, int, float, bool)) or value is None:
                    values.add(str(value))
                    
            if len(values) == 1:
                # All bodies have the same value for this field
                field_values[field] = next(iter(values))
                
        patterns.append({
            "common_fields": list(common_fields),
            "common_values": field_values
        })
        
        return patterns
        
    def _analyze_form_bodies(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze form request bodies.
        
        Args:
            requests: List of request entries
            
        Returns:
            List of form body pattern information
        """
        from urllib.parse import parse_qs
        
        patterns = []
        
        # Parse form bodies
        parsed_bodies = []
        for request in requests:
            if request["body"]:
                try:
                    body = parse_qs(request["body"])
                    parsed_bodies.append(body)
                except Exception:
                    pass
                    
        if not parsed_bodies:
            return patterns
            
        # Find common fields
        common_fields = set(parsed_bodies[0].keys())
        for body in parsed_bodies[1:]:
            common_fields &= set(body.keys())
            
        # Analyze common field values
        field_values = {}
        for field in common_fields:
            values = set()
            for body in parsed_bodies:
                value = body[field][0] if body[field] else ""
                values.add(value)
                
            if len(values) == 1:
                # All bodies have the same value for this field
                field_values[field] = next(iter(values))
                
        patterns.append({
            "common_fields": list(common_fields),
            "common_values": field_values
        })
        
        return patterns
        
    def _analyze_text_bodies(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze text request bodies.
        
        Args:
            requests: List of request entries
            
        Returns:
            List of text body pattern information
        """
        patterns = []
        
        # Extract text bodies
        bodies = [request["body"] for request in requests if request["body"]]
        
        if not bodies:
            return patterns
            
        # Check if all bodies are identical
        if len(set(bodies)) == 1:
            patterns.append({
                "identical": True,
                "value": bodies[0]
            })
        else:
            # Find common prefix
            common_prefix = os.path.commonprefix(bodies)
            
            # Find common suffix
            reversed_bodies = [body[::-1] for body in bodies]
            common_suffix = os.path.commonprefix(reversed_bodies)[::-1]
            
            if common_prefix or common_suffix:
                patterns.append({
                    "identical": False,
                    "common_prefix": common_prefix if len(common_prefix) > 5 else "",
                    "common_suffix": common_suffix if len(common_suffix) > 5 else ""
                })
                
        return patterns
        
    def suggest_filter_rules(self, traffic_entries: List[Dict[str, Any]], max_rules: int = 5) -> List[FilterRule]:
        """Suggest filter rules based on traffic patterns.
        
        Args:
            traffic_entries: List of traffic history entries
            max_rules: Maximum number of rules to suggest
            
        Returns:
            List of suggested FilterRule objects
        """
        suggested_rules = []
        
        # Analyze traffic
        analysis_results = self.analyze_traffic(traffic_entries)
        
        # Extract request patterns
        request_patterns = []
        for result in analysis_results:
            if result["type"] == "request_patterns":
                request_patterns = result["patterns"]
                break
                
        # Extract response patterns
        response_patterns = []
        for result in analysis_results:
            if result["type"] == "response_patterns":
                response_patterns = result["patterns"]
                break
                
        # Create rules from request patterns
        for pattern in request_patterns:
            # Skip patterns with low count
            if pattern["count"] < 2:
                continue
                
            conditions = []
            
            # Add method condition
            conditions.append(FilterCondition("method", "equals", pattern["method"]))
            
            # Add path condition
            path_pattern = pattern["path_pattern"]
            if path_pattern.endswith('/'):
                # Path prefix
                conditions.append(FilterCondition("path", "starts_with", path_pattern))
            else:
                # Exact path
                conditions.append(FilterCondition("path", "equals", path_pattern))
                
            # Add header conditions for important headers
            for header, value in pattern["common_headers"].items():
                if header in ["User-Agent", "Content-Type", "Authorization"]:
                    conditions.append(FilterCondition(f"headers.{header}", "equals", value))
                    
            # Create the rule
            rule = FilterRule(
                name=f"Block {pattern['method']} requests to {path_pattern}",
                description=f"Automatically suggested rule based on {pattern['count']} requests",
                conditions=conditions,
                tags=["request", "auto-suggested"]
            )
            
            suggested_rules.append(rule)
            
        # Create rules from response patterns
        for pattern in response_patterns:
            # Skip patterns with low count or non-error status codes
            if pattern["count"] < 2 or pattern["status_code"] < 400:
                continue
                
            conditions = []
            
            # Add status code condition
            conditions.append(FilterCondition("status_code", "equals", pattern["status_code"]))
            
            # Add request path condition
            path_pattern = pattern["request_path_pattern"]
            if path_pattern.endswith('/'):
                # Path prefix
                conditions.append(FilterCondition("request_path", "starts_with", path_pattern))
            else:
                # Exact path
                conditions.append(FilterCondition("request_path", "equals", path_pattern))
                
            # Add content type condition if available
            if pattern["content_types"] and len(pattern["content_types"]) == 1:
                conditions.append(FilterCondition("headers.Content-Type", "equals", pattern["content_types"][0]))
                
            # Create the rule
            rule = FilterRule(
                name=f"Block {pattern['status_code']} responses from {path_pattern}",
                description=f"Automatically suggested rule based on {pattern['count']} responses",
                conditions=conditions,
                tags=["response", "auto-suggested"]
            )
            
            suggested_rules.append(rule)
            
        # Sort by count (descending) and return top N
        suggested_rules.sort(key=lambda r: int(r.description.split()[-2]), reverse=True)
        return suggested_rules[:max_rules]
        
    def create_rule_from_traffic(self, traffic_entry: Dict[str, Any], name: str = "", description: str = "") -> FilterRule:
        """Create a filter rule from a traffic history entry.
        
        Args:
            traffic_entry: The traffic history entry
            name: Optional name for the rule
            description: Optional description for the rule
            
        Returns:
            A new FilterRule based on the traffic
        """
        conditions = []
        
        if traffic_entry["type"] == "request":
            # Create conditions based on request
            conditions.append(FilterCondition("method", "equals", traffic_entry["method"]))
            conditions.append(FilterCondition("path", "equals", traffic_entry["path"]))
            
            # Add some headers as conditions
            for header in ["User-Agent", "Content-Type", "Authorization"]:
                if header in traffic_entry["headers"]:
                    conditions.append(FilterCondition(f"headers.{header}", "equals", traffic_entry["headers"][header]))
                    
        elif traffic_entry["type"] == "response":
            # Create conditions based on response
            conditions.append(FilterCondition("request_method", "equals", traffic_entry["request_method"]))
            conditions.append(FilterCondition("request_path", "equals", traffic_entry["request_path"]))
            conditions.append(FilterCondition("status_code", "equals", traffic_entry["status_code"]))
            
            # Add some headers as conditions
            for header in ["Content-Type", "Server"]:
                if header in traffic_entry["headers"]:
                    conditions.append(FilterCondition(f"headers.{header}", "equals", traffic_entry["headers"][header]))
                    
        # Create the rule
        return FilterRule(
            name=name or f"Rule from {traffic_entry['type']} to {traffic_entry.get('path', traffic_entry.get('request_path', ''))}",
            description=description or f"Automatically generated from {traffic_entry['type']} traffic",
            conditions=conditions,
            tags=[traffic_entry["type"], "auto-generated"]
        )

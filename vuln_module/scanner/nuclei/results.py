"""Nuclei results parsing and processing utilities."""
from typing import Dict, Any, Optional
from datetime import datetime

from ...models import (
    VulnResult,
    VulnSeverity,
    PayloadResult,
    PayloadType,
    Payload
)

class NucleiResultParser:
    """Parser for Nuclei scan results."""

    SEVERITY_MAP = {
        "info": VulnSeverity.INFO,
        "low": VulnSeverity.LOW,
        "medium": VulnSeverity.MEDIUM,
        "high": VulnSeverity.HIGH,
        "critical": VulnSeverity.CRITICAL
    }

    @classmethod
    def _get_payload_type(cls, result: Dict[str, Any]) -> PayloadType:
        """Determine payload type from template ID and info."""
        template_id = result.get("template-id", "").lower()
        tags = result.get("info", {}).get("tags", [])

        # Check tags first
        if any("xss" in tag.lower() for tag in tags):
            return PayloadType.XSS
        elif any("sqli" in tag.lower() for tag in tags):
            return PayloadType.SQLI
        elif any("cmdi" in tag.lower() or "command-injection" in tag.lower() for tag in tags):
            return PayloadType.COMMAND_INJECTION
        elif any("ssrf" in tag.lower() for tag in tags):
            return PayloadType.SSRF
        elif any("lfi" in tag.lower() or "path-traversal" in tag.lower() for tag in tags):
            return PayloadType.PATH_TRAVERSAL

        # Fallback to template ID checks
        if "xss" in template_id:
            return PayloadType.XSS
        elif "sqli" in template_id:
            return PayloadType.SQLI
        elif "command-injection" in template_id:
            return PayloadType.COMMAND_INJECTION
        elif "ssrf" in template_id:
            return PayloadType.SSRF
        elif "path-traversal" in template_id or "lfi" in template_id:
            return PayloadType.PATH_TRAVERSAL

        return PayloadType.CUSTOM

    @classmethod
    def _extract_matched_payload(cls, result: Dict[str, Any]) -> str:
        """Extract the matched payload from result."""
        # Try getting from matched data
        if matched := result.get("matched"):
            return str(matched)
        
        # Try getting from extracted values
        if extracted := result.get("extracted-values"):
            if isinstance(extracted, dict):
                return str(next(iter(extracted.values()), ""))
            return str(extracted)

        # Fallback to matcher name
        return result.get("matcher-name", "")

    @classmethod
    def parse_result(cls, result: Dict[str, Any]) -> Optional[VulnResult]:
        """Convert Nuclei result to VulnResult model."""
        try:
            severity = cls.SEVERITY_MAP.get(
                result.get("severity", "").lower(),
                VulnSeverity.INFO
            )

            metadata = {
                "template": result.get("template", ""),
                "template-url": result.get("template-url", ""),
                "curl-command": result.get("curl-command", ""),
                "matcher-name": result.get("matcher-name", ""),
                "extracted-values": result.get("extracted-values", {}),
                "ip": result.get("ip", ""),
                "host": result.get("host", "")
            }

            payload_result = PayloadResult(
                payload=Payload(
                    content=cls._extract_matched_payload(result),
                    type=cls._get_payload_type(result),
                ),
                success=True,
                response_data={
                    "request": result.get("request", ""),
                    "response": result.get("response", ""),
                    **metadata
                }
            )

            return VulnResult(
                name=result.get("template-id", "Unknown"),
                type=result.get("type", "Unknown"),
                severity=severity,
                description=result.get("info", {}).get("description", ""),
                endpoint=result.get("matched-at", ""),
                payloads=[payload_result],
                found_at=datetime.fromisoformat(result.get("timestamp", datetime.now().isoformat())),
                tags=set(result.get("info", {}).get("tags", [])),
                references=result.get("info", {}).get("reference", []),
                metadata=metadata
            )
        except Exception:
            return None

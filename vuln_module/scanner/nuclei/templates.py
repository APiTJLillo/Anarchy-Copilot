"""Nuclei template management utilities."""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json

from ...models import PayloadType

@dataclass
class NucleiTemplate:
    """Represents a Nuclei vulnerability scanning template."""
    template_id: str
    name: str
    author: str
    severity: str 
    description: str
    requests: List[Dict[str, Any]]
    matchers: List[Dict[str, Any]]
    matchers_condition: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_yaml(self) -> str:
        """Convert template to YAML format."""
        template = {
            "id": self.template_id,
            "info": {
                "name": self.name,
                "author": self.author,
                "severity": self.severity,
                "description": self.description,
                "tags": self.tags or []
            },
            "requests": self.requests
        }
        
        if self.matchers:
            if self.matchers_condition:
                template["matchers-condition"] = self.matchers_condition
            template["matchers"] = self.matchers
            
        return json.dumps(template, indent=2)

class NucleiTemplateGenerator:
    """Generator for common vulnerability test templates."""
    
    @staticmethod
    def create_xss_template(payloads: List[str]) -> NucleiTemplate:
        """Create an XSS testing template."""
        return NucleiTemplate(
            template_id="custom-xss-template",
            name="Custom XSS Test",
            author="Anarchy Copilot",
            severity="medium",
            description="Custom XSS payload testing",
            requests=[{
                "method": "GET",
                "path": ["{{BaseURL}}"],
                "payloads": {
                    "injection": payloads
                },
                "iterations": "{{injection}}"
            }],
            matchers=[
                {
                    "type": "word",
                    "words": ["{{injection}}"],
                    "part": "response"
                },
                {
                    "type": "word",
                    "words": ["text/html"],
                    "part": "header"
                }
            ],
            matchers_condition="and",
            tags=["xss", "custom"]
        )

    @staticmethod
    def create_sqli_template(payloads: List[str]) -> NucleiTemplate:
        """Create an SQL injection testing template."""
        return NucleiTemplate(
            template_id="custom-sqli-template",
            name="Custom SQL Injection Test",
            author="Anarchy Copilot",
            severity="high",
            description="Custom SQL injection payload testing",
            requests=[{
                "method": "GET",
                "path": ["{{BaseURL}}"],
                "payloads": {
                    "injection": payloads
                },
                "iterations": "{{injection}}"
            }],
            matchers=[{
                "type": "regex",
                "regex": [
                    "SQL syntax.*MySQL|Warning.*mysql_.*|valid MySQL result|MySqlClient\\.",
                    "PostgreSQL.*ERROR|Warning.*Pg_.*|valid PostgreSQL result",
                    "Driver.* SQL[\\-\\_\\ ]*Server|OLE DB.* SQL Server|SQLServer.*Driver",
                    "Warning.*sybase.*|Sybase message",
                    "Oracle error",
                    "SQLite/JDBCDriver|SQLite.Exception"
                ],
                "part": "response"
            }],
            matchers_condition="or",
            tags=["sqli", "custom"]
        )

    @staticmethod
    def create_generic_template(
        payload_type: PayloadType,
        payloads: List[str]
    ) -> NucleiTemplate:
        """Create a generic testing template."""
        return NucleiTemplate(
            template_id=f"custom-{payload_type.name.lower()}-template",
            name=f"Custom {payload_type.name} Test",
            author="Anarchy Copilot",
            severity="medium",
            description=f"Custom {payload_type.name} payload testing",
            requests=[{
                "method": "GET",
                "path": ["{{BaseURL}}"],
                "payloads": {
                    "injection": payloads
                },
                "iterations": "{{injection}}"
            }],
            matchers=[{
                "type": "dsl",
                "dsl": ["contains(body, '{{injection}}')"]
            }],
            tags=[payload_type.name.lower(), "custom"]
        )

"""Nuclei vulnerability scanner integration."""

import json
import asyncio
from typing import Dict, List, Any, AsyncIterator, Optional
from pathlib import Path
import tempfile
import os
import aiofiles
from datetime import datetime

from ..models import (
    VulnResult,
    VulnSeverity,
    PayloadResult,
    PayloadType,
    Payload,
    ScanConfig
)
from .base import BaseVulnScanner

class NucleiScanner(BaseVulnScanner):
    """Scanner implementation using Nuclei."""

    SEVERITY_MAP = {
        "info": VulnSeverity.INFO,
        "low": VulnSeverity.LOW,
        "medium": VulnSeverity.MEDIUM,
        "high": VulnSeverity.HIGH,
        "critical": VulnSeverity.CRITICAL
    }

    def __init__(self, config: ScanConfig):
        """Initialize Nuclei scanner."""
        super().__init__(config)
        self._templates_dir: Optional[str] = None
        self._output_file: Optional[str] = None
        self._process: Optional[asyncio.subprocess.Process] = None
        self._custom_templates: List[str] = []

    async def setup(self) -> None:
        """Prepare Nuclei scanner."""
        # Create temporary directory for outputs
        self._templates_dir = tempfile.mkdtemp()
        self._output_file = os.path.join(self._templates_dir, "nuclei_output.json")

        # Write custom headers to file if provided
        if self.config.custom_headers:
            headers_file = os.path.join(self._templates_dir, "headers.txt")
            async with aiofiles.open(headers_file, 'w') as f:
                for k, v in self.config.custom_headers.items():
                    await f.write(f"{k}: {v}\n")

        # Generate custom templates for payload types
        self._custom_templates = await self._generate_templates()

    async def cleanup(self) -> None:
        """Clean up temporary files."""
        if self._process:
            try:
                self._process.terminate()
                await self._process.wait()
            except ProcessLookupError:
                pass  # Process already terminated

        # Clean up temporary files
        if self._templates_dir and os.path.exists(self._templates_dir):
            import shutil
            shutil.rmtree(self._templates_dir)

    async def _generate_templates(self) -> List[str]:
        """Generate Nuclei templates for custom payloads."""
        templates = []
        template_base = os.path.join(self._templates_dir, "custom_templates")
        os.makedirs(template_base, exist_ok=True)

        # Generate templates for each payload type
        supported_payloads = await self.get_supported_payloads()
        for payload_type, payloads in supported_payloads.items():
            template_file = os.path.join(
                template_base,
                f"custom_{payload_type.name.lower()}.yaml"
            )
            template_content = self._create_template(payload_type, payloads)
            
            async with aiofiles.open(template_file, 'w') as f:
                await f.write(template_content)
            templates.append(template_file)

        return templates

    def _create_template(self, payload_type: PayloadType, payloads: List[str]) -> str:
        """Create a Nuclei template for a payload type."""
        # Template structure varies based on payload type
        if payload_type == PayloadType.XSS:
            return self._create_xss_template(payloads)
        elif payload_type == PayloadType.SQLI:
            return self._create_sqli_template(payloads)
        # Add more template types as needed
        return self._create_generic_template(payload_type, payloads)

    async def _scan_target(self) -> AsyncIterator[VulnResult]:
        """Run Nuclei scan and yield results."""
        cmd = [
            "nuclei",
            "-target", self.config.target,
            "-json",
            "-output", self._output_file,
            "-templates", *self._custom_templates
        ]

        if self.config.proxy:
            cmd.extend(["-proxy", self.config.proxy])
        
        if not self.config.verify_ssl:
            cmd.append("-insecure")

        if self.config.rate_limit:
            cmd.extend(["-rate-limit", str(self.config.rate_limit)])

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Read output file as it's being written
            while self._running:
                try:
                    async with aiofiles.open(self._output_file, 'r') as f:
                        content = await f.read()
                        for line in content.splitlines():
                            if line.strip():
                                try:
                                    result = json.loads(line)
                                    vuln = self._parse_nuclei_result(result)
                                    if vuln:
                                        yield vuln
                                except json.JSONDecodeError:
                                    continue
                except FileNotFoundError:
                    pass  # File not created yet

                await asyncio.sleep(1)

        finally:
            if self._process:
                try:
                    self._process.terminate()
                    await self._process.wait()
                except ProcessLookupError:
                    pass

    def _parse_nuclei_result(self, result: Dict[str, Any]) -> Optional[VulnResult]:
        """Convert Nuclei output to VulnResult."""
        try:
            severity = self.SEVERITY_MAP.get(
                result.get("severity", "").lower(),
                VulnSeverity.INFO
            )

            payload_result = PayloadResult(
                payload=Payload(
                    content=result.get("matcher-name", ""),
                    type=self._get_payload_type(result),
                ),
                success=True,
                response_data={
                    "matched": result.get("matched", ""),
                    "extracted": result.get("extracted-values", {}),
                    "ip": result.get("ip", ""),
                    "host": result.get("host", ""),
                    "request": result.get("request", ""),
                    "response": result.get("response", "")
                }
            )

            return VulnResult(
                name=result.get("template-id", "Unknown"),
                type=result.get("type", "Unknown"),
                severity=severity,
                description=result.get("info", {}).get("description", ""),
                endpoint=result.get("matched-at", ""),
                payloads=[payload_result],
                tags=set(result.get("tags", [])),
                references=result.get("info", {}).get("reference", []),
                metadata={
                    "template": result.get("template", ""),
                    "template-url": result.get("template-url", ""),
                    "curl-command": result.get("curl-command", ""),
                }
            )
        except Exception:
            return None

    def _get_payload_type(self, result: Dict[str, Any]) -> PayloadType:
        """Determine payload type from Nuclei result."""
        template_id = result.get("template-id", "").lower()
        if "xss" in template_id:
            return PayloadType.XSS
        elif "sqli" in template_id:
            return PayloadType.SQLI
        elif "command-injection" in template_id:
            return PayloadType.COMMAND_INJECTION
        elif "ssrf" in template_id:
            return PayloadType.SSRF
        elif "path-traversal" in template_id:
            return PayloadType.PATH_TRAVERSAL
        return PayloadType.CUSTOM

    async def verify_vulnerability(self, result: VulnResult) -> bool:
        """Verify if a vulnerability is a true positive."""
        # For now, we trust Nuclei's results
        # TODO: Implement additional verification logic
        return True

    async def test_payload(self, payload: str, payload_type: PayloadType) -> PayloadResult:
        """Test a specific payload using a custom Nuclei template."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            template_content = self._create_generic_template(payload_type, [payload])
            f.write(template_content)
            template_path = f.name

        try:
            cmd = [
                "nuclei",
                "-target", self.config.target,
                "-json",
                "-template", template_path
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, _ = await proc.communicate()
            success = proc.returncode == 0 and bool(stdout)

            return PayloadResult(
                payload=Payload(
                    content=payload,
                    type=payload_type
                ),
                success=success,
                response_data=json.loads(stdout.decode()) if stdout else {}
            )
        finally:
            os.unlink(template_path)

    async def get_supported_payloads(self) -> Dict[PayloadType, List[str]]:
        """Get built-in Nuclei payloads by type."""
        # For now, return basic examples. In practice, these would be loaded 
        # from Nuclei's template database
        return {
            PayloadType.XSS: [
                "<script>alert(1)</script>",
                "javascript:alert(1)",
            ],
            PayloadType.SQLI: [
                "' OR '1'='1",
                "1 UNION SELECT NULL--",
            ],
            PayloadType.COMMAND_INJECTION: [
                "$(id)",
                "`id`",
            ],
            PayloadType.PATH_TRAVERSAL: [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\win.ini",
            ],
        }

    def _create_xss_template(self, payloads: List[str]) -> str:
        """Create a Nuclei template for XSS testing."""
        return f"""
id: custom-xss-template
info:
  name: Custom XSS Test
  author: Anarchy Copilot
  severity: medium
  description: Custom XSS payload testing

requests:
  - method: GET
    path:
      - "{{{{BaseURL}}}}"
    payloads:
      injection:
        - {json.dumps(payloads)}
    
    iterations: "{{{{injection}}}}"
    
    matchers-condition: and
    matchers:
      - type: word
        words:
          - "{{{{injection}}}}"
        part: response
      - type: word
        words:
          - "text/html"
        part: header
"""

    def _create_sqli_template(self, payloads: List[str]) -> str:
        """Create a Nuclei template for SQL injection testing."""
        return f"""
id: custom-sqli-template
info:
  name: Custom SQL Injection Test
  author: Anarchy Copilot
  severity: high
  description: Custom SQL injection payload testing

requests:
  - method: GET
    path:
      - "{{{{BaseURL}}}}"
    payloads:
      injection:
        - {json.dumps(payloads)}
    
    iterations: "{{{{injection}}}}"
    
    matchers-condition: or
    matchers:
      - type: regex
        regex:
          - "SQL syntax.*MySQL|Warning.*mysql_.*|valid MySQL result|MySqlClient\\."|
          - "PostgreSQL.*ERROR|Warning.*Pg_.*|valid PostgreSQL result"|
          - "Driver.* SQL[\\-\\_\\ ]*Server|OLE DB.* SQL Server|SQLServer.*Driver"|
          - "Warning.*sybase.*|Sybase message"|
          - "Oracle error"|
          - "SQLite/JDBCDriver|SQLite.Exception"
        part: response
"""

    def _create_generic_template(self, payload_type: PayloadType, payloads: List[str]) -> str:
        """Create a generic Nuclei template for testing."""
        return f"""
id: custom-{payload_type.name.lower()}-template
info:
  name: Custom {payload_type.name} Test
  author: Anarchy Copilot
  severity: medium
  description: Custom {payload_type.name} payload testing

requests:
  - method: GET
    path:
      - "{{{{BaseURL}}}}"
    payloads:
      injection:
        - {json.dumps(payloads)}
    
    iterations: "{{{{injection}}}}"
    
    matchers:
      - type: dsl
        dsl:
          - 'contains(body, "{{{{injection}}}}")'
"""

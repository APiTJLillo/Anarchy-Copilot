"""Main Nuclei scanner implementation."""
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, AsyncIterator, Any
import aiofiles
import asyncio

from ..base import BaseVulnScanner
from ...models import (
    VulnResult,
    PayloadResult,
    PayloadType,
    Payload,
    ScanConfig
)
from .templates import NucleiTemplateGenerator, NucleiTemplate
from .process import NucleiProcess
from .results import NucleiResultParser

logger = logging.getLogger(__name__)

class NucleiScanner(BaseVulnScanner):
    """Scanner implementation using Nuclei."""

    def __init__(self, config: ScanConfig):
        """Initialize Nuclei scanner."""
        super().__init__(config)
        self._templates_dir: Optional[Path] = None
        self._output_file: Optional[Path] = None
        self._process: Optional[NucleiProcess] = None
        self._custom_templates: List[str] = []
        self._template_generator = NucleiTemplateGenerator()

    async def setup(self) -> None:
        """Prepare Nuclei scanner."""
        # Create temporary directory for outputs
        self._templates_dir = Path(tempfile.mkdtemp())
        self._output_file = self._templates_dir / "nuclei_output.json"
        self._process = NucleiProcess()
        self._process.set_output_file(self._output_file)

        # Write custom headers to file if provided
        if self.config.custom_headers:
            headers_file = self._templates_dir / "headers.txt"
            async with aiofiles.open(str(headers_file), 'w') as f:
                for k, v in self.config.custom_headers.items():
                    await f.write(f"{k}: {v}\n")

        # Generate custom templates for payload types
        template_dir = self._templates_dir / "templates"
        template_dir.mkdir(exist_ok=True)
        
        # Generate templates based on configured payload types
        supported_payloads = await self.get_supported_payloads()
        for payload_type, payloads in supported_payloads.items():
            if payload_type not in self.config.payload_types:
                continue

            template = self._create_template(payload_type, payloads)
            if template:
                template_path = template_dir / f"{payload_type.name.lower()}.yaml"
                async with aiofiles.open(str(template_path), 'w') as f:
                    await f.write(template.to_yaml())
                self._custom_templates.append(str(template_path))

        logger.info(f"Generated {len(self._custom_templates)} custom templates")

    async def cleanup(self) -> None:
        """Clean up temporary files."""
        if self._process:
            await self._process.stop()
            self._process = None
        
        if self._templates_dir and self._templates_dir.exists():
            shutil.rmtree(str(self._templates_dir))
            self._templates_dir = None
            self._output_file = None
            self._custom_templates = []

    def _create_template(
        self, 
        payload_type: PayloadType,
        payloads: List[str]
    ) -> Optional[NucleiTemplate]:
        """Create appropriate template for payload type."""
        if payload_type == PayloadType.XSS:
            return self._template_generator.create_xss_template(payloads)
        elif payload_type == PayloadType.SQLI:
            return self._template_generator.create_sqli_template(payloads)
        else:
            return self._template_generator.create_generic_template(payload_type, payloads)

    async def _scan_target(self) -> AsyncIterator[VulnResult]:
        """Run Nuclei scan and yield results."""
        if not self._process:
            return

        try:
            await self._process.run(
                target=self.config.target,
                templates=self._custom_templates,
                proxy=self.config.proxy,
                verify_ssl=self.config.verify_ssl,
                rate_limit=self.config.rate_limit
            )

            async for result in self._process.read_results():
                if vuln := NucleiResultParser.parse_result(result):
                    self._stats["vulnerabilities_found"] += 1
                    yield vuln
                await self.rate_limit()

        except FileNotFoundError:
            logger.error("Nuclei binary not found. Please ensure nuclei is installed.")
        except Exception as e:
            logger.error(f"Error during scan: {e}")
            raise

    async def verify_vulnerability(self, result: VulnResult) -> bool:
        """Verify if a vulnerability is a true positive."""
        # For now, we trust Nuclei's results
        # TODO: Implement additional verification logic
        return True

    async def test_payload(self, payload: str, payload_type: PayloadType) -> PayloadResult:
        """Test a specific payload using a custom Nuclei template."""
        template = self._create_template(payload_type, [payload])
        if not template:
            return PayloadResult(
                payload=Payload(content=payload, type=payload_type),
                success=False,
                response_data={},
                error="Unsupported payload type"
            )

        # Create temporary template file
        template_file = Path(tempfile.mktemp(suffix='.yaml'))
        output_file = Path(tempfile.mktemp())

        try:
            # Write template to file
            async with aiofiles.open(str(template_file), 'w') as f:
                await f.write(template.to_yaml())

            # Configure and run nuclei process
            process = NucleiProcess()
            process.set_output_file(output_file)

            try:
                await process.run(
                    target=self.config.target,
                    templates=[str(template_file)],
                    proxy=self.config.proxy,
                    verify_ssl=self.config.verify_ssl
                )

                found = False
                async for result in process.read_results():
                    if vuln := NucleiResultParser.parse_result(result):
                        if vuln.payloads and len(vuln.payloads) > 0:
                            await process.stop()
                            return vuln.payloads[0]
                        found = True
                        break

                return PayloadResult(
                    payload=Payload(content=payload, type=payload_type),
                    success=found,
                    response_data={},
                    error="No match found"
                )

            finally:
                await process.stop()

        except FileNotFoundError:
            return PayloadResult(
                payload=Payload(content=payload, type=payload_type),
                success=False,
                response_data={},
                error="Nuclei binary not found"
            )
        finally:
            # Clean up temporary files
            if template_file.exists():
                template_file.unlink()
            if output_file.exists():
                output_file.unlink()

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
            PayloadType.SSRF: [
                "http://169.254.169.254/latest/meta-data/",
                "file:///etc/passwd",
            ],
            PayloadType.TEMPLATE_INJECTION: [
                "${7*7}",
                "{{7*7}}",
            ],
            PayloadType.XXE: [
                '<?xml version="1.0"?><!DOCTYPE root [<!ENTITY test SYSTEM "file:///etc/passwd">]><root>&test;</root>',
            ],
            PayloadType.DESERIALIZATION: [
                'O:8:"stdClass":0:{}',
            ],
            PayloadType.FILE_UPLOAD: [
                "test.php",
                "test.jsp",
            ],
        }

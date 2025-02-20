"""Models package."""
# Base models
from .base import Base, User, Project, project_collaborators

# Recon models
from .recon import ReconResult, ReconTarget, ReconModule

# Vulnerability models
from .vulnerability import (
    Vulnerability, VulnerabilityComment, VulnerabilityResult, 
    VulnerabilityScan, Tag, Report,
    vulnerability_tags, report_vulnerabilities,
    SeverityLevel, VulnerabilityStatus
)

# Import proxy models last to avoid circular imports
from api.proxy.database_models import ProxySession, ProxyHistoryEntry

__all__ = [
    # Base models
    'Base', 'User', 'Project', 'project_collaborators',
    
    # Recon models
    'ReconResult', 'ReconTarget', 'ReconModule',
    
    # Vulnerability models
    'Vulnerability', 'VulnerabilityComment', 'VulnerabilityResult',
    'VulnerabilityScan', 'Tag', 'Report',
    'vulnerability_tags', 'report_vulnerabilities',
    'SeverityLevel', 'VulnerabilityStatus',
    
    # Proxy models
    'ProxySession', 'ProxyHistoryEntry'
]

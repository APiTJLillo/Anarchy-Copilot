"""Recon API endpoints."""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from recon_module.models import ScanConfig
from recon_module.recon_manager import ReconManager

router = APIRouter()

class ScanRequest(BaseModel):
    """Scan request model."""
    domain: str = Field(..., description="Target domain to scan")
    tool: str = Field(..., description="Tool to use for scanning (subdomain, portscan, service)")
    
class ScanResponse(BaseModel):
    """Scan response model."""
    scan_id: str = Field(..., description="Unique identifier for the scan")
    status: str = Field(..., description="Status of the scan")
    message: str = Field(..., description="Additional information about the scan")

class ScanResult(BaseModel):
    """Scan result model."""
    id: str = Field(..., description="Unique identifier for the result")
    domain: str = Field(..., description="Target domain")
    tool: str = Field(..., description="Tool used for scanning")
    timestamp: str = Field(..., description="Timestamp of the scan")
    results: Dict[str, Any] = Field(..., description="Scan results")

# Store active scans in memory (in a real app, this would be in a database)
active_scans = {}
scan_history = []
scan_results = {}

@router.post("/scan", response_model=ScanResponse)
async def start_scan(scan_request: ScanRequest, background_tasks: BackgroundTasks):
    """Start a new scan."""
    # Generate a unique scan ID
    import uuid
    scan_id = str(uuid.uuid4())
    
    # Create scan config based on the requested tool
    config = ScanConfig(
        target=scan_request.domain,
        network_scan_enabled=scan_request.tool == "portscan",
        port_scan_enabled=scan_request.tool == "portscan",
        dns_scan_enabled=scan_request.tool == "subdomain",
        subdomain_scan_enabled=scan_request.tool == "subdomain",
        service_scan_enabled=scan_request.tool == "service"
    )
    
    # Store scan in active scans
    active_scans[scan_id] = {
        "domain": scan_request.domain,
        "tool": scan_request.tool,
        "status": "pending",
        "timestamp": str(import_datetime().now())
    }
    
    # Run scan in background
    background_tasks.add_task(run_scan_task, scan_id, config)
    
    return ScanResponse(
        scan_id=scan_id,
        status="pending",
        message=f"Scan started for {scan_request.domain} using {scan_request.tool}"
    )

def import_datetime():
    """Import datetime module."""
    from datetime import datetime
    return datetime

async def run_scan_task(scan_id: str, config: ScanConfig):
    """Run scan task in background."""
    try:
        # Update scan status
        active_scans[scan_id]["status"] = "running"
        
        # Initialize recon manager
        recon_manager = ReconManager()
        
        # Run scan
        results = await recon_manager.run_scan(config)
        
        # Store results
        scan_results[scan_id] = results
        
        # Update scan status
        active_scans[scan_id]["status"] = "completed"
        
        # Add to history
        scan_history.append({
            "id": scan_id,
            "domain": active_scans[scan_id]["domain"],
            "tool": active_scans[scan_id]["tool"],
            "timestamp": active_scans[scan_id]["timestamp"],
            "status": "completed"
        })
    except Exception as e:
        # Update scan status on error
        active_scans[scan_id]["status"] = "failed"
        active_scans[scan_id]["error"] = str(e)

@router.get("/scans/active", response_model=List[Dict[str, Any]])
async def get_active_scans():
    """Get all active scans."""
    return [{"id": k, **v} for k, v in active_scans.items()]

@router.get("/scans/history", response_model=List[Dict[str, Any]])
async def get_scan_history():
    """Get scan history."""
    return scan_history

@router.get("/scans/{scan_id}", response_model=Dict[str, Any])
async def get_scan_status(scan_id: str):
    """Get scan status."""
    if scan_id in active_scans:
        return {"id": scan_id, **active_scans[scan_id]}
    
    # Check history
    for scan in scan_history:
        if scan["id"] == scan_id:
            return scan
    
    raise HTTPException(status_code=404, detail="Scan not found")

@router.get("/scans/{scan_id}/results", response_model=List[Dict[str, Any]])
async def get_scan_results(scan_id: str):
    """Get scan results."""
    if scan_id in scan_results:
        return scan_results[scan_id]
    
    raise HTTPException(status_code=404, detail="Scan results not found")

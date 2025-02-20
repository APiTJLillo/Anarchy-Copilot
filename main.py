from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import logging
import typing

from recon_module.recon_manager import ReconManager
from models.recon import ReconResult
from database import get_db
from api.proxy import router as proxy_router
from api.websocket import router as websocket_router
from api.projects import router as projects_router
from api import create_app

import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_and_configure_app():
    """Create and configure FastAPI application with enhanced logging."""
    logger.info("Starting application creation")
    
    # Create fresh app instance
    # Create base app with minimal config
    app = create_app({
        "debug": True, 
        "cors_origins": ["http://localhost:3000"]
    })

    logger.info("Base app created")

    # Add routers
    logger.info("Adding routers")
    app.include_router(projects_router)  # Projects router already has /api/projects prefix
    app.include_router(proxy_router, prefix="/api/proxy")
    app.include_router(websocket_router, prefix="/api/proxy/websocket")

    # Add diagnostic routes
    @app.get("/api/health")
    async def health_check():
        """Basic health check endpoint."""
        return {"status": "healthy"}
    
    @app.get("/api/debug/routes")
    async def list_routes():
        """List all registered routes."""
        return {
            "routes": [str(route) for route in app.routes],
            "count": len(app.routes)
        }

    logger.info("Application configured successfully")
    return app

# Create the FastAPI application instance
app = create_and_configure_app()

# Register startup event
@app.on_event("startup")
async def startup_event():
    """Log when application starts."""
    logger.info("FastAPI application starting up")
    # Test the proxy status endpoint
    from api.proxy.endpoints import get_proxy_status
    try:
        status = await get_proxy_status()
        logger.info(f"Initial proxy status: {status}")
    except Exception as e:
        logger.error(f"Error getting proxy status: {e}")

# Store active scans
active_scans: Dict[str, ReconManager] = {}

@app.get("/recon_results/")
async def get_recon_results(
    tool: str,
    domain: str,
    project_id: Optional[int] = None,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    if not domain:
        raise HTTPException(status_code=400, detail="Domain is required")
    
    scan_id = f"{tool}_{domain}_{project_id}"
    
    try:
        recon_manager = ReconManager(project_id=project_id)
        active_scans[scan_id] = recon_manager
        
        recon_result = await recon_manager.run_tool(tool, domain, db)
        
        # Update result with additional metadata
        categorized_domains = recon_manager.categorize_domains(recon_result.results.get("domains", []))
        
        # Add categories to results
        result_dict = {
            "id": recon_result.id,
            "tool": recon_result.tool,
            "domain": recon_result.domain,
            "results": recon_result.results,
            "status": recon_result.status,
            "error_message": recon_result.error_message,
            "start_time": recon_result.start_time.isoformat() if recon_result.start_time else None,
            "end_time": recon_result.end_time.isoformat() if recon_result.end_time else None,
            "project_id": recon_result.project_id,
            "categories": categorized_domains,
            "total_found": len(recon_result.results.get("domains", [])),
            "response_time": (recon_result.end_time - recon_result.start_time).total_seconds() if recon_result.end_time and recon_result.start_time else None
        }
        
        # Clean up after scan completes
        del active_scans[scan_id]
        
        return result_dict
    
    except Exception as e:
        if scan_id in active_scans:
            del active_scans[scan_id]
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recon_results/progress/{scan_id}")
async def get_scan_progress(scan_id: str) -> Dict[str, Any]:
    """Get progress information for an active scan."""
    if scan_id not in active_scans:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    return active_scans[scan_id].get_progress()

@app.get("/recon_results/history/")
async def get_recon_history(
    db: Session = Depends(get_db),
    project_id: Optional[int] = None,
    tool: Optional[str] = None,
    domain: Optional[str] = None,
    date_range: Optional[str] = None,  # Format: "YYYY-MM-DD,YYYY-MM-DD"
    min_results: Optional[int] = None,
    category: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get historical recon results with enhanced filtering."""
    query = db.query(ReconResult)
    
    if project_id:
        query = query.filter(ReconResult.project_id == project_id)
    if tool:
        query = query.filter(ReconResult.tool == tool)
    if domain:
        query = query.filter(ReconResult.domain.ilike(f"%{domain}%"))
    
    # Handle date range filtering
    if date_range:
        try:
            start_date, end_date = date_range.split(",")
            query = query.filter(
                ReconResult.start_time >= datetime.strptime(start_date, "%Y-%m-%d"),
                ReconResult.start_time <= datetime.strptime(end_date, "%Y-%m-%d")
            )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date range format")
    
    results = query.all()
    
    # Post-query filtering
    filtered_results = []
    for result in results:
        result_dict = result.to_dict()
        domains = result.results.get("domains", [])
        total_results = len(domains)
        
        # Filter by minimum results
        if min_results is not None and total_results < min_results:
            continue
            
        # Add categories
        recon_manager = ReconManager()
        result_dict["categories"] = recon_manager.categorize_domains(domains)
        
        # Filter by category
        if category and not any(domain for domain in domains if category in domain):
            continue
            
        # Add metadata
        result_dict["total_found"] = total_results
        if result.start_time and result.end_time:
            result_dict["response_time"] = (result.end_time - result.start_time).total_seconds()
            
        filtered_results.append(result_dict)
    
    return filtered_results

if __name__ == "__main__":
    import uvicorn
    # Ensure we bind to all interfaces when running in Docker
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile="app/certs/ca.key",
        ssl_certfile="app/certs/ca.crt"
    )

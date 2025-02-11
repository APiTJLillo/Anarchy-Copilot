from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from fastapi import Depends

from recon_module.recon_manager import ReconManager
from models import Project, User, ReconResult
from database import engine, get_db

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    uvicorn.run(app, host="0.0.0.0", port=8000)

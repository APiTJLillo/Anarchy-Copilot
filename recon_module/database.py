"""Database operations for reconnaissance results."""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import sqlalchemy.orm  # type: ignore
from sqlalchemy import desc  # type: ignore

from models import ReconResult

class ReconDatabase:
    """Handles database operations for reconnaissance results."""

    def __init__(self, db: sqlalchemy.orm.Session, project_id: Optional[int] = None):
        """Initialize database handler with project context."""
        self.db = db
        self.project_id = project_id

    async def save_result(self, tool: str, domain: str, results: Dict[str, Any], error: Optional[str] = None) -> ReconResult:
        """Save scan results to database."""
        now = datetime.utcnow()
        recon_result = ReconResult(
            scan_type=tool,
            domain=domain,
            project_id=self.project_id,
            status="completed" if not error else "failed",
            results=results,
            error_message=error,
            start_time=now,  # Would be more accurate if passed from scanner
            end_time=now,
            metadata={"scheduled": True}  # Mark as scheduled scan
        )
        self.db.add(recon_result)
        self.db.commit()
        self.db.refresh(recon_result)
        return recon_result

    async def get_results_since(
        self, 
        domain: str, 
        since: datetime,
        tools: Optional[List[str]] = None
    ) -> List[ReconResult]:
        """Get scan results for a domain since a specific time."""
        query = self.db.query(ReconResult).filter(
            ReconResult.domain == domain,
            ReconResult.end_time >= since
        )

        if self.project_id is not None:
            query = query.filter(ReconResult.project_id == self.project_id)

        if tools:
            query = query.filter(ReconResult.scan_type.in_(tools))

        # Order by end_time to get most recent first
        return query.order_by(desc(ReconResult.end_time)).all()

    async def get_latest_result(
        self, 
        domain: str, 
        tool: Optional[str] = None
    ) -> Optional[ReconResult]:
        """Get the most recent scan result for a domain."""
        query = self.db.query(ReconResult).filter(
            ReconResult.domain == domain
        )

        if tool:
            query = query.filter(ReconResult.scan_type == tool)
        if self.project_id is not None:
            query = query.filter(ReconResult.project_id == self.project_id)

        return query.order_by(desc(ReconResult.end_time)).first()

    async def get_results_between(
        self,
        domain: str,
        start: datetime,
        end: datetime,
        tools: Optional[List[str]] = None
    ) -> List[ReconResult]:
        """Get scan results between two timestamps."""
        query = self.db.query(ReconResult).filter(
            ReconResult.domain == domain,
            ReconResult.end_time >= start,
            ReconResult.end_time <= end
        )

        if self.project_id is not None:
            query = query.filter(ReconResult.project_id == self.project_id)
        if tools:
            query = query.filter(ReconResult.scan_type.in_(tools))

        return query.order_by(desc(ReconResult.end_time)).all()

    async def get_change_history(
        self,
        domain: str,
        since: Optional[datetime] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get history of changes for a domain."""
        query = self.db.query(ReconResult).filter(
            ReconResult.domain == domain
        )

        if since:
            query = query.filter(ReconResult.end_time >= since)
        if self.project_id is not None:
            query = query.filter(ReconResult.project_id == self.project_id)

        results = query.order_by(desc(ReconResult.end_time)).limit(limit).all()
        
        # Extract changes from metadata
        changes = []
        for result in results:
            if result.metadata and "changes" in result.metadata:
                changes.append({
                    "timestamp": result.end_time,
                    "scan_type": result.scan_type,
                    "changes": result.metadata["changes"]
                })
        
        return changes

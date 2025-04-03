"""
API endpoints for the Traffic Analysis Engine.

This module provides API endpoints for accessing traffic analysis functionality,
including security issues, behavior patterns, and analysis rules.
"""
from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Dict, List, Optional, Any
import logging

from ..proxy.analysis_manager import TrafficAnalysisManager, AnalysisResult, BehaviorPattern
from ..proxy.analysis_rules import AnalysisRulesEngine, AnalysisRule

# Initialize the router
router = APIRouter(prefix="/api/analysis", tags=["analysis"])

# Initialize the analysis manager and rules engine
analysis_manager = TrafficAnalysisManager()
rules_engine = AnalysisRulesEngine()

# Logger
logger = logging.getLogger(__name__)

# Helper function to get analysis manager
async def get_analysis_manager():
    return analysis_manager

# Helper function to get rules engine
async def get_rules_engine():
    return rules_engine

@router.get("/security-issues")
async def get_security_issues(
    severity: Optional[str] = None,
    manager: TrafficAnalysisManager = Depends(get_analysis_manager)
):
    """Get security issues detected in traffic."""
    try:
        issues = await manager.get_security_issues(severity)
        return {
            "success": True,
            "issues": [issue.to_dict() for issue in issues]
        }
    except Exception as e:
        logger.error(f"Error getting security issues: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/behavior-patterns")
async def get_behavior_patterns(
    pattern_type: Optional[str] = None,
    manager: TrafficAnalysisManager = Depends(get_analysis_manager)
):
    """Get behavior patterns detected in traffic."""
    try:
        patterns = await manager.get_behavior_patterns(pattern_type)
        return {
            "success": True,
            "patterns": [pattern.to_dict() for pattern in patterns]
        }
    except Exception as e:
        logger.error(f"Error getting behavior patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results")
async def get_analysis_results(
    request_id: Optional[str] = None,
    manager: TrafficAnalysisManager = Depends(get_analysis_manager)
):
    """Get analysis results for traffic."""
    try:
        if request_id:
            result = await manager.get_analysis_result(request_id)
            if not result:
                return {"success": False, "error": "Analysis result not found"}
            return {
                "success": True,
                "result": result.to_dict()
            }
        else:
            results = await manager.get_all_analysis_results()
            return {
                "success": True,
                "results": {req_id: result.to_dict() for req_id, result in results.items()}
            }
    except Exception as e:
        logger.error(f"Error getting analysis results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rules")
async def get_analysis_rules(
    rule_type: Optional[str] = None,
    engine: AnalysisRulesEngine = Depends(get_rules_engine)
):
    """Get analysis rules."""
    try:
        if rule_type:
            rules = engine.get_rules_by_type(rule_type)
        else:
            rules = engine.get_all_rules()
        return {
            "success": True,
            "rules": [rule.to_dict() for rule in rules]
        }
    except Exception as e:
        logger.error(f"Error getting analysis rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/rules/{rule_id}")
async def get_analysis_rule(
    rule_id: str,
    engine: AnalysisRulesEngine = Depends(get_rules_engine)
):
    """Get a specific analysis rule."""
    try:
        rule = engine.get_rule(rule_id)
        if not rule:
            return {"success": False, "error": "Rule not found"}
        return {
            "success": True,
            "rule": rule.to_dict()
        }
    except Exception as e:
        logger.error(f"Error getting analysis rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rules")
async def add_analysis_rule(
    rule_data: Dict[str, Any] = Body(...),
    engine: AnalysisRulesEngine = Depends(get_rules_engine)
):
    """Add a new analysis rule."""
    try:
        rule = AnalysisRule.from_dict(rule_data)
        success = engine.add_rule(rule)
        return {
            "success": success,
            "rule_id": rule.id if success else None
        }
    except Exception as e:
        logger.error(f"Error adding analysis rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/rules/{rule_id}")
async def delete_analysis_rule(
    rule_id: str,
    engine: AnalysisRulesEngine = Depends(get_rules_engine)
):
    """Delete an analysis rule."""
    try:
        success = engine.remove_rule(rule_id)
        return {
            "success": success
        }
    except Exception as e:
        logger.error(f"Error deleting analysis rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rules/import")
async def import_rules_from_file(
    file_path: str = Body(..., embed=True),
    engine: AnalysisRulesEngine = Depends(get_rules_engine)
):
    """Import rules from a file."""
    try:
        count = engine.load_rules_from_file(file_path)
        return {
            "success": True,
            "rules_imported": count
        }
    except Exception as e:
        logger.error(f"Error importing rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rules/export")
async def export_rules_to_file(
    file_path: str = Body(..., embed=True),
    engine: AnalysisRulesEngine = Depends(get_rules_engine)
):
    """Export rules to a file."""
    try:
        success = engine.save_rules_to_file(file_path)
        return {
            "success": success
        }
    except Exception as e:
        logger.error(f"Error exporting rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear")
async def clear_analysis_data(
    manager: TrafficAnalysisManager = Depends(get_analysis_manager)
):
    """Clear all analysis data."""
    try:
        await manager.clear_analysis_data()
        return {
            "success": True
        }
    except Exception as e:
        logger.error(f"Error clearing analysis data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

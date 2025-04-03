"""
API endpoints for the advanced filtering system with bypass mode.

This module provides API endpoints for:
1. Managing filter rules (CRUD operations)
2. Switching filtering modes
3. Accessing traffic history
4. Creating rules from traffic
5. Getting filter suggestions
"""

import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field

from proxy.filter import FilterMode, FilterRule, FilterCondition, FilterManager
from proxy.analysis.filter_analysis import FilterAnalyzer
from database import get_db, AsyncSession

# Initialize router
router = APIRouter(
    prefix="/api/filter",
    tags=["filter"],
)

logger = logging.getLogger(__name__)

# Initialize filter manager and analyzer
filter_manager = None
filter_analyzer = FilterAnalyzer()

# Pydantic models for API requests and responses
class FilterConditionModel(BaseModel):
    """Model for filter condition."""
    field: str = Field(..., description="Field to match against")
    operator: str = Field(..., description="Operator to use")
    value: Any = Field(..., description="Value to compare against")
    
    class Config:
        schema_extra = {
            "example": {
                "field": "path",
                "operator": "contains",
                "value": "/admin"
            }
        }

class FilterRuleModel(BaseModel):
    """Model for filter rule."""
    id: Optional[str] = Field(None, description="Unique identifier for the rule")
    name: str = Field(..., description="Human-readable name for the rule")
    description: Optional[str] = Field("", description="Detailed description of the rule")
    conditions: List[FilterConditionModel] = Field(..., description="List of conditions that must be met for the rule to match")
    enabled: Optional[bool] = Field(True, description="Whether the rule is enabled")
    priority: Optional[int] = Field(0, description="Priority of the rule (higher values have higher priority)")
    tags: Optional[List[str]] = Field([], description="List of tags for categorizing the rule")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Block Admin Access",
                "description": "Block access to admin pages",
                "conditions": [
                    {
                        "field": "path",
                        "operator": "contains",
                        "value": "/admin"
                    }
                ],
                "enabled": True,
                "priority": 10,
                "tags": ["admin", "security"]
            }
        }

class FilterModeModel(BaseModel):
    """Model for filter mode."""
    mode: str = Field(..., description="Filter mode (ACTIVE or BYPASS)")
    
    class Config:
        schema_extra = {
            "example": {
                "mode": "ACTIVE"
            }
        }

class TrafficItemModel(BaseModel):
    """Model for traffic history item."""
    id: int = Field(..., description="Unique identifier for the traffic item")
    type: str = Field(..., description="Type of traffic (request or response)")
    method: Optional[str] = Field(None, description="HTTP method (for requests)")
    path: Optional[str] = Field(None, description="Request path (for requests)")
    request_method: Optional[str] = Field(None, description="Original request method (for responses)")
    request_path: Optional[str] = Field(None, description="Original request path (for responses)")
    status_code: Optional[int] = Field(None, description="HTTP status code (for responses)")
    headers: Dict[str, str] = Field(..., description="HTTP headers")
    body: Optional[str] = Field(None, description="Request or response body")
    timestamp: str = Field(..., description="Timestamp when the traffic was recorded")
    matched_rules: Optional[List[str]] = Field(None, description="List of rule IDs that matched this traffic")
    filtered: Optional[bool] = Field(None, description="Whether this traffic was filtered")

class TrafficHistoryModel(BaseModel):
    """Model for traffic history response."""
    items: List[TrafficItemModel] = Field(..., description="List of traffic history items")
    total: int = Field(..., description="Total number of items")

class FilterSuggestionModel(BaseModel):
    """Model for filter suggestion."""
    rule: FilterRuleModel = Field(..., description="Suggested filter rule")
    match_count: int = Field(..., description="Number of traffic items that would match this rule")
    sample_matches: List[Dict[str, Any]] = Field(..., description="Sample of traffic items that would match this rule")

class FilterSuggestionsModel(BaseModel):
    """Model for filter suggestions response."""
    suggestions: List[FilterSuggestionModel] = Field(..., description="List of filter suggestions")

class PreviewResultModel(BaseModel):
    """Model for rule preview result."""
    would_match: bool = Field(..., description="Whether the rule would match any traffic")
    traffic_count: int = Field(..., description="Number of traffic items that would match this rule")
    sample_matches: List[Dict[str, Any]] = Field(..., description="Sample of traffic items that would match this rule")

# Helper function to get filter manager
async def get_filter_manager(db: AsyncSession = Depends(get_db)) -> FilterManager:
    """Get or initialize the filter manager."""
    global filter_manager
    if filter_manager is None:
        from proxy.filter import DatabaseFilterStorage
        storage = DatabaseFilterStorage()
        filter_manager = FilterManager(storage)
    return filter_manager

# API endpoints
@router.get("/mode", response_model=FilterModeModel)
async def get_filter_mode(
    filter_manager: FilterManager = Depends(get_filter_manager)
) -> FilterModeModel:
    """Get the current filter mode."""
    mode = await filter_manager.get_mode()
    return FilterModeModel(mode=mode.name)

@router.put("/mode", response_model=FilterModeModel)
async def set_filter_mode(
    mode_data: FilterModeModel,
    filter_manager: FilterManager = Depends(get_filter_manager)
) -> FilterModeModel:
    """Set the filter mode."""
    try:
        mode = FilterMode[mode_data.mode]
        success = await filter_manager.set_mode(mode)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to set filter mode")
        return FilterModeModel(mode=mode.name)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode_data.mode}. Must be ACTIVE or BYPASS")

@router.get("/rules", response_model=List[FilterRuleModel])
async def get_filter_rules(
    filter_manager: FilterManager = Depends(get_filter_manager)
) -> List[FilterRuleModel]:
    """Get all filter rules."""
    rules = await filter_manager.get_all_rules()
    return [
        FilterRuleModel(
            id=rule.id,
            name=rule.name,
            description=rule.description,
            conditions=[
                FilterConditionModel(
                    field=condition.field,
                    operator=condition.operator,
                    value=condition.value
                )
                for condition in rule.conditions
            ],
            enabled=rule.enabled,
            priority=rule.priority,
            tags=rule.tags
        )
        for rule in rules
    ]

@router.get("/rules/{rule_id}", response_model=FilterRuleModel)
async def get_filter_rule(
    rule_id: str = Path(..., description="Unique identifier of the rule"),
    filter_manager: FilterManager = Depends(get_filter_manager)
) -> FilterRuleModel:
    """Get a filter rule by ID."""
    rule = await filter_manager.get_rule(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")
    
    return FilterRuleModel(
        id=rule.id,
        name=rule.name,
        description=rule.description,
        conditions=[
            FilterConditionModel(
                field=condition.field,
                operator=condition.operator,
                value=condition.value
            )
            for condition in rule.conditions
        ],
        enabled=rule.enabled,
        priority=rule.priority,
        tags=rule.tags
    )

@router.post("/rules", response_model=FilterRuleModel)
async def create_filter_rule(
    rule_data: FilterRuleModel,
    filter_manager: FilterManager = Depends(get_filter_manager)
) -> FilterRuleModel:
    """Create a new filter rule."""
    # Convert from API model to domain model
    rule = FilterRule(
        id=rule_data.id,  # Will generate a new ID if None
        name=rule_data.name,
        description=rule_data.description,
        conditions=[
            FilterCondition(
                field=condition.field,
                operator=condition.operator,
                value=condition.value
            )
            for condition in rule_data.conditions
        ],
        enabled=rule_data.enabled,
        priority=rule_data.priority,
        tags=rule_data.tags
    )
    
    success = await filter_manager.add_rule(rule)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to create filter rule")
    
    # Return the created rule with its generated ID
    return FilterRuleModel(
        id=rule.id,
        name=rule.name,
        description=rule.description,
        conditions=[
            FilterConditionModel(
                field=condition.field,
                operator=condition.operator,
                value=condition.value
            )
            for condition in rule.conditions
        ],
        enabled=rule.enabled,
        priority=rule.priority,
        tags=rule.tags
    )

@router.put("/rules/{rule_id}", response_model=FilterRuleModel)
async def update_filter_rule(
    rule_data: FilterRuleModel,
    rule_id: str = Path(..., description="Unique identifier of the rule"),
    filter_manager: FilterManager = Depends(get_filter_manager)
) -> FilterRuleModel:
    """Update an existing filter rule."""
    # Check if rule exists
    existing_rule = await filter_manager.get_rule(rule_id)
    if existing_rule is None:
        raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")
    
    # Convert from API model to domain model
    rule = FilterRule(
        id=rule_id,  # Use the path parameter ID
        name=rule_data.name,
        description=rule_data.description,
        conditions=[
            FilterCondition(
                field=condition.field,
                operator=condition.operator,
                value=condition.value
            )
            for condition in rule_data.conditions
        ],
        enabled=rule_data.enabled,
        priority=rule_data.priority,
        tags=rule_data.tags
    )
    
    success = await filter_manager.update_rule(rule)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update filter rule")
    
    return rule_data

@router.delete("/rules/{rule_id}", response_model=Dict[str, bool])
async def delete_filter_rule(
    rule_id: str = Path(..., description="Unique identifier of the rule"),
    filter_manager: FilterManager = Depends(get_filter_manager)
) -> Dict[str, bool]:
    """Delete a filter rule."""
    # Check if rule exists
    existing_rule = await filter_manager.get_rule(rule_id)
    if existing_rule is None:
        raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")
    
    success = await filter_manager.delete_rule(rule_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete filter rule")
    
    return {"success": True}

@router.get("/traffic", response_model=TrafficHistoryModel)
async def get_traffic_history(
    page: int = Query(1, description="Page number"),
    page_size: int = Query(10, description="Number of items per page"),
    filter_type: Optional[str] = Query(None, description="Filter by traffic type (request or response)"),
    filter_path: Optional[str] = Query(None, description="Filter by path (contains)"),
    filter_method: Optional[str] = Query(None, description="Filter by method"),
    filter_status: Optional[int] = Query(None, description="Filter by status code"),
    filter_manager: FilterManager = Depends(get_filter_manager)
) -> TrafficHistoryModel:
    """Get traffic history."""
    # Get all traffic history
    traffic = filter_manager.get_traffic_history()
    
    # Apply filters
    if filter_type:
        traffic = [item for item in traffic if item["type"] == filter_type]
    
    if filter_path:
        traffic = [
            item for item in traffic 
            if (item["type"] == "request" and filter_path in item["path"]) or
               (item["type"] == "response" and filter_path in item["request_path"])
        ]
    
    if filter_method:
        traffic = [
            item for item in traffic 
            if (item["type"] == "request" and item["method"] == filter_method) or
               (item["type"] == "response" and item["request_method"] == filter_method)
        ]
    
    if filter_status:
        traffic = [
            item for item in traffic 
            if item["type"] == "response" and item["status_code"] == filter_status
        ]
    
    # Calculate pagination
    total = len(traffic)
    start = (page - 1) * page_size
    end = start + page_size
    paginated_traffic = traffic[start:end]
    
    # Convert to API model
    items = []
    for i, item in enumerate(paginated_traffic):
        if item["type"] == "request":
            items.append(TrafficItemModel(
                id=start + i,
                type=item["type"],
                method=item["method"],
                path=item["path"],
                headers=item["headers"],
                body=item["body"],
                timestamp=item["timestamp"],
                matched_rules=item.get("matched_rules"),
                filtered=item.get("filtered")
            ))
        else:  # response
            items.append(TrafficItemModel(
                id=start + i,
                type=item["type"],
                request_method=item["request_method"],
                request_path=item["request_path"],
                status_code=item["status_code"],
                headers=item["headers"],
                body=item["body"],
                timestamp=item["timestamp"],
                matched_rules=item.get("matched_rules"),
                filtered=item.get("filtered")
            ))
    
    return TrafficHistoryModel(
        items=items,
        total=total
    )

@router.post("/traffic/{traffic_id}/create-rule", response_model=FilterRuleModel)
async def create_rule_from_traffic(
    traffic_id: int = Path(..., description="ID of the traffic item"),
    name: Optional[str] = Query(None, description="Name for the new rule"),
    description: Optional[str] = Query(None, description="Description for the new rule"),
    filter_manager: FilterManager = Depends(get_filter_manager)
) -> FilterRuleModel:
    """Create a filter rule from a traffic history item."""
    try:
        # Get traffic history
        traffic = filter_manager.get_traffic_history()
        
        if traffic_id < 0 or traffic_id >= len(traffic):
            raise HTTPException(status_code=404, detail=f"Traffic item not found: {traffic_id}")
        
        traffic_item = traffic[traffic_id]
        
        # Create rule from traffic
        rule = filter_manager.create_rule_from_traffic(
            traffic_id,
            name or f"Rule from {traffic_item['type']} to {traffic_item.get('path', traffic_item.get('request_path', ''))}",
            description or f"Automatically generated from {traffic_item['type']} traffic"
        )
        
        # Add the rule
        success = await filter_manager.add_rule(rule)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create filter rule")
        
        # Convert to API model
        return FilterRuleModel(
            id=rule.id,
            name=rule.name,
            description=rule.description,
            conditions=[
                FilterConditionModel(
                    field=condition.field,
                    operator=condition.operator,
                    value=condition.value
                )
                for condition in rule.conditions
            ],
            enabled=rule.enabled,
            priority=rule.priority,
            tags=rule.tags
        )
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Traffic item not found: {traffic_id}")
    except Exception as e:
        logger.error(f"Error creating rule from traffic: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create rule: {str(e)}")

@router.get("/suggestions", response_model=FilterSuggestionsModel)
async def get_filter_suggestions(
    max_suggestions: int = Query(5, description="Maximum number of suggestions to return"),
    filter_manager: FilterManager = Depends(get_filter_manager)
) -> FilterSuggestionsModel:
    """Get filter suggestions based on traffic patterns."""
    # Get traffic history
    traffic = filter_manager.get_traffic_history()
    
    if not traffic:
        return FilterSuggestionsModel(suggestions=[])
    
    # Use the analyzer to suggest rules
    suggested_rules = filter_analyzer.suggest_filter_rules(traffic, max_rules=max_suggestions)
    
    # Convert to API model
    suggestions = []
    for rule in suggested_rules:
        # Find traffic items that would match this rule
        matching_traffic = []
        match_count = 0
        
        for item in traffic:
            # Convert traffic item to format expected by condition evaluator
            if item["type"] == "request":
                data = {
                    "method": item["method"],
                    "path": item["path"],
                    "headers": item["headers"],
                    "body": item["body"]
                }
            else:  # response
                data = {
                    "request_method": item["request_method"],
                    "request_path": item["request_path"],
                    "status_code": item["status_code"],
                    "headers": item["headers"],
                    "body": item["body"]
                }
            
            # Check if all conditions match
            all_match = True
            for condition in rule.conditions:
                if not filter_manager.condition_evaluator.evaluate(condition, data):
                    all_match = False
                    break
            
            if all_match:
                match_count += 1
                if len(matching_traffic) < 5:  # Limit to 5 samples
                    matching_traffic.append(item)
        
        suggestions.append(FilterSuggestionModel(
            rule=FilterRuleModel(
                id=rule.id,
                name=rule.name,
                description=rule.description,
                conditions=[
                    FilterConditionModel(
                        field=condition.field,
                        operator=condition.operator,
                        value=condition.value
                    )
                    for condition in rule.conditions
                ],
                enabled=rule.enabled,
                priority=rule.priority,
                tags=rule.tags
            ),
            match_count=match_count,
            sample_matches=matching_traffic
        ))
    
    return FilterSuggestionsModel(suggestions=suggestions)

@router.post("/preview", response_model=PreviewResultModel)
async def preview_filter_rule(
    rule_data: FilterRuleModel,
    filter_manager: FilterManager = Depends(get_filter_manager)
) -> PreviewResultModel:
    """Preview how a filter rule would affect traffic."""
    # Convert from API model to domain model
    rule = FilterRule(
        id=rule_data.id,
        name=rule_data.name,
        description=rule_data.description,
        conditions=[
            FilterCondition(
                field=condition.field,
                operator=condition.operator,
                value=condition.value
            )
            for condition in rule_data.conditions
        ],
        enabled=rule_data.enabled,
        priority=rule_data.priority,
        tags=rule_data.tags
    )
    
    # Get traffic history
    traffic = filter_manager.get_traffic_history()
    
    # Find traffic items that would match this rule
    matching_traffic = []
    
    for item in traffic:
        # Convert traffic item to format expected by condition evaluator
        if item["type"] == "request":
            data = {
                "method": item["method"],
                "path": item["path"],
                "headers": item["headers"],
                "body": item["body"]
            }
        else:  # response
            data = {
                "request_method": item["request_method"],
                "request_path": item["request_path"],
                "status_code": item["status_code"],
                "headers": item["headers"],
                "body": item["body"]
            }
        
        # Check if all conditions match
        all_match = True
        for condition in rule.conditions:
            if not filter_manager.condition_evaluator.evaluate(condition, data):
                all_match = False
                break
        
        if all_match:
            matching_traffic.append(item)
    
    return PreviewResultModel(
        would_match=len(matching_traffic) > 0,
        traffic_count=len(matching_traffic),
        sample_matches=matching_traffic[:5]  # Limit to 5 samples
    )

@router.post("/clear-traffic", response_model=Dict[str, bool])
async def clear_traffic_history(
    filter_manager: FilterManager = Depends(get_filter_manager)
) -> Dict[str, bool]:
    """Clear the traffic history."""
    filter_manager.clear_traffic_history()
    return {"success": True}

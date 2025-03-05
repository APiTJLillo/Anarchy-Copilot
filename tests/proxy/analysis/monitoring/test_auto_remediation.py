"""Tests for auto-remediation functionality."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import json
import tempfile
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from proxy.analysis.monitoring.auto_remediation import (
    RemediationAction,
    RemediationPlan,
    RemediationKnowledgeBase,
    AutoRemediation,
    suggest_remediation
)
from proxy.analysis.monitoring.alert_correlation import (
    AlertPattern,
    AlertCorrelation
)
from proxy.analysis.monitoring.alerts import AlertSeverity

@pytest.fixture
def sample_actions() -> List[RemediationAction]:
    """Create sample remediation actions."""
    return [
        RemediationAction(
            action_type="scale",
            target="web_server",
            parameters={"replicas": 3},
            priority=1,
            estimated_impact=0.8,
            risk_level="low",
            prerequisites=["cpu_below_80"],
            validation_steps=["check_redundancy"]
        ),
        RemediationAction(
            action_type="config",
            target="database",
            parameters={"max_connections": 1000},
            priority=2,
            estimated_impact=0.6,
            risk_level="medium",
            prerequisites=["memory_available"],
            validation_steps=["check_backup"]
        ),
        RemediationAction(
            action_type="restart",
            target="cache",
            parameters={},
            priority=3,
            estimated_impact=0.4,
            risk_level="high",
            prerequisites=["cpu_below_80", "memory_available"],
            validation_steps=["check_redundancy", "check_backup"]
        )
    ]

@pytest.fixture
def sample_patterns() -> List[AlertPattern]:
    """Create sample alert patterns."""
    return [
        AlertPattern(
            alerts=["alert_1", "alert_2"],
            confidence=0.8,
            support=0.7,
            time_window=timedelta(minutes=30),
            root_cause="High CPU usage",
            impact_score=0.9
        ),
        AlertPattern(
            alerts=["alert_3"],
            confidence=0.6,
            support=0.5,
            time_window=timedelta(minutes=15),
            root_cause="Memory leak",
            impact_score=0.7
        )
    ]

@pytest.fixture
def sample_correlations() -> List[AlertCorrelation]:
    """Create sample alert correlations."""
    return [
        AlertCorrelation(
            source_alert="alert_1",
            target_alert="alert_2",
            correlation_type="temporal",
            strength=0.9,
            time_lag=timedelta(minutes=5),
            description="Strong temporal correlation"
        ),
        AlertCorrelation(
            source_alert="alert_2",
            target_alert="alert_3",
            correlation_type="metric",
            strength=0.7,
            time_lag=None,
            description="Metric value correlation"
        )
    ]

@pytest.fixture
def kb_file() -> Path:
    """Create temporary knowledge base file."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        test_kb = {
            "solutions": {
                "cpu_usage_warning": [
                    {
                        "action_type": "scale",
                        "target": "web_server",
                        "parameters": {"replicas": 3},
                        "priority": 1,
                        "estimated_impact": 0.8,
                        "risk_level": "low",
                        "prerequisites": ["cpu_below_80"],
                        "validation_steps": ["check_redundancy"]
                    }
                ],
                "memory_leak": [
                    {
                        "action_type": "restart",
                        "target": "service",
                        "parameters": {},
                        "priority": 2,
                        "estimated_impact": 0.6,
                        "risk_level": "medium",
                        "prerequisites": ["memory_available"],
                        "validation_steps": ["check_backup"]
                    }
                ]
            },
            "history": [
                {
                    "timestamp": "2025-03-02T09:00:00",
                    "pattern": "alert_1,alert_2",
                    "action": {
                        "type": "scale",
                        "target": "web_server",
                        "parameters": {"replicas": 3}
                    },
                    "success": True,
                    "recovery_time": "timedelta(minutes=10)"
                }
            ]
        }
        json.dump(test_kb, f)
        return Path(f.name)

class TestRemediationKnowledgeBase:
    """Test remediation knowledge base functionality."""
    
    def test_load_kb(self, kb_file):
        """Test loading knowledge base from file."""
        kb = RemediationKnowledgeBase(kb_file)
        assert len(kb.solutions) == 2
        assert len(kb.success_history) == 1
        
        assert "cpu_usage_warning" in kb.solutions
        assert kb.solutions["cpu_usage_warning"][0]["action_type"] == "scale"
    
    def test_add_success_case(self, kb_file):
        """Test adding success case to history."""
        kb = RemediationKnowledgeBase(kb_file)
        initial_count = len(kb.success_history)
        
        action = RemediationAction(
            action_type="scale",
            target="web_server",
            parameters={"replicas": 3},
            priority=1,
            estimated_impact=0.8,
            risk_level="low",
            prerequisites=["cpu_below_80"],
            validation_steps=["check_redundancy"]
        )
        
        kb.add_success_case(
            pattern="test_pattern",
            action=action,
            outcome=True,
            recovery_time=timedelta(minutes=5)
        )
        
        assert len(kb.success_history) == initial_count + 1
        assert kb.success_history[-1]["success"] is True
    
    def test_get_solution(self, kb_file):
        """Test getting solutions for issues."""
        kb = RemediationKnowledgeBase(kb_file)
        solutions = kb.get_solution(
            metric="cpu_usage",
            severity=AlertSeverity.WARNING,
            contributing_factors=[("memory_leak", 0.8)]
        )
        
        assert len(solutions) > 0
        assert isinstance(solutions[0], RemediationAction)

class TestAutoRemediation:
    """Test auto-remediation system."""
    
    def test_predictor_training(self, kb_file):
        """Test training success predictor."""
        remediator = AutoRemediation(
            kb=RemediationKnowledgeBase(kb_file)
        )
        assert remediator.success_predictor is not None
    
    def test_success_prediction(self, kb_file, sample_actions):
        """Test predicting action success."""
        remediator = AutoRemediation(
            kb=RemediationKnowledgeBase(kb_file)
        )
        
        prob = remediator.predict_success(
            pattern="alert_1,alert_2",
            action=sample_actions[0]
        )
        assert 0 <= prob <= 1
    
    def test_plan_generation(
        self,
        kb_file,
        sample_patterns,
        sample_correlations
    ):
        """Test remediation plan generation."""
        remediator = AutoRemediation(
            kb=RemediationKnowledgeBase(kb_file)
        )
        
        plan = remediator.generate_plan(
            patterns=sample_patterns,
            correlations=sample_correlations
        )
        
        assert isinstance(plan, RemediationPlan)
        assert len(plan.actions) > 0
        assert len(plan.root_causes) > 0
        assert isinstance(plan.estimated_recovery_time, timedelta)
    
    def test_action_validation(self, kb_file, sample_actions):
        """Test action validation."""
        remediator = AutoRemediation(
            kb=RemediationKnowledgeBase(kb_file)
        )
        
        # Test with valid metrics
        valid_metrics = {
            "cpu_usage": 70,
            "memory_usage": 80,
            "replica_count": 2,
            "backup_status": 1
        }
        
        assert remediator.validate_action(
            sample_actions[0],
            valid_metrics
        )
        
        # Test with invalid metrics
        invalid_metrics = {
            "cpu_usage": 90,
            "memory_usage": 95,
            "replica_count": 1,
            "backup_status": 0
        }
        
        assert not remediator.validate_action(
            sample_actions[2],  # High-risk action
            invalid_metrics
        )

@pytest.mark.integration
class TestRemediationIntegration:
    """Integration tests for remediation system."""
    
    async def test_full_remediation_flow(
        self,
        kb_file,
        sample_patterns,
        sample_correlations
    ):
        """Test complete remediation workflow."""
        # Setup mocked metrics collector
        mock_metrics = {
            "cpu_usage": 75,
            "memory_usage": 85,
            "replica_count": 2,
            "backup_status": 1
        }
        
        # Generate remediation plan
        plan = suggest_remediation(
            patterns=sample_patterns,
            correlations=sample_correlations,
            current_metrics=mock_metrics
        )
        
        assert isinstance(plan, RemediationPlan)
        assert plan.incident_id
        assert len(plan.actions) > 0
        
        # Verify actions are validated
        for action in plan.actions:
            assert action.risk_level in ["low", "medium", "high"]
            if action.risk_level == "high":
                assert len(action.validation_steps) > 1
    
    async def test_concurrent_remediation(
        self,
        kb_file,
        sample_patterns,
        sample_correlations
    ):
        """Test handling multiple concurrent remediations."""
        # Generate multiple plans concurrently
        async def generate_plan():
            return suggest_remediation(
                patterns=sample_patterns,
                correlations=sample_correlations,
                current_metrics={
                    "cpu_usage": np.random.randint(60, 90),
                    "memory_usage": np.random.randint(70, 95),
                    "replica_count": 2,
                    "backup_status": 1
                }
            )
        
        plans = await asyncio.gather(*[
            generate_plan()
            for _ in range(5)
        ])
        
        # Verify plans are unique
        plan_ids = set(plan.incident_id for plan in plans)
        assert len(plan_ids) == 5
    
    async def test_error_handling(
        self,
        kb_file,
        sample_patterns,
        sample_correlations
    ):
        """Test error handling in remediation process."""
        # Test with invalid metrics
        with pytest.raises(ValueError):
            suggest_remediation(
                patterns=sample_patterns,
                correlations=sample_correlations,
                current_metrics={"invalid": "metrics"}
            )
        
        # Test with empty patterns
        plan = suggest_remediation(
            patterns=[],
            correlations=sample_correlations,
            current_metrics={"cpu_usage": 75}
        )
        assert not plan.actions

def test_remediation_safety(kb_file, sample_actions):
    """Test remediation safety mechanisms."""
    remediator = AutoRemediation(
        kb=RemediationKnowledgeBase(kb_file)
    )
    
    # Test high-risk action safety
    high_risk_action = sample_actions[2]  # restart action
    
    # Should fail without all prerequisites
    assert not remediator.validate_action(
        high_risk_action,
        {"cpu_usage": 70}  # Missing memory check
    )
    
    # Should fail without all validations
    assert not remediator.validate_action(
        high_risk_action,
        {
            "cpu_usage": 70,
            "memory_usage": 80,
            "replica_count": 1  # Missing backup
        }
    )
    
    # Should pass with all requirements
    assert remediator.validate_action(
        high_risk_action,
        {
            "cpu_usage": 70,
            "memory_usage": 80,
            "replica_count": 2,
            "backup_status": 1
        }
    )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

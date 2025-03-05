"""Validation and context management for LLM responses."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from enum import Enum
import json
import logging
import re
from pathlib import Path

from .test_costbenefit_llm import (
    LLMEnhancedGenerator,
    LLMConfig,
    PreferenceExplanation
)

class ValidationLevel(Enum):
    """Validation strictness levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    STRICT = "strict"

class ValidationResult(Enum):
    """Validation result states."""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"

@dataclass
class ValidationConfig:
    """Configuration for response validation."""
    level: ValidationLevel = ValidationLevel.STANDARD
    max_length: int = 2000
    min_length: int = 50
    required_elements: Set[str] = field(default_factory=lambda: {
        "features",
        "rules",
        "confidence"
    })
    style_rules: Dict[str, Any] = field(default_factory=lambda: {
        "sentence_case": True,
        "no_repetition": True,
        "proper_punctuation": True
    })
    sentiment_bounds: Tuple[float, float] = (-0.3, 0.3)
    max_repair_attempts: int = 3
    repair_timeout: float = 5.0
    enable_semantic_validation: bool = True
    semantic_similarity_threshold: float = 0.7

@dataclass
class ContextConfig:
    """Configuration for context management."""
    max_history: int = 5
    context_window: int = 3
    relevance_threshold: float = 0.5
    enable_memory: bool = True
    memory_path: Optional[str] = "context_memory"
    compression_enabled: bool = True
    compression_ratio: float = 0.5
    merge_similar: bool = True
    similarity_threshold: float = 0.8

class ResponseValidator:
    """Validate and repair LLM responses."""

    def __init__(
        self,
        config: ValidationConfig
    ):
        self.config = config
        self.validators = {
            ValidationLevel.MINIMAL: self._validate_minimal,
            ValidationLevel.STANDARD: self._validate_standard,
            ValidationLevel.STRICT: self._validate_strict
        }
        
        # Initialize NLP tools
        self._setup_nlp()

    def _setup_nlp(self) -> None:
        """Setup NLP tools for validation."""
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except ImportError:
            logging.warning("Spacy not installed - using basic validation")
            self.nlp = None

    async def validate_response(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> Tuple[ValidationResult, Optional[str]]:
        """Validate LLM response."""
        try:
            # Run level-specific validation
            validator = self.validators[self.config.level]
            result = await validator(response, context)
            
            if result != ValidationResult.PASS:
                # Attempt repair if validation fails
                fixed_response = await self._repair_response(
                    response,
                    context,
                    result
                )
                if fixed_response:
                    # Validate repaired response
                    new_result = await validator(fixed_response, context)
                    if new_result == ValidationResult.PASS:
                        return ValidationResult.PASS, fixed_response
                    return new_result, None
            
            return result, response if result == ValidationResult.PASS else None
            
        except Exception as e:
            logging.error(f"Validation error: {e}")
            return ValidationResult.FAIL, None

    async def _validate_minimal(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> ValidationResult:
        """Perform minimal validation checks."""
        # Check length bounds
        if not self.config.min_length <= len(response) <= self.config.max_length:
            return ValidationResult.FAIL
        
        # Check for required elements
        for element in self.config.required_elements:
            if element.lower() not in response.lower():
                return ValidationResult.WARN
        
        return ValidationResult.PASS

    async def _validate_standard(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> ValidationResult:
        """Perform standard validation checks."""
        # Run minimal checks first
        minimal_result = await self._validate_minimal(response, context)
        if minimal_result == ValidationResult.FAIL:
            return minimal_result
        
        # Check style rules
        if self.config.style_rules["sentence_case"]:
            sentences = self._get_sentences(response)
            if not all(s[0].isupper() for s in sentences if s):
                return ValidationResult.WARN
        
        if self.config.style_rules["no_repetition"]:
            if self._has_repetition(response):
                return ValidationResult.WARN
        
        if self.config.style_rules["proper_punctuation"]:
            if not self._check_punctuation(response):
                return ValidationResult.WARN
        
        # Check sentiment if enabled
        if self.nlp and self.config.sentiment_bounds:
            sentiment = self._analyze_sentiment(response)
            if not (
                self.config.sentiment_bounds[0] <=
                sentiment <=
                self.config.sentiment_bounds[1]
            ):
                return ValidationResult.WARN
        
        return ValidationResult.PASS

    async def _validate_strict(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> ValidationResult:
        """Perform strict validation checks."""
        # Run standard checks first
        standard_result = await self._validate_standard(response, context)
        if standard_result != ValidationResult.PASS:
            return standard_result
        
        # Check semantic similarity if enabled
        if (
            self.config.enable_semantic_validation and
            self.nlp and
            "original_text" in context
        ):
            similarity = self._calculate_similarity(
                response,
                context["original_text"]
            )
            if similarity < self.config.semantic_similarity_threshold:
                return ValidationResult.FAIL
        
        return ValidationResult.PASS

    def _get_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        return [s.strip() for s in text.split(".") if s.strip()]

    def _has_repetition(self, text: str) -> bool:
        """Check for obvious repetition."""
        sentences = self._get_sentences(text)
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                if self._calculate_similarity(sentences[i], sentences[j]) > 0.8:
                    return True
        return False

    def _check_punctuation(self, text: str) -> bool:
        """Check for proper punctuation."""
        sentences = self._get_sentences(text)
        for sentence in sentences:
            if not sentence.endswith((".", "!", "?")):
                return False
        return True

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze text sentiment."""
        if not self.nlp:
            return 0.0
        doc = self.nlp(text)
        return sum(token.sentiment for token in doc) / len(doc)

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts."""
        if not self.nlp:
            return 0.0
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        return doc1.similarity(doc2)

    async def _repair_response(
        self,
        response: str,
        context: Dict[str, Any],
        validation_result: ValidationResult
    ) -> Optional[str]:
        """Attempt to repair invalid response."""
        for attempt in range(self.config.max_repair_attempts):
            try:
                # Apply repairs based on validation level
                if validation_result == ValidationResult.WARN:
                    repaired = await self._apply_style_repairs(response)
                else:
                    repaired = await self._apply_content_repairs(
                        response,
                        context
                    )
                
                if repaired != response:
                    return repaired
                
            except Exception as e:
                logging.error(f"Repair error: {e}")
                continue
        
        return None

    async def _apply_style_repairs(self, text: str) -> str:
        """Apply style-based repairs."""
        # Fix sentence case
        sentences = self._get_sentences(text)
        fixed_sentences = [
            s[0].upper() + s[1:] if s else s
            for s in sentences
        ]
        
        # Remove repetition
        unique_sentences = []
        for sentence in fixed_sentences:
            if not any(
                self._calculate_similarity(sentence, existing) > 0.8
                for existing in unique_sentences
            ):
                unique_sentences.append(sentence)
        
        # Fix punctuation
        final_sentences = [
            s + "." if not s.endswith((".", "!", "?")) else s
            for s in unique_sentences
        ]
        
        return " ".join(final_sentences)

    async def _apply_content_repairs(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> str:
        """Apply content-based repairs."""
        # Ensure required elements
        fixed_text = text
        for element in self.config.required_elements:
            if element.lower() not in text.lower():
                if element in context:
                    fixed_text += f"\n{element}: {context[element]}"
        
        return fixed_text

class ContextManager:
    """Manage context for LLM interactions."""

    def __init__(
        self,
        config: ContextConfig
    ):
        self.config = config
        self.history: List[Dict[str, Any]] = []
        self.memory: Dict[str, Any] = {}
        
        # Load memory
        self._load_memory()

    def add_context(
        self,
        context: Dict[str, Any],
        explanation: PreferenceExplanation
    ) -> Dict[str, Any]:
        """Add new context and get relevant history."""
        # Add to history
        entry = {
            "context": context,
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(entry)
        
        # Trim history if needed
        if len(self.history) > self.config.max_history:
            self.history = self.history[-self.config.max_history:]
        
        # Get relevant context
        relevant = self._get_relevant_context(explanation)
        
        # Update memory
        if self.config.enable_memory:
            self._update_memory(entry)
        
        return relevant

    def _get_relevant_context(
        self,
        explanation: PreferenceExplanation
    ) -> Dict[str, Any]:
        """Get relevant historical context."""
        if not self.history:
            return {}
        
        relevant = []
        for entry in reversed(self.history[-self.config.context_window:]):
            similarity = self._calculate_relevance(
                explanation,
                entry["explanation"]
            )
            if similarity >= self.config.relevance_threshold:
                relevant.append(entry)
        
        # Merge similar contexts if enabled
        if self.config.merge_similar:
            relevant = self._merge_similar_contexts(relevant)
        
        # Compress if enabled
        if (
            self.config.compression_enabled and
            len(relevant) > 1
        ):
            relevant = self._compress_contexts(
                relevant,
                self.config.compression_ratio
            )
        
        return self._combine_contexts(relevant)

    def _calculate_relevance(
        self,
        current: PreferenceExplanation,
        historical: PreferenceExplanation
    ) -> float:
        """Calculate relevance between explanations."""
        # Compare feature importance
        feature_similarity = self._compare_features(
            current.feature_importance,
            historical.feature_importance
        )
        
        # Compare rules
        rule_similarity = self._compare_rules(
            current.rules,
            historical.rules
        )
        
        return 0.7 * feature_similarity + 0.3 * rule_similarity

    def _compare_features(
        self,
        current: Dict[str, float],
        historical: Dict[str, float]
    ) -> float:
        """Compare feature importance distributions."""
        all_features = set(current) | set(historical)
        if not all_features:
            return 0.0
        
        similarity = 0.0
        for feature in all_features:
            curr_val = current.get(feature, 0.0)
            hist_val = historical.get(feature, 0.0)
            similarity += 1.0 - abs(curr_val - hist_val)
        
        return similarity / len(all_features)

    def _compare_rules(
        self,
        current: List[str],
        historical: List[str]
    ) -> float:
        """Compare decision rules."""
        if not current or not historical:
            return 0.0
        
        matches = sum(
            1 for rule in current
            if any(self._rules_match(rule, hist_rule) for hist_rule in historical)
        )
        
        return matches / max(len(current), len(historical))

    def _rules_match(self, rule1: str, rule2: str) -> bool:
        """Check if rules are semantically similar."""
        # Basic string similarity for now
        # Could be enhanced with proper parsing
        return (
            rule1.lower().replace(" ", "") ==
            rule2.lower().replace(" ", "")
        )

    def _merge_similar_contexts(
        self,
        contexts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge similar context entries."""
        merged = []
        used = set()
        
        for i, ctx1 in enumerate(contexts):
            if i in used:
                continue
            
            merged_ctx = ctx1.copy()
            for j, ctx2 in enumerate(contexts[i+1:], i+1):
                if j in used:
                    continue
                
                similarity = self._calculate_relevance(
                    ctx1["explanation"],
                    ctx2["explanation"]
                )
                
                if similarity >= self.config.similarity_threshold:
                    # Merge contexts
                    merged_ctx = self._combine_contexts([merged_ctx, ctx2])
                    used.add(j)
            
            merged.append(merged_ctx)
        
        return merged

    def _compress_contexts(
        self,
        contexts: List[Dict[str, Any]],
        ratio: float
    ) -> List[Dict[str, Any]]:
        """Compress context list."""
        n = max(1, int(len(contexts) * ratio))
        return contexts[:n]

    def _combine_contexts(
        self,
        contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Combine multiple contexts."""
        if not contexts:
            return {}
        
        combined = {
            "features": {},
            "rules": set(),
            "examples": []
        }
        
        for ctx in contexts:
            # Merge features
            for feature, importance in ctx["explanation"].feature_importance.items():
                if feature in combined["features"]:
                    combined["features"][feature] = max(
                        combined["features"][feature],
                        importance
                    )
                else:
                    combined["features"][feature] = importance
            
            # Merge rules
            combined["rules"].update(ctx["explanation"].rules)
            
            # Merge examples
            if hasattr(ctx["explanation"], "counterfactuals"):
                combined["examples"].extend(ctx["explanation"].counterfactuals)
        
        # Convert sets back to lists
        combined["rules"] = list(combined["rules"])
        
        return combined

    def _load_memory(self) -> None:
        """Load context memory from disk."""
        if not self.config.enable_memory or not self.config.memory_path:
            return
        
        try:
            memory_file = Path(self.config.memory_path) / "context_memory.json"
            if memory_file.exists():
                with open(memory_file, 'r') as f:
                    self.memory = json.load(f)
        except Exception as e:
            logging.error(f"Error loading memory: {e}")

    def _update_memory(self, entry: Dict[str, Any]) -> None:
        """Update context memory."""
        if not self.config.enable_memory:
            return
        
        try:
            # Update memory
            key = self._get_memory_key(entry["explanation"])
            self.memory[key] = {
                "frequency": self.memory.get(key, {}).get("frequency", 0) + 1,
                "last_seen": datetime.now().isoformat(),
                "context": entry["context"]
            }
            
            # Save to disk
            if self.config.memory_path:
                memory_file = Path(self.config.memory_path) / "context_memory.json"
                memory_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(memory_file, 'w') as f:
                    json.dump(self.memory, f, indent=2)
                    
        except Exception as e:
            logging.error(f"Error updating memory: {e}")

    def _get_memory_key(self, explanation: PreferenceExplanation) -> str:
        """Generate memory key from explanation."""
        return f"{hash(str(explanation.feature_importance))}_{hash(str(explanation.rules))}"

@pytest.fixture
def validator():
    """Create response validator for testing."""
    config = ValidationConfig()
    return ResponseValidator(config)

@pytest.fixture
def context_manager():
    """Create context manager for testing."""
    config = ContextConfig()
    return ContextManager(config)

@pytest.mark.asyncio
async def test_response_validation(validator):
    """Test response validation."""
    context = {"original_text": "Test response about features."}
    
    # Test valid response
    response = "The response discusses Features and Rules with proper formatting."
    result, validated = await validator.validate_response(response, context)
    assert result == ValidationResult.PASS
    assert validated == response
    
    # Test invalid response
    response = "x" * 3000  # Too long
    result, validated = await validator.validate_response(response, context)
    assert result == ValidationResult.FAIL
    assert validated is None

@pytest.mark.asyncio
async def test_context_management(context_manager):
    """Test context management."""
    explanation1 = PreferenceExplanation(
        feature_importance={"response_time": 0.8},
        rules=["response_time > 100"],
        counterfactuals=[],
        shap_values=None,
        confidence=0.9
    )
    
    explanation2 = PreferenceExplanation(
        feature_importance={"response_time": 0.7},
        rules=["response_time > 90"],
        counterfactuals=[],
        shap_values=None,
        confidence=0.8
    )
    
    # Add contexts
    context1 = {"test": "context1"}
    relevant1 = context_manager.add_context(context1, explanation1)
    assert len(context_manager.history) == 1
    
    context2 = {"test": "context2"}
    relevant2 = context_manager.add_context(context2, explanation2)
    assert len(context_manager.history) == 2
    
    # Verify relevance calculation
    assert context_manager._calculate_relevance(explanation1, explanation2) > 0.5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

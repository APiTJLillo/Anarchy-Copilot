"""Language model integration for natural language generation."""

import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import logging
import json
from pathlib import Path

from .test_costbenefit_nlg import (
    NaturalLanguageGenerator,
    NLGConfig,
    PreferenceExplanation
)

@dataclass
class LLMConfig:
    """Configuration for language model integration."""
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 500
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    enable_streaming: bool = True
    cache_responses: bool = True
    cache_path: Optional[str] = "llm_cache"
    retry_attempts: int = 3
    timeout: float = 10.0
    prompt_templates: Dict[str, str] = field(default_factory=lambda: {
        "explanation": """
        Given the following preference decision data, generate a natural explanation:
        Confidence: {confidence}
        Important Features: {features}
        Rules: {rules}
        Examples: {examples}
        
        Response should be:
        - Clear and concise
        - Natural sounding
        - Well-structured
        - Focused on key insights
        """,
        "summary": """
        Summarize the key points of this preference decision:
        {explanation}
        
        Focus on:
        - Main factors
        - Critical thresholds
        - Important trade-offs
        """,
        "comparison": """
        Compare this preference decision to the previous one:
        Current: {current}
        Previous: {previous}
        
        Highlight:
        - Key differences
        - Changing factors
        - Trend implications
        """
    })

class LLMEnhancedGenerator:
    """Enhanced natural language generation with LLM integration."""

    def __init__(
        self,
        base_generator: NaturalLanguageGenerator,
        config: LLMConfig
    ):
        self.base_generator = base_generator
        self.config = config
        
        # State
        self.response_cache: Dict[str, str] = {}
        self.llm_client = None
        
        # Load cache
        self._load_cache()
        
        # Initialize client
        self._setup_client()

    def _setup_client(self) -> None:
        """Setup LLM client."""
        try:
            import openai
            self.llm_client = openai.AsyncOpenAI()
        except ImportError:
            logging.warning("OpenAI package not installed - using fallback generation")
            self.llm_client = None

    async def generate_enhanced_explanation(
        self,
        explanation: PreferenceExplanation
    ) -> str:
        """Generate enhanced natural language explanation."""
        # Get base explanation
        base_text = self.base_generator.generate_explanation(explanation)
        
        if not self.llm_client:
            return base_text
        
        try:
            # Check cache
            cache_key = self._get_cache_key(explanation)
            if cache_key in self.response_cache:
                return self.response_cache[cache_key]
            
            # Prepare prompt
            prompt = self._prepare_prompt(
                "explanation",
                explanation,
                base_text
            )
            
            # Get LLM response
            response = await self._get_llm_response(prompt)
            
            # Cache response
            if self.config.cache_responses:
                self.response_cache[cache_key] = response
                await self._save_cache()
            
            return response
            
        except Exception as e:
            logging.error(f"Error in enhanced generation: {e}")
            return base_text

    async def generate_summary(
        self,
        explanation: PreferenceExplanation
    ) -> str:
        """Generate concise summary of explanation."""
        base_text = self.base_generator.generate_explanation(explanation)
        
        if not self.llm_client:
            return self._fallback_summary(base_text)
        
        try:
            prompt = self._prepare_prompt(
                "summary",
                explanation,
                base_text
            )
            return await self._get_llm_response(prompt)
            
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            return self._fallback_summary(base_text)

    async def generate_comparison(
        self,
        current: PreferenceExplanation,
        previous: PreferenceExplanation
    ) -> str:
        """Generate comparison between explanations."""
        if not self.llm_client:
            return self._fallback_comparison(current, previous)
        
        try:
            current_text = self.base_generator.generate_explanation(current)
            previous_text = self.base_generator.generate_explanation(previous)
            
            prompt = self.config.prompt_templates["comparison"].format(
                current=current_text,
                previous=previous_text
            )
            
            return await self._get_llm_response(prompt)
            
        except Exception as e:
            logging.error(f"Error generating comparison: {e}")
            return self._fallback_comparison(current, previous)

    def _prepare_prompt(
        self,
        template_name: str,
        explanation: PreferenceExplanation,
        base_text: Optional[str] = None
    ) -> str:
        """Prepare prompt for LLM."""
        template = self.config.prompt_templates[template_name]
        
        # Format feature importance
        features = [
            f"{feature}: {importance:.2%}"
            for feature, importance in explanation.feature_importance.items()
        ]
        
        context = {
            "confidence": explanation.confidence,
            "features": "\n".join(features),
            "rules": "\n".join(explanation.rules),
            "examples": base_text if base_text else ""
        }
        
        return template.format(**context)

    async def _get_llm_response(self, prompt: str) -> str:
        """Get response from language model."""
        for attempt in range(self.config.retry_attempts):
            try:
                response = await asyncio.wait_for(
                    self.llm_client.chat.completions.create(
                        model=self.config.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        top_p=self.config.top_p,
                        frequency_penalty=self.config.frequency_penalty,
                        presence_penalty=self.config.presence_penalty,
                        stream=self.config.enable_streaming
                    ),
                    timeout=self.config.timeout
                )
                
                if self.config.enable_streaming:
                    text = ""
                    async for chunk in response:
                        if chunk.choices[0].delta.content:
                            text += chunk.choices[0].delta.content
                    return text
                else:
                    return response.choices[0].message.content
                
            except asyncio.TimeoutError:
                logging.warning(f"LLM request timeout (attempt {attempt + 1})")
                continue
            except Exception as e:
                logging.error(f"LLM request error: {e}")
                if attempt == self.config.retry_attempts - 1:
                    raise
                continue

    def _fallback_summary(self, text: str) -> str:
        """Generate fallback summary."""
        # Simple extractive summary
        sentences = text.split(". ")
        if len(sentences) <= 3:
            return text
        
        # Take first sentence and two most important ones based on feature mentions
        feature_sentences = []
        for sentence in sentences[1:]:
            feature_count = sum(
                1 for feature in self.base_generator.learner.adapter.config.objectives
                if feature.name.lower() in sentence.lower()
            )
            feature_sentences.append((sentence, feature_count))
        
        important_sentences = sorted(
            feature_sentences,
            key=lambda x: x[1],
            reverse=True
        )[:2]
        
        return ". ".join([
            sentences[0],
            *[s[0] for s in important_sentences]
        ]) + "."

    def _fallback_comparison(
        self,
        current: PreferenceExplanation,
        previous: PreferenceExplanation
    ) -> str:
        """Generate fallback comparison."""
        changes = []
        
        # Compare feature importance
        for feature in set(current.feature_importance) | set(previous.feature_importance):
            curr_imp = current.feature_importance.get(feature, 0)
            prev_imp = previous.feature_importance.get(feature, 0)
            
            if abs(curr_imp - prev_imp) > 0.1:
                change = "increased" if curr_imp > prev_imp else "decreased"
                changes.append(
                    f"{feature} importance {change} "
                    f"from {prev_imp:.1%} to {curr_imp:.1%}"
                )
        
        if not changes:
            return "No significant changes in preferences."
        
        return "Key changes:\n- " + "\n- ".join(changes)

    def _get_cache_key(self, explanation: PreferenceExplanation) -> str:
        """Generate cache key for explanation."""
        return f"{hash(str(explanation.feature_importance))}_{hash(str(explanation.rules))}"

    def _load_cache(self) -> None:
        """Load response cache from disk."""
        if not self.config.cache_path:
            return
        
        try:
            cache_file = Path(self.config.cache_path) / "responses.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    self.response_cache = json.load(f)
        except Exception as e:
            logging.error(f"Error loading cache: {e}")

    async def _save_cache(self) -> None:
        """Save response cache to disk."""
        if not self.config.cache_path:
            return
        
        try:
            cache_file = Path(self.config.cache_path) / "responses.json"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_file, 'w') as f:
                json.dump(self.response_cache, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving cache: {e}")

@pytest.fixture
def llm_generator(natural_language_generator):
    """Create LLM-enhanced generator for testing."""
    config = LLMConfig()
    return LLMEnhancedGenerator(natural_language_generator, config)

@pytest.mark.asyncio
async def test_enhanced_explanation(llm_generator):
    """Test enhanced explanation generation."""
    explanation = PreferenceExplanation(
        feature_importance={"response_time": 0.8, "memory_usage": 0.2},
        rules=["response_time > 100"],
        counterfactuals=[{"response_time": 50.0}],
        shap_values=None,
        confidence=0.9
    )
    
    text = await llm_generator.generate_enhanced_explanation(explanation)
    
    assert text
    assert isinstance(text, str)
    assert len(text) > 0

@pytest.mark.asyncio
async def test_fallbacks(llm_generator):
    """Test fallback generation methods."""
    # Test summary fallback
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    summary = llm_generator._fallback_summary(text)
    assert isinstance(summary, str)
    assert len(summary.split(". ")) <= 3
    
    # Test comparison fallback
    current = PreferenceExplanation(
        feature_importance={"response_time": 0.8},
        rules=[],
        counterfactuals=[],
        shap_values=None,
        confidence=0.9
    )
    previous = PreferenceExplanation(
        feature_importance={"response_time": 0.4},
        rules=[],
        counterfactuals=[],
        shap_values=None,
        confidence=0.9
    )
    
    comparison = llm_generator._fallback_comparison(current, previous)
    assert isinstance(comparison, str)
    assert "changes" in comparison.lower()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

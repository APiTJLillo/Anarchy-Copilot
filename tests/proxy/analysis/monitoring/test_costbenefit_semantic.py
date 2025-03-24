"""Advanced semantic analysis for LLM responses."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cosine
import torch
from transformers import AutoTokenizer, AutoModel
import logging
import json
from pathlib import Path

from .test_costbenefit_llm_validation import (
    ResponseValidator,
    ContextManager,
    ValidationConfig,
    ContextConfig
)

@dataclass
class SemanticConfig:
    """Configuration for semantic analysis."""
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    cache_embeddings: bool = True
    cache_path: Optional[str] = "semantic_cache"
    batch_size: int = 32
    max_length: int = 512
    use_gpu: bool = True
    pooling_strategy: str = "mean"  # mean, max, cls
    threshold_high: float = 0.8
    threshold_medium: float = 0.6
    contextual_window: int = 3
    enable_concepts: bool = True
    concept_extraction: bool = True
    concept_linking: bool = True
    min_concept_freq: int = 2
    semantic_metrics: List[str] = field(default_factory=lambda: [
        "similarity",
        "entailment",
        "coherence",
        "consistency"
    ])

class SemanticAnalyzer:
    """Enhanced semantic analysis for responses and context."""

    def __init__(
        self,
        config: SemanticConfig
    ):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.use_gpu else "cpu"
        )
        
        # Initialize models
        self.tokenizer = None
        self.model = None
        self.concept_cache: Dict[str, torch.Tensor] = {}
        self.embedding_cache: Dict[str, torch.Tensor] = {}
        
        # Load cache
        self._load_cache()
        
        # Initialize models lazily
        self._setup_ready = False

    async def setup(self) -> None:
        """Lazy initialization of models."""
        if not self._setup_ready:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name
                )
                self.model = AutoModel.from_pretrained(
                    self.config.model_name
                ).to(self.device)
                self._setup_ready = True
            except Exception as e:
                logging.error(f"Error initializing semantic models: {e}")

    async def analyze_response(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform semantic analysis of response."""
        if not self._setup_ready:
            await self.setup()
        
        results = {}
        
        try:
            # Get embeddings
            response_embedding = await self._get_embedding(response)
            context_embedding = await self._get_embedding(
                context.get("original_text", "")
            )
            
            # Calculate metrics
            results["similarity"] = self._calculate_similarity(
                response_embedding,
                context_embedding
            )
            
            if "entailment" in self.config.semantic_metrics:
                results["entailment"] = await self._check_entailment(
                    response,
                    context
                )
            
            if "coherence" in self.config.semantic_metrics:
                results["coherence"] = await self._measure_coherence(response)
            
            if "consistency" in self.config.semantic_metrics:
                results["consistency"] = await self._check_consistency(
                    response,
                    context
                )
            
            # Extract concepts if enabled
            if self.config.concept_extraction:
                results["concepts"] = await self._extract_concepts(response)
                
                if self.config.concept_linking:
                    results["concept_links"] = await self._link_concepts(
                        results["concepts"],
                        context
                    )
            
            return results
            
        except Exception as e:
            logging.error(f"Error in semantic analysis: {e}")
            return {"error": str(e)}

    async def enhance_context(
        self,
        context: Dict[str, Any],
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Enhance context with semantic information."""
        if not self._setup_ready:
            await self.setup()
        
        enhanced = context.copy()
        
        try:
            # Get embeddings for current context
            current_embedding = await self._get_embedding(
                context.get("text", "")
            )
            
            # Find semantically relevant historical context
            relevant = await self._find_relevant_context(
                current_embedding,
                history
            )
            
            # Extract and merge concepts
            if self.config.concept_extraction:
                current_concepts = await self._extract_concepts(
                    context.get("text", "")
                )
                historical_concepts = await self._extract_concepts_from_history(
                    relevant
                )
                
                enhanced["concepts"] = self._merge_concepts(
                    current_concepts,
                    historical_concepts
                )
            
            # Add semantic metrics
            enhanced["semantic_context"] = {
                "relevance_scores": relevant["scores"],
                "concept_frequency": self._get_concept_frequency(
                    enhanced.get("concepts", [])
                ),
                "context_coherence": await self._measure_context_coherence(
                    context,
                    relevant
                )
            }
            
            return enhanced
            
        except Exception as e:
            logging.error(f"Error enhancing context: {e}")
            return context

    async def _get_embedding(
        self,
        text: str,
        use_cache: bool = True
    ) -> torch.Tensor:
        """Get text embedding using model."""
        if not text:
            return torch.zeros(self.model.config.hidden_size)
        
        # Check cache
        cache_key = hash(text)
        if use_cache and self.config.cache_embeddings:
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]
        
        # Tokenize and encode
        inputs = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Pool embeddings
        if self.config.pooling_strategy == "mean":
            embedding = torch.mean(outputs.last_hidden_state, dim=1)
        elif self.config.pooling_strategy == "max":
            embedding = torch.max(outputs.last_hidden_state, dim=1)[0]
        else:  # cls
            embedding = outputs.last_hidden_state[:, 0]
        
        # Cache result
        if use_cache and self.config.cache_embeddings:
            self.embedding_cache[cache_key] = embedding
            await self._save_cache()
        
        return embedding

    def _calculate_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor
    ) -> float:
        """Calculate cosine similarity between embeddings."""
        if embedding1.shape != embedding2.shape:
            return 0.0
        
        similarity = 1 - cosine(
            embedding1.cpu().numpy(),
            embedding2.cpu().numpy()
        )
        return float(similarity)

    async def _check_entailment(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> float:
        """Check if text entails context."""
        # Simple approximation using similarity
        text_embedding = await self._get_embedding(text)
        context_embedding = await self._get_embedding(
            context.get("original_text", "")
        )
        
        return max(
            0.0,
            self._calculate_similarity(text_embedding, context_embedding) - 0.5
        ) * 2

    async def _measure_coherence(self, text: str) -> float:
        """Measure internal coherence of text."""
        sentences = text.split(".")
        if len(sentences) < 2:
            return 1.0
        
        # Calculate pairwise similarities between consecutive sentences
        similarities = []
        for i in range(len(sentences) - 1):
            emb1 = await self._get_embedding(sentences[i])
            emb2 = await self._get_embedding(sentences[i + 1])
            similarities.append(self._calculate_similarity(emb1, emb2))
        
        return float(np.mean(similarities))

    async def _check_consistency(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> float:
        """Check consistency with context."""
        if not context:
            return 1.0
        
        # Compare with all context elements
        consistencies = []
        for key, value in context.items():
            if isinstance(value, str):
                text_emb = await self._get_embedding(text)
                ctx_emb = await self._get_embedding(value)
                consistency = self._calculate_similarity(text_emb, ctx_emb)
                consistencies.append(consistency)
        
        return float(np.mean(consistencies)) if consistencies else 1.0

    async def _extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Extract key concepts from text."""
        if not text:
            return []
        
        concepts = []
        
        # Use model for concept extraction
        inputs = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Extract noun phrases as concepts
        if self.config.concept_extraction:
            try:
                import spacy
                nlp = spacy.load("en_core_web_sm")
                doc = nlp(text)
                
                for chunk in doc.noun_chunks:
                    concept = {
                        "text": chunk.text,
                        "root": chunk.root.text,
                        "embedding": await self._get_embedding(chunk.text),
                        "frequency": text.lower().count(chunk.text.lower())
                    }
                    if concept["frequency"] >= self.config.min_concept_freq:
                        concepts.append(concept)
            except ImportError:
                logging.warning("Spacy not available for concept extraction")
        
        return concepts

    async def _extract_concepts_from_history(
        self,
        history: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract concepts from historical context."""
        all_concepts = []
        
        for entry in history.get("entries", []):
            if isinstance(entry, dict) and "text" in entry:
                concepts = await self._extract_concepts(entry["text"])
                all_concepts.extend(concepts)
        
        return all_concepts

    def _merge_concepts(
        self,
        current: List[Dict[str, Any]],
        historical: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge current and historical concepts."""
        merged = {}
        
        # Process current concepts
        for concept in current:
            key = concept["root"].lower()
            if key not in merged:
                merged[key] = concept.copy()
            else:
                merged[key]["frequency"] += concept["frequency"]
        
        # Merge historical concepts
        for concept in historical:
            key = concept["root"].lower()
            if key not in merged:
                merged[key] = concept.copy()
            else:
                merged[key]["frequency"] += concept["frequency"]
        
        return list(merged.values())

    async def _find_relevant_context(
        self,
        embedding: torch.Tensor,
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Find semantically relevant historical context."""
        relevant = {
            "entries": [],
            "scores": []
        }
        
        for entry in history:
            if isinstance(entry, dict) and "text" in entry:
                hist_emb = await self._get_embedding(entry["text"])
                score = self._calculate_similarity(embedding, hist_emb)
                
                if score > self.config.threshold_medium:
                    relevant["entries"].append(entry)
                    relevant["scores"].append(float(score))
        
        # Sort by relevance
        sorted_indices = np.argsort(relevant["scores"])[::-1]
        relevant["entries"] = [relevant["entries"][i] for i in sorted_indices]
        relevant["scores"] = [relevant["scores"][i] for i in sorted_indices]
        
        return relevant

    async def _measure_context_coherence(
        self,
        context: Dict[str, Any],
        history: Dict[str, Any]
    ) -> float:
        """Measure coherence between context and history."""
        if not context or not history:
            return 1.0
        
        coherence_scores = []
        current_emb = await self._get_embedding(context.get("text", ""))
        
        for entry, score in zip(history["entries"], history["scores"]):
            if "text" in entry:
                hist_emb = await self._get_embedding(entry["text"])
                coherence = self._calculate_similarity(current_emb, hist_emb)
                coherence_scores.append(coherence * score)  # Weight by relevance
        
        return float(np.mean(coherence_scores)) if coherence_scores else 1.0

    def _get_concept_frequency(
        self,
        concepts: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Get frequency distribution of concepts."""
        frequencies = {}
        for concept in concepts:
            root = concept["root"].lower()
            frequencies[root] = frequencies.get(root, 0) + concept["frequency"]
        return frequencies

    def _load_cache(self) -> None:
        """Load embedding cache from disk."""
        if not self.config.cache_embeddings or not self.config.cache_path:
            return
        
        try:
            cache_file = Path(self.config.cache_path) / "embeddings.pt"
            if cache_file.exists():
                self.embedding_cache = torch.load(cache_file)
        except Exception as e:
            logging.error(f"Error loading cache: {e}")

    async def _save_cache(self) -> None:
        """Save embedding cache to disk."""
        if not self.config.cache_embeddings or not self.config.cache_path:
            return
        
        try:
            cache_file = Path(self.config.cache_path) / "embeddings.pt"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.embedding_cache, cache_file)
        except Exception as e:
            logging.error(f"Error saving cache: {e}")

@pytest.fixture
def semantic_analyzer():
    """Create semantic analyzer for testing."""
    config = SemanticConfig()
    return SemanticAnalyzer(config)

@pytest.mark.asyncio
async def test_semantic_analysis(semantic_analyzer):
    """Test semantic analysis capabilities."""
    # Setup analyzer
    await semantic_analyzer.setup()
    
    # Test response analysis
    response = "The system performance improved significantly."
    context = {"original_text": "Performance metrics show positive trends."}
    
    results = await semantic_analyzer.analyze_response(response, context)
    
    assert "similarity" in results
    assert results["similarity"] >= 0
    
    if "concepts" in results:
        assert len(results["concepts"]) > 0
        assert all("text" in c for c in results["concepts"])

@pytest.mark.asyncio
async def test_context_enhancement(semantic_analyzer):
    """Test context enhancement."""
    # Setup analyzer
    await semantic_analyzer.setup()
    
    # Test context enhancement
    context = {
        "text": "Response time decreased by 50%"
    }
    
    history = [
        {"text": "Previous response time was high"},
        {"text": "System optimization improved metrics"}
    ]
    
    enhanced = await semantic_analyzer.enhance_context(context, history)
    
    assert "semantic_context" in enhanced
    assert "relevance_scores" in enhanced["semantic_context"]
    
    if "concepts" in enhanced:
        assert isinstance(enhanced["concepts"], list)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

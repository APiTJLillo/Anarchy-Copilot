"""Cross-lingual support for semantic analysis."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cosine
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    MarianTokenizer,
    MarianMTModel
)
import logging
import json
from pathlib import Path
import langdetect

from .test_costbenefit_semantic import (
    SemanticAnalyzer,
    SemanticConfig
)

@dataclass
class LanguageConfig:
    """Configuration for language support."""
    source_language: str = "en"
    target_languages: List[str] = field(default_factory=lambda: ["en", "es", "fr", "de"])
    translation_model: str = "Helsinki-NLP/opus-mt-{src}-{tgt}"
    semantic_model: str = "xlm-roberta-base"
    enable_auto_detection: bool = True
    min_detection_confidence: float = 0.8
    batch_translation: bool = True
    max_batch_size: int = 16
    cache_translations: bool = True
    cache_path: Optional[str] = "translation_cache"
    fallback_language: str = "en"
    enable_alignment: bool = True
    alignment_threshold: float = 0.7
    parallel_processing: bool = True
    max_parallel: int = 4

class MultilingualAnalyzer:
    """Cross-lingual semantic analysis."""

    def __init__(
        self,
        base_analyzer: SemanticAnalyzer,
        config: LanguageConfig
    ):
        self.base_analyzer = base_analyzer
        self.config = config
        
        # Translation models
        self.translation_models: Dict[str, MarianMTModel] = {}
        self.translation_tokenizers: Dict[str, MarianTokenizer] = {}
        
        # Caches
        self.translation_cache: Dict[str, Dict[str, str]] = {}
        self.alignment_cache: Dict[str, Dict[str, List[Tuple[int, int]]]] = {}
        
        # Load cache
        self._load_cache()
        
        # Initialize models lazily
        self._setup_ready = False

    async def setup(self) -> None:
        """Lazy initialization of models."""
        if not self._setup_ready:
            try:
                # Load translation models for target languages
                for lang in self.config.target_languages:
                    if lang != self.config.source_language:
                        await self._load_translation_model(
                            self.config.source_language,
                            lang
                        )
                
                self._setup_ready = True
                
            except Exception as e:
                logging.error(f"Error initializing multilingual models: {e}")

    async def analyze_multilingual(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform cross-lingual semantic analysis."""
        if not self._setup_ready:
            await self.setup()
        
        results = {}
        
        try:
            # Detect language if enabled
            source_lang = (
                await self._detect_language(text)
                if self.config.enable_auto_detection
                else self.config.source_language
            )
            
            # Translate if needed
            translations = {}
            if source_lang != self.config.source_language:
                translations = await self._translate_text(
                    text,
                    source_lang,
                    [self.config.source_language]
                )
                analysis_text = translations[self.config.source_language]
            else:
                analysis_text = text
            
            # Perform base analysis
            base_results = await self.base_analyzer.analyze_response(
                analysis_text,
                context
            )
            results.update(base_results)
            
            # Add multilingual information
            results["language"] = {
                "detected": source_lang,
                "confidence": await self._get_language_confidence(text, source_lang),
                "translations": translations
            }
            
            # Add cross-lingual metrics if available
            if source_lang != self.config.source_language:
                cross_results = await self._analyze_cross_lingual(
                    text,
                    analysis_text,
                    source_lang
                )
                results["cross_lingual"] = cross_results
            
            return results
            
        except Exception as e:
            logging.error(f"Error in multilingual analysis: {e}")
            return {"error": str(e)}

    async def _detect_language(self, text: str) -> str:
        """Detect text language."""
        try:
            lang = langdetect.detect(text)
            confidence = await self._get_language_confidence(text, lang)
            
            if confidence >= self.config.min_detection_confidence:
                return lang
            
            return self.config.fallback_language
            
        except Exception as e:
            logging.warning(f"Language detection failed: {e}")
            return self.config.fallback_language

    async def _get_language_confidence(
        self,
        text: str,
        lang: str
    ) -> float:
        """Get confidence score for language detection."""
        try:
            # Use langdetect probabilities
            probs = langdetect.detect_langs(text)
            for prob in probs:
                if prob.lang == lang:
                    return prob.prob
            return 0.0
            
        except Exception:
            return 0.0

    async def _load_translation_model(
        self,
        src_lang: str,
        tgt_lang: str
    ) -> None:
        """Load translation model for language pair."""
        model_name = self.config.translation_model.format(
            src=src_lang,
            tgt=tgt_lang
        )
        
        key = f"{src_lang}-{tgt_lang}"
        if key not in self.translation_models:
            try:
                self.translation_tokenizers[key] = MarianTokenizer.from_pretrained(
                    model_name
                )
                self.translation_models[key] = MarianMTModel.from_pretrained(
                    model_name
                ).to(self.base_analyzer.device)
            except Exception as e:
                logging.error(f"Error loading translation model {model_name}: {e}")

    async def _translate_text(
        self,
        text: str,
        src_lang: str,
        tgt_langs: List[str]
    ) -> Dict[str, str]:
        """Translate text to target languages."""
        translations = {}
        
        for tgt_lang in tgt_langs:
            if tgt_lang == src_lang:
                translations[tgt_lang] = text
                continue
            
            # Check cache
            cache_key = f"{src_lang}:{tgt_lang}:{hash(text)}"
            if (
                self.config.cache_translations and
                cache_key in self.translation_cache
            ):
                translations[tgt_lang] = self.translation_cache[cache_key]
                continue
            
            # Translate
            key = f"{src_lang}-{tgt_lang}"
            if key in self.translation_models:
                try:
                    inputs = self.translation_tokenizers[key](
                        text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.base_analyzer.config.max_length
                    ).to(self.base_analyzer.device)
                    
                    outputs = self.translation_models[key].generate(
                        **inputs,
                        max_length=self.base_analyzer.config.max_length
                    )
                    
                    translation = self.translation_tokenizers[key].decode(
                        outputs[0],
                        skip_special_tokens=True
                    )
                    
                    translations[tgt_lang] = translation
                    
                    # Cache translation
                    if self.config.cache_translations:
                        self.translation_cache[cache_key] = translation
                        await self._save_cache()
                        
                except Exception as e:
                    logging.error(f"Translation error {src_lang}->{tgt_lang}: {e}")
                    translations[tgt_lang] = text  # Fallback to original
            
            else:
                translations[tgt_lang] = text  # Fallback if model not available
        
        return translations

    async def _analyze_cross_lingual(
        self,
        original_text: str,
        translated_text: str,
        source_lang: str
    ) -> Dict[str, Any]:
        """Analyze cross-lingual aspects."""
        results = {}
        
        # Get embeddings in both languages
        orig_emb = await self.base_analyzer._get_embedding(original_text)
        trans_emb = await self.base_analyzer._get_embedding(translated_text)
        
        # Calculate cross-lingual similarity
        results["semantic_preservation"] = float(
            self.base_analyzer._calculate_similarity(orig_emb, trans_emb)
        )
        
        # Add alignment information if enabled
        if self.config.enable_alignment:
            alignments = await self._get_alignments(
                original_text,
                translated_text,
                source_lang
            )
            results["alignments"] = alignments
        
        return results

    async def _get_alignments(
        self,
        source_text: str,
        target_text: str,
        source_lang: str
    ) -> List[Tuple[int, int]]:
        """Get word alignments between texts."""
        cache_key = f"{source_lang}:{hash(source_text)}:{hash(target_text)}"
        
        if cache_key in self.alignment_cache:
            return self.alignment_cache[cache_key]
        
        try:
            # Use fast_align or similar alignment tool
            from fast_align import align
            alignments = align(source_text, target_text)
            
            # Filter by confidence
            alignments = [
                (i, j) for i, j, score in alignments
                if score >= self.config.alignment_threshold
            ]
            
            # Cache results
            self.alignment_cache[cache_key] = alignments
            return alignments
            
        except ImportError:
            logging.warning("fast_align not available for word alignment")
            return []

    def _load_cache(self) -> None:
        """Load translation cache from disk."""
        if not self.config.cache_translations or not self.config.cache_path:
            return
        
        try:
            cache_file = Path(self.config.cache_path) / "translations.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    self.translation_cache = json.load(f)
        except Exception as e:
            logging.error(f"Error loading translation cache: {e}")

    async def _save_cache(self) -> None:
        """Save translation cache to disk."""
        if not self.config.cache_translations or not self.config.cache_path:
            return
        
        try:
            cache_file = Path(self.config.cache_path) / "translations.json"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_file, 'w') as f:
                json.dump(self.translation_cache, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving translation cache: {e}")

@pytest.fixture
def multilingual_analyzer(semantic_analyzer):
    """Create multilingual analyzer for testing."""
    config = LanguageConfig()
    return MultilingualAnalyzer(semantic_analyzer, config)

@pytest.mark.asyncio
async def test_language_detection(multilingual_analyzer):
    """Test language detection."""
    # Test English
    text_en = "The system performance has improved."
    lang_en = await multilingual_analyzer._detect_language(text_en)
    assert lang_en == "en"
    
    # Test Spanish
    text_es = "El rendimiento del sistema ha mejorado."
    lang_es = await multilingual_analyzer._detect_language(text_es)
    assert lang_es == "es"

@pytest.mark.asyncio
async def test_translation(multilingual_analyzer):
    """Test text translation."""
    await multilingual_analyzer.setup()
    
    text = "The system is working well."
    translations = await multilingual_analyzer._translate_text(
        text,
        "en",
        ["es", "fr"]
    )
    
    assert "es" in translations
    assert "fr" in translations
    assert all(isinstance(t, str) for t in translations.values())

@pytest.mark.asyncio
async def test_multilingual_analysis(multilingual_analyzer):
    """Test full multilingual analysis."""
    await multilingual_analyzer.setup()
    
    # Test non-English text
    text = "El sistema está funcionando correctamente."
    context = {"original_text": "System performance is optimal."}
    
    results = await multilingual_analyzer.analyze_multilingual(text, context)
    
    assert "language" in results
    assert results["language"]["detected"] == "es"
    assert "translations" in results["language"]
    if "cross_lingual" in results:
        assert "semantic_preservation" in results["cross_lingual"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

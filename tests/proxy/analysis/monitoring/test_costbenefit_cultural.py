"""Cultural context awareness for multilingual analysis."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cosine
import torch
import logging
import json
from pathlib import Path

from .test_costbenefit_multilingual import (
    MultilingualAnalyzer,
    LanguageConfig
)

@dataclass
class CultureConfig:
    """Configuration for cultural context."""
    enable_cultural_adaptation: bool = True
    region_mappings: Dict[str, List[str]] = field(default_factory=lambda: {
        "en": ["US", "UK", "AU", "CA"],
        "es": ["ES", "MX", "AR", "CO"],
        "fr": ["FR", "CA", "BE", "CH"],
        "de": ["DE", "AT", "CH"]
    })
    formality_levels: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "en": {"formal": 0.7, "informal": 0.3},
        "es": {"formal": 0.8, "informal": 0.2},
        "fr": {"formal": 0.9, "informal": 0.1},
        "de": {"formal": 0.8, "informal": 0.2}
    })
    cultural_markers: Dict[str, List[str]] = field(default_factory=lambda: {
        "en": ["please", "thank you", "would", "could"],
        "es": ["por favor", "gracias", "podría", "debería"],
        "fr": ["s'il vous plaît", "merci", "pourriez", "devriez"],
        "de": ["bitte", "danke", "würden", "könnten"]
    })
    idiom_mappings: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "en": {
            "piece of cake": "very easy",
            "break a leg": "good luck"
        },
        "es": {
            "pan comido": "very easy",
            "mucha mierda": "good luck"
        }
    })
    context_db_path: Optional[str] = "cultural_context.db"
    min_cultural_confidence: float = 0.7
    enable_learning: bool = True
    learning_rate: float = 0.1
    update_interval: float = 86400  # 24 hours

class CulturalAnalyzer:
    """Cultural context analysis for multilingual content."""

    def __init__(
        self,
        base_analyzer: MultilingualAnalyzer,
        config: CultureConfig
    ):
        self.base_analyzer = base_analyzer
        self.config = config
        
        # Cultural knowledge base
        self.cultural_db: Dict[str, Any] = {}
        self.formality_models: Dict[str, Any] = {}
        self.region_classifiers: Dict[str, Any] = {}
        
        # Learning state
        self.context_history: List[Dict[str, Any]] = []
        self.last_update: Optional[datetime] = None
        
        # Load database
        self._load_cultural_db()

    async def analyze_cultural_context(
        self,
        text: str,
        language: str,
        region: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze cultural aspects of text."""
        results = {
            "language": language,
            "region": region or await self._detect_region(text, language)
        }
        
        try:
            # Analyze formality
            formality = await self._analyze_formality(text, language)
            results["formality"] = {
                "level": formality,
                "appropriate": self._check_formality_appropriateness(
                    formality,
                    language,
                    results["region"]
                )
            }
            
            # Detect cultural markers
            markers = await self._detect_cultural_markers(text, language)
            results["cultural_markers"] = {
                "detected": markers,
                "relevance": self._calculate_marker_relevance(
                    markers,
                    language,
                    results["region"]
                )
            }
            
            # Identify idioms
            idioms = await self._identify_idioms(text, language)
            if idioms:
                results["idioms"] = {
                    "detected": idioms,
                    "translations": await self._translate_idioms(
                        idioms,
                        language,
                        self.base_analyzer.config.source_language
                    )
                }
            
            # Add cultural context score
            results["context_score"] = self._calculate_context_score(results)
            
            # Update learning if enabled
            if self.config.enable_learning:
                await self._update_cultural_knowledge(text, results)
            
            return results
            
        except Exception as e:
            logging.error(f"Error in cultural analysis: {e}")
            return {"error": str(e)}

    async def enhance_translation(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Dict[str, Any]
    ) -> str:
        """Enhance translation with cultural context."""
        # Get base translation
        translations = await self.base_analyzer._translate_text(
            text,
            source_lang,
            [target_lang]
        )
        translation = translations.get(target_lang, text)
        
        try:
            # Analyze cultural context
            source_context = await self.analyze_cultural_context(
                text,
                source_lang
            )
            
            # Adapt formality
            if source_context.get("formality"):
                translation = await self._adapt_formality(
                    translation,
                    target_lang,
                    source_context["formality"]["level"]
                )
            
            # Adapt cultural markers
            if source_context.get("cultural_markers"):
                translation = await self._adapt_cultural_markers(
                    translation,
                    source_lang,
                    target_lang,
                    source_context["cultural_markers"]["detected"]
                )
            
            # Adapt idioms
            if source_context.get("idioms"):
                translation = await self._adapt_idioms(
                    translation,
                    source_lang,
                    target_lang,
                    source_context["idioms"]["detected"]
                )
            
            return translation
            
        except Exception as e:
            logging.error(f"Error enhancing translation: {e}")
            return translation

    async def _detect_region(
        self,
        text: str,
        language: str
    ) -> Optional[str]:
        """Detect likely region for language variant."""
        if language not in self.region_classifiers:
            return self.config.region_mappings.get(language, [None])[0]
        
        try:
            # Use region classifier
            classifier = self.region_classifiers[language]
            scores = classifier.predict_proba([text])[0]
            regions = self.config.region_mappings[language]
            
            # Get highest scoring region
            region_idx = np.argmax(scores)
            if scores[region_idx] >= self.config.min_cultural_confidence:
                return regions[region_idx]
            
        except Exception as e:
            logging.warning(f"Region detection failed: {e}")
        
        return None

    async def _analyze_formality(
        self,
        text: str,
        language: str
    ) -> float:
        """Analyze text formality level."""
        if language not in self.formality_models:
            # Use heuristic analysis
            markers = self.config.cultural_markers.get(language, [])
            formal_markers = [m for m in markers if m in text.lower()]
            informal_markers = len(text.split()) - len(formal_markers)
            
            return len(formal_markers) / max(len(text.split()), 1)
        
        try:
            # Use formality model
            model = self.formality_models[language]
            score = model.predict_proba([text])[0][1]  # Assume binary classification
            return float(score)
            
        except Exception as e:
            logging.warning(f"Formality analysis failed: {e}")
            return 0.5

    def _check_formality_appropriateness(
        self,
        formality: float,
        language: str,
        region: Optional[str]
    ) -> bool:
        """Check if formality level is appropriate."""
        if language not in self.config.formality_levels:
            return True
        
        expected = self.config.formality_levels[language]["formal"]
        if region:
            # Adjust expectation based on region
            region_adjustment = {
                "US": -0.1,  # Less formal
                "UK": 0.1,   # More formal
                "ES": 0.2,   # More formal
                "MX": -0.1   # Less formal
            }.get(region, 0.0)
            expected += region_adjustment
        
        return abs(formality - expected) <= 0.2

    async def _detect_cultural_markers(
        self,
        text: str,
        language: str
    ) -> List[str]:
        """Detect cultural markers in text."""
        markers = []
        text_lower = text.lower()
        
        # Check configured markers
        for marker in self.config.cultural_markers.get(language, []):
            if marker in text_lower:
                markers.append(marker)
        
        # Check learned markers from database
        db_markers = self.cultural_db.get(language, {}).get("markers", [])
        for marker in db_markers:
            if marker["text"] in text_lower:
                markers.append(marker["text"])
        
        return markers

    def _calculate_marker_relevance(
        self,
        markers: List[str],
        language: str,
        region: Optional[str]
    ) -> float:
        """Calculate relevance of detected markers."""
        if not markers:
            return 0.0
        
        relevant_count = 0
        total_weight = 0
        
        for marker in markers:
            weight = 1.0
            
            # Check regional relevance
            if region:
                db_info = self.cultural_db.get(language, {}).get("markers", [])
                for info in db_info:
                    if info["text"] == marker:
                        if region in info.get("regions", []):
                            weight *= 1.5
                        break
            
            # Check frequency in database
            db_freq = self._get_marker_frequency(marker, language)
            if db_freq > 0:
                weight *= min(1 + np.log(db_freq), 2.0)
            
            relevant_count += weight
            total_weight += 1.0
        
        return relevant_count / total_weight if total_weight > 0 else 0.0

    async def _identify_idioms(
        self,
        text: str,
        language: str
    ) -> List[Dict[str, Any]]:
        """Identify idioms in text."""
        idioms = []
        text_lower = text.lower()
        
        # Check configured idioms
        for idiom, meaning in self.config.idiom_mappings.get(language, {}).items():
            if idiom in text_lower:
                idioms.append({
                    "text": idiom,
                    "meaning": meaning,
                    "confidence": 1.0
                })
        
        # Check learned idioms
        db_idioms = self.cultural_db.get(language, {}).get("idioms", [])
        for idiom in db_idioms:
            if idiom["text"] in text_lower:
                idioms.append({
                    "text": idiom["text"],
                    "meaning": idiom["meaning"],
                    "confidence": idiom.get("confidence", 0.8)
                })
        
        return idioms

    async def _translate_idioms(
        self,
        idioms: List[Dict[str, Any]],
        source_lang: str,
        target_lang: str
    ) -> Dict[str, str]:
        """Translate idioms to target language."""
        translations = {}
        
        for idiom in idioms:
            # Check if direct mapping exists
            target_idioms = self.config.idiom_mappings.get(target_lang, {})
            source_meaning = idiom["meaning"]
            
            for target_idiom, target_meaning in target_idioms.items():
                if target_meaning == source_meaning:
                    translations[idiom["text"]] = target_idiom
                    break
            
            # Fallback to direct translation of meaning
            if idiom["text"] not in translations:
                meaning_translation = await self.base_analyzer._translate_text(
                    source_meaning,
                    source_lang,
                    [target_lang]
                )
                translations[idiom["text"]] = meaning_translation.get(
                    target_lang,
                    idiom["text"]
                )
        
        return translations

    def _calculate_context_score(
        self,
        analysis: Dict[str, Any]
    ) -> float:
        """Calculate overall cultural context score."""
        scores = []
        weights = {
            "formality": 0.4,
            "markers": 0.4,
            "idioms": 0.2
        }
        
        # Formality score
        if "formality" in analysis:
            formality_score = float(analysis["formality"]["appropriate"])
            scores.append((formality_score, weights["formality"]))
        
        # Cultural markers score
        if "cultural_markers" in analysis:
            marker_score = analysis["cultural_markers"]["relevance"]
            scores.append((marker_score, weights["markers"]))
        
        # Idiom score
        if "idioms" in analysis:
            idiom_score = np.mean([
                idiom["confidence"]
                for idiom in analysis["idioms"]["detected"]
            ]) if analysis["idioms"]["detected"] else 0.0
            scores.append((idiom_score, weights["idioms"]))
        
        # Calculate weighted average
        if scores:
            total_score = sum(score * weight for score, weight in scores)
            total_weight = sum(weight for _, weight in scores)
            return total_score / total_weight if total_weight > 0 else 0.0
        
        return 0.0

    async def _update_cultural_knowledge(
        self,
        text: str,
        analysis: Dict[str, Any]
    ) -> None:
        """Update cultural knowledge base."""
        if not self.config.enable_learning:
            return
        
        current_time = datetime.now()
        if (
            self.last_update and
            (current_time - self.last_update).total_seconds() <
            self.config.update_interval
        ):
            return
        
        try:
            language = analysis["language"]
            region = analysis["region"]
            
            # Update marker frequencies
            for marker in analysis.get("cultural_markers", {}).get("detected", []):
                self._update_marker_frequency(marker, language, region)
            
            # Update idiom confidence
            for idiom in analysis.get("idioms", {}).get("detected", []):
                self._update_idiom_confidence(idiom, language, region)
            
            # Save updates
            self._save_cultural_db()
            self.last_update = current_time
            
        except Exception as e:
            logging.error(f"Error updating cultural knowledge: {e}")

    def _get_marker_frequency(self, marker: str, language: str) -> int:
        """Get frequency of cultural marker."""
        markers = self.cultural_db.get(language, {}).get("markers", [])
        for m in markers:
            if m["text"] == marker:
                return m.get("frequency", 0)
        return 0

    def _update_marker_frequency(
        self,
        marker: str,
        language: str,
        region: Optional[str]
    ) -> None:
        """Update frequency count for cultural marker."""
        if language not in self.cultural_db:
            self.cultural_db[language] = {"markers": [], "idioms": []}
        
        markers = self.cultural_db[language]["markers"]
        for m in markers:
            if m["text"] == marker:
                m["frequency"] = m.get("frequency", 0) + 1
                if region and region not in m.get("regions", []):
                    m.setdefault("regions", []).append(region)
                return
        
        # Add new marker
        markers.append({
            "text": marker,
            "frequency": 1,
            "regions": [region] if region else []
        })

    def _update_idiom_confidence(
        self,
        idiom: Dict[str, Any],
        language: str,
        region: Optional[str]
    ) -> None:
        """Update confidence score for idiom."""
        if language not in self.cultural_db:
            self.cultural_db[language] = {"markers": [], "idioms": []}
        
        idioms = self.cultural_db[language]["idioms"]
        for i in idioms:
            if i["text"] == idiom["text"]:
                # Update confidence with exponential moving average
                i["confidence"] = (
                    i.get("confidence", 0.8) * (1 - self.config.learning_rate) +
                    idiom["confidence"] * self.config.learning_rate
                )
                if region:
                    i.setdefault("regions", []).append(region)
                return
        
        # Add new idiom
        idioms.append({
            "text": idiom["text"],
            "meaning": idiom["meaning"],
            "confidence": idiom["confidence"],
            "regions": [region] if region else []
        })

    def _load_cultural_db(self) -> None:
        """Load cultural knowledge base."""
        if not self.config.context_db_path:
            return
        
        try:
            db_path = Path(self.config.context_db_path)
            if db_path.exists():
                with open(db_path, 'r') as f:
                    self.cultural_db = json.load(f)
        except Exception as e:
            logging.error(f"Error loading cultural database: {e}")

    def _save_cultural_db(self) -> None:
        """Save cultural knowledge base."""
        if not self.config.context_db_path:
            return
        
        try:
            db_path = Path(self.config.context_db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(db_path, 'w') as f:
                json.dump(self.cultural_db, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving cultural database: {e}")

@pytest.fixture
def cultural_analyzer(multilingual_analyzer):
    """Create cultural analyzer for testing."""
    config = CultureConfig()
    return CulturalAnalyzer(multilingual_analyzer, config)

@pytest.mark.asyncio
async def test_cultural_analysis(cultural_analyzer):
    """Test cultural context analysis."""
    # Test English formal text
    text_en = "Would you please review the system performance?"
    analysis = await cultural_analyzer.analyze_cultural_context(text_en, "en")
    
    assert "formality" in analysis
    assert analysis["formality"]["level"] > 0.5
    assert "cultural_markers" in analysis
    assert len(analysis["cultural_markers"]["detected"]) > 0

@pytest.mark.asyncio
async def test_translation_enhancement(cultural_analyzer):
    """Test culturally-aware translation enhancement."""
    text = "It was a piece of cake."
    enhanced = await cultural_analyzer.enhance_translation(
        text,
        "en",
        "es",
        {}
    )
    
    assert isinstance(enhanced, str)
    assert len(enhanced) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

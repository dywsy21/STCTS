"""Main prosody feature extractor."""

from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import parselmouth

from src.prosody.pitch import extract_pitch, PitchNormalizer
from src.prosody.energy import extract_energy, EnergyNormalizer
from src.prosody.rate import extract_speaking_rate, RateNormalizer
from src.prosody.emotion import EmotionDetector
from src.prosody.emphasis import EmphasisDetector, EmphasisRegion
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProsodyFeatures:
    """Prosody features container."""
    pitch: Optional[np.ndarray] = None  # F0 contour (normalized)
    energy: Optional[np.ndarray] = None  # Energy/volume contour (normalized)
    speaking_rate: Optional[np.ndarray] = None  # Syllables/sec trajectory (normalized)
    emotion: Optional[str] = None  # Emotion label
    emphasis: Optional[List[EmphasisRegion]] = None  # Emphasis regions
    timestamp: float = 0.0


class ProsodyExtractor:
    """Extract prosody features from audio."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        enable_emotion: bool = True,
        enable_emphasis: bool = True
    ):
        """Initialize prosody extractor.
        
        Args:
            sample_rate: Audio sample rate
            enable_emotion: Enable emotion detection
            enable_emphasis: Enable emphasis detection
        """
        self.sample_rate = sample_rate
        self.enable_emotion = enable_emotion
        self.enable_emphasis = enable_emphasis
        
        # Normalizers
        self.pitch_normalizer = PitchNormalizer()
        self.energy_normalizer = EnergyNormalizer()
        self.rate_normalizer = RateNormalizer()
        
        # Lazy load detectors
        self.emotion_detector = EmotionDetector() if enable_emotion else None
        self.emphasis_detector = EmphasisDetector(sample_rate) if enable_emphasis else None
        
        logger.info(f"Prosody extractor initialized (emotion={enable_emotion}, emphasis={enable_emphasis})")
    
    def extract(
        self,
        audio: np.ndarray,
        extract_features: List[str],
        words: Optional[List[Tuple[str, float, float]]] = None
    ) -> ProsodyFeatures:
        """Extract prosody features from audio.
        
        Args:
            audio: Audio data as numpy array (float32, -1 to 1)
            extract_features: List of features to extract
            words: Optional list of (word, start_time, end_time) tuples for emphasis
        
        Returns:
            Prosody features
        """
        features = ProsodyFeatures()
        
        # Extract requested features
        if "pitch" in extract_features:
            raw_pitch = extract_pitch(audio, self.sample_rate)
            features.pitch = self.pitch_normalizer.normalize(raw_pitch)
        
        if "energy" in extract_features:
            raw_energy = extract_energy(audio, self.sample_rate)
            features.energy = self.energy_normalizer.normalize(raw_energy)
        
        if "speaking_rate" in extract_features:
            raw_rate = extract_speaking_rate(audio, self.sample_rate)
            features.speaking_rate = self.rate_normalizer.normalize(raw_rate)
        
        if "emotion" in extract_features and self.emotion_detector:
            features.emotion = self.emotion_detector.get_dominant_emotion(audio, self.sample_rate)
        
        if "emphasis" in extract_features and self.emphasis_detector:
            features.emphasis = self.emphasis_detector.detect(audio, words)
        
        return features
    
    def extract_all(
        self,
        audio: np.ndarray,
        words: Optional[List[Tuple[str, float, float]]] = None
    ) -> ProsodyFeatures:
        """Extract all available prosody features.
        
        Args:
            audio: Audio data
            words: Optional word timestamps
        
        Returns:
            Complete prosody features
        """
        features_list = ["pitch", "energy", "speaking_rate"]
        
        if self.enable_emotion:
            features_list.append("emotion")
        
        if self.enable_emphasis:
            features_list.append("emphasis")
        
        return self.extract(audio, features_list, words)


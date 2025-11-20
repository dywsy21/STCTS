"""Unit tests for prosody extractor."""
import numpy as np
from src.prosody.extractor import ProsodyExtractor, ProsodyFeatures


def test_prosody_extractor_creation():
    """Test creating a prosody extractor."""
    extractor = ProsodyExtractor(sample_rate=16000)
    
    assert extractor.sample_rate == 16000
    assert extractor.enable_emotion is True
    assert extractor.enable_emphasis is True


def test_prosody_extractor_disabled_features():
    """Test creating extractor with disabled features."""
    extractor = ProsodyExtractor(
        sample_rate=16000,
        enable_emotion=False,
        enable_emphasis=False
    )
    
    assert extractor.emotion_detector is None
    assert extractor.emphasis_detector is None


def test_prosody_features_dataclass():
    """Test ProsodyFeatures dataclass."""
    features = ProsodyFeatures()
    
    assert features.pitch is None
    assert features.energy is None
    assert features.speaking_rate is None
    assert features.emotion is None
    assert features.emphasis is None
    assert features.timestamp == 0.0


def test_extract_pitch_feature():
    """Test extracting pitch feature."""
    extractor = ProsodyExtractor(
        sample_rate=16000,
        enable_emotion=False,
        enable_emphasis=False
    )
    
    # Generate test audio
    duration = 0.5
    t = np.linspace(0, duration, int(16000 * duration))
    audio = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    
    features = extractor.extract(audio, extract_features=["pitch"])
    
    assert features.pitch is not None
    assert len(features.pitch) > 0


def test_extract_energy_feature():
    """Test extracting energy feature."""
    extractor = ProsodyExtractor(
        sample_rate=16000,
        enable_emotion=False,
        enable_emphasis=False
    )
    
    # Generate test audio
    duration = 0.5
    t = np.linspace(0, duration, int(16000 * duration))
    audio = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    
    features = extractor.extract(audio, extract_features=["energy"])
    
    assert features.energy is not None
    assert len(features.energy) > 0
    assert np.all(features.energy >= 0)


def test_extract_rate_feature():
    """Test extracting speaking rate feature."""
    extractor = ProsodyExtractor(
        sample_rate=16000,
        enable_emotion=False,
        enable_emphasis=False
    )
    
    # Generate test audio
    duration = 1.0
    t = np.linspace(0, duration, int(16000 * duration))
    audio = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    
    features = extractor.extract(audio, extract_features=["speaking_rate"])
    
    assert features.speaking_rate is not None
    assert features.speaking_rate >= 0


def test_extract_multiple_features():
    """Test extracting multiple features at once."""
    extractor = ProsodyExtractor(
        sample_rate=16000,
        enable_emotion=False,
        enable_emphasis=False
    )
    
    # Generate test audio
    duration = 1.0
    t = np.linspace(0, duration, int(16000 * duration))
    audio = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    
    features = extractor.extract(audio, extract_features=["pitch", "energy", "speaking_rate"])
    
    assert features.pitch is not None
    assert features.energy is not None
    assert features.speaking_rate is not None


def test_extract_no_features():
    """Test extracting with empty feature list."""
    extractor = ProsodyExtractor(
        sample_rate=16000,
        enable_emotion=False,
        enable_emphasis=False
    )
    
    # Generate test audio
    duration = 0.5
    t = np.linspace(0, duration, int(16000 * duration))
    audio = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    
    features = extractor.extract(audio, extract_features=[])
    
    # Should return empty features
    assert features.pitch is None
    assert features.energy is None
    assert features.speaking_rate is None


def test_extract_from_silence():
    """Test extracting features from silence."""
    extractor = ProsodyExtractor(
        sample_rate=16000,
        enable_emotion=False,
        enable_emphasis=False
    )
    
    # Silent audio
    audio = np.zeros(8000, dtype=np.float32)
    
    features = extractor.extract(audio, extract_features=["pitch", "energy", "speaking_rate"])
    
    # Should still return features even for silence
    assert features.energy is not None
    # Energy should be near zero
    assert np.allclose(features.energy, 0, atol=1e-6)

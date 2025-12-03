"""Pytest configuration and shared fixtures."""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def sample_audio():
    """Generate 1-second sample audio at 16kHz."""
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Generate a simple sine wave (440 Hz)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sample_rate

@pytest.fixture
def sample_audio_16k():
    """Generate 1-second sample audio at 16kHz (alias for sample_audio)."""
    return sample_audio()

@pytest.fixture
def sample_audio_48k():
    """Generate 1-second sample audio at 48kHz."""
    sample_rate = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sample_rate

@pytest.fixture
def test_config():
    """Load minimal mode configuration."""
    from src.utils.config import load_config
    config = load_config("configs/minimal_mode.yaml")
    return config

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)

@pytest.fixture
def sample_text():
    """Sample text for compression testing."""
    return "Hello, this is a test message for voice communication."

@pytest.fixture
def sample_prosody():
    """Sample prosody features."""
    return {
        'pitch': np.array([200.0, 210.0, 205.0, 195.0]),
        'energy': np.array([0.5, 0.6, 0.55, 0.45]),
        'rate': 4.5
    }

@pytest.fixture
def sample_timbre():
    """Sample speaker embedding."""
    return np.random.randn(192).astype(np.float32)

@pytest.fixture(scope="session")
def stt_model_mock():
    """Mock STT model for testing."""
    from unittest.mock import MagicMock
    mock = MagicMock()
    mock.transcribe.return_value = [
        {"text": "test transcription", "segments": []}
    ]
    return mock

@pytest.fixture(scope="session")
def tts_model_mock():
    """Mock TTS model for testing."""
    from unittest.mock import MagicMock
    mock = MagicMock()
    mock.tts_to_file.return_value = None
    return mock

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "requires_model: marks tests that require downloaded models"
    )

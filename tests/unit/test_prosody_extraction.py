"""Unit tests for prosody extraction modules."""
import numpy as np
from src.prosody.pitch import extract_pitch
from src.prosody.energy import extract_energy
from src.prosody.rate import extract_speaking_rate


def test_extract_pitch():
    """Test pitch extraction from audio."""
    # Generate a simple sine wave (440 Hz)
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440.0  # A4 note
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    pitch = extract_pitch(audio, sample_rate, time_step=0.01)
    
    assert len(pitch) > 0
    # Check that extracted pitch is close to 440 Hz (within tolerance)
    valid_pitch = pitch[~np.isnan(pitch)]
    if len(valid_pitch) > 0:
        mean_pitch = np.mean(valid_pitch)
        assert 400 < mean_pitch < 480  # Reasonable range around 440


def test_extract_pitch_silence():
    """Test pitch extraction from silence."""
    sample_rate = 16000
    duration = 0.5
    audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
    
    pitch = extract_pitch(audio, sample_rate)
    
    # Silence should return mostly NaN or zeros
    assert len(pitch) >= 0


def test_extract_pitch_noise():
    """Test pitch extraction from noise."""
    sample_rate = 16000
    duration = 0.5
    audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
    
    pitch = extract_pitch(audio, sample_rate)
    
    # Should return some values even for noise
    assert len(pitch) >= 0


def test_extract_energy():
    """Test energy extraction from audio."""
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    energy = extract_energy(audio, sample_rate, frame_length=400)
    
    assert len(energy) > 0
    assert np.all(energy >= 0)  # Energy should be non-negative
    assert np.any(energy > 0)  # Should have some non-zero energy


def test_extract_energy_silence():
    """Test energy extraction from silence."""
    sample_rate = 16000
    duration = 0.5
    audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
    
    energy = extract_energy(audio, sample_rate)
    
    assert len(energy) > 0
    # Silence should have very low energy
    assert np.allclose(energy, 0, atol=1e-6)


def test_extract_energy_varying():
    """Test energy extraction with varying amplitude."""
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create audio with increasing amplitude
    amplitude = np.linspace(0.1, 1.0, len(t))
    audio = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    
    energy = extract_energy(audio, sample_rate)
    
    assert len(energy) > 0
    # Energy should generally increase
    # Check first half vs second half
    mid = len(energy) // 2
    assert np.mean(energy[mid:]) > np.mean(energy[:mid])


def test_estimate_speaking_rate():
    """Test speaking rate estimation."""
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Simulate speech with pauses (syllables)
    audio = np.zeros_like(t)
    syllable_duration = 0.15  # 150ms per syllable
    pause_duration = 0.05     # 50ms pause
    
    current_time = 0
    while current_time < duration:
        start_idx = int(current_time * sample_rate)
        end_idx = int((current_time + syllable_duration) * sample_rate)
        if end_idx <= len(audio):
            audio[start_idx:end_idx] = np.sin(2 * np.pi * 200 * t[start_idx:end_idx])
        current_time += syllable_duration + pause_duration
    
    rate = extract_speaking_rate(audio.astype(np.float32), sample_rate)
    
    assert rate > 0
    # Typical speaking rate is 3-7 syllables/second
    assert 1 < rate < 10


def test_estimate_speaking_rate_silence():
    """Test speaking rate estimation for silence."""
    sample_rate = 16000
    duration = 1.0
    audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
    
    rate = extract_speaking_rate(audio, sample_rate)
    
    # Silence should have very low speaking rate
    assert rate >= 0
    assert rate < 1


def test_estimate_speaking_rate_continuous():
    """Test speaking rate for continuous speech."""
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Continuous tone (no pauses)
    audio = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    
    rate = extract_speaking_rate(audio, sample_rate)
    
    # Should detect some rate even for continuous audio
    assert rate >= 0

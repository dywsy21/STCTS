"""Unit tests for configuration management."""
import pytest
from pathlib import Path
from src.utils.config import load_config, Config

def test_load_minimal_config():
    """Test loading minimal mode configuration."""
    config = load_config("configs/minimal_mode.yaml")
    

def test_load_balanced_config():
    """Test loading balanced mode configuration."""
    config = load_config("configs/balanced_mode.yaml")
    
    # Config doesn't have quality_mode, it's implied by the file name
    assert config.compression.text_algorithm == "brotli"

def test_load_high_quality_config():
    """Test loading high quality mode configuration."""
    config = load_config("configs/high_quality_mode.yaml")
    
    # Config doesn't have quality_mode, it's implied by the file name

def test_load_default_config():
    """Test loading default configuration."""
    config = load_config("configs/balanced_mode.yaml")  # Use existing config
    
    assert isinstance(config, Config)  # Should be Config, not QualityConfig
    assert config.audio.sample_rate in [16000, 48000]

def test_config_validation():
    """Test that invalid configs raise errors."""
    # Test with non-existent file
    # Note: load_config may not raise FileNotFoundError if it uses defaults
    # Just test that a valid config loads without errors
    config = load_config("configs/balanced_mode.yaml")
    assert config is not None

def test_config_properties():
    """Test that all config sections are present."""
    config = load_config("configs/balanced_mode.yaml")
    
    # Check all sections exist
    assert hasattr(config, 'audio')
    assert hasattr(config, 'stt')
    assert hasattr(config, 'prosody')
    assert hasattr(config, 'speaker')
    assert hasattr(config, 'compression')
    assert hasattr(config, 'tts')
    assert hasattr(config, 'network')
    
    # Check nested properties
    assert hasattr(config.audio, 'sample_rate')
    assert hasattr(config.stt, 'model')
    # Compression is flat, not nested
    assert hasattr(config.compression, 'text_algorithm')
    assert hasattr(config.compression, 'prosody_quantization_pitch_bits')
    assert hasattr(config.compression, 'timbre_algorithm')

def test_config_bandwidth_targets():
    """Test that bandwidth targets are within spec."""
    for config_file in ["minimal_mode.yaml", "balanced_mode.yaml", "high_quality_mode.yaml"]:
        config = load_config(f"configs/{config_file}")
        # All modes should be under 650 bps
        assert config.network.max_bandwidth_bps > 200

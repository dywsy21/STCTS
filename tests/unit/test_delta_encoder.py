"""Unit tests for delta encoder."""
import pytest
import numpy as np
from src.compression.delta_encoder import DeltaEncoder

def test_delta_encoder_basic():
    """Test basic delta encoding and decoding."""
    encoder = DeltaEncoder()
    
    data = np.array([100.0, 105.0, 103.0, 108.0, 110.0])
    encoded = encoder.encode(data)
    decoded = encoder.decode(encoded)
    
    assert np.array_equal(data, decoded)

def test_delta_encoder_constant_values():
    """Test delta encoding with constant values."""
    encoder = DeltaEncoder()
    
    data = np.array([5.0, 5.0, 5.0, 5.0])
    encoded = encoder.encode(data)
    decoded = encoder.decode(encoded)
    
    assert np.array_equal(data, decoded)
    # All deltas should be zero except first value
    assert encoded[0] == 5.0
    assert np.all(encoded[1:] == 0)

def test_delta_encoder_linear_increase():
    """Test delta encoding with linear increase."""
    encoder = DeltaEncoder()
    
    data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    encoded = encoder.encode(data)
    decoded = encoder.decode(encoded)
    
    assert np.array_equal(data, decoded)
    # All deltas should be 1.0
    assert np.all(encoded[1:] == 1.0)

def test_delta_encoder_compression_benefit():
    """Test that delta encoding provides compression benefit."""
    encoder = DeltaEncoder()
    
    # Slowly varying signal
    data = np.array([200.0, 201.0, 199.0, 200.0, 202.0])
    encoded = encoder.encode(data)
    
    # Deltas should be small
    assert np.all(np.abs(encoded[1:]) <= 3.0)
    # When quantized, small deltas should compress better

def test_delta_encoder_large_changes():
    """Test delta encoding with large changes."""
    encoder = DeltaEncoder()
    
    data = np.array([100.0, 200.0, 50.0, 300.0])
    encoded = encoder.encode(data)
    decoded = encoder.decode(encoded)
    
    assert np.allclose(data, decoded)

def test_delta_encoder_single_value():
    """Test delta encoding with single value."""
    encoder = DeltaEncoder()
    
    data = np.array([42.0])
    encoded = encoder.encode(data)
    decoded = encoder.decode(encoded)
    
    assert np.array_equal(data, decoded)
    assert encoded[0] == 42.0
    assert len(encoded) == 1  # Only first value, no deltas

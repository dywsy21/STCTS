"""Unit tests for quantizer."""
import pytest
import numpy as np
from src.compression.quantizer import Quantizer

def test_quantizer_basic():
    """Test basic quantization and dequantization."""
    quantizer = Quantizer()
    
    data = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    quantized = quantizer.quantize(data, num_bits=4, min_val=0.0, max_val=1.0)
    dequantized = quantizer.dequantize(quantized, num_bits=4, min_val=0.0, max_val=1.0)
    
    assert dequantized.shape == data.shape
    assert np.allclose(data, dequantized, atol=0.1)

def test_quantizer_pack_unpack():
    """Test packing and unpacking quantized values."""
    quantizer = Quantizer()
    
    quantized = np.array([0, 7, 15, 3, 11], dtype=np.uint8)
    packed = quantizer.pack_to_bytes(quantized, num_bits=4)
    unpacked = quantizer.unpack_from_bytes(packed, num_bits=4, length=5)
    
    assert np.array_equal(quantized, unpacked)

def test_quantizer_compression_ratio():
    """Test that quantization reduces size."""
    quantizer = Quantizer()
    
    # 100 float32 values = 400 bytes
    data = np.random.uniform(0, 1, 100).astype(np.float32)
    original_size = data.nbytes
    
    # Quantize to 4 bits
    quantized = quantizer.quantize(data, num_bits=4, min_val=0.0, max_val=1.0)
    packed = quantizer.pack_to_bytes(quantized, num_bits=4)
    compressed_size = len(packed)
    
    # With proper bit-packing: 100 values * 4 bits = 400 bits = 50 bytes
    assert compressed_size == 50  # 100 * 4 bits / 8 = 50 bytes
    assert compressed_size < original_size * 0.15  # Much better compression!

def test_quantizer_different_bit_depths():
    """Test quantization with different bit depths."""
    quantizer = Quantizer()
    data = np.array([0.0, 0.5, 1.0])
    
    for n_bits in [2, 4, 6, 8]:
        quantized = quantizer.quantize(data, num_bits=n_bits, min_val=0.0, max_val=1.0)
        dequantized = quantizer.dequantize(quantized, num_bits=n_bits, min_val=0.0, max_val=1.0)
        
        # Higher bit depth should have lower error
        error = np.mean(np.abs(data - dequantized))
        assert error < 0.2

def test_quantizer_custom_range():
    """Test quantization with custom value range."""
    quantizer = Quantizer()
    
    data = np.array([50.0, 150.0, 250.0, 350.0])
    quantized = quantizer.quantize(data, num_bits=8, min_val=0.0, max_val=400.0)
    dequantized = quantizer.dequantize(quantized, num_bits=8, min_val=0.0, max_val=400.0)
    
    assert np.allclose(data, dequantized, rtol=0.05)

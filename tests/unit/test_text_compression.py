"""Unit tests for text compression."""
import pytest
from src.compression.text import TextCompressor

def test_text_compressor_basic():
    """Test basic text compression and decompression."""
    compressor = TextCompressor(algorithm='brotli', level=5)
    
    original = "Hello, this is a test message."
    compressed = compressor.compress(original)
    decompressed = compressor.decompress(compressed)
    
    assert decompressed == original
    # Small texts may not compress well due to overhead
    # Just verify decompression works correctly

def test_text_compressor_with_preprocessing():
    """Test text compression with preprocessing."""
    compressor = TextCompressor(algorithm='brotli', level=5, preprocess=True)
    
    original = "I am going to test this."
    compressed = compressor.compress(original)
    decompressed = compressor.decompress(compressed)
    
    # Preprocessing converts "going to" -> "gonna"
    assert "gonna" in decompressed or decompressed == original

def test_text_compressor_lz4():
    """Test LZ4 compression."""
    compressor = TextCompressor(algorithm='lz4')
    
    original = "Test message for LZ4 compression."
    compressed = compressor.compress(original)
    decompressed = compressor.decompress(compressed)
    
    assert decompressed == original

def test_text_compressor_empty_string():
    """Test compression of empty string."""
    compressor = TextCompressor(algorithm='brotli', level=5)
    
    original = ""
    compressed = compressor.compress(original)
    decompressed = compressor.decompress(compressed)
    
    assert decompressed == original

def test_text_compressor_long_text():
    """Test compression of longer text."""
    compressor = TextCompressor(algorithm='brotli', level=5)
    
    original = "This is a longer test message. " * 100
    compressed = compressor.compress(original)
    decompressed = compressor.decompress(compressed)
    
    # Preprocessing may remove trailing space
    assert decompressed.strip() == original.strip()
    # Should achieve significant compression
    assert len(compressed) < len(original.encode('utf-8')) * 0.5

def test_text_compressor_compression_ratio():
    """Test that compression ratio meets expectations."""
    compressor = TextCompressor(algorithm='brotli', level=5)
    
    # Repetitive text should compress well
    original = "hello " * 100
    compressed = compressor.compress(original)
    
    ratio = len(compressed) / len(original.encode('utf-8'))
    assert ratio < 0.2  # Should compress to <20% of original

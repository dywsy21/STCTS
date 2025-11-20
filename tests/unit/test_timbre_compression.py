"""Unit tests for timbre compression."""
import pytest
import numpy as np
from src.compression.timbre import TimbreCompressor

def test_timbre_compressor_basic():
    """Test basic timbre compression and decompression."""
    compressor = TimbreCompressor()
    
    # Generate random embedding (192-dim)
    embedding = np.random.randn(192).astype(np.float32)
    
    compressed = compressor.compress(embedding)
    decompressed = compressor.decompress(compressed, dim=192)
    
    assert decompressed.shape == embedding.shape
    assert np.allclose(embedding, decompressed, rtol=1e-3)

def test_timbre_compressor_compression_ratio():
    """Test that timbre compression achieves 50% reduction."""
    compressor = TimbreCompressor()
    
    embedding = np.random.randn(192).astype(np.float32)
    original_size = embedding.nbytes  # 192 * 4 = 768 bytes
    
    compressed = compressor.compress(embedding)
    compressed_size = len(compressed)
    
    # float16 should be exactly 50% of float32
    assert compressed_size == original_size // 2
    assert compressed_size == 384  # 192 * 2 bytes

def test_timbre_compressor_different_sizes():
    """Test timbre compression with different embedding sizes."""
    compressor = TimbreCompressor()
    
    for size in [128, 192, 256, 512]:
        embedding = np.random.randn(size).astype(np.float32)
        compressed = compressor.compress(embedding)
        decompressed = compressor.decompress(compressed, dim=size)
        
        assert decompressed.shape == embedding.shape
        assert np.allclose(embedding, decompressed, rtol=1e-3)

def test_timbre_compressor_preserves_similarity():
    """Test that compression preserves embedding similarity."""
    compressor = TimbreCompressor()
    
    # Create two similar embeddings
    embedding1 = np.random.randn(192).astype(np.float32)
    embedding2 = embedding1 + np.random.randn(192).astype(np.float32) * 0.1
    
    # Compute original cosine similarity
    sim_original = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )
    
    # Compress and decompress
    compressed1 = compressor.compress(embedding1)
    compressed2 = compressor.compress(embedding2)
    decompressed1 = compressor.decompress(compressed1, dim=192)
    decompressed2 = compressor.decompress(compressed2, dim=192)
    
    # Compute similarity after compression
    sim_compressed = np.dot(decompressed1, decompressed2) / (
        np.linalg.norm(decompressed1) * np.linalg.norm(decompressed2)
    )
    
    # Similarity should be preserved (within 1%)
    assert abs(sim_original - sim_compressed) < 0.01

def test_timbre_compressor_zero_embedding():
    """Test compression of zero embedding."""
    compressor = TimbreCompressor()
    
    embedding = np.zeros(192, dtype=np.float32)
    compressed = compressor.compress(embedding)
    decompressed = compressor.decompress(compressed, dim=192)
    
    assert np.allclose(decompressed, embedding)

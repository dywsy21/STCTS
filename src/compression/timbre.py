"""Timbre (speaker embedding) compression."""

import numpy as np
import zlib
try:
    import brotli
except ImportError:
    brotli = None

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TimbreCompressor:
    """Compress speaker embeddings."""
    
    def __init__(self, algorithm: str = "float16"):
        """Initialize timbre compressor.
        
        Args:
            algorithm: Compression algorithm (float16, float32, float16+zlib, float16+brotli, etc.)
        """
        self.algorithm = algorithm
        logger.info(f"Timbre compressor: {algorithm}")
    
    def compress(self, embedding: np.ndarray) -> bytes:
        """Compress speaker embedding.
        
        Args:
            embedding: Speaker embedding (float32)
        
        Returns:
            Compressed embedding
        """
        # 1. Quantization / Type Conversion
        if self.algorithm.startswith("float16"):
            data = embedding.astype(np.float16).tobytes()
        elif self.algorithm.startswith("float32"):
            data = embedding.astype(np.float32).tobytes()
        else:
            # Default to float16 if unknown prefix, or raise error?
            # For safety, let's assume float16 if not specified, or raise.
            # Given existing code, let's be strict about the base type.
            raise ValueError(f"Unknown base format in algorithm: {self.algorithm}")

        # 2. Universal Compression
        if "+zlib" in self.algorithm:
            return zlib.compress(data)
        elif "+brotli" in self.algorithm:
            if brotli is None:
                logger.warning("Brotli not installed, falling back to uncompressed")
                return data
            return brotli.compress(data)
        
        return data
    
    def decompress(self, data: bytes, dim: int) -> np.ndarray:
        """Decompress speaker embedding.
        
        Args:
            data: Compressed data
            dim: Embedding dimension
        
        Returns:
            Speaker embedding
        """
        # 1. Universal Decompression
        if "+zlib" in self.algorithm:
            try:
                data = zlib.decompress(data)
            except zlib.error as e:
                logger.error(f"Zlib decompression failed: {e}")
                raise
        elif "+brotli" in self.algorithm:
            if brotli is None:
                raise ImportError("Brotli required for decompression but not installed")
            try:
                data = brotli.decompress(data)
            except brotli.error as e:
                logger.error(f"Brotli decompression failed: {e}")
                raise

        # 2. Type Conversion
        if self.algorithm.startswith("float16"):
            embedding_fp16 = np.frombuffer(data, dtype=np.float16, count=dim)
            return embedding_fp16.astype(np.float32)
        elif self.algorithm.startswith("float32"):
            return np.frombuffer(data, dtype=np.float32, count=dim)
        else:
            raise ValueError(f"Unknown base format in algorithm: {self.algorithm}")

"""Text compression."""

import brotli
import lz4.frame

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TextCompressor:
    """Compress text data."""
    
    def __init__(self, algorithm: str = "brotli", level: int = 5, preprocess: bool = True):
        """Initialize text compressor.
        
        Args:
            algorithm: Compression algorithm (brotli, lz4)
            level: Compression level
            preprocess: Enable text preprocessing
        """
        self.algorithm = algorithm.lower()
        self.level = level
        self.preprocess = preprocess
        
        logger.info(f"Text compressor: {algorithm} (level={level}, preprocess={preprocess})")
    
    def _preprocess(self, text: str) -> str:
        """Preprocess text before compression.
        
        Args:
            text: Input text
        
        Returns:
            Preprocessed text
        """
        if not self.preprocess:
            return text
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Common abbreviations (saves bytes)
        replacements = {
            "going to": "gonna",
            "want to": "wanna",
            "have to": "gotta",
            "out of": "outta",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def compress(self, text: str) -> bytes:
        """Compress text.
        
        Args:
            text: Text to compress
        
        Returns:
            Compressed data
        """
        # Preprocess
        text = self._preprocess(text)
        
        # Encode to bytes
        text_bytes = text.encode('utf-8')
        
        # Compress
        if self.algorithm == "brotli":
            compressed = brotli.compress(text_bytes, quality=self.level)
        elif self.algorithm == "lz4":
            compressed = lz4.frame.compress(text_bytes, compression_level=self.level)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        logger.debug(f"Compressed {len(text_bytes)} -> {len(compressed)} bytes")
        return compressed
    
    def decompress(self, compressed: bytes) -> str:
        """Decompress text.
        
        Args:
            compressed: Compressed data
        
        Returns:
            Decompressed text
        """
        # Decompress
        if self.algorithm == "brotli":
            text_bytes = brotli.decompress(compressed)
        elif self.algorithm == "lz4":
            text_bytes = lz4.frame.decompress(compressed)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Decode
        text = text_bytes.decode('utf-8')
        
        return text

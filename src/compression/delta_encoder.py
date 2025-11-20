"""Delta encoder for prosody features."""

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DeltaEncoder:
    """Encode values as deltas from previous values."""
    
    def encode(self, values: np.ndarray) -> np.ndarray:
        """Encode values as deltas.
        
        Args:
            values: Input values
        
        Returns:
            Delta-encoded values (first value unchanged, rest are deltas)
        """
        if len(values) == 0:
            return values
        
        deltas = np.zeros_like(values)
        deltas[0] = values[0]  # Keep first value as-is
        
        # Calculate deltas
        deltas[1:] = np.diff(values)
        
        return deltas
    
    def decode(self, deltas: np.ndarray) -> np.ndarray:
        """Decode delta-encoded values.
        
        Args:
            deltas: Delta-encoded values
        
        Returns:
            Original values
        """
        if len(deltas) == 0:
            return deltas
        
        # Cumulative sum to reconstruct original values
        values = np.cumsum(deltas)
        
        return values

"""Speaker change detection."""

import numpy as np
from scipy.spatial.distance import cosine

from src.utils.logger import get_logger

logger = get_logger(__name__)


def detect_speaker_change(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    threshold: float = 0.3
) -> bool:
    """Detect if speaker has changed based on embeddings.
    
    Args:
        embedding1: First speaker embedding
        embedding2: Second speaker embedding
        threshold: Cosine distance threshold
    
    Returns:
        True if speaker changed, False otherwise
    """
    try:
        distance = cosine(embedding1, embedding2)
        return distance > threshold
    except Exception as e:
        logger.error(f"Error detecting speaker change: {e}")
        return False

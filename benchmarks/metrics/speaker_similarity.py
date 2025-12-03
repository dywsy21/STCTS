"""Speaker similarity metrics for voice quality assessment."""

import numpy as np
from typing import Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def calculate_speaker_similarity(
    original_audio: np.ndarray,
    reconstructed_audio: np.ndarray,
    sample_rate: int = 16000
) -> dict:
    """Calculate speaker similarity between original and reconstructed audio.
    
    Uses speaker embedding similarity (cosine similarity) to measure
    how well the speaker identity is preserved.
    
    Args:
        original_audio: Original audio signal
        reconstructed_audio: Reconstructed audio signal  
        sample_rate: Audio sample rate
        
    Returns:
        Dictionary with similarity metrics
    """
    from src.speaker import SpeakerEmbedding
    
    logger.info("Extracting speaker embeddings...")
    speaker_model = SpeakerEmbedding()
    
    # Extract embeddings
    original_embedding = speaker_model.extract(original_audio, sample_rate)
    reconstructed_embedding = speaker_model.extract(reconstructed_audio, sample_rate)
    
    # Calculate similarity
    similarity = cosine_similarity(original_embedding, reconstructed_embedding)
    
    logger.info(f"Speaker similarity: {similarity:.4f}")
    
    # EER (Equal Error Rate) threshold is typically ~0.85 for same speaker
    is_same_speaker = similarity > 0.85
    
    return {
        'speaker_similarity': float(similarity),
        'is_same_speaker': bool(is_same_speaker),
        'original_embedding': original_embedding,
        'reconstructed_embedding': reconstructed_embedding
    }


def calculate_speaker_recognition_accuracy(
    embeddings_original: list,
    embeddings_reconstructed: list,
    labels: list
) -> dict:
    """Calculate speaker recognition accuracy for multiple speakers.
    
    Useful for evaluating if the system preserves speaker identity
    across multiple speakers.
    
    Args:
        embeddings_original: List of original speaker embeddings
        embeddings_reconstructed: List of reconstructed speaker embeddings
        labels: List of speaker labels (0, 1, 2, ...)
        
    Returns:
        Dictionary with recognition metrics
    """
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import normalize
    
    # Normalize embeddings
    emb_orig = normalize(np.array(embeddings_original))
    emb_recon = normalize(np.array(embeddings_reconstructed))
    
    # For each reconstructed, find closest original (nearest neighbor)
    predicted_labels = []
    for recon_emb in emb_recon:
        similarities = [cosine_similarity(recon_emb, orig_emb) for orig_emb in emb_orig]
        predicted_label = labels[np.argmax(similarities)]
        predicted_labels.append(predicted_label)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predicted_labels)
    f1 = f1_score(labels, predicted_labels, average='weighted')
    
    logger.info(f"Speaker recognition accuracy: {accuracy:.4f}")
    logger.info(f"Speaker recognition F1: {f1:.4f}")
    
    return {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'predicted_labels': predicted_labels
    }

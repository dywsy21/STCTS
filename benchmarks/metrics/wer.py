"""Word Error Rate (WER) metric for speech recognition quality."""

import numpy as np
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate between reference and hypothesis transcripts.
    
    WER = (S + D + I) / N
    where:
    - S = number of substitutions
    - D = number of deletions
    - I = number of insertions
    - N = number of words in reference
    
    Args:
        reference: Ground truth transcript
        hypothesis: Predicted transcript
        
    Returns:
        WER as a float (0.0 = perfect, higher = worse)
    """
    # Normalize texts
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # Build edit distance matrix
    n, m = len(ref_words), len(hyp_words)
    dp = np.zeros((n + 1, m + 1))
    
    # Initialize
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    # Fill matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                substitution = dp[i-1][j-1] + 1
                insertion = dp[i][j-1] + 1
                deletion = dp[i-1][j] + 1
                dp[i][j] = min(substitution, insertion, deletion)
    
    # Calculate WER
    if n == 0:
        return 0.0 if m == 0 else float('inf')
    
    wer = dp[n][m] / n
    return float(wer)


def calculate_wer_from_audio(
    original_audio: np.ndarray,
    reconstructed_audio: np.ndarray,
    sample_rate: int = 16000,
    model_name: str = "distil-large-v3"
) -> dict:
    """Calculate WER by transcribing both original and reconstructed audio.
    
    Args:
        original_audio: Original audio signal
        reconstructed_audio: Reconstructed audio signal
        sample_rate: Audio sample rate
        model_name: Whisper model to use for transcription
        
    Returns:
        Dictionary with WER metrics
    """
    from src.stt import FasterWhisperSTT
    
    logger.info("Transcribing original audio...")
    stt = FasterWhisperSTT(model_size=model_name)
    
    # Transcribe original
    original_result = stt.transcribe(original_audio, sample_rate)
    original_text = original_result.text
    logger.info(f"Original: {original_text[:100]}...")
    
    # Transcribe reconstructed
    logger.info("Transcribing reconstructed audio...")
    reconstructed_result = stt.transcribe(reconstructed_audio, sample_rate)
    reconstructed_text = reconstructed_result.text
    logger.info(f"Reconstructed: {reconstructed_text[:100]}...")
    
    # Calculate WER
    wer = calculate_wer(original_text, reconstructed_text)
    
    return {
        'wer': wer,
        'original_text': original_text,
        'reconstructed_text': reconstructed_text,
        'original_words': len(original_text.split()),
        'reconstructed_words': len(reconstructed_text.split())
    }

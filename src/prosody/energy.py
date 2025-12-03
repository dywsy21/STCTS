"""Energy extraction and normalization."""

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_energy(audio: np.ndarray, sample_rate: int, frame_length_ms: int = 40, hop_length_ms: int = 10) -> np.ndarray:
    """Extract RMS energy contour from audio.
    
    Args:
        audio: Audio data
        sample_rate: Sample rate
        frame_length_ms: Frame length in ms (default 40ms per paper)
        hop_length_ms: Hop length in ms (default 10ms per paper)
    
    Returns:
        Energy contour
    """
    try:
        frame_length = int(sample_rate * frame_length_ms / 1000)
        hop_length = int(sample_rate * hop_length_ms / 1000)
        
        if len(audio) < frame_length:
            return np.array([])

        # Calculate RMS energy per frame
        num_frames = (len(audio) - frame_length) // hop_length + 1
        energy = np.zeros(num_frames)
        
        # Vectorized implementation for speed
        # (Simple loop is fine for short chunks, but let's be robust)
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            frame = audio[start:end]
            energy[i] = np.sqrt(np.mean(frame ** 2))
        
        return energy
    
    except Exception as e:
        logger.error(f"Error extracting energy: {e}")
        return np.array([])


class EnergyNormalizer:
    """Dynamic range normalization for energy."""
    
    def __init__(self, window_size: int = 1000):
        # Window size for percentile calculation (e.g., 1000 frames = 10 seconds at 100Hz)
        self.window_size = window_size
        self.history = []
        self.e_min = 1e-6
        self.e_max = 1.0
        self.epsilon = 1e-6

    def normalize(self, energy: np.ndarray) -> np.ndarray:
        """Normalize energy to speaker's dynamic range.
        
        Eq 3:
        E_hat[t] = (log(E[t] + eps) - log(E_min)) / (log(E_max) - log(E_min))
        """
        if len(energy) == 0:
            return energy
            
        # Update history
        self.history.extend(energy.tolist())
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]
        
        # Update stats (5th and 95th percentiles)
        if len(self.history) > 10:
            self.e_min = np.percentile(self.history, 5)
            self.e_max = np.percentile(self.history, 95)
            
            # Ensure valid range
            if self.e_min < self.epsilon:
                self.e_min = self.epsilon
            if self.e_max <= self.e_min:
                self.e_max = self.e_min + 1e-6
            
        # Apply normalization
        log_e = np.log(energy + self.epsilon)
        log_min = np.log(self.e_min)
        log_max = np.log(self.e_max)
        
        normalized = (log_e - log_min) / (log_max - log_min)
        
        # Clip to [0, 1]
        return np.clip(normalized, 0.0, 1.0)

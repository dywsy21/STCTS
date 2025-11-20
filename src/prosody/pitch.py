"""Pitch extraction and normalization."""

import numpy as np
import parselmouth

from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_pitch(audio: np.ndarray, sample_rate: int, time_step: float = 0.01) -> np.ndarray:
    """Extract pitch contour from audio using Parselmouth (Praat).
    
    Args:
        audio: Audio data
        sample_rate: Sample rate
        time_step: Time step for pitch extraction (seconds)
    
    Returns:
        Pitch contour in Hz (0 for unvoiced)
    """
    try:
        # Create Praat sound object
        sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
        
        # Extract pitch
        pitch = sound.to_pitch(time_step=time_step)
        
        # Get pitch values
        pitch_values = pitch.selected_array['frequency']
        
        # Replace unvoiced (0) with 0 (already 0 in selected_array usually, but ensure)
        # The paper says: "For unvoiced frames... we set F0[t] = 0"
        pitch_values[pitch_values == 0] = 0.0
        
        return pitch_values
    
    except Exception as e:
        logger.error(f"Error extracting pitch: {e}")
        return np.array([])


class PitchNormalizer:
    """Log-scale and z-score normalization for pitch."""
    
    def __init__(self, calibration_seconds: float = 3.0):
        self.calibration_seconds = calibration_seconds
        self.buffer = []
        self.mu_f0 = None
        self.sigma_f0 = None
        self.is_calibrated = False
        self.total_buffered_duration = 0.0

    def normalize(self, pitch: np.ndarray) -> np.ndarray:
        """Normalize pitch trajectory.
        
        Eq 1:
        F_hat[t] = (log(F0[t]) - log(mu)) / sigma  if F0[t] > 0
                 = 0                               otherwise
        """
        if len(pitch) == 0:
            return pitch

        # Filter unvoiced for statistics
        voiced = pitch[pitch > 0]
        
        # Calibration phase
        if not self.is_calibrated:
            if len(voiced) > 0:
                self.buffer.extend(voiced.tolist())
                # Estimate duration based on frame count (assuming 10ms frame shift)
                self.total_buffered_duration += len(pitch) * 0.01 
            
            if self.total_buffered_duration >= self.calibration_seconds and len(self.buffer) > 0:
                # Compute baseline statistics in log domain
                log_f0 = np.log(np.array(self.buffer))
                self.mu_f0 = np.mean(log_f0)
                self.sigma_f0 = np.std(log_f0)
                if self.sigma_f0 < 1e-6:
                    self.sigma_f0 = 1.0 # Avoid division by zero
                self.is_calibrated = True
                logger.info(f"Pitch calibrated: mu={self.mu_f0:.2f}, sigma={self.sigma_f0:.2f}")
                self.buffer = [] # Clear buffer
        
        # If not yet calibrated, return zeros or raw log (here we return zeros to be safe/consistent)
        if not self.is_calibrated:
            return np.zeros_like(pitch)
            
        # Apply normalization
        normalized = np.zeros_like(pitch)
        mask = pitch > 0
        
        if np.any(mask):
            # Log-scale and z-score normalize
            normalized[mask] = (np.log(pitch[mask]) - self.mu_f0) / self.sigma_f0
            
        return normalized

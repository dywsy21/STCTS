"""Speaking rate estimation and normalization."""

import numpy as np
from scipy import signal

from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_speaking_rate(audio: np.ndarray, sample_rate: int, hop_length_ms: int = 10) -> np.ndarray:
    """Estimate instantaneous speaking rate trajectory.
    
    Algorithm:
    1. Bandpass filter (300-3000 Hz)
    2. Extract envelope
    3. Detect syllable nuclei (peaks)
    4. Count nuclei in sliding window (1s)
    
    Args:
        audio: Audio data
        sample_rate: Sample rate
        hop_length_ms: Output frame hop in ms (default 10ms to match pitch/energy)
    
    Returns:
        Speaking rate trajectory (syllables/sec)
    """
    try:
        if len(audio) == 0:
            return np.array([])

        # 1. Bandpass filter (300-3000 Hz) to emphasize speech energy
        sos = signal.butter(4, [300, 3000], btype='bandpass', fs=sample_rate, output='sos')
        filtered = signal.sosfilt(sos, audio)
        
        # 2. Extract envelope (Hilbert transform or simple rectification + lowpass)
        # Using simple rectification + smoothing for efficiency
        envelope = np.abs(filtered)
        # Smooth envelope (e.g., 50ms window)
        smooth_window = int(sample_rate * 0.05)
        envelope = np.convolve(envelope, np.ones(smooth_window)/smooth_window, mode='same')
        
        # 3. Detect peaks (syllable nuclei)
        # Adaptive thresholding could be used, here we use relative height
        peaks, _ = signal.find_peaks(envelope, distance=int(sample_rate*0.1), height=np.mean(envelope))
        
        # Create a binary impulse train of nuclei
        nuclei_train = np.zeros_like(audio)
        nuclei_train[peaks] = 1.0
        
        # 4. Count nuclei in sliding window (1 second)
        # We want output at 100Hz (10ms hop)
        hop_length = int(sample_rate * hop_length_ms / 1000)
        window_size = int(sample_rate * 1.0) # 1 second window
        
        # Convolve impulse train with rectangular window of size 1s
        # This effectively counts peaks in the window centered at each sample
        # Result is syllables per second (since window is 1s)
        rate_continuous = np.convolve(nuclei_train, np.ones(window_size), mode='same')
        
        # Downsample to target frame rate
        num_frames = (len(audio) - 1) // hop_length + 1
        rate_trajectory = np.zeros(num_frames)
        
        for i in range(num_frames):
            idx = i * hop_length
            if idx < len(rate_continuous):
                rate_trajectory[i] = rate_continuous[idx]
                
        return rate_trajectory
    
    except Exception as e:
        logger.error(f"Error extracting speaking rate: {e}")
        return np.array([])


class RateNormalizer:
    """Z-score normalization for speaking rate."""
    
    def __init__(self, calibration_seconds: float = 5.0):
        self.calibration_seconds = calibration_seconds
        self.buffer = []
        self.mu_r = 4.0 # Default fallback (syllables/sec)
        self.sigma_r = 1.0
        self.is_calibrated = False
        self.total_buffered_duration = 0.0

    def normalize(self, rate: np.ndarray) -> np.ndarray:
        """Normalize speaking rate.
        
        Eq 5:
        R_hat[t] = (R[t] - mu_R) / sigma_R
        """
        if len(rate) == 0:
            return rate
            
        # Calibration
        if not self.is_calibrated:
            # Only consider active speech segments (rate > 0)
            active_rate = rate[rate > 0.5] # Threshold to ignore silence
            if len(active_rate) > 0:
                self.buffer.extend(active_rate.tolist())
                self.total_buffered_duration += len(rate) * 0.01
            
            if self.total_buffered_duration >= self.calibration_seconds and len(self.buffer) > 0:
                self.mu_r = np.mean(self.buffer)
                self.sigma_r = np.std(self.buffer)
                if self.sigma_r < 0.1:
                    self.sigma_r = 0.1
                self.is_calibrated = True
                logger.info(f"Rate calibrated: mu={self.mu_r:.2f}, sigma={self.sigma_r:.2f}")
                self.buffer = []
        
        # Normalize
        normalized = (rate - self.mu_r) / self.sigma_r
        return normalized

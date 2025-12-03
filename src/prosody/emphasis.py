"""Emphasis and stress detection from audio."""

from typing import List, Tuple
import numpy as np
from scipy import signal
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EmphasisRegion:
    """Emphasis region in audio."""
    start_time: float  # Start time in seconds
    end_time: float    # End time in seconds
    strength: float    # Emphasis strength (0-1)
    word_index: int = -1  # Word index if available


class EmphasisDetector:
    """Detect emphasis and stress in speech."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: float = 0.025,  # 25ms
        frame_shift: float = 0.010,   # 10ms
        emphasis_threshold: float = 0.6
    ):
        """Initialize emphasis detector.
        
        Args:
            sample_rate: Audio sample rate
            frame_length: Frame length in seconds
            frame_shift: Frame shift in seconds
            emphasis_threshold: Threshold for emphasis detection
        """
        self.sample_rate = sample_rate
        self.frame_length = int(frame_length * sample_rate)
        self.frame_shift = int(frame_shift * sample_rate)
        self.emphasis_threshold = emphasis_threshold
        logger.info("Emphasis detector initialized")
    
    def detect(self, audio: np.ndarray, words: List[Tuple[str, float, float]] = None) -> List[EmphasisRegion]:
        """Detect emphasis regions in audio.
        
        Args:
            audio: Audio data (float32, -1 to 1)
            words: Optional list of (word, start_time, end_time) tuples
        
        Returns:
            List of emphasis regions
        """
        # Calculate energy envelope
        energy = self._calculate_energy_envelope(audio)
        
        # Smooth energy
        smoothed_energy = self._smooth_signal(energy)
        
        # Normalize energy
        if np.max(smoothed_energy) > 0:
            normalized_energy = smoothed_energy / np.max(smoothed_energy)
        else:
            normalized_energy = smoothed_energy
        
        # Detect peaks
        peaks = self._find_peaks(normalized_energy)
        
        # Convert to emphasis regions
        regions = []
        for peak_idx, strength in peaks:
            start_time = (peak_idx * self.frame_shift) / self.sample_rate
            end_time = ((peak_idx + 1) * self.frame_shift) / self.sample_rate
            
            # Extend region slightly
            start_time = max(0, start_time - 0.05)
            end_time = min(len(audio) / self.sample_rate, end_time + 0.05)
            
            region = EmphasisRegion(
                start_time=start_time,
                end_time=end_time,
                strength=strength
            )
            
            # Match to words if provided
            if words:
                region.word_index = self._find_word_index(start_time, end_time, words)
            
            regions.append(region)
        
        logger.debug(f"Detected {len(regions)} emphasis regions")
        return regions
    
    def _calculate_energy_envelope(self, audio: np.ndarray) -> np.ndarray:
        """Calculate energy envelope.
        
        Args:
            audio: Audio signal
        
        Returns:
            Energy envelope
        """
        # Frame the signal
        n_frames = (len(audio) - self.frame_length) // self.frame_shift + 1
        energy = np.zeros(n_frames)
        
        for i in range(n_frames):
            start = i * self.frame_shift
            end = start + self.frame_length
            frame = audio[start:end]
            
            # Calculate RMS energy
            energy[i] = np.sqrt(np.mean(frame ** 2))
        
        return energy
    
    def _smooth_signal(self, signal_data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Smooth signal with moving average.
        
        Args:
            signal_data: Input signal
            window_size: Window size for smoothing
        
        Returns:
            Smoothed signal
        """
        if len(signal_data) < window_size:
            return signal_data
        
        window = np.ones(window_size) / window_size
        smoothed = np.convolve(signal_data, window, mode='same')
        return smoothed
    
    def _find_peaks(self, energy: np.ndarray) -> List[Tuple[int, float]]:
        """Find energy peaks indicating emphasis.
        
        Args:
            energy: Energy signal
        
        Returns:
            List of (peak_index, strength) tuples
        """
        # Find peaks using scipy
        peak_indices, properties = signal.find_peaks(
            energy,
            height=self.emphasis_threshold,
            distance=int(0.2 * self.sample_rate / self.frame_shift)  # Min 200ms apart
        )
        
        peaks = []
        for idx in peak_indices:
            strength = float(energy[idx])
            peaks.append((int(idx), strength))
        
        return peaks
    
    def _find_word_index(
        self,
        start_time: float,
        end_time: float,
        words: List[Tuple[str, float, float]]
    ) -> int:
        """Find which word this emphasis region belongs to.
        
        Args:
            start_time: Region start time
            end_time: Region end time
            words: List of (word, start, end) tuples
        
        Returns:
            Word index, or -1 if not found
        """
        region_center = (start_time + end_time) / 2
        
        for i, (word, word_start, word_end) in enumerate(words):
            if word_start <= region_center <= word_end:
                return i
        
        return -1
    
    def detect_stressed_words(
        self,
        audio: np.ndarray,
        words: List[Tuple[str, float, float]]
    ) -> List[int]:
        """Detect which words are stressed/emphasized.
        
        Args:
            audio: Audio data
            words: List of (word, start_time, end_time) tuples
        
        Returns:
            List of word indices that are emphasized
        """
        regions = self.detect(audio, words)
        
        # Get unique word indices
        emphasized_words = set()
        for region in regions:
            if region.word_index >= 0:
                emphasized_words.add(region.word_index)
        
        return sorted(list(emphasized_words))


def extract_emphasis(
    audio: np.ndarray,
    sample_rate: int = 16000,
    words: List[Tuple[str, float, float]] = None
) -> List[EmphasisRegion]:
    """Extract emphasis regions from audio (simple interface).
    
    Args:
        audio: Audio data
        sample_rate: Sample rate
        words: Optional word timestamps
    
    Returns:
        List of emphasis regions
    """
    detector = EmphasisDetector(sample_rate=sample_rate)
    return detector.detect(audio, words)


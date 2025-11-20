"""Prosody compression."""

import numpy as np

from src.compression.quantizer import Quantizer
from src.compression.delta_encoder import DeltaEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ProsodyCompressor:
    """Compress prosody features using sparse keyframes and delta encoding."""
    
    def __init__(
        self,
        pitch_bits: int = 6,
        energy_bits: int = 5,
        rate_bits: int = 5,
        keyframe_rate: float = 1.0, # Hz
        input_rate: float = 100.0, # Hz
    ):
        """Initialize prosody compressor.
        
        Args:
            pitch_bits: Bits for pitch quantization
            energy_bits: Bits for energy quantization
            rate_bits: Bits for speaking rate quantization
            keyframe_rate: Rate of keyframes (sparse updates)
            input_rate: Input feature rate
        """
        self.quantizer = Quantizer()
        self.delta_encoder = DeltaEncoder()
        
        self.pitch_bits = pitch_bits
        self.energy_bits = energy_bits
        self.rate_bits = rate_bits
        
        self.keyframe_interval = int(input_rate / keyframe_rate)
        
        # Sender state
        self.last_pitch_keyframe = 0.0
        self.last_energy_keyframe = 0.0
        self.last_rate_keyframe = 0.0
        
        # Receiver state
        self.last_pitch_keyframe_rx = 0.0
        self.last_energy_keyframe_rx = 0.0
        self.last_rate_keyframe_rx = 0.0
        
        logger.info(
            f"Prosody compressor: pitch={pitch_bits}b, energy={energy_bits}b, "
            f"rate={rate_bits}b, keyframe_rate={keyframe_rate}Hz"
        )
    
    def _sample_keyframes(self, values: np.ndarray) -> np.ndarray:
        """Sample keyframes from continuous trajectory."""
        if len(values) == 0:
            return np.array([])
            
        # Simple decimation
        # In a real stream, we would need to maintain phase/counter
        # Here we assume each chunk starts aligned or we just take indices
        indices = np.arange(0, len(values), self.keyframe_interval)
        return values[indices]

    def compress_pitch(self, pitch: np.ndarray) -> bytes:
        """Compress pitch contour (sparse).
        
        Args:
            pitch: Normalized pitch contour (z-score)
        
        Returns:
            Compressed pitch data
        """
        # 1. Sparse Sampling
        keyframes = self._sample_keyframes(pitch)
        if len(keyframes) == 0:
            return b""
            
        # 2. Delta Encoding (Keyframe to Keyframe)
        # We need to handle state across chunks
        deltas = np.zeros_like(keyframes)
        deltas[0] = keyframes[0] - self.last_pitch_keyframe
        deltas[1:] = np.diff(keyframes)
        
        # Update state
        self.last_pitch_keyframe = keyframes[-1]
        
        # 3. Non-uniform Quantization (Deadzone)
        # Pitch is z-score normalized, so range is approx -3 to 3
        # We use quantize_deadzone
        quantized = self.quantizer.quantize_deadzone(
            deltas, 
            self.pitch_bits, 
            deadzone_threshold=0.05,
            scale_factor=1.0
        )
        
        # 4. Pack to bytes
        return self.quantizer.pack_to_bytes(quantized, self.pitch_bits)
    
    def decompress_pitch(self, data: bytes, length: int) -> np.ndarray:
        """Decompress pitch contour (with interpolation).
        
        Args:
            data: Compressed data
            length: Expected output length (frames)
        
        Returns:
            Pitch contour (interpolated)
        """
        # Unpack
        num_keyframes = (length + self.keyframe_interval - 1) // self.keyframe_interval
        if num_keyframes == 0:
            return np.zeros(length)

        quantized = self.quantizer.unpack_from_bytes(data, self.pitch_bits, num_keyframes)
        
        # Dequantize
        deltas = self.quantizer.dequantize_deadzone(quantized, self.pitch_bits)
        
        # Delta Decode (Reconstruct Keyframes)
        keyframes = np.zeros_like(deltas)
        current_val = self.last_pitch_keyframe_rx
        
        for i in range(len(deltas)):
            current_val += deltas[i]
            keyframes[i] = current_val
            
        self.last_pitch_keyframe_rx = current_val
        
        # Interpolate to full rate (Linear)
        x_indices = np.arange(0, len(keyframes) * self.keyframe_interval, self.keyframe_interval)
        target_indices = np.arange(length)
        
        # Linear interpolation for robustness
        # Note: This assumes keyframes are at 0, interval, 2*interval...
        # Ideally we should know the exact timing, but for fixed rate this is approx correct.
        full_contour = np.interp(target_indices, x_indices, keyframes)
        
        return full_contour

    def compress_energy(self, energy: np.ndarray) -> bytes:
        """Compress energy contour (sparse)."""
        keyframes = self._sample_keyframes(energy)
        if len(keyframes) == 0:
            return b""
        
        deltas = np.zeros_like(keyframes)
        deltas[0] = keyframes[0] - self.last_energy_keyframe
        deltas[1:] = np.diff(keyframes)
        self.last_energy_keyframe = keyframes[-1]
        
        # Energy is [0, 1], deltas are [-1, 1]
        quantized = self.quantizer.quantize_deadzone(deltas, self.energy_bits, deadzone_threshold=0.02)
        return self.quantizer.pack_to_bytes(quantized, self.energy_bits)

    def decompress_energy(self, data: bytes, length: int) -> np.ndarray:
        """Decompress energy contour."""
        num_keyframes = (length + self.keyframe_interval - 1) // self.keyframe_interval
        if num_keyframes == 0:
            return np.zeros(length)
        
        quantized = self.quantizer.unpack_from_bytes(data, self.energy_bits, num_keyframes)
        deltas = self.quantizer.dequantize_deadzone(quantized, self.energy_bits)
        
        keyframes = np.zeros_like(deltas)
        current_val = self.last_energy_keyframe_rx
        
        for i in range(len(deltas)):
            current_val += deltas[i]
            keyframes[i] = current_val
            
        self.last_energy_keyframe_rx = current_val
        
        x_indices = np.arange(0, len(keyframes) * self.keyframe_interval, self.keyframe_interval)
        target_indices = np.arange(length)
        return np.interp(target_indices, x_indices, keyframes)


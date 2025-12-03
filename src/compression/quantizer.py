"""Quantizer for feature compression."""

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Quantizer:
    """Quantize continuous values to discrete levels."""
    
    def quantize(
        self,
        values: np.ndarray,
        num_bits: int,
        min_val: float,
        max_val: float,
    ) -> np.ndarray:
        """Quantize values to specified bit depth (Linear Uniform).
        
        Args:
            values: Input values
            num_bits: Number of bits for quantization
            min_val: Minimum value of range
            max_val: Maximum value of range
        
        Returns:
            Quantized values (integers)
        """
        # Clip to range
        values = np.clip(values, min_val, max_val)
        
        # Normalize to [0, 1]
        normalized = (values - min_val) / (max_val - min_val)
        
        # Quantize
        num_levels = 2 ** num_bits
        quantized = np.round(normalized * (num_levels - 1)).astype(np.int32)
        
        return quantized

    def quantize_deadzone(
        self,
        values: np.ndarray,
        num_bits: int,
        deadzone_threshold: float = 0.05,
        scale_factor: float = 1.0
    ) -> np.ndarray:
        """Quantize values using dead-zone uniform quantizer.
        
        Eq 7:
        q[t] = 0 if |x| < threshold
             = sign(x) * ceil(|x| / step) otherwise
             
        Args:
            values: Input values (deltas)
            num_bits: Number of bits
            deadzone_threshold: Threshold below which values are quantized to 0
            scale_factor: Scaling factor (step size alpha)
            
        Returns:
            Quantized integer indices
        """
        # Deadzone
        mask_zero = np.abs(values) < deadzone_threshold
        
        # Quantization step size
        # Max range is roughly +/- 3 sigma (since inputs are z-scores)
        # With num_bits, we have 2^(num_bits-1) levels per side
        max_range = 3.0
        num_levels = 2 ** (num_bits - 1)
        step_size = max_range / num_levels * scale_factor
        
        # Quantize
        # q = sign(x) * ceil(|x| / step)
        quantized = np.sign(values) * np.ceil(np.abs(values) / step_size)
        
        # Apply deadzone
        quantized[mask_zero] = 0
        
        # Clip to available bits range
        max_int = 2 ** (num_bits - 1) - 1
        min_int = -(2 ** (num_bits - 1))
        quantized = np.clip(quantized, min_int, max_int).astype(np.int32)
        
        return quantized

    def dequantize_deadzone(
        self,
        quantized: np.ndarray,
        num_bits: int,
        scale_factor: float = 1.0
    ) -> np.ndarray:
        """Dequantize dead-zone values.
        
        Args:
            quantized: Quantized integer values
            num_bits: Number of bits
            scale_factor: Scaling factor
            
        Returns:
            Reconstructed values
        """
        max_range = 3.0
        num_levels = 2 ** (num_bits - 1)
        step_size = max_range / num_levels * scale_factor
        
        # Reconstruction
        # x_hat = q * step
        # (Simple reconstruction, could be improved with centroid)
        values = quantized * step_size
        
        return values
    
    def dequantize(
        self,
        quantized: np.ndarray,
        num_bits: int,
        min_val: float,
        max_val: float,
    ) -> np.ndarray:
        """Dequantize values.
        
        Args:
            quantized: Quantized integer values
            num_bits: Number of bits used for quantization
            min_val: Minimum value of range
            max_val: Maximum value of range
        
        Returns:
            Dequantized values (floats)
        """
        num_levels = 2 ** num_bits
        
        # Normalize
        normalized = quantized.astype(np.float32) / (num_levels - 1)
        
        # Scale back
        values = normalized * (max_val - min_val) + min_val
        
        return values
    
    def pack_to_bytes(self, quantized: np.ndarray, num_bits: int) -> bytes:
        """Pack quantized values into bytes using bit-packing.
        
        Args:
            quantized: Quantized integer values
            num_bits: Number of bits per value
        
        Returns:
            Packed bytes
        """
        if len(quantized) == 0:
            return b""
        
        # Handle 0 bits - return empty bytes
        if num_bits == 0:
            return b""
        
        # Use proper bit-packing for efficiency
        if num_bits == 8:
            return quantized.astype(np.uint8).tobytes()
        elif num_bits >= 16:
            return quantized.astype(np.uint16).tobytes()
        
        # For num_bits < 8, pack multiple values per byte
        max_value = (1 << num_bits) - 1  # Maximum value for num_bits
        quantized_clipped = np.clip(quantized, 0, max_value).astype(np.uint32)
        
        # Pack bits into a byte array
        bit_string = ''.join(format(val, f'0{num_bits}b') for val in quantized_clipped)
        
        # Pad to byte boundary
        remainder = len(bit_string) % 8
        if remainder != 0:
            bit_string += '0' * (8 - remainder)
        
        # Convert bit string to bytes
        packed_bytes = bytearray()
        for i in range(0, len(bit_string), 8):
            byte_str = bit_string[i:i+8]
            packed_bytes.append(int(byte_str, 2))
        
        return bytes(packed_bytes)
    
    def unpack_from_bytes(self, data: bytes, num_bits: int, length: int) -> np.ndarray:
        """Unpack bytes into quantized values using bit-unpacking.
        
        Args:
            data: Packed bytes
            num_bits: Number of bits per value
            length: Number of values to unpack
        
        Returns:
            Quantized values
        """
        if len(data) == 0:
            return np.array([], dtype=np.int32)
        
        # Handle 0 bits - return zeros
        if num_bits == 0:
            return np.zeros(length, dtype=np.int32)
        
        # Use proper bit-unpacking
        if num_bits == 8:
            return np.frombuffer(data, dtype=np.uint8, count=length).astype(np.int32)
        elif num_bits >= 16:
            return np.frombuffer(data, dtype=np.uint16, count=length).astype(np.int32)
        
        # For num_bits < 8, unpack multiple values per byte
        # Convert bytes to bit string
        bit_string = ''.join(format(byte, '08b') for byte in data)
        
        # Extract values
        values = []
        for i in range(length):
            start_bit = i * num_bits
            end_bit = start_bit + num_bits
            if end_bit <= len(bit_string):
                value_bits = bit_string[start_bit:end_bit]
                values.append(int(value_bits, 2))
            else:
                # Ran out of bits, pad with zeros
                values.append(0)
        
        return np.array(values, dtype=np.int32)

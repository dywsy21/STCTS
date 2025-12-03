"""Noise injection utilities for bitstream resilience testing."""

import numpy as np
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


def inject_bit_errors(data: bytes, bit_error_rate: float, seed: Optional[int] = None) -> bytes:
    """Inject random bit errors into a byte stream.
    
    Args:
        data: Input byte stream
        bit_error_rate: Probability of bit flip (e.g., 0.001 for 0.1%)
        seed: Random seed for reproducibility
        
    Returns:
        Corrupted byte stream
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert bytes to bit array
    byte_array = np.frombuffer(data, dtype=np.uint8)
    
    # Create a copy to modify
    corrupted_array = byte_array.copy()
    
    # Calculate number of bits
    num_bits = len(byte_array) * 8
    
    # Generate random bit positions to flip
    num_errors = int(num_bits * bit_error_rate)
    
    if num_errors == 0:
        logger.info(f"No errors to inject (BER={bit_error_rate:.2e}, {num_bits} bits)")
        return data
    
    # Random bit positions to corrupt
    bit_positions = np.random.choice(num_bits, size=num_errors, replace=False)
    
    # Flip bits
    for bit_pos in bit_positions:
        byte_idx = bit_pos // 8
        bit_idx = bit_pos % 8
        
        # Flip the bit
        corrupted_array[byte_idx] ^= (1 << bit_idx)
    
    logger.info(f"💥 Injected {num_errors} bit errors ({bit_error_rate:.2e} BER) into {len(data)} bytes")
    
    return corrupted_array.tobytes()


def inject_errors_into_components(
    text_compressed: bytes,
    prosody_packets: list,
    timbre_compressed: bytes,
    bit_error_rate: float,
    seed: Optional[int] = None
) -> tuple:
    """Inject bit errors into all compressed components.
    
    Note: Text compression is skipped because bit errors in compressed text
    (Brotli/LZ4) typically cause complete decompression failure. We only
    inject errors into prosody and timbre components which are more resilient.
    
    Args:
        text_compressed: Compressed text bytes (NOT corrupted)
        prosody_packets: List of compressed prosody packets
        timbre_compressed: Compressed timbre bytes
        bit_error_rate: Bit error rate to apply
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (text, corrupted_prosody_packets, corrupted_timbre)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # DO NOT inject errors into text - it causes decompression failure
    # Text compression algorithms (Brotli/LZ4) are not error-resilient
    corrupted_text = text_compressed
    
    logger.info("⚠️  Skipping text corruption (would cause decompression failure)")
    
    # Inject errors into each prosody packet
    corrupted_prosody_packets = []
    for i, packet in enumerate(prosody_packets):
        corrupted_packet = {}
        for key, value in packet.items():
            if isinstance(value, bytes):
                # Use different seed for each packet component
                packet_seed = seed + i * 100 + hash(key) % 100 if seed is not None else None
                corrupted_packet[key] = inject_bit_errors(value, bit_error_rate, packet_seed)
            else:
                # Non-bytes values (metadata) are not corrupted
                corrupted_packet[key] = value
        corrupted_prosody_packets.append(corrupted_packet)
    
    # Inject errors into timbre
    timbre_seed = seed + 10000 if seed is not None else None
    corrupted_timbre = inject_bit_errors(timbre_compressed, bit_error_rate, timbre_seed)
    
    return corrupted_text, corrupted_prosody_packets, corrupted_timbre


def get_ber_levels():
    """Get standard bit error rate levels for testing.
    
    Returns:
        List of (BER value, description string) tuples
    """
    return [
        (1e-3, "0.1%"),
        (1e-2, "1%"),
        (1e-1, "10%")
    ]

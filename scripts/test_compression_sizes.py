#!/usr/bin/env python3
"""Quick test to compare compression sizes between configs."""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.compression.prosody import ProsodyCompressor
from src.utils.config import load_config

# Test data: 100 random values
test_data = np.random.uniform(50, 400, 100)

configs = {
    "minimal": "configs/minimal_mode.yaml",
    "balanced": "configs/balanced_mode.yaml",
    "high_quality": "configs/high_quality_mode.yaml"
}

print("=" * 80)
print("COMPRESSION SIZE COMPARISON")
print("=" * 80)
print()

for name, config_path in configs.items():
    config = load_config(config_path)
    
    # Create compressor with config settings
    compressor = ProsodyCompressor(
        pitch_bits=config.compression.prosody_quantization_pitch_bits,
        energy_bits=config.compression.prosody_quantization_energy_bits,
        rate_bits=config.compression.prosody_quantization_rate_bits,
        use_delta_encoding=config.compression.prosody_delta_encoding
    )
    
    # Compress test data
    compressed = compressor.compress_pitch(test_data)
    
    print(f"{name.upper():15} | {config.compression.prosody_quantization_pitch_bits}-bit | {len(compressed):5} bytes")

print()
print("=" * 80)
print("Result: If sizes are different, configs are working correctly!")
print("        If sizes are the same, there's still a problem.")
print("=" * 80)

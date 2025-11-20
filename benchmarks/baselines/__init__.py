"""Baseline codec implementations for comparison.

This module contains implementations of standard audio codecs:

- Opus: Open-source low-latency codec (6-510 kbps)
- Encodec: Meta's neural audio codec (1.5-24 kbps)
- Vevo: (TODO) High-quality neural codec
"""

from .opus import OpusCodec
from .encodec import EncodecCodec

__all__ = [
    "OpusCodec",
    "EncodecCodec",
]

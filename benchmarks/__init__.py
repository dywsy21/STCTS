"""Benchmarking suite for STT-Compress-TTS."""

from benchmarks.metrics.wer import calculate_wer
from benchmarks.metrics.speaker_similarity import calculate_speaker_similarity
from benchmarks.metrics.perceptual_quality import (
    calculate_pesq,
    calculate_stoi,
    calculate_utmos,
    calculate_nisqa
)

__all__ = [
    'calculate_wer',
    'calculate_speaker_similarity',
    'calculate_pesq',
    'calculate_stoi',
    'calculate_utmos',
    'calculate_nisqa',
]

"""Benchmarking metrics module.

This module contains implementations of various metrics for evaluating
speech compression and reconstruction quality:

- Word Error Rate (WER): Intelligibility metric
- Speaker Similarity: Voice identity preservation
- Perceptual Quality: PESQ, STOI, UTMOS, NISQA
"""

from .wer import calculate_wer, calculate_wer_from_audio
from .speaker_similarity import (
    cosine_similarity,
    calculate_speaker_similarity,
    calculate_speaker_recognition_accuracy,
)
from .perceptual_quality import (
    calculate_pesq,
    calculate_stoi,
    calculate_utmos,
    calculate_nisqa,
    calculate_all_perceptual_metrics,
)

__all__ = [
    # WER
    "calculate_wer",
    "calculate_wer_from_audio",
    # Speaker Similarity
    "cosine_similarity",
    "calculate_speaker_similarity",
    "calculate_speaker_recognition_accuracy",
    # Perceptual Quality
    "calculate_pesq",
    "calculate_stoi",
    "calculate_utmos",
    "calculate_nisqa",
    "calculate_all_perceptual_metrics",
]

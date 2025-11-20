"""Base STT interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np


@dataclass
class STTResult:
    """STT transcription result."""
    text: str
    confidence: float
    start_time: float
    end_time: float
    language: Optional[str] = None
    words: Optional[list] = None


class BaseSTT(ABC):
    """Base class for Speech-to-Text implementations."""
    
    def __init__(self, model_name: str, language: Optional[str] = None):
        """Initialize STT model.
        
        Args:
            model_name: Name/path of the model
            language: Language code (optional)
        """
        self.model_name = model_name
        self.language = language
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the STT model."""
        pass
    
    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> STTResult:
        """Transcribe audio to text.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Audio sample rate
        
        Returns:
            Transcription result
        """
        pass
    
    @abstractmethod
    def transcribe_stream(
        self,
        audio_stream: Iterator[np.ndarray],
        sample_rate: int = 16000,
    ) -> Iterator[STTResult]:
        """Transcribe streaming audio.
        
        Args:
            audio_stream: Iterator of audio chunks
            sample_rate: Audio sample rate
        
        Yields:
            Transcription results
        """
        pass
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        self.is_loaded = False

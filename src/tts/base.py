"""Base TTS interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TTSResult:
    """TTS synthesis result."""
    audio: np.ndarray
    sample_rate: int
    duration: float


class BaseTTS(ABC):
    """Base class for Text-to-Speech implementations."""
    
    def __init__(self, model_name: str):
        """Initialize TTS model.
        
        Args:
            model_name: Name/path of the model
        """
        self.model_name = model_name
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the TTS model."""
        pass
    
    @abstractmethod
    def synthesize(
        self,
        text: str,
        speaker_embedding: Optional[np.ndarray] = None,
        prosody: Optional[dict] = None,
    ) -> TTSResult:
        """Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            speaker_embedding: Speaker embedding for voice cloning
            prosody: Prosody parameters (pitch, energy, rate)
        
        Returns:
            Synthesized audio
        """
        pass
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        self.is_loaded = False

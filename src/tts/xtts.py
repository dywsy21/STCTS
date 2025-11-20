"""XTTS-v2 TTS implementation."""

from typing import Optional

import numpy as np
import torch

from src.tts.base import BaseTTS, TTSResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class XTTSTS(BaseTTS):
    """Coqui XTTS-v2 implementation."""
    
    def __init__(self, device: str = "auto"):
        """Initialize XTTS model.
        
        Args:
            device: Device to run on (cpu, cuda, auto)
        """
        super().__init__("tts_models/multilingual/multi-dataset/xtts_v2")
        
        # Auto-detect device if set to auto
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = None
    
    def load_model(self) -> None:
        """Load the XTTS model."""
        if self.is_loaded:
            logger.info("Model already loaded")
            return
        
        logger.info(f"Loading XTTS-v2 model (device={self.device})")
        
        try:
            from TTS.api import TTS
            
            # Patch torch.load to use weights_only=False for TTS models
            # This is safe because we trust the TTS library from Coqui/HuggingFace
            import torch
            original_load = torch.load
            
            def patched_load(*args, **kwargs):
                # Force weights_only=False for TTS model loading
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            # Temporarily replace torch.load
            torch.load = patched_load
            
            try:
                self.model = TTS(self.model_name, gpu=(self.device == "cuda"))
                self.is_loaded = True
                logger.info("XTTS model loaded successfully")
            finally:
                # Restore original torch.load
                torch.load = original_load
                
        except Exception as e:
            logger.error(f"Error loading XTTS model: {e}")
            raise
    
    def synthesize(
        self,
        text: str,
        speaker_embedding: Optional[np.ndarray] = None,
        prosody: Optional[dict] = None,
        speaker_wav: Optional[str] = None,
    ) -> TTSResult:
        """Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            speaker_embedding: Speaker embedding (192-dim from ECAPA-TDNN) - not used directly
            prosody: Prosody parameters
            speaker_wav: Path to reference audio file for voice cloning (recommended)
        
        Returns:
            Synthesized audio
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            # XTTS-v2 is a voice cloning model that requires a reference audio file
            # If speaker_wav is provided, use it directly for voice cloning
            if speaker_wav:
                logger.info(f"Using provided reference audio: {speaker_wav}")
                audio = self.model.tts(
                    text=text,
                    language="en",
                    speaker_wav=speaker_wav
                )
            else:
                # No reference audio provided
                # XTTS will use a default/neutral voice or fail
                logger.warning("No speaker reference provided, using model defaults")
                # Try to synthesize without speaker (will likely fail for XTTS-v2)
                try:
                    audio = self.model.tts(text=text, language="en")
                except Exception as e:
                    logger.error(f"Failed to synthesize without speaker reference: {e}")
                    logger.info("Tip: Provide speaker_wav parameter with a reference audio file")
                    raise
            
            # Get sample rate
            sample_rate = self.model.synthesizer.output_sample_rate
            
            # Convert to numpy
            if isinstance(audio, list):
                audio = np.array(audio, dtype=np.float32)
            elif isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            
            duration = len(audio) / sample_rate
            
            logger.info(f"Synthesized {duration:.2f}s of audio at {sample_rate}Hz")
            
            return TTSResult(
                audio=audio,
                sample_rate=sample_rate,
                duration=duration,
            )
        
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            raise
    
    def unload_model(self) -> None:
        """Unload the model."""
        self.model = None
        self.is_loaded = False
        logger.info("XTTS model unloaded")

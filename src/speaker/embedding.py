"""Speaker embedding extraction."""

import numpy as np
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SpeakerEmbedding:
    """Extract speaker embeddings using SpeechBrain."""
    
    def __init__(self, model_name: str = "speechbrain/spkrec-ecapa-voxceleb", device: str = "auto"):
        """Initialize speaker embedding model.
        
        Args:
            model_name: SpeechBrain model name
            device: Device to run on (cpu, cuda, auto)
        """
        self.model_name = model_name
        
        # Auto-detect device if set to auto
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = None
        logger.info(f"Speaker embedding model: {model_name} (device={self.device})")
    
    def load_model(self):
        """Load the model."""
        if self.model is not None:
            return
        
        try:
            from speechbrain.inference import EncoderClassifier
            self.model = EncoderClassifier.from_hparams(
                source=self.model_name,
                run_opts={"device": self.device}
            )
            logger.info("Speaker embedding model loaded")
        except Exception as e:
            logger.error(f"Error loading speaker model: {e}")
            raise
    
    def extract(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract speaker embedding.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
        
        Returns:
            Speaker embedding vector
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model.encode_batch(audio_tensor)
            
            # Convert to numpy
            embedding_np = embedding.squeeze().cpu().numpy()
            
            return embedding_np
        
        except Exception as e:
            logger.error(f"Error extracting speaker embedding: {e}")
            return np.array([])

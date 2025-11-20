"""Emotion detection from audio."""

from typing import Optional, Dict
import numpy as np
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmotionDetector:
    """Detect emotion from speech using SpeechBrain."""
    
    # Emotion categories
    EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
    
    def __init__(self, model_name: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"):
        """Initialize emotion detector.
        
        Args:
            model_name: Model name from SpeechBrain hub
        """
        self.model_name = model_name
        self.classifier = None
        self._initialized = False
        logger.info(f"Emotion detector created with model: {model_name}")
    
    def _lazy_load(self):
        """Lazy load the model on first use."""
        if not self._initialized:
            try:
                from speechbrain.inference.classifiers import EncoderClassifier
                
                self.classifier = EncoderClassifier.from_hparams(
                    source=self.model_name,
                    savedir="./models/emotion"
                )
                self._initialized = True
                logger.info("Emotion model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load emotion model: {e}")
                self._initialized = False
    
    def detect(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict[str, float]:
        """Detect emotion from audio.
        
        Args:
            audio: Audio data (float32, -1 to 1)
            sample_rate: Sample rate
        
        Returns:
            Dictionary of emotion probabilities
        """
        self._lazy_load()
        
        if not self._initialized or self.classifier is None:
            # Return neutral if model not available
            return {"neutral": 1.0}
        
        try:
            # Convert to torch tensor
            if isinstance(audio, np.ndarray):
                audio_tensor = torch.from_numpy(audio).float()
            else:
                audio_tensor = audio
            
            # Ensure 1D
            if audio_tensor.dim() > 1:
                audio_tensor = audio_tensor.squeeze()
            
            # Add batch dimension
            audio_batch = audio_tensor.unsqueeze(0)
            wav_lens = torch.tensor([1.0])
            
            # Manual forward pass through the model pipeline
            # This model doesn't have compute_features, so we process manually
            with torch.no_grad():
                # 1. Extract features with wav2vec2
                feats = self.classifier.mods.wav2vec2(audio_batch, wav_lens)
                
                # 2. Apply pooling
                pooled = self.classifier.mods.avg_pool(feats, wav_lens)
                
                # 3. Get logits from output layer
                logits = self.classifier.mods.output_mlp(pooled)
                
                # 4. Apply softmax to get probabilities
                if hasattr(self.classifier.hparams, 'softmax'):
                    probs = self.classifier.hparams.softmax(logits)
                else:
                    # Manual softmax if not in hparams
                    probs = torch.nn.functional.softmax(logits, dim=1)
                
                # Convert to numpy
                probs_np = probs[0].cpu().numpy()
                
                # Ensure 1D array
                if probs_np.ndim > 1:
                    probs_np = probs_np.flatten()
            
            # Get emotion labels from label encoder
            if hasattr(self.classifier.hparams, 'label_encoder'):
                # Use ind2lab mapping directly
                encoder = self.classifier.hparams.label_encoder
                if hasattr(encoder, 'ind2lab'):
                    ind2lab = encoder.ind2lab
                    labels = [ind2lab.get(i, f"class{i}") for i in range(len(probs_np))]
                else:
                    labels = [f"class{i}" for i in range(len(probs_np))]
            else:
                # Fallback to default IEMOCAP labels
                labels = ["neu", "hap", "sad", "ang"][:len(probs_np)]
            
            # Create probability dict
            result = {}
            for i, (label, prob) in enumerate(zip(labels, probs_np)):
                # Convert numpy scalar to Python float
                prob_val = prob.item() if hasattr(prob, 'item') else float(prob)
                result[label] = prob_val
            
            return result
                
        except Exception as e:
            logger.warning(f"Emotion detection failed: {e}")
            return {"neutral": 1.0}
    
    def get_dominant_emotion(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Get the dominant emotion label.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
        
        Returns:
            Emotion label (e.g., "neutral", "happy")
        """
        emotions = self.detect(audio, sample_rate)
        return max(emotions.items(), key=lambda x: x[1])[0]


def extract_emotion(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """Extract emotion from audio (simple interface).
    
    Args:
        audio: Audio data
        sample_rate: Sample rate
    
    Returns:
        Emotion label
    """
    detector = EmotionDetector()
    return detector.get_dominant_emotion(audio, sample_rate)


"""Encodec baseline (Meta's neural audio codec)."""

import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EncodecCodec:
    """Encodec codec for baseline comparison."""
    
    def __init__(self, bandwidth: float = 1.5, sample_rate: int = 24000):
        """Initialize Encodec codec.
        
        Args:
            bandwidth: Target bandwidth in kbps (1.5, 3, 6, 12, 24)
            sample_rate: Audio sample rate (24kHz or 48kHz for Encodec)
        """
        self.bandwidth = bandwidth
        self.sample_rate = sample_rate
        self.model = None
        
    def _load_model(self):
        """Lazy load Encodec model."""
        if self.model is not None:
            return
            
        try:
            from encodec import EncodecModel
            from encodec.utils import convert_audio
            import torch
            
            logger.info(f"Loading Encodec model (bandwidth={self.bandwidth}kbps)...")
            
            # Load model
            if self.sample_rate == 24000:
                self.model = EncodecModel.encodec_model_24khz()
            else:
                self.model = EncodecModel.encodec_model_48khz()
            
            self.model.set_target_bandwidth(self.bandwidth)
            self.convert_audio = convert_audio
            self.torch = torch
            
            logger.info("Encodec model loaded")
            
        except ImportError:
            logger.error("encodec package not installed. Install with: pip install encodec")
            raise
    
    def encode_decode(self, audio: np.ndarray) -> tuple:
        """Encode and decode audio using Encodec.
        
        Args:
            audio: Input audio signal (mono)
            
        Returns:
            Tuple of (reconstructed_audio, compressed_size_bytes)
        """
        self._load_model()
        
        if self.model is None:
            return audio, 0
        
        try:
            import torch
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
            
            # Resample if needed (Encodec expects 24kHz or 48kHz)
            if self.sample_rate not in [24000, 48000]:
                import librosa
                audio_resampled = librosa.resample(
                    audio,
                    orig_sr=self.sample_rate,
                    target_sr=24000
                )
                audio_tensor = torch.from_numpy(audio_resampled).float().unsqueeze(0).unsqueeze(0)
                sr = 24000
            else:
                sr = self.sample_rate
            
            # Encode
            with torch.no_grad():
                encoded_frames = self.model.encode(audio_tensor)
            
            # Calculate compressed size
            # encoded_frames is a list of tuples: [(codes, scale), ...]
            # codes shape: [batch, num_codebooks, num_frames]
            codes, scale = encoded_frames[0]
            batch_size, num_codebooks, num_frames = codes.shape
            
            # The easiest and most accurate way: use the bandwidth directly
            # bandwidth is in kbps, duration is in seconds
            duration = len(audio) / self.sample_rate
            compressed_size = int((self.bandwidth * 1000 * duration) / 8)  # kbps -> bytes
            
            # Decode
            with torch.no_grad():
                reconstructed_tensor = self.model.decode(encoded_frames)
            
            # Convert back to numpy
            reconstructed = reconstructed_tensor.squeeze().cpu().numpy()
            
            # Resample back if needed
            if sr != self.sample_rate:
                import librosa
                reconstructed = librosa.resample(
                    reconstructed,
                    orig_sr=sr,
                    target_sr=self.sample_rate
                )
            
            logger.info(f"Encodec: {len(audio)/self.sample_rate:.1f}s audio -> {compressed_size}B")
            
            return reconstructed, compressed_size
            
        except Exception as e:
            logger.error(f"Error in Encodec codec: {e}")
            return audio, 0

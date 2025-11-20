"""Configuration settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # STT settings
    stt_model: str = "distil-large-v3"
    stt_device: str = "cpu"
    
    # TTS settings
    tts_model: str = "xtts-v2"
    tts_device: str = "cpu"
    
    # Compression settings
    text_algorithm: str = "brotli"
    text_level: int = 5
    prosody_quantization_pitch_bits: int = 6
    prosody_quantization_energy_bits: int = 4
    prosody_quantization_rate_bits: int = 4
    
    # Audio settings
    sample_rate: int = 16000
    chunk_size: int = 4000  # 250ms at 16kHz
    
    # Network settings
    signaling_server: str = "ws://localhost:8080"
    target_bitrate: int = 500  # bps
    
    # Quality mode
    quality_mode: str = "balanced"
    
    class Config:
        """Pydantic config."""
        extra = "ignore"
        env_prefix = "STT_"

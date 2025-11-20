"""Configuration management utilities."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class AudioConfig(BaseModel):
    """Audio configuration."""
    sample_rate: int = Field(16000, description="Audio sample rate in Hz")
    chunk_size: int = Field(1024, description="Audio chunk size")
    channels: int = Field(1, description="Number of audio channels")


class STTConfig(BaseModel):
    """STT configuration."""
    model: str = Field("faster-whisper", description="STT model type")
    model_size: str = Field("distil-large-v3", description="Model size")
    chunk_duration_ms: int = Field(400, description="Chunk duration in ms")
    vad_threshold: float = Field(0.5, description="VAD threshold")
    language: Optional[str] = Field(None, description="Language code")


class ProsodyConfig(BaseModel):
    """Prosody extraction configuration."""
    update_rate_hz: float = Field(2.5, description="Prosody update rate")
    features: list[str] = Field(
        ["pitch", "energy", "speaking_rate"],
        description="Features to extract"
    )
    emotion_rate_hz: float = Field(0.5, description="Emotion update rate")


class SpeakerConfig(BaseModel):
    """Speaker embedding configuration."""
    embedding_model: str = Field(
        "speechbrain/spkrec-ecapa-voxceleb",
        description="Speaker embedding model"
    )
    embedding_dim: int = Field(192, description="Embedding dimension")
    update_on_change: bool = Field(True, description="Update on speaker change")
    change_threshold: float = Field(0.3, description="Speaker change threshold")


class CompressionConfig(BaseModel):
    """Compression configuration."""
    text_algorithm: str = Field("brotli", description="Text compression algorithm")
    text_level: int = Field(5, description="Text compression level")
    text_preprocess: bool = Field(True, description="Enable text preprocessing")
    
    prosody_quantization_pitch_bits: int = Field(6, description="Pitch quantization bits")
    prosody_quantization_energy_bits: int = Field(4, description="Energy quantization bits")
    prosody_quantization_rate_bits: int = Field(4, description="Rate quantization bits")
    prosody_delta_encoding: bool = Field(True, description="Enable delta encoding")
    
    timbre_algorithm: str = Field("float16", description="Timbre compression algorithm")


class TTSConfig(BaseModel):
    """TTS configuration."""
    model: str = Field("xtts-v2", description="TTS model type")
    streaming: bool = Field(True, description="Enable streaming synthesis")
    vocoder: str = Field("hifigan", description="Vocoder type")


class NetworkConfig(BaseModel):
    """Network configuration."""
    protocol: str = Field("webrtc", description="Network protocol")
    fec_enabled: bool = Field(True, description="Enable FEC")
    priority_queue: bool = Field(True, description="Enable priority queue")
    max_bandwidth_bps: int = Field(450, description="Max bandwidth in bps")


class QualityConfig(BaseModel):
    """Quality control configuration."""
    target_latency_ms: int = Field(500, description="Target latency in ms")
    graceful_degradation: bool = Field(True, description="Enable graceful degradation")
    interpolate_missing_prosody: bool = Field(True, description="Interpolate missing prosody")


class Config(BaseModel):
    """Main application configuration."""
    audio: AudioConfig = Field(default_factory=AudioConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    prosody: ProsodyConfig = Field(default_factory=ProsodyConfig)
    speaker: SpeakerConfig = Field(default_factory=SpeakerConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)


class Settings(BaseSettings):
    """Environment-based settings."""
    app_name: str = Field("stt-compress-tts", alias="APP_NAME")
    app_version: str = Field("0.1.0", alias="APP_VERSION")
    environment: str = Field("development", alias="ENVIRONMENT")
    
    signaling_server_host: str = Field("0.0.0.0", alias="SIGNALING_SERVER_HOST")
    signaling_server_port: int = Field(8080, alias="SIGNALING_SERVER_PORT")
    signaling_server_url: str = Field("ws://localhost:8080", alias="SIGNALING_SERVER_URL")
    
    backend_host: str = Field("0.0.0.0", alias="BACKEND_HOST")
    backend_port: int = Field(8000, alias="BACKEND_PORT")
    
    quality_mode: str = Field("balanced", alias="QUALITY_MODE")
    
    models_dir: Path = Field(Path("./models"), alias="MODELS_DIR")
    
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    log_file: Optional[Path] = Field(None, alias="LOG_FILE")
    
    debug: bool = Field(False, alias="DEBUG")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables not defined in schema


def load_config(config_path: Optional[Union[Path, str]] = None, quality_mode: Optional[str] = None) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file (can be string or Path)
        quality_mode: Quality mode (minimal, balanced, high)
    
    Returns:
        Configuration object
    """
    if config_path is None:
        settings = Settings()
        quality_mode = quality_mode or settings.quality_mode
        config_path = Path(f"configs/{quality_mode}_mode.yaml")
    else:
        # Convert string to Path if needed
        config_path = Path(config_path) if isinstance(config_path, str) else config_path
        
        # If it's a simple name (no directory separator), look in configs/
        if not config_path.is_absolute() and "/" not in str(config_path) and "\\" not in str(config_path):
            # Check if it needs .yaml suffix
            if not str(config_path).endswith(".yaml"):
                # If it doesn't end with _mode, add it
                if not str(config_path).endswith("_mode"):
                    config_path = Path(f"configs/{config_path}_mode.yaml")
                else:
                    config_path = Path(f"configs/{config_path}.yaml")
            else:
                config_path = Path(f"configs/{config_path}")
    
    if not config_path.exists():
        # Return default config
        logger.warning(f"Config file not found: {config_path}, using default config")
        return Config()
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    return Config(**config_dict)


def save_config(config: Config, output_path: Path) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object
        output_path: Output file path
    """
    config_dict = config.model_dump()
    
    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


# Global settings instance
settings = Settings()

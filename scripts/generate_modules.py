"""
Script to generate all project source files
This creates the complete project structure with all modules
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"

# Module templates
MODULES = {
    # Prosody module
    "prosody/__init__.py": '''"""Prosody extraction module."""

from src.prosody.extractor import ProsodyExtractor, ProsodyFeatures

__all__ = ["ProsodyExtractor", "ProsodyFeatures"]
''',
    
    "prosody/extractor.py": '''"""Main prosody feature extractor."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import parselmouth

from src.prosody.pitch import extract_pitch
from src.prosody.energy import extract_energy
from src.prosody.rate import extract_speaking_rate
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProsodyFeatures:
    """Prosody features container."""
    pitch: Optional[np.ndarray] = None  # F0 contour
    energy: Optional[np.ndarray] = None  # Energy/volume contour
    speaking_rate: Optional[float] = None  # Words per minute
    emotion: Optional[str] = None  # Emotion label
    emphasis: Optional[list] = None  # Word emphasis markers
    timestamp: float = 0.0


class ProsodyExtractor:
    """Extract prosody features from audio."""
    
    def __init__(self, sample_rate: int = 16000):
        """Initialize prosody extractor.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        logger.info("Prosody extractor initialized")
    
    def extract(self, audio: np.ndarray, extract_features: list[str]) -> ProsodyFeatures:
        """Extract prosody features from audio.
        
        Args:
            audio: Audio data as numpy array (float32, -1 to 1)
            extract_features: List of features to extract
        
        Returns:
            Prosody features
        """
        features = ProsodyFeatures()
        
        # Extract requested features
        if "pitch" in extract_features:
            features.pitch = extract_pitch(audio, self.sample_rate)
        
        if "energy" in extract_features:
            features.energy = extract_energy(audio, self.sample_rate)
        
        if "speaking_rate" in extract_features:
            features.speaking_rate = extract_speaking_rate(audio, self.sample_rate)
        
        return features
''',

    "prosody/pitch.py": '''"""Pitch extraction using Parselmouth."""

import numpy as np
import parselmouth

from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_pitch(audio: np.ndarray, sample_rate: int, time_step: float = 0.01) -> np.ndarray:
    """Extract pitch contour from audio.
    
    Args:
        audio: Audio data
        sample_rate: Sample rate
        time_step: Time step for pitch extraction (seconds)
    
    Returns:
        Pitch contour in Hz
    """
    try:
        # Create Praat sound object
        sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
        
        # Extract pitch
        pitch = sound.to_pitch(time_step=time_step)
        
        # Get pitch values
        pitch_values = pitch.selected_array['frequency']
        
        # Replace unvoiced (0) with NaN for easier processing
        pitch_values[pitch_values == 0] = np.nan
        
        return pitch_values
    
    except Exception as e:
        logger.error(f"Error extracting pitch: {e}")
        return np.array([])
''',

    "prosody/energy.py": '''"""Energy extraction."""

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_energy(audio: np.ndarray, sample_rate: int, frame_length: int = 400) -> np.ndarray:
    """Extract energy contour from audio.
    
    Args:
        audio: Audio data
        sample_rate: Sample rate
        frame_length: Frame length in samples
    
    Returns:
        Energy contour
    """
    try:
        # Calculate RMS energy per frame
        frame_hop = frame_length // 2
        num_frames = (len(audio) - frame_length) // frame_hop + 1
        
        energy = np.zeros(num_frames)
        
        for i in range(num_frames):
            start = i * frame_hop
            end = start + frame_length
            frame = audio[start:end]
            
            # RMS energy
            energy[i] = np.sqrt(np.mean(frame ** 2))
        
        return energy
    
    except Exception as e:
        logger.error(f"Error extracting energy: {e}")
        return np.array([])
''',

    "prosody/rate.py": '''"""Speaking rate estimation."""

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_speaking_rate(audio: np.ndarray, sample_rate: int) -> float:
    """Estimate speaking rate from audio.
    
    This is a simple approximation based on zero-crossing rate and energy.
    
    Args:
        audio: Audio data
        sample_rate: Sample rate
    
    Returns:
        Estimated speaking rate (syllables per second)
    """
    try:
        # Simple approximation: count energy peaks
        from scipy import signal
        
        # Calculate energy envelope
        frame_length = int(sample_rate * 0.02)  # 20ms frames
        hop_length = frame_length // 2
        
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i+frame_length]
            energy.append(np.sum(frame ** 2))
        
        energy = np.array(energy)
        
        # Find peaks in energy (approximate syllables)
        peaks, _ = signal.find_peaks(energy, distance=5, height=np.mean(energy))
        
        # Calculate rate
        duration = len(audio) / sample_rate
        rate = len(peaks) / duration if duration > 0 else 0.0
        
        return float(rate)
    
    except Exception as e:
        logger.error(f"Error extracting speaking rate: {e}")
        return 0.0
''',

    "prosody/emotion.py": '''"""Emotion detection (placeholder)."""

from src.utils.logger import get_logger

logger = get_logger(__name__)

# TODO: Implement with SpeechBrain emotion recognition
''',

    "prosody/emphasis.py": '''"""Emphasis detection (placeholder)."""

from src.utils.logger import get_logger

logger = get_logger(__name__)

# TODO: Implement emphasis detection
''',

    # Speaker module
    "speaker/__init__.py": '''"""Speaker embedding module."""

from src.speaker.embedding import SpeakerEmbedding
from src.speaker.profile import TimbreProfile

__all__ = ["SpeakerEmbedding", "TimbreProfile"]
''',

    "speaker/embedding.py": '''"""Speaker embedding extraction."""

import numpy as np
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SpeakerEmbedding:
    """Extract speaker embeddings using SpeechBrain."""
    
    def __init__(self, model_name: str = "speechbrain/spkrec-ecapa-voxceleb", device: str = "cpu"):
        """Initialize speaker embedding model.
        
        Args:
            model_name: SpeechBrain model name
            device: Device to run on
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        logger.info(f"Speaker embedding model: {model_name}")
    
    def load_model(self):
        """Load the model."""
        if self.model is not None:
            return
        
        try:
            from speechbrain.pretrained import EncoderClassifier
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
''',

    "speaker/profile.py": '''"""Timbre profile management."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TimbreProfile:
    """Speaker timbre profile."""
    speaker_id: str
    embedding: np.ndarray
    avg_pitch: float
    avg_speaking_rate: float
    timestamp: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "speaker_id": self.speaker_id,
            "embedding": self.embedding.tolist(),
            "avg_pitch": self.avg_pitch,
            "avg_speaking_rate": self.avg_speaking_rate,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TimbreProfile":
        """Create from dictionary."""
        data["embedding"] = np.array(data["embedding"])
        return cls(**data)
    
    def save(self, filepath: Path):
        """Save profile to file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f)
        logger.info(f"Saved timbre profile to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> "TimbreProfile":
        """Load profile from file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded timbre profile from {filepath}")
        return cls.from_dict(data)
''',

    "speaker/change_detection.py": '''"""Speaker change detection."""

import numpy as np
from scipy.spatial.distance import cosine

from src.utils.logger import get_logger

logger = get_logger(__name__)


def detect_speaker_change(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    threshold: float = 0.3
) -> bool:
    """Detect if speaker has changed based on embeddings.
    
    Args:
        embedding1: First speaker embedding
        embedding2: Second speaker embedding
        threshold: Cosine distance threshold
    
    Returns:
        True if speaker changed, False otherwise
    """
    try:
        distance = cosine(embedding1, embedding2)
        return distance > threshold
    except Exception as e:
        logger.error(f"Error detecting speaker change: {e}")
        return False
''',

}

def create_file(filepath: Path, content: str):
    """Create a file with content."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {filepath}")

def main():
    """Generate all module files."""
    print("Generating project modules...")
    print()
    
    for rel_path, content in MODULES.items():
        filepath = SRC_DIR / rel_path
        create_file(filepath, content)
    
    print()
    print(f"Generated {len(MODULES)} files successfully!")
    print("Project structure is ready.")

if __name__ == "__main__":
    main()

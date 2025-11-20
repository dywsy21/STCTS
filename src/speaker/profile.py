"""Timbre profile management."""

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

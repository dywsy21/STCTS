"""Speaker embedding module."""

from src.speaker.embedding import SpeakerEmbedding
from src.speaker.profile import TimbreProfile
from src.speaker.change_detection import detect_speaker_change
from src.speaker.manager import SpeakerManager

__all__ = [
    "SpeakerEmbedding",
    "TimbreProfile",
    "detect_speaker_change",
    "SpeakerManager",
]
